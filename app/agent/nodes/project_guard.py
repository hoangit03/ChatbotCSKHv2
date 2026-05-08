"""
app/agent/nodes/project_guard.py

Node: Project Guard (Người gác cổng).
Nhiệm vụ: 
  - Đảm bảo state["project_name"] được xác định.
  - Nếu chưa có: trích xuất từ query bằng LLM.

Tối ưu:
  - Cache danh sách dự án (TTL 5 phút) — tránh gọi API mỗi request
  - Chỉ chạy LLM NER khi cần (chưa có project hoặc detect keyword chuyển dự án)
"""
from __future__ import annotations

import json
import time
from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import ToolRegistry
from app.core.config.settings import get_settings
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)



# ── Keywords bypass — cho phép đi qua khi hỏi về danh sách dự án ──
_PROJECT_LISTING_KEYWORDS = [
    "bao nhiêu dự án", "danh sách dự án", "kể tên dự án",
    "có những dự án nào", "dự án hiện tại", "liệt kê dự án",
    "show me projects", "dự án nào đang", "dự án gì", "các dự án",
    "tất cả dự án", "mấy dự án", "bất động sản", "dự án nào phù hợp",
    "gợi ý dự án", "tìm dự án", "giới thiệu dự án", "dự án nào"
]

# ── Keywords gợi ý khách đang chuyển/nhắc tên dự án khác ──
_PROJECT_SWITCH_HINTS = [
    "dự án", "tìm hiểu về", "chuyển sang", "hỏi về",
    "quan tâm", "muốn xem", "thông tin",
]

async def _get_available_projects(registry: ToolRegistry) -> list[dict]:
    """Lấy danh sách dự án (dict chứa id và name) với cache TTL 5 phút từ Redis."""
    redis_pool = registry.get_redis_pool()
    client = None
    cache_key = "system_project_list"
    _CACHE_TTL = 300

    if redis_pool:
        import redis.asyncio as aioredis
        client = aioredis.Redis(connection_pool=redis_pool)
        try:
            cached = await client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            log.warning("redis_project_cache_get_failed", error=str(e))

    projects = []

    # Ưu tiên từ Sales API
    project_tool = registry.get("list_projects")
    if project_tool:
        res = await project_tool.run({"raw_query": "", "project_name": "", "sales_data": {}})
        if res.success and res.data:
            # data là list[dict] chứa 'id' và 'name'
            projects = res.data

    # Fallback sang Vector DB (nếu list_projects fail)
    if not projects:
        vdb = registry.get_vdb()
        if vdb:
            names = await vdb.list_unique_projects()
            projects = [{"id": "unknown", "name": n} for n in names if n]

    if projects and client:
        try:
            await client.set(cache_key, json.dumps(projects, ensure_ascii=False), ex=_CACHE_TTL)
            log.debug("project_cache_refreshed_in_redis", count=len(projects))
        except Exception as e:
            log.warning("redis_project_cache_set_failed", error=str(e))

    return projects


def _query_hints_project_switch(query: str) -> bool:
    """Fast check: query có gợi ý nhắc đến dự án không (dùng trước khi gọi LLM)."""
    q = query.lower()
    return any(hint in q for hint in _PROJECT_SWITCH_HINTS)


async def project_guard_node(state: AgentState, registry: ToolRegistry, llm: ChatPort) -> AgentState:
    """
    Chạy sau classify_intent. 
    Nhiệm vụ: 
      - Đảm bảo dự án được xác định.
      - KHÔNG chặn nếu khách hỏi câu hỏi chung khi đã có sẵn dự án trong session.
    Tối ưu:
      - Cache list_projects (TTL 5 phút)
    """
    cfg = get_settings()
    current_project = state.get("project_name")
    query = state.get("raw_query", "")

    # [BYPASS] Chitchat → cho qua ngay
    from app.agent.state.agent_state import Intent
    if state.get("intent") == Intent.CHITCHAT:
        log.info("project_guard_bypassed_for_chitchat", session=state.get("session_id"))
        return state

    # ── 1. Lấy danh sách dự án (cached — TTL 5 phút) ──
    available_projects = await _get_available_projects(registry)

    # ── [BYPASS] Yêu cầu liệt kê dự án ──
    if any(k in query.lower() for k in _PROJECT_LISTING_KEYWORDS):
        log.info("project_guard_bypassed_for_listing", query=query)
        return state

    # ── 3. Kiểm tra nếu vẫn chưa có dự án nào ──
    if not current_project or current_project.lower() in ["", "string", "none", "unknown"]:
        projects = available_projects
        if not projects:
            state["final_answer"] = (
                f"Chào bạn! Tôi là {cfg.bot_name}. Hiện tại tôi đang cập nhật dữ liệu. "
                "Bạn vui lòng để lại thông tin để em hỗ trợ mình sau nhé!"
            )
            return state

        project_list_str = ", ".join([p["name"] for p in projects])
        state["final_answer"] = cfg.project_suggestion_prompt.format(projects=project_list_str)
        log.info("project_guard_interruption", session=state.get("session_id"))
        return state

    return state