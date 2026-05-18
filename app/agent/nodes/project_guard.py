"""
app/agent/nodes/project_guard.py

Node: Customer Guard (Luồng A — Khách hàng).

Nhiệm vụ:
  1. Chặn các câu hỏi về giá/tồn kho chi tiết → mời liên hệ Sale.
  2. Đảm bảo project_name được xác định trước khi RAG/QA search.
  3. Cache danh sách dự án (TTL 5 phút, Redis).

v2 — Xóa bỏ: logic bypass cho user_type == "sale" (Sale có Graph riêng).
"""
from __future__ import annotations

import json
import time

from app.agent.state.agent_state import AgentState, Intent
from app.agent.tools.base_tool import ToolRegistry
from app.core.config.settings import get_settings
from app.core.interfaces.llm_port import ChatPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


# ── Cache local (in-process fallback nếu Redis lỗi) ───────────────
_local_project_cache: list[dict] = []
_local_cache_ts: float = 0.0
_LOCAL_CACHE_TTL = 300  # 5 phút


async def _get_available_projects(registry: ToolRegistry) -> list[dict]:
    """Lấy danh sách dự án với cache Redis TTL 5 phút."""
    global _local_project_cache, _local_cache_ts

    import redis.asyncio as aioredis

    redis_pool = registry.get_redis_pool()
    cache_key  = "system_project_list"
    _CACHE_TTL = 300

    # 1. Thử Redis cache
    if redis_pool:
        client = aioredis.Redis(connection_pool=redis_pool)
        try:
            cached = await client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            log.warning("redis_project_cache_get_failed", error=str(e))

    # 2. In-process cache fallback
    if _local_project_cache and (time.monotonic() - _local_cache_ts) < _LOCAL_CACHE_TTL:
        return _local_project_cache

    # 3. Gọi Sales API
    projects: list[dict] = []
    project_tool = registry.get("list_projects")
    if project_tool:
        minimal_state = {
            "raw_query": "", "project_name": None, "project_id": None,
            "session_id": "__system__", "sales_data": {}, "tool_kwargs": {},
            "messages": [], "intent": None, "was_injected": False,
            "rag_results": [], "qa_result": None, "qa_hit": False,
            "query_embedding": None, "final_answer": "", "sources": [],
            "tool_calls": [], "fallback": False, "fallback_reason": "",
            "iteration": 0, "error": None, "project_newly_confirmed": False,
            "customer_name": None, "customer_phone": None,
            "suggested_questions": [], "stream_queue": None,
            "human_handover_requested": False,
        }
        res = await project_tool.run(minimal_state)
        if res.success and res.data:
            projects = res.data

    # 4. Fallback: Vector DB
    if not projects:
        vdb = registry.get_vdb()
        if vdb:
            names = await vdb.list_unique_projects()
            projects = [{"id": "unknown", "name": n} for n in names if n]

    # 5. Ghi cache
    if projects:
        _local_project_cache = projects
        _local_cache_ts = time.monotonic()
        if redis_pool:
            try:
                client = aioredis.Redis(connection_pool=redis_pool)
                await client.set(cache_key, json.dumps(projects, ensure_ascii=False), ex=_CACHE_TTL)
            except Exception as e:
                log.warning("redis_project_cache_set_failed", error=str(e))

    return projects


# ── Câu trả lời chặn sales inquiry ─────────────────────────────
_SALES_BLOCK_MSG = (
    "Dạ, để được tư vấn chi tiết về **giá cả**, **tình trạng căn hộ** và **bảng hàng** "
    "của từng dự án, anh/chị vui lòng:\n"
    "• Để lại số điện thoại để chuyên viên liên hệ tư vấn trực tiếp\n"
    "• Hoặc đặt lịch hẹn xem nhà mẫu để được báo giá chính xác kèm ưu đãi\n\n"
    "Anh/chị có thể hỏi em về **thông tin tổng quan**, **tiện ích**, "
    "**chính sách** hay **danh sách các dự án** nhé!"
)


async def project_guard_node(state: AgentState, registry: ToolRegistry, llm: ChatPort) -> AgentState:
    """
    Customer Guard — chạy sau classify_intent.

    Luồng quyết định:
      CHITCHAT            → cho qua (synthesizer tự xử lý)
      SALES_INQUIRY       → chặn giá/tồn kho chi tiết, trả thông báo redirect
      (chưa có project)   → hỏi khách chọn dự án
      còn lại             → cho qua (support_node xử lý)
    """
    cfg     = get_settings()
    intent  = state.get("intent")
    current = state.get("project_name")

    # Chitchat — pass thẳng
    if intent == Intent.CHITCHAT:
        return state

    # SALES_INQUIRY → chặn (khách hàng không được xem giá/tồn kho chi tiết)
    if intent == Intent.SALES_INQUIRY and current and current.lower() not in ("", "none", "unknown"):
        state["final_answer"] = _SALES_BLOCK_MSG
        state["suggested_questions"] = [
            "Để lại số điện thoại để được tư vấn miễn phí",
            "Dự án nào đang mở bán?",
        ]
        log.info("customer_guard_blocked_sales_inquiry", session=state.get("session_id"))
        return state

    # CONSULTATION_INTENT → pass (support_node tự điều hướng)
    if intent == Intent.CONSULTATION_INTENT:
        return state

    # Nếu đã có project → pass
    if current and current.lower() not in ("", "string", "none", "unknown"):
        return state

    # SALES_INQUIRY không có project cụ thể → pass (để list_projects chạy)
    if intent == Intent.SALES_INQUIRY:
        return state

    # CUSTOMER_SUPPORT / COMPARISON / UNKNOWN chưa có project → hỏi
    available = await _get_available_projects(registry)
    if not available:
        state["final_answer"] = (
            f"Chào bạn! Tôi là {cfg.bot_name}. "
            "Hiện tôi đang cập nhật danh sách dự án. "
            "Bạn vui lòng để lại thông tin để em hỗ trợ nhé!"
        )
        return state

    state["final_answer"] = cfg.project_suggestion_prompt
    log.info("customer_guard_asked_project", session=state.get("session_id"))
    return state