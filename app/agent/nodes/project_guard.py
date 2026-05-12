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


# Nếu quá rộng (vd: "bất động sản"), khách hàng sẽ vào sales_node mà không có project context.
_PROJECT_LISTING_KEYWORDS = [
    "bao nhiêu dự án", "danh sách dự án", "kể tên dự án",
    "có những dự án nào", "liệt kê dự án", "liệt kê tất cả dự án",
    "show me projects", "tất cả dự án", "các dự án hiện có",
]

# ── Keywords gợi ý khách đang chuyển/nhắc tên dự án khác ──
# Chỉ giữ những hint cực kỳ cụ thể về chuyển ngữ cảnh dự án.
_PROJECT_SWITCH_HINTS = [
    "chuyển sang dự án", "hỏi về dự án", "tìm hiểu dự án",
    "xem dự án",
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
        # FIX BUG-06: Truyền AgentState-compatible dict đầy đủ các key bắt buộc
        # ProjectListTool.run() cần: sales_data (dict), raw_query (str)
        minimal_state = {
            "raw_query": "",
            "project_name": None,
            "project_id": None,
            "session_id": "__system__",
            "sales_data": {},
            "tool_kwargs": {},
            "user_type": "sale",
            "messages": [],
            "intent": None,
            "was_injected": False,
            "rag_results": [],
            "qa_result": None,
            "qa_hit": False,
            "query_embedding": None,
            "final_answer": "",
            "sources": [],
            "tool_calls": [],
            "fallback": False,
            "fallback_reason": "",
            "iteration": 0,
            "error": None,
            "project_newly_confirmed": False,
            "customer_name": None,
            "customer_phone": None,
            "customer_stage": "awareness",
            "stage_signals": [],
            "usps_used": [],
            "appointment_booked": False,
            "scarcity_level": "none",
            "cross_sell_suggestions": [],
            "booking_confirmation": False,
            "price_disclosed": False,
            "suggested_questions": [],
            "stream_queue": None,
            "human_handover_requested": False,
        }
        res = await project_tool.run(minimal_state)
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
      - Đảm bảo dự án được xác định trước khi xử lý các intent cần context dự án.
      - KHÔNG chặn nếu intent là SALES_INQUIRY/CONSULTATION → để sales_node tự gọi đúng tool.
      - Chỉ hỏi lại project khi intent là CUSTOMER_SUPPORT / COMPARISON / UNKNOWN.
    Tối ưu:
      - Cache list_projects (TTL 5 phút) — chỉ gọi khi thực sự cần kiểm tra.
    """
    cfg = get_settings()
    current_project = state.get("project_name")

    from app.agent.state.agent_state import Intent
    intent = state.get("intent")

    # [BYPASS] Chitchat → cho qua ngay
    if intent == Intent.CHITCHAT:
        log.info("project_guard_bypassed_for_chitchat", session=state.get("session_id"))
        return state

    # [BYPASS] SALES_INQUIRY → sales_node sẽ tự gọi list_projects / search_units / get_inventory
    # với đúng filter. Guard không cần xen vào.
    if intent == Intent.SALES_INQUIRY:
        log.info("project_guard_bypassed_for_sales_inquiry", session=state.get("session_id"))
        return state

    # [BYPASS] CONSULTATION_INTENT → ConsultationTool tự xử lý slot-filling (hỏi dự án khi cần).
    if intent == Intent.CONSULTATION_INTENT:
        log.info("project_guard_bypassed_for_consultation", session=state.get("session_id"))
        return state

    # Từ đây chỉ còn: CUSTOMER_SUPPORT, COMPARISON_INTENT, UNKNOWN
    # → các intent này CẦN project context để RAG/QA search đúng collection.

    # Nếu đã có project hợp lệ trong session → cho qua
    if current_project and current_project.lower() not in ["", "string", "none", "unknown"]:
        return state

    # ── Chưa có project: lấy danh sách và hỏi khách chọn ──
    # Lấy danh sách dự án (cached — TTL 5 phút)
    available_projects = await _get_available_projects(registry)

    if not available_projects:
        state["final_answer"] = (
            f"Chào bạn! Tôi là {cfg.bot_name}. Hiện tại tôi đang cập nhật dữ liệu. "
            "Bạn vui lòng để lại thông tin để em hỗ trợ mình sau nhé!"
        )
        return state

    state["final_answer"] = cfg.project_suggestion_prompt
    log.info("project_guard_interruption", session=state.get("session_id"))
    return state