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

NER_PROMPT = """Bạn là trợ lý trích xuất thực thể tên dự án bất động sản.
Dưới đây là danh sách các dự án hiện có trong hệ thống:
{projects}

Khách hàng sẽ đặt một câu hỏi. Nhiệm vụ của bạn là kiểm tra xem trong câu hỏi có nhắc đến dự án nào trong danh sách trên hay không.
Chú ý: Khách có thể viết tắt, viết sai chính tả một chút. Hãy suy luận cẩn thận.
Bạn PHẢI trả về JSON với định dạng sau, không kèm bất kỳ markdown hay chữ nào khác:
{{"found": true_hoặc_false, "project_name": "Tên_dự_án_chính_xác_trong_danh_sách_nếu_có_ngược_lại_để_trống"}}
"""

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

# ── Project List Cache (module-level, shared across requests) ──
_project_cache: dict = {"projects": [], "ts": 0.0}
_CACHE_TTL = 300  # 5 phút


async def _get_available_projects(registry: ToolRegistry) -> list[dict]:
    """Lấy danh sách dự án (dict chứa id và name) với cache TTL 5 phút."""
    now = time.monotonic()
    if _project_cache["projects"] and (now - _project_cache["ts"]) < _CACHE_TTL:
        return _project_cache["projects"]

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

    if projects:
        _project_cache["projects"] = projects
        _project_cache["ts"] = now
        log.debug("project_cache_refreshed", count=len(projects))

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
      - Hỗ trợ đổi ngữ cảnh nếu khách nhắc tên dự án khác (Dùng LLM Extract).
      - KHÔNG chặn nếu khách hỏi câu hỏi chung khi đã có sẵn dự án trong session.
    Tối ưu:
      - Cache list_projects (TTL 5 phút)
      - Chỉ chạy LLM NER khi cần (chưa có project hoặc query hint chuyển dự án)
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
    project_names = [p["name"] for p in available_projects]

    # ── [BYPASS] Yêu cầu liệt kê dự án ──
    if any(k in query.lower() for k in _PROJECT_LISTING_KEYWORDS):
        log.info("project_guard_bypassed_for_listing", query=query)
        return state

    # ── 2. Trích xuất dự án từ query ──
    # Chỉ gọi LLM NER khi:
    #   a) Chưa có project (phải detect)
    #   b) Đã có project NHƯNG query gợi ý nhắc tên dự án khác
    detected_project_name = None
    need_ner = (
        not current_project 
        or current_project.lower() in ["", "string", "none", "unknown"]
        or _query_hints_project_switch(query)
    )

    if need_ner and query and project_names:
        try:
            system_msg = NER_PROMPT.format(projects=", ".join(project_names))
            resp = await llm.chat(
                messages=[LLMMessage(role="user", content=query)],
                system=system_msg,
                temperature=0.0
            )
            content = resp.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            if data.get("found") and data.get("project_name") in project_names:
                detected_project_name = data.get("project_name")
        except Exception as e:
            log.error("project_extraction_failed", error=str(e))
            # Fallback to exact match as safety net
            for p_name in project_names:
                if p_name.lower() in query.lower():
                    detected_project_name = p_name
                    break
    elif not need_ner:
        log.debug("project_ner_skipped", reason="project_confirmed_no_switch_hint")

    # Nếu phát hiện dự án mới trong query -> Cập nhật context
    if detected_project_name:
        # Tìm ID tương ứng
        p_id = next((p["id"] for p in available_projects if p["name"] == detected_project_name), "unknown")
        
        if current_project != detected_project_name:
            log.info("project_context_switched", old=current_project, new=detected_project_name, id=p_id)
            state["project_name"] = detected_project_name
            state["project_id"] = p_id
            state["project_newly_confirmed"] = True
            current_project = detected_project_name
        else:
            state["project_id"] = p_id
            log.info("project_confirmed_in_query", project=detected_project_name, id=p_id)

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