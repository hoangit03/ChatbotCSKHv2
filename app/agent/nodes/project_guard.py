"""
app/agent/nodes/project_guard.py

Node: Project Guard (Người gác cổng).
Nhiệm vụ: 
  - Đảm bảo state["project_name"] được xác định.
  - Nếu chưa có: trích xuất từ query hoặc yêu cầu khách chọn (kèm gợi ý).
"""
from __future__ import annotations

from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import ToolRegistry
from app.core.config.settings import get_settings
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


async def project_guard_node(state: AgentState, registry: ToolRegistry) -> AgentState:
    """
    Chạy sau classify_intent. 
    Nếu không có project_name -> ngắt luồng và yêu cầu chọn dự án.
    """
    cfg = get_settings()
    current_project = state.get("project_name")
    query = state.get("raw_query", "").lower()

    # ── 1. Cố gắng trích xuất tên dự án từ query nếu state chưa có ──────
    # (Trong tương lai có thể dùng LLM Entity Extraction ở đây)
    if not current_project or current_project.lower() in ["string", "default", "none"]:
        # Lấy danh sách dự án thực tế từ DB để so khớp
        vdb = registry.get_vdb() # Giả định registry có helper lấy vdb
        available_projects = await vdb.list_unique_projects()
        
        for p in available_projects:
            if p.lower() in query:
                state["project_name"] = p
                current_project = p
                log.info("project_detected_from_query", project=p, session=state.get("session_id"))
                break

    # ── 2. Nếu vẫn không có project -> Ngắt luồng và yêu cầu chọn ────────
    if not current_project or current_project.lower() in ["string", "default", "none"]:
        vdb = registry.get_vdb()
        projects = await vdb.list_unique_projects()
        
        if not projects:
            # Trường hợp database trống
            state["final_answer"] = "Chào bạn! Hiện tại hệ thống đang cập nhật dữ liệu dự án. Vui lòng quay lại sau ít phút."
            return state

        # Soạn câu trả lời tự nhiên kèm gợi ý (dùng template từ settings)
        project_list_str = ", ".join(projects)
        state["final_answer"] = cfg.project_suggestion_prompt.format(projects=project_list_str)
        
        log.info("project_guard_interruption", session=state.get("session_id"))
        # Khi set final_answer, các node sau (Support/Sales/Synthesizer) sẽ skip hoặc END sớm
        return state

    return state
