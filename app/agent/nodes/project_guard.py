"""
app/agent/nodes/project_guard.py

Node: Project Guard (Người gác cổng).
Nhiệm vụ: 
  - Đảm bảo state["project_name"] được xác định.
  - Nếu chưa có: trích xuất từ query bằng LLM.
"""
from __future__ import annotations

import json
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

async def project_guard_node(state: AgentState, registry: ToolRegistry, llm: ChatPort) -> AgentState:
    """
    Chạy sau classify_intent. 
    Nhiệm vụ: 
      - Đảm bảo dự án được xác định.
      - Hỗ trợ đổi ngữ cảnh nếu khách nhắc tên dự án khác (Dùng LLM Extract).
      - KHÔNG chặn nếu khách hỏi câu hỏi chung khi đã có sẵn dự án trong session.
    """
    cfg = get_settings()
    current_project = state.get("project_name")
    query = state.get("raw_query", "")

    vdb = registry.get_vdb()
    available_projects = await vdb.list_unique_projects()
    
    detected_project = None

    if query and available_projects:
        try:
            # LLM Extraction
            system_msg = NER_PROMPT.format(projects=", ".join(available_projects))
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
            if data.get("found") and data.get("project_name") in available_projects:
                detected_project = data.get("project_name")
        except Exception as e:
            log.error("project_extraction_failed", error=str(e))
            # Fallback to exact match as safety net
            for p in available_projects:
                if p.lower() in query.lower():
                    detected_project = p
                    break

    # Nếu phát hiện dự án mới trong query -> Cập nhật context
    if detected_project:
        if current_project != detected_project:
            log.info("project_context_switched", old=current_project, new=detected_project)
            state["project_name"] = detected_project
            state["project_newly_confirmed"] = True
            current_project = detected_project
        else:
            log.info("project_confirmed_in_query", project=detected_project)

    # ── 2. Kiểm tra nếu vẫn chưa có dự án nào (cả trong state lẫn session) ──
    if not current_project or current_project.lower() in ["", "string", "none", "unknown"]:
        # Nếu không có dự án VÀ không tìm thấy dự án trong query -> Mới yêu cầu chọn
        projects = available_projects
        if not projects:
            state["final_answer"] = (
                "Chào bạn! Tôi là chatbot CTlotus. Hiện tại tôi đang cập nhật dữ liệu. "
                "Bạn vui lòng để lại thông tin để em hỗ trợ mình sau nhé!"
            )
            return state

        project_list_str = ", ".join(projects)
        state["final_answer"] = cfg.project_suggestion_prompt.format(projects=project_list_str)
        log.info("project_guard_interruption", session=state.get("session_id"))
        return state

    # Nếu đã có current_project -> Tiếp tục luồng (không quan tâm query có nhắc lại tên dự án hay không)
    return state