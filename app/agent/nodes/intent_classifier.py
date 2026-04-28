"""
app/agent/nodes/intent_classifier.py

Node 1: Phân loại intent.
Quyết định luồng nào tiếp theo trong graph.
Dùng LLM (Zero-shot) để phân loại intent một cách linh hoạt, thay vì Regex.
"""
from __future__ import annotations

import json
from app.agent.state.agent_state import AgentState, Intent
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger
from app.shared.security.guards import sanitize_input

log = get_logger(__name__)

CLASSIFIER_PROMPT = """Bạn là một trợ lý thông minh cho chatbot bất động sản.
Nhiệm vụ của bạn là:
1. Đọc lịch sử hội thoại và câu hỏi mới nhất của khách hàng.
2. Phân loại ý định của câu hỏi mới nhất vào 1 trong các nhóm sau:
   - "customer_support": Khách hỏi thông tin dự án, pháp lý, tiện ích, tiến độ, chính sách bán hàng.
   - "sales_inquiry": Khách hỏi giá, bao nhiêu tiền, tồn kho, còn căn không.
   - "booking_intent": Khách thể hiện ý định muốn đặt cọc, giữ chỗ, mua căn, xác nhận mua.
   - "chitchat": Khách chào hỏi, đồng ý/từ chối giao tiếp chung (vd: "có tôi muốn", "ok", "dạ"), hoặc các câu hỏi không liên quan đến BĐS.
   - "unknown": Không thể phân loại.
3. Nếu câu hỏi mới nhất bị thiếu ngữ cảnh (ví dụ: "có tôi muốn", "cái đó giá bao nhiêu", "nó ở đâu"), hãy viết lại câu hỏi (rewritten_query) bằng cách kết hợp với lịch sử hội thoại để tạo thành một câu hoàn chỉnh, dùng để tìm kiếm tài liệu. Nếu câu hỏi đã đủ ý, giữ nguyên.

Bạn PHẢI trả về duy nhất một chuỗi JSON có format như sau, không có markdown:
{
  "intent": "tên_intent",
  "rewritten_query": "câu hỏi đã được viết lại cho đầy đủ ý nghĩa"
}

LỊCH SỬ HỘI THOẠI:
{history}
"""


async def classify_intent(state: AgentState, llm: ChatPort) -> AgentState:
    """
    Node: sanitize input, classify intent bằng LLM.
    Output: state với intent và raw_query đã set.
    """
    raw = state.get("raw_query", "")
    clean, injected = sanitize_input(raw)

    if injected:
        log.warning(
            "injection_attempt",
            session=state.get("session_id"),
            prefix=raw[:80],
        )

    state["raw_query"] = clean
    state["was_injected"] = injected

    # Build history string
    history_str = ""
    messages = state.get("messages") or []
    for msg in messages[-4:]:  # Lấy 2 turns gần nhất (4 messages) để tối ưu token (GPT-4 mini)
        role = "Khách" if msg.get("role") == "user" else "Bot"
        content = msg.get("content", "")
        if content:
            history_str += f"{role}: {content}\n"
    if not history_str:
        history_str = "(Không có lịch sử)"

    try:
        system_msg = CLASSIFIER_PROMPT.format(history=history_str)
        resp = await llm.chat(
            messages=[LLMMessage(role="user", content=clean)],
            system=system_msg,
            temperature=0.0
        )
        content = resp.content.strip()
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
        else:
            data = {}
            
        intent_str = data.get("intent", "unknown").lower()
        rewritten = data.get("rewritten_query", clean)
        
        # Nếu LLM quyết định viết lại câu hỏi, cập nhật raw_query để RAG lấy đúng tài liệu
        if rewritten and rewritten != clean:
            log.info("query_rewritten", original=clean, rewritten=rewritten)
            state["raw_query"] = rewritten
        
        # Map string to Enum
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.UNKNOWN
            
    except Exception as e:
        log.error("intent_classification_failed", error=str(e), fallback="UNKNOWN")
        intent = Intent.UNKNOWN

    state["intent"] = intent
    log.info(
        "intent_classified",
        session=state.get("session_id"),
        intent=intent.value,
        query_len=len(clean),
    )
    return state


def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge — trả về tên node tiếp theo.
    LangGraph gọi function này để quyết định branch.
    """
    intent = state.get("intent", Intent.UNKNOWN)

    if intent == Intent.BOOKING_INTENT:
        return "sales_node"        # Booking → thẳng vào sales
    if intent == Intent.SALES_INQUIRY:
        return "sales_node"        # Hỏi giá/tồn kho → sales
    if intent == Intent.CUSTOMER_SUPPORT:
        return "support_node"      # Hỏi thông tin dự án → RAG + QA
    if intent == Intent.CHITCHAT:
        return "synthesizer"       # Chitchat → trả lời ngay
    return "support_node"          # Unknown → thử support trước