"""
app/agent/nodes/intent_classifier.py

Node 1: Phân loại intent.
Quyết định luồng nào tiếp theo trong graph.
Dùng rule-based + LLM fallback — tránh hallucination khi rule đủ.
"""
from __future__ import annotations

import re

from app.agent.state.agent_state import AgentState, Intent
from app.shared.logging.logger import get_logger
from app.shared.security.guards import sanitize_input

log = get_logger(__name__)

# Keywords cho rule-based classifier (nhanh, rẻ)
_SALES_KW = re.compile(
    r"giá|price|bao nhiêu tiền|còn căn|tồn kho|còn hàng|đặt cọc|"
    r"giữ chỗ|booking|mua căn|thanh toán|vay|trả góp|diện tích|"
    r"phòng ngủ|căn hộ còn|inventory|available",
    re.IGNORECASE,
)
_BOOKING_KW = re.compile(
    r"đặt cọc|đặt chỗ|giữ căn|booking|tôi muốn mua|tôi quyết định|"
    r"cho tôi đặt|xác nhận mua",
    re.IGNORECASE,
)
_SUPPORT_KW = re.compile(
    r"pháp lý|tiện ích|vị trí|tiến độ|bàn giao|quy hoạch|"
    r"sổ hồng|sổ đỏ|hạ tầng|trường học|bệnh viện|an ninh|"
    r"chính sách bán hàng|ưu đãi|brochure|mặt bằng|faq|hỏi đáp",
    re.IGNORECASE,
)


async def classify_intent(state: AgentState) -> AgentState:
    """
    Node: sanitize input, classify intent.
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

    # Rule-based — ưu tiên BOOKING vì nó cụ thể nhất
    if _BOOKING_KW.search(clean):
        intent = Intent.BOOKING_INTENT
    elif _SALES_KW.search(clean):
        intent = Intent.SALES_INQUIRY
    elif _SUPPORT_KW.search(clean):
        intent = Intent.CUSTOMER_SUPPORT
    else:
        intent = Intent.UNKNOWN   # Graph sẽ thử cả RAG lẫn QA

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
    return "support_node"          # Unknown → thử support trước