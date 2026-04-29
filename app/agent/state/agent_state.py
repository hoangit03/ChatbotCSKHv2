"""
app/agent/state/agent_state.py

AgentState là trái tim của LangGraph.
Mọi node đọc/ghi vào đây — không dùng biến global.

Tại sao LangGraph tốt hơn LangChain Agent:
  - State machine rõ ràng, mọi bước được kiểm soát
  - Dễ debug: log từng node
  - Controllable: có thể interrupt, replay, branch
  - Không bị "hallucinate tool call" vì graph định sẵn luồng
  - Dễ test từng node riêng biệt
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypedDict


class Intent(str, Enum):
    """
    Agent phân loại intent trước khi chọn tool.
    Tránh dùng LLM gọi tool sai.
    """
    CUSTOMER_SUPPORT = "customer_support"   # hỏi về dự án, pháp lý, tiện ích
    SALES_INQUIRY    = "sales_inquiry"      # hỏi giá, tồn kho, đặt cọc
    BOOKING_INTENT   = "booking_intent"     # muốn đặt cọc/giữ chỗ
    CHITCHAT         = "chitchat"           # chào hỏi, tán gẫu, câu hỏi chung
    UNKNOWN          = "unknown"


@dataclass
class SourceRef:
    """Trích dẫn nguồn trả lời."""
    document_code: str
    document_name: str
    doc_group: str
    excerpt: str
    page: Optional[int] = None


@dataclass
class ToolCall:
    """Record một lần gọi tool — để audit/debug."""
    tool_name: str
    input_summary: str
    output_summary: str
    duration_ms: int
    success: bool


class AgentState(TypedDict):
    """
    LangGraph State — kế thừa TypedDict để compat với StateGraph.

    Keys:
      messages       : lịch sử hội thoại (list message dạng dict)
      session_id     : định danh session
      project_name   : dự án đang hỏi (optional filter)
      intent         : Intent enum sau khi classify
      raw_query      : query gốc (sau sanitize)
      was_injected   : cờ injection attempt
      rag_results    : list[dict] từ vector search
      qa_result      : QAItem nếu match
      sales_data     : dict từ sales API
      final_answer   : câu trả lời cuối
      sources        : list[SourceRef]
      tool_calls     : list[ToolCall] — audit trail
      fallback       : True nếu không đủ dữ liệu
      fallback_reason: lý do fallback
      iteration      : số iteration hiện tại
      error          : lỗi nếu có
    """
    messages: list
    session_id: str
    project_name: Optional[str]
    intent: Any                  # Intent enum
    raw_query: str
    was_injected: bool
    rag_results: list
    qa_result: Optional[Any]     # QAItem | None
    qa_hit: bool                 # True nếu QATool tìm thấy match
    query_embedding: Optional[list[float]] # Lưu embedding của query để dùng chung
    sales_data: dict
    final_answer: str
    sources: list
    tool_calls: list
    fallback: bool
    fallback_reason: str
    iteration: int
    error: Optional[str]
    project_newly_confirmed: bool       # True khi guard vừa switch sang project mới
    customer_name: Optional[str]        # Tên khách hàng (từ ChatRequest)
    customer_phone: Optional[str]       # SĐT khách hàng (từ ChatRequest)




def make_initial_state(
    session_id: str,
    raw_query: str,
    project_name: str | None = None,
    customer_name: str | None = None,
    customer_phone: str | None = None,
) -> AgentState:
    return AgentState(
        messages=[],
        session_id=session_id,
        project_name=project_name,
        intent=Intent.UNKNOWN,
        raw_query=raw_query,
        was_injected=False,
        rag_results=[],
        qa_result=None,
        qa_hit=False,
        query_embedding=None,
        sales_data={},
        final_answer="",
        sources=[],
        tool_calls=[],
        fallback=False,
        fallback_reason="",
        iteration=0,
        error=None,
        project_newly_confirmed=False,
        customer_name=customer_name,
        customer_phone=customer_phone,
    )