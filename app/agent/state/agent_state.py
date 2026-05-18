"""
app/agent/state/agent_state.py

AgentState cho CustomerGraph (Luồng A — Khách hàng).

v2 — Loại bỏ:
  - CustomerStage, ScarcityLevel, USPItem (không còn inject USP/Stage)
  - customer_stage, stage_signals, usps_used, scarcity_level,
    cross_sell_suggestions, booking_confirmation, price_disclosed, appointment_booked
  → Giảm ~35% field, giảm token rác gửi LLM.

SaleAgentState cho Luồng B nằm ở app/agent/state/sale_state.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TypedDict


class Intent(str, Enum):
    """Ý định người dùng — dùng chung cho cả Customer và Sale classifier."""
    CUSTOMER_SUPPORT    = "customer_support"
    SALES_INQUIRY       = "sales_inquiry"
    CONSULTATION_INTENT = "consultation_intent"
    COMPARISON_INTENT   = "comparison_intent"
    CHITCHAT            = "chitchat"
    UNKNOWN             = "unknown"


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
    """Record một lần gọi tool — audit/debug trail."""
    tool_name: str
    input_summary: str
    output_summary: str
    duration_ms: int
    success: bool


class AgentState(TypedDict):
    """
    LangGraph State cho CustomerGraph.

    Chỉ chứa các field thực sự cần thiết cho luồng khách hàng.
    Không có USP, CustomerStage, ScarcityLevel.
    """
    # ── Core ─────────────────────────────────────────
    messages: list
    session_id: str
    project_name: Optional[str]
    project_id: Optional[str]
    intent: Any                         # Intent enum
    raw_query: str
    was_injected: bool

    # ── RAG / QA ─────────────────────────────────────
    rag_results: list
    qa_result: Optional[Any]
    qa_hit: bool
    query_embedding: Optional[list[float]]

    # ── Output ───────────────────────────────────────
    sales_data: dict
    final_answer: str
    sources: list
    tool_calls: list
    fallback: bool
    fallback_reason: str
    iteration: int
    error: Optional[str]

    # ── Identity ─────────────────────────────────────
    project_newly_confirmed: bool
    customer_name: Optional[str]
    customer_phone: Optional[str]
    tool_kwargs: dict

    # ── UI ───────────────────────────────────────────
    suggested_questions: list
    stream_queue: Optional[Any]
    human_handover_requested: bool


# ── Factory ───────────────────────────────────────────────────────

def make_initial_state(
    session_id: str,
    raw_query: str,
    project_name: str | None = None,
    project_id: str | None = None,
    customer_name: str | None = None,
    customer_phone: str | None = None,
    # Deprecated params — kept for backward compat, ignored
    user_type: str = "customer",
    min_role_level: int | None = None,
    customer_stage: str | None = None,
    usps_used: list | None = None,
    appointment_booked: bool = False,
) -> AgentState:
    """Factory khởi tạo AgentState cho CustomerGraph."""
    return AgentState(
        messages=[],
        session_id=session_id,
        project_name=project_name,
        project_id=project_id,
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
        tool_kwargs={},
        suggested_questions=[],
        stream_queue=None,
        human_handover_requested=False,
    )