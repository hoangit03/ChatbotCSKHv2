"""
app/agent/state/sale_state.py

SaleAgentState — State riêng cho SaleGraph (Luồng B — Sale nội bộ).

Gọn nhẹ: không có CustomerStage, USP, ScarcityLevel.
Sale chỉ cần: query → tool data → synthesize.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from app.agent.state.agent_state import SourceRef, ToolCall  # noqa: F401 — re-export


class SaleAgentState(TypedDict):
    """
    LangGraph State cho SaleGraph.

    Không có:
    - customer_stage / stage_signals (Sale không cần tâm lý funnel)
    - usps_used / scarcity_level     (Sale không inject USP)
    - price_disclosed / price_disclosure_blocked (Sale toàn quyền về giá)
    - human_handover_requested       (Sale là người — không cần handover)
    """
    # ── Core ─────────────────────────────────────────
    messages: list
    session_id: str
    project_name: Optional[str]
    project_id: Optional[str]
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

    # ── Customer info (khi Sale tra cứu cho KH cụ thể) ───
    customer_name: Optional[str]
    customer_phone: Optional[str]
    tool_kwargs: dict

    # ── UI ───────────────────────────────────────────
    suggested_questions: list
    stream_queue: Optional[Any]


def make_sale_initial_state(
    session_id: str,
    raw_query: str,
    project_name: str | None = None,
    project_id: str | None = None,
    customer_name: str | None = None,
    customer_phone: str | None = None,
) -> SaleAgentState:
    """Factory khởi tạo SaleAgentState."""
    return SaleAgentState(
        messages=[],
        session_id=session_id,
        project_name=project_name,
        project_id=project_id,
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
        customer_name=customer_name,
        customer_phone=customer_phone,
        tool_kwargs={},
        suggested_questions=[],
        stream_queue=None,
    )
