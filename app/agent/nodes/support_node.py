"""
app/agent/nodes/support_node.py

Node: Customer Support.
Chiến lược: QA lookup (inject context) → RAG search → Synthesizer tổng hợp.

RAG-style v4:
  - QATool hit → rag_results được populate với Q&A context (không early-return)
  - RAG search → append thêm document context vào rag_results
  - SynthesizerNode nhận combined context → LLM tổng hợp câu trả lời
"""
from __future__ import annotations

from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import ToolRegistry
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class SupportNode:
    """
    SRP: chỉ lo luồng customer support.
    Không biết gì về LLM hay graph — chỉ điều phối tools.

    Thứ tự:
    1. QA lookup → nếu hit, inject Q&A answer vào rag_results (RAG-style)
    2. RAG search → run RAG để bổ sung context (luôn chạy để lấy context phong phú nhất)
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1

        # ── 1. QA lookup (semantic search trên qa_pairs collection) ───
        qa_tool = self._registry.get("qa_lookup")
        if qa_tool:
            result, call = await qa_tool.execute(state)
            state["tool_calls"] = state.get("tool_calls", []) + [call]

            if result.success and state.get("qa_hit"):
                best_score = state["qa_result"].score
                log.info(
                    "support_qa_hit",
                    session=state.get("session_id"),
                    score=best_score,
                )

        # ── 2. RAG search (document vector search trên realestate_kb) ──
        rag_tool = self._registry.get("rag_search")
        if rag_tool:
            result, call = await rag_tool.execute(state)
            state["tool_calls"] = state.get("tool_calls", []) + [call]

        # ── Fallback: chỉ khi cả Q&A lẫn RAG đều không có kết quả ──
        # Không check result.success vì Q&A có thể đã inject context rồi
        if not state.get("rag_results"):
            state["fallback"] = True
            state["fallback_reason"] = "Không tìm thấy thông tin trong tài liệu và bộ Q&A"
            log.info("support_no_context", session=state.get("session_id"))

        return state