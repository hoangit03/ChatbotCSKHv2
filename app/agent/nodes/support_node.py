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
import asyncio

from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import ToolRegistry
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class SupportNode:
    """
    SRP: chỉ lo luồng customer support.
    Không biết gì về LLM hay graph — chỉ điều phối tools.

    Thứ tự:
    1. Pre-compute embedding nếu chưa có.
    2. Chạy QA lookup và RAG search song song.
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1

        qa_tool = self._registry.get("qa_lookup")
        rag_tool = self._registry.get("rag_search")

        # ── Pre-compute query embedding ──
        if not state.get("query_embedding") and rag_tool:
            try:
                state["query_embedding"] = await rag_tool._embed.embed_one(state["raw_query"])
            except Exception as e:
                log.error("support_node_embed_failed", error=str(e))

        # ── Chạy song song Q&A và RAG ──
        tasks = []
        if qa_tool:
            tasks.append(qa_tool.execute(state))
        if rag_tool:
            tasks.append(rag_tool.execute(state))

        if tasks:
            results = await asyncio.gather(*tasks)
            # Log tool calls
            for _, call in results:
                if "tool_calls" not in state or state["tool_calls"] is None:
                    state["tool_calls"] = []
                state["tool_calls"].append(call)

            # Check QA hit specifically from results
            for result, _ in results:
                if getattr(result, "success", False) and state.get("qa_hit"):
                    # We assume qa_result is updated by QATool.
                    if state.get("qa_result"):
                        best_score = state["qa_result"].score
                        log.info(
                            "support_qa_hit",
                            session=state.get("session_id"),
                            score=best_score,
                        )

        # ── Fallback: chỉ khi cả Q&A lẫn RAG đều không có kết quả ──
        if not state.get("rag_results"):
            state["fallback"] = True
            state["fallback_reason"] = "Không tìm thấy thông tin trong tài liệu và bộ Q&A"
            log.info("support_no_context", session=state.get("session_id"))

        return state