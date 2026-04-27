"""
app/agent/nodes/support_node.py

Node: Customer Support.
Chiến lược: QA lookup trước (nhanh) → RAG nếu không match → fallback.
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
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1

        # 1. QA lookup (nhanh, chính xác)
        qa_tool = self._registry.get("qa_lookup")
        if qa_tool:
            result, call = await qa_tool.execute(state)
            state["tool_calls"] = state.get("tool_calls", []) + [call]
            if result.success and state.get("qa_result"):
                # Direct answer — không cần RAG hay LLM
                qa = state["qa_result"]
                state["final_answer"] = qa.answer
                state["sources"] = []
                log.info("support_qa_hit", session=state.get("session_id"))
                return state

        # 2. RAG search
        rag_tool = self._registry.get("rag_search")
        if rag_tool:
            result, call = await rag_tool.execute(state)
            state["tool_calls"] = state.get("tool_calls", []) + [call]
            if not result.success or not state.get("rag_results"):
                state["fallback"] = True
                state["fallback_reason"] = "Không tìm thấy thông tin trong tài liệu"
                log.info("support_no_rag_results", session=state.get("session_id"))

        return state