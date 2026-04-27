"""
app/agent/graph/agent_graph.py

LangGraph StateGraph — định nghĩa toàn bộ luồng Agent.

Tại sao LangGraph > LangChain Agent:
  ┌─────────────────────────────────────────────────────────┐
  │  LangChain Agent              │  LangGraph              │
  │  ──────────────────────────── │  ───────────────────── │
  │  Black-box ReAct loop         │  Explicit state machine │
  │  Khó debug từng bước          │  Từng node có thể test  │
  │  LLM tự chọn tool (lỗi đc)   │  Graph kiểm soát luồng  │
  │  Không control được iteration │  Max iteration rõ ràng  │
  │  Khó thêm logic branching     │  Conditional edge đơn  │
  └─────────────────────────────────────────────────────────┘

Graph structure:
  START
    ↓
  [classify_intent]  ← sanitize + phân loại
    ↓
  (conditional edge: route_by_intent)
    ├── "support_node" → [support_node]  → [synthesizer]
    └── "sales_node"   → [sales_node]   → [synthesizer]
                                              ↓
                                            END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.intent_classifier import classify_intent, route_by_intent
from app.agent.nodes.sales_node import SalesNode
from app.agent.nodes.support_node import SupportNode
from app.agent.nodes.synthesizer_node import SynthesizerNode
from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


def build_agent_graph(
    llm: ChatPort,
    tool_registry: ToolRegistry,
    max_iterations: int = 8,
) -> "CompiledGraph":
    """
    Factory function — build và compile LangGraph.
    DIP: nhận interface, không import implementation cụ thể.
    """
    support_node = SupportNode(tool_registry)
    sales_node = SalesNode(tool_registry)
    synthesizer = SynthesizerNode(llm)

    # Guard: max iterations
    async def guarded_support(state: AgentState) -> AgentState:
        if state.get("iteration", 0) >= max_iterations:
            state["fallback"] = True
            state["fallback_reason"] = "Max iterations reached"
            state["final_answer"] = (
                "Xin lỗi, tôi cần thêm thông tin để trả lời câu hỏi này. "
                "Vui lòng liên hệ Sales để được hỗ trợ."
            )
            return state
        return await support_node(state)

    async def guarded_sales(state: AgentState) -> AgentState:
        if state.get("iteration", 0) >= max_iterations:
            state["fallback"] = True
            state["final_answer"] = "Vui lòng liên hệ Sales để biết thêm chi tiết."
            return state
        return await sales_node(state)

    # Build graph
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("support_node", guarded_support)
    builder.add_node("sales_node", guarded_sales)
    builder.add_node("synthesizer", synthesizer)

    # Edges
    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "support_node": "support_node",
            "sales_node": "sales_node",
        },
    )

    builder.add_edge("support_node", "synthesizer")
    builder.add_edge("sales_node", "synthesizer")
    builder.add_edge("synthesizer", END)

    compiled = builder.compile()
    log.info("agent_graph_compiled", max_iterations=max_iterations)
    return compiled