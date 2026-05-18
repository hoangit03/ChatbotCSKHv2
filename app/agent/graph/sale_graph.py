"""
app/agent/graph/sale_graph.py

SaleGraph — LangGraph cho Luồng B (Sale nội bộ).

Graph cực kỳ đơn giản — Sale hỏi gì AI trả nấy:

  START
    ↓
  [sale_sanitize]   ← sanitize input (không LLM call, không classify)
    ↓
  [sale_agent]      ← RAG + QA + LLM tool calling (toàn quyền)
    ↓
  [sale_synthesizer] ← tổng hợp data, format rõ ràng
    ↓
  END

Lợi ích so với CustomerGraph:
  - Không có classify_intent node (tiết kiệm 1 LLM call/request)
  - Không có project_guard node
  - Không có stage classifier
  → Giảm 2 LLM calls/request → tốc độ nhanh hơn + rẻ hơn
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.sale_agent_node import SaleAgentNode
from app.agent.nodes.sale_synthesizer_node import SaleSynthesizerNode
from app.agent.state.sale_state import SaleAgentState
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort
from app.shared.logging.logger import get_logger
from app.shared.security.guards import sanitize_input

log = get_logger(__name__)


async def _sanitize_node(state: SaleAgentState) -> SaleAgentState:
    """Sanitize input — không cần LLM, chỉ regex/rule-based."""
    raw = state.get("raw_query", "")
    clean, injected = sanitize_input(raw)
    state["raw_query"]     = clean
    state["was_injected"]  = injected
    if injected:
        log.warning("sale_injection_attempt", session=state.get("session_id"), prefix=raw[:80])
    return state


def build_sale_graph(
    llm: ChatPort,
    tool_registry: ToolRegistry,
    max_iterations: int = 6,
) -> "CompiledGraph":
    """
    Build và compile SaleGraph.

    Nodes: sanitize → sale_agent → sale_synthesizer → END
    Không có classify_intent hay project_guard.
    """
    sale_agent      = SaleAgentNode(tool_registry, llm)
    sale_synthesizer = SaleSynthesizerNode(llm)

    builder = StateGraph(SaleAgentState)
    builder.add_node("sanitize",          _sanitize_node)
    builder.add_node("sale_agent",        sale_agent)
    builder.add_node("sale_synthesizer",  sale_synthesizer)

    builder.add_edge(START,             "sanitize")
    builder.add_edge("sanitize",        "sale_agent")
    builder.add_edge("sale_agent",      "sale_synthesizer")
    builder.add_edge("sale_synthesizer", END)

    compiled = builder.compile()
    compiled.recursion_limit = max_iterations * 2
    log.info("sale_graph_compiled", max_iterations=max_iterations)
    return compiled
