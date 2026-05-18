"""
app/agent/graph/agent_graph.py

CustomerGraph — LangGraph cho Luồng A (Khách hàng).

Graph structure:
  START
    ↓
  [classify_intent]   ← sanitize + phân loại intent
    ↓
  [customer_guard]    ← chặn price query, hỏi project nếu chưa chọn
    ↓ (conditional)
    ├── "end"          → END       (guard đã trả lời — block hoặc hỏi project)
    ├── "synthesizer"  → [synth]   (chitchat — không cần RAG)
    └── "support_node" → [support] → [customer_node] → [synthesizer] → END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.intent_classifier import classify_intent, route_by_intent
from app.agent.nodes.support_node import SupportNode
from app.agent.nodes.sales_node import CustomerSalesNode
from app.agent.nodes.synthesizer_node import SynthesizerNode
from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


def build_customer_graph(
    llm: ChatPort,
    tool_registry: ToolRegistry,
    max_iterations: int = 8,
) -> "CompiledGraph":
    """
    Build và compile CustomerGraph.

    Nodes:
      classify_intent → customer_guard → support_node → customer_node → synthesizer → END
    """
    from app.agent.nodes.project_guard import project_guard_node

    support_node  = SupportNode(tool_registry)
    customer_node = CustomerSalesNode(tool_registry, llm)
    synthesizer   = SynthesizerNode(llm)

    async def wrapped_classifier(state: AgentState) -> AgentState:
        return await classify_intent(state, llm, tool_registry)

    async def wrapped_guard(state: AgentState) -> AgentState:
        return await project_guard_node(state, tool_registry, llm)

    def route_after_guard(state: AgentState) -> str:
        if state.get("final_answer"):
            return "end"           # Guard đã trả lời (block / hỏi project)
        return route_by_intent(state)  # → "synthesizer" hoặc "support_node"

    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", wrapped_classifier)
    builder.add_node("customer_guard",  wrapped_guard)
    builder.add_node("support_node",    support_node)
    builder.add_node("customer_node",   customer_node)
    builder.add_node("synthesizer",     synthesizer)

    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "customer_guard")

    builder.add_conditional_edges(
        "customer_guard",
        route_after_guard,
        {
            "support_node": "support_node",
            "synthesizer":  "synthesizer",
            "end":          END,
        },
    )
    # support_node  : RAG + QA (tài liệu)
    # customer_node : list_projects / register_consultation (Sales API - whitelist)
    builder.add_edge("support_node",  "customer_node")
    builder.add_edge("customer_node", "synthesizer")
    builder.add_edge("synthesizer",   END)

    compiled = builder.compile()
    compiled.recursion_limit = max_iterations * 2
    log.info("customer_graph_compiled", max_iterations=max_iterations)
    return compiled


# Backward-compat alias
build_agent_graph = build_customer_graph
