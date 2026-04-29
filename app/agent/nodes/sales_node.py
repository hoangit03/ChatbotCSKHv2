"""
app/agent/nodes/sales_node.py

Node: Sales Inquiry & Booking.
Chiến lược: Dùng LLM để quyết định gọi tool sales phù hợp (Check Availability, Inventory, Booking).
"""
from __future__ import annotations

import asyncio
import json
from app.agent.state.agent_state import AgentState, Intent
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

TOOL_DECISION_PROMPT = """Bạn là trợ lý điều phối công cụ Sales bất động sản.
Câu hỏi của khách hàng: {query}
Bạn có các công cụ sau:
- "check_availability": Gọi khi khách hỏi tình trạng trống của một căn CỤ THỂ (vd: "căn A102 còn không?").
- "get_inventory": Gọi khi khách hỏi thông tin chung về tồn kho (tổng số căn, bao nhiêu căn trống).
- "search_units": Gọi khi khách muốn tìm kiếm danh sách các căn theo các tiêu chí (vd: tầng 8, 2 phòng ngủ, hướng đông, dưới 3 tỷ, ...).
- "get_payment_policy": Gọi khi khách hỏi về chính sách thanh toán, vay vốn, trả góp.
- "booking_intent": Gọi khi khách có ý định đặt cọc, giữ chỗ, muốn mua.

Dựa vào câu hỏi, hãy quyết định xem cần gọi công cụ nào (có thể chọn nhiều). 
Bạn PHẢI trả về duy nhất một mảng JSON các tên công cụ cần gọi (ví dụ: ["search_units", "get_inventory"]). 
Trả về [] nếu không cần công cụ nào. Không kèm text hay markdown nào khác.
"""

class SalesNode:
    def __init__(self, registry: ToolRegistry, llm: ChatPort):
        self._registry = registry
        self._llm = llm

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1
        query = state["raw_query"]
        intent = state.get("intent", Intent.SALES_INQUIRY)

        if not isinstance(state.get("sales_data"), dict):
            state["sales_data"] = {}

        if state.get("customer_name"):
            state["sales_data"].setdefault("customer_name", state["customer_name"])
        if state.get("customer_phone"):
            state["sales_data"].setdefault("customer_phone", state["customer_phone"])

        # 1. Luôn chạy Q&A và RAG để lấy chính sách/thông tin chung
        await self._run_doc_tools(state)

        # 2. Dùng LLM để quyết định tool Sales
        system_msg = TOOL_DECISION_PROMPT.format(query=query)
        tools_to_run = []
        try:
            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=query)],
                system=system_msg,
                temperature=0.0
            )
            content = resp.content.strip()
            import re
            
            # Remove markdown
            clean_content = re.sub(r'^```(?:json)?\n', '', content)
            clean_content = re.sub(r'\n```$', '', clean_content)
            clean_content = clean_content.strip()

            match = re.search(r'\[.*\]', clean_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                tools_to_run = json.loads(json_str)
            else:
                tools_to_run = []
                
            if not isinstance(tools_to_run, list):
                tools_to_run = []
                
            if not tools_to_run:
                # Fallback string matching
                cl = content.lower()
                if "check_availability" in cl:
                    tools_to_run.append("check_availability")
                if "get_inventory" in cl:
                    tools_to_run.append("get_inventory")
                if "booking_intent" in cl:
                    tools_to_run.append("booking_intent")
                if "search_units" in cl:
                    tools_to_run.append("search_units")
                if "get_payment_policy" in cl:
                    tools_to_run.append("get_payment_policy")
                    
        except Exception as e:
            log.error("sales_tool_decision_failed", error=str(e))
            # Fallback behavior
            if intent == Intent.BOOKING_INTENT:
                tools_to_run = ["booking_intent"]
            else:
                tools_to_run = ["check_availability", "get_inventory"]

        # Chạy các tools được chọn song song
        sales_tasks = []
        for tool_name in tools_to_run:
            if isinstance(tool_name, str):
                sales_tasks.append(self._run_tool(tool_name.strip(), state))
                
        if sales_tasks:
            await asyncio.gather(*sales_tasks)

        if not state.get("rag_results") and not state.get("sales_data"):
            log.info("sales_node_no_info", query=query)

        return state

    async def _run_doc_tools(self, state: AgentState) -> None:
        qa_tool = self._registry.get("qa_lookup")
        rag_tool = self._registry.get("rag_search")

        # Pre-compute query embedding
        if not state.get("query_embedding") and rag_tool:
            try:
                state["query_embedding"] = await rag_tool._embed.embed_one(state["raw_query"])
            except Exception as e:
                log.error("sales_node_embed_failed", error=str(e))

        tasks = []
        if qa_tool:
            tasks.append(qa_tool.execute(state))
        if rag_tool:
            tasks.append(rag_tool.execute(state))

        if tasks:
            results = await asyncio.gather(*tasks)
            for _, call in results:
                if "tool_calls" not in state or state["tool_calls"] is None:
                    state["tool_calls"] = []
                state["tool_calls"].append(call)

    async def _run_tool(self, name: str, state: AgentState) -> None:
        tool = self._registry.get(name)
        if not tool:
            return
        result, call = await tool.execute(state)
        if "tool_calls" not in state or state["tool_calls"] is None:
            state["tool_calls"] = []
        state["tool_calls"].append(call)