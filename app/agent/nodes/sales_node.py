"""
app/agent/nodes/sales_node.py

Node: Sales Inquiry & Booking.
Chiến lược: Dùng LLM Native Tool Calling để gọi tool sales phù hợp (Check Availability, Inventory, Booking).
Có xử lý Slot Filling trước khi gọi Booking API.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.agent.state.agent_state import AgentState, Intent
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

SALES_SYSTEM_PROMPT = """Bạn là chuyên viên tư vấn bất động sản.
Nhiệm vụ của bạn là phân tích câu hỏi của khách hàng và gọi công cụ (tool) phù hợp nhất để lấy dữ liệu.
Nếu khách hàng muốn tìm kiếm căn hộ, hãy gọi `search_units` với các tiêu chí tương ứng.
Nếu khách hàng hỏi về một căn cụ thể (vd: "căn A1-05"), hãy gọi `check_availability`.
Nếu khách hàng hỏi tổng quan tồn kho, gọi `get_inventory`.
Nếu khách hàng muốn biết chính sách thanh toán, vay, gọi `get_payment_policy`.
Nếu khách hàng có ý định cọc/mua, gọi `booking_intent`.

LƯU Ý QUAN TRỌNG:
- KHÔNG BAO GIỜ bịa ra thông tin. Chỉ trích xuất các tiêu chí có sẵn trong câu hỏi.
- Nếu câu hỏi không yêu cầu gọi công cụ bán hàng nào, bạn không cần gọi công cụ.
"""

SALES_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Kiểm tra căn hộ còn trống không. Dùng khi hỏi 'còn căn không', 'căn X còn chưa'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_code": {
                        "type": "string",
                        "description": "Mã căn hộ cần kiểm tra (ví dụ: A1-05, B02, C12-A)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory",
            "description": "Lấy tổng số căn hộ còn lại của dự án.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_units",
            "description": "Tìm căn hộ theo tiêu chí: số phòng ngủ, diện tích, giá tối đa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bedrooms": {"type": "integer", "description": "Số lượng phòng ngủ (vd: 1, 2, 3)"},
                    "min_price_vnd": {"type": "number", "description": "Giá tối thiểu bằng VNĐ (vd: 3000000000 cho 3 tỷ)"},
                    "max_price_vnd": {"type": "number", "description": "Giá tối đa bằng VNĐ (vd: 5000000000 cho 5 tỷ)"},
                    "min_area_m2": {"type": "number", "description": "Diện tích tối thiểu (m2)"},
                    "max_area_m2": {"type": "number", "description": "Diện tích tối đa (m2)"},
                    "floor": {"type": "string", "description": "Tầng của căn hộ (vd: 8, 12A, 15)"},
                    "direction": {"type": "string", "description": "Hướng của căn hộ (vd: Đông, Tây Nam)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_payment_policy",
            "description": "Lấy chính sách thanh toán, vay vốn, trả góp của dự án.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "booking_intent",
            "description": "Gửi yêu cầu đặt cọc/giữ chỗ khi khách hàng đã quyết định mua.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_code": {"type": "string", "description": "Mã căn hộ muốn đặt cọc"}
                }
            }
        }
    }
]

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
        if not isinstance(state.get("tool_kwargs"), dict):
            state["tool_kwargs"] = {}

        if state.get("customer_name"):
            state["sales_data"].setdefault("customer_name", state["customer_name"])
        if state.get("customer_phone"):
            state["sales_data"].setdefault("customer_phone", state["customer_phone"])

        # 1. Luôn chạy Q&A và RAG để lấy chính sách/thông tin chung
        await self._run_doc_tools(state)

        # 2. Dùng LLM Native Tool Calling để quyết định tool Sales
        try:
            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=query)],
                system=SALES_SYSTEM_PROMPT,
                temperature=0.0,
                tools=SALES_TOOLS_SCHEMA
            )
            
            tool_calls = resp.tool_calls or []
            
            # Fallback nếu intent là booking nhưng LLM không gọi tool
            if not tool_calls and intent == Intent.BOOKING_INTENT:
                tool_calls = [{"name": "booking_intent", "arguments": {}}]

            # Chạy tuần tự các tools được chọn để tránh race conditions (vd: availability rồi mới booking)
            for tc in tool_calls:
                tool_name = tc.get("name")
                args = tc.get("arguments", {})
                
                if tool_name == "booking_intent":
                    # SLOT FILLING: Kiểm tra xem đã đủ thông tin khách hàng và mã căn chưa
                    customer_phone = state.get("customer_phone") or state["sales_data"].get("customer_phone")
                    customer_name = state.get("customer_name") or state["sales_data"].get("customer_name")
                    unit_code = args.get("unit_code") or state["sales_data"].get("selected_unit_code")
                    
                    missing = []
                    if not customer_phone: missing.append("số điện thoại")
                    if not unit_code: missing.append("mã căn hộ chính xác")
                    
                    if missing:
                        # Gán final answer yêu cầu thông tin, bỏ qua việc gọi booking api
                        missing_str = " và ".join(missing)
                        state["final_answer"] = f"Dạ, để tiến hành giữ chỗ/đặt cọc, anh/chị vui lòng cung cấp thêm {missing_str} để em báo hệ thống nhé!"
                        state["sales_data"]["booking_pending"] = True
                        continue
                    else:
                        # Nếu đủ thông tin, lưu unit_code vào kwargs
                        state["tool_kwargs"][tool_name] = {"unit_code": unit_code}
                else:
                    # Lưu arguments vào state để tool đọc
                    state["tool_kwargs"][tool_name] = args

                await self._run_tool(tool_name, state)

        except Exception as e:
            log.error("sales_tool_decision_failed", error=str(e))
            # Nếu LLM call lỗi, chạy dự phòng
            if intent == Intent.BOOKING_INTENT:
                state["final_answer"] = "Dạ, anh/chị vui lòng cho em xin số điện thoại để chuyên viên hỗ trợ đặt cọc cho mình nhé."
            else:
                await self._run_tool("check_availability", state)
                await self._run_tool("get_inventory", state)

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