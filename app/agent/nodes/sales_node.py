"""
app/agent/nodes/sales_node.py

Node: Sales Inquiry & Booking.
Gọi Sales API backend, không dùng RAG cho dữ liệu động.
"""
from __future__ import annotations

from app.agent.state.agent_state import AgentState, Intent
from app.agent.tools.base_tool import ToolRegistry
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

_PRICE_KW = ("giá", "price", "bao nhiêu", "chi phí", "tổng tiền")
_INVENTORY_KW = ("còn hàng", "tồn kho", "available", "còn căn", "còn bao nhiêu")
_PAYMENT_KW = ("thanh toán", "vay", "trả góp", "payment", "ngân hàng", "lãi suất")
_SEARCH_KW = ("tìm căn", "muốn căn", "cần căn", "phòng ngủ", "diện tích", "tìm hộ")


class SalesNode:

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1
        query = state["raw_query"].lower()
        intent = state.get("intent", Intent.SALES_INQUIRY)

        # Booking → chạy booking tool
        if intent == Intent.BOOKING_INTENT:
            await self._run_tool("booking_intent", state)
            return state

        # Chạy các tool phù hợp theo keyword
        ran_any = False

        if any(kw in query for kw in _INVENTORY_KW) or any(kw in query for kw in _PRICE_KW):
            await self._run_tool("check_availability", state)
            ran_any = True

        if any(kw in query for kw in _PRICE_KW):
            # Inventory đã có giá, nhưng cũng lấy tổng tồn kho
            await self._run_tool("get_inventory", state)
            ran_any = True

        if any(kw in query for kw in _PAYMENT_KW):
            await self._run_tool("get_payment_policy", state)
            ran_any = True

        if any(kw in query for kw in _SEARCH_KW):
            await self._run_tool("search_units", state)
            ran_any = True

        if not ran_any:
            # Fallback: lấy tổng quan tồn kho
            await self._run_tool("get_inventory", state)
            await self._run_tool("check_availability", state)

        return state

    async def _run_tool(self, name: str, state: AgentState) -> None:
        tool = self._registry.get(name)
        if not tool:
            log.warning("tool_not_found", name=name)
            return
        result, call = await tool.execute(state)
        state["tool_calls"] = state.get("tool_calls", []) + [call]
        if not result.success:
            state["fallback"] = True
            state["fallback_reason"] = result.summary