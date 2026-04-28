"""
app/agent/nodes/sales_node.py

Node: Sales Inquiry & Booking.
Chiến lược: Kết hợp dữ liệu động từ Sales API + dữ liệu tĩnh từ RAG/Q&A.
"""
from __future__ import annotations

from app.agent.state.agent_state import AgentState, Intent
from app.agent.tools.base_tool import ToolRegistry
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

_PAYMENT_KW = ("thanh toán", "vay", "trả góp", "lãi suất", "ngân hàng", "ân hạn")

class SalesNode:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1
        query = state["raw_query"].lower()
        intent = state.get("intent", Intent.SALES_INQUIRY)

        # ── 1. LUÔN LUÔN tìm trong Q&A và RAG trước ─────────────
        # Vì thông tin về chính sách vay, bảng giá, pháp lý... thường nằm trong tài liệu
        await self._run_doc_tools(state)

        # ── 2. Xử lý Booking ─────────────────────────────────────
        if intent == Intent.BOOKING_INTENT:
            await self._run_tool("booking_intent", state)
            return state

        # ── 3. Gọi Sales API cho dữ liệu real-time (tồn kho, giá căn cụ thể) ──
        # Ví dụ: nếu hỏi "còn căn nào", "giá căn A102"
        if any(kw in query for kw in ["còn căn", "tồn kho", "mã căn", "vị trí căn"]):
            await self._run_tool("check_availability", state)
            await self._run_tool("get_inventory", state)

        # ── 4. Xử lý Fallback context ────────────────────────────
        if not state.get("rag_results") and not state.get("sales_data"):
            log.info("sales_node_no_info", query=query)
            # Không cần set fallback=True ngay vì Synthesizer sẽ lo

        return state

    async def _run_doc_tools(self, state: AgentState) -> None:
        """Chạy Q&A và RAG để lấy context từ tài liệu đã upload."""
        # Q&A Lookup
        qa_tool = self._registry.get("qa_lookup")
        if qa_tool:
            result, call = await qa_tool.execute(state)
            state["tool_calls"] = state.get("tool_calls", []) + [call]
            
            # Nếu Q&A hit với điểm cực cao, có thể return sớm để tiết kiệm RAG
            if result.success and state.get("qa_hit"):
                if state["qa_result"].score >= 0.92:
                    return

        # RAG Search (Documents)
        rag_tool = self._registry.get("rag_search")
        if rag_tool:
            result, call = await rag_tool.execute(state)
            state["tool_calls"] = state.get("tool_calls", []) + [call]

    async def _run_tool(self, name: str, state: AgentState) -> None:
        tool = self._registry.get(name)
        if not tool:
            return
        result, call = await tool.execute(state)
        state["tool_calls"] = state.get("tool_calls", []) + [call]