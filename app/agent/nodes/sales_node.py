"""
app/agent/nodes/sales_node.py

Node: Sales Inquiry & Booking.
Chiến lược: Dùng LLM Native Tool Calling để gọi tool sales phù hợp (Check Availability, Inventory, Booking).
Có xử lý Slot Filling trước khi gọi Booking API.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.agent.sales.usp_registry import usp_registry
from app.agent.state.agent_state import AgentState, CustomerStage, Intent, ScarcityLevel
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

_BASE_SALES_PROMPT = """Bạn là chuyên viên tư vấn bất động sản.
Nhiệm vụ của bạn là phân tích câu hỏi của khách hàng và gọi công cụ (tool) phù hợp nhất để lấy dữ liệu.
 
LƯU Ý QUAN TRỌNG VỀ TOOL:
- Hỏi danh sách dự án / có những dự án nào: gọi `list_projects`.
- Hỏi dự án (cụ thể) còn căn trống không / có bao nhiêu căn: gọi `get_inventory`. TUYỆT ĐỐI KHÔNG gọi `list_projects` nếu khách đang hỏi về một dự án cụ thể.
- Hỏi một căn cụ thể (VD: căn góc, mã căn T1-04): gọi `check_availability`.
- Tìm căn theo tiêu chí (giá, số phòng): gọi `search_units`.
- Nếu không cần truy vấn số liệu bán hàng, không gọi tool.
"""
 
_STAGE_TOOL_GUIDANCE = {
    CustomerStage.AWARENESS: """
GIAI ĐOẠN KHÁCH HÀNG: Giai đoạn 1 — Mới Quan Tâm.
CHIẾN LƯỢC: Mục tiêu là thu thập thông tin khách hàng để tư vấn (Consultation).
- Nếu khách hỏi về dự án, tiện ích, vị trí: dùng kiến thức đã có.
- Nếu khách hỏi muốn xem nhà, muốn liên hệ sale: gọi `register_consultation`.
- Nếu khách hỏi tổng quan tồn kho: gọi `get_inventory`.
- TUYỆT ĐỐI KHÔNG báo giá chi tiết ở giai đoạn này. Ưu tiên mời đăng ký tư vấn.
""",
    CustomerStage.CONSIDERATION: """
GIAI ĐOẠN KHÁCH HÀNG: Giai đoạn 2 — Đang Đánh Giá.
CHIẾN LƯỢC: Chứng minh đẳng cấp và quy mô dự án. 
- Nếu khách muốn liên hệ tư vấn: gọi `register_consultation`.
- Nếu hỏi căn cụ thể/view/hướng: gọi `check_availability` hoặc `search_units`.
- Nếu hỏi tổng quan: gọi `get_inventory`.
- Có thể cung cấp thông tin giá ở giai đoạn này.
""",
    CustomerStage.DECISION: """
GIAI ĐOẠN KHÁCH HÀNG: Giai đoạn 3 — Sắp Chốt Deal.
CHIẾN LƯỢC: Xóa bỏ rủi ro, tạo sức ép khan hiếm.
- Nếu khách muốn chốt, muốn liên hệ sale ngay: gọi `register_consultation`.
- Nếu hỏi căn cụ thể: gọi `check_availability`.
- Nếu tìm theo tiêu chí: gọi `search_units`.
- Cung cấp đầy đủ thông tin giá, pháp lý.
""",
}


# SALES_TOOLS_SCHEMA — giờ được auto-generate từ ToolRegistry.generate_schemas()
# Xem base_tool.py:ToolRegistry.generate_schemas() và sales_tool.py:tool_schema property

class SalesNode:
    def __init__(self, registry: ToolRegistry, llm: ChatPort):
        self._registry = registry
        self._llm = llm

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1
        query = state["raw_query"]
        intent = state.get("intent", Intent.SALES_INQUIRY)
        stage  = CustomerStage(state.get("customer_stage", CustomerStage.AWARENESS))

        if not isinstance(state.get("sales_data"), dict):
            state["sales_data"] = {}
        if not isinstance(state.get("tool_kwargs"), dict):
            state["tool_kwargs"] = {}

        if state.get("customer_name"):
            state["sales_data"].setdefault("customer_name", state["customer_name"])
        if state.get("customer_phone"):
            state["sales_data"].setdefault("customer_phone", state["customer_phone"])

        # ── 1. Inject USP theo stage vào context ─────────────────
        self._inject_usps(state, stage)

        # ── 2. RAG + QA luôn chạy để lấy thông tin chung ─────────
        await self._run_doc_tools(state)

        # ── 3. Xử lý đặc biệt cho Comparison Intent ──────────────
        # Comparison → force stage lên DECISION, load thêm USP pháp lý
        if intent == Intent.COMPARISON_INTENT:
            state["customer_stage"] = CustomerStage.DECISION
            stage = CustomerStage.DECISION
            self._inject_usps(state, stage)  # Re-inject với stage mới
            log.info("comparison_intent_stage_upgraded", session=state.get("session_id"))
 
        # ── 4. (Đã gỡ bỏ: Xử lý đặc biệt cho Consultation Intent vì làm mất arguments) ─────────────

        # ── 5. Gọi LLM để quyết định gọi tool hay trả lời trực tiếp ──
        stage_guidance = _STAGE_TOOL_GUIDANCE.get(stage, "")
        system_prompt = _BASE_SALES_PROMPT + stage_guidance
 
        try:
            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=query)],
                system=system_prompt,
                temperature=0.0,
                tools=self._registry.generate_schemas(),
            )
 
            tool_calls = resp.tool_calls or []
 
            # Fallback nếu intent là tư vấn nhưng LLM không gọi tool
            if not tool_calls and intent == Intent.CONSULTATION_INTENT:
                tool_calls = [{"name": "register_consultation", "arguments": {}}]

            new_tool_calls = []
            
            # [NEW] Price Guard: Giai đoạn 1 không tiết lộ giá chi tiết
            if stage == CustomerStage.AWARENESS:
                state["sales_data"]["price_disclosure_blocked"] = True

            # Chạy tuần tự — tránh race condition
            for tc in tool_calls:
                tool_name = tc.get("name")
                args = tc.get("arguments", {})
                
                state["tool_kwargs"][tool_name] = args
                await self._run_tool(tool_name, state)
 
        except Exception as e:
            log.error("sales_tool_decision_failed", error=str(e))
            if intent == Intent.BOOKING_INTENT:
                state["final_answer"] = (
                    "Dạ, anh/chị vui lòng cho em xin số điện thoại "
                    "để chuyên viên hỗ trợ đặt cọc cho mình nhé."
                )
            elif intent == Intent.APPOINTMENT_INTENT:
                await self._run_tool("book_appointment", state)
            else:
                await self._run_tool("get_inventory", state)
 
        return state

    def _inject_usps(self, state: AgentState, stage: CustomerStage) -> None:
        """
        [NEW] Lấy USP phù hợp với stage và inject vào sales_data.
        Synthesizer sẽ đọc key "usp_context" này để inject vào prompt.
        """
        project = state.get("project_name", "")
        if not project:
            return
 
        used_ids: list[str] = state.get("usps_used", [])
 
        usps = usp_registry.get_usps_for_stage(
            project_name=project,
            stage=stage,
            used_ids=used_ids,
            max_usps=3,
        )
 
        if not usps:
            return
 
        usp_context = usp_registry.format_usps_for_context(usps)
        state["sales_data"]["usp_context"] = usp_context
        state["sales_data"]["usp_stage"]   = stage.value
 
        # Đánh dấu USP đã dùng để tránh lặp trong session
        new_used = used_ids + [u.usp_id for u in usps if u.usp_id not in used_ids]
        state["usps_used"] = new_used
 
        log.info(
            "usps_injected",
            stage=stage.value,
            usp_ids=[u.usp_id for u in usps],
            session=state.get("session_id"),
        )
 

    async def _run_doc_tools(self, state: AgentState) -> None:
        qa_tool = self._registry.get("qa_lookup")
        rag_tool = self._registry.get("rag_search")

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