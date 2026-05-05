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
 
LƯU Ý QUAN TRỌNG:
- KHÔNG BAO GIỜ bịa ra thông tin. Chỉ trích xuất các tiêu chí có sẵn trong câu hỏi.
- Nếu khách hỏi hiện có bao nhiêu dự án, hoặc danh sách dự án, hãy gọi tool `list_projects`.
- Nếu câu hỏi không yêu cầu gọi công cụ bán hàng nào, bạn không cần gọi công cụ.
"""
 
_STAGE_TOOL_GUIDANCE = {
    CustomerStage.AWARENESS: """
GIAI ĐOẠN KHÁCH HÀNG: Giai đoạn 1 — Mới Quan Tâm.
CHIẾN LƯỢC: Mục tiêu là chốt lịch hẹn, KHÔNG bán nhà qua điện thoại.
- Nếu khách hỏi xem nhà, muốn đến xem: gọi `book_appointment`.
- Nếu khách hỏi tổng quan tồn kho: gọi `get_inventory`.
- TUYỆT ĐỐI KHÔNG gọi `check_availability` với giá chi tiết ở giai đoạn này.
- Nếu khách hỏi giá: trả lời chung chung, ưu tiên mời đến xem sa bàn.
""",
    CustomerStage.CONSIDERATION: """
GIAI ĐOẠN KHÁCH HÀNG: Giai đoạn 2 — Đang Đánh Giá.
CHIẾN LƯỢC: Đẩy cảm xúc, chứng minh đẳng cấp và quy mô dự án.
- Nếu hỏi căn cụ thể/view/hướng: gọi `check_availability` hoặc `search_units`.
- Nếu hỏi tổng quan: gọi `get_inventory`.
- Có thể cung cấp thông tin giá ở giai đoạn này.
""",
    CustomerStage.DECISION: """
GIAI ĐOẠN KHÁCH HÀNG: Giai đoạn 3 — Sắp Chốt Deal.
CHIẾN LƯỢC: Xóa bỏ rủi ro, cung cấp số liệu pháp lý, tạo sức ép khan hiếm.
- Nếu hỏi căn cụ thể: gọi `check_availability`.
- Nếu tìm theo tiêu chí: gọi `search_units`.
- Nếu hỏi thanh toán/vay: gọi `get_payment_policy`.
- Nếu muốn đặt cọc: gọi `booking_intent`.
- Cung cấp đầy đủ thông tin giá, pháp lý, bảo lãnh.
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
 
        # ── 4. Xử lý đặc biệt cho Appointment Intent ─────────────
        if intent == Intent.APPOINTMENT_INTENT:
            await self._run_tool("book_appointment", state)
            return state

        # ── 5. LLM Native Tool Calling với stage-aware prompt ─────
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
 
            # Fallback nếu intent là booking nhưng LLM không gọi tool
            if not tool_calls and intent == Intent.BOOKING_INTENT:
                tool_calls = [{"name": "booking_intent", "arguments": {}}]
 
            # Chạy tuần tự — tránh race condition
            for tc in tool_calls:
                tool_name = tc.get("name")
                args = tc.get("arguments", {})
 
                # ── [NEW] Price Guard: Giai đoạn 1 không tiết lộ giá ──
                if stage == CustomerStage.AWARENESS and tool_name == "check_availability":
                    log.info(
                        "price_guard_blocked",
                        stage=stage.value,
                        tool=tool_name,
                        session=state.get("session_id"),
                    )
                    # Chặn tiết lộ giá chi tiết căn hộ
                    state["sales_data"]["price_disclosure_blocked"] = True
                    # Chuyển sang get_inventory thay thế (không có giá chi tiết)
                    tool_name = "get_inventory"
                    args = {}
 
                if tool_name == "booking_intent":
                    result = self._handle_booking_slot_filling(state, args)
                    if result:  # Có final_answer → thiếu thông tin hoặc cần confirm
                        state["final_answer"] = result
                        continue
                    state["tool_kwargs"][tool_name] = args
                else:
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
 
    def _handle_booking_slot_filling(
        self,
        state: AgentState,
        args: dict,
    ) -> str | None:
        """
        [NEW] Multi-turn slot filling cho booking.
        Trả về final_answer nếu cần thêm thông tin hoặc cần xác nhận.
        Trả về None nếu đã đủ → cho phép tiếp tục gọi tool.
        """
        sales_data     = state.get("sales_data", {})
        customer_phone = state.get("customer_phone") or sales_data.get("customer_phone")
        customer_name  = state.get("customer_name")  or sales_data.get("customer_name")
        unit_code      = args.get("unit_code") or sales_data.get("selected_unit_code")
 
        missing = []
        if not customer_name:
            missing.append("họ và tên")
        if not customer_phone:
            missing.append("số điện thoại")
        if not unit_code:
            missing.append("mã căn hộ chính xác")
 
        if missing:
            missing_str = " và ".join(missing)
            state["sales_data"]["booking_pending"] = True
            state["sales_data"]["booking_missing_fields"] = missing
            return (
                f"Dạ, để tiến hành giữ chỗ/đặt cọc, "
                f"anh/chị vui lòng cung cấp thêm {missing_str} để em báo hệ thống nhé!"
            )
 
        # [NEW] Confirmation step — lần đầu hỏi booking
        if not state.get("booking_confirmation"):
            state["sales_data"]["booking_pending"] = True
            state["sales_data"]["booking_confirm_required"] = {
                "unit_code":      unit_code,
                "customer_name":  customer_name,
                "customer_phone": customer_phone,
            }
            # Lưu unit_code để turn tiếp theo dùng
            state["tool_kwargs"]["booking_intent"] = {"unit_code": unit_code}
            return (
                f"Dạ, em xin xác nhận lại thông tin:\n"
                f"• Căn hộ: **{unit_code}**\n"
                f"• Họ tên: **{customer_name}**\n"
                f"• SĐT: **{customer_phone}**\n\n"
                f"Anh/chị xác nhận đặt cọc giữ chỗ căn này nhé? "
                f"(Trả lời 'xác nhận' hoặc 'đồng ý' để hoàn tất)"
            )
 
        # Đã confirm → cho phép gọi tool
        state["tool_kwargs"]["booking_intent"] = {"unit_code": unit_code}
        return None

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