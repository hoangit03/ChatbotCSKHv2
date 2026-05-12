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

NGUYÊN TẮC CHỌN TOOL:
① Khách hỏi DANH SÁCH DỰ ÁN (liệt kê, kể tên, có những dự án nào...): gọi `list_projects`.
   • Khách hỏi "đang mở bán", "đang bán", "hiện có", "đang kinh doanh": truyền status_filter="dang_mo_ban".
   • Khách hỏi "sắp mở bán", "sắp ra mắt", "sắp tới": truyền status_filter="sap_mo_ban".
   • Khách hỏi TẤT CẢ không phân biệt trạng thái: KHÔNG truyền status_filter.

② Khách hỏi TỒN KHO của MỘT DỰ ÁN CỤ THỂ (còn bao nhiêu căn, còn trống không):
   → gọi `get_inventory`. TUYỆT ĐỐI KHÔNG gọi `list_projects` khi khách đang hỏi dự án cụ thể.

③ Khách hỏi MỘT CĂN CỤ THỂ theo mã căn (VD: T1-A14-03, căn góc tầng 5): gọi `check_availability`.

④ Khách TÌM CĂN theo tiêu chí (số phòng, giá tối đa, tầng, hướng...): gọi `search_units`.

⑤ Khách hỏi thông tin chung (tiện ích, pháp lý, vị trí...) mà KHÔNG cần số liệu real-time: KHÔNG gọi tool.
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

# System prompt riêng cho Sale nội bộ — không có price guard hay stage restriction.
# Sale cần thấy đầy đủ thông tin để tư vấn khách hàng chính xác.
_SALE_EXTRA_NOTE = """
BẠN ĐANG TRẢ LỜI CHO NHÂN VIÊN SALE NỘI BỘ (không phải khách hàng).
Yêu cầu: Cung cấp DỮ LIỆU ĐẦY ĐỦ, chính xác và trực tiếp. Không che giấu giá, không hạn chế thông tin.
• Có thể tra cứu nhiều căn cùng lúc bằng nhiều tool call song song.
• Báo giá chi tiết, số phòng, diện tích, tầng, hướng, chương trình ưu đãi.
• Nếu sale hỏi nhiều mã căn (VD: A101 và B202), hãy gọi `check_availability` cho từng căn riêng biệt.
"""

class SalesNode:
    def __init__(self, registry: ToolRegistry, llm: ChatPort):
        self._registry = registry
        self._llm = llm

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1
        query = state["raw_query"]
        intent = state.get("intent", Intent.SALES_INQUIRY)
        stage  = CustomerStage(state.get("customer_stage", CustomerStage.AWARENESS))
        user_type = state.get("user_type", "customer")  # "customer" | "sale"

        if not isinstance(state.get("sales_data"), dict):
            state["sales_data"] = {}
        if not isinstance(state.get("tool_kwargs"), dict):
            state["tool_kwargs"] = {}

        if state.get("customer_name"):
            state["sales_data"].setdefault("customer_name", state["customer_name"])
        if state.get("customer_phone"):
            state["sales_data"].setdefault("customer_phone", state["customer_phone"])

        # ── 1. Inject USP theo stage vào context (chỉ luồng khách hàng) ─────────────────
        # Sale nội bộ không cần USP — họ cần raw data để tư vấn chính xác.
        if user_type == "customer":
            self._inject_usps(state, stage)

        # ── 2. Precompute query embedding (reuse giữa QATool và RAGTool) ─────────────────────────
        # Chỉ precompute khi cần doc tools (không phải Consultation)
        if intent != Intent.CONSULTATION_INTENT and not state.get("query_embedding"):
            try:
                rag_tool = self._registry.get("rag_search")
                if rag_tool and hasattr(rag_tool, "_embedder"):
                    state["query_embedding"] = await rag_tool._embedder.embed_one(query)
            except Exception as emb_err:
                log.debug("sales_node_preembed_skip", reason=str(emb_err))

        # ── 3. RAG + QA — Bỏ qua nếu là Consultation Intent ─────────────────────────
        if intent != Intent.CONSULTATION_INTENT:
            await self._run_doc_tools(state)

        # ── 4. Comparison Intent: chỉ upgrade stage cho khách hàng ──────────────
        # Sale hỏi so sánh → chỉ lấy data, không cần stage/USP manipulation.
        if intent == Intent.COMPARISON_INTENT and user_type == "customer":
            state["customer_stage"] = CustomerStage.DECISION.value
            stage = CustomerStage.DECISION
            self._inject_usps(state, stage)
            log.info("comparison_intent_stage_upgraded", session=state.get("session_id"))


        # ── 5. Gọi LLM để quyết định gọi tool hay trả lời trực tiếp ──
        stage_guidance = _STAGE_TOOL_GUIDANCE.get(stage, "")

        # Xây dựng danh sách tools phù hợp theo user_type
        # Khách hàng (Luồng A): chỉ có register_consultation và list_projects
        # Sale (Luồng B): đầy đủ tools bao gồm cả bảng hàng
        if user_type == "customer":
            allowed_tools = ["register_consultation", "list_projects"]
            tools_schemas = self._registry.generate_schemas(allowed_tools)
            sale_only_note = (
                "\nLƯU Ý: Đây là kênh tư vấn khách hàng. TUYỆT ĐỐI KHÔNG gọi các tool: "
                "check_availability, get_inventory, search_units. "
                "Nếu khách hỏi giá hoặc tồn kho cụ thể, mời đăng ký tư vấn để được hỗ trợ trực tiếp."
            )
            system_prompt = _BASE_SALES_PROMPT + stage_guidance + sale_only_note
        else:
            # Sale nội bộ: đầy đủ tools, không có stage restriction về giá, có thêm note sale
            tools_schemas = self._registry.generate_schemas()
            system_prompt = _BASE_SALES_PROMPT + _SALE_EXTRA_NOTE
 
        try:
            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=query)],
                system=system_prompt,
                temperature=0.0,
                tools=tools_schemas,
            )
 
            tool_calls = resp.tool_calls or []
 
            # Fallback nếu intent là tư vấn nhưng LLM không gọi tool
            if not tool_calls and intent == Intent.CONSULTATION_INTENT:
                tool_calls = [{"name": "register_consultation", "arguments": {}}]

            # Price Guard: chỉ áp dụng cho KHÁCH HÀNG ở giai đoạn AWARENESS
            # Sale nội bộ luôn thấy giá đầy đủ để tư vấn chính xác.
            if stage == CustomerStage.AWARENESS and user_type == "customer":
                state["sales_data"]["price_disclosure_blocked"] = True

            # Bảo vệ thêm: lọc bỏ tool call trái phép cho customer (defense-in-depth)
            if user_type == "customer":
                restricted = {"check_availability", "get_inventory", "search_units"}
                tool_calls = [tc for tc in tool_calls if tc.get("name") not in restricted]
                if not tool_calls and intent == Intent.CONSULTATION_INTENT:
                    tool_calls = [{"name": "register_consultation", "arguments": {}}]

            # Chạy song song — tối ưu tốc độ
            tasks = []
            for tc in tool_calls:
                tool_name = tc.get("name")
                args = tc.get("arguments", {})
                tasks.append(self._run_tool_returning_call(tool_name, state, args))
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    log.error("sales_tool_parallel_execution_failed", error=str(res))
                elif res is not None:
                    if "tool_calls" not in state or state["tool_calls"] is None:
                        state["tool_calls"] = []
                    state["tool_calls"].append(res)
 
        except Exception as e:
            log.error("sales_tool_decision_failed", error=str(e))
            if intent == Intent.CONSULTATION_INTENT:
                state["final_answer"] = (
                    "Dạ, anh/chị vui lòng cho em xin số điện thoại "
                    "để chuyên viên hỗ trợ tư vấn cho mình nhé."
                )
            elif user_type == "sale":  # chỉ fallback inventory cho sale
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
        qa_tool  = self._registry.get("qa_lookup")
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

    async def _run_tool_returning_call(self, name: str, state: AgentState, arguments: dict = None) -> Any:
        tool = self._registry.get(name)
        if not tool:
            return None
        result, call = await tool.execute(state, tool_kwargs=arguments)
        return call

    async def _run_tool(self, name: str, state: AgentState, arguments: dict = None) -> None:
        call = await self._run_tool_returning_call(name, state, arguments)
        if call is not None:
            if "tool_calls" not in state or state["tool_calls"] is None:
                state["tool_calls"] = []
            state["tool_calls"].append(call)