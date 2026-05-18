"""
app/agent/nodes/sales_node.py

Node: Customer Support Node (Luồng A — Khách hàng).

Token savings: không inject USP context (~200-500 token/request).
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.agent.state.agent_state import AgentState, Intent
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

# ── System prompt cho Customer — ngắn gọn, không có stage/USP ────────
_CUSTOMER_TOOL_PROMPT = """Bạn là trợ lý AI tư vấn bất động sản phục vụ khách hàng.
Nhiệm vụ: phân tích câu hỏi và gọi tool phù hợp để lấy dữ liệu.

NGUYÊN TẮC CHỌN TOOL:
① Khách hỏi DANH SÁCH DỰ ÁN (kể tên, có những dự án nào, đang mở bán):
   → Gọi `list_projects`. Truyền status_filter nếu khách đề cập trạng thái.
② Khách muốn ĐĂNG KÝ TƯ VẤN, gặp sale, xem nhà mẫu, để lại thông tin:
   → Gọi `register_consultation`.
③ Khách hỏi thông tin chung (tiện ích, pháp lý, vị trí, chính sách):
   → KHÔNG gọi tool. Dùng tài liệu đã có trong context.

LƯU Ý QUAN TRỌNG:
- Đây là kênh khách hàng. TUYỆT ĐỐI KHÔNG gọi: check_availability, get_inventory, search_units.
- Nếu khách hỏi giá hoặc tồn kho căn cụ thể: KHÔNG gọi tool, trả lời rỗng để Synthesizer xử lý.
"""


class CustomerSalesNode:
    """
    Node xử lý Sales/Consultation intent cho Khách hàng.

    Tools được phép: list_projects, register_consultation.
    RAG + QA chạy song song trước để có context đầy đủ.
    """

    _ALLOWED_TOOLS = {"list_projects", "register_consultation"}

    def __init__(self, registry: ToolRegistry, llm: ChatPort):
        self._registry = registry
        self._llm = llm

    async def __call__(self, state: AgentState) -> AgentState:
        state["iteration"] = state.get("iteration", 0) + 1

        if not isinstance(state.get("sales_data"), dict):
            state["sales_data"] = {}
        if not isinstance(state.get("tool_kwargs"), dict):
            state["tool_kwargs"] = {}

        # Prefill customer info nếu có
        if state.get("customer_name"):
            state["sales_data"].setdefault("customer_name", state["customer_name"])
        if state.get("customer_phone"):
            state["sales_data"].setdefault("customer_phone", state["customer_phone"])

        intent = state.get("intent", Intent.SALES_INQUIRY)
        query  = state["raw_query"]

        # ── 1. Pre-compute embedding cho RAG/QA ───────────────────
        if intent != Intent.CONSULTATION_INTENT and not state.get("query_embedding"):
            rag_tool = self._registry.get("rag_search")
            if rag_tool and hasattr(rag_tool, "_embedder"):
                try:
                    state["query_embedding"] = await rag_tool._embedder.embed_one(query)
                except Exception:
                    pass

        # ── 2. RAG + QA (bỏ qua nếu là Consultation) ─────────────
        if intent != Intent.CONSULTATION_INTENT:
            await self._run_doc_tools(state)

        # ── 3. LLM tool calling (list_projects / register_consultation) ──
        tools_schemas = self._registry.generate_schemas(list(self._ALLOWED_TOOLS))
        try:
            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=query)],
                system=_CUSTOMER_TOOL_PROMPT,
                temperature=0.0,
                tools=tools_schemas,
            )
            tool_calls = resp.tool_calls or []

            # Fallback: Consultation intent không gọi tool → ép gọi
            if not tool_calls and intent == Intent.CONSULTATION_INTENT:
                tool_calls = [{"name": "register_consultation", "arguments": {}}]

            # Defense-in-depth: lọc tool không nằm trong whitelist
            tool_calls = [tc for tc in tool_calls if tc.get("name") in self._ALLOWED_TOOLS]

            # Chạy song song
            tasks = [
                self._run_tool_returning_call(tc["name"], state, tc.get("arguments", {}))
                for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    log.error("customer_tool_exec_failed", error=str(res))
                elif res is not None:
                    state.setdefault("tool_calls", [])
                    state["tool_calls"].append(res)

        except Exception as e:
            log.error("customer_tool_decision_failed", error=str(e))
            if intent == Intent.CONSULTATION_INTENT:
                state["final_answer"] = (
                    "Dạ, anh/chị vui lòng cho em xin số điện thoại "
                    "để chuyên viên hỗ trợ tư vấn nhé."
                )

        return state

    async def _run_doc_tools(self, state: AgentState) -> None:
        """Chạy QA + RAG song song."""
        qa_tool  = self._registry.get("qa_lookup")
        rag_tool = self._registry.get("rag_search")
        tasks = []
        if qa_tool:
            tasks.append(qa_tool.execute(state))
        if rag_tool:
            tasks.append(rag_tool.execute(state))
        if tasks:
            # FIX B2: return_exceptions=True — nếu 1 tool fail, tool kia vẫn chạy
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for item in results:
                if isinstance(item, Exception):
                    log.warning("customer_doc_tool_failed", error=str(item))
                    continue
                _, call = item
                state.setdefault("tool_calls", [])
                state["tool_calls"].append(call)

    async def _run_tool_returning_call(self, name: str, state: AgentState, arguments: dict) -> Any:
        tool = self._registry.get(name)
        if not tool:
            return None
        _, call = await tool.execute(state, tool_kwargs=arguments)
        return call