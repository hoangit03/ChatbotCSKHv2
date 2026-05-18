"""
app/agent/nodes/sale_agent_node.py

Node chính của SaleGraph — Data Assistant cho Sale nội bộ.

Thiết kế:
  - Toàn quyền: tất cả Sales API tools (bảng hàng, giá, tồn kho)
  - Native tool calling: LLM tự quyết định gọi tool nào
  - RAG + QA chạy AUTO song song trước (không expose cho LLM để tránh gọi 2 lần)
  - Prompt "Data Assistant": cung cấp dữ liệu thô, chính xác, không hạn chế
  - Không có stage, không có USP, không có price guard

Fix B1: generate_schemas() chỉ lấy SALES_TOOLS, không truyền rag_search/qa_lookup
         vào tool_names của LLM để tránh LLM gọi lại RAG/QA lần 2.
Fix B7: bỏ fallback ngẫu nhiên get_inventory khi LLM call fail — chỉ log error.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.agent.state.sale_state import SaleAgentState
from app.agent.tools.base_tool import ToolRegistry
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

# Tools mà LLM được phép gọi cho Sale (không bao gồm rag_search / qa_lookup
# vì chúng đã chạy auto trước — tránh double-call tốn token)
_SALE_LLM_TOOLS = {
    "list_projects",
    "check_availability",
    "get_inventory",
    "search_units",
    "register_consultation",
}

_SALE_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu dữ liệu bất động sản cho nhân viên Sale nội bộ.

Tài liệu chính sách / pháp lý / tiện ích đã được cung cấp sẵn trong CONTEXT (phần trên).
Chỉ gọi tool khi cần dữ liệu BẢNG HÀNG / TỒN KHO / GIÁ THỰC TẾ.

NGUYÊN TẮC CHỌN TOOL:
① Hỏi DANH SÁCH DỰ ÁN → `list_projects` (truyền status_filter nếu có).
② Hỏi TỒN KHO một dự án cụ thể → `get_inventory`.
③ Hỏi một CĂN CỤ THỂ theo mã căn → `check_availability`.
④ TÌM CĂN theo tiêu chí (phòng, giá, tầng, hướng) → `search_units`.
⑤ Thông tin pháp lý / tiện ích / chính sách → KHÔNG gọi tool (đã có trong CONTEXT).
⑥ Đăng ký tư vấn cho khách → `register_consultation`.
⑦ Hỏi nhiều mã căn → gọi `check_availability` cho từng căn riêng biệt.

QUY TẮC BẮT BUỘC:
- Cung cấp DỮ LIỆU ĐẦY ĐỦ: giá, diện tích, tầng, hướng, mã căn, chính sách.
- Căn đã bán (Sold): HIỂN THỊ ĐẦY ĐỦ thông tin kèm ghi chú "(Đã bán)".
- Nếu sale viết tắt tên dự án (VD: "metro", "diamond"), suy luận và gọi tool đúng.
- KHÔNG dùng câu "liên hệ bộ phận sale" vì người chat là Sale.
- Nếu thiếu dữ liệu: báo thẳng "Chưa có dữ liệu về [X] trong hệ thống".
"""


class SaleAgentNode:
    """
    Node duy nhất xử lý query của SaleGraph.

    Luồng:
      1. Pre-compute embedding (nếu chưa có)
      2. Chạy RAG + QA auto song song → nạp context tài liệu
      3. LLM tool calling với SALES_TOOLS only (không expose RAG/QA cho LLM)
      4. Chạy sales tool calls song song → nạp data bảng hàng
    """

    def __init__(self, registry: ToolRegistry, llm: ChatPort):
        self._registry = registry
        self._llm = llm

    async def __call__(self, state: SaleAgentState) -> SaleAgentState:
        state["iteration"] = state.get("iteration", 0) + 1

        if not isinstance(state.get("sales_data"), dict):
            state["sales_data"] = {}
        if not isinstance(state.get("tool_kwargs"), dict):
            state["tool_kwargs"] = {}

        # Prefill customer info
        if state.get("customer_name"):
            state["sales_data"].setdefault("customer_name", state["customer_name"])
        if state.get("customer_phone"):
            state["sales_data"].setdefault("customer_phone", state["customer_phone"])

        query = state["raw_query"]

        # ── 1. Pre-compute embedding ───────────────────────────────
        if not state.get("query_embedding"):
            rag_tool = self._registry.get("rag_search")
            if rag_tool and hasattr(rag_tool, "_embedder"):
                try:
                    state["query_embedding"] = await rag_tool._embedder.embed_one(query)
                except Exception as e:
                    log.warning("sale_embed_failed", error=str(e))

        # ── 2. RAG + QA auto (không expose cho LLM) ───────────────
        # Chạy để nạp context tài liệu vào state trước khi LLM tổng hợp
        await self._run_doc_tools(state)

        # ── 3. LLM tool calling — chỉ SALES tools ─────────────────
        # FIX B1: không truyền rag_search / qa_lookup vào đây
        tools_schemas = self._registry.generate_schemas(list(_SALE_LLM_TOOLS))
        if not tools_schemas:
            # Không có sales tool nào được cấu hình → skip LLM tool call
            log.warning("sale_no_tools_available", session=state.get("session_id"))
            return state

        try:
            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=query)],
                system=_SALE_SYSTEM_PROMPT,
                temperature=0.0,
                tools=tools_schemas,
            )
            tool_calls = resp.tool_calls or []

            if not tool_calls:
                # LLM quyết định không cần gọi tool (câu hỏi về chính sách, v.v.)
                log.info("sale_no_tool_call_needed", session=state.get("session_id"))
                return state

            # Defense: lọc tool không nằm trong whitelist
            tool_calls = [tc for tc in tool_calls if tc.get("name") in _SALE_LLM_TOOLS]

            # Chạy song song
            tasks = [
                self._run_tool_call(tc["name"], state, tc.get("arguments", {}))
                for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    log.error("sale_tool_exec_failed", error=str(res))
                elif res is not None:
                    state.setdefault("tool_calls", [])
                    state["tool_calls"].append(res)

        except Exception as e:
            # FIX B7: không fallback bằng get_inventory ngẫu nhiên
            # Chỉ log lỗi — synthesizer sẽ dùng RAG context có sẵn để trả lời
            log.error("sale_llm_tool_decision_failed", error=str(e))

        return state

    async def _run_doc_tools(self, state: SaleAgentState) -> None:
        """Chạy QA + RAG song song với return_exceptions để không crash toàn bộ."""
        qa_tool  = self._registry.get("qa_lookup")
        rag_tool = self._registry.get("rag_search")
        tasks = []
        if qa_tool:
            tasks.append(qa_tool.execute(state))
        if rag_tool:
            tasks.append(rag_tool.execute(state))
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                log.warning("sale_doc_tool_failed", error=str(item))
                continue
            _, call = item
            state.setdefault("tool_calls", [])
            state["tool_calls"].append(call)

    async def _run_tool_call(self, name: str, state: SaleAgentState, arguments: dict) -> Any:
        tool = self._registry.get(name)
        if not tool:
            log.warning("sale_tool_not_found", tool=name)
            return None
        _, call = await tool.execute(state, tool_kwargs=arguments)
        return call
