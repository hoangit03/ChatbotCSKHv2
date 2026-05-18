"""
app/agent/nodes/sale_synthesizer_node.py

Node tổng hợp câu trả lời cho SaleGraph (Luồng B — Sale nội bộ).

Khác biệt so với SynthesizerNode (Customer):
  - Không giới hạn độ dài (120 từ)
  - Không có CTA (mời đặt cọc, để lại số điện thoại)
  - Ưu tiên format bảng, bullet rõ ràng: giá / diện tích / tầng / hướng / mã căn
  - Hiển thị căn đã bán đầy đủ thông tin
  - Không có handover logic (Sale là người thật)
  - Thẳng thắn khi thiếu data: "Chưa có dữ liệu về X"
"""
from __future__ import annotations

import asyncio
import json
import time

from app.agent.state.sale_state import SaleAgentState
from app.agent.state.agent_state import ToolCall
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

# ── Prompt constants — tách rõ để tránh re.sub fragile (FIX B4) ──
_BASE_PROMPT = """Bạn là trợ lý AI tổng hợp dữ liệu bất động sản cho nhân viên Sale nội bộ.
Người dùng là Sale — KHÔNG PHẢI khách hàng.

NGUYÊN TẮC TUYỆT ĐỐI:
1. TOÀN QUYỀN THÔNG TIN: Hiển thị đầy đủ giá, diện tích, tầng, hướng, mã căn, chính sách.
2. KHÔNG GIỚI HẠN ĐỘ DÀI: Trình bày chi tiết, rõ ràng, chuyên nghiệp.
3. CĂN ĐÃ BÁN: Hiển thị toàn bộ thông tin + ghi chú "(Đã bán)". KHÔNG ẩn hay rút gọn.
4. KHÔNG CTA: Không mời đặt cọc, không kêu gọi "để lại số điện thoại". Chỉ cung cấp dữ liệu.
5. THIẾU DỮ LIỆU: Thông báo thẳng "Chưa có dữ liệu về [X] trong hệ thống."
6. KHÔNG dùng "liên hệ bộ phận sale" — người chat là Sale nội bộ.

FORMAT ƯU TIÊN:
- Bảng hoặc bullet có cấu trúc (Mã căn | Tầng | Hướng | Diện tích | Giá | Trạng thái).
- JSON/Dict từ API: format thành text dễ đọc, KHÔNG in raw JSON.
- Gợi ý 2 câu hỏi tiếp theo liên quan đến công việc tư vấn."""

_JSON_SUFFIX = """

ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC (JSON, không markdown):
{{
  "answer": "<nội dung trả lời>",
  "suggested_questions": ["<câu hỏi 1>", "<câu hỏi 2>"]
}}"""

_STREAM_SUFFIX = """

TRẢ LỜI bằng VĂN BẢN THUẦN TÚY. TUYỆT ĐỐI không trả về JSON."""

# Alias cho code ngoài import (FIX B4: không còn re.sub)
_SALE_SYSTEM_PROMPT_JSON   = _BASE_PROMPT + _JSON_SUFFIX
_SALE_SYSTEM_PROMPT_STREAM = _BASE_PROMPT + _STREAM_SUFFIX

_SYNTHESIS_TEMPLATE = """LỊCH SỬ HỘI THOẠI:
{history}

DỮ LIỆU TỪ HỆ THỐNG:
{context}

CÂU HỎI CỦA SALE:
---
{question}
---

Tổng hợp và trình bày dữ liệu đầy đủ cho Sale."""

_FALLBACK_MESSAGE = (
    "Hệ thống hiện không thể xử lý yêu cầu. "
    "Vui lòng kiểm tra lại kết nối API hoặc thử lại sau."
)


def _parse_output(content: str) -> tuple[str, list[str]]:
    try:
        data      = json.loads(content)
        answer    = data.get("answer", "").strip()
        suggested = data.get("suggested_questions", [])
        if isinstance(suggested, list):
            suggested = [str(q).strip() for q in suggested[:2] if q]
        else:
            suggested = []
        if answer:
            return answer, suggested
    except Exception as e:
        log.warning("sale_synthesizer_json_parse_failed", error=str(e), content=content[:100])
    return content.strip(), []


class SaleSynthesizerNode:

    def __init__(self, llm: ChatPort):
        self._llm = llm

    async def __call__(self, state: SaleAgentState) -> SaleAgentState:
        # Đã có final_answer → skip
        if state.get("final_answer"):
            if not state.get("suggested_questions"):
                state["suggested_questions"] = [
                    "Tìm căn khác theo tiêu chí?",
                    "Xem thêm thông tin dự án?",
                ]
            await self._push_to_queue_if_stream(state)
            return state
        ctx_str     = self._build_context(state)
        history_str = self._build_history(state)
        prompt      = _SYNTHESIS_TEMPLATE.format(
            history=history_str or "(Chưa có lịch sử)",
            context=ctx_str or "(Không có dữ liệu từ hệ thống)",
            question=state["raw_query"],
        )

        t0 = time.monotonic()
        try:
            if state.get("stream_queue"):
                await self._handle_stream(state, prompt)
            else:
                await self._handle_normal(state, prompt, t0, ctx_len=len(ctx_str))
        except Exception as e:
            log.error("sale_synthesizer_error", error=str(e))
            state["final_answer"]        = _FALLBACK_MESSAGE
            state["suggested_questions"] = []
            state["fallback"]            = True
            state["error"]               = str(e)
            if state.get("stream_queue"):
                try:
                    state["stream_queue"].put_nowait({"type": "token", "content": _FALLBACK_MESSAGE})
                except Exception:
                    pass

        return state

    async def _handle_normal(
        self,
        state: SaleAgentState,
        prompt: str,
        t0: float,
        ctx_len: int = 0,
    ) -> None:
        try:
            resp = await asyncio.wait_for(
                self._llm.chat(
                    messages=[LLMMessage(role="user", content=prompt)],
                    system=_SALE_SYSTEM_PROMPT_JSON,
                    response_format={"type": "json_object"},
                ),
                timeout=45.0,  # Sale cần thêm thời gian
            )
            answer, suggested = _parse_output(resp.content)
            state["final_answer"]        = answer
            state["suggested_questions"] = suggested

            duration_ms = int((time.monotonic() - t0) * 1000)
            # FIX B10: dùng ctx_len đã có, không gọi lại _build_context()
            state["tool_calls"] = state.get("tool_calls", []) + [ToolCall(
                tool_name="sale_llm_synthesizer",
                input_summary=f"ctx_len={ctx_len}, query={state['raw_query'][:60]!r}",
                output_summary=f"ans_len={len(answer)}, tok={resp.input_tokens}+{resp.output_tokens}",
                duration_ms=duration_ms,
                success=True,
            )]
            log.info("sale_synthesizer_done", session=state.get("session_id"), ms=duration_ms)

        except asyncio.TimeoutError:
            log.error("sale_synthesizer_timeout", session=state.get("session_id"))
            state["final_answer"]        = _FALLBACK_MESSAGE
            state["suggested_questions"] = []
            state["fallback"]            = True
            state["fallback_reason"]     = "LLM timeout"

    async def _handle_stream(self, state: SaleAgentState, prompt: str) -> None:
        queue  = state["stream_queue"]
        chunks = []
        try:
            async for token in self._llm.chat_stream(
                messages=[LLMMessage(role="user", content=prompt)],
                system=_SALE_SYSTEM_PROMPT_STREAM,
            ):
                chunks.append(token)
                await queue.put({"type": "token", "content": token})
            state["final_answer"]        = "".join(chunks)
            state["suggested_questions"] = await self._gen_suggestions(prompt)
            if state["suggested_questions"]:
                await queue.put({"type": "suggestions", "content": state["suggested_questions"]})
        except asyncio.TimeoutError:
            log.error("sale_synthesizer_stream_timeout")
            state["final_answer"] = _FALLBACK_MESSAGE
            state["fallback"]     = True

    async def _gen_suggestions(self, prompt: str) -> list[str]:
        try:
            sug_prompt = (
                prompt + "\n\nGợi ý 2 câu hỏi tiếp theo hữu ích cho Sale. "
                'JSON: {"suggested_questions": ["cau 1", "cau 2"]}'
            )
            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=sug_prompt)],
                system=_SALE_SYSTEM_PROMPT_JSON,
                response_format={"type": "json_object"},
            )
            _, sug = _parse_output(resp.content)
            return sug
        except Exception:
            return []

    async def _push_to_queue_if_stream(self, state: SaleAgentState) -> None:
        if state.get("stream_queue"):
            queue = state["stream_queue"]
            async def _push():
                await queue.put({"type": "token",       "content": state["final_answer"]})
                await queue.put({"type": "suggestions", "content": state.get("suggested_questions", [])})
                await queue.put({"type": "done"})
            asyncio.ensure_future(_push())

    def _build_history(self, state: SaleAgentState) -> str:
        messages = state.get("messages") or []
        MAX_CHARS = 4000
        lines, cur_len = [], 0
        for msg in reversed(messages):
            role    = "Sale" if msg.get("role") == "user" else "Bot"
            content = msg.get("content", "")
            if content:
                line = f"{role}: {content}"
                if cur_len + len(line) > MAX_CHARS:
                    break
                lines.insert(0, line)
                cur_len += len(line)
        return "\n".join(lines)

    def _build_context(self, state: SaleAgentState) -> str:
        parts: list[str] = []
        MAX_CHARS = 20_000

        sales = state.get("sales_data") or {}
        if sales:
            parts.append("=== DỮ LIỆU BẢNG HÀNG / TỒN KHO ===\n" +
                         json.dumps(sales, ensure_ascii=False, indent=2))

        rag = state.get("rag_results") or []
        if rag:
            qa_chunks  = [r for r in rag if r.get("source_type") == "qa"]
            doc_chunks = [r for r in rag if r.get("source_type") != "qa"]
            if qa_chunks:
                parts.append("=== Q&A CHUẨN ===\n" + "\n\n".join(r["text"] for r in qa_chunks))
            if doc_chunks:
                parts.append("=== TÀI LIỆU DỰ ÁN ===\n" + "\n\n".join(
                    f"[{r.get('document_name', '')} — {r.get('doc_group', '')}]\n{r['text']}"
                    for r in doc_chunks
                ))

        full = "\n\n".join(parts)
        if len(full) > MAX_CHARS:
            full = full[:MAX_CHARS] + "\n...[Nội dung đã rút gọn]"
        return full
