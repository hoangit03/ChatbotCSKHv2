"""
app/agent/nodes/synthesizer_node.py

Node cuối: dùng LLM tổng hợp câu trả lời từ context thu thập được.

RAG-style v4:
  - Q&A chunks (source_type="qa") được đặt đầu context với label ưu tiên cao
  - Document chunks theo sau để bổ sung thông tin chi tiết
  - LLM nhận combined context → câu trả lời tự nhiên, chính xác
"""
from __future__ import annotations

import json

from app.agent.state.agent_state import AgentState, Intent, ToolCall
from app.core.interfaces.llm_port import ChatPort, LLMMessage
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

# ── System prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp của công ty bất động sản. 
Nhiệm vụ của bạn là tư vấn cho khách hàng về dự án: {project_name}.

NGUYÊN TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên dữ liệu context của dự án {project_name} được cung cấp — KHÔNG bịa thêm hoặc lấy thông tin dự án khác.
2. Luôn giữ thái độ lịch sự, chuyên nghiệp và thân thiện.
3. Khi context có "Bộ Q&A chuẩn", hãy ưu tiên dùng thông tin đó làm nền tảng trả lời.
4. Nếu khách hỏi về dự án khác dự án đang chọn, hãy nhẹ nhàng nhắc khách và đề nghị chuyển sang bối cảnh dự án mới.
5. Nếu không có thông tin trong context -> nói thẳng bạn chưa có thông tin cụ thể về vấn đề này của dự án {project_name} và đề nghị khách để lại liên hệ.
8. Trả lời bằng ngôn ngữ của khách hàng (Việt hoặc Anh).
"""

SYNTHESIS_TEMPLATE = """CONTEXT:
{context}

CÂU HỎI CỦA KHÁCH: {question}

Hãy trả lời dựa trên context trên. Nếu context không đủ thông tin, hãy nói rõ."""

FALLBACK_MESSAGE = (
    "Xin lỗi, tôi chưa tìm thấy thông tin chính xác cho câu hỏi này. "
    "Để được hỗ trợ tốt nhất, bạn vui lòng:\n"
    "• Liên hệ trực tiếp với bộ phận Sales của chúng tôi\n"
    "• Hoặc để lại số điện thoại, chúng tôi sẽ gọi lại trong vòng 30 phút."
)


class SynthesizerNode:

    def __init__(self, llm: ChatPort):
        self._llm = llm

    async def __call__(self, state: AgentState) -> AgentState:
        # Đã có final_answer từ trước (ví dụ: intent=booking đã xử lý xong)
        if state.get("final_answer"):
            log.info("synthesizer_skip_already_answered", session=state.get("session_id"))
            return state

        context = self._build_context(state)
        if not context.strip():
            state["final_answer"] = FALLBACK_MESSAGE
            state["fallback"] = True
            log.info("synthesizer_fallback_no_context", session=state.get("session_id"))
            return state

        import time
        t0 = time.monotonic()
        prompt = SYNTHESIS_TEMPLATE.format(
            context=context,
            question=state["raw_query"],
        )
        try:
            # Format system prompt với tên dự án thực tế
            project_name = state.get("project_name", "dự án")
            system_msg = SYSTEM_PROMPT.format(project_name=project_name)

            resp = await self._llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                system=system_msg,
            )

            state["final_answer"] = resp.content

            duration_ms = int((time.monotonic() - t0) * 1000)
            call = ToolCall(
                tool_name="llm_synthesizer",
                input_summary=f"context_len={len(context)}, qa_hit={state.get('qa_hit', False)}, query={state['raw_query'][:60]!r}",
                output_summary=f"answer_len={len(resp.content)}, tokens={resp.input_tokens}+{resp.output_tokens}",
                duration_ms=duration_ms,
                success=True,
            )
            state["tool_calls"] = state.get("tool_calls", []) + [call]
            log.info(
                "synthesizer_done",
                session=state.get("session_id"),
                duration_ms=duration_ms,
                qa_hit=state.get("qa_hit", False),
                provider=self._llm.provider_name,
            )

        except Exception as e:
            log.error("synthesizer_llm_error", error=str(e))
            state["final_answer"] = FALLBACK_MESSAGE
            state["fallback"] = True
            state["error"] = str(e)

        return state

    def _build_context(self, state: AgentState) -> str:
        """
        Xây dựng context cho LLM từ tất cả nguồn.
        Thứ tự ưu tiên:
          1. Q&A chunks (source_type="qa") — câu trả lời chuẩn
          2. Document chunks — thông tin chi tiết từ tài liệu
          3. Sales API data — dữ liệu real-time
        """
        parts: list[str] = []

        rag = state.get("rag_results", [])
        if rag:
            # Tách Q&A chunks và document chunks
            qa_chunks  = [r for r in rag if r.get("source_type") == "qa"]
            doc_chunks = [r for r in rag if r.get("source_type") != "qa"]

            # Q&A chunks — đặt đầu tiên với label ưu tiên
            if qa_chunks:
                qa_text = "\n\n".join(r["text"] for r in qa_chunks)
                parts.append(f"=== CÂU TRẢ LỜI CHUẨN (Q&A) ===\n{qa_text}")

            # Document chunks — bổ sung thông tin chi tiết
            if doc_chunks:
                doc_text = "\n\n".join(
                    f"[Tài liệu: {r.get('document_name', '')} — {r.get('doc_group', '')}]\n{r['text']}"
                    for r in doc_chunks
                )
                parts.append(f"=== TÀI LIỆU DỰ ÁN ===\n{doc_text}")

        # Sales API context
        sales = state.get("sales_data", {})
        if sales:
            sales_text = json.dumps(sales, ensure_ascii=False, indent=2)
            parts.append(f"=== DỮ LIỆU TỪ HỆ THỐNG BÁN HÀNG ===\n{sales_text}")

        return "\n\n".join(parts)