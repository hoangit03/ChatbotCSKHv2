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
SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh tên là 'chatbot CTlotus' của công ty bất động sản. 
Nhiệm vụ của bạn là hỗ trợ khách hàng tìm hiểu về dự án: {project_name}, hỗ trợ bán hàng và giải đáp thắc mắc.

PHONG CÁCH GIAO TIẾP:
1. Giao tiếp tự nhiên, thân thiện và linh hoạt. Tuyệt đối không trả lời máy móc hoặc dùng mãi một câu kết thúc rập khuôn.
2. Nếu là câu hỏi xã giao (chào hỏi, hỏi tên, toán học cơ bản), hãy trả lời vui vẻ, ngắn gọn và mời khách hỏi về dự án nếu cần.
3. Luôn giữ thái độ lịch sự nhưng gần gũi, như một chuyên viên tư vấn thực thụ.

NGUYÊN TẮC NỘI DUNG:
1. Đối với thông tin dự án {project_name}: Hãy sử dụng dữ liệu CONTEXT được cung cấp để trả lời. Nếu dữ liệu có phần trùng khớp (dù ít), hãy cố gắng suy luận để trả lời tự nhiên nhất thay vì từ chối.
2. NẾU KHÁCH CHỈ CHÀO HOẶC QUAN TÂM CHUNG CHUNG (ví dụ: "chào bạn", "tôi quan tâm dự án", "dự án này ở đâu"): Hãy chào mừng, giới thiệu ngắn gọn và HỎI XEM khách cần tìm hiểu cụ thể về mảng nào (pháp lý, giá bán, tiến độ...). TUYỆT ĐỐI KHÔNG báo lỗi "chưa có thông tin" trong trường hợp này.
3. CHỈ KHI khách hỏi một CÂU HỎI CHI TIẾT (ví dụ: "giá bao nhiêu", "tiến độ đến đâu") mà trong CONTEXT hoàn toàn KHÔNG CÓ dữ liệu, bạn mới từ chối khéo léo và mời khách để lại SĐT để chuyên viên hỗ trợ.
4. Luôn ưu tiên "Bộ Q&A chuẩn" nếu có thông tin khớp.
5. Trả lời bằng đúng ngôn ngữ của khách hàng.

NGUYÊN TẮC BÁN HÀNG (SALES):
1. KHI CÓ DỮ LIỆU TỪ HỆ THỐNG BÁN HÀNG (Inventory/Search/Availability): Hãy thể hiện vai trò là người tư vấn. Nếu có thông tin `total_price`, `maintenance_fee`, hãy báo giá chi tiết và minh bạch.
2. NẾU CÓ CHƯƠNG TRÌNH KHUYẾN MÃI (`sale_program`), BẮT BUỘC phải nhắc nhở khách hàng để tạo cảm giác cấp bách (FOMO).
3. NẾU CĂN TRỐNG (Available): Luôn kết thúc bằng việc hỏi khách hàng có muốn đặt cọc / giữ chỗ (booking) căn này không để không bị bỏ lỡ.
4. NẾU CĂN ĐÃ BÁN (Reserved/Sold/Hợp đồng): Phải tỏ ra tiếc nuối, sau đó CHỦ ĐỘNG đề xuất tìm kiếm một căn hộ khác có thông số tương đương (Cross-sell).

TRÁNH RẬP KHUÔN:
- TUYỆT ĐỐI KHÔNG dùng mãi câu "Anh/chị cần thêm thông tin gì về dự án, tôi rất sẵn lòng hỗ trợ!".
- Hãy đa dạng hóa lời chào và lời kết để tạo cảm giác thoải mái, thân thiện.
"""

SYNTHESIS_TEMPLATE = """LỊCH SỬ HỘI THOẠI:
{history}

CONTEXT DỮ LIỆU:
{context}

CÂU HỎI MỚI NHẤT CỦA KHÁCH: {question}

Hãy trả lời câu hỏi mới nhất dựa trên context và lịch sử hội thoại trên. Nếu context không đủ thông tin, hãy nói rõ."""

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
        
        # Nếu là Chitchat hoặc câu hỏi chung, không cần context gắt gao
        is_chitchat = state.get("intent") == Intent.CHITCHAT
        
        import asyncio
        import time
        t0 = time.monotonic()
        
        # Format lịch sử hội thoại dựa trên độ dài ký tự
        history_str = ""
        messages = state.get("messages") or []
        MAX_HISTORY_CHARS = 1500  # Giảm xuống ~400 tokens để phù hợp LLM limit
        current_len = 0
        history_lines = []
        
        for msg in reversed(messages):
            role = "Khách" if msg.get("role") == "user" else "Bot"
            content = msg.get("content", "")
            if content:
                line = f"{role}: {content}"
                if current_len + len(line) > MAX_HISTORY_CHARS:
                    break
                history_lines.insert(0, line)
                current_len += len(line)
                
        history_str = "\n".join(history_lines)
            
        prompt = SYNTHESIS_TEMPLATE.format(
            history=history_str or "(Chưa có hội thoại trước đó)",
            context=context,
            question=state["raw_query"],
        )
        try:
            # Format system prompt với tên dự án thực tế
            project_name = state.get("project_name", "dự án")
            system_msg = SYSTEM_PROMPT.format(project_name=project_name)
            
            # Nếu dự án vừa được xác nhận mới, thêm yêu cầu chào mừng vào prompt
            if state.get("project_newly_confirmed"):
                system_msg += (
                    f"\nLƯU Ý QUAN TRỌNG: Khách hàng vừa nhắc đến dự án {project_name}. "
                    f"Hãy bắt đầu câu trả lời bằng một câu chào thân thiện như: "
                    f"'Cảm ơn bạn đã quan tâm về dự án {project_name}, chúng tôi có thể giúp gì cho bạn' "
                    f"hoặc tương tự, sau đó mới trả lời câu hỏi của khách."
                )

            try:
                resp = await asyncio.wait_for(
                    self._llm.chat(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system=system_msg,
                    ),
                    timeout=30.0,   # 30s hard timeout — tránh hang vô hạn
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

            except asyncio.TimeoutError:
                log.error("synthesizer_llm_timeout", session=state.get("session_id"))
                state["final_answer"] = FALLBACK_MESSAGE
                state["fallback"] = True
                state["fallback_reason"] = "LLM timeout"
        except Exception as e:
            log.error("synthesizer_llm_error", error=str(e))
            state["final_answer"] = FALLBACK_MESSAGE
            state["fallback"] = True
            state["error"] = str(e)

        return state

    def _build_context(self, state: AgentState) -> str:
        """
        Xây dựng context cho LLM từ tất cả nguồn.
        Thứ tự ưu tiên MỚI:
          1. Sales API data — dữ liệu real-time (Quan trọng nhất để chốt sale)
          2. Q&A chunks (source_type="qa") — câu trả lời chuẩn
          3. Document chunks — thông tin chi tiết từ tài liệu
        """
        parts: list[str] = []
        MAX_CONTEXT_CHARS = 6000  # Giới hạn ~1500 tokens để tổng prompt (Context + History) luôn nằm an toàn dưới 2048 tokens.

        # 1. Ưu tiên cao nhất: Sales API context
        sales = state.get("sales_data", {})
        if sales:
            sales_text = json.dumps(sales, ensure_ascii=False, indent=2)
            parts.append(f"=== DỮ LIỆU TỪ HỆ THỐNG BÁN HÀNG ===\n{sales_text}")

        # 2 & 3. RAG Results
        rag = state.get("rag_results", [])
        if rag:
            # Tách Q&A chunks và document chunks
            qa_chunks  = [r for r in rag if r.get("source_type") == "qa"]
            doc_chunks = [r for r in rag if r.get("source_type") != "qa"]

            # Q&A chunks
            if qa_chunks:
                qa_text = "\n\n".join(r["text"] for r in qa_chunks)
                parts.append(f"=== CÂU TRẢ LỜI CHUẨN (Q&A) ===\n{qa_text}")

            # Document chunks
            if doc_chunks:
                doc_text = "\n\n".join(
                    f"[Tài liệu: {r.get('document_name', '')} — {r.get('doc_group', '')}]\n{r['text']}"
                    for r in doc_chunks
                )
                parts.append(f"=== TÀI LIỆU DỰ ÁN ===\n{doc_text}")

        full_context = "\n\n".join(parts)
        if len(full_context) > MAX_CONTEXT_CHARS:
            full_context = full_context[:MAX_CONTEXT_CHARS] + "\n...[Nội dung đã được rút gọn để tránh quá tải]"
            
        return full_context