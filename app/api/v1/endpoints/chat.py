"""
app/api/v1/endpoints/chat.py

CB-02: Hỏi đáp qua Agent RAG.

Endpoint:
  POST /api/v1/chat  — gửi tin nhắn, nhận câu trả lời từ agent
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.application.usecases.handle_chat import ChatRequest, ChatResponse, HandleChatUseCase
from app.shared.logging.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat — CB-02"])


# ── Request / Response schemas (Pydantic → OpenAPI docs) ─────────

class ChatIn(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu hỏi hoặc tin nhắn của khách hàng",
        examples=["Căn hộ 2PN tại Vinhomes giá bao nhiêu?"],
    )
    session_id: Optional[str] = Field(
        None,
        description="ID phiên hội thoại. Bỏ trống để tạo session mới.",
    )
    project_name: Optional[str] = Field(
        None,
        description="Tên dự án để filter (ví dụ: Vinhomes_GrandPark). "
                    "Bỏ trống nếu muốn search toàn bộ dự án.",
    )
    customer_name: Optional[str] = Field(None, description="Tên khách hàng (dùng khi đặt cọc)")
    customer_phone: Optional[str] = Field(None, description="SĐT khách hàng (dùng khi đặt cọc)")


class SourceRefOut(BaseModel):
    document_code: str
    document_name: str
    doc_group: str
    excerpt: str
    page: Optional[int] = None


class ToolCallOut(BaseModel):
    tool_name: str
    input_summary: str
    output_summary: str
    duration_ms: int
    success: bool


class ChatOut(BaseModel):
    session_id: str
    answer: str
    intent: str
    sources: list[SourceRefOut] = []
    tool_calls: list[ToolCallOut] = []
    fallback: bool = False
    fallback_reason: str = ""
    was_injected: bool = False
    response_time_ms: int


# ── Endpoint ──────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatOut,
    summary="Gửi câu hỏi đến Agent RAG",
    description="""
Agent tự động phân loại intent và chọn luồng xử lý:

- **Customer Support**: Tìm trong tài liệu (PDF/DOCX/Excel đã upload) + Q&A chuẩn
- **Sales Inquiry**: Truy vấn real-time giá / tồn kho / chính sách từ Sales API backend
- **Booking Intent**: Trigger đặt cọc / giữ chỗ qua Sales API backend

Khi không đủ dữ liệu, `fallback=true` và `fallback_reason` giải thích lý do.
    """,
)
async def chat(body: ChatIn, request: Request) -> ChatOut:
    uc: HandleChatUseCase = request.app.state.handle_chat_uc

    chat_req = ChatRequest(
        message=body.message,
        session_id=body.session_id,
        project_name=body.project_name,
        customer_name=body.customer_name,
        customer_phone=body.customer_phone,
    )

    try:
        resp: ChatResponse = await uc.execute(chat_req)
    except Exception as e:
        log.error("chat_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau.",
        )

    return ChatOut(
        session_id=resp.session_id,
        answer=resp.answer,
        intent=str(resp.intent.value if hasattr(resp.intent, "value") else resp.intent),
        sources=[
            SourceRefOut(
                document_code=s.document_code,
                document_name=s.document_name,
                doc_group=s.doc_group,
                excerpt=s.excerpt,
                page=s.page,
            )
            for s in resp.sources
        ],
        tool_calls=[
            ToolCallOut(
                tool_name=t.tool_name,
                input_summary=t.input_summary,
                output_summary=t.output_summary,
                duration_ms=t.duration_ms,
                success=t.success,
            )
            for t in resp.tool_calls
        ],
        fallback=resp.fallback,
        fallback_reason=resp.fallback_reason,
        was_injected=resp.was_injected,
        response_time_ms=resp.response_time_ms,
    )