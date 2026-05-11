"""
app/api/v1/endpoints/sale_chat.py

CB-02 — Luồng B (Sale nội bộ): Chat endpoint với xác thực X-API-Key.

Khác biệt so với /api/v1/chat (Luồng A — Khách hàng):
  - Yêu cầu header X-API-Key (được kiểm soát bởi APIKeyMiddleware)
  - Truyền user_type="sale" → mở khóa toàn bộ tools (bảng hàng, giá, tồn kho)
  - Role được set qua X-Role-Level header (>1 = sale, mặc định = 2)
  - Hỗ trợ cả regular và stream endpoint (giống /api/v1/chat)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi import Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.application.usecases.handle_chat import ChatRequest, ChatResponse, HandleChatUseCase
from app.shared.logging.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/sale/chat", tags=["Sale Chat — CB-02 (Luồng B)"])


# ── Request / Response schemas ─────────────────────────────────────

class SaleChatIn(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu hỏi hoặc tin nhắn của sale",
        examples=["Dự án Prime Diamond còn bao nhiêu căn 2PN tầng cao?"],
    )
    session_id: Optional[str] = Field(
        None,
        description="ID phiên hội thoại. Bỏ trống để tạo session mới.",
    )
    project_name: Optional[str] = Field(
        None,
        description="Tên dự án để filter. Bỏ trống để search toàn bộ.",
    )
    # Thông tin khách hàng — sale có thể truyền sẵn để prefill
    customer_name: Optional[str] = Field(None, description="Tên khách hàng")
    customer_phone: Optional[str] = Field(None, description="SĐT khách hàng")


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


class SaleChatOut(BaseModel):
    session_id: str
    answer: str
    intent: str
    sources: list[SourceRefOut] = []
    tool_calls: list[ToolCallOut] = []
    fallback: bool = False
    fallback_reason: str = ""
    was_injected: bool = False
    project_name: Optional[str] = None
    response_time_ms: int
    suggested_questions: list[str] = []
    sales_data: dict = {}


# ── Endpoints ──────────────────────────────────────────────────────

@router.post(
    "",
    response_model=SaleChatOut,
    summary="[Sale] Gửi câu hỏi đến Agent RAG với quyền đầy đủ",
    description="""
Endpoint dành riêng cho **Sale nội bộ** — yêu cầu `X-API-Key` hợp lệ.

Khác biệt so với endpoint khách hàng (`/api/v1/chat`):
- Truy cập đầy đủ vào **bảng hàng real-time**: giá, tồn kho, tầng, hướng
- Có thể tra cứu **căn cụ thể** theo mã căn
- Toàn bộ USP injection theo CustomerStage
- Không bị chặn Price Guard
    """,
)
async def sale_chat(
    body: SaleChatIn,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None),
    x_role_level: Optional[str] = Header("2"),   # Default 2 = sale
    x_session_id: Optional[str] = Header(None),
    x_project_name: Optional[str] = Header(None),
) -> SaleChatOut:
    uc: HandleChatUseCase = request.app.state.handle_chat_uc

    # Ưu tiên session_id từ Header (do Gateway proxy xuống)
    final_session_id = x_session_id or body.session_id

    # Đảm bảo role_level >= 2 để luôn là "sale"
    role_level = x_role_level if x_role_level else "2"
    if role_level.isdigit() and int(role_level) < 2:
        role_level = "2"

    final_project_name = x_project_name or body.project_name

    chat_req = ChatRequest(
        message=body.message,
        session_id=final_session_id,
        project_name=final_project_name,
        customer_name=body.customer_name,
        customer_phone=body.customer_phone,
        user_id=x_user_id,
        tenant_id=x_tenant_id,
        role_level=role_level,  # >= 2 → user_type = "sale"
    )

    try:
        resp: ChatResponse = await uc.execute(chat_req)
    except Exception as e:
        log.error("sale_chat_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau.",
        )

    return SaleChatOut(
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
        project_name=resp.project_name,
        response_time_ms=resp.response_time_ms,
        suggested_questions=resp.suggested_questions,
        sales_data=resp.sales_data,
    )


@router.post(
    "/stream",
    summary="[Sale] Gửi câu hỏi và stream câu trả lời theo SSE",
    description=(
        "Giống `/sale/chat` nhưng trả về từng token qua SSE. "
        "Yêu cầu `X-API-Key` hợp lệ. Phù hợp cho UI sale dashboard."
    ),
)
async def sale_chat_stream(
    body: SaleChatIn,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None),
    x_role_level: Optional[str] = Header("2"),
    x_session_id: Optional[str] = Header(None),
    x_project_name: Optional[str] = Header(None),
):
    uc: HandleChatUseCase = request.app.state.handle_chat_uc

    final_session_id = x_session_id or body.session_id

    role_level = x_role_level if x_role_level else "2"
    if role_level.isdigit() and int(role_level) < 2:
        role_level = "2"

    final_project_name = x_project_name or body.project_name

    chat_req = ChatRequest(
        message=body.message,
        session_id=final_session_id,
        project_name=final_project_name,
        customer_name=body.customer_name,
        customer_phone=body.customer_phone,
        user_id=x_user_id,
        tenant_id=x_tenant_id,
        role_level=role_level,
    )

    return StreamingResponse(
        uc.execute_stream(chat_req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering
        },
    )
