"""
app/api/v1/endpoints/document.py

CB-01: Upload và quản lý tài liệu.

Endpoints:
  POST   /api/v1/documents/upload           — Upload tài liệu mới
  POST   /api/v1/documents/supersede/{code} — Upload thay thế doc cũ
  DELETE /api/v1/documents/{code}           — Xoá doc (admin only)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.api.middleware.auth import require_role
from app.application.usecases.upload_document import UploadDocumentRequest, UploadDocumentUseCase
from app.shared.errors.exceptions import (
    FileMagicMismatchError,
    FileTooLargeError,
    ParseError,
    UnsupportedFileTypeError,
)
from app.shared.logging.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents — CB-01"])


# ── Response schemas ──────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_code: str
    chunk_count: int
    superseded_count: int
    file_checksum: str
    message: str


# ── Dependency injection helper ───────────────────────────────────

def _get_use_case(request: Request) -> UploadDocumentUseCase:
    """Lấy use case đã được khởi tạo từ app state (set trong main.py)."""
    return request.app.state.upload_doc_uc


def _uploader(request: Request) -> str:
    principal = getattr(request.state, "principal", {})
    return principal.get("user_id", "anonymous")


# ── Endpoints ─────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload tài liệu mới vào knowledge base",
    description="""
Upload tài liệu (PDF / DOCX / Excel / ảnh) và index vào vector DB.

**Tài liệu thay thế tài liệu cũ:** truyền `supersedes_code` — hệ thống tự động
vô hiệu hoá các vector cũ (đánh dấu `superseded`) và index vector mới.
    """,
)
async def upload_document(
    request:         Request,
    file:            UploadFile = File(..., description="File tài liệu"),
    project_name:    str        = Form(..., description="Tên dự án, ví dụ: Vinhomes_GrandPark"),
    doc_group:       str        = Form(..., description="brochure | price_list | sales_policy | faq | legal | floor_plan | progress"),
    version:         str        = Form(..., description="Phiên bản, ví dụ: 1.0"),
    effective_date:  datetime   = Form(..., description="Ngày hiệu lực (ISO 8601)"),
    description:     Optional[str] = Form(None),
    supersedes_code: Optional[str] = Form(None, description="Mã doc cũ bị thay thế"),
    uc:              UploadDocumentUseCase = Depends(_get_use_case),
):
    file_bytes = await file.read()
    req = UploadDocumentRequest(
        file_bytes=file_bytes,
        file_name=file.filename or "unknown",
        project_name=project_name,
        doc_group=doc_group,
        version=version,
        effective_date=effective_date,
        uploaded_by=_uploader(request),
        description=description,
        supersedes_code=supersedes_code,
    )
    return await _run_upload(uc, req)


@router.post(
    "/supersede/{old_document_code}",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload tài liệu mới thay thế tài liệu cũ theo mã",
)
async def supersede_document(
    old_document_code: str,
    request:        Request,
    file:           UploadFile = File(...),
    project_name:   str        = Form(...),
    doc_group:      str        = Form(...),
    version:        str        = Form(...),
    effective_date: datetime   = Form(...),
    description:    Optional[str] = Form(None),
    uc:             UploadDocumentUseCase = Depends(_get_use_case),
):
    """Shortcut: supersedes_code = old_document_code từ path."""
    file_bytes = await file.read()
    req = UploadDocumentRequest(
        file_bytes=file_bytes,
        file_name=file.filename or "unknown",
        project_name=project_name,
        doc_group=doc_group,
        version=version,
        effective_date=effective_date,
        uploaded_by=_uploader(request),
        description=description,
        supersedes_code=old_document_code,
    )
    return await _run_upload(uc, req)


@router.delete(
    "/{document_code}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Xoá tài liệu và toàn bộ vector (admin only)",
)
async def delete_document(
    document_code: str,
    request: Request,
):
    require_role(request, "admin")
    vector_db = request.app.state.vector_db
    await vector_db.delete_by_document(document_code)
    log.info(
        "document_deleted",
        document_code=document_code,
        by=getattr(request.state, "principal", {}).get("user_id"),
    )


# ── Shared upload runner với error mapping ────────────────────────

async def _run_upload(
    uc: UploadDocumentUseCase,
    req: UploadDocumentRequest,
) -> UploadResponse:
    try:
        result = await uc.execute(req)
        return UploadResponse(
            document_code=result.document_code,
            chunk_count=result.chunk_count,
            superseded_count=result.superseded_count,
            file_checksum=result.file_checksum,
            message=result.message,
        )
    except (UnsupportedFileTypeError, FileTooLargeError, FileMagicMismatchError, ParseError) as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        log.error("upload_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi xử lý tài liệu. Vui lòng thử lại.",
        )