"""
app/api/v1/endpoints/qa_import.py

Quản lý bộ Q&A chuẩn (import từ Excel, list, deactivate).

Endpoints:
  POST   /api/v1/qa/import    — Upload file Excel Q&A (async, bulk embed)
  GET    /api/v1/qa           — List Q&A đang active theo project (scroll API)
  DELETE /api/v1/qa/{qa_id}  — Vô hiệu hoá Q&A (soft-delete)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.api.middleware.auth import require_role
from app.application.usecases.import_qa import ImportQARequest, ImportQAUseCase
from app.shared.logging.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/qa", tags=["Q&A Management"])


# ── Response schemas ──────────────────────────────────────────────

class ImportQAResponse(BaseModel):
    total_rows: int
    imported: int
    skipped: int
    errors: list[str] = []


class QAItemOut(BaseModel):
    id: str
    project_name: str
    question: str
    answer: str
    keywords: list[str] = []
    doc_group: Optional[str] = None
    is_active: bool


# ── Endpoints ─────────────────────────────────────────────────────

@router.post(
    "/import",
    response_model=ImportQAResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import bộ Q&A chuẩn từ file Excel",
    description="""
File Excel cần có các cột (không phân biệt hoa/thường, hỗ trợ cả tiếng Việt):

| Question / Câu hỏi | Answer / Câu trả lời | Keywords (tuỳ chọn) | DocGroup (tuỳ chọn) |
|---|---|---|---|

**Lưu ý:**
- Import nhiều file khác nhau, hoặc re-import cùng file → **không bị duplicate** (idempotent)
- Q&A được embed và lưu vào Qdrant collection `qa_pairs` riêng biệt
- Hỗ trợ import số lượng lớn (bulk embedding, 1 API call/batch)
    """,
)
async def import_qa(
    request:      Request,
    file:         UploadFile = File(..., description="File Excel (.xlsx hoặc .xls)"),
    project_name: str        = Form(..., description="Tên dự án"),
    sheet_name:   Optional[str] = Form(None, description="Tên sheet (mặc định: sheet đầu tiên)"),
):
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Chỉ hỗ trợ file Excel (.xlsx, .xls)",
        )

    file_bytes = await file.read()
    uc: ImportQAUseCase = request.app.state.import_qa_uc

    import_req = ImportQARequest(
        file_bytes=file_bytes,
        file_name=file.filename,
        project_name=project_name,
        sheet_name=sheet_name,
    )

    try:
        result = await uc.execute(import_req)   # async — bulk embed + upsert Qdrant
    except Exception as e:
        log.error("qa_import_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý file Excel: {e}",
        )

    return ImportQAResponse(
        total_rows=result.total_rows,
        imported=result.imported,
        skipped=result.skipped,
        errors=result.errors,
    )


@router.get(
    "",
    response_model=list[QAItemOut],
    summary="Danh sách Q&A đang active theo dự án",
    description="Dùng Qdrant scroll API — không cần vector search, hỗ trợ dataset lớn.",
)
async def list_qa(project_name: str, request: Request) -> list[QAItemOut]:
    qa_store = request.app.state.qa_store
    items = await qa_store.list_by_project(project_name)   # async scroll
    return [
        QAItemOut(
            id=i.id,
            project_name=i.project_name,
            question=i.question,
            answer=i.answer,
            keywords=i.keywords,
            doc_group=i.doc_group,
            is_active=i.is_active,
        )
        for i in items
    ]


@router.delete(
    "/{qa_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Vô hiệu hoá một Q&A (soft-delete)",
)
async def deactivate_qa(qa_id: str, request: Request):
    require_role(request, "admin", "sales")
    qa_store = request.app.state.qa_store
    success = await qa_store.deactivate(qa_id)   # async
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy Q&A với id='{qa_id}'",
        )