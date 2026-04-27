"""
app/application/usecases/upload_document.py

Use Case: CB-01 — Upload tài liệu vào knowledge base.

Pipeline:
  1. Validate file (extension, size, magic bytes)
  2. Parse nội dung (PDF / DOCX / Excel / Image)
  3. Chunk text theo token
  4. Embed từng chunk (batch)
  5. Upsert vào Qdrant với metadata đầy đủ
  6. Nếu có supersedes_code → mark_superseded tài liệu cũ
  7. Lưu file vật lý vào storage
  8. Trả về DocumentUploadResult

SRP: class này chỉ lo orchestration, không biết Qdrant / Anthropic / disk.
DIP: nhận port interfaces, không import adapter trực tiếp.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config.settings import Settings
from app.core.interfaces.llm_port import EmbedPort
from app.core.interfaces.parser_port import ParserRegistry
from app.core.interfaces.storage_port import StoragePort
from app.core.interfaces.vector_port import VectorPoint, VectorPort
from app.infrastructure.parser.chunker import Chunker
from app.shared.errors.exceptions import (
    FileMagicMismatchError,
    FileTooLargeError,
    ParseError,
    UnsupportedFileTypeError,
)
from app.shared.logging.logger import get_logger
from app.shared.security.guards import check_magic_bytes, make_document_code

log = get_logger(__name__)


# ── Request / Response DTOs ───────────────────────────────────────

@dataclass
class UploadDocumentRequest:
    file_bytes: bytes
    file_name: str
    project_name: str
    doc_group: str            # brochure | price_list | sales_policy | faq | legal | floor_plan | progress
    version: str              # e.g. "1.0", "2.1"
    effective_date: datetime
    uploaded_by: str
    description: Optional[str] = None
    supersedes_code: Optional[str] = None   # mã doc cũ bị thay thế


@dataclass
class UploadDocumentResult:
    document_code: str
    chunk_count: int
    superseded_count: int     # số vector cũ bị đánh dấu superseded
    file_checksum: str
    message: str


# ── Use Case ──────────────────────────────────────────────────────

class UploadDocumentUseCase:
    """
    Orchestrates toàn bộ pipeline CB-01.
    Không biết provider LLM là gì, Qdrant hay Chroma, local hay S3.
    """

    def __init__(
        self,
        cfg: Settings,
        parser_registry: ParserRegistry,
        embedder: EmbedPort,
        vector_db: VectorPort,
        storage: StoragePort,
    ) -> None:
        self._cfg = cfg
        self._parsers = parser_registry
        self._embedder = embedder
        self._vector_db = vector_db
        self._storage = storage
        self._chunker = Chunker(
            chunk_size=cfg.chunk_size_tokens,
            chunk_overlap=cfg.chunk_overlap_tokens,
        )

    async def execute(self, req: UploadDocumentRequest) -> UploadDocumentResult:
        ext = Path(req.file_name).suffix.lower()
        doc_code = make_document_code(req.project_name, req.doc_group)

        log.info(
            "upload_start",
            doc_code=doc_code,
            file=req.file_name,
            project=req.project_name,
            supersedes=req.supersedes_code,
        )

        # ── Step 1: Validate ──────────────────────────────────────
        self._validate(req.file_bytes, req.file_name, ext)

        checksum = _sha256(req.file_bytes)

        # ── Step 2: Parse ─────────────────────────────────────────
        parser = self._parsers.get(ext)
        if parser is None:
            raise UnsupportedFileTypeError(
                f"Không tìm thấy parser cho '{ext}'. "
                f"Hỗ trợ: {self._parsers.supported()}"
            )

        try:
            parsed = await parser.parse(req.file_bytes, req.file_name)
        except Exception as e:
            raise ParseError(f"Parse thất bại: {e}", file=req.file_name) from e

        log.info("upload_parsed", doc_code=doc_code, pages=len(parsed.chunks))

        # ── Step 3: Chunk ─────────────────────────────────────────
        base_meta = {
            "document_code": doc_code,
            "project_name":  req.project_name,
            "doc_group":     req.doc_group,
            "version":       req.version,
            "effective_date": req.effective_date.isoformat(),
            "file_name":     req.file_name,
            "uploaded_by":   req.uploaded_by,
        }

        all_chunks = []
        for page_chunk in parsed.chunks:
            chunks = self._chunker.chunk(
                text=page_chunk.text,
                base_metadata=base_meta,
                page=page_chunk.page,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            log.warning("upload_no_chunks", doc_code=doc_code)
            return UploadDocumentResult(
                document_code=doc_code,
                chunk_count=0,
                superseded_count=0,
                file_checksum=checksum,
                message="Tài liệu không có nội dung text để index.",
            )

        # ── Step 4: Embed (batch, tránh rate limit) ───────────────
        texts = [c.text for c in all_chunks]
        embeddings = await self._batch_embed(texts)

        # ── Step 5: Upsert Qdrant ─────────────────────────────────
        points = [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    **all_chunks[i].metadata,
                    "text":        all_chunks[i].text,
                    "chunk_index": all_chunks[i].index,
                    "page_number": all_chunks[i].page,
                    "token_count": all_chunks[i].token_count,
                    "checksum":    checksum,
                },
            )
            for i in range(len(all_chunks))
        ]

        chunk_count = await self._vector_db.upsert(points)

        # ── Step 6: Supersede doc cũ ──────────────────────────────
        superseded_count = 0
        if req.supersedes_code:
            superseded_count = await self._vector_db.mark_superseded(req.supersedes_code)
            log.info(
                "upload_superseded",
                old_code=req.supersedes_code,
                count=superseded_count,
            )

        # ── Step 7: Lưu file vật lý ───────────────────────────────
        dest_name = f"{doc_code}{ext}"
        await self._storage.save(req.file_bytes, dest_name)

        log.info(
            "upload_complete",
            doc_code=doc_code,
            chunks=chunk_count,
            superseded=superseded_count,
        )

        return UploadDocumentResult(
            document_code=doc_code,
            chunk_count=chunk_count,
            superseded_count=superseded_count,
            file_checksum=checksum,
            message=f"Upload thành công. Đã index {chunk_count} đoạn văn bản.",
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _validate(self, data: bytes, name: str, ext: str) -> None:
        if ext not in self._cfg.allowed_ext_set:
            raise UnsupportedFileTypeError(
                f"Định dạng '{ext}' không được hỗ trợ. "
                f"Cho phép: {self._cfg.allowed_extensions}",
                extension=ext,
            )
        if len(data) > self._cfg.max_file_bytes:
            raise FileTooLargeError(
                f"File vượt giới hạn {self._cfg.max_file_size_mb}MB "
                f"(thực tế: {len(data) // 1024 // 1024}MB)",
                limit_mb=self._cfg.max_file_size_mb,
            )
        if not check_magic_bytes(data, ext):
            raise FileMagicMismatchError(
                f"Nội dung file không khớp định dạng '{ext}'. "
                "Có thể file bị đổi tên hoặc bị lỗi.",
                extension=ext,
            )

    async def _batch_embed(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Chia batch để tránh rate limit embedding API."""
        result: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = await self._embedder.embed(batch)
            result.extend(vecs)
        return result


# ── Utility ───────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()