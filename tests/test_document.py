"""
tests/test_document.py
Unit test cho document upload pipeline.
"""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.document_service import DocumentService, DocumentValidationError
from app.models.document import DocGroup, DocumentUploadMeta
from datetime import datetime


@pytest.fixture
def service():
    return DocumentService()


@pytest.fixture
def sample_meta():
    return DocumentUploadMeta(
        project_name="Vinhomes",
        doc_group=DocGroup.BROCHURE,
        version="1.0",
        effective_date=datetime(2024, 1, 1),
        description="Test brochure",
    )


# ─────────────────────────────────────────────────────────────────
# File Validation
# ─────────────────────────────────────────────────────────────────

class TestFileValidation:

    def test_valid_pdf(self, service):
        """PDF hợp lệ không raise exception."""
        pdf_bytes = b"%PDF-1.4 sample content"
        service.validate_file(pdf_bytes, "test.pdf", "application/pdf")

    def test_invalid_extension(self, service):
        with pytest.raises(DocumentValidationError, match="Định dạng"):
            service.validate_file(b"content", "test.exe", "application/octet-stream")

    def test_file_too_large(self, service):
        # 51MB > 50MB limit
        large_bytes = b"%PDF" + b"0" * (51 * 1024 * 1024)
        with pytest.raises(DocumentValidationError, match="vượt quá giới hạn"):
            service.validate_file(large_bytes, "huge.pdf", "application/pdf")

    def test_magic_bytes_mismatch(self, service):
        """File đặt tên .pdf nhưng nội dung không phải PDF."""
        fake_pdf = b"This is not a PDF at all"
        with pytest.raises(DocumentValidationError, match="không khớp"):
            service.validate_file(fake_pdf, "fake.pdf", "application/pdf")

    def test_valid_docx(self, service):
        """DOCX (ZIP) bắt đầu bằng PK magic bytes."""
        docx_bytes = b"PK\x03\x04" + b"\x00" * 100
        service.validate_file(docx_bytes, "test.docx", "application/vnd.openxmlformats")

    def test_valid_png(self, service):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        service.validate_file(png_bytes, "image.png", "image/png")


# ─────────────────────────────────────────────────────────────────
# Checksum
# ─────────────────────────────────────────────────────────────────

class TestChecksum:

    def test_same_content_same_checksum(self, service):
        data = b"hello world"
        assert service.compute_checksum(data) == service.compute_checksum(data)

    def test_different_content_different_checksum(self, service):
        assert service.compute_checksum(b"abc") != service.compute_checksum(b"def")


# ─────────────────────────────────────────────────────────────────
# Upload Pipeline (mock external deps)
# ─────────────────────────────────────────────────────────────────

class TestUploadPipeline:

    @pytest.mark.asyncio
    async def test_upload_success(self, service, sample_meta):
        pdf_bytes = b"%PDF-1.4 test document with some content to parse"

        with (
            patch("app.services.document_service.FileParser.parse") as mock_parse,
            patch("app.services.document_service.text_processor") as mock_chunker,
            patch("app.services.document_service.llm_service.embed", new_callable=AsyncMock) as mock_embed,
            patch("app.services.document_service.vector_service.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
            patch.object(service, "_save_file", new_callable=AsyncMock),
        ):
            # Setup mocks
            mock_page = MagicMock()
            mock_page.text = "Tài liệu bất động sản test content"
            mock_page.page_number = 1
            mock_parse.return_value = MagicMock(pages=[mock_page])

            mock_chunk = MagicMock()
            mock_chunk.text = "chunk text"
            mock_chunker.chunk_document.return_value = [mock_chunk]

            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1

            result = await service.process_upload(
                file_bytes=pdf_bytes,
                file_name="brochure.pdf",
                meta=sample_meta,
                uploaded_by="test-user",
            )

        assert result.chunk_count == 1
        assert result.document_code.startswith("DOC-VINHOMES")

    @pytest.mark.asyncio
    async def test_upload_with_supersede(self, service, sample_meta):
        """Khi có supersedes_code, vector cũ phải bị supersede."""
        sample_meta.supersedes_code = "DOC-OLD-CODE-abc123"
        pdf_bytes = b"%PDF-1.4 new policy content"

        with (
            patch("app.services.document_service.FileParser.parse") as mock_parse,
            patch("app.services.document_service.text_processor") as mock_chunker,
            patch("app.services.document_service.llm_service.embed", new_callable=AsyncMock) as mock_embed,
            patch("app.services.document_service.vector_service.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
            patch("app.services.document_service.vector_service.supersede_document") as mock_supersede,
            patch.object(service, "_save_file", new_callable=AsyncMock),
        ):
            mock_page = MagicMock(text="new content", page_number=1)
            mock_parse.return_value = MagicMock(pages=[mock_page])
            mock_chunker.chunk_document.return_value = [MagicMock(text="chunk")]
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1
            mock_supersede.return_value = 5  # 5 vector cũ bị supersede

            result = await service.process_upload(
                file_bytes=pdf_bytes,
                file_name="new_policy.pdf",
                meta=sample_meta,
                uploaded_by="admin",
            )

        mock_supersede.assert_called_once_with("DOC-OLD-CODE-abc123")
        assert result.chunk_count == 1