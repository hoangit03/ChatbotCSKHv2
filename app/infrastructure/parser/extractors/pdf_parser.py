"""
app/infrastructure/parser/extractors/pdf_parser.py
"""
from __future__ import annotations
import io
from app.core.interfaces.parser_port import ParsedChunk, ParsedDocument, ParserPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class PDFParser(ParserPort):

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    async def parse(self, content: bytes, file_name: str) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pip install pypdf")

        reader = PdfReader(io.BytesIO(content))
        chunks: list[ParsedChunk] = []

        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append(ParsedChunk(page=i, text=_clean(text)))

        # Nếu không extract được text (scanned PDF) → OCR
        if not chunks:
            log.warning("pdf_no_text_ocr_fallback", file=file_name)
            chunks = await _ocr_pdf(content)

        return ParsedDocument(
            file_name=file_name,
            file_type="pdf",
            chunks=chunks,
            raw_metadata=dict(reader.metadata or {}),
        )


async def _ocr_pdf(content: bytes) -> list[ParsedChunk]:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError:
        return [ParsedChunk(page=1, text="[OCR not available]")]

    images = convert_from_bytes(content, dpi=200)
    return [
        ParsedChunk(page=i, text=_clean(pytesseract.image_to_string(img, lang="vie+eng")))
        for i, img in enumerate(images, start=1)
        if pytesseract.image_to_string(img, lang="vie+eng").strip()
    ]


def _clean(text: str) -> str:
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()