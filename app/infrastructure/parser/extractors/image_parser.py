"""
app/infrastructure/parser/extractors/image_parser.py
"""
from app.core.interfaces.parser_port import ParsedChunk, ParsedDocument, ParserPort
 
 
class ImageParser(ParserPort):
 
    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".png", ".jpg", ".jpeg", ".webp"})
 
    async def parse(self, content: bytes, file_name: str) -> ParsedDocument:
        try:
            import pytesseract
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(content))
            text = pytesseract.image_to_string(img, lang="vie+eng").strip()
        except Exception:
            text = "[Image OCR unavailable]"
 
        return ParsedDocument(
            file_name=file_name,
            file_type="image",
            chunks=[ParsedChunk(page=1, text=text)],
        )