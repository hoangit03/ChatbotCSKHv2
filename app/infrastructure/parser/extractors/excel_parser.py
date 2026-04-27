from __future__ import annotations
import io, re
from app.core.interfaces.parser_port import ParsedChunk, ParsedDocument, ParserPort
 
 
class ExcelParser(ParserPort):
 
    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".xlsx", ".xls"})
 
    async def parse(self, content: bytes, file_name: str) -> ParsedDocument:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        chunks: list[ParsedChunk] = []
 
        for idx, ws in enumerate(wb.worksheets, start=1):
            rows: list[list[str]] = []
            for row in ws.iter_rows(values_only=True):
                vals = [str(c) if c is not None else "" for c in row]
                if any(v.strip() for v in vals):
                    rows.append(vals)
 
            if not rows:
                continue
 
            # Detect Q&A pattern: 2-column sheets = Question / Answer
            text = _tbl_to_text(rows)
            chunks.append(ParsedChunk(
                page=idx,
                text=_clean(text),
                tables=[rows],
                metadata={"sheet_name": ws.title},
            ))
 
        return ParsedDocument(file_name=file_name, file_type="xlsx", chunks=chunks)
 
 
def _tbl_to_text(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = [
            f"{h.strip()}: {v.strip()}"
            for h, v in zip(header, row) if h.strip() and v.strip()
        ]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)
 
 
def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()