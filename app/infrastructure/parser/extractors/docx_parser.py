"""
app/infrastructure/parser/extractors/docx_parser.py
"""
from __future__ import annotations
import io, re
from app.core.interfaces.parser_port import ParsedChunk, ParsedDocument, ParserPort


class DocxParser(ParserPort):

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".docx"})

    async def parse(self, content: bytes, file_name: str) -> ParsedDocument:
        from docx import Document
        doc = Document(io.BytesIO(content))
        parts: list[str] = []
        tables: list[list[list[str]]] = []

        for block in doc.element.body:
            tag = block.tag.split("}")[-1]
            if tag == "p":
                text = "".join(
                    r.text for r in block.iterchildren() if r.tag.endswith("}r")
                ).strip()
                if text:
                    parts.append(text)
            elif tag == "tbl":
                tbl: list[list[str]] = []
                for row in block.iterchildren():
                    if not row.tag.endswith("}tr"):
                        continue
                    cells = [
                        "".join(t.text or "" for t in cell.itertext())
                        for cell in row.iterchildren() if cell.tag.endswith("}tc")
                    ]
                    tbl.append(cells)
                if tbl:
                    tables.append(tbl)
                    parts.append(_tbl_to_text(tbl))

        full = "\n".join(parts)
        return ParsedDocument(
            file_name=file_name,
            file_type="docx",
            chunks=[ParsedChunk(page=1, text=_clean(full), tables=tables)],
        )


def _tbl_to_text(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    lines = [
        " | ".join(f"{h}: {v}" for h, v in zip(header, row) if h.strip() and v.strip())
        for row in rows[1:]
    ]
    return "\n".join(l for l in lines if l)


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()