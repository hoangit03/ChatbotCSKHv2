"""
app/core/interfaces/parser_port.py

Port cho document parser.
OCP: thêm format mới → implement ParserPort mới, đăng ký vào ParserRegistry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedChunk:
    """Một đơn vị nội dung sau parse — trang hoặc sheet."""
    page: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    file_name: str
    file_type: str
    chunks: list[ParsedChunk]
    raw_metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks if c.text.strip())


class ParserPort(ABC):
    """Một parser cho một hoặc nhiều file type."""

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """{'.pdf'}, {'.docx'}, v.v."""
        ...

    @abstractmethod
    async def parse(self, content: bytes, file_name: str) -> ParsedDocument:
        ...


class ParserRegistry:
    """
    Registry pattern — tra cứu parser theo extension.
    OCP: thêm parser mới bằng register(), không sửa if-else.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, ParserPort] = {}

    def register(self, parser: ParserPort) -> None:
        for ext in parser.supported_extensions:
            self._parsers[ext.lower()] = parser

    def get(self, extension: str) -> Optional[ParserPort]:
        return self._parsers.get(extension.lower())

    def supported(self) -> set[str]:
        return set(self._parsers.keys())