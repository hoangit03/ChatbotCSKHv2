"""
app/core/interfaces/vector_port.py

Port cho Vector Database.
Qdrant là implementation mặc định — nhưng có thể swap Chroma/Weaviate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorPoint:
    """Một document chunk được index."""
    id: str
    vector: list[float]
    payload: dict            # metadata: doc_code, text, project, group, v.v.


@dataclass
class SearchResult:
    """Kết quả tìm kiếm vector."""
    id: str
    score: float
    text: str
    document_code: str
    document_name: str
    doc_group: str
    project_name: str
    page_number: Optional[int]
    extra: dict = field(default_factory=dict)   # full payload — dùng cho QA lookup


@dataclass
class SearchFilter:
    """Filter áp dụng khi search — không expose Qdrant type ra ngoài."""
    project_name: Optional[str] = None
    doc_group: Optional[str] = None
    status: str = "active"     # "active" | "superseded"


class VectorPort(ABC):

    @abstractmethod
    async def upsert(self, points: list[VectorPoint]) -> int:
        """Insert hoặc update points. Trả về số point upserted."""
        ...

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: SearchFilter,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def scroll(
        self,
        filter: SearchFilter,
        limit: int = 100,
        offset: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str]]:
        """
        Cuộn qua tất cả points khớp filter (không cần vector).
        Dùng cho list_by_project, export, admin APIs.
        Trả về (results, next_offset) — next_offset=None nghĩa là hết data.
        """
        ...

    @abstractmethod
    async def list_unique_projects(self) -> list[str]:
        """Lấy danh sách các dự án (unique project_name) đang có trong DB."""
        ...

    @abstractmethod
    async def mark_superseded(self, document_code: str) -> int:
        """Đánh dấu tất cả chunk của doc cũ thành superseded. Trả về count."""
        ...

    @abstractmethod
    async def delete_by_document(self, document_code: str) -> None:
        ...

    @abstractmethod
    async def ensure_collection(self, dimension: int) -> None:
        """Tạo collection nếu chưa có."""
        ...