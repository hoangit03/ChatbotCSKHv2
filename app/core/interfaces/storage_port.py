"""
app/core/interfaces/storage_port.py

Port cho file storage (local disk / S3).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StoredFile:
    path: str
    size_bytes: int
    checksum: str   # SHA-256


class StoragePort(ABC):

    @abstractmethod
    async def save(
        self,
        file_bytes: bytes,
        destination: str,   # e.g. "DOC-VINH-BROC-abc123.pdf"
    ) -> StoredFile:
        ...

    @abstractmethod
    async def read(self, path: str) -> bytes:
        ...

    @abstractmethod
    async def delete(self, path: str) -> None:
        ...