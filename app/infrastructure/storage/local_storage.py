"""
app/infrastructure/storage/local_storage.py

Local disk implementation của StoragePort.
Swap sang S3 → tạo S3StorageAdapter implement cùng interface, không sửa gì khác.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import aiofiles

from app.core.interfaces.storage_port import StoragePort, StoredFile
from app.shared.errors.exceptions import StorageError
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class LocalStorageAdapter(StoragePort):

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        log.info("local_storage_init", path=str(self._base))

    async def save(self, file_bytes: bytes, destination: str) -> StoredFile:
        dest = self._base / destination
        try:
            async with aiofiles.open(dest, "wb") as f:
                await f.write(file_bytes)
            checksum = hashlib.sha256(file_bytes).hexdigest()
            log.debug("storage_save", path=str(dest), size=len(file_bytes))
            return StoredFile(
                path=str(dest),
                size_bytes=len(file_bytes),
                checksum=checksum,
            )
        except OSError as e:
            raise StorageError(f"Không thể lưu file '{destination}': {e}") from e

    async def read(self, path: str) -> bytes:
        try:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except OSError as e:
            raise StorageError(f"Không thể đọc file '{path}': {e}") from e

    async def delete(self, path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
            log.debug("storage_delete", path=path)
        except OSError as e:
            raise StorageError(f"Không thể xoá file '{path}': {e}") from e