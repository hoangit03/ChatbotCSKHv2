"""
app/infrastructure/cache/pg_history.py

Lưu lịch sử hội thoại vào PostgreSQL (fire-and-forget background task).

FIX v2:
  - Dùng asyncpg.create_pool() singleton thay vì tạo connection mới mỗi lần.
  - async with pool.acquire() → tự động trả connection về pool dù có exception.
  - Không còn connection leak khi exception xảy ra giữa chừng.
  - Dùng structlog nhất quán với toàn project.
"""
from __future__ import annotations

import os
import asyncio
from typing import Optional

import asyncpg
from app.shared.logging.logger import get_logger

log = get_logger(__name__)

# Chuẩn hóa URL: SQLAlchemy dùng postgresql+asyncpg://, asyncpg dùng postgresql://
_RAW_DB_URL = os.getenv("DATABASE_URL", "postgresql://raguser:ragpassword@localhost:5432/ragdb")
DATABASE_URL = (
    _RAW_DB_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
    if _RAW_DB_URL.startswith("postgresql+asyncpg://")
    else _RAW_DB_URL
)

# Connection pool singleton — khởi tạo lazy khi cần lần đầu tiên
_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


async def _get_pool() -> asyncpg.Pool:
    """Lazy init connection pool (singleton). Thread-safe qua asyncio.Lock."""
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:  # Double-check locking pattern
                try:
                    _pool = await asyncpg.create_pool(
                        DATABASE_URL,
                        min_size=1,
                        max_size=5,        # Giới hạn tránh quá tải PG
                        command_timeout=10,
                    )
                    log.info("pg_pool_created", url=DATABASE_URL.split("@")[-1])
                except Exception as e:
                    log.error("pg_pool_create_failed", error=str(e))
                    raise
    return _pool


async def save_chat_message_async(
    session_id: str,
    role: str,
    content: str,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Lưu tin nhắn vào PostgreSQL bất đồng bộ.
    Fire-and-forget — được gọi qua asyncio.create_task().
    Không raise exception (chỉ log lỗi) để không ảnh hưởng luồng chat chính.
    """
    if not session_id:
        return

    try:
        import uuid
        # Validate UUID format — tránh ValueError
        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            log.warning("pg_history_invalid_session_id", session_id=session_id[:36])
            return

        pool = await _get_pool()

        # async with pool.acquire() tự đóng connection khi ra khỏi block
        async with pool.acquire() as conn:
            # Tạo session nếu chưa tồn tại (chỉ khi có user_id và tenant_id)
            if user_id and tenant_id:
                try:
                    uid = int(user_id)
                    title = content[:50] + "..." if len(content) > 50 else content
                    await conn.execute(
                        """
                        INSERT INTO chat_sessions (id, user_id, tenant_id, title)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        session_uuid, uid, tenant_id, title,
                    )
                except (ValueError, asyncpg.PostgresError) as e:
                    # Không fail toàn bộ function nếu tạo session lỗi
                    log.warning("pg_create_session_failed", error=str(e))

            await conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES ($1, $2, $3)
                """,
                session_uuid, role, content,
            )

    except asyncpg.PostgresConnectionError as e:
        log.error("pg_history_connection_error", error=str(e))
    except Exception as e:
        log.error("pg_history_save_failed", error=str(e), session=session_id[:36])
