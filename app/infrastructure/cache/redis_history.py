"""
app/infrastructure/cache/redis_history.py

Lưu trữ và truy xuất lịch sử hội thoại từ Redis.
Giúp chatbot có trí nhớ ngắn hạn trong cùng một session.
"""
from __future__ import annotations

import json
from typing import Any

from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class RedisHistoryStore:
    def __init__(self, redis_pool, ttl: int = 3600):
        self._pool = redis_pool
        self._ttl = ttl

    def _get_client(self):
        import redis.asyncio as aioredis
        return aioredis.Redis(connection_pool=self._pool)

    async def get_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """Lấy N tin nhắn gần nhất của session."""
        try:
            client = self._get_client()
            key = f"chat_history:{session_id}"
            # Lấy list từ Redis
            data = await client.lrange(key, -limit, -1)
            if not data:
                return []
            return [json.loads(m) for m in data]
        except Exception as e:
            log.error("redis_history_get_failed", session=session_id, error=str(e))
            return []

    async def append(self, session_id: str, role: str, content: str):
        """Thêm một tin nhắn mới vào history."""
        try:
            client = self._get_client()
            key = f"chat_history:{session_id}"
            message = {"role": role, "content": content}
            
            # Đẩy vào list
            await client.rpush(key, json.dumps(message, ensure_ascii=False))
            # Cắt bớt nếu quá dài (ví dụ giữ 20 tin nhắn)
            await client.ltrim(key, -20, -1)
            # Gia hạn TTL
            await client.expire(key, self._ttl)
        except Exception as e:
            log.error("redis_history_append_failed", session=session_id, error=str(e))

    async def clear(self, session_id: str):
        try:
            client = self._get_client()
            await client.delete(f"chat_history:{session_id}")
            await client.delete(f"chat_context:{session_id}")
        except Exception as e:
            log.error("redis_history_clear_failed", session=session_id, error=str(e))

    async def get_context(self, session_id: str) -> dict[str, Any]:
        """Lấy các metadata của session (ví dụ: project_name)."""
        try:
            client = self._get_client()
            key = f"chat_context:{session_id}"
            data = await client.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            log.error("redis_context_get_failed", session=session_id, error=str(e))
            return {}

    async def set_context(self, session_id: str, context: dict[str, Any]):
        """Lưu metadata cho session."""
        try:
            client = self._get_client()
            key = f"chat_context:{session_id}"
            # Lấy context cũ và merge
            old = await self.get_context(session_id)
            old.update(context)
            await client.set(key, json.dumps(old, ensure_ascii=False), ex=self._ttl)
        except Exception as e:
            log.error("redis_context_set_failed", session=session_id, error=str(e))
