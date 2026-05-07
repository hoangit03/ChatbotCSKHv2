import os
import asyncpg
import json
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@core_postgres:5432/llmerp")

async def save_chat_message_async(session_id: str, role: str, content: str, user_id: str = None, tenant_id: str = None):
    """Lưu tin nhắn vào PostgreSQL bất đồng bộ"""
    if not session_id:
        return
        
    try:
        import uuid
        session_uuid = uuid.UUID(session_id)
        
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Tạo session nếu chưa tồn tại
        if user_id and tenant_id:
            uid = int(user_id)
            session_exists = await conn.fetchval("SELECT id FROM chat_sessions WHERE id = $1", session_uuid)
            if not session_exists:
                # Cắt content làm title
                title = content[:50] + "..." if len(content) > 50 else content
                await conn.execute(
                    """
                    INSERT INTO chat_sessions (id, user_id, tenant_id, title)
                    VALUES ($1, $2, $3, $4)
                    """,
                    session_uuid, uid, tenant_id, title
                )
        await conn.execute(
            """
            INSERT INTO chat_messages (session_id, role, content)
            VALUES ($1, $2, $3)
            """,
            session_uuid, role, content
        )
        await conn.close()
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
