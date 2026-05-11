import os
import asyncpg
import json
import logging

import logging
import asyncio
import httpx

logger = logging.getLogger(__name__)

async def _sync_to_frappe(session_id: str, role: str, content: str, title: str, user_id: str, tenant_id: str):
    """
    Fire-and-forget sync to Frappe App Core
    """
    try:
        url = "http://ct_hub_be_v2:8000/api/method/ct_agent_hub.admin_api.sync_chatbot_message"
        payload = {
            "session_id": session_id,
            "agent_name": tenant_id or "default",
            "role": role,
            "content": content,
            "user": "guest_3rd_party" if user_id == "0" else (user_id or "guest@local"),
            "title": title
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json=payload, headers={"Host": "app.ctpai.vn"})
    except Exception as e:
        logger.error(f"Failed to sync message to Frappe: {e}")


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@core_postgres:5432/llmerp")
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

async def save_chat_message_async(session_id: str, role: str, content: str, user_id: str = None, tenant_id: str = None):
    """Lưu tin nhắn vào PostgreSQL bất đồng bộ và đồng bộ lên Frappe"""
    if not session_id:
        return
        
    try:
        import uuid
        session_uuid = uuid.UUID(session_id)
        
        # 1. Fire and forget sync to Frappe
        title = content[:50] + "..." if len(content) > 50 else content
        asyncio.create_task(
            _sync_to_frappe(session_id, role, content, title, str(user_id) if user_id else "0", tenant_id)
        )

        # 2. Lưu vào DB nội bộ (PostgreSQL) - Bọc try/except riêng biệt
        try:
            conn = await asyncpg.connect(DATABASE_URL)
            
            # Tạo session nếu chưa tồn tại
            session_exists = await conn.fetchval("SELECT id FROM chat_sessions WHERE id = $1", session_uuid)
            if not session_exists:
                # Nếu không có user_id hợp lệ, truyền None để tránh vi phạm khóa ngoại (0 không tồn tại)
                uid = int(user_id) if user_id and str(user_id).isdigit() and user_id != "0" else None
                tid = tenant_id if tenant_id else "default"
                await conn.execute(
                    """
                    INSERT INTO chat_sessions (id, user_id, tenant_id, title)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    session_uuid, uid, tid, title
                )
            await conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES ($1, $2, $3)
                """,
                session_uuid, role, content
            )
            await conn.close()
        except Exception as pg_err:
            logger.error(f"Error saving to Postgres (chat_sessions/messages): {pg_err}")

    except Exception as e:
        logger.error(f"Error in save_chat_message_async flow: {e}")
