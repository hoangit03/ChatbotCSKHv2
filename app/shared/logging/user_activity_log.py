"""
app/shared/logging/user_activity_log.py

Ghi audit log hành động người dùng ra file JSON Lines.
Phân biệt với application log (logger.py) — đây là business audit trail.

Format mỗi record:
  {
    "ts":          "2026-05-05T11:30:00+07:00",
    "session_id":  "sess_xxx",
    "project":     "Prime Diamond",
    "intent":      "sales_inquiry",
    "query_len":   42,
    "answer_len":  120,
    "response_ms": 850,
    "fallback":    false,
    "tool_count":  3
  }

NOTE: KHÔNG ghi raw query/answer để bảo mật nội dung. Chỉ ghi metadata.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.application.usecases.handle_chat import ChatRequest, ChatResponse


class UserActivityLogger:
    """
    Async-safe logger ghi user activity ra file JSONL (JSON Lines).
    File xoay theo ngày: storage/logs/user_activity_YYYY-MM-DD.jsonl
    """

    def __init__(self, log_dir: str = "./storage/logs"):
        self._log_dir = Path(log_dir)
        self._lock = asyncio.Lock()
        # Đảm bảo thư mục tồn tại
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        """Trả về path file log theo ngày hôm nay."""
        today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
        return self._log_dir / f"user_activity_{today}.jsonl"

    async def log_chat_event(
        self,
        req: "ChatRequest",
        resp: "ChatResponse",
    ) -> None:
        """
        Ghi một event chat vào file JSONL.
        Async-safe — dùng asyncio.Lock để tránh concurrent write corruption.
        """
        record = {
            "ts":          datetime.now(timezone.utc).astimezone().isoformat(),
            "session_id":  resp.session_id,
            "project":     resp.project_name or "unknown",
            "intent":      resp.intent,
            "query_len":   len(req.message),
            "answer_len":  len(resp.answer),
            "response_ms": resp.response_time_ms,
            "fallback":    resp.fallback,
            "was_injected": resp.was_injected,
            "tool_count":  len(resp.tool_calls),
            "source_count": len(resp.sources),
        }
        line = json.dumps(record, ensure_ascii=False)

        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._write_line,
                str(self._log_path()),
                line,
            )

    @staticmethod
    def _write_line(filepath: str, line: str) -> None:
        """Ghi đồng bộ trong executor để không block event loop."""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
