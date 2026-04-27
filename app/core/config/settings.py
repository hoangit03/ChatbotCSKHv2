"""
app/core/config/settings.py

Single source of truth cho toàn bộ config.
Dùng Pydantic Settings — validate chặt, không có magic string.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Set

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LLMProvider = Literal["anthropic", "openai", "gemini", "ollama", "qwen"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "dev-secret-key-change-in-production-32ch"
    debug: bool = False
    # Key để gọi API trong dev/staging. Đặt trong .env: DEV_API_KEY=...
    dev_api_key: str = "chatbot-dev-key-2024"

    # ── LLM ──────────────────────────────────────────────────────
    llm_provider: LLMProvider = "anthropic"
    llm_model: str = "claude-opus-4-5"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # API keys — chỉ validate key của provider được chọn
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"

    # ── Embedding ─────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # ── Vector DB ─────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "realestate_kb"

    # ── Database ──────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb"

    # ── Redis ─────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600

    # ── Sales Backend API ─────────────────────────────────────────
    sales_api_base_url: str = "https://sales-backend.internal.company.com"
    sales_api_key: str = ""
    sales_api_timeout: int = 10
    sales_api_max_retries: int = 3

    # ── Storage ───────────────────────────────────────────────────
    storage_path: str = "./storage/documents"
    max_file_size_mb: int = 50
    allowed_extensions: str = "pdf,docx,xlsx,png,jpg,jpeg"

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # ── Rate Limit ────────────────────────────────────────────────
    rate_limit_chat: int = 30
    rate_limit_upload: int = 20

    # ── Agent ─────────────────────────────────────────────────────
    agent_max_iterations: int = 8
    rag_score_threshold: float = 0.70
    qa_score_threshold: float = 0.30   # multi-signal scoring (0.30 thay vì 0.45 của Jaccard)

    # ── Q&A Auto-load (dev/staging) ───────────────────────────────
    # Đường dẫn tới file Excel Q&A load khi khởi động (bỏ trống để bỏ qua)
    qa_autoload_file: str = ""
    # Tên project gán cho các Q&A item được load tự động
    qa_autoload_project: str = "default"

    # ── Sales API toggle ──────────────────────────────────────────
    # False = bỏ qua toàn bộ Sales tools (khi chưa có API backend)
    sales_api_enabled: bool = True

    # ── Computed ──────────────────────────────────────────────────
    @property
    def allowed_ext_set(self) -> Set[str]:
        return {f".{e.strip()}" for e in self.allowed_extensions.split(",")}

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def sales_api_configured(self) -> bool:
        """True nếu Sales API đã được cấu hình (không dùng placeholder URL)."""
        _PLACEHOLDER = "https://sales-backend.internal.company.com"
        return (
            self.sales_api_enabled
            and self.sales_api_base_url != _PLACEHOLDER
            and bool(self.sales_api_key)
        )

    # ── Validation ────────────────────────────────────────────────
    @field_validator("app_secret_key")
    @classmethod
    def secret_min_length(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("app_secret_key must be >= 32 characters")
        return v

    @model_validator(mode="after")
    def check_llm_key(self) -> "Settings":
        required = {
            "anthropic": ("anthropic_api_key", self.anthropic_api_key),
            "openai": ("openai_api_key", self.openai_api_key),
            "gemini": ("google_api_key", self.google_api_key),
            "ollama": None,
            "qwen": None,
        }
        entry = required.get(self.llm_provider)
        if entry and not entry[1]:
            raise ValueError(
                f"{entry[0]} is required when llm_provider='{self.llm_provider}'"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton — parse .env một lần duy nhất."""
    return Settings()