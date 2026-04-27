"""
app/infrastructure/llm/llm_factory.py

Factory tạo LLM provider dựa vào settings.
─────────────────────────────────────────────────────────────
OCP: thêm provider mới → thêm case, không sửa gì khác.
DIP: caller nhận ChatPort interface, không biết impl là gì.

Cách swap provider:
  .env → LLM_PROVIDER=gemini, LLM_MODEL=gemini-1.5-pro
  → restart app → xong, không sửa code.
"""
from __future__ import annotations

from app.core.config.settings import Settings
from app.core.interfaces.llm_port import ChatPort, EmbedPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


def create_chat_provider(cfg: Settings) -> ChatPort:
    """Factory method — trả về ChatPort phù hợp với config."""
    provider = cfg.llm_provider
    log.info("llm_provider_init", provider=provider, model=cfg.llm_model)

    if provider == "anthropic":
        from app.infrastructure.llm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=cfg.anthropic_api_key,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )

    if provider == "openai":
        from app.infrastructure.llm.providers.openai_provider import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=cfg.openai_api_key,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            provider_label="openai",
        )

    if provider == "gemini":
        # Gemini hỗ trợ OpenAI-compat endpoint
        from app.infrastructure.llm.providers.openai_provider import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=cfg.google_api_key,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            provider_label="gemini",
        )

    if provider in ("ollama", "qwen"):
        # Local models via Ollama OpenAI-compat
        from app.infrastructure.llm.providers.openai_provider import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=cfg.ollama_api_key,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            base_url=cfg.ollama_base_url,
            provider_label=provider,
        )

    raise ValueError(f"Unknown LLM provider: '{provider}'")


def create_embed_provider(cfg: Settings) -> EmbedPort:
    """Embedding luôn dùng OpenAI — chất lượng ổn định nhất."""
    from app.infrastructure.llm.providers.embed_provider import OpenAIEmbedProvider

    # Nếu không có OpenAI key (e.g., pure Anthropic setup) → raise sớm
    if not cfg.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY bắt buộc cho embedding dù dùng provider LLM khác. "
            "Embedding dùng text-embedding-3-small của OpenAI."
        )

    return OpenAIEmbedProvider(
        api_key=cfg.openai_api_key,
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
    )