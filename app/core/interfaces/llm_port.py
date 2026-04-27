"""
app/core/interfaces/llm_port.py

Interface (Port) cho LLM.
─────────────────────────────────────────────────────────────
SOLID áp dụng:
  - ISP: tách ChatPort và EmbedPort — không force implement cả hai
  - DIP: tầng application phụ thuộc vào interface này, không phải impl cụ thể
  - OCP: thêm provider mới → tạo class mới implement interface, không sửa code cũ

LangGraph sẽ gọi ChatPort.chat() — không biết provider là gì.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMMessage:
    """Chuẩn hoá message, không phụ thuộc format của từng provider."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str   # "end_turn" | "max_tokens" | "tool_use"


class ChatPort(ABC):
    """
    Port cho chat completion.
    Implement với: Anthropic, OpenAI, Gemini, Ollama, Qwen.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Gửi messages, nhận response. Không streaming."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
    ) -> AsyncIterator[str]:
        """Streaming version — yield từng token."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Tên provider để log/debug."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class EmbedPort(ABC):
    """
    Port cho embedding.
    Tách biệt với ChatPort vì không phải mọi chat provider đều embed tốt.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed, trả list vectors."""
        ...

    @abstractmethod
    async def embed_one(self, text: str) -> list[float]:
        """Single embed."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension."""
        ...