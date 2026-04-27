"""
app/infrastructure/llm/providers/anthropic_provider.py

Implement ChatPort cho Anthropic Claude.
Chỉ class này biết về anthropic SDK — tầng trên không import anthropic trực tiếp.
"""
from __future__ import annotations

from typing import AsyncIterator

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.interfaces.llm_port import ChatPort, LLMMessage, LLMResponse
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class AnthropicProvider(ChatPort):

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        import anthropic as _sdk
        self._client = _sdk.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def chat(
        self,
        messages: list[LLMMessage],
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        api_msgs = [{"role": m.role, "content": m.content} for m in messages]
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
            system=system or "",
            messages=api_msgs,
        )
        return LLMResponse(
            content=resp.content[0].text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason or "end_turn",
        )

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
    ) -> AsyncIterator[str]:
        api_msgs = [{"role": m.role, "content": m.content} for m in messages]
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=api_msgs,
        ) as stream:
            async for text in stream.text_stream:
                yield text