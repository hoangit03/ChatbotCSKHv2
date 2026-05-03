"""
app/infrastructure/llm/providers/openai_provider.py

Implement ChatPort cho OpenAI-compatible API.
Covers: GPT-4o, Gemini (via openai-compat), Ollama, Qwen (local).
Một provider class xử lý được nhiều backend nhờ base_url config.
"""
from __future__ import annotations

from typing import AsyncIterator, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.interfaces.llm_port import ChatPort, LLMMessage, LLMResponse
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class OpenAICompatProvider(ChatPort):
    """
    Dùng cho: OpenAI, Gemini (openai-compat endpoint), Ollama, Qwen.
    Chỉ cần thay base_url + api_key + model.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str | None = None,   # None = OpenAI default
        provider_label: str = "openai",
    ):
        from openai import AsyncOpenAI
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._label = provider_label

    @property
    def provider_name(self) -> str:
        return self._label

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
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        api_msgs: list[dict] = []
        if system:
            api_msgs.append({"role": "system", "content": system})
        for m in messages:
            api_msgs.append({"role": m.role, "content": m.content})

        kwargs_api = {
            "model": self._model,
            "messages": api_msgs,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if tools:
            kwargs_api["tools"] = tools

        resp = await self._client.chat.completions.create(**kwargs_api)
        choice = resp.choices[0]
        usage = resp.usage
        
        parsed_tool_calls = None
        if choice.message.tool_calls:
            import json
            parsed_tool_calls = []
            for tc in choice.message.tool_calls:
                parsed_tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
                })

        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason or "stop",
            tool_calls=parsed_tool_calls,
        )

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
    ) -> AsyncIterator[str]:
        api_msgs: list[dict] = []
        if system:
            api_msgs.append({"role": "system", "content": system})
        for m in messages:
            api_msgs.append({"role": m.role, "content": m.content})

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=api_msgs,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content