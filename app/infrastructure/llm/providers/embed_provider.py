"""
app/infrastructure/llm/providers/embed_provider.py

EmbedPort implementation dùng OpenAI text-embedding-3-small.
Dùng cho mọi LLM provider vì embedding chất lượng ổn định.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.interfaces.llm_port import EmbedPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)
_BATCH_SIZE = 64   # OpenAI rate limit safe


class OpenAIEmbedProvider(EmbedPort):

    def __init__(self, api_key: str, model: str, dimension: int):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed với auto-batching để tránh rate limit."""
        if not texts:
            return []
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            resp = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            all_vecs.extend(item.embedding for item in resp.data)
        return all_vecs

    async def embed_one(self, text: str) -> list[float]:
        vecs = await self.embed([text])
        return vecs[0]