"""
app/infrastructure/parser/chunker.py

Token-based sliding window chunker.
SRP: chỉ làm một việc — chia text thành chunks với overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)


class Chunker:
    """
    Chiến lược:
      1. Tách theo đoạn văn (semantic boundary)
      2. Ghép đoạn cho đến khi đầy chunk_size
      3. Overlap = giữ lại chunk_overlap token cuối của chunk trước

    SRP: không biết gì về vector DB hay LLM.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        tokenizer: Callable[[str], list] | None = None,
    ):
        self._size = chunk_size
        self._overlap = chunk_overlap
        self._tok = tokenizer or _default_tokenizer()

    def chunk(
        self,
        text: str,
        base_metadata: dict,
        page: int | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        paragraphs = _split_paragraphs(text)
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tok: list = []
        idx = 0

        for para in paragraphs:
            para_tok = self._tok(para)

            # Đoạn quá dài → force split
            if len(para_tok) > self._size:
                if current_parts:
                    chunks.append(self._make(idx, current_parts, current_tok, page, base_metadata))
                    idx += 1
                for sub in self._force_split(para):
                    chunks.append(Chunk(
                        index=idx,
                        text=sub,
                        token_count=len(self._tok(sub)),
                        page=page,
                        metadata={**base_metadata, "chunk_index": idx},
                    ))
                    idx += 1
                current_parts, current_tok = [], []
                continue

            # Flush nếu thêm vào sẽ vượt limit
            if len(current_tok) + len(para_tok) > self._size and current_parts:
                chunks.append(self._make(idx, current_parts, current_tok, page, base_metadata))
                idx += 1
                # Keep overlap
                overlap_text = " ".join(current_tok[-self._overlap:])
                current_parts = [overlap_text] if overlap_text else []
                current_tok = self._tok(overlap_text)

            current_parts.append(para)
            current_tok = self._tok(" ".join(current_parts))

        if current_parts:
            chunks.append(self._make(idx, current_parts, current_tok, page, base_metadata))

        return chunks

    def _make(
        self,
        idx: int,
        parts: list[str],
        tokens: list,
        page: int | None,
        meta: dict,
    ) -> Chunk:
        text = " ".join(parts)
        return Chunk(
            index=idx,
            text=text,
            token_count=len(tokens),
            page=page,
            metadata={**meta, "chunk_index": idx},
        )

    def _force_split(self, text: str) -> list[str]:
        tokens = self._tok(text)
        result = []
        start = 0
        while start < len(tokens):
            end = min(start + self._size, len(tokens))
            result.append(" ".join(str(t) for t in tokens[start:end]))
            start += self._size - self._overlap
        return result


def _split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _default_tokenizer() -> Callable[[str], list]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode
    except Exception:
        return str.split