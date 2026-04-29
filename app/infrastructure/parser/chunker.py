"""
app/infrastructure/parser/chunker.py

Token-based sliding window chunker.
Sửa lỗi: decode tokens về lại text khi force split hoặc xử lý overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)


class Chunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        tokenizer: Any = None,
    ):
        self._size = chunk_size
        self._overlap = chunk_overlap
        # Mặc định dùng tiktoken nếu có
        self._tokenizer_obj = tokenizer or _get_default_tokenizer_obj()
        
        # Helper để lấy encode/decode methods
        if hasattr(self._tokenizer_obj, "encode") and hasattr(self._tokenizer_obj, "decode"):
            self._encode = self._tokenizer_obj.encode
            self._decode = self._tokenizer_obj.decode
        else:
            # Fallback nếu dùng string split đơn giản
            self._encode = lambda x: x.split()
            self._decode = lambda x: " ".join(x)

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
        current_tok_len = 0
        idx = 0

        for para in paragraphs:
            para_tok = self._encode(para)
            para_len = len(para_tok)

            # 1. Đoạn quá dài (vượt cả chunk size) -> force split theo token
            if para_len > self._size:
                if current_parts:
                    chunks.append(self._make(idx, current_parts, current_tok_len, page, base_metadata))
                    idx += 1
                
                for sub_text in self._force_split(para):
                    sub_tok = self._encode(sub_text)
                    chunks.append(Chunk(
                        index=idx,
                        text=sub_text,
                        token_count=len(sub_tok),
                        page=page,
                        metadata={**base_metadata, "chunk_index": idx},
                    ))
                    idx += 1
                current_parts, current_tok_len = [], 0
                continue

            # 2. Nếu thêm para này vào sẽ vượt size -> đóng chunk hiện tại
            if current_tok_len + para_len > self._size and current_parts:
                chunks.append(self._make(idx, current_parts, current_tok_len, page, base_metadata))
                idx += 1
                
                # Tạo overlap bằng cách lấy X tokens cuối của chunk vừa rồi
                full_text_so_far = " ".join(current_parts)
                tokens_so_far = self._encode(full_text_so_far)
                overlap_tokens = tokens_so_far[-self._overlap:]
                overlap_text = self._decode(overlap_tokens)
                
                current_parts = [overlap_text]
                current_tok_len = len(overlap_tokens)

            current_parts.append(para)
            current_tok_len += para_len

        if current_parts:
            chunks.append(self._make(idx, current_parts, current_tok_len, page, base_metadata))

        return chunks

    def _make(self, idx: int, parts: list[str], tok_len: int, page: int | None, meta: dict) -> Chunk:
        return Chunk(
            index=idx,
            text="\n".join(parts),
            token_count=tok_len,
            page=page,
            metadata={**meta, "chunk_index": idx},
        )

    def _force_split(self, text: str) -> list[str]:
        """Băm nhỏ một paragraph cực dài thành các chunks có overlap."""
        tokens = self._encode(text)
        result = []
        start = 0
        while start < len(tokens):
            end = min(start + self._size, len(tokens))
            chunk_tokens = tokens[start:end]
            # QUAN TRỌNG: Phải decode tokens về lại text!
            result.append(self._decode(chunk_tokens))
            start += self._size - self._overlap
        return result


def _split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _get_default_tokenizer_obj():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None