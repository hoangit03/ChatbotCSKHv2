"""
app/agent/tools/qa_tool.py

Tool: Tra cứu bộ Q&A chuẩn (import từ Excel).
Ưu tiên gọi trước RAG — nhanh hơn và chính xác nhất.

Thuật toán tìm kiếm (v2 — thay Jaccard bằng multi-signal scoring):
  1. Exact match (sau normalize)         → score = 1.0
  2. Keyword match (từ cột keywords)     → bonus score
  3. Token overlap (bi-gram + unigram)   → partial score
  4. Prefix/substring match              → partial score
  Kết hợp cả 4 tín hiệu → lấy score cao nhất.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from app.agent.state.agent_state import AgentState
from app.agent.tools.base_tool import AgentTool, ToolResult
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


@dataclass
class QAItem:
    id: str
    project_name: str
    question: str
    answer: str
    keywords: list[str] = field(default_factory=list)
    doc_group: Optional[str] = None
    is_active: bool = True


class QAStore:
    """In-memory Q&A store. Production: thay bằng PostgreSQL + full-text search."""

    def __init__(self) -> None:
        self._items: dict[str, QAItem] = {}

    def add(self, item: QAItem) -> None:
        self._items[item.id] = item

    def bulk_add(self, items: list[QAItem]) -> None:
        for i in items:
            self._items[i.id] = i

    def search(
        self,
        query: str,
        project: str | None = None,
        threshold: float = 0.30,   # Ngưỡng thấp hơn cho multi-signal
        top_k: int = 1,
    ) -> list[QAItem]:
        q_norm = _normalize(query)
        q_tokens = set(_tokenize(q_norm))
        q_bigrams = set(_bigrams(q_tokens))

        scored: list[tuple[float, QAItem]] = []

        for item in self._items.values():
            if not item.is_active:
                continue
            if project and item.project_name != project:
                continue

            score = _score(q_norm, q_tokens, q_bigrams, item)
            if score >= threshold:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [qa for _, qa in scored[:top_k]]

    def deactivate(self, qa_id: str) -> bool:
        if qa_id in self._items:
            self._items[qa_id].is_active = False
            return True
        return False

    def list_by_project(self, project: str) -> list[QAItem]:
        return [i for i in self._items.values() if i.project_name == project and i.is_active]

    def count(self) -> int:
        return sum(1 for i in self._items.values() if i.is_active)


class QATool(AgentTool):

    def __init__(self, store: QAStore, threshold: float = 0.30):
        self._store = store
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "qa_lookup"

    @property
    def description(self) -> str:
        return (
            "Tra cứu bộ câu hỏi/trả lời chuẩn đã được biên soạn sẵn. "
            "Dùng khi câu hỏi phổ biến về dự án."
        )

    async def run(self, state: AgentState) -> ToolResult:
        results = self._store.search(
            query=state["raw_query"],
            project=state.get("project_name"),
            threshold=self._threshold,
        )

        if not results:
            return ToolResult(success=False, data=None, summary="Không match Q&A chuẩn")

        best = results[0]
        state["qa_result"] = best

        return ToolResult(
            success=True,
            data=best,
            summary=f"Match Q&A: {best.question[:60]!r}",
        )


# ─────────────────────────────────────────────────────────────────
# Text processing helpers
# ─────────────────────────────────────────────────────────────────

# Các từ viết tắt phổ biến trong bất động sản → expand để so sánh tốt hơn
_ABBREVIATIONS = {
    "cđt": "chủ đầu tư",
    "bđs": "bất động sản",
    "hđmb": "hợp đồng mua bán",
    "gpxd": "giấy phép xây dựng",
    "gpkd": "giấy phép kinh doanh",
    "gcn": "giấy chứng nhận",
    "qsd": "quyền sử dụng",
    "shr": "sổ hồng riêng",
    "sh": "sổ hồng",
    "pn": "phòng ngủ",
    "2pn": "hai phòng ngủ",
    "3pn": "ba phòng ngủ",
    "vp": "văn phòng",
}


def _normalize(text: str) -> str:
    """Normalize text: lowercase, remove diacritics partially, expand abbreviations."""
    text = text.lower().strip()
    # Xóa dấu câu thừa nhưng giữ khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Expand abbreviations
    tokens = text.split()
    expanded = [_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def _tokenize(text: str) -> list[str]:
    """Tách từ, lọc stop words ngắn."""
    _STOP = {"là", "và", "có", "không", "của", "cho", "được", "với", "về",
             "đã", "này", "các", "hay", "hoặc", "theo", "bằng", "tại",
             "trong", "ngoài", "trên", "dưới", "đến", "từ", "ra", "vào"}
    tokens = [t for t in text.split() if len(t) > 1 and t not in _STOP]
    return tokens


def _bigrams(tokens: list[str] | set[str]) -> list[tuple[str, str]]:
    tlist = list(tokens)
    return [(tlist[i], tlist[i + 1]) for i in range(len(tlist) - 1)]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _score(
    q_norm: str,
    q_tokens: set[str],
    q_bigrams: set[tuple],
    item: QAItem,
) -> float:
    """
    Multi-signal scoring:
      - exact match       → 1.0 (ngay lập tức trả về)
      - keyword hit       → +0.4 per keyword hit
      - token Jaccard     → weight 0.5
      - bigram Jaccard    → weight 0.3
      - substring         → +0.2 bonus
    """
    item_norm = _normalize(item.question)

    # 1. Exact match (sau normalize)
    if q_norm == item_norm:
        return 1.0

    # 2. Keyword exact match (keywords từ cột Keywords trong Excel)
    if item.keywords:
        kw_norm = [_normalize(kw) for kw in item.keywords]
        kw_hits = sum(1 for kw in kw_norm if kw and kw in q_norm)
        if kw_hits > 0:
            kw_score = min(0.4 * kw_hits, 0.8)   # max bonus 0.8
        else:
            kw_score = 0.0
    else:
        kw_score = 0.0

    # 3. Token Jaccard (unigram)
    item_tokens = set(_tokenize(item_norm))
    unigram_score = _jaccard(q_tokens, item_tokens)

    # 4. Bigram Jaccard
    item_bigrams = set(_bigrams(item_tokens))
    bigram_score = _jaccard(q_bigrams, item_bigrams) if q_bigrams and item_bigrams else 0.0

    # 5. Substring bonus: query chứa trong question hoặc ngược lại
    sub_bonus = 0.0
    if len(q_norm) > 5 and (q_norm in item_norm or item_norm in q_norm):
        sub_bonus = 0.25

    # Tổng hợp
    combined = (
        kw_score
        + unigram_score * 0.5
        + bigram_score * 0.3
        + sub_bonus
    )

    return min(combined, 1.0)  # clamp về [0, 1]