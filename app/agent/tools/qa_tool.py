"""
app/agent/tools/qa_tool.py

Tool: Tra cứu bộ Q&A chuẩn (import từ Excel) qua Qdrant semantic search.
Ưu tiên gọi trước RAG — nhanh và chính xác nhất.

Kiến trúc v4 (RAG-style):
  - QAVectorStore: lưu Q&A vào Qdrant collection riêng ("qa_pairs")
    Tách hoàn toàn với document collection ("realestate_kb")
  - Scale: Qdrant xử lý triệu Q&A pairs, persist qua restart
  - Upsert idempotent: deterministic UUID từ (project, question)
    → Import nhiều file / re-import không duplicate

  Luồng khi có Q&A hit (RAG-style v4):
    embed(query) → Qdrant search qa_pairs → score ≥ threshold
    → inject vào state["rag_results"] như context chunk ưu tiên cao nhất
    → SynthesizerNode nhận context → LLM tổng hợp → trả lời tự nhiên
    (không skip LLM — câu trả lời tự nhiên hơn, có thể combine với doc RAG)
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.agent.state.agent_state import AgentState, SourceRef
from app.agent.tools.base_tool import AgentTool, ToolResult
from app.core.interfaces.llm_port import EmbedPort
from app.core.interfaces.vector_port import SearchFilter, VectorPoint, VectorPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Domain model
# ─────────────────────────────────────────────────────────────────

@dataclass
class QAItem:
    """Một cặp Q&A chuẩn, map 1-1 với 1 row trong Qdrant qa_pairs."""
    id: str
    project_name: str
    question: str
    answer: str
    keywords: list[str] = field(default_factory=list)
    doc_group: Optional[str] = None
    is_active: bool = True
    score: float = 0.0          # similarity score từ vector search

    @staticmethod
    def make_id(project_name: str, question: str) -> str:
        """Deterministic UUID từ (project, question) — upsert idempotent."""
        key = f"{project_name}::{question.strip().lower()}"
        return str(uuid.UUID(hashlib.md5(key.encode("utf-8")).hexdigest()))


# ─────────────────────────────────────────────────────────────────
# QAVectorStore — Qdrant-backed, dedicated "qa_pairs" collection
# ─────────────────────────────────────────────────────────────────

class QAVectorStore:
    """
    Q&A store backed by Qdrant dedicated collection.

    Design principles:
    - Collection "qa_pairs" tách hoàn toàn với "realestate_kb" (document RAG)
    - Upsert idempotent: deterministic UUID từ (project, question)
      → Import cùng file nhiều lần / nhiều file khác nhau → không duplicate
    - Batch embed: 1 API call cho toàn bộ batch (tiết kiệm chi phí embedding)
    - Scale: Qdrant xử lý triệu Q&A pairs, không tốn RAM server
    - Scroll API: list_by_project không cần vector search
    """

    def __init__(
        self,
        vector_db: VectorPort,
        embedder: EmbedPort,
        threshold: float = 0.75,
    ) -> None:
        self._vdb = vector_db
        self._embed = embedder
        self._threshold = threshold

    async def search(
        self,
        query: str,
        project: str | None = None,
        top_k: int = 3,
    ) -> list[QAItem]:
        """
        Semantic search trong Qdrant qa_pairs collection.
        Trả danh sách QAItem đã vượt ngưỡng cosine similarity, kèm score.
        # Nếu project rỗng hoặc là placeholder từ Swagger, bỏ qua filter dự án để tìm rộng hơn"""
        search_project = project
        if not project or project.lower() in ["string", "", "none", "default"]:
            search_project = None

        vec = await self._embed.embed_one(query)
        results = await self._vdb.search(

            vector=vec,
            top_k=top_k + 2,        # lấy dư để lọc threshold
            filter=SearchFilter(project_name=search_project, status="active"),
        )


        items: list[QAItem] = []
        for r in results:
            if r.score < self._threshold:
                continue
            payload = r.extra       # full payload từ QdrantAdapter (đã fix)
            items.append(QAItem(
                id=r.id,
                project_name=payload.get("project_name", r.project_name),
                question=payload.get("question", r.text),
                answer=payload.get("answer", ""),
                keywords=payload.get("keywords", []),
                doc_group=payload.get("doc_group") or None,
                score=r.score,
            ))
            if len(items) >= top_k:
                break

        return items

    async def bulk_add(self, items: list[QAItem]) -> int:
        """
        Batch embed + upsert tất cả items vào Qdrant.
        - 1 API call embedding cho toàn batch → tiết kiệm cost
        - ID deterministic từ (project, question) → re-import an toàn
        - Qdrant upsert = insert or update → idempotent
        """
        if not items:
            return 0

        questions = [i.question for i in items]
        try:
            vectors = await self._embed.embed(questions)
        except Exception as e:
            log.error("qa_bulk_embed_failed", error=str(e), count=len(items))
            raise

        points = [
            VectorPoint(
                id=i.id,                    # deterministic → idempotent
                vector=v,
                payload={
                    "question":      i.question,
                    "answer":        i.answer,
                    "project_name":  i.project_name,
                    "keywords":      i.keywords,
                    "doc_group":     i.doc_group or "",
                    "text":          i.question,   # compat với SearchResult.text
                    "document_code": i.project_name,   # dùng cho delete_by_document
                    "type":          "qa",
                },
            )
            for i, v in zip(items, vectors)
        ]

        count = await self._vdb.upsert(points)
        log.info(
            "qa_bulk_add_done",
            count=count,
            project=items[0].project_name if items else "",
        )
        return count

    async def delete_by_project(self, project_name: str) -> None:
        """Xóa toàn bộ Q&A của một project (dùng khi re-import fresh)."""
        await self._vdb.delete_by_document(project_name)

    async def list_by_project(
        self,
        project_name: str,
        limit: int = 500,
    ) -> list[QAItem]:
        """
        List Q&A của một project — dùng cho API GET /qa.
        Dùng Qdrant scroll (không cần vector) → chính xác, không cần embedding.
        Hỗ trợ limit lớn cho project có nhiều Q&A.
        """
        results, _ = await self._vdb.scroll(
            filter=SearchFilter(project_name=project_name, status="active"),
            limit=limit,
        )
        return [
            QAItem(
                id=r.id,
                project_name=r.extra.get("project_name", project_name),
                question=r.extra.get("question", r.text),
                answer=r.extra.get("answer", ""),
                keywords=r.extra.get("keywords", []),
                doc_group=r.extra.get("doc_group") or None,
            )
            for r in results
        ]

    async def deactivate(self, qa_id: str) -> bool:
        """Soft-delete: đặt status='superseded' trong Qdrant payload."""
        try:
            from app.infrastructure.vector.qdrant_adapter import QdrantAdapter
            if isinstance(self._vdb, QdrantAdapter):
                client = self._vdb._get_client()
                from qdrant_client.models import PointIdsList
                await client.set_payload(
                    collection_name=self._vdb._collection,
                    payload={"status": "superseded"},
                    points=PointIdsList(points=[qa_id]),
                )
            return True
        except Exception as e:
            log.error("qa_deactivate_failed", qa_id=qa_id, error=str(e))
        return False


# ─────────────────────────────────────────────────────────────────
# QATool — Agent tool interface (RAG-style v4)
# ─────────────────────────────────────────────────────────────────

class QATool(AgentTool):
    """
    Agent tool: tra cứu Q&A qua Qdrant semantic search.

    RAG-style v4: Khi có Q&A match, inject answer vào state["rag_results"]
    như một context chunk ưu tiên cao (thay vì set final_answer trực tiếp).
    SynthesizerNode sẽ nhận context → LLM tổng hợp câu trả lời tự nhiên.

    Ưu điểm so với direct-answer:
    - Câu trả lời tự nhiên hơn, không cứng nhắc
    - LLM có thể combine Q&A + document RAG context
    - Dễ audit và debug qua sources list
    """

    def __init__(self, store: QAVectorStore):
        self._store = store

    @property
    def name(self) -> str:
        return "qa_lookup"

    @property
    def description(self) -> str:
        return (
            "Tra cứu bộ câu hỏi/trả lời chuẩn đã được biên soạn sẵn. "
            "Dùng khi câu hỏi phổ biến về dự án bất động sản."
        )

    async def run(self, state: AgentState) -> ToolResult:
        results = await self._store.search(
            query=state["raw_query"],
            project=state.get("project_name"),
            top_k=3,
        )

        if not results:
            return ToolResult(
                success=False,
                data=None,
                summary="Không match Q&A chuẩn",
            )

        # ── Inject Q&A answers vào rag_results (RAG-style) ──────────
        # Q&A chunks đặt đầu danh sách → LLM ưu tiên sử dụng
        qa_chunks = [
            {
                "score":         item.score,
                "text":          f"Câu hỏi: {item.question}\nTrả lời: {item.answer}",
                "source_type":   "qa",          # phân biệt với document chunks
                "document_code": f"qa::{item.id}",
                "document_name": "Bộ Q&A chuẩn",
                "doc_group":     item.doc_group or "qa",
                "project_name":  item.project_name,
                "page_number":   None,
            }
            for item in results
        ]

        # Prepend Q&A context (ưu tiên cao hơn document RAG)
        existing_rag = state.get("rag_results", [])
        state["rag_results"] = qa_chunks + existing_rag

        # Gắn sources để hiển thị cho user
        qa_sources = [
            SourceRef(
                document_code=f"qa::{item.id}",
                document_name="Bộ Q&A chuẩn",
                doc_group=item.doc_group or "qa",
                excerpt=item.answer[:200],
                page=None,
            )
            for item in results
        ]
        existing_sources = state.get("sources", [])
        state["sources"] = qa_sources + existing_sources

        # Giữ qa_result cho backward compat / logging
        state["qa_result"] = results[0]
        state["qa_hit"] = True

        best_score = results[0].score
        return ToolResult(
            success=True,
            data=results,
            summary=(
                f"Match {len(results)} Q&A (top score={best_score:.2f}): "
                f"{results[0].question[:60]!r}"
            ),
        )