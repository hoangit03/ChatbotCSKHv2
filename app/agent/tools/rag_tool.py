"""
app/agent/tools/rag_tool.py

Tool: Tìm kiếm tài liệu qua RAG (vector similarity search).
Dùng cho: Customer Support — tìm thông tin từ PDF/DOCX/Image đã upload.
"""
from __future__ import annotations

from app.agent.state.agent_state import AgentState, SourceRef
from app.agent.tools.base_tool import AgentTool, ToolResult
from app.core.interfaces.llm_port import EmbedPort
from app.core.interfaces.vector_port import SearchFilter, VectorPort
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class RAGTool(AgentTool):

    def __init__(
        self,
        vector_db: VectorPort,
        embedder: EmbedPort,
        score_threshold: float = 0.70,
        top_k: int = 5,
    ):
        self._vdb = vector_db
        self._embed = embedder
        self._threshold = score_threshold
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Tìm kiếm thông tin từ tài liệu dự án (brochure, FAQ, pháp lý, "
            "mặt bằng, tiến độ). Dùng khi câu hỏi về thông tin dự án, tiện ích, "
            "vị trí, pháp lý, chính sách bán hàng."
        )

    async def run(self, state: AgentState) -> ToolResult:
        query = state["raw_query"]
        project = state.get("project_name")

        query_vec = await self._embed.embed_one(query)
        results = await self._vdb.search(
            vector=query_vec,
            top_k=self._top_k,
            filter=SearchFilter(project_name=project, status="active"),
        )

        relevant = [r for r in results if r.score >= self._threshold]

        if not relevant:
            return ToolResult(
                success=False,
                data=[],
                summary=f"Không tìm thấy tài liệu phù hợp (threshold={self._threshold})",
            )

        # APPEND document chunks — không overwrite Q&A chunks đã inject trước đó
        doc_chunks = [
            {
                "score": r.score,
                "text": r.text,
                "source_type": "document",      # phân biệt với qa chunks
                "document_code": r.document_code,
                "document_name": r.document_name,
                "doc_group": r.doc_group,
                "project_name": r.project_name,
                "page_number": r.page_number,
            }
            for r in relevant
        ]
        existing_rag = state.get("rag_results") or []
        state["rag_results"] = existing_rag + doc_chunks

        # APPEND source refs — không overwrite Q&A sources
        doc_sources = [
            SourceRef(
                document_code=r.document_code,
                document_name=r.document_name,
                doc_group=r.doc_group,
                excerpt=r.text[:200],
                page=r.page_number,
            )
            for r in relevant
        ]
        existing_sources = state.get("sources") or []
        state["sources"] = existing_sources + doc_sources

        return ToolResult(
            success=True,
            data=doc_chunks,
            summary=f"Tìm được {len(relevant)} đoạn tài liệu liên quan (top score={relevant[0].score:.2f})",
        )