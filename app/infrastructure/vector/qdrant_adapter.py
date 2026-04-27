"""
app/infrastructure/vector/qdrant_adapter.py

Qdrant implementation của VectorPort.
Mọi Qdrant-specific type được giữ trong file này.
"""
from __future__ import annotations

import uuid

from app.core.interfaces.vector_port import (
    SearchFilter, SearchResult, VectorPoint, VectorPort,
)
from app.shared.errors.exceptions import VectorDBError
from app.shared.logging.logger import get_logger

log = get_logger(__name__)


class QdrantAdapter(VectorPort):

    def __init__(self, url: str, api_key: str, collection: str):
        self._url = url
        self._api_key = api_key
        self._collection = collection
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            if self._url == ":memory:":
                # Chế độ in-memory — không cần Docker/server, dùng khi dev/test
                self._client = AsyncQdrantClient(location=":memory:")
                log.info("qdrant_inmemory_mode")
            else:
                kwargs: dict = {"url": self._url}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                self._client = AsyncQdrantClient(**kwargs)
        return self._client

    async def ensure_collection(self, dimension: int) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams
            client = self._get_client()
            collections = await client.get_collections()
            existing = [c.name for c in collections.collections]
            if self._collection not in existing:
                await client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
                log.info("qdrant_collection_created", collection=self._collection)
        except Exception as e:
            raise VectorDBError(f"Cannot create collection: {e}") from e

    async def upsert(self, points: list[VectorPoint]) -> int:
        try:
            from qdrant_client.models import PointStruct
            client = self._get_client()
            structs = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=p.vector,
                    payload={**p.payload, "status": "active"},
                )
                for p in points
            ]
            await client.upsert(collection_name=self._collection, points=structs)
            log.info("qdrant_upsert", count=len(structs))
            return len(structs)
        except Exception as e:
            raise VectorDBError(f"Upsert failed: {e}") from e

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: SearchFilter,
    ) -> list[SearchResult]:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            client = self._get_client()

            must = [FieldCondition(key="status", match=MatchValue(value=filter.status))]
            if filter.project_name:
                must.append(
                    FieldCondition(key="project_name", match=MatchValue(value=filter.project_name))
                )
            if filter.doc_group:
                must.append(
                    FieldCondition(key="doc_group", match=MatchValue(value=filter.doc_group))
                )

            results = await client.search(
                collection_name=self._collection,
                query_vector=vector,
                query_filter=Filter(must=must),
                limit=top_k,
                with_payload=True,
            )
            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    text=r.payload.get("text", ""),
                    document_code=r.payload.get("document_code", ""),
                    document_name=r.payload.get("file_name", ""),
                    doc_group=r.payload.get("doc_group", ""),
                    project_name=r.payload.get("project_name", ""),
                    page_number=r.payload.get("page_number"),
                )
                for r in results
            ]
        except Exception as e:
            raise VectorDBError(f"Search failed: {e}") from e

    async def mark_superseded(self, document_code: str) -> int:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            client = self._get_client()
            scroll_filter = Filter(
                must=[
                    FieldCondition(key="document_code", match=MatchValue(value=document_code)),
                    FieldCondition(key="status", match=MatchValue(value="active")),
                ]
            )
            ids, offset = [], None
            while True:
                batch, offset = await client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=200,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                ids.extend(str(p.id) for p in batch)
                if offset is None:
                    break

            if ids:
                await client.set_payload(
                    collection_name=self._collection,
                    payload={"status": "superseded"},
                    points=ids,
                )
            log.info("qdrant_superseded", document_code=document_code, count=len(ids))
            return len(ids)
        except Exception as e:
            raise VectorDBError(f"mark_superseded failed: {e}") from e

    async def delete_by_document(self, document_code: str) -> None:
        try:
            from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
            client = self._get_client()
            await client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(
                    filter=Filter(must=[
                        FieldCondition(key="document_code", match=MatchValue(value=document_code))
                    ])
                ),
            )
            log.info("qdrant_deleted", document_code=document_code)
        except Exception as e:
            raise VectorDBError(f"delete failed: {e}") from e
