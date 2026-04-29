import asyncio
from app.core.config.settings import get_settings
from qdrant_client import AsyncQdrantClient

async def clear_rag_collection():
    cfg = get_settings()
    client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    
    print(f"⚠️ Đang xóa toàn bộ dữ liệu trong collection: {cfg.qdrant_collection}...")
    
    # Xóa và tạo mới collection
    await client.recreate_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config={"size": cfg.embedding_dimension, "distance": "Cosine"}
    )
    
    print("✅ Đã làm sạch collection. Bây giờ bạn có thể upload lại file.")

if __name__ == "__main__":
    asyncio.run(clear_rag_collection())
