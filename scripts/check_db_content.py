import asyncio
from app.core.config.settings import get_settings
from app.infrastructure.vector.qdrant_adapter import QdrantAdapter
from app.core.interfaces.vector_port import SearchFilter

async def check_database():
    cfg = get_settings()
    qa_db = QdrantAdapter(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qa_qdrant_collection,
    )
    
    print(f"\n🔍 Đang kiểm tra collection: {cfg.qa_qdrant_collection}")
    
    # Lấy thử 10 bản ghi để xem project_name
    results, _ = await qa_db.scroll(
        filter=SearchFilter(status="active"),
        limit=10
    )
    
    if not results:
        print("❌ Database Q&A rỗng! Hãy chạy import trước.")
        return

    projects = set()
    print("\n--- 10 bản ghi đầu tiên ---")
    for r in results:
        p_name = r.extra.get('project_name')
        projects.add(p_name)
        print(f"ID: {r.id} | Project: '{p_name}' | Q: {r.text[:50]}...")
    
    print("\n--- Danh sách Project Name bạn có thể dùng để chat ---")
    for p in projects:
        print(f"👉 '{p}'")
    print("\nLưu ý: Khi gọi API Chat, bạn PHẢI truyền project_name khớp chính xác với danh sách trên.")

if __name__ == "__main__":
    asyncio.run(check_database())
