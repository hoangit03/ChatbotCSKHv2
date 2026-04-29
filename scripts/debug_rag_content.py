import asyncio
import os
import sys
from app.core.config.settings import get_settings
from app.infrastructure.llm.llm_factory import create_embed_provider
from app.infrastructure.vector.qdrant_adapter import QdrantAdapter
from app.core.interfaces.vector_port import SearchFilter

# --- CẤU HÌNH CÂU HỎI CỦA BẠN ---
QUERY = "Tôi cần thanh toán bao nhiêu ban đầu?"
PROJECT = None # Để None để tìm toàn bộ

async def debug_rag():
    cfg = get_settings()
    embedder = create_embed_provider(cfg)
    vdb = QdrantAdapter(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
    )

    print(f"\n🔍 Đang tra cứu tài liệu: '{QUERY}'")
    
    # 1. Tạo vector từ câu hỏi
    vec = await embedder.embed_one(QUERY)
    
    # 2. Tìm kiếm trong Qdrant (lấy top 5 kết quả bất kể điểm số)
    results = await vdb.search(
        vector=vec,
        top_k=5,
        filter=SearchFilter(project_name=PROJECT, status="active")
    )
    
    if not results:
        print("\n❌ KHÔNG TÌM THẤY BẤT KỲ ĐOẠN VĂN BẢN NÀO!")
        print("-> Có thể quá trình Upload file .docx bị lỗi hoặc collection rỗng.")
        return

    print(f"\n✅ Tìm thấy {len(results)} đoạn văn bản tiềm năng:")
    print("-" * 50)
    
    for i, r in enumerate(results):
        print(f"[{i+1}] Score: {r.score:.4f}")
        print(f"    Dự án: {r.project_name}")
        print(f"    File: {r.document_name}")
        print(f"    Nhóm: {r.doc_group}")
        print(f"    Nội dung: {r.text[:200]}...")
        print("-" * 50)

    best_score = results[0].score
    if best_score < cfg.rag_score_threshold:
        print(f"\n⚠️ CẢNH BÁO: Điểm cao nhất ({best_score:.4f}) thấp hơn ngưỡng threshold ({cfg.rag_score_threshold})")
        print(f"-> Hệ thống đã lọc bỏ kết quả này trong Chat API.")
        print(f"-> Giải pháp: Giảm RAG_SCORE_THRESHOLD trong .env xuống khoảng {best_score - 0.05:.2f}")

if __name__ == "__main__":
    asyncio.run(debug_rag())
