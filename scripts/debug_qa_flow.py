"""
scripts/debug_qa_flow.py

Debug script: kiểm tra toàn bộ luồng Q&A từ search → context → LLM.
Chạy: python scripts/debug_qa_flow.py

Bước kiểm tra:
  1. Kết nối Qdrant + xem collection qa_pairs có data không
  2. Embedding một câu hỏi mẫu → search qa_pairs → kiểm tra score
  3. Simulate toàn bộ luồng: QATool → RAGTool → context build
  4. In chi tiết từng bước để debug
"""
from __future__ import annotations

import asyncio
import os
import sys

# Thêm root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Câu hỏi test — đổi sang câu hỏi từ file Excel của bạn ──────────
TEST_QUESTION = "Chủ đầu tư (CĐT) dự án là ai? Tên pháp nhân chính xác trên giấy phép?"   # ← SỬA chỗ này
TEST_PROJECT  = None   # None = tất cả project, hoặc điền "Tên Dự Án"


# ── Color helpers ────────────────────────────────────────────────────
def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def cyan(s):  return f"\033[96m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"

def hr(title=""):
    line = "─" * 60
    if title:
        print(f"\n{cyan(line)}")
        print(f"  {bold(title)}")
        print(f"{cyan(line)}")
    else:
        print(f"{cyan(line)}")


async def main():
    print(bold("\n🔍 DEBUG Q&A RAG FLOW\n"))

    # ── Load settings ────────────────────────────────────────────────
    from app.core.config.settings import get_settings
    cfg = get_settings()

    hr("1. SETTINGS")
    print(f"  qdrant_url         = {cfg.qdrant_url}")
    print(f"  qdrant_collection  = {cfg.qdrant_collection}  (documents)")
    print(f"  qa_collection      = {cfg.qa_qdrant_collection}  (Q&A)")
    print(f"  qa_score_threshold = {cfg.qa_score_threshold}")
    print(f"  rag_score_threshold= {cfg.rag_score_threshold}")
    print(f"  embedding_model    = {cfg.embedding_model}")
    print(f"  embedding_dim      = {cfg.embedding_dimension}")

    # ── Embedder ─────────────────────────────────────────────────────
    hr("2. EMBEDDER")
    from app.infrastructure.llm.llm_factory import create_embed_provider
    embedder = create_embed_provider(cfg)
    print(f"  Provider: {type(embedder).__name__}, dim={embedder.dimension}")

    try:
        test_vec = await embedder.embed_one("test")
        print(green(f"  ✓ embed_one() OK — vector len={len(test_vec)}, first5={[round(x,4) for x in test_vec[:5]]}"))
    except Exception as e:
        print(red(f"  ✗ embed_one() FAILED: {e}"))
        return

    # ── Qdrant connections ────────────────────────────────────────────
    hr("3. QDRANT COLLECTIONS")
    from app.infrastructure.vector.qdrant_adapter import QdrantAdapter

    # Document collection
    doc_db = QdrantAdapter(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
    )
    # Q&A collection
    qa_db = QdrantAdapter(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qa_qdrant_collection,
    )

    try:
        from qdrant_client import AsyncQdrantClient
        if cfg.qdrant_url == ":memory:":
            client = AsyncQdrantClient(location=":memory:")
        else:
            client_kwargs: dict = {"url": cfg.qdrant_url}
            if cfg.qdrant_api_key:
                client_kwargs["api_key"] = cfg.qdrant_api_key
            client = AsyncQdrantClient(**client_kwargs)

        collections = await client.get_collections()
        existing = [c.name for c in collections.collections]
        print(f"  Existing collections: {existing}")

        for col_name in [cfg.qdrant_collection, cfg.qa_qdrant_collection]:
            if col_name in existing:
                info = await client.get_collection(col_name)
                count = info.points_count
                print(green(f"  ✓ '{col_name}': {count} points"))
                if count == 0:
                    print(yellow(f"    ⚠ Collection rỗng! Import dữ liệu trước khi test."))
            else:
                print(red(f"  ✗ '{col_name}': KHÔNG TỒN TẠI"))
    except Exception as e:
        print(red(f"  ✗ Qdrant connection FAILED: {e}"))
        return

    # ── Q&A Search ───────────────────────────────────────────────────
    hr(f"4. Q&A SEARCH — '{TEST_QUESTION[:50]}'")
    from app.agent.tools.qa_tool import QAVectorStore

    qa_store = QAVectorStore(
        vector_db=qa_db,
        embedder=embedder,
        threshold=cfg.qa_score_threshold,
    )

    try:
        # Thử với threshold = 0 để xem tất cả kết quả (debug)
        vec = await embedder.embed_one(TEST_QUESTION)
        from app.core.interfaces.vector_port import SearchFilter
        raw_results = await qa_db.search(
            vector=vec,
            top_k=5,
            filter=SearchFilter(project_name=TEST_PROJECT, status="active"),
        )

        print(f"\n  Raw search results (top 5, threshold=0):")
        if not raw_results:
            print(red("  ✗ Không có kết quả! qa_pairs collection có thể rỗng hoặc filter sai."))
        for i, r in enumerate(raw_results):
            score_color = green if r.score >= cfg.qa_score_threshold else yellow
            print(f"    [{i+1}] score={score_color(f'{r.score:.4f}')}  "
                  f"project={r.extra.get('project_name','?')}  "
                  f"question={r.extra.get('question', r.text)[:60]!r}")
            print(f"         answer={r.extra.get('answer', '[EMPTY!]')[:80]!r}")
            if not r.extra:
                print(red(f"         ⚠ extra={{}}: payload không được trả về! Bug trong qdrant_adapter.search()"))
            if not r.extra.get("answer"):
                print(red(f"         ⚠ answer rỗng trong payload!"))

        # Kết quả với threshold chuẩn
        items = await qa_store.search(TEST_QUESTION, project=TEST_PROJECT, top_k=3)
        print(f"\n  Filtered (threshold={cfg.qa_score_threshold}): {len(items)} matches")
        for item in items:
            print(green(f"    ✓ score={item.score:.4f} | Q: {item.question[:60]!r}"))
            print(f"       A: {item.answer[:100]!r}")

        if not items:
            print(red(f"  ✗ Không có match vượt threshold={cfg.qa_score_threshold}"))
            if raw_results:
                best = raw_results[0].score
                print(yellow(f"    Best score={best:.4f}. Xem xét giảm qa_score_threshold xuống dưới {best:.2f}"))

    except Exception as e:
        import traceback
        print(red(f"  ✗ Q&A search FAILED: {e}"))
        traceback.print_exc()
        return

    # ── Simulate QATool ──────────────────────────────────────────────
    hr("5. SIMULATE QATool.run()")
    from app.agent.state.agent_state import make_initial_state
    from app.agent.tools.qa_tool import QATool

    state = make_initial_state(
        session_id="debug-session",
        raw_query=TEST_QUESTION,
        project_name=TEST_PROJECT,
    )

    qa_tool = QATool(store=qa_store)
    try:
        result, call = await qa_tool.execute(state)
        print(f"  result.success = {green('True') if result.success else red('False')}")
        print(f"  result.summary = {result.summary!r}")
        print(f"  state['qa_hit'] = {state.get('qa_hit')}")
        print(f"  state['rag_results'] len = {len(state.get('rag_results', []))}")
        print(f"  state['sources'] len = {len(state.get('sources', []))}")

        rag_results = state.get("rag_results", [])
        for i, chunk in enumerate(rag_results):
            src = chunk.get("source_type", "?")
            print(f"    chunk[{i}] source_type={cyan(src)} text={chunk.get('text','')[:80]!r}")

        if not result.success:
            print(red("  ✗ QATool FAILED — Q&A context không được inject!"))
        else:
            print(green("  ✓ QATool OK — Q&A context đã inject vào rag_results"))
    except Exception as e:
        import traceback
        print(red(f"  ✗ QATool FAILED: {e}"))
        traceback.print_exc()

    # ── Simulate RAGTool ─────────────────────────────────────────────
    hr("6. SIMULATE RAGTool.run() (document search)")
    from app.agent.tools.rag_tool import RAGTool

    rag_tool = RAGTool(
        vector_db=doc_db,
        embedder=embedder,
        score_threshold=cfg.rag_score_threshold,
        top_k=5,
    )
    try:
        result2, call2 = await rag_tool.execute(state)
        print(f"  result.success = {green('True') if result2.success else yellow('False (no doc match)')}")
        print(f"  result.summary = {result2.summary!r}")
        print(f"  state['rag_results'] len = {len(state.get('rag_results', []))} (Q&A + doc combined)")
        print(f"  state['fallback'] = {state.get('fallback')}")

        rag_results_after = state.get("rag_results", [])
        qa_chunks  = [r for r in rag_results_after if r.get("source_type") == "qa"]
        doc_chunks = [r for r in rag_results_after if r.get("source_type") != "qa"]
        print(f"  → qa_chunks={len(qa_chunks)}, doc_chunks={len(doc_chunks)}")

        if qa_chunks and len(rag_results_after) >= len(qa_chunks):
            print(green("  ✓ Q&A context được preserve sau RAG search"))
        else:
            print(red("  ✗ Q&A context BỊ MẤT sau RAG search — Bug overwrite!"))

    except Exception as e:
        import traceback
        print(red(f"  ✗ RAGTool FAILED: {e}"))
        traceback.print_exc()

    # ── Context build ────────────────────────────────────────────────
    hr("7. CONTEXT BUILD (SynthesizerNode._build_context)")
    from app.agent.nodes.synthesizer_node import SynthesizerNode

    import json

    def build_context_debug(state):
        parts = []
        rag = state.get("rag_results", [])
        if rag:
            qa_chunks  = [r for r in rag if r.get("source_type") == "qa"]
            doc_chunks = [r for r in rag if r.get("source_type") != "qa"]
            if qa_chunks:
                qa_text = "\n\n".join(r["text"] for r in qa_chunks)
                parts.append(f"=== CÂU TRẢ LỜI CHUẨN (Q&A) ===\n{qa_text}")
            if doc_chunks:
                doc_text = "\n\n".join(
                    f"[Tài liệu: {r.get('document_name','')} — {r.get('doc_group','')}]\n{r['text']}"
                    for r in doc_chunks
                )
                parts.append(f"=== TÀI LIỆU DỰ ÁN ===\n{doc_text}")
        sales = state.get("sales_data", {})
        if sales:
            parts.append(f"=== DỮ LIỆU BÁN HÀNG ===\n{json.dumps(sales, ensure_ascii=False)}")
        return "\n\n".join(parts)

    context = build_context_debug(state)
    if context.strip():
        print(green(f"  ✓ Context built ({len(context)} chars)"))
        print(f"\n  Preview (first 400 chars):\n")
        print(f"  {context[:400]!r}")
    else:
        print(red("  ✗ Context RỖNG — LLM sẽ trả về fallback message!"))
        print(red("  Kiểm tra các bước trên để tìm điểm thất bại."))

    # ── Summary ──────────────────────────────────────────────────────
    hr("SUMMARY")
    qa_items_found = len([r for r in state.get("rag_results", []) if r.get("source_type") == "qa"])
    doc_items_found = len([r for r in state.get("rag_results", []) if r.get("source_type") != "qa"])
    context_ok = bool(context.strip())

    checks = [
        ("Qdrant qa_pairs có data",      bool(raw_results)),
        ("Q&A search score >= threshold", bool(items)),
        ("QATool inject vào rag_results", qa_items_found > 0),
        ("Q&A context không bị overwrite", qa_items_found > 0),
        ("Context cho LLM không rỗng",   context_ok),
    ]
    all_ok = True
    for label, ok in checks:
        icon = green("✓") if ok else red("✗")
        print(f"  {icon}  {label}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print(green(bold("  🎉 Toàn bộ luồng Q&A OK! Nếu vẫn sai, kiểm tra LLM response.")))
    else:
        print(red(bold("  ❌ Có vấn đề trong luồng — xem chi tiết từng bước ở trên.")))
    print()


if __name__ == "__main__":
    asyncio.run(main())
