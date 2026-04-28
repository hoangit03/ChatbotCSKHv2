"""
app/main.py

FastAPI application — entry point và Dependency Injection wiring.

Đây là nơi DUY NHẤT tạo ra concrete implementations và inject vào use cases.
Mọi nơi khác chỉ biết interfaces.

Thứ tự khởi tạo (lifespan):
  1. Settings
  2. Logging
  3. Redis connection pool
  4. Infrastructure adapters (LLM, Vector, Storage, SalesAPI, Parsers)
  5. Tools + ToolRegistry
  6. Agent graph (LangGraph)
  7. Use cases (inject dependencies)
  8. API key store (seed dev key nếu dev mode)
  9. Store vào app.state để endpoints lấy ra
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware.auth import APIKeyMiddleware, APIKeyStore
from app.api.v1.endpoints import chat, document, health, qa_import
from app.core.config.settings import get_settings
from app.shared.errors.exceptions import AppError
from app.shared.logging.logger import get_logger, setup_logging

cfg = get_settings()
setup_logging(level=cfg.debug and "DEBUG" or "INFO", fmt="text" if cfg.debug else "json")
log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# LIFESPAN — khởi tạo tất cả dependencies
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app_starting", env=cfg.app_env, debug=cfg.debug)

    # ── 1. Redis connection pool ───────────────────────────────────
    import redis.asyncio as aioredis
    redis_pool = aioredis.ConnectionPool.from_url(
        cfg.redis_url,
        decode_responses=True,
        max_connections=20,
    )
    app.state.redis_pool = redis_pool
    log.info("redis_pool_ready", url=cfg.redis_url.split("@")[-1])  # log host only

    # ── 2. LLM providers ──────────────────────────────────────────
    from app.infrastructure.llm.llm_factory import create_chat_provider, create_embed_provider
    llm      = create_chat_provider(cfg)
    embedder = create_embed_provider(cfg)
    log.info("llm_ready", provider=llm.provider_name, model=llm.model_name)

    # ── 3. Vector DB ──────────────────────────────────────────────
    from app.infrastructure.vector.qdrant_adapter import QdrantAdapter
    vector_db = QdrantAdapter(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qdrant_collection,
    )
    await vector_db.ensure_collection(dimension=cfg.embedding_dimension)
    app.state.vector_db = vector_db

    # ── 4. Storage ────────────────────────────────────────────────
    from app.infrastructure.storage.local_storage import LocalStorageAdapter
    storage = LocalStorageAdapter(base_path=cfg.storage_path)

    # ── 5. Parser registry (OCP: đăng ký parser, không sửa code cũ) ──
    from app.core.interfaces.parser_port import ParserRegistry
    from app.infrastructure.parser.extractors.pdf_parser import PDFParser
    from app.infrastructure.parser.extractors.docx_parser import DocxParser
    from app.infrastructure.parser.extractors.excel_parser import ExcelParser
    from app.infrastructure.parser.extractors.image_parser import ImageParser

    parsers = ParserRegistry()
    parsers.register(PDFParser())
    parsers.register(DocxParser())
    parsers.register(ExcelParser())
    parsers.register(ImageParser())

    # ── 6. Sales API ──────────────────────────────────────────────
    from app.infrastructure.sql_api.sales_api_adapter import SalesAPIAdapter
    sales_api = SalesAPIAdapter(
        base_url=cfg.sales_api_base_url,
        api_key=cfg.sales_api_key,
        timeout=cfg.sales_api_timeout,
        max_retries=cfg.sales_api_max_retries,
    )
    app.state.sales_api = sales_api

    # ── 7. Q&A Vector Store (collection riêng: "qa_pairs") ────────
    # Tách hoàn toàn với realestate_kb → scale độc lập, không ảnh hưởng nhau
    from app.agent.tools.qa_tool import QAVectorStore
    qa_vector_db = QdrantAdapter(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection=cfg.qa_qdrant_collection,
    )
    await qa_vector_db.ensure_collection(dimension=cfg.embedding_dimension)
    qa_store = QAVectorStore(
        vector_db=qa_vector_db,
        embedder=embedder,
        threshold=cfg.qa_score_threshold,
    )
    app.state.qa_store = qa_store

    # ── 8. Tool Registry (OCP: thêm tool = register(), không sửa gì) ──
    from app.agent.tools.base_tool import ToolRegistry
    from app.agent.tools.qa_tool import QATool
    from app.agent.tools.rag_tool import RAGTool
    from app.agent.tools.sales_tool import (
        AvailabilityTool,
        BookingIntentTool,
        InventoryTool,
        PaymentTool,
        UnitSearchTool,
    )

    registry = ToolRegistry()
    registry.register(QATool(store=qa_store))   # threshold đã cấu hình trong QAVectorStore
    registry.register(RAGTool(
        vector_db=vector_db,
        embedder=embedder,
        score_threshold=cfg.rag_score_threshold,
        top_k=5,
    ))

    # Chỉ đăng ký Sales tools khi API được cấu hình thực tế
    # (tránh timeout 10s/call khi Sales API chưa sẵn sàng)
    if cfg.sales_api_configured:
        registry.register(AvailabilityTool(api=sales_api))
        registry.register(InventoryTool(api=sales_api))
        registry.register(PaymentTool(api=sales_api))
        registry.register(UnitSearchTool(api=sales_api))
        registry.register(BookingIntentTool(api=sales_api))
        log.info("sales_tools_registered")
    else:
        log.warning(
            "sales_tools_disabled",
            reason="SALES_API_BASE_URL is placeholder or SALES_API_ENABLED=false",
            note="Cấu hình SALES_API_BASE_URL + SALES_API_KEY trong .env khi có API backend",
        )

    log.info("tools_registered", tools=registry.names())

    # ── 9. Agent graph ────────────────────────────────────────────
    from app.agent.graph.agent_graph import build_agent_graph
    agent_graph = build_agent_graph(
        llm=llm,
        tool_registry=registry,
        max_iterations=cfg.agent_max_iterations,
    )

    # ── 10. Use Cases ─────────────────────────────────────────────
    from app.application.usecases.upload_document import UploadDocumentUseCase
    from app.application.usecases.handle_chat import HandleChatUseCase
    from app.application.usecases.import_qa import ImportQAUseCase

    app.state.upload_doc_uc  = UploadDocumentUseCase(
        cfg=cfg,
        parser_registry=parsers,
        embedder=embedder,
        vector_db=vector_db,
        storage=storage,
    )
    app.state.handle_chat_uc = HandleChatUseCase(agent_graph=agent_graph)
    import_qa_uc = ImportQAUseCase(qa_store=qa_store)
    app.state.import_qa_uc  = import_qa_uc

    # ── 10.5. Auto-load Q&A Excel khi khởi động ───────────────────────────
    # Đặt QA_AUTOLOAD_FILE=./data/your_qa.xlsx trong .env để kích hoạt
    if cfg.qa_autoload_file:
        from pathlib import Path
        from app.application.usecases.import_qa import ImportQARequest
        _qa_path = Path(cfg.qa_autoload_file)
        if _qa_path.exists():
            try:
                _qa_result = await import_qa_uc.execute(
                    ImportQARequest(
                        file_bytes=_qa_path.read_bytes(),
                        file_name=_qa_path.name,
                        project_name=cfg.qa_autoload_project,
                    )
                )
                log.info(
                    "qa_autoload_done",
                    file=cfg.qa_autoload_file,
                    project=cfg.qa_autoload_project,
                    imported=_qa_result.imported,
                    skipped=_qa_result.skipped,
                    errors=len(_qa_result.errors),
                )
                if _qa_result.errors:
                    for err in _qa_result.errors[:5]:
                        log.warning("qa_autoload_row_error", detail=err)
            except Exception as _e:
                log.error("qa_autoload_failed", error=str(_e))
        else:
            log.warning(
                "qa_autoload_file_not_found",
                path=cfg.qa_autoload_file,
                note="Kiểm tra đường dẫn trong QA_AUTOLOAD_FILE",
            )


    # ── 11. API Key Store ─────────────────────────────────────────
    api_key_store = APIKeyStore()
    if not cfg.is_production:
        api_key_store.register(
            raw_key=cfg.dev_api_key,   # đọc từ DEV_API_KEY trong .env
            user_id="dev",
            role="admin",
        )
        log.warning(
            "dev_api_key_active",
            key_preview=cfg.dev_api_key[:8] + "...",
            note="XOÁ key này trước khi deploy production!",
        )
    app.state.api_key_store = api_key_store

    log.info("app_ready", env=cfg.app_env)
    yield

    # ── Cleanup ───────────────────────────────────────────────────
    await sales_api.close()
    await redis_pool.aclose()
    log.info("app_shutdown")



# ─────────────────────────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Real Estate Chatbot — Agent RAG",
    version="1.0.0",
    description="""
## Chatbot AI Bất Động Sản

Hệ thống kết hợp **Customer Support** (RAG từ tài liệu) và **Sales** (dữ liệu real-time).

### Authentication
Mọi request cần header: `X-API-Key: <your-key>`

### Agent Flow
```
Câu hỏi → Intent Classifier → Support Node (RAG + Q&A)
                             → Sales Node  (API backend)
                             → Synthesizer (LLM) → Trả lời
```
    """,
    docs_url="/docs"        if not cfg.is_production else None,
    redoc_url="/redoc"      if not cfg.is_production else None,
    openapi_url="/openapi.json" if not cfg.is_production else None,
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────
# MIDDLEWARE (thứ tự: ngoài → trong)
# ─────────────────────────────────────────────────────────────────

# 1. CORS
_origins = (
    ["*"]
    if not cfg.is_production
    else [
        "https://your-frontend.company.com",   # ← đổi theo domain thực
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# 2. Rate limiting — functional middleware đọc Redis pool từ app.state
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    from app.api.middleware.rate_limit import check_rate_limit
    return await check_rate_limit(request, call_next)

# 3. Auth — instance tạo 1 lần, lazy-load store từ app.state mỗi request
_auth_mw = APIKeyMiddleware()

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    return await _auth_mw.dispatch(request, call_next)


# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────

API_V1 = "/api/v1"
app.include_router(health.router)                               # /health
app.include_router(chat.router,       prefix=API_V1)           # /api/v1/chat
app.include_router(document.router,   prefix=API_V1)           # /api/v1/documents
app.include_router(qa_import.router,  prefix=API_V1)           # /api/v1/qa


# ─────────────────────────────────────────────────────────────────
# ROOT + GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"service": "Real Estate Chatbot", "version": "1.0.0", "status": "running"}


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    log.warning(
        "app_error",
        code=exc.code,
        message=exc.message,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.code, "detail": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log.error(
        "unhandled_error",
        error=type(exc).__name__,
        detail=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "detail": "Lỗi hệ thống. Vui lòng thử lại sau."},
    )


# ─────────────────────────────────────────────────────────────────
# SWAGGER UI — Security scheme (nút "Authorize 🔒")
# ─────────────────────────────────────────────────────────────────

def _custom_openapi():
    """
    Thêm ApiKeyAuth vào OpenAPI spec để Swagger UI hiện nút "Authorize".
    User nhập key 1 lần → tự động gắn vào mọi request trong Swagger.
    """
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Khai báo security scheme kiểu API Key truyền qua header
    schema.setdefault("components", {})["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": (
                f"**Dev key (development only):** `{cfg.dev_api_key}`\n\n"
                "Nhập vào ô **Value** rồi bấm **Authorize**."
            ),
        }
    }

    # Áp dụng global cho tất cả endpoint
    schema["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi  # type: ignore[method-assign]
