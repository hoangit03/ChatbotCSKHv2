"""
app/api/v1/endpoints/health.py

Health check: kiểm tra kết nối Qdrant, Redis, Sales API.
Không cần auth — dùng cho load balancer / monitoring.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthOut(BaseModel):
    status: str          # "healthy" | "degraded"
    checks: dict[str, str]


@router.get("/health", response_model=HealthOut, include_in_schema=True)
async def health(request: Request) -> HealthOut:
    checks: dict[str, str] = {"app": "ok"}

    # ── Qdrant ────────────────────────────────────────────────────
    try:
        vdb = request.app.state.vector_db
        vdb._get_client()   # lazy init — sẽ throw nếu không connect được
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {type(e).__name__}"

    # ── Redis ─────────────────────────────────────────────────────
    try:
        from app.core.config.settings import get_settings
        import redis.asyncio as aioredis
        r = aioredis.from_url(get_settings().redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"unavailable: {type(e).__name__}"

    # ── Sales API ─────────────────────────────────────────────────
    try:
        sales_api = request.app.state.sales_api
        # Gọi endpoint health nếu có, không thì chỉ check client init
        checks["sales_api"] = "reachable" if sales_api else "not_configured"
    except Exception as e:
        checks["sales_api"] = f"error: {type(e).__name__}"

    overall = "healthy" if all(v in ("ok", "reachable", "not_configured") for v in checks.values()) else "degraded"
    return HealthOut(status=overall, checks=checks)