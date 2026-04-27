"""
app/api/middleware/rate_limit.py

Rate limiting dùng Redis Sorted Set (sliding window).
Tách giới hạn riêng cho chat vs upload.
Fail-open nếu Redis không kết nối được (không block service).

FIX (v1.1):
  - Không tạo Redis connection mới mỗi request.
  - Dùng ConnectionPool đã khởi tạo trong lifespan (lưu tại app.state.redis_pool).
  - Cung cấp check_rate_limit() dưới dạng functional middleware thay vì
    BaseHTTPMiddleware để có thể access app.state.
"""
from __future__ import annotations

import time

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.config.settings import get_settings
from app.shared.logging.logger import get_logger

log = get_logger(__name__)
_cfg = get_settings()

# (path_prefix, limit, window_seconds, bucket_prefix)
_RULES: list[tuple[str, int, int, str]] = [
    ("/api/v1/chat",      _cfg.rate_limit_chat,   60,   "rl:chat"),
    ("/api/v1/documents", _cfg.rate_limit_upload, 3600, "rl:upload"),
    ("/api/v1/qa",        _cfg.rate_limit_upload, 3600, "rl:qa"),
]


async def check_rate_limit(request: Request, call_next):
    """
    Functional middleware — gọi từ @app.middleware("http") trong main.py.
    Đọc redis_pool từ app.state (được tạo trong lifespan).
    """
    path = request.url.path
    ip   = _ip(request)

    for prefix, limit, window, bucket_prefix in _RULES:
        if path.startswith(prefix):
            bucket = f"{bucket_prefix}:{ip}"
            # Lấy pool từ app.state (fail gracefully nếu chưa có)
            pool = getattr(request.app.state, "redis_pool", None)
            allowed = await _check(bucket, limit, window, pool)
            if not allowed:
                log.warning(
                    "rate_limit_exceeded",
                    ip=ip,
                    path=path,
                    limit=limit,
                    window=window,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "RATE_LIMIT",
                        "detail": f"Quá nhiều request. Giới hạn {limit} req / {window}s.",
                    },
                    headers={"Retry-After": str(window)},
                )
            break  # chỉ match rule đầu tiên

    return await call_next(request)


async def _check(bucket: str, limit: int, window: int, pool) -> bool:
    """Sliding window counter với Redis ZADD / ZREMRANGEBYSCORE.
    Dùng pool có sẵn thay vì tạo connection mới mỗi request.
    """
    try:
        import redis.asyncio as aioredis
        if pool is not None:
            r = aioredis.Redis(connection_pool=pool)
        else:
            # Fallback: tạo connection tạm (chỉ xảy ra khi lifespan chưa chạy)
            r = aioredis.from_url(_cfg.redis_url, decode_responses=True)

        now = int(time.time() * 1000)   # milliseconds
        cutoff = now - window * 1000

        pipe = r.pipeline()
        pipe.zremrangebyscore(bucket, 0, cutoff)
        pipe.zadd(bucket, {str(now): now})
        pipe.zcard(bucket)
        pipe.expire(bucket, window + 10)
        results = await pipe.execute()
        count = results[2]

        # Không gọi r.aclose() — connection trả về pool tự động
        return count <= limit

    except Exception as e:
        # Fail-open: Redis không chạy → không block request
        log.warning("rate_limit_redis_unavailable", error=type(e).__name__)
        return True


def _ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"