"""
app/api/middleware/auth.py

Middleware xác thực API key cho mọi request đến hệ thống.

Bảo mật:
  - Client gửi key qua header X-API-Key (không qua URL / query string)
  - Key được hash PBKDF2-HMAC-SHA256 trước khi so sánh với DB
  - Timing-safe comparison (hmac.compare_digest)
  - Không log raw key — chỉ log prefix 8 ký tự
  - Public paths (health, docs) miễn auth

Phân biệt rõ:
  - X-API-Key: key của CLIENT gọi vào RAG system này
  - Anthropic/OpenAI keys: chỉ nằm trong Settings, KHÔNG BAO GIỜ ra ngoài

FIX (v1.1):
  - APIKeyMiddleware KHÔNG kế thừa BaseHTTPMiddleware nữa.
    Dùng như plain callable — instance tạo 1 lần, reuse mỗi request.
  - Middleware lazy-load APIKeyStore từ app.state (giải quyết race condition
    khi request đến trước lifespan hoàn thành).
  - Thay import private _hash_key → dùng public hash_api_key.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from app.shared.logging.logger import get_logger
from app.shared.security.guards import hash_api_key, verify_api_key

log = get_logger(__name__)

# Paths không cần auth
_PUBLIC = frozenset({
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
})


class APIKeyStore:
    """
    Lưu trữ API key (đã hash) và thông tin principal.
    Production: thay bằng PostgreSQL + Redis cache (TTL 5 phút).
    """

    def __init__(self) -> None:
        # { hashed_key: principal_dict }
        self._keys: dict[str, dict] = {}

    def register(self, raw_key: str, user_id: str, role: str = "sales") -> None:
        """Đăng ký key mới. Chỉ gọi lúc khởi động hoặc từ admin endpoint."""
        hashed = hash_api_key(raw_key)           # ← dùng public function
        self._keys[hashed] = {"user_id": user_id, "role": role}
        log.info("api_key_registered", user_id=user_id, role=role)

    def lookup(self, raw_key: str) -> dict | None:
        """Tìm principal từ raw key. Return None nếu không hợp lệ."""
        hashed = hash_api_key(raw_key)           # ← dùng public function
        return self._keys.get(hashed)

    def revoke(self, raw_key: str) -> bool:
        hashed = hash_api_key(raw_key)
        if hashed in self._keys:
            del self._keys[hashed]
            return True
        return False


class APIKeyMiddleware:
    """
    Plain callable middleware — KHÔNG kế thừa BaseHTTPMiddleware.

    Lý do:
      - BaseHTTPMiddleware yêu cầu store phải sẵn sàng lúc đăng ký (app startup).
        Nhưng APIKeyStore được khởi tạo trong lifespan SAU khi middleware đã đăng ký.
      - Giải pháp: lazy-load store từ request.app.state mỗi request.
      - Tạo instance 1 lần trong main.py, reuse cho mọi request (không tạo mới mỗi call).

    Sử dụng trong main.py:
        _auth_mw = APIKeyMiddleware()

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            return await _auth_mw.dispatch(request, call_next)
    """

    async def dispatch(self, request: Request, call_next):
        # Public paths — bỏ qua auth
        if request.url.path in _PUBLIC:
            return await call_next(request)

        # Lazy-load store — guard race condition khi service đang khởi động
        store: APIKeyStore | None = getattr(request.app.state, "api_key_store", None)
        if store is None:
            log.warning("auth_service_not_ready", path=request.url.path)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "SERVICE_STARTING",
                    "detail": "Hệ thống đang khởi động. Vui lòng thử lại sau.",
                },
            )

        raw_key = request.headers.get("X-API-Key", "").strip()

        if not raw_key:
            log.warning(
                "auth_missing_key",
                path=request.url.path,
                ip=_ip(request),
            )
            return JSONResponse(
                status_code=401,
                content={"error": "AUTH_ERROR", "detail": "Thiếu API key. Thêm header: X-API-Key: <your-key>"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        principal = store.lookup(raw_key)
        if principal is None:
            log.warning(
                "auth_invalid_key",
                path=request.url.path,
                ip=_ip(request),
                key_prefix=raw_key[:8] + "...",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "AUTH_ERROR", "detail": "API key không hợp lệ hoặc đã bị thu hồi."},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        request.state.principal = principal
        log.debug("auth_ok", user=principal["user_id"], path=request.url.path)
        return await call_next(request)


def require_role(request: Request, *roles: str) -> dict:
    """
    Helper dùng trong endpoint để kiểm tra role.
    Raise 403 nếu không đủ quyền.
    """
    from fastapi import HTTPException, status
    principal = getattr(request.state, "principal", {})
    if principal.get("role") not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cần role: {', '.join(roles)}. Role hiện tại: {principal.get('role')}",
        )
    return principal


def _ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"