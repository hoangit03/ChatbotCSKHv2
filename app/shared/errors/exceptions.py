"""
app/shared/errors/exceptions.py

Hierarchy lỗi — phân biệt rõ loại lỗi để handler phản hồi đúng HTTP status.
"""
from __future__ import annotations


class AppError(Exception):
    """Base cho mọi lỗi của ứng dụng."""
    http_status: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = context


# ── Domain errors (4xx) ───────────────────────────────────────────

class ValidationError(AppError):
    http_status = 422
    code = "VALIDATION_ERROR"


class NotFoundError(AppError):
    http_status = 404
    code = "NOT_FOUND"


class PermissionError(AppError):
    http_status = 403
    code = "PERMISSION_DENIED"


class AuthError(AppError):
    http_status = 401
    code = "AUTH_ERROR"


class RateLimitError(AppError):
    http_status = 429
    code = "RATE_LIMIT"


# ── File errors ───────────────────────────────────────────────────

class FileValidationError(ValidationError):
    code = "FILE_INVALID"


class FileTooLargeError(FileValidationError):
    code = "FILE_TOO_LARGE"


class UnsupportedFileTypeError(FileValidationError):
    code = "FILE_TYPE_UNSUPPORTED"


class FileMagicMismatchError(FileValidationError):
    code = "FILE_MAGIC_MISMATCH"


# ── Infrastructure errors (5xx) ───────────────────────────────────

class LLMError(AppError):
    http_status = 502
    code = "LLM_ERROR"


class VectorDBError(AppError):
    http_status = 502
    code = "VECTOR_DB_ERROR"


class SalesAPIError(AppError):
    http_status = 502
    code = "SALES_API_ERROR"

    def __init__(self, message: str, upstream_status: int = 0, **context):
        super().__init__(message, **context)
        self.upstream_status = upstream_status


class StorageError(AppError):
    http_status = 500
    code = "STORAGE_ERROR"


class ParseError(AppError):
    http_status = 422
    code = "PARSE_ERROR"


# ── Agent errors ──────────────────────────────────────────────────

class AgentMaxIterationsError(AppError):
    http_status = 500
    code = "AGENT_MAX_ITER"


class AgentToolError(AppError):
    http_status = 500
    code = "AGENT_TOOL_ERROR"