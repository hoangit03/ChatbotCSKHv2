"""
app/shared/security/guards.py

Bảo mật tập trung:
  1. Sanitize input người dùng (chống prompt injection)
  2. Hash/verify API key (PBKDF2-HMAC-SHA256)
  3. Mask sensitive data trong logs
  4. Magic-byte check cho file upload
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import Any

from app.core.config.settings import get_settings

_cfg = get_settings()


# ─────────────────────────────────────────────────────────────────
# PROMPT INJECTION GUARD
# ─────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions?"
    r"|disregard\s+.{0,30}instructions?"
    r"|you\s+are\s+now\s+(a\s+)?(jailbreak|dan|evil|unrestricted)"
    r"|<\s*/?\s*(system|human|assistant)\s*>"
    r"|###\s*(new|override)\s*(instruction|system|prompt)"
    r"|\[INST\]|\[/INST\]",
    flags=re.IGNORECASE,
)


def sanitize_input(text: str, max_len: int = 2000) -> tuple[str, bool]:
    """
    Returns (cleaned_text, was_injected).
    Caller nên log nếu was_injected=True.
    """
    text = text[:max_len].replace("\x00", "")
    injected = bool(_INJECTION_PATTERNS.search(text))
    if injected:
        text = _INJECTION_PATTERNS.sub("[BLOCKED]", text)
    return text.strip(), injected


# ─────────────────────────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────────────────────────

_PBKDF2_ITERATIONS = 260_000   # OWASP 2024


def generate_api_key() -> tuple[str, str]:
    """Returns (raw_key, hashed_key). raw chỉ show 1 lần."""
    raw = f"rag_{secrets.token_urlsafe(32)}"
    return raw, _hash_key(raw)


def verify_api_key(raw: str, hashed: str) -> bool:
    return hmac.compare_digest(_hash_key(raw), hashed)


def hash_api_key(raw: str) -> str:
    """Public alias cho _hash_key — dùng khi cần hash key từ module khác."""
    return _hash_key(raw)


def _hash_key(raw: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        raw.encode(),
        _cfg.app_secret_key.encode(),
        iterations=_PBKDF2_ITERATIONS,
    )
    return dk.hex()



# ─────────────────────────────────────────────────────────────────
# DOCUMENT CODE GENERATOR
# ─────────────────────────────────────────────────────────────────

def make_document_code(project: str, group: str) -> str:
    """DOC-<PROJECT>-<GROUP>-<8hex>  e.g. DOC-VINH-BROC-a3b2c1d0"""
    p = re.sub(r"[^A-Z0-9]", "", project.upper())[:8]
    g = re.sub(r"[^A-Z0-9]", "", group.upper())[:6]
    uid = secrets.token_hex(4)
    return f"DOC-{p}-{g}-{uid}"


# ─────────────────────────────────────────────────────────────────
# LOG MASKING
# ─────────────────────────────────────────────────────────────────

_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key", "x-api-key",
    "authorization", "password", "secret", "token",
    "x-internal-key",
})


def mask_value(v: str) -> str:
    if not v or len(v) < 8:
        return "***"
    return v[:4] + "***" + v[-4:]


def mask_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Shallow-safe copy với sensitive values masked."""
    result = {}
    for k, v in d.items():
        if k.lower().replace("_", "-") in _SENSITIVE_KEYS:
            result[k] = mask_value(str(v))
        elif isinstance(v, dict):
            result[k] = mask_dict(v)
        else:
            result[k] = v
    return result


# ─────────────────────────────────────────────────────────────────
# FILE MAGIC BYTES
# ─────────────────────────────────────────────────────────────────

_MAGIC: dict[str, bytes] = {
    ".pdf":  b"%PDF",
    ".docx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
    ".png":  b"\x89PNG",
    ".jpg":  b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
}


def check_magic_bytes(data: bytes, ext: str) -> bool:
    """True nếu magic bytes khớp."""
    expected = _MAGIC.get(ext.lower())
    if expected is None:
        return True  # không biết → không chặn
    return data[:len(expected)] == expected