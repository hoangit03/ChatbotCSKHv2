"""
app/shared/logging/logger.py

Structured logging với structlog.
JSON trong production, màu trong dev.
Auto-mask sensitive keys.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def _mask_processor(logger: Any, method: str, event: dict) -> dict:
    from app.shared.security.guards import _SENSITIVE_KEYS, mask_value
    for k in list(event.keys()):
        if k.lower().replace("_", "-") in _SENSITIVE_KEYS:
            event[k] = mask_value(str(event[k]))
    return event


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _mask_processor,
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    for noisy in ["httpx", "httpcore", "asyncio", "multipart", "qdrant_client"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)