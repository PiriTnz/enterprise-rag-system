"""Structured logging setup. Uses stdlib logging with a JSON-friendly format
when in non-dev environments so logs are ingestible by ELK/Datadog/etc.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import settings


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.app = settings.app_name
        record.env = settings.environment
        return True


def setup_logging(level: str | None = None) -> None:
    log_level = (level or settings.log_level).upper()

    fmt = (
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        if settings.environment == "dev"
        else '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s",'
        '"app":"%(app)s","env":"%(env)s","msg":"%(message)s"}'
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Quiet down noisy libs
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured event line. Useful for metrics-style logs."""
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"event={event} {payload}")
