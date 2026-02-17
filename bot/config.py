"""Application configuration for the Telegram bot runtime.

Loads environment variables and centralizes runtime constants used across
handlers, services, and middleware modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _as_int(name: str, default: int) -> int:
    """Parse an integer environment value with a safe fallback."""
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _as_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment value with safe defaults."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(name: str, default: float) -> float:
    """Parse a float environment value with a safe fallback."""
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _as_int_set(name: str) -> set[int]:
    """Parse comma-separated integer IDs from environment."""
    raw = os.getenv(name, "")
    result: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.add(int(token))
        except ValueError:
            continue
    return result


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Typed runtime configuration."""

    owner_id: int
    admin_ids: frozenset[int]
    telegram_bot_token: str
    auto_sync_interval: int
    assignment_check_interval: int
    stars_notify_interval: int
    rate_limit_max: int
    rate_limit_window: int
    log_level: str
    rag_similarity_threshold: float
    rag_min_chunks: int
    rag_top_k: int
    memory_max_messages: int
    memory_ttl_minutes: int
    healthcheck_enabled: bool
    healthcheck_host: str
    healthcheck_port: int


CONFIG = AppConfig(
    owner_id=_as_int("TELEGRAM_OWNER_ID", 0),
    admin_ids=frozenset(_as_int_set("TELEGRAM_ADMIN_IDS")),
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    auto_sync_interval=_as_int("AUTO_SYNC_INTERVAL", 600),
    assignment_check_interval=_as_int("ASSIGNMENT_CHECK_INTERVAL", 600),
    stars_notify_interval=_as_int("STARS_NOTIFY_INTERVAL", 43200),
    rate_limit_max=_as_int("RATE_LIMIT_MAX", 30),
    rate_limit_window=_as_int("RATE_LIMIT_WINDOW", 60),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    rag_similarity_threshold=_as_float("RAG_SIMILARITY_THRESHOLD", 0.65),
    rag_min_chunks=_as_int("RAG_MIN_CHUNKS", 2),
    rag_top_k=_as_int("RAG_TOP_K", 5),
    memory_max_messages=_as_int("MEMORY_MAX_MESSAGES", 5),
    memory_ttl_minutes=_as_int("MEMORY_TTL_MINUTES", 30),
    healthcheck_enabled=_as_bool("HEALTHCHECK_ENABLED", True),
    healthcheck_host=os.getenv("HEALTHCHECK_HOST", "0.0.0.0"),
    healthcheck_port=_as_int("HEALTHCHECK_PORT", 9090),
)
