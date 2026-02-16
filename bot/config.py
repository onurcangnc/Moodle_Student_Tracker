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


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Typed runtime configuration."""

    owner_id: int
    telegram_bot_token: str
    auto_sync_interval: int
    assignment_check_interval: int
    stars_notify_interval: int
    rate_limit_max: int
    rate_limit_window: int
    log_level: str
    healthcheck_enabled: bool
    healthcheck_host: str
    healthcheck_port: int


CONFIG = AppConfig(
    owner_id=_as_int("TELEGRAM_OWNER_ID", 0),
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    auto_sync_interval=_as_int("AUTO_SYNC_INTERVAL", 600),
    assignment_check_interval=_as_int("ASSIGNMENT_CHECK_INTERVAL", 600),
    stars_notify_interval=_as_int("STARS_NOTIFY_INTERVAL", 43200),
    rate_limit_max=_as_int("RATE_LIMIT_MAX", 30),
    rate_limit_window=_as_int("RATE_LIMIT_WINDOW", 60),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    healthcheck_enabled=_as_bool("HEALTHCHECK_ENABLED", True),
    healthcheck_host=os.getenv("HEALTHCHECK_HOST", "0.0.0.0"),
    healthcheck_port=_as_int("HEALTHCHECK_PORT", 8080),
)
