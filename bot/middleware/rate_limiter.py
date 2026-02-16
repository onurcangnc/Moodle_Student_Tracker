"""Rate limiting middleware abstractions."""

from __future__ import annotations

from bot.services.user_service import check_rate_limit


def allow_message(user_id: int) -> bool:
    """Return whether user can send another message in current window."""
    return check_rate_limit(user_id)
