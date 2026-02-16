"""Authorization middleware helpers for Telegram updates."""

from __future__ import annotations

from telegram import Update

from bot import legacy


def is_owner(update: Update) -> bool:
    """Return whether the update sender matches configured owner."""
    return legacy.is_owner(update)


async def owner_only(update: Update) -> bool:
    """Enforce owner-only checks and emit user feedback when denied."""
    return await legacy.owner_only(update)
