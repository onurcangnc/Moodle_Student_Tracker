"""Presentation and formatting helpers for Telegram responses."""

from __future__ import annotations

from typing import Any


def _legacy():
    from bot import legacy

    return legacy


def format_for_telegram(text: str) -> str:
    """Normalize model output for Telegram HTML mode."""
    return _legacy().format_for_telegram(text)


async def send_long_message(
    update: Any,
    text: str,
    reply_markup: Any | None = None,
    parse_mode: str | None = None,
) -> None:
    """Send long messages in chunks while preserving markup on final chunk."""
    await _legacy().send_long_message(update=update, text=text, reply_markup=reply_markup, parse_mode=parse_mode)


def format_progress(current: int, total: int) -> str:
    """Return reading progress bar text."""
    return _legacy()._format_progress(current=current, total=total)


def format_source_footer(chunks: list[dict], source_type: str) -> str:
    """Return source attribution footer for response chunks."""
    return _legacy()._format_source_footer(chunks=chunks, source_type=source_type)
