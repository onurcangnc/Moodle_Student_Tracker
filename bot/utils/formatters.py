"""Text formatting helpers for Telegram-friendly responses."""

from __future__ import annotations

from telegram import Message
from telegram.error import TelegramError


def to_markdown_safe(text: str) -> str:
    """Return text unchanged while keeping explicit formatting entry point."""
    return text.strip()


async def send_text(message: Message, text: str) -> None:
    """Send text with Markdown first, then plain fallback."""
    payload = to_markdown_safe(text)
    try:
        await message.reply_text(payload, parse_mode="Markdown")
    except TelegramError:
        await message.reply_text(payload)
