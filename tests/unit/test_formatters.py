"""Unit tests for outbound text formatting helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from telegram.error import TelegramError

from bot.utils import formatters


def test_to_markdown_safe():
    """Formatter should trim message text."""
    assert formatters.to_markdown_safe("  abc  ") == "abc"


def test_send_text_fallback():
    """Sender should fallback to plain text when Markdown send fails."""
    message = type("M", (), {})()
    message.reply_text = AsyncMock(side_effect=[TelegramError("bad markdown"), None])
    asyncio.run(formatters.send_text(message, "hello"))
    assert message.reply_text.await_count == 2
