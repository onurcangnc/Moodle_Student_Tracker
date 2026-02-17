"""Integration tests for Telegram handler registration wiring."""

from __future__ import annotations

import pytest

pytest.importorskip("telegram")

from bot.handlers.commands import register_command_handlers
from bot.handlers.messages import register_message_handlers


class DummyApp:
    """Simple app stub that records registered handlers."""

    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)


@pytest.mark.integration
def test_register_handlers_adds_expected_groups():
    """All modular registration functions should append handlers to the app."""
    app = DummyApp()
    register_command_handlers(app)
    register_message_handlers(app)

    assert len(app.handlers) == 4  # /start, /upload, Document, Text
    command_handlers = [h for h in app.handlers if hasattr(h, "commands")]
    assert any("start" in h.commands for h in command_handlers)
    assert any("upload" in h.commands for h in command_handlers)
