"""Unit tests for structured logging configuration."""

from __future__ import annotations

import logging

from bot.logging_config import setup_logging


def test_setup_logging_configures_root_logger():
    """setup_logging should configure root level and stdout handler."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    try:
        setup_logging("DEBUG")
        assert root.level == logging.DEBUG
        assert root.handlers
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("chromadb").level == logging.WARNING
    finally:
        root.handlers.clear()
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
