"""Unit tests for config parsing helpers and defaults."""

from __future__ import annotations

import importlib
import os

from bot import config


def test_as_int_fallback(monkeypatch):
    """Invalid integer env values should fall back to provided default."""
    monkeypatch.setenv("TEST_INT", "not-a-number")
    assert config._as_int("TEST_INT", 42) == 42


def test_as_float_fallback(monkeypatch):
    """Invalid float env values should fall back to provided default."""
    monkeypatch.setenv("TEST_FLOAT", "nan-value")
    assert config._as_float("TEST_FLOAT", 0.65) == 0.65


def test_as_int_set_ignores_invalid_tokens(monkeypatch):
    """Integer set parser should skip malformed comma-separated entries."""
    monkeypatch.setenv("TEST_IDS", "12, abc, 34, , 56x")
    assert config._as_int_set("TEST_IDS") == {12, 34}


def test_config_reload_reads_environment_overrides(monkeypatch):
    """Reloaded config should reflect explicit env overrides."""
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "123")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token-abc")
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.75")
    monkeypatch.setenv("RAG_MIN_CHUNKS", "3")
    monkeypatch.setenv("RAG_TOP_K", "7")

    reloaded = importlib.reload(config)
    assert reloaded.CONFIG.owner_id == 123
    assert reloaded.CONFIG.telegram_bot_token == "token-abc"
    assert reloaded.CONFIG.rag_similarity_threshold == 0.75
    assert reloaded.CONFIG.rag_min_chunks == 3
    assert reloaded.CONFIG.rag_top_k == 7

    # Restore module-level state for later tests that import bot.config.
    importlib.reload(config)
    os.environ.pop("TEST_INT", None)
    os.environ.pop("TEST_FLOAT", None)
    os.environ.pop("TEST_IDS", None)
