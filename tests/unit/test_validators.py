"""Unit tests for validation helpers."""

from __future__ import annotations

from bot.utils import validators


def test_is_non_empty_text():
    """Validator should accept meaningful text and reject empty payloads."""
    assert validators.is_non_empty_text(" merhaba ")
    assert not validators.is_non_empty_text("   ")
    assert not validators.is_non_empty_text(None)


def test_normalize_course_query():
    """Whitespace normalization should collapse repeated spaces."""
    assert validators.normalize_course_query("  CTIS   363  ") == "CTIS 363"
