"""Unit tests for custom bot exception hierarchy."""

from __future__ import annotations

from bot.exceptions import (
    AuthorizationError,
    BotBaseError,
    DocumentProcessingError,
    InsufficientContextError,
    RAGPipelineError,
    RateLimitExceededError,
)


def test_exception_hierarchy():
    """All custom exceptions should derive from BotBaseError."""
    assert issubclass(RAGPipelineError, BotBaseError)
    assert issubclass(DocumentProcessingError, BotBaseError)
    assert issubclass(RateLimitExceededError, BotBaseError)
    assert issubclass(InsufficientContextError, BotBaseError)
    assert issubclass(AuthorizationError, BotBaseError)


def test_exception_instantiation():
    """Custom exceptions should be constructible with messages."""
    err = InsufficientContextError("context yetersiz")
    assert str(err) == "context yetersiz"
