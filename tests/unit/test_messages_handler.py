"""Unit tests for unified message handler branching logic."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from bot.handlers import messages


@dataclass(slots=True)
class _Course:
    course_id: str


def _build_update(text: str = "Polimorfizm nedir?") -> SimpleNamespace:
    message = SimpleNamespace(text=text, reply_text=AsyncMock())
    user = SimpleNamespace(id=12345)
    return SimpleNamespace(effective_message=message, effective_user=user)


@pytest.mark.asyncio
async def test_handler_no_active_course(monkeypatch):
    """If user has no active course, handler should direct them to /courses."""
    update = _build_update()
    context = SimpleNamespace()

    monkeypatch.setattr(messages.user_service, "check_rate_limit", lambda user_id: True)
    monkeypatch.setattr(messages.user_service, "get_active_course", lambda user_id: None)
    retrieve_mock = AsyncMock()
    monkeypatch.setattr(messages.rag_service, "retrieve_context", retrieve_mock)

    await messages.handle_message(update, context)

    update.effective_message.reply_text.assert_awaited_once_with(
        "Henuz bir kurs secmediniz. /courses ile kurslari gorebilirsiniz."
    )
    retrieve_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_sufficient_context_calls_teaching(monkeypatch):
    """Sufficient retrieval should trigger teaching mode and send its response."""
    update = _build_update("Kalitim nedir?")
    context = SimpleNamespace()

    monkeypatch.setattr(messages.user_service, "check_rate_limit", lambda user_id: True)
    monkeypatch.setattr(messages.user_service, "get_active_course", lambda user_id: _Course("CTIS 363"))
    monkeypatch.setattr(messages.user_service, "get_conversation_history", lambda user_id: [])

    retrieval = SimpleNamespace(has_sufficient_context=True, chunks=["chunk1"])
    monkeypatch.setattr(messages.rag_service, "retrieve_context", AsyncMock(return_value=retrieval))
    teaching_mock = AsyncMock(return_value="Ogretim cevabi")
    guidance_mock = AsyncMock(return_value="Yonlendirme cevabi")
    monkeypatch.setattr(messages.llm_service, "generate_teaching_response", teaching_mock)
    monkeypatch.setattr(messages.llm_service, "generate_guidance_response", guidance_mock)

    add_turn_calls: list[tuple[int, str, str]] = []

    def _add_turn(user_id: int, role: str, content: str) -> None:
        add_turn_calls.append((user_id, role, content))

    monkeypatch.setattr(messages.user_service, "add_conversation_turn", _add_turn)

    await messages.handle_message(update, context)

    teaching_mock.assert_awaited_once()
    guidance_mock.assert_not_awaited()
    assert add_turn_calls == [
        (12345, "user", "Kalitim nedir?"),
        (12345, "assistant", "Ogretim cevabi"),
    ]
    update.effective_message.reply_text.assert_awaited_once_with("Ogretim cevabi", parse_mode="Markdown")


@pytest.mark.asyncio
async def test_handler_insufficient_context_calls_guidance(monkeypatch):
    """Insufficient retrieval should load topics and call guidance mode."""
    update = _build_update("Kuantum fizigi nedir?")
    context = SimpleNamespace()

    monkeypatch.setattr(messages.user_service, "check_rate_limit", lambda user_id: True)
    monkeypatch.setattr(messages.user_service, "get_active_course", lambda user_id: _Course("CTIS 363"))
    monkeypatch.setattr(messages.user_service, "get_conversation_history", lambda user_id: [])

    retrieval = SimpleNamespace(has_sufficient_context=False, chunks=[])
    monkeypatch.setattr(messages.rag_service, "retrieve_context", AsyncMock(return_value=retrieval))
    monkeypatch.setattr(
        messages,
        "TOPIC_CACHE",
        SimpleNamespace(get_topics=AsyncMock(return_value=["Kalitim", "Polimorfizm"])),
    )
    teaching_mock = AsyncMock(return_value="Ogretim cevabi")
    guidance_mock = AsyncMock(return_value="Yonlendirme cevabi")
    monkeypatch.setattr(messages.llm_service, "generate_teaching_response", teaching_mock)
    monkeypatch.setattr(messages.llm_service, "generate_guidance_response", guidance_mock)
    monkeypatch.setattr(messages.user_service, "add_conversation_turn", lambda user_id, role, content: None)

    await messages.handle_message(update, context)

    teaching_mock.assert_not_awaited()
    guidance_mock.assert_awaited_once_with(
        query="Kuantum fizigi nedir?",
        available_topics=["Kalitim", "Polimorfizm"],
        conversation_history=[],
    )
    update.effective_message.reply_text.assert_has_awaits([call("Yonlendirme cevabi", parse_mode="Markdown")])
