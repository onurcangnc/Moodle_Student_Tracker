"""Unit tests for LLM teaching/guidance prompt construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.services import llm_service
from bot.services.rag_service import Chunk
from bot.state import STATE


class _CapturingEngine:
    """Engine stub capturing completion payload for assertions."""

    def __init__(self) -> None:
        self.last_task: str | None = None
        self.last_system: str | None = None
        self.last_messages: list[dict] | None = None
        self.last_max_tokens: int | None = None

    def complete(self, task: str, system: str, messages: list[dict], max_tokens: int) -> str:
        self.last_task = task
        self.last_system = system
        self.last_messages = messages
        self.last_max_tokens = max_tokens
        return "Test cevabi"


@pytest.mark.asyncio
async def test_teaching_prompt_includes_chunks(monkeypatch):
    """Teaching mode should include retrieved chunk text in user prompt."""
    engine = _CapturingEngine()
    monkeypatch.setattr(STATE, "llm", SimpleNamespace(engine=engine))
    chunks = [Chunk(chunk_id="c1", text="Polimorfizm cok bicimliliktir.", similarity=0.9, metadata={"filename": "oop.pdf"})]

    response = await llm_service.generate_teaching_response(
        query="Polimorfizm nedir?",
        chunks=chunks,
        conversation_history=[],
    )

    assert response == "Test cevabi"
    assert engine.last_task == "study"
    payload = engine.last_messages[0]["content"]
    assert "Polimorfizm cok bicimliliktir." in payload
    assert "Kaynak 1: oop.pdf" in payload


@pytest.mark.asyncio
async def test_guidance_prompt_includes_topics(monkeypatch):
    """Guidance mode should include available topics in the prompt."""
    engine = _CapturingEngine()
    monkeypatch.setattr(STATE, "llm", SimpleNamespace(engine=engine))

    await llm_service.generate_guidance_response(
        query="Ne var burada?",
        available_topics=["Kalitim", "Polimorfizm"],
        conversation_history=[],
    )

    assert engine.last_task == "chat"
    payload = engine.last_messages[0]["content"]
    assert "- Kalitim" in payload
    assert "- Polimorfizm" in payload


@pytest.mark.asyncio
async def test_teaching_prompt_includes_conversation_history(monkeypatch):
    """Conversation history should be included in teaching prompt context."""
    engine = _CapturingEngine()
    monkeypatch.setattr(STATE, "llm", SimpleNamespace(engine=engine))
    chunks = [Chunk(chunk_id="c1", text="Metin", similarity=0.9, metadata={"filename": "oop.pdf"})]
    history = [
        {"role": "user", "content": "Polimorfizm nedir?"},
        {"role": "assistant", "content": "Cok bicimlilik olarak aciklanir."},
    ]

    await llm_service.generate_teaching_response(
        query="Bir ornek verir misin?",
        chunks=chunks,
        conversation_history=history,
    )

    payload = engine.last_messages[0]["content"]
    assert "ONCEKI KONUSMA" in payload
    assert "user: Polimorfizm nedir?" in payload
    assert "assistant: Cok bicimlilik olarak aciklanir." in payload
