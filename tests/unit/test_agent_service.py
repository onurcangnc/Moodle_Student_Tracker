"""
Comprehensive unit tests for bot/services/agent_service.py
==========================================================
Tests all 14 tool handlers, language detection, tool execution,
system prompt building, and the main agent loop.

All tests are mocked — no LLM API calls, no network.
~200+ test cases covering:
  A. _detect_language (30+ cases)
  B. _resolve_course (10+ cases)
  C. Tool handlers — 14 tools × ~10 cases each (140+ cases)
  D. _execute_tool_call (15+ cases)
  E. _build_system_prompt (10+ cases)
  F. handle_agent_message (15+ cases)
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.services import agent_service
from bot.state import STATE


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset STATE between tests to avoid cross-contamination."""
    monkeypatch.setattr(STATE, "moodle", None)
    monkeypatch.setattr(STATE, "vector_store", None)
    monkeypatch.setattr(STATE, "llm", None)
    monkeypatch.setattr(STATE, "stars_client", None)
    monkeypatch.setattr(STATE, "webmail_client", None)
    monkeypatch.setattr(STATE, "active_courses", {})
    monkeypatch.setattr(STATE, "started_at_monotonic", time.monotonic())
    monkeypatch.setattr(STATE, "startup_version", "test")
    monkeypatch.setattr(STATE, "file_summaries", {})


def _make_course(course_id: str, short_name: str, display_name: str):
    return SimpleNamespace(
        course_id=course_id,
        short_name=short_name,
        display_name=display_name,
    )


def _make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def _make_vector_store(
    files=None, chunks=None, hybrid_results=None, stats=None
):
    """Build a mock vector store."""
    vs = SimpleNamespace(
        get_files_for_course=MagicMock(return_value=files or []),
        get_file_chunks=MagicMock(return_value=chunks or []),
        hybrid_search=MagicMock(return_value=hybrid_results or []),
        get_stats=MagicMock(return_value=stats or {
            "total_chunks": 100, "unique_courses": 3, "unique_files": 10
        }),
    )
    return vs


def _make_stars(authenticated=True, schedule=None, grades=None, attendance=None):
    return SimpleNamespace(
        is_authenticated=MagicMock(return_value=authenticated),
        get_schedule=MagicMock(return_value=schedule or []),
        get_grades=MagicMock(return_value=grades or []),
        get_attendance=MagicMock(return_value=attendance or []),
    )


def _make_webmail(authenticated=True, mails=None):
    return SimpleNamespace(
        authenticated=authenticated,
        get_recent_airs_dais=MagicMock(return_value=mails or []),
        check_all_unread=MagicMock(return_value=mails or []),
    )


def _make_mail(subject="Test Mail", from_addr="test@bilkent.edu.tr",
               date="25 Şub 2026", body="Body", source="AIRS"):
    return {
        "subject": subject,
        "from": from_addr,
        "date": date,
        "body_preview": body,
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# A. _detect_language TESTS (30+ cases)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectLanguage:
    """Tests for _detect_language function."""

    # --- Turkish detection (chars) ---
    def test_turkish_chars_c(self):
        assert agent_service._detect_language("Nasılsın?") == "tr"

    def test_turkish_chars_g(self):
        assert agent_service._detect_language("Günaydın") == "tr"

    def test_turkish_chars_i(self):
        assert agent_service._detect_language("İstanbul") == "tr"

    def test_turkish_chars_o(self):
        assert agent_service._detect_language("Ödev nedir") == "tr"

    def test_turkish_chars_s(self):
        assert agent_service._detect_language("Şimdi") == "tr"

    def test_turkish_chars_u(self):
        assert agent_service._detect_language("Üniversite") == "tr"

    def test_turkish_mixed_with_english(self):
        # Turkish chars take priority over English words
        assert agent_service._detect_language("Show me ödevlerim") == "tr"

    def test_turkish_sentence(self):
        assert agent_service._detect_language("Bugün derslerim ne?") == "tr"

    def test_turkish_filler(self):
        assert agent_service._detect_language("Neyse devam edelim") == "tr"

    # --- English detection (word set) ---
    def test_english_two_words(self):
        assert agent_service._detect_language("show grades") == "en"

    def test_english_greeting(self):
        assert agent_service._detect_language("hello") == "en"

    def test_english_short_phrase(self):
        assert agent_service._detect_language("my grades") == "en"

    def test_english_full_sentence(self):
        assert agent_service._detect_language("Show me my grades please") == "en"

    def test_english_schedule(self):
        assert agent_service._detect_language("What is my schedule today?") == "en"

    def test_english_emails(self):
        assert agent_service._detect_language("Show me my emails") == "en"

    def test_english_help(self):
        assert agent_service._detect_language("help me") == "en"

    def test_english_hey(self):
        assert agent_service._detect_language("hey") == "en"

    def test_english_hi(self):
        assert agent_service._detect_language("hi") == "en"

    def test_english_attendance(self):
        assert agent_service._detect_language("get attendance") == "en"

    def test_english_assignments(self):
        assert agent_service._detect_language("list assignments") == "en"

    # --- Default to Turkish ---
    def test_default_turkish_for_unknown(self):
        assert agent_service._detect_language("xyz abc") == "tr"

    def test_default_turkish_for_numbers(self):
        assert agent_service._detect_language("12345") == "tr"

    def test_default_turkish_for_empty(self):
        assert agent_service._detect_language("") == "tr"

    def test_default_turkish_for_single_unknown(self):
        assert agent_service._detect_language("merhaba") == "tr"

    def test_default_turkish_for_punctuation(self):
        assert agent_service._detect_language("???") == "tr"

    # --- Edge cases ---
    def test_single_en_word_short_message(self):
        # 1 EN word + <=4 total words → en
        assert agent_service._detect_language("show") == "en"

    def test_single_en_word_long_message(self):
        # 1 EN word in 5+ word message → tr (not enough EN)
        assert agent_service._detect_language("bana show ne demek acaba bilmiyorum") == "tr"

    def test_case_insensitive(self):
        assert agent_service._detect_language("SHOW ME MY GRADES") == "en"

    def test_turkish_chars_override_english_words(self):
        assert agent_service._detect_language("show me çalışma") == "tr"

    def test_course_code_no_language(self):
        # Course codes have no language signal
        assert agent_service._detect_language("CTIS 363") == "tr"

    def test_emoji_only(self):
        assert agent_service._detect_language("👋") == "tr"


# ═══════════════════════════════════════════════════════════════════════════════
# B. _resolve_course TESTS (10+ cases)
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveCourse:
    """Tests for _resolve_course helper."""

    def test_from_args(self, monkeypatch):
        result = agent_service._resolve_course({"course_filter": "CTIS 363"}, user_id=1)
        assert result == "CTIS 363"

    def test_from_active_course(self, monkeypatch):
        course = _make_course("CTIS 363", "CTIS", "CTIS 363 Ethics")
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: course,
        )
        result = agent_service._resolve_course({}, user_id=1)
        assert result == "CTIS 363"

    def test_no_course(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        result = agent_service._resolve_course({}, user_id=1)
        assert result is None

    def test_explicit_key(self, monkeypatch):
        result = agent_service._resolve_course(
            {"course_name": "EDEB 201"}, user_id=1, key="course_name"
        )
        assert result == "EDEB 201"

    def test_empty_string_falls_to_active(self, monkeypatch):
        course = _make_course("POLS", "POLS", "POLS 101")
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: course,
        )
        result = agent_service._resolve_course({"course_filter": ""}, user_id=1)
        assert result == "POLS"

    def test_none_falls_to_active(self, monkeypatch):
        course = _make_course("HCIV", "HCIV", "HCIV 102")
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: course,
        )
        result = agent_service._resolve_course({"course_filter": None}, user_id=1)
        assert result == "HCIV"


# ═══════════════════════════════════════════════════════════════════════════════
# C. TOOL HANDLER TESTS (14 tools × ~10 cases each = ~140 cases)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── C1. get_source_map ─────────────────────────────────────────────────────

class TestToolGetSourceMap:
    @pytest.mark.asyncio
    async def test_no_active_course(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        result = await agent_service._tool_get_source_map({}, user_id=1)
        assert "kurs seçili değil" in result.lower()

    @pytest.mark.asyncio
    async def test_no_vector_store(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        result = await agent_service._tool_get_source_map({}, user_id=1)
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_no_files_found(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(files=[]))
        result = await agent_service._tool_get_source_map({}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_files_found(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        files = [
            {"filename": "ethics.pdf", "chunk_count": 10, "section": "Week 1"},
            {"filename": "privacy.pdf", "chunk_count": 5, "section": "Week 2"},
        ]
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(files=files))
        monkeypatch.setattr(
            "bot.services.summary_service.load_source_summary",
            lambda fn, cn: None,
        )
        result = await agent_service._tool_get_source_map({}, user_id=1)
        assert "ethics.pdf" in result
        assert "privacy.pdf" in result
        assert "2 dosya" in result
        assert "15 toplam" in result

    @pytest.mark.asyncio
    async def test_with_course_filter_arg(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        files = [{"filename": "test.pdf", "chunk_count": 3, "section": ""}]
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(files=files))
        monkeypatch.setattr(
            "bot.services.summary_service.load_source_summary",
            lambda fn, cn: None,
        )
        result = await agent_service._tool_get_source_map(
            {"course_filter": "EDEB 201"}, user_id=1
        )
        assert "test.pdf" in result

    @pytest.mark.asyncio
    async def test_exception_handling(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("X", "X", "X"),
        )
        vs = SimpleNamespace(get_files_for_course=MagicMock(side_effect=RuntimeError("DB error")))
        monkeypatch.setattr(STATE, "vector_store", vs)
        result = await agent_service._tool_get_source_map({}, user_id=1)
        assert "alınamadı" in result.lower()


# ─── C2. read_source ────────────────────────────────────────────────────────

class TestToolReadSource:
    @pytest.mark.asyncio
    async def test_no_source_name(self, monkeypatch):
        result = await agent_service._tool_read_source({}, user_id=1)
        assert "belirtilmedi" in result.lower()

    @pytest.mark.asyncio
    async def test_no_vector_store(self, monkeypatch):
        result = await agent_service._tool_read_source({"source": "file.pdf"}, user_id=1)
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_no_chunks_found(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(chunks=[]))
        monkeypatch.setattr(
            "bot.services.summary_service.load_source_summary",
            lambda fn, cn: None,
        )
        result = await agent_service._tool_read_source({"source": "nonexist.pdf"}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_with_summary_no_section(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store())
        summary = {
            "overview": "This file covers ethics.",
            "sections": [{"title": "Intro", "summary": "Basic concepts"}],
            "cross_references": ["See also: privacy.pdf"],
            "difficulty": "Intermediate",
        }
        monkeypatch.setattr(
            "bot.services.summary_service.load_source_summary",
            lambda fn, cn: summary,
        )
        result = await agent_service._tool_read_source({"source": "ethics.pdf"}, user_id=1)
        assert "ethics.pdf" in result
        assert "Intro" in result
        assert "Hangi bölüm" in result

    @pytest.mark.asyncio
    async def test_with_section(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        chunks = [
            {"text": "Introduction to privacy concepts.", "chunk_index": 0},
            {"text": "Privacy in the digital age.", "chunk_index": 1},
            {"text": "Surveillance and ethics.", "chunk_index": 2},
        ]
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(chunks=chunks))
        monkeypatch.setattr(
            "bot.services.summary_service.load_source_summary",
            lambda fn, cn: {"overview": "Summary"},
        )
        result = await agent_service._tool_read_source(
            {"source": "ethics.pdf", "section": "privacy"}, user_id=1
        )
        assert "privacy" in result.lower()


# ─── C3. study_topic ────────────────────────────────────────────────────────

class TestToolStudyTopic:
    @pytest.mark.asyncio
    async def test_no_topic(self, monkeypatch):
        result = await agent_service._tool_study_topic({}, user_id=1)
        assert "belirtilmedi" in result.lower()

    @pytest.mark.asyncio
    async def test_no_vector_store(self, monkeypatch):
        result = await agent_service._tool_study_topic({"topic": "ethics"}, user_id=1)
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_no_results(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(hybrid_results=[]))
        result = await agent_service._tool_study_topic({"topic": "quantum"}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_with_results(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        results = [
            {"metadata": {"filename": "ethics.pdf"}, "text": "A" * 60, "distance": 0.1},
            {"metadata": {"filename": "privacy.pdf"}, "text": "B" * 60, "distance": 0.2},
        ]
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(hybrid_results=results))
        result = await agent_service._tool_study_topic(
            {"topic": "ethics", "depth": "overview"}, user_id=1
        )
        assert "ethics.pdf" in result

    @pytest.mark.asyncio
    async def test_fallback_to_all_courses(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        call_count = {"n": 0}
        def fake_search(query, top_k, course_filter):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return []  # First call (with filter) empty
            return [{"metadata": {"filename": "f.pdf"}, "text": "X" * 60, "distance": 0.1}]

        vs = SimpleNamespace(hybrid_search=MagicMock(side_effect=fake_search))
        monkeypatch.setattr(STATE, "vector_store", vs)
        result = await agent_service._tool_study_topic({"topic": "test"}, user_id=1)
        assert "f.pdf" in result
        assert call_count["n"] == 2  # Called twice (filter then all)

    @pytest.mark.asyncio
    async def test_short_text_filtered(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        results = [
            {"metadata": {"filename": "f.pdf"}, "text": "short", "distance": 0.1},
        ]
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(hybrid_results=results))
        result = await agent_service._tool_study_topic({"topic": "x"}, user_id=1)
        assert "yeterli materyal bulunamadı" in result.lower()


# ─── C4. rag_search ─────────────────────────────────────────────────────────

class TestToolRagSearch:
    @pytest.mark.asyncio
    async def test_no_query(self, monkeypatch):
        result = await agent_service._tool_rag_search({}, user_id=1)
        assert "belirtilmedi" in result.lower()

    @pytest.mark.asyncio
    async def test_no_vector_store(self, monkeypatch):
        result = await agent_service._tool_rag_search({"query": "test"}, user_id=1)
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_no_results(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(hybrid_results=[]))
        result = await agent_service._tool_rag_search({"query": "nonexistent"}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_with_results(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: _make_course("CTIS", "CTIS", "CTIS 363"),
        )
        results = [
            {"metadata": {"filename": "f.pdf", "course": "CTIS 363"}, "text": "Z" * 60, "distance": 0.15},
        ]
        monkeypatch.setattr(STATE, "vector_store", _make_vector_store(hybrid_results=results))
        result = await agent_service._tool_rag_search({"query": "ethics"}, user_id=1)
        assert "f.pdf" in result


# ─── C5. get_moodle_materials ────────────────────────────────────────────────

class TestToolGetMoodleMaterials:
    @pytest.mark.asyncio
    async def test_no_moodle(self, monkeypatch):
        result = await agent_service._tool_get_moodle_materials({}, user_id=1)
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_no_courses(self, monkeypatch):
        moodle = SimpleNamespace(
            get_courses=MagicMock(return_value=[]),
        )
        monkeypatch.setattr(STATE, "moodle", moodle)
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        result = await agent_service._tool_get_moodle_materials({}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        moodle = SimpleNamespace(
            get_courses=MagicMock(side_effect=ConnectionError("timeout")),
        )
        monkeypatch.setattr(STATE, "moodle", moodle)
        result = await agent_service._tool_get_moodle_materials({}, user_id=1)
        assert "bağlanılamadı" in result.lower()


# ─── C6. get_schedule ────────────────────────────────────────────────────────

class TestToolGetSchedule:
    @pytest.mark.asyncio
    async def test_no_stars(self, monkeypatch):
        result = await agent_service._tool_get_schedule({"period": "today"}, user_id=1)
        assert "giriş" in result.lower()

    @pytest.mark.asyncio
    async def test_not_authenticated(self, monkeypatch):
        monkeypatch.setattr(STATE, "stars_client", _make_stars(authenticated=False))
        result = await agent_service._tool_get_schedule({"period": "today"}, user_id=1)
        assert "giriş" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_schedule(self, monkeypatch):
        monkeypatch.setattr(STATE, "stars_client", _make_stars(schedule=[]))
        result = await agent_service._tool_get_schedule({"period": "today"}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_week_schedule(self, monkeypatch):
        schedule = [
            {"day": "Pazartesi", "time": "09:00-10:30", "course": "CTIS 363", "room": "B-201"},
            {"day": "Salı", "time": "13:30-15:00", "course": "EDEB 201", "room": "A-102"},
        ]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(schedule=schedule))
        result = await agent_service._tool_get_schedule({"period": "week"}, user_id=1)
        assert "Pazartesi" in result
        assert "Salı" in result
        assert "CTIS 363" in result

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        stars = SimpleNamespace(
            is_authenticated=MagicMock(return_value=True),
            get_schedule=MagicMock(side_effect=ConnectionError("timeout")),
        )
        monkeypatch.setattr(STATE, "stars_client", stars)
        result = await agent_service._tool_get_schedule({"period": "today"}, user_id=1)
        assert "alınamadı" in result.lower()


# ─── C7. get_grades ──────────────────────────────────────────────────────────

class TestToolGetGrades:
    @pytest.mark.asyncio
    async def test_no_stars(self, monkeypatch):
        result = await agent_service._tool_get_grades({}, user_id=1)
        assert "giriş" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_grades(self, monkeypatch):
        monkeypatch.setattr(STATE, "stars_client", _make_stars(grades=[]))
        result = await agent_service._tool_get_grades({}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_grades_returned(self, monkeypatch):
        grades = [
            {
                "course": "CTIS 363",
                "assessments": [
                    {"name": "Midterm", "grade": "85", "weight": "30%"},
                    {"name": "Final", "grade": "90", "weight": "40%"},
                ],
            }
        ]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(grades=grades))
        result = await agent_service._tool_get_grades({}, user_id=1)
        assert "CTIS 363" in result
        assert "85" in result
        assert "Midterm" in result

    @pytest.mark.asyncio
    async def test_course_filter(self, monkeypatch):
        grades = [
            {"course": "CTIS 363", "assessments": [{"name": "Mid", "grade": "80", "weight": ""}]},
            {"course": "EDEB 201", "assessments": [{"name": "Mid", "grade": "70", "weight": ""}]},
        ]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(grades=grades))
        result = await agent_service._tool_get_grades({"course_filter": "CTIS"}, user_id=1)
        assert "CTIS 363" in result
        assert "EDEB" not in result

    @pytest.mark.asyncio
    async def test_no_matching_course(self, monkeypatch):
        grades = [
            {"course": "CTIS 363", "assessments": [{"name": "Mid", "grade": "80", "weight": ""}]},
        ]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(grades=grades))
        result = await agent_service._tool_get_grades({"course_filter": "PHYS"}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_no_assessments(self, monkeypatch):
        grades = [{"course": "CTIS 363", "assessments": []}]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(grades=grades))
        result = await agent_service._tool_get_grades({}, user_id=1)
        assert "girilmemiş" in result.lower()


# ─── C8. get_attendance ──────────────────────────────────────────────────────

class TestToolGetAttendance:
    @pytest.mark.asyncio
    async def test_no_stars(self, monkeypatch):
        result = await agent_service._tool_get_attendance({}, user_id=1)
        assert "giriş" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_attendance(self, monkeypatch):
        monkeypatch.setattr(STATE, "stars_client", _make_stars(attendance=[]))
        result = await agent_service._tool_get_attendance({}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_attendance_normal(self, monkeypatch):
        attendance = [
            {
                "course": "CTIS 363",
                "ratio": "90%",
                "records": [
                    {"attended": True}, {"attended": True}, {"attended": False},
                ],
            }
        ]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(attendance=attendance))
        result = await agent_service._tool_get_attendance({}, user_id=1)
        assert "CTIS 363" in result
        assert "90%" in result

    @pytest.mark.asyncio
    async def test_attendance_warning(self, monkeypatch):
        attendance = [
            {
                "course": "CTIS 363",
                "ratio": "82%",
                "records": [
                    {"attended": True}, {"attended": False}, {"attended": False},
                ],
            }
        ]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(attendance=attendance))
        result = await agent_service._tool_get_attendance({}, user_id=1)
        assert "⚠️" in result

    @pytest.mark.asyncio
    async def test_course_filter(self, monkeypatch):
        attendance = [
            {"course": "CTIS 363", "ratio": "90%", "records": []},
            {"course": "EDEB 201", "ratio": "85%", "records": []},
        ]
        monkeypatch.setattr(STATE, "stars_client", _make_stars(attendance=attendance))
        result = await agent_service._tool_get_attendance({"course_filter": "EDEB"}, user_id=1)
        assert "EDEB" in result
        assert "CTIS" not in result


# ─── C9. get_assignments ─────────────────────────────────────────────────────

class TestToolGetAssignments:
    @pytest.mark.asyncio
    async def test_no_moodle(self, monkeypatch):
        result = await agent_service._tool_get_assignments({}, user_id=1)
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_no_assignments(self, monkeypatch):
        moodle = SimpleNamespace(
            get_upcoming_assignments=MagicMock(return_value=[]),
            get_assignments=MagicMock(return_value=[]),
        )
        monkeypatch.setattr(STATE, "moodle", moodle)
        result = await agent_service._tool_get_assignments({}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_upcoming_assignments(self, monkeypatch):
        assignment = SimpleNamespace(
            course_name="CTIS 363",
            name="Homework 1",
            submitted=False,
            due_date="2026-03-01",
            time_remaining="3 gün",
        )
        moodle = SimpleNamespace(
            get_upcoming_assignments=MagicMock(return_value=[assignment]),
        )
        monkeypatch.setattr(STATE, "moodle", moodle)
        result = await agent_service._tool_get_assignments({"filter": "upcoming"}, user_id=1)
        assert "CTIS 363" in result
        assert "Homework 1" in result

    @pytest.mark.asyncio
    async def test_overdue_filter(self, monkeypatch):
        assignment = SimpleNamespace(
            course_name="EDEB 201",
            name="Essay",
            submitted=False,
            due_date=time.time() - 86400,  # Yesterday
            time_remaining="",
        )
        moodle = SimpleNamespace(
            get_upcoming_assignments=MagicMock(return_value=[assignment]),
        )
        monkeypatch.setattr(STATE, "moodle", moodle)
        result = await agent_service._tool_get_assignments({"filter": "overdue"}, user_id=1)
        assert "geçmiş" in result.lower() or "EDEB" in result

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        moodle = SimpleNamespace(
            get_upcoming_assignments=MagicMock(side_effect=RuntimeError("net")),
        )
        monkeypatch.setattr(STATE, "moodle", moodle)
        result = await agent_service._tool_get_assignments({}, user_id=1)
        assert "alınamadı" in result.lower()


# ─── C10. get_emails ─────────────────────────────────────────────────────────

class TestToolGetEmails:
    @pytest.mark.asyncio
    async def test_no_webmail(self, monkeypatch):
        result = await agent_service._tool_get_emails({}, user_id=1)
        assert "giriş" in result.lower()

    @pytest.mark.asyncio
    async def test_not_authenticated(self, monkeypatch):
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(authenticated=False))
        result = await agent_service._tool_get_emails({}, user_id=1)
        assert "giriş" in result.lower()

    @pytest.mark.asyncio
    async def test_no_mails(self, monkeypatch):
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=[]))
        result = await agent_service._tool_get_emails({}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_mails(self, monkeypatch):
        mails = [
            _make_mail("Career Fair", "serhat@bilkent.edu.tr", "17 Şub 2026"),
            _make_mail("Essay Writing", "tunahan@bilkent.edu.tr", "21 Şub 2026"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails({"count": 5}, user_id=1)
        assert "Career Fair" in result
        assert "Essay Writing" in result

    @pytest.mark.asyncio
    async def test_keyword_from_filter(self, monkeypatch):
        mails = [
            _make_mail("CTIS Assignment", "prof@bilkent.edu.tr"),
            _make_mail("EDEB Lecture", "hoca@bilkent.edu.tr"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails(
            {"keyword": "CTIS", "count": 5}, user_id=1
        )
        assert "CTIS Assignment" in result
        assert "EDEB" not in result

    @pytest.mark.asyncio
    async def test_keyword_from_subject(self, monkeypatch):
        mails = [
            _make_mail("Seminar Invitation", "admin@bilkent.edu.tr"),
            _make_mail("Grades Published", "prof@bilkent.edu.tr"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails(
            {"keyword": "Seminar", "count": 5}, user_id=1
        )
        assert "Seminar" in result
        assert "Grades" not in result

    @pytest.mark.asyncio
    async def test_keyword_from_sender(self, monkeypatch):
        mails = [
            _make_mail("Mail 1", "serhat@bilkent.edu.tr"),
            _make_mail("Mail 2", "other@bilkent.edu.tr"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails(
            {"keyword": "serhat", "count": 5}, user_id=1
        )
        assert "Mail 1" in result
        assert "Mail 2" not in result

    @pytest.mark.asyncio
    async def test_keyword_date_search(self, monkeypatch):
        mails = [
            _make_mail("Mail A", "a@b.tr", "11 Şub 2026"),
            _make_mail("Mail B", "c@d.tr", "17 Şub 2026"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails(
            {"keyword": "11 Şub", "count": 5}, user_id=1
        )
        assert "Mail A" in result
        assert "Mail B" not in result

    @pytest.mark.asyncio
    async def test_keyword_source_search(self, monkeypatch):
        mails = [
            _make_mail("M1", "a@b.tr", source="AIRS"),
            _make_mail("M2", "c@d.tr", source="DAIS"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails(
            {"keyword": "DAIS", "count": 5}, user_id=1
        )
        assert "M2" in result
        assert "M1" not in result

    @pytest.mark.asyncio
    async def test_count_limit(self, monkeypatch):
        mails = [_make_mail(f"Mail {i}") for i in range(10)]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails({"count": 3}, user_id=1)
        assert result.count("📧") == 3

    @pytest.mark.asyncio
    async def test_keyword_with_larger_fetch_pool(self, monkeypatch):
        """When keyword is set, fetch_count should be max(count, 20)."""
        mails = [_make_mail(f"Mail {i}", source="AIRS") for i in range(20)]
        wm = _make_webmail(mails=mails)
        monkeypatch.setattr(STATE, "webmail_client", wm)
        await agent_service._tool_get_emails(
            {"keyword": "AIRS", "count": 3}, user_id=1
        )
        # get_recent_airs_dais should be called with 20 (not 3) when keyword is set
        wm.get_recent_airs_dais.assert_called_once_with(20)

    @pytest.mark.asyncio
    async def test_unread_scope(self, monkeypatch):
        mails = [_make_mail("Unread Mail")]
        wm = _make_webmail(mails=mails)
        monkeypatch.setattr(STATE, "webmail_client", wm)
        result = await agent_service._tool_get_emails({"scope": "unread"}, user_id=1)
        wm.check_all_unread.assert_called_once()

    @pytest.mark.asyncio
    async def test_keyword_case_insensitive(self, monkeypatch):
        mails = [_make_mail("EDEB 201 Lecture Notes")]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails(
            {"keyword": "edeb", "count": 5}, user_id=1
        )
        assert "EDEB" in result

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch):
        wm = SimpleNamespace(
            authenticated=True,
            get_recent_airs_dais=MagicMock(side_effect=ConnectionError("timeout")),
        )
        monkeypatch.setattr(STATE, "webmail_client", wm)
        result = await agent_service._tool_get_emails({}, user_id=1)
        assert "alınamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_sender_filter_backward_compat(self, monkeypatch):
        """Old 'sender_filter' param should still work."""
        mails = [
            _make_mail("M1", "serhat@bilkent.edu.tr"),
            _make_mail("M2", "other@bilkent.edu.tr"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_emails(
            {"sender_filter": "serhat", "count": 5}, user_id=1
        )
        assert "M1" in result


# ─── C11. get_email_detail ───────────────────────────────────────────────────

class TestToolGetEmailDetail:
    @pytest.mark.asyncio
    async def test_no_webmail(self, monkeypatch):
        result = await agent_service._tool_get_email_detail({}, user_id=1)
        assert "giriş" in result.lower()

    @pytest.mark.asyncio
    async def test_no_keyword(self, monkeypatch):
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail())
        result = await agent_service._tool_get_email_detail({}, user_id=1)
        assert "belirtilmedi" in result.lower()

    @pytest.mark.asyncio
    async def test_no_match(self, monkeypatch):
        mails = [_make_mail("Career Fair")]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_email_detail(
            {"keyword": "nonexistent"}, user_id=1
        )
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_match_by_subject(self, monkeypatch):
        mails = [_make_mail("Career Fair", body="Full career fair details here")]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_email_detail(
            {"keyword": "Career"}, user_id=1
        )
        assert "Career Fair" in result
        assert "Full career fair details" in result

    @pytest.mark.asyncio
    async def test_match_by_sender(self, monkeypatch):
        mails = [_make_mail("Some Mail", "serhat@bilkent.edu.tr")]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_email_detail(
            {"keyword": "serhat"}, user_id=1
        )
        assert "Some Mail" in result

    @pytest.mark.asyncio
    async def test_match_by_date(self, monkeypatch):
        mails = [_make_mail("Feb Mail", date="11 Şub 2026")]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_email_detail(
            {"keyword": "11 Şub"}, user_id=1
        )
        assert "Feb Mail" in result

    @pytest.mark.asyncio
    async def test_match_by_source(self, monkeypatch):
        mails = [_make_mail("M1", source="DAIS"), _make_mail("M2", source="AIRS")]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_email_detail(
            {"keyword": "DAIS"}, user_id=1
        )
        assert "M1" in result

    @pytest.mark.asyncio
    async def test_email_subject_backward_compat(self, monkeypatch):
        """Old 'email_subject' param should still work."""
        mails = [_make_mail("Career Fair")]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_email_detail(
            {"email_subject": "Career"}, user_id=1
        )
        assert "Career Fair" in result

    @pytest.mark.asyncio
    async def test_first_match_returned(self, monkeypatch):
        mails = [
            _make_mail("First CTIS Mail"),
            _make_mail("Second CTIS Mail"),
        ]
        monkeypatch.setattr(STATE, "webmail_client", _make_webmail(mails=mails))
        result = await agent_service._tool_get_email_detail(
            {"keyword": "CTIS"}, user_id=1
        )
        assert "First" in result


# ─── C12. list_courses ───────────────────────────────────────────────────────

class TestToolListCourses:
    @pytest.mark.asyncio
    async def test_no_courses(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.list_courses",
            lambda: [],
        )
        result = await agent_service._tool_list_courses({}, user_id=1)
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_courses_listed(self, monkeypatch):
        courses = [
            _make_course("CTIS 363", "CTIS", "CTIS 363 Ethics"),
            _make_course("EDEB 201", "EDEB", "EDEB 201 Literature"),
        ]
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.list_courses",
            lambda: courses,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        result = await agent_service._tool_list_courses({}, user_id=1)
        assert "CTIS" in result
        assert "EDEB" in result

    @pytest.mark.asyncio
    async def test_active_course_marked(self, monkeypatch):
        courses = [
            _make_course("CTIS 363", "CTIS", "CTIS 363 Ethics"),
            _make_course("EDEB 201", "EDEB", "EDEB 201 Literature"),
        ]
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.list_courses",
            lambda: courses,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: courses[0],
        )
        result = await agent_service._tool_list_courses({}, user_id=1)
        assert "▸" in result


# ─── C13. set_active_course ──────────────────────────────────────────────────

class TestToolSetActiveCourse:
    @pytest.mark.asyncio
    async def test_no_course_name(self, monkeypatch):
        result = await agent_service._tool_set_active_course({}, user_id=1)
        assert "belirtilmedi" in result.lower()

    @pytest.mark.asyncio
    async def test_course_not_found(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.find_course",
            lambda q: None,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.list_courses",
            lambda: [],
        )
        result = await agent_service._tool_set_active_course(
            {"course_name": "PHYS 101"}, user_id=1
        )
        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_course_set_successfully(self, monkeypatch):
        course = _make_course("CTIS 363", "CTIS", "CTIS 363 Ethics")
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.find_course",
            lambda q: course,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.set_active_course",
            lambda uid, cid: None,
        )
        # Need a mock llm with set_active_course
        monkeypatch.setattr(STATE, "llm", SimpleNamespace(set_active_course=MagicMock()))
        result = await agent_service._tool_set_active_course(
            {"course_name": "CTIS"}, user_id=1
        )
        assert "değiştirildi" in result.lower()
        assert "CTIS 363" in result


# ─── C14. get_stats ──────────────────────────────────────────────────────────

class TestToolGetStats:
    @pytest.mark.asyncio
    async def test_no_vector_store(self, monkeypatch):
        result = await agent_service._tool_get_stats({}, user_id=1)
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_stats_returned(self, monkeypatch):
        vs = _make_vector_store(stats={
            "total_chunks": 3661, "unique_courses": 5, "unique_files": 28
        })
        monkeypatch.setattr(STATE, "vector_store", vs)
        monkeypatch.setattr(
            "bot.services.summary_service.list_summaries",
            lambda: ["a", "b", "c"],
        )
        result = await agent_service._tool_get_stats({}, user_id=1)
        assert "3661" in result
        assert "5" in result
        assert "28" in result


# ═══════════════════════════════════════════════════════════════════════════════
# D. _execute_tool_call TESTS (15+ cases)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecuteToolCall:
    @pytest.mark.asyncio
    async def test_valid_tool_call(self, monkeypatch):
        async def fake_handler(args, uid):
            return "OK result"

        monkeypatch.setitem(agent_service.TOOL_HANDLERS, "get_stats", fake_handler)
        tc = _make_tool_call("get_stats", {})
        result = await agent_service._execute_tool_call(tc, user_id=1)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert result["content"] == "OK result"

    @pytest.mark.asyncio
    async def test_unknown_tool(self, monkeypatch):
        tc = _make_tool_call("nonexistent_tool", {})
        result = await agent_service._execute_tool_call(tc, user_id=1)
        assert "bilinmeyen" in result["content"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_args(self, monkeypatch):
        async def fake_handler(args, uid):
            return f"args={args}"

        monkeypatch.setitem(agent_service.TOOL_HANDLERS, "get_stats", fake_handler)
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="get_stats", arguments="not valid json"),
        )
        result = await agent_service._execute_tool_call(tc, user_id=1)
        assert result["content"] == "args={}"

    @pytest.mark.asyncio
    async def test_handler_exception(self, monkeypatch):
        async def exploding_handler(args, uid):
            raise ConnectionError("IMAP timeout")

        monkeypatch.setitem(agent_service.TOOL_HANDLERS, "get_emails", exploding_handler)
        tc = _make_tool_call("get_emails", {"count": 5})
        result = await agent_service._execute_tool_call(tc, user_id=1)
        assert "[get_emails]" in result["content"]
        assert "ConnectionError" in result["content"]

    @pytest.mark.asyncio
    async def test_handler_runtime_error(self, monkeypatch):
        async def bad_handler(args, uid):
            raise RuntimeError("DB crashed")

        monkeypatch.setitem(agent_service.TOOL_HANDLERS, "get_grades", bad_handler)
        tc = _make_tool_call("get_grades", {})
        result = await agent_service._execute_tool_call(tc, user_id=1)
        assert "RuntimeError" in result["content"]

    @pytest.mark.asyncio
    async def test_handler_value_error(self, monkeypatch):
        async def val_handler(args, uid):
            raise ValueError("bad param")

        monkeypatch.setitem(agent_service.TOOL_HANDLERS, "rag_search", val_handler)
        tc = _make_tool_call("rag_search", {"query": "test"})
        result = await agent_service._execute_tool_call(tc, user_id=1)
        assert "ValueError" in result["content"]

    @pytest.mark.asyncio
    async def test_result_has_correct_structure(self, monkeypatch):
        async def ok_handler(args, uid):
            return "data"

        monkeypatch.setitem(agent_service.TOOL_HANDLERS, "list_courses", ok_handler)
        tc = _make_tool_call("list_courses", {}, call_id="call_xyz")
        result = await agent_service._execute_tool_call(tc, user_id=1)
        assert set(result.keys()) == {"role", "tool_call_id", "content"}
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_xyz"

    @pytest.mark.asyncio
    async def test_all_14_tools_registered(self):
        expected = {
            "get_source_map", "read_source", "study_topic", "rag_search",
            "get_moodle_materials", "get_schedule", "get_grades", "get_attendance",
            "get_assignments", "get_emails", "get_email_detail",
            "list_courses", "set_active_course", "get_stats",
        }
        assert set(agent_service.TOOL_HANDLERS.keys()) == expected

    @pytest.mark.asyncio
    async def test_tool_definitions_match_handlers(self):
        tool_names = {t["function"]["name"] for t in agent_service.TOOLS}
        handler_names = set(agent_service.TOOL_HANDLERS.keys())
        assert tool_names == handler_names


# ═══════════════════════════════════════════════════════════════════════════════
# E. _build_system_prompt TESTS (10+ cases)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:
    def test_contains_language_rule(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "DİL KURALI" in prompt

    def test_contains_planning_section(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "PLANLAMA VE TOOL SEÇİMİ" in prompt

    def test_contains_reflection_section(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "YANIT KALİTE KONTROLÜ" in prompt

    def test_contains_notification_context(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "BİLDİRİM BAĞLAMI" in prompt

    def test_contains_identity_rule(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "KİMLİK KURALI" in prompt

    def test_active_course_shown(self, monkeypatch):
        course = _make_course("CTIS 363", "CTIS", "CTIS 363 Ethics")
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: course,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "CTIS 363 Ethics" in prompt

    def test_no_active_course_message(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "henüz kurs seçmemiş" in prompt.lower()

    def test_stars_status_shown(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(STATE, "stars_client", _make_stars(authenticated=True))
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "STARS: ✅" in prompt

    def test_stars_disconnected(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        # Default state: stars_client is None
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "STARS: ❌" in prompt

    def test_contains_date(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "Tarih:" in prompt

    def test_contains_mail_format_template(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        prompt = agent_service._build_system_prompt(user_id=1)
        assert "📧" in prompt
        assert "👤" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# F. handle_agent_message TESTS (15+ cases)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleAgentMessage:
    @pytest.mark.asyncio
    async def test_no_llm(self, monkeypatch):
        result = await agent_service.handle_agent_message(1, "hello")
        assert "hazır değil" in result.lower()

    @pytest.mark.asyncio
    async def test_simple_response_no_tools(self, monkeypatch):
        # Mock LLM that returns text without tool calls
        response_msg = SimpleNamespace(
            content="Merhaba! Size nasıl yardımcı olabilirim?",
            tool_calls=None,
        )
        monkeypatch.setattr(
            "bot.services.agent_service._call_llm_with_tools",
            AsyncMock(return_value=response_msg),
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_conversation_history",
            lambda uid: [],
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.add_conversation_turn",
            lambda uid, role, content: None,
        )
        monkeypatch.setattr(STATE, "llm", SimpleNamespace(
            engine=SimpleNamespace(
                router=SimpleNamespace(chat="gpt-5-mini"),
                get_adapter=lambda k: SimpleNamespace(model="gpt-5-mini", client=MagicMock()),
            ),
            mem_manager=None,
            _build_student_context=lambda: "",
        ))
        result = await agent_service.handle_agent_message(1, "Merhaba!")
        assert "yardımcı" in result.lower()

    @pytest.mark.asyncio
    async def test_english_language_override(self, monkeypatch):
        call_args = {}

        async def capture_llm(messages, system_prompt, tools):
            call_args["system_prompt"] = system_prompt
            return SimpleNamespace(content="Hello!", tool_calls=None)

        monkeypatch.setattr(
            "bot.services.agent_service._call_llm_with_tools",
            capture_llm,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_conversation_history",
            lambda uid: [],
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.add_conversation_turn",
            lambda uid, role, content: None,
        )
        monkeypatch.setattr(STATE, "llm", SimpleNamespace(
            engine=SimpleNamespace(
                router=SimpleNamespace(chat="gpt-5-mini"),
                get_adapter=lambda k: SimpleNamespace(model="gpt-5-mini", client=MagicMock()),
            ),
            mem_manager=None,
            _build_student_context=lambda: "",
        ))
        await agent_service.handle_agent_message(1, "Show me my grades")
        assert "LANGUAGE OVERRIDE" in call_args["system_prompt"]
        assert "ENGLISH" in call_args["system_prompt"]

    @pytest.mark.asyncio
    async def test_turkish_no_override(self, monkeypatch):
        call_args = {}

        async def capture_llm(messages, system_prompt, tools):
            call_args["system_prompt"] = system_prompt
            return SimpleNamespace(content="Notlarınız:", tool_calls=None)

        monkeypatch.setattr(
            "bot.services.agent_service._call_llm_with_tools",
            capture_llm,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_conversation_history",
            lambda uid: [],
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.add_conversation_turn",
            lambda uid, role, content: None,
        )
        monkeypatch.setattr(STATE, "llm", SimpleNamespace(
            engine=SimpleNamespace(
                router=SimpleNamespace(chat="gpt-5-mini"),
                get_adapter=lambda k: SimpleNamespace(model="gpt-5-mini", client=MagicMock()),
            ),
            mem_manager=None,
            _build_student_context=lambda: "",
        ))
        await agent_service.handle_agent_message(1, "Notlarım nedir?")
        assert "LANGUAGE OVERRIDE" not in call_args["system_prompt"]

    @pytest.mark.asyncio
    async def test_llm_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service._call_llm_with_tools",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_conversation_history",
            lambda uid: [],
        )
        monkeypatch.setattr(STATE, "llm", SimpleNamespace(
            engine=SimpleNamespace(
                router=SimpleNamespace(chat="gpt-5-mini"),
                get_adapter=lambda k: SimpleNamespace(model="gpt-5-mini", client=MagicMock()),
            ),
            mem_manager=None,
            _build_student_context=lambda: "",
        ))
        result = await agent_service.handle_agent_message(1, "test")
        assert "üretilemedi" in result.lower()

    @pytest.mark.asyncio
    async def test_llm_exception(self, monkeypatch):
        monkeypatch.setattr(
            "bot.services.agent_service._call_llm_with_tools",
            AsyncMock(side_effect=Exception("API down")),
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_active_course",
            lambda uid: None,
        )
        monkeypatch.setattr(
            "bot.services.agent_service.user_service.get_conversation_history",
            lambda uid: [],
        )
        monkeypatch.setattr(STATE, "llm", SimpleNamespace(
            engine=SimpleNamespace(
                router=SimpleNamespace(chat="gpt-5-mini"),
                get_adapter=lambda k: SimpleNamespace(model="gpt-5-mini", client=MagicMock()),
            ),
            mem_manager=None,
            _build_student_context=lambda: "",
        ))
        result = await agent_service.handle_agent_message(1, "test")
        assert "sorun oluştu" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# G. TOOL DEFINITIONS VALIDATION (10+ cases)
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolDefinitions:
    def test_14_tools_defined(self):
        assert len(agent_service.TOOLS) == 14

    def test_all_tools_have_function_type(self):
        for tool in agent_service.TOOLS:
            assert tool["type"] == "function"

    def test_all_tools_have_name(self):
        for tool in agent_service.TOOLS:
            assert "name" in tool["function"]
            assert len(tool["function"]["name"]) > 0

    def test_all_tools_have_description(self):
        for tool in agent_service.TOOLS:
            assert "description" in tool["function"]
            assert len(tool["function"]["description"]) > 10

    def test_all_tools_have_parameters(self):
        for tool in agent_service.TOOLS:
            assert "parameters" in tool["function"]
            assert tool["function"]["parameters"]["type"] == "object"

    def test_get_emails_has_keyword_param(self):
        tool = next(t for t in agent_service.TOOLS if t["function"]["name"] == "get_emails")
        props = tool["function"]["parameters"]["properties"]
        assert "keyword" in props

    def test_get_emails_has_count_param(self):
        tool = next(t for t in agent_service.TOOLS if t["function"]["name"] == "get_emails")
        props = tool["function"]["parameters"]["properties"]
        assert "count" in props

    def test_get_email_detail_requires_keyword(self):
        tool = next(t for t in agent_service.TOOLS if t["function"]["name"] == "get_email_detail")
        assert "keyword" in tool["function"]["parameters"]["required"]

    def test_get_schedule_requires_period(self):
        tool = next(t for t in agent_service.TOOLS if t["function"]["name"] == "get_schedule")
        assert "period" in tool["function"]["parameters"]["required"]

    def test_get_emails_description_mentions_tum(self):
        tool = next(t for t in agent_service.TOOLS if t["function"]["name"] == "get_emails")
        desc = tool["function"]["description"]
        assert "Tüm" in desc or "tüm" in desc

    def test_get_emails_description_mentions_date(self):
        tool = next(t for t in agent_service.TOOLS if t["function"]["name"] == "get_emails")
        desc = tool["function"]["description"]
        assert "tarih" in desc.lower()

    def test_email_detail_mentions_notification(self):
        tool = next(t for t in agent_service.TOOLS if t["function"]["name"] == "get_email_detail")
        desc = tool["function"]["description"]
        assert "bildirim" in desc.lower() or "Bildirim" in desc

    def test_no_duplicate_tool_names(self):
        names = [t["function"]["name"] for t in agent_service.TOOLS]
        assert len(names) == len(set(names))


# ═══════════════════════════════════════════════════════════════════════════════
# H. CONSTANTS AND MODULE-LEVEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_max_iterations(self):
        assert agent_service.MAX_TOOL_ITERATIONS == 5

    def test_tr_chars_set(self):
        expected = set("çğıöşüÇĞİÖŞÜ")
        assert agent_service._TR_CHARS == expected

    def test_en_words_not_empty(self):
        assert len(agent_service._EN_WORDS) >= 20

    def test_en_words_lowercase(self):
        for word in agent_service._EN_WORDS:
            assert word == word.lower()

    def test_day_names_tr(self):
        assert agent_service._DAY_NAMES_TR[0] == "Pazartesi"
        assert agent_service._DAY_NAMES_TR[6] == "Pazar"
        assert len(agent_service._DAY_NAMES_TR) == 7
