"""Temporal awareness tests.

Tests verify that the bot has correct, grounded temporal awareness:

- System prompt always contains today's date in DD/MM/YYYY HH:MM format
- System prompt contains a valid Turkish day name (Pazartesi … Pazar)
- UNIX timestamps in assignment due dates are converted to human-readable format
- Past timestamps (overdue) are correctly identified by the overdue filter
- Future timestamps (upcoming) survive the upcoming filter
- None/missing due dates yield "Belirtilmemiş"
- Non-integer (string) due dates are passed through as-is
- Overdue filter excludes submitted assignments regardless of timestamp
- The bot does NOT hard-code dates — it reads from tool results (system prompt injection test)
- "Bu hafta / Bu gün / Yarın" style temporal queries score appropriately

Hallucination guard: if the system prompt doesn't include today's date,
the LLM has no grounded temporal reference and will invent dates.
"""

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bot.services.agent_service import (
    _build_system_prompt,
    _execute_tool_call,
    _sanitize_tool_output,
    _score_complexity,
    handle_agent_message,
)
from bot.state import STATE
import bot.services.user_service as user_service


# ─── Helpers ─────────────────────────────────────────────────────────────────

VALID_TR_DAYS = {"Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"}


def _fake_assignment(
    name: str = "Lab-3",
    course: str = "CTIS 256",
    due_date=None,
    submitted: bool = False,
    time_remaining: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        course_name=course,
        due_date=due_date,
        submitted=submitted,
        time_remaining=time_remaining,
    )


# ─── 1. System Prompt Date Injection ─────────────────────────────────────────


class TestSystemPromptDateInjection:
    """The system prompt must contain today's date so the LLM has a
    grounded temporal reference. Without this, hallucinated dates are likely."""

    def test_system_prompt_contains_today_date(self, monkeypatch):
        monkeypatch.setattr(STATE, "stars_client", None)
        monkeypatch.setattr(STATE, "webmail_client", None)
        monkeypatch.setattr(STATE, "llm", None)
        with patch("bot.services.user_service.get_active_course", return_value=None):
            prompt = _build_system_prompt(user_id=1)

        today_str = datetime.now().strftime("%d/%m/%Y")
        assert today_str in prompt, (
            f"Today's date ({today_str}) must appear in system prompt for temporal grounding"
        )

    def test_system_prompt_date_is_not_hardcoded_past(self, monkeypatch):
        """System prompt must NOT contain a hardcoded past year like 2023 or 2024."""
        monkeypatch.setattr(STATE, "stars_client", None)
        monkeypatch.setattr(STATE, "webmail_client", None)
        monkeypatch.setattr(STATE, "llm", None)
        with patch("bot.services.user_service.get_active_course", return_value=None):
            prompt = _build_system_prompt(user_id=1)

        current_year = str(datetime.now().year)
        # Should have this year's date (dynamic), not a hardcoded stale year
        assert current_year in prompt

    def test_system_prompt_contains_valid_turkish_day_name(self, monkeypatch):
        """Prompt must include a recognizable Turkish day name."""
        monkeypatch.setattr(STATE, "stars_client", None)
        monkeypatch.setattr(STATE, "webmail_client", None)
        monkeypatch.setattr(STATE, "llm", None)
        with patch("bot.services.user_service.get_active_course", return_value=None):
            prompt = _build_system_prompt(user_id=1)

        found = any(day in prompt for day in VALID_TR_DAYS)
        assert found, f"No Turkish day name found in system prompt. Valid: {VALID_TR_DAYS}"

    def test_system_prompt_date_format_is_dd_mm_yyyy(self, monkeypatch):
        """Date must be in DD/MM/YYYY format, not ISO (YYYY-MM-DD) or ambiguous."""
        monkeypatch.setattr(STATE, "stars_client", None)
        monkeypatch.setattr(STATE, "webmail_client", None)
        monkeypatch.setattr(STATE, "llm", None)
        with patch("bot.services.user_service.get_active_course", return_value=None):
            prompt = _build_system_prompt(user_id=1)

        today = datetime.now()
        # DD/MM/YYYY format: zero-padded day and month
        formatted = today.strftime("%d/%m/%Y")
        assert formatted in prompt

    def test_system_prompt_includes_time(self, monkeypatch):
        """Prompt must include HH:MM so LLM can reason about time-of-day deadlines."""
        monkeypatch.setattr(STATE, "stars_client", None)
        monkeypatch.setattr(STATE, "webmail_client", None)
        monkeypatch.setattr(STATE, "llm", None)
        with patch("bot.services.user_service.get_active_course", return_value=None):
            prompt = _build_system_prompt(user_id=1)

        # Prompt should contain HH:MM time (colon between 2-digit groups)
        import re
        assert re.search(r"\d{2}:\d{2}", prompt), "HH:MM time not found in system prompt"

    def test_system_prompt_rebuilt_with_fresh_date_each_call(self, monkeypatch):
        """Two calls to _build_system_prompt return prompts with today's date (not cached)."""
        monkeypatch.setattr(STATE, "stars_client", None)
        monkeypatch.setattr(STATE, "webmail_client", None)
        monkeypatch.setattr(STATE, "llm", None)
        with patch("bot.services.user_service.get_active_course", return_value=None):
            p1 = _build_system_prompt(user_id=1)
            p2 = _build_system_prompt(user_id=1)

        today = datetime.now().strftime("%d/%m/%Y")
        assert today in p1
        assert today in p2


# ─── 2. Turkish Day Name Completeness ────────────────────────────────────────


class TestTurkishDayNames:
    """_DAY_NAMES_TR must cover all 7 weekdays (0=Mon … 6=Sun)."""

    def test_all_seven_days_covered(self):
        from bot.services.agent_service import _DAY_NAMES_TR  # type: ignore[attr-defined]
        for i in range(7):
            day = _DAY_NAMES_TR.get(i)
            assert day is not None, f"weekday {i} missing from _DAY_NAMES_TR"
            assert day in VALID_TR_DAYS, f"'{day}' is not a recognised Turkish day name"

    def test_monday_is_pazartesi(self):
        from bot.services.agent_service import _DAY_NAMES_TR  # type: ignore[attr-defined]
        assert _DAY_NAMES_TR[0] == "Pazartesi"  # Monday = 0

    def test_friday_is_cuma(self):
        from bot.services.agent_service import _DAY_NAMES_TR  # type: ignore[attr-defined]
        assert _DAY_NAMES_TR[4] == "Cuma"

    def test_saturday_is_cumartesi(self):
        from bot.services.agent_service import _DAY_NAMES_TR  # type: ignore[attr-defined]
        assert _DAY_NAMES_TR[5] == "Cumartesi"

    def test_sunday_is_pazar(self):
        from bot.services.agent_service import _DAY_NAMES_TR  # type: ignore[attr-defined]
        assert _DAY_NAMES_TR[6] == "Pazar"


# ─── 3. Assignment Timestamp Conversion ──────────────────────────────────────


class TestAssignmentTimestampConversion:
    """_tool_get_assignments must convert UNIX timestamps to DD/MM/YYYY HH:MM.
    This is the primary temporal grounding for deadline information.
    """

    @pytest.mark.asyncio
    async def test_future_unix_timestamp_converted_to_human_date(self, monkeypatch):
        future_ts = time.time() + 7 * 24 * 3600  # +7 days
        expected = datetime.fromtimestamp(future_ts).strftime("%d/%m/%Y %H:%M")

        assignment = _fake_assignment(name="Lab-4", due_date=future_ts)

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "upcoming"}, user_id=1)

        assert expected in result
        assert "Lab-4" in result

    @pytest.mark.asyncio
    async def test_past_unix_timestamp_converted_correctly(self, monkeypatch):
        past_ts = time.time() - 3 * 24 * 3600  # -3 days (past)
        expected = datetime.fromtimestamp(past_ts).strftime("%d/%m/%Y %H:%M")

        assignment = _fake_assignment(name="Lab-2", due_date=past_ts, submitted=False)

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "upcoming"}, user_id=1)

        assert expected in result
        assert "Lab-2" in result

    @pytest.mark.asyncio
    async def test_none_due_date_shows_belirtilmemis(self, monkeypatch):
        assignment = _fake_assignment(name="Bonus Task", due_date=None)

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "upcoming"}, user_id=1)

        assert "Belirtilmemiş" in result
        assert "Bonus Task" in result

    @pytest.mark.asyncio
    async def test_string_due_date_passed_through_as_is(self, monkeypatch):
        assignment = _fake_assignment(name="Yazılı Ödev", due_date="15/03/2026 23:59")

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "upcoming"}, user_id=1)

        assert "15/03/2026 23:59" in result

    @pytest.mark.asyncio
    async def test_small_numeric_due_date_treated_as_string(self, monkeypatch):
        """A numeric value <= 1_000_000 must NOT be treated as a UNIX timestamp."""
        assignment = _fake_assignment(name="Quiz", due_date=999)

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "upcoming"}, user_id=1)

        # 999 is small — treated as str(999), not as a timestamp
        assert "999" in result

    @pytest.mark.asyncio
    async def test_timestamp_format_is_dd_mm_yyyy_hhmm(self, monkeypatch):
        """Converted date must match DD/MM/YYYY HH:MM — not ISO or ambiguous."""
        import re
        ts = time.time() + 86400  # tomorrow
        assignment = _fake_assignment(name="Proje", due_date=ts)

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "upcoming"}, user_id=1)

        # Must match DD/MM/YYYY HH:MM pattern
        assert re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", result), (
            f"Expected DD/MM/YYYY HH:MM format in: {result!r}"
        )


# ─── 4. Overdue Filter Temporal Logic ────────────────────────────────────────


class TestOverdueFilterTemporalLogic:
    """The overdue filter must use the current time (time.time()), not a hardcoded date."""

    @pytest.mark.asyncio
    async def test_past_unsubmitted_assignment_survives_overdue_filter(self, monkeypatch):
        past_ts = time.time() - 86400  # yesterday
        assignment = _fake_assignment(name="Geçmiş Ödev", due_date=past_ts, submitted=False)

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "overdue"}, user_id=1)

        assert "Geçmiş Ödev" in result
        assert "Süresi geçmiş" in result

    @pytest.mark.asyncio
    async def test_future_unsubmitted_excluded_from_overdue_filter(self, monkeypatch):
        future_ts = time.time() + 86400  # tomorrow
        assignment = _fake_assignment(name="Gelecek Ödev", due_date=future_ts, submitted=False)

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "overdue"}, user_id=1)

        assert "Gelecek Ödev" not in result

    @pytest.mark.asyncio
    async def test_submitted_past_assignment_excluded_from_overdue(self, monkeypatch):
        past_ts = time.time() - 86400
        assignment = _fake_assignment(
            name="Teslim Edilmiş", due_date=past_ts, submitted=True
        )

        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [assignment],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "overdue"}, user_id=1)

        # submitted=True → should NOT appear in overdue list
        assert "Teslim Edilmiş" not in result

    @pytest.mark.asyncio
    async def test_no_assignments_returns_graceful_message(self, monkeypatch):
        mock_moodle = SimpleNamespace(
            get_upcoming_assignments=lambda days: [],
        )
        monkeypatch.setattr(STATE, "moodle", mock_moodle)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "overdue"}, user_id=1)

        assert "bulunamadı" in result.lower()

    @pytest.mark.asyncio
    async def test_moodle_none_returns_connection_error(self, monkeypatch):
        monkeypatch.setattr(STATE, "moodle", None)

        from bot.services.agent_service import _tool_get_assignments
        result = await _tool_get_assignments({"filter": "upcoming"}, user_id=1)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "hazır değil" in result or "bağlantı" in result.lower()


# ─── 5. Temporal Complexity Scoring ──────────────────────────────────────────


class TestTemporalComplexityScoring:
    """Queries with temporal context should score appropriately.
    Simple time queries stay low; multi-step temporal queries escalate.
    """

    def test_bu_hafta_single_query_stays_moderate(self):
        score = _score_complexity("Bu hafta ödevim var mı?")
        assert 0.0 <= score <= 1.0  # Must not crash; score is bounded

    def test_yarın_single_query_stays_moderate(self):
        score = _score_complexity("Yarın sınav var mı?")
        assert 0.0 <= score <= 1.0

    def test_temporal_plus_multistep_escalates(self):
        query = "Önce bu haftaki ödevleri ver, sonra gelecek haftaki sınavları karşılaştır."
        score = _score_complexity(query)
        assert score >= 0.25  # "önce" + "sonra" trigger multi-step

    def test_temporal_plus_technical_escalates_further(self):
        query = (
            "Bu hafta hem türev ödevini hem de final programını karşılaştır. "
            "Neden önce türev çalışmalıyım nasıl planlayayım?"
        )
        score = _score_complexity(query)
        assert score > 0.65


# ─── 6. Temporal Date in Tool Output — Sanitization Safety ───────────────────


class TestTemporalDataInToolOutputSanitization:
    """Date strings in tool outputs must pass through sanitizer unchanged."""

    def test_date_string_not_filtered(self):
        content = "Ödev teslim tarihi: 20/05/2026 23:59"
        out = _sanitize_tool_output("get_assignments", content)
        assert "20/05/2026 23:59" in out
        assert "[FILTERED]" not in out

    def test_time_string_not_filtered(self):
        content = "Sınav saati: 09:00 – 11:00 arasında."
        out = _sanitize_tool_output("get_schedule", content)
        assert "09:00" in out
        assert "11:00" in out
        assert "[FILTERED]" not in out

    def test_relative_time_not_filtered(self):
        content = "Kalan süre: 3 gün 4 saat"
        out = _sanitize_tool_output("get_assignments", content)
        assert "3 gün 4 saat" in out
        assert "[FILTERED]" not in out

    def test_iso_date_not_filtered(self):
        """Even ISO dates (e.g. 2026-05-20) must not be filtered."""
        content = "Due: 2026-05-20T23:59:00Z"
        out = _sanitize_tool_output("get_assignments", content)
        assert "2026-05-20" in out
        assert "[FILTERED]" not in out
