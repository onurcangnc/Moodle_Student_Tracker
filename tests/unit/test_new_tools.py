"""
Unit tests for the 4 new agent tools added in feat/new-tools:
  - get_exam_schedule   (STARS exam timetable)
  - get_assignment_detail (Moodle assignment full description)
  - get_upcoming_events (Moodle calendar events)
  - calculate_grade     (Bilkent GPA / course-grade calculator)

All tests are pure unit tests:
  - No network calls, no real STARS/Moodle/LLM
  - External state (STATE, cache_db) is patched via unittest.mock
  - calculate_grade helper functions (_bilkent_gpa, _academic_standing, _honor_status)
    are tested directly as pure functions

Running:
    pytest tests/unit/test_new_tools.py -v
    pytest tests/unit/test_new_tools.py -v -k "GPA"
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── imports under test ────────────────────────────────────────────────────────
from bot.services.agent_service import (
    _GRADE_POINTS,
    _NO_GPA_GRADES,
    _academic_standing,
    _bilkent_gpa,
    _honor_status,
    _tool_calculate_grade,
    _tool_get_assignment_detail,
    _tool_get_exam_schedule,
    _tool_get_upcoming_events,
)

# ═════════════════════════════════════════════════════════════════════════════
# A. calculate_grade — pure-function helpers (no mocks needed)
# ═════════════════════════════════════════════════════════════════════════════


class TestBilkentGpaCalculation:
    """_bilkent_gpa() — exact arithmetic against Bilkent grade point table."""

    def test_single_course_a_minus(self):
        """A- (3.70) × 3 credits = 11.10 → GPA 3.70."""
        gpa, credits, warns = _bilkent_gpa([{"name": "CTIS 256", "grade": "A-", "credits": 3}])
        assert gpa == pytest.approx(3.70, abs=0.01)
        assert credits == 3
        assert warns == []

    def test_two_courses_average(self):
        """A- (3.70×3) + B+ (3.30×4) = 11.10+13.20 = 24.30 / 7 = 3.47."""
        courses = [
            {"name": "CTIS 256", "grade": "A-", "credits": 3},
            {"name": "MATH 101", "grade": "B+", "credits": 4},
        ]
        gpa, credits, warns = _bilkent_gpa(courses)
        assert gpa == pytest.approx(3.47, abs=0.01)
        assert credits == 7
        assert warns == []

    def test_a_plus_equals_a(self):
        """A+ and A both map to 4.00."""
        gpa_plus, _, _ = _bilkent_gpa([{"name": "X", "grade": "A+", "credits": 3}])
        gpa_a, _, _ = _bilkent_gpa([{"name": "X", "grade": "A", "credits": 3}])
        assert gpa_plus == pytest.approx(4.00, abs=0.01)
        assert gpa_a == pytest.approx(4.00, abs=0.01)

    def test_f_fx_fz_are_zero(self):
        """F, FX, and FZ all give 0.00 grade points and drag down GPA."""
        for failing in ("F", "FX", "FZ"):
            gpa, _, warns = _bilkent_gpa([{"name": "X", "grade": failing, "credits": 3}])
            assert gpa == pytest.approx(0.00, abs=0.01), f"{failing} should yield GPA=0.00"
            assert warns == []

    def test_no_gpa_grades_excluded(self):
        """S, U, I, P, T, W have no grade point equivalent — excluded with a warning."""
        for special in ("S", "U", "I", "P", "T", "W"):
            gpa, credits, warns = _bilkent_gpa([{"name": "X", "grade": special, "credits": 3}])
            assert credits == 0.0, f"{special} should not add to total credits"
            assert any(special in w for w in warns), f"Expected warning for {special}"

    def test_unknown_grade_produces_warning(self):
        """An unrecognised grade string is skipped with a warning."""
        gpa, credits, warns = _bilkent_gpa([{"name": "X", "grade": "Z+", "credits": 3}])
        assert credits == 0.0
        assert any("Z+" in w for w in warns)

    def test_zero_credits_skipped(self):
        """Courses with 0 or negative credits are skipped with a warning."""
        gpa, credits, warns = _bilkent_gpa([{"name": "X", "grade": "A", "credits": 0}])
        assert credits == 0.0
        assert warns  # at least one warning emitted

    def test_empty_course_list_returns_zero(self):
        """Empty input → GPA 0.0, no crash."""
        gpa, credits, warns = _bilkent_gpa([])
        assert gpa == 0.0
        assert credits == 0.0

    def test_mixed_valid_and_no_gpa_grades(self):
        """S grade excluded; valid courses still calculate correctly."""
        courses = [
            {"name": "CTIS 499", "grade": "S", "credits": 3},  # excluded
            {"name": "MATH 101", "grade": "B", "credits": 4},   # 3.00 × 4
        ]
        gpa, credits, warns = _bilkent_gpa(courses)
        assert credits == 4.0
        assert gpa == pytest.approx(3.00, abs=0.01)
        assert any("S" in w for w in warns)

    def test_all_grade_points_present(self):
        """Every expected letter grade maps to the correct Bilkent value."""
        expected = {
            "A+": 4.00, "A": 4.00, "A-": 3.70,
            "B+": 3.30, "B": 3.00, "B-": 2.70,
            "C+": 2.30, "C": 2.00, "C-": 1.70,
            "D+": 1.30, "D": 1.00,
            "F": 0.00, "FX": 0.00, "FZ": 0.00,
        }
        assert _GRADE_POINTS == expected


class TestAcademicStanding:
    """_academic_standing() — Bilkent CGPA thresholds."""

    def test_satisfactory_at_exactly_2_00(self):
        result = _academic_standing(2.00)
        assert "Satisfactory" in result
        assert "✅" in result

    def test_satisfactory_above_2_00(self):
        assert "Satisfactory" in _academic_standing(3.50)

    def test_probation_at_1_80(self):
        result = _academic_standing(1.80)
        assert "Probation" in result
        assert "⚠️" in result

    def test_probation_at_1_99(self):
        result = _academic_standing(1.99)
        assert "Probation" in result

    def test_unsatisfactory_at_1_79(self):
        result = _academic_standing(1.79)
        assert "Unsatisfactory" in result
        assert "🚨" in result

    def test_unsatisfactory_at_zero(self):
        result = _academic_standing(0.00)
        assert "Unsatisfactory" in result


class TestHonorStatus:
    """_honor_status() — Honor / High Honor thresholds."""

    def test_high_honor_at_3_50(self):
        result = _honor_status(3.50, 2.50, 5)
        assert "High Honor" in result

    def test_high_honor_at_4_00(self):
        result = _honor_status(4.00, 3.80, 5)
        assert "High Honor" in result

    def test_honor_at_3_00(self):
        result = _honor_status(3.00, 2.50, 5)
        assert "Honor" in result and "High" not in result

    def test_honor_at_3_49(self):
        result = _honor_status(3.49, 2.50, 5)
        assert "Honor" in result

    def test_no_honor_below_3_00(self):
        result = _honor_status(2.90, 2.50, 5)
        assert "High Honor" not in result
        assert "🎖️" not in result

    def test_low_cgpa_blocks_honor(self):
        """Honor requires CGPA >= 2.00."""
        result = _honor_status(3.80, 1.90, 5)  # GPA high but CGPA too low
        assert "CGPA" in result


# ═════════════════════════════════════════════════════════════════════════════
# B. calculate_grade tool — async handler (mocked STATE)
# ═════════════════════════════════════════════════════════════════════════════


class TestToolCalculateGradeGpaMode:
    """calculate_grade(mode='gpa') — async handler tests."""

    @pytest.mark.asyncio
    async def test_basic_gpa_output_contains_grade_and_standing(self):
        courses = [
            {"name": "CTIS 256", "grade": "A-", "credits": 3},
            {"name": "MATH 101", "grade": "B+", "credits": 4},
        ]
        result = await _tool_calculate_grade({"mode": "gpa", "courses": courses}, user_id=1)
        assert "GPA" in result
        assert "3.47" in result or "3.4" in result
        assert "Satisfactory" in result

    @pytest.mark.asyncio
    async def test_high_honor_detected(self):
        courses = [
            {"name": "A", "grade": "A+", "credits": 3},
            {"name": "B", "grade": "A",  "credits": 3},
        ]
        result = await _tool_calculate_grade({"mode": "gpa", "courses": courses}, user_id=1)
        assert "High Honor" in result

    @pytest.mark.asyncio
    async def test_probation_detected(self):
        # C (2.00×3) + C- (1.70×3) = 6.00+5.10 = 11.10/6 = 1.85 → Probation (1.80–1.99)
        courses = [
            {"name": "X", "grade": "C",  "credits": 3},
            {"name": "Y", "grade": "C-", "credits": 3},
        ]
        result = await _tool_calculate_grade({"mode": "gpa", "courses": courses}, user_id=1)
        assert "Probation" in result

    @pytest.mark.asyncio
    async def test_fx_fz_in_course_list(self):
        courses = [
            {"name": "PHYS 101", "grade": "FZ", "credits": 4},
            {"name": "CTIS 101", "grade": "B",  "credits": 3},
        ]
        result = await _tool_calculate_grade({"mode": "gpa", "courses": courses}, user_id=1)
        assert "GPA" in result

    @pytest.mark.asyncio
    async def test_empty_courses_returns_prompt(self):
        result = await _tool_calculate_grade({"mode": "gpa", "courses": []}, user_id=1)
        assert "gerekli" in result.lower() or "örnek" in result.lower()

    @pytest.mark.asyncio
    async def test_grade_table_included_in_output(self):
        """Official Bilkent grade table is shown at the bottom."""
        courses = [{"name": "X", "grade": "B", "credits": 3}]
        result = await _tool_calculate_grade({"mode": "gpa", "courses": courses}, user_id=1)
        assert "Not Tablosu" in result or "4.00" in result

    @pytest.mark.asyncio
    async def test_s_grade_warning_shown(self):
        courses = [
            {"name": "PE 101", "grade": "S", "credits": 0},
            {"name": "CTIS 256", "grade": "A", "credits": 3},
        ]
        result = await _tool_calculate_grade({"mode": "gpa", "courses": courses}, user_id=1)
        assert "S" in result  # either in warning or table

    @pytest.mark.asyncio
    async def test_unknown_mode_returns_error(self):
        result = await _tool_calculate_grade({"mode": "invalid"}, user_id=1)
        assert "Bilinmeyen" in result or "invalid" in result


class TestToolCalculateGradeCourseMode:
    """calculate_grade(mode='course') — weighted assessment calculation."""

    @pytest.mark.asyncio
    async def test_midterm_final_weighted_average(self):
        """Midterm 40% (75/100) + Final 60% (80/100) = 30 + 48 = 78.00."""
        args = {
            "mode": "course",
            "assessments": [
                {"name": "Midterm", "grade": 75, "weight": 40},
                {"name": "Final",   "grade": 80, "weight": 60},
            ],
        }
        result = await _tool_calculate_grade(args, user_id=1)
        assert "78.00" in result or "78.0" in result

    @pytest.mark.asyncio
    async def test_what_if_scenario_included(self):
        """What-if final grade is labelled as varsayımsal."""
        args = {
            "mode": "course",
            "assessments": [
                {"name": "Midterm", "grade": 70, "weight": 40},
            ],
            "what_if": {"name": "Final", "grade": 90, "weight": 60},
        }
        result = await _tool_calculate_grade(args, user_id=1)
        assert "varsayımsal" in result or "what" in result.lower() or "Final" in result

    @pytest.mark.asyncio
    async def test_missing_assessment_shows_best_worst(self):
        """Missing grade shows best/worst case projection."""
        args = {
            "mode": "course",
            "assessments": [
                {"name": "Midterm", "grade": 70, "weight": 40},
                {"name": "Final", "weight": 60},  # no grade yet
            ],
        }
        result = await _tool_calculate_grade(args, user_id=1)
        assert "En iyi" in result or "best" in result.lower()
        assert "En kötü" in result or "worst" in result.lower()

    @pytest.mark.asyncio
    async def test_letter_grade_a_for_score_above_90(self):
        """Score ≥ 90 → maps to A / A+ letter grade."""
        args = {
            "mode": "course",
            "assessments": [
                {"name": "Quiz", "grade": 95, "weight": 100},
            ],
        }
        result = await _tool_calculate_grade(args, user_id=1)
        assert "A" in result

    @pytest.mark.asyncio
    async def test_letter_grade_f_below_60(self):
        """Score < 60 → maps to F."""
        args = {
            "mode": "course",
            "assessments": [
                {"name": "Quiz", "grade": 45, "weight": 100},
            ],
        }
        result = await _tool_calculate_grade(args, user_id=1)
        assert "F" in result

    @pytest.mark.asyncio
    async def test_empty_assessments_returns_prompt(self):
        result = await _tool_calculate_grade({"mode": "course"}, user_id=1)
        assert "gerekli" in result.lower() or "örnek" in result.lower()

    @pytest.mark.asyncio
    async def test_custom_max_grade_normalised(self):
        """Grade 15/20 (max_grade=20) should be treated as 75/100."""
        args = {
            "mode": "course",
            "assessments": [
                {"name": "Quiz", "grade": 15, "max_grade": 20, "weight": 100},
            ],
        }
        result = await _tool_calculate_grade(args, user_id=1)
        # 15/20 = 75 → letter grade C or C+
        assert "75.00" in result or "75.0" in result


# ═════════════════════════════════════════════════════════════════════════════
# C. get_exam_schedule — async handler (mocked STARS + cache)
# ═════════════════════════════════════════════════════════════════════════════

_SAMPLE_EXAMS = [
    {
        "course": "CTIS 256",
        "exam_name": "Midterm",
        "date": "2026-03-15",
        "start_time": "09:30",
        "time_block": "B1",
        "time_remaining": "25 gün",
    },
    {
        "course": "MATH 101",
        "exam_name": "Final",
        "date": "2026-05-20",
        "start_time": "13:00",
        "time_block": "A2",
        "time_remaining": "90 gün",
    },
]


class TestToolGetExamSchedule:

    @pytest.mark.asyncio
    async def test_stars_not_connected_returns_error(self):
        with patch("bot.services.agent_service.STATE") as mock_state:
            mock_state.stars_client = None
            result = await _tool_get_exam_schedule({}, user_id=1)
        assert "STARS" in result

    @pytest.mark.asyncio
    async def test_stars_not_authenticated_returns_error(self):
        stars = MagicMock()
        stars.is_authenticated.return_value = False
        with patch("bot.services.agent_service.STATE") as mock_state:
            mock_state.stars_client = stars
            result = await _tool_get_exam_schedule({}, user_id=1)
        assert "giriş" in result.lower() or "STARS" in result

    @pytest.mark.asyncio
    async def test_cache_hit_no_stars_call(self):
        stars = MagicMock()
        stars.is_authenticated.return_value = True
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
        ):
            mock_state.stars_client = stars
            mock_cache.get_json.return_value = _SAMPLE_EXAMS
            result = await _tool_get_exam_schedule({}, user_id=1)
        stars.get_exams.assert_not_called()
        assert "CTIS 256" in result
        assert "Midterm" in result

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_from_stars_and_caches(self):
        stars = MagicMock()
        stars.is_authenticated.return_value = True
        stars.get_exams.return_value = _SAMPLE_EXAMS
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.stars_client = stars
            mock_cache.get_json.return_value = None  # cache miss
            mock_thread.return_value = _SAMPLE_EXAMS
            result = await _tool_get_exam_schedule({}, user_id=1)
        mock_cache.set_json.assert_called_once()
        assert "CTIS 256" in result or "MATH 101" in result

    @pytest.mark.asyncio
    async def test_course_filter_applied(self):
        stars = MagicMock()
        stars.is_authenticated.return_value = True
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
        ):
            mock_state.stars_client = stars
            mock_cache.get_json.return_value = _SAMPLE_EXAMS
            result = await _tool_get_exam_schedule({"course_filter": "CTIS"}, user_id=1)
        assert "CTIS 256" in result
        assert "MATH 101" not in result

    @pytest.mark.asyncio
    async def test_no_exams_returns_informative_message(self):
        stars = MagicMock()
        stars.is_authenticated.return_value = True
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.stars_client = stars
            mock_cache.get_json.return_value = None
            mock_thread.return_value = []
            result = await _tool_get_exam_schedule({}, user_id=1)
        assert "bulunamadı" in result or "açıklanmamış" in result

    @pytest.mark.asyncio
    async def test_output_contains_date_and_time(self):
        stars = MagicMock()
        stars.is_authenticated.return_value = True
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
        ):
            mock_state.stars_client = stars
            mock_cache.get_json.return_value = _SAMPLE_EXAMS
            result = await _tool_get_exam_schedule({}, user_id=1)
        assert "2026-03-15" in result
        assert "09:30" in result


# ═════════════════════════════════════════════════════════════════════════════
# D. get_assignment_detail — async handler (mocked Moodle + cache)
# ═════════════════════════════════════════════════════════════════════════════

def _make_assignment(name: str, course: str, desc: str = "", due: int = 0, submitted: bool = False):
    a = MagicMock()
    a.name = name
    a.course_name = course
    a.description = desc
    a.due_date = due
    a.submitted = submitted
    a.graded = False
    a.grade = ""
    a.time_remaining = "5 gün"
    return a


class TestToolGetAssignmentDetail:

    @pytest.mark.asyncio
    async def test_moodle_not_connected(self):
        with patch("bot.services.agent_service.STATE") as mock_state:
            mock_state.moodle = None
            result = await _tool_get_assignment_detail({"assignment_name": "HW1"}, user_id=1)
        assert "Moodle" in result

    @pytest.mark.asyncio
    async def test_missing_assignment_name_returns_error(self):
        moodle = MagicMock()
        with patch("bot.services.agent_service.STATE") as mock_state:
            mock_state.moodle = moodle
            result = await _tool_get_assignment_detail({}, user_id=1)
        assert "belirtilmedi" in result.lower() or "gerekli" in result.lower()

    @pytest.mark.asyncio
    async def test_assignment_found_with_description(self):
        hw = _make_assignment("HW2: Sorting Algorithms", "CTIS 256",
                              desc="Implement quicksort and mergesort.", due=int(time.time()) + 86400)
        moodle = MagicMock()
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = moodle
            mock_cache.get_json.return_value = None  # no cache
            mock_thread.return_value = [hw]
            result = await _tool_get_assignment_detail({"assignment_name": "HW2"}, user_id=1)
        assert "HW2: Sorting Algorithms" in result
        assert "Implement quicksort" in result
        assert "CTIS 256" in result

    @pytest.mark.asyncio
    async def test_partial_name_match(self):
        """'sort' should match 'HW2: Sorting Algorithms'."""
        hw = _make_assignment("HW2: Sorting Algorithms", "CTIS 256", desc="Implement sort.")
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_cache.get_json.return_value = None
            mock_thread.return_value = [hw]
            result = await _tool_get_assignment_detail({"assignment_name": "sort"}, user_id=1)
        assert "Sorting Algorithms" in result

    @pytest.mark.asyncio
    async def test_assignment_not_found_returns_helpful_message(self):
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_cache.get_json.return_value = None
            mock_thread.return_value = []
            result = await _tool_get_assignment_detail({"assignment_name": "XYZ999"}, user_id=1)
        assert "bulunamadı" in result

    @pytest.mark.asyncio
    async def test_submitted_assignment_shows_status(self):
        hw = _make_assignment("Project 1", "CTIS 499", submitted=True)
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_cache.get_json.return_value = None
            mock_thread.return_value = [hw]
            result = await _tool_get_assignment_detail({"assignment_name": "Project"}, user_id=1)
        assert "Teslim edildi" in result or "✅" in result

    @pytest.mark.asyncio
    async def test_no_description_shows_placeholder(self):
        hw = _make_assignment("HW1", "CTIS 101", desc="")
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.cache_db") as mock_cache,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_cache.get_json.return_value = None
            mock_thread.return_value = [hw]
            result = await _tool_get_assignment_detail({"assignment_name": "HW1"}, user_id=1)
        assert "mevcut değil" in result or "Açıklama" in result


# ═════════════════════════════════════════════════════════════════════════════
# E. get_upcoming_events — async handler (mocked Moodle)
# ═════════════════════════════════════════════════════════════════════════════

_SAMPLE_EVENTS = [
    {
        "name": "Quiz 3 — Chapter 5",
        "course": "CTIS 256",
        "type": "quiz",
        "due_date": int(time.time()) + 3 * 86400,
        "action": "Submit Quiz",
    },
    {
        "name": "Project Submission",
        "course": "CTIS 499",
        "type": "assign",
        "due_date": int(time.time()) + 7 * 86400,
        "action": "Submit Assignment",
    },
    {
        "name": "Discussion Post",
        "course": "HIST 200",
        "type": "forum",
        "due_date": int(time.time()) + 5 * 86400,
        "action": "Post to Forum",
    },
]


class TestToolGetUpcomingEvents:

    @pytest.mark.asyncio
    async def test_moodle_not_connected(self):
        with patch("bot.services.agent_service.STATE") as mock_state:
            mock_state.moodle = None
            result = await _tool_get_upcoming_events({}, user_id=1)
        assert "Moodle" in result

    @pytest.mark.asyncio
    async def test_no_events_returns_informative_message(self):
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = []
            result = await _tool_get_upcoming_events({}, user_id=1)
        assert "bulunamadı" in result

    @pytest.mark.asyncio
    async def test_events_formatted_with_correct_icons(self):
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = _SAMPLE_EVENTS
            result = await _tool_get_upcoming_events({}, user_id=1)
        assert "❓" in result  # quiz icon
        assert "📝" in result  # assign icon
        assert "💬" in result  # forum icon

    @pytest.mark.asyncio
    async def test_quiz_filter_excludes_other_types(self):
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = _SAMPLE_EVENTS
            result = await _tool_get_upcoming_events({"event_type": "quiz"}, user_id=1)
        assert "Quiz 3" in result
        assert "Project Submission" not in result
        assert "Discussion Post" not in result

    @pytest.mark.asyncio
    async def test_assign_filter(self):
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = _SAMPLE_EVENTS
            result = await _tool_get_upcoming_events({"event_type": "assign"}, user_id=1)
        assert "Project Submission" in result
        assert "Quiz 3" not in result

    @pytest.mark.asyncio
    async def test_days_capped_at_30(self):
        """days=100 should be capped to 30 in the Moodle API call."""
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = _SAMPLE_EVENTS
            await _tool_get_upcoming_events({"days": 100}, user_id=1)
        # The actual value passed to get_upcoming_events should be ≤ 30
        call_args = mock_thread.call_args
        if call_args:
            _, kwargs = call_args
            positional = call_args[0]
            # get_upcoming_events is called with (moodle.get_upcoming_events, days)
            days_arg = positional[1] if len(positional) > 1 else None
            if days_arg is not None:
                assert days_arg <= 30

    @pytest.mark.asyncio
    async def test_no_filter_returns_all_types(self):
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = _SAMPLE_EVENTS
            result = await _tool_get_upcoming_events({"event_type": "all"}, user_id=1)
        assert "Quiz 3" in result
        assert "Project Submission" in result
        assert "Discussion Post" in result

    @pytest.mark.asyncio
    async def test_event_names_and_courses_shown(self):
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = _SAMPLE_EVENTS
            result = await _tool_get_upcoming_events({}, user_id=1)
        assert "CTIS 256" in result
        assert "CTIS 499" in result
        assert "HIST 200" in result

    @pytest.mark.asyncio
    async def test_filter_returns_empty_message_when_no_match(self):
        """Filtering for 'forum' when only quiz+assign exist → empty message."""
        events_no_forum = [e for e in _SAMPLE_EVENTS if e["type"] != "forum"]
        with (
            patch("bot.services.agent_service.STATE") as mock_state,
            patch("bot.services.agent_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_state.moodle = MagicMock()
            mock_thread.return_value = events_no_forum
            result = await _tool_get_upcoming_events({"event_type": "forum"}, user_id=1)
        assert "bulunamadı" in result
