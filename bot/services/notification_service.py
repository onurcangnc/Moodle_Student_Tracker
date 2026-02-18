"""
Background notification service using PTB job queue.
=====================================================
Every job does two things:
  1. Detect changes → notify owner via Telegram
  2. Write fresh data to SQLite cache (so tool handlers never block on live API)

Job schedule:
  assignment_check   — 10 min  (new assignments + cache refresh)
  email_check        — 5 min   (new mails + cache refresh)
  grades_sync        — 30 min  (new grades detected + cache refresh)
  attendance_sync    — 60 min  (low attendance alert + cache refresh)
  schedule_sync      — 6 h     (cache refresh only, no notification)
  deadline_reminder  — 30 min  (upcoming deadline alerts)
  session_refresh    — 60 min  (re-login webmail + STARS)
  summary_generation — 60 min  (KATMAN 2 source summaries)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta

from telegram.ext import Application, ContextTypes

from bot.config import CONFIG
from bot.state import STATE
from core import cache_db

logger = logging.getLogger(__name__)

OWNER_ID = CONFIG.owner_id

# Attendance warning threshold (%) — fallback when no syllabus limit found
_ATTENDANCE_WARN_THRESHOLD = 85.0

# Notify when this many absence slots remain (syllabus-based tracking)
_ABSENCE_WARN_REMAINING = 3   # ⚠️ warning
_ABSENCE_CRIT_REMAINING = 1   # 🚨 critical

# Regex patterns to extract max absence hours from syllabus text.
# Ordered most-specific → least-specific; first match wins.
_ABSENCE_PATTERNS = [
    # "miss more than 12 hrs of lecture"
    # "miss more than 10-class hours"   ← dash+word before "hours"
    # "miss more than 10 class hours"
    re.compile(r"miss\s+more\s+than\s+(\d+)[^.\n]{0,20}?hours?", re.IGNORECASE),
    # Same but with "hrs" abbreviation: "miss more than 12 hrs"
    re.compile(r"miss\s+more\s+than\s+(\d+)[^.\n]{0,10}?hrs?\b", re.IGNORECASE),
    # "maximum 12 hours of absence"
    re.compile(r"maximum\s+(\d+)\s*hours?", re.IGNORECASE),
    # "absence limit: 12" / "absence limit 12 hours"
    re.compile(r"absence\s+limit[:\s]+(\d+)", re.IGNORECASE),
    # "(\d+) hours of absence"
    re.compile(r"(\d+)\s*hours?\s+of\s+absence", re.IGNORECASE),
    # Turkish: "devamsızlık hakkı: 12 saat" / "12 saatlik devamsızlık hakkı"
    re.compile(r"devams[ıi]zl[ıi]k\s+hakk[ıi][:\s]+(\d+)\s*saat", re.IGNORECASE),
    re.compile(r"(\d+)\s*saatlik\s+devams[ıi]zl[ıi]k", re.IGNORECASE),
    # Loose: "12 hrs" anywhere near "lecture" in the same line
    re.compile(r"(\d+)\s*hrs?[^.\n]{0,30}lecture", re.IGNORECASE),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _serialize_assignments(assignments: list) -> list[dict]:
    """Convert assignment objects → JSON-serializable dicts."""
    result = []
    for a in assignments or []:
        result.append({
            "name":          getattr(a, "name", ""),
            "course_name":   getattr(a, "course_name", ""),
            "submitted":     getattr(a, "submitted", False),
            "due_date":      getattr(a, "due_date", None),
            "time_remaining": getattr(a, "time_remaining", ""),
        })
    return result


def _grade_keys(grades: list[dict]) -> set[tuple]:
    """Build set of (course, assessment_name) tuples that have a non-empty grade."""
    keys = set()
    for course in grades or []:
        cname = course.get("course", "")
        for a in course.get("assessments", []):
            if a.get("grade"):
                keys.add((cname, a.get("name", "")))
    return keys


def _attendance_ratios(attendance: list[dict]) -> dict[str, float]:
    """Build {course_name: ratio_float} dict from attendance list."""
    ratios = {}
    for cd in attendance or []:
        cname = cd.get("course", "")
        try:
            ratio = float(cd.get("ratio", "100").replace("%", ""))
        except (ValueError, AttributeError):
            ratio = 100.0
        ratios[cname] = ratio
    return ratios


def _extract_syllabus_attendance_limit(course_name: str) -> int | None:
    """
    Search the RAG vector store for the course's syllabus and extract the max
    absence hours/sessions. Returns an integer limit or None if not found.

    Uses course_filter to stay within the course's documents, then tries
    broader queries for syllabi that may be indexed under a different course label.
    """
    store = STATE.vector_store
    if store is None:
        return None

    # (query, use_course_filter)
    searches = [
        ("syllabus attendance absence limit hours", True),
        ("devamsızlık saat limit hakkı", True),
        ("minimum requirements qualify final exam miss lecture", True),
        # Broader fallback without course filter
        (f"{course_name} syllabus attendance miss hours", False),
    ]

    seen_texts: list[str] = []
    for query, use_filter in searches:
        try:
            hits = store.query(
                query,
                n_results=5,
                course_filter=course_name if use_filter else None,
            )
            for hit in hits or []:
                text = hit.get("text", "")
                if text and text not in seen_texts:
                    seen_texts.append(text)
        except Exception as exc:
            logger.debug("Syllabus RAG query failed for %s: %s", course_name, exc)

    combined = "\n".join(seen_texts)
    for pattern in _ABSENCE_PATTERNS:
        m = pattern.search(combined)
        if m:
            val = int(m.group(1))
            # Sanity check: Bilkent semesters ~28-42 sessions; limits are 10-25% of that
            if 4 <= val <= 50:
                logger.info(
                    "Syllabus attendance limit found for %s: %d (pattern: %s)",
                    course_name, val, pattern.pattern,
                )
                return val

    return None


def _count_absences(records: list[dict]) -> int:
    """Count sessions where the student was absent."""
    return sum(1 for r in records if not r.get("attended", True))


async def _send(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    try:
        await context.bot.send_message(
            chat_id=OWNER_ID, text=text, parse_mode="Markdown"
        )
    except Exception as exc:
        logger.error("Notification send failed: %s", exc)


# ─── Jobs ─────────────────────────────────────────────────────────────────────

async def _check_new_assignments(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check Moodle for new assignments → notify + cache."""
    moodle = STATE.moodle
    if moodle is None:
        return

    try:
        raw = moodle.get_upcoming_assignments(days=14)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Notification: assignment check failed: %s", exc)
        return

    # Cache refresh — always write even if no new assignments
    serialized = _serialize_assignments(raw)
    cache_db.set_json("assignments", OWNER_ID, serialized)
    logger.debug("Assignments cached: %d entries", len(serialized))

    # Detect truly new assignments (not yet seen this session)
    new_assignments = []
    for a in raw or []:
        aid = f"{a.course_name}_{a.name}"
        if aid not in STATE.known_assignment_ids:
            STATE.known_assignment_ids.add(aid)
            new_assignments.append(a)

    if not new_assignments:
        return

    lines = ["📋 *Yeni Ödev Bildirimi*\n"]
    for a in new_assignments:
        due = a.due_date if hasattr(a, "due_date") else "?"
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        lines.append(f"• *{a.course_name}* — {a.name}\n  Teslim: {due}")
        if remaining:
            lines.append(f"  Kalan: {remaining}")

    await _send(context, "\n".join(lines))
    logger.info("Assignment notification sent: %d new", len(new_assignments))


async def _check_new_emails(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check for new AIRS/DAIS emails → notify + cache."""
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return

    try:
        new_mails = webmail.check_new_airs_dais()
    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("Notification: email check failed: %s", exc)
        return

    # Cache refresh — always write recent 20 mails
    try:
        recent = webmail.get_recent_airs_dais(20)
        stored = cache_db.store_emails(recent)
        logger.debug("Email cache refreshed: %d mails stored", stored)
    except Exception as exc:
        logger.warning("Email cache refresh failed: %s", exc)

    if not new_mails:
        return

    lines = ["📧 *Yeni Mail Bildirimi*\n"]
    for m in new_mails[:5]:
        subject = m.get("subject", "Konusuz")
        source = m.get("source", "")
        date = m.get("date", "")
        lines.append(f"• [{source}] *{subject}*\n  {date}")

    await _send(context, "\n".join(lines))
    logger.info("Email notification sent: %d new mails", len(new_mails))


async def _sync_grades(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch grades from STARS → detect new grades → notify + cache."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    # Snapshot previous grades for change detection
    prev = cache_db.get_json("grades", OWNER_ID)
    prev_keys = _grade_keys(prev)

    try:
        grades = await asyncio.to_thread(stars.get_grades, OWNER_ID)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Grades sync failed: %s", exc)
        return

    if not grades:
        return

    # Cache refresh
    cache_db.set_json("grades", OWNER_ID, grades)
    logger.debug("Grades cached: %d courses", len(grades))

    # Notify for new grade entries
    new_keys = _grade_keys(grades)
    truly_new = new_keys - prev_keys
    if not truly_new:
        return

    lines = ["📊 *Yeni Not Girişi*\n"]
    for course in grades:
        cname = course.get("course", "")
        for a in course.get("assessments", []):
            if (cname, a.get("name", "")) in truly_new:
                lines.append(
                    f"• *{cname}* — {a.get('name', '')}: *{a.get('grade', '')}*"
                )

    await _send(context, "\n".join(lines))
    logger.info("Grade notification sent: %d new entries", len(truly_new))


async def _sync_attendance(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Fetch attendance from STARS → detect drops → notify + cache.

    Per-course logic:
    - If syllabus limit found in RAG: hour-based tracking (remaining = limit - absences)
      Notify at ≤3 remaining (⚠️) and ≤1 remaining (🚨).
    - Fallback: ratio-based (notify on first drop below 85%).
    """
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    # Snapshot previous state for change detection
    prev = cache_db.get_json("attendance", OWNER_ID)
    prev_ratios = _attendance_ratios(prev)
    prev_abs_counts: dict[str, int] = {
        cd.get("course", ""): _count_absences(cd.get("records", []))
        for cd in (prev or [])
    }

    # Load cached syllabus limits {course_name: max_hours}
    syllabus_limits: dict[str, int] = cache_db.get_json("syllabus_limits", OWNER_ID) or {}

    try:
        attendance = await asyncio.to_thread(stars.get_attendance, OWNER_ID)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Attendance sync failed: %s", exc)
        return

    if not attendance:
        return

    # Cache refresh
    cache_db.set_json("attendance", OWNER_ID, attendance)
    logger.debug("Attendance cached: %d courses", len(attendance))

    warnings: list[str] = []

    for cd in attendance:
        course = cd.get("course", "")
        records = cd.get("records", [])
        absent_now = _count_absences(records)
        absent_prev = prev_abs_counts.get(course, 0)

        limit = syllabus_limits.get(course) or None  # 0 = "not found" sentinel → None

        if limit is not None:
            # ── Syllabus-based tracking ──────────────────────────────
            remaining = limit - absent_now
            prev_remaining = limit - absent_prev

            # Notify only when we cross a threshold (not every sync)
            crossed_warn = prev_remaining > _ABSENCE_WARN_REMAINING >= remaining
            crossed_crit = prev_remaining > _ABSENCE_CRIT_REMAINING >= remaining

            if crossed_crit:
                warnings.append(
                    f"🚨 *{course}*: {absent_now}/{limit} saat devamsızlık — "
                    f"yalnızca *{remaining} saat* kaldı! KRİTİK!"
                )
            elif crossed_warn:
                warnings.append(
                    f"⚠️ *{course}*: {absent_now}/{limit} saat devamsızlık — "
                    f"*{remaining} saat* kaldı."
                )
        else:
            # ── Fallback: ratio-based (existing logic) ───────────────
            try:
                ratio = float(cd.get("ratio", "100").replace("%", ""))
            except (ValueError, AttributeError):
                ratio = 100.0
            was_ok = prev_ratios.get(course, 100.0) >= _ATTENDANCE_WARN_THRESHOLD
            now_low = ratio < _ATTENDANCE_WARN_THRESHOLD
            if was_ok and now_low:
                warnings.append(
                    f"⚠️ *{course}*: %{ratio:.1f} devam oranı — limit yaklaşıyor!"
                )

    if not warnings:
        return

    lines = ["⚠️ *Devamsızlık Uyarısı*\n"] + warnings
    await _send(context, "\n".join(lines))
    logger.info("Attendance warning sent: %d courses", len(warnings))


async def _sync_schedule(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch schedule from STARS → cache only (no notification)."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    try:
        schedule = await asyncio.to_thread(stars.get_schedule, OWNER_ID)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Schedule sync failed: %s", exc)
        return

    if schedule:
        cache_db.set_json("schedule", OWNER_ID, schedule)
        logger.debug("Schedule cached: %d entries", len(schedule))


async def _check_deadline_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remind about assignments due within 24 hours."""
    moodle = STATE.moodle
    if moodle is None:
        return

    try:
        assignments = moodle.get_upcoming_assignments(days=1)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Notification: deadline reminder check failed: %s", exc)
        return

    urgent = [a for a in assignments if not a.submitted]
    if not urgent:
        return

    lines = ["⏰ *Yaklaşan Deadline'lar (24 saat içinde)*\n"]
    for a in urgent:
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        lines.append(f"• *{a.course_name}* — {a.name}")
        if remaining:
            lines.append(f"  Kalan: {remaining}")

    await _send(context, "\n".join(lines))
    logger.info("Deadline reminder sent: %d urgent", len(urgent))


async def _cleanup_old_cache(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Weekly job: delete emails older than 90 days from SQLite."""
    deleted = cache_db.clean_old_emails()
    if deleted:
        logger.info("Weekly cache cleanup: removed %d old emails", deleted)


async def _refresh_sessions(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hourly re-login for webmail IMAP and STARS to keep sessions fresh."""
    from bot.main import refresh_external_sessions

    try:
        await asyncio.to_thread(refresh_external_sessions)
    except Exception as exc:
        logger.error("Session refresh failed: %s", exc, exc_info=True)


async def _generate_missing_summaries(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Background job: generate KATMAN 2 summaries for files that don't have one."""
    store = STATE.vector_store
    if store is None or STATE.llm is None:
        return

    try:
        from bot.services.summary_service import generate_missing_summaries

        count = await asyncio.to_thread(generate_missing_summaries)
        if count > 0:
            logger.info("Background summary generation: %d new summaries", count)
    except Exception as exc:
        logger.error("Background summary generation failed: %s", exc, exc_info=True)


async def _sync_syllabus_limits(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Daily job: for each course in attendance cache, search RAG for the syllabus
    and extract the max absence limit. Caches results as {course_name: max_hours}.

    Runs once at startup (after 5 min) then every 24h. Results persist in SQLite
    so they survive bot restarts.
    """
    attendance = cache_db.get_json("attendance", OWNER_ID) or []
    if not attendance:
        # No attendance data yet — skip silently
        return

    existing: dict[str, int] = cache_db.get_json("syllabus_limits", OWNER_ID) or {}
    updated = dict(existing)
    found = 0

    for cd in attendance:
        course = cd.get("course", "")
        if not course:
            continue
        # Re-scan even if already cached (syllabus might be uploaded mid-semester)
        limit = await asyncio.to_thread(_extract_syllabus_attendance_limit, course)
        if limit is not None:
            updated[course] = limit
            found += 1
        elif course not in updated:
            # Explicitly mark as "no limit found" so we don't re-scan every 24h
            # Use 0 as sentinel → treated as None in _sync_attendance
            updated[course] = 0

    cache_db.set_json("syllabus_limits", OWNER_ID, updated)
    logger.info(
        "Syllabus limits synced: %d courses checked, %d limits found", len(attendance), found
    )


# ─── Registration ─────────────────────────────────────────────────────────────

def register_notification_jobs(app: Application) -> None:
    """Register all periodic background jobs on the PTB job queue."""
    jq = app.job_queue
    if jq is None:
        logger.warning("Job queue not available — notifications disabled")
        return

    jq.run_repeating(
        _check_new_assignments,
        interval=timedelta(seconds=CONFIG.assignment_check_interval),
        first=timedelta(seconds=30),
        name="assignment_check",
    )
    jq.run_repeating(
        _check_new_emails,
        interval=timedelta(minutes=5),
        first=timedelta(seconds=60),
        name="email_check",
    )
    jq.run_repeating(
        _sync_grades,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=3),
        name="grades_sync",
    )
    jq.run_repeating(
        _sync_attendance,
        interval=timedelta(minutes=60),
        first=timedelta(minutes=4),
        name="attendance_sync",
    )
    jq.run_repeating(
        _sync_schedule,
        interval=timedelta(hours=6),
        first=timedelta(minutes=5),
        name="schedule_sync",
    )
    jq.run_repeating(
        _check_deadline_reminders,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=2),
        name="deadline_reminder",
    )
    jq.run_repeating(
        _refresh_sessions,
        interval=timedelta(minutes=60),
        first=timedelta(minutes=60),
        name="session_refresh",
    )
    jq.run_repeating(
        _generate_missing_summaries,
        interval=timedelta(minutes=60),
        first=timedelta(minutes=5),
        name="summary_generation",
    )
    jq.run_repeating(
        _cleanup_old_cache,
        interval=timedelta(weeks=1),
        first=timedelta(hours=1),  # First run 1h after startup (non-urgent)
        name="cache_cleanup",
    )
    jq.run_repeating(
        _sync_syllabus_limits,
        interval=timedelta(hours=24),
        first=timedelta(minutes=5),  # Run soon after startup so limits are ready
        name="syllabus_limits_sync",
    )

    logger.info(
        "Notification jobs registered: assignments=10m, emails=5m, grades=30m, "
        "attendance=60m, schedule=6h, deadlines=30m, session=60m, summaries=60m, "
        "cache_cleanup=weekly, syllabus_limits=24h"
    )
