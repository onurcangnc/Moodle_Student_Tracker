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
  exams_sync         — 6 h     (cache refresh only, no notification)
  exam_reminder      — 1 h     (1-day-before exam alerts with room info from mail)
  deadline_reminder  — 30 min  (upcoming deadline alerts)
  session_refresh    — 24 h   (re-login webmail + STARS once per day)
  summary_generation — 60 min  (KATMAN 2 source summaries)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timedelta

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


_COURSE_CODE_RE = re.compile(r"^([A-Z]{2,}\s*\d{3}[A-Z]?)\b")


def _short_course_code(course_name: str) -> str:
    """
    Extract the short course code from a full STARS course name.

    STARS returns long names like "HCIV 201 Science and Technology in History".
    RAG metadata stores the short code "HCIV 201" (or similar prefix).
    The vector_store course_filter checks: filter.lower() in meta["course"].lower()
    So we need the SHORT code so it is a substring of the metadata value.
    """
    m = _COURSE_CODE_RE.match(course_name.strip())
    return m.group(1).strip() if m else course_name


def _extract_syllabus_attendance_limit(course_name: str) -> int | None:
    """
    Dynamically search RAG for any course's syllabus and extract the max
    absence limit. Works for every enrolled course — nothing hardcoded.

    Strategy:
    1. Derive short code from STARS full name (e.g. "HCIV 201 Sci..." → "HCIV 201")
    2. Query RAG with course_filter=short_code (substring match in metadata)
    3. Fallback: query without filter but embed course name in query text
    Returns integer hour limit, or None if not found / no syllabus uploaded.
    """
    store = STATE.vector_store
    if store is None:
        return None

    short_code = _short_course_code(course_name)

    # (query_text, course_filter)  — ordered best-first
    searches = [
        ("syllabus attendance absence limit hours miss lecture", short_code),
        ("devamsızlık saat limit hakkı miss", short_code),
        ("minimum requirements qualify final exam", short_code),
        # Fallback: no filter, course code embedded in query text
        (f"{short_code} syllabus attendance miss hours absence limit", None),
    ]

    seen_texts: list[str] = []
    for query, cf in searches:
        try:
            hits = store.query(query, n_results=5, course_filter=cf)
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
            # Sanity check: Bilkent semesters ~28-42 sessions; absence limits 4-50h
            if 4 <= val <= 50:
                logger.info(
                    "Syllabus attendance limit found for %s (code=%s): %d h",
                    course_name, short_code, val,
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


async def _keep_alive_stars(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ping STARS every 15 min to extend session beyond 58-min expiry."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return
    try:
        alive = await asyncio.to_thread(stars.keep_alive, OWNER_ID)
        if alive:
            logger.debug("STARS keep-alive OK")
        else:
            logger.warning("STARS keep-alive failed — session may have expired")
    except (ConnectionError, RuntimeError, OSError) as exc:
        logger.warning("STARS keep-alive error: %s", exc)


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


async def _sync_exams(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch exams from STARS → cache (no notification)."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(OWNER_ID):
        return

    try:
        exams = await asyncio.to_thread(stars.get_exams, OWNER_ID)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Exam sync failed: %s", exc)
        return

    if exams:
        cache_db.set_json("exams", OWNER_ID, exams)
        logger.debug("Exams cached: %d entries", len(exams))


# Regex patterns to extract room info from mail body
_ROOM_PATTERNS = [
    re.compile(r"(?:Room|Salon|Sınıf|Derslik|Hall)[:\s]+([A-Z0-9]+-?\d{1,4}[A-Z]?)", re.IGNORECASE),
    re.compile(r"\b([A-Z]-\d{2,3})\b"),  # e.g. B-201, A-05
    re.compile(r"\b(EA-\d{2,3})\b", re.IGNORECASE),  # e.g. EA-409
]


def _find_exam_room_in_mails(course_code: str) -> str | None:
    """Search cached emails for exam room info matching a course code."""
    emails = cache_db.get_emails(limit=30)
    if not emails:
        return None

    # Exam keywords to identify relevant mails
    exam_kw = re.compile(r"exam|sınav|midterm|final|quiz|qme", re.IGNORECASE)
    code_pattern = re.compile(re.escape(course_code), re.IGNORECASE)

    for mail in emails:
        subject = mail.get("subject", "")
        body = mail.get("body_preview", "") or mail.get("body_full", "")
        text = f"{subject} {body}"

        # Must mention the course code AND an exam keyword
        if not code_pattern.search(text) or not exam_kw.search(text):
            continue

        # Try to extract room
        for pat in _ROOM_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1)

    return None


def _parse_exam_date(exam: dict) -> datetime | None:
    """Parse exam date string into datetime. Handles common STARS formats."""
    date_str = exam.get("date", "")
    if not date_str:
        return None

    # Try common formats
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


async def _check_exam_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check for exams happening tomorrow and send reminder with room info."""
    exams = cache_db.get_json("exams", OWNER_ID)
    if not exams:
        return

    now = datetime.now()
    tomorrow = now.date() + timedelta(days=1)

    # Track sent reminders to avoid duplicates
    sent: list[str] = cache_db.get_json("exam_reminders_sent", OWNER_ID) or []

    notifications = []
    for exam in exams:
        exam_dt = _parse_exam_date(exam)
        if exam_dt is None or exam_dt.date() != tomorrow:
            continue

        # Build unique key for dedup
        key = f"{exam.get('course', '')}_{exam.get('exam_name', '')}_{exam.get('date', '')}"
        if key in sent:
            continue

        # Extract course code for mail matching (e.g. "CTIS 256" from "CTIS 256 Discrete Structures")
        course = exam.get("course", "")
        code_match = re.match(r"([A-Z]{2,}\s*\d{3})", course)
        course_code = code_match.group(1) if code_match else course

        # Search mails for room info
        room = _find_exam_room_in_mails(course_code)

        line = f"*{course}* — {exam.get('exam_name', 'Sınav')}"
        date_info = exam.get("date", "")
        time_info = exam.get("start_time", "") or exam.get("time_block", "")
        if date_info:
            line += f"\n📅 {date_info}"
            if time_info:
                line += f", {time_info}"
        if room:
            line += f"\n🏫 Salon: {room}"

        notifications.append(line)
        sent.append(key)

    if notifications:
        header = "📝 *Yarın sınavın var!*\n"
        msg = header + "\n\n".join(notifications) + "\n\nBaşarılar!"
        await _send(context, msg)
        cache_db.set_json("exam_reminders_sent", OWNER_ID, sent)
        logger.info("Exam reminder sent: %d exams", len(notifications))


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
    """Monthly job: delete emails older than 365 days from SQLite."""
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


# Polling watchdog: if no Telegram update received for 30 min, self-kill.
# systemd Restart=always will restart the process.
_WATCHDOG_TIMEOUT = 1800  # 30 minutes
_WATCHDOG_GRACE = 600     # 10 minutes after startup before checking


async def _polling_watchdog(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Kill the process if Telegram polling appears stuck."""
    now = time.monotonic()
    uptime = now - STATE.started_at_monotonic

    # Don't check during startup grace period
    if uptime < _WATCHDOG_GRACE:
        return

    last = STATE.last_update_received
    # If never received any update, use startup time as baseline
    if last == 0.0:
        last = STATE.started_at_monotonic

    silence = now - last
    if silence > _WATCHDOG_TIMEOUT:
        logger.critical(
            "WATCHDOG: No Telegram update for %.0f seconds — killing process for restart",
            silence,
        )
        await _send(context, "⚠️ Bot polling stuck — otomatik restart yapılıyor...")
        os._exit(1)  # noqa: SLF001 — hard kill, systemd restarts


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
        _keep_alive_stars,
        interval=timedelta(minutes=15),
        first=timedelta(minutes=5),
        name="stars_keep_alive",
    )
    jq.run_repeating(
        _sync_schedule,
        interval=timedelta(hours=6),
        first=timedelta(minutes=5),
        name="schedule_sync",
    )
    jq.run_repeating(
        _sync_exams,
        interval=timedelta(hours=6),
        first=timedelta(minutes=6),
        name="exams_sync",
    )
    jq.run_repeating(
        _check_exam_reminders,
        interval=timedelta(hours=1),
        first=timedelta(minutes=10),
        name="exam_reminder",
    )
    jq.run_repeating(
        _check_deadline_reminders,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=2),
        name="deadline_reminder",
    )
    jq.run_repeating(
        _refresh_sessions,
        interval=timedelta(hours=24),
        first=timedelta(hours=23),
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
        interval=timedelta(weeks=4),  # Monthly — 365-day retention means no rush
        first=timedelta(hours=1),
        name="cache_cleanup",
    )
    jq.run_repeating(
        _sync_syllabus_limits,
        interval=timedelta(hours=24),
        first=timedelta(minutes=5),  # Run soon after startup so limits are ready
        name="syllabus_limits_sync",
    )
    jq.run_repeating(
        _polling_watchdog,
        interval=timedelta(minutes=5),
        first=timedelta(minutes=10),
        name="polling_watchdog",
    )

    logger.info(
        "Notification jobs registered: assignments=10m, emails=5m, grades=30m, "
        "attendance=60m, schedule=6h, exams=6h, exam_reminder=1h, deadlines=30m, "
        "session=24h, summaries=60m, cache_cleanup=weekly, syllabus_limits=24h, "
        "polling_watchdog=5m"
    )
