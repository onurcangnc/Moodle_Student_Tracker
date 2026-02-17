"""
Background notification service using PTB job queue.
=====================================================
Periodic checks for:
- New Moodle assignments (every 10 min)
- New AIRS/DAIS emails (every 5 min, if webmail authenticated)
- Upcoming deadline reminders (every 30 min)
- Missing source summaries (every 60 min, KATMAN 2)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from telegram.ext import Application, ContextTypes

from bot.config import CONFIG
from bot.state import STATE

logger = logging.getLogger(__name__)

# Owner receives all notifications
OWNER_ID = CONFIG.owner_id


async def _check_new_assignments(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check Moodle for new assignments and notify owner."""
    moodle = STATE.moodle
    if moodle is None:
        return

    try:
        assignments = moodle.get_upcoming_assignments(days=14)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Notification: assignment check failed: %s", exc)
        return

    if not assignments:
        return

    # Track known assignments to avoid duplicate notifications
    new_assignments = []
    for a in assignments:
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

    try:
        await context.bot.send_message(
            chat_id=OWNER_ID,
            text="\n".join(lines),
            parse_mode="Markdown",
        )
        logger.info("Assignment notification sent: %d new", len(new_assignments))
    except Exception as exc:
        logger.error("Failed to send assignment notification: %s", exc)


async def _check_new_emails(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check for new AIRS/DAIS emails and notify owner."""
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return

    try:
        new_mails = webmail.check_new_airs_dais()
    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("Notification: email check failed: %s", exc)
        return

    if not new_mails:
        return

    lines = ["📧 *Yeni Mail Bildirimi*\n"]
    for m in new_mails[:5]:  # Limit to 5 per notification
        subject = m.get("subject", "Konusuz")
        source = m.get("source", "")
        date = m.get("date", "")
        lines.append(f"• [{source}] *{subject}*\n  {date}")

    try:
        await context.bot.send_message(
            chat_id=OWNER_ID,
            text="\n".join(lines),
            parse_mode="Markdown",
        )
        logger.info("Email notification sent: %d new mails", len(new_mails))
    except Exception as exc:
        logger.error("Failed to send email notification: %s", exc)


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

    # Filter only unsubmitted
    urgent = [a for a in assignments if not a.submitted]
    if not urgent:
        return

    lines = ["⚠️ *Yaklaşan Deadline'lar (24 saat içinde)*\n"]
    for a in urgent:
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        lines.append(f"• *{a.course_name}* — {a.name}")
        if remaining:
            lines.append(f"  ⏰ Kalan: {remaining}")

    try:
        await context.bot.send_message(
            chat_id=OWNER_ID,
            text="\n".join(lines),
            parse_mode="Markdown",
        )
        logger.info("Deadline reminder sent: %d urgent", len(urgent))
    except Exception as exc:
        logger.error("Failed to send deadline reminder: %s", exc)


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


def register_notification_jobs(app: Application) -> None:
    """Register periodic background jobs on the PTB job queue."""
    jq = app.job_queue
    if jq is None:
        logger.warning("Job queue not available — notifications disabled")
        return

    # Assignment check — every 10 minutes
    jq.run_repeating(
        _check_new_assignments,
        interval=timedelta(seconds=CONFIG.assignment_check_interval),
        first=timedelta(seconds=30),  # Wait 30s after startup
        name="assignment_check",
    )

    # Email check — every 5 minutes
    jq.run_repeating(
        _check_new_emails,
        interval=timedelta(minutes=5),
        first=timedelta(seconds=60),
        name="email_check",
    )

    # Deadline reminders — every 30 minutes
    jq.run_repeating(
        _check_deadline_reminders,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=2),
        name="deadline_reminder",
    )

    # Session refresh — every 60 min (re-login webmail + STARS)
    jq.run_repeating(
        _refresh_sessions,
        interval=timedelta(minutes=60),
        first=timedelta(minutes=60),  # First run after 1 hour (startup already logged in)
        name="session_refresh",
    )

    # KATMAN 2: Generate missing source summaries — every 60 min
    jq.run_repeating(
        _generate_missing_summaries,
        interval=timedelta(minutes=60),
        first=timedelta(minutes=5),  # Wait 5 min after startup
        name="summary_generation",
    )

    logger.info(
        "Notification jobs registered: assignment_check=%ds, email_check=300s, "
        "deadline_reminder=1800s, session_refresh=3600s, summary_generation=3600s",
        CONFIG.assignment_check_interval,
    )
