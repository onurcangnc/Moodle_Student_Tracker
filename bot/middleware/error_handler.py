"""Centralized Telegram error handling middleware."""

from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.error import Conflict, TelegramError
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unhandled exceptions and send a generic error message to user."""
    tg_update: Update | None = update if isinstance(update, Update) else None
    err = context.error if hasattr(context, "error") else None

    logger.exception(
        "Unhandled exception in Telegram update pipeline",
        exc_info=err or True,
        extra={
            "update_id": tg_update.update_id if tg_update else None,
            "user_id": tg_update.effective_user.id if tg_update and tg_update.effective_user else None,
        },
    )

    # Conflict error = another bot instance is polling → restart immediately.
    # Without this, polling stops but process stays alive (zombie state).
    if isinstance(err, Conflict):
        logger.critical("Conflict error detected — another instance is running. Restarting...")
        os._exit(1)  # noqa: SLF001 — hard kill, systemd restarts

    if tg_update and tg_update.effective_message:
        try:
            await tg_update.effective_message.reply_text("Bir hata oluştu. Lütfen tekrar deneyin.")
        except TelegramError as exc:
            logger.exception("Failed to send generic error response: %s", exc)
