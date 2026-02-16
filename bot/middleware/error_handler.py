"""Centralized Telegram error handling middleware."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unhandled exceptions and send a generic error message to user."""
    tg_update: Update | None = update if isinstance(update, Update) else None
    logger.exception(
        "Unhandled exception in Telegram update pipeline",
        exc_info=context.error if hasattr(context, "error") else True,
        extra={
            "update_id": tg_update.update_id if tg_update else None,
            "user_id": tg_update.effective_user.id if tg_update and tg_update.effective_user else None,
        },
    )

    if tg_update and tg_update.effective_message:
        try:
            await tg_update.effective_message.reply_text("Bir hata oluştu. Lütfen tekrar deneyin.")
        except TelegramError as exc:
            logger.exception("Failed to send generic error response: %s", exc)
