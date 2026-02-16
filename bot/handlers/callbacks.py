"""Callback query handlers for inline keyboard actions."""

from __future__ import annotations

from telegram.ext import Application, CallbackQueryHandler

from bot import legacy

handle_callback = legacy.handle_callback


def register_callback_handlers(app: Application) -> None:
    """Register callback query handler."""
    app.add_handler(CallbackQueryHandler(handle_callback))
