"""Message and document handlers for conversational interactions."""

from __future__ import annotations

from telegram.ext import Application, MessageHandler, filters

from bot import legacy

handle_document = legacy.handle_document
handle_message = legacy.handle_message


def register_message_handlers(app: Application) -> None:
    """Register document and text message handlers."""
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
