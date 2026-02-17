"""Command handlers — only /start and /upload remain. Everything else is agentic."""

from __future__ import annotations

import logging

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from bot.middleware.auth import admin_only
from bot.services import user_service

logger = logging.getLogger(__name__)


async def post_init(app: Application) -> None:
    """Register visible command list in Telegram client UI."""
    commands = [
        BotCommand("start", "Botu başlat"),
        BotCommand("upload", "Admin materyal yükleme"),
    ]
    await app.bot.set_my_commands(commands)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message for agentic workflow."""
    await update.effective_message.reply_text(
        "Merhaba! 👋\n\n"
        "Ben Bilkent akademik asistanınım. Bana her şeyi doğal dilde sorabilirsin:\n\n"
        "📚 \"Derslerimi göster\"\n"
        "📖 \"CTIS 256 hakkında konuşalım\"\n"
        "❓ \"Etik nedir?\" — materyallerden cevap\n"
        "📅 \"Ödevlerim ne zaman?\"\n"
        "📊 \"Notlarım ne?\" — STARS\n"
        "📧 \"Son mailleri göster\" — AIRS/DAIS\n"
        "🕐 \"Bugün hangi dersim var?\"\n"
        "📝 \"Devamsızlığım ne?\"\n"
        "📄 \"Nelere çalışabilirim?\" — konu haritası\n\n"
        "Sadece yaz, ben anlayacağım! 🚀"
    )


async def cmd_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable one-shot admin upload session for the next document message."""
    if not await admin_only(update):
        return

    user = update.effective_user
    if user is None:
        return
    user_service.begin_upload_session(user.id)
    await update.effective_message.reply_text(
        "📤 Yükleme modu açıldı. Şimdi dokümanı gönderin.\n"
        "Doküman aktif kursa veya dosya adından tespit edilen kursa indexlenecek."
    )


def register_command_handlers(app: Application) -> None:
    """Register only /start and /upload commands."""
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("upload", cmd_upload))
