"""Command handlers — only /start and /upload remain. Everything else is agentic."""

from __future__ import annotations

import logging
import time

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from bot.middleware.auth import admin_only
from bot.services import user_service
from bot.state import STATE

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
    STATE.last_update_received = time.monotonic()
    await update.effective_message.reply_text(
        "Merhaba! 👋\n\n"
        "Ben akademik asistanınım — komut ezberleme, benimle konuş!\n\n"
        '📚 "Bu dersi çalışmak istiyorum" → Konu haritası + adım adım öğretim\n'
        '📖 "Privacy konusunu anlat" → Materyali OKUYUP öğretir\n'
        '📂 "Hangi materyaller var?" → Kaynak listesi + çalışma sırası\n'
        '❓ "Polimorfizm nedir?" → Materyallerden cevap\n'
        '📅 "Bugün derslerim ne?" → Ders programı\n'
        '📝 "Yaklaşan ödevlerim?" → Deadline\'lar\n'
        '📊 "Notlarım nasıl?" → Akademik durum\n'
        '📋 "Devamsızlıklarım?" → Devamsızlık + limit uyarısı\n'
        '📧 "Son maillerimi göster" → DAIS & AIRS mailleri\n'
        '🔍 "Erkan hoca mail attı mı?" → Hoca bazlı mail arama\n\n'
        'Başlamak için "kurslarımı göster" yaz!'
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
