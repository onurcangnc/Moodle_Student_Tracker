"""Command handlers for the simplified chat-first learning interface."""

from __future__ import annotations

import logging

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from bot.middleware.auth import admin_only
from bot.services import user_service
from bot.state import STATE

logger = logging.getLogger(__name__)


async def post_init(app: Application) -> None:
    """Register visible command list in Telegram client UI."""
    commands = [
        BotCommand("start", "Botu baslat"),
        BotCommand("help", "Kullanim rehberi"),
        BotCommand("courses", "Kurslari listele ve sec"),
        BotCommand("upload", "Admin materyal yukleme"),
        BotCommand("stats", "Admin bot istatistikleri"),
    ]
    await app.bot.set_my_commands(commands)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send short welcome message for chat-first workflow."""
    await update.effective_message.reply_text(
        "Merhaba. Bu bot ders materyallerini kullanarak sohbet seklinde ogretir.\n"
        "Bir kurs secmek icin /courses yazin, sonra sorunu normal mesaj olarak gonderin."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send concise usage guidance."""
    await update.effective_message.reply_text(
        "Kullanim cok basit: once /courses ile aktif kurs secin, sonra sorularinizi mesaj olarak yazin.\n"
        "Bot materyale dayali aciklama yapar; konu materyalde yoksa sizi uygun basliklara yonlendirir."
    )


async def cmd_courses(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List courses or set active course when argument is provided."""
    user = update.effective_user
    if user is None:
        return

    courses = user_service.list_courses()
    if not courses:
        await update.effective_message.reply_text("Henuz yuklu kurs bulunamadi.")
        return

    if context.args:
        query = " ".join(context.args).strip()
        match = user_service.find_course(query)
        if match is None:
            await update.effective_message.reply_text("Kurs eslesmedi. Ornek: /courses CTIS 363 veya /courses POLS")
            return

        user_service.set_active_course(user.id, match.course_id)
        await update.effective_message.reply_text(f"Aktif kurs secildi: {match.display_name}")
        return

    active = user_service.get_active_course(user.id)
    lines = ["Yuklu kurslar:"]
    for course in courses:
        prefix = "* " if active and active.course_id == course.course_id else "- "
        lines.append(f"{prefix}{course.short_name} | {course.display_name}")
    lines.append("\nKurs secmek icin: /courses <kisa_ad_veya_ad>")
    await update.effective_message.reply_text("\n".join(lines))


async def cmd_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable one-shot admin upload session for the next document message."""
    if not await admin_only(update):
        return

    user = update.effective_user
    if user is None:
        return
    user_service.begin_upload_session(user.id)
    await update.effective_message.reply_text(
        "Yukleme modu acildi. Simdi dokumani gonderin. "
        "Dokuman aktif kursa veya dosya adindan tespit edilen kursa indexlenecek."
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show lightweight operational stats for admins."""
    if not await admin_only(update):
        return

    store = STATE.vector_store
    if store is None:
        await update.effective_message.reply_text("Vector store henuz hazir degil.")
        return

    stats = store.get_stats()
    lines = [
        "Bot istatistikleri:",
        f"- Toplam chunk: {stats.get('total_chunks', 0)}",
        f"- Kurs sayisi: {stats.get('unique_courses', 0)}",
        f"- Dosya sayisi: {stats.get('unique_files', 0)}",
        f"- Aktif kurs secimi olan kullanici: {len(STATE.active_courses)}",
        f"- Bekleyen upload oturumu: {len(STATE.pending_upload_users)}",
    ]
    await update.effective_message.reply_text("\n".join(lines))


def register_command_handlers(app: Application) -> None:
    """Register only the allowed minimal command set."""
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("courses", cmd_courses))
    app.add_handler(CommandHandler("upload", cmd_upload))
    app.add_handler(CommandHandler("stats", cmd_stats))
