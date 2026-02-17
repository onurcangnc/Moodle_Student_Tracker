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
        BotCommand("start", "Botu başlat"),
        BotCommand("help", "Kullanım rehberi"),
        BotCommand("courses", "Kursları listele ve seç"),
        BotCommand("upload", "Admin materyal yükleme"),
        BotCommand("stats", "Admin bot istatistikleri"),
    ]
    await app.bot.set_my_commands(commands)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message for chat-first workflow."""
    await update.effective_message.reply_text(
        "Merhaba! 👋\n\n"
        "Ben ders materyallerinden öğrenmenizi kolaylaştıran bir asistanım.\n\n"
        "📚 /courses — Kurslarınızı listeleyin ve aktif kurs seçin\n"
        "❓ Soru sorun — Aktif kurstaki materyallerden cevap alırsınız\n\n"
        "📤 /upload — Doküman yükle (admin)\n"
        "📊 /stats — Bot istatistikleri (admin)\n"
        "ℹ️ /help — Yardım\n\n"
        "Başlamak için /courses ile bir kurs seçin, sonra sorunuzu yazın!"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send usage guidance."""
    await update.effective_message.reply_text(
        "📖 Nasıl Kullanılır?\n\n"
        "1️⃣ /courses ile kurslarınızı görün\n"
        "2️⃣ /courses <kurs_adı> ile aktif kurs seçin\n"
        "3️⃣ Sorunuzu mesaj olarak yazın\n\n"
        "Bot, seçtiğiniz kurstaki materyallerden cevap üretir.\n"
        "Yeterli materyal bulamazsa sizi doğru konulara yönlendirir.\n\n"
        "Komutlar:\n"
        "• /courses — Kurs listesi ve seçimi\n"
        "• /upload — Doküman yükle (admin)\n"
        "• /stats — İstatistikler (admin)"
    )


async def cmd_courses(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List courses or set active course when argument is provided."""
    user = update.effective_user
    if user is None:
        return

    courses = user_service.list_courses()
    if not courses:
        await update.effective_message.reply_text("Henüz yüklü kurs bulunamadı.")
        return

    if context.args:
        query = " ".join(context.args).strip()
        match = user_service.find_course(query)
        if match is None:
            await update.effective_message.reply_text(
                "Kurs eşleşmedi. Örnek: /courses CTIS 363 veya /courses POLS"
            )
            return

        user_service.set_active_course(user.id, match.course_id)
        await update.effective_message.reply_text(f"✅ Aktif kurs seçildi: {match.display_name}")
        return

    active = user_service.get_active_course(user.id)
    lines = ["📚 Yüklü kurslar:\n"]
    for course in courses:
        prefix = "▸ " if active and active.course_id == course.course_id else "  "
        lines.append(f"{prefix}{course.short_name} — {course.display_name}")
    lines.append("\nKurs seçmek için: /courses <kurs_adı>")
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
        "📤 Yükleme modu açıldı. Şimdi dokümanı gönderin.\n"
        "Doküman aktif kursa veya dosya adından tespit edilen kursa indexlenecek."
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show lightweight operational stats for admins."""
    if not await admin_only(update):
        return

    store = STATE.vector_store
    if store is None:
        await update.effective_message.reply_text("Vector store henüz hazır değil.")
        return

    stats = store.get_stats()
    lines = [
        "📊 Bot İstatistikleri:\n",
        f"Toplam chunk: {stats.get('total_chunks', 0)}",
        f"Kurs sayısı: {stats.get('unique_courses', 0)}",
        f"Dosya sayısı: {stats.get('unique_files', 0)}",
        f"Aktif kurs seçimi olan kullanıcı: {len(STATE.active_courses)}",
        f"Bekleyen upload oturumu: {len(STATE.pending_upload_users)}",
    ]
    await update.effective_message.reply_text("\n".join(lines))


def register_command_handlers(app: Application) -> None:
    """Register only the allowed minimal command set."""
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("courses", cmd_courses))
    app.add_handler(CommandHandler("upload", cmd_upload))
    app.add_handler(CommandHandler("stats", cmd_stats))
