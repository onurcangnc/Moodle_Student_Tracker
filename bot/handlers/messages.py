"""Unified message handlers for chat-first learning and admin uploads."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from telegram import Message, Update
from telegram.error import TelegramError
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.middleware.auth import admin_only
from bot.services import document_service, llm_service, rag_service, user_service
from bot.services.topic_cache import TOPIC_CACHE
from core import config as core_config

logger = logging.getLogger(__name__)


async def _reply_message(message: Message, text: str) -> None:
    """Send response with Markdown fallback to plain text on parse errors."""
    try:
        await message.reply_text(text, parse_mode="Markdown")
    except TelegramError:
        await message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle all text messages via one flow.

    Flow:
    1) active course check
    2) retrieve context
    3) teaching mode (sufficient) or guidance mode (insufficient)
    4) send response
    """
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None or not message.text:
        return

    if not user_service.check_rate_limit(user.id):
        await message.reply_text("Cok hizli mesaj gonderdiniz. Lutfen bir dakika sonra tekrar deneyin.")
        return

    query = message.text.strip()
    if not query:
        return

    active_course = user_service.get_active_course(user.id)
    if active_course is None:
        await message.reply_text("Henuz bir kurs secmediniz. /courses ile kurslari gorebilirsiniz.")
        return

    history = user_service.get_conversation_history(user.id)
    retrieval = await rag_service.retrieve_context(query=query, course_id=active_course.course_id)
    if retrieval.has_sufficient_context:
        response = await llm_service.generate_teaching_response(
            query=query,
            chunks=retrieval.chunks,
            conversation_history=history,
        )
    else:
        topics = await TOPIC_CACHE.get_topics(active_course.course_id)
        response = await llm_service.generate_guidance_response(
            query=query,
            available_topics=topics,
            conversation_history=history,
        )

    user_service.add_conversation_turn(user.id, role="user", content=query)
    user_service.add_conversation_turn(user.id, role="assistant", content=response)
    await _reply_message(message, response)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle admin-triggered document uploads after `/upload` command."""
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None or message.document is None:
        return

    if not await admin_only(update):
        return

    if not user_service.is_upload_session_active(user.id):
        await message.reply_text("Dokuman yuklemek icin once /upload komutunu kullanin.")
        return

    filename = message.document.file_name or f"upload_{int(time.time())}.bin"
    upload_dir = core_config.downloads_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    local_path = upload_dir / f"{int(time.time())}_{filename}"

    try:
        telegram_file = await message.document.get_file()
        await telegram_file.download_to_drive(custom_path=str(local_path))

        active_course = user_service.get_active_course(user.id)
        detected_course = document_service.detect_course(filename)
        course_name = detected_course or (active_course.course_id if active_course else "")
        if not course_name:
            await message.reply_text("Kurs tespit edilemedi. Once /courses ile aktif kurs secin.")
            return

        added = await asyncio.to_thread(document_service.index_uploaded_file, Path(local_path), course_name, filename)
        await TOPIC_CACHE.refresh(course_name)
        await message.reply_text(f"Yukleme tamamlandi. {added} yeni parcacik indexlendi. (Kurs: {course_name})")
    except (RuntimeError, ValueError, OSError, TelegramError):
        logger.error("Upload processing failed", exc_info=True, extra={"filename": filename, "user_id": user.id})
        await message.reply_text("Dokuman islenirken hata olustu. Lutfen tekrar deneyin.")
    finally:
        user_service.clear_upload_session(user.id)


def register_message_handlers(app: Application) -> None:
    """Register document and text handlers for the simplified flow."""
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
