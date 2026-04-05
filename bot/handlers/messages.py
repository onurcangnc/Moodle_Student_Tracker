"""Unified message handlers for agentic chat and admin uploads."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.middleware.auth import admin_only
from bot.services import document_service, user_service
from bot.services.agent_service import handle_agent_message, handle_agent_message_streaming
from bot.services.topic_cache import TOPIC_CACHE
from core import config as core_config

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_LEN = 4096
_STREAM_EDIT_INTERVAL = 1.5  # seconds between Telegram edits (rate limit safe)


def _split_message(text: str, max_len: int = _TELEGRAM_MAX_LEN) -> list[str]:
    """Split a long message into chunks, preferring paragraph → line → word breaks."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while len(text) > max_len:
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    if text:
        chunks.append(text)
    return chunks


async def _reply_message(message: Message, text: str) -> None:
    """Send response with Markdown fallback and automatic chunking for long messages."""
    chunks = _split_message(text)
    for chunk in chunks:
        try:
            await message.reply_text(chunk, parse_mode="Markdown")
        except TelegramError:
            try:
                await message.reply_text(chunk)
            except TelegramError as exc:
                logger.error("Failed to send message chunk: %s", exc)
                await message.reply_text("Yanıt gönderilirken hata oluştu.")


async def _typing_keepalive(message: Message, stop_event: asyncio.Event) -> None:
    """Re-send typing action every 4 seconds until stop_event is set."""
    try:
        while not stop_event.is_set():
            await message.reply_chat_action(ChatAction.TYPING)
            await asyncio.sleep(4)
    except Exception:
        pass


async def _stream_to_telegram(message: Message, stream) -> None:
    """Consume a streaming async iterator and progressively edit a Telegram message.

    Sends an initial placeholder, then edits it as text accumulates.
    Handles __TOOL_START__/__TOOL_DONE__ sentinels for status updates.
    """
    sent_msg: Message | None = None
    accumulated = ""
    last_edit = 0.0
    in_tool_phase = False

    async for chunk in stream:
        # Handle sentinels
        if chunk == "__TOOL_START__":
            in_tool_phase = True
            if sent_msg is None:
                sent_msg = await message.reply_text("🔄 Veri çekiliyor...")
            continue
        if chunk == "__TOOL_DONE__":
            in_tool_phase = False
            continue

        accumulated += chunk

        # First real text chunk — send or replace placeholder
        if sent_msg is None:
            try:
                sent_msg = await message.reply_text(accumulated, parse_mode="Markdown")
            except TelegramError:
                sent_msg = await message.reply_text(accumulated)
            last_edit = time.monotonic()
            continue

        # Throttle edits to respect Telegram rate limits
        now = time.monotonic()
        if now - last_edit >= _STREAM_EDIT_INTERVAL:
            try:
                await sent_msg.edit_text(accumulated, parse_mode="Markdown")
            except TelegramError:
                try:
                    await sent_msg.edit_text(accumulated)
                except TelegramError:
                    pass  # skip this edit cycle
            last_edit = now

    # Final edit with complete text
    if accumulated and sent_msg is not None:
        try:
            await sent_msg.edit_text(accumulated, parse_mode="Markdown")
        except TelegramError:
            try:
                await sent_msg.edit_text(accumulated)
            except TelegramError:
                pass
    elif accumulated and sent_msg is None:
        # Stream produced text but we never sent a message
        await _reply_message(message, accumulated)
    elif not accumulated:
        # No text produced at all
        if sent_msg:
            try:
                await sent_msg.edit_text("Yanıt üretilemedi. Lütfen tekrar deneyin.")
            except TelegramError:
                pass
        else:
            await message.reply_text("Yanıt üretilemedi. Lütfen tekrar deneyin.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle all text messages via streaming agentic LLM.

    Flow:
    1) rate limit check
    2) stream response chunks → progressive Telegram message edits
    """
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None or not message.text:
        return

    user_service.record_user_activity(user.id)
    if not user_service.check_rate_limit(user.id):
        await message.reply_text("Çok hızlı mesaj gönderdiniz. Lütfen bir dakika sonra tekrar deneyin.")
        return

    query = message.text.strip()
    if not query:
        return

    stream = handle_agent_message_streaming(user_id=user.id, user_text=query)
    await _stream_to_telegram(message, stream)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle admin-triggered document uploads after `/upload` command."""
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None or message.document is None:
        return

    user_service.record_user_activity(user.id)
    if not await admin_only(update):
        return

    if not user_service.is_upload_session_active(user.id):
        await message.reply_text("Dokuman yuklemek icin once /upload komutunu kullanin.")
        return

    raw_name = message.document.file_name or f"upload_{int(time.time())}.bin"
    filename = raw_name.replace("/", "_").replace("\\", "_").replace("..", "_")
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
