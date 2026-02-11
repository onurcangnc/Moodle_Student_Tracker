"""
Telegram Bot for Moodle AI Assistant
=====================================
Pure conversational bot — student writes, bot answers via RAG + LLM.
No session states, no mode switches.  Buttons are info shortcuts only.

Features kept:
  - Auto-sync every 10 minutes
  - Assignment check & deadline reminders
  - File upload + indexing
  - Conversation history (in-memory, last 20 messages)
  - /menu, /sync, /odevler, /help commands
"""

import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

from core import config
from core.moodle_client import MoodleClient
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.llm_engine import LLMEngine
from core.sync_engine import SyncEngine
from core.memory import StaticProfile, MemoryManager
from core.stars_client import StarsClient
from core.webmail_client import WebmailClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("telegram-bot")

# ─── Config ──────────────────────────────────────────────────────────────────

OWNER_ID = int(os.getenv("TELEGRAM_OWNER_ID", "0"))
AUTO_SYNC_INTERVAL = int(os.getenv("AUTO_SYNC_INTERVAL", "600"))  # 10 min

# ─── Global Components ──────────────────────────────────────────────────────

moodle: MoodleClient = None
processor: DocumentProcessor = None
vector_store: VectorStore = None
llm: LLMEngine = None
sync_engine: SyncEngine = None
memory: MemoryManager = None
stars_client: StarsClient = StarsClient()
webmail_client: WebmailClient = WebmailClient()

last_sync_time: str = "Henüz yapılmadı"
last_sync_new_files: int = 0
known_assignment_ids: set = set()
sync_lock = asyncio.Lock()

# ─── Conversation History ────────────────────────────────────────────────────

# user_id → {"messages": [...], "active_course": "fullname" | None}
conversation_history: dict[int, dict] = {}


def get_conversation_history(user_id: int, limit: int = 5) -> list[dict]:
    entry = conversation_history.get(user_id, {})
    return entry.get("messages", [])[-limit:]


def get_user_active_course(user_id: int) -> str | None:
    """Get the course this user is currently talking about."""
    entry = conversation_history.get(user_id, {})
    return entry.get("active_course")


def save_to_history(
    user_id: int, user_msg: str, bot_response: str, active_course: str | None = None
):
    if user_id not in conversation_history:
        conversation_history[user_id] = {"messages": [], "active_course": None}
    conv = conversation_history[user_id]
    conv["messages"].append({"role": "user", "content": user_msg})
    conv["messages"].append({"role": "assistant", "content": bot_response})
    if active_course is not None:
        conv["active_course"] = active_course
    # Max 20 messages
    if len(conv["messages"]) > 20:
        conv["messages"] = conv["messages"][-20:]


def detect_active_course(user_msg: str, user_id: int) -> str | None:
    """
    Detect which course the user is talking about.
    Priority: 1) course name in new message → 2) history's active_course → 3) None
    """
    try:
        courses = moodle.get_courses()
    except Exception:
        return get_user_active_course(user_id)

    msg_lower = user_msg.lower().replace("-", " ").replace("_", " ")

    for c in courses:
        # Match shortname code parts (e.g. "hciv", "edeb", "ctis")
        short = c.shortname.lower().replace("-", " ")
        code_parts = short.split()
        for part in code_parts:
            if len(part) >= 3 and part.isalpha() and part in msg_lower:
                return c.fullname
        # Match full shortname (e.g. "hciv 102")
        code_key = " ".join(p for p in code_parts if not p.isdigit())[:20]
        if code_key and code_key in msg_lower:
            return c.fullname

    # No match in new message → keep history course
    return get_user_active_course(user_id)


def build_smart_query(user_msg: str, history: list[dict]) -> str:
    """For short/ambiguous messages, prepend recent context."""
    if len(user_msg.split()) < 5 and history:
        recent = " ".join(m["content"] for m in history[-4:])
        return f"{recent} {user_msg}"
    return user_msg


# ─── Sync Blocking ──────────────────────────────────────────────────────────

def _sync_blocking() -> dict | None:
    """Run full sync pipeline (blocking). Called via asyncio.to_thread()."""
    global last_sync_time, last_sync_new_files

    if not moodle.connect():
        return None

    stats_before = vector_store.get_stats()
    chunks_before = stats_before.get("total_chunks", 0)

    courses = moodle.get_courses()
    profile = StaticProfile()
    profile.auto_populate_from_moodle(
        site_info=moodle.site_info, courses=[c.fullname for c in courses]
    )

    sync_engine.sync_all()

    stats_after = vector_store.get_stats()
    chunks_after = stats_after.get("total_chunks", 0)
    new_chunks = chunks_after - chunks_before

    last_sync_time = datetime.now().strftime("%H:%M:%S")
    last_sync_new_files = new_chunks

    return {
        "chunks_after": chunks_after,
        "new_chunks": new_chunks,
        "unique_courses": stats_after.get("unique_courses", 0),
        "unique_files": stats_after.get("unique_files", 0),
    }


def init_components():
    global moodle, processor, vector_store, llm, sync_engine, memory

    errors = config.validate()
    if errors:
        for e in errors:
            logger.error(f"Config error: {e}")
        sys.exit(1)

    moodle = MoodleClient()
    processor = DocumentProcessor()
    vector_store = VectorStore()
    vector_store.initialize()
    llm = LLMEngine(vector_store)
    sync_engine = SyncEngine(moodle, processor, vector_store)
    memory = MemoryManager()

    if moodle.connect():
        logger.info("Moodle connected successfully.")
        courses = moodle.get_courses()
        profile = StaticProfile()
        profile.auto_populate_from_moodle(
            site_info=moodle.site_info,
            courses=[c.fullname for c in courses],
        )
        try:
            existing = moodle.get_assignments()
            known_assignment_ids.update(a.id for a in existing)
            logger.info(f"Loaded {len(known_assignment_ids)} existing assignments.")
        except Exception as e:
            logger.debug(f"Could not load assignments: {e}")
    else:
        logger.warning("Moodle connection failed — chat works with cached data.")


# ─── Owner Check ─────────────────────────────────────────────────────────────

def is_owner(update: Update) -> bool:
    if OWNER_ID == 0:
        return True
    return update.effective_user.id == OWNER_ID


async def owner_only(update: Update) -> bool:
    if not is_owner(update):
        await update.message.reply_text("⛔ Bu bot sadece sahibi tarafından kullanılabilir.")
        return False
    return True


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_for_telegram(text: str) -> str:
    """Convert LLM Markdown output to Telegram HTML format."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = re.sub(r'```\w*\n?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = re.sub(r'^#{1,3}\s*(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(?!\s)(.+?)(?<!\s)\*', r'<i>\1</i>', text)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|[-:| ]+\|$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\|$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\|\s*', '  │  ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def send_long_message(update, text: str, reply_markup=None, parse_mode=None):
    """Send a long text, splitting at 4000 chars."""
    if parse_mode == ParseMode.HTML:
        text = format_for_telegram(text)

    send_func = update.message.reply_text
    chunks = []
    while text:
        if len(text) <= 4000:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, 4000)
        if split_at < 2000:
            split_at = 4000
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    for i, chunk in enumerate(chunks):
        kwargs = {"parse_mode": parse_mode}
        if i == len(chunks) - 1 and reply_markup:
            kwargs["reply_markup"] = reply_markup
        try:
            await send_func(chunk, **kwargs)
        except Exception:
            kwargs.pop("parse_mode", None)
            await send_func(chunk, **kwargs)


# ─── Keyboards ───────────────────────────────────────────────────────────────

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📚 Kurslar", callback_data="cmd_kurslar"),
            InlineKeyboardButton("📝 Ödevler", callback_data="cmd_odevler"),
        ],
        [
            InlineKeyboardButton("🔄 Sync", callback_data="cmd_sync"),
            InlineKeyboardButton("📊 Durum", callback_data="cmd_stats"),
        ],
    ])


def courses_keyboard():
    try:
        courses = moodle.get_courses()
        buttons = []
        for c in courses:
            short = c.shortname.split("-")[0].strip() if "-" in c.shortname else c.shortname
            buttons.append([
                InlineKeyboardButton(f"📋 {short} Özet", callback_data=f"ozet_{c.shortname}"),
                InlineKeyboardButton(f"🔒 Odaklan", callback_data=f"focus_{c.shortname}"),
            ])
        buttons.append([InlineKeyboardButton("🔓 Odağı Kaldır", callback_data="focus_clear")])
        buttons.append([InlineKeyboardButton("◀️ Ana Menü", callback_data="main_menu")])
        return InlineKeyboardMarkup(buttons)
    except Exception:
        return InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Ana Menü", callback_data="main_menu")]])


def back_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("◀️ Ana Menü", callback_data="main_menu")]
    ])


# ─── Typing Indicator ───────────────────────────────────────────────────────

class _TypingIndicator:
    """Send typing action every 5 seconds until stopped."""

    def __init__(self, bot, chat_id: int):
        self._bot = bot
        self._chat_id = chat_id
        self._task: asyncio.Task | None = None

    async def _loop(self):
        try:
            while True:
                try:
                    await self._bot.send_chat_action(chat_id=self._chat_id, action=ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    def start(self):
        self._task = asyncio.create_task(self._loop())

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None


# ─── Command Handlers ───────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    stats = vector_store.get_stats()
    await update.message.reply_text(
        f"🎓 *Moodle AI Asistan*\n\n"
        f"📦 {stats.get('total_chunks', 0)} chunk indeksli\n"
        f"📚 {stats.get('unique_courses', 0)} kurs\n"
        f"🔄 Otomatik sync: Her {AUTO_SYNC_INTERVAL // 60} dakika\n\n"
        f"Direkt mesaj yaz, sohbet ederek öğren!\n\n"
        f"💡 *İpuçları:*\n"
        f"• \"EDEB 201 bu hafta ne işledik?\" → Haftalık özet\n"
        f"• \"Felatun Bey'i anlat\" → Konu anlatımı\n"
        f"• \"Beni test et\" → Soru sorar\n"
        f"• \"Anlamadım, tekrar açıkla\" → Daha basit anlatır",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    await update.message.reply_text(
        "📖 *Nasıl Kullanılır?*\n\n"
        "Direkt mesaj yaz → Ders materyallerinden cevap alırsın.\n\n"
        "*Örnek mesajlar:*\n"
        "• \"Columbian Exchange nedir?\"\n"
        "• \"HCIV 102 öğret\"\n"
        "• \"Beni test et\"\n"
        "• \"Özet ver\"\n"
        "• \"Anlamadım, daha basit anlat\"\n\n"
        "*Komutlar:*\n"
        "/menu — Kurs listesi ve kısayollar\n"
        "/odevler — Ödev durumu\n"
        "/sync — Materyalleri güncelle\n"
        "/temizle — Sohbet geçmişini sıfırla",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    try:
        courses = moodle.get_courses()
        lines = ["📚 *Kursların*\n"]
        for c in courses:
            short = c.shortname.split("-")[0].strip() if "-" in c.shortname else c.shortname
            lines.append(f"• {short} — {c.fullname}")
        lines.append("\n💡 Ders hakkında soru sormak için direkt yaz.")
        await update.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=courses_keyboard(),
        )
    except Exception as e:
        await update.message.reply_text(f"❌ {e}", reply_markup=main_menu_keyboard())


async def cmd_courses(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    await cmd_menu(update, context)


async def cmd_focus_course(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    if not context.args:
        await update.message.reply_text(
            "Kullanım: /kurs <kurs adı>\nÖrnek: /kurs CTIS 465",
            reply_markup=courses_keyboard(),
        )
        return

    query = " ".join(context.args).lower()
    courses = moodle.get_courses()
    match = next((c for c in courses if query in c.fullname.lower() or query in c.shortname.lower()), None)

    if match:
        llm.active_course = match.fullname
        await update.message.reply_text(
            f"🔒 Odak: *{match.fullname}*\n\nArtık tüm sorular bu derse odaklanacak.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=back_keyboard(),
        )
    else:
        await update.message.reply_text(f"❌ Kurs bulunamadı: {' '.join(context.args)}")


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    if not context.args:
        await update.message.reply_text(
            "Kullanım: /ozet <kurs adı>\nÖrnek: /ozet EDEB 201-2\n\nVeya aşağıdan seç:",
            reply_markup=courses_keyboard(),
        )
        return
    await _generate_summary_msg(update, " ".join(context.args))


async def _generate_summary_msg(update: Update, course_query: str):
    courses = moodle.get_courses()
    query_lower = course_query.lower().replace("-", " ").replace("_", " ")
    match = None
    for c in courses:
        sn = c.shortname.lower().replace("-", " ")
        if query_lower in c.fullname.lower() or query_lower in sn or query_lower.replace(" ", "") in sn.replace(" ", ""):
            match = c
            break

    if not match:
        await update.message.reply_text(f"❌ Kurs bulunamadı: {course_query}")
        return

    msg = await update.message.reply_text(f"⏳ *{match.fullname}* özeti...", parse_mode=ParseMode.MARKDOWN)
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        sections = moodle.get_course_content(match.id)
        topics_text = f"DERS: {match.fullname}\n\n"
        for s in sections:
            if s.name and s.name.lower() not in ("general", "genel"):
                topics_text += f"• {s.name}"
                if s.summary:
                    topics_text += f": {s.summary[:200]}"
                topics_text += "\n"
        summary = llm.generate_course_overview(topics_text)
        await msg.delete()
        await send_long_message(
            update, f"📋 **{match.fullname}**\n\n{summary}",
            reply_markup=back_keyboard(), parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await msg.edit_text(f"❌ Özet hatası: {e}")


async def cmd_questions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    if not context.args:
        await update.message.reply_text("Kullanım: /sorular <konu>\nÖrnek: /sorular microservice patterns")
        return

    topic = " ".join(context.args)
    course = getattr(llm, "active_course", None)
    msg = await update.message.reply_text(f"⏳ Sorular: *{topic}*...", parse_mode=ParseMode.MARKDOWN)
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        questions = llm.generate_practice_questions(topic, course=course)
        await msg.delete()
        await send_long_message(
            update, f"📝 **{topic}**\n\n{questions}",
            reply_markup=back_keyboard(), parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await msg.edit_text(f"❌ Hata: {e}")


async def cmd_sync(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return

    if sync_lock.locked():
        await update.message.reply_text("⏳ Sync zaten devam ediyor...")
        return

    msg = await update.message.reply_text("🔄 Sync başladı...")

    async with sync_lock:
        try:
            result = await asyncio.to_thread(_sync_blocking)
            if result is None:
                await msg.edit_text("❌ Moodle bağlantısı başarısız!", reply_markup=back_keyboard())
                return
            text = _format_sync_result(result)
            await msg.edit_text(text, reply_markup=back_keyboard())
        except Exception as e:
            logger.error(f"Sync error: {e}")
            await msg.edit_text(f"❌ Sync hatası: {e}", reply_markup=back_keyboard())


def _format_sync_result(result: dict) -> str:
    text = (
        f"✅ Sync tamamlandı ({last_sync_time})\n\n"
        f"📦 Toplam: {result['chunks_after']} chunk\n"
        f"📚 Kurslar: {result['unique_courses']}\n"
        f"📄 Dosyalar: {result['unique_files']}\n"
    )
    if result["new_chunks"] > 0:
        text += f"\n🆕 Yeni: {result['new_chunks']} chunk eklendi!"
    else:
        text += "\n✓ Yeni materyal yok."
    return text


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return

    stats = vector_store.get_stats()
    mem_stats = memory.get_stats()
    active = getattr(llm, "active_course", None)

    text = (
        f"📊 *İstatistikler*\n\n"
        f"📦 Chunks: {stats.get('total_chunks', 0)}\n"
        f"📚 Kurslar: {stats.get('unique_courses', 0)}\n"
        f"📄 Dosyalar: {stats.get('unique_files', 0)}\n"
        f"───────────────\n"
        f"💾 Oturumlar: {mem_stats.get('sessions', 0)}\n"
        f"💬 Mesajlar: {mem_stats.get('messages', 0)}\n"
        f"🧠 Anılar: {mem_stats.get('memories', 0)}\n"
        f"📈 Konular: {mem_stats.get('topics', 0)}\n"
        f"───────────────\n"
        f"🔒 Odak: {active or 'Yok'}\n"
        f"🔄 Son sync: {last_sync_time}\n"
        f"⏱️ Auto-sync: Her {AUTO_SYNC_INTERVAL // 60} dk"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard())


async def cmd_cost(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    from core.llm_providers import MultiProviderEngine
    engine = MultiProviderEngine()
    costs = engine.estimate_costs(turns_per_day=20)
    lines = ["💰 *Tahmini Aylık Maliyet (20 tur/gün)*\n"]
    total = 0
    for task, info in costs.items():
        lines.append(f"• {task}: ${info['monthly']:.2f} ({info['model']})")
        total += info["monthly"]
    lines.append(f"\n*TOPLAM: ${total:.2f}/ay*")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard())


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    from core.llm_providers import MultiProviderEngine
    engine = MultiProviderEngine()
    models = engine.get_available_models()
    lines = ["🤖 *Model Routing*\n"]
    for m in models:
        status = "✅" if m["has_key"] else "❌"
        lines.append(f"{status} *{m['key']}*: `{m['model_id']}` ({m['provider']})")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard())


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    memories = memory.get_relevant_memories("genel")
    if not memories:
        await update.message.reply_text(
            "Henüz kayıtlı hafıza yok.\n\n💡 Konuştukça öğrenirim:\n• Zorlandığın konular\n• Tercihlerin\n• Sınav tarihleri",
            reply_markup=back_keyboard(),
        )
        return
    lines = ["🧠 *Kayıtlı Anılar*\n"]
    for m in memories[:20]:
        lines.append(f"• {m.to_text()}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard())


async def cmd_assignments(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    await update.message.chat.send_action(ChatAction.TYPING)
    text = _format_assignments()
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard())


def _format_assignments() -> str:
    import time as _time

    try:
        assignments = moodle.get_assignments()
    except Exception as e:
        return f"❌ Ödev bilgisi alınamadı: {e}"

    if not assignments:
        return "✅ Hiç ödev bulunamadı."

    now = int(_time.time())

    pending = []
    overdue = []
    submitted = []
    no_deadline = []

    for a in assignments:
        if a.submitted:
            submitted.append(a)
        elif a.due_date == 0:
            no_deadline.append(a)
        elif a.due_date < now:
            overdue.append(a)
        else:
            pending.append(a)

    lines = ["📝 *Ödevler*\n"]

    if pending:
        lines.append("🔴 *Bekleyen:*")
        for a in pending:
            due_str = datetime.fromtimestamp(a.due_date).strftime("%d/%m %H:%M")
            lines.append(f"  • *{a.name}*")
            lines.append(f"    📚 {a.course_name}")
            lines.append(f"    ⏰ {due_str} ({a.time_remaining})")
            if a.description:
                desc = a.description[:100] + "..." if len(a.description) > 100 else a.description
                lines.append(f"    📄 {desc}")
            lines.append("")

    if overdue:
        lines.append("⚠️ *Süresi Geçmiş:*")
        for a in overdue:
            due_str = datetime.fromtimestamp(a.due_date).strftime("%d/%m %H:%M")
            lines.append(f"  • *{a.name}*")
            lines.append(f"    📚 {a.course_name}")
            lines.append(f"    ❌ {due_str} (GEÇMİŞ)")
            lines.append("")

    if submitted:
        lines.append("✅ *Teslim Edilenler:*")
        for a in submitted:
            grade_str = f" — Not: {a.grade}" if a.graded else ""
            lines.append(f"  • {a.name} ({a.course_name}){grade_str}")

    if no_deadline:
        lines.append("\n📌 *Son Tarihi Olmayan:*")
        for a in no_deadline:
            status = "✅ Teslim" if a.submitted else "⏳ Bekliyor"
            lines.append(f"  • {a.name} ({a.course_name}) — {status}")

    lines.append(f"\n───────────────")
    lines.append(f"📊 Toplam: {len(assignments)} | Bekleyen: {len(pending)} | Geçmiş: {len(overdue)} | Teslim: {len(submitted)}")

    return "\n".join(lines)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return
    uid = update.effective_user.id
    llm.memory.clear()
    llm.active_course = None
    conversation_history.pop(uid, None)
    logger.info(f"Cleared history and course focus for user {uid}")
    await update.message.reply_text("🗑️ Sohbet geçmişi temizlendi.", reply_markup=back_keyboard())


# ─── STARS Commands ───────────────────────────────────────────────────────────

async def cmd_login(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual STARS login — triggers SMS 2FA."""
    if not await owner_only(update):
        return
    uid = update.effective_user.id
    stars_user = os.getenv("STARS_USERNAME")
    stars_pass = os.getenv("STARS_PASSWORD")

    if not stars_user or not stars_pass:
        await update.message.reply_text("❌ .env'de STARS_USERNAME / STARS_PASSWORD tanımlı değil.")
        return

    msg = await update.message.reply_text("🔄 STARS'a bağlanılıyor...")
    result = await asyncio.to_thread(stars_client.start_login, uid, stars_user, stars_pass)

    if result["status"] == "sms_sent":
        await msg.edit_text("📱 SMS kodu gönderildi. Kodu buraya yaz:")
    elif result["status"] == "ok":
        # No 2FA — fetch data immediately
        await msg.edit_text("🔄 STARS verileri çekiliyor...")
        cache = await asyncio.to_thread(stars_client.fetch_all_data, uid)
        if cache:
            exam_count = len(cache.exams)
            await msg.edit_text(
                f"✅ STARS verileri güncellendi!\n"
                f"📊 CGPA: {cache.user_info.get('cgpa', '?')} | {cache.user_info.get('standing', '?')}\n"
                f"📅 {exam_count} yaklaşan sınav\n"
                f"📋 {len(cache.attendance)} ders takip ediliyor",
            )
        else:
            await msg.edit_text("✅ STARS oturumu açıldı ama veri çekilemedi.")
    else:
        await msg.edit_text(f"❌ Giriş başarısız: {result.get('message', '')}")


async def cmd_stars(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show STARS cache status."""
    if not await owner_only(update):
        return
    uid = update.effective_user.id
    cache = stars_client.get_cache(uid)

    if cache and cache.fetched_at:
        from datetime import datetime as _dt
        fetched = _dt.fromtimestamp(cache.fetched_at).strftime("%d/%m %H:%M")
        await update.message.reply_text(
            f"📊 <b>STARS Verileri</b> (son güncelleme: {fetched})\n\n"
            f"🎓 CGPA: <b>{cache.user_info.get('cgpa', '?')}</b> | {cache.user_info.get('standing', '?')}\n"
            f"📅 {len(cache.exams)} sınav | 📋 {len(cache.attendance)} ders\n\n"
            f"Güncellemek için /login yaz.",
            parse_mode=ParseMode.HTML,
        )
    elif stars_client.is_awaiting_sms(uid):
        await update.message.reply_text("📱 SMS doğrulama kodu bekleniyor. Kodu buraya yaz:")
    else:
        await update.message.reply_text("❌ STARS verileri yok. /login ile giriş yap.")


# ─── STARS Display Helpers ───────────────────────────────────────────────────


async def _stars_reply_exams(update: Update, cache):
    """Format and send upcoming exams from STARS cache."""
    if not cache.exams:
        await update.message.reply_text("📅 Yaklaşan sınav bulunamadı.")
        return

    from datetime import datetime as _dt

    lines = ["📅 <b>Yaklaşan Sınavlar</b>\n"]
    for ex in cache.exams:
        days_str = ""
        if ex.get("date"):
            try:
                exam_date = _dt.strptime(ex["date"], "%d.%m.%Y")
                days_left = (exam_date - _dt.now()).days
                if days_left >= 0:
                    days_str = f"({days_left} gün kaldı)"
                else:
                    days_str = "(geçti)"
            except ValueError:
                pass

        lines.append(f"📌 <b>{ex.get('exam_name', '')} — {ex['course']}</b>")
        if ex.get("date"):
            lines.append(f"   📆 {ex['date']} {days_str}")
        if ex.get("time_block"):
            lines.append(f"   ⏰ {ex['time_block']}")
        if ex.get("time_remaining"):
            lines.append(f"   ⏳ {ex['time_remaining']}")
        lines.append("")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def _stars_reply_attendance(update: Update, cache):
    """Format and send attendance data from STARS cache."""
    if not cache.attendance:
        await update.message.reply_text("📋 Devamsızlık verisi yok.")
        return

    lines = ["📋 <b>Devamsızlık Durumu</b>\n"]
    for course in cache.attendance:
        ratio = course.get("ratio", "")
        name = course["course"]
        records = course.get("records", [])

        if not records:
            lines.append(f"➖ {name} — Henüz yoklama alınmamış")
            continue

        attended = sum(1 for r in records if r["attended"])
        total = len(records)

        try:
            pct = float(ratio.replace("%", ""))
            emoji = "✅" if pct >= 90 else "⚠️" if pct >= 70 else "❌"
        except (ValueError, AttributeError):
            emoji = "📋"

        lines.append(f"{emoji} <b>{name}</b> — {ratio}")
        lines.append(f"   {attended}/{total} derse katıldın")

        missed = [r for r in records if not r["attended"]]
        for m in missed:
            lines.append(f"   ❌ {m['date']} {m['title']}")
        lines.append("")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def _stars_reply_academic(update: Update, cache):
    """Format and send academic status from STARS cache."""
    info = cache.user_info or {}
    name = info.get("full_name", f"{info.get('name', '')} {info.get('surname', '')}".strip())
    text = (
        f"🎓 <b>Akademik Durum</b>\n\n"
        f"👤 {name}\n"
        f"📊 CGPA: <b>{info.get('cgpa', '?')}</b>\n"
        f"📈 Standing: {info.get('standing', '?')}\n"
        f"🎒 Sınıf: {info.get('class', '?')}"
    )
    if info.get("email"):
        text += f"\n📧 {info['email']}"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def _stars_reply_grades(update: Update, cache):
    """Format and send current semester grades from STARS cache."""
    grades = cache.grades or []
    if not grades:
        await update.message.reply_text("📝 Henüz not verisi yok.")
        return

    lines = ["📊 <b>Notlarım</b>\n"]
    for c in grades:
        lines.append(f"<b>{c['course']}</b>")
        if not c["assessments"]:
            lines.append("  Henüz not girilmemiş")
        else:
            for a in c["assessments"]:
                weight = f" (%{a['weight']})" if a.get("weight") else ""
                lines.append(f"  • {a['name']}: <b>{a['grade']}</b>{weight}")
        lines.append("")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def _stars_reply_course_detail(update: Update, cache, course_code: str):
    """Show all available info for a specific course: attendance + grades + exams."""
    lines = [f"📚 <b>{course_code} Detay</b>\n"]
    found = False

    # Attendance
    if cache.attendance:
        for course in cache.attendance:
            if course_code.lower() in course["course"].lower():
                found = True
                ratio = course.get("ratio", "")
                records = course.get("records", [])
                if records:
                    attended = sum(1 for r in records if r["attended"])
                    total = len(records)
                    try:
                        pct = float(ratio.replace("%", ""))
                        emoji = "✅" if pct >= 90 else "⚠️" if pct >= 70 else "❌"
                    except (ValueError, AttributeError):
                        emoji = "📋"
                    lines.append(f"{emoji} <b>Devamsızlık:</b> {ratio} ({attended}/{total})")
                    for r in records:
                        icon = "✅" if r["attended"] else "❌"
                        lines.append(f"   {icon} {r['date']} — {r['title']}")
                else:
                    lines.append("📋 Henüz yoklama alınmamış")
                lines.append("")
                break

    # Grades
    if cache.grades:
        for course in cache.grades:
            if course_code.lower() in course["course"].lower():
                found = True
                if course["assessments"]:
                    lines.append("<b>📝 Notlar:</b>")
                    for a in course["assessments"]:
                        weight = f" (%{a['weight']})" if a.get("weight") else ""
                        lines.append(f"   {a['name']}: <b>{a['grade']}</b>{weight}")
                else:
                    lines.append("📝 Henüz not girilmemiş")
                lines.append("")
                break

    # Exams
    if cache.exams:
        from datetime import datetime as _dt
        for ex in cache.exams:
            if course_code.lower() in ex["course"].lower():
                found = True
                days_str = ""
                if ex.get("date"):
                    try:
                        exam_date = _dt.strptime(ex["date"], "%d.%m.%Y")
                        days_left = (exam_date - _dt.now()).days
                        days_str = f"({days_left} gün kaldı)" if days_left >= 0 else "(geçti)"
                    except ValueError:
                        pass
                lines.append(f"📅 <b>Sınav:</b> {ex.get('exam_name', '')}")
                lines.append(f"   📆 {ex['date']} {days_str}")
                if ex.get("time_block"):
                    lines.append(f"   ⏰ {ex['time_block']}")
                lines.append("")

    if not found:
        lines.append("Bu ders için veri bulunamadı.")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


# ─── Webmail Commands ─────────────────────────────────────────────────────────

async def cmd_mail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return

    if not webmail_client.authenticated:
        await update.message.reply_text("📬 Webmail bağlantısı yok. .env'de WEBMAIL_EMAIL ve WEBMAIL_PASSWORD tanımlayın.")
        return

    msg = await update.message.reply_text("🔄 Mailler kontrol ediliyor...")
    try:
        mails = await asyncio.wait_for(
            asyncio.to_thread(webmail_client.check_all_unread),
            timeout=45,
        )
    except asyncio.TimeoutError:
        await msg.edit_text("⚠️ Mail sunucusu yanıt vermedi. Tekrar dene.")
        return
    except Exception as e:
        await msg.edit_text(f"⚠️ Mail hatası: {e}")
        return

    if not mails:
        await msg.edit_text("📬 Okunmamış AIRS/DAIS maili yok.")
        return

    lines = [f"📬 <b>{len(mails)} okunmamış AIRS/DAIS maili:</b>\n"]
    for i, m in enumerate(mails, 1):
        subject = m.get("subject", "(Konusuz)")[:60]
        sender = m.get("from", "?")
        if "<" in sender:
            sender = sender.split("<")[0].strip().strip('"')
        source = m.get("source", "")
        emoji = "👨‍🏫" if source == "AIRS" else "🏛️"
        lines.append(f"{i}. {emoji} <b>{subject}</b>\n   {sender}")

    await msg.edit_text("\n".join(lines), parse_mode=ParseMode.HTML)


# ─── Keepalive Jobs ──────────────────────────────────────────────────────────

async def moodle_keepalive_job(context: ContextTypes.DEFAULT_TYPE):
    """Keep Moodle session alive (every 2 min)."""
    try:
        await asyncio.to_thread(moodle.keepalive)
    except Exception as e:
        logger.error(f"Moodle keepalive error: {e}")




# ─── Mail Background Job ─────────────────────────────────────────────────────

async def mail_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Check for new AIRS/DAIS mails every 5 minutes."""
    if not webmail_client.authenticated:
        return

    try:
        new_mails = await asyncio.to_thread(webmail_client.check_new_airs_dais)
        if not new_mails:
            return

        for mail in new_mails:
            # Summarize body with LLM
            try:
                summary = await asyncio.to_thread(
                    llm.engine.complete,
                    task="chat",
                    system="Aşağıdaki e-postayı Türkçe olarak 2-3 cümleyle özetle. Sadece özet ver.",
                    messages=[{
                        "role": "user",
                        "content": f"Konu: {mail['subject']}\nGönderen: {mail['from']}\n\nİçerik:\n{mail.get('body_preview', '')}",
                    }],
                    max_tokens=200,
                )
            except Exception:
                summary = mail.get("body_preview", "")[:200]

            source_emoji = "👨‍🏫" if mail["source"] == "AIRS" else "🏛️"
            source_label = "Hoca" if mail["source"] == "AIRS" else "Bölüm"

            notification = (
                f"{source_emoji} <b>Yeni {mail['source']} Maili ({source_label})</b>\n\n"
                f"📧 <b>Konu:</b> {mail['subject']}\n"
                f"👤 <b>Gönderen:</b> {mail['from']}\n"
                f"📅 {mail.get('date', '')}\n\n"
                f"📝 <b>Özet:</b> {summary}"
            )

            try:
                await context.bot.send_message(
                    chat_id=OWNER_ID,
                    text=notification,
                    parse_mode=ParseMode.HTML,
                )
                logger.info(f"Mail notification: {mail['source']} - {mail['subject'][:50]}")
            except Exception as e:
                logger.error(f"Mail notification send error: {e}")

    except Exception as e:
        logger.error(f"Mail check error: {e}")


# ─── Callback Query Handler (Buttons) ───────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if OWNER_ID and query.from_user.id != OWNER_ID:
        await query.edit_message_text("⛔ Yetkisiz.")
        return

    data = query.data

    if data == "main_menu":
        await query.edit_message_text("📋 *Ana Menü*", parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_keyboard())
        return

    if data == "cmd_kurslar":
        try:
            courses = moodle.get_courses()
            lines = ["📚 *Kayıtlı Kurslar:*\n"]
            for c in courses:
                lines.append(f"• {c.fullname}")
            await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=courses_keyboard())
        except Exception as e:
            await query.edit_message_text(f"❌ {e}", reply_markup=back_keyboard())
        return

    if data == "cmd_stats":
        stats = vector_store.get_stats()
        mem_stats = memory.get_stats()
        active = getattr(llm, "active_course", None)
        text = (
            f"📊 *İstatistikler*\n\n"
            f"📦 Chunks: {stats.get('total_chunks', 0)}\n"
            f"📚 Kurslar: {stats.get('unique_courses', 0)}\n"
            f"📄 Dosyalar: {stats.get('unique_files', 0)}\n"
            f"───────────────\n"
            f"🔒 Odak: {active or 'Yok'}\n"
            f"🔄 Son sync: {last_sync_time}\n"
            f"⏱️ Auto-sync: Her {AUTO_SYNC_INTERVAL // 60} dk"
        )
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard())
        return

    if data == "cmd_sync":
        if sync_lock.locked():
            await query.edit_message_text("⏳ Sync zaten devam ediyor...", reply_markup=back_keyboard())
            return
        await query.edit_message_text("🔄 Sync başladı...")

        async def _bg_sync_callback():
            async with sync_lock:
                try:
                    result = await asyncio.to_thread(_sync_blocking)
                    if result is None:
                        await query.edit_message_text("❌ Moodle bağlantısı başarısız!", reply_markup=back_keyboard())
                        return
                    text = _format_sync_result(result)
                    await query.edit_message_text(text, reply_markup=back_keyboard())
                except Exception as e:
                    await query.edit_message_text(f"❌ {e}", reply_markup=back_keyboard())

        asyncio.create_task(_bg_sync_callback())
        return

    if data == "cmd_odevler":
        await query.edit_message_text("⏳ Ödevler yükleniyor...")
        text = _format_assignments()
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard())
        return

    # Course actions
    if data.startswith("ozet_"):
        shortname = data[5:]
        await query.edit_message_text(f"⏳ Özet: {shortname}...")
        try:
            courses = moodle.get_courses()
            match = next((c for c in courses if c.shortname == shortname), None)
            if match:
                sections = moodle.get_course_content(match.id)
                topics_text = f"DERS: {match.fullname}\n\n"
                for s in sections:
                    if s.name and s.name.lower() not in ("general", "genel"):
                        topics_text += f"• {s.name}"
                        if s.summary:
                            topics_text += f": {s.summary[:200]}"
                        topics_text += "\n"
                summary = llm.generate_course_overview(topics_text)
                formatted = format_for_telegram(f"📋 **{match.fullname}**\n\n{summary}")
                await query.edit_message_text(formatted[:4000], parse_mode=ParseMode.HTML)
                remaining = formatted[4000:]
                while remaining:
                    chunk = remaining[:4000]
                    remaining = remaining[4000:]
                    try:
                        await query.message.reply_text(chunk, parse_mode=ParseMode.HTML)
                    except Exception:
                        await query.message.reply_text(chunk)
            else:
                await query.edit_message_text(f"❌ Bulunamadı: {shortname}", reply_markup=back_keyboard())
        except Exception as e:
            await query.edit_message_text(f"❌ {e}", reply_markup=back_keyboard())
        return

    if data.startswith("focus_"):
        if data == "focus_clear":
            llm.active_course = None
            await query.edit_message_text("🔓 Odak kaldırıldı.", reply_markup=back_keyboard())
        else:
            shortname = data[6:]
            courses = moodle.get_courses()
            match = next((c for c in courses if c.shortname == shortname), None)
            if match:
                llm.active_course = match.fullname
                await query.edit_message_text(
                    f"🔒 Odak: *{match.fullname}*",
                    parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard(),
                )
            else:
                await query.edit_message_text(f"❌ Bulunamadı: {shortname}", reply_markup=back_keyboard())
        return

    # ─── Upload callbacks ──────────────────────────────────────────
    if data.startswith("upassign_"):
        course_id = int(data.split("_")[1])
        upload_path = context.user_data.get("pending_upload_path")
        upload_name = context.user_data.get("pending_upload_name", "unknown")

        if not upload_path or not Path(upload_path).exists():
            await query.edit_message_text("❌ Dosya bulunamadı veya süresi doldu.", reply_markup=back_keyboard())
            return

        courses = moodle.get_courses()
        match = next((c for c in courses if c.id == course_id), None)
        if not match:
            await query.edit_message_text("❌ Kurs bulunamadı.", reply_markup=back_keyboard())
            return

        await query.edit_message_text(f"⏳ '{upload_name}' işleniyor...\n📚 Kurs: {match.fullname}")
        try:
            count = await asyncio.to_thread(
                _index_uploaded_file_sync, Path(upload_path), match.fullname, upload_name
            )
            await query.edit_message_text(
                f"✅ '{upload_name}' başarıyla indekslendi!\n\n"
                f"📚 Kurs: {match.fullname}\n"
                f"📦 {count} chunk eklendi.",
                reply_markup=back_keyboard(),
            )
        except Exception as e:
            await query.edit_message_text(f"❌ İşleme hatası: {e}", reply_markup=back_keyboard())
        finally:
            Path(upload_path).unlink(missing_ok=True)
            context.user_data.pop("pending_upload_path", None)
            context.user_data.pop("pending_upload_name", None)
        return

    if data == "upload_cancel":
        upload_path = context.user_data.get("pending_upload_path")
        if upload_path:
            Path(upload_path).unlink(missing_ok=True)
        context.user_data.pop("pending_upload_path", None)
        context.user_data.pop("pending_upload_name", None)
        await query.edit_message_text("❌ Yükleme iptal edildi.", reply_markup=back_keyboard())
        return


# ─── File Upload Handler ─────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}


def _detect_course(filename: str) -> str | None:
    try:
        courses = moodle.get_courses()
    except Exception:
        return None

    fn_lower = filename.lower().replace("_", " ").replace("-", " ")
    for c in courses:
        short = c.shortname.lower().replace("-", " ")
        code = short.split()[0] if short else ""
        if code and code in fn_lower:
            return c.fullname
        if short and short in fn_lower:
            return c.fullname
    return None


def _index_uploaded_file_sync(file_path: Path, course_name: str, filename: str) -> int:
    chunks = processor.process_file(
        file_path=file_path,
        course_name=course_name,
        section_name="Kullanıcı Yüklemesi",
        module_name=filename,
    )
    for c in chunks:
        c.metadata["source_type"] = "user_upload"
    vector_store.add_chunks(chunks)
    return len(chunks)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return

    doc = update.message.document
    if not doc:
        return

    filename = doc.file_name or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            f"❌ Desteklenmeyen format: {ext}\n"
            f"Desteklenen: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
        return

    if doc.file_size and doc.file_size > 50 * 1024 * 1024:
        await update.message.reply_text("❌ Dosya çok büyük (max 50 MB).")
        return

    msg = await update.message.reply_text(f"📥 '{filename}' indiriliyor...")

    try:
        tg_file = await doc.get_file()
        tmp_dir = Path("/tmp/moodle_uploads")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_path = tmp_dir / filename
        await tg_file.download_to_drive(str(local_path))
    except Exception as e:
        await msg.edit_text(f"❌ İndirme hatası: {e}")
        return

    detected = _detect_course(filename)

    if detected:
        await msg.edit_text(f"⏳ '{filename}' işleniyor...\n📚 Kurs: {detected}")
        try:
            count = await asyncio.to_thread(_index_uploaded_file_sync, local_path, detected, filename)
            await msg.edit_text(
                f"✅ '{filename}' başarıyla indekslendi!\n\n"
                f"📚 Kurs: {detected}\n"
                f"📦 {count} chunk eklendi.",
                reply_markup=back_keyboard(),
            )
        except Exception as e:
            await msg.edit_text(f"❌ İşleme hatası: {e}", reply_markup=back_keyboard())
        finally:
            local_path.unlink(missing_ok=True)
    else:
        context.user_data["pending_upload_path"] = str(local_path)
        context.user_data["pending_upload_name"] = filename

        try:
            courses = moodle.get_courses()
            buttons = []
            for c in courses:
                short = c.shortname.split("-")[0].strip() if "-" in c.shortname else c.shortname
                cb_data = f"upassign_{c.id}"
                buttons.append([InlineKeyboardButton(f"📚 {short}", callback_data=cb_data)])
            buttons.append([InlineKeyboardButton("❌ İptal", callback_data="upload_cancel")])
            keyboard = InlineKeyboardMarkup(buttons)

            await msg.edit_text(
                f"📄 '{filename}'\n\n"
                f"Kurs otomatik tespit edilemedi.\nHangi kursa eklensin?",
                reply_markup=keyboard,
            )
        except Exception:
            await msg.edit_text("❌ Kurs listesi alınamadı.", reply_markup=back_keyboard())
            local_path.unlink(missing_ok=True)


# ─── Main Chat Handler (Conversational RAG) ──────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await owner_only(update):
        return

    user_msg = update.message.text
    if not user_msg:
        return

    uid = update.effective_user.id
    chat_id = update.message.chat_id

    # ── STARS SMS verification intercept ──
    if stars_client.is_awaiting_sms(uid):
        code = user_msg.strip()
        if code.isdigit() and 4 <= len(code) <= 6:
            msg = await update.message.reply_text("🔄 Doğrulama kodu kontrol ediliyor...")
            result = await asyncio.to_thread(stars_client.verify_sms, uid, code)
            if result["status"] == "ok":
                await msg.edit_text("🔄 STARS verileri çekiliyor...")
                cache = await asyncio.to_thread(stars_client.fetch_all_data, uid)
                if cache:
                    exam_count = len(cache.exams)
                    await msg.edit_text(
                        f"✅ STARS verileri güncellendi!\n"
                        f"📊 CGPA: {cache.user_info.get('cgpa', '?')} | {cache.user_info.get('standing', '?')}\n"
                        f"📅 {exam_count} yaklaşan sınav\n"
                        f"📋 {len(cache.attendance)} ders takip ediliyor",
                    )
                else:
                    await msg.edit_text("✅ STARS oturumu açıldı ama veri çekilemedi.")
            else:
                await msg.edit_text(f"❌ {result.get('message', 'Doğrulama başarısız.')}")
            return

    # ── STARS natural language detection (from cache) ──
    cache = stars_client.get_cache(uid)
    if cache and cache.fetched_at:
        msg_lower = user_msg.lower()
        stars_keywords = {
            "exam": ["sınav", "midterm", "final", "exam", "sinavlar"],
            "cgpa": ["cgpa", "not ortalam", "gpa", "akademik durum"],
            "grades": ["notlar", "not", "grade", "puan"],
            "attendance": ["devamsızlık", "devamsizlik", "yoklama", "attendance"],
        }
        # Check for course-specific query (e.g. "CTIS 465 detay")
        import re as _re
        course_match = _re.search(r'\b([A-Z]{2,5}\s*\d{3})\b', user_msg)
        if course_match and any(kw in msg_lower for kw in ["detay", "detail", "bilgi", "info"]):
            await _stars_reply_course_detail(update, cache, course_match.group(1))
            return

        for intent, keywords in stars_keywords.items():
            if any(kw in msg_lower for kw in keywords):
                if intent == "exam":
                    await _stars_reply_exams(update, cache)
                    return
                elif intent == "cgpa":
                    await _stars_reply_academic(update, cache)
                    return
                elif intent == "grades":
                    await _stars_reply_grades(update, cache)
                    return
                elif intent == "attendance":
                    await _stars_reply_attendance(update, cache)
                    return
                break

    # ── Regular RAG chat flow ──

    # Typing indicator
    typing = _TypingIndicator(update.message.get_bot(), chat_id)
    typing.start()

    try:
        # Get conversation history for context
        history = get_conversation_history(uid, limit=5)

        # Detect active course: manual focus > message detection > history
        course_filter = llm.active_course or detect_active_course(user_msg, uid)

        # Build smart query (enriches short messages with recent context)
        smart_query = build_smart_query(user_msg, history)

        # RAG: retrieve relevant chunks (filtered by course, with fallback)
        results = vector_store.query(
            query_text=smart_query,
            n_results=15,
            course_filter=course_filter,
        )

        # Fallback: if filtered results are weak, search all courses
        top_score = (1 - results[0]["distance"]) if results else 0
        if course_filter and (len(results) < 2 or top_score < 0.35):
            all_results = vector_store.query(
                query_text=smart_query, n_results=15,
            )
            all_top = (1 - all_results[0]["distance"]) if all_results else 0
            if all_top > top_score:
                results = all_results
                logger.info(f"RAG fallback: filtered score {top_score:.2f} → all-course score {all_top:.2f}")

        # Build history messages for LLM
        llm_history = history.copy()
        llm_history.append({"role": "user", "content": user_msg})

        # Call LLM with history + RAG context
        response = await asyncio.to_thread(
            llm.chat_with_history,
            messages=llm_history,
            context_chunks=results,
        )

        typing.stop()

        await send_long_message(update, response, parse_mode=ParseMode.HTML)

        # Save to history (with active course)
        save_to_history(uid, user_msg, response, active_course=course_filter)

    except Exception as e:
        typing.stop()
        logger.error(f"Chat error: {e}")
        await update.message.reply_text(f"❌ Hata: {e}")


# ─── Auto-Sync Background Job ───────────────────────────────────────────────

async def auto_sync_job(context: ContextTypes.DEFAULT_TYPE):
    if sync_lock.locked():
        logger.info("Auto-sync skipped: sync already running.")
        return

    logger.info("Auto-sync starting (background)...")

    async with sync_lock:
        try:
            result = await asyncio.to_thread(_sync_blocking)
            if result is None:
                logger.warning("Auto-sync: Moodle connection failed.")
                return

            new_chunks = result["new_chunks"]
            if new_chunks > 0:
                logger.info(f"Auto-sync: {new_chunks} new chunks!")
                if OWNER_ID:
                    try:
                        await context.bot.send_message(
                            chat_id=OWNER_ID,
                            text=(
                                f"🆕 *Yeni Materyal!*\n\n"
                                f"🔄 Auto-sync ({last_sync_time})\n"
                                f"📦 +{new_chunks} yeni chunk\n"
                                f"📚 Toplam: {result['chunks_after']}"
                            ),
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    except Exception:
                        pass
            else:
                logger.info("Auto-sync: No new materials.")

        except Exception as e:
            logger.error(f"Auto-sync error: {e}")


# ─── Auto Assignment Check ───────────────────────────────────────────────────

async def assignment_check_job(context: ContextTypes.DEFAULT_TYPE):
    global known_assignment_ids

    logger.info("Assignment check starting...")

    try:
        if not moodle.connect():
            return

        current = moodle.get_assignments()
        current_ids = {a.id for a in current}
        new_ids = current_ids - known_assignment_ids

        if new_ids:
            new_assignments = [a for a in current if a.id in new_ids]
            known_assignment_ids.update(new_ids)

            if OWNER_ID:
                lines = ["🆕 *Yeni Ödev Tespit Edildi!*\n"]
                for a in new_assignments:
                    lines.append(f"📝 *{a.name}*")
                    lines.append(f"   📚 {a.course_name}")
                    if a.due_date > 0:
                        due_str = datetime.fromtimestamp(a.due_date).strftime("%d/%m/%Y %H:%M")
                        lines.append(f"   ⏰ Son tarih: {due_str} ({a.time_remaining})")
                    if a.description:
                        desc = a.description[:150] + "..." if len(a.description) > 150 else a.description
                        lines.append(f"   📄 {desc}")
                    lines.append("")

                try:
                    await context.bot.send_message(
                        chat_id=OWNER_ID,
                        text="\n".join(lines),
                        parse_mode=ParseMode.MARKDOWN,
                    )
                except Exception as e:
                    logger.debug(f"Could not send assignment notification: {e}")

            logger.info(f"Assignment check: {len(new_ids)} new assignments!")
        else:
            known_assignment_ids = current_ids
            logger.info("Assignment check: No new assignments.")

    except Exception as e:
        logger.error(f"Assignment check error: {e}")


async def deadline_reminder_job(context: ContextTypes.DEFAULT_TYPE):
    import time as _time

    logger.info("Deadline reminder check...")

    try:
        if not moodle.connect():
            return

        now = int(_time.time())
        three_days = now + (3 * 86400)

        assignments = moodle.get_assignments()
        urgent = [
            a for a in assignments
            if not a.submitted
            and 0 < a.due_date <= three_days
            and a.due_date > now
        ]

        if urgent and OWNER_ID:
            lines = ["⚠️ *Yaklaşan Ödev Tarihleri!*\n"]
            for a in urgent:
                due_str = datetime.fromtimestamp(a.due_date).strftime("%d/%m %H:%M")
                emoji = "🔴" if a.due_date - now < 86400 else "🟡"
                lines.append(f"{emoji} *{a.name}*")
                lines.append(f"   📚 {a.course_name}")
                lines.append(f"   ⏰ {due_str} ({a.time_remaining})")
                lines.append("")

            try:
                await context.bot.send_message(
                    chat_id=OWNER_ID,
                    text="\n".join(lines),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Deadline reminder error: {e}")


# ─── Bot Setup ───────────────────────────────────────────────────────────────

async def post_init(app: Application):
    await app.bot.set_my_commands([
        BotCommand("menu", "Kurs listesi"),
        BotCommand("odevler", "Ödev durumu"),
        BotCommand("login", "STARS giriş"),
        BotCommand("stars", "STARS verileri"),
        BotCommand("mail", "Okunmamış mailler"),
        BotCommand("sync", "Materyalleri senkronla"),
        BotCommand("temizle", "Sohbet geçmişini sıfırla"),
        BotCommand("help", "Yardım"),
    ])

    # ── Background Jobs ──
    app.job_queue.run_repeating(auto_sync_job, interval=AUTO_SYNC_INTERVAL, first=AUTO_SYNC_INTERVAL, name="auto_sync")
    app.job_queue.run_repeating(assignment_check_job, interval=1800, first=60, name="assignment_check")
    app.job_queue.run_repeating(mail_check_job, interval=300, first=60, name="mail_check")
    app.job_queue.run_repeating(moodle_keepalive_job, interval=120, first=120, name="moodle_keepalive")

    from datetime import time as dtime
    app.job_queue.run_daily(deadline_reminder_job, time=dtime(hour=9, minute=0), name="deadline_reminder")

    # ── Auto-connect Webmail from .env (non-blocking) ──
    webmail_email = os.getenv("WEBMAIL_EMAIL")
    webmail_pass = os.getenv("WEBMAIL_PASSWORD")
    if webmail_email and webmail_pass:
        try:
            ok = await asyncio.wait_for(
                asyncio.to_thread(webmail_client.login, webmail_email, webmail_pass),
                timeout=15,
            )
            if ok:
                logger.info("Webmail auto-connected from .env")
            else:
                logger.error("Webmail auto-connect failed")
        except asyncio.TimeoutError:
            logger.error("Webmail auto-connect timed out (15s)")

    logger.info(f"Auto-sync: every {AUTO_SYNC_INTERVAL // 60} min")
    logger.info("Moodle/Webmail keepalive: every 2 min")
    logger.info("Mail check: every 5 min")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set in .env")
        print("   1. Telegram → @BotFather → /newbot")
        print("   2. .env'e ekle: TELEGRAM_BOT_TOKEN=xxx")
        sys.exit(1)

    if not OWNER_ID:
        print("⚠️  TELEGRAM_OWNER_ID not set — bot herkese açık!")

    print("🔧 Bileşenler yükleniyor...")
    init_components()

    app = Application.builder().token(token).post_init(post_init).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("kurslar", cmd_courses))
    app.add_handler(CommandHandler("kurs", cmd_focus_course))
    app.add_handler(CommandHandler("ozet", cmd_summary))
    app.add_handler(CommandHandler("sorular", cmd_questions))
    app.add_handler(CommandHandler("odevler", cmd_assignments))
    app.add_handler(CommandHandler("sync", cmd_sync))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("maliyet", cmd_cost))
    app.add_handler(CommandHandler("modeller", cmd_models))
    app.add_handler(CommandHandler("hafiza", cmd_memory))
    app.add_handler(CommandHandler("temizle", cmd_clear))

    # STARS + Webmail commands
    app.add_handler(CommandHandler("login", cmd_login))
    app.add_handler(CommandHandler("stars", cmd_stars))
    app.add_handler(CommandHandler("mail", cmd_mail))

    # Button callbacks
    app.add_handler(CallbackQueryHandler(handle_callback))

    # File uploads
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Conversational chat (main handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(f"🚀 Bot çalışıyor! (Owner: {OWNER_ID or 'herkes'})")
    print(f"🔄 Auto-sync: Her {AUTO_SYNC_INTERVAL // 60} dakika | Keepalive: 2 dakika")
    print(f"📧 Mail kontrolü: Her 5 dakika | STARS: Manuel /login")
    print("   Ctrl+C ile durdur")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
