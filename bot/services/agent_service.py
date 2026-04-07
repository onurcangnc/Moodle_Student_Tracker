"""
Agentic LLM service with OpenAI function calling — v4.
========================================================
Refactored for SOLID compliance:
- Tools extracted to bot/services/tools/ (Strategy pattern)
- LLM routing extracted to bot/services/llm_router.py
- Uses ToolRegistry for tool management
- Uses ServiceContainer for dependency injection

The bot's brain: 3-Layer Knowledge Architecture + 18 tools.

KATMAN 1 — Index: metadata aggregation (get_source_map, instant, free)
KATMAN 2 — Summary: pre-generated teaching overviews (read_source, stored JSON)
KATMAN 3 — Deep read: chunk-based content (rag_search, study_topic, read_source)

Tool loop: user → LLM (with tools) → tool exec → LLM (with results) → reply
Max iterations: 5, parallel_tool_calls=True
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from telegram import Message
from telegram.error import TelegramError

from bot.services import user_service
from bot.state import STATE
from core import cache_db

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

# ─── Instant Responses (LLM bypass for simple queries) ───────────────────────

_INSTANT_RESPONSES: dict[str, str] = {
    # Greetings
    "merhaba": "Merhaba! Nasıl yardımcı olabilirim?",
    "selam": "Selam! Ne yapmak istersin?",
    "slm": "Selam! Ne yapmak istersin?",
    "mrb": "Merhaba! Nasıl yardımcı olabilirim?",
    "hi": "Hi! How can I help you?",
    "hello": "Hello! What can I do for you?",
    "hey": "Hey! How can I help?",
    # Gratitude
    "teşekkürler": "Rica ederim! Başka bir şey lazım olursa yaz.",
    "teşekkür ederim": "Rica ederim! Başka bir şey lazım olursa yaz.",
    "sağol": "Ne demek! Başka sorun varsa sor.",
    "sağ ol": "Ne demek! Başka sorun varsa sor.",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Let me know if you need anything else.",
    # Acknowledgments
    "tamam": "Tamam! Başka bir şey var mı?",
    "ok": "Okay! Anything else?",
    "anladım": "Güzel! Başka sorun olursa yaz.",
    "peki": "Peki! Başka bir konuda yardım edebilir miyim?",
    # Farewells
    "görüşürüz": "Görüşürüz! İyi çalışmalar!",
    "bye": "Bye! Good luck!",
    "bb": "Görüşürüz!",
}


def _check_instant_response(text: str) -> str | None:
    """Check if message matches an instant response pattern."""
    normalized = text.strip().lower()
    if normalized in _INSTANT_RESPONSES:
        return _INSTANT_RESPONSES[normalized]
    return None


# ─── Smart Model Selection ───────────────────────────────────────────────────

_COMPLEXITY_KEYWORDS = {
    "detaylı", "ayrıntılı", "derinlemesine", "kapsamlı", "analiz",
    "karşılaştır", "compare", "explain in detail", "thoroughly",
    "akademik durumum", "genel durum", "özet", "summary",
    "tüm dersler", "all courses", "everything",
    "anlat", "öğret", "açıkla", "explain", "teach",
    "nasıl hazırlanayım", "strateji", "plan", "tavsiye",
}


def _is_complex_query(user_text: str, tool_count: int = 0) -> bool:
    """Detect if query needs higher-quality model."""
    text_lower = user_text.lower()
    if tool_count >= 2:
        return True
    if len(user_text) > 150:
        return True
    if any(kw in text_lower for kw in _COMPLEXITY_KEYWORDS):
        return True
    return False


# ─── Smart Error Messages ────────────────────────────────────────────────────

def _extract_topic(text: str) -> str | None:
    """Extract main topic from user query for profile tracking."""
    text_lower = text.lower()
    if len(text) < 10:
        return None

    topic_patterns = [
        ("not", "notlar"),
        ("devamsızlık", "devamsızlık"),
        ("ders program", "program"),
        ("ödev", "ödevler"),
        ("sınav", "sınavlar"),
        ("mail", "mailler"),
        ("çalış", "ders çalışma"),
        ("anlat", "konu açıklama"),
        ("öğret", "öğretim"),
        ("privacy", "privacy"),
        ("ethics", "ethics"),
        ("güvenlik", "güvenlik"),
    ]

    for pattern, topic in topic_patterns:
        if pattern in text_lower:
            return topic
    return None


def _smart_error(error_type: str, context: str = "", user_id: int | None = None) -> str:
    """Generate helpful error messages with recovery suggestions."""
    if error_type == "llm_failed":
        return (
            "Yanıt oluştururken bir sorun oluştu.\n\n"
            "Şunları deneyebilirsin:\n"
            "- Soruyu daha kısa/basit yaz\n"
            "- Biraz bekleyip tekrar dene\n"
            f"{context}"
        )
    if error_type == "llm_null":
        return (
            "Yanıt üretilemedi. Sistem meşgul olabilir.\n"
            "Birkaç saniye bekleyip tekrar dene."
        )
    if error_type == "stars_session":
        cache_hint = ""
        if user_id:
            cached = cache_db.get_json("grades", user_id)
            if cached:
                cache_hint = "\nSon bilinen veriler mevcut (cache). Temel bilgiler için tekrar sorabilirsin."
        return f"STARS bağlantısı sona ermiş.{cache_hint}\n\nYeniden bağlanmak için /start yaz."
    if error_type == "no_data":
        return f"{context}\n\nFarklı bir sorgu denemek ister misin?"
    return "Bir sorun oluştu. Lütfen tekrar dene."


# ─── Day Names ───────────────────────────────────────────────────────────────

_DAY_NAMES_TR = {
    0: "Pazartesi",
    1: "Salı",
    2: "Çarşamba",
    3: "Perşembe",
    4: "Cuma",
    5: "Cumartesi",
    6: "Pazar",
}


# ─── System Prompt Builder ────────────────────────────────────────────────────

def _build_system_prompt(user_id: int) -> str:
    """Build dynamic system prompt with 3-layer teaching methodology."""
    active_course = user_service.get_active_course(user_id)
    course_section = (
        f"Kullanıcının aktif kursu: *{active_course.display_name}*"
        if active_course
        else "Kullanıcı henüz kurs seçmemiş. Ders içeriği sorulursa 'Kurslarımı göster' demesini öner."
    )

    stars_ok = STATE.stars is not None and STATE.stars.is_authenticated(user_id)
    webmail_ok = STATE.webmail is not None and STATE.webmail.authenticated

    services = []
    if stars_ok:
        services.append("STARS: Bağlı")
    else:
        services.append("STARS: Bağlı değil — get_schedule, get_grades, get_attendance çalışmaz")
    if webmail_ok:
        services.append("Webmail: Bağlı")
    else:
        services.append("Webmail: Bağlı değil — get_emails, get_email_detail çalışmaz")

    now = datetime.now()
    today_tr = _DAY_NAMES_TR.get(now.weekday(), "")
    date_str = now.strftime("%d/%m/%Y %H:%M")

    student_ctx = ""
    if STATE.llm:
        student_ctx = STATE.llm._build_student_context()

    profile_ctx = cache_db.get_profile_context(user_id)

    return f"""Sen Bilkent Üniversitesi öğrencileri için bir akademik asistan botsun.

## DİL KURALI (KRİTİK — HER MESAJDA UYGULA)
Kullanıcının SON mesajının dili yanıt dilini belirler. Konuşma geçmişi farklı dilde olsa bile SON mesaja bak:
- Son mesaj Türkçe → Türkçe yanıt
- Son mesaj İngilizce → İngilizce yanıt
- Karışık → mesajın ağırlıklı diline göre

{course_section}
Aktif servisler: {chr(10).join(services)}
Tarih: {date_str} ({today_tr})
{student_ctx}
{profile_ctx}

## KONUŞMA BAĞLAMI (KRİTİK)
Her mesajı KONUŞMADAKİ ÖNCEKI MESAJLARLA BİRLİKTE değerlendir.
- "neysi", "neyse", "hani", "işte" gibi bağlaç/dolgu kelimeleri arama terimi DEĞİLDİR
- "Hoca farklı bir anlamından bahsetti neysi" → önceki konuşmadaki konuyu devam ettir
- "Detaylandır", "devam et", "daha fazla" → önceki yanıtı derinleştir, yeni arama YAPMA
- Belirsiz referanslarda ("bunu", "şunu", "o konuyu") konuşma geçmişinden bağlamı çıkar

## KİŞİLİĞİN
- Samimi, yardımsever, motive edici
- Emoji kullan ama abartma
- Kısa ve öz ol — Telegram'da max 3-4 paragraf
- Slash komut sorulursa "Benimle doğal dilde konuşabilirsin!" de

## KİMLİK KURALI
Sen bir Bilkent akademik asistanısın. GPT, Claude, Gemini, OpenAI gibi model isimlerini ASLA söyleme.

## PLANLAMA VE TOOL SEÇİMİ
Her mesajda ÖNCE düşün:
1. Ne soruyor? (veri sorgusu / ders çalışma / sohbet / bilgi)
2. Hangi tool(lar) gerekli? Paralel mi sıralı mı?
3. Aktif kurs bağlamında mı, genel mi?

Karmaşık sorularda tool'ları paralel çağır:
- "Sınavlara nasıl hazırlanayım?" → get_assignments + get_schedule + get_source_map
- "Bugün ne var?" → get_schedule(today) + get_assignments(upcoming)
- "Akademik durumum?" → get_grades + get_attendance + get_assignments

Basit sorularda TEK tool yeterli — fazla tool çağırma.
Sohbet/selamlama → HİÇ tool çağırma, doğrudan cevap ver.

## DERS ÇALIŞMA — ÖĞRETİM YAKLAŞIMI

Sen bir ÖĞRETMENSİN, arama motoru değilsin. Materyali OKUYUP ÖĞRETİYORSUN.

Çalışma akışı:
1. "Çalışmak istiyorum" → get_source_map ile materyal haritası çıkar
2. Önerilen çalışma sırası sun (temelden ileriye)
3. Öğrenci kaynak seçince → read_source ile dosyayı OKU
4. Pedagojik öğretim yap:
   - Konuyu basitçe açıkla
   - Gerçek hayat örnekleri ver
   - Düşündürücü sorular sor
   - İlişkili kavramları bağla

## AGENTIC RAG — İTERATİF ARAMA
Tek bir RAG sorgusuyla yetinme. Karmaşık sorularda İTERATİF çalış:
- İlk sonuç yetersizse → farklı anahtar kelimeyle tekrar ara
- Karşılaştırma sorusu ("A vs B") → her kavramı AYRI ara, sonra sentezle

## FORMAT KURALLARI
1. Telegram Markdown: *bold*, _italic_, `code`
2. Tool sonuçlarını doğal dille sun, JSON/teknik format GÖSTERME
3. Tool sonucu boş gelirse nazikçe bildir
4. RAG sonuçlarını kullanırken kaynak etiketi ekle

## TEKNİK TERİM YASAĞI
ASLA kullanma: chunk, RAG, retrieval, embedding, vector, tool, function call, token, pipeline, LLM, model, API

## SON KURAL — DİL
Kullanıcının SON mesajı İngilizce ise yanıtın %100 İngilizce olmalı.
Kullanıcının SON mesajı Türkçe ise yanıtın %100 Türkçe olmalı."""


# ─── Language Detection ───────────────────────────────────────────────────────

_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
_EN_WORDS = {
    "show", "me", "my", "what", "how", "the", "is", "are", "do", "does",
    "can", "get", "list", "which", "from", "about", "please", "tell",
    "help", "hello", "hi", "hey", "give", "want", "need", "today",
    "grades", "schedule", "emails", "courses", "assignments", "attendance",
}


def _detect_language(text: str) -> str:
    """Detect if user message is English or Turkish. Returns 'en' or 'tr'."""
    if any(c in _TR_CHARS for c in text):
        return "tr"
    words = set(text.lower().split())
    en_matches = len(words & _EN_WORDS)
    if en_matches >= 2 or (en_matches >= 1 and len(words) <= 4):
        return "en"
    return "tr"


# ─── Progressive Send ────────────────────────────────────────────────────────

async def _send_progressive(message: Message, text: str) -> None:
    """Send pre-generated text progressively for perceived speed."""
    if not text:
        return

    if len(text) < 100:
        await message.reply_text(text, parse_mode="Markdown")
        return

    chunk_size = max(50, len(text) // 5)
    sent_msg = None

    try:
        for i in range(0, len(text), chunk_size):
            partial = text[: i + chunk_size]
            if sent_msg is None:
                sent_msg = await message.reply_text(partial, parse_mode=None)
            else:
                try:
                    await sent_msg.edit_text(partial, parse_mode=None)
                except TelegramError:
                    pass
            await asyncio.sleep(0.15)

        if sent_msg:
            try:
                await sent_msg.edit_text(text, parse_mode="Markdown")
            except TelegramError:
                try:
                    await sent_msg.edit_text(text, parse_mode=None)
                except TelegramError:
                    pass
    except Exception as exc:
        logger.warning("Progressive send failed: %s", exc)
        if sent_msg is None:
            await message.reply_text(text, parse_mode="Markdown")


# ─── Tool Execution ──────────────────────────────────────────────────────────

async def _execute_tool_call(tool_call: Any, user_id: int) -> dict[str, str]:
    """Execute a single tool call via ToolRegistry."""
    fn_name = tool_call.function.name
    try:
        fn_args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        fn_args = {}

    registry = STATE.tool_registry
    if registry is None:
        result = "Tool registry not initialized"
    else:
        result = await registry.execute(fn_name, fn_args, user_id, STATE)

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }


# ─── Main Entry Point ────────────────────────────────────────────────────────

async def handle_agent_message(
    user_id: int,
    user_text: str,
    message: Message | None = None,
) -> str:
    """
    Main agentic handler: takes user message, runs tool loop, returns response.

    Flow:
    1. Build system prompt with 3-layer teaching methodology
    2. Get conversation history
    3. Call LLM with tools + parallel_tool_calls=True
    4. If tool calls → execute in parallel → feed results → repeat (max 5)
    5. Stream final text response to Telegram (if message provided)

    Returns empty string if response was already streamed to the user.
    """
    # Instant response bypass (no LLM call)
    instant = _check_instant_response(user_text)
    if instant:
        user_service.add_conversation_turn(user_id, "user", user_text)
        user_service.add_conversation_turn(user_id, "assistant", instant)
        logger.info("Instant response for: %s", user_text[:30])
        return instant

    if STATE.llm is None:
        return "Sistem henüz hazır değil. Lütfen birazdan tekrar deneyin."

    router = STATE.llm_router
    registry = STATE.tool_registry
    if router is None or registry is None:
        return "Sistem bileşenleri henüz hazır değil."

    t_start = time.time()
    system_prompt = _build_system_prompt(user_id)

    # Detect language and inject directive
    lang = _detect_language(user_text)
    if lang == "en":
        system_prompt += "\n\n[LANGUAGE OVERRIDE] The user's current message is in ENGLISH. You MUST respond entirely in English."

    available_tools = registry.get_definitions()

    history = user_service.get_conversation_history(user_id)
    messages: list[dict[str, Any]] = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    is_complex = _is_complex_query(user_text)
    tools_used: list[str] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        # Refresh typing indicator
        if message:
            try:
                await message.chat.send_action("typing")
            except TelegramError:
                pass

        try:
            t_llm = time.time()
            max_tokens = 1024 if available_tools else 4096
            response_msg = await router.complete(
                messages, system_prompt, available_tools, max_tokens
            )
            logger.info("LLM call (iter %d): %.2fs", iteration + 1, time.time() - t_llm)
        except Exception as exc:
            logger.error("LLM call failed (iteration %d): %s", iteration, exc, exc_info=True)
            return _smart_error("llm_failed", f"Hata: {type(exc).__name__}")

        if response_msg is None:
            return _smart_error("llm_null")

        tool_calls = getattr(response_msg, "tool_calls", None)
        if not tool_calls:
            # Final text response
            final_text = router.sanitize_output(response_msg.content or "")

            if message and final_text:
                await _send_progressive(message, final_text)
                user_service.add_conversation_turn(user_id, "user", user_text)
                user_service.add_conversation_turn(user_id, "assistant", final_text)
                active = user_service.get_active_course(user_id)
                cache_db.track_query(user_id, course=active.course_id if active else None, topic=_extract_topic(user_text))
                logger.info("Total response time: %.2fs (progressive)", time.time() - t_start)
                return ""

            user_service.add_conversation_turn(user_id, "user", user_text)
            user_service.add_conversation_turn(user_id, "assistant", final_text)
            active = user_service.get_active_course(user_id)
            cache_db.track_query(user_id, course=active.course_id if active else None, topic=_extract_topic(user_text))
            logger.info("Total response time: %.2fs (no tools)", time.time() - t_start)
            return final_text

        # LLM wants tools — execute in parallel
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": response_msg.content or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        messages.append(assistant_msg)

        # Refresh typing before tool execution
        if message:
            try:
                await message.chat.send_action("typing")
            except TelegramError:
                pass

        t_tools = time.time()
        tool_results = await asyncio.gather(
            *[_execute_tool_call(tc, user_id) for tc in tool_calls]
        )
        messages.extend(tool_results)
        tool_names = [tc.function.name for tc in tool_calls]
        tools_used.extend(tool_names)

        logger.info(
            "Tools executed (iter %d): %s in %.2fs",
            iteration + 1,
            tool_names,
            time.time() - t_tools,
        )

    # Max iterations exceeded — stream final response
    if message:
        try:
            await message.chat.send_action("typing")
        except TelegramError:
            pass

    try:
        if message:
            t_stream = time.time()
            final_text = await router.stream(messages, system_prompt, message)
            if final_text:
                logger.info("Streaming response: %.2fs", time.time() - t_stream)
                logger.info("Total response time: %.2fs (streamed)", time.time() - t_start)
                user_service.add_conversation_turn(user_id, "user", user_text)
                user_service.add_conversation_turn(user_id, "assistant", final_text)
                return ""

        # Non-streaming fallback
        response_msg = await router.complete(messages, system_prompt, None, 4096)
        final_text = router.sanitize_output(response_msg.content) if response_msg else "Yanıt üretilemedi."
    except Exception:
        final_text = "İşlem zaman aşımına uğradı. Lütfen tekrar deneyin."

    user_service.add_conversation_turn(user_id, "user", user_text)
    user_service.add_conversation_turn(user_id, "assistant", final_text)

    active = user_service.get_active_course(user_id)
    cache_db.track_query(
        user_id,
        course=active.course_id if active else None,
        topic=_extract_topic(user_text),
    )

    logger.info("Total response time: %.2fs (with tools)", time.time() - t_start)
    return final_text


# ─── Warmup (Backwards Compatibility) ────────────────────────────────────────

async def warmup_llm_connections() -> None:
    """Pre-warm LLM connections at startup."""
    router = STATE.llm_router
    if router:
        await router.warmup()
