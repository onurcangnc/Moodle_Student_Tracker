"""
Agentic LLM service with OpenAI function calling.
===================================================
The bot's brain: receives user messages, decides which tools to call via LLM,
executes them, and returns a natural language response.

Tool loop: user message → LLM (with tools) → tool execution → LLM (with results) → reply
Max iterations: 5 (prevents infinite loops)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from bot.services import user_service
from bot.state import STATE

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

# ─── Tool Definitions (OpenAI function calling format) ────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Ders materyallerinde arama yapar. Öğrencinin ders içeriğiyle ilgili sorularını cevaplamak için kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Aranacak sorgu (Türkçe veya İngilizce)",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Kurs adı filtresi (opsiyonel, aktif kurs otomatik kullanılır)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_assignments",
            "description": "Moodle'daki ödevleri ve teslim tarihlerini getirir. Deadline, ödev, teslim soruları için kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "upcoming_only": {
                        "type": "boolean",
                        "description": "True ise sadece yaklaşan (14 gün içinde, teslim edilmemiş) ödevleri getirir",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schedule",
            "description": "Öğrencinin haftalık ders programını getirir. 'Bugün hangi dersim var?', 'Yarın ne var?' gibi sorular için kullan. STARS girişi gerektirir.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grades",
            "description": "Öğrencinin not durumunu (assessment grades) getirir. 'Notlarım ne?', 'Kaç aldım?' gibi sorular için kullan. STARS girişi gerektirir.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_emails",
            "description": "Bilkent AIRS/DAIS e-postalarını getirir ve özetler. 'Mailler ne diyor?', 'Son maillar?' gibi sorular için kullan. Webmail girişi gerektirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Kaç mail getirilsin (varsayılan 5)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_courses",
            "description": "Kayıtlı kursları listeler. 'Hangi derslerim var?', 'Kurslarım?' gibi sorular için kullan.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_active_course",
            "description": "Aktif kursu değiştirir. Öğrenci başka bir ders hakkında konuşmak istediğinde kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Seçilecek kurs adı veya kısa adı (örn: 'CTIS 256', 'POLS')",
                    },
                },
                "required": ["course_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Bot istatistiklerini getirir: chunk sayısı, kurs sayısı, dosya sayısı. Admin soruları için kullan.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "study_overview",
            "description": "Bir kurstaki tüm materyallerin konu haritasını çıkarır. 'Bu derste neler var?', 'Nelere çalışabilirim?' gibi sorular için kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Kurs adı (opsiyonel, aktif kurs otomatik kullanılır)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "study_topic",
            "description": "Belirli bir konuyu derinlemesine araştırır, daha fazla materyal çeker. 'X konusunu anlat', 'X hakkında detaylı bilgi' gibi sorular için kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Çalışılacak konu",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Kurs adı (opsiyonel)",
                    },
                },
                "required": ["topic"],
            },
        },
    },
]


# ─── System Prompt Builder ────────────────────────────────────────────────────


def _build_system_prompt(user_id: int) -> str:
    """Build dynamic system prompt based on user state and available services."""
    active_course = user_service.get_active_course(user_id)
    course_info = f"Aktif kurs: {active_course.display_name}" if active_course else "Aktif kurs seçili değil."

    # Check which services are available
    stars_available = STATE.stars_client is not None and STATE.stars_client.is_authenticated(user_id)
    webmail_available = STATE.webmail_client is not None and STATE.webmail_client.authenticated

    services_status = []
    if stars_available:
        services_status.append("STARS: ✅ Bağlı")
    else:
        services_status.append("STARS: ❌ Giriş yapılmamış (get_schedule, get_grades çalışmaz)")
    if webmail_available:
        services_status.append("Webmail: ✅ Bağlı")
    else:
        services_status.append("Webmail: ❌ Giriş yapılmamış (get_emails çalışmaz)")

    # Student context from LLM engine (date, schedule, STARS data, assignments)
    student_ctx = ""
    if STATE.llm:
        student_ctx = STATE.llm._build_student_context()

    return f"""Sen Bilkent Üniversitesi öğrencisinin kişisel akademik asistanısın.
Telegram üzerinden sohbet ediyorsun. Adın "Moodle Student Tracker".

KİMLİĞİN: GPT, Claude, Gemini gibi model adları SENİN adın DEĞİL — onları hiç söyleme.

GÖREV: Öğrencinin sorularını anla ve DOĞRU TOOL'U çağır.
- Ders içeriği soruları → rag_search (veya study_topic derinlik için)
- Ödev/deadline soruları → get_assignments
- Not soruları → get_grades
- Ders programı → get_schedule
- Mail soruları → get_emails
- Kurs listesi → list_courses
- Kurs değiştirme → set_active_course
- Konu haritası → study_overview
- Genel sohbet (selamlama, teşekkür, vs.) → tool çağırma, direkt cevap ver

KURALLAR:
1. KISA OL: Telegram'da max 3-4 paragraf. Duvar yazısı YAZMA.
2. Veri sorguları (not, program, ödev) → SADECE istenen veriyi ver, ders anlatma.
3. Ders materyali soruları → Socratic method: anlat, sonra kontrol sorusu sor.
4. Tool sonuçlarını doğal dille özetle, JSON/teknik format GÖSTERME.
5. Birden fazla tool gerekiyorsa sırayla çağır (örn: kurs değiştir + arama yap).
6. Tool sonucu boş gelirse kullanıcıya nazikçe bildir.
7. KAYNAK ETİKETLEME: RAG sonuçlarını kullanırken 📖 [dosya_adı] etiketi ekle.
8. FORMAT: Telegram Markdown kullan (*bold*, _italic_, `code`).

ÖĞRENCİ DURUMU:
{course_info}
Servis Durumu: {chr(10).join(services_status)}
{student_ctx}"""


# ─── Tool Availability Filter ────────────────────────────────────────────────


def _get_available_tools(user_id: int) -> list[dict[str, Any]]:
    """Filter tools based on what services are actually usable."""
    available = []
    for tool in TOOLS:
        name = tool["function"]["name"]
        # Always include these
        if name in ("rag_search", "get_assignments", "list_courses",
                     "set_active_course", "get_stats", "study_overview",
                     "study_topic"):
            available.append(tool)
        elif name == "get_schedule" or name == "get_grades":
            # Include even if not authenticated — tool handler returns helpful message
            available.append(tool)
        elif name == "get_emails":
            available.append(tool)
    return available


# ─── LLM Call with Tools ─────────────────────────────────────────────────────


async def _call_llm_with_tools(
    messages: list[dict[str, Any]],
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> Any:
    """Call LLM with function calling via the adapter's OpenAI client."""
    llm = STATE.llm
    if llm is None:
        return None

    # Get the adapter for the chat task
    model_key = llm.engine.router.chat
    adapter = llm.engine.get_adapter(model_key)

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    # Use adapter's OpenAI-compatible client directly for tool calling
    response = await asyncio.to_thread(
        adapter.client.chat.completions.create,
        model=adapter.model,
        messages=full_messages,
        tools=tools if tools else None,
        tool_choice="auto" if tools else None,
        max_tokens=4096,
    )
    return response.choices[0].message


# ─── Tool Handlers ───────────────────────────────────────────────────────────


async def _tool_rag_search(args: dict, user_id: int) -> str:
    """Search course materials via hybrid RAG."""
    query = args.get("query", "")
    if not query:
        return "Arama sorgusu belirtilmedi."

    course_name = args.get("course_name")
    if not course_name:
        active = user_service.get_active_course(user_id)
        course_name = active.course_id if active else None

    store = STATE.vector_store
    if store is None:
        return "Materyal veritabanı henüz hazır değil."

    results = await asyncio.to_thread(
        store.hybrid_search, query, 10, course_name
    )

    if not results:
        # Fallback: search all courses
        if course_name:
            results = await asyncio.to_thread(
                store.hybrid_search, query, 10, None
            )
        if not results:
            return "Bu konuyla ilgili materyal bulunamadı."

    # Format chunks for LLM consumption
    parts = []
    for r in results:
        meta = r.get("metadata", {})
        filename = meta.get("filename", "bilinmeyen")
        course = meta.get("course", "")
        text = r.get("text", "")
        dist = r.get("distance", 0)
        if len(text.strip()) < 50:
            continue
        parts.append(f"[📖 {filename} | Kurs: {course} | Skor: {1-dist:.2f}]\n{text}")

    return "\n\n---\n\n".join(parts) if parts else "İlgili materyal bulunamadı."


async def _tool_get_assignments(args: dict, user_id: int) -> str:
    """Get Moodle assignments."""
    moodle = STATE.moodle
    if moodle is None:
        return "Moodle bağlantısı hazır değil."

    upcoming_only = args.get("upcoming_only", True)

    try:
        if upcoming_only:
            assignments = await asyncio.to_thread(moodle.get_upcoming_assignments, 14)
        else:
            assignments = await asyncio.to_thread(moodle.get_assignments)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Assignment fetch failed: %s", exc, exc_info=True)
        return f"Ödev bilgileri alınamadı: {exc}"

    if not assignments:
        return "Yaklaşan ödev bulunamadı." if upcoming_only else "Hiç ödev bulunamadı."

    lines = []
    for a in assignments:
        status = "✅ Teslim edildi" if a.submitted else "⏳ Teslim edilmedi"
        due = a.due_date if hasattr(a, "due_date") else "Bilinmiyor"
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        line = f"• {a.course_name} — {a.name}\n  Tarih: {due} | {status}"
        if remaining and not a.submitted:
            line += f" | Kalan: {remaining}"
        lines.append(line)

    return "\n".join(lines)


async def _tool_get_schedule(args: dict, user_id: int) -> str:
    """Get weekly schedule from STARS."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(user_id):
        return "STARS girişi yapılmamış. Ders programını görmek için önce /start ile STARS'a giriş yapman gerekiyor."

    try:
        schedule = await asyncio.to_thread(stars.get_schedule, user_id)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Schedule fetch failed: %s", exc, exc_info=True)
        return f"Ders programı alınamadı: {exc}"

    if not schedule:
        return "Ders programı bilgisi bulunamadı."

    lines = []
    for entry in schedule:
        course = entry.get("course", "")
        day = entry.get("day", "")
        hours = entry.get("hours", "")
        room = entry.get("room", "")
        lines.append(f"• {day} {hours} — {course} ({room})")

    return "\n".join(lines) if lines else "Ders programı boş."


async def _tool_get_grades(args: dict, user_id: int) -> str:
    """Get grades from STARS."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(user_id):
        return "STARS girişi yapılmamış. Not bilgilerini görmek için önce /start ile STARS'a giriş yapman gerekiyor."

    try:
        grades = await asyncio.to_thread(stars.get_grades, user_id)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Grades fetch failed: %s", exc, exc_info=True)
        return f"Not bilgileri alınamadı: {exc}"

    if not grades:
        return "Not bilgisi bulunamadı."

    lines = []
    for course in grades:
        course_name = course.get("course", "Bilinmeyen")
        assessments = course.get("assessments", [])
        if not assessments:
            lines.append(f"📚 {course_name}: Henüz not girilmemiş")
            continue
        lines.append(f"📚 {course_name}:")
        for a in assessments:
            name = a.get("name", "")
            grade = a.get("grade", "")
            weight = a.get("weight", "")
            w_str = f" (Ağırlık: {weight})" if weight else ""
            lines.append(f"  • {name}: {grade}{w_str}")

    return "\n".join(lines)


async def _tool_get_emails(args: dict, user_id: int) -> str:
    """Get recent AIRS/DAIS emails."""
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return "Webmail girişi yapılmamış. Mailleri görmek için önce /start ile webmail'e giriş yapman gerekiyor."

    limit = args.get("limit", 5)
    try:
        mails = await asyncio.to_thread(webmail.get_recent_airs_dais, limit)
    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("Email fetch failed: %s", exc, exc_info=True)
        return f"E-postalar alınamadı: {exc}"

    if not mails:
        return "AIRS/DAIS e-postası bulunamadı."

    lines = []
    for m in mails:
        subject = m.get("subject", "Konusuz")
        from_addr = m.get("from", "")
        date = m.get("date", "")
        body = m.get("body_preview", "")
        source = m.get("source", "")
        lines.append(
            f"📧 [{source}] {subject}\n"
            f"  Kimden: {from_addr}\n"
            f"  Tarih: {date}\n"
            f"  İçerik: {body[:200]}{'...' if len(body) > 200 else ''}"
        )

    return "\n\n".join(lines)


async def _tool_list_courses(args: dict, user_id: int) -> str:
    """List available courses."""
    courses = user_service.list_courses()
    if not courses:
        return "Henüz yüklü kurs bulunamadı."

    active = user_service.get_active_course(user_id)
    lines = []
    for c in courses:
        prefix = "▸ " if active and active.course_id == c.course_id else "  "
        lines.append(f"{prefix}{c.short_name} — {c.display_name}")

    return "\n".join(lines)


async def _tool_set_active_course(args: dict, user_id: int) -> str:
    """Set active course."""
    course_name = args.get("course_name", "")
    if not course_name:
        return "Kurs adı belirtilmedi."

    match = user_service.find_course(course_name)
    if match is None:
        courses = user_service.list_courses()
        available = ", ".join(c.short_name for c in courses) if courses else "Yok"
        return f"'{course_name}' ile eşleşen kurs bulunamadı. Mevcut kurslar: {available}"

    user_service.set_active_course(user_id, match.course_id)
    # Also update LLM engine's active course
    if STATE.llm:
        STATE.llm.set_active_course(match.course_id)
    return f"Aktif kurs değiştirildi: {match.display_name}"


async def _tool_get_stats(args: dict, user_id: int) -> str:
    """Get bot statistics."""
    store = STATE.vector_store
    if store is None:
        return "Vector store hazır değil."

    stats = store.get_stats()
    uptime = int(time.monotonic() - STATE.started_at_monotonic)
    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"Toplam chunk: {stats.get('total_chunks', 0)}\n"
        f"Kurs sayısı: {stats.get('unique_courses', 0)}\n"
        f"Dosya sayısı: {stats.get('unique_files', 0)}\n"
        f"Aktif kullanıcı: {len(STATE.active_courses)}\n"
        f"Uptime: {hours}s {minutes}dk {seconds}sn\n"
        f"Versiyon: {STATE.startup_version}"
    )


async def _tool_study_overview(args: dict, user_id: int) -> str:
    """Get course topic map from file metadata."""
    course_name = args.get("course_name")
    if not course_name:
        active = user_service.get_active_course(user_id)
        course_name = active.course_id if active else None

    if not course_name:
        return "Aktif kurs seçili değil. Önce bir kurs seç."

    store = STATE.vector_store
    if store is None:
        return "Materyal veritabanı hazır değil."

    try:
        files = await asyncio.to_thread(store.get_files_for_course, course_name)
    except (AttributeError, RuntimeError, ValueError) as exc:
        logger.error("Study overview failed: %s", exc, exc_info=True)
        return f"Konu haritası alınamadı: {exc}"

    if not files:
        return f"'{course_name}' kursu için yüklü materyal bulunamadı."

    # Also include file summaries if available
    summaries = STATE.file_summaries or {}
    lines = []
    for f in files:
        filename = f.get("filename", "")
        chunk_count = f.get("chunk_count", 0)
        section = f.get("section", "")
        summary = summaries.get(filename, {}).get("summary", "")
        line = f"📄 {filename} ({chunk_count} parça)"
        if section:
            line += f" — Bölüm: {section}"
        if summary:
            line += f"\n   Özet: {summary[:150]}..."
        lines.append(line)

    return f"📚 {course_name} — Materyal Haritası:\n\n" + "\n\n".join(lines)


async def _tool_study_topic(args: dict, user_id: int) -> str:
    """Deep search for a specific topic with higher top_k."""
    topic = args.get("topic", "")
    if not topic:
        return "Konu belirtilmedi."

    course_name = args.get("course_name")
    if not course_name:
        active = user_service.get_active_course(user_id)
        course_name = active.course_id if active else None

    store = STATE.vector_store
    if store is None:
        return "Materyal veritabanı hazır değil."

    # Higher top_k for deeper study
    results = await asyncio.to_thread(
        store.hybrid_search, topic, 25, course_name
    )

    if not results:
        if course_name:
            results = await asyncio.to_thread(
                store.hybrid_search, topic, 25, None
            )
        if not results:
            return f"'{topic}' konusuyla ilgili materyal bulunamadı."

    # Format with file summaries
    summaries = STATE.file_summaries or {}
    parts = []
    seen_files = set()
    for r in results:
        meta = r.get("metadata", {})
        filename = meta.get("filename", "bilinmeyen")
        text = r.get("text", "")
        dist = r.get("distance", 0)
        if len(text.strip()) < 50:
            continue

        # Add file summary header once per file
        if filename not in seen_files:
            seen_files.add(filename)
            file_summary = summaries.get(filename, {}).get("summary", "")
            if file_summary:
                parts.append(f"[📄 {filename} — Dosya Özeti: {file_summary[:200]}]")

        parts.append(f"[📖 {filename} | Skor: {1-dist:.2f}]\n{text}")

    return "\n\n---\n\n".join(parts) if parts else f"'{topic}' ile ilgili yeterli materyal bulunamadı."


# ─── Tool Dispatcher ─────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "rag_search": _tool_rag_search,
    "get_assignments": _tool_get_assignments,
    "get_schedule": _tool_get_schedule,
    "get_grades": _tool_get_grades,
    "get_emails": _tool_get_emails,
    "list_courses": _tool_list_courses,
    "set_active_course": _tool_set_active_course,
    "get_stats": _tool_get_stats,
    "study_overview": _tool_study_overview,
    "study_topic": _tool_study_topic,
}


async def _execute_tool_call(tool_call: Any, user_id: int) -> dict[str, str]:
    """Execute a single tool call and return the result message."""
    fn_name = tool_call.function.name
    try:
        fn_args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        fn_args = {}

    handler = TOOL_HANDLERS.get(fn_name)
    if handler is None:
        logger.warning("Unknown tool called: %s", fn_name)
        result = f"Bilinmeyen araç: {fn_name}"
    else:
        try:
            result = await handler(fn_args, user_id)
        except Exception as exc:
            logger.error("Tool %s failed: %s", fn_name, exc, exc_info=True)
            result = f"Araç hatası ({fn_name}): {exc}"

    logger.info(
        "Tool executed",
        extra={
            "tool": fn_name,
            "args": fn_args,
            "result_len": len(result),
            "user_id": user_id,
        },
    )

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }


# ─── Main Entry Point ────────────────────────────────────────────────────────


async def handle_agent_message(user_id: int, user_text: str) -> str:
    """
    Main agentic handler: takes user message, runs tool loop, returns final response.

    Flow:
    1. Build system prompt with user state
    2. Get conversation history
    3. Call LLM with tools
    4. If tool calls → execute → feed results back → repeat (max 5 iterations)
    5. Return final text response
    """
    if STATE.llm is None:
        return "Sistem henüz hazır değil. Lütfen birazdan tekrar deneyin."

    # Build context
    system_prompt = _build_system_prompt(user_id)
    available_tools = _get_available_tools(user_id)

    # Get conversation history
    history = user_service.get_conversation_history(user_id)
    messages: list[dict[str, Any]] = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    # Tool loop
    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response_msg = await _call_llm_with_tools(
                messages, system_prompt, available_tools
            )
        except Exception as exc:
            logger.error("LLM call failed (iteration %d): %s", iteration, exc, exc_info=True)
            return "Bir hata oluştu. Lütfen tekrar deneyin."

        if response_msg is None:
            return "Yanıt üretilemedi. Lütfen tekrar deneyin."

        # Check if LLM wants to call tools
        tool_calls = getattr(response_msg, "tool_calls", None)
        if not tool_calls:
            # No tool calls — return the text response
            final_text = response_msg.content or ""
            # Record conversation
            user_service.add_conversation_turn(user_id, "user", user_text)
            user_service.add_conversation_turn(user_id, "assistant", final_text)

            # Record for persistent memory
            if STATE.llm and STATE.llm.mem_manager:
                active = user_service.get_active_course(user_id)
                STATE.llm.mem_manager.record_exchange(
                    user_message=user_text,
                    assistant_response=final_text,
                    course=active.course_id if active else "",
                    rag_sources="",
                )

            return final_text

        # LLM wants tools — add assistant message with tool calls
        # Build assistant message dict from the response
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

        # Execute all tool calls
        tool_results = await asyncio.gather(
            *[_execute_tool_call(tc, user_id) for tc in tool_calls]
        )
        messages.extend(tool_results)

        logger.info(
            "Tool loop iteration %d: %d tool calls executed",
            iteration + 1,
            len(tool_calls),
            extra={"user_id": user_id, "tools": [tc.function.name for tc in tool_calls]},
        )

    # Exceeded max iterations — ask LLM for final response without tools
    try:
        response_msg = await _call_llm_with_tools(messages, system_prompt, [])
        final_text = response_msg.content if response_msg else "Yanıt üretilemedi."
    except Exception:
        final_text = "İşlem zaman aşımına uğradı. Lütfen tekrar deneyin."

    user_service.add_conversation_turn(user_id, "user", user_text)
    user_service.add_conversation_turn(user_id, "assistant", final_text)
    return final_text
