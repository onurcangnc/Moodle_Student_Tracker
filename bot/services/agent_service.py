"""
Agentic LLM service with OpenAI function calling — v2.
========================================================
The bot's brain: receives user messages, decides which tools to call via LLM,
executes them, and returns a natural language response.

14 tools:
  rag_search, get_assignments, get_schedule, get_grades, get_emails,
  get_email_detail, list_courses, set_active_course, get_stats,
  study_overview, study_topic, study_source, list_course_materials,
  get_attendance

Tool loop: user → LLM (with tools) → tool exec → LLM (with results) → reply
Max iterations: 5 (prevents infinite loops)
Supports parallel_tool_calls for multi-tool queries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
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
            "description": (
                "Ders materyallerinde arama yapar. Öğrencinin ders içeriğiyle ilgili "
                "sorularını cevaplamak için kullan. Aktif kurs yoksa tüm kurslarda arar."
            ),
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
            "description": (
                "Moodle'daki ödevleri ve teslim tarihlerini getirir. "
                "'Ödevlerim neler?', 'Deadline ne zaman?', 'Teslim edilmemiş ödevler' gibi sorular için kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "enum": ["upcoming", "overdue", "all"],
                        "description": (
                            "upcoming: 14 gün içindeki teslim edilmemiş ödevler (varsayılan). "
                            "overdue: süresi geçmiş ama teslim edilmemiş ödevler. "
                            "all: tüm ödevler."
                        ),
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
            "description": (
                "Öğrencinin haftalık ders programını getirir. "
                "'Bugün hangi dersim var?', 'Yarın ne var?', 'Cuma programım?' gibi sorular için kullan. "
                "STARS girişi gerektirir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["today", "tomorrow", "week"],
                        "description": (
                            "today: sadece bugünün dersleri. "
                            "tomorrow: yarının dersleri. "
                            "week: tüm haftalık program (varsayılan)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grades",
            "description": (
                "Öğrencinin not durumunu (assessment grades) getirir. "
                "'Notlarım ne?', 'Kaç aldım?', 'CTIS 256 notlarım?' gibi sorular için kullan. "
                "STARS girişi gerektirir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_filter": {
                        "type": "string",
                        "description": "Belirli bir kursun notlarını filtrelemek için kurs adı (opsiyonel, boş bırakılırsa tüm kurslar)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_emails",
            "description": (
                "Bilkent AIRS/DAIS e-postalarını getirir. "
                "Kullanıcı 'son mailleri göster' derse, ÖNCELİKLE 'Kaç mail görmek istersin?' diye sor — "
                "bu tool'u hemen çağırma. Sayı belirtildiğinde limit parametresiyle çağır. "
                "scope='unread' sadece okunmamış mailleri getirir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Kaç mail getirilsin (varsayılan 5)",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["recent", "unread"],
                        "description": "recent: son mailleri getirir (varsayılan). unread: sadece okunmamış.",
                    },
                    "sender_filter": {
                        "type": "string",
                        "description": "Gönderici filtresi — AIRS veya DAIS (opsiyonel)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_email_detail",
            "description": (
                "Belirli bir e-postanın tam içeriğini getirir. "
                "Kullanıcı bir mailin detayını görmek istediğinde kullan. "
                "subject parametresi ile eşleşen maili bulur."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Detayı görülmek istenen mailin konusu (kısmi eşleşme yeterli)",
                    },
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_courses",
            "description": (
                "Kayıtlı Moodle kurslarını listeler. Aktif kurs işaretli gösterilir. "
                "'Hangi derslerim var?', 'Kurslarım?' gibi sorular için kullan."
            ),
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
            "description": (
                "Aktif kursu değiştirir. Öğrenci başka bir ders hakkında konuşmak istediğinde "
                "veya kurs adı belirttiğinde kullan. RAG araması ve study tool'ları aktif kursu kullanır."
            ),
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
            "description": (
                "Bot istatistiklerini getirir: chunk sayısı, kurs sayısı, dosya sayısı, uptime. "
                "Admin soruları veya 'botun durumu ne?' gibi sorular için kullan."
            ),
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
            "description": (
                "Bir kurstaki tüm materyallerin konu haritasını çıkarır. Dosya listesi ve özetleri gösterir. "
                "'Bu derste neler var?', 'Nelere çalışabilirim?', 'Konu listesi' gibi sorular için kullan. "
                "Daha sonra study_topic ile derinleşebilirsin."
            ),
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
            "description": (
                "Belirli bir konuyu derinlemesine araştırır, daha fazla materyal çeker. "
                "'X konusunu anlat', 'X hakkında detaylı bilgi', 'X nedir?' gibi sorular için kullan. "
                "depth=deep daha fazla chunk getirir."
            ),
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
                    "depth": {
                        "type": "string",
                        "enum": ["normal", "deep"],
                        "description": "normal: 15 chunk (varsayılan). deep: 25 chunk, dosya özetleri dahil.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "study_source",
            "description": (
                "Belirli bir dosyanın tüm içeriğini chunk'lar halinde getirir. "
                "'Bu PDF'i oku', 'Dosyanın tamamını göster', 'Chapter 5'i göster' gibi istekler için kullan. "
                "Dosya adını study_overview veya rag_search sonuçlarından al."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Kaynak dosya adı (ör: 'hafta3.pdf', 'lecture5.pptx')",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maksimum chunk sayısı (0 = tümü, varsayılan 20)",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_course_materials",
            "description": (
                "Bir kurstaki tüm dosya ve materyalleri listeler (chunk sayılarıyla). "
                "'Derste hangi dosyalar var?', 'Materyalleri göster' gibi sorular için kullan. "
                "study_overview'dan farklı olarak sadece dosya listesi verir, özet vermez."
            ),
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
            "name": "get_attendance",
            "description": (
                "Öğrencinin devamsızlık durumunu getirir (kurs bazlı, yüzde oranıyla). "
                "'Devamsızlığım ne?', 'Kaç derse girmedim?' gibi sorular için kullan. "
                "Devamsızlık %20'ye yaklaşıyorsa uyar. STARS girişi gerektirir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_filter": {
                        "type": "string",
                        "description": "Belirli bir kursun devamsızlığı (opsiyonel, boşsa tümü)",
                    },
                },
                "required": [],
            },
        },
    },
]


# ─── System Prompt Builder ────────────────────────────────────────────────────

_DAY_NAMES_TR = {
    0: "Pazartesi",
    1: "Salı",
    2: "Çarşamba",
    3: "Perşembe",
    4: "Cuma",
    5: "Cumartesi",
    6: "Pazar",
}


def _build_system_prompt(user_id: int) -> str:
    """Build dynamic system prompt based on user state and available services."""
    active_course = user_service.get_active_course(user_id)
    course_info = f"Aktif kurs: {active_course.display_name}" if active_course else "Aktif kurs seçili değil."

    stars_available = STATE.stars_client is not None and STATE.stars_client.is_authenticated(user_id)
    webmail_available = STATE.webmail_client is not None and STATE.webmail_client.authenticated

    services = []
    if stars_available:
        services.append("STARS: ✅ Bağlı (program, not, devamsızlık erişilebilir)")
    else:
        services.append("STARS: ❌ Giriş yapılmamış → get_schedule, get_grades, get_attendance çalışmaz")
    if webmail_available:
        services.append("Webmail: ✅ Bağlı (mail erişilebilir)")
    else:
        services.append("Webmail: ❌ Giriş yapılmamış → get_emails, get_email_detail çalışmaz")

    now = datetime.now()
    today_tr = _DAY_NAMES_TR.get(now.weekday(), "")
    date_str = now.strftime("%d/%m/%Y %H:%M")

    student_ctx = ""
    if STATE.llm:
        student_ctx = STATE.llm._build_student_context()

    return f"""Sen Bilkent Üniversitesi öğrencisinin kişisel akademik asistanısın.
Telegram üzerinden sohbet ediyorsun.

KİMLİK KURALI: Sen bir Bilkent akademik asistanısın. GPT, Claude, Gemini, OpenAI gibi model isimlerini ASLA söyleme — sen onlar değilsin.

GÖREV: Öğrencinin doğal dildeki mesajını anla ve doğru tool'u çağır.

TOOL SEÇİM REHBERİ:
• Ders içeriği sorusu → rag_search (genel soru) veya study_topic (derinlemesine)
• Ödev/deadline sorusu → get_assignments
• Not sorusu → get_grades
• Ders programı → get_schedule
• Mail sorusu → get_emails (ama önce limit sor!) veya get_email_detail
• Devamsızlık → get_attendance
• Kurs listesi → list_courses
• Kurs değiştirme → set_active_course
• Materyal listesi → list_course_materials veya study_overview (özetli)
• Dosya içeriği → study_source
• Genel sohbet (selam, teşekkür, günlük) → tool çağırmadan direkt cevap ver

MAİL AKIŞI (KRİTİK):
Kullanıcı "son maillerimi göster" / "mailler ne diyor?" gibi bir şey derse:
→ Tool çağırma! Önce "Kaç mail görmek istersin? (1-10)" diye sor.
→ Kullanıcı sayı söyleyince o sayıyla get_emails(limit=N) çağır.
→ Kullanıcı "AIRS maillerini göster" derse sender_filter="AIRS" kullan.

ÇALIŞMA MODU AKIŞI:
1. "Nelere çalışabilirim?" → study_overview (konu haritası)
2. "X konusunu anlat" → study_topic (konu detayı)
3. "Dosyayı oku" / "PDF'i göster" → study_source (tam dosya)
Bu sıralama önerilir ama zorunlu değil — öğrenci direkt konu sorabilir.

DEVAMSIZLIK UYARISI:
get_attendance sonucu %15 üzeri devamsızlık gösteriyorsa uyar:
"⚠️ Dikkat: [Kurs] devamsızlığın %X — limit %20."

FORMAT KURALLARI:
1. Telegram Markdown kullan: *bold*, _italic_, `code`
2. Kısa ol — Telegram'da max 3-4 paragraf. Duvar yazısı YAZMA.
3. Veri sorguları (not, program, ödev) → SADECE istenen veriyi ver, ders ANLATMA.
4. RAG sonuçlarını kullanırken 📖 [dosya_adı] kaynak etiketi ekle.
5. Tool sonuçlarını doğal dille özetle, JSON/teknik format GÖSTERME.
6. Tool sonucu boş gelirse nazikçe bildir.
7. Birden fazla tool gerekiyorsa paralel çağır (örn: "bugün ne var?" → get_schedule + get_assignments).

ÖĞRENCİ DURUMU:
{course_info}
Servis Durumu: {chr(10).join(services)}
Tarih: {date_str} ({today_tr})
{student_ctx}"""


# ─── Tool Availability Filter ────────────────────────────────────────────────


def _get_available_tools(user_id: int) -> list[dict[str, Any]]:
    """Return all tools — unavailable services are handled by tool handlers with helpful messages."""
    return list(TOOLS)


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

    model_key = llm.engine.router.chat
    adapter = llm.engine.get_adapter(model_key)

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    kwargs: dict[str, Any] = {
        "model": adapter.model,
        "messages": full_messages,
        "max_tokens": 4096,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = True

    response = await asyncio.to_thread(
        adapter.client.chat.completions.create,
        **kwargs,
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

    results = await asyncio.to_thread(store.hybrid_search, query, 10, course_name)

    if not results and course_name:
        results = await asyncio.to_thread(store.hybrid_search, query, 10, None)

    if not results:
        return "Bu konuyla ilgili materyal bulunamadı."

    parts = []
    for r in results:
        meta = r.get("metadata", {})
        filename = meta.get("filename", "bilinmeyen")
        course = meta.get("course", "")
        text = r.get("text", "")
        dist = r.get("distance", 0)
        if len(text.strip()) < 50:
            continue
        parts.append(f"[📖 {filename} | Kurs: {course} | Skor: {1 - dist:.2f}]\n{text}")

    return "\n\n---\n\n".join(parts) if parts else "İlgili materyal bulunamadı."


async def _tool_get_assignments(args: dict, user_id: int) -> str:
    """Get Moodle assignments with optional filtering."""
    moodle = STATE.moodle
    if moodle is None:
        return "Moodle bağlantısı hazır değil."

    filter_mode = args.get("filter", "upcoming")
    now_ts = time.time()

    try:
        if filter_mode == "all":
            assignments = await asyncio.to_thread(moodle.get_assignments)
        else:
            assignments = await asyncio.to_thread(moodle.get_upcoming_assignments, 14)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Assignment fetch failed: %s", exc, exc_info=True)
        return f"Ödev bilgileri alınamadı: {exc}"

    if filter_mode == "overdue":
        assignments = [
            a for a in (assignments or [])
            if not a.submitted and a.due_date and a.due_date < now_ts
        ]

    if not assignments:
        labels = {"upcoming": "Yaklaşan", "overdue": "Süresi geçmiş", "all": "Hiç"}
        return f"{labels.get(filter_mode, 'Yaklaşan')} ödev bulunamadı."

    lines = []
    for a in assignments:
        status = "✅ Teslim edildi" if a.submitted else "⏳ Teslim edilmedi"
        due = a.due_date if hasattr(a, "due_date") else "Bilinmiyor"
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        line = f"• {a.course_name} — {a.name}\n  Tarih: {due} | {status}"
        if remaining and not a.submitted:
            line += f" | Kalan: {remaining}"
        if filter_mode == "overdue":
            line += " | ⚠️ Süresi geçmiş!"
        lines.append(line)

    return "\n".join(lines)


async def _tool_get_schedule(args: dict, user_id: int) -> str:
    """Get weekly schedule from STARS with optional day filter."""
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

    period = args.get("period", "week")

    if period in ("today", "tomorrow"):
        now = datetime.now()
        if period == "tomorrow":
            from datetime import timedelta
            target = now + timedelta(days=1)
        else:
            target = now
        target_day = _DAY_NAMES_TR.get(target.weekday(), "")
        schedule = [e for e in schedule if e.get("day", "") == target_day]
        if not schedule:
            return f"{target_day} günü için ders bulunamadı."

    lines = []
    current_day = ""
    for entry in schedule:
        day = entry.get("day", "")
        time_slot = entry.get("time", "")
        course = entry.get("course", "")
        room = entry.get("room", "")
        if day != current_day:
            current_day = day
            lines.append(f"\n*{day}*")
        room_str = f" ({room})" if room else ""
        lines.append(f"  • {time_slot} — {course}{room_str}")

    return "\n".join(lines).strip() if lines else "Ders programı boş."


async def _tool_get_grades(args: dict, user_id: int) -> str:
    """Get grades from STARS with optional course filter."""
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

    course_filter = args.get("course_filter", "")
    if course_filter:
        cf_lower = course_filter.lower()
        grades = [g for g in grades if cf_lower in g.get("course", "").lower()]
        if not grades:
            return f"'{course_filter}' ile eşleşen kurs notu bulunamadı."

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
    scope = args.get("scope", "recent")
    sender_filter = args.get("sender_filter", "")

    try:
        if scope == "unread":
            mails = await asyncio.to_thread(webmail.check_all_unread)
        else:
            mails = await asyncio.to_thread(webmail.get_recent_airs_dais, limit)
    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("Email fetch failed: %s", exc, exc_info=True)
        return f"E-postalar alınamadı: {exc}"

    if sender_filter:
        sf = sender_filter.upper()
        mails = [m for m in mails if m.get("source", "").upper() == sf]

    if scope != "unread":
        mails = mails[:limit]

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
            f"  Özet: {body[:200]}{'...' if len(body) > 200 else ''}"
        )

    return "\n\n".join(lines)


async def _tool_get_email_detail(args: dict, user_id: int) -> str:
    """Get full content of a specific email by subject match."""
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return "Webmail girişi yapılmamış."

    subject_query = args.get("subject", "")
    if not subject_query:
        return "Mail konusu belirtilmedi."

    try:
        mails = await asyncio.to_thread(webmail.get_recent_airs_dais, 10)
    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("Email detail fetch failed: %s", exc, exc_info=True)
        return f"Mail detayı alınamadı: {exc}"

    sq = subject_query.lower()
    match = None
    for m in mails:
        if sq in m.get("subject", "").lower():
            match = m
            break

    if not match:
        return f"'{subject_query}' konusuyla eşleşen mail bulunamadı."

    subject = match.get("subject", "Konusuz")
    from_addr = match.get("from", "")
    date = match.get("date", "")
    body = match.get("body_preview", "")

    return (
        f"📧 *{subject}*\n"
        f"Kimden: {from_addr}\n"
        f"Tarih: {date}\n\n"
        f"{body}"
    )


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
    """Get course topic map from file metadata + summaries."""
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
    """Deep search for a specific topic with configurable depth."""
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

    depth = args.get("depth", "normal")
    top_k = 25 if depth == "deep" else 15

    results = await asyncio.to_thread(store.hybrid_search, topic, top_k, course_name)

    if not results and course_name:
        results = await asyncio.to_thread(store.hybrid_search, topic, top_k, None)

    if not results:
        return f"'{topic}' konusuyla ilgili materyal bulunamadı."

    summaries = STATE.file_summaries or {}
    parts = []
    seen_files: set[str] = set()
    for r in results:
        meta = r.get("metadata", {})
        filename = meta.get("filename", "bilinmeyen")
        text = r.get("text", "")
        dist = r.get("distance", 0)
        if len(text.strip()) < 50:
            continue

        if filename not in seen_files and depth == "deep":
            seen_files.add(filename)
            file_summary = summaries.get(filename, {}).get("summary", "")
            if file_summary:
                parts.append(f"[📄 {filename} — Dosya Özeti: {file_summary[:200]}]")

        parts.append(f"[📖 {filename} | Skor: {1 - dist:.2f}]\n{text}")

    return "\n\n---\n\n".join(parts) if parts else f"'{topic}' ile ilgili yeterli materyal bulunamadı."


async def _tool_study_source(args: dict, user_id: int) -> str:
    """Get full file content chunk by chunk."""
    filename = args.get("filename", "")
    if not filename:
        return "Dosya adı belirtilmedi."

    store = STATE.vector_store
    if store is None:
        return "Materyal veritabanı hazır değil."

    max_chunks = args.get("max_chunks", 20)

    try:
        chunks = await asyncio.to_thread(store.get_file_chunks, filename, max_chunks)
    except (AttributeError, RuntimeError, ValueError) as exc:
        logger.error("Study source failed: %s", exc, exc_info=True)
        return f"Dosya içeriği alınamadı: {exc}"

    if not chunks:
        return f"'{filename}' dosyası bulunamadı. study_overview veya rag_search ile doğru dosya adını kontrol edin."

    total_chunks = len(chunks)
    parts = []
    for c in chunks:
        text = c.get("text", "")
        idx = c.get("chunk_index", 0)
        if text.strip():
            parts.append(f"[Parça {idx + 1}]\n{text}")

    header = f"📄 *{filename}* — {total_chunks} parça"
    if max_chunks and total_chunks == max_chunks:
        header += f" (ilk {max_chunks} gösteriliyor)"

    return header + "\n\n" + "\n\n---\n\n".join(parts)


async def _tool_list_course_materials(args: dict, user_id: int) -> str:
    """List all files for a course (lightweight, no summaries)."""
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
        logger.error("List materials failed: %s", exc, exc_info=True)
        return f"Materyal listesi alınamadı: {exc}"

    if not files:
        return f"'{course_name}' kursu için yüklü materyal bulunamadı."

    lines = []
    total_chunks = 0
    for f in files:
        filename = f.get("filename", "")
        chunk_count = f.get("chunk_count", 0)
        total_chunks += chunk_count
        section = f.get("section", "")
        line = f"  • {filename} ({chunk_count} parça)"
        if section:
            line += f" — {section}"
        lines.append(line)

    header = f"📚 {course_name} — {len(files)} dosya, {total_chunks} toplam parça:\n"
    return header + "\n".join(lines)


async def _tool_get_attendance(args: dict, user_id: int) -> str:
    """Get attendance records from STARS."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(user_id):
        return "STARS girişi yapılmamış. Devamsızlık bilgisi için önce /start ile STARS'a giriş yapman gerekiyor."

    try:
        attendance = await asyncio.to_thread(stars.get_attendance, user_id)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Attendance fetch failed: %s", exc, exc_info=True)
        return f"Devamsızlık bilgisi alınamadı: {exc}"

    if not attendance:
        return "Devamsızlık bilgisi bulunamadı."

    course_filter = args.get("course_filter", "")
    if course_filter:
        cf_lower = course_filter.lower()
        attendance = [a for a in attendance if cf_lower in a.get("course", "").lower()]
        if not attendance:
            return f"'{course_filter}' ile eşleşen kurs devamsızlığı bulunamadı."

    lines = []
    for course_data in attendance:
        course_name = course_data.get("course", "Bilinmeyen")
        records = course_data.get("records", [])
        ratio = course_data.get("ratio", "")

        total = len(records)
        absent = sum(1 for r in records if not r.get("attended", True))

        line = f"📚 {course_name}:"
        if ratio:
            line += f" Devam oranı: {ratio}"
        line += f" ({absent}/{total} devamsız)"

        # Warn if approaching limit
        try:
            ratio_num = float(ratio.replace("%", "")) if ratio else 100
            if ratio_num < 85:
                line += "\n  ⚠️ Dikkat: Devamsızlık limiti %20'ye yaklaşıyor!"
        except (ValueError, AttributeError):
            pass

        lines.append(line)

    return "\n".join(lines)


# ─── Tool Dispatcher ─────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "rag_search": _tool_rag_search,
    "get_assignments": _tool_get_assignments,
    "get_schedule": _tool_get_schedule,
    "get_grades": _tool_get_grades,
    "get_emails": _tool_get_emails,
    "get_email_detail": _tool_get_email_detail,
    "list_courses": _tool_list_courses,
    "set_active_course": _tool_set_active_course,
    "get_stats": _tool_get_stats,
    "study_overview": _tool_study_overview,
    "study_topic": _tool_study_topic,
    "study_source": _tool_study_source,
    "list_course_materials": _tool_list_course_materials,
    "get_attendance": _tool_get_attendance,
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
    4. If tool calls → execute (parallel) → feed results back → repeat (max 5 iterations)
    5. Return final text response
    """
    if STATE.llm is None:
        return "Sistem henüz hazır değil. Lütfen birazdan tekrar deneyin."

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
            final_text = response_msg.content or ""
            user_service.add_conversation_turn(user_id, "user", user_text)
            user_service.add_conversation_turn(user_id, "assistant", final_text)

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

        # Execute all tool calls in parallel
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
