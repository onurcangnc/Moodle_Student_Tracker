"""
Agentic LLM service with OpenAI function calling — v3.
========================================================
The bot's brain: 3-Layer Knowledge Architecture + 14 tools.

KATMAN 1 — Index: metadata aggregation (get_source_map, instant, free)
KATMAN 2 — Summary: pre-generated teaching overviews (read_source, stored JSON)
KATMAN 3 — Deep read: chunk-based content (rag_search, study_topic, read_source)

14 tools:
  get_source_map, read_source, study_topic, rag_search, get_moodle_materials,
  get_schedule, get_grades, get_attendance, get_assignments,
  get_emails, get_email_detail, list_courses, set_active_course, get_stats

Tool loop: user → LLM (with tools) → tool exec → LLM (with results) → reply
Max iterations: 5, parallel_tool_calls=True
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

from litellm import Router
from telegram import Message
from telegram.error import TelegramError

from bot.services import user_service
from bot.state import STATE
from core import cache_db

logger = logging.getLogger(__name__)

# ─── LiteLLM Router (latency-based model selection) ─────────────────────────────
# Automatically picks the fastest responding model, with fallback on failure.

_litellm_router: Router | None = None


def _get_fast_router() -> Router:
    """Lazy-init LiteLLM router with latency-based routing."""
    global _litellm_router
    if _litellm_router is not None:
        return _litellm_router

    # Build model list — only models with RELIABLE tool calling support
    model_list = []

    # OpenAI GPT-4.1-mini (fast, reliable, best tool support)
    if os.getenv("OPENAI_API_KEY"):
        model_list.append({
            "model_name": "fast",
            "litellm_params": {
                "model": "gpt-4.1-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        })

    # Google Gemini 2.5 Flash (free tier: 15 RPM, good tool support)
    if os.getenv("GEMINI_API_KEY"):
        model_list.append({
            "model_name": "fast",
            "litellm_params": {
                "model": "gemini/gemini-2.5-flash",
                "api_key": os.getenv("GEMINI_API_KEY"),
            },
        })

    # Mistral Large — DISABLED: LiteLLM generates tool call IDs that don't match
    # Mistral's required format (9 char alphanumeric). This is a LiteLLM bug.
    # if os.getenv("MISTRAL_API_KEY"): ...

    # DeepSeek V3 (very cheap: $0.14/M input, good tool support)
    if os.getenv("DEEPSEEK_API_KEY"):
        model_list.append({
            "model_name": "fast",
            "litellm_params": {
                "model": "deepseek/deepseek-chat",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
            },
        })

    # GLM-5 — DISABLED: doesn't support parallel_tool_calls parameter
    # if os.getenv("GLM_API_KEY"): ...

    if not model_list:
        raise RuntimeError("No LLM API keys configured!")

    _litellm_router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        num_retries=2,
        timeout=30,
        enable_pre_call_checks=True,
    )
    logger.info("LiteLLM router initialized with %d models", len(model_list))
    return _litellm_router


async def warmup_llm_connections() -> None:
    """
    Pre-warm LLM connections at startup to eliminate cold start latency.
    Makes a lightweight call to establish TCP/TLS connections and populate
    the router's latency measurements.
    """
    try:
        router = _get_fast_router()
        start = time.time()
        # Minimal call to warm up connections
        await router.acompletion(
            model="fast",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        elapsed = (time.time() - start) * 1000
        logger.info("LLM connections warmed up in %.0fms", elapsed)
    except Exception as exc:
        logger.warning("LLM warmup failed (non-critical): %s", exc)

MAX_TOOL_ITERATIONS = 5

# ─── Output Sanitization ─────────────────────────────────────────────────────

# DeepSeek V3 sometimes leaks internal control tokens (DSML tags) in output.
# These are not meant for the user and should be stripped.
import re as _re
_DEEPSEEK_LEAK_RE = _re.compile(r"<｜[A-Z]+｜[^>]*>")
_CONTROL_TAG_RE = _re.compile(r"<\|[a-z_]+\|>")


def _sanitize_llm_output(text: str) -> str:
    """Strip internal model control tokens from LLM output."""
    if not text:
        return text
    text = _DEEPSEEK_LEAK_RE.sub("", text)
    text = _CONTROL_TAG_RE.sub("", text)
    return text.strip()


# ─── Instant Responses (LLM bypass for simple queries) ───────────────────────

_INSTANT_RESPONSES: dict[str, str] = {
    # Greetings
    "merhaba": "Merhaba! 👋 Nasıl yardımcı olabilirim?",
    "selam": "Selam! Ne yapmak istersin?",
    "slm": "Selam! Ne yapmak istersin?",
    "mrb": "Merhaba! 👋 Nasıl yardımcı olabilirim?",
    "hi": "Hi! How can I help you?",
    "hello": "Hello! What can I do for you?",
    "hey": "Hey! How can I help?",
    # Gratitude
    "teşekkürler": "Rica ederim! Başka bir şey lazım olursa yaz. 🙂",
    "teşekkür ederim": "Rica ederim! Başka bir şey lazım olursa yaz. 🙂",
    "sağol": "Ne demek! Başka sorun varsa sor. 🙂",
    "sağ ol": "Ne demek! Başka sorun varsa sor. 🙂",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Let me know if you need anything else.",
    # Acknowledgments
    "tamam": "Tamam! Başka bir şey var mı?",
    "ok": "Okay! Anything else?",
    "anladım": "Güzel! Başka sorun olursa yaz.",
    "peki": "Peki! Başka bir konuda yardım edebilir miyim?",
    # Farewells
    "görüşürüz": "Görüşürüz! İyi çalışmalar! 👋",
    "bye": "Bye! Good luck! 👋",
    "bb": "Görüşürüz! 👋",
}


def _check_instant_response(text: str) -> str | None:
    """Check if message matches an instant response pattern."""
    normalized = text.strip().lower()
    # Exact match
    if normalized in _INSTANT_RESPONSES:
        return _INSTANT_RESPONSES[normalized]
    return None


# ─── Smart Model Selection ───────────────────────────────────────────────────

_COMPLEXITY_KEYWORDS = {
    # Deep analysis requests
    "detaylı", "ayrıntılı", "derinlemesine", "kapsamlı", "analiz",
    "karşılaştır", "compare", "explain in detail", "thoroughly",
    # Multi-aspect queries
    "akademik durumum", "genel durum", "özet", "summary",
    "tüm dersler", "all courses", "everything",
    # Teaching requests
    "anlat", "öğret", "açıkla", "explain", "teach",
    # Planning requests
    "nasıl hazırlanayım", "strateji", "plan", "tavsiye",
}


def _is_complex_query(user_text: str, tool_count: int = 0) -> bool:
    """Detect if query needs higher-quality model."""
    text_lower = user_text.lower()

    # Multiple tools requested = complex
    if tool_count >= 2:
        return True

    # Long query = likely complex
    if len(user_text) > 150:
        return True

    # Complexity keywords
    if any(kw in text_lower for kw in _COMPLEXITY_KEYWORDS):
        return True

    return False


# ─── Smart Error Messages ────────────────────────────────────────────────────

def _extract_topic(text: str) -> str | None:
    """Extract main topic from user query for profile tracking."""
    text_lower = text.lower()

    # Skip greetings and short messages
    if len(text) < 10:
        return None

    # Common topic patterns
    topic_patterns = [
        # Course-related
        ("not", "notlar"),
        ("devamsızlık", "devamsızlık"),
        ("ders program", "program"),
        ("ödev", "ödevler"),
        ("sınav", "sınavlar"),
        ("mail", "mailler"),
        # Study-related
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
            "⚠️ Yanıt oluştururken bir sorun oluştu.\n\n"
            "Şunları deneyebilirsin:\n"
            "• Soruyu daha kısa/basit yaz\n"
            "• Biraz bekleyip tekrar dene\n"
            f"{context}"
        )

    if error_type == "llm_null":
        return (
            "⚠️ Yanıt üretilemedi. Sistem meşgul olabilir.\n"
            "Birkaç saniye bekleyip tekrar dene."
        )

    if error_type == "stars_session":
        # Try to provide cached data info
        cache_hint = ""
        if user_id:
            cached = cache_db.get_json("grades", user_id)
            if cached:
                cache_hint = f"\n💡 Son bilinen veriler mevcut (cache). Temel bilgiler için tekrar sorabilirsin."
        return (
            f"⚠️ STARS bağlantısı sona ermiş.{cache_hint}\n\n"
            "Yeniden bağlanmak için /start yaz."
        )

    if error_type == "no_data":
        return (
            f"📭 {context}\n\n"
            "Farklı bir sorgu denemek ister misin?"
        )

    return "Bir sorun oluştu. Lütfen tekrar dene."

# ─── Tool Definitions (OpenAI function calling format) ────────────────────────

TOOLS: list[dict[str, Any]] = [
    # ═══ A. Teaching & Materials (5 tools) ═══
    {
        "type": "function",
        "function": {
            "name": "get_source_map",
            "description": (
                "Aktif kurstaki TÜM materyallerin haritasını çıkarır. Dosya adları, chunk sayıları, "
                "hafta/konu gruplaması, dosya özetleri. 'Bu dersi çalışmak istiyorum', 'konular ne', "
                "'materyaller ne', 'neler var', 'nelere çalışabilirim' gibi isteklerde kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_filter": {
                        "type": "string",
                        "description": "Kurs adı (opsiyonel, aktif kurs kullanılır)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": (
                "Belirli bir kaynak dosyayı OKUR. Önce hazır öğretim özetini yükler (büyük resim), "
                "sonra ilgili chunk'ları çeker (detay). Dosyayı baştan sona anlayarak gerçek öğretim "
                "yapabilirsin. 'X.pdf'i çalışayım', 'şu materyali oku', 'X dosyasını anlat' gibi "
                "isteklerde kullan. section parametresi verilirse sadece o bölümü okur."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Dosya adı (lecture_05_privacy.pdf gibi)",
                    },
                    "section": {
                        "type": "string",
                        "description": "Belirli bölüm/konu adı (opsiyonel — verilmezse tüm dosya özeti)",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "study_topic",
            "description": (
                "Belirli bir konuyu TÜM kaynaklarda arar ve öğretir. read_source'dan farkı: tek dosya "
                "değil, tüm materyallerde o konuyu arar. 'Ethics nedir', 'privacy konusunu çalışayım' "
                "gibi KONU bazlı isteklerde kullan. Dosya adı belirtilmemişse bu tool'u kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Konu",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["overview", "detailed", "deep"],
                        "description": (
                            "overview: genel bakış (top-10). "
                            "detailed: detaylı (top-25, varsayılan). "
                            "deep: kapsamlı (top-50, dosya özetleri dahil)."
                        ),
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Ders materyallerinde spesifik soru/kavram arar. KISA, odaklı sorular için. "
                "Konu çalışma değil, bilgi arama. 'X nedir?', 'Y'nin tanımı ne?' gibi sorularda kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Soru veya kavram",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Kurs filtresi (opsiyonel, aktif kurs kullanılır)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_moodle_materials",
            "description": (
                "Moodle'dan kursun materyal/kaynak listesini doğrudan Moodle API'sinden getirir. "
                "'Moodle'da ne var', 'en güncel materyaller', 'haftalık içerik' gibi isteklerde kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_filter": {
                        "type": "string",
                        "description": "Kurs adı (opsiyonel)",
                    },
                },
                "required": [],
            },
        },
    },
    # ═══ B. STARS — Academic Info (3 tools) ═══
    {
        "type": "function",
        "function": {
            "name": "get_schedule",
            "description": (
                "Ders programı. 'Bugün derslerim' → today, 'yarın ne var' → tomorrow, "
                "'haftalık' → week. SADECE sorulan dönemi getir. STARS girişi gerektirir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["today", "tomorrow", "week"],
                        "description": "today/tomorrow/week (varsayılan: today)",
                    },
                },
                "required": ["period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grades",
            "description": (
                "Not bilgileri. Spesifik ders sorulursa SADECE o dersi getir. "
                "'Notlarım' → tüm dersler. STARS girişi gerektirir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_filter": {
                        "type": "string",
                        "description": "Ders adı (opsiyonel)",
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
                "Devamsızlık bilgisi + syllabus'tan max limit + kalan hak hesabı. "
                "STARS'tan mevcut devamsızlık, RAG'den syllabus limiti çeker. "
                "Spesifik ders sorulursa SADECE o dersi getir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_filter": {
                        "type": "string",
                        "description": "Ders adı (opsiyonel)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exams",
            "description": (
                "Sınav takvimi. 'sınavlarım', 'exam schedule', 'ne zaman sınav', "
                "'midterm ne zaman', 'final tarihleri' gibi isteklerde çağır."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_filter": {
                        "type": "string",
                        "description": "Ders adı filtresi (opsiyonel)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transcript",
            "description": (
                "Transkript — alınan dersler, notlar, krediler. "
                "'transkriptim', 'transcript', 'aldığım dersler', 'GPA' gibi isteklerde çağır."
            ),
            "parameters": {"type": "object", "properties": {"_": {"type": "string", "description": "Unused"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_letter_grades",
            "description": (
                "Harf notları dönem bazlı. 'harf notlarım', 'letter grades', "
                "'dönem notlarım' gibi isteklerde çağır."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "semester_filter": {
                        "type": "string",
                        "description": "Dönem filtresi (opsiyonel)",
                    },
                },
                "required": [],
            },
        },
    },
    # ═══ C. Moodle — Assignments (1 tool) ═══
    {
        "type": "function",
        "function": {
            "name": "get_assignments",
            "description": (
                "Ödev/deadline bilgisi. "
                "Ders adı formatı: 'CTIS 363 (E. Uçar)', 'HCIV 102 (T. Durmaz)' — hoca adı parantez içinde KISALTMA olarak var. "
                "Hoca adı sorulursa → keyword olarak SOYADI kullan (ör: 'Tunahan hoca' → keyword='Durmaz'). "
                "filter: upcoming=yaklaşan, overdue=geciken, all=tümü."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "enum": ["upcoming", "overdue", "all"],
                        "description": "upcoming (varsayılan), overdue, all",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Arama: hoca SOYADI, ders kodu veya ödev adı (ör: 'Durmaz', 'HCIV', 'Video')",
                    },
                },
                "required": [],
            },
        },
    },
    # ═══ D. Mail — DAIS & AIRS (2 tools) ═══
    {
        "type": "function",
        "function": {
            "name": "get_emails",
            "description": (
                "Bilkent DAIS & AIRS mailleri. "
                "Sayı belirtilmişse (ör: 'Son 3 mail') → count kullan. "
                "'Tüm mailler' → count=20. Belirsizse → count=5 (scope=auto: önce okunmamış, yoksa son mailler). "
                "Hoca adı sorulursa → keyword olarak AD veya SOYAD kullan (from alanında 'Tunahan Durmaz' şeklinde tam ad var). "
                "Ders kodu, konu veya tarih varsa keyword kullan. "
                "Tarih: '11 şubat' → keyword='11 Şub'. "
                "Sonuç boşsa 'istersen son maillerine bakabilirim' de."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Kaç mail (varsayılan 5, max 20)",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Arama: gönderici ad/soyad, ders kodu, konu veya tarih (ör: 'Durmaz', 'CTIS', 'Video', '11 Şub')",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["auto", "recent", "unread"],
                        "description": "auto=önce okunmamış, yoksa son mailler (varsayılan). recent=sadece son mailler. unread=sadece okunmamış.",
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
                "Mailin tam içeriğini getirir. Konu, gönderici adı, ders kodu veya tarih ile arar. "
                "'Şu mailin detayını göster' dediğinde kullan. "
                "Bildirimden sonra 'detayını göster' denirse → bildirimdeki konu/göndericiyi keyword olarak kullan. "
                "Seminerin saati/detayı sorulursa → ilgili maili bu tool ile aç."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Arama terimi — konu, gönderici adı, ders kodu veya tarih (kısmi eşleşme yeterli)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Kaç mailin detayı (varsayılan 5, birden fazla eşleşme varsa hepsini getirir)",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    # ═══ E. Bot Management (3 tools) ═══
    {
        "type": "function",
        "function": {
            "name": "list_courses",
            "description": (
                "Kayıtlı tüm kursları listeler. 'List my courses', 'kurslarım', 'derslerim ne', "
                "'kurslarımı göster', 'hangi derslere kayıtlıyım' gibi isteklerde MUTLAKA çağır. "
                "Aktif kurs işaretli gösterilir."
            ),
            "parameters": {"type": "object", "properties": {"_": {"type": "string", "description": "Unused"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_active_course",
            "description": (
                "Aktif kursu değiştirir. Kısmi eşleşme destekler. "
                "Öğrenci başka bir ders hakkında konuşmak istediğinde kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Kurs adı veya kısa adı (örn: 'CTIS 256', 'POLS')",
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
            "description": "Bot istatistikleri: chunk, kurs, dosya sayısı, uptime.",
            "parameters": {"type": "object", "properties": {"_": {"type": "string", "description": "Unused"}}},
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
    """Build dynamic system prompt with 3-layer teaching methodology."""
    active_course = user_service.get_active_course(user_id)
    course_section = (
        f"Kullanıcının aktif kursu: *{active_course.display_name}*"
        if active_course
        else "Kullanıcı henüz kurs seçmemiş. Ders içeriği sorulursa 'Kurslarımı göster' demesini öner."
    )

    stars_ok = STATE.stars_client is not None and STATE.stars_client.is_authenticated(user_id)
    webmail_ok = STATE.webmail_client is not None and STATE.webmail_client.authenticated

    services = []
    if stars_ok:
        services.append("STARS: ✅ Bağlı")
    else:
        services.append("STARS: ❌ → get_schedule, get_grades, get_attendance çalışmaz")
    if webmail_ok:
        services.append("Webmail: ✅ Bağlı")
    else:
        services.append("Webmail: ❌ → get_emails, get_email_detail çalışmaz")

    now = datetime.now()
    today_tr = _DAY_NAMES_TR.get(now.weekday(), "")
    date_str = now.strftime("%d/%m/%Y %H:%M")

    student_ctx = ""
    if STATE.llm:
        student_ctx = STATE.llm._build_student_context()

    # Add student profile context
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

"Help", "yardım", "ne yapabilirsin" → GENEL yardım yanıtı ver (yapabileceklerini listele).
Önceki konuşmada mail/ders/not konuşulmuş olsa bile "help" isteğini mail/ders bağlamına BAĞLAMA.
Genel yardım: ders programı, notlar, devamsızlık, mailler, ders çalışma, ödev takibi yapabileceğini söyle.

Bildirim sonrası sorularda:
- "Detayını ver" → bildirimdeki konu/göndericiyi bul → get_email_detail(keyword=...)
- "Seminer kaçta?" → bildirimdeki seminer mailini aç → get_email_detail(keyword="Seminar")
- ASLA kullanıcıya "hangi mail?" diye sorma — konuşma geçmişini OKU

## DERS ÇALIŞMA — ÖĞRETİM YAKLAŞIMI

Sen bir ÖĞRETMENSİN, arama motoru değilsin. Materyali OKUYUP ÖĞRETİYORSUN.

Çalışma akışı:
1. "Çalışmak istiyorum" → get_source_map ile materyal haritası çıkar
2. Önerilen çalışma sırası sun (temelden ileriye)
3. Öğrenci kaynak seçince → read_source ile dosyayı OKU
   - Dosya özeti + bölüm haritası sun
4. Öğrenci bölüm seçince → read_source(section=...) ile derinleş
5. Pedagojik öğretim yap:
   - Konuyu basitçe açıkla
   - Gerçek hayat örnekleri ver
   - Düşündürücü sorular sor ("Sence bu neden önemli?")
   - İlişkili kavramları bağla
6. "Soru sor" denirse → materyalden quiz üret (tool ÇAĞIRMA, zaten biliyorsun)
7. Bölüm bitince "Devam edelim mi, başka bölüm mü?" sor

read_source kullandığında:
- Hem dosya özeti hem spesifik içerik gelir
- Özet: tüm dosyanın yapısını, bölümler arası ilişkileri gösterir
- İçerik: o anki bölümün detaylarını içerir
- Öğrenciye öğretirken her ikisini de kullan

Bölümler arası bağlantıları MUTLAKA belirt:
- "Bu konu Bölüm 3'teki GDPR detaylarıyla ilişkili"
- "Az önce gördüğümüz privacy kavramı burada uygulanıyor"

Konu bazlı çalışma (dosya adı belirtilmemişse):
- study_topic kullan — tüm kaynaklarda konuyu arar
- depth: overview → detailed → deep adım adım derinleş

## AGENTIC RAG — İTERATİF ARAMA (KRİTİK)
Tek bir RAG sorgusuyla yetinme. Karmaşık sorularda İTERATİF çalış:
- İlk sonuç yetersizse → farklı anahtar kelimeyle tekrar ara
- Karşılaştırma sorusu ("A vs B") → her kavramı AYRI ara, sonra sentezle
- Geniş konu ("güvenlik konuları") → önce study_topic(overview), sonra read_source(section) ile derinleş
- Sonuçlarda referans edilen başka konu/bölüm varsa → o bölümü de çek
- Birden fazla dosyada bilgi varsa → hepsinden topla, çapraz referans yap
Örnek:
  "Privacy ve encryption farkı" →
  1. study_topic("privacy") → chunk'lar
  2. study_topic("encryption") → chunk'lar
  3. LLM: iki sonucu sentezle → karşılaştırmalı yanıt

## NOT VE DEVAMSIZLIK
- Spesifik ders sorulursa → SADECE o ders
- Genel sorulursa → tüm dersler
- Devamsızlık limitine yaklaşıyorsa → ⚠️ UYAR
- NOT GÖSTERİMİ: Tool'dan gelen veriyi OLDUĞU GİBİ göster. Her ödevi/sınavı tek tek listele, ÖZETLEME.
  YANLIŞ: "Tüm ödevler tam puan 1/1"
  DOĞRU: "Homework 01: 1/1, Homework 02: 1/1, Homework 03: 1/1..."
- Mail detayı gösterilirken de tam içeriği göster, ÖZETLEME.

## MAİL — DAIS & AIRS
- Sayı belirtilmişse ("Son 3 mail", "5 mailimi göster") → DOĞRUDAN get_emails(count=N) çağır
- "Tüm mailleri göster", "hepsi", "bütün" → get_emails(count=20) çağır (SORU SORMA, hemen getir)
- Sayısız isteklerde ("Maillerimi göster") → count=5 varsayılan kullan, soru SORMA
- Hoca adı, ders kodu, konu VEYA TARİH: keyword parametresi kullan (gönderici, konu, kaynak, tarih hepsinde arar)
- "EDEB maili" → keyword="EDEB", "Adem hoca" → keyword="Adem"
- "11 şubat maili" → keyword="11 Şub" (tarih formatı: "GG Ay_kısaltma", ör: "25 Şub", "11 Oca")
- "Serhat hoca 11 şubat" → İKİ keyword birleşemez, ÖNCE keyword="Serhat" ile çek, sonra tarih sonuçlardan filtrele
- Mail detayı isterse: get_email_detail(keyword=...) — konu, hoca adı, ders kodu veya tarih ile arar
- Sonuç boşsa: "Yakın zamanda yok, istersen son maillerini gösterebilirim"
- Mail listesi gösterdikten sonra "hepsini göster", "tümünü göster", "hepsini aç" → TÜM maillerin detayını get_email_detail ile sırayla göster, SORU SORMA
- "Detayını göster" + numara/konu → o mailin detayını aç

## BİLDİRİM BAĞLAMI (KRİTİK)
Bot bildirim gönderdiğinde (📧 Yeni Mail, ⚠️ Devamsızlık vb.) bu bildirim konuşma geçmişinde kalır.
- "Mailin detayını ver/göster" → son bildirimdeki konu/göndericiyi keyword olarak kullan
- "Seminer kaçta?" → son bildirimde seminer maili varsa get_email_detail ile aç (get_schedule DEĞİL)
- "Bu ne?" → son bildirimin içeriğini açıkla
- ASLA "hangi mail?" diye sorma — konuşma geçmişinde bildirim varsa onu kullan

Mail sonuçlarını AŞAĞIDAKİ FORMATTA göster (her mail için):
📧 *Konu başlığı*
  👤 Gönderen adı
  📅 Tarih
  💬 Kısa özet (1-2 cümle)

Mailler arasında boş satır bırak. Özetleme YAPMA, her maili ayrı ayrı göster.

## YANIT KALİTE KONTROLÜ (her yanıtta uygula)
Yanıtını göndermeden önce kontrol et:
1. Soruya doğrudan cevap veriyor musun? Konu dışına çıkma
2. Doğru dilde mi? (Son mesajın dili)
3. Tool sonucu boş geldiyse → uydurma, açıkça belirt
4. Gereksiz bilgi var mı? → Kısa ve öz ol (max 3-4 paragraf)
5. Kaynak gerekiyorsa → 📖 [dosya] etiketi ekle
6. Sayısal veri (not, devamsızlık) → doğrudan tool sonucunu kullan, yuvarlama YAPMA

## FORMAT KURALLARI
1. Telegram Markdown: *bold*, _italic_, `code`
2. Veri sorguları (not, program, ödev) → SADECE istenen veriyi ver
3. RAG sonuçlarını kullanırken 📖 [dosya_adı] kaynak etiketi ekle
4. Tool sonuçlarını doğal dille sun, JSON/teknik format GÖSTERME (mail hariç — mailler yapılandırılmış formatta gösterilmeli)
5. Tool sonucu boş gelirse nazikçe bildir

## TEKNİK TERİM YASAĞI
ASLA kullanma: chunk, RAG, retrieval, embedding, vector, tool, function call, token, pipeline, LLM, model, API, context window, top-k
Bunlar yerine: materyal, kaynak, bilgi, arama, içerik

## FOLLOW-UP ÖNERİLERİ (KRİTİK — KULLANICI DENEYİMİ)
Her yanıtın sonunda BAĞLAMSAL bir öneri ekle (zorunlu değil ama faydalı):

Veri gösterdikten sonra (notlar, program, ödev):
→ "Başka bir şey görmek ister misin? (devamsızlık, transkript vb.)"

Ders çalışma sonrası:
→ "Devam edelim mi, yoksa başka bir konuya mı geçelim?"

Sorun/düşük not görürsen:
→ "Bu dersle ilgili materyallere bakabilir veya çalışma planı yapabiliriz."

Mail gösterdikten sonra:
→ "Detayını görmek istersen numarasını veya konusunu yaz."

ÖNEMLİ: Önerileri KISA tut (1 cümle), abartma. Soru sor ama baskıcı olma.

## SON KURAL — DİL (BU KURALI ASLA İHLAL ETME)
Kullanıcının SON mesajı İngilizce ise yanıtın %100 İngilizce olmalı.
Kullanıcının SON mesajı Türkçe ise yanıtın %100 Türkçe olmalı.
Önceki mesajların dili ÖNEMSİZ — sadece SON mesaja bak."""


# ─── Tool Availability Filter ────────────────────────────────────────────────


def _get_available_tools(user_id: int) -> list[dict[str, Any]]:
    """Return all tools — unavailable services handled by tool handlers."""
    return list(TOOLS)


# ─── LLM Call with Tools ─────────────────────────────────────────────────────


async def _call_llm_with_tools(
    messages: list[dict[str, Any]],
    system_prompt: str,
    tools: list[dict[str, Any]],
    use_complex_model: bool = False,
) -> Any:
    """Call LLM with function calling via LiteLLM Router (latency-based)."""
    router = _get_fast_router()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    # Tool-selection calls need short output; final responses need more
    max_tokens = 1024 if tools else 4096
    kwargs: dict[str, Any] = {
        "model": "fast",  # LiteLLM picks fastest available model
        "messages": full_messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = True  # GPT, Gemini, DeepSeek all support this

    try:
        response = await router.acompletion(**kwargs)
        logger.debug("LiteLLM response from: %s", response.model)
        return response.choices[0].message
    except Exception as exc:
        logger.error("LiteLLM call failed: %s", exc)
        return None


# ─── Streaming ───────────────────────────────────────────────────────────────

_STREAM_EDIT_INTERVAL = 1.0  # seconds between Telegram message edits


async def _stream_final_response(
    messages: list[dict[str, Any]],
    system_prompt: str,
    message: Message,
) -> str:
    """Stream LLM response directly to Telegram via progressive message edits."""
    router = _get_fast_router()

    full_messages = [{"role": "system", "content": system_prompt}] + messages
    kwargs: dict[str, Any] = {
        "model": "fast",
        "messages": full_messages,
        "max_tokens": 4096,
        "stream": True,
    }

    try:
        stream = await router.acompletion(**kwargs)
    except Exception as exc:
        logger.error("LiteLLM streaming failed: %s", exc)
        return ""

    accumulated = ""
    sent_msg = None
    last_edit = 0.0

    try:
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                accumulated += delta.content

            now = time.monotonic()
            # Edit message periodically (not on every token)
            if accumulated and (now - last_edit) >= _STREAM_EDIT_INTERVAL:
                try:
                    if sent_msg is None:
                        sent_msg = await message.reply_text(accumulated, parse_mode=None)
                    else:
                        await sent_msg.edit_text(accumulated, parse_mode=None)
                    last_edit = now
                except TelegramError:
                    pass  # edit rate limit or unchanged text — ignore

        # Final edit with Markdown formatting
        if accumulated and sent_msg is not None:
            try:
                await sent_msg.edit_text(accumulated, parse_mode="Markdown")
            except TelegramError:
                try:
                    await sent_msg.edit_text(accumulated, parse_mode=None)
                except TelegramError:
                    pass
        elif accumulated and sent_msg is None:
            # Stream was too fast, never sent — send now
            await message.reply_text(accumulated, parse_mode="Markdown")
    except Exception as exc:
        logger.warning("Streaming failed, falling back: %s", exc)
        if not accumulated:
            return ""  # caller will use non-streaming fallback

    return accumulated


async def _send_progressive(message: Message, text: str) -> None:
    """
    Send pre-generated text progressively for perceived speed.
    Simulates streaming by revealing text in chunks.
    """
    if not text:
        return

    # For short messages, just send directly
    if len(text) < 100:
        await message.reply_text(text, parse_mode="Markdown")
        return

    # Progressive reveal: start with first sentence/chunk, expand
    chunk_size = max(50, len(text) // 5)  # ~5 updates total
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
                    pass  # Rate limit or unchanged — continue
            await asyncio.sleep(0.15)  # Brief pause between updates

        # Final edit with Markdown
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
        # Fallback: send full text
        if sent_msg is None:
            await message.reply_text(text, parse_mode="Markdown")


# ─── Tool Handlers ───────────────────────────────────────────────────────────


def _resolve_course(args: dict, user_id: int, key: str = "course_filter") -> str | None:
    """Resolve course name from args or active course."""
    name = args.get(key)
    if not name:
        active = user_service.get_active_course(user_id)
        name = active.course_id if active else None
    return name


async def _tool_get_source_map(args: dict, user_id: int) -> str:
    """KATMAN 1 — Metadata aggregation + KATMAN 2 summaries."""
    course_name = _resolve_course(args, user_id)
    if not course_name:
        return "Aktif kurs seçili değil. Önce bir kurs seç."

    store = STATE.vector_store
    if store is None:
        return "Materyal veritabanı hazır değil."

    try:
        files = await asyncio.to_thread(store.get_files_for_course, course_name)
    except (AttributeError, RuntimeError, ValueError) as exc:
        logger.error("Source map failed: %s", exc, exc_info=True)
        return f"Materyal haritası alınamadı: {exc}"

    if not files:
        return f"'{course_name}' kursu için yüklü materyal bulunamadı."

    from bot.services.summary_service import load_source_summary

    lines = []
    total_chunks = 0
    for f in files:
        filename = f.get("filename", "")
        chunk_count = f.get("chunk_count", 0)
        total_chunks += chunk_count
        section = f.get("section", "")

        line = f"📄 {filename} ({chunk_count} parça)"
        if section:
            line += f" — {section}"

        # KATMAN 2: Add summary if available
        summary = load_source_summary(filename, course_name)
        if summary and not summary.get("fallback"):
            overview = summary.get("overview", "")
            if overview:
                line += f"\n   Özet: {overview[:200]}"
            sections = summary.get("sections", [])
            if sections:
                sec_names = [s.get("title", "") for s in sections[:5] if s.get("title")]
                if sec_names:
                    line += f"\n   Bölümler: {', '.join(sec_names)}"
            difficulty = summary.get("difficulty", "")
            if difficulty:
                line += f"\n   Seviye: {difficulty}"

        lines.append(line)

    study_order = ""
    # Check first file's summary for study order
    if files:
        first_summary = load_source_summary(files[0].get("filename", ""), course_name)
        if first_summary:
            study_order = first_summary.get("suggested_study_order", "")

    header = f"📚 {course_name} — {len(files)} dosya, {total_chunks} toplam parça\n"
    result = header + "\n\n".join(lines)
    if study_order:
        result += f"\n\n💡 Önerilen çalışma sırası: {study_order}"

    return result


async def _tool_read_source(args: dict, user_id: int) -> str:
    """KATMAN 2 + KATMAN 3 birleşik okuma — en kritik tool."""
    source = args.get("source", "")
    if not source:
        return "Dosya adı belirtilmedi."

    section = args.get("section")
    course_name = _resolve_course(args, user_id)

    store = STATE.vector_store
    if store is None:
        return "Materyal veritabanı hazır değil."

    # KATMAN 2: Load pre-generated summary
    from bot.services.summary_service import load_source_summary

    summary = load_source_summary(source, course_name or "")

    if summary and not section:
        # Return full summary — file introduction
        overview = summary.get("overview", "")
        sections = summary.get("sections", [])
        cross_refs = summary.get("cross_references", [])
        study_order = summary.get("suggested_study_order", "")
        difficulty = summary.get("difficulty", "")

        parts = [f"📖 *{source}*\n"]
        if overview:
            parts.append(overview)
        if difficulty:
            parts.append(f"Seviye: {difficulty}")
        if sections:
            parts.append("\n*Bölümler:*")
            for i, s in enumerate(sections, 1):
                title = s.get("title", f"Bölüm {i}")
                sec_summary = s.get("summary", "")
                concepts = s.get("key_concepts", [])
                parts.append(f"\n{i}. *{title}*")
                if sec_summary:
                    parts.append(f"   {sec_summary[:200]}")
                if concepts:
                    parts.append(f"   Kavramlar: {', '.join(concepts[:6])}")
        if cross_refs:
            parts.append("\n*Bölümler arası bağlantılar:*")
            for ref in cross_refs[:5]:
                parts.append(f"  → {ref}")
        if study_order:
            parts.append(f"\n💡 {study_order}")
        parts.append("\nHangi bölümle başlamak istersin?")

        return "\n".join(parts)

    # KATMAN 3: Get chunks
    if section:
        # Section-specific: search within the file
        chunks = await asyncio.to_thread(store.get_file_chunks, source, 0)
        if not chunks:
            return f"'{source}' dosyası bulunamadı."

        # Filter by section keyword
        sec_lower = section.lower()
        filtered = [c for c in chunks if sec_lower in c.get("text", "").lower()]
        if not filtered:
            # Fallback: return all chunks (section not found as keyword)
            filtered = chunks[:30]

        chunk_texts = "\n\n---\n\n".join(
            f"[Parça {c.get('chunk_index', 0) + 1}]\n{c.get('text', '')}"
            for c in filtered[:30]
            if c.get("text", "").strip()
        )

        # Prepend summary if available
        result = ""
        if summary:
            result = f"DOSYA ÖZETİ:\n{json.dumps(summary, ensure_ascii=False)}\n\nBÖLÜM DETAYI:\n"
        result += chunk_texts
        return result

    # No summary, no section: return all chunks (fallback)
    chunks = await asyncio.to_thread(store.get_file_chunks, source, 0)
    if not chunks:
        return f"'{source}' dosyası bulunamadı. get_source_map ile doğru dosya adını kontrol et."

    if len(chunks) > 80:
        return f"Dosya çok büyük ({len(chunks)} parça). Lütfen bir bölüm belirt veya get_source_map ile bölümlere bak."

    parts = [f"📄 *{source}* — {len(chunks)} parça\n"]
    for c in chunks[:40]:
        text = c.get("text", "")
        idx = c.get("chunk_index", 0)
        if text.strip():
            parts.append(f"[Parça {idx + 1}]\n{text}")

    return "\n\n---\n\n".join(parts)


async def _tool_study_topic(args: dict, user_id: int) -> str:
    """Cross-source topic search with configurable depth."""
    topic = args.get("topic", "")
    if not topic:
        return "Konu belirtilmedi."

    course_name = _resolve_course(args, user_id)

    store = STATE.vector_store
    if store is None:
        return "Materyal veritabanı hazır değil."

    depth = args.get("depth", "detailed")
    top_k = {"overview": 10, "detailed": 25, "deep": 50}.get(depth, 25)

    results = await asyncio.to_thread(store.hybrid_search, topic, top_k, course_name)

    if not results and course_name:
        results = await asyncio.to_thread(store.hybrid_search, topic, top_k, None)

    if not results:
        return f"'{topic}' konusuyla ilgili materyal bulunamadı."

    parts = []
    seen_files: set[str] = set()

    if depth == "deep":
        from bot.services.summary_service import load_source_summary

    for r in results:
        meta = r.get("metadata", {})
        filename = meta.get("filename", "bilinmeyen")
        text = r.get("text", "")
        dist = r.get("distance", 0)
        if len(text.strip()) < 50:
            continue

        # Deep mode: add file summary header once per file
        if depth == "deep" and filename not in seen_files:
            seen_files.add(filename)
            summary = load_source_summary(filename, course_name or "")
            if summary and not summary.get("fallback"):
                overview = summary.get("overview", "")
                if overview:
                    parts.append(f"[📄 {filename} — Dosya Özeti: {overview[:200]}]")

        parts.append(f"[📖 {filename} | Skor: {1 - dist:.2f}]\n{text}")

    return "\n\n---\n\n".join(parts) if parts else f"'{topic}' ile ilgili yeterli materyal bulunamadı."


async def _tool_rag_search(args: dict, user_id: int) -> str:
    """Standard RAG search for specific questions."""
    query = args.get("query", "")
    if not query:
        return "Arama sorgusu belirtilmedi."

    course_name = args.get("course_name")
    if not course_name:
        course_name = _resolve_course(args, user_id)

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


async def _tool_get_moodle_materials(args: dict, user_id: int) -> str:
    """Get materials directly from Moodle API (not vector store)."""
    moodle = STATE.moodle
    if moodle is None:
        return "Moodle bağlantısı hazır değil."

    course_name = _resolve_course(args, user_id)

    try:
        courses = await asyncio.to_thread(moodle.get_courses)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Moodle courses fetch failed: %s", exc, exc_info=True)
        return f"Moodle'a bağlanılamadı: {exc}"

    # Find matching course
    target = None
    if course_name:
        cn_lower = course_name.lower()
        for c in courses:
            if cn_lower in c.fullname.lower() or cn_lower in c.shortname.lower():
                target = c
                break

    if not target and courses:
        target = courses[0]

    if not target:
        return "Kurs bulunamadı."

    try:
        text = await asyncio.to_thread(moodle.get_course_topics_text, target)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Moodle topics fetch failed: %s", exc, exc_info=True)
        return f"Moodle içeriği alınamadı: {exc}"

    if not text:
        return f"'{target.fullname}' kursunda içerik bulunamadı."

    # Truncate if too long
    if len(text) > 3000:
        text = text[:3000] + "\n\n[... kısaltıldı ...]"

    return text


async def _tool_get_schedule(args: dict, user_id: int) -> str:
    """Get schedule from cache (updated every 1 min by background sync)."""
    # Cache-first: background job updates cache every 1 minute
    schedule = cache_db.get_json("schedule", user_id)

    # 3. No data at all
    if not schedule:
        return "Ders programı bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

    period = args.get("period", "today")

    if period in ("today", "tomorrow"):
        now = datetime.now()
        target = now + timedelta(days=1) if period == "tomorrow" else now
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
    """Get grades from cache (updated every 1 min by background sync)."""
    # Cache-first: background job updates cache every 1 minute
    grades = cache_db.get_json("grades", user_id)

    if not grades:
        return "Not bilgisi bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

    course_filter = args.get("course_filter", "")
    if course_filter:
        cf_lower = course_filter.lower()
        grades = [g for g in grades if cf_lower in g.get("course", "").lower()]
        if not grades:
            return f"'{course_filter}' ile eşleşen kurs notu bulunamadı."

    lines = []
    for course in grades:
        cname = course.get("course", "Bilinmeyen")
        assessments = course.get("assessments", [])
        if not assessments:
            lines.append(f"📚 {cname}: Henüz not girilmemiş")
            continue
        lines.append(f"📚 {cname}:")
        for a in assessments:
            name = a.get("name", "")
            grade = a.get("grade", "")
            atype = a.get("type", "")
            date = a.get("date", "")
            weight = a.get("weight", "")
            extras = []
            if atype:
                extras.append(atype)
            if date:
                extras.append(date)
            if weight:
                extras.append(f"Ağırlık: {weight}")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"  • {name}: {grade}{extra_str}")

    return "\n".join(lines)


async def _tool_get_attendance(args: dict, user_id: int) -> str:
    """Get attendance from STARS + syllabus limits from cache."""
    # Cache-first: background job updates cache every 1 minute
    attendance = cache_db.get_json("attendance", user_id)

    if not attendance:
        return "Devamsızlık bilgisi bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

    course_filter = args.get("course_filter", "")
    if course_filter:
        cf_lower = course_filter.lower()
        attendance = [a for a in attendance if cf_lower in a.get("course", "").lower()]
        if not attendance:
            return f"'{course_filter}' ile eşleşen kurs devamsızlığı bulunamadı."

    # Load cached syllabus limits (populated by notification_service daily job)
    # Format: {course_name: max_hours} — 0 means "checked but not found"
    syllabus_limits: dict[str, int] = cache_db.get_json("syllabus_limits", user_id) or {}

    def _calc_missed_hours(records: list[dict]) -> int:
        """Calculate actual missed hours from STARS raw data (e.g., '0/ 2' = 2h missed)."""
        total_missed = 0
        for r in records:
            raw = r.get("raw", "")
            if "/" in raw:
                try:
                    parts = raw.replace(" ", "").split("/")
                    attended_h = int(parts[0])
                    total_h = int(parts[1])
                    total_missed += total_h - attended_h
                except (ValueError, IndexError):
                    pass
        return total_missed

    lines = []
    for cd in attendance:
        cname = cd.get("course", "Bilinmeyen")
        records = cd.get("records", [])
        ratio = cd.get("ratio", "")

        total_sessions = len(records)
        absent_sessions = sum(1 for r in records if not r.get("attended", True))
        hours_absent = _calc_missed_hours(records)

        line = f"📚 {cname}:"
        if ratio:
            line += f" Devam: {ratio}"
        line += f" ({hours_absent} saat devamsız / {absent_sessions} ders)"

        # Check cached syllabus limit (0 = not found sentinel)
        max_hours = syllabus_limits.get(cname)
        if max_hours and max_hours > 0:
            remaining_hours = max(0, max_hours - hours_absent)
            line += f"\n  📋 Syllabus limiti: max {max_hours} saat"
            if remaining_hours > 0:
                line += f" → {remaining_hours} saat hakkın kaldı ✅"
            else:
                line += f" → ⚠️ LİMİT AŞILDI! ({hours_absent - max_hours} saat fazla)"
        else:
            # Default warning if no syllabus found
            try:
                ratio_num = float(ratio.replace("%", "")) if ratio else 100
                if ratio_num < 85:
                    line += "\n  ⚠️ Dikkat: Devamsızlık limiti %20'ye yaklaşıyor!"
            except (ValueError, AttributeError):
                pass

        lines.append(line)

    return "\n".join(lines)


async def _tool_get_exams(args: dict, user_id: int) -> str:
    """Get exam schedule from cache (updated every 1 min by background sync)."""
    # Cache-first: background job updates cache every 1 minute
    exams = cache_db.get_json("exams", user_id)

    if not exams:
        return "Sınav takvimi bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

    course_filter = args.get("course_filter", "")
    if course_filter:
        cf_lower = course_filter.lower()
        exams = [e for e in exams if cf_lower in e.get("course", "").lower()]
        if not exams:
            return f"'{course_filter}' ile eşleşen sınav bulunamadı."

    lines = []
    for e in exams:
        course = e.get("course", "")
        exam_name = e.get("exam_name", "Sınav")
        date = e.get("date", "")
        start_time = e.get("start_time", "")
        time_block = e.get("time_block", "")
        time_remaining = e.get("time_remaining", "")

        line = f"📝 *{course}* — {exam_name}"
        if date:
            line += f"\n  📅 {date}"
            if start_time:
                line += f", {start_time}"
            elif time_block:
                line += f", {time_block}"
        if time_remaining:
            line += f"\n  ⏳ {time_remaining}"
        lines.append(line)

    return "\n\n".join(lines)


async def _tool_get_transcript(args: dict, user_id: int) -> str:
    """Get academic transcript from STARS with cache fallback."""
    stars = STATE.stars_client
    transcript = None

    if stars and stars.is_authenticated(user_id):
        try:
            transcript = await asyncio.to_thread(stars.get_transcript, user_id)
        except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("Transcript live fetch failed, trying cache: %s", exc)

    if not transcript:
        transcript = cache_db.get_json("transcript", user_id)
        if transcript:
            logger.info("Transcript served from cache for user %d", user_id)

    if not transcript:
        return "Transkript bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

    lines = ["📋 *Transkript*\n"]
    current_semester = ""
    for entry in transcript:
        semester = entry.get("semester", "")
        if semester != current_semester:
            current_semester = semester
            lines.append(f"\n*{semester}*")

        code = entry.get("code", "")
        name = entry.get("name", "")
        grade = entry.get("grade", "")
        credits = entry.get("credits", "")

        line = f"  • {code} {name}"
        if grade:
            line += f" — {grade}"
        if credits:
            line += f" ({credits} kr)"
        lines.append(line)

    return "\n".join(lines)


async def _tool_get_letter_grades(args: dict, user_id: int) -> str:
    """Get letter grades from STARS with cache fallback."""
    stars = STATE.stars_client
    letter_grades = None

    if stars and stars.is_authenticated(user_id):
        try:
            letter_grades = await asyncio.to_thread(stars.get_letter_grades, user_id)
        except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("Letter grades live fetch failed, trying cache: %s", exc)

    if not letter_grades:
        letter_grades = cache_db.get_json("letter_grades", user_id)
        if letter_grades:
            logger.info("Letter grades served from cache for user %d", user_id)

    if not letter_grades:
        return "Harf notları bulunamadı. STARS session süresi dolmuş olabilir — /start ile tekrar giriş yap."

    semester_filter = args.get("semester_filter", "")

    lines = ["📊 *Harf Notları*\n"]
    for sem in letter_grades:
        semester = sem.get("semester", "")
        if semester_filter and semester_filter.lower() not in semester.lower():
            continue

        lines.append(f"\n*{semester}*")
        for c in sem.get("courses", []):
            code = c.get("code", "")
            name = c.get("name", "")
            grade = c.get("grade", "")
            lines.append(f"  • {code} {name} — {grade}")

    if len(lines) == 1:
        return f"'{semester_filter}' dönemi ile eşleşen not bulunamadı."

    return "\n".join(lines)


async def _tool_get_assignments(args: dict, user_id: int) -> str:
    """Get Moodle assignments with optional filtering."""
    moodle = STATE.moodle
    if moodle is None:
        return "Moodle bağlantısı hazır değil."

    filter_mode = args.get("filter", "upcoming")
    keyword = args.get("keyword", "").lower().strip()
    now_ts = time.time()

    # Strip common noise words that don't help search
    NOISE_WORDS = ["hoca", "hocanın", "hocam", "öğretmen", "prof", "dersi", "dersinin", "ödevi", "ödevini"]
    if keyword:
        tokens = keyword.split()
        keyword = " ".join(w for w in tokens if w not in NOISE_WORDS)

    try:
        if filter_mode == "all" or keyword:
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

    # Keyword filtering — partial match on course name or assignment name
    if keyword and assignments:
        search_tokens = keyword.split()
        def matches(a) -> bool:
            searchable = f"{a.course_name} {a.name}".lower()
            return any(tok in searchable for tok in search_tokens)
        assignments = [a for a in assignments if matches(a)]

    if not assignments:
        labels = {"upcoming": "Yaklaşan", "overdue": "Süresi geçmiş", "all": "Hiç"}
        if keyword:
            return f"'{keyword}' ile eşleşen ödev bulunamadı."
        return f"{labels.get(filter_mode, 'Yaklaşan')} ödev bulunamadı."

    lines = []
    for a in assignments:
        status = "✅ Teslim edildi" if a.submitted else "⏳ Teslim edilmedi"
        # Format due_date from Unix timestamp to readable date
        if hasattr(a, "due_date") and a.due_date and a.due_date > 0:
            due_dt = datetime.fromtimestamp(a.due_date)
            due = due_dt.strftime("%d/%m/%Y %H:%M")
        else:
            due = "Son tarih yok"
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        line = f"• {a.course_name} — {a.name}\n  Tarih: {due} | {status}"
        if remaining and not a.submitted:
            line += f" | Kalan: {remaining}"
        if filter_mode == "overdue":
            line += " | ⚠️ Süresi geçmiş!"
        lines.append(line)

    return "\n".join(lines)


async def _tool_get_emails(args: dict, user_id: int) -> str:
    """Get AIRS/DAIS emails from SQLite cache (instant, no IMAP)."""
    # Check if cache is populated
    email_count = cache_db.get_email_count()
    if email_count == 0:
        return "Mail cache henüz doldurulmadı. Birkaç saniye bekleyip tekrar dene."

    count = args.get("count", 5)
    scope = args.get("scope", "auto")  # auto: unread first, then recent
    keyword = args.get("keyword", "") or args.get("sender_filter", "")

    # Fetch from SQLite cache — instant!
    if keyword:
        # Direct DB search — no client-side filtering needed
        mails = cache_db.search_emails(keyword, limit=max(count, 50))
    elif scope == "unread":
        mails = cache_db.get_unread_emails()
    elif scope == "auto":
        mails = cache_db.get_unread_emails()
        if not mails:
            mails = cache_db.get_emails(count) or []
    else:
        mails = cache_db.get_emails(count) or []

    mails = mails[:count]

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
    """Get full content of matching emails from SQLite cache."""
    keyword = args.get("keyword", "") or args.get("email_subject", "")
    if not keyword:
        return "Mail arama terimi belirtilmedi."

    count = args.get("count", 5)

    # Direct DB search — same as get_emails
    mails = cache_db.search_emails(keyword, limit=count)
    if not mails:
        return f"'{keyword}' ile eşleşen mail bulunamadı."

    lines = []
    for m in mails:
        body = m.get("body_full") or m.get("body_preview", "")
        lines.append(
            f"📧 *{m.get('subject', 'Konusuz')}*\n"
            f"Kimden: {m.get('from', '')}\n"
            f"Tarih: {m.get('date', '')}\n\n"
            f"{body}"
        )

    return "\n\n---\n\n".join(lines)


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

    # Count source summaries
    from bot.services.summary_service import list_summaries

    summaries = list_summaries()

    return (
        f"Toplam chunk: {stats.get('total_chunks', 0)}\n"
        f"Kurs sayısı: {stats.get('unique_courses', 0)}\n"
        f"Dosya sayısı: {stats.get('unique_files', 0)}\n"
        f"Kaynak özetleri: {len(summaries)}\n"
        f"Aktif kullanıcı: {len(STATE.active_courses)}\n"
        f"Uptime: {hours}s {minutes}dk {seconds}sn\n"
        f"Versiyon: {STATE.startup_version}"
    )


# ─── Tool Dispatcher ─────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "get_source_map": _tool_get_source_map,
    "read_source": _tool_read_source,
    "study_topic": _tool_study_topic,
    "rag_search": _tool_rag_search,
    "get_moodle_materials": _tool_get_moodle_materials,
    "get_schedule": _tool_get_schedule,
    "get_grades": _tool_get_grades,
    "get_attendance": _tool_get_attendance,
    "get_exams": _tool_get_exams,
    "get_transcript": _tool_get_transcript,
    "get_letter_grades": _tool_get_letter_grades,
    "get_assignments": _tool_get_assignments,
    "get_emails": _tool_get_emails,
    "get_email_detail": _tool_get_email_detail,
    "list_courses": _tool_list_courses,
    "set_active_course": _tool_set_active_course,
    "get_stats": _tool_get_stats,
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
            result = f"[{fn_name}] şu anda çalışmıyor ({type(exc).__name__}). Alternatif bilgi kaynağı dene veya kullanıcıya bildir."

    logger.info(
        "Tool executed: %s (result_len=%d)",
        fn_name,
        len(result),
        extra={"tool": fn_name, "user_id": user_id},
    )

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }


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
    # ═══ Instant response bypass (no LLM call) ═══
    instant = _check_instant_response(user_text)
    if instant:
        user_service.add_conversation_turn(user_id, "user", user_text)
        user_service.add_conversation_turn(user_id, "assistant", instant)
        logger.info("Instant response for: %s", user_text[:30])
        return instant

    if STATE.llm is None:
        return "Sistem henüz hazır değil. Lütfen birazdan tekrar deneyin."

    t_start = time.time()
    system_prompt = _build_system_prompt(user_id)

    # Detect language of current message and inject directive
    lang = _detect_language(user_text)
    if lang == "en":
        system_prompt += "\n\n[LANGUAGE OVERRIDE] The user's current message is in ENGLISH. You MUST respond entirely in English."

    available_tools = _get_available_tools(user_id)

    history = user_service.get_conversation_history(user_id)
    messages: list[dict[str, Any]] = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    # Detect if query needs higher-quality model
    is_complex = _is_complex_query(user_text)
    tools_used: list[str] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        # Refresh typing indicator before each LLM call
        if message:
            try:
                await message.chat.send_action("typing")
            except TelegramError:
                pass

        try:
            t_llm = time.time()
            # Use complex model for final response if query is complex or multiple tools used
            use_complex = is_complex or len(tools_used) >= 2
            response_msg = await _call_llm_with_tools(
                messages, system_prompt, available_tools, use_complex_model=use_complex
            )
            logger.info("LLM call (iter %d): %.2fs%s", iteration + 1, time.time() - t_llm,
                       " [complex]" if use_complex else "")
        except Exception as exc:
            logger.error("LLM call failed (iteration %d): %s", iteration, exc, exc_info=True)
            return _smart_error("llm_failed", f"Hata: {type(exc).__name__}")

        if response_msg is None:
            return _smart_error("llm_null")

        tool_calls = getattr(response_msg, "tool_calls", None)
        if not tool_calls:
            # LLM returned final text — send progressively for perceived speed
            final_text = _sanitize_llm_output(response_msg.content or "")

            if message and final_text:
                # Progressive send: show text appearing quickly
                await _send_progressive(message, final_text)
                user_service.add_conversation_turn(user_id, "user", user_text)
                user_service.add_conversation_turn(user_id, "assistant", final_text)
                # Track query for profile
                active = user_service.get_active_course(user_id)
                cache_db.track_query(user_id, course=active.course_id if active else None, topic=_extract_topic(user_text))
                logger.info("Total response time: %.2fs (progressive)", time.time() - t_start)
                return ""  # Already sent

            # No message object — return text directly
            user_service.add_conversation_turn(user_id, "user", user_text)
            user_service.add_conversation_turn(user_id, "assistant", final_text)
            # Track query for profile
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
        tools_used.extend(tool_names)  # Track for complexity detection

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
        # Try streaming the final response
        if message:
            t_stream = time.time()
            final_text = await _stream_final_response(messages, system_prompt, message)
            if final_text:
                logger.info("Streaming response: %.2fs", time.time() - t_stream)
                logger.info("Total response time: %.2fs (streamed)", time.time() - t_start)
                user_service.add_conversation_turn(user_id, "user", user_text)
                user_service.add_conversation_turn(user_id, "assistant", final_text)
                return ""  # already sent via streaming

        # Non-streaming fallback
        response_msg = await _call_llm_with_tools(messages, system_prompt, [])
        final_text = _sanitize_llm_output(response_msg.content) if response_msg else "Yanıt üretilemedi."
    except Exception:
        final_text = "İşlem zaman aşımına uğradı. Lütfen tekrar deneyin."

    user_service.add_conversation_turn(user_id, "user", user_text)
    user_service.add_conversation_turn(user_id, "assistant", final_text)

    # Track query for profile building
    active = user_service.get_active_course(user_id)
    cache_db.track_query(
        user_id,
        course=active.course_id if active else None,
        topic=_extract_topic(user_text),
    )

    logger.info("Total response time: %.2fs (with tools)", time.time() - t_start)
    return final_text
