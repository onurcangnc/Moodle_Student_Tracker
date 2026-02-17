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
import re
import time
from datetime import datetime, timedelta
from typing import Any

from bot.services import user_service
from bot.state import STATE

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

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
                    "offset": {
                        "type": "integer",
                        "description": "Başlangıç parça indeksi, sayfalama için (varsayılan 0)",
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
                "Devamsızlık bilgisi. Spesifik ders sorulursa SADECE o dersi getir. "
                "Limite yaklaşıyorsa UYAR. STARS girişi gerektirir."
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
    # ═══ C. Moodle — Assignments (1 tool) ═══
    {
        "type": "function",
        "function": {
            "name": "get_assignments",
            "description": (
                "Ödev/deadline. upcoming=yaklaşan, overdue=geciken, all=tümü."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "enum": ["upcoming", "overdue", "all"],
                        "description": "upcoming (varsayılan), overdue, all",
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
                "Bilkent DAIS & AIRS mailleri. KRİTİK KURALLAR: "
                "(1) Mail sorulursa varsayılan count=5 ile çağır. Kullanıcı farklı sayı isterse o sayıyı kullan. "
                "(2) Hoca adıyla sorulursa sender_filter kullan. "
                "(3) Sonuç boşsa 'Yakın zamanda yok, istersen son maillerini gösterebilirim' de."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Kaç mail (varsayılan 5, max 20)",
                    },
                    "sender_filter": {
                        "type": "string",
                        "description": "Gönderici adı filtresi",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["recent", "unread"],
                        "description": "recent=son mailler (varsayılan). unread=okunmamış.",
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
                "Mailin tam içeriğini getirir. 'Şu mailin detayını göster' dediğinde kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email_subject": {
                        "type": "string",
                        "description": "Mail konusu (kısmi eşleşme yeterli)",
                    },
                },
                "required": ["email_subject"],
            },
        },
    },
    # ═══ E. Bot Management (3 tools) ═══
    {
        "type": "function",
        "function": {
            "name": "list_courses",
            "description": "Kayıtlı kursları listeler. Aktif kurs işaretli gösterilir.",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
            "parameters": {"type": "object", "properties": {}, "required": []},
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

    return f"""Sen Bilkent Üniversitesi öğrencileri için bir akademik asistan botsun.
Türkçe yanıt ver (teknik terimler İngilizce kalabilir).

{course_section}
Aktif servisler: {chr(10).join(services)}
Tarih: {date_str} ({today_tr})
{student_ctx}

## KİŞİLİĞİN
- Samimi, yardımsever, motive edici
- Mesaj başına MAX 1 emoji. Emoji'yi sadece başlıklarda kullan (📚📧📋), cümle sonuna koyma. 😊🚀 gibi yüz/eğlence emojileri KULLANMA.
- Kısa ve öz ol — Telegram'da max 3-4 paragraf
- Slash komut sorulursa "Benimle doğal dilde konuşabilirsin!" de

## KİMLİK KURALI
Sen bir Bilkent akademik asistanısın. GPT, Claude, Gemini, OpenAI gibi model isimlerini ASLA söyleme.

## ÇOKLU TOOL
Birden fazla bilgi gerekiyorsa tool'ları paralel çağır.
"Bugün ne var?" → get_schedule(today) + get_assignments(upcoming) paralel

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

## NOT VE DEVAMSIZLIK
- Spesifik ders sorulursa → SADECE o ders
- Genel sorulursa → tüm dersler
- Devamsızlık limitine yaklaşıyorsa → ⚠️ UYAR

## MAİL — DAIS & AIRS
- Mail sorulursa → get_emails(count=5) direkt çağır
- Daha fazla istenirse → belirtilen sayıyla çağır
- Hoca adıyla: sender_filter kullan, sonuç yoksa "Yakın zamanda yok" de
- Mail detayı: get_email_detail
- Ödev sorusunda mail de kontrol et (çapraz sorgu)

Mail sonuçlarını AŞAĞIDAKİ FORMATTA göster (her mail için):
📧 *Konu başlığı*
  👤 Gönderen adı
  📅 Tarih
  💬 Kısa özet (1-2 cümle)

Mailler arasında boş satır bırak. Özetleme YAPMA, her maili ayrı ayrı göster.

## AKILLI ÇAPRAZ SORGU
- "Ödev var mı?" sorulursa → get_assignments + get_emails paralel çağır
- Moodle'da resmi ödev yoksa maillerde ödev duyurusu olabilir — MUTLAKA kontrol et
- Bilgi farklı kaynaklardan geliyorsa hepsini birleştirip sun
- Ödev bilgisi mailde varsa "Moodle'da resmi ödev yok ama mailinizde şu ödev duyurusu var" de

## HATA DÜZELTME PROTOKOLÜ
Kullanıcı bir tarih, isim veya bilgiyi düzelttiğinde:
1. İlgili tool'u tekrar çağırarak kaynağa dön
2. Doğru bilgiyi KAYNAKTAN al
3. "Kontrol ettim, haklısın" de ve doğru bilgiyi kaynak referansıyla sun
Kullanıcının düzeltmesini doğrulamadan KABUL ETME — her zaman kaynaktan teyit et.

## TARİH KURALI
- Tarih bilgisini SADECE tool sonuçlarından al, asla kendin hesaplama/tahmin yapma
- Tool sonucunda tarih varsa BİREBİR aktar, format değiştirme

## FORMAT KURALLARI
1. Telegram Markdown: *bold*, _italic_, `code`
2. Veri sorguları (not, program, ödev) → SADECE istenen veriyi ver
3. RAG sonuçlarını kullanırken 📖 [dosya_adı] kaynak etiketi ekle
4. Tool sonuçlarını doğal dille sun, JSON/teknik format GÖSTERME (mail hariç — mailler yapılandırılmış formatta gösterilmeli)
5. Tool sonucu boş gelirse nazikçe bildir

## TEKNİK TERİM YASAĞI
ASLA kullanma: chunk, RAG, retrieval, embedding, vector, tool, function call, token, pipeline, LLM, model, API, context window, top-k
Bunlar yerine: materyal, kaynak, bilgi, arama, içerik"""


# ─── Tool Availability Filter ────────────────────────────────────────────────


def _get_available_tools(user_id: int) -> list[dict[str, Any]]:
    """Return all tools — unavailable services handled by tool handlers."""
    return list(TOOLS)


# ─── LLM Call with Tools ─────────────────────────────────────────────────────


# ─── Security: Tool Output Sanitization ──────────────────────────────────────

_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|above|prior)\s+instructions?|"
    r"new\s+(role|task|system|instruction)|"
    r"you\s+are\s+now|disregard\s+(all|previous)|"
    r"forget\s+(everything|all|previous)|"
    r"act\s+as\s+(?!a\s+student|an?\s+assistant)|"
    r"pretend\s+(you\s+are|to\s+be))",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]{1,100}>")


def _sanitize_tool_output(tool_name: str, output: str) -> str:
    """Strip prompt injection patterns and HTML from tool results before feeding to LLM."""
    sanitized = _INJECTION_RE.sub("[FILTERED]", output)
    if tool_name in ("get_emails", "get_email_detail"):
        sanitized = _HTML_TAG_RE.sub("", sanitized)
    return sanitized


# ─── Complexity Scoring ───────────────────────────────────────────────────────

_MULTI_STEP_KW = frozenset(
    ["hem", "hem de", "ayrıca", "bunun yanı sıra", "önce", "sonra", "buna ek",
     "and also", "additionally", "first", "then", "compare", "karşılaştır",
     "farkı nedir", "farkları", "hem...hem"]
)
_TECHNICAL_KW = frozenset(
    ["türev", "integral", "kompleks", "algoritma", "kanıtla", "ispat",
     "proof", "derive", "algorithm", "complexity", "o(n)", "theorem", "teorem",
     "matematiksel", "formül", "denklem"]
)


def _score_complexity(query: str) -> float:
    """Return 0.0–1.0 heuristic complexity score for a query (no LLM required)."""
    q = query.lower()
    score = 0.0
    score += min(len(query) / 600, 0.3)
    if any(kw in q for kw in _MULTI_STEP_KW):
        score += 0.25
    if any(kw in q for kw in _TECHNICAL_KW):
        score += 0.25
    if q.count("?") >= 2 or ("neden" in q and "nasıl" in q):
        score += 0.15
    return min(score, 1.0)


# ─── Planner Step ─────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You are a planning assistant for an academic Telegram bot. "
    "Given a student's question and the available tool names, output a short execution plan "
    "as JSON: {\"plan\": [\"step 1\", \"step 2\", ...]} (max 4 steps, be specific about tool names). "
    "Return ONLY the JSON object, no explanation."
)


async def _plan_agent(user_text: str, history: list[dict], tool_names: list[str]) -> str:
    """
    Generate a short execution plan before the tool loop.
    Uses the cheapest model (extraction → gpt-4.1-nano). Returns empty string on any failure.
    """
    llm = STATE.llm
    if llm is None:
        return ""
    context = ""
    if history:
        last = history[-1].get("content", "")[:200]
        context = f"Recent context: {last}\n\n"
    user_prompt = f"{context}Available tools: {', '.join(tool_names)}\n\nStudent question: {user_text}"
    try:
        raw = await asyncio.to_thread(
            llm.engine.complete,
            "extraction",
            _PLANNER_SYSTEM,
            [{"role": "user", "content": user_prompt}],
            300,
        )
        data = json.loads(raw)
        steps = data.get("plan", [])
        if isinstance(steps, list) and steps:
            plan_lines = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps[:4]))
            return f"Execution plan:\n{plan_lines}"
    except Exception as exc:
        logger.debug("Planner step skipped: %s", exc)
    return ""


async def _call_llm_with_tools(
    messages: list[dict[str, Any]],
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> Any:
    """Call LLM with function calling via the adapter's OpenAI client."""
    llm = STATE.llm
    if llm is None:
        return None

    # Adaptive model escalation: complex queries get a more capable model
    last_user_content = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    complexity = _score_complexity(last_user_content)
    if complexity > 0.65:
        model_key = getattr(llm.engine.router, "complexity", llm.engine.router.chat)
        logger.debug("Complexity %.2f → escalating to %s", complexity, model_key)
    else:
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


async def _fuzzy_find_source(source: str, course_name: str | None) -> str | None:
    """Fuzzy filename match: case-insensitive substring search across course files."""
    store = STATE.vector_store
    if store is None or not course_name:
        return None
    try:
        files = await asyncio.to_thread(store.get_files_for_course, course_name)
    except (AttributeError, RuntimeError, ValueError):
        return None
    if not files:
        return None
    src_lower = source.lower()
    matches = [
        f.get("filename", "")
        for f in files
        if src_lower in f.get("filename", "").lower()
    ]
    if not matches:
        return None
    # Prefer shortest filename (most specific match)
    return min(matches, key=len)


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

    # Try fuzzy filename match if exact summary/chunks not found directly
    summary = load_source_summary(source, course_name or "")
    if not summary:
        fuzzy_match = await _fuzzy_find_source(source, course_name)
        if fuzzy_match and fuzzy_match != source:
            logger.info("Fuzzy filename match: '%s' → '%s'", source, fuzzy_match)
            source = fuzzy_match
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
            fuzzy_match = await _fuzzy_find_source(source, course_name)
            if fuzzy_match:
                source = fuzzy_match
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
        fuzzy_match = await _fuzzy_find_source(source, course_name)
        if fuzzy_match:
            source = fuzzy_match
            chunks = await asyncio.to_thread(store.get_file_chunks, source, 0)
    if not chunks:
        return f"'{source}' dosyası bulunamadı. get_source_map ile doğru dosya adını kontrol et."

    offset = args.get("offset", 0)
    page_size = 30
    chunks_page = chunks[offset:offset + page_size]
    total = len(chunks)

    parts = [f"📄 *{source}* — {total} parça\n"]
    for c in chunks_page:
        text = c.get("text", "")
        idx = c.get("chunk_index", 0)
        if text.strip():
            parts.append(f"[Parça {idx + 1}]\n{text}")

    result = "\n\n---\n\n".join(parts)
    shown_end = min(offset + page_size, total)
    result += f"\n\n[Toplam {total} parça. Gösterilen: {offset + 1}–{shown_end}."
    if shown_end < total:
        result += f" Devam için offset={shown_end} kullan.]"
    else:
        result += " Tüm parçalar gösterildi.]"
    return result


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
    """Get schedule from STARS with period filter."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(user_id):
        return "STARS girişi yapılmamış. Ders programını görmek için önce /start ile STARS'a giriş yap."

    try:
        schedule = await asyncio.to_thread(stars.get_schedule, user_id)
    except (ConnectionError, RuntimeError, OSError, ValueError) as exc:
        logger.error("Schedule fetch failed: %s", exc, exc_info=True)
        return f"Ders programı alınamadı: {exc}"

    if not schedule:
        return "Ders programı bilgisi bulunamadı."

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
    """Get grades from STARS with optional course filter."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(user_id):
        return "STARS girişi yapılmamış. Not bilgileri için önce /start ile STARS'a giriş yap."

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
        cname = course.get("course", "Bilinmeyen")
        assessments = course.get("assessments", [])
        if not assessments:
            lines.append(f"📚 {cname}: Henüz not girilmemiş")
            continue
        lines.append(f"📚 {cname}:")
        for a in assessments:
            name = a.get("name", "")
            grade = a.get("grade", "")
            weight = a.get("weight", "")
            w_str = f" (Ağırlık: {weight})" if weight else ""
            lines.append(f"  • {name}: {grade}{w_str}")

    return "\n".join(lines)


async def _tool_get_attendance(args: dict, user_id: int) -> str:
    """Get attendance from STARS with limit warnings."""
    stars = STATE.stars_client
    if stars is None or not stars.is_authenticated(user_id):
        return "STARS girişi yapılmamış. Devamsızlık bilgisi için önce /start ile STARS'a giriş yap."

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
    for cd in attendance:
        cname = cd.get("course", "Bilinmeyen")
        records = cd.get("records", [])
        ratio = cd.get("ratio", "")

        total = len(records)
        absent = sum(1 for r in records if not r.get("attended", True))

        line = f"📚 {cname}:"
        if ratio:
            line += f" Devam oranı: {ratio}"
        line += f" ({absent}/{total} devamsız)"

        try:
            ratio_num = float(ratio.replace("%", "")) if ratio else 100
            if ratio_num < 85:
                line += "\n  ⚠️ Dikkat: Devamsızlık limiti %20'ye yaklaşıyor!"
        except (ValueError, AttributeError):
            pass

        lines.append(line)

    return "\n".join(lines)


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
        raw_due = a.due_date if hasattr(a, "due_date") else None
        if isinstance(raw_due, (int, float)) and raw_due > 1_000_000:
            due = datetime.fromtimestamp(raw_due).strftime("%d/%m/%Y %H:%M")
        elif raw_due:
            due = str(raw_due)
        else:
            due = "Belirtilmemiş"
        remaining = a.time_remaining if hasattr(a, "time_remaining") else ""
        line = f"• {a.course_name} — {a.name}\n  Tarih: {due} | {status}"
        if remaining and not a.submitted:
            line += f" | Kalan: {remaining}"
        if filter_mode == "overdue":
            line += " | ⚠️ Süresi geçmiş!"
        lines.append(line)

    return "\n".join(lines)


async def _tool_get_emails(args: dict, user_id: int) -> str:
    """Get AIRS/DAIS emails."""
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return "Webmail girişi yapılmamış. Mailleri görmek için önce /start ile webmail'e giriş yap."

    count = args.get("count", 5)
    scope = args.get("scope", "recent")
    sender_filter = args.get("sender_filter", "")

    try:
        if scope == "unread":
            mails = await asyncio.to_thread(webmail.check_all_unread)
        else:
            mails = await asyncio.to_thread(webmail.get_recent_airs_dais, count)
    except (ConnectionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("Email fetch failed: %s", exc, exc_info=True)
        return f"E-postalar alınamadı: {exc}"

    if sender_filter:
        sf = sender_filter.lower()
        mails = [m for m in mails if sf in m.get("from", "").lower() or sf in m.get("source", "").lower()]

    if scope != "unread":
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
    """Get full content of a specific email."""
    webmail = STATE.webmail_client
    if webmail is None or not webmail.authenticated:
        return "Webmail girişi yapılmamış."

    subject_query = args.get("email_subject", "")
    if not subject_query:
        return "Mail konusu belirtilmedi."

    try:
        mails = await asyncio.to_thread(webmail.get_recent_airs_dais, 20)
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
        for m in mails:
            if sq in m.get("body_preview", "").lower():
                match = m
                break

    if not match:
        return f"'{subject_query}' konusuyla eşleşen mail bulunamadı."

    return (
        f"📧 *{match.get('subject', 'Konusuz')}*\n"
        f"Kimden: {match.get('from', '')}\n"
        f"Tarih: {match.get('date', '')}\n\n"
        f"{match.get('body_preview', '')}"
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
            result = "Bu bilgiye şu anda ulaşılamıyor."

    result = _sanitize_tool_output(fn_name, result)

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


# ─── Main Entry Point ────────────────────────────────────────────────────────


async def handle_agent_message(user_id: int, user_text: str) -> str:
    """
    Main agentic handler: takes user message, runs tool loop, returns response.

    Flow:
    1. Build system prompt with 3-layer teaching methodology
    2. Get conversation history
    3. Call LLM with 14 tools + parallel_tool_calls=True
    4. If tool calls → execute in parallel → feed results → repeat (max 5)
    5. Return final text response
    """
    if STATE.llm is None:
        return "Sistem henüz hazır değil. Lütfen birazdan tekrar deneyin."

    system_prompt = _build_system_prompt(user_id)
    available_tools = _get_available_tools(user_id)

    history = user_service.get_conversation_history(user_id)
    messages: list[dict[str, Any]] = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    # Planner step: generate a short execution plan and inject into system prompt
    tool_names = [t["function"]["name"] for t in available_tools]
    plan_hint = await _plan_agent(user_text, history, tool_names)
    if plan_hint:
        system_prompt = system_prompt + f"\n\n{plan_hint}"
        logger.debug("Planner hint injected (%d chars)", len(plan_hint))

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response_msg = await _call_llm_with_tools(
                messages, system_prompt, available_tools
            )
        except Exception as exc:
            logger.error("LLM call failed (iteration %d): %s", iteration, exc, exc_info=True)
            return "Bir sorun oluştu. Lütfen tekrar deneyin."

        if response_msg is None:
            return "Yanıt üretilemedi. Lütfen tekrar deneyin."

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

        tool_results = await asyncio.gather(
            *[_execute_tool_call(tc, user_id) for tc in tool_calls]
        )
        messages.extend(tool_results)

        logger.info(
            "Tool loop iteration %d: %d tools",
            iteration + 1,
            len(tool_calls),
            extra={"user_id": user_id, "tools": [tc.function.name for tc in tool_calls]},
        )

    # Max iterations exceeded
    try:
        response_msg = await _call_llm_with_tools(messages, system_prompt, [])
        final_text = response_msg.content if response_msg else "Yanıt üretilemedi."
    except Exception:
        final_text = "İşlem zaman aşımına uğradı. Lütfen tekrar deneyin."

    user_service.add_conversation_turn(user_id, "user", user_text)
    user_service.add_conversation_turn(user_id, "assistant", final_text)
    return final_text
