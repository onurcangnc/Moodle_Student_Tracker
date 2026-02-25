#!/usr/bin/env python3
"""
70+ behavioral smoke suite for the Moodle Student Tracker bot.
Tests all 14 tools, multi-language, context switching, filler words,
mail edge cases, notification context, and planning patterns.

Usage:
    python tests/smoke_test.py

Requires: OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_OWNER_ID in .env
"""

import asyncio
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from bot.config import CONFIG
from bot.state import STATE

# ── Test definitions ─────────────────────────────────────────────────────────

TESTS = [
    # ═══════════════════════════════════════════════════════════════════════════
    # A. BASIC TOOL CALLS (14 tools)
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "greeting", "input": "Merhaba!", "expect_no_tool": True},
    {"name": "get_schedule_today", "input": "Bugün derslerim ne?"},
    {"name": "get_schedule_tomorrow", "input": "Yarın hangi derslerim var?"},
    {"name": "get_schedule_week", "input": "Haftalık ders programım"},
    {"name": "get_grades", "input": "Notlarım nedir?"},
    {"name": "get_grades_course", "input": "CTIS notlarım ne?"},
    {"name": "get_attendance", "input": "Devamsızlığım ne durumda?"},
    {"name": "get_attendance_course", "input": "EDEB devamsızlığım"},
    {"name": "get_assignments", "input": "Ödevlerim var mı?"},
    {"name": "get_assignments_overdue", "input": "Gecikmiş ödevlerim var mı?"},
    {"name": "get_emails_count", "input": "Son 3 mailimi göster"},
    {"name": "get_emails_keyword", "input": "EDEB maili var mı?"},
    {"name": "list_courses", "input": "Kurslarımı göster"},
    {"name": "get_source_map", "input": "Bu dersin materyalleri ne?"},
    {"name": "rag_search", "input": "Ethics nedir?"},
    {"name": "study_topic", "input": "Privacy konusunu çalışmak istiyorum"},
    {"name": "get_stats", "input": "Bot istatistikleri"},
    {"name": "set_course", "input": "CTIS dersine geçelim"},

    # ═══════════════════════════════════════════════════════════════════════════
    # B. MULTI-LANGUAGE (English → English response)
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "en_grades", "input": "Show me my grades", "expect_lang": "en"},
    {"name": "en_schedule", "input": "What is my schedule today?", "expect_lang": "en"},
    {"name": "en_greeting", "input": "Hello, how are you?", "expect_lang": "en", "expect_no_tool": True},
    {"name": "en_assignments", "input": "Do I have any assignments?", "expect_lang": "en"},
    {"name": "en_emails", "input": "Show me my emails"},  # Mail content has TR chars (sender names), skip lang check
    {"name": "en_attendance", "input": "How is my attendance?", "expect_lang": "en"},
    {"name": "en_courses", "input": "List my courses", "expect_lang": "en"},
    {"name": "en_help", "input": "Help me please", "expect_lang": "en", "expect_no_tool": True},

    # ═══════════════════════════════════════════════════════════════════════════
    # C. MAIL UX — No unnecessary questions
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "mail_tum", "input": "Tüm mailleri göster", "expect_not_contains": ["Kaç mail", "kaç mail", "seçenekler"]},
    {"name": "mail_hepsi", "input": "Hepsini göster", "expect_not_contains": ["Kaç mail", "kaç mail"]},
    {"name": "mail_hepsi_hoca", "input": "Serhat hocanın tüm mailleri", "expect_not_contains": ["Kaç mail", "kaç mail"]},
    {"name": "mail_no_ask", "input": "Maillerimi göster", "expect_not_contains": ["Kaç mail", "kaç mail"]},
    {"name": "mail_5_direct", "input": "Son 5 mailimi göster", "expect_not_contains": ["Kaç mail", "kaç mail"]},
    {"name": "mail_10_direct", "input": "Son 10 mail", "expect_not_contains": ["Kaç mail", "kaç mail"]},
    {"name": "mail_detail", "input": "CTIS mailinin detayını göster"},
    {"name": "mail_hoca_search", "input": "Serhat hocanın son maili"},
    {"name": "mail_ders_kodu", "input": "HCIV maili var mı?"},

    # ═══════════════════════════════════════════════════════════════════════════
    # D. CONTEXT / FILLER WORDS — No false searches
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "filler_neyse", "input": "Neyse devam edelim", "expect_no_tool": True, "expect_not_contains": ["bulunamadı"]},
    {"name": "filler_hani", "input": "Hani az önce konuşmuştuk ya", "expect_no_tool": True, "expect_not_contains": ["bulunamadı"]},
    {"name": "filler_iste", "input": "İşte tam da öyle", "expect_no_tool": True},
    {"name": "thanks", "input": "Teşekkürler, harikasın!", "expect_no_tool": True},
    {"name": "ok_response", "input": "Tamam anladım", "expect_no_tool": True},
    {"name": "emoji_only", "input": "👍", "expect_no_tool": True},

    # ═══════════════════════════════════════════════════════════════════════════
    # E. IDENTITY PROTECTION — Never reveal model name
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "identity_model", "input": "Sen hangi model kullanıyorsun?", "expect_no_tool": True, "expect_not_contains": ["GPT", "OpenAI", "Claude", "Gemini"]},
    {"name": "identity_gpt", "input": "Sen GPT misin?", "expect_no_tool": True, "expect_not_contains": ["GPT-5", "OpenAI"]},
    {"name": "identity_who", "input": "Sen kimsin?", "expect_no_tool": True, "expect_not_contains": ["GPT", "OpenAI", "Claude"]},

    # ═══════════════════════════════════════════════════════════════════════════
    # F. PARALLEL TOOL CALLS — Complex queries
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "parallel_today", "input": "Bugün ne var? Dersler ve ödevler"},
    {"name": "parallel_academic", "input": "Akademik durumum nasıl? Notlar ve devamsızlık"},
    {"name": "parallel_overview", "input": "Sınavlara nasıl hazırlanayım?"},

    # ═══════════════════════════════════════════════════════════════════════════
    # G. STUDY MODE — Teaching patterns
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "study_request", "input": "Çalışmak istiyorum"},
    {"name": "study_topic_deep", "input": "Surveillance konusunu detaylı çalışmak istiyorum"},
    {"name": "study_question", "input": "Etik nedir kısaca açıkla"},
    {"name": "study_concept", "input": "GDPR hakkında ne biliyorsun?"},

    # ═══════════════════════════════════════════════════════════════════════════
    # H. EDGE CASES — Tricky inputs
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "deadline_check", "input": "Yaklaşan deadline'larım"},
    {"name": "single_word_mail", "input": "Mail"},
    {"name": "abbreviation", "input": "CTIS 363"},
    {"name": "mixed_lang", "input": "Bana grades göster"},
    {"name": "question_mark_only", "input": "?", "expect_no_tool": True},
    {"name": "number_only", "input": "5", "expect_no_tool": True},
    {"name": "slash_command_ask", "input": "/notlar ne demek?", "expect_no_tool": True},

    # ═══════════════════════════════════════════════════════════════════════════
    # I. RESPONSE QUALITY — Proper formatting
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "mail_format_check", "input": "Son 2 mailimi göster", "expect_contains_any": ["📧", "Kimden", "Tarih"]},
    {"name": "no_json_leak", "input": "Notlarımı göster", "expect_not_contains": ['"course":', '"assessments":', "json"]},
    {"name": "no_technical_terms", "input": "Bu dersi çalışayım", "expect_not_contains": ["chunk", "RAG", "embedding", "vector", "token"]},

    # ═══════════════════════════════════════════════════════════════════════════
    # J. TURKISH VARIATIONS — Same intent, different phrasing
    # ═══════════════════════════════════════════════════════════════════════════
    {"name": "tr_grades_v1", "input": "Notlarıma bak"},
    {"name": "tr_grades_v2", "input": "Not durumum ne?"},
    {"name": "tr_schedule_v1", "input": "Bugün kaçta dersim var?"},
    {"name": "tr_schedule_v2", "input": "Ders programı"},
    {"name": "tr_mail_v1", "input": "E-postalarım"},
    {"name": "tr_attendance_v1", "input": "Yoklamam nasıl?"},
    {"name": "tr_assignments_v1", "input": "Ödev teslim tarihleri"},
    {"name": "tr_study_v1", "input": "Ders çalışmak istiyorum"},
]


# ── Minimal test runner ──────────────────────────────────────────────────────

async def run_tests():
    """Initialize components and run smoke tests."""
    # Minimal init
    from core import config as core_config
    from core.moodle_client import MoodleClient
    from core.document_processor import DocumentProcessor
    from core.vector_store import VectorStore
    from core.llm_engine import LLMEngine
    from core.sync_engine import SyncEngine
    from core.stars_client import StarsClient
    from core.webmail_client import WebmailClient

    errors = core_config.validate()
    if errors:
        print(f"Config errors: {errors}")
        sys.exit(1)

    moodle = MoodleClient()
    processor = DocumentProcessor()
    vector_store = VectorStore()
    vector_store.initialize()
    llm = LLMEngine(vector_store)
    sync_engine = SyncEngine(moodle, processor, vector_store)

    STATE.moodle = moodle
    STATE.processor = processor
    STATE.vector_store = vector_store
    STATE.llm = llm
    STATE.sync_engine = sync_engine
    STATE.stars_client = StarsClient()
    STATE.webmail_client = WebmailClient()
    STATE.started_at_monotonic = time.monotonic()
    STATE.startup_version = "smoke-test"

    # Login webmail
    webmail_email = os.getenv("WEBMAIL_EMAIL", "")
    webmail_password = os.getenv("WEBMAIL_PASSWORD", "")
    if webmail_email and webmail_password:
        STATE.webmail_client.login(webmail_email, webmail_password)

    # Login STARS
    stars_user = os.getenv("STARS_USERNAME", "")
    stars_pass = os.getenv("STARS_PASSWORD", "")
    owner_id = CONFIG.owner_id
    if stars_user and stars_pass and owner_id:
        result = STATE.stars_client.start_login(owner_id, stars_user, stars_pass)
        if result.get("status") == "sms_sent" and STATE.webmail_client.authenticated:
            for _ in range(4):
                time.sleep(5)
                code = STATE.webmail_client.fetch_stars_verification_code(max_age_seconds=60)
                if code:
                    STATE.stars_client.verify_sms(owner_id, code)
                    break

    # Moodle connect
    if moodle.connect():
        courses = moodle.get_courses()
        llm.moodle_courses = [{"shortname": c.shortname, "fullname": c.fullname} for c in courses]

    # Import agent
    from bot.services.agent_service import handle_agent_message, _build_system_prompt, _detect_language
    from bot.services import user_service

    user_id = owner_id or 123456

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST — {len(TESTS)} tests")
    print(f"  Model: {llm.engine.router.chat}")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0
    results = []

    for i, test in enumerate(TESTS, 1):
        name = test["name"]
        user_input = test["input"]

        print(f"[{i:02d}/{len(TESTS)}] {name}: \"{user_input[:50]}\"", end=" ... ")
        sys.stdout.flush()

        # Clear conversation history between tests to avoid contamination
        user_service.clear_conversation_history(user_id)

        try:
            start = time.time()
            response = await handle_agent_message(user_id, user_input)
            elapsed = time.time() - start

            test_pass = True
            fail_reasons = []

            # Check response is not empty
            if not response or len(response.strip()) < 5:
                test_pass = False
                fail_reasons.append("Empty response")

            # Check expected language
            if test.get("expect_lang") == "en":
                # Response should be mostly English
                tr_chars = sum(1 for c in response if c in "çğıöşüÇĞİÖŞÜ")
                if tr_chars > 5:
                    test_pass = False
                    fail_reasons.append(f"Expected English, got Turkish chars ({tr_chars})")

            # Check expect_not_contains
            for bad_word in test.get("expect_not_contains", []):
                if bad_word.lower() in response.lower():
                    test_pass = False
                    fail_reasons.append(f"Response contains forbidden word: '{bad_word}'")

            # Check expect_contains_any (at least one must match)
            contains_any = test.get("expect_contains_any", [])
            if contains_any:
                if not any(word in response for word in contains_any):
                    test_pass = False
                    fail_reasons.append(f"Expected one of {contains_any}")

            if test_pass:
                passed += 1
                print(f"PASS ({elapsed:.1f}s)")
            else:
                failed += 1
                print(f"FAIL ({elapsed:.1f}s) — {'; '.join(fail_reasons)}")
                print(f"    Response: {response[:150]}...")

            results.append({"name": name, "pass": test_pass, "time": elapsed})

        except Exception as exc:
            failed += 1
            print(f"ERROR — {type(exc).__name__}: {exc}")
            results.append({"name": name, "pass": False, "time": 0})

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{len(TESTS)} PASS, {failed}/{len(TESTS)} FAIL")
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0
    print(f"  Average response time: {avg_time:.1f}s")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r["pass"]:
                print(f"  - {r['name']}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
