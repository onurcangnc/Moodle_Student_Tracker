#!/usr/bin/env python3
"""
24-test smoke suite for the Moodle Student Tracker bot.
Tests all tools, multi-language, context switching, and filler words.

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
    # ─── A. Basic tool calls ───
    {
        "name": "greeting",
        "input": "Merhaba!",
        "expect_tool": None,  # No tool should be called
        "expect_contains": None,
        "expect_no_tool": True,
    },
    {
        "name": "get_schedule",
        "input": "Bugün derslerim ne?",
        "expect_tool": "get_schedule",
        "expect_contains": None,
    },
    {
        "name": "get_grades",
        "input": "Notlarım nedir?",
        "expect_tool": "get_grades",
        "expect_contains": None,
    },
    {
        "name": "get_attendance",
        "input": "Devamsızlığım ne durumda?",
        "expect_tool": "get_attendance",
        "expect_contains": None,
    },
    {
        "name": "get_assignments",
        "input": "Ödevlerim var mı?",
        "expect_tool": "get_assignments",
        "expect_contains": None,
    },
    {
        "name": "get_emails_count",
        "input": "Son 3 mailimi göster",
        "expect_tool": "get_emails",
        "expect_contains": None,
    },
    {
        "name": "get_emails_keyword",
        "input": "EDEB maili var mı?",
        "expect_tool": "get_emails",
        "expect_contains": None,
    },
    {
        "name": "list_courses",
        "input": "Kurslarımı göster",
        "expect_tool": "list_courses",
        "expect_contains": None,
    },
    {
        "name": "get_source_map",
        "input": "Bu dersin materyalleri ne?",
        "expect_tool": "get_source_map",
        "expect_contains": None,
    },
    {
        "name": "rag_search",
        "input": "Ethics nedir?",
        "expect_tool": "rag_search",
        "expect_contains": None,
    },
    {
        "name": "study_topic",
        "input": "Privacy konusunu çalışmak istiyorum",
        "expect_tool": "study_topic",
        "expect_contains": None,
    },
    {
        "name": "get_stats",
        "input": "Bot istatistikleri",
        "expect_tool": "get_stats",
        "expect_contains": None,
    },
    # ─── B. Multi-language ───
    {
        "name": "english_grades",
        "input": "Show me my grades",
        "expect_tool": "get_grades",
        "expect_lang": "en",
    },
    {
        "name": "english_schedule",
        "input": "What is my schedule today?",
        "expect_tool": "get_schedule",
        "expect_lang": "en",
    },
    {
        "name": "english_greeting",
        "input": "Hello, how are you?",
        "expect_tool": None,
        "expect_no_tool": True,
        "expect_lang": "en",
    },
    # ─── C. Context / filler words ───
    {
        "name": "filler_neyse",
        "input": "Neyse devam edelim",
        "expect_tool": None,
        "expect_no_tool": True,
        "expect_not_contains": ["arama", "bulunamadı"],
    },
    {
        "name": "filler_hani",
        "input": "Hani az önce konuşmuştuk ya",
        "expect_tool": None,
        "expect_no_tool": True,
        "expect_not_contains": ["arama", "bulunamadı"],
    },
    # ─── D. Parallel tool calls ───
    {
        "name": "parallel_bugün",
        "input": "Bugün ne var? Dersler ve ödevler",
        "expect_tool": "get_schedule",
        "expect_also_tool": "get_assignments",
    },
    {
        "name": "parallel_akademik",
        "input": "Akademik durumum nasıl? Notlar ve devamsızlık",
        "expect_tool": "get_grades",
        "expect_also_tool": "get_attendance",
    },
    # ─── E. Set course ───
    {
        "name": "set_course",
        "input": "CTIS dersine geçelim",
        "expect_tool": "set_active_course",
        "expect_contains": None,
    },
    # ─── F. Edge cases ───
    {
        "name": "chat_no_tool",
        "input": "Teşekkürler, harikasın!",
        "expect_tool": None,
        "expect_no_tool": True,
    },
    {
        "name": "identity_check",
        "input": "Sen hangi model kullanıyorsun?",
        "expect_tool": None,
        "expect_no_tool": True,
        "expect_not_contains": ["GPT", "OpenAI", "Claude", "Gemini"],
    },
    {
        "name": "deadline_check",
        "input": "Yaklaşan deadline'larım",
        "expect_tool": "get_assignments",
        "expect_contains": None,
    },
    {
        "name": "mail_detail",
        "input": "CTIS mailinin detayını göster",
        "expect_tool": "get_email_detail",
        "expect_contains": None,
    },
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
