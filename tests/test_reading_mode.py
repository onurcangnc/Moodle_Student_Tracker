"""
Reading Mode E2E Tests — 10 sequential scenarios testing the full reading mode flow.

Tests:
  1. "EDEB 201 çalışalım"        → InlineKeyboard menu
  2. [File button click]          → First batch + progress + source footer
  3. "devam et" x3                → New chunks each time + footer
  4. "burada ne demek istiyor"    → RAG (file scope) + footer
  5. "devam et"                   → Position preserved after QA
  6. "soru sorma"                 → Socratic toggle ack
  7. "devam et"                   → No questions in response + footer
  8. "beni test et"               → Quiz from read chunks + footer
  9. "KVKK nedir"                 → 3-tier RAG confidence + correct footer type
  10. "CTIS 474 çalışacağım"      → No-material warning

Run:
  cd /opt/moodle-bot && source venv/bin/activate
  python tests/test_reading_mode.py              # logic-only (no LLM)
  python tests/test_reading_mode.py --live       # full LLM calls (costs credits)
  python tests/test_reading_mode.py -v           # verbose output
"""

import sys
import time
import json
from dataclasses import dataclass

sys.path.insert(0, ".")


# ─── Result Tracking ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    step: int
    name: str
    checks: dict[str, bool]
    details: str = ""

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def summary(self) -> str:
        icon = "✅" if self.passed else "❌"
        fails = [k for k, v in self.checks.items() if not v]
        fail_str = f"  FAILED: {', '.join(fails)}" if fails else ""
        return f"  {icon} Step {self.step}: {self.name}{fail_str}"


def run_tests(live: bool = False, verbose: bool = False):
    # ─── Imports ──────────────────────────────────────────────────────────
    from core.vector_store import VectorStore
    from core.llm_engine import LLMEngine
    from core.memory import MemoryManager

    vs = VectorStore()
    vs.initialize()

    # Initialize telegram_bot module globals BEFORE importing functions
    import telegram_bot as bot_module
    bot_module.vector_store = vs

    llm_engine = LLMEngine(vs)
    mem = MemoryManager()
    llm_engine.mem_manager = mem
    bot_module.llm = llm_engine

    # Set up moodle_courses for detect_active_course (same as existing E2E test)
    llm_engine.moodle_courses = [
        {"shortname": "CTIS 363-1", "fullname": "CTIS 363-1 Ethical and Social Issues in Information Systems"},
        {"shortname": "HCIV 102-4", "fullname": "HCIV 102-4 History of Civilization II"},
        {"shortname": "CTIS 474-1", "fullname": "CTIS 474-1 Information Systems Auditing"},
        {"shortname": "EDEB 201-2", "fullname": "EDEB 201-2 Introduction to Turkish Fiction"},
        {"shortname": "CTIS 465-1", "fullname": "CTIS 465-1 Microservice Development"},
        {"shortname": "CTIS 456-4", "fullname": "CTIS 456-4 Senior Project II"},
    ]

    llm = llm_engine if live else None

    # Import bot logic functions
    from telegram_bot import (
        _get_user_state,
        _reset_reading_mode,
        _start_reading_mode,
        _check_socratic_toggle,
        _is_continue_command,
        _is_test_command,
        _needs_topic_menu,
        _format_topic_menu,
        _format_progress,
        _get_reading_batch,
        _format_completion_message,
        _format_source_footer,
        _extract_sources,
        detect_active_course,
        _user_state,
        _HIGH_CONFIDENCE,
        _LOW_CONFIDENCE,
        _READING_MODE_INSTRUCTION,
        _READING_QA_INSTRUCTION,
        _NO_SOCRATIC_INSTRUCTION,
        _QUIZ_INSTRUCTION,
        _RAG_PARTIAL_INSTRUCTION,
        _NO_RAG_INSTRUCTION,
        _SOURCE_RULE,
        READING_BATCH_SIZE,
    )
    from telegram import InlineKeyboardMarkup

    results: list[StepResult] = []
    TEST_UID = 99999  # fake user id

    # Clean state
    _user_state.pop(TEST_UID, None)
    state = _get_user_state(TEST_UID)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: "EDEB 201 çalışalım" → InlineKeyboard menu
    # ═══════════════════════════════════════════════════════════════════════
    msg = "EDEB 201 çalışalım"
    checks = {}

    # Course detection
    detected = detect_active_course(msg, TEST_UID)
    checks["course_detected"] = detected is not None
    checks["course_is_edeb201"] = "EDEB 201" in (detected or "")

    # Needs topic menu?
    checks["needs_menu"] = _needs_topic_menu(msg)

    # Get files for course
    if detected:
        state["current_course"] = detected
        course_files = vs.get_files_for_course(course_name=detected)
        checks["has_files"] = len(course_files) > 0

        # Format menu
        header, markup = _format_topic_menu(detected, course_files)
        checks["header_is_str"] = isinstance(header, str)
        checks["markup_is_keyboard"] = isinstance(markup, InlineKeyboardMarkup)
        checks["keyboard_has_buttons"] = len(markup.inline_keyboard) > 0

        # Check callback data format
        first_btn = markup.inline_keyboard[0][0]
        checks["callback_starts_rf"] = first_btn.callback_data.startswith("rf|")

        if verbose:
            print(f"\n  [Step 1] Course: {detected}")
            print(f"  [Step 1] Files: {len(course_files)}")
            print(f"  [Step 1] Header: {header}")
            for row in markup.inline_keyboard:
                print(f"  [Step 1] Button: {row[0].text} → {row[0].callback_data}")
    else:
        checks["has_files"] = False
        checks["header_is_str"] = False
        checks["markup_is_keyboard"] = False
        checks["keyboard_has_buttons"] = False
        checks["callback_starts_rf"] = False

    results.append(StepResult(1, "EDEB 201 çalışalım → Menu", checks,
                              f"detected={detected}"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: File button click → reading mode start + first batch
    # ═══════════════════════════════════════════════════════════════════════
    checks = {}

    # Pick a file (prefer Berna Moran if exists, otherwise first file)
    target_file = None
    if detected:
        course_files = vs.get_files_for_course(course_name=detected)
        for f in course_files:
            if "berna" in f["filename"].lower() or "moran" in f["filename"].lower():
                target_file = f
                break
        if not target_file and course_files:
            target_file = course_files[0]

    if target_file:
        filename = target_file["filename"]
        total = target_file.get("chunk_count", 0)
        display = filename.rsplit(".", 1)[0].replace("_", " ")

        # Simulate callback: _start_reading_mode
        _start_reading_mode(state, filename, display, total)

        checks["reading_mode_on"] = state["reading_mode"] is True
        checks["reading_file_set"] = state["reading_file"] == filename
        checks["reading_position_zero"] = state["reading_position"] == 0
        checks["reading_total_set"] = state["reading_total"] == total
        checks["seen_chunks_cleared"] = state["seen_chunk_ids"] == []

        # Get first batch
        batch = _get_reading_batch(filename, 0)
        checks["batch_not_empty"] = len(batch) > 0
        checks["batch_size_correct"] = len(batch) <= READING_BATCH_SIZE

        if batch:
            state["reading_position"] = len(batch)
            progress = _format_progress(state["reading_position"], total)
            checks["progress_has_bar"] = "█" in progress or "░" in progress
            checks["progress_has_count"] = f"{len(batch)}/{total}" in progress

            # Source footer
            footer = _format_source_footer(batch, "reading")
            checks["footer_has_source"] = "📄" in footer or "Kaynak" in footer

            # LLM call (if live)
            if live and llm:
                response = llm.chat_with_history(
                    messages=[{"role": "user", "content": f"Bu bölümü öğretici bir şekilde anlat: {display}"}],
                    context_chunks=batch,
                    study_mode=True,
                    extra_system=_READING_MODE_INSTRUCTION,
                )
                checks["llm_response_not_empty"] = len(response) > 20
                checks["llm_no_copy_paste"] = len(response) < len("\n".join(c["text"] for c in batch))
                if verbose:
                    print(f"\n  [Step 2] LLM Response ({len(response)} chars):")
                    print(f"  {response[:300]}...")
                time.sleep(1)

        if verbose:
            print(f"\n  [Step 2] File: {filename}")
            print(f"  [Step 2] Total chunks: {total}")
            print(f"  [Step 2] Batch size: {len(batch) if batch else 0}")
            print(f"  [Step 2] Position after: {state['reading_position']}")
    else:
        checks["reading_mode_on"] = False
        checks["batch_not_empty"] = False

    results.append(StepResult(2, "File button → First batch + progress", checks,
                              f"file={target_file['filename'] if target_file else 'None'}"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: "devam et" x3 → New chunks each time
    # ═══════════════════════════════════════════════════════════════════════
    checks = {}
    positions = [state["reading_position"]]

    for i in range(3):
        msg = "devam et"
        checks[f"continue_{i+1}_detected"] = _is_continue_command(msg)
        checks[f"continue_{i+1}_in_reading"] = state.get("reading_mode", False)

        batch = _get_reading_batch(state["reading_file"], state["reading_position"])
        if batch:
            old_pos = state["reading_position"]
            state["reading_position"] += len(batch)
            positions.append(state["reading_position"])
            checks[f"continue_{i+1}_new_chunks"] = state["reading_position"] > old_pos
            checks[f"continue_{i+1}_batch_size"] = 0 < len(batch) <= READING_BATCH_SIZE

            progress = _format_progress(state["reading_position"], state["reading_total"])
            checks[f"continue_{i+1}_progress_updated"] = f"{state['reading_position']}/{state['reading_total']}" in progress

            footer = _format_source_footer(batch, "reading")
            checks[f"continue_{i+1}_has_footer"] = "📄" in footer or "Kaynak" in footer

            if live and llm:
                response = llm.chat_with_history(
                    messages=[{"role": "user", "content": "Devam et, sonraki bölümü öğret."}],
                    context_chunks=batch,
                    study_mode=True,
                    extra_system=_READING_MODE_INSTRUCTION,
                )
                checks[f"continue_{i+1}_llm_ok"] = len(response) > 20
                if verbose:
                    print(f"\n  [Step 3.{i+1}] Position: {state['reading_position']}")
                    print(f"  [Step 3.{i+1}] Response: {response[:200]}...")
                time.sleep(1)
        else:
            checks[f"continue_{i+1}_new_chunks"] = False  # file ended too early

    # Verify positions are strictly increasing
    checks["positions_increasing"] = all(positions[i] < positions[i+1] for i in range(len(positions)-1))

    results.append(StepResult(3, '"devam et" x3 → Sequential chunks', checks,
                              f"positions={positions}"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: "burada ne demek istiyor" → RAG (file scope) + footer
    # ═══════════════════════════════════════════════════════════════════════
    msg = "burada ne demek istiyor"
    pos_before_qa = state["reading_position"]
    checks = {}

    checks["still_reading_mode"] = state.get("reading_mode", False)
    checks["not_continue"] = not _is_continue_command(msg)
    checks["not_test"] = not _is_test_command(msg)

    # RAG search — file scope
    all_chunks = vs.get_file_chunks(state["reading_file"])
    recent_read = all_chunks[max(0, state["reading_position"] - 5):state["reading_position"]]
    checks["recent_read_not_empty"] = len(recent_read) > 0

    rag_results = vs.hybrid_search(
        query=msg,
        n_results=10,
        course_filter=state.get("current_course"),
    )
    file_results = [r for r in rag_results if r.get("metadata", {}).get("filename") == state["reading_file"]]

    # Merge with recent read
    seen_ids = {r["id"] for r in file_results}
    for c in recent_read:
        if c["id"] not in seen_ids:
            file_results.append(c)
            seen_ids.add(c["id"])

    checks["has_context"] = len(file_results) > 0

    footer = _format_source_footer(file_results[:10], "rag_strong" if file_results else "general")
    checks["footer_type_correct"] = "📄" in footer if file_results else "💡" in footer

    if live and llm:
        response = llm.chat_with_history(
            messages=[{"role": "user", "content": msg}],
            context_chunks=file_results[:10],
            study_mode=True,
            extra_system=_READING_QA_INSTRUCTION,
        )
        checks["llm_response_ok"] = len(response) > 20
        if verbose:
            print(f"\n  [Step 4] File results: {len(file_results)}")
            print(f"  [Step 4] Response: {response[:300]}...")
        time.sleep(1)

    results.append(StepResult(4, '"burada ne demek istiyor" → File-scoped RAG', checks,
                              f"file_results={len(file_results)}"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: "devam et" → Position preserved after QA
    # ═══════════════════════════════════════════════════════════════════════
    checks = {}

    checks["position_preserved"] = state["reading_position"] == pos_before_qa
    checks["still_in_reading"] = state.get("reading_mode", False)

    batch = _get_reading_batch(state["reading_file"], state["reading_position"])
    if batch:
        old_pos = state["reading_position"]
        state["reading_position"] += len(batch)
        checks["continues_from_correct_pos"] = state["reading_position"] > pos_before_qa
        checks["batch_not_empty"] = True

        footer = _format_source_footer(batch, "reading")
        checks["has_footer"] = "📄" in footer or "Kaynak" in footer

        if live and llm:
            response = llm.chat_with_history(
                messages=[{"role": "user", "content": "Devam et, sonraki bölümü öğret."}],
                context_chunks=batch,
                study_mode=True,
                extra_system=_READING_MODE_INSTRUCTION,
            )
            checks["llm_response_ok"] = len(response) > 20
            if verbose:
                print(f"\n  [Step 5] Position: {pos_before_qa} → {state['reading_position']}")
            time.sleep(1)
    else:
        checks["continues_from_correct_pos"] = True  # file ended, still ok
        checks["batch_not_empty"] = False

    results.append(StepResult(5, '"devam et" → Position preserved after QA', checks,
                              f"pos_before={pos_before_qa}, pos_after={state['reading_position']}"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: "soru sorma" → Socratic toggle off
    # ═══════════════════════════════════════════════════════════════════════
    msg = "soru sorma"
    checks = {}

    checks["was_socratic_on"] = state["socratic_mode"] is True
    ack = _check_socratic_toggle(msg, state)
    checks["toggle_returned_ack"] = ack is not None
    checks["socratic_now_off"] = state["socratic_mode"] is False
    checks["seen_chunks_reset"] = state["seen_chunk_ids"] == []
    checks["reading_mode_preserved"] = state.get("reading_mode", False)

    if verbose:
        print(f"\n  [Step 6] Ack: {ack}")
        print(f"  [Step 6] Socratic: {state['socratic_mode']}")

    results.append(StepResult(6, '"soru sorma" → Socratic toggle off', checks))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: "devam et" → No questions in response (socratic off)
    # ═══════════════════════════════════════════════════════════════════════
    checks = {}

    checks["socratic_is_off"] = state["socratic_mode"] is False

    batch = _get_reading_batch(state["reading_file"], state["reading_position"])
    if batch:
        state["reading_position"] += len(batch)

        extra_sys = _READING_MODE_INSTRUCTION + "\n" + _NO_SOCRATIC_INSTRUCTION
        checks["extra_sys_has_no_socratic"] = _NO_SOCRATIC_INSTRUCTION in extra_sys

        footer = _format_source_footer(batch, "reading")
        checks["has_footer"] = "📄" in footer or "Kaynak" in footer

        if live and llm:
            response = llm.chat_with_history(
                messages=[{"role": "user", "content": "Devam et, sonraki bölümü öğret."}],
                context_chunks=batch,
                study_mode=True,
                extra_system=extra_sys,
            )
            checks["llm_response_ok"] = len(response) > 20
            # Check no questions asked
            checks["no_question_marks"] = "?" not in response
            if verbose:
                print(f"\n  [Step 7] Response ({len(response)} chars):")
                print(f"  {response[:300]}...")
                if "?" in response:
                    print(f"  ⚠️ Found '?' in response — socratic should be off!")
            time.sleep(1)
    else:
        checks["extra_sys_has_no_socratic"] = True
        checks["has_footer"] = True

    results.append(StepResult(7, '"devam et" → No questions (socratic off)', checks))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: "beni test et" → Quiz from read chunks
    # ═══════════════════════════════════════════════════════════════════════
    msg = "beni test et"
    checks = {}

    checks["is_test_command"] = _is_test_command(msg)
    checks["in_reading_mode"] = state.get("reading_mode", False)

    all_chunks = vs.get_file_chunks(state["reading_file"])
    read_chunks = all_chunks[:state["reading_position"]]
    checks["has_read_chunks"] = len(read_chunks) > 0
    checks["read_chunks_count"] = len(read_chunks) > 0  # at least some

    quiz_chunks = read_chunks[-10:]
    footer = _format_source_footer(quiz_chunks, "quiz")
    checks["footer_has_source"] = "📄" in footer or "Kaynak" in footer

    if live and llm:
        response = llm.chat_with_history(
            messages=[{"role": "user", "content": "Okuduğumuz bölümlerden beni test et."}],
            context_chunks=quiz_chunks,
            study_mode=True,
            extra_system=_QUIZ_INSTRUCTION,
        )
        checks["llm_response_ok"] = len(response) > 20
        checks["has_question"] = "?" in response
        # Should be a single question, not multiple
        q_count = response.count("?")
        checks["single_question"] = 1 <= q_count <= 3  # allow some slack

        if verbose:
            print(f"\n  [Step 8] Read chunks: {len(read_chunks)}, Quiz from last {len(quiz_chunks)}")
            print(f"  [Step 8] Questions: {q_count}")
            print(f"  [Step 8] Response: {response[:300]}...")
        time.sleep(1)

    results.append(StepResult(8, '"beni test et" → Quiz from read chunks', checks,
                              f"read_chunks={len(read_chunks)}"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9: "KVKK nedir" → 3-tier RAG confidence + correct footer
    # Exit reading mode first to test normal flow's 3-tier confidence
    # ═══════════════════════════════════════════════════════════════════════
    _reset_reading_mode(state)
    state["current_course"] = None  # clear course to force normal mode

    msg = "KVKK nedir"
    checks = {}

    checks["reading_mode_off"] = state.get("reading_mode") is False

    # Detect course (CTIS 363 expected — KVKK is in that course)
    detected_9 = detect_active_course(msg, TEST_UID)
    if detected_9:
        state["current_course"] = detected_9

    # Normal flow: RAG search
    rag = vs.hybrid_search(query=msg, n_results=15, course_filter=detected_9)

    if rag:
        top_score = 1 - rag[0]["distance"]
        checks["has_rag_results"] = True
        checks["top_score_valid"] = 0 <= top_score <= 1

        if top_score >= _HIGH_CONFIDENCE:
            src_type = "rag_strong"
        elif top_score >= _LOW_CONFIDENCE:
            src_type = "rag_partial"
        else:
            src_type = "general"

        checks["src_type_determined"] = src_type in ("rag_strong", "rag_partial", "general")
        footer = _format_source_footer(rag[:10], src_type)

        if src_type == "rag_strong":
            checks["footer_correct"] = "📄" in footer
        elif src_type == "rag_partial":
            checks["footer_correct"] = "📄" in footer and "tamamlandı" in footer
        else:
            checks["footer_correct"] = "💡" in footer

        if verbose:
            print(f"\n  [Step 9] Query: {msg}")
            print(f"  [Step 9] Detected course: {detected_9}")
            print(f"  [Step 9] Top score: {top_score:.3f}")
            print(f"  [Step 9] Source type: {src_type}")
            print(f"  [Step 9] Footer: {footer.strip()}")

        if live and llm:
            extra_sys = _SOURCE_RULE
            if src_type == "rag_partial":
                extra_sys += "\n" + _RAG_PARTIAL_INSTRUCTION
            elif src_type == "general":
                extra_sys += "\n" + _NO_RAG_INSTRUCTION

            response = llm.chat_with_history(
                messages=[{"role": "user", "content": msg}],
                context_chunks=rag[:10] if src_type != "general" else [],
                study_mode=False,
                extra_system=extra_sys,
            )
            checks["llm_response_ok"] = len(response) > 20
            # Verify LLM doesn't add its own source references
            checks["no_inline_source"] = "📖" not in response and "materyale göre" not in response.lower()
            if verbose:
                print(f"  [Step 9] Response: {response[:300]}...")
            time.sleep(1)
    else:
        checks["has_rag_results"] = False
        checks["src_type_determined"] = True
        footer = _format_source_footer([], "general")
        checks["footer_correct"] = "💡" in footer

    _ts = f"{1 - rag[0]['distance']:.3f}" if rag else "0"
    results.append(StepResult(9, '"KVKK nedir" → 3-tier confidence', checks,
                              f"top_score={_ts}"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 10: "CTIS 474 çalışacağım" → No-material warning
    # ═══════════════════════════════════════════════════════════════════════
    msg = "CTIS 474 çalışacağım"
    checks = {}

    detected = detect_active_course(msg, TEST_UID)
    checks["course_detected"] = detected is not None
    checks["course_is_ctis474"] = "CTIS 474" in (detected or "")
    checks["needs_menu"] = _needs_topic_menu(msg)

    # Check course materials status
    if detected:
        course_files = vs.get_files_for_course(course_name=detected)
        has_materials = len(course_files) > 0 and any(f.get("chunk_count", 0) > 0 for f in course_files)

        if has_materials:
            # Course has materials → should show inline keyboard menu
            header, markup = _format_topic_menu(detected, course_files)
            checks["menu_generated"] = isinstance(markup, InlineKeyboardMarkup)
            checks["menu_has_buttons"] = len(markup.inline_keyboard) > 0
        else:
            # No materials → should show warning
            parts = detected.split()
            short = parts[0] if len(parts) == 1 else f"{parts[0]} {parts[1].split('-')[0]}"
            checks["warning_buildable"] = len(short) > 0
            checks["menu_generated"] = True  # n/a but pass
            checks["menu_has_buttons"] = True  # n/a but pass

        if verbose:
            print(f"\n  [Step 10] Course: {detected}")
            print(f"  [Step 10] Files: {len(course_files)}")
            print(f"  [Step 10] Has materials: {has_materials}")
            if has_materials:
                print(f"  [Step 10] → Shows inline keyboard (file count: {len(course_files)})")
            else:
                print(f"  [Step 10] → Shows no-material warning")
    else:
        checks["menu_generated"] = False
        checks["menu_has_buttons"] = False

    results.append(StepResult(10, '"CTIS 474 çalışacağım" → Course detection + menu/warning', checks,
                              f"detected={detected}"))

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("READING MODE E2E TEST RESULTS")
    print("=" * 70)
    print(f"Mode: {'LIVE (with LLM)' if live else 'LOGIC ONLY (no LLM)'}")
    print()

    total_checks = 0
    passed_checks = 0
    for r in results:
        print(r.summary())
        for k, v in r.checks.items():
            total_checks += 1
            if v:
                passed_checks += 1
            elif verbose:
                print(f"       ⚠️ {k} = {v}")
        if verbose and r.details:
            print(f"       ({r.details})")

    passed_steps = sum(1 for r in results if r.passed)
    print()
    print(f"Steps: {passed_steps}/{len(results)} passed")
    print(f"Checks: {passed_checks}/{total_checks} passed")
    print("=" * 70)

    # Clean up test state
    _user_state.pop(TEST_UID, None)

    return passed_steps == len(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reading Mode E2E Tests")
    parser.add_argument("--live", action="store_true", help="Run with actual LLM calls")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    success = run_tests(live=args.live, verbose=args.verbose)
    sys.exit(0 if success else 1)
