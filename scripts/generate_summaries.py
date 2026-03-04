#!/usr/bin/env python3
"""
Retroactive summary generation for existing indexed materials.
=============================================================
Generates KATMAN 2 teaching summaries for all indexed files
that don't already have a summary.

Usage (on server):
    cd /opt/moodle-bot
    python -m scripts.generate_summaries          # all courses
    python -m scripts.generate_summaries --course "CTIS 256"  # single course
    python -m scripts.generate_summaries --dry-run  # list what would be generated
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KATMAN 2 source summaries")
    parser.add_argument("--course", type=str, default=None, help="Filter by course name")
    parser.add_argument("--dry-run", action="store_true", help="List files without generating")
    args = parser.parse_args()

    # Initialize core components
    from core.llm_engine import MultiProviderEngine
    from core.vector_store import VectorStore

    logger.info("Loading vector store...")
    store = VectorStore()
    store.initialize()

    stats = store.get_stats()
    courses = stats.get("courses", [])
    logger.info("Found %d courses, %d total chunks", len(courses), stats.get("total_chunks", 0))

    if args.course:
        courses = [c for c in courses if args.course.lower() in c.lower()]
        if not courses:
            logger.error("No matching course for: %s", args.course)
            sys.exit(1)

    # Patch STATE so summary_service can access llm
    from bot.state import STATE

    if not args.dry_run:
        logger.info("Initializing LLM engine...")
        engine = MultiProviderEngine()
        STATE.llm = type("LLMShim", (), {"engine": engine})()  # type: ignore[assignment]
        STATE.vector_store = store

    from bot.services.summary_service import generate_source_summary, summary_exists

    total = 0
    skipped = 0
    generated = 0
    errors = 0

    for course in courses:
        files = store.get_files_for_course(course)
        logger.info("Course: %s — %d files", course, len(files))

        for file_info in files:
            filename = file_info.get("filename", "")
            if not filename:
                continue
            total += 1

            if summary_exists(filename, course):
                skipped += 1
                continue

            chunks = store.get_file_chunks(filename, max_chunks=0)
            chunk_texts = [c.get("text", "") for c in chunks if c.get("text", "").strip()]
            if not chunk_texts:
                logger.warning("  SKIP %s — no text chunks", filename)
                skipped += 1
                continue

            if args.dry_run:
                logger.info("  WOULD generate: %s (%d chunks)", filename, len(chunk_texts))
                continue

            try:
                logger.info("  Generating: %s (%d chunks)...", filename, len(chunk_texts))
                t0 = time.time()
                generate_source_summary(filename, course, chunk_texts)
                elapsed = time.time() - t0
                generated += 1
                logger.info("  OK — %.1fs", elapsed)

                # Rate limit: 15s between LLM calls
                if elapsed < 15:
                    time.sleep(15 - elapsed)
            except Exception as exc:
                errors += 1
                logger.error("  FAIL %s: %s", filename, exc)

    logger.info(
        "Done. total=%d, skipped=%d, generated=%d, errors=%d",
        total, skipped, generated, errors,
    )


if __name__ == "__main__":
    main()
