"""Document indexing service for admin uploads."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bot.services import user_service
from bot.state import STATE

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize text for fuzzy filename/course matching."""
    return re.sub(r"[\W_]+", "", text.casefold())


def detect_course(filename: str) -> str | None:
    """Infer course name from filename using known Moodle course labels."""
    normalized_name = _normalize(filename)
    for course in user_service.list_courses():
        short = _normalize(course.short_name)
        display = _normalize(course.display_name)
        if short and short in normalized_name:
            return course.course_id
        if display and display in normalized_name:
            return course.course_id
    return None


def index_uploaded_file(file_path: Path, course_name: str, filename: str) -> int:
    """Process uploaded file, add chunks to vector store, then generate summary."""
    processor = STATE.processor
    vector_store = STATE.vector_store
    if processor is None or vector_store is None:
        raise RuntimeError("Document pipeline is not initialized.")

    chunks = processor.process_file(file_path=file_path, course_name=course_name, module_name=filename)
    if not chunks:
        return 0
    vector_store.add_chunks(chunks)

    # KATMAN 2: Generate teaching summary for newly uploaded file
    try:
        from bot.services.summary_service import generate_source_summary, summary_exists

        if not summary_exists(filename, course_name):
            chunk_texts = [c.text for c in chunks if c.text.strip()]
            if chunk_texts:
                generate_source_summary(filename, course_name, chunk_texts)
                logger.info("Summary generated for uploaded file: %s", filename)
    except Exception as exc:
        logger.warning("Summary generation skipped for %s: %s", filename, exc)

    return len(chunks)
