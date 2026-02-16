"""Document service wrappers.

Provides a stable service API for document indexing and summary generation.
"""

from __future__ import annotations

from pathlib import Path


def _legacy():
    from bot import legacy

    return legacy


async def generate_missing_summaries() -> int:
    """Generate missing per-file summaries and return generated count."""
    return await _legacy()._generate_missing_summaries()


def detect_course(filename: str) -> str | None:
    """Detect course from uploaded filename using existing heuristics."""
    return _legacy()._detect_course(filename)


def index_uploaded_file(file_path: Path, course_name: str, filename: str) -> int:
    """Index an uploaded file and return number of newly added chunks."""
    return _legacy()._index_uploaded_file_sync(file_path=file_path, course_name=course_name, filename=filename)
