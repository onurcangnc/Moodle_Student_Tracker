"""
Source Summary Service — KATMAN 2
==================================
Generates, stores, and loads structured teaching summaries for source files.
Summaries are JSON with: overview, sections, key_concepts, cross_references.

Integration points:
- After document upload → generate_source_summary()
- After Moodle sync → generate_missing_summaries()
- Agent read_source tool → load_source_summary()
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from core import config as core_config

logger = logging.getLogger(__name__)

SUMMARY_DIR = core_config.data_dir / "source_summaries"

SUMMARY_GENERATION_PROMPT = """Bu bir üniversite ders materyali. Tamamını oku ve aşağıdaki JSON formatında
detaylı bir öğretim özeti oluştur.

ÖNEMLİ:
- Materyal dilinde yaz (Türkçe veya İngilizce — hangisindeyse)
- Bölüm ayrımlarını içerikten çıkar
- Her bölümün anahtar kavramlarını listele
- Bölümler arası ilişkileri (cross-references) mutlaka belirt
- Önerilen çalışma sırası ekle

JSON FORMAT:
{
  "overview": "2-3 cümle genel özet",
  "sections": [
    {
      "title": "Bölüm başlığı",
      "summary": "3-5 cümle bölüm özeti",
      "key_concepts": ["kavram1", "kavram2"]
    }
  ],
  "cross_references": [
    "Bölüm 1'deki X kavramı → Bölüm 3'te Y olarak detaylandırılıyor"
  ],
  "prerequisite_knowledge": ["Gerekli ön bilgiler"],
  "difficulty": "beginner|intermediate|advanced",
  "suggested_study_order": "Önerilen çalışma sırası açıklaması"
}

SADECE JSON döndür, başka bir şey yazma.

MATERYALİN İÇERİĞİ:
"""


def _safe_filename(course: str, filename: str) -> str:
    """Generate filesystem-safe summary filename."""
    safe_course = re.sub(r"[^\w\-]", "_", course)
    safe_file = re.sub(r"[^\w\-.]", "_", Path(filename).stem)
    return f"{safe_course}__{safe_file}.json"


def summary_exists(filename: str, course: str) -> bool:
    """Check if a summary already exists for this source file."""
    path = SUMMARY_DIR / _safe_filename(course, filename)
    return path.exists()


def load_source_summary(filename: str, course: str) -> dict | None:
    """Load a saved summary. Returns None if not found."""
    path = SUMMARY_DIR / _safe_filename(course, filename)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load summary %s: %s", path.name, exc)
        return None


def save_source_summary(filename: str, course: str, summary: dict) -> Path:
    """Save a summary to disk."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / _safe_filename(course, filename)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _make_fallback_summary(filename: str, course: str, chunk_count: int) -> dict:
    """Create a minimal fallback summary when LLM generation fails."""
    return {
        "overview": f"{filename} materyalinin detaylı özeti henüz oluşturulamamıştır.",
        "sections": [],
        "cross_references": [],
        "prerequisite_knowledge": [],
        "difficulty": "intermediate",
        "suggested_study_order": "",
        "source": filename,
        "course": course,
        "chunk_count": chunk_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fallback": True,
    }


def generate_source_summary(filename: str, course: str, chunk_texts: list[str]) -> dict:
    """
    Generate a teaching summary for a source file.

    Uses engine.complete(task="summary") for LLM call — synchronous.
    Called from background threads (asyncio.to_thread).

    Args:
        filename: Source file name (e.g. "lecture_05_privacy.pdf")
        course: Course full name
        chunk_texts: List of chunk text content, ordered by chunk_index
    """
    from bot.state import STATE

    if not chunk_texts:
        logger.warning("No chunks to summarize for %s/%s", course, filename)
        return _make_fallback_summary(filename, course, 0)

    # Combine chunk texts
    combined = ""
    for i, text in enumerate(chunk_texts):
        combined += f"\n\n--- Parça {i + 1} ---\n{text}"

    # Truncate if too long (avoid exceeding context window)
    # ~500K chars ≈ ~125K tokens — safe for Gemini Flash 1M context
    if len(combined) > 500_000:
        combined = combined[:500_000] + "\n\n[... kısaltıldı ...]"

    prompt = SUMMARY_GENERATION_PROMPT + combined

    llm = STATE.llm
    if llm is None:
        logger.error("LLM engine not available for summary generation")
        return _make_fallback_summary(filename, course, len(chunk_texts))

    try:
        response = llm.engine.complete(
            task="summary",
            system="Sen bir akademik içerik özetleyicisin. SADECE istenen JSON formatında yanıt ver.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )

        summary = _parse_llm_json(response)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("Summary JSON parse failed for %s: %s", filename, exc)
        summary = _make_fallback_summary(filename, course, len(chunk_texts))
    except Exception as exc:
        logger.error("Summary generation LLM call failed for %s: %s", filename, exc, exc_info=True)
        summary = _make_fallback_summary(filename, course, len(chunk_texts))

    # Add metadata
    summary["source"] = filename
    summary["course"] = course
    summary["chunk_count"] = len(chunk_texts)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary.pop("fallback", None)

    # Save
    path = save_source_summary(filename, course, summary)
    logger.info("Summary generated: %s (%d chunks)", path.name, len(chunk_texts))

    return summary


def generate_missing_summaries() -> int:
    """
    Generate summaries for all indexed files that don't have one yet.
    Returns number of summaries generated.

    Synchronous — meant to be called via asyncio.to_thread().
    """
    from bot.state import STATE

    store = STATE.vector_store
    if store is None:
        return 0

    generated = 0
    stats = store.get_stats()
    courses = stats.get("courses", [])

    for course in courses:
        files = store.get_files_for_course(course)
        for file_info in files:
            filename = file_info.get("filename", "")
            if not filename or summary_exists(filename, course):
                continue

            # Get all chunks for this file
            chunks = store.get_file_chunks(filename, max_chunks=0)
            if not chunks:
                continue

            chunk_texts = [c.get("text", "") for c in chunks if c.get("text", "").strip()]
            if not chunk_texts:
                continue

            try:
                generate_source_summary(filename, course, chunk_texts)
                generated += 1
            except Exception as exc:
                logger.error("Summary generation failed for %s/%s: %s", course, filename, exc, exc_info=True)

    logger.info("Missing summaries generation complete: %d new summaries", generated)
    return generated


def list_summaries(course: str | None = None) -> list[dict]:
    """List all available summaries, optionally filtered by course."""
    if not SUMMARY_DIR.exists():
        return []

    summaries = []
    for path in SUMMARY_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if course and course.lower() not in data.get("course", "").lower():
                continue
            summaries.append({
                "filename": data.get("source", path.stem),
                "course": data.get("course", ""),
                "overview": data.get("overview", ""),
                "sections": len(data.get("sections", [])),
                "chunk_count": data.get("chunk_count", 0),
                "difficulty": data.get("difficulty", ""),
                "generated_at": data.get("generated_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return summaries
