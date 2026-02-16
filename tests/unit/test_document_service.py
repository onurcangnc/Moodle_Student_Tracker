"""Unit tests for document upload indexing service."""

from __future__ import annotations

from pathlib import Path

from bot.services import document_service
from bot.state import STATE


def test_detect_course_matches_short_name(monkeypatch):
    """Filename course detection should match known short course names."""
    fake_course = type("Course", (), {"course_id": "CTIS 363", "short_name": "CTIS363", "display_name": "CTIS 363"})()
    monkeypatch.setattr(document_service.user_service, "list_courses", lambda: [fake_course])
    assert document_service.detect_course("ctis363_notes_week1.pdf") == "CTIS 363"


def test_index_uploaded_file_adds_chunks(monkeypatch, tmp_path):
    """Indexing should process file and forward chunks into vector store."""
    captured: dict[str, object] = {}

    class FakeProcessor:
        def process_file(self, file_path: Path, course_name: str, module_name: str):
            captured["file_path"] = file_path
            captured["course_name"] = course_name
            captured["module_name"] = module_name
            return ["chunk-a", "chunk-b"]

    class FakeStore:
        def add_chunks(self, chunks):
            captured["chunks"] = list(chunks)

    monkeypatch.setattr(STATE, "processor", FakeProcessor())
    monkeypatch.setattr(STATE, "vector_store", FakeStore())

    target = tmp_path / "x.pdf"
    target.write_text("sample")
    count = document_service.index_uploaded_file(target, "CTIS 363", "x.pdf")
    assert count == 2
    assert captured["course_name"] == "CTIS 363"
    assert captured["module_name"] == "x.pdf"
    assert captured["chunks"] == ["chunk-a", "chunk-b"]
