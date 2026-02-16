"""Unit tests for document service wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from bot.services import document_service


def test_detect_course_delegates(monkeypatch):
    """Document service should delegate filename-course detection."""
    fake_legacy = SimpleNamespace(_detect_course=lambda filename: f"COURSE::{filename}")
    monkeypatch.setattr(document_service, "_legacy", lambda: fake_legacy)
    assert document_service.detect_course("ctis363_notes.pdf") == "COURSE::ctis363_notes.pdf"


def test_index_uploaded_file_delegates(monkeypatch, tmp_path):
    """Document indexing should pass through all expected arguments."""
    called: dict[str, object] = {}

    def fake_index(file_path, course_name, filename):
        called["file_path"] = file_path
        called["course_name"] = course_name
        called["filename"] = filename
        return 42

    fake_legacy = SimpleNamespace(_index_uploaded_file_sync=fake_index)
    monkeypatch.setattr(document_service, "_legacy", lambda: fake_legacy)
    path = tmp_path / "x.pdf"
    path.write_text("sample")
    result = document_service.index_uploaded_file(path, "CTIS 363", "x.pdf")
    assert result == 42
    assert called == {"file_path": path, "course_name": "CTIS 363", "filename": "x.pdf"}
