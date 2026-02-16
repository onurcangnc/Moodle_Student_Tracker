"""E2E-style scenario tests migrated to pytest markers."""

from __future__ import annotations

import pytest

from tests import test_e2e_scenarios as legacy_e2e


class FakeVectorStore:
    """Simple vector-store stub for scenario-level retrieval checks."""

    def hybrid_search(self, query: str, n_results: int = 15, course_filter: str | None = None):
        text = f"{query.lower()} privacy ethics veri koruma gdpr kvkk docker container microservice"
        return [
            {
                "id": "fake-1",
                "text": text,
                "metadata": {"filename": "fake.pdf", "course": course_filter or "GENERIC"},
                "distance": 0.10,
            }
        ]

    def query(self, query_text: str, n_results: int = 15, course_filter: str | None = None):
        return self.hybrid_search(query=query_text, n_results=n_results, course_filter=course_filter)


@pytest.mark.e2e
def test_course_detection_eval_subset_passes():
    """Legacy scenario evaluator should pass when detector returns expected courses."""
    scenarios = [s for s in legacy_e2e.SCENARIOS if s.expected_course][:8]
    index = {s.user_message: s.expected_course for s in scenarios}
    results = legacy_e2e.eval_course_detection(scenarios, detect_fn=lambda message: index.get(message))
    pass_rate = sum(1 for r in results if r["pass"]) / len(results)
    assert pass_rate == 1.0


@pytest.mark.e2e
def test_rag_eval_subset_has_nonzero_precision():
    """Legacy retrieval evaluator should produce measurable precision on subset."""
    scenarios = [s for s in legacy_e2e.SCENARIOS if s.expected_keywords][:10]
    results = legacy_e2e.eval_rag_retrieval(scenarios, vs=FakeVectorStore())
    avg_precision = sum(r["keyword_precision"] for r in results) / len(results)
    assert avg_precision > 0.15
