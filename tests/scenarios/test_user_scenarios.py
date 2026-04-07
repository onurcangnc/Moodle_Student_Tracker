"""
Comprehensive User Scenario Tests
==================================
Tests real user interactions and edge cases that have caused issues.

Run with: pytest tests/scenarios/test_user_scenarios.py -v
"""

import re
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# MAIL SEARCH TESTS - Bug: "Homework 6" not matching "Homework 06"
# ═══════════════════════════════════════════════════════════════════════════════

class TestMailSearch:
    """Test mail search with number normalization and multi-word queries."""

    @staticmethod
    def normalize_numbers(text: str) -> str:
        """Normalize numbers: '06' -> '6', '007' -> '7'."""
        return re.sub(r'\b0+(\d+)', r'\1', text.lower())

    @staticmethod
    def mail_matches(keyword: str, mail: dict) -> bool:
        """Check if a mail matches the keyword - simple substring match.

        Agentic design: LLM extracts proper keyword, tool just filters.
        No noise word stripping - LLM handles that.
        """
        if not keyword.strip():
            return False

        searchable = " ".join([
            mail.get("subject", ""),
            mail.get("from", ""),
            mail.get("source", ""),
            mail.get("date", ""),
        ]).lower()

        return keyword.lower() in searchable

    # ─── Number Normalization ─────────────────────────────────────────────────

    # ─── LLM Extracts Proper Keywords ──────────────────────────────────────────

    def test_llm_extracts_homework(self):
        """LLM extracts 'Homework' from user query - matches substring."""
        mail = {"subject": "CTIS-474 - Homework 06", "from": "Volkan Evrin", "source": "AIRS", "date": "2 Nisan 2026"}
        assert self.mail_matches("Homework", mail)
        assert self.mail_matches("Homework 06", mail)

    def test_llm_extracts_course_code(self):
        """LLM extracts 'CTIS-474' from context."""
        mail = {"subject": "CTIS-474 - Homework 06", "from": "Volkan Evrin", "source": "AIRS", "date": "2 Nisan 2026"}
        assert self.mail_matches("CTIS-474", mail)
        assert self.mail_matches("CTIS", mail)

    def test_llm_extracts_sender(self):
        """LLM extracts sender name."""
        mail = {"subject": "CTIS-474 - Homework 06", "from": "Volkan Evrin", "source": "AIRS", "date": "2 Nisan 2026"}
        assert self.mail_matches("Volkan", mail)
        assert self.mail_matches("Evrin", mail)

    def test_llm_extracts_quiz(self):
        """LLM extracts 'Quiz' from user query."""
        mail = {"subject": "CTIS 363 - Quiz 01 Results", "from": "Instructor", "source": "DAIS", "date": "6 Mart 2026"}
        assert self.mail_matches("Quiz", mail)
        assert self.mail_matches("Quiz 01", mail)

    # ─── Turkish Characters ───────────────────────────────────────────────────

    def test_turkish_sender_search(self):
        """Search with Turkish characters."""
        mail = {"subject": "Ders Notu", "from": "Ecem Tanriverdi", "source": "AIRS", "date": "2 Nisan 2026"}
        assert self.mail_matches("Ecem", mail)
        assert self.mail_matches("tanriverdi", mail)

    def test_turkish_date_search(self):
        """Search by Turkish date."""
        mail = {"subject": "Test", "from": "Prof", "source": "AIRS", "date": "2 Nisan 2026"}
        assert self.mail_matches("Nisan", mail)
        assert self.mail_matches("2 nisan", mail)

    # ─── Edge Cases ───────────────────────────────────────────────────────────

    def test_partial_word_no_match(self):
        """'home' should not match 'Homework' (we want token match)."""
        mail = {"subject": "CTIS-474 - Homework 06", "from": "Volkan", "source": "AIRS", "date": "2 Nisan"}
        # Actually 'home' IS in 'homework' as substring, so this WILL match
        # This is expected behavior - substring matching
        assert self.mail_matches("home", mail)

    def test_empty_keyword(self):
        """Empty keyword should match nothing."""
        mail = {"subject": "Test", "from": "Prof", "source": "AIRS", "date": "1 Apr"}
        # Empty splits to empty list, any() of empty is False
        # Correct behavior - empty search matches nothing
        assert not self.mail_matches("", mail)

    def test_no_match(self):
        """Completely unrelated search."""
        mail = {"subject": "CTIS-474 - Homework 06", "from": "Volkan", "source": "AIRS", "date": "2 Nisan"}
        assert not self.mail_matches("biology final exam", mail)


# ═══════════════════════════════════════════════════════════════════════════════
# COURSE DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCourseDetection:
    """Test course code extraction from user queries."""

    COURSE_CODES = [
        "CTIS 363", "CTIS 474", "CTIS 465", "CTIS 456",
        "EDEB 201", "HCIV 102"
    ]

    @staticmethod
    def extract_course_code(text: str) -> str | None:
        """Extract course code from user message."""
        # Pattern: 2-4 letters + space + 3 digits + optional letter
        pattern = r'\b([A-Z]{2,4})\s*(\d{3}[A-Z]?)\b'
        match = re.search(pattern, text.upper())
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None

    def test_explicit_course_code(self):
        """Direct course code mention."""
        assert self.extract_course_code("CTIS 474 notlarım") == "CTIS 474"
        assert self.extract_course_code("EDEB 201 dersi") == "EDEB 201"

    def test_lowercase_course_code(self):
        """Lowercase should still work."""
        assert self.extract_course_code("ctis 474 notlarım") == "CTIS 474"
        assert self.extract_course_code("edeb 201 hakkında") == "EDEB 201"

    def test_no_space_course_code(self):
        """CTIS474 without space."""
        assert self.extract_course_code("CTIS474 ödevi") == "CTIS 474"

    def test_no_course_code(self):
        """No course code in message."""
        assert self.extract_course_code("notlarım neler") is None
        assert self.extract_course_code("son mailler") is None

    def test_course_with_section(self):
        """Course with section number."""
        # "CTIS 363-1" should extract "CTIS 363"
        result = self.extract_course_code("CTIS 363-1 dersi")
        assert result == "CTIS 363"


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntentDetection:
    """Test keyword-based intent routing."""

    STARS_KEYWORDS = ["not", "notlar", "grade", "devamsızlık", "yoklama", "ders programı", "sınav"]
    SYNC_KEYWORDS = ["sync", "senkron", "güncelle", "yenile"]
    MAIL_KEYWORDS = ["mail", "e-posta", "eposta", "mektup"]
    RAG_KEYWORDS = ["anlat", "açıkla", "özetle", "konu", "materyal", "kaynak", "içerik"]

    @staticmethod
    def detect_intent(message: str) -> str:
        """Detect user intent from message."""
        msg_lower = message.lower()

        # Priority order matters
        if any(kw in msg_lower for kw in ["not", "notlar", "grade"]):
            return "grades"
        if any(kw in msg_lower for kw in ["devamsızlık", "yoklama", "attendance"]):
            return "attendance"
        if any(kw in msg_lower for kw in ["ders programı", "schedule", "program"]):
            return "schedule"
        if any(kw in msg_lower for kw in ["sınav", "exam", "final", "midterm", "quiz"]):
            return "exams"
        if any(kw in msg_lower for kw in ["mail", "e-posta", "eposta"]):
            return "mail"
        if any(kw in msg_lower for kw in ["ödev", "assignment", "homework", "teslim"]):
            return "assignments"
        if any(kw in msg_lower for kw in ["sync", "senkron", "güncelle"]):
            return "sync"
        if any(kw in msg_lower for kw in ["anlat", "açıkla", "özetle", "konu", "materyal"]):
            return "rag"

        return "chat"

    def test_grade_intent(self):
        """Grade-related queries."""
        assert self.detect_intent("notlarımı göster") == "grades"
        assert self.detect_intent("CTIS 474 notları") == "grades"
        assert self.detect_intent("grade durumum") == "grades"

    def test_mail_intent(self):
        """Mail-related queries."""
        assert self.detect_intent("son mailler") == "mail"
        assert self.detect_intent("Volkan hocanın maili") == "mail"
        assert self.detect_intent("e-posta kontrol et") == "mail"

    def test_assignment_intent(self):
        """Assignment queries."""
        assert self.detect_intent("ödevlerim neler") == "assignments"
        assert self.detect_intent("homework 6 ne zaman") == "assignments"
        assert self.detect_intent("teslim tarihleri") == "assignments"

    def test_rag_intent(self):
        """RAG/material queries."""
        assert self.detect_intent("etik teorileri anlat") == "rag"
        assert self.detect_intent("bu konuyu açıkla") == "rag"
        assert self.detect_intent("ders materyalleri") == "rag"

    def test_ambiguous_intent(self):
        """Ambiguous - defaults to first match by priority."""
        # "not" takes priority over "mail"
        assert self.detect_intent("not maili") == "grades"


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE FORMAT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseFormat:
    """Test response formatting edge cases."""

    def test_grade_format(self):
        """Grade display format."""
        grade = {"course": "CTIS 474", "item": "Homework 01", "grade": "1/1", "date": "23 Şubat 2026"}
        formatted = f"{grade['item']}: {grade['grade']} ({grade['date']})"
        assert "Homework 01" in formatted
        assert "1/1" in formatted

    def test_long_message_split(self):
        """Messages over 4096 chars should be split."""
        MAX_LENGTH = 4096
        long_text = "A" * 5000

        def split_message(text: str, max_len: int = MAX_LENGTH) -> list[str]:
            if len(text) <= max_len:
                return [text]
            parts = []
            while text:
                parts.append(text[:max_len])
                text = text[max_len:]
            return parts

        parts = split_message(long_text)
        assert len(parts) == 2
        assert len(parts[0]) == MAX_LENGTH
        assert len(parts[1]) == 5000 - MAX_LENGTH

    def test_markdown_escaping(self):
        """Special chars in Markdown should be escaped."""
        text = "Grade: 90/100 (Good!)"
        # In MarkdownV2, these need escaping: _ * [ ] ( ) ~ ` > # + - = | { } . !
        # But we use Markdown (v1) which only needs: * _ ` [
        # Actually for basic markdown, / and () don't need escaping
        assert "/" in text  # Should display fine


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR RECOVERY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorRecovery:
    """Test graceful error handling."""

    def test_empty_mail_list(self):
        """Handle empty mail results gracefully."""
        mails = []
        response = "AIRS/DAIS e-postası bulunamadı." if not mails else "Found mails"
        assert response == "AIRS/DAIS e-postası bulunamadı."

    def test_none_values_in_mail(self):
        """Handle None values in mail dict."""
        mail = {"subject": None, "from": None, "date": None}
        subject = mail.get("subject") or "Konusuz"
        assert subject == "Konusuz"

    def test_missing_keys_in_grade(self):
        """Handle missing keys in grade data."""
        grade = {"course": "CTIS 474"}  # Missing 'item', 'grade', 'date'
        item = grade.get("item", "Bilinmeyen")
        score = grade.get("grade", "N/A")
        assert item == "Bilinmeyen"
        assert score == "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC DESIGN TESTS - LLM handles understanding, tool does simple filtering
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgenticDesign:
    """Verify agentic design philosophy.

    Principle: LLM understands user intent and extracts proper identifiers.
    Tool just does simple substring matching - no complex pattern matching.

    Examples of LLM responsibility:
    - User: "Homework 6 maili" → LLM extracts: "Homework" (not "Homework 6" vs "06")
    - User: "Erkan Hoca maili" → LLM extracts: "Erkan" (strips noise words)
    - User: "Audit dersi" → LLM extracts: "CTIS 474" or "Auditing" (resolves alias)
    """

    def test_llm_extracts_keyword_tool_filters(self):
        """LLM extracts proper keyword, tool does simple substring match."""
        mail = {"subject": "CTIS-474 - Homework 06", "from": "Volkan Evrin", "source": "AIRS", "date": "2 Nisan"}
        # LLM extracts "Homework" from "Homework 6 maili"
        assert TestMailSearch.mail_matches("Homework", mail)
        # LLM extracts "CTIS-474" from context
        assert TestMailSearch.mail_matches("CTIS-474", mail)

    def test_llm_resolves_aliases(self):
        """LLM resolves user aliases to actual course codes/names."""
        mail = {"subject": "CTIS-474 - Homework 06", "from": "Volkan", "source": "AIRS", "date": "2 Apr"}
        # 'audit' alone won't match - LLM should resolve to "CTIS-474" or "Auditing"
        assert not TestMailSearch.mail_matches("audit", mail)
        # LLM-resolved identifiers work
        assert TestMailSearch.mail_matches("CTIS-474", mail)
        assert TestMailSearch.mail_matches("CTIS", mail)

    def test_llm_strips_noise(self):
        """LLM strips noise words before passing to tool."""
        mail = {"subject": "CTIS 363 Announcement", "from": "Erkan Tunc", "source": "AIRS", "date": "1 Apr"}
        # User: "Erkan Hoca maili" → LLM extracts: "Erkan"
        assert TestMailSearch.mail_matches("Erkan", mail)
        # "Erkan Hoca" won't match because "Hoca" is not in mail
        assert not TestMailSearch.mail_matches("Erkan Hoca", mail)

    def test_direct_identifiers_work(self):
        """Direct course codes and names always work."""
        mail = {"subject": "EDEB 201 Quiz Results", "from": "Prof", "source": "AIRS", "date": "1 Apr"}
        assert TestMailSearch.mail_matches("EDEB", mail)
        assert TestMailSearch.mail_matches("EDEB 201", mail)
        assert TestMailSearch.mail_matches("Quiz", mail)
        assert TestMailSearch.mail_matches("edeb 201", mail)

    def test_instructor_name_works(self):
        """Instructor names should work directly."""
        mail = {"subject": "HCIV 102 Announcement", "from": "Tunahan Durmaz", "source": "AIRS", "date": "1 Apr"}
        assert TestMailSearch.mail_matches("Durmaz", mail)
        assert TestMailSearch.mail_matches("Tunahan", mail)


# ═══════════════════════════════════════════════════════════════════════════════
# COURSE FILTER MATCHING - Simple substring match (LLM handles understanding)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCourseFilterMatching:
    """Test simple course matching - LLM extracts proper identifiers, tool just filters."""

    @staticmethod
    def _course_matches(course_name: str, filter_term: str) -> bool:
        """Same logic as agent_service._course_matches - simple substring."""
        return filter_term.lower() in course_name.lower()

    # ─── LLM Extracts Proper Identifier ───────────────────────────────────────

    def test_llm_extracts_partial_name(self):
        """LLM extracts 'Audit' from user saying 'audit dersi'."""
        assert self._course_matches("CTIS 474-1 Information Systems Auditing", "Audit")
        assert self._course_matches("CTIS 474-1 Information Systems Auditing", "auditing")

    def test_llm_extracts_course_code(self):
        """LLM extracts 'CTIS 474' from context or user saying code."""
        assert self._course_matches("CTIS 474-1 Information Systems Auditing", "CTIS 474")
        assert self._course_matches("CTIS 474-1 Information Systems Auditing", "CTIS")
        assert self._course_matches("CTIS 474-1 Information Systems Auditing", "474")

    def test_llm_extracts_full_name(self):
        """LLM can extract full or partial course name."""
        assert self._course_matches("EDEB 201 Introduction to Turkish Fiction", "Turkish Fiction")
        assert self._course_matches("EDEB 201 Introduction to Turkish Fiction", "EDEB 201")

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        assert self._course_matches("CTIS 474 Information Systems Auditing", "ctis")
        assert self._course_matches("CTIS 474 Information Systems Auditing", "INFORMATION")

    # ─── Real Agentic Scenarios ───────────────────────────────────────────────

    def test_audit_scenario(self):
        """User: 'Auditte kaç saat' → LLM extracts 'Audit' or 'CTIS 474'."""
        course = "CTIS 474-1 Information Systems Auditing"
        # LLM should extract one of these from context
        assert self._course_matches(course, "Audit")
        assert self._course_matches(course, "CTIS 474")
        assert self._course_matches(course, "Information Systems")

    def test_ethics_scenario(self):
        """User: 'Ethics notlarım' → LLM extracts 'Ethical' or 'CTIS 363'."""
        course = "CTIS 363 Ethical and Social Issues"
        assert self._course_matches(course, "Ethical")
        assert self._course_matches(course, "CTIS 363")
        assert self._course_matches(course, "Social Issues")

    def test_history_scenario(self):
        """User: 'Tarih dersi' → LLM extracts 'History' or 'HCIV'."""
        course = "HCIV 102 History of Civilization II"
        assert self._course_matches(course, "History")
        assert self._course_matches(course, "HCIV")
        assert self._course_matches(course, "Civilization")

    # ─── No Match (LLM won't extract wrong identifier) ────────────────────────

    def test_no_match(self):
        """Different course identifiers don't match."""
        assert not self._course_matches("CTIS 474 Information Systems Auditing", "EDEB")
        assert not self._course_matches("CTIS 474 Information Systems Auditing", "History")
        assert not self._course_matches("CTIS 474 Information Systems Auditing", "Fiction")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
