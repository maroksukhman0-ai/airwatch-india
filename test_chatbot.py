"""
tests/test_chatbot.py
---------------------
Unit tests for src/chatbot.py

The chatbot is pure Python (no DB or files needed) so tests are simple
and fast — just call answer() and check what comes back.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from chatbot import answer


class TestChatbotAnswer:

    # ── Returns a string ──────────────────────────────────────────────────────

    def test_returns_string(self):
        """answer() should always return a string, never None or an int."""
        result = answer("What does AQI mean?")
        assert isinstance(result, str)

    def test_returns_nonempty_string(self):
        """answer() should never return an empty string."""
        result = answer("hello")
        assert len(result.strip()) > 0

    # ── AQI question ─────────────────────────────────────────────────────────

    def test_aqi_mean_question(self):
        """Asking what AQI means should mention 'Air Quality Index'."""
        result = answer("What does AQI mean?")
        assert "Air Quality Index" in result or "AQI" in result

    def test_aqi_keyword_case_insensitive(self):
        """The match should work with upper or lower case."""
        lower = answer("what does aqi mean")
        upper = answer("What does AQI mean")
        # Both should give a non-default reply (contain 'Air' or 'AQI')
        assert "Air" in lower or "AQI" in lower
        assert "Air" in upper or "AQI" in upper

    # ── PM2.5 question ────────────────────────────────────────────────────────

    def test_pm25_question(self):
        """Asking about PM2.5 should mention particles or micrometres."""
        result = answer("Tell me about PM2.5")
        # The answer should contain something about particles or size
        assert any(word in result for word in ["particle", "micro", "PM2.5", "lung"])

    def test_pm25_lowercase(self):
        """pm2.5 (lowercase) should also be recognised."""
        result = answer("what is pm2.5?")
        assert len(result) > 20   # more than the default "I don't know" length

    # ── Safe / good AQI question ─────────────────────────────────────────────

    def test_safe_question(self):
        """Asking what is safe should mention 50 or 'Good'."""
        result = answer("What is a safe AQI level?")
        assert "50" in result or "Good" in result

    def test_good_keyword(self):
        """The word 'good' alone should trigger the safe-AQI answer."""
        result = answer("good")
        assert "50" in result or "Good" in result or "safe" in result.lower()

    # ── Unknown / fallback ───────────────────────────────────────────────────

    def test_unknown_question_returns_fallback(self):
        """An unrecognised question should return the fallback message."""
        result = answer("xyzzy nonsense question 12345")
        assert "Sorry" in result or "don't know" in result or "try" in result.lower()

    def test_empty_string(self):
        """An empty input should not crash; it should return the fallback."""
        result = answer("")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_whitespace_only(self):
        """Whitespace-only input should not crash."""
        result = answer("   ")
        assert isinstance(result, str)

    # ── Multiple questions / edge cases ──────────────────────────────────────

    @pytest.mark.parametrize("question,expected_fragment", [
        ("What does AQI mean?",       "Air Quality Index"),
        ("Tell me about pm2.5",       "2.5"),   # µm size appears in answer
        ("Is 30 a good AQI?",         "50"),    # Good is ≤ 50
        ("what is safe air quality",  "50"),
    ])
    def test_parametrized_known_questions(self, question, expected_fragment):
        """Parametrized check: each known question produces a sensible answer."""
        result = answer(question)
        assert expected_fragment in result, (
            f"Question: '{question}'\n"
            f"Expected '{expected_fragment}' in answer but got:\n{result}"
        )
