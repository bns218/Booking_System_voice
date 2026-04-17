"""
tests/test_agent.py
-------------------
Unit tests for persona prompt building and booking extraction logic.
Run with: pytest tests/
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import build_persona_prompt, is_confirmation


# ── Persona prompt ────────────────────────────────────────────────────────────

def test_persona_includes_restaurant_name():
    prompt = build_persona_prompt()
    assert "Bella Notte" in prompt


def test_persona_includes_agent_name():
    prompt = build_persona_prompt()
    assert "Sofia" in prompt


def test_persona_includes_rag_context():
    prompt = build_persona_prompt(rag_context="Truffle risotto is our signature dish.")
    assert "Truffle risotto" in prompt


def test_persona_without_rag_has_no_knowledge_section():
    prompt = build_persona_prompt(rag_context="")
    assert "KNOWLEDGE BASE" not in prompt


def test_persona_contains_confirmation_instruction():
    prompt = build_persona_prompt()
    assert "your booking is confirmed" in prompt.lower()


# ── Confirmation detection ────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "your booking is confirmed",
    "Your Booking Is Confirmed — ref BN-1234-2026",
    "reservation is confirmed",
    "Booking has been confirmed, thank you!",
    "You're all set for Saturday evening.",
    "reference number BN-5678-2026",
    "We look forward to seeing you this Saturday!",
])
def test_is_confirmation_true(text):
    assert is_confirmation(text) is True


@pytest.mark.parametrize("text", [
    "Would you like to make a reservation?",
    "Could I take your name please?",
    "We have availability at 7pm.",
    "Thank you for calling Bella Notte.",
    "",
])
def test_is_confirmation_false(text):
    assert is_confirmation(text) is False
