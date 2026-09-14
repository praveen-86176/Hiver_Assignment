"""
Tests for Phase 3-6 modules:
- Retrieval
- Intent classification
- Policy routing
- Baselines
- Eval harness ROUGE

These tests are all offline (no API calls). Pipeline/LLM tests require GROQ_API_KEY.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# ─── Retrieval ──────────────────────────────────────────────────────────────

def test_clean_for_retrieval():
    from src.retrieval.retriever import clean_for_retrieval
    text = "@AppleSupport my battery https://t.co/abc is dying!!!"
    cleaned = clean_for_retrieval(text)
    assert "applesupp" not in cleaned
    assert "http" not in cleaned
    assert "battery" in cleaned


def test_format_grounding_context():
    from src.retrieval.retriever import format_grounding_context
    retrieved = [
        {"rank": 1, "similarity": 0.85, "customer_message": "my battery dies", "brand_reply": "Please DM us", "thread_id": 1},
        {"rank": 2, "similarity": 0.75, "customer_message": "battery draining", "brand_reply": "Try this fix", "thread_id": 2},
    ]
    ctx = format_grounding_context(retrieved)
    assert "Rank 1" in ctx or "Example 1" in ctx
    assert "DM" in ctx or "battery" in ctx.lower()
    assert len(ctx) <= 850  # Respects max_chars limit with some tolerance


# ─── Routing ────────────────────────────────────────────────────────────────

def test_route_auto_handle():
    from src.agent.route import route
    taxonomy = {
        "escalation_policy": {
            "auto_escalate_intents": ["apple_id_account", "hardware_repair", "app_store_billing"],
            "conditional_escalate_signals": [],
            "auto_handle_intents": ["battery_power"],
        }
    }
    result = route("battery_power", "my battery is dying fast", taxonomy)
    assert result["decision"] == "auto_handle"
    assert result["triggered_signals"] == []


def test_route_escalate_by_intent():
    from src.agent.route import route
    taxonomy = {
        "escalation_policy": {
            "auto_escalate_intents": ["apple_id_account", "hardware_repair", "app_store_billing"],
            "conditional_escalate_signals": [],
            "auto_handle_intents": [],
        }
    }
    result = route("hardware_repair", "my screen is cracked", taxonomy)
    assert result["decision"] == "escalate"


def test_route_escalate_by_signal():
    from src.agent.route import route
    taxonomy = {
        "escalation_policy": {
            "auto_escalate_intents": [],
            "conditional_escalate_signals": [],
            "auto_handle_intents": ["battery_power"],
        }
    }
    # Legal threat signal
    result = route("battery_power", "I will sue Apple if this isn't fixed", taxonomy)
    assert result["decision"] == "escalate"
    assert "legal_threat" in result["triggered_signals"]


def test_route_security_signal():
    from src.agent.route import route
    taxonomy = {
        "escalation_policy": {
            "auto_escalate_intents": [],
            "conditional_escalate_signals": [],
            "auto_handle_intents": ["general_inquiry"],
        }
    }
    result = route("general_inquiry", "my account was hacked and compromised by someone else", taxonomy)
    assert result["decision"] == "escalate"
    assert "security_compromise" in result["triggered_signals"]


# ─── Baselines ──────────────────────────────────────────────────────────────

def test_trivial_baseline_always_auto():
    from src.baselines.baselines import TrivialBaseline
    b = TrivialBaseline()
    result = b.run("some customer message")
    assert result["routing"]["decision"] == "auto_handle"
    assert len(result["reply"]) > 10


def test_keyword_baseline_battery():
    from src.baselines.baselines import KeywordBaseline
    b = KeywordBaseline()
    result = b.run("my iPhone battery is dying so fast")
    assert result["classification"]["intent"] == "battery_power"
    assert result["routing"]["decision"] == "auto_handle"


def test_keyword_baseline_escalate():
    from src.baselines.baselines import KeywordBaseline
    b = KeywordBaseline()
    result = b.run("my Apple ID is locked and I need to recover my account asap")
    assert result["classification"]["intent"] == "apple_id_account"
    assert result["routing"]["decision"] == "escalate"


# ─── Eval Harness (ROUGE) ──────────────────────────────────────────────────

def test_rouge1_recall_identical():
    from src.eval.harness import rouge1_recall
    score = rouge1_recall("the quick brown fox", "the quick brown fox")
    assert score == 1.0


def test_rouge1_recall_partial():
    from src.eval.harness import rouge1_recall
    score = rouge1_recall("quick fox", "the quick brown fox")
    assert 0.3 < score < 0.7


def test_rouge1_recall_empty_reference():
    from src.eval.harness import rouge1_recall
    score = rouge1_recall("some text", "")
    assert score == 0.0


# ─── Taxonomy ───────────────────────────────────────────────────────────────

def test_taxonomy_valid_json():
    taxonomy_path = Path("src/taxonomy/intents.json")
    assert taxonomy_path.exists(), "intents.json missing"
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    assert "intents" in taxonomy
    assert len(taxonomy["intents"]) >= 6
    for intent in taxonomy["intents"]:
        assert "id" in intent
        assert "label" in intent
        assert "keywords" in intent


def test_taxonomy_escalation_policy():
    taxonomy_path = Path("src/taxonomy/intents.json")
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    policy = taxonomy["escalation_policy"]
    assert "auto_escalate_intents" in policy
    assert "apple_id_account" in policy["auto_escalate_intents"]
    assert "hardware_repair" in policy["auto_escalate_intents"]
    assert "app_store_billing" in policy["auto_escalate_intents"]
