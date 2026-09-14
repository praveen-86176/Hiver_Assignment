"""
Phase 5: Baselines.

Implements two baselines to compare against the full agent:

1. Trivial Baseline: Always returns the single most-common AppleSupport reply template.
   Routing: Always auto_handle.

2. Keyword Baseline: Uses keyword matching for intent classification + canned response per intent.
   Routing: Policy-based (same as agent), but no LLM, no retrieval.

These baselines set the floor for evaluation. The agent should beat both on quality metrics.
"""

import json
import re
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.agent.route import route, load_taxonomy
from scripts.build_golden_set import classify_by_keywords

PAIRS_PATH = Path("data/processed/AppleSupport_pairs.parquet")
TAXONOMY_PATH = Path("src/taxonomy/intents.json")

# Hardcoded canned responses per intent (mirrors real AppleSupport Twitter patterns)
CANNED_REPLIES = {
    "battery_power": (
        "We're sorry about your battery issues! Check your Battery settings for usage details. "
        "If you need more help, please DM us with your device model and iOS version."
    ),
    "ios_update_glitch": (
        "We're aware of some issues after recent updates. Try restarting your device. "
        "If the problem continues, please DM us and we'll help troubleshoot."
    ),
    "apple_id_account": (
        "We can help you regain access to your Apple ID. "
        "Please DM us so we can securely verify your account and walk you through recovery."
    ),
    "hardware_repair": (
        "Sorry to hear about the physical damage. "
        "Please DM us with your device model and we'll help you find the best repair option."
    ),
    "app_store_billing": (
        "We understand billing concerns can be frustrating. "
        "Please DM us with your purchase details and we'll look into this right away."
    ),
    "connectivity_network": (
        "Sorry about the connection issues! Try toggling Airplane Mode or resetting Network Settings. "
        "Still happening? DM us and we'll help further."
    ),
    "device_setup_restore": (
        "We're here to help with your device setup! "
        "Please DM us with your device model and what step you're stuck on."
    ),
    "performance_storage": (
        "Sorry your device is running slowly! Try restarting and clearing unused apps. "
        "DM us if the issue persists and we'll assist further."
    ),
    "general_inquiry": (
        "Thanks for reaching out! We'd love to help. "
        "Could you share a bit more detail about your issue? DM us and we'll get to the bottom of it."
    ),
}

TRIVIAL_REPLY = (
    "Thanks for reaching out to Apple Support! "
    "Please DM us with your device details and we'll help you resolve this as quickly as possible."
)


class TrivialBaseline:
    """
    Always returns the same generic reply, always routes to auto_handle.
    Represents the 'do nothing, just answer with a template' floor.
    """

    def run(self, customer_message: str) -> dict[str, Any]:
        return {
            "customer_message": customer_message,
            "classification": {"intent": "general_inquiry", "confidence": 0.0, "reasoning": "trivial_baseline"},
            "routing": {"decision": "auto_handle", "triggered_signals": [], "policy_basis": "trivial_baseline"},
            "reply": TRIVIAL_REPLY,
            "character_count": len(TRIVIAL_REPLY),
            "baseline": "trivial",
        }


class KeywordBaseline:
    """
    Keyword-based classifier + policy-based routing + canned reply per intent.
    No LLM, no retrieval. Deterministic and reproducible.
    """

    def __init__(self):
        with open(TAXONOMY_PATH) as f:
            self.taxonomy = json.load(f)

    def run(self, customer_message: str) -> dict[str, Any]:
        intent, confidence = classify_by_keywords(customer_message, self.taxonomy)
        routing = route(intent, customer_message, taxonomy=self.taxonomy)
        reply = CANNED_REPLIES.get(intent, CANNED_REPLIES["general_inquiry"])

        if routing["decision"] == "escalate":
            reply = "[ESCALATED: Routed to human agent]"

        return {
            "customer_message": customer_message,
            "classification": {"intent": intent, "confidence": confidence, "reasoning": "keyword_match"},
            "routing": routing,
            "reply": reply,
            "character_count": len(reply),
            "baseline": "keyword",
        }


if __name__ == "__main__":
    test_msg = "@AppleSupport my battery is dying so fast after the iOS 11 update!"

    trivial = TrivialBaseline()
    print("Trivial baseline:")
    r = trivial.run(test_msg)
    print(f"  Intent: {r['classification']['intent']}, Reply: {r['reply'][:80]}")

    keyword = KeywordBaseline()
    print("\nKeyword baseline:")
    r = keyword.run(test_msg)
    print(f"  Intent: {r['classification']['intent']}, Conf: {r['classification']['confidence']}, Reply: {r['reply'][:80]}")
