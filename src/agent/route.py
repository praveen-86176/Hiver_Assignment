"""
Phase 4: Routing / Escalation Decision Module.

Applies deterministic policy rules first (no LLM cost for clear cases),
then uses LLM judgment only for ambiguous cases.
"""

import json
import re
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TAXONOMY_PATH = Path("src/taxonomy/intents.json")

# Pre-compiled escalation signal patterns
ESCALATION_SIGNAL_PATTERNS = [
    (re.compile(r"\b(sue|lawsuit|legal|lawyer|attorney|court|consumer protection)\b", re.I), "legal_threat"),
    (re.compile(r"\b(media|journalist|reporter|news|broadcast|press)\b", re.I), "media_threat"),
    (re.compile(r"\b(data loss|lost all|lost everything|all data|backup failed.*lost)\b", re.I), "data_loss"),
    (re.compile(r"\$\s*[0-9]{3,}|\b(hundred|thousand)\s+dollars?\b", re.I), "financial_loss"),
    (re.compile(r"\b(medical|hospital|emergency|ambulance|life support|safety|pacemaker|911)\b", re.I), "safety_critical"),
    (re.compile(r"\b(hack|hacked|unauthorized|someone else|account.*(compromised|stolen|accessed))\b", re.I), "security_compromise"),
]


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def route(
    intent: str,
    customer_message: str,
    taxonomy: dict | None = None,
) -> dict[str, Any]:
    """
    Determine routing decision for a classified customer message.

    Applies deterministic rules in priority order:
    1. Hard escalate: intent is in always-escalate list
    2. Conditional escalate: message contains escalation signals
    3. Auto-handle: all clear

    Returns:
        dict with keys: decision, triggered_signals, policy_basis
    """
    if taxonomy is None:
        taxonomy = load_taxonomy()

    policy = taxonomy["escalation_policy"]
    auto_escalate_intents = set(policy["auto_escalate_intents"])

    # Check conditional signals in message text
    triggered_signals = []
    for pattern, signal_name in ESCALATION_SIGNAL_PATTERNS:
        if pattern.search(customer_message):
            triggered_signals.append(signal_name)

    # Priority 1: Intent-level escalation
    if intent in auto_escalate_intents:
        basis = f"Intent '{intent}' requires human handling (policy: always-escalate)"
        if triggered_signals:
            basis += f"; also: {', '.join(triggered_signals)}"
        return {
            "decision": "escalate",
            "triggered_signals": triggered_signals,
            "policy_basis": basis,
        }

    # Priority 2: Conditional signal escalation
    if triggered_signals:
        return {
            "decision": "escalate",
            "triggered_signals": triggered_signals,
            "policy_basis": f"Conditional escalation triggered: {', '.join(triggered_signals)}",
        }

    # Priority 3: Auto-handle
    return {
        "decision": "auto_handle",
        "triggered_signals": [],
        "policy_basis": f"Intent '{intent}' is auto-handleable; no escalation signals detected",
    }
