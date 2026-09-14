"""
Golden Evaluation Set Builder for AppleSupport Customer Intent Dataset.

Stratified-samples 200 conversation pairs from the AppleSupport cleaned corpus,
auto-labels them using keyword heuristics, and saves as data/eval/golden_set.jsonl.

Labels are derived deterministically from the taxonomy defined in src/taxonomy/intents.json.
A human reviewer should verify and correct labels before using for final evaluation.
"""

import json
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INTENTS_PATH = Path("src/taxonomy/intents.json")
PAIRS_PATH = Path("data/processed/AppleSupport_pairs.parquet")
GOLDEN_SET_PATH = Path("data/eval/golden_set.jsonl")
SAMPLING_NOTES_PATH = Path("data/eval/golden_set_sampling_method.md")

TOTAL_EXAMPLES = 200  # Target golden set size


def load_taxonomy():
    with open(INTENTS_PATH) as f:
        return json.load(f)


def clean_text(text: str) -> str:
    t = re.sub(r"@[A-Za-z0-9_]+", "", str(text))
    t = re.sub(r"https?://\S+", "", t)
    return t.lower().strip()


def classify_by_keywords(text: str, taxonomy: dict) -> tuple[str, float]:
    """
    Heuristic keyword-based intent classification.
    Returns (intent_id, confidence_score).
    """
    cleaned = clean_text(text)
    scores = {}

    for intent in taxonomy["intents"]:
        iid = intent["id"]
        keywords = intent["keywords"]
        hits = sum(1 for kw in keywords if kw.lower() in cleaned)
        scores[iid] = hits / max(len(keywords), 1)

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score == 0:
        return "general_inquiry", 0.0
    return best_intent, round(best_score, 3)


def get_escalation_decision(intent_id: str, text: str, taxonomy: dict) -> tuple[str, str]:
    """
    Determine auto_handle vs escalate based on taxonomy policy and text signals.
    Returns (decision, reasoning).
    """
    policy = taxonomy["escalation_policy"]
    cleaned = clean_text(text)

    # Check escalation signals
    conditional_signals = policy["conditional_escalate_signals"]
    triggered_signals = []

    signal_patterns = [
        (r"\b(sue|legal|lawyer|attorney|court|consumer protection)\b", "legal action threat"),
        (r"\b(media|news|broadcast|journalist|reporter|twitter)\b", "media escalation threat"),
        (r"\b(lost all|deleted everything|data loss|lost data)\b", "data loss"),
        (r"\b(\$[0-9]{3,}|hundred|thousand) dollar\b", "financial loss > $100"),
        (r"\b(medical|hospital|doctor|emergency|life support|pacemaker)\b", "safety-critical device use"),
    ]
    for pattern, label in signal_patterns:
        if re.search(pattern, cleaned):
            triggered_signals.append(label)

    if intent_id in policy["auto_escalate_intents"]:
        reason = f"Intent '{intent_id}' requires human handling per policy"
        if triggered_signals:
            reason += f"; additional escalation signals: {', '.join(triggered_signals)}"
        return "escalate", reason

    if triggered_signals:
        return "escalate", f"Conditional escalation triggered: {', '.join(triggered_signals)}"

    return "auto_handle", f"Intent '{intent_id}' falls within auto-handle scope; no escalation signals detected"


def build_golden_set():
    print("Loading taxonomy...")
    taxonomy = load_taxonomy()

    print("Loading AppleSupport pairs...")
    df = pd.read_parquet(PAIRS_PATH)
    df = df.dropna(subset=["customer_message", "brand_reply"]).reset_index(drop=True)
    print(f"Total available pairs: {len(df):,}")

    # Classify all examples
    print("Applying heuristic classification...")
    df["intent"], df["confidence"] = zip(*df["customer_message"].apply(
        lambda t: classify_by_keywords(t, taxonomy)
    ))
    df["decision"], df["decision_reason"] = zip(*df.apply(
        lambda row: get_escalation_decision(row["intent"], row["customer_message"], taxonomy),
        axis=1
    ))

    # Stratified sampling: proportional across intents
    intent_counts = df["intent"].value_counts()
    print("\nIntent distribution in full corpus:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Per-intent target: proportional, minimum 10 per intent
    n_intents = len(taxonomy["intents"])
    min_per_intent = 10
    per_intent_target = max(min_per_intent, TOTAL_EXAMPLES // n_intents)

    sampled_frames = []

    # Deliberate inclusion of edge cases: low confidence examples
    low_confidence = df[df["confidence"] < 0.05].sample(
        min(15, len(df[df["confidence"] < 0.05])), random_state=42
    )
    low_confidence = low_confidence.copy()
    low_confidence["sample_note"] = "edge_case: low_keyword_confidence"
    sampled_frames.append(low_confidence)

    # Deliberately include escalation triggers
    escalation_examples = df[df["decision"] == "escalate"].sample(
        min(30, len(df[df["decision"] == "escalate"])), random_state=42
    )
    escalation_examples = escalation_examples.copy()
    escalation_examples["sample_note"] = "deliberate: escalation_sample"
    sampled_frames.append(escalation_examples)

    # Stratified per-intent sample
    already_sampled_ids = set(low_confidence.index) | set(escalation_examples.index)
    remaining = df[~df.index.isin(already_sampled_ids)]

    for intent_data in taxonomy["intents"]:
        iid = intent_data["id"]
        subset = remaining[remaining["intent"] == iid]
        n = min(per_intent_target, len(subset))
        if n > 0:
            s = subset.sample(n, random_state=42).copy()
            s["sample_note"] = f"stratified: {iid}"
            sampled_frames.append(s)

    golden_df = pd.concat(sampled_frames, ignore_index=True)
    golden_df = golden_df.drop_duplicates(subset=["thread_id"]).head(TOTAL_EXAMPLES).reset_index(drop=True)

    print(f"\nFinal golden set: {len(golden_df)} examples")
    print(f"Intent distribution in golden set:")
    print(golden_df["intent"].value_counts())
    print(f"\nDecision distribution:")
    print(golden_df["decision"].value_counts())

    # Save as JSONL
    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, row in golden_df.iterrows():
        records.append({
            "id": int(idx),
            "thread_id": int(row["thread_id"]),
            "customer_message": str(row["customer_message"]),
            "brand_reply_reference": str(row["brand_reply"]),
            "intent": row["intent"],
            "routing_decision": row["decision"],
            "routing_reason": row["decision_reason"],
            "confidence": float(row["confidence"]),
            "sample_note": row.get("sample_note", "stratified"),
            "label_method": "heuristic_keyword",
            "human_verified": False
        })

    with open(GOLDEN_SET_PATH, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved golden set to {GOLDEN_SET_PATH}")
    return len(records)


if __name__ == "__main__":
    n = build_golden_set()
    print(f"\nGolden set complete: {n} examples")
