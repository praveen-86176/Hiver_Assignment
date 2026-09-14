"""
Phase 6: Evaluation Harness.

Evaluates the agent (and baselines) against the golden set using:
1. Intent classification accuracy (against heuristic labels in golden set)
2. Routing accuracy (auto_handle vs escalate)
3. Reply quality via LLM-as-judge (Groq, with structured rubric)
4. ROUGE-1 recall vs reference replies (fast, deterministic)

Outputs results to reports/eval_results.json and reports/eval_report.md.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from groq import Groq
import numpy as np

load_dotenv()

GOLDEN_SET_PATH = Path("data/eval/golden_set.jsonl")
RESULTS_PATH = Path("reports/eval_results.json")
REPORT_PATH = Path("reports/eval_report.md")


# ---------------------------------------------------------------------------
# ROUGE-1 recall (lightweight, no NLTK needed)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def rouge1_recall(candidate: str, reference: str) -> float:
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return 0.0
    cand_tokens = tokenize(candidate)
    overlap = cand_tokens & ref_tokens
    return len(overlap) / len(ref_tokens)


# ---------------------------------------------------------------------------
# LLM-as-Judge (Groq)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of customer support reply quality.
Given a customer message and a generated reply, score the reply on these dimensions (each 1-5):

- empathy: Does the reply acknowledge the customer's frustration or problem warmly?
- actionability: Does the reply give a clear next step or action for the customer?
- brand_voice: Does the reply sound like professional Apple Support (calm, concise, helpful)?
- relevance: Is the reply specific to the customer's actual problem (not generic)?

Respond ONLY with valid JSON:
{"empathy": <int 1-5>, "actionability": <int 1-5>, "brand_voice": <int 1-5>, "relevance": <int 1-5>, "comment": "<one sentence>"}"""


def llm_judge(
    customer_message: str,
    generated_reply: str,
    client: Groq,
    model: str,
) -> dict:
    """Use LLM-as-judge to score a generated reply."""
    prompt = f"""Customer message:
"{customer_message[:300]}"

Generated reply:
"{generated_reply[:300]}"

Score this reply:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw)

        for dim in ["empathy", "actionability", "brand_voice", "relevance"]:
            result[dim] = max(1, min(5, int(result.get(dim, 3))))

        result["overall"] = round(np.mean([result[d] for d in ["empathy", "actionability", "brand_voice", "relevance"]]), 2)
        return result

    except Exception as e:
        return {"empathy": 0, "actionability": 0, "brand_voice": 0, "relevance": 0, "overall": 0,
                "comment": f"Judge failed: {str(e)[:80]}"}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def load_golden_set() -> list[dict]:
    records = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def evaluate_system(
    system_name: str,
    run_fn,  # callable: customer_message -> pipeline result dict
    golden: list[dict],
    judge_client: Groq,
    judge_model: str,
    judge_every_n: int = 1,  # Judge every Nth sample (to control cost)
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a system (agent or baseline) against the golden set.

    Args:
        system_name: Name of the system being evaluated.
        run_fn: Function that takes a customer_message string and returns a result dict.
        golden: List of golden set records.
        judge_client: Groq client for LLM-as-judge.
        judge_model: Model name for LLM-as-judge.
        judge_every_n: Only run LLM judge on every Nth example (default 1 = all).
        max_samples: Limit evaluation to first N samples (for quick testing).

    Returns:
        Aggregated metrics dict.
    """
    if max_samples:
        golden = golden[:max_samples]

    print(f"\nEvaluating '{system_name}' on {len(golden)} examples...")

    results = []
    intent_correct = []
    routing_correct = []
    rouge_scores = []
    judge_scores = []

    for i, example in enumerate(golden):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(golden)}")

        result = run_fn(example["customer_message"])

        # Intent accuracy
        pred_intent = result["classification"]["intent"]
        true_intent = example["intent"]
        intent_ok = int(pred_intent == true_intent)
        intent_correct.append(intent_ok)

        # Routing accuracy
        pred_decision = result["routing"]["decision"]
        true_decision = example["routing_decision"]
        routing_ok = int(pred_decision == true_decision)
        routing_correct.append(routing_ok)

        # ROUGE-1
        r1 = rouge1_recall(result["reply"], example["brand_reply_reference"])
        rouge_scores.append(r1)

        # LLM-as-judge (skip for escalated cases)
        judge_score = None
        if (i % judge_every_n == 0) and result["routing"]["decision"] == "auto_handle":
            judge_score = llm_judge(
                example["customer_message"],
                result["reply"],
                judge_client,
                judge_model,
            )
            judge_scores.append(judge_score)

        results.append({
            "id": example["id"],
            "thread_id": example["thread_id"],
            "customer_message": example["customer_message"][:120],
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "intent_correct": intent_ok,
            "true_decision": true_decision,
            "pred_decision": pred_decision,
            "routing_correct": routing_ok,
            "reply": result.get("reply", "")[:200],
            "rouge1_recall": round(r1, 4),
            "judge": judge_score,
        })

    # Aggregated metrics
    n = len(results)
    n_judge = len(judge_scores)

    judge_agg = {}
    if judge_scores:
        for dim in ["empathy", "actionability", "brand_voice", "relevance", "overall"]:
            judge_agg[f"judge_{dim}_mean"] = round(float(np.mean([s[dim] for s in judge_scores])), 3)

    escalation_confusion = {
        "tp": sum(1 for r in results if r["true_decision"] == "escalate" and r["pred_decision"] == "escalate"),
        "fp": sum(1 for r in results if r["true_decision"] == "auto_handle" and r["pred_decision"] == "escalate"),
        "fn": sum(1 for r in results if r["true_decision"] == "escalate" and r["pred_decision"] == "auto_handle"),
        "tn": sum(1 for r in results if r["true_decision"] == "auto_handle" and r["pred_decision"] == "auto_handle"),
    }
    tp = escalation_confusion["tp"]
    fp = escalation_confusion["fp"]
    fn = escalation_confusion["fn"]
    esc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    esc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    esc_f1 = 2 * esc_precision * esc_recall / (esc_precision + esc_recall) if (esc_precision + esc_recall) > 0 else 0

    metrics = {
        "system": system_name,
        "n_samples": n,
        "intent_accuracy": round(float(np.mean(intent_correct)), 4),
        "routing_accuracy": round(float(np.mean(routing_correct)), 4),
        "escalation_precision": round(esc_precision, 4),
        "escalation_recall": round(esc_recall, 4),
        "escalation_f1": round(esc_f1, 4),
        "escalation_confusion": escalation_confusion,
        "rouge1_recall_mean": round(float(np.mean(rouge_scores)), 4),
        "n_judge_samples": n_judge,
        **judge_agg,
    }

    return {"metrics": metrics, "per_sample_results": results}


def run_evaluation(systems: dict, judge_every_n: int = 5, max_samples: int | None = None):
    """
    Run evaluation for all provided systems and save reports.

    Args:
        systems: Dict of {system_name: run_fn}
        judge_every_n: Only run LLM judge every N samples (cost control)
        max_samples: Limit to first N samples
    """
    judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    judge_model = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")

    golden = load_golden_set()
    print(f"Loaded {len(golden)} golden set examples.")

    all_results = {}
    for system_name, run_fn in systems.items():
        result = evaluate_system(
            system_name=system_name,
            run_fn=run_fn,
            golden=golden,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_every_n=judge_every_n,
            max_samples=max_samples,
        )
        all_results[system_name] = result

    # Save raw results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {RESULTS_PATH}")

    # Generate markdown report
    generate_report(all_results, golden)
    return all_results


def generate_report(all_results: dict, golden: list[dict]):
    """Generate a human-readable eval report in markdown."""
    lines = ["# Agent Evaluation Report\n"]
    lines.append(f"**Golden set size**: {len(golden)} examples\n")
    lines.append(f"**Brand**: AppleSupport\n")
    lines.append("")

    # Summary metrics table
    lines.append("## Summary Metrics\n")
    headers = [
        "System", "Intent Acc", "Routing Acc", "Esc Precision", "Esc Recall",
        "Esc F1", "ROUGE-1 Recall", "Judge Overall"
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for sys_name, data in all_results.items():
        m = data["metrics"]
        row = [
            sys_name,
            f"{m['intent_accuracy']:.1%}",
            f"{m['routing_accuracy']:.1%}",
            f"{m['escalation_precision']:.1%}",
            f"{m['escalation_recall']:.1%}",
            f"{m['escalation_f1']:.1%}",
            f"{m['rouge1_recall_mean']:.3f}",
            str(m.get("judge_overall_mean", "N/A")),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Per-system escalation confusion matrices
    lines.append("## Escalation Confusion Matrices\n")
    for sys_name, data in all_results.items():
        m = data["metrics"]
        cm = m["escalation_confusion"]
        lines.append(f"### {sys_name}\n")
        lines.append("```")
        lines.append(f"               Predicted Escalate  Predicted Auto-Handle")
        lines.append(f"True Escalate     {cm['tp']:>4}                {cm['fn']:>4}")
        lines.append(f"True Auto-Handle  {cm['fp']:>4}                {cm['tn']:>4}")
        lines.append("```\n")

    # Failure analysis — top misclassified intents
    lines.append("## Failure Analysis — Intent Misclassifications\n")
    for sys_name, data in all_results.items():
        lines.append(f"### {sys_name}\n")
        wrong = [r for r in data["per_sample_results"] if not r["intent_correct"]]
        if not wrong:
            lines.append("No misclassifications.\n")
            continue
        from collections import Counter
        confusion_pairs = Counter(
            (r["true_intent"], r["pred_intent"]) for r in wrong
        ).most_common(10)
        lines.append("Top misclassified intent pairs (true → predicted):\n")
        for (true, pred), count in confusion_pairs:
            lines.append(f"- `{true}` → `{pred}`: {count} times")
        lines.append("")

    # Low judge score examples
    lines.append("## Low-Quality Reply Examples (Judge Score < 2.5)\n")
    for sys_name, data in all_results.items():
        lines.append(f"### {sys_name}\n")
        low_q = [
            r for r in data["per_sample_results"]
            if r.get("judge") and r["judge"].get("overall", 5) < 2.5
        ][:5]
        if not low_q:
            lines.append("No low-quality replies found.\n")
            continue
        for r in low_q:
            lines.append(f"**Customer**: {r['customer_message'][:100]}...")
            lines.append(f"**Reply**: {r['reply'][:120]}")
            j = r["judge"]
            lines.append(f"**Scores**: empathy={j['empathy']}, actionability={j['actionability']}, brand_voice={j['brand_voice']}, relevance={j['relevance']}")
            lines.append(f"**Comment**: {j.get('comment', '')}\n")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"Evaluation report saved to {REPORT_PATH}")
