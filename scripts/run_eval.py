#!/usr/bin/env python3
"""
scripts/run_eval.py — Full evaluation runner.

Evaluates the agent and both baselines against the 200-example golden set.
Produces reports/eval_results.json and reports/eval_report.md.

Usage:
    python scripts/run_eval.py               # Full eval, judge every 5th sample
    python scripts/run_eval.py --quick       # Quick smoke test (20 samples)
    python scripts/run_eval.py --judge-n 1   # Judge every sample (more accurate, slower)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.pipeline import SupportAgentPipeline
from src.baselines.baselines import TrivialBaseline, KeywordBaseline
from src.eval.harness import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--quick", action="store_true", help="Limit to 20 examples for speed")
    parser.add_argument("--judge-n", type=int, default=5, help="Run LLM judge every N samples")
    parser.add_argument("--no-agent", action="store_true", help="Skip agent (baselines only)")
    args = parser.parse_args()

    max_samples = 20 if args.quick else None

    # Initialize systems
    trivial = TrivialBaseline()
    keyword = KeywordBaseline()

    systems = {
        "trivial_baseline": trivial.run,
        "keyword_baseline": keyword.run,
    }

    if not args.no_agent:
        agent = SupportAgentPipeline()
        systems["agent"] = agent.run

    print(f"\nRunning evaluation:")
    print(f"  Systems: {list(systems.keys())}")
    print(f"  Judge every: {args.judge_n} samples")
    if max_samples:
        print(f"  Quick mode: {max_samples} samples only")

    all_results = run_evaluation(
        systems=systems,
        judge_every_n=args.judge_n,
        max_samples=max_samples,
    )

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for sys_name, data in all_results.items():
        m = data["metrics"]
        print(f"\n{sys_name}:")
        print(f"  Intent accuracy:     {m['intent_accuracy']:.1%}")
        print(f"  Routing accuracy:    {m['routing_accuracy']:.1%}")
        print(f"  Escalation F1:       {m['escalation_f1']:.1%}")
        print(f"  ROUGE-1 recall:      {m['rouge1_recall_mean']:.3f}")
        if "judge_overall_mean" in m:
            print(f"  Judge overall:       {m['judge_overall_mean']}/5")


if __name__ == "__main__":
    main()
