#!/usr/bin/env python3
"""
demo.py — Interactive demo of the AppleSupport AI Agent.

Run this to see the agent classify, route, and draft replies
for any customer message you provide.

Usage:
    python demo.py
    python demo.py --message "my battery dies after iOS 11 update"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent.pipeline import SupportAgentPipeline


DEMO_MESSAGES = [
    "@AppleSupport my battery drains from 100% to 20% in under 3 hours since iOS 11. iPhone 7.",
    "@AppleSupport I got charged twice for the same app purchase. How do I get a refund?",
    "@AppleSupport I've been locked out of my Apple ID for 2 days. 2FA code isn't working.",
    "@AppleSupport my AirPods keep disconnecting from my Mac every 10 minutes. Very frustrating.",
    "@AppleSupport how do I transfer data from my old iPhone X to new iPhone 14?",
    "@AppleSupport my home button stopped working completely after iOS 11 update.",
]


def pretty_print_result(result: dict):
    c = result["classification"]
    r = result["routing"]

    print("\n" + "═"*65)
    print(f"📨 CUSTOMER:  {result['customer_message'][:100]}")
    print("─"*65)
    print(f"🏷  Intent:    {c['intent']}")
    print(f"   Confidence: {c['confidence']:.0%}")
    print(f"   Reasoning:  {c['reasoning'][:80]}")
    print(f"🔀 Decision:  {r['decision'].upper()}")
    print(f"   Basis:      {r['policy_basis'][:80]}")
    if r["triggered_signals"]:
        print(f"   Signals:    {', '.join(r['triggered_signals'])}")
    print(f"⏱  Latency:   {result['latency_s']}s")
    print()
    if result["routing"]["decision"] == "auto_handle":
        print(f"✉  REPLY ({result['character_count']} chars):")
        print(f"   {result['reply']}")
    else:
        print("🚨 ESCALATED — Routing to human agent.")
    print("═"*65)


def run_interactive(pipeline: SupportAgentPipeline):
    print("\n🍎 Apple Support AI Agent — Interactive Demo")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            msg = input("Customer message: ").strip()
            if msg.lower() in ("quit", "exit", "q"):
                break
            if not msg:
                continue
            result = pipeline.run(msg)
            pretty_print_result(result)
        except KeyboardInterrupt:
            break
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="AppleSupport AI Agent Demo")
    parser.add_argument("--message", type=str, help="Run a single message instead of interactive mode")
    parser.add_argument("--demo", action="store_true", help="Run all built-in demo messages")
    args = parser.parse_args()

    pipeline = SupportAgentPipeline()

    if args.message:
        result = pipeline.run(args.message)
        pretty_print_result(result)

    elif args.demo:
        print(f"\n🍎 Apple Support AI Agent — Running {len(DEMO_MESSAGES)} demo messages\n")
        for msg in DEMO_MESSAGES:
            result = pipeline.run(msg)
            pretty_print_result(result)

    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
