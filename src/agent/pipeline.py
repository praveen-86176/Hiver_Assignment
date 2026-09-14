"""
Phase 4: Agent Pipeline (Orchestrator).

End-to-end pipeline: customer_message -> classify -> retrieve -> route -> draft reply

The pipeline is designed to be:
- Modular: each component (classify, retrieve, route, draft) can be tested independently
- Cost-efficient: only calls LLM for auto-handle cases that need a reply draft
- Deterministic routing: routing is done purely via policy rules (no LLM cost)
"""

import json
import os
import time
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from groq import Groq

from src.agent.classify import classify_intent, load_taxonomy
from src.agent.draft import draft_reply
from src.agent.route import route
from src.retrieval.retriever import build_index, retrieve, format_grounding_context

load_dotenv()


class SupportAgentPipeline:
    """
    Apple Support AI Agent pipeline.

    Usage:
        pipeline = SupportAgentPipeline()
        result = pipeline.run("@AppleSupport my battery is dying after iOS 11 update")
    """

    def __init__(self, top_k_retrieval: int = 3):
        print("Initializing support agent pipeline...")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.taxonomy = load_taxonomy()
        self.index = build_index()
        self.top_k = top_k_retrieval
        print("Pipeline ready.")

    def run(self, customer_message: str) -> dict[str, Any]:
        """
        Run the full agent pipeline for a single customer message.

        Returns:
            dict with full pipeline trace: classification, retrieval, routing, reply
        """
        t_start = time.time()

        # Step 1: Classify intent
        classification = classify_intent(
            message=customer_message,
            taxonomy=self.taxonomy,
            client=self.client,
        )
        intent = classification["intent"]

        # Step 2: Route (deterministic, no LLM)
        routing = route(
            intent=intent,
            customer_message=customer_message,
            taxonomy=self.taxonomy,
        )

        # Step 3: Retrieve similar historical pairs
        retrieved = retrieve(customer_message, index=self.index, top_k=self.top_k)
        grounding_context = format_grounding_context(retrieved)

        # Step 4: Draft reply (only if auto-handle)
        if routing["decision"] == "auto_handle":
            drafted = draft_reply(
                customer_message=customer_message,
                intent=intent,
                grounding_context=grounding_context,
                client=self.client,
            )
        else:
            drafted = {
                "reply": "[ESCALATED: Routed to human agent]",
                "character_count": 0,
            }

        elapsed = round(time.time() - t_start, 2)

        return {
            "customer_message": customer_message,
            "classification": classification,
            "routing": routing,
            "retrieved_pairs": retrieved,
            "reply": drafted["reply"],
            "character_count": drafted["character_count"],
            "latency_s": elapsed,
        }


def run_demo():
    """Quick smoke test — runs 3 messages through the full pipeline."""
    pipeline = SupportAgentPipeline()

    test_cases = [
        "@AppleSupport my battery drains so fast since iOS 11 update. From 80% to 10% in 2 hours.",
        "@AppleSupport someone hacked my Apple ID and I can't get back in. This is urgent!",
        "@AppleSupport my AirPods keep disconnecting from my iPhone 13. Already reset them twice.",
    ]

    print("\n" + "="*70)
    for msg in test_cases:
        result = pipeline.run(msg)
        print(f"\n📨 Customer: {msg[:80]}")
        print(f"   Intent:   {result['classification']['intent']} (confidence={result['classification']['confidence']:.2f})")
        print(f"   Routing:  {result['routing']['decision'].upper()}")
        print(f"   Reply:    {result['reply'][:120]}")
        print(f"   Latency:  {result['latency_s']}s")
        print("-"*70)


if __name__ == "__main__":
    run_demo()
