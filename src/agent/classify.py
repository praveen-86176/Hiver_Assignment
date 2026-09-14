"""
Phase 4: Intent Classification Module.

Uses Groq LLM with structured JSON output to classify incoming
customer messages into the 9-intent AppleSupport taxonomy.
"""

import json
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

TAXONOMY_PATH = Path("src/taxonomy/intents.json")


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def _build_classify_prompt(message: str, taxonomy: dict) -> str:
    intent_descriptions = "\n".join(
        f"- \"{i['id']}\": {i['label']} — {i['description']}"
        for i in taxonomy["intents"]
    )
    valid_ids = [i["id"] for i in taxonomy["intents"]]

    return f"""You are an expert Apple Support triage agent. Classify the customer message below into exactly ONE of these intent categories:

{intent_descriptions}

Respond ONLY with valid JSON in this exact format:
{{
  "intent": "<one of: {', '.join(valid_ids)}>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining why>"
}}

Customer message:
\"\"\"{message}\"\"\"

JSON response:"""


def classify_intent(
    message: str,
    taxonomy: dict | None = None,
    client: Groq | None = None,
    model: str | None = None,
) -> dict:
    """
    Classify a customer message into an intent using Groq LLM.

    Returns:
        dict with keys: intent, confidence, reasoning
    """
    if taxonomy is None:
        taxonomy = load_taxonomy()
    if client is None:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    if model is None:
        model = os.getenv("CLASSIFIER_MODEL", "qwen/qwen3.8-27b")

    valid_ids = {i["id"] for i in taxonomy["intents"]}
    prompt = _build_classify_prompt(message, taxonomy)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw)

        # Validate
        if result.get("intent") not in valid_ids:
            result["intent"] = "general_inquiry"
        result["confidence"] = float(result.get("confidence", 0.5))

        return result

    except Exception as e:
        return {
            "intent": "general_inquiry",
            "confidence": 0.0,
            "reasoning": f"Classification failed: {str(e)[:100]}",
        }
