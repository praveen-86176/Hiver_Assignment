"""
Phase 4: Reply Drafter Module.

Generates a grounded, on-brand Apple Support reply using retrieved similar
historical replies as grounding context.
"""

import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

REPLY_SYSTEM_PROMPT = """You are an Apple Support agent. You write concise, empathetic, and professional customer support replies in the style of @AppleSupport on Twitter.

Guidelines:
- Maximum 280 characters (Twitter limit) — be brief and actionable
- Start with empathy, then provide a clear next step
- If recommending a DM, say "Please DM us so we can look into this"
- Do NOT make up URLs — use generic action phrases instead
- Reference the customer's specific issue (battery, app crash, etc.)
- Do NOT include hashtags or @mentions
- Tone: warm, calm, professional — never dismissive or over-enthusiastic"""


def _build_draft_prompt(customer_message: str, intent: str, grounding_context: str) -> str:
    return f"""Customer message (Intent: {intent}):
\"\"\"{customer_message}\"\"\"

{grounding_context}

Using the examples above as style and content guidance, write a new, original Apple Support reply to this specific customer. Keep it under 280 characters.

Reply:"""


def draft_reply(
    customer_message: str,
    intent: str,
    grounding_context: str,
    client: Groq | None = None,
    model: str | None = None,
) -> dict:
    """
    Draft a grounded Apple Support reply.

    Returns:
        dict with keys: reply, character_count
    """
    if client is None:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    if model is None:
        model = os.getenv("DRAFTER_MODEL", "qwen/qwen3.8-27b")

    prompt = _build_draft_prompt(customer_message, intent, grounding_context)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        reply = response.choices[0].message.content.strip()

        # Clean up any quotes or "Reply:" prefix
        reply = re.sub(r'^["\'Reply:\s]+', '', reply).strip().strip('"\'')

        # Truncate to Twitter limit
        if len(reply) > 280:
            reply = reply[:277] + "..."

        return {
            "reply": reply,
            "character_count": len(reply),
        }

    except Exception as e:
        return {
            "reply": "We're sorry you're experiencing this issue. Please DM us so we can help you directly.",
            "character_count": 88,
        }
