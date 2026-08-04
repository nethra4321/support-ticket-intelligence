import json
import logging
import os
from typing import Tuple

from anthropic import Anthropic

logger = logging.getLogger(__name__)

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_INPUT_CHARS = 4000

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def claude_summary_reply(ticket_text: str) -> Tuple[str, str]:
    """
    Generate a ticket summary and suggested reply using Claude.
    Returns (summary, suggested_reply).
    """

    ticket_text = (ticket_text or "").strip()

    if len(ticket_text) > MAX_INPUT_CHARS:
        ticket_text = ticket_text[:MAX_INPUT_CHARS]

    prompt = f"""
Analyze the following customer support ticket.

Return ONLY valid JSON in this exact format:
{{
  "summary": "string",
  "suggested_reply": "string"
}}

Rules:
- summary must be concise and factual
- suggested_reply must be polite, empathetic, and helpful
- do not include markdown
- do not include additional fields

Ticket:
{ticket_text}
""".strip()

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="You are a customer support assistant.",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    raw_output = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()

    if not raw_output:
        raise ValueError("Claude returned an empty response")

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")

        if start == -1 or end == -1 or end <= start:
            logger.error("Claude output was not JSON: %r", raw_output[:300])
            raise ValueError("Claude returned non-JSON output")

        data = json.loads(raw_output[start : end + 1])

    summary = str(data.get("summary") or "").strip()
    reply = str(data.get("suggested_reply") or "").strip()

    if not summary or not reply:
        raise ValueError("Claude response is missing required fields")

    return summary, reply