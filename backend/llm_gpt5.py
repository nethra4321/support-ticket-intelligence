import json
import logging
import os
from typing import Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_INPUT_CHARS = 4000

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt5_summary_reply(ticket_text: str) -> Tuple[str, str]:
    """
    Generate a ticket summary and suggested reply using GPT-5.
    Returns (summary, suggested_reply).
    """

    ticket_text = (ticket_text or "").strip()

    if len(ticket_text) > MAX_INPUT_CHARS:
        ticket_text = ticket_text[:MAX_INPUT_CHARS]

    prompt = f"""
You are a customer support assistant.

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

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    raw_output = response.output_text.strip()

    if not raw_output:
        raise ValueError("GPT-5 returned an empty response")

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")

        if start == -1 or end == -1 or end <= start:
            logger.error("GPT-5 output was not JSON: %r", raw_output[:300])
            raise ValueError("GPT-5 returned non-JSON output")

        data = json.loads(raw_output[start : end + 1])

    summary = str(data.get("summary") or "").strip()
    reply = str(data.get("suggested_reply") or "").strip()

    if not summary or not reply:
        raise ValueError("GPT-5 response is missing required fields")

    return summary, reply