import json
import logging
import os
import requests
from typing import Tuple

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

MODEL = "mistral"

REQUEST_TIMEOUT_SECONDS = 500
MAX_INPUT_CHARS = 4000
MAX_OUTPUT_TOKENS = 180


def mistral_summary_reply(ticket_text: str) -> Tuple[str, str]:
    """
    Generate a short summary and suggested reply using Ollama Mistral.
    Returns (summary, suggested_reply)
    """

    # ---- Safety: cap very long tickets ----
    ticket_text = (ticket_text or "").strip()
    if len(ticket_text) > MAX_INPUT_CHARS:
        ticket_text = ticket_text[:MAX_INPUT_CHARS]

    prompt = f"""
You are a customer support assistant.

Return ONLY valid JSON.
Do not include explanations, markdown, or extra text.

Ticket:
{ticket_text}

JSON format:
{{"summary":"string","suggested_reply":"string"}}

Rules:
- summary: 1–2 concise lines
- suggested_reply: polite, helpful, 2–4 lines
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": MAX_OUTPUT_TOKENS,
                    "temperature": 0.2,
                },
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    except requests.exceptions.ReadTimeout:
        logger.error("Ollama request timed out after %s seconds", REQUEST_TIMEOUT_SECONDS)
        raise

    except requests.exceptions.RequestException as e:
        logger.error("Ollama request failed: %s", e)
        raise

    payload = response.json()
    raw_output = (payload.get("response") or "").strip()

    if not raw_output:
        logger.error("Empty response from Ollama")
        raise ValueError("Empty response from LLM")


    try:
        data = json.loads(raw_output)
        return (
            (data.get("summary") or "").strip(),
            (data.get("suggested_reply") or "").strip(),
        )
    except json.JSONDecodeError:
        pass


    start = raw_output.find("{")
    end = raw_output.rfind("}")

    if start == -1 or end == -1 or end <= start:
        logger.error("LLM output not valid JSON: %r", raw_output[:300])
        raise ValueError("LLM returned non-JSON output")

    try:
        data = json.loads(raw_output[start : end + 1])
        return (
            (data.get("summary") or "").strip(),
            (data.get("suggested_reply") or "").strip(),
        )
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extracted JSON: %s", e)
        raise
