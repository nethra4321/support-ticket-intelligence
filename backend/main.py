# backend/main.py
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import requests
import snowflake.connector
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from llm_mistral import mistral_summary_reply
from llm_gpt2 import gpt2_summary_reply

from bert_lora_result import classify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sti-api")


load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SNOWFLAKE_DB = os.environ.get("SNOWFLAKE_DATABASE", "STI")
SNOWFLAKE_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")


def get_conn():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ.get("SNOWFLAKE_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=SNOWFLAKE_DB,
        schema=SNOWFLAKE_SCHEMA,
    )


def run_llm(model_name: str, text: str) -> Tuple[str, str]:
    m = (model_name or "").lower().strip()
    if m == "mistral":
        return mistral_summary_reply(text)
    if m == "gpt2":
        return gpt2_summary_reply(text)
    raise HTTPException(status_code=400, detail="model must be 'mistral' or 'gpt2'")


def fetch_one(cur, sql: str, params: Tuple = ()) -> Optional[Tuple]:
    cur.execute(sql, params)
    return cur.fetchone()


def fetch_all(cur, sql: str, params: Tuple = ()) -> List[Tuple]:
    cur.execute(sql, params)
    return cur.fetchall()


app = FastAPI(title="Support Ticket Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|192\.168\.56\.1):3000$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/tickets")
def list_tickets(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        rows = fetch_all(
            cur,
            f"""
            SELECT ticket_id, created_at, customer_id, text
            FROM {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.RAW_TICKETS
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )

        # Optional: classify each ticket text (can be slow on CPU).
        # For list view, keep Unknown to stay fast.
        return [
            {
                "ticket_id": r[0],
                "created_at": str(r[1]),
                "customer_id": r[2],
                "text": r[3],
                "label": "Unknown",
                "confidence": 0.0,
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, model: str = "mistral"):
    conn = get_conn()
    cur = conn.cursor()
    try:
        row = fetch_one(
            cur,
            f"""
            SELECT ticket_id, created_at, customer_id, text
            FROM {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.RAW_TICKETS
            WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        if not row:
            raise HTTPException(status_code=404, detail="Ticket not found")

        try:
            label, confidence = classify(row[3])
        except Exception:
            logger.exception("Classifier failed")
            label, confidence = "Unknown", 0.0
        model_name = (model or "mistral").lower().strip()
        g = fetch_one(
            cur,
            f"""
            SELECT summary, suggested_reply
            FROM {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.TICKET_GPT_OUTPUTS
            WHERE ticket_id=%s AND model_name=%s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (ticket_id, model_name),
        )
        summary = g[0] if g else "Not generated yet."
        suggested_reply = g[1] if g else "Not generated yet."

        return {
            "ticket_id": row[0],
            "created_at": str(row[1]),
            "customer_id": row[2],
            "text": row[3],
            "label": label,
            "confidence": round(float(confidence), 4),
            "model": model_name,  # for summary/reply view in UI
            "summary": summary,
            "suggested_reply": suggested_reply,
        }
    finally:
        cur.close()
        conn.close()


@app.post("/tickets/{ticket_id}/generate")
def generate_ticket_ai(ticket_id: str, model: str = "mistral") -> Dict[str, Any]:
    model = (model or "").lower().strip()

    conn = get_conn()
    cur = conn.cursor()
    try:
        row = fetch_one(
            cur,
            f"""
            SELECT text
            FROM {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.RAW_TICKETS
            WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        if not row:
            raise HTTPException(status_code=404, detail="Ticket not found")

        text = row[0]


        try:
            summary, reply = run_llm(model, text)
        except requests.RequestException:
            logger.exception("LLM request failed")
            raise HTTPException(
                status_code=502,
                detail="LLM service is not reachable. Check that Ollama is running.",
            )
        except json.JSONDecodeError:
            logger.exception("LLM returned invalid JSON")
            raise HTTPException(
                status_code=502,
                detail="LLM returned invalid output format (not JSON).",
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception("Unexpected error in run_llm")
            raise HTTPException(
                status_code=500,
                detail="Internal error while generating AI response.",
            )

    
        cur.execute(
            f"""
            DELETE FROM {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.TICKET_GPT_OUTPUTS
            WHERE ticket_id = %s AND model_name = %s
            """,
            (ticket_id, model),
        )

        cur.execute(
            f"""
            INSERT INTO {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.TICKET_GPT_OUTPUTS
            (ticket_id, model_name, summary, suggested_reply, generated_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (ticket_id, model, summary, reply, datetime.utcnow()),
        )
        conn.commit()

        return {
            "ticket_id": ticket_id,
            "model": model,
            "summary": summary,
            "suggested_reply": reply,
        }
    finally:
        cur.close()
        conn.close()


@app.post("/tickets/{ticket_id}/classify")
def classify_ticket(ticket_id: str) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        row = fetch_one(
            cur,
            f"""
            SELECT text
            FROM {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.RAW_TICKETS
            WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        if not row:
            raise HTTPException(status_code=404, detail="Ticket not found")

        label, confidence = classify(row[0])

        return {
            "ticket_id": ticket_id,
            "model": "bert+lora",
            "label": label,
            "confidence": round(float(confidence), 4),
        }
    finally:
        cur.close()
        conn.close()
