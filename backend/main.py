import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import snowflake.connector

# Load root .env (repo root)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def get_conn():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ.get("SNOWFLAKE_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "STI"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
    )

# Initialize FastAPI application
app = FastAPI(title="Support Ticket Intelligence API")

# Enable CORS so the Next.js frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  health check endpoint used to verify the API is running.
@app.get("/health")
def health():
    return {"ok": True}

# get a list of tickets 
@app.get("/tickets")
def list_tickets(limit: int = 50):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT ticket_id, created_at, customer_id, text
            FROM STI.PUBLIC.RAW_TICKETS
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
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

# Fetch details for a single ticket by ID.
@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT ticket_id, created_at, customer_id, text
            FROM STI.PUBLIC.RAW_TICKETS
            WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return {
            "ticket_id": row[0],
            "created_at": str(row[1]),
            "customer_id": row[2],
            "text": row[3],
            "label": "Unknown",
            "confidence": 0.0,
            "summary": "Not generated yet (GPT step comes next).",
            "suggested_reply": "Not generated yet.",
        }
    finally:
        cur.close()
        conn.close()
