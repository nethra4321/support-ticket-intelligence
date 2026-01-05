import os
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector

# connecting to snowflake
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


def main():
    # loads .env from repo root
    load_dotenv()

    csv_path = os.environ["TWITTER_CSV_PATH"]
    limit = int(os.environ.get("LOAD_LIMIT", "10000"))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # ThoughtVector / Kaggle Twitter customer support dataset includes tweet_id, author_id, inbound, created_at, text
    required = {"tweet_id", "author_id", "inbound", "created_at", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Keep only inbound customer tweets
    df = df[df["inbound"] == True].copy()  
    df = df.dropna(subset=["tweet_id", "author_id", "created_at", "text"])
    df = df.head(limit)

    # Map to RAW_TICKETS schema
    df["ticket_id"] = df["tweet_id"].astype(str)
    df["customer_id"] = df["author_id"].astype(str)
    df["source"] = "twitter"
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])

    rows = list(
        zip(
            df["ticket_id"].tolist(),
            df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            df["customer_id"].tolist(),
            df["text"].astype(str).tolist(),
            df["source"].tolist(),
        )
    )

    print(f"Inserting {len(rows)} rows into STI.PUBLIC.RAW_TICKETS ...")

    conn = get_conn()
    cur = conn.cursor()
    try:
        # For repeat runs during dev
        cur.execute("TRUNCATE TABLE STI.PUBLIC.RAW_TICKETS")

        insert_sql = """
        INSERT INTO STI.PUBLIC.RAW_TICKETS (ticket_id, created_at, customer_id, text, source)
        VALUES (%s, %s, %s, %s, %s)
        """
        cur.executemany(insert_sql, rows)
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM STI.PUBLIC.RAW_TICKETS")
        count = cur.fetchone()[0]
        print(f"Done. RAW_TICKETS row count = {count}")

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
