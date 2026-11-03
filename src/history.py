"""Locally stored, anonymous prediction history. No guest names or emails."""
import sqlite3
from datetime import datetime, timezone
from .config import HISTORY_PATH


def connection():
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(HISTORY_PATH), timeout=15)
    db.execute("""CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        checked_at TEXT NOT NULL,
        hotel TEXT NOT NULL,
        arrival_month TEXT NOT NULL,
        lead_time INTEGER NOT NULL,
        guests INTEGER NOT NULL,
        nights INTEGER NOT NULL,
        cancellation_probability REAL NOT NULL,
        risk_level TEXT NOT NULL
    )""")
    db.commit()
    return db


def record_assessment(b, probability, bucket):
    with connection() as db:
        db.execute("""INSERT INTO assessments
            (checked_at, hotel, arrival_month, lead_time, guests, nights, cancellation_probability, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            b["hotel"], b["arrival_date_month"], int(b["lead_time"]),
            int(b["adults"] + b["children"] + b["babies"]),
            int(b["stays_in_weekend_nights"] + b["stays_in_week_nights"]),
            float(probability), bucket
        ))
        db.commit()


def get_history():
    import pandas as pd
    with connection() as db:
        return pd.read_sql_query(
            "SELECT id AS 'ID', checked_at AS 'Checked at (UTC)', hotel AS 'Hotel', "
            "arrival_month AS 'Arrival month', lead_time AS 'Lead (days)', "
            "guests AS 'Guests', nights AS 'Nights', "
            "ROUND(cancellation_probability * 100, 1) AS 'Risk (%)', "
            "risk_level AS 'Risk level' FROM assessments ORDER BY id DESC LIMIT 1000", db
        )


def total_assessments():
    with connection() as db:
        return int(db.execute("SELECT COUNT(*) FROM assessments").fetchone()[0])


def clear_history():
    with connection() as db:
        db.execute("DELETE FROM assessments")
        db.commit()
