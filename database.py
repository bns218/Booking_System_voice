"""
database.py
-----------
Handles all database operations for bookings.
Defaults to SQLite. Swap save_booking() for PostgreSQL / MySQL / ORM.
"""

import sqlite3
from datetime import datetime
from config import DATABASE_URL


def init_db() -> None:
    """Create the bookings table if it does not already exist."""
    conn = sqlite3.connect(DATABASE_URL)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            reference       TEXT UNIQUE,
            customer_name   TEXT,
            date            TEXT,
            time            TEXT,
            guests          INTEGER,
            dietary         TEXT,
            contact         TEXT,
            special_request TEXT,
            created_at      TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_booking(booking: dict) -> bool:
    """
    Persist a confirmed booking to the database.
    Returns True on success, False on failure.

    ── To switch databases ──────────────────────────────────────────────────
    PostgreSQL (psycopg2):
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO bookings (reference, customer_name, ...) VALUES (%s, %s, ...)",
            (booking["reference"], booking["customer_name"], ...)
        )
        conn.commit()

    SQLAlchemy ORM:
        session.add(Booking(**booking))
        session.commit()
    ────────────────────────────────────────────────────────────────────────
    """
    conn = sqlite3.connect(DATABASE_URL)
    try:
        conn.execute(
            """INSERT OR IGNORE INTO bookings
               (reference, customer_name, date, time, guests,
                dietary, contact, special_request, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                booking.get("reference"),
                booking.get("customer_name"),
                booking.get("date"),
                booking.get("time"),
                booking.get("guests"),
                booking.get("dietary"),
                booking.get("contact"),
                booking.get("special_request"),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        print(f"✅ Booking saved → ref: {booking.get('reference')}")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    finally:
        conn.close()


def get_booking(reference: str) -> dict | None:
    """Retrieve a booking by reference number."""
    conn = sqlite3.connect(DATABASE_URL)
    try:
        cur = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    finally:
        conn.close()


def list_bookings(date: str | None = None) -> list[dict]:
    """List all bookings, optionally filtered by date string."""
    conn = sqlite3.connect(DATABASE_URL)
    try:
        if date:
            cur = conn.execute(
                "SELECT * FROM bookings WHERE date = ? ORDER BY time", (date,)
            )
        else:
            cur = conn.execute("SELECT * FROM bookings ORDER BY created_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
