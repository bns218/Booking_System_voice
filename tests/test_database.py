"""
tests/test_database.py
----------------------
Unit tests for database save, retrieve, and list operations.
Uses a temporary SQLite DB so it does not touch production data.
Run with: pytest tests/
"""

import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Override DATABASE_URL before importing database module
os.environ["DATABASE_URL"] = os.path.join(tempfile.gettempdir(), "test_bookings.db")

import config
config.DATABASE_URL = os.environ["DATABASE_URL"]

from database import init_db, save_booking, get_booking, list_bookings

SAMPLE = {
    "reference":       "BN-TEST-2026",
    "customer_name":   "Alice Smith",
    "date":            "2026-05-10",
    "time":            "19:00",
    "guests":          4,
    "dietary":         "Gluten-free",
    "contact":         "alice@example.com",
    "special_request": "Window table please",
}


@pytest.fixture(autouse=True)
def fresh_db():
    """Recreate DB before each test."""
    db_path = config.DATABASE_URL
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


def test_save_booking_returns_true():
    assert save_booking(SAMPLE) is True


def test_save_booking_duplicate_ignored():
    save_booking(SAMPLE)
    result = save_booking(SAMPLE)  # second save — INSERT OR IGNORE
    assert result is True          # no crash, just no-op


def test_get_booking_returns_correct_record():
    save_booking(SAMPLE)
    row = get_booking("BN-TEST-2026")
    assert row is not None
    assert row["customer_name"] == "Alice Smith"
    assert row["guests"] == 4


def test_get_booking_missing_returns_none():
    result = get_booking("BN-NONEXISTENT")
    assert result is None


def test_list_bookings_returns_all():
    save_booking(SAMPLE)
    second = dict(SAMPLE, reference="BN-TEST2-2026", customer_name="Bob Jones")
    save_booking(second)
    rows = list_bookings()
    assert len(rows) == 2


def test_list_bookings_filtered_by_date():
    save_booking(SAMPLE)
    other = dict(SAMPLE, reference="BN-OTHER-2026", date="2026-06-01")
    save_booking(other)
    rows = list_bookings(date="2026-05-10")
    assert len(rows) == 1
    assert rows[0]["reference"] == "BN-TEST-2026"
