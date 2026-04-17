"""
scripts/list_bookings.py
------------------------
CLI to view bookings stored in the database.

Usage:
    python scripts/list_bookings.py                     # all bookings
    python scripts/list_bookings.py --date 2026-05-10   # filter by date
    python scripts/list_bookings.py --ref BN-1234-2026  # lookup by reference
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import init_db, get_booking, list_bookings


def fmt_row(r: dict) -> str:
    return (
        f"  Ref:      {r.get('reference','—')}\n"
        f"  Name:     {r.get('customer_name','—')}\n"
        f"  Date:     {r.get('date','—')}  {r.get('time','—')}\n"
        f"  Guests:   {r.get('guests','—')}\n"
        f"  Dietary:  {r.get('dietary') or '—'}\n"
        f"  Contact:  {r.get('contact') or '—'}\n"
        f"  Special:  {r.get('special_request') or '—'}\n"
        f"  Saved at: {r.get('created_at','—')}"
    )


def main():
    parser = argparse.ArgumentParser(description="List bookings from the database")
    parser.add_argument("--date", help="Filter by date (YYYY-MM-DD)")
    parser.add_argument("--ref",  help="Look up a single booking by reference")
    args = parser.parse_args()

    init_db()

    if args.ref:
        row = get_booking(args.ref)
        if row:
            print(f"\n── Booking {args.ref} ──")
            print(fmt_row(row))
        else:
            print(f"❌ No booking found for reference: {args.ref}")
        return

    rows = list_bookings(date=args.date)
    if not rows:
        msg = f"No bookings found for {args.date}" if args.date else "No bookings found."
        print(msg)
        return

    header = f"── Bookings{' for ' + args.date if args.date else ''} ({len(rows)} total) ──"
    print(f"\n{header}")
    for i, row in enumerate(rows, 1):
        print(f"\n[{i}]")
        print(fmt_row(row))
    print()


if __name__ == "__main__":
    main()
