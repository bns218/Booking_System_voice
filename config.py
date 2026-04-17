"""
config.py
---------
Central configuration — all settings loaded from environment variables.
Copy .env.example to .env and fill in your values before running.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Anthropic ────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]

# ── PersonaPlex server ───────────────────────────────────────────────────────
PERSONAPLEX_SERVER_URL: str = os.getenv(
    "PERSONAPLEX_SERVER_URL", "wss://localhost:8998/api/chat"
)

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "bookings.db")

# ── RAG ──────────────────────────────────────────────────────────────────────
DOCS_FOLDER:   Path = Path(os.getenv("DOCS_FOLDER",   "docs"))
CHROMA_FOLDER: Path = Path(os.getenv("CHROMA_FOLDER", ".chroma_db"))
EMBED_MODEL:   str  = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE:    int  = int(os.getenv("CHUNK_SIZE",    "400"))
CHUNK_OVERLAP: int  = int(os.getenv("CHUNK_OVERLAP", "80"))
RAG_TOP_K:     int  = int(os.getenv("RAG_TOP_K",     "6"))

# ── Restaurant ───────────────────────────────────────────────────────────────
RESTAURANT: dict = {
    "name":          os.getenv("RESTAURANT_NAME", "Bella Notte"),
    "agent_name":    os.getenv("AGENT_NAME",      "Sofia"),
    "cuisine":       os.getenv("CUISINE",          "Italian"),
    "opening_hours": os.getenv("OPENING_HOURS",    "Monday to Sunday, 12:00 PM to 10:00 PM"),
    "phone":         os.getenv("PHONE",            "+1-800-555-0199"),
    "tables": {
        "small":  {"capacity": 2, "count": 6},
        "medium": {"capacity": 4, "count": 8},
        "large":  {"capacity": 8, "count": 3},
    },
    "special_notes": (
        "We offer a fixed-price tasting menu on Fridays and Saturdays. "
        "Outdoor seating is available. "
        "We accommodate dietary requirements including vegan and gluten-free options. "
        "Reservations can be made up to 30 days in advance. "
        "We require a credit card to hold bookings for groups of 6 or more."
    ),
}

# ── Voice ────────────────────────────────────────────────────────────────────
VOICE_ID:    str = os.getenv("VOICE_ID",    "NATF1")
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "24000"))
CHUNK_AUDIO: int = int(os.getenv("CHUNK_AUDIO", "1920"))   # 80ms at 24kHz

# ── Booking confirmation triggers ────────────────────────────────────────────
CONFIRMATION_PHRASES: list[str] = [
    "your booking is confirmed",
    "reservation is confirmed",
    "booking has been confirmed",
    "you're all set",
    "we look forward to seeing you",
    "reference number",
]

# ── RAG seed queries ─────────────────────────────────────────────────────────
RAG_SEED_QUERIES: list[str] = [
    "menu dishes starters mains desserts",
    "drink wine cocktail beverages",
    "pricing set menu tasting prix fixe",
    "dietary vegan vegetarian gluten free allergens",
    "booking reservation cancellation policy deposit",
    "opening hours location parking directions",
    "private dining events group bookings",
    "dress code gift voucher loyalty",
]
