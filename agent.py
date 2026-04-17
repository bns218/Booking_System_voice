"""
agent.py
--------
Persona prompt builder and booking detail extractor (Claude API).
"""

import json
import re
from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, RESTAURANT, CONFIRMATION_PHRASES

claude = Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Persona prompt ────────────────────────────────────────────────────────────

def build_persona_prompt(rag_context: str = "") -> str:
    """Build the full system prompt, optionally enriched with RAG context."""
    r = RESTAURANT
    rag_section = ""
    if rag_context:
        rag_section = f"""
KNOWLEDGE BASE (from your restaurant's documents):
Use the information below to answer customer questions accurately.
Do not make up details beyond what is provided here or in the restaurant details above.

{rag_context}
"""

    return f"""You are {r["agent_name"]}, a warm and professional booking agent for {r["name"]}, \
an upscale {r["cuisine"]} restaurant.

YOUR ROLE:
- Help customers make, modify, or cancel dining reservations
- Answer questions about the restaurant, menu, and facilities
- Collect all required booking details clearly and politely

RESTAURANT DETAILS:
- Name: {r["name"]}
- Cuisine: {r["cuisine"]}
- Hours: {r["opening_hours"]}
- Contact: {r["phone"]}

TABLE AVAILABILITY:
- Tables for 2: {r["tables"]["small"]["count"]} available
- Tables for 4: {r["tables"]["medium"]["count"]} available
- Tables for up to 8: {r["tables"]["large"]["count"]} available

IMPORTANT INFORMATION:
{r["special_notes"]}

BOOKING INFORMATION TO COLLECT:
1. Customer full name
2. Date and time of reservation
3. Number of guests
4. Any dietary requirements or special requests
5. Phone number or email for confirmation

CONVERSATION STYLE:
- Be warm, friendly, and professional at all times
- Speak clearly and at a natural conversational pace
- If a requested time slot is unavailable, proactively suggest the nearest alternatives
- Always confirm booking details back to the customer before finalising
- End every confirmed booking by giving the customer a reference number (e.g. BN-XXXX-2026)
- Say the phrase "your booking is confirmed" clearly when finalising a reservation

LIMITATIONS:
- You cannot process payments over the phone
- For same-day bookings with less than 2 hours notice, ask the customer to call back or walk in
- Always apologise politely if you cannot fulfil a request
{rag_section}
Start the conversation by greeting the customer and asking how you can help them today."""


# ── Booking extraction ────────────────────────────────────────────────────────

def is_confirmation(text: str) -> bool:
    """Return True if the agent text contains a booking confirmation phrase."""
    lower = text.lower()
    return any(phrase in lower for phrase in CONFIRMATION_PHRASES)


def extract_booking_from_transcript(transcript: str) -> dict | None:
    """
    Use Claude to extract structured booking details from the conversation
    transcript after a confirmation is detected.
    Returns a dict or None if extraction fails.
    """
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=(
            "You are a data extraction assistant. "
            "Extract booking details from a restaurant call transcript. "
            "Return ONLY a valid JSON object with these keys: "
            "reference, customer_name, date, time, guests (integer), "
            "dietary, contact, special_request. "
            "Use null for any field not mentioned. "
            "Do not include any explanation or markdown."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Extract the confirmed booking from this transcript:\n\n{transcript}",
            }
        ],
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        clean = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(clean)
        except Exception:
            print(f"⚠️  Could not parse booking JSON:\n{raw}")
            return None
