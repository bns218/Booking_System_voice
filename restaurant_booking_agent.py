"""
Restaurant Dining Booking Agent — PersonaPlex (Local GPU) + RAG
===============================================================
Requirements:
    pip install moshi sounddevice numpy websockets anthropic
    pip install chromadb sentence-transformers pypdf python-docx

Documents folder:
    Place your .txt, .pdf, or .docx files inside a  docs/  folder
    next to this script. Examples:
        docs/menu.pdf
        docs/policies.txt
        docs/faq.docx
        docs/wine_list.pdf

Run PersonaPlex server first:
    SSL_DIR=$(mktemp -d)
    python -m moshi.server --ssl "$SSL_DIR" --hf-repo nvidia/personaplex-7b-v1

Then run this script:
    python restaurant_booking_agent.py
"""

import asyncio
import json
import os
import re
import sqlite3
import numpy as np
import sounddevice as sd
import websockets
import ssl
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic

# RAG imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 1. RESTAURANT CONFIGURATION
#    Edit this section to match your restaurant
# ──────────────────────────────────────────────

RESTAURANT = {
    "name": "Bella Notte",
    "cuisine": "Italian",
    "agent_name": "Sofia",
    "phone": "+1-800-555-0199",
    "opening_hours": "Monday to Sunday, 12:00 PM to 10:00 PM",
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

# ──────────────────────────────────────────────
# 2. RAG — DOCUMENT KNOWLEDGE BASE
#    Drop any .txt / .pdf / .docx files into
#    the  docs/  folder and they are auto-loaded
#    at startup and injected into the agent.
# ──────────────────────────────────────────────

DOCS_FOLDER   = Path("docs")          # folder containing your documents
CHROMA_FOLDER = Path(".chroma_db")    # persisted vector store (auto-created)
EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast local embedding model (~80MB)
CHUNK_SIZE    = 400                   # characters per chunk
CHUNK_OVERLAP = 80                    # overlap between chunks
TOP_K         = 6                     # chunks to retrieve per seed query

# Pre-seeded queries used to fetch relevant context at startup
# Add more if your documents cover additional topics
RAG_SEED_QUERIES = [
    "menu dishes starters mains desserts",
    "drink wine cocktail beverages",
    "pricing set menu tasting prix fixe",
    "dietary vegan vegetarian gluten free allergens",
    "booking reservation cancellation policy deposit",
    "opening hours location parking directions",
    "private dining events group bookings",
    "dress code gift voucher loyalty",
]


# ── Document loaders ──────────────────────────

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print(f"  ⚠️  pypdf not installed — skipping {path.name}")
        return ""


def _load_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        print(f"  ⚠️  python-docx not installed — skipping {path.name}")
        return ""


LOADERS = {".txt": _load_txt, ".pdf": _load_pdf, ".docx": _load_docx}


def load_documents(folder: Path) -> list[dict]:
    """Load all supported documents from folder, return list of {text, source}."""
    if not folder.exists():
        print(f"  ℹ️  No docs/ folder found — RAG context will be empty.")
        return []

    docs = []
    for path in sorted(folder.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if not loader:
            continue
        print(f"  📄 Loading {path.name}...")
        text = loader(path).strip()
        if text:
            docs.append({"text": text, "source": path.name})

    print(f"  ✅ Loaded {len(docs)} document(s)")
    return docs


# ── Chunking ──────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 40]  # drop tiny fragments


# ── Vector store ──────────────────────────────

def build_vector_store(docs: list[dict]) -> chromadb.Collection:
    """Embed all document chunks and store in ChromaDB."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_FOLDER),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name="restaurant_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # If docs already indexed, skip re-embedding
    existing_sources = set()
    if collection.count() > 0:
        results = collection.get(include=["metadatas"])
        existing_sources = {m["source"] for m in results["metadatas"]}

    model = SentenceTransformer(EMBED_MODEL)
    new_chunks, new_ids, new_metas = [], [], []

    for doc in docs:
        if doc["source"] in existing_sources:
            print(f"  ♻️  {doc['source']} already indexed — skipping")
            continue
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            new_chunks.append(chunk)
            new_ids.append(f"{doc['source']}::{i}")
            new_metas.append({"source": doc["source"], "chunk": i})

    if new_chunks:
        print(f"  🔢 Embedding {len(new_chunks)} new chunks...")
        embeddings = model.encode(new_chunks, show_progress_bar=False).tolist()
        collection.add(
            documents=new_chunks,
            embeddings=embeddings,
            ids=new_ids,
            metadatas=new_metas,
        )
        print(f"  ✅ Vector store updated ({collection.count()} total chunks)")

    return collection, model


def retrieve_context(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    queries: list[str],
    top_k: int = TOP_K,
) -> str:
    """
    Query the vector store with multiple seed queries and return
    deduplicated top chunks as a single context string.
    """
    if collection.count() == 0:
        return ""

    seen, chunks = set(), []
    for query in queries:
        embedding = model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            key = f"{meta['source']}::{meta['chunk']}"
            if key not in seen:
                seen.add(key)
                chunks.append(f"[{meta['source']}]\n{doc}")

    return "\n\n".join(chunks)


def setup_rag() -> str:
    """
    Full RAG pipeline: load docs → embed → retrieve context.
    Returns a formatted string ready to inject into the system prompt.
    """
    print("\n📚 Setting up RAG knowledge base...")
    docs = load_documents(DOCS_FOLDER)

    if not docs:
        return ""

    collection, model = build_vector_store(docs)
    context = retrieve_context(collection, model, RAG_SEED_QUERIES)

    if context:
        print(f"  ✅ RAG context ready ({len(context)} chars injected into prompt)\n")
    return context


# ──────────────────────────────────────────────
# 3. DATABASE CONFIGURATION
#    Swap the save_booking() function below to
#    connect to PostgreSQL, MySQL, or any ORM.
# ──────────────────────────────────────────────

DB_PATH = "bookings.db"

# Keywords the agent says when a booking is confirmed
CONFIRMATION_PHRASES = [
    "your booking is confirmed",
    "reservation is confirmed",
    "booking has been confirmed",
    "you're all set",
    "we look forward to seeing you",
    "reference number",
]


def init_db():
    """Create bookings table if it doesn't exist (SQLite example)."""
    conn = sqlite3.connect(DB_PATH)
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


def save_booking(booking: dict):
    """
    Save confirmed booking to the database.

    Replace this function body to use your own DB:
    ─────────────────────────────────────────────
    PostgreSQL example (psycopg2):
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO bookings (reference, customer_name, ...) VALUES (%s, %s, ...)",
            (booking["reference"], booking["customer_name"], ...)
        )
        conn.commit()

    SQLAlchemy ORM example:
        session.add(Booking(**booking))
        session.commit()
    ─────────────────────────────────────────────
    """
    conn = sqlite3.connect(DB_PATH)
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
        print(f"\n✅ Booking saved to database → ref: {booking.get('reference')}")
    except Exception as e:
        print(f"\n❌ Database error: {e}")
    finally:
        conn.close()


# ──────────────────────────────────────────────
# 3. BOOKING EXTRACTOR (Claude API)
#    Parses the conversation transcript and
#    returns structured booking details as JSON.
# ──────────────────────────────────────────────

claude = Anthropic()


def extract_booking_from_transcript(transcript: str) -> dict | None:
    """
    Use Claude to extract structured booking details from the
    conversation transcript after confirmation is detected.
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
        # Strip any accidental markdown fences
        clean = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(clean)
        except Exception:
            print(f"\n⚠️  Could not parse booking JSON:\n{raw}")
            return None


# ──────────────────────────────────────────────
# 4. PERSONA SYSTEM PROMPT
# ──────────────────────────────────────────────

def build_persona_prompt(restaurant: dict, rag_context: str = "") -> str:
    rag_section = ""
    if rag_context:
        rag_section = f"""
KNOWLEDGE BASE (from your restaurant's documents):
Use the information below to answer customer questions accurately.
Do not make up details beyond what is provided here or in the restaurant details above.

{rag_context}
"""

    return f"""You are {restaurant["agent_name"]}, a warm and professional booking agent for {restaurant["name"]}, 
an upscale {restaurant["cuisine"]} restaurant.

YOUR ROLE:
- Help customers make, modify, or cancel dining reservations
- Answer questions about the restaurant, menu, and facilities
- Collect all required booking details clearly and politely

RESTAURANT DETAILS:
- Name: {restaurant["name"]}
- Cuisine: {restaurant["cuisine"]}
- Hours: {restaurant["opening_hours"]}
- Contact: {restaurant["phone"]}

TABLE AVAILABILITY:
- Tables for 2: {restaurant["tables"]["small"]["count"]} available
- Tables for 4: {restaurant["tables"]["medium"]["count"]} available
- Tables for up to 8: {restaurant["tables"]["large"]["count"]} available

IMPORTANT INFORMATION:
{restaurant["special_notes"]}

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
- End every confirmed booking by giving the customer a reference number (e.g. BN-{{}}-2026)
- Say the phrase "your booking is confirmed" clearly when finalising a reservation

LIMITATIONS:
- You cannot process payments over the phone
- For same-day bookings with less than 2 hours notice, ask the customer to call back or walk in
- Always apologise politely if you cannot fulfil a request
{rag_section}
Start the conversation by greeting the customer and asking how you can help them today."""


# ──────────────────────────────────────────────
# 5. VOICE CONFIGURATION
# ──────────────────────────────────────────────

VOICE_CONFIG = {
    # Natural female voice — sounds warm and professional
    # Options: NATF0-NATF3, NATM0-NATM3, VARF0-VARF4, VARM0-VARM4
    "voice_id": "NATF1",
    "sample_rate": 24000,
    "chunk_size": 1920,  # 80ms of audio at 24kHz
}


# ──────────────────────────────────────────────
# 6. PERSONAPLEX CLIENT
# ──────────────────────────────────────────────

class RestaurantBookingAgent:
    def __init__(
        self,
        server_url: str = "wss://localhost:8998/api/chat",
        persona_prompt: str = "",
        voice_id: str = "NATF1",
    ):
        self.server_url = server_url
        self.persona_prompt = persona_prompt
        self.voice_id = voice_id
        self.sample_rate = VOICE_CONFIG["sample_rate"]
        self.chunk_size = VOICE_CONFIG["chunk_size"]

        # Transcript is built from text tokens sent by PersonaPlex
        self.transcript: list[str] = []
        self.booking_saved = False

    def _make_ssl_context(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _build_init_message(self) -> dict:
        return {
            "type": "init",
            "voice": self.voice_id,
            "system_prompt": self.persona_prompt,
        }

    def _is_confirmation(self, text: str) -> bool:
        """Return True if the agent has just confirmed a booking."""
        lower = text.lower()
        return any(phrase in lower for phrase in CONFIRMATION_PHRASES)

    def _handle_text_token(self, token: str):
        """Append agent text to transcript and watch for confirmation."""
        self.transcript.append(token)

        if not self.booking_saved and self._is_confirmation(token):
            full_transcript = "".join(self.transcript)
            print("\n🔍 Confirmation detected — extracting booking details...")
            booking = extract_booking_from_transcript(full_transcript)
            if booking:
                save_booking(booking)
                self.booking_saved = True
            else:
                print("⚠️  Could not extract booking details from transcript.")

    async def run(self):
        ssl_ctx = self._make_ssl_context()

        print(f"\n🍽️  {RESTAURANT['name']} Booking Agent")
        print(f"   Agent: {RESTAURANT['agent_name']} | Voice: {self.voice_id}")
        print("   Connecting to PersonaPlex server...\n")

        async with websockets.connect(self.server_url, ssl=ssl_ctx) as ws:
            await ws.send(json.dumps(self._build_init_message()))
            print("✅ Connected. Speak to start your reservation.\n")
            print("   (Press Ctrl+C to end the call)\n")

            input_queue: asyncio.Queue = asyncio.Queue()
            output_queue: asyncio.Queue = asyncio.Queue()

            def mic_callback(indata, frames, time, status):
                input_queue.put_nowait(indata.copy())

            def speaker_callback(outdata, frames, time, status):
                try:
                    chunk = output_queue.get_nowait()
                    outdata[:] = chunk.reshape(-1, 1)
                except asyncio.QueueEmpty:
                    outdata[:] = np.zeros((frames, 1), dtype=np.float32)

            async def send_audio():
                while True:
                    chunk = await input_queue.get()
                    pcm = (chunk * 32767).astype(np.int16).tobytes()
                    await ws.send(pcm)

            async def receive_messages():
                """Handle both audio bytes and JSON text tokens from PersonaPlex."""
                async for message in ws:
                    if isinstance(message, bytes):
                        # Audio response — play it
                        pcm = np.frombuffer(message, dtype=np.int16)
                        audio = pcm.astype(np.float32) / 32767.0
                        await output_queue.put(audio)
                    elif isinstance(message, str):
                        # Text token from PersonaPlex transcript stream
                        try:
                            data = json.loads(message)
                            token = data.get("text", "")
                            if token:
                                self._handle_text_token(token)
                        except json.JSONDecodeError:
                            self._handle_text_token(message)

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=mic_callback,
            ), sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=speaker_callback,
            ):
                await asyncio.gather(send_audio(), receive_messages())


# ──────────────────────────────────────────────
# 7. ENTRY POINT
# ──────────────────────────────────────────────

async def main():
    init_db()

    # Build RAG context from docs/ folder
    rag_context = setup_rag()

    # Build persona prompt enriched with document knowledge
    persona = build_persona_prompt(RESTAURANT, rag_context=rag_context)

    agent = RestaurantBookingAgent(
        server_url="wss://localhost:8998/api/chat",
        persona_prompt=persona,
        voice_id=VOICE_CONFIG["voice_id"],
    )

    try:
        await agent.run()
    except KeyboardInterrupt:
        # On call end, save full transcript to a log file for audit
        if agent.transcript:
            log_path = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_path, "w") as f:
                f.write("".join(agent.transcript))
            print(f"\n📝 Transcript saved → {log_path}")

        print("\n📞 Call ended. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
