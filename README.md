# 🍽️ Bella Notte — Voice Booking Agent

A real-time, full-duplex AI voice agent for restaurant reservations, built on **NVIDIA PersonaPlex** + **Claude API** + **RAG**.

Customers call in, speak naturally, and the agent books their table — interruptions, dietary questions, and all — then saves the confirmed reservation to your database automatically.

---

## Architecture

```
Customer Voice
     │
     ▼
PersonaPlex (local GPU)          ← full-duplex voice model
  │  speech-to-speech, ~170ms latency
  │
  ▼
Booking Agent (main.py)
  ├── RAG (rag.py)               ← menu / wine / policy / FAQ embedded in ChromaDB
  ├── Persona Prompt (agent.py)  ← Sofia, Italian restaurant agent
  ├── Transcript monitor         ← watches for "your booking is confirmed"
  └── Booking Extractor          ← Claude API parses transcript → structured JSON
       │
       ▼
  Database (database.py)         ← SQLite by default, swap for PostgreSQL / MySQL
```

---

## Project Structure

```
bella-notte-agent/
├── main.py                   # Entry point — wires everything together
├── agent.py                  # Persona prompt builder + Claude booking extractor
├── rag.py                    # RAG pipeline: load → chunk → embed → retrieve
├── database.py               # DB init, save, get, list bookings
├── config.py                 # All config loaded from .env
│
├── booking_agent_tester.html # Browser-based test console (no server needed)
│
├── docs/                     # RAG knowledge base — drop your files here
│   ├── menu.pdf
│   ├── wine_list.pdf
│   ├── booking_policy.txt
│   └── faq.docx
│
├── tests/
│   ├── test_agent.py         # Persona + confirmation detection tests
│   ├── test_database.py      # DB CRUD tests
│   └── test_rag.py           # Chunking + document loading tests
│
├── scripts/
│   ├── add_document.py       # CLI: add a new doc to the RAG store
│   └── list_bookings.py      # CLI: view bookings in the database
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/bella-notte-agent.git
cd bella-notte-agent

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set:

```env
ANTHROPIC_API_KEY=sk-ant-...
RESTAURANT_NAME=Bella Notte
AGENT_NAME=Sofia
```

### 3. Start the PersonaPlex server

Requires an NVIDIA GPU (RTX 2000 series or above).

```bash
# Accept the model license on Hugging Face first:
# https://huggingface.co/nvidia/personaplex-7b-v1

SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR" --hf-repo nvidia/personaplex-7b-v1
```

If your GPU has limited VRAM, add `--cpu-offload`:

```bash
python -m moshi.server --ssl "$SSL_DIR" --hf-repo nvidia/personaplex-7b-v1 --cpu-offload
```

### 4. Run the agent

```bash
python main.py
```

The agent will:
1. Load and embed all documents in `docs/`
2. Connect to the PersonaPlex server
3. Start listening — speak to begin the reservation
4. Save confirmed bookings to `bookings.db` automatically
5. Save a timestamped transcript on call end

---

## Testing Without a GPU

Open `booking_agent_tester.html` directly in your browser.

- **Simulation mode** (default) — no server needed, pre-scripted agent responses
- **Live WebSocket mode** — connects to your running PersonaPlex instance
- Booking details auto-fill as the customer speaks
- One-click save to database when booking is confirmed

---

## RAG Knowledge Base

Drop any `.txt`, `.pdf`, or `.docx` files into `docs/` and they are automatically indexed at startup.

| File | Contents |
|------|----------|
| `menu.pdf` | Full à la carte menu with descriptions, prices, dietary tags |
| `wine_list.pdf` | 20+ wines — regions, tasting notes, glass & bottle prices |
| `booking_policy.txt` | Cancellation, deposit, dress code, and contact policy |
| `faq.docx` | 25 Q&A pairs covering reservations, dietary, events, and more |

### Add a document at runtime

```bash
python scripts/add_document.py docs/new_menu_summer.pdf
```

Preview extracted text before indexing:

```bash
python scripts/add_document.py docs/new_menu_summer.pdf --preview
```

ChromaDB persists the index to `.chroma_db/` — existing documents are not re-embedded on restart.

---

## Database

Default: **SQLite** (`bookings.db` in the project root).

### View bookings

```bash
python scripts/list_bookings.py                      # all bookings
python scripts/list_bookings.py --date 2026-05-10    # by date
python scripts/list_bookings.py --ref BN-1234-2026   # by reference
```

### Switch to PostgreSQL

In `database.py`, replace the body of `save_booking()`:

```python
import psycopg2
conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute(
    "INSERT INTO bookings (reference, customer_name, date, time, guests, "
    "dietary, contact, special_request, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
    (booking["reference"], booking["customer_name"], ...)
)
conn.commit()
```

And update `.env`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/bella_notte
```

---

## Voices

Set `VOICE_ID` in `.env`. All 16 PersonaPlex voices:

| Natural Female | Natural Male | Variety Female | Variety Male |
|---------------|--------------|----------------|--------------|
| NATF0 | NATM0 | VARF0 | VARM0 |
| NATF1 | NATM1 | VARF1 | VARM1 |
| NATF2 | NATM2 | VARF2 | VARM2 |
| NATF3 | NATM3 | VARF3 | VARM3 |
| | | VARF4 | VARM4 |

Default is `NATF1` — natural female, warm and professional.

---

## Customising for Your Restaurant

Everything is driven by `.env` and the `RESTAURANT` dict in `config.py`.

| Setting | Where |
|---------|-------|
| Restaurant name, cuisine, hours, phone | `.env` |
| Table sizes and counts | `config.py` → `RESTAURANT["tables"]` |
| Confirmation trigger phrases | `config.py` → `CONFIRMATION_PHRASES` |
| RAG seed queries | `config.py` → `RAG_SEED_QUERIES` |
| Agent persona instructions | `agent.py` → `build_persona_prompt()` |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover:
- Persona prompt content and RAG injection
- Confirmation phrase detection (true/false cases)
- Database save, retrieve, list, and duplicate handling
- RAG chunking, document loading, and unknown file filtering

---

## Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.11+ |
| GPU | NVIDIA GPU with CUDA (RTX 2000+) for PersonaPlex |
| VRAM | 16GB recommended; 8GB with `--cpu-offload` |
| Anthropic API | For booking extraction and RAG |
| Internet | First run only — to download the embedding model |

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required** |
| `PERSONAPLEX_SERVER_URL` | `wss://localhost:8998/api/chat` | PersonaPlex WebSocket URL |
| `DATABASE_URL` | `bookings.db` | SQLite path or DB connection string |
| `DOCS_FOLDER` | `docs` | Folder containing RAG documents |
| `CHROMA_FOLDER` | `.chroma_db` | ChromaDB persistence folder |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `VOICE_ID` | `NATF1` | PersonaPlex voice |
| `RESTAURANT_NAME` | `Bella Notte` | Restaurant name |
| `AGENT_NAME` | `Sofia` | Agent's name |
| `CUISINE` | `Italian` | Cuisine type |
| `OPENING_HOURS` | `Mon–Sun 12:00–22:00` | Opening hours string |
| `PHONE` | `+1-800-555-0199` | Contact number |

---

## Licence

MIT — see `LICENSE` for details.
