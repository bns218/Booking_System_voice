"""
server.py
---------
FastAPI WebSocket server that bridges the browser frontend to PersonaPlex
for full-duplex voice calls, with real-time transcript monitoring,
Gemini-powered booking extraction, and SQLite persistence.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Architecture:
    Browser (mic/speaker via WebAudio)
        ↕  WebSocket  ↕
    FastAPI server (this file)
        ↕  WebSocket  ↕
    PersonaPlex server (GPU, speech-to-speech)
"""

import asyncio
import json
import ssl
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import (
    PERSONAPLEX_SERVER_URL,
    VOICE_ID,
    SAMPLE_RATE,
    RESTAURANT,
)
from database import init_db, save_booking, list_bookings, get_booking
from rag import setup_rag
from agent import build_persona_prompt, is_confirmation, extract_booking_from_transcript

import websockets


# ── Global state ─────────────────────────────────────────────────────────────

rag_context: str = ""
persona_prompt: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, build RAG, prepare persona prompt."""
    global rag_context, persona_prompt

    print("\n=== Bella Notte Server Starting ===\n")

    # 1. Init database
    init_db()
    print("[OK] Database initialised")

    # 2. Build RAG context
    rag_context = setup_rag()
    print(f"[OK] RAG context ready ({len(rag_context):,} chars)")

    # 3. Build persona prompt
    persona_prompt = build_persona_prompt(rag_context=rag_context)
    print(f"[OK] Persona prompt built ({len(persona_prompt):,} chars)")

    print("\n=== Server ready — open http://localhost:8000 ===\n")
    yield


app = FastAPI(
    title="Bella Notte Booking Agent",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow browser connections) ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SSL context for PersonaPlex (self-signed cert) ───────────────────────────

def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── Active sessions tracker ──────────────────────────────────────────────────

class CallSession:
    """Tracks a single active call between browser and PersonaPlex."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.transcript: list[str] = []
        self.booking_data: dict = {}
        self.booking_saved: bool = False
        self.booking_status: str = "none"  # none | pending | confirmed
        self.started_at: datetime = datetime.now()

    def add_transcript(self, speaker: str, text: str) -> dict | None:
        """Add text to transcript, check for booking confirmation.
        Returns booking data dict if extraction succeeds, else None."""
        self.transcript.append(f"[{speaker}] {text}")

        if speaker == "agent" and not self.booking_saved and is_confirmation(text):
            full_transcript = "\n".join(self.transcript)
            print(f"\n[Session {self.session_id}] Confirmation detected — extracting booking...")
            booking = extract_booking_from_transcript(full_transcript)
            if booking:
                self.booking_data = booking
                self.booking_status = "confirmed"
                return booking
            else:
                print(f"[Session {self.session_id}] Could not extract booking from transcript.")

        # Update status to pending if we have some data patterns
        if self.booking_status == "none" and len(self.transcript) > 2:
            self.booking_status = "pending"

        return None


active_sessions: dict[str, CallSession] = {}


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoint: Full voice pipeline (browser ↔ PersonaPlex)
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    """
    Full-duplex voice call bridge.

    Protocol (browser → server):
        - JSON: { "type": "init", "voice": "NATF1" }     → start session
        - Binary: raw PCM int16 audio frames               → proxy to PersonaPlex
        - JSON: { "type": "end" }                          → end call

    Protocol (server → browser):
        - JSON: { "type": "transcript", "speaker": "agent", "text": "..." }
        - JSON: { "type": "booking_update", "status": "...", "data": {...} }
        - JSON: { "type": "status", "state": "connected" | "error", "message": "..." }
        - Binary: PCM int16 audio frames from PersonaPlex  → play in browser
    """
    await ws.accept()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = CallSession(session_id)
    active_sessions[session_id] = session
    pp_ws = None  # PersonaPlex WebSocket

    print(f"\n[Session {session_id}] Browser connected")

    try:
        # Wait for init message
        init_raw = await ws.receive_text()
        init_msg = json.loads(init_raw)
        voice = init_msg.get("voice", VOICE_ID)
        custom_prompt = init_msg.get("system_prompt", persona_prompt)

        print(f"[Session {session_id}] Init — voice={voice}")

        # Connect to PersonaPlex
        try:
            pp_ws = await websockets.connect(
                PERSONAPLEX_SERVER_URL,
                ssl=_ssl_ctx(),
                max_size=2**20,
            )
            await pp_ws.send(json.dumps({
                "type": "init",
                "voice": voice,
                "system_prompt": custom_prompt,
            }))

            await ws.send_json({
                "type": "status",
                "state": "connected",
                "message": f"Connected to PersonaPlex — voice {voice}",
            })
            print(f"[Session {session_id}] PersonaPlex connected")

        except Exception as e:
            await ws.send_json({
                "type": "status",
                "state": "error",
                "message": f"PersonaPlex connection failed: {str(e)}",
            })
            print(f"[Session {session_id}] PersonaPlex error: {e}")
            return

        # ── Bidirectional proxy ──────────────────────────────────────────

        async def browser_to_personaplex():
            """Forward audio/commands from browser to PersonaPlex."""
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        # Binary audio → forward to PersonaPlex
                        await pp_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        data = json.loads(msg["text"])
                        if data.get("type") == "end":
                            break
                        elif data.get("type") == "customer_text":
                            # Text simulation message from browser
                            text = data.get("text", "")
                            session.add_transcript("customer", text)
                            # Forward as text to PersonaPlex if supported
                            await pp_ws.send(json.dumps({
                                "type": "text",
                                "text": text,
                            }))
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"[Session {session_id}] browser→pp error: {e}")

        async def personaplex_to_browser():
            """Forward audio/text from PersonaPlex to browser."""
            try:
                async for msg in pp_ws:
                    if isinstance(msg, bytes):
                        # Audio from PersonaPlex → browser
                        await ws.send_bytes(msg)
                    elif isinstance(msg, str):
                        # Text token from PersonaPlex
                        try:
                            data = json.loads(msg)
                            text = data.get("text", "")
                        except json.JSONDecodeError:
                            text = msg

                        if text:
                            # Send transcript to browser
                            await ws.send_json({
                                "type": "transcript",
                                "speaker": "agent",
                                "text": text,
                            })

                            # Check for booking confirmation
                            booking = session.add_transcript("agent", text)
                            if booking:
                                await ws.send_json({
                                    "type": "booking_update",
                                    "status": "confirmed",
                                    "data": booking,
                                })
                            elif session.booking_status == "pending":
                                await ws.send_json({
                                    "type": "booking_update",
                                    "status": "pending",
                                    "data": {},
                                })
            except websockets.exceptions.ConnectionClosed:
                print(f"[Session {session_id}] PersonaPlex disconnected")
            except Exception as e:
                print(f"[Session {session_id}] pp→browser error: {e}")

        # Run both directions concurrently
        await asyncio.gather(
            browser_to_personaplex(),
            personaplex_to_browser(),
            return_exceptions=True,
        )

    except WebSocketDisconnect:
        print(f"[Session {session_id}] Browser disconnected")
    except Exception as e:
        print(f"[Session {session_id}] Error: {e}")
    finally:
        if pp_ws:
            await pp_ws.close()
        active_sessions.pop(session_id, None)

        # Save transcript log
        if session.transcript:
            log_dir = Path("transcripts")
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / f"call_{session_id}.txt"
            log_path.write_text("\n".join(session.transcript))
            print(f"[Session {session_id}] Transcript saved → {log_path}")

        print(f"[Session {session_id}] Session ended\n")


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoint: Text-only simulation (no PersonaPlex needed)
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    """
    Text-only simulation endpoint for testing without GPU/PersonaPlex.
    Uses Gemini to generate agent responses based on the persona prompt.

    Protocol (browser → server):
        - JSON: { "type": "message", "text": "..." }

    Protocol (server → browser):
        - JSON: { "type": "transcript", "speaker": "agent"|"customer", "text": "..." }
        - JSON: { "type": "booking_update", "status": "...", "data": {...} }
    """
    await ws.accept()
    session_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = CallSession(session_id)
    active_sessions[session_id] = session

    print(f"\n[Session {session_id}] Simulation started")

    # Import Gemini for text responses
    import google.generativeai as genai
    from config import GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=persona_prompt,
    )
    chat = model.start_chat()

    # Send initial greeting
    try:
        greeting_response = chat.send_message(
            "A customer has just called. Greet them warmly."
        )
        greeting = greeting_response.text
        session.add_transcript("agent", greeting)
        await ws.send_json({
            "type": "transcript",
            "speaker": "agent",
            "text": greeting,
        })
    except Exception as e:
        print(f"[Session {session_id}] Greeting error: {e}")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            if data.get("type") == "end":
                break

            if data.get("type") == "message":
                customer_text = data.get("text", "").strip()
                if not customer_text:
                    continue

                session.add_transcript("customer", customer_text)

                # Get agent response from Gemini
                try:
                    response = chat.send_message(customer_text)
                    agent_text = response.text

                    await ws.send_json({
                        "type": "transcript",
                        "speaker": "agent",
                        "text": agent_text,
                    })

                    # Check for booking
                    booking = session.add_transcript("agent", agent_text)
                    if booking:
                        await ws.send_json({
                            "type": "booking_update",
                            "status": "confirmed",
                            "data": booking,
                        })
                    elif session.booking_status == "pending":
                        await ws.send_json({
                            "type": "booking_update",
                            "status": "pending",
                            "data": {},
                        })

                except Exception as e:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Agent error: {str(e)}",
                    })

    except WebSocketDisconnect:
        pass
    finally:
        active_sessions.pop(session_id, None)
        if session.transcript:
            log_dir = Path("transcripts")
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / f"call_{session_id}.txt"
            log_path.write_text("\n".join(session.transcript))
        print(f"[Session {session_id}] Simulation ended\n")


# ══════════════════════════════════════════════════════════════════════════════
# REST endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/bookings")
async def api_list_bookings(date: str | None = None):
    """List all bookings, optionally filtered by date."""
    return JSONResponse(list_bookings(date))


@app.get("/api/bookings/{reference}")
async def api_get_booking(reference: str):
    """Get a single booking by reference."""
    booking = get_booking(reference)
    if not booking:
        return JSONResponse({"error": "Booking not found"}, status_code=404)
    return JSONResponse(booking)


@app.post("/api/bookings/save")
async def api_save_booking(request: Request):
    """Manually save a booking from the frontend."""
    booking = await request.json()
    success = save_booking(booking)
    if success:
        return JSONResponse({"status": "saved", "reference": booking.get("reference")})
    return JSONResponse({"error": "Failed to save booking"}, status_code=500)


@app.get("/api/status")
async def api_status():
    """Server health check and status."""
    return {
        "status": "running",
        "restaurant": RESTAURANT["name"],
        "agent": RESTAURANT["agent_name"],
        "rag_context_length": len(rag_context),
        "persona_prompt_length": len(persona_prompt),
        "active_sessions": len(active_sessions),
        "personaplex_url": PERSONAPLEX_SERVER_URL,
    }


@app.get("/api/sessions")
async def api_sessions():
    """List active call sessions."""
    return [
        {
            "id": s.session_id,
            "started_at": s.started_at.isoformat(),
            "transcript_lines": len(s.transcript),
            "booking_status": s.booking_status,
        }
        for s in active_sessions.values()
    ]


# ── Serve the frontend ──────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the test console HTML."""
    return FileResponse("booking_agent_tester.html")
