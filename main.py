"""
main.py
-------
Entry point for the Bella Notte voice booking agent.

Usage:
    python main.py

Prerequisites:
    1. Copy .env.example to .env and fill in your API key
    2. Start the PersonaPlex server:
         SSL_DIR=$(mktemp -d)
         python -m moshi.server --ssl "$SSL_DIR" --hf-repo nvidia/personaplex-7b-v1
    3. Drop your documents into the docs/ folder
    4. Run: python main.py
"""

import asyncio
import json
import ssl
from datetime import datetime

import numpy as np
import sounddevice as sd
import websockets

from config import PERSONAPLEX_SERVER_URL, VOICE_ID, SAMPLE_RATE, CHUNK_AUDIO, RESTAURANT
from database import init_db, save_booking
from rag import setup_rag
from agent import build_persona_prompt, is_confirmation, extract_booking_from_transcript


class BookingAgent:
    def __init__(self, persona_prompt: str):
        self.persona_prompt = persona_prompt
        self.transcript:    list[str] = []
        self.booking_saved: bool = False

    def _ssl_ctx(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _on_text_token(self, token: str) -> None:
        """Accumulate transcript and trigger DB save on confirmation."""
        self.transcript.append(token)

        if not self.booking_saved and is_confirmation(token):
            full = "".join(self.transcript)
            print("\n🔍 Confirmation detected — extracting booking details…")
            booking = extract_booking_from_transcript(full)
            if booking:
                save_booking(booking)
                self.booking_saved = True
            else:
                print("⚠️  Could not extract booking from transcript.")

    async def run(self) -> None:
        print(f"\n🍽️  {RESTAURANT['name']} — Booking Agent")
        print(f"   Agent : {RESTAURANT['agent_name']}  |  Voice: {VOICE_ID}")
        print(f"   Server: {PERSONAPLEX_SERVER_URL}")
        print("   Connecting to PersonaPlex…\n")

        async with websockets.connect(PERSONAPLEX_SERVER_URL, ssl=self._ssl_ctx()) as ws:
            await ws.send(json.dumps({
                "type":          "init",
                "voice":         VOICE_ID,
                "system_prompt": self.persona_prompt,
            }))
            print("✅ Connected. Speak to start the reservation.\n")
            print("   (Press Ctrl+C to end the call)\n")

            input_q:  asyncio.Queue = asyncio.Queue()
            output_q: asyncio.Queue = asyncio.Queue()

            def mic_cb(indata, frames, time, status):
                input_q.put_nowait(indata.copy())

            def spk_cb(outdata, frames, time, status):
                try:
                    chunk = output_q.get_nowait()
                    outdata[:] = chunk.reshape(-1, 1)
                except asyncio.QueueEmpty:
                    outdata[:] = np.zeros((frames, 1), dtype=np.float32)

            async def send_audio():
                while True:
                    chunk = await input_q.get()
                    pcm = (chunk * 32767).astype(np.int16).tobytes()
                    await ws.send(pcm)

            async def recv_messages():
                async for msg in ws:
                    if isinstance(msg, bytes):
                        pcm = np.frombuffer(msg, dtype=np.int16)
                        await output_q.put(pcm.astype(np.float32) / 32767.0)
                    elif isinstance(msg, str):
                        try:
                            data = json.loads(msg)
                            token = data.get("text", "")
                        except json.JSONDecodeError:
                            token = msg
                        if token:
                            self._on_text_token(token)

            with (
                sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                               blocksize=CHUNK_AUDIO, callback=mic_cb),
                sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=CHUNK_AUDIO, callback=spk_cb),
            ):
                await asyncio.gather(send_audio(), recv_messages())


async def main() -> None:
    # 1. Init DB
    init_db()

    # 2. Build RAG context from docs/
    rag_context = setup_rag()

    # 3. Build persona prompt
    persona = build_persona_prompt(rag_context=rag_context)

    # 4. Run agent
    agent = BookingAgent(persona_prompt=persona)
    try:
        await agent.run()
    except KeyboardInterrupt:
        if agent.transcript:
            log_path = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_path, "w") as f:
                f.write("".join(agent.transcript))
            print(f"\n📝 Transcript saved → {log_path}")
        print("\n📞 Call ended. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
