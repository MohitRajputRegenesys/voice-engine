# Voice Engine — Realtime conversational voice runtime

Shared **STT / TTS** module powering speech + listen for the **AI Facilitator**
chatbot. Implements a production-oriented session/state machine, bounded
queues, cancellation/barge-in handling, and pluggable providers.

## Providers

| Layer | Default (mock) | Real (opt-in) |
|-------|----------------|---------------|
| **STT** | `MockSTTProvider` | `DeepgramSTTProvider` (needs `DEEPGRAM_API_KEY`) |
| **LLM** | `MockLLMProvider` | `FacilitatorLLMProvider` → calls the AI Facilitator `/api/chat` (RAG) |
| **TTS** | `MockTTSProvider` | `EdgeTTSProvider` (free Microsoft Edge TTS, no key) |

Provider selection is driven by environment variables — see `.env.example`.

## Setup (uv)

```bash
cp .env.example .env   # then fill in keys you want to enable
uv sync
```

## Run the voice gateway (port 8001)

```bash
uv run uvicorn voice_engine.server:app --port 8001 --reload
```

> Port 8001 is used so the voice gateway
> backend, which runs on 8000.

## Exercise the mocked flow

```bash
uv run python example_client.py
```

The client opens `ws://localhost:8001/ws`, sends audio frames + an `<end>`
marker, and prints the `transcript.partial` / `transcript.final` →
`llm.delta` → `audio.chunk` events.

## WebSocket protocol

**Client → server**
- `{"type":"audio","data":"<end>"}` — mark end of utterance (finalize STT)
- `{"type":"audio","data":"<base64 raw PCM16 16kHz>"}` — real audio frame
- `{"type":"audio","data":"text"}` — demo text frame (mock path)

**Server → client**
- `transcript.partial` / `transcript.final` — live + committed user speech
- `llm.delta` — streaming LLM (facilitator) text
- `audio.chunk` — `data` is base64 MP3 audio to play back (TTS)
- `ping` / `pong` — heartbeat

## Wiring into the AI Facilitator

1. Start the facilitator backend on `http://localhost:8000` (its `/api/chat`
   becomes the voice pipeline's brain via `FacilitatorLLMProvider`).
2. Run this voice gateway on `8001`.
3. Enable real speech in `.env`:
   - `DEEPGRAM_API_KEY=...` for real listening
   - `TTS_PROVIDER=edge` for real speaking
   - (leave `FACILITATOR_API_URL=http://localhost:8000` to answer from the RAG)
4. The facilitator's Next.js frontend connects to `ws://localhost:8001/ws` —
   the mic streams audio there, and returning `audio.chunk` is played aloud.

## Using it as a "voice layer" for your own AI project

The engine can be used as just **listen + speak** for an AI project that already
has its own brain (i.e. no facilitator). Connect with `?auto_llm=0` so the engine
does STT only and lets **your** AI decide the answer; push that answer back and the
engine speaks it. The `VoiceClient` SDK (`voice_engine.client`) hides the protocol.

```python
import asyncio
from voice_engine.client import VoiceClient

async def run():
    async with VoiceClient() as vc:                       # ws://.../ws?auto_llm=0
        await vc.send_audio(audio_bytes)                  # your mic (PCM16 16kHz)
        await vc.end_utterance()                          # finalize STT
        async for msg in vc.messages():
            if msg.get("type") == "transcript.final":
                answer = await my_model.generate(msg["text"])   # YOUR AI
                await vc.speak(answer)                          # engine speaks it
```

Run a full working demo: `uv run python external_ai_example.py`.

### One-shot REST (no WebSocket) — easiest for any language/framework

- `POST /stt` — `{"data":"<base64 PCM16 16kHz audio>", "encoding":"base64"}` →
  `{"text":"recognized transcript"}` (with `"encoding":"text"` it echoes a mock
  transcript). Real STT needs `DEEPGRAM_API_KEY`.
- `POST /tts` — `{"text":"hello"}` → raw audio bytes
  (`audio/mpeg` when `TTS_PROVIDER=edge`, else `application/octet-stream`).

Client helpers: `await transcribe(...)` and `await synthesize(...)` (HTTP).

### WebSocket protocol (new messages)

- **Client → server**: `{"type":"speak","data":"<text>"}` — external AI asks the
  engine to speak arbitrary text (TTS + `audio.chunk` back).
- **Server → client**: `speak.done` — emitted when an external `speak` turn ends.

The existing `audio`/`transcript`/`llm.delta`/`audio.chunk` messages are unchanged,
so the facilitator flow and the external-AI flow share the same pipeline.

## Tests

```bash
.venv\Scripts\python -m pytest -q
```


