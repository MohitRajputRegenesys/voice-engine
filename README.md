# Voice Engine — drop-in **listen + speak** layer for any AI project

A small realtime voice runtime that adds speech I/O around *your* AI brain.
It exposes a WebSocket gateway + one-shot REST endpoints backed by pluggable
providers — **Deepgram** for STT *and* TTS (Aura) by default, with mocks for
zero-config development and free Edge-TTS as an opt-in fallback.

```
mic ──PCM16 16kHz──►  /ws or /stt ──► transcript.final ──► YOUR AI ──► speak(text)
                                                                    ◄── audio.chunk (MP3)
```

## Two integration modes

| Mode | Best for | API |
|------|----------|-----|
| **Realtime duplex** | streaming UX, partial transcripts, barge-in | `VoiceClient` over `/ws` |
| **One-shot REST** | simplest wiring from any language/framework | `POST /stt`, `POST /tts` |

### Providers

| Layer | Auto-selected real backend | Mock (default when no keys) | Opt-in |
|-------|---------------------------|------------------------------|--------|
| **STT** | `DeepgramSTTProvider` (WS streaming, needs `DEEPGRAM_API_KEY`) | `MockSTTProvider` | — |
| **TTS** | `DeepgramTTSProvider` (**Deepgram Aura**, same key) | `MockTTSProvider` | `EdgeTTSProvider` (`TTS_PROVIDER=edge`, free) |
| **LLM brain** | yours (see modes below) | `MockLLMProvider` | optional HTTP adapter (`FacilitatorLLMProvider`) |

Selection lives in one place: `voice_engine/providers.py::build_stt_provider()`
and `build_tts_provider()` — see `.env.example` for every knob.

## Install

```bash
pip install git+https://github.com/<you>/voice-engine.git            # SDK only (client)
pip install "git+https://github.com/<you>/voice-engine.git[server]" # incl. gateway
```

Local development with [uv](https://docs.astral.sh/uv/):

```bash
cp .env.example .env    # add your DEEPGRAM_API_KEY
uv sync --all-extras
```

## Run the gateway (port 8001)

```bash
uv run uvicorn voice_engine.server:app --port 8001 --reload
```

Check your live setup end-to-end (TTS + optional STT on a WAV):

```bash
uv run python smoke_live.py              # synthesizes smoke_tts.mp3 via configured TTS
uv run python smoke_live.py sample.wav   # also transcribes a 16k mono PCM16 WAV
uv run python probe_stt_ws.py            # verifies Deepgram WS auth / model / language
```

## Mode 1 — bring your own AI (realtime)

The engine listens, streams `transcript.final` events, and speaks whatever
**your** model decides — it never answers by itself when you connect with
`?auto_llm=0`. The `VoiceClient` SDK hides the protocol:

```python
import asyncio
from voice_engine.client import VoiceClient

async def run():
    async with VoiceClient() as vc:                     # ws://localhost:8001/ws?auto_llm=0
        await vc.send_audio(mic_pcm16_16khz_frame())    # stream user speech
        await vc.end_utterance()                        # finalize STT
        async for msg in vc.messages():
            if msg.get("type") == "transcript.final":
                answer = await my_model.generate(msg["text"])   # YOUR AI here
                await vc.speak(answer)                          # engine speaks it
                await vc.read_speech()                          # consume MP3 chunks until done

asyncio.run(run())
```

Full runnable demo: `uv run python external_ai_example.py`
(engine-own-LLM demo: `example_client.py`).

### One-shot REST (any language/framework)

```bash
curl -X POST localhost:8001/tts -H 'Content-Type: application/json' \
     -d '{"text":"hello"}' -o hello.mp3        # audio/mpeg via Aura

curl -X POST localhost:8001/stt -H 'Content-Type: application/json' \
     -d '{"data":"<base64 PCM16 16kHz>","encoding":"base64"}'
```

Python helpers: `from voice_engine.client import transcribe, synthesize`.

## WebSocket protocol

**Client → server**
- `{"type":"audio","data":"<end>"}` — end of utterance (finalize STT)
- `{"type":"audio","data":"<base64 raw PCM16 16kHz>"}` — audio frame
- `{"type":"speak","data":"<text>"}` — external AI drives a spoken turn
- `{"type":"ping"}` → `"pong"`

**Server → client**
- `transcript.partial` / `transcript.final` — live + committed user speech
- `llm.delta` — streamed answer text (only in auto-LLM mode)
- `audio.chunk` — base64 MP3 to play back
- `speak.done` — end of an externally-driven spoken turn
- `ping` heartbeat every 10 s (ignore if not needed)

Barge-in: sending audio while the engine is speaking cancels the current utterance.

## Twilio calling feature (phone calls)

The engine can also run a full **phone call** through Twilio Media Streams using
the engine's own STT/TTS providers and the saleKnowledgeBase app-backend RAG API
as the brain -- no code outside this package is required.

```
phone ──mu-law 8kHz──► Twilio Media Stream ──► /media-stream ──► Deepgram STT (mulaw/8000)
                                                                        │
phone ◄──mu-law 8kHz── Twilio ◄── base64 media ── /media-stream ◄──── Deepgram Aura TTS (mulaw/8000)
                                                                        ▲
                                             RAG answer from app backend (POST /api/v1/rag/chat)
```

Audio stays in native 8 kHz mu-law end-to-end (no PCM conversions), and the
turn flow mirrors the reference calling agent: 180 ms pre-buffering, mark-based
sequencing, and instant RMS barge-in.

### Endpoints added

| Endpoint | Purpose |
|----------|---------|
| `WS /media-stream` | Twilio Media Streams bidirectional audio |
| `POST /api/twilio/voice` | Inbound call TwiML (X-Twilio-Signature validated) |
| `POST /api/twilio/status` | Call status webhook (signature validated) |
| `POST /api/calls/outbound` | Outbound call trigger (X-API-Key protected) |

### Configuration

See `.env.example` — the Twilio block:

```bash
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
TWILIO_BASE_URL=your-public-host.ngrok-free.dev  # PUBLIC host Twilio can reach (NOT localhost!)
CALLS_API_KEY=secret-api-key
FACILITATOR_API_URL=http://localhost:8000   # saleKnowledgeBase app backend
FACILITATOR_API_PATH=/api/v1/rag/chat       # RAG chat endpoint (the brain)
```

> **Important:** `TWILIO_BASE_URL` must be a **public**, reachable host (e.g. an
> ngrok tunnel: `ngrok http 8001`) — *not* `localhost`. Twilio rejects local
> hosts with error **11100 "Invalid URL"** because it cannot reach your machine
> directly.

### Troubleshooting: `Deepgram rejected connection ... HTTP 400`

An HTTP 400 from Deepgram means the **request parameters** were rejected (a bad
key gives 401 instead). The most common cause is a deprecated `STT_MODEL` —
`nova-2` is no longer accepted by Deepgram on newer accounts; use `nova-3` (the
model the reference calling agent runs on). On any rejection the engine now
logs the full request URL plus Deepgram's response body, which names the exact
offending parameter.

### Triggering an outbound call

```bash
curl -X POST localhost:8001/api/calls/outbound \
     -H 'Content-Type: application/json' -H 'X-API-Key: secret-api-key' \
     -d '{"to":"+1234567890"}'
```

For inbound calls, point your Twilio Voice webhook at
`https://<your-host>/api/twilio/voice`.

### Hold music while the AI thinks

While the RAG answer is being generated, the engine streams a soft, seamless
music bed to the caller instead of dead air, and stops it the instant the first
spoken byte is ready (music and speech can never overlap). If the caller starts
speaking during the gap, the music stops immediately. Configure via:

| Variable | Default | Purpose |
|----------|---------|---------|
| `TWILIO_HOLD_MUSIC` | `true` | Enable/disable the hold-music bed |
| `TWILIO_HOLD_MUSIC_FILE` | *(built-in pad)* | Custom track: raw mu-law 8 kHz mono, or 16-bit 8 kHz mono WAV |
| `TWILIO_HOLD_MUSIC_GAIN` | `0.3` | Volume, 0.0 (silent) to 1.0 |
| `TWILIO_HOLD_MUSIC_DELAY_MS` | `400` | Wait this long before music starts (fast answers stay clean) |

The built-in pad is procedurally synthesized (royalty-free by construction) and
loops sample-exact seamlessly.

## Tests

```bash
uv run pytest -q
```

Unit tests are fully mocked (no network/key required); use `smoke_live.py` to
verify your real credentials.


