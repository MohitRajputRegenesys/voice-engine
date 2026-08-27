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

## Tests

```bash
uv run pytest -q
```

Unit tests are fully mocked (no network/key required); use `smoke_live.py` to
verify your real credentials.


