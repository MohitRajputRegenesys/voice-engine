"""Provider abstractions and lightweight mocks for STT/LLM/TTS streaming."""
from __future__ import annotations
import asyncio
import uuid
from typing import AsyncIterator, Dict, Any, Optional
import os
import json
import logging
import websockets
from urllib.parse import urlencode
from websockets import ConnectionClosed, ConnectionClosedError
from websockets.exceptions import InvalidStatus


class STTEvent:
    def __init__(self, response_id: str, sequence: int, partial: Optional[str]=None, final: Optional[str]=None):
        self.response_id = response_id
        self.sequence = sequence
        self.partial = partial
        self.final = final


class LLMDelta:
    def __init__(self, response_id: str, sequence: int, text: str):
        self.response_id = response_id
        self.sequence = sequence
        self.text = text


class AudioChunk:
    def __init__(self, response_id: str, sequence: int, data: bytes):
        self.response_id = response_id
        self.sequence = sequence
        self.data = data


class BaseSTTProvider:
    async def consume_audio(self, audio_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
        raise NotImplementedError()


class MockSTTProvider(BaseSTTProvider):
    """Mock STT that emits partials then final transcript for each 'utterance' boundary.

    Protocol: audio_queue yields b"<end>" to mark end of utterance.
    """
    async def consume_audio(self, audio_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
        seq = 0
        while True:
            audio = await audio_queue.get()
            if audio is None:
                return
            # treat special marker
            if audio == b"<end>":
                # emit final
                response_id = str(uuid.uuid4())
                yield STTEvent(response_id=response_id, sequence=seq, final="mock final transcript")
                seq += 1
                continue
            # emit a partial for other audio chunks
            response_id = str(uuid.uuid4())
            yield STTEvent(response_id=response_id, sequence=seq, partial="mock partial")
            seq += 1


class BaseLLMProvider:
    async def stream_response(self, prompt: str, response_id: str) -> AsyncIterator[LLMDelta]:
        raise NotImplementedError()


class MockLLMProvider(BaseLLMProvider):
    async def stream_response(self, prompt: str, response_id: str) -> AsyncIterator[LLMDelta]:
        # stream token-like deltas
        seq = 0
        parts = ("Hello ", "this is a ", "streamed response.")
        for p in parts:
            await asyncio.sleep(0.2)
            yield LLMDelta(response_id=response_id, sequence=seq, text=p)
            seq += 1


class BaseTTSProvider:
    # Content type describing bytes returned by `synthesize`; used by the REST
    # /tts endpoint so clients know how to play the payload.
    media_type = "application/octet-stream"

    async def synthesize(self, text_iter: AsyncIterator[str], response_id: str) -> AsyncIterator[AudioChunk]:
        raise NotImplementedError()


class MockTTSProvider(BaseTTSProvider):
    async def synthesize(self, text_iter: AsyncIterator[str], response_id: str) -> AsyncIterator[AudioChunk]:
        seq = 0
        async for chunk in text_iter:
            # pretend to synthesize chunk into audio
            await asyncio.sleep(0.1)
            data = f"AUDIO({chunk})".encode("utf-8")
            yield AudioChunk(response_id=response_id, sequence=seq, data=data)
            seq += 1


def _chunk_text(text: str, size: int = 120) -> list:
    """Split text into reasonably sized word chunks for streaming deltas."""
    text = text.strip()
    if not text:
        return [""]
    words = text.split()
    chunks: list = []
    cur = ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > size:
            chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur)
    return chunks or [""]


class FacilitatorLLMProvider(BaseLLMProvider):
    """LLM provider backed by the AI Facilitator's RAG `/api/chat` endpoint.

    Instead of a canned mock reply, this sends the user's transcript to the
    facilitator backend, retrieves a grounded teacher answer, and streams it
    back as sentence-level `LLMDelta` objects. Conversational history is kept
    so multi-turn context is preserved across questions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        top_k: int = 5,
        alpha: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.top_k = top_k
        self.alpha = alpha
        self.history: list = []
        self.logger = logging.getLogger("FacilitatorLLMProvider")
        self._client: Optional[Any] = None  # httpx.AsyncClient, created lazily

    def _get_client(self):
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def stream_response(self, prompt: str, response_id: str) -> AsyncIterator[LLMDelta]:
        import httpx

        payload = {
            "question": prompt,
            "top_k": self.top_k,
            "alpha": self.alpha,
            "conversation_history": self.history,
        }
        if self._get_client() is None:
            return
        resp = await self._get_client().post(f"{self.base_url}/api/chat", json=payload)
        if resp.status_code != 200:
            self.logger.error("Facilitator /api/chat returned %s: %s", resp.status_code, resp.text)
            yield LLMDelta(
                response_id=response_id,
                sequence=0,
                text="I'm sorry, I could not retrieve an answer right now.",
            )
            return
        data = resp.json()
        answer = data.get("answer", "") or ""
        # commit the turn to the conversation history
        self.history.append({"role": "student", "content": prompt})
        self.history.append({"role": "teacher", "content": answer})
        # stream as small deltas for a natural token-like flow
        for seq, piece in enumerate(_chunk_text(answer)):
            await asyncio.sleep(0.05)
            yield LLMDelta(response_id=response_id, sequence=seq, text=piece)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


class EdgeTTSProvider(BaseTTSProvider):
    """TTS provider using Microsoft Edge TTS (free, no API key required).

    Accumulates the streamed answer text, synthesizes it to MP3 audio via
    `edge-tts`, and yields the resulting audio as a single `AudioChunk`.
    """

    media_type = "audio/mpeg"

    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", volume: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume

    async def synthesize(self, text_iter: AsyncIterator[str], response_id: str) -> AsyncIterator[AudioChunk]:
        from edge_tts import Communicate  # lazy import keeps optional dependency

        parts = []
        async for chunk in text_iter:
            parts.append(chunk)
        text = "".join(parts).strip()
        if not text:
            return
        communicate = Communicate(text, voice=self.voice, rate=self.rate, volume=self.volume)
        audio = bytearray()
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio.extend(message["data"])
        if audio:
            yield AudioChunk(response_id=response_id, sequence=0, data=bytes(audio))


class DeepgramTTSProvider(BaseTTSProvider):
    """TTS provider using Deepgram's Aura voices (REST /v1/speak endpoint).

    Streams MP3 audio back for the given text. Aura models accept up to 2000
    characters per request, so longer text is word-packed into segments below
    that limit and each synthesized segment is yielded as its own AudioChunk.
    """

    media_type = "audio/mpeg"
    API_URL = "https://api.deepgram.com/v1/speak"
    MAX_CHARS = 2000

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        self.model = model or os.environ.get("TTS_MODEL", "") or "aura-2-thalia-en"
        self.logger = logging.getLogger("DeepgramTTSProvider")
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set in environment")

    async def _synthesize_segment(self, client: Any, text: str) -> Optional[bytes]:
        resp = await client.post(
            self.API_URL,
            params={"model": self.model},
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
        if resp.status_code != 200:
            self.logger.error("Deepgram TTS returned %s: %s", resp.status_code, resp.text[:300])
            return None
        return resp.content

    async def synthesize(self, text_iter: AsyncIterator[str], response_id: str) -> AsyncIterator[AudioChunk]:
        import httpx

        parts = []
        async for chunk in text_iter:
            parts.append(chunk)
        text = "".join(parts).strip()
        if not text:
            return

        # Respect Aura's per-request character limit.
        if len(text) <= self.MAX_CHARS:
            segments = [text]
        else:
            segments = [s for s in _chunk_text(text, size=self.MAX_CHARS - 200) if s.strip()]

        seq = 0
        async with httpx.AsyncClient(timeout=60.0) as client:
            for segment in segments:
                try:
                    audio = await self._synthesize_segment(client, segment)
                except httpx.HTTPError as exc:
                    self.logger.error("Deepgram TTS request failed: %s", exc)
                    continue
                if audio:
                    yield AudioChunk(response_id=response_id, sequence=seq, data=audio)
                    seq += 1


def build_stt_provider():
    """Select the STT backend from environment configuration.

    DEEPGRAM_API_KEY set -> Deepgram streaming STT; otherwise mock.
    """
    if os.environ.get("DEEPGRAM_API_KEY"):
        return DeepgramSTTProvider(
            model=os.environ.get("STT_MODEL", "nova-2") or "nova-2",
            language=os.environ.get("STT_LANGUAGE") or None,
            sample_rate=int(os.environ.get("STT_SAMPLE_RATE", "16000") or 16000),
        )
    return MockSTTProvider()


def build_tts_provider():
    """Select the TTS backend from environment configuration.

    Priority:
      - TTS_PROVIDER="deepgram" -> Deepgram Aura (requires DEEPGRAM_API_KEY)
      - TTS_PROVIDER="edge"     -> free Microsoft Edge TTS
      - TTS_PROVIDER=""         -> auto: Deepgram Aura when the key is present,
                                   otherwise mock (no vendor lock-in, no cost)
    """
    prov = os.environ.get("TTS_PROVIDER", "").strip().lower()
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")

    if prov == "":
        if deepgram_key:
            return DeepgramTTSProvider()
        return MockTTSProvider()

    if prov in ("deepgram", "deepgram-tts", "aura"):
        if not deepgram_key:
            raise RuntimeError("TTS_PROVIDER=deepgram requires DEEPGRAM_API_KEY")
        return DeepgramTTSProvider(model=os.environ.get("TTS_MODEL") or None)

    if prov in ("edge", "edge-tts"):
        return EdgeTTSProvider(voice=os.environ.get("TTS_VOICE", "en-US-AriaNeural"))

    logging.getLogger(__name__).warning("Unknown TTS_PROVIDER=%r, falling back to mock", prov)
    return MockTTSProvider()


class DeepgramSTTProvider(BaseSTTProvider):
    """Deepgram realtime STT provider using WebSocket. Requires DEEPGRAM_API_KEY env var.

    This implementation streams binary audio frames from `audio_queue` to Deepgram
    and yields `STTEvent` objects for partial and final transcripts. It supports
    reconnection with backoff and cleans up on queue termination (None).
    """
    def __init__(self, model: str = "nova-2", language: Optional[str] = None, sample_rate: int = 16000):
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.logger = logging.getLogger("DeepgramSTTProvider")

    async def consume_audio(self, audio_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set in environment")

        params = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "smart_format": "true",
        }
        if self.language:
            params["language"] = self.language
        url = "wss://api.deepgram.com/v1/listen?" + urlencode(params)

        backoff = 1.0
        sequence = 0
        while True:
            try:
                headers = {"Authorization": f"Token {self.api_key}"}
                async with websockets.connect(url, additional_headers=headers, ping_interval=20) as ws:
                    self.logger.info("Deepgram websocket connected")

                    async def sender():
                        while True:
                            data = await audio_queue.get()
                            if data is None:
                                try:
                                    await ws.send(json.dumps({"type": "CloseStream"}))
                                except Exception:
                                    pass
                                return
                            if data == b"<end>":
                                # Deterministic finalization: ask Deepgram to wrap
                                # up the current stream so a final transcript is
                                # emitted promptly (instead of waiting on silence).
                                try:
                                    await ws.send(json.dumps({"type": "CloseStream"}))
                                except ConnectionClosedError:
                                    pass
                                return
                            try:
                                await ws.send(data)
                            except ConnectionClosedError:
                                return

                    send_task = asyncio.create_task(sender())
                    try:
                        async for msg in ws:
                            try:
                                payload = json.loads(msg)
                            except Exception:
                                continue

                            if not isinstance(payload, dict):
                                continue

                            transcript_text = None
                            is_final = False
                            channel = payload.get("channel") or {}
                            alternatives = channel.get("alternatives") if isinstance(channel, dict) else None
                            if alternatives and isinstance(alternatives, list) and len(alternatives) > 0:
                                alt = alternatives[0]
                                transcript_text = alt.get("transcript")
                                is_final = alt.get("is_final") or payload.get("is_final") or False

                            if transcript_text is None and "transcript" in payload:
                                transcript_text = payload.get("transcript")
                                is_final = payload.get("is_final", False)

                            if transcript_text is None:
                                continue

                            if is_final:
                                yield STTEvent(response_id=str(uuid.uuid4()), sequence=sequence, final=transcript_text)
                            else:
                                yield STTEvent(response_id=str(uuid.uuid4()), sequence=sequence, partial=transcript_text)
                            sequence += 1
                    finally:
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
            except ConnectionClosed:
                # Normal end-of-stream after CloseStream; reconnect quietly for
                # the next utterance rather than logging a scary traceback.
                await asyncio.sleep(0.5)
                continue
            except InvalidStatus as exc:
                # Auth/model/config problems won't heal by retrying - fail fast
                # with an actionable error instead of an endless reconnect loop.
                self.logger.error("Deepgram rejected connection (check DEEPGRAM_API_KEY / STT_MODEL): %s", exc)
                raise RuntimeError(f"Deepgram connection rejected: {exc}") from exc
            except Exception as exc:
                self.logger.exception("Deepgram connection failed: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
