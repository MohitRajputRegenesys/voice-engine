"""Provider abstractions and lightweight mocks for STT/LLM/TTS streaming."""
from __future__ import annotations
import asyncio
import re
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


def clean_spoken_text(text: str) -> str:
    """Strip Markdown/markup markers so the TTS engine never *speaks* them.

    Voice synthesis engines literally read out punctuation such as
    asterisks, underscores and backticks -- e.g. a Markdown ``" **bold** "``
    sentence is spoken as "asterisk asterisk bold asterisk asterisk". This
    normalises text for *speech* only: it removes the formatting markers
    while preserving the readable word content.

    NOTE: this is intentionally speech-only. Display/markup text returned to
    the browser (e.g. ``llm.delta`` websocket frames) is left untouched so the
    UI can still render Markdown.
    """
    if not text:
        return text

    # 1. Bold -- run before italic so `**` pairs are consumed first.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)

    # 2. Italic / emphasis.
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    # 3. Inline code -> code.
    text = re.sub(r"`(.+?)`", r"\1", text)

    # 4. Strikethrough.
    text = re.sub(r"~~(.+?)~~", r"\1", text)

    # 5. Headings / blockquotes / bullets at line start.
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", " ", text, flags=re.MULTILINE)

    # 6. Any stray asterisks/underscores/backticks left behind (e.g. unbalanced).
    text = text.replace("*", "").replace("_", "").replace("`", "")

    # 7. Collapse the whitespace left behind by the removals.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


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
            # Strip Markdown/markup markers so the mock never "speaks" asterisks etc.
            chunk = clean_spoken_text(chunk)
            if not chunk:
                continue
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
    """LLM provider backed by the saleKnowledgeBase app backend RAG API.

    Instead of a canned mock reply, this sends the user's transcript to the RAG
    backend (default ``POST /api/v1/rag/chat``), retrieves a grounded answer, and
    streams it back as word-sized `LLMDelta` objects. Multi-turn context is
    preserved by keeping a ``session_id`` so the app backend persists the
    conversation across turns; the backend derives conversation history from the
    session rather than from a client-supplied list.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_path: str = "/api/v1/rag/chat",
        top_k: int = 5,
        alpha: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_path = api_path
        self.top_k = top_k
        self.alpha = alpha
        self.session_id: Optional[str] = None
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

        # Legacy facilitator backend expected a question + full history; the app
        # backend RAG API takes a message + limit and persists its own session.
        if self.api_path.rstrip("/").endswith("/api/chat"):
            payload = {
                "question": prompt,
                "top_k": self.top_k,
                "alpha": self.alpha,
                "conversation_history": list(self.history),
            }
        else:
            payload = {"message": prompt, "limit": self.top_k}
            if self.session_id:
                payload["session_id"] = self.session_id

        resp = await self._get_client().post(f"{self.base_url}{self.api_path}", json=payload)
        if resp.status_code != 200:
            self.logger.error("RAG backend returned %s: %s", resp.status_code, resp.text[:300])
            yield LLMDelta(
                response_id=response_id,
                sequence=0,
                text="I'm sorry, I could not retrieve an answer right now.",
            )
            return
        data = resp.json()
        self.session_id = data.get("session_id", self.session_id)
        answer = data.get("answer", "") or ""
        # commit the turn to the local history (mirrors the backend session)
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
        text = clean_spoken_text("".join(parts))
        if not text:
            return
        communicate = Communicate(text, voice=self.voice, rate=self.rate, volume=self.volume)
        audio = bytearray()
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio.extend(message["data"])
        if audio:
            yield AudioChunk(response_id=response_id, sequence=0, data=bytes(audio))


def _deepgram_rejection_detail(exc) -> tuple:
    """Extract ``(status_code, body_text)`` from a websockets ``InvalidStatus``.

    Deepgram's HTTP 4xx rejection body names the exact offending parameter (e.g.
    an unavailable model), so surfacing it makes connection failures diagnosable
    without guessing.
    """
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", "?")
    raw = getattr(response, "body", b"") or b""
    try:
        body = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - body may be arbitrary bytes
        body = repr(raw)
    return status_code, body.strip()


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
        text = clean_spoken_text("".join(parts))
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
            model=os.environ.get("STT_MODEL", "nova-3") or "nova-3",
            language=os.environ.get("STT_LANGUAGE") or None,
            sample_rate=int(os.environ.get("STT_SAMPLE_RATE", "16000") or 16000),
            encoding=os.environ.get("STT_ENCODING", "linear16") or "linear16",
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


def build_twilio_stt_provider():
    """Build an STT provider tuned for Twilio phone calls (8 kHz mu-law).

    Audio stays in native mu-law end-to-end (no PCM conversion), and utterances
    auto-finalize via Deepgram endpointing/VAD instead of explicit <end> markers
    (Twilio streams continuous audio).

    The request mirrors the proven-working reference calling agent exactly:
    ``model=nova-3`` (nova-2 is deprecated and rejected by Deepgram with HTTP
    400 on newer accounts), ``channels=1`` and ``interim_results=true``.
    Language is intentionally not auto-sent on the phone path -- override with
    ``TWILIO_STT_LANGUAGE`` if needed.
    """
    if os.environ.get("DEEPGRAM_API_KEY"):
        return DeepgramSTTProvider(
            model=(
                os.environ.get("TWILIO_STT_MODEL")
                or os.environ.get("STT_MODEL")
                or "nova-3"
            ),
            language=os.environ.get("TWILIO_STT_LANGUAGE") or None,
            sample_rate=8000,
            encoding="mulaw",
            channels=1,
            interim_results=True,
            endpointing_ms=int(os.environ.get("TWILIO_SILENCE_ENDPOINT_MS", "800") or 800),
            utterance_end_ms=int(os.environ.get("TWILIO_UTTERANCE_END_MS", "1500") or 1500),
            vad_events=True,
        )
    return MockSTTProvider()


def build_twilio_tts_provider():
    """Build a TTS provider tuned for Twilio phone calls (8 kHz mu-law).

    Uses the streaming WebSocket Aura provider so raw mu-law bytes flow straight
    back to the phone (the REST Aura provider returns MP3, which is not phone-usable).
    Falls back to the mock for zero-config development/testing.
    """
    if os.environ.get("DEEPGRAM_API_KEY"):
        return DeepgramStreamingTTSProvider(
            model=os.environ.get("TTS_MODEL") or None,
            encoding="mulaw",
            sample_rate=8000,
        )
    return MockTTSProvider()


class DeepgramStreamingTTSProvider(BaseTTSProvider):
    """Deepgram Aura TTS over WebSocket with configurable output encoding.

    Unlike the REST `DeepgramTTSProvider` (which returns MP3), this provider
    streams raw audio bytes over a WebSocket using the ``Speak``/``Flush``/``Clear``
    protocol. The output encoding is configurable so Twilio calls request mu-law
    at 8 kHz (native for the phone) while browser sessions keep linear16/16 kHz.
    """

    media_type = "application/octet-stream"

    def __init__(
        self,
        model: Optional[str] = None,
        encoding: str = "linear16",
        sample_rate: int = 16000,
        container: str = "none",
    ):
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        self.model = model or os.environ.get("TTS_MODEL", "") or "aura-2-thalia-en"
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.container = container
        self.logger = logging.getLogger("DeepgramStreamingTTSProvider")
        self._ws: Optional[Any] = None
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set in environment")

    def _get_url(self) -> str:
        params = {
            "model": self.model,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
            "container": self.container,
        }
        return "wss://api.deepgram.com/v1/speak?" + urlencode(params)

    async def synthesize(self, text_iter: AsyncIterator[str], response_id: str) -> AsyncIterator[AudioChunk]:
        import websockets
        from websockets.exceptions import InvalidStatus

        url = self._get_url()
        try:
            async for chunk in self._synthesize_stream(url, text_iter, response_id):
                yield chunk
        except InvalidStatus as exc:
            status_code, body = _deepgram_rejection_detail(exc)
            self.logger.error(
                "Deepgram rejected TTS connection (HTTP %s): %s | url=%s | body=%s",
                status_code,
                exc,
                url,
                body or "<no body>",
            )
            raise RuntimeError(
                f"Deepgram TTS connection rejected (HTTP {status_code}): "
                f"{body or exc}. Check DEEPGRAM_API_KEY / TTS_MODEL / request "
                "parameters (url logged above)."
            ) from exc

    async def _synthesize_stream(
        self,
        url: str,
        text_iter: AsyncIterator[str],
        response_id: str,
    ) -> AsyncIterator[AudioChunk]:
        import websockets

        headers = {"Authorization": f"Token {self.api_key}"}
        async with websockets.connect(url, additional_headers=headers, ping_interval=20) as ws:
            self._ws = ws

            async def sender() -> None:
                try:
                    async for text in text_iter:
                        if text and text.strip():
                            await ws.send(json.dumps({"type": "Speak", "text": text}))
                    await ws.send(json.dumps({"type": "Flush"}))
                except Exception:  # noqa: BLE001 - connector teardown races
                    pass

            send_task = asyncio.create_task(sender())
            seq = 0
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        yield AudioChunk(response_id=response_id, sequence=seq, data=msg)
                        seq += 1
                    else:
                        try:
                            payload = json.loads(msg)
                        except Exception:
                            continue
                        if payload.get("type") in ("Flushed", "Cleared"):
                            break
            finally:
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                self._ws = None

    async def abort(self) -> None:
        """Send Clear to Deepgram so it stops synthesizing immediately (barge-in)."""
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.send(json.dumps({"type": "Clear"}))
            except Exception:  # noqa: BLE001
                pass


class DeepgramSTTProvider(BaseSTTProvider):
    """Deepgram realtime STT provider using WebSocket. Requires DEEPGRAM_API_KEY env var.

    This implementation streams binary audio frames from `audio_queue` to Deepgram
    and yields `STTEvent` objects for partial and final transcripts. It supports
    reconnection with backoff and cleans up on queue termination (None).

    The wire encoding is configurable: browser sessions use ``linear16`` at 16 kHz
    while Twilio phone calls use ``mulaw`` at 8 kHz (native for the phone, no PCM
    conversion and therefore no audio-format mismatch beeps).
    """
    def __init__(
        self,
        model: str = "nova-3",
        language: Optional[str] = None,
        sample_rate: int = 16000,
        encoding: str = "linear16",
        channels: Optional[int] = None,
        interim_results: bool = False,
        endpointing_ms: Optional[int] = None,
        utterance_end_ms: Optional[int] = None,
        vad_events: bool = False,
    ):
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.vad_events = vad_events
        self.logger = logging.getLogger("DeepgramSTTProvider")

    def build_request_url(self) -> str:
        """Build the Deepgram listen websocket URL from the provider configuration.

        The API key travels in the Authorization header (never in the URL), so
        the URL is safe to log in full when diagnosing connection rejections.
        """
        params = {
            "model": self.model,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
            "smart_format": "true",
        }
        if self.language:
            params["language"] = self.language
        if self.channels:
            params["channels"] = str(self.channels)
        if self.interim_results:
            params["interim_results"] = "true"
        # Auto-finalize utterances on silence/vad (used by the Twilio call path,
        # where no explicit <end> marker is ever sent).
        if self.endpointing_ms:
            params["endpointing"] = str(self.endpointing_ms)
        if self.utterance_end_ms:
            params["utterance_end_ms"] = str(self.utterance_end_ms)
        if self.vad_events:
            params["vad_events"] = "true"
        return "wss://api.deepgram.com/v1/listen?" + urlencode(params)

    async def consume_audio(self, audio_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set in environment")

        url = self.build_request_url()

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
                status_code, body = _deepgram_rejection_detail(exc)
                self.logger.error(
                    "Deepgram rejected STT connection (HTTP %s): %s | url=%s | body=%s",
                    status_code,
                    exc,
                    url,
                    body or "<no body>",
                )
                raise RuntimeError(
                    f"Deepgram STT connection rejected (HTTP {status_code}): "
                    f"{body or exc}. Check DEEPGRAM_API_KEY / STT_MODEL / request "
                    "parameters (url logged above)."
                ) from exc
            except Exception as exc:
                self.logger.exception("Deepgram connection failed: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
