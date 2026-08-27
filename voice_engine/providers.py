"""Provider abstractions and lightweight mocks for STT/LLM/TTS streaming."""
from __future__ import annotations
import asyncio
import uuid
from typing import AsyncIterator, Dict, Any, Optional
import os
import json
import logging
import websockets
from websockets import ConnectionClosed, ConnectionClosedError


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


class DeepgramSTTProvider(BaseSTTProvider):
    """Deepgram realtime STT provider using WebSocket. Requires DEEPGRAM_API_KEY env var.

    This implementation streams binary audio frames from `audio_queue` to Deepgram
    and yields `STTEvent` objects for partial and final transcripts. It supports
    reconnection with backoff and cleans up on queue termination (None).
    """
    def __init__(self, model: str = "general/enhanced", sample_rate: int = 16000):
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        self.model = model
        self.sample_rate = sample_rate
        self.logger = logging.getLogger("DeepgramSTTProvider")

    async def consume_audio(self, audio_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set in environment")

        url = f"wss://api.deepgram.com/v1/listen?model={self.model}&encoding=linear16&sample_rate={self.sample_rate}"

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
            except Exception as exc:
                self.logger.exception("Deepgram connection failed: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
