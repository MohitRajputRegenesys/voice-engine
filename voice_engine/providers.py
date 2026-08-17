"""Provider abstractions and lightweight mocks for STT/LLM/TTS streaming."""
from __future__ import annotations
import asyncio
import uuid
from typing import AsyncIterator, Dict, Any, Optional
import os
import json
import logging
import websockets
from websockets import ConnectionClosedError


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
                headers = [("Authorization", f"Token {self.api_key}")]
                async with websockets.connect(url, extra_headers=headers, ping_interval=20) as ws:
                    self.logger.info("Deepgram websocket connected")

                    async def sender():
                        while True:
                            data = await audio_queue.get()
                            if data is None:
                                return
                            if data == b"<end>":
                                continue
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
            except Exception as exc:
                self.logger.exception("Deepgram connection failed: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
