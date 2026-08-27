"""One-shot, REST-friendly STT/TTS helpers used by the HTTP endpoints.

These wrap the same streaming providers as the realtime WebSocket path but run a
single, bounded request so external AI projects can call them over plain HTTP
("give me the text of this audio" / "give me audio for this text").
"""
from __future__ import annotations

import asyncio
import uuid

from .providers import build_stt_provider, build_tts_provider  # noqa: F401  (re-exported for convenience)


async def transcribe_once(data: bytes, stt_provider=None) -> str:
    """Run a single finalized audio buffer through STT and return the text.

    Works uniformly for Mock and Deepgram providers: the bytes are fed in followed
    by the ``<end>`` marker, then the first final transcript is collected.
    """
    provider = stt_provider or build_stt_provider()
    queue: asyncio.Queue = asyncio.Queue()

    async def collect():
        async for event in provider.consume_audio(queue):
            if event.final:
                return event.final
        return None

    task = asyncio.create_task(collect())
    queue.put_nowait(data)
    queue.put_nowait(b"<end>")
    try:
        result = await asyncio.wait_for(task, timeout=30.0)
    finally:
        if not task.done():
            queue.put_nowait(None)
            task.cancel()
    return result or ""


async def synthesize_once(text: str, tts_provider=None) -> bytes:
    """Synthesize a single string of text to audio bytes via the TTS provider."""
    provider = tts_provider or build_tts_provider()

    async def text_iter():
        yield text

    chunks = []
    async for audio in provider.synthesize(text_iter(), str(uuid.uuid4())):
        chunks.append(audio.data)
    return b"".join(chunks)
