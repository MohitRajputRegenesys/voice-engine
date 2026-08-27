"""High-level client for external AI projects to add "listen + speak".

Two integration styles are provided:

* **Realtime duplex** -- ``VoiceClient`` (WebSocket). Stream audio in, receive
  transcripts, and push your own AI's answer back to be spoken.
* **One-shot REST** -- ``transcribe()`` and ``synthesize()`` helpers (plain HTTP)
  for projects that just want "text of this audio" / "audio for this text".

Example (realtime, external AI drives the response):

    async with VoiceClient() as vc:
        await vc.send_audio(audio_bytes)      # from your mic
        await vc.end_utterance()              # finalize STT
        async for msg in vc.messages():       # transcript.final / audio.chunk / speak.done
            if msg.get("type") == "transcript.final":
                answer = await my_model.generate(msg["text"])   # YOUR AI
                await vc.speak(answer)        # engine speaks it aloud
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator, Optional

import httpx
import websockets


class VoiceClient:
    """Minimal wrapper around the voice-engine WebSocket gateway."""

    def __init__(self, ws_url: str = "ws://localhost:8001/ws"):
        # `auto_llm=0` keeps the engine from auto-answering; YOUR AI drives speak().
        self.ws_url = ws_url + ("&" if "?" in ws_url else "?") + "auto_llm=0"
        self._ws = None

    async def connect(self) -> "VoiceClient":
        self._ws = await websockets.connect(self.ws_url)
        return self

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> "VoiceClient":
        return await self.connect()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def send_message(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload))

    async def send_audio(self, data: bytes) -> None:
        """Stream a raw PCM16 16kHz audio frame to the engine."""
        await self.send_message({"type": "audio", "data": base64.b64encode(data).decode("ascii")})

    async def end_utterance(self) -> None:
        """Mark the end of the current utterance (finalizes STT)."""
        await self.send_message({"type": "audio", "data": "<end>"})

    async def speak(self, text: str) -> None:
        """Ask the engine to speak YOUR AI's text (synthesized via TTS)."""
        await self.send_message({"type": "speak", "data": text})

    async def messages(self) -> AsyncIterator[dict]:
        """Async iterator over decoded server messages.

        Relevant types: ``transcript.partial`` / ``transcript.final``,
        ``audio.chunk`` (base64 in ``data``), ``speak.done``, ``ping`` / ``pong``.
        """
        while True:
            raw = await self._ws.recv()
            if not isinstance(raw, str):
                # binary frames are not expected for this client
                continue
            msg = json.loads(raw)
            yield msg

    async def listen(self) -> AsyncIterator[str]:
        """Convenience: yield only final transcripts."""
        async for msg in self.messages():
            if msg.get("type") == "transcript.final":
                yield msg["text"]

    async def read_speech(self) -> AsyncIterator[bytes]:
        """Yield decoded audio.chunk bytes until a speak.done for the current turn.

        For streaming an external AI's spoken answer as it is synthesized.
        """
        async for msg in self.messages():
            if msg.get("type") == "audio.chunk":
                yield base64.b64decode(msg["data"])
            elif msg.get("type") == "speak.done":
                return


async def transcribe(data: str, base_url: str = "http://localhost:8001",
                     encoding: str = "text") -> str:
    """One-shot STT over HTTP: `data` is base64 audio (encoding="base64") or text."""
    resp = await httpx.AsyncClient().post(
        f"{base_url.rstrip('/')}/stt", json={"data": data, "encoding": encoding}
    )
    resp.raise_for_status()
    return resp.json()["text"]


async def synthesize(text: str, base_url: str = "http://localhost:8001") -> bytes:
    """One-shot TTS over HTTP: returns raw synthesized audio bytes."""
    resp = await httpx.AsyncClient().post(
        f"{base_url.rstrip('/')}/tts", json={"text": text}
    )
    resp.raise_for_status()
    return resp.content
