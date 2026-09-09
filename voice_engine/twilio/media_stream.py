"""FastAPI WebSocket endpoint for Twilio Media Streams (/media-stream).

The Twilio call path keeps audio in native 8 kHz mu-law end-to-end:
Twilio frames are decoded and fed straight to the engine's STT provider (Deepgram
with ``encoding=mulaw&sample_rate=8000``), and the engine's streaming Aura TTS
provider emits mu-law bytes that are sent straight back to Twilio -- no PCM
conversions anywhere in the hot path, so there are no audio-format mismatch beeps.
"""
import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..providers import (
    FacilitatorLLMProvider,
    build_twilio_stt_provider,
    build_twilio_tts_provider,
)
from .audio import decode_mulaw_base64, encode_mulaw_base64
from .session import twilio_call_registry
from .turn import TwilioTurnOrchestrator

logger = logging.getLogger("voice_engine.twilio.media_stream")
router = APIRouter(tags=["twilio"])


async def send_twilio_media(websocket: WebSocket, stream_sid: str, mulaw_bytes: bytes) -> None:
    """Send a base64 mu-law audio frame to the Twilio Media Stream WebSocket."""
    if not mulaw_bytes or not stream_sid:
        return

    payload = encode_mulaw_base64(mulaw_bytes)
    message = {"event": "media", "streamSid": stream_sid, "media": {"payload": payload}}
    try:
        await websocket.send_text(json.dumps(message))
    except Exception as e:
        logger.error("Failed to send media frame to Twilio", extra={"stream_sid": stream_sid, "error": str(e)})


async def send_twilio_mark(websocket: WebSocket, stream_sid: str, mark_name: str) -> None:
    """Send a mark sequencing event to the Twilio Media Stream WebSocket."""
    if not stream_sid or not mark_name:
        return

    message = {"event": "mark", "streamSid": stream_sid, "mark": {"name": mark_name}}
    try:
        await websocket.send_text(json.dumps(message))
    except Exception as e:
        logger.error("Failed to send mark event to Twilio", extra={"stream_sid": stream_sid, "error": str(e)})


async def send_twilio_clear(websocket: WebSocket, stream_sid: str) -> None:
    """Send a clear buffer event to instantly stop speaker playback on barge-in."""
    if not stream_sid:
        return

    message = {"event": "clear", "streamSid": stream_sid}
    try:
        await websocket.send_text(json.dumps(message))
    except Exception as e:
        logger.error("Failed to send clear event to Twilio", extra={"stream_sid": stream_sid, "error": str(e)})


def _build_call_providers():
    """Build the STT/TTS/LLM providers for a Twilio call.

    Tests monkeypatch this function to inject fakes. The LLM brain comes from
    the saleKnowledgeBase app backend RAG API.
    """
    stt_provider = build_twilio_stt_provider()
    tts_provider = build_twilio_tts_provider()
    llm_provider = FacilitatorLLMProvider(
        base_url=os.environ.get("FACILITATOR_API_URL", "http://localhost:8000"),
        api_path=os.environ.get("FACILITATOR_API_PATH", "/api/v1/rag/chat"),
        top_k=int(os.environ.get("FACILITATOR_TOP_K", "5")),
        alpha=float(os.environ.get("FACILITATOR_ALPHA", "0.5")),
    )
    return stt_provider, tts_provider, llm_provider
@router.websocket("/media-stream")
async def websocket_media_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint handling bidirectional audio stream with Twilio.


    Protocol (Twilio -> engine -> Twilio):
      * ``start``   -- register a call session, boot STT/TTS/RAG providers, greet the caller
      * ``media``   -- decode mu-law frame, feed to STT (and barge-in detection)
      * ``mark``    -- Twilio finished playing our audio; advance the turn state machine
      * ``stop``    -- tear down the stream
    """
    await websocket.accept()
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    session = None
    orchestrator = None
    stt_queue = None
    stt_task = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            if not raw_message:
                continue

            try:
                data = json.loads(raw_message)
            except Exception:
                logger.warning("Received invalid JSON payload from Twilio")
                continue

            event = data.get("event")

            if event == "start":
                stream_sid = data.get("streamSid", "")
                start_payload = data.get("start", {}) or {}
                call_sid = start_payload.get("callSid", "")

                session = await twilio_call_registry.register_session(call_sid, stream_sid)
                stt_provider, tts_provider, llm_provider = _build_call_providers()
                stt_queue = asyncio.Queue(maxsize=64)

                async def outbound_media(mulaw_bytes: bytes) -> None:
                    if stream_sid:
                        await send_twilio_media(websocket, stream_sid, mulaw_bytes)

                async def outbound_mark(mark_name: str) -> None:
                    if stream_sid:
                        await send_twilio_mark(websocket, stream_sid, mark_name)

                async def outbound_clear() -> None:
                    if stream_sid:
                        await send_twilio_clear(websocket, stream_sid)

                orchestrator = TwilioTurnOrchestrator(
                    session=session,
                    tts_provider=tts_provider,
                    llm_provider=llm_provider,
                    send_media_func=outbound_media,
                    send_mark_func=outbound_mark,
                    send_clear_func=outbound_clear,
                )

                async def stt_consumer():
                    try:
                        async for stt_event in stt_provider.consume_audio(stt_queue):
                            if stt_event.final:
                                await orchestrator.handle_user_transcript(stt_event.final)
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.error("STT consumer error", extra={"error": str(exc)})

                stt_task = asyncio.create_task(stt_consumer())

                asyncio.create_task(orchestrator.send_welcome_greeting())
                logger.info("Twilio media stream started", extra={"stream_sid": stream_sid, "call_sid": call_sid})

            elif event == "media":
                if orchestrator is not None and stt_queue is not None:
                    media_payload = data.get("media", {}).get("payload", "")

                    if media_payload:
                        mulaw_bytes = decode_mulaw_base64(media_payload)
                        try:
                            stt_queue.put_nowait(mulaw_bytes)
                        except asyncio.QueueFull:
                            # drop oldest frame to keep the pipeline moving
                            try:
                                stt_queue.get_nowait()
                                stt_queue.put_nowait(mulaw_bytes)
                            except Exception:
                                pass
                        await orchestrator.handle_inbound_audio(mulaw_bytes)

            elif event == "mark":
                if orchestrator is not None:
                    mark_name = data.get("mark", {}).get("name", "")

                    if mark_name:
                        await orchestrator.handle_mark_event(mark_name)

            elif event == "stop":
                logger.info("Twilio stream stop event received", extra={"stream_sid": stream_sid})
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected", extra={"stream_sid": stream_sid})
    except Exception as exc:
        logger.error("Unexpected error in Twilio media stream handler", extra={"error": str(exc), "stream_sid": stream_sid})
    finally:
        if stt_task is not None:
            stt_task.cancel()
            try:
                await stt_task
            except asyncio.CancelledError:
                pass
        if stt_queue is not None:
            try:
                stt_queue.put_nowait(None)
            except Exception:
                pass
        if orchestrator is not None:
            await orchestrator.close()
        if stream_sid:
            await twilio_call_registry.remove_session(stream_sid)
        logger.info("Cleaned up Twilio media stream resources", extra={"stream_sid": stream_sid})