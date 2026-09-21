"""FastAPI WebSocket server exposing realtime voice session endpoint."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
import base64
import logging
import os
from pathlib import Path
from .session import Session
from .state import SessionState
from .providers import build_tts_provider
from .rest import transcribe_once, synthesize_once
from .twilio.media_stream import router as twilio_media_stream_router
from .twilio.routes import router as twilio_router
from .asterisk.routes import router as asterisk_router
from .asterisk.calls import asterisk_call_manager
from .asterisk.config import get_settings as get_asterisk_settings

for env_path in (Path(__file__).resolve().parent.parent / ".env", Path(__file__).resolve().parent.parent / "env"):
    if env_path.exists():
        load_dotenv(env_path)


def _configure_logging() -> None:
    """Give the ``voice_engine`` loggers a console handler.

    Uvicorn only configures its own loggers, so without this every
    ``voice_engine.*`` INFO record (call lifecycle, media transport, STT/LLM/TTS)
    was silently dropped -- which makes a live call impossible to debug.
    Uvicorn keeps its own handlers; ``basicConfig`` only adds a root handler
    when none exists yet.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    class DiagnosticFormatter(logging.Formatter):
        def format(self, record):
            for field in ("channel_id", "turn", "state", "queue_depth"):
                if not hasattr(record, field):
                    setattr(record, field, "-")
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(
        DiagnosticFormatter(
            "%(asctime)s %(levelname)s %(name)s "
            "[channel=%(channel_id)s turn=%(turn)s state=%(state)s queue=%(queue_depth)s]: %(message)s"
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    if not root.handlers:
        root.addHandler(handler)
    else:
        for existing in root.handlers:
            existing.setFormatter(handler.formatter)
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
    )


_configure_logging()


async def _start_asterisk_integration() -> None:
    """Connect to Asterisk ARI at startup when ASTERISK_ENABLED=true.

    Never raises: a missing/unreachable Asterisk must not take the gateway
    down (the events websocket keeps retrying with backoff; REST calls fail
    per-request and surface as 502/503 from the API).
    """
    settings = get_asterisk_settings()
    if not settings.enabled:
        return
    try:
        await asterisk_call_manager.start()
    except Exception as exc:  # noqa: BLE001 - startup must never crash
        logging.getLogger("voice_engine.server").error(
            "Asterisk/3CX integration failed to start: %s", exc
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _start_asterisk_integration()
    try:
        yield
    finally:
        await asterisk_call_manager.stop()


app = FastAPI(lifespan=lifespan)

connections = {}


class TranscribeRequest(BaseModel):
    # For `encoding="text"` the payload is echoed back as the transcript (demo/mock
    # path). For `encoding="base64"` it is raw PCM16 16kHz audio for real STT.
    data: str
    encoding: str = "text"


class TTSRequest(BaseModel):
    text: str


@app.get("/")
async def index():
    return HTMLResponse("""<html><body>Voice Engine WebSocket endpoint at /ws</body></html>""")


@app.post("/stt")
async def one_shot_stt(req: TranscribeRequest):
    """One-shot speech-to-text for external AI projects.

    `POST /stt` {"data":"<base64 audio or plain text>", "encoding":"text|base64"}
    -> {"text": "recognized transcript"}
    """
    if req.encoding in ("base64", "audio"):
        try:
            raw = base64.b64decode(req.data, validate=False)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64 payload")
        text = await transcribe_once(raw)
    else:
        text = req.data  # echo path for mock/demo flows
    return {"text": text, "encoding": "text"}


@app.post("/tts")
async def one_shot_tts(req: TTSRequest):
    """One-shot text-to-speech for external AI projects.

    `POST /tts` {"text":"hello"} -> raw audio bytes. The media type reflects the
    configured provider (`audio/mpeg` for Deepgram Aura / Edge MP3 output).
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    tts_provider = build_tts_provider()
    audio = await synthesize_once(req.text, tts_provider=tts_provider)
    return Response(content=bytes(audio), media_type=getattr(tts_provider, "media_type", "application/octet-stream"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # `?auto_llm=0` → external client drives the response via speak(); default keeps
    # the configured facilitator LLM auto-answering.
    auto_llm_param = ws.query_params.get("auto_llm")
    auto_llm = True
    if auto_llm_param is not None and auto_llm_param.lower() in ("0", "false", "no", "off"):
        auto_llm = False
    session = Session(ws, auto_llm=auto_llm)
    await session.start()
    connections[session.id] = session

    # heartbeat
    heartbeat_task = asyncio.create_task(_heartbeat(ws, session))
    session.tasks.append(heartbeat_task)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                # send ping/pong handled by heartbeat
                continue
            typ = msg.get("type")
            if typ == "audio":
                # client sends base64-encoded raw PCM (real STT) or a marker;
                # for mocked/demo flows we also accept plain text frames.
                data = msg.get("data")
                if data == "<end>":
                    await session.post_audio(b"<end>")
                else:
                    try:
                        raw = base64.b64decode(data, validate=False)
                        await session.post_audio(raw)
                    except Exception:
                        await session.post_audio(str(data).encode("utf-8"))
                # if receiving audio while speaking, that's a barge-in
                if session.state.state == SessionState.SPEAKING:
                    session.metrics.interruption_count += 1
                    if session.active_response_cancel:
                        session.active_response_cancel.set()
            elif typ == "speak":
                # External AI drives the response: synthesize and speak this text.
                text = str(msg.get("data", ""))
                if text.strip():
                    await session.speak(text)
            elif typ == "close":
                break
            elif typ == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        connections.pop(session.id, None)


async def _heartbeat(ws: WebSocket, session: Session):
    try:
        while True:
            await asyncio.sleep(10)
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break
    except asyncio.CancelledError:
        return

# Include the Twilio calling feature routers (media stream WS + webhooks/outbound API).
app.include_router(twilio_media_stream_router)
app.include_router(twilio_router)

# Include the 3CX/Asterisk calling feature (ARI + ExternalMedia RTP).
# Disabled unless ASTERISK_ENABLED=true -- dev machines without Asterisk are
# completely unaffected and all Twilio endpoints keep working unchanged.
app.include_router(asterisk_router)
