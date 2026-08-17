"""FastAPI WebSocket server exposing realtime voice session endpoint."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import asyncio
from pathlib import Path
from .session import Session
from .state import SessionState

for env_path in (Path(__file__).resolve().parent.parent / ".env", Path(__file__).resolve().parent.parent / "env"):
    if env_path.exists():
        load_dotenv(env_path)

app = FastAPI()

connections = {}


@app.get("/")
async def index():
    return HTMLResponse("""<html><body>Voice Engine WebSocket endpoint at /ws</body></html>""")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
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
                # client sends base64 or marker; for mocked flows accept marker <end>
                data = msg.get("data")
                if data == "<end>":
                    await session.post_audio(b"<end>")
                else:
                    await session.post_audio(data.encode("utf-8"))
                # if receiving audio while speaking, that's a barge-in
                if session.state.state == SessionState.SPEAKING:
                    session.metrics.interruption_count += 1
                    if session.active_response_cancel:
                        session.active_response_cancel.set()
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
