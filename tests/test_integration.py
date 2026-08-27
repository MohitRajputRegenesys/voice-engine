import asyncio
import base64

import pytest
from fastapi.testclient import TestClient

from voice_engine.server import app
from voice_engine.providers import MockSTTProvider, MockLLMProvider, MockTTSProvider
from voice_engine.session import Session
from voice_engine.state import SessionState


class DummyWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    # Force mock providers regardless of any real .env keys (no network in tests).
    monkeypatch.setenv("DEEPGRAM_API_KEY", "")
    monkeypatch.setenv("FACILITATOR_API_URL", "")
    monkeypatch.setenv("TTS_PROVIDER", "")


def test_stt_text_echo(client):
    resp = client.post("/stt", json={"data": "hello world", "encoding": "text"})
    assert resp.status_code == 200
    assert resp.json()["text"] == "hello world"


def test_tts_returns_audio_bytes(client):
    resp = client.post("/tts", json={"text": "hello"})
    assert resp.status_code == 200
    assert resp.content  # non-empty; mock provider returns b"AUDIO(hello)"


def test_tts_empty_text_rejected(client):
    resp = client.post("/tts", json={"text": "   "})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_session_auto_llm_off_does_not_auto_speak():
    """In external-AI mode a final transcript must NOT auto-trigger the LLM."""
    ws = DummyWebSocket()
    session = Session(
        ws,
        stt_provider=MockSTTProvider(),
        llm_provider=MockLLMProvider(),  # present, but must not be used
        tts_provider=MockTTSProvider(),
        auto_llm=False,
    )
    await session.start()
    await session.post_audio(b"hello")
    await session.post_audio(b"<end>")
    await asyncio.sleep(0.3)

    types = [m["type"] for m in ws.sent]
    assert "transcript.final" in types
    # No auto-generated response / no auto TTS when auto_llm=False
    assert "llm.delta" not in types
    assert "audio.chunk" not in types

    await session.close()


@pytest.mark.asyncio
async def test_session_speak_streams_audio():
    """External speak(text) must stream audio.chunk then emit speak.done."""
    ws = DummyWebSocket()
    session = Session(
        ws,
        stt_provider=MockSTTProvider(),
        llm_provider=MockLLMProvider(),
        tts_provider=MockTTSProvider(),
        auto_llm=False,
    )
    await session.start()
    await session.speak("external answer")
    await asyncio.sleep(0.5)

    types = [m["type"] for m in ws.sent]
    assert "audio.chunk" in types
    assert "speak.done" in types
    assert session.state.state == SessionState.LISTENING

    await session.close()


def test_websocket_external_ai_full_flow(client):
    """End-to-end over the real WS endpoint: listen -> external speak -> audio out."""
    with client.websocket_connect("/ws?auto_llm=0") as ws:
        # -- 1. send a frame then finalize the utterance (mock STT) --
        ws.send_json({"type": "audio", "data": base64.b64encode(b"\x00" * 320).decode()})
        ws.send_json({"type": "audio", "data": "<end>"})

        # read until transcript.final
        saw_final = False
        while not saw_final:
            msg = ws.receive_json()
            if msg["type"] == "transcript.final":
                assert msg["text"]  # non-empty mock final
                saw_final = True
        # auto_llm=0 -> no auto llm.delta / audio.chunk before we speak
        assert not any(m["type"] in ("llm.delta",) for m in [])

        # -- 2. external AI asks the engine to speak its answer --
        ws.send_json({"type": "speak", "data": "external answer"})

        saw_audio = False
        saw_done = False
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "audio.chunk":
                saw_audio = True
            elif msg["type"] == "speak.done":
                saw_done = True
                break
        assert saw_audio
        assert saw_done

