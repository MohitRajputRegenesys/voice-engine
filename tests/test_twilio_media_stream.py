"""End-to-end test of the /media-stream WebSocket with fake providers.

Simulates a Twilio call: start -> welcome greeting (media + mark) -> caller
speaks (STT final) -> app-backend RAG answer -> TTS media -> mark.
"""
import base64

import pytest
from fastapi.testclient import TestClient

from voice_engine.providers import AudioChunk, LLMDelta, STTEvent
from voice_engine.server import app
from voice_engine.twilio.session import twilio_call_registry


class FakeTwilioSTT:
    def __init__(self, final="What courses do you offer?"):
        self.final = final
        self.audio_frames = 0

    async def consume_audio(self, queue):
        while True:
            data = await queue.get()
            if data is None:
                return
            self.audio_frames += 1
            yield STTEvent(response_id="stt-1", sequence=0, final=self.final)


class FakeTwilioTTS:
    def __init__(self, audio=b"\x00" * 200):
        self.audio = audio

    async def synthesize(self, text_iter, response_id):
        async for _ in text_iter:
            yield AudioChunk(response_id=response_id, sequence=0, data=self.audio)


class FakeRAG:
    def __init__(self, answer="We offer a Machine Learning 6 month course."):
        self.answer = answer
        self.prompts = []

    async def stream_response(self, prompt, response_id):
        self.prompts.append(prompt)
        yield LLMDelta(response_id=response_id, sequence=0, text=self.answer)


def _fake_providers():
    """Injected STT/TTS/RAG; used by the endpoint via monkeypatch."""
    return FakeTwilioSTT(), FakeTwilioTTS(), FakeRAG()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("TWILIO_PREBUFFER_MS", "10")  # tiny so audio flows at once
    monkeypatch.setenv("TWILIO_HOLD_MUSIC", "false")  # keep the WS flow assertions audio-only
    monkeypatch.setattr(
        "voice_engine.twilio.media_stream._build_call_providers", _fake_providers
    )
    # fresh registry so tests don't leak sessions
    twilio_call_registry._sessions.clear()
    return TestClient(app)


def _read_until_mark(ws, max_messages=20):
    """Consume messages until a mark event; return (saw_media, mark_name, messages)."""
    saw_media = False
    messages = []
    for _ in range(max_messages):
        msg = ws.receive_json()
        messages.append(msg)
        if msg.get("event") == "media":
            saw_media = True
        elif msg.get("event") == "mark":
            return saw_media, msg["mark"]["name"], messages
    raise AssertionError("no mark event received")


def test_media_stream_welcome_then_rag_turn(client):
    with client.websocket_connect("/media-stream") as ws:
        # ---- start the call ----
        ws.send_json(
            {"event": "start", "streamSid": "MOCK_STREAM", "start": {"callSid": "CA1"}}
        )

        # ---- welcome greeting: audio media + mark ----
        saw_welcome_media, welcome_mark, _ = _read_until_mark(ws)
        assert saw_welcome_media
        assert welcome_mark

        # Twilio acknowledges playback -> turn 1 completes, LISTENING
        ws.send_json(
            {"event": "mark", "streamSid": "MOCK_STREAM", "mark": {"name": welcome_mark}}
        )

        # ---- caller speaks (8 kHz mu-law frame, base64) ----
        ws.send_json(
            {
                "event": "media",
                "streamSid": "MOCK_STREAM",
                "media": {"payload": base64.b64encode(b"\x00" * 160).decode()},
            }
        )

        # ---- RAG answer is spoken: audio media + mark ----
        saw_turn_media, turn_mark, _ = _read_until_mark(ws)
        assert saw_turn_media
        assert turn_mark

        # ---- teardown ----
        ws.send_json({"event": "stop", "streamSid": "MOCK_STREAM"})

    assert twilio_call_registry.active_call_count() == 0


def test_media_stream_handles_stop_when_idle(client):
    """A stream that connects and immediately stops must clean up cleanly."""
    with client.websocket_connect("/media-stream") as ws:
        ws.send_json(
            {"event": "start", "streamSid": "MOCK_STREAM_2", "start": {"callSid": "CA2"}}
        )
        # don't wait for the welcome; just hang up
        ws.send_json({"event": "stop", "streamSid": "MOCK_STREAM_2"})

    assert twilio_call_registry.active_call_count() == 0