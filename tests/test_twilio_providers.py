"""Tests for the Twilio provider factories, the streaming Aura TTS provider,
and the app-backend RAG brain adapter (FacilitatorLLMProvider)."""
import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest

from voice_engine import providers as prov_mod
from voice_engine.providers import (
    DeepgramSTTProvider,
    DeepgramStreamingTTSProvider,
    FacilitatorLLMProvider,
    MockSTTProvider,
    MockTTSProvider,
    build_twilio_stt_provider,
    build_twilio_tts_provider,
)


def _query_params(url: str) -> dict:
    return parse_qs(urlparse(url).query)


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)
    monkeypatch.delenv("STT_LANGUAGE", raising=False)
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    monkeypatch.delenv("TTS_MODEL", raising=False)
    monkeypatch.delenv("TWILIO_STT_MODEL", raising=False)
    monkeypatch.delenv("TWILIO_STT_LANGUAGE", raising=False)
    monkeypatch.delenv("TWILIO_SILENCE_ENDPOINT_MS", raising=False)
    monkeypatch.delenv("TWILIO_UTTERANCE_END_MS", raising=False)


# ---------------------------------------------------------------- factories

def test_twilio_stt_tuned_for_phone_calls(clean_env, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = build_twilio_stt_provider()
    assert isinstance(p, DeepgramSTTProvider)
    assert p.model == "nova-3"  # mirrors the working reference (nova-2 -> HTTP 400)
    assert p.encoding == "mulaw"  # native phone encoding -> no PCM conversion
    assert p.sample_rate == 8000
    assert p.channels == 1
    assert p.interim_results is True
    assert p.endpointing_ms == 800
    assert p.utterance_end_ms == 1500
    assert p.vad_events is True
    assert p.language is None  # phone path sends no language unless TWILIO_STT_LANGUAGE


def test_twilio_stt_request_url_matches_working_reference(clean_env, monkeypatch):
    """The Twilio STT request must match the proven-working reference URL set."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = build_twilio_stt_provider()
    q = _query_params(p.build_request_url())
    assert q["model"] == ["nova-3"]
    assert q["encoding"] == ["mulaw"]
    assert q["sample_rate"] == ["8000"]
    assert q["channels"] == ["1"]
    assert q["interim_results"] == ["true"]
    assert q["endpointing"] == ["800"]
    assert q["utterance_end_ms"] == ["1500"]
    assert q["vad_events"] == ["true"]
    assert q["smart_format"] == ["true"]
    assert "language" not in q


def test_twilio_stt_model_and_language_overrides(clean_env, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setenv("TWILIO_STT_MODEL", "nova-2-phonecall")
    monkeypatch.setenv("TWILIO_STT_LANGUAGE", "en-US")
    p = build_twilio_stt_provider()
    assert p.model == "nova-2-phonecall"
    assert p.language == "en-US"


def test_twilio_stt_mock_without_key(clean_env):
    assert isinstance(build_twilio_stt_provider(), MockSTTProvider)


def test_twilio_tts_streaming_mulaw_8k(clean_env, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = build_twilio_tts_provider()
    assert isinstance(p, DeepgramStreamingTTSProvider)
    assert p.encoding == "mulaw"
    assert p.sample_rate == 8000


def test_twilio_tts_mock_without_key(clean_env):
    assert isinstance(build_twilio_tts_provider(), MockTTSProvider)


def test_streaming_tts_requires_key(clean_env):
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        DeepgramStreamingTTSProvider()


# ---------------------------------------------------------------- streaming Aura TTS

class FakeWS:
    """Minimal stand-in for a websockets connection (sync-iterable messages)."""

    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for m in self.messages:
            # tiny delay so the background sender task can deliver its Speak/Flush
            # frames before the receiver reaches the terminal Flushed message.
            await asyncio.sleep(0.01)
            yield m


class FakeConnect:
    def __init__(self, ws):
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_streaming_tts_yields_audio_and_sends_speak_flush(clean_env, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = DeepgramStreamingTTSProvider(encoding="mulaw", sample_rate=8000)

    ws = FakeWS(messages=[b"\x00" * 160, b"\x01" * 160, json.dumps({"type": "Flushed"})])
    monkeypatch.setattr(prov_mod.websockets, "connect", lambda *a, **k: FakeConnect(ws))

    async def text_iter():
        yield "Hello there."

    chunks = [c async for c in p.synthesize(text_iter(), "resp-1")]
    assert len(chunks) == 2
    assert chunks[0].data == b"\x00" * 160
    assert [c.sequence for c in chunks] == [0, 1]

    # let the background sender task deliver its frames before asserting
    await asyncio.sleep(0.05)
    sent = "\n".join(ws.sent)
    assert '"type": "Speak"' in sent
    assert '"type": "Flush"' in sent


@pytest.mark.asyncio
async def test_streaming_tts_abort_sends_clear(clean_env, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = DeepgramStreamingTTSProvider()
    ws = FakeWS(messages=[])
    p._ws = ws
    await p.abort()
    assert p._ws is None
    assert any("Clear" in m for m in ws.sent)
# ---------------------------------------------------------------- FacilitatorLLMProvider (app backend RAG)

class FakeResp:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {
            "session_id": "sess-1",
            "answer": "We offer a Machine Learning 6 month course.",
        }
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, resp):
        self.resp = resp
        self.posts = []

    async def post(self, url, json=None):
        self.posts.append((url, json))
        return self.resp

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_facilitator_hits_app_backend_chat_api():
    provider = FacilitatorLLMProvider(
        base_url="http://localhost:8000",
        api_path="/api/v1/rag/chat",
        top_k=7,
    )
    fake = FakeClient(FakeResp())
    provider._client = fake

    deltas = [d async for d in provider.stream_response("What do you offer?", "resp-1")]
    assert "".join(d.text for d in deltas) == "We offer a Machine Learning 6 month course."

    url, payload = fake.posts[0]
    assert url == "http://localhost:8000/api/v1/rag/chat"
    assert payload == {"message": "What do you offer?", "limit": 7}
    assert provider.session_id == "sess-1"


@pytest.mark.asyncio
async def test_facilitator_sends_session_id_on_later_turns():
    provider = FacilitatorLLMProvider(api_path="/api/v1/rag/chat")
    provider.session_id = "sess-9"
    fake = FakeClient(FakeResp(data={"session_id": "sess-9", "answer": "Sure."}))
    provider._client = fake

    [d async for d in provider.stream_response("hi", "resp")]
    url, payload = fake.posts[0]
    assert payload["message"] == "hi"
    assert payload["session_id"] == "sess-9"


@pytest.mark.asyncio
async def test_facilitator_legacy_api_chat_payload():
    provider = FacilitatorLLMProvider(api_path="/api/chat", top_k=3, alpha=0.4)
    fake = FakeClient(FakeResp(data={"answer": "Legacy answer."}))
    provider._client = fake

    [d async for d in provider.stream_response("q", "resp")]
    url, payload = fake.posts[0]
    assert url == "http://localhost:8000/api/chat"
    assert payload == {
        "question": "q",
        "top_k": 3,
        "alpha": 0.4,
        "conversation_history": [],
    }


@pytest.mark.asyncio
async def test_facilitator_returns_apology_on_error():
    provider = FacilitatorLLMProvider(api_path="/api/v1/rag/chat")
    fake = FakeClient(FakeResp(status_code=500))
    provider._client = fake

    deltas = [d async for d in provider.stream_response("q", "resp")]
    assert deltas and "I'm sorry" in deltas[0].text