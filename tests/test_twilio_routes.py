"""Tests for the Twilio webhook routes and the outbound call API."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from voice_engine.server import app
from voice_engine.twilio.twiml import outbound_response


class FakeCallsManager:
    instances = []

    def __init__(self):
        self.created = []
        FakeCallsManager.instances.append(self)

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(sid="CA_OUTBOUND_1")


class FakeTwilioClient:
    def __init__(self, *args, **kwargs):
        self.calls = FakeCallsManager()


@pytest.fixture
def client():
    return TestClient(app)


def _twilio_env(monkeypatch):
    monkeypatch.setenv("ENV", "test")  # bypass X-Twilio-Signature checks
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token-test")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15551234567")
    # bare public host -- exactly the format ALCALLINGAGENT uses (BASE_URL
    # has no protocol), so the produced URLs match the working reference setup.
    monkeypatch.setenv("TWILIO_BASE_URL", "voice.example.com")
    monkeypatch.setenv("CALLS_API_KEY", "secret-test-key")


# ---------------------------------------------------------------- webhooks

def test_voice_webhook_returns_twiml(client, monkeypatch):
    _twilio_env(monkeypatch)
    resp = client.post("/api/twilio/voice", data={"CallSid": "CA123"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    assert "wss://voice.example.com/media-stream" in resp.text
    assert "CA123" in resp.text


def test_voice_webhook_rejects_missing_signature(client, monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token-test")
    resp = client.post("/api/twilio/voice", data={"CallSid": "CA123"})
    assert resp.status_code == 403


def test_status_webhook_logs_and_returns_empty_response(client, monkeypatch):
    _twilio_env(monkeypatch)
    resp = client.post(
        "/api/twilio/status",
        data={"CallSid": "CA1", "CallStatus": "completed", "CallDuration": "42"},
    )
    assert resp.status_code == 200
    assert resp.text == "<Response></Response>"


# ---------------------------------------------------------------- outbound

def test_outbound_requires_api_key(client, monkeypatch):
    _twilio_env(monkeypatch)
    resp = client.post("/api/calls/outbound", json={"to": "+15550000000"})
    assert resp.status_code == 401


def test_outbound_requires_phone_number(client, monkeypatch):
    _twilio_env(monkeypatch)
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "")
    resp = client.post(
        "/api/calls/outbound",
        json={"to": "+15550000000"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 400


def test_outbound_call_success(client, monkeypatch):
    _twilio_env(monkeypatch)
    FakeCallsManager.instances = []
    monkeypatch.setattr("voice_engine.twilio.routes.TwilioClient", FakeTwilioClient)

    resp = client.post(
        "/api/calls/outbound",
        json={"to": "+15550000000"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "initiated"
    assert data["call_sid"] == "CA_OUTBOUND_1"
    assert data["from_number"] == "+15551234567"

    kwargs = FakeCallsManager.instances[0].created[0]
    assert kwargs["to"] == "+15550000000"
    assert kwargs["from_"] == "+15551234567"
    # the TwiML must point back at the engine's public media stream URL
    assert kwargs["twiml"] == outbound_response(stream_url="voice.example.com")
    assert "wss://voice.example.com/media-stream" in kwargs["twiml"]
    # the status callback must be a well-formed absolute https URL (no localhost,
    # no doubled scheme -- both trigger Twilio error 11100 "Invalid URL")
    assert kwargs["status_callback"] == "https://voice.example.com/api/twilio/status"
    assert "localhost" not in kwargs["status_callback"]
    assert "https://https://" not in kwargs["status_callback"]


def test_outbound_call_success_with_scheme_prefixed_host(client, monkeypatch):
    """A full https:// host in TWILIO_BASE_URL must never produce https://https://..."""
    _twilio_env(monkeypatch)
    monkeypatch.setenv("TWILIO_BASE_URL", "https://voice.example.com")
    FakeCallsManager.instances = []
    monkeypatch.setattr("voice_engine.twilio.routes.TwilioClient", FakeTwilioClient)

    resp = client.post(
        "/api/calls/outbound",
        json={"to": "+15550000000"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 200
    kwargs = FakeCallsManager.instances[0].created[0]
    assert kwargs["status_callback"] == "https://voice.example.com/api/twilio/status"
    assert "wss://voice.example.com/media-stream" in kwargs["twiml"]
    assert "https://https://" not in kwargs["status_callback"]