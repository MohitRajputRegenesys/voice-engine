"""Tests for the 3CX outbound/status API routes."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from voice_engine.asterisk import routes as routes_mod
from voice_engine.asterisk.calls import AsteriskCallManager
from voice_engine.asterisk.session import CallState
from voice_engine.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def enabled(monkeypatch):
    # NOTE: deliberately no ARI_PASSWORD so the real singleton manager stays
    # "not configured" during the app lifespan startup (no background tasks).
    monkeypatch.setenv("ASTERISK_ENABLED", "true")
    monkeypatch.setenv("CALLS_API_KEY", "secret-test-key")


@pytest.fixture
def manager(monkeypatch, enabled):
    """Replace the routes' manager with a fresh, fully controlled instance."""
    mgr = AsteriskCallManager()
    monkeypatch.setattr(routes_mod, "asterisk_call_manager", mgr)
    return mgr


def make_ready_manager(manager, monkeypatch):
    """Mark the manager configured + connected for success-path tests."""
    monkeypatch.setattr(manager, "is_configured", lambda: True)
    connected = asyncio.Event()
    connected.set()
    manager.ari = SimpleNamespace(connected=connected)
    return manager


# ------------------------------------------------------------------- auth

def test_outbound_requires_api_key(client, enabled):
    resp = client.post("/api/threecx/calls/outbound", json={"phone": "+919876543210"})
    assert resp.status_code == 401


def test_outbound_rejects_wrong_api_key(client, enabled):
    resp = client.post(
        "/api/threecx/calls/outbound",
        json={"phone": "+919876543210"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------- gating

def test_outbound_disabled_returns_503(client, monkeypatch):
    monkeypatch.delenv("ASTERISK_ENABLED", raising=False)
    monkeypatch.setenv("CALLS_API_KEY", "secret-test-key")
    resp = client.post(
        "/api/threecx/calls/outbound",
        json={"phone": "+919876543210"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"]


def test_outbound_not_configured_returns_503(client, enabled, manager, monkeypatch):
    # enabled but is_configured() False (missing ARI credentials); the real
    # .env may carry credentials, so clear them explicitly for this case
    for var in ("ARI_PASSWORD", "ARI_USERNAME", "ARI_BASE_URL", "ARI_APP"):
        monkeypatch.delenv(var, raising=False)
    resp = client.post(
        "/api/threecx/calls/outbound",
        json={"phone": "+919876543210"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_outbound_not_connected_returns_503(client, enabled, manager, monkeypatch):
    monkeypatch.setattr(manager, "is_configured", lambda: True)
    # manager.ari is None -> the ARI client never connected
    resp = client.post(
        "/api/threecx/calls/outbound",
        json={"phone": "+919876543210"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 503
    assert "not connected" in resp.json()["detail"]

# ---------------------------------------------------------------- success

def test_outbound_success(client, enabled, manager, monkeypatch):
    make_ready_manager(manager, monkeypatch)

    async def fake_originate(phone, caller_id=""):
        return SimpleNamespace(
            channel_id="CH-9", phone="919876543210", state=CallState.DIALING
        )

    monkeypatch.setattr(manager, "originate_call", fake_originate)

    resp = client.post(
        "/api/threecx/calls/outbound",
        json={"phone": "+91 98765 43210", "customer_id": "cust_123"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "status": "dialing",
        "channel_id": "CH-9",
        "to": "919876543210",
        "from_extension": "900",
    }


def test_outbound_alias_path(client, enabled, manager, monkeypatch):
    make_ready_manager(manager, monkeypatch)

    async def fake_originate(phone, caller_id=""):
        return SimpleNamespace(
            channel_id="CH-A", phone="15550000000", state=CallState.DIALING
        )

    monkeypatch.setattr(manager, "originate_call", fake_originate)
    resp = client.post(
        "/api/asterisk/calls/outbound",
        json={"phone": "15550000000"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["channel_id"] == "CH-A"


def test_outbound_invalid_phone_returns_400(client, enabled, manager, monkeypatch):
    make_ready_manager(manager, monkeypatch)

    async def fake_originate(phone, caller_id=""):
        raise ValueError("phone number contains no digits")

    monkeypatch.setattr(manager, "originate_call", fake_originate)
    resp = client.post(
        "/api/threecx/calls/outbound",
        json={"phone": "  "},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 400


def test_outbound_ari_failure_returns_502(client, enabled, manager, monkeypatch):
    make_ready_manager(manager, monkeypatch)

    async def fake_originate(phone, caller_id=""):
        raise RuntimeError("ARI request failed")

    monkeypatch.setattr(manager, "originate_call", fake_originate)
    resp = client.post(
        "/api/threecx/calls/outbound",
        json={"phone": "+919876543210"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert resp.status_code == 502


# ----------------------------------------------------------------- status

def test_status_endpoint(client, enabled, manager, monkeypatch):
    make_ready_manager(manager, monkeypatch)
    resp = client.get("/api/threecx/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["configured"] is True
    assert data["connected"] is True
    assert data["threecx_extension"] == "900"
    assert data["active_call_count"] == 0


def test_status_alias_endpoint(client, enabled):
    resp = client.get("/api/asterisk/status")
    assert resp.status_code == 200
    assert "enabled" in resp.json()


# ----------------------------------------------------------------- hangup

def test_hangup_success_and_unknown_channel(client, enabled, manager, monkeypatch):
    make_ready_manager(manager, monkeypatch)

    async def fake_hangup(channel_id):
        return channel_id == "CH-OK"

    monkeypatch.setattr(manager, "hangup", fake_hangup)
    headers = {"X-API-Key": "secret-test-key"}

    resp = client.post("/api/threecx/calls/CH-OK/hangup", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"status": "hangup_requested", "channel_id": "CH-OK"}

    resp = client.post("/api/threecx/calls/CH-GONE/hangup", headers=headers)
    assert resp.status_code == 404


def test_hangup_requires_api_key(client, enabled):
    resp = client.post("/api/threecx/calls/CH-1/hangup")
    assert resp.status_code == 401

