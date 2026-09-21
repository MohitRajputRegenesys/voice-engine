"""Tests for the minimal ARI client (request building + event dispatch)."""
import asyncio

import pytest

from voice_engine.asterisk.ari import AriClient, AriError


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text or str(self._data)
        self.content = str(self._data).encode() if self._data else b""

    def json(self):
        return self._data


class FakeHttp:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    async def request(self, method, path, params=None, json=None):
        self.calls.append((method, path, params, json))
        return self.response


def make_client(fake=None):
    client = AriClient("http://ari:8088", "user", "pass", "app")
    client._client = fake or FakeHttp()
    return client


@pytest.mark.asyncio
async def test_create_channel_params():
    fake = FakeHttp(FakeResponse(data={"id": "CH1"}))
    client = make_client(fake)
    await client.create_channel(
        endpoint="PJSIP/3cx/9876543210",
        app_args="direction=outbound,phone=9876543210",
        channel_id="CH1",
        caller_id="AI Agent",
    )
    method, path, params, _ = fake.calls[0]
    assert (method, path) == ("POST", "/ari/channels/create")
    assert params["endpoint"] == "PJSIP/3cx/9876543210"
    assert params["app"] == "app"
    assert params["appArgs"] == "direction=outbound,phone=9876543210"
    assert params["channelId"] == "CH1"
    assert params["callerId"] == "AI Agent"


@pytest.mark.asyncio
async def test_dial_sends_timeout():
    fake = FakeHttp(FakeResponse())
    client = make_client(fake)
    await client.dial("CH-1", timeout=30)
    method, path, params, _ = fake.calls[0]
    assert path == "/ari/channels/CH-1/dial"
    assert params == {"timeout": 30}


@pytest.mark.asyncio
async def test_answer_and_hangup():
    fake = FakeHttp(FakeResponse())
    client = make_client(fake)
    await client.answer("CH1")
    await client.hangup("CH1")
    assert fake.calls[0][1] == "/ari/channels/CH1/answer"
    assert fake.calls[1][0] == "DELETE"
    assert fake.calls[1][1] == "/ari/channels/CH1"
    assert fake.calls[1][2] == {"reason": "normal"}


@pytest.mark.asyncio
async def test_create_external_media_params():
    fake = FakeHttp(FakeResponse(data={"channelid": "M1"}))
    client = make_client(fake)
    result = await client.create_external_media(
        external_host="10.0.0.5:16000", fmt="ulaw", channel_id="M1"
    )
    method, path, params, _ = fake.calls[0]
    assert path == "/ari/channels/externalMedia"
    assert params["external_host"] == "10.0.0.5:16000"
    assert params["encapsulation"] == "none"
    assert params["transport"] == "udp"
    assert params["connection_type"] == "client"
    assert params["format"] == "ulaw"
    assert params["app"] == "app"
    # Asterisk versions differ in the id field name; both must normalise
    assert result["id"] == "M1"

@pytest.mark.asyncio
async def test_bridge_operations():
    fake = FakeHttp(FakeResponse(data={"id": "BR1"}))
    client = make_client(fake)
    bridge = await client.create_bridge(name="voice-engine-x")
    assert bridge["id"] == "BR1"
    await client.add_channel_to_bridge("BR1", "CH1")
    method, path, params, _ = fake.calls[1]
    assert path == "/ari/bridges/BR1/addChannel"
    assert params["channel"] == "CH1"
    await client.remove_channel_from_bridge("BR1", "CH1")
    await client.destroy_bridge("BR1")
    assert fake.calls[3][1] == "/ari/bridges/BR1"


@pytest.mark.asyncio
async def test_http_error_raises_ari_error():
    client = make_client(FakeHttp(FakeResponse(status_code=500, text="boom")))
    with pytest.raises(AriError) as excinfo:
        await client.create_channel(endpoint="PJSIP/3cx/1")
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_event_dispatch_calls_handlers():
    client = AriClient("http://ari:8088", "u", "p", "app")
    seen = []

    async def handler(event):
        seen.append(event)

    client.on("StasisStart", handler)
    await client._dispatch({"type": "StasisStart", "channel": {"id": "1"}})
    assert seen and seen[0]["channel"]["id"] == "1"


@pytest.mark.asyncio
async def test_dispatch_isolates_handler_errors():
    client = AriClient("http://ari:8088", "u", "p", "app")
    seen = []

    async def bad_handler(event):
        raise RuntimeError("handler bug")

    async def good_handler(event):
        seen.append(event)

    client.on("StasisStart", bad_handler)
    client.on("StasisStart", good_handler)
    await client._dispatch({"type": "StasisStart", "channel": {}})
    assert seen  # the good handler still ran despite the bad one


def test_ws_url_carries_app_and_api_key():
    client = AriClient("http://ari:8088", "user", "p@ss", "voice-engine")
    url = client.ws_url()
    assert url.startswith("ws://ari:8088/ari/events")
    assert "app=voice-engine" in url
    # username:password -- special chars percent-encoded, separator colon kept
    assert "api_key=user:p%40ss" in url


@pytest.mark.asyncio
async def test_close_is_idempotent():
    client = make_client()
    await client.close()
    await client.close()
    assert client._stop is True

