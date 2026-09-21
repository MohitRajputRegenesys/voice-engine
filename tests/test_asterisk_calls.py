"""Lifecycle tests for the ARI call manager with a fully faked ARI client."""
import asyncio

import pytest

from voice_engine.asterisk import calls as calls_mod
from voice_engine.asterisk.calls import AsteriskCallManager
from voice_engine.asterisk.session import CallState


class FakeAri:
    def __init__(self):
        self.calls = []
        self.handlers = {}
        self.connected = asyncio.Event()
        self.connected.set()

    def on(self, event_type, handler):
        self.handlers.setdefault(event_type, []).append(handler)

    async def create_channel(self, **kwargs):
        self.calls.append(("create_channel", kwargs))
        return {"id": kwargs.get("channel_id")}

    async def dial(self, channel_id, timeout=None):
        self.calls.append(("dial", channel_id, timeout))

    async def answer(self, channel_id):
        self.calls.append(("answer", channel_id))

    async def create_bridge(self, name="", bridge_type="mixing"):
        self.calls.append(("create_bridge", name))
        return {"id": "BR-1"}

    async def add_channel_to_bridge(self, bridge_id, channel_id, role=None):
        self.calls.append(("add_channel", bridge_id, channel_id))

    async def remove_channel_from_bridge(self, bridge_id, channel_id):
        self.calls.append(("remove_channel", bridge_id, channel_id))

    async def destroy_bridge(self, bridge_id):
        self.calls.append(("destroy_bridge", bridge_id))

    async def hangup(self, channel_id, reason="normal"):
        self.calls.append(("hangup", channel_id, reason))

    async def create_external_media(self, **kwargs):
        self.calls.append(("external_media", kwargs))
        return {"id": kwargs.get("channel_id")}


class FakeRtp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.on_frame = None
        self.closed = False
        self.bound_port = 16000
        self.local_addr = ("127.0.0.1", 16000)
        self.stats = {"rx_packets": 0, "tx_packets": 0, "dropped": 0}

    async def queue_mulaw(self, chunk):
        pass

    async def drain(self, timeout=10.0):
        pass

    def clear(self):
        pass

    async def close(self):
        self.closed = True


class FakeSTT:
    async def consume_audio(self, queue):
        while True:
            item = await queue.get()
            if item is None:
                return
            yield


class FakeTTS:
    async def synthesize(self, text_iter, response_id):
        async for _ in text_iter:
            yield


class FakeLLM:
    async def stream_response(self, prompt, response_id):
        yield


def fake_providers():
    return FakeSTT(), FakeTTS(), FakeLLM()


@pytest.fixture
def enabled_env(monkeypatch):
    monkeypatch.setenv("ASTERISK_ENABLED", "true")
    monkeypatch.setenv("ARI_BASE_URL", "http://asterisk:8088")
    monkeypatch.setenv("ARI_USERNAME", "user")
    monkeypatch.setenv("ARI_PASSWORD", "pw")
    monkeypatch.setenv("ARI_APP", "voice-engine")
    monkeypatch.setenv("THREECX_PJSIP_ENDPOINT", "3cx")
    monkeypatch.setenv("ASTERISK_MEDIA_HOST", "10.0.0.5")
    monkeypatch.setenv("ASTERISK_HOLD_MUSIC", "false")  # keep fakes simple


def fresh_manager(monkeypatch):
    manager = AsteriskCallManager()
    fake_ari = FakeAri()
    manager.ari = fake_ari

    async def fake_rtp(s):
        return FakeRtp()

    monkeypatch.setattr(manager, "_create_rtp_session", fake_rtp)
    monkeypatch.setattr(calls_mod, "build_call_providers", fake_providers)
    return manager, fake_ari

@pytest.mark.asyncio
async def test_originate_call_dials_3cx_endpoint(enabled_env, monkeypatch):
    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("+91 98765 43210")

    assert session.phone == "919876543210"  # '+'-stripped digits
    assert session.state == CallState.DIALING
    create = next(c for c in fake_ari.calls if c[0] == "create_channel")
    assert create[1]["endpoint"] == "PJSIP/919876543210@3cx"
    assert create[1]["app_args"] == "direction=outbound,phone=919876543210"
    assert create[1]["channel_id"] == session.channel_id
    dial = next(c for c in fake_ari.calls if c[0] == "dial")
    assert dial[1] == session.channel_id
    assert dial[2] == 30  # ASTERISK_CHANNEL_TIMEOUT default
    assert manager.registry.active_call_count() == 1


@pytest.mark.asyncio
async def test_originate_applies_outbound_prefix(enabled_env, monkeypatch):
    monkeypatch.setenv("ASTERISK_OUTBOUND_PREFIX", "0")
    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    assert session.phone == "09876543210"
    create = next(c for c in fake_ari.calls if c[0] == "create_channel")
    assert create[1]["endpoint"] == "PJSIP/09876543210@3cx"


@pytest.mark.asyncio
async def test_originate_requires_running_client(enabled_env):
    manager = AsteriskCallManager()
    with pytest.raises(RuntimeError):
        await manager.originate_call("123")


@pytest.mark.asyncio
async def test_originate_arms_dial_watchdog(enabled_env, monkeypatch):
    """An outbound leg always gets a watchdog so it can never sit unanswered."""
    manager, _ = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    assert session.watchdog_task is not None
    session.cancel_watchdog()


@pytest.mark.asyncio
async def test_unanswered_call_watchdog_hangs_up_and_tears_down(enabled_env, monkeypatch):
    """Regression: a call that is never answered must not stay DIALING forever.

    Observed against Asterisk 20.21.0: ARI's dial timeout does not destroy the
    channel, so without the watchdog the session leaked in DIALING indefinitely.
    """
    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    session.cancel_watchdog()  # stop the real timer; drive the watchdog directly

    await manager._dial_watchdog(session, grace=0.05)

    assert session.final_status == "NO_ANSWER"
    assert session.state == CallState.ENDED
    assert any(
        c[0] == "hangup" and c[1] == session.channel_id for c in fake_ari.calls
    )
    assert manager.registry.active_call_count() == 0


@pytest.mark.asyncio
async def test_answer_disarms_watchdog(enabled_env, monkeypatch):
    """Answering cancels the watchdog, and a late watchdog run is a no-op."""
    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    channel = {"id": session.channel_id, "name": "PJSIP/3cx-x"}

    await manager._on_channel_state_change({"channel": {**channel, "state": "Up"}})
    assert session.watchdog_task is None  # disarmed by mark_answered()

    await manager._dial_watchdog(session, grace=0.05)  # must not touch the call
    assert session.final_status is None
    # the call is alive and past the greeting (LISTENING), never NO_ANSWER
    assert session.state == CallState.LISTENING

    await manager._on_stasis_end({"channel": channel})
    assert manager.registry.active_call_count() == 0


@pytest.mark.asyncio
async def test_teardown_hangs_up_media_leg_before_bridge(enabled_env, monkeypatch):
    """Regression: the AudioSocket leg is not dropped by the call leg's hangup.

    It must be hung up explicitly, otherwise every call leaks a channel and its
    TCP peer (seen live: two stale channels left on the PBX).
    """
    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    channel = {"id": session.channel_id, "name": "PJSIP/3cx-x"}
    await manager._on_channel_state_change({"channel": {**channel, "state": "Up"}})
    media_id = session.media_channel_id

    await manager._on_stasis_end({"channel": channel})

    events = [c[0] for c in fake_ari.calls]
    assert (
        "hangup",
        media_id,
        "normal",
    ) in [tuple(c) for c in fake_ari.calls]
    # media leg is hung up before the bridge is destroyed
    assert events.index("hangup") < len(events) - 1
    assert ("destroy_bridge", "BR-1") in [tuple(c) for c in fake_ari.calls]
@pytest.mark.asyncio
async def test_to_dict_reports_media_stats_for_verification(enabled_env, monkeypatch):
    """The call report exposes transport counters so audio flow is verifiable."""
    manager, _ = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    channel = {"id": session.channel_id, "name": "PJSIP/3cx-x"}
    await manager._on_channel_state_change({"channel": {**channel, "state": "Up"}})

    session.rtp.stats["rx_packets"] = 50
    session.rtp.stats["tx_packets"] = 75
    payload = session.to_dict()
    assert payload["media"] == {"rx_packets": 50, "tx_packets": 75, "dropped": 0}
    assert payload["conversation"] == []

    await manager._on_stasis_end({"channel": channel})
    assert session.to_dict()["media"] is None  # released with the transport


@pytest.mark.asyncio
async def test_outbound_lifecycle_answer_then_teardown(enabled_env, monkeypatch):

    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    channel_id = session.channel_id
    channel = {"id": channel_id, "name": f"PJSIP/3cx-{channel_id}"}

    # Ringing -> RINGING
    await manager._on_channel_state_change({"channel": {**channel, "state": "Ring"}})
    assert session.state == CallState.RINGING

    # Answered -> bridge + external media + conversation pipeline
    await manager._on_channel_state_change({"channel": {**channel, "state": "Up"}})
    assert session.state == CallState.ANSWERED
    assert session.conversation_started
    assert session.bridge_id == "BR-1"
    assert session.media_channel_id is not None
    assert session.rtp is not None
    assert session.orchestrator is not None
    assert [c[0] for c in fake_ari.calls].count("add_channel") == 2
    external = next(c for c in fake_ari.calls if c[0] == "external_media")
    assert external[1]["external_host"] == "10.0.0.5:16000"
    assert external[1]["fmt"] == "ulaw"
    # default transport is AudioSocket over TCP (Asterisk 18/20/21 externalMedia)
    assert external[1]["encapsulation"] == "AUDIOSOCKET"
    assert external[1]["transport"] == "TCP"
    assert external[1]["data"] == session.media_channel_id

    # re-entrant events must not start a second pipeline
    await manager._on_channel_state_change({"channel": {**channel, "state": "Up"}})
    assert [c[0] for c in fake_ari.calls].count("create_bridge") == 1

    # Hangup -> full teardown (references are nulled after cleanup)
    rtp = session.rtp
    await manager._on_stasis_end({"channel": channel})
    assert manager.registry.active_call_count() == 0
    assert session.state == CallState.ENDED
    assert rtp.closed
    assert session.rtp is None
    assert session.orchestrator is None
    assert any(
        c[0] == "hangup" and c[1] == session.media_channel_id for c in fake_ari.calls
    )
    assert any(c[0] == "destroy_bridge" and c[1] == "BR-1" for c in fake_ari.calls)


@pytest.mark.asyncio
async def test_teardown_hangs_up_media_channel_before_closing_transport(
    enabled_env, monkeypatch
):
    """Regression guard for the orphaned AudioSocket channel leak.

    Asterisk destroys the external-media channel through its TCP leg, so the ARI
    hangup must happen while our transport is still open. Closing the transport
    first (the original order) made Asterisk fail the DELETE and left a channel
    behind in Stasis after every call -- verified live on Asterisk 20.21.0, where
    ``core show channels`` kept reporting the AudioSocket channel after hangup.
    """
    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    channel = {"id": session.channel_id, "name": f"PJSIP/3cx-{session.channel_id}"}
    await manager._on_channel_state_change({"channel": {**channel, "state": "Up"}})

    transport_alive_at_hangup = []
    original_hangup = fake_ari.hangup

    async def recording_hangup(channel_id, reason="normal"):
        if channel_id == session.media_channel_id:
            transport_alive_at_hangup.append(
                session.rtp is not None and not session.rtp.closed
            )
        await original_hangup(channel_id, reason)

    fake_ari.hangup = recording_hangup
    await manager._on_stasis_end({"channel": channel})

    assert transport_alive_at_hangup == [True], (
        "media channel was hung up after the media transport had been closed"
    )
    assert session.rtp is None
    assert any(
        c[0] == "hangup" and c[1] == session.media_channel_id for c in fake_ari.calls
    )



@pytest.mark.asyncio
async def test_inbound_call_answers_and_starts(enabled_env, monkeypatch):
    manager, fake_ari = fresh_manager(monkeypatch)
    event = {
        "args": ["inbound"],
        "channel": {
            "id": "IN-1",
            "name": "PJSIP/3cx-08ab",
            "caller": {"number": "9811111111", "name": "Customer"},
        },
    }
    await manager._on_stasis_start(event)

    session = await manager.registry.get_session("IN-1")
    assert session is not None
    assert session.direction == "inbound"
    assert session.phone == "9811111111"
    assert ("answer", "IN-1") in fake_ari.calls
    assert session.conversation_started

    await manager._on_stasis_end({"channel": {"id": "IN-1", "name": "PJSIP/3cx-08ab"}})
    assert manager.registry.active_call_count() == 0


@pytest.mark.asyncio
async def test_media_channel_events_are_ignored(enabled_env, monkeypatch):
    manager, fake_ari = fresh_manager(monkeypatch)
    # externalMedia channels announce themselves as UnicastRTP/...
    await manager._on_stasis_start(
        {
            "args": [],
            "channel": {"id": "M-1", "name": "UnicastRTP/10.0.0.5:16000-08ab"},
        }
    )
    assert manager.registry.active_call_count() == 0
    # ...and are also filtered by explicit id tracking
    manager._media_channel_ids.add("M-2")
    await manager._on_stasis_start({"args": [], "channel": {"id": "M-2", "name": "x"}})
    assert manager.registry.active_call_count() == 0


@pytest.mark.asyncio
async def test_unanswered_destroy_maps_to_no_answer(enabled_env, monkeypatch):
    manager, fake_ari = fresh_manager(monkeypatch)
    session = await manager.originate_call("9876543210")
    await manager._on_channel_destroyed(
        {
            "channel": {
                "id": session.channel_id,
                "name": f"PJSIP/3cx-{session.channel_id}",
                "cause": 19,
                "cause_txt": "NO_ANSWER",
            }
        }
    )
    assert manager.registry.active_call_count() == 0
    assert session.final_status == "NO_ANSWER"


@pytest.mark.asyncio
async def test_get_status_shape(enabled_env, monkeypatch):
    manager, fake_ari = fresh_manager(monkeypatch)
    status = manager.get_status()
    assert status["enabled"] is True
    assert status["configured"] is True
    assert status["connected"] is True  # FakeAri's connected event is set
    assert status["app"] == "voice-engine"
    assert status["threecx_extension"] == "900"
    assert status["media_format"] == "ulaw"
    assert status["active_call_count"] == 0

