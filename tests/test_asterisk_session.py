"""Tests for the Asterisk call session state machine and registry."""
import pytest

from voice_engine.asterisk.session import (
    AsteriskCallRegistry,
    AsteriskCallSession,
    CallState,
)


@pytest.mark.asyncio
async def test_register_get_remove_outbound():
    registry = AsteriskCallRegistry()
    session = await registry.register_outbound("CH1", "919876543210", caller_id="AI")
    assert session.direction == "outbound"
    assert session.state == CallState.NEW
    assert await registry.get_session("CH1") is session
    assert await registry.get_session("NOPE") is None
    assert registry.active_call_count() == 1
    removed = await registry.remove_session("CH1")
    assert removed is session
    assert session.state == CallState.ENDED
    assert registry.active_call_count() == 0
    assert await registry.remove_session("CH1") is None  # idempotent


@pytest.mark.asyncio
async def test_media_channel_index_follows_call():
    registry = AsteriskCallRegistry()
    await registry.register_inbound("CH2", "919870000000")
    await registry.link_media_channel("CH2", "MEDIA-1")
    session = await registry.get_by_media_channel("MEDIA-1")
    assert session is not None and session.channel_id == "CH2"
    # unknown media channels resolve to nothing
    assert await registry.get_by_media_channel("MEDIA-X") is None
    # removing the call clears the media index too
    await registry.remove_session("CH2")
    assert await registry.get_by_media_channel("MEDIA-1") is None


def test_mark_answered_only_moves_telephony_states():
    session = AsteriskCallSession("CH3", "outbound", "919876543210")
    assert session.answered_at is None
    session.transition_to(CallState.DIALING)
    session.transition_to(CallState.RINGING)
    session.mark_answered()
    assert session.state == CallState.ANSWERED
    assert session.answered_at is not None
    # once the conversation starts, mark_answered must not stomp the state
    session.transition_to(CallState.LISTENING)
    session.mark_answered()
    assert session.state == CallState.LISTENING


def test_to_dict_shape():
    session = AsteriskCallSession("CH4", "inbound", "100")
    data = session.to_dict()
    assert data["channel_id"] == "CH4"
    assert data["direction"] == "inbound"
    assert data["phone"] == "100"
    assert data["state"] == "NEW"
    assert data["final_status"] is None
    assert data["turn_number"] == 0
    assert "duration_sec" in data


def test_conversation_history():
    session = AsteriskCallSession("CH5", "outbound")
    session.add_user_message("hi")
    session.add_assistant_message("hello")
    assert session.conversation == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
