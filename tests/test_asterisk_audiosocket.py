"""Tests for the AudioSocket external-media transport."""
import asyncio
import uuid

import pytest

from voice_engine.asterisk.audio import linear16_to_mulaw, mulaw_from_wire
from voice_engine.asterisk.audiosocket import (
    TYPE_AUDIO,
    TYPE_HANGUP,
    TYPE_UUID,
    AudioSocketSession,
    build_frame,
    parse_frame,
)


def test_frame_roundtrip():
    payload = bytes(range(160))
    frame = build_frame(TYPE_AUDIO, payload)
    assert frame[0] == TYPE_AUDIO
    assert int.from_bytes(frame[1:3], "big") == 160
    assert parse_frame(frame) == (TYPE_AUDIO, payload, 163)


def test_parse_partial_returns_none():
    assert parse_frame(b"") is None
    assert parse_frame(b"\x10\x00") is None
    assert parse_frame(build_frame(TYPE_AUDIO, b"\x00" * 160)[:-10]) is None


@pytest.mark.asyncio
async def test_audiosocket_loopback_decodes_configured_ulaw_and_sends_slin16():
    received = []
    done = asyncio.Event()

    async def on_frame(mulaw):
        received.append(mulaw)
        done.set()

    session = AudioSocketSession(
        bind_ip="127.0.0.1", fmt="ulaw", frame_ms=5, on_frame=on_frame
    )
    await session.start()
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", session.bound_port
        )
        call_uuid = uuid.uuid4()
        ulaw_frame = b"\xff" * 160
        writer.write(build_frame(TYPE_UUID, call_uuid.bytes))
        writer.write(build_frame(TYPE_AUDIO, ulaw_frame))
        await writer.drain()

        for _ in range(40):
            if session.peer_uuid:
                break
            await asyncio.sleep(0.02)
        assert session.peer_uuid == str(call_uuid)
        await asyncio.wait_for(done.wait(), 1)
        assert session.fmt == "ulaw"
        assert received == [mulaw_from_wire(ulaw_frame, "ulaw")]
        assert len(received[0]) == 160
        assert session.stats["rx_packets"] == 1

        await session.queue_mulaw(b"\x00" * 160)
        await session.drain(timeout=2)
        header = await asyncio.wait_for(reader.readexactly(3), 1)
        payload = await reader.readexactly(int.from_bytes(header[1:3], "big"))
        assert header[0] == TYPE_AUDIO
        assert len(payload) == 80
        assert linear16_to_mulaw(payload) == b"\x00" * 40
        assert session.stats["tx_packets"] == 4

        writer.write(build_frame(TYPE_HANGUP, b""))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    finally:
        await session.close()


def test_default_audio_formats_use_ulaw_inbound_and_slin16_outbound():
    session = AudioSocketSession(fmt="ulaw", frame_ms=20)
    assert session.fmt == "ulaw"
    assert session.tx_format == "slin16"
    assert session._framer.fmt == "slin16"
