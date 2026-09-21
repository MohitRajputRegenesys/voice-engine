"""Tests for the minimal RTP implementation (build/parse/latch/paced sending)."""
import asyncio

import pytest

from voice_engine.asterisk.rtp import RtpSession, build_rtp_packet, parse_rtp_packet


def _make_packet(cc=0, ext_words=0, payload=b"\x01" * 8, pt=0, marker=False):
    header = bytearray(12)
    header[0] = 0x80 | (0x10 if ext_words else 0) | cc
    header[1] = (0x80 if marker else 0) | pt
    header[2:4] = (1).to_bytes(2, "big")    # sequence
    header[4:8] = (5).to_bytes(4, "big")    # timestamp
    header[8:12] = b"\xde\xad\xbe\xef"      # ssrc
    packet = bytes(header)
    packet += b"\xaa\xbb\xcc\xdd" * cc      # CSRC entries
    if ext_words:
        packet += b"\xbe\xef" + ext_words.to_bytes(2, "big")
        packet += b"\x00" * (4 * ext_words)
    return packet + payload


def test_build_parse_roundtrip():
    payload = bytes(range(160))
    packet = build_rtp_packet(0, 7, 1000, 42, True, payload)
    parsed = parse_rtp_packet(packet)
    assert parsed is not None
    payload_type, sequence, timestamp, ssrc, marker, parsed_payload = parsed
    assert (payload_type, sequence, timestamp, ssrc, marker) == (0, 7, 1000, 42, True)
    assert parsed_payload == payload


def test_parse_rejects_garbage():
    assert parse_rtp_packet(b"") is None
    assert parse_rtp_packet(b"\x00" * 12) is None  # wrong version
    assert parse_rtp_packet(b"\x80\x00\x00") is None  # too short


def test_parse_skips_csrc_and_extension():
    parsed = parse_rtp_packet(_make_packet(cc=2, ext_words=1, payload=b"\x07" * 160))
    assert parsed is not None
    assert parsed[0] == 0
    assert parsed[5] == b"\x07" * 160


def test_parse_strips_padding():
    payload = b"\x03" * 8
    packet = bytearray(build_rtp_packet(0, 1, 1, 1, False, payload + b"\x00\x00\x00\x04"))
    packet[0] |= 0x20  # padding bit; last byte says 4 padding bytes
    parsed = parse_rtp_packet(bytes(packet))
    assert parsed is not None
    assert parsed[5] == payload


class Capture(asyncio.DatagramProtocol):
    """UDP capture endpoint; the protocol instance is stored by the factory."""

    def __init__(self):
        self.queue = asyncio.Queue()
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.queue.put_nowait((data, addr))


async def make_capture():
    loop = asyncio.get_running_loop()
    holder = {}

    def factory():
        protocol = Capture()
        holder["protocol"] = protocol
        return protocol

    transport, _ = await loop.create_datagram_endpoint(
        factory, local_addr=("127.0.0.1", 0)
    )
    addr = transport.get_extra_info("sockname")
    return transport, holder["protocol"], addr

@pytest.mark.asyncio
async def test_datagram_latches_remote_and_converts():
    received = []
    done = asyncio.Event()

    async def on_frame(mulaw):
        received.append(mulaw)
        done.set()

    session = RtpSession(fmt="ulaw", frame_ms=20, on_frame=on_frame)
    await session.start()
    try:
        mulaw = b"\x55" * 160
        session._on_datagram(
            build_rtp_packet(0, 1, 160, 9, False, mulaw), ("10.1.2.3", 5010)
        )
        await asyncio.wait_for(done.wait(), 1)
        assert received[0] == mulaw
        # first packet's source becomes the outbound target (RTP latching)
        assert session.remote_addr == ("10.1.2.3", 5010)
        assert session.stats["rx_packets"] == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_outbound_frames_are_paced_rtp():
    capture_transport, capture, capture_addr = await make_capture()
    # bind to loopback so the observed source address matches local_addr
    session = RtpSession(
        bind_ip="127.0.0.1", fmt="ulaw", frame_ms=5, remote_addr=capture_addr
    )
    await session.start()
    try:
        await session.queue_mulaw(b"\x00" * 320)  # 5 ms frames -> 8 packets
        await session.drain(timeout=2)
        await asyncio.sleep(0.05)  # let the sender flush the last packet
        packets = []
        while not capture.queue.empty():
            packets.append(capture.queue.get_nowait())
        assert len(packets) == 8
        data, addr = packets[0]
        # the source address is the RTP session's own socket (we send TO the capture)
        assert addr == session.local_addr
        parsed = parse_rtp_packet(data)
        assert parsed is not None
        assert parsed[0] == 0  # PCMU static payload type
        assert len(parsed[5]) == 40
        assert session.stats["tx_packets"] == 8
    finally:
        await session.close()
        capture_transport.close()


@pytest.mark.asyncio
async def test_clear_drops_queued_frames():
    session = RtpSession(fmt="ulaw", frame_ms=20)
    session.remote_addr = ("10.0.0.1", 5004)  # no socket needed for queue ops
    await session.queue_mulaw(b"\x00" * 480)  # 3 frames
    assert session.pending_frames() == 3
    session.clear()
    assert session.pending_frames() == 0


@pytest.mark.asyncio
async def test_slin16_mirrors_dynamic_payload_type():
    capture_transport, capture, capture_addr = await make_capture()
    received = []
    done = asyncio.Event()

    async def on_frame(mulaw):
        received.append(mulaw)
        done.set()

    session = RtpSession(
        fmt="slin16", frame_ms=5, remote_addr=capture_addr, on_frame=on_frame
    )
    await session.start()
    try:
        # 1. Asterisk -> engine: L16 payload arriving with dynamic PT 96
        session._on_datagram(
            build_rtp_packet(96, 1, 160, 5, False, b"\x10\x00" * 40),
            ("127.0.0.1", capture_addr[1]),
        )
        await asyncio.wait_for(done.wait(), 1)
        assert len(received[0]) == 40  # converted to mu-law, same sample count
        # 2. engine -> Asterisk: the dynamic PT is mirrored back
        await session.queue_mulaw(b"\x00" * 40)
        await session.drain(timeout=2)
        await asyncio.sleep(0.05)
        data, _ = capture.queue.get_nowait()
        parsed = parse_rtp_packet(data)
        assert parsed[0] == 96
        assert len(parsed[5]) == 80  # 40 samples * 2 bytes
    finally:
        await session.close()
        capture_transport.close()

