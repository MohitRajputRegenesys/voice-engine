"""Minimal RTP endpoint for Asterisk ExternalMedia (encapsulation=none, transport=udp).

Asterisk sends the call audio as RTP to the address passed to
``POST /ari/channels/externalMedia`` (this socket). Audio injected back into
the call is sent as RTP to Asterisk, addressed to the *source* of the first
received packet (RTP latching -- the same trick the Asterisk external-media
samples use) or to the statically configured ``ASTERISK_MEDIA_REMOTE_ADDR`` /
``ASTERISK_MEDIA_REMOTE_PORT`` when latching is not desirable.

The sender paces frames at ``frame_ms`` (20 ms default) so the call audio stays
in real time even though TTS generates much faster than real time.
"""
import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, Tuple

from .audio import SAMPLE_RATE, WIRE_PAYLOAD_TYPES, WireFramer, mulaw_from_wire

logger = logging.getLogger("voice_engine.asterisk.rtp")


def build_rtp_packet(
    payload_type: int,
    sequence: int,
    timestamp: int,
    ssrc: int,
    marker: bool,
    payload: bytes,
) -> bytes:
    """Build a minimal RTP/AVP packet (V2, no padding, no extension, 0 CSRC)."""
    header = bytearray(12)
    header[0] = 0x80  # version 2
    header[1] = (0x80 if marker else 0x00) | (payload_type & 0x7F)
    header[2:4] = sequence.to_bytes(2, "big")
    header[4:8] = timestamp.to_bytes(4, "big")
    header[8:12] = ssrc.to_bytes(4, "big")
    return bytes(header) + payload


def parse_rtp_packet(data: bytes) -> Optional[Tuple[int, int, int, int, bool, bytes]]:
    """Parse an RTP packet.

    Returns ``(payload_type, sequence, timestamp, ssrc, marker, payload)`` or
    ``None`` for anything this minimal implementation should ignore.
    """
    if len(data) < 12:
        return None
    b0, b1 = data[0], data[1]
    version = b0 >> 6
    if version != 2:
        return None
    padding = (b0 >> 5) & 0x01
    extension = (b0 >> 4) & 0x01
    csrc_count = b0 & 0x0F
    marker = bool((b1 >> 7) & 0x01)
    payload_type = b1 & 0x7F
    sequence = int.from_bytes(data[2:4], "big")
    timestamp = int.from_bytes(data[4:8], "big")
    ssrc = int.from_bytes(data[8:12], "big")
    offset = 12 + 4 * csrc_count
    if extension:
        if len(data) < offset + 4:
            return None
        ext_len = int.from_bytes(data[offset + 2 : offset + 4], "big")
        offset += 4 + 4 * ext_len
    if offset > len(data):
        return None
    payload = data[offset:]
    if padding:
        pad_len = data[-1]
        if 0 < pad_len < len(payload):
            payload = payload[:-pad_len]
        else:
            payload = b""
    if not payload:
        return None
    return payload_type, sequence, timestamp, ssrc, marker, payload


class _RtpProtocol(asyncio.DatagramProtocol):
    def __init__(self, session: "RtpSession") -> None:
        self.session = session

    def datagram_received(self, data: bytes, addr) -> None:
        self.session._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:  # pragma: no cover
        logger.debug("RTP socket error: %s", exc)

class RtpSession:
    """One ExternalMedia RTP leg: Asterisk <-> this engine (mu-law pipeline)."""

    def __init__(
        self,
        *,
        bind_ip: str = "0.0.0.0",
        bind_port: int = 0,
        fmt: str = "ulaw",
        frame_ms: int = 20,
        remote_addr: Optional[Tuple[str, int]] = None,
        on_frame: Optional[Callable[[bytes], Awaitable[None]]] = None,
        max_queue: int = 400,
    ) -> None:
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.fmt = (fmt or "ulaw").strip().lower()
        self.frame_ms = frame_ms
        self.on_frame = on_frame  # async callback receiving mu-law frames

        self._configured_payload_type = WIRE_PAYLOAD_TYPES.get(self.fmt)
        self._mirrored_payload_type: Optional[int] = None

        self._fixed_remote = remote_addr
        self.remote_addr: Optional[Tuple[str, int]] = remote_addr

        self._framer = WireFramer(self.fmt, frame_ms)
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._rx_queue: asyncio.Queue = asyncio.Queue()
        self._transport = None
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._closed = False
        self.local_addr: Optional[Tuple[str, int]] = None

        self.sequence = random.randint(0, 0xFFFF)
        self.timestamp = random.randint(0, 0x7FFFFFFF)
        self.ssrc = random.randint(1, 0x7FFFFFFF)
        self._last_send_monotonic = 0.0

        self.stats = {"rx_packets": 0, "tx_packets": 0, "dropped": 0}

    async def start(self) -> None:
        """Bind the UDP socket and start the paced sender + receiver loops."""
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _RtpProtocol(self), local_addr=(self.bind_ip, self.bind_port)
        )
        self.local_addr = self._transport.get_extra_info("sockname")
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        self._sender_task = asyncio.create_task(self._sender_loop())
        logger.info(
            "RTP session listening on %s:%s (format=%s, frame=%sms)",
            self.local_addr[0],
            self.local_addr[1],
            self.fmt,
            self.frame_ms,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in (self._sender_task, self._receiver_task):
            if task is not None:
                task.cancel()
        for task in (self._sender_task, self._receiver_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    pass
        self._sender_task = None
        self._receiver_task = None
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:  # noqa: BLE001
                pass
            self._transport = None

    @property
    def bound_port(self) -> int:
        return self.local_addr[1] if self.local_addr else self.bind_port

    # -------------------------------------------------------------- inbound

    def _on_datagram(self, data: bytes, addr) -> None:
        """Sync callback from the event loop: parse RTP, convert, enqueue."""
        if self._closed:
            return
        parsed = parse_rtp_packet(data)
        if parsed is None:
            return
        payload_type, _seq, _ts, _ssrc, _marker, payload = parsed
        if self._configured_payload_type is None and self._mirrored_payload_type is None:
            # dynamic payload type (slin16): mirror whatever Asterisk uses
            self._mirrored_payload_type = payload_type
        try:
            mulaw = mulaw_from_wire(payload, self.fmt)
        except Exception:  # noqa: BLE001 - never let one bad packet kill the loop
            return
        if self.remote_addr is None and self._fixed_remote is None:
            self.remote_addr = addr  # latch the Asterisk RTP source
            logger.info("RTP source latched: %s:%s", addr[0], addr[1])
        self.stats["rx_packets"] += 1
        try:
            self._rx_queue.put_nowait((mulaw, addr))
        except asyncio.QueueFull:  # pragma: no cover
            self.stats["dropped"] += 1

    async def _receiver_loop(self) -> None:
        while not self._closed:
            item = await self._rx_queue.get()
            if item is None:
                return
            mulaw, _addr = item
            if self.on_frame is not None:
                try:
                    await self.on_frame(mulaw)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("RTP frame handler error: %s", exc)

    # ------------------------------------------------------------- outbound

    async def queue_mulaw(self, mulaw: bytes) -> None:
        """Buffer mu-law audio; framed, converted to the wire format, paced out."""
        if self._closed or not mulaw:
            return
        for frame in self._framer.push(mulaw):
            await self._enqueue_frame(frame)

    async def flush_mulaw(self) -> None:
        """Flush any partial buffered frame (padded with silence)."""
        if self._closed:
            return
        for frame in self._framer.flush():
            await self._enqueue_frame(frame)

    async def _enqueue_frame(self, frame: bytes) -> None:
        while True:
            try:
                self._send_queue.put_nowait(frame)
                return
            except asyncio.QueueFull:
                try:
                    self._send_queue.get_nowait()
                    self.stats["dropped"] += 1
                except asyncio.QueueEmpty:  # pragma: no cover
                    pass

    async def _sender_loop(self) -> None:
        interval = max(self.frame_ms, 1) / 1000.0
        loop = asyncio.get_running_loop()
        next_deadline = loop.time()
        while not self._closed:
            frame = await self._send_queue.get()
            self._send_frame(frame, interval)
            next_deadline += interval
            delay = next_deadline - loop.time()
            if delay < -interval:
                # fell far behind (stall/event-loop hiccup): restart the cadence
                next_deadline = loop.time()
                delay = 0
            if delay > 0:
                await asyncio.sleep(delay)

    def _send_frame(self, frame: bytes, interval: float) -> None:
        if self._closed or self._transport is None:
            return
        remote = self.remote_addr
        if remote is None:
            # Nothing received from Asterisk yet and no static target: we do not
            # know where to send. Asterisk starts streaming silence as soon as
            # the external media channel joins the bridge, so the latch happens
            # within milliseconds; the first frames are dropped until then.
            self.stats["dropped"] += 1
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        # Marker bit on the first packet of a talkspurt (after an idle gap).
        marker = now - self._last_send_monotonic > max(2 * interval, 0.05)
        self._last_send_monotonic = now
        payload_type = self._configured_payload_type
        if payload_type is None:
            payload_type = self._mirrored_payload_type or 96
        packet = build_rtp_packet(
            payload_type,
            self.sequence & 0xFFFF,
            self.timestamp & 0xFFFFFFFF,
            self.ssrc,
            marker,
            frame,
        )
        self.sequence += 1
        self.timestamp += SAMPLE_RATE * self.frame_ms // 1000  # 160 samples @ 8 kHz
        try:
            self._transport.sendto(packet, remote)
            self.stats["tx_packets"] += 1
        except Exception as exc:  # pragma: no cover
            logger.error("RTP send failed: %s", exc)

    # -------------------------------------------------------------- control

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait until every queued frame has been handed to the socket.

        Used by the turn orchestrator in place of Twilio's mark acknowledgement:
        once the send queue is empty, the caller has (within network/jitter
        delay) heard everything queued for this turn.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._closed and not self._send_queue.empty():
            if loop.time() >= deadline:
                return
            await asyncio.sleep(min(self.frame_ms, 10) / 1000.0)

    def clear(self) -> None:
        """Drop every queued (not yet sent) frame -- used on barge-in."""
        while True:
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._framer.reset()

    def pending_frames(self) -> int:
        return self._send_queue.qsize()


