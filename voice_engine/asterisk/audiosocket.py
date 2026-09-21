"""AudioSocket transport for Asterisk ExternalMedia (encapsulation=AUDIOSOCKET, transport=TCP).

Asterisk 20's ``POST /ari/channels/externalMedia`` only implements the TCP
transports (raw RTP over TCP is rejected with 501), so the media leg on this
deployment uses the AudioSocket protocol: a plain TCP stream where every frame
is ``1 byte type + 2 bytes big-endian length + payload``.

Types: 0x01 UUID identification, 0x10 audio, 0x00 hangup, others ignored.
 AudioSocket inbound audio is signed linear 16-bit PCM at 8 kHz for this
 Asterisk deployment. Outbound audio sent by this service uses the same wire
 format. The conversation pipeline remains mu-law, so conversion happens at
 this boundary.

AudioSocket protocol quirk (verified against the Asterisk 20.21.0 deployment):
the generic 0x10 AUDIO kind arriving from this external-media channel is
signed-linear PCM16 even though the ARI request uses ``format=ulaw``. Outbound
audio is also emitted as slin16; see ``tx_format``.

With ``connection_type=client`` (our default) Asterisk connects OUT to this
server, so the peer address is known on accept -- no RTP latching needed and
audio can flow immediately.

Direction of the UUID frame (verified against Asterisk 20.21.0): Asterisk
sends the call's UUID frame to us right after connecting, and
``res_audiosocket`` only accepts AUDIO (0x10) / HANGUP (0x00) frames *from*
this side. Replying with our own UUID frame makes it fail the channel with
"Received AudioSocket message other than hangup or audio" and the call drops,
so ``announce_uuid`` defaults to False; the flag only exists for non-Asterisk
peers that want the identity frame.

Exposes the same interface as :class:`RtpSession` so the call manager can use
either transport (``ASTERISK_MEDIA_TRANSPORT=audiosocket|rtp``).
"""
import asyncio
import logging
import time
import uuid
from typing import Awaitable, Callable, Optional

from .audio import WireFramer, mulaw_from_wire
from ..twilio.audio import MULAW_TO_PCM16, rms

logger = logging.getLogger("voice_engine.asterisk.audiosocket")

TYPE_UUID = 0x01
TYPE_AUDIO = 0x10
TYPE_HANGUP = 0x00
TYPE_ERROR = 0xFF


def build_frame(frame_type: int, payload: bytes) -> bytes:
    """Encode one AudioSocket frame (1-byte type, 2-byte BE length, payload)."""
    return bytes([frame_type & 0xFF]) + len(payload).to_bytes(2, "big") + payload


def parse_frame(data: bytes):
    """Split a buffer at the first complete frame; returns (type, payload, consumed) or None."""
    if len(data) < 3:
        return None
    frame_type = data[0]
    length = int.from_bytes(data[1:3], "big")
    if len(data) < 3 + length:
        return None
    return frame_type, data[3 : 3 + length], 3 + length


class AudioSocketSession:
    """One ExternalMedia AudioSocket leg: Asterisk <-> this engine (mu-law pipeline)."""

    def __init__(
        self,
        *,
        bind_ip: str = "0.0.0.0",
        bind_port: int = 0,
        fmt: str = "ulaw",
        frame_ms: int = 20,
        # AudioSocket kind 0x10 (AUDIO) is ALWAYS read back by Asterisk as
        # ast_format_slin (16-bit signed linear, 8 kHz) -- see the module
        # docstring quirk note. That is a fixed property of the wire protocol,
        # so the OUTBOUND (tx) wire format is slin16 regardless of ``fmt``.
        # ``fmt`` only governs the INBOUND direction (the codec Asterisk streams
        # *to* us), which stays ulaw by default to match the mulaw STT feed.
        tx_format: str = "slin16",
        on_frame: Optional[Callable[[bytes], Awaitable[None]]] = None,
        max_queue: int = 400,
        announce_uuid: bool = False,
    ) -> None:
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        # The deployed external-media channel sends 160-byte 8 kHz mu-law
        # frames. Keep the pipeline format aligned with the observed wire data.
        self.fmt = (fmt or "ulaw").strip().lower()
        self.frame_ms = frame_ms
        self.tx_format = (tx_format or "slin16").strip().lower()
        self.on_frame = on_frame
        self.uuid = str(uuid.uuid4())
        self.announce_uuid = announce_uuid
        self.peer_uuid: Optional[str] = None

        self._framer = WireFramer(self.tx_format, frame_ms)
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._rx_queue: asyncio.Queue = asyncio.Queue()
        self._server = None
        self._reader = None
        self._writer = None
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._rx_task: Optional[asyncio.Task] = None
        self._closed = False
        self.local_addr: Optional[tuple] = None
        self.peer_addr: Optional[tuple] = None
        self.stats = {
            "rx_packets": 0,
            "rx_bytes": 0,
            "tx_packets": 0,
            "tx_bytes": 0,
            "dropped": 0,
            "last_rx_at": None,
            "last_tx_at": None,
        }

    async def start(self) -> None:
        """Bind the TCP listener; Asterisk connects in (connection_type=client)."""
        self._server = await asyncio.start_server(
            self._handle_client, self.bind_ip, self.bind_port
        )
        sock = self._server.sockets[0]
        self.local_addr = sock.getsockname()[:2]
        logger.info(
            "AudioSocket server listening on %s:%s (rx=%s tx=%s, uuid=%s)",
            self.local_addr[0], self.local_addr[1], self.fmt, self.tx_format, self.uuid,
        )

    async def close(self, skip_task: Optional[asyncio.Task] = None) -> None:
        if self._closed:
            return
        self._closed = True
        current_task = asyncio.current_task()
        skip_tasks = {task for task in (current_task, skip_task) if task is not None}
        for task in (self._sender_task, self._receiver_task, self._rx_task):
            if task is not None and task not in skip_tasks:
                task.cancel()
        for task in (self._sender_task, self._receiver_task, self._rx_task):
            if task is not None and task not in skip_tasks:
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._sender_task = None
        self._receiver_task = None
        self._rx_task = None
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
            self._writer = None

    @property
    def bound_port(self) -> int:
        return self.local_addr[1] if self.local_addr else self.bind_port

    # -------------------------------------------------------------- inbound

    async def _handle_client(self, reader, writer) -> None:
        """Accept the single Asterisk connection for this call."""
        if self._reader is not None or self._closed:
            writer.close()
            return
        self._reader = reader
        self._writer = writer
        self.peer_addr = writer.get_extra_info("peername")
        logger.info(
            "AudioSocket peer connected: %s (uuid=%s, rx_format=%s, tx_format=%s)",
            self.peer_addr,
            self.uuid,
            self.fmt,
            self.tx_format,
        )
        # Asterisk already sent us the call's UUID frame and rejects any
        # non-audio/hangup frame from this side, so announcing ours is opt-in.
        if self.announce_uuid:
            try:
                raw_uuid = uuid.UUID(self.uuid).bytes
                self._writer.write(build_frame(TYPE_UUID, raw_uuid))
                await self._writer.drain()
            except Exception:  # noqa: BLE001
                pass
        self._receiver_task = asyncio.create_task(self._reader_loop())
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._rx_task = asyncio.create_task(self._rx_pump())

    async def _reader_loop(self) -> None:
        buffer = b""
        try:
            while not self._closed:
                chunk = await self._reader.read(4096)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    parsed = parse_frame(buffer)
                    if parsed is None:
                        break
                    frame_type, payload, consumed = parsed
                    buffer = buffer[consumed:]
                    if frame_type == TYPE_UUID:
                        # Asterisk identifies the call this way; read-only.
                        if len(payload) == 16:
                            self.peer_uuid = str(uuid.UUID(bytes=payload))
                            logger.info("AudioSocket call UUID %s", self.peer_uuid)
                        continue
                    if frame_type == TYPE_AUDIO and payload:
                        try:
                            mulaw = mulaw_from_wire(payload, self.fmt)
                        except Exception as exc:  # noqa: BLE001
                            logger.error(
                                "AudioSocket audio decode failed: wire_bytes=%s format=%s error=%s",
                                len(payload),
                                self.fmt,
                                exc,
                            )
                            continue
                        self.stats["rx_packets"] += 1
                        self.stats["rx_bytes"] += len(payload)
                        self.stats["last_rx_at"] = time.time()
                        if self.stats["rx_packets"] == 1 or self.stats["rx_packets"] % 100 == 0:
                            wire_samples = [
                                int.from_bytes(payload[index : index + 2], "little", signed=True)
                                for index in range(0, len(payload) - 1, 2)
                            ]
                            wire_rms = (
                                sum(sample * sample for sample in wire_samples) / len(wire_samples)
                            ) ** 0.5 if wire_samples else 0.0
                            wire_peak = max((abs(sample) for sample in wire_samples), default=0)
                            mulaw_rms = rms(mulaw)
                            mulaw_peak = max(
                                (abs(MULAW_TO_PCM16[value]) for value in mulaw),
                                default=0,
                            )
                            logger.info(
                                "AudioSocket inbound audio flowing: packets=%s bytes=%s payload_bytes=%s wire_rms=%.1f wire_peak=%s mulaw_rms=%.1f mulaw_peak=%s",
                                self.stats["rx_packets"],
                                self.stats["rx_bytes"],
                                len(payload),
                                wire_rms,
                                wire_peak,
                                mulaw_rms,
                                mulaw_peak,
                            )
                        try:
                            self._rx_queue.put_nowait(mulaw)
                        except asyncio.QueueFull:  # pragma: no cover
                            self.stats["dropped"] += 1
                    elif frame_type in (TYPE_HANGUP, TYPE_ERROR):
                        logger.info(
                            "AudioSocket peer signalled end (type=%s, rx_packets=%s, tx_packets=%s)",
                            frame_type,
                            self.stats["rx_packets"],
                            self.stats["tx_packets"],
                        )
                        return
                    # UUID/keepalive/unknown frames are ignored
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except asyncio.CancelledError:
            raise
        finally:
            logger.info(
                "AudioSocket reader stopped (rx_packets=%s rx_bytes=%s tx_packets=%s tx_bytes=%s)",
                self.stats["rx_packets"],
                self.stats["rx_bytes"],
                self.stats["tx_packets"],
                self.stats["tx_bytes"],
            )
            if not self._closed:
                self._closed = True
                for task in (self._sender_task, self._rx_task):
                    if task is not None:
                        task.cancel()
                if self._server is not None:
                    self._server.close()
                if self._writer is not None:
                    self._writer.close()

    async def _rx_pump(self) -> None:
        while not self._closed:
            mulaw = await self._rx_queue.get()
            if mulaw is None:
                return
            if self.on_frame is not None:
                try:
                    await self.on_frame(mulaw)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("AudioSocket frame handler error: %s", exc)
    # ------------------------------------------------------------- outbound

    async def queue_mulaw(self, mulaw: bytes) -> None:
        """Buffer mu-law audio; framed and paced out over TCP."""
        if self._closed or not mulaw:
            return
        for frame in self._framer.push(mulaw):
            await self._enqueue_frame(frame)

    async def flush_mulaw(self) -> None:
        if self._closed:
            return
        for frame in self._framer.flush():
            await self._enqueue_frame(frame)

    async def _enqueue_frame(self, frame: bytes) -> None:
        if self._closed:
            return
        # Preserve speech frames. Dropping the oldest frame when a long TTS
        # response fills the queue makes words disappear and sounds rushed.
        await self._send_queue.put(frame)

    async def _sender_loop(self) -> None:
        interval = max(self.frame_ms, 1) / 1000.0
        loop = asyncio.get_running_loop()
        next_deadline = loop.time()
        try:
            while not self._closed:
                frame = await self._send_queue.get()
                # wait for the Asterisk connection before writing
                while self._writer is None and not self._closed:
                    await asyncio.sleep(0.02)
                if self._closed or self._writer is None:
                    return
                self._writer.write(build_frame(TYPE_AUDIO, frame))
                await self._writer.drain()
                self.stats["tx_packets"] += 1
                self.stats["tx_bytes"] += len(frame)
                self.stats["last_tx_at"] = time.time()
                if self.stats["tx_packets"] == 1 or self.stats["tx_packets"] % 100 == 0:
                    logger.info(
                        "AudioSocket outbound audio flowing: packets=%s bytes=%s payload_bytes=%s pending=%s",
                        self.stats["tx_packets"],
                        self.stats["tx_bytes"],
                        len(frame),
                        self._send_queue.qsize(),
                    )
                next_deadline += interval
                delay = next_deadline - loop.time()
                if delay < -interval:
                    next_deadline = loop.time()
                    delay = 0
                if delay > 0:
                    await asyncio.sleep(delay)
        except (ConnectionError, asyncio.IncompleteReadError):
            logger.warning("AudioSocket sender connection closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("AudioSocket sender stopped unexpectedly: %s", exc)

    # -------------------------------------------------------------- control

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait until every queued frame has been written to Asterisk."""
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

    async def wait_for_queue_room(self, max_pending: int = 10) -> None:
        """Keep outbound lookahead short so speech is not delayed by old audio."""
        while not self._closed and self._send_queue.qsize() > max_pending:
            await asyncio.sleep(max(self.frame_ms, 1) / 1000.0)

    def pending_frames(self) -> int:
        return self._send_queue.qsize()

