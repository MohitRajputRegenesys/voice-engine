"""Audio conversion utilities for the Asterisk/3CX call path.

The conversation pipeline runs entirely in 8 kHz mu-law -- the same format the
Twilio path and the Deepgram phone-tuned STT/TTS providers already use -- so
every existing component (barge-in RMS, hold music, STT queue) works unchanged.

The RTP wire format toward/from Asterisk is configurable (``ulaw`` | ``alaw`` |
``slin16``) and every frame is converted at the RTP boundary only, so the rest
of the pipeline never sees the wire codec.
"""
from bisect import bisect_left
from typing import List

from ..twilio.audio import MULAW_TO_PCM16, linear16_to_mulaw

SAMPLE_RATE = 8000

# RTP static payload types for G.711. slin16 (L16/8000) has no static type --
# its dynamic payload type is mirrored from whatever Asterisk sends us.
WIRE_PAYLOAD_TYPES = {"ulaw": 0, "alaw": 8, "slin16": None}
BYTES_PER_SAMPLE = {"ulaw": 1, "alaw": 1, "slin16": 2}


def _build_alaw_decode_table() -> List[int]:
    """Generate G.711 A-law to 16-bit linear PCM lookup table (256 entries)."""
    table = [0] * 256
    for i in range(256):
        a_val = i ^ 0x55  # A-law alternates even/odd bits
        # 13-bit magnitude expansion (ITU G.711 reference algorithm)
        t = (a_val & 0x0F) << 4
        seg = (a_val & 0x70) >> 4
        if seg == 0:
            t += 8
        elif seg == 1:
            t += 0x108
        else:
            t += 0x108
            t <<= seg - 1
        # After XOR, bit 0x80 set means positive for A-law
        table[i] = t if (a_val & 0x80) else -t
    return table


ALAW_TO_PCM16: List[int] = _build_alaw_decode_table()

# Sorted (decoded_value, byte) pairs for nearest-neighbour A-law encoding.
_ALAW_ORDERED = sorted((value, byte) for byte, value in enumerate(ALAW_TO_PCM16))
_ALAW_VALUES = [value for value, _ in _ALAW_ORDERED]


def _alaw_byte_for(sample: int) -> int:
    """Return the A-law byte whose decoded value is nearest to ``sample``."""
    idx = bisect_left(_ALAW_VALUES, sample)
    candidates = []
    if idx < 256:
        candidates.append(_ALAW_ORDERED[idx])
    if idx > 0:
        candidates.append(_ALAW_ORDERED[idx - 1])
    best = min(candidates, key=lambda pair: (abs(pair[0] - sample), pair[1]))
    return best[1]


def linear16_to_alaw(pcm16: bytes) -> bytes:
    """Encode 16-bit little-endian linear PCM into 8-bit G.711 A-law."""
    out = bytearray(len(pcm16) // 2)
    for i in range(len(out)):
        sample = int.from_bytes(pcm16[2 * i : 2 * i + 2], "little", signed=True)
        out[i] = _alaw_byte_for(sample)
    return bytes(out)


def alaw_to_linear16(alaw: bytes) -> bytes:
    """Decode 8-bit G.711 A-law into 16-bit little-endian linear PCM."""
    out = bytearray(len(alaw) * 2)
    for i, byte in enumerate(alaw):
        out[2 * i : 2 * i + 2] = ALAW_TO_PCM16[byte].to_bytes(2, "little", signed=True)
    return bytes(out)


def mulaw_to_linear16(mulaw: bytes) -> bytes:
    """Decode 8-bit G.711 mu-law into 16-bit little-endian linear PCM."""
    out = bytearray(len(mulaw) * 2)
    for i, byte in enumerate(mulaw):
        out[2 * i : 2 * i + 2] = MULAW_TO_PCM16[byte].to_bytes(2, "little", signed=True)
    return bytes(out)


def _normalized(fmt: str) -> str:
    fmt = (fmt or "ulaw").strip().lower()
    if fmt not in WIRE_PAYLOAD_TYPES:
        raise ValueError(f"unsupported media format: {fmt!r}")
    return fmt


def wire_from_mulaw(mulaw: bytes, fmt: str) -> bytes:
    """Convert a mu-law frame into the configured RTP wire format."""
    fmt = _normalized(fmt)
    if fmt == "ulaw":
        return mulaw
    if fmt == "alaw":
        return linear16_to_alaw(mulaw_to_linear16(mulaw))
    return mulaw_to_linear16(mulaw)  # slin16


def mulaw_from_wire(payload: bytes, fmt: str) -> bytes:
    """Convert an RTP payload in the wire format into mu-law (pipeline format)."""
    fmt = _normalized(fmt)
    if fmt == "ulaw":
        return payload
    if fmt == "alaw":
        return linear16_to_mulaw(alaw_to_linear16(payload))
    return linear16_to_mulaw(payload)  # slin16: payload is 16-bit LE PCM


class WireFramer:
    """Buffers mu-law audio and emits exact RTP-frame payloads in the wire format.

    RTP packets should carry a fixed number of samples (one ptime frame, 160
    samples @ 8 kHz for 20 ms). The Deepgram streaming TTS emits arbitrary
    chunk sizes, so this buffers and slices; the final partial frame of a turn
    is padded with mu-law silence (0xFF) to keep every packet the same length.
    """

    def __init__(self, fmt: str = "ulaw", frame_ms: int = 20) -> None:
        self.fmt = _normalized(fmt)
        self.samples_per_frame = SAMPLE_RATE * frame_ms // 1000  # 160 @ 20 ms
        self.bytes_per_sample = BYTES_PER_SAMPLE[self.fmt]
        self._pending = bytearray()

    @property
    def frame_wire_bytes(self) -> int:
        return self.samples_per_frame * self.bytes_per_sample

    def push(self, mulaw: bytes) -> List[bytes]:
        """Feed mu-law audio; returns complete wire-format frames."""
        self._pending.extend(mulaw or b"")
        frames: List[bytes] = []
        n = self.samples_per_frame
        while len(self._pending) >= n:
            frame = bytes(self._pending[:n])
            del self._pending[:n]
            frames.append(wire_from_mulaw(frame, self.fmt))
        return frames

    def flush(self) -> List[bytes]:
        """Emit the remaining partial frame, silence-padded (end of a turn)."""
        if not self._pending:
            return []
        frame = bytes(self._pending)
        self._pending.clear()
        pad = self.samples_per_frame - len(frame)
        if pad > 0:
            frame += b"\xff" * pad  # mu-law silence
        return [wire_from_mulaw(frame, self.fmt)]

    def reset(self) -> None:
        """Drop any buffered partial frame (used on barge-in)."""
        self._pending.clear()
