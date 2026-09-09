"""Audio utilities for the Twilio call path.

Twilio Media Streams delivers and expects raw G.711 mu-law (8 kHz, 8-bit).
These helpers decode base64 mu-law frames, compute RMS energy for barge-in
detection, and pre-buffer outbound audio so speech starts without stutter.
"""
import base64
import math
from typing import List


def _build_mulaw_table() -> List[int]:
    """Generate G.711 mu-law to 16-bit linear PCM lookup table (256 entries)."""
    table = [0] * 256
    for i in range(256):
        # Bitwise complement of input byte
        mu_val = i ^ 0xFF
        # Sign bit
        sign = -1 if (mu_val & 0x80) else 1
        # Exponent (bits 6-4)
        exponent = (mu_val >> 4) & 0x07
        # Mantissa (bits 3-0)
        mantissa = mu_val & 0x0F
        # G.711 expansion formula
        sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
        table[i] = sign * sample
    return table


MULAW_TO_PCM16: List[int] = _build_mulaw_table()


def rms(mulaw_bytes: bytes) -> float:
    """Calculate Root Mean Square (RMS) energy from 8 kHz 8-bit mu-law bytes."""
    if not mulaw_bytes:
        return 0.0

    sum_squares = 0.0
    for byte in mulaw_bytes:
        pcm_val = MULAW_TO_PCM16[byte]
        sum_squares += pcm_val * pcm_val

    mean_square = sum_squares / len(mulaw_bytes)
    return math.sqrt(mean_square)


def decode_mulaw_base64(payload: str) -> bytes:
    """Decode a base64 string payload to raw mu-law bytes."""
    return base64.b64decode(payload)


def encode_mulaw_base64(data: bytes) -> str:
    """Encode raw mu-law bytes to a base64 string payload."""
    return base64.b64encode(data).decode("ascii")


_MU_BIAS = 0x84   # G.711 mu-law encoding bias
_MU_CLIP = 32635  # largest magnitude that survives bias addition


def linear16_to_mulaw(pcm16: bytes) -> bytes:
    """Encode 16-bit little-endian linear PCM into 8-bit G.711 mu-law.

    Pure-Python counterpart of the :data:`MULAW_TO_PCM16` decode table
    (``audioop`` was removed in Python 3.13). Round-trips with the decoder
    within the expected mu-law quantization error.
    """
    out = bytearray(len(pcm16) // 2)
    for i in range(len(out)):
        sample = int.from_bytes(pcm16[2 * i : 2 * i + 2], "little", signed=True)
        sign = 0x80 if sample < 0 else 0x00
        if sign:
            sample = -sample
        if sample > _MU_CLIP:
            sample = _MU_CLIP
        sample += _MU_BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and (sample & mask) == 0:
            exponent -= 1
            mask >>= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        out[i] = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return bytes(out)


class PreBuffer:
    """Pre-buffers initial TTS audio chunks before streaming to prevent start-of-turn stutter.

    At 8 kHz 8-bit mu-law, 1 sample = 1 byte = 1/8000 s = 0.125 ms.
    """

    def __init__(self, target_ms: int = 180, sample_rate: int = 8000) -> None:
        self.target_bytes: int = int((target_ms / 1000.0) * sample_rate)
        self._buffer: bytearray = bytearray()
        self.is_flushed: bool = False

    def push(self, chunk: bytes) -> List[bytes]:
        """Push a chunk of audio. Returns a list of ready chunks to transmit."""
        if not chunk:
            return []

        if self.is_flushed:
            return [chunk]

        self._buffer.extend(chunk)
        if len(self._buffer) >= self.target_bytes:
            ready_data = bytes(self._buffer)
            self._buffer.clear()
            self.is_flushed = True
            return [ready_data]

        return []

    def flush(self) -> List[bytes]:
        """Flush remaining buffered audio chunks immediately."""
        if self._buffer:
            ready_data = bytes(self._buffer)
            self._buffer.clear()
            self.is_flushed = True
            return [ready_data]
        self.is_flushed = True
        return []

    def reset(self) -> None:
        """Reset pre-buffer state for the next turn."""
        self._buffer.clear()
        self.is_flushed = False


def is_ws_open(ws: object) -> bool:
    """Check if a WebSocket connection object is active and open across websockets library versions."""
    if ws is None:
        return False
    if hasattr(ws, "closed"):
        return not ws.closed
    if hasattr(ws, "state"):
        try:
            import websockets

            return ws.state == websockets.State.OPEN
        except Exception:
            pass
    return getattr(ws, "close_code", None) is None