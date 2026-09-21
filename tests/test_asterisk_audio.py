"""Tests for A-law conversion, wire-format helpers and the RTP wire framer."""
import pytest

from voice_engine.asterisk.audio import (
    ALAW_TO_PCM16,
    WIRE_PAYLOAD_TYPES,
    WireFramer,
    alaw_to_linear16,
    linear16_to_alaw,
    mulaw_from_wire,
    mulaw_to_linear16,
    wire_from_mulaw,
)
from voice_engine.twilio.audio import MULAW_TO_PCM16


# ------------------------------------------------------------------ A-law

def test_alaw_known_values():
    # Standard ITU G.711 A-law decode points.
    assert ALAW_TO_PCM16[0xD5] == 8      # "positive zero"
    assert ALAW_TO_PCM16[0x55] == -8     # "negative zero"
    assert ALAW_TO_PCM16[0x00] == -5504  # negative, segment 5, mantissa 5
    assert ALAW_TO_PCM16[0x2A] == -32256  # negative full scale
    assert ALAW_TO_PCM16[0xFF] == 848


def test_alaw_encode_decode_bijective():
    """Nearest-neighbour encode must invert the decode table exactly."""
    for byte in range(256):
        pcm = alaw_to_linear16(bytes([byte]))
        assert linear16_to_alaw(pcm) == bytes([byte])


def test_alaw_quantization_error_bounded():
    """Round-trip error must stay within half of the largest A-law step."""
    largest_step_half = 1024  # top segment step is 2048 (16-bit domain)
    for sample in range(-32768, 32769, 11):
        encoded = linear16_to_alaw(sample.to_bytes(2, "little", signed=True))
        decoded = int.from_bytes(alaw_to_linear16(encoded), "little", signed=True)
        assert abs(decoded - sample) <= largest_step_half


# ------------------------------------------------------------ wire formats

def test_wire_payload_types():
    assert WIRE_PAYLOAD_TYPES["ulaw"] == 0   # PCMU static
    assert WIRE_PAYLOAD_TYPES["alaw"] == 8   # PCMA static
    assert WIRE_PAYLOAD_TYPES["slin16"] is None  # dynamic -> mirrored


def test_ulaw_is_passthrough():
    payload = bytes(range(160))
    assert wire_from_mulaw(payload, "ulaw") == payload
    assert mulaw_from_wire(payload, "ulaw") == payload


def test_slin16_conversion():
    silence = b"\xff" * 8  # mu-law silence decodes to 0
    assert wire_from_mulaw(silence, "slin16") == b"\x00" * 16
    assert mulaw_from_wire(b"\x00" * 16, "slin16") == b"\xff" * 8
    # 16-bit little-endian round trip keeps sample count
    pcm = b"\x10\x00" * 160
    mulaw = mulaw_from_wire(pcm, "slin16")
    assert len(mulaw) == 160
    assert MULAW_TO_PCM16[mulaw[0]] == 16  # exact for small values


def test_alaw_conversion_round_trip_error_bounded():
    mulaw = bytes((i * 7) % 256 for i in range(160))
    wire = wire_from_mulaw(mulaw, "alaw")
    back = mulaw_from_wire(wire, "alaw")
    assert len(back) == 160
    for original, result in zip(mulaw, back):
        diff = abs(MULAW_TO_PCM16[original] - MULAW_TO_PCM16[result])
        # two quantizations (mu-law -> A-law -> mu-law): A-law nearest distance
        # is <= 1024 and the mu-law re-encode adds <= 512.
        assert diff <= 1600


def test_unknown_format_rejected():
    with pytest.raises(ValueError):
        wire_from_mulaw(b"\x00" * 8, "opus")
    with pytest.raises(ValueError):
        mulaw_from_wire(b"\x00" * 8, "gsm")


# ------------------------------------------------------------- WireFramer

def test_framer_emits_exact_frames_and_pads_tail():
    framer = WireFramer("ulaw", frame_ms=20)  # 160 samples -> 160 bytes
    frames = framer.push(b"\x00" * 350)
    assert [len(f) for f in frames] == [160, 160]
    assert frames[0] == b"\x00" * 160
    tail = framer.flush()
    assert len(tail) == 1
    assert len(tail[0]) == 160
    assert tail[0][:30] == b"\x00" * 30       # remaining real audio
    assert tail[0][30:] == b"\xff" * 130      # mu-law silence padding
    assert framer.flush() == []               # nothing left


def test_framer_slin16_frame_size():
    framer = WireFramer("slin16", frame_ms=20)
    frames = framer.push(b"\xff" * 160)  # 160 mu-law samples
    assert len(frames) == 1
    assert frames[0] == b"\x00" * 320    # 160 samples * 2 bytes, silence


def test_framer_reset_drops_partial():
    framer = WireFramer("ulaw", frame_ms=20)
    assert framer.push(b"\x00" * 100) == []
    framer.reset()
    assert framer.flush() == []
