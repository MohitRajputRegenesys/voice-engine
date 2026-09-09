"""Tests for the hold-music bed: mu-law encoder, pad synthesis, file loading."""
import math
import struct
import wave

import pytest

from voice_engine.twilio.audio import MULAW_TO_PCM16, linear16_to_mulaw
from voice_engine.twilio.config import get_settings
from voice_engine.twilio.hold_music import (
    LOOP_SECONDS,
    SAMPLE_RATE,
    _synth_pad_pcm,
    load_hold_music,
)


def _decode_mulaw(raw: bytes) -> list:
    return [MULAW_TO_PCM16[b] for b in raw]


# ---------------------------------------------------------------- mu-law encoder

@pytest.mark.parametrize(
    "sample,expected",
    [
        (0, 0xFF),
        (8, 0xFE),
        (-8, 0x7E),
        (132, 0xEF),
        (32124, 0x80),
        (-32124, 0x00),
        (32767, 0x80),   # clipped
        (-32768, 0x00),  # clipped
    ],
)
def test_linear16_to_mulaw_known_values(sample, expected):
    assert linear16_to_mulaw(struct.pack("<h", sample)) == bytes([expected])


def test_mulaw_roundtrip_low_amplitude_sine():
    """Encode/decode a low-amplitude sine and verify quantization error stays small."""
    amp = 2000
    pcm = b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * 50 * i / SAMPLE_RATE)))
        for i in range(SAMPLE_RATE)
    )
    decoded = _decode_mulaw(linear16_to_mulaw(pcm))
    original = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    errors = [abs(d - o) for d, o in zip(decoded, original)]
    assert max(errors) <= 80  # within half a mu-law step in the used segments


# ---------------------------------------------------------------- built-in pad

def test_pad_is_seamless_and_frame_aligned():
    pcm = _synth_pad_pcm()
    assert len(pcm) == SAMPLE_RATE * LOOP_SECONDS * 2  # 16-bit samples
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    # A seamless loop join is no sharper than the waveform's own slew rate:
    # the last->first step must stay within the natural adjacent-sample delta
    # (all partials complete whole cycles over the integer-second window, so
    # the waveform continues exactly where it left off).
    adjacent = [abs(samples[i + 1] - samples[i]) for i in range(len(samples) - 1)]
    join_delta = abs(samples[0] - samples[-1])
    assert join_delta <= 2 * max(adjacent)
    # 100 ms pump frames (800 samples) divide the loop evenly
    assert (len(pcm) // 2) % 800 == 0


def test_pad_is_audible_but_not_clipping():
    pcm = _synth_pad_pcm()
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    peak = max(abs(s) for s in samples)
    assert peak > 500      # audible, not silence
    assert peak < 30000    # no clipping


# ---------------------------------------------------------------- loading

def test_load_builtin_pad_by_default(monkeypatch):
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_FILE", "")
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_GAIN", "0.3")
    audio = load_hold_music(get_settings())
    assert len(audio) == SAMPLE_RATE * LOOP_SECONDS  # mu-law: 1 byte per sample
    decoded = _decode_mulaw(audio)
    peak = max(abs(s) for s in decoded)
    assert 200 < peak < 12000  # audible background level at default gain


def test_load_is_cached(monkeypatch):
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_FILE", "")
    first = load_hold_music(get_settings())
    second = load_hold_music(get_settings())
    assert first is second


def test_zero_gain_is_mu_law_silence(monkeypatch):
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_FILE", "")
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_GAIN", "0.0")
    audio = load_hold_music(get_settings())
    assert audio == b"\xff" * len(audio)  # 0xFF decodes to exactly 0


def test_load_raw_mulaw_file(tmp_path, monkeypatch):
    raw = bytes([0xFF]) * 1600  # 200 ms of mu-law silence
    path = tmp_path / "hold.mulaw"
    path.write_bytes(raw)
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_FILE", str(path))
    audio = load_hold_music(get_settings())
    assert len(audio) == 1600  # silence passes through unchanged


def test_load_wav_file(tmp_path, monkeypatch):
    path = tmp_path / "hold.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack("<" + "h" * 800, *([1000] * 800)))
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_FILE", str(path))
    audio = load_hold_music(get_settings())
    assert len(audio) == 800  # 800 samples -> 800 mu-law bytes
    decoded = _decode_mulaw(audio)
    assert all(abs(s - 300) <= 30 for s in decoded)  # 1000 * default gain 0.3


def test_load_missing_file_falls_back_to_pad(tmp_path, monkeypatch):
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_FILE", str(tmp_path / "missing.mulaw"))
    audio = load_hold_music(get_settings())
    assert len(audio) == SAMPLE_RATE * LOOP_SECONDS  # built-in pad


def test_load_rejects_stereo_wav_and_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack("<" + "h" * 1600, *([100] * 1600)))
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_FILE", str(path))
    audio = load_hold_music(get_settings())
    assert len(audio) == SAMPLE_RATE * LOOP_SECONDS  # built-in pad, no raise