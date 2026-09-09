"""Hold music for the Twilio call path.

While the RAG brain / TTS pipeline generates an answer, the engine streams a
soft, seamless music bed to the phone instead of dead air, and stops it the
instant the first spoken byte is ready. Two sources are supported:

* ``TWILIO_HOLD_MUSIC_FILE`` -- a custom track as raw mu-law (8 kHz mono) or a
  16-bit 8 kHz mono WAV;
* otherwise a built-in procedurally synthesized ambient pad (royalty-free by
  construction, generated once per process and cached).

Everything is produced as raw mu-law 8 kHz mono -- the exact format the Twilio
Media Stream consumes -- so no conversion happens on the hot path.
"""
import logging
import math
import wave

from .audio import MULAW_TO_PCM16, linear16_to_mulaw
from .config import TwilioSettings

logger = logging.getLogger("voice_engine.twilio.hold_music")

SAMPLE_RATE = 8000
LOOP_SECONDS = 4  # integer seconds keep the synthesized loop seamless

_CACHE: dict = {}


def _synth_pad_pcm(seconds: int = LOOP_SECONDS, peak: float = 0.6) -> bytes:
    """Synthesize the built-in soft ambient pad as 16-bit PCM.

    Every partial uses an integer frequency over an integer-second window, so
    each completes whole cycles and the loop point is sample-exact seamless.
    """
    total = SAMPLE_RATE * seconds
    # Warm A-major-ish pad: (frequency_hz, relative_gain) -- all integer Hz.
    voices = (
        (110.0, 0.30),  # A2
        (165.0, 0.22),  # E3
        (220.0, 0.20),  # A3
        (275.0, 0.12),  # C#4 (a touch flat; inaudible in a pad)
        (330.0, 0.12),  # E4
        (440.0, 0.08),  # A4
    )
    lfo_freq = 0.25   # 1 cycle per 4 s loop -> seamless
    lfo_depth = 0.35
    norm = sum(gain for _, gain in voices)

    two_pi = 2.0 * math.pi
    out = bytearray()
    for n in range(total):
        t = n / SAMPLE_RATE
        mod = 1.0 + lfo_depth * math.sin(two_pi * lfo_freq * t)
        value = 0.0
        for freq, gain in voices:
            value += gain * math.sin(two_pi * freq * t)
        value = value * mod * peak / norm
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        out += int(value * 32767).to_bytes(2, "little", signed=True)
    return bytes(out)


def _pcm_from_mulaw(raw: bytes) -> bytes:
    pcm = bytearray(len(raw) * 2)
    for i, byte in enumerate(raw):
        pcm[2 * i : 2 * i + 2] = MULAW_TO_PCM16[byte].to_bytes(2, "little", signed=True)
    return bytes(pcm)


def _load_music_file_pcm(path: str) -> bytes:
    """Load a custom hold-music track as 16-bit PCM.

    Accepts raw mu-law (8 kHz mono) or a 16-bit 8 kHz mono WAV.
    """
    with open(path, "rb") as fh:
        header = fh.read(12)
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        with wave.open(path, "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
        if channels != 1:
            raise ValueError(f"hold music WAV must be mono (got {channels} channels)")
        if rate != SAMPLE_RATE:
            raise ValueError(f"hold music WAV must be {SAMPLE_RATE} Hz (got {rate} Hz)")
        if width != 2:
            raise ValueError(f"hold music WAV must be 16-bit (got {width * 8}-bit)")
        return frames

    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw:
        raise ValueError("hold music file is empty")
    return _pcm_from_mulaw(raw)


def _apply_gain(pcm16: bytes, gain: float) -> bytes:
    if gain >= 0.999:
        return pcm16
    out = bytearray(len(pcm16))
    for i in range(0, len(pcm16), 2):
        sample = int.from_bytes(pcm16[i : i + 2], "little", signed=True)
        scaled = int(sample * gain)
        if scaled > 32767:
            scaled = 32767
        elif scaled < -32768:
            scaled = -32768
        out[i : i + 2] = scaled.to_bytes(2, "little", signed=True)
    return bytes(out)


def load_hold_music(settings: TwilioSettings) -> bytes:
    """Return the hold-music loop as raw mu-law 8 kHz mono bytes (cached).

    Prefers ``TWILIO_HOLD_MUSIC_FILE`` when configured; falls back to the
    built-in pad. Never raises -- a broken custom file simply degrades to the
    built-in pad so calls are never affected.
    """
    gain = max(0.0, min(1.0, settings.hold_music_gain))
    key = (settings.hold_music_file or "", round(gain, 3))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    pcm16 = b""
    if settings.hold_music_file:
        try:
            pcm16 = _load_music_file_pcm(settings.hold_music_file)
        except Exception as exc:  # noqa: BLE001 - degrade to built-in pad
            logger.warning(
                "Could not load hold music file %r (%s) -- using built-in pad",
                settings.hold_music_file,
                exc,
            )
    if not pcm16:
        pcm16 = _synth_pad_pcm()
    if gain <= 0.0:
        audio = b"\xff" * len(pcm16)  # mu-law silence
    else:
        audio = linear16_to_mulaw(_apply_gain(pcm16, gain))

    _CACHE[key] = audio
    return audio