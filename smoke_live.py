"""Live smoke test against the providers configured in .env.

Verifies the real vendor round-trip without needing the gateway running:

  uv run python smoke_live.py                 # TTS check -> writes smoke_tts.mp3
  uv run python smoke_live.py sample.wav      # + STT check on a 16 kHz mono PCM16 WAV
"""
import asyncio
import os
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

from voice_engine.providers import build_stt_provider, build_tts_provider  # noqa: E402
from voice_engine.rest import synthesize_once, transcribe_once  # noqa: E402


def describe(provider):
    return f"{type(provider).__name__} (model={getattr(provider, 'model', getattr(provider, 'voice', '-'))})"


async def main():
    out_path = pathlib.Path(__file__).resolve().parent / "smoke_tts.mp3"
    tts = build_tts_provider()
    print(f"[TTS] provider : {describe(tts)}")
    data = await synthesize_once("Hello! The voice engine is alive.", tts_provider=tts)
    out_path.write_bytes(data)
    head = ", ".join(f"{b:#04x}" for b in data[:3])
    print(f"[TTS] wrote {out_path.name}: {len(data)} bytes, starts with [{head}] "
          f"(MP3 frames usually begin with ID3 or 0xff)")
    if len(data) < 1000:
        print("[TTS] WARNING: suspiciously small payload - check logs above.")

    if len(sys.argv) > 1:
        wav = sys.argv[1]
        stt = build_stt_provider()
        print(f"[STT] provider : {describe(stt)}")
        raw = pathlib.Path(wav).read_bytes()
        if raw[:4] == b"RIFF":
            raw = raw[44:]  # strip standard WAV header -> raw PCM16
        text = await transcribe_once(raw, stt_provider=stt)
        print(f"[STT] transcript: {text!r}")

    if os.environ.get("DEEPGRAM_API_KEY"):
        print("[OK] DEEPGRAM_API_KEY present - both STT/TTS runs were live.")
    else:
        print("[NOTE] No DEEPGRAM_API_KEY found - ran against mock providers.")


if __name__ == "__main__":
    asyncio.run(main())