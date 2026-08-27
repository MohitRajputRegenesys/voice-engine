"""One-off live probe: does our Deepgram STT websocket accept auth/model/language?

Streams 100 ms of silence + <end>, waits briefly for events. Success criteria:
no RuntimeError raised (an auth/model failure triggers fail-fast), so the
WS handshake + params are correct.
"""
import asyncio

from dotenv import load_dotenv

load_dotenv(".env")

from voice_engine.providers import build_stt_provider


async def main():
    p = build_stt_provider()
    print(f"[STT] probing {type(p).__name__} model={p.model} language={p.language}")

    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait(b"\x00\x00" * 1600)   # 100 ms of digital silence, PCM16 @16 kHz
    q.put_nowait(b"<end>")

    events = []

    async def consume():
        async for ev in p.consume_audio(q):
            kind = "final" if ev.final else "partial"
            text = ev.final or ev.partial
            events.append((kind, text))
            if ev.final:
                break   # enough proof; leave the reconnect loop early

    try:
        # After <end> we send CloseStream; silence normally emits no transcript,
        # so cap the wait and cancel the (infinite) consumer afterwards.
        await asyncio.wait_for(consume(), timeout=12.0)
    except asyncio.TimeoutError:
        pass
    except RuntimeError as exc:
        print(f"[FAIL] provider rejected config/key: {exc}")
        return
    except Exception as exc:
        print(f"[FAIL] unexpected {type(exc).__name__}: {exc}")
        return
    finally:
        await q.put(None)

    print("[OK] Deepgram websocket connected (auth + model + params accepted).")
    if events:
        print("     events:", events)


if __name__ == "__main__":
    asyncio.run(main())