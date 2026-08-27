"""Demo: how an EXTERNAL AI project adds "speak + listen" to its own brain.

Runs against the voice engine (start with `uv run uvicorn voice_engine.server:app
--port 8001`). It exercises the decoupled mode: the session does STT only, this
demo's "AI" (a trivial echo) generates the answer, and the engine speaks it.

Compare with example_client.py, which relies on the engine's OWN built-in LLM.
"""
import asyncio

from voice_engine.client import VoiceClient


async def my_model_generate(user_text: str) -> str:
    # In a real project this is your own LLM / chatbot endpoint.
    return f"You said: {user_text}"


async def run():
    async with VoiceClient() as vc:
        # 1. Simulate a user speaking.
        await vc.send_audio(b"\x00\x00" * 160)  # ~10ms of silence (mock STT)
        await vc.end_utterance()

        async for msg in vc.messages():
            typ = msg.get("type")
            if typ == "transcript.final":
                print("[transcript.final]", msg["text"])
                # 2. YOUR AI decides the answer...
                answer = await my_model_generate(msg["text"])
                print("[my AI says]", answer)
                # 3. ...and the voice engine speaks it.
                await vc.speak(answer)
            elif typ == "audio.chunk":
                print("[audio.chunk] received (play over your speaker)")
                break
            elif typ == "speak.done":
                print("[speak.done] finished")
                break


if __name__ == "__main__":
    asyncio.run(run())
