import asyncio
import websockets
import json

URI = "ws://localhost:8001/ws"  # voice-engine runs on 8001 so it doesn't clash with the facilitator on 8000


async def run():
    async with websockets.connect(URI) as ws:
        # Simulate a user speaking: send partial audio frames then the end marker.
        await ws.send(json.dumps({"type": "audio", "data": "chunk1"}))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "audio", "data": "chunk2"}))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "audio", "data": "<end>"}))

        # Read responses for a bit.
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                print("recv:", msg)
        except asyncio.TimeoutError:
            pass


if __name__ == "__main__":
    asyncio.run(run())

