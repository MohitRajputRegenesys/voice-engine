"""Simple example WebSocket client to exercise the mocked realtime flow."""
import asyncio
import websockets
import json


async def run():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # simulate user speaks: send partial audio then end
        await ws.send(json.dumps({"type": "audio", "data": "chunk1"}))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "audio", "data": "chunk2"}))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "audio", "data": "<end>"}))

        # read responses for a bit
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                print("recv:", msg)
        except asyncio.TimeoutError:
            pass

if __name__ == "__main__":
    asyncio.run(run())
