Voice Engine — Realtime conversational voice runtime

Quickstart

1. Install:

```
python -m pip install -e .
pip install -r requirements.txt || true
```

2. Run:

```
uvicorn voice_engine.server:app --reload
```

3. Connect a WebSocket client to `ws://localhost:8000/ws` and send JSON messages:

- `{ "type": "audio", "data": "<end>" }` to mark end of utterance (mock STT)
- `{ "type": "audio", "data": "..." }` to send audio chunks

This repository contains a production-oriented session/state machine, bounded queues, cancellation propagation, and mock providers to exercise the realtime acceptance test described in the project plan.
