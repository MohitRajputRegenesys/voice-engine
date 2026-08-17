import asyncio

import pytest

from voice_engine.providers import MockSTTProvider, MockLLMProvider, MockTTSProvider
from voice_engine.session import Session
from voice_engine.state import SessionState, StateMachine, InvalidTransition


class DummyWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass


def test_state_machine_transition_validation():
    sm = StateMachine()
    assert sm.state == SessionState.IDLE

    sm.transition(SessionState.CONNECTING)
    assert sm.state == SessionState.CONNECTING

    sm.transition(SessionState.LISTENING)
    assert sm.state == SessionState.LISTENING

    with pytest.raises(InvalidTransition):
        sm.transition(SessionState.SPEAKING)


@pytest.mark.asyncio
async def test_session_starts_and_streams_mock_response():
    ws = DummyWebSocket()
    session = Session(
        ws,
        stt_provider=MockSTTProvider(),
        llm_provider=MockLLMProvider(),
        tts_provider=MockTTSProvider(),
    )

    await session.start()
    await session.post_audio(b"hello")
    await asyncio.sleep(0.05)
    await session.post_audio(b"<end>")
    await asyncio.sleep(1.0)

    assert session.state.state in {SessionState.LISTENING, SessionState.SPEAKING}
    assert len(ws.sent) > 0

    await session.close()
