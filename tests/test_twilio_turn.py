"""Unit tests for the Twilio turn orchestrator (greeting, RAG turn, barge-in, hold music)."""
import asyncio

import pytest

from voice_engine.providers import AudioChunk, LLMDelta
from voice_engine.twilio.session import CallState, TwilioCallSession
from voice_engine.twilio.turn import TwilioTurnOrchestrator, sanitize_speech_text


class FakeTTSProvider:
    def __init__(self, chunk=b"\x00" * 200):
        self.chunk = chunk
        self.aborted = False
        self.synthesized_text = []

    async def synthesize(self, text_iter, response_id):
        async for text in text_iter:
            self.synthesized_text.append(text)
            yield AudioChunk(response_id=response_id, sequence=0, data=self.chunk)

    async def abort(self):
        self.aborted = True


class SlowFakeTTSProvider(FakeTTSProvider):
    """Simulates the LLM + TTS generation gap before the first audio chunk."""

    def __init__(self, chunk=b"\xEE" * 200, delay: float = 0.3):
        super().__init__(chunk=chunk)
        self.delay = delay

    async def synthesize(self, text_iter, response_id):
        await asyncio.sleep(self.delay)
        async for text in text_iter:
            self.synthesized_text.append(text)
            yield AudioChunk(response_id=response_id, sequence=0, data=self.chunk)


class FakeLLMProvider:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    async def stream_response(self, prompt, response_id):
        self.prompts.append(prompt)
        yield LLMDelta(response_id=response_id, sequence=0, text=self.answer)


@pytest.fixture
def call_settings(monkeypatch):
    """Small prebuffer + instant barge-in + instant hold music for determinism."""
    monkeypatch.setenv("TWILIO_PREBUFFER_MS", "10")
    monkeypatch.setenv("TWILIO_BARGE_IN_RMS_THRESHOLD", "1000")
    monkeypatch.setenv("TWILIO_BARGE_IN_CONSECUTIVE_FRAMES", "1")
    monkeypatch.setenv("TWILIO_WELCOME_GREETING", "Hello there!")
    monkeypatch.setenv("TWILIO_HOLD_MUSIC", "true")
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_DELAY_MS", "0")


def build_harness(llm_answer="We offer a Machine Learning 6 month course.", tts=None, llm=None):
    session = TwilioCallSession(call_sid="CA1", stream_sid="STREAM1")
    tts = tts if tts is not None else FakeTTSProvider()
    llm = llm if llm is not None else FakeLLMProvider(llm_answer)
    media, marks, clears = [], [], []

    async def send_media(chunk):
        media.append(chunk)

    async def send_mark(name):
        marks.append(name)

    async def send_clear():
        clears.append(True)

    orch = TwilioTurnOrchestrator(session, tts, llm, send_media, send_mark, send_clear)
    return session, tts, llm, media, marks, clears, orch


def test_sanitize_speech_text_cleans_markdown_and_currency():
    assert sanitize_speech_text("**Bold** and Rs 85,000") == "Bold and 85000 rupees"
    assert sanitize_speech_text("Hello world") == "Hello world"


@pytest.mark.asyncio
async def test_welcome_greeting_plays_and_marks(call_settings):
    session, tts, llm, media, marks, clears, orch = build_harness()
    await orch.send_welcome_greeting()

    assert media  # TTS audio was pre-buffered and sent
    assert marks  # mark sent for turn sequencing
    assert session.state == CallState.WELCOME  # waiting for the mark ack

    await orch.handle_mark_event(marks[0])
    assert session.state == CallState.LISTENING
    assert session.turn_number == 1
    # welcome greeting committed as the assistant's first message
    assert session.conversation[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_user_transcript_runs_rag_turn_and_marks(call_settings):
    session, tts, llm, media, marks, clears, orch = build_harness()
    session.transition_to(CallState.LISTENING)

    await orch.handle_user_transcript("What courses do you offer?")

    assert llm.prompts == ["What courses do you offer?"]
    assert tts.synthesized_text  # RAG answer text was fed to TTS
    assert media  # audio out to the phone
    assert marks  # mark sent
    assert session.pending_assistant_reply  # reply saved until mark ack

    await orch.handle_mark_event(marks[0])
    assert session.state == CallState.LISTENING
    assert session.turn_number == 1
    assert session.conversation[-1]["role"] == "assistant"
    assert "Machine Learning" in session.conversation[-1]["content"]


@pytest.mark.asyncio
async def test_barge_in_during_speaking(call_settings):
    session, tts, llm, media, marks, clears, orch = build_harness()
    session.transition_to(CallState.SPEAKING)

    # 0x00 is the loudest mu-law byte; 1 frame triggers with consecutive=1
    await orch.handle_inbound_audio(b"\x00" * 160)

    assert clears  # Twilio buffer was flushed
    assert tts.aborted  # the engine's TTS provider was aborted
    assert session.state == CallState.LISTENING


@pytest.mark.asyncio
async def test_quiet_audio_does_not_barge_in(call_settings):
    session, tts, llm, media, marks, clears, orch = build_harness()
    session.transition_to(CallState.SPEAKING)

    # 0xff is mu-law silence (RMS 0) -> below threshold, no barge-in
    for _ in range(5):
        await orch.handle_inbound_audio(b"\xff" * 160)

    assert not clears
    assert not tts.aborted
    assert session.state == CallState.SPEAKING


@pytest.mark.asyncio
async def test_transcript_ignored_while_speaking(call_settings):
    session, tts, llm, media, marks, clears, orch = build_harness()
    session.transition_to(CallState.SPEAKING)

    await orch.handle_user_transcript("ignore me")
    assert not llm.prompts
    assert not media


# ---------------------------------------------------------------- hold music

@pytest.mark.asyncio
async def test_hold_music_plays_during_gap_and_stops_before_speech(call_settings):
    slow_tts = SlowFakeTTSProvider(delay=0.3)
    session, tts, llm, media, marks, clears, orch = build_harness(tts=slow_tts)
    session.transition_to(CallState.LISTENING)

    await orch.handle_user_transcript("What courses do you offer?")
    await orch.handle_mark_event(marks[0])

    speech = [i for i, c in enumerate(media) if c == b"\xEE" * 200]
    music = [i for i, c in enumerate(media) if c != b"\xEE" * 200]
    assert music, "no hold music was streamed during the generation gap"
    assert speech, "answer audio missing"
    # every music frame strictly precedes the first speech frame: never interleaved
    assert max(music) < min(speech)
    assert orch._hold_music_task is None  # pump fully stopped after the turn


@pytest.mark.asyncio
async def test_hold_music_skipped_for_fast_answers(call_settings, monkeypatch):
    monkeypatch.setenv("TWILIO_HOLD_MUSIC_DELAY_MS", "10000")
    session, tts, llm, media, marks, clears, orch = build_harness(
        tts=SlowFakeTTSProvider(delay=0.0)
    )
    session.transition_to(CallState.LISTENING)

    await orch.handle_user_transcript("What courses do you offer?")
    await orch.handle_mark_event(marks[0])

    assert media
    assert all(c == b"\xEE" * 200 for c in media)  # answer only, zero music frames
    assert orch._hold_music_task is None


@pytest.mark.asyncio
async def test_hold_music_disabled(call_settings, monkeypatch):
    monkeypatch.setenv("TWILIO_HOLD_MUSIC", "false")
    session, tts, llm, media, marks, clears, orch = build_harness(
        tts=SlowFakeTTSProvider(delay=0.2)
    )
    session.transition_to(CallState.LISTENING)

    await orch.handle_user_transcript("What courses do you offer?")
    await orch.handle_mark_event(marks[0])

    assert media
    assert all(c == b"\xEE" * 200 for c in media)  # no music frames
    assert orch._hold_music_loop == b""  # never even loaded


@pytest.mark.asyncio
async def test_barge_in_during_hold_music_stops_music(call_settings):
    session, tts, llm, media, marks, clears, orch = build_harness(
        tts=SlowFakeTTSProvider(delay=0.5)
    )
    session.transition_to(CallState.LISTENING)

    turn_task = asyncio.create_task(orch.handle_user_transcript("Wait, actually one more thing"))
    await asyncio.sleep(0.1)  # music is playing now
    assert orch._hold_music_task is not None

    await orch.handle_inbound_audio(b"\x00" * 160)  # loud -> full barge-in (consecutive=1)
    assert clears
    assert session.state == CallState.LISTENING
    assert orch._hold_music_task is None  # music stopped instantly
    assert all(c != b"\xEE" * 200 for c in media)  # the interrupted answer never played

    await turn_task


@pytest.mark.asyncio
async def test_caller_speech_during_thinking_stops_music_only(call_settings):
    session, tts, llm, media, marks, clears, orch = build_harness(
        tts=SlowFakeTTSProvider(delay=0.5)
    )
    session.transition_to(CallState.THINKING)
    orch._start_hold_music()
    await asyncio.sleep(0.05)
    assert orch._hold_music_task is not None

    await orch.handle_inbound_audio(b"\x00" * 160)  # loud
    assert orch._hold_music_task is None           # music stopped...
    assert session.state == CallState.THINKING     # ...but no full barge-in
    assert not clears
    await orch._stop_hold_music()  # idempotent