"""Session manager implementing realtime voice session logic."""
from __future__ import annotations
import asyncio
import uuid
import time
import base64
from typing import Optional, Dict, Any, AsyncIterator, List
from .state import StateMachine, SessionState, InvalidTransition
from .providers import (
    MockSTTProvider,
    MockLLMProvider,
    MockTTSProvider,
    STTEvent,
    LLMDelta,
    AudioChunk,
    DeepgramSTTProvider,
    FacilitatorLLMProvider,
    EdgeTTSProvider,
)
import os


class SessionMetrics:
    def __init__(self):
        self.start_time = time.monotonic()
        self.first_transcript_ts = None
        self.final_transcript_ts = None
        self.llm_first_token_ts = None
        self.tts_first_audio_ts = None
        self.interruption_count = 0

    def as_dict(self):
        return {k: getattr(self, k) for k in vars(self)}


class Session:
    def __init__(self, websocket, stt_provider=None, llm_provider=None, tts_provider=None, auto_llm: bool = True):
        self.id = str(uuid.uuid4())
        self.ws = websocket
        self.state = StateMachine()
        # When True (default), a final transcript auto-triggers the configured LLM
        # provider which then speaks its answer. When False, the session only does
        # STT and emits `transcript.final`; an external client drives the response
        # by calling `speak(...)`. This is the "just add voice to my own AI" mode.
        self.auto_llm = auto_llm
        # Prefer explicit provider, then Deepgram if configured, otherwise mock
        if stt_provider:
            self.stt_provider = stt_provider
        elif os.environ.get("DEEPGRAM_API_KEY"):
            self.stt_provider = DeepgramSTTProvider(
                model=os.environ.get("STT_MODEL", "nova-2-general") or "nova-2-general"
            )
        else:
            self.stt_provider = MockSTTProvider()

        # LLM: explicit > facilitator (RAG) if configured > mock
        if llm_provider:
            self.llm_provider = llm_provider
        elif os.environ.get("FACILITATOR_API_URL"):
            self.llm_provider = FacilitatorLLMProvider(
                base_url=os.environ.get("FACILITATOR_API_URL"),
                top_k=int(os.environ.get("FACILITATOR_TOP_K", "5")),
                alpha=float(os.environ.get("FACILITATOR_ALPHA", "0.5")),
            )
        else:
            self.llm_provider = MockLLMProvider()

        # TTS: explicit > edge-tts > mock (driven by TTS_PROVIDER)
        if tts_provider:
            self.tts_provider = tts_provider
        else:
            _prov = os.environ.get("TTS_PROVIDER", "").lower()
            if _prov in ("edge", "edge-tts"):
                self.tts_provider = EdgeTTSProvider(
                    voice=os.environ.get("TTS_VOICE", "en-US-AriaNeural")
                )
            else:
                self.tts_provider = MockTTSProvider()

        # Conversation history retained for the facilitator LLM provider
        self.conversation_history: List[Dict[str, str]] = []

        # Bounded queues
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.llm_text_queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self.tts_audio_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        # Active response tracking
        self.active_response_id: Optional[str] = None
        self.active_response_cancel: Optional[asyncio.Event] = None

        self.metrics = SessionMetrics()
        self.tasks = []
        self.closed = False

    async def start(self):
        self.state.transition(SessionState.CONNECTING)
        # start provider tasks
        self.tasks.append(asyncio.create_task(self._stt_consumer()))
        self.state.transition(SessionState.LISTENING)

    async def close(self):
        if self.closed:
            return
        self.state.transition(SessionState.CLOSING)
        # cancel tasks
        for t in self.tasks:
            t.cancel()
        # notify providers by putting None
        await self._drain_queues()
        self.state.transition(SessionState.CLOSED)
        self.closed = True

    async def _drain_queues(self):
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass

    async def post_audio(self, data: bytes):
        # backpressure -> drop oldest if full
        try:
            self.audio_queue.put_nowait(data)
        except asyncio.QueueFull:
            try:
                _ = self.audio_queue.get_nowait()
            except Exception:
                pass
            await self.audio_queue.put(data)

    async def _stt_consumer(self):
        try:
            async for event in self.stt_provider.consume_audio(self.audio_queue):
                # record times
                if event.partial and self.metrics.first_transcript_ts is None:
                    self.metrics.first_transcript_ts = time.monotonic()
                if event.final:
                    self.metrics.final_transcript_ts = time.monotonic()
                    # surface the committed user turn to the client, then start LLM
                    await self._send_ws({
                        "type": "transcript.final",
                        "response_id": event.response_id,
                        "text": event.final,
                        "encoding": "text",
                    })
                    # In auto-LLM mode the engine picks its own answer and speaks it.
                    # Otherwise the external client decides what to say and calls speak().
                    if self.auto_llm:
                        await self._start_llm_for(event.final)
                else:
                    # forward partials to client (but don't add to conversation)
                    await self._send_ws({"type": "transcript.partial", "response_id": event.response_id, "sequence": event.sequence, "text": event.partial})
        except asyncio.CancelledError:
            return

    async def _start_llm_for(self, user_text: str):
        # ensure single active response
        if self.active_response_cancel:
            # interrupt previous
            self.metrics.interruption_count += 1
            self.active_response_cancel.set()

        response_id = str(uuid.uuid4())
        self.active_response_id = response_id
        cancel_event = asyncio.Event()
        self.active_response_cancel = cancel_event
        self.state.transition(SessionState.PROCESSING)

        async def llm_runner():
            seq = 0
            try:
                async for delta in self.llm_provider.stream_response(user_text, response_id):
                    if cancel_event.is_set() or response_id != self.active_response_id:
                        break
                    if self.metrics.llm_first_token_ts is None:
                        self.metrics.llm_first_token_ts = time.monotonic()
                    await self._handle_llm_delta(delta)
                    seq += 1
                # finished LLM -> after streaming, mark speaking
                if not cancel_event.is_set():
                    self.state.transition(SessionState.SPEAKING)
                    # start TTS runner
                    t = asyncio.create_task(self._tts_runner(response_id))
                    self.tasks.append(t)
            except asyncio.CancelledError:
                return

        t = asyncio.create_task(llm_runner())
        self.tasks.append(t)

    async def speak(self, text: str) -> None:
        """Speak arbitrary text coming from an EXTERNAL AI (the client's own brain).

        This is the entry point for "just add voice to my own AI": the connected
        project sends `{"type":"speak","data":"<text>"}` and the engine synthesizes
        that text with the configured TTS provider and streams `audio.chunk` frames.
        It respects barge-in: a new speak (or incoming audio) cancels the previous.
        """
        if self.active_response_cancel:
            self.active_response_cancel.set()
            self.active_response_cancel = None

        response_id = str(uuid.uuid4())
        self.active_response_id = response_id
        cancel_event = asyncio.Event()
        self.active_response_cancel = cancel_event
        self.metrics.interruption_count = 0  # fresh turn for external responses
        self.state.transition(SessionState.PROCESSING)

        async def speak_runner():
            async def text_iter() -> AsyncIterator[str]:
                yield text

            try:
                self.state.transition(SessionState.SPEAKING)
                if self.metrics.tts_first_audio_ts is None:
                    self.metrics.tts_first_audio_ts = time.monotonic()
                async for audio in self.tts_provider.synthesize(text_iter(), response_id):
                    if cancel_event.is_set() or response_id != self.active_response_id:
                        break
                    await self._send_ws({
                        "type": "audio.chunk",
                        "response_id": audio.response_id,
                        "sequence": audio.sequence,
                        "encoding": "base64",
                        "data": base64.b64encode(audio.data).decode("ascii"),
                    })
                # finished speaking
                if not cancel_event.is_set() and self.active_response_id == response_id:
                    self.active_response_id = None
                    await self._send_ws({"type": "speak.done", "response_id": response_id})
                    self.state.transition(SessionState.LISTENING)
            except asyncio.CancelledError:
                return

        self.tasks.append(asyncio.create_task(speak_runner()))

    async def _handle_llm_delta(self, delta: LLMDelta):
        # buffer text into llm_text_queue for TTS
        try:
            self.llm_text_queue.put_nowait((delta.response_id, delta.sequence, delta.text))
        except asyncio.QueueFull:
            # drop or merge; for now drop oldest
            try:
                _ = self.llm_text_queue.get_nowait()
            except Exception:
                pass
            await self.llm_text_queue.put((delta.response_id, delta.sequence, delta.text))
        await self._send_ws({"type": "llm.delta", "response_id": delta.response_id, "sequence": delta.sequence, "text": delta.text})

    async def _tts_runner(self, response_id: str):
        # create async iterator that yields buffered text chunks grouped sensibly
        async def text_iter() -> AsyncIterator[str]:
            buf = []
            last_seq = -1
            while True:
                try:
                    rid, seq, text = await asyncio.wait_for(self.llm_text_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    if buf:
                        yield "".join(buf)
                        buf = []
                    else:
                        break
                    continue
                if rid != response_id:
                    # stale chunk
                    continue
                buf.append(text)
                last_seq = seq
                # simple heuristic: flush when sentence end or buffer large
                if text.endswith(".") or len("".join(buf)) > 80:
                    yield "".join(buf)
                    buf = []
            if buf:
                yield "".join(buf)

        # consume text_iter by TTS provider
        try:
            if self.metrics.tts_first_audio_ts is None:
                self.metrics.tts_first_audio_ts = time.monotonic()
            async for audio in self.tts_provider.synthesize(text_iter(), response_id):
                # protect stale audio
                if response_id != self.active_response_id:
                    continue
                await self._send_ws({
                    "type": "audio.chunk",
                    "response_id": audio.response_id,
                    "sequence": audio.sequence,
                    "encoding": "base64",
                    "data": base64.b64encode(audio.data).decode("ascii"),
                })
            # finished speaking
            if self.active_response_id == response_id:
                self.active_response_id = None
                self.state.transition(SessionState.LISTENING)
        except asyncio.CancelledError:
            return

    async def _send_ws(self, payload: Dict[str, Any]):
        try:
            await self.ws.send_json(payload)
        except Exception:
            pass
