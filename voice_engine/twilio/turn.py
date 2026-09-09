"""Turn-taking orchestrator for Twilio calls (listen -> think -> speak).

The orchestrator drives the engine's own TTS provider and the RAG brain from
the saleKnowledgeBase app backend (via the configured LLM provider). It enforces
the listen/think/speak sequence, 180 ms pre-buffering, mark-based sequencing,
and instant RMS barge-in handling -- the same behavior as the reference calling
agent, but implemented entirely with voice-engine providers.
"""
import asyncio
import re
import uuid
import logging
from typing import Awaitable, Callable, List, Optional

from ..providers import AudioChunk, LLMDelta
from .audio import PreBuffer, rms
from .config import get_settings
from .hold_music import SAMPLE_RATE, load_hold_music
from .session import CallState, TwilioCallSession

logger = logging.getLogger("voice_engine.twilio.turn")


def sanitize_speech_text(text: str) -> str:
    """Sanitize LLM output so TTS reads natural spoken English.

    Fixes fragmented-model artifacts: markdown symbols, unicode dashes, spaces
    before punctuation ("choice !"), and currency written as "Rs 85,000".
    """
    if not text:
        return ""
    # Strip markdown symbols (**bold**, # heading, `code`, bullets)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+", "", text)
    text = re.sub(r"_+", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"^\s*[\-\*]\s+", "", text, flags=re.MULTILINE)
    # Unicode dashes (the model writes "six ‑ month") -> plain hyphen
    text = re.sub(r"\s*[\u2010\u2011\u2012\u2013\u2014]\s*", "-", text)
    # Remove stray spaces BEFORE punctuation so "choice !" -> "choice!"
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    # Rejoin numbers broken by spaced commas: "85 , 000" -> "85,000"
    text = re.sub(r"(\d)\s*,\s*(?=\d)", r"\1,", text)
    # Currency: "Rs 85,000" / "Rs. 75000" -> "85000 rupees"
    text = re.sub(
        r"\bRs\.?\s*(\d[\d,]*)",
        lambda m: m.group(1).replace(",", "") + " rupees",
        text,
        flags=re.IGNORECASE,
    )
    # Strip commas inside numbers: "85,000" -> "85000"
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SentenceChunker:
    """Buffers streamed LLM tokens and emits sentence-sized chunks.

    The TTS provider must never receive single characters or lone punctuation --
    otherwise it spells letters ("g r e a t") and reads symbols aloud
    ("!" -> "exclamation mark"). Chunks are emitted at sentence boundaries,
    with a minimum length so audio starts fast but never fragmented.
    """

    MIN_CHARS = 60   # start speaking after ~1 short sentence
    MAX_CHARS = 240  # force a clause break on very long run-ons

    def __init__(self) -> None:
        self._buf = ""

    def add(self, raw_token: str) -> List[str]:
        """Feed a raw streamed token; returns any ready-to-speak chunks."""
        self._buf += raw_token or ""
        return self._drain(force=False)

    def flush(self) -> List[str]:
        """Emit whatever remains (end of the LLM stream)."""
        return self._drain(force=True)

    def _drain(self, force: bool) -> List[str]:
        text = self._buf
        if not text.strip():
            self._buf = ""
            return []

        cut = 0
        # Prefer the last completed sentence at/after MIN_CHARS
        for m in re.finditer(r"[.!?][\"')\]]?(?=\s|$)", text):
            if m.end() >= self.MIN_CHARS:
                cut = m.end()

        if not force and cut == 0:
            # Very long run-on: break at a comma/semicolon to keep chunks bounded
            if len(text) > self.MAX_CHARS:
                pos = max(
                    text.rfind(", ", 0, self.MAX_CHARS),
                    text.rfind("; ", 0, self.MAX_CHARS),
                    text.rfind(" and ", 0, self.MAX_CHARS),
                )
                if pos >= self.MIN_CHARS:
                    cut = pos + len(", ") if text[pos : pos + 2] == ", " else pos + 1
            if cut == 0:
                return []  # keep buffering until a boundary arrives

        if force:
            cut = len(text)

        chunk = sanitize_speech_text(text[:cut])
        self._buf = text[cut:].lstrip()

        # Never emit a fragment without any real content (pure symbols/spaces)
        if chunk and re.search(r"[A-Za-z0-9]", chunk):
            return [chunk]
        return []


class TwilioTurnOrchestrator:
    """Orchestrates Twilio call turns: LISTEN -> THINK -> SPEAK -> mark -> LISTEN, plus instant barge-in."""

    def __init__(
        self,
        session: TwilioCallSession,
        tts_provider: object,
        llm_provider: object,
        send_media_func: Callable[[bytes], Awaitable[None]],
        send_mark_func: Callable[[str], Awaitable[None]],
        send_clear_func: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self.session = session
        self.tts_provider = tts_provider
        self.llm_provider = llm_provider
        self.send_media_func = send_media_func
        self.send_mark_func = send_mark_func
        self.send_clear_func = send_clear_func
        settings = get_settings()
        self.pre_buffer = PreBuffer(target_ms=settings.prebuffer_ms)
        self.barge_in_rms_threshold = settings.barge_in_rms_threshold
        self.barge_in_consecutive_frames = settings.barge_in_consecutive_frames
        self.welcome_greeting = settings.welcome_greeting
        self._current_reply_parts: List[str] = []

        # Hold music -- streamed while the answer is generated (Twilio path only).
        self.hold_music_enabled = settings.hold_music_enabled
        self.hold_music_start_delay_ms = settings.hold_music_start_delay_ms
        self._hold_music_task: Optional[asyncio.Task] = None
        self._hold_music_loop: bytes = b""
        if self.hold_music_enabled:
            try:
                self._hold_music_loop = load_hold_music(settings)
            except Exception as exc:  # noqa: BLE001 - never break calls over music
                logger.error("Failed to load hold music", extra={"error": str(exc)})

    def _start_hold_music(self) -> None:
        """Start streaming the hold-music bed (no-op when disabled/unavailable)."""
        if not self.hold_music_enabled or not self._hold_music_loop:
            return
        if self._hold_music_task is not None and not self._hold_music_task.done():
            return
        self._hold_music_task = asyncio.create_task(self._hold_music_pump())

    async def _stop_hold_music(self) -> None:
        """Stop the hold-music pump and wait, so no music frame is in flight."""
        task = self._hold_music_task
        self._hold_music_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _hold_music_pump(self) -> None:
        """Stream the hold-music loop in ~real time while the answer is generated.

        Frames are paced (about 100 ms each, tiny lookahead) because Twilio plays
        whatever has already arrived -- over-sending would make the music
        unstoppable without also flushing the TTS pre-buffer.
        """
        loop_audio = self._hold_music_loop
        if not loop_audio:
            return
        frame_ms = 0.1
        frame_bytes = int(SAMPLE_RATE * frame_ms)  # 800 mu-law bytes ~= 100 ms
        loop = asyncio.get_running_loop()
        await asyncio.sleep(self.hold_music_start_delay_ms / 1000.0)
        deadline = loop.time() + frame_ms
        offset = 0
        while self.session.state in (CallState.THINKING, CallState.SPEAKING):
            end = offset + frame_bytes
            if end >= len(loop_audio):
                chunk = loop_audio[offset:]  # loop tail, then wrap seamlessly
                offset = 0
            else:
                chunk = loop_audio[offset:end]
                offset = end
            await self.send_media_func(chunk)
            delay = deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            deadline += frame_ms

    async def close(self) -> None:
        """Release background work when the media stream ends."""
        await self._stop_hold_music()

    async def send_welcome_greeting(self, greeting_text: Optional[str] = None) -> None:
        """Play the initial welcome greeting upon stream start."""
        greeting = greeting_text or self.welcome_greeting
        async with self.session.lock:
            if self.session.state != CallState.NEW:
                return
            self.session.transition_to(CallState.WELCOME)

        self.pre_buffer.reset()
        mark_name = f"mark_welcome_{uuid.uuid4().hex[:8]}"
        self.session.pending_mark_name = mark_name
        self.session.pending_assistant_reply = greeting

        async def handle_audio_chunk(chunk: bytes) -> None:
            ready_chunks = self.pre_buffer.push(chunk)
            for ready in ready_chunks:
                await self.send_media_func(ready)

        try:
            async def single_text_iter():
                yield greeting

            response_id = str(uuid.uuid4())
            async for audio in self.tts_provider.synthesize(single_text_iter(), response_id):
                await handle_audio_chunk(audio.data)

            for chunk in self.pre_buffer.flush():
                await self.send_media_func(chunk)

            await self.send_mark_func(mark_name)
            logger.info("Welcome greeting sent, awaiting mark", extra={"mark_name": mark_name})
        except Exception as e:
            logger.error("Error sending welcome greeting", extra={"error": str(e)})
            self.session.transition_to(CallState.LISTENING)

    async def handle_user_transcript(self, transcript: str) -> None:
        """Handle a finalized user transcript from STT (triggers THINK -> SPEAK turn)."""
        if not transcript or not transcript.strip():
            return

        async with self.session.lock:
            if self.session.state not in (CallState.LISTENING, CallState.WELCOME):
                logger.warning(
                    "Ignoring transcript while not in LISTENING state",
                    extra={
                        "state": self.session.state.value,
                        "transcript": transcript,
                    },
                )
                return

            self.session.transition_to(CallState.THINKING)
            self.session.add_user_message(transcript)

        turn = self.session.turn_number
        self._current_reply_parts = []
        self.pre_buffer.reset()

        async def llm_text_stream():
            response_id = str(uuid.uuid4())
            chunker = SentenceChunker()
            # RAG brain comes from the app backend via the LLM provider.
            async for delta in self.llm_provider.stream_response(transcript, response_id):
                if self.session.state != CallState.SPEAKING:
                    return
                if delta.text:
                    self._current_reply_parts.append(delta.text)
                    for piece in chunker.add(delta.text):
                        yield piece
            for piece in chunker.flush():
                yield piece

        async def handle_audio_chunk(chunk: bytes) -> None:
            if self.session.state != CallState.SPEAKING:
                return
            # The first answer byte ends the hold music; awaiting the pump keeps
            # ordering strict so music and speech can never interleave.
            if self._hold_music_task is not None:
                await self._stop_hold_music()
            ready_chunks = self.pre_buffer.push(chunk)
            for ready in ready_chunks:
                await self.send_media_func(ready)

        try:
            async with self.session.lock:
                self.session.transition_to(CallState.SPEAKING)
            self._start_hold_music()

            response_id = str(uuid.uuid4())
            async for audio in self.tts_provider.synthesize(llm_text_stream(), response_id):
                await handle_audio_chunk(audio.data)
            await self._stop_hold_music()

            if self.session.state == CallState.SPEAKING:
                for chunk in self.pre_buffer.flush():
                    await self.send_media_func(chunk)

                full_reply = sanitize_speech_text("".join(self._current_reply_parts))
                self.session.pending_assistant_reply = full_reply
                mark_name = f"mark_turn_{uuid.uuid4().hex[:8]}"
                self.session.pending_mark_name = mark_name
                await self.send_mark_func(mark_name)
                logger.info("Turn TTS completed, sent mark", extra={"mark_name": mark_name, "turn": turn})
        except Exception as e:
            logger.error("Error executing speaking turn", extra={"error": str(e)})
            await self._stop_hold_music()
            self.session.transition_to(CallState.LISTENING)

    async def handle_mark_event(self, mark_name: str) -> None:
        """Process a mark event returned from Twilio Media Stream."""
        if self.session.pending_mark_name and self.session.pending_mark_name == mark_name:
            self.session.pending_mark_name = None
            if self.session.pending_assistant_reply:
                self.session.add_assistant_message(self.session.pending_assistant_reply)
                self.session.pending_assistant_reply = ""

            self.session.turn_number += 1
            self.session.transition_to(CallState.LISTENING)
            logger.info("Mark matched, turn transition to LISTENING complete", extra={"turn": self.session.turn_number})

    async def handle_inbound_audio(self, mulaw_bytes: bytes) -> None:
        """Process an incoming raw mu-law audio frame from the caller (barge-in detection only)."""
        if not mulaw_bytes:
            return

        # Sensitive RMS & framing barge-in detection during SPEAKING / WELCOME state
        if self.session.state in (CallState.SPEAKING, CallState.WELCOME):
            energy = rms(mulaw_bytes)
            if energy >= self.barge_in_rms_threshold:
                self.session.consecutive_barge_in_frames += 1
                if self.session.consecutive_barge_in_frames >= self.barge_in_consecutive_frames:
                    logger.info(
                        "Barge-in triggered by caller speech energy!",
                        extra={"rms": round(energy, 2), "frames": self.session.consecutive_barge_in_frames},
                    )
                    await self.trigger_barge_in()
            else:
                self.session.consecutive_barge_in_frames = 0
        # While hold music plays, caller speech stops the music immediately so
        # the caller never has to talk over it (a full barge-in still only
        # happens in SPEAKING/WELCOME, matching the reference agent).
        elif self.session.state == CallState.THINKING and self._hold_music_task is not None:
            if rms(mulaw_bytes) >= self.barge_in_rms_threshold:
                self.session.consecutive_barge_in_frames += 1
                if self.session.consecutive_barge_in_frames >= self.barge_in_consecutive_frames:
                    logger.info(
                        "Caller speech detected during hold music -- stopping music",
                        extra={"rms": round(rms(mulaw_bytes), 2)},
                    )
                    await self._stop_hold_music()
                    self.session.consecutive_barge_in_frames = 0
            else:
                self.session.consecutive_barge_in_frames = 0

    async def trigger_barge_in(self) -> None:
        """Instantly flush the Twilio audio buffer, abort active TTS, and return to LISTENING."""
        self.session.consecutive_barge_in_frames = 0

        # 1. Instruct Twilio to instantly purge its speaker playback buffer on the phone
        if self.send_clear_func:
            try:
                await self.send_clear_func()
            except Exception as e:
                logger.error("Failed to send clear message on barge-in", extra={"error": str(e)})

        # 2. Abort TTS generation and drain queues
        if hasattr(self.tts_provider, "abort"):
            try:
                await self.tts_provider.abort()
            except Exception as e:
                logger.error("Failed to abort TTS on barge-in", extra={"error": str(e)})
        self.pre_buffer.reset()
        await self._stop_hold_music()

        # 3. Save the partial reply so the next turn still has context (e.g. "what is that?").
        partial = sanitize_speech_text("".join(self._current_reply_parts)) if self._current_reply_parts else ""
        if partial:
            self.session.add_assistant_message(partial)

        self.session.pending_mark_name = None
        self.session.pending_assistant_reply = ""

        # 4. Instantly transition state to LISTENING
        self.session.transition_to(CallState.LISTENING)
