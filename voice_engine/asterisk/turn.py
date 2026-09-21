"""Turn-taking orchestrator for 3CX/Asterisk calls (listen -> think -> speak).

Functionally mirrors the Twilio orchestrator (same pre-buffering, sentence
chunking, hold music and RMS barge-in) with the transport differences that
ARI + ExternalMedia RTP brings:

* no mark events -- a turn is complete once the paced RTP sender has handed
  every frame to the network (``wait_media_drained_func`` replaces Twilio's
  mark acknowledgement);
* barge-in cannot flush Asterisk's jitter buffer remotely, but dropping the
  locally queued frames and aborting TTS stops the remaining speech within a
  few tens of milliseconds, which callers experience the same way.

All audio in this layer is 8 kHz mu-law; wire-format conversion happens in the
RTP session only.
"""
import asyncio
import inspect
import logging
import uuid
from typing import Awaitable, Callable, List, Optional

from ..twilio.audio import PreBuffer
from ..twilio.hold_music import SAMPLE_RATE, load_hold_music
from ..twilio.turn import SentenceChunker, sanitize_speech_text
from .config import AsteriskSettings, get_settings
from .session import AsteriskCallSession, CallState

logger = logging.getLogger("voice_engine.asterisk.turn")


class AsteriskTurnOrchestrator:
    """Orchestrates 3CX call turns: LISTEN -> THINK -> SPEAK -> drained -> LISTEN."""

    def __init__(
        self,
        session: AsteriskCallSession,
        tts_provider: object,
        llm_provider: object,
        send_media_func: Callable[[bytes], Awaitable[None]],
        wait_media_drained_func: Optional[Callable[..., Awaitable[None]]] = None,
        wait_media_room_func: Optional[Callable[..., Awaitable[None]]] = None,
        clear_media_func: Optional[Callable[[], Awaitable[None]]] = None,
        flush_media_func: Optional[Callable[[], Awaitable[None]]] = None,
        settings: Optional[AsteriskSettings] = None,
    ) -> None:
        self.session = session
        self.tts_provider = tts_provider
        self.llm_provider = llm_provider
        self.send_media_func = send_media_func
        self.wait_media_drained_func = wait_media_drained_func
        self.wait_media_room_func = wait_media_room_func
        self.clear_media_func = clear_media_func
        self.flush_media_func = flush_media_func
        settings = settings or get_settings()
        self.pre_buffer = PreBuffer(target_ms=settings.prebuffer_ms)
        self.barge_in_rms_threshold = settings.barge_in_rms_threshold
        self.barge_in_consecutive_frames = settings.barge_in_consecutive_frames
        self.welcome_greeting = settings.welcome_greeting
        self._current_reply_parts: List[str] = []

        # Hold music -- streamed while the answer is generated (same as Twilio).
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
        """Stream the hold-music loop in ~real time while the answer is generated."""
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

    async def _flush_outbound(self) -> None:
        """Flush the wire-framer tail partial at turn end.

        push() only emits complete ptime frames; the remainder stays in _pending.
        Never flushing it makes the next turn first frame stitch from stale
        mu-law bytes -> Asterisk decodes 0x10 as slin -> full-scale PCM noise
        (the post-greeting noise). Padding with mu-law silence gives a clean
        boundary and empties _pending for the next turn.
        """
        if self.flush_media_func is None:
            return
        try:
            result = self.flush_media_func()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Flushing wire framer failed: %s", exc)

    async def _wait_for_media_drained(self) -> None:
        """Wait until the RTP sender has handed every queued frame to Asterisk."""
        if self.wait_media_drained_func is None:
            return
        try:
            await self.wait_media_drained_func()
        except Exception as exc:  # noqa: BLE001 - draining must never kill a turn
            logger.warning("Waiting for media drain failed: %s", exc)

    async def close(self) -> None:
        """Release background work when the call ends."""
        await self._stop_hold_music()

    async def send_welcome_greeting(self, greeting_text: Optional[str] = None) -> None:
        """Play the initial welcome greeting when the call is answered.

        Reached from ``NEW`` (inbound: the channel enters Stasis before audio),
        but on an outbound leg ``mark_answered()`` has already moved the session
        to ``ANSWERED``/``RINGING`` before the media bridge exists. Guarding on
        ``NEW`` alone silently skipped the greeting on every outbound 3CX call
        (no TTS, ``turn_number`` stuck at 0, state stuck at ANSWERED).
        """
        greeting = greeting_text or self.welcome_greeting
        logger.info(
            "Welcome greeting starting",
            extra={
                "channel_id": self.session.channel_id,
                "text_length": len(greeting or ""),
                "state": self.session.state.value,
            },
        )
        if not greeting:
            # Fail loudly instead of sending an empty/None frame over the media
            # wire: a None greeting previously produced nothing the caller could
            # recognize as speech (and an empty TTS request can yield silence or
            # garbage audio). The greeting is configured via ASTERISK_WELCOME_
            # GREETING; if it is missing we abort here and let the normal
            # LISTEN->STT flow handle the call instead.
            logger.error(
                "Cannot play welcome greeting: welcome_greeting is not configured "
                "(set ASTERISK_WELCOME_GREETING or pass greeting_text); no TTS "
                "request sent, media wire left untouched."
            )
            return
        async with self.session.lock:
            if self.session.state not in (
                CallState.NEW,
                CallState.DIALING,
                CallState.RINGING,
                CallState.ANSWERED,
            ):
                return
            self.session.transition_to(CallState.WELCOME)

        self.pre_buffer.reset()
        try:
            async def single_text_iter():
                yield greeting

            response_id = str(uuid.uuid4())
            async for audio in self.tts_provider.synthesize(single_text_iter(), response_id):
                logger.info(
                    "Welcome TTS audio chunk received",
                    extra={
                        "channel_id": self.session.channel_id,
                        "bytes": len(audio.data),
                        "response_id": response_id,
                    },
                )
                for ready in self.pre_buffer.push(audio.data):
                    await self.send_media_func(ready)

            for chunk in self.pre_buffer.flush():
                await self.send_media_func(chunk)
            await self._flush_outbound()

            await self._wait_for_media_drained()
            if self.session.state == CallState.WELCOME:
                self.session.add_assistant_message(sanitize_speech_text(greeting))
                self.session.turn_number += 1
                self.session.transition_to(CallState.LISTENING)
                logger.info(
                    "Welcome greeting played, now LISTENING",
                    extra={
                        "channel_id": self.session.channel_id,
                        "turn": self.session.turn_number,
                    },
                )
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "Error sending welcome greeting",
                extra={"channel_id": self.session.channel_id, "error": str(e)},
            )
            if self.session.state == CallState.WELCOME:
                self.session.transition_to(CallState.LISTENING)

    async def handle_user_transcript(self, transcript: str) -> None:
        """Handle a finalized user transcript from STT (triggers THINK -> SPEAK turn)."""
        if not transcript or not transcript.strip():
            logger.warning(
                "Ignoring empty STT transcript",
                extra={"channel_id": self.session.channel_id},
            )
            return

        async with self.session.lock:
            if self.session.state not in (CallState.LISTENING, CallState.WELCOME):
                logger.warning(
                    "Ignoring transcript while not in LISTENING state",
                    extra={
                        "channel_id": self.session.channel_id,
                        "state": self.session.state.value,
                        "text_length": len(transcript),
                    },
                )
                return
            self.session.transition_to(CallState.THINKING)
            self.session.add_user_message(transcript)

        logger.info(
            "User transcript accepted; starting response turn",
            extra={
                "channel_id": self.session.channel_id,
                "text_length": len(transcript),
                "turn": self.session.turn_number + 1,
            },
        )
        logger.info(
            "CALLER TRANSCRIPT: %s",
            transcript,
            extra={
                "channel_id": self.session.channel_id,
                "turn": self.session.turn_number + 1,
                "state": self.session.state.value,
            },
        )

        turn = self.session.turn_number
        self._current_reply_parts = []
        self.pre_buffer.reset()

        async def llm_text_stream():
            response_id = str(uuid.uuid4())
            chunker = SentenceChunker()
            delta_count = 0
            # RAG brain comes from the app backend via the LLM provider.
            async for delta in self.llm_provider.stream_response(transcript, response_id):
                delta_count += 1
                if delta_count == 1:
                    logger.info(
                        "LLM first delta received",
                        extra={"channel_id": self.session.channel_id, "turn": turn},
                    )
                if self.session.state != CallState.SPEAKING:
                    logger.warning(
                        "Stopping LLM stream because call state changed",
                        extra={
                            "channel_id": self.session.channel_id,
                            "turn": turn,
                            "state": self.session.state.value,
                        },
                    )
                    return
                if delta.text:
                    self._current_reply_parts.append(delta.text)
                    for piece in chunker.add(delta.text):
                        yield piece
            for piece in chunker.flush():
                yield piece
            logger.info(
                "LLM stream completed",
                extra={
                    "channel_id": self.session.channel_id,
                    "turn": turn,
                    "delta_count": delta_count,
                    "text_length": len("".join(self._current_reply_parts)),
                },
            )

        async def handle_audio_chunk(chunk: bytes) -> None:
            if self.session.state != CallState.SPEAKING:
                return
            # The first answer byte ends the hold music; awaiting the pump keeps
            # ordering strict so music and speech can never interleave. Clear
            # frames already queued by the hold pump before enqueueing speech.
            if self._hold_music_task is not None:
                await self._stop_hold_music()
                if self.clear_media_func is not None:
                    try:
                        result = self.clear_media_func()
                        if inspect.isawaitable(result):
                            await result
                        logger.info(
                            "Hold music stopped and queued media cleared before TTS",
                            extra={
                                "channel_id": self.session.channel_id,
                                "turn": turn,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Could not clear queued hold music before TTS: %s",
                            exc,
                            extra={
                                "channel_id": self.session.channel_id,
                                "turn": turn,
                            },
                        )
            # AudioSocket can expose a small bounded lookahead helper. Wait
            # until the sender is close to real time before adding speech.
            wait_for_room = self.wait_media_room_func
            if wait_for_room is not None:
                try:
                    await wait_for_room(2)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not pace speech handoff: %s", exc)
            for ready in self.pre_buffer.push(chunk):
                await self.send_media_func(ready)

        try:
            async with self.session.lock:
                self.session.transition_to(CallState.SPEAKING)
            self._start_hold_music()

            response_id = str(uuid.uuid4())
            logger.info(
                "TTS response starting",
                extra={"channel_id": self.session.channel_id, "turn": turn},
            )
            tts_chunks = 0
            async for audio in self.tts_provider.synthesize(llm_text_stream(), response_id):
                tts_chunks += 1
                logger.info(
                    "TTS audio chunk received",
                    extra={
                        "channel_id": self.session.channel_id,
                        "turn": turn,
                        "chunk": tts_chunks,
                        "bytes": len(audio.data),
                    },
                )
                await handle_audio_chunk(audio.data)
            await self._stop_hold_music()

            if self.session.state == CallState.SPEAKING:
                for chunk in self.pre_buffer.flush():
                    await self.send_media_func(chunk)
                await self._flush_outbound()

                full_reply = sanitize_speech_text("".join(self._current_reply_parts))
                logger.info(
                    "AI WILL SAY: %s",
                    full_reply or "<empty response>",
                    extra={
                        "channel_id": self.session.channel_id,
                        "turn": turn,
                        "state": self.session.state.value,
                    },
                )
                # No marks in Asterisk: the turn is done once the paced RTP
                # sender has handed every frame to the network.
                await self._wait_for_media_drained()
                self.session.add_assistant_message(full_reply)
                self.session.turn_number += 1
                self.session.transition_to(CallState.LISTENING)
                logger.info(
                    "Turn complete (media drained)",
                    extra={
                        "channel_id": self.session.channel_id,
                        "turn": turn,
                        "tts_chunks": tts_chunks,
                        "reply_length": len(full_reply),
                    },
                )
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "Error executing speaking turn",
                extra={
                    "channel_id": self.session.channel_id,
                    "turn": turn,
                    "error": str(e),
                },
            )
            await self._stop_hold_music()
            self.session.transition_to(CallState.LISTENING)

    async def handle_inbound_audio(self, mulaw_bytes: bytes) -> None:
        """Process an incoming mu-law frame from the caller (barge-in detection)."""
        if not mulaw_bytes:
            return

        from ..twilio.audio import rms  # local import keeps the hot path lean

        # Sensitive RMS & framing barge-in detection during SPEAKING / WELCOME
        if self.session.state in (CallState.SPEAKING, CallState.WELCOME):
            energy = rms(mulaw_bytes)
            if energy >= self.barge_in_rms_threshold:
                self.session.consecutive_barge_in_frames += 1
                if self.session.consecutive_barge_in_frames >= self.barge_in_consecutive_frames:
                    logger.info(
                        "Barge-in triggered by caller speech energy!",
                        extra={
                            "channel_id": self.session.channel_id,
                            "state": self.session.state.value,
                            "rms": round(energy, 2),
                            "frames": self.session.consecutive_barge_in_frames,
                        },
                    )
                    await self.trigger_barge_in()
            else:
                self.session.consecutive_barge_in_frames = 0
        # While hold music plays, caller speech stops the music immediately
        # (a full barge-in still only happens in SPEAKING/WELCOME).
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
        """Stop outgoing speech instantly and return to LISTENING."""
        self.session.consecutive_barge_in_frames = 0

        # 1. Drop every locally queued (not yet sent) frame -- the closest
        #    equivalent of Twilio's "clear" message for the RTP/AudioSocket path.
        #    ``clear()`` is synchronous on both transports, so the result is only
        #    awaited when it is awaitable (awaiting a non-awaitable raised
        #    TypeError, which left the queued speech playing through barge-in).
        if self.clear_media_func:
            try:
                result = self.clear_media_func()
                if inspect.isawaitable(result):
                    await result
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to clear media on barge-in", extra={"error": str(e)})

        # 2. Abort TTS generation and drain the pre-buffer.
        if hasattr(self.tts_provider, "abort"):
            try:
                await self.tts_provider.abort()
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to abort TTS on barge-in", extra={"error": str(e)})
        self.pre_buffer.reset()
        await self._stop_hold_music()

        # 3. Save the partial reply so the next turn still has context
        #    (e.g. "what is that?").
        partial = (
            sanitize_speech_text("".join(self._current_reply_parts))
            if self._current_reply_parts
            else ""
        )
        if partial:
            self.session.add_assistant_message(partial)
        self.session.pending_assistant_reply = ""

        # 4. Instantly transition state to LISTENING.
        self.session.transition_to(CallState.LISTENING)


