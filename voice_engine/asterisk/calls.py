"""Call orchestration for 3CX calls via Asterisk ARI (the app-control layer).

Outbound lifecycle::

    POST /api/threecx/calls/outbound {"phone": "+919876543210"}
      -> ARI create channel  (endpoint PJSIP/3cx/<digits>, in the Stasis app)
      -> ARI dial            (Asterisk INVITEs 3CX as extension 900;
                              3CX outbound rule -> Airtel -> customer phone)
      -> ChannelStateChange "Ring"            -> RINGING
      -> ChannelStateChange "Up"              -> ANSWERED
      -> mixing bridge + externalMedia channel (RTP to this engine)
      -> welcome greeting + STT -> RAG LLM -> TTS turn loop over RTP

Inbound lifecycle::

    3CX routes a call to extension 900 -> Asterisk dialplan runs
    ``Stasis(voice-engine,inbound)`` -> answer -> same media pipeline.

Teardown on StasisEnd / ChannelDestroyed: bridge destroyed, external media
channel hung up, RTP socket closed, STT task cancelled, registry cleaned.
"""
import asyncio
import logging
import os
import uuid
from typing import Optional

from ..providers import (
    DeepgramSTTProvider,
    DeepgramStreamingTTSProvider,
    FacilitatorLLMProvider,
    MockSTTProvider,
    MockTTSProvider,
)
from .ari import AriClient, AriError
from .audiosocket import AudioSocketSession
from .config import get_settings, normalize_outbound_number
from .rtp import RtpSession
from .session import AsteriskCallRegistry, AsteriskCallSession, CallState
from .turn import AsteriskTurnOrchestrator

logger = logging.getLogger("voice_engine.asterisk.calls")

# Q.850 hangup cause -> call status (for unanswered outbound attempts).
_HANGUP_CAUSE_STATUS = {
    16: "COMPLETED",  # normal clearing
    17: "BUSY",       # user busy
    18: "NO_ANSWER",  # no user responding
    19: "NO_ANSWER",  # no answer from the user
    21: "FAILED",     # call rejected
    27: "FAILED",     # destination out of order
    34: "FAILED",     # no circuit/channel available
}


def build_asterisk_stt_provider():
    """STT tuned for the 3CX phone path (8 kHz mu-law, like the Twilio path)."""
    if os.environ.get("DEEPGRAM_API_KEY"):
        return DeepgramSTTProvider(
            model=(
                os.environ.get("ASTERISK_STT_MODEL")
                or os.environ.get("STT_MODEL")
                or "nova-3"
            ),
            language=os.environ.get("ASTERISK_STT_LANGUAGE") or None,
            sample_rate=8000,
            encoding="mulaw",
            channels=1,
            interim_results=True,
            endpointing_ms=int(os.environ.get("ASTERISK_SILENCE_ENDPOINT_MS", "800") or 800),
            utterance_end_ms=int(os.environ.get("ASTERISK_UTTERANCE_END_MS", "1500") or 1500),
            vad_events=True,
        )
    return MockSTTProvider()


def build_asterisk_tts_provider():
    """TTS tuned for the 3CX phone path (8 kHz mu-law streaming Aura)."""
    if os.environ.get("DEEPGRAM_API_KEY"):
        return DeepgramStreamingTTSProvider(
            model=os.environ.get("TTS_MODEL") or None,
            encoding="mulaw",
            sample_rate=8000,
        )
    return MockTTSProvider()

def build_call_providers():
    """Build the STT/TTS/LLM providers for one 3CX call.

    Tests monkeypatch this function to inject fakes. The LLM brain comes from
    the saleKnowledgeBase app backend RAG API (same as the Twilio path).
    """
    stt_provider = build_asterisk_stt_provider()
    tts_provider = build_asterisk_tts_provider()
    llm_provider = FacilitatorLLMProvider(
        base_url=os.environ.get("FACILITATOR_API_URL", "http://localhost:8000"),
        api_path=os.environ.get("FACILITATOR_API_PATH", "/api/v1/rag/chat"),
        top_k=int(os.environ.get("FACILITATOR_TOP_K", "5")),
        alpha=float(os.environ.get("FACILITATOR_ALPHA", "0.5")),
    )
    return stt_provider, tts_provider, llm_provider


class AsteriskCallManager:
    """Owns the ARI connection, the call registry and per-call media pipelines."""

    def __init__(self) -> None:
        self.registry = AsteriskCallRegistry()
        self.ari: Optional[AriClient] = None
        self._media_channel_ids: set = set()

    # ------------------------------------------------------------ lifecycle

    def is_configured(self) -> bool:
        s = get_settings()
        return bool(
            s.enabled
            and s.ari_base_url
            and s.ari_username
            and s.ari_password
            and s.ari_app
        )

    async def start(self) -> bool:
        """Connect to ARI and start event processing.

        Returns False (instead of raising) when disabled or not configured so
        the gateway startup never fails because Asterisk is missing.
        """
        if not self.is_configured():
            logger.warning(
                "Asterisk/3CX integration disabled or incomplete -- set "
                "ASTERISK_ENABLED=true and the ARI_* environment variables"
            )
            return False
        if self.ari is None:
            s = get_settings()
            client = AriClient(
                base_url=s.ari_base_url,
                username=s.ari_username,
                password=s.ari_password,
                app=s.ari_app,
            )
            self._register_handlers(client)
            self.ari = client
            logger.info(
                "Asterisk/3CX integration starting (app=%s, ari=%s)",
                s.ari_app,
                s.ari_base_url,
            )
        await self.ari.start_events()
        return True

    async def stop(self) -> None:
        if self.ari is not None:
            await self.ari.close()

    # ------------------------------------------------------------ originate

    async def originate_call(self, phone: str, caller_id: str = "") -> AsteriskCallSession:
        """Dial ``phone`` through Asterisk -> 3CX (extension 900) -> Airtel."""
        if self.ari is None:
            raise RuntimeError("Asterisk integration is not running (ASTERISK_ENABLED=false?)")
        s = get_settings()
        digits = normalize_outbound_number(phone, s.outbound_prefix)
        channel_id = str(uuid.uuid4())
        session = await self.registry.register_outbound(
            channel_id=channel_id, phone=digits, caller_id=caller_id or s.caller_id
        )
        session.transition_to(CallState.DIALING)
        endpoint = f"PJSIP/{digits}@{s.threecx_pjsip_endpoint}"  # at-form: the proven dialstring shape for 3CX
        app_args = f"direction=outbound,phone={digits}"
        try:
            await self.ari.create_channel(
                endpoint=endpoint,
                app_args=app_args,
                channel_id=channel_id,
                caller_id=session.caller_id or None,
            )
            await self.ari.dial(channel_id, timeout=s.channel_timeout)
        except AriError:
            await self.registry.remove_session(channel_id)
            raise
        logger.info(
            "Outbound 3CX call initiated",
            extra={"channel_id": channel_id, "phone": digits, "endpoint": endpoint},
        )
        self._arm_dial_watchdog(session, s.channel_timeout)
        return session

    def _arm_dial_watchdog(self, session: AsteriskCallSession, timeout: int) -> None:
        """Bound how long a call may sit unanswered.

        ARI's ``dial`` timeout does not reliably destroy the channel (observed on
        Asterisk 20.21.0: a leg stayed held in Stasis after the dial timeout), so
        an unanswered call would keep its session in DIALING forever and never be
        cleaned up. The watchdog hangs the leg up and tears the session down,
        reporting ``NO_ANSWER``.
        """
        grace = max(int(timeout or 0), 1) + 5
        session.cancel_watchdog()
        session.watchdog_task = asyncio.create_task(
            self._dial_watchdog(session, grace)
        )

    async def _dial_watchdog(self, session: AsteriskCallSession, grace: int) -> None:
        try:
            await asyncio.sleep(grace)
            if session.state not in (
                CallState.NEW,
                CallState.DIALING,
                CallState.RINGING,
            ):
                return  # answered (and possibly already finished) in time
            logger.warning(
                "Call not answered within %ss, hanging up",
                grace,
                extra={"channel_id": session.channel_id, "phone": session.phone},
            )
            session.final_status = "NO_ANSWER"
            if self.ari is not None:
                try:
                    await self.ari.hangup(session.channel_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Watchdog hangup failed: %s",
                        exc,
                        extra={"channel_id": session.channel_id},
                    )
            await self._teardown(session.channel_id)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - a watchdog must never crash
            logger.error(
                "Dial watchdog failed: %s", exc, extra={"channel_id": session.channel_id}
            )

    async def hangup(self, channel_id: str) -> bool:
        """Hang up an active call (best effort)."""
        if self.ari is None:
            return False
        session = await self.registry.get_session(channel_id)
        if session is None:
            return False
        await self.ari.hangup(channel_id)
        return True

    # ------------------------------------------------------- ARI event wiring

    def _register_handlers(self, client: AriClient) -> None:
        client.on("StasisStart", self._on_stasis_start)
        client.on("StasisEnd", self._on_stasis_end)
        client.on("ChannelStateChange", self._on_channel_state_change)
        client.on("ChannelDestroyed", self._on_channel_destroyed)
        client.on("ChannelDtmfReceived", self._on_dtmf_received)

    @staticmethod
    def _parse_app_args(args) -> dict:
        """Parse ARI appArgs into a dict.

        Accepts ``["direction=outbound", "phone=123"]`` lists, comma-joined
        strings (``"direction=outbound,phone=123"``) and bare direction tokens
        (``["inbound"]`` from the dialplan's ``Stasis(voice-engine,inbound)``).
        """
        parsed: dict = {}
        for raw in args or []:
            for part in str(raw).split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" in part:
                    key, value = part.split("=", 1)
                    parsed[key.strip()] = value.strip()
                else:
                    parsed.setdefault("direction", part)
        return parsed

    def _is_media_channel(self, channel: dict) -> bool:
        channel_id = channel.get("id", "")
        name = channel.get("name", "")
        # RTP legs are "UnicastRTP/..."; AudioSocket legs are
        # "AudioSocket/host:port-<uuid>". Both belong to us, never a call leg.
        return (
            channel_id in self._media_channel_ids
            or name.startswith("UnicastRTP")
            or name.startswith("AudioSocket")
        )

    async def _on_stasis_start(self, event: dict) -> None:
        channel = event.get("channel") or {}
        channel_id = channel.get("id")
        if not channel_id:
            return
        if self._is_media_channel(channel):
            return  # our own externalMedia channel, not a call leg

        parsed = self._parse_app_args(event.get("args"))
        direction = parsed.get("direction", "inbound")
        caller = channel.get("caller") or {}
        session = await self.registry.get_session(channel_id)
        if session is None:
            session = await self.registry.register_inbound(
                channel_id=channel_id,
                phone=caller.get("number", "") or "",
                caller_id=caller.get("name", "") or "",
            )
        if parsed.get("phone"):
            session.phone = parsed["phone"]
        session.direction = direction
        logger.info(
            "StasisStart",
            extra={
                "channel_id": channel_id,
                "direction": direction,
                "phone": session.phone,
                "state": session.state.value,
            },
        )
        if direction == "inbound":
            await self._start_conversation(session)
        # outbound: the conversation starts when the channel is answered (Up).

    async def _on_channel_state_change(self, event: dict) -> None:
        channel = event.get("channel") or {}
        channel_id = channel.get("id")
        state = channel.get("state")
        if not channel_id or self._is_media_channel(channel):
            return
        session = await self.registry.get_session(channel_id)
        if session is None:
            return
        if state == "Ring":
            if session.state in (CallState.NEW, CallState.DIALING):
                session.transition_to(CallState.RINGING)
                logger.info("Call ringing", extra={"channel_id": channel_id})
        elif state == "Up":
            session.mark_answered()
            logger.info("Call answered", extra={"channel_id": channel_id})
            if session.direction == "outbound":
                await self._start_conversation(session)

    async def _on_stasis_end(self, event: dict) -> None:
        channel = event.get("channel") or {}
        channel_id = channel.get("id")
        if not channel_id:
            return
        if self._is_media_channel(channel):
            self._media_channel_ids.discard(channel_id)
            return
        await self._teardown(channel_id)

    async def _on_channel_destroyed(self, event: dict) -> None:
        channel = event.get("channel") or {}
        channel_id = channel.get("id")
        if not channel_id:
            return
        if self._is_media_channel(channel):
            self._media_channel_ids.discard(channel_id)
            return
        cause = channel.get("cause")
        if cause is not None:
            try:
                cause = int(cause)
            except (TypeError, ValueError):
                cause = None
        session = await self.registry.get_session(channel_id)
        if session is not None and cause is not None:
            status = _HANGUP_CAUSE_STATUS.get(cause)
            if status:
                session.final_status = status if session.answered_at else (
                    status if status != "COMPLETED" else "FAILED"
                )
        await self._teardown(channel_id)

    async def _on_dtmf_received(self, event: dict) -> None:
        channel = event.get("channel") or {}
        session = await self.registry.get_session(channel.get("id", ""))
        if session is None or self._is_media_channel(channel):
            return
        digit = event.get("digit", "")
        logger.info(
            "DTMF received",
            extra={"channel_id": channel.get("id"), "digit": digit},
        )

    # ------------------------------------------------- conversation pipeline

    async def _create_rtp_session(self, s):
        """Create the media session for a call (monkeypatched in tests).

        Transport is selected by ``ASTERISK_MEDIA_TRANSPORT``:
        * ``audiosocket`` (default) -- TCP AudioSocket, the only encapsulation
          Asterisk's externalMedia implements on 18/20/21;
        * ``rtp`` -- UDP RTP, for builds/custom versions that support it.
        Both expose the same interface, so the rest of the pipeline is agnostic.
        """
        remote_addr = None
        if s.media_remote_addr and s.media_remote_port:
            remote_addr = (s.media_remote_addr, s.media_remote_port)
        if getattr(s, "media_transport", "audiosocket") == "rtp":
            session = RtpSession(
                bind_ip=s.media_bind_ip,
                bind_port=s.media_port,
                fmt=s.media_format,
                frame_ms=s.frame_ms,
                remote_addr=remote_addr,
            )
        else:
            session = AudioSocketSession(
                bind_ip=s.media_bind_ip,
                bind_port=s.media_port,
                fmt=s.media_format,
                frame_ms=s.frame_ms,
            )
        await session.start()
        return session

    async def _start_conversation(self, session: AsteriskCallSession) -> None:
        """Answer (inbound), bridge the call to an ExternalMedia channel, run AI."""
        if session.conversation_started:
            return
        session.conversation_started = True
        s = get_settings()
        try:
            if session.direction == "inbound" and self.ari is not None:
                # Inbound legs arrive from the 3CX dialplan unanswered.
                await self.ari.answer(session.channel_id)

            bridge = await self.ari.create_bridge(
                name=f"voice-engine-{session.channel_id[:8]}"
            )
            session.bridge_id = bridge.get("id")

            # RTP socket first: the externalMedia channel needs our address.
            rtp = await self._create_rtp_session(s)
            session.rtp = rtp

            media_channel_id = str(uuid.uuid4())
            self._media_channel_ids.add(media_channel_id)
            if getattr(s, "media_transport", "audiosocket") == "audiosocket":
                # AudioSocket over TCP: the only transport combination Asterisk
                # 18/20/21 externalMedia actually implements (UDP -> HTTP 501).
                await self.ari.create_external_media(
                    external_host=f"{s.media_host}:{rtp.bound_port}",
                    # Asterisk requires the external-media channel to be
                    # negotiated as ulaw for bidirectional bridge audio. The
                    # AudioSocket transport still receives the resulting
                    # 0x10 frames as slin16 and decodes them at its boundary.
                    fmt=s.media_format,
                    channel_id=media_channel_id,
                    encapsulation="AUDIOSOCKET",
                    transport="TCP",
                    data=media_channel_id,
                )
            else:
                await self.ari.create_external_media(
                    external_host=f"{s.media_host}:{rtp.bound_port}",
                    fmt=s.media_format,
                    channel_id=media_channel_id,
                    encapsulation="NONE",
                    transport="UDP",
                    connection_type="client",
                )
            session.media_channel_id = media_channel_id
            await self.registry.link_media_channel(session.channel_id, media_channel_id)

            await self.ari.add_channel_to_bridge(session.bridge_id, session.channel_id)
            await self.ari.add_channel_to_bridge(session.bridge_id, media_channel_id)

            self._wire_pipeline(session, rtp, s)
            logger.info(
                "3CX conversation started",
                extra={
                    "channel_id": session.channel_id,
                    "bridge_id": session.bridge_id,
                    "media_channel_id": media_channel_id,
                    "rtp": f"{s.media_host}:{rtp.bound_port}",
                    "format": s.media_format,
                },
            )
        except Exception as exc:  # noqa: BLE001 - fail the call, not the gateway
            logger.error(
                "Failed to start conversation",
                extra={"channel_id": session.channel_id, "error": str(exc)},
            )
            session.conversation_started = False
            await self._teardown(session.channel_id)

    def _wire_pipeline(self, session: AsteriskCallSession, rtp: RtpSession, s) -> None:
        """Wire STT -> RAG LLM -> TTS between the RTP leg and the conversation."""
        stt_provider, tts_provider, llm_provider = build_call_providers()
        stt_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        # Provider loggers otherwise have no knowledge of the telephony leg,
        # which makes simultaneous calls impossible to distinguish in logs.
        if hasattr(stt_provider, "logger"):
            stt_provider.logger = logging.LoggerAdapter(
                stt_provider.logger,
                {"channel_id": session.channel_id},
            )
        if hasattr(llm_provider, "logger"):
            llm_provider.logger = logging.LoggerAdapter(
                llm_provider.logger,
                {"channel_id": session.channel_id},
            )
        logger.info(
            "Call pipeline providers ready",
            extra={
                "channel_id": session.channel_id,
                "stt": type(stt_provider).__name__,
                "tts": type(tts_provider).__name__,
                "llm": type(llm_provider).__name__,
                "queue_limit": stt_queue.maxsize,
            },
        )

        orchestrator = AsteriskTurnOrchestrator(
            session=session,
            tts_provider=tts_provider,
            llm_provider=llm_provider,
            send_media_func=rtp.queue_mulaw,
                flush_media_func=getattr(rtp, "flush_mulaw", None),
            wait_media_drained_func=rtp.drain,
                wait_media_room_func=getattr(rtp, "wait_for_queue_room", None),
            clear_media_func=rtp.clear,
            settings=s,
        )
        session.orchestrator = orchestrator
        session.stt_provider = stt_provider
        session.stt_queue = stt_queue

        async def enqueue_stt_audio(audio: bytes) -> None:
            try:
                stt_queue.put_nowait(audio)
                if session.rtp is not None:
                    rx_packets = getattr(session.rtp, "stats", {}).get("rx_packets", 0)
                    if rx_packets == 1 or rx_packets % 100 == 0:
                        logger.info(
                            "Call inbound audio queued",
                            extra={
                                "channel_id": session.channel_id,
                                "rx_packets": rx_packets,
                                "queue_depth": stt_queue.qsize(),
                                "frame_bytes": len(audio),
                                "state": session.state.value,
                            },
                        )
            except asyncio.QueueFull:
                # drop the oldest frame to keep the pipeline moving
                try:
                    stt_queue.get_nowait()
                    stt_queue.put_nowait(audio)
                    logger.warning(
                        "STT queue full; dropped oldest audio frame",
                        extra={
                            "channel_id": session.channel_id,
                            "queue_depth": stt_queue.qsize(),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass

        async def on_mulaw_frame(mulaw: bytes) -> None:
            await enqueue_stt_audio(mulaw)
            await orchestrator.handle_inbound_audio(mulaw)

        rtp.on_frame = on_mulaw_frame

        async def stt_consumer():
            logger.info(
                "STT consumer started",
                extra={
                    "channel_id": session.channel_id,
                    "provider": type(stt_provider).__name__,
                },
            )
            try:
                async for stt_event in stt_provider.consume_audio(stt_queue):
                    logger.info(
                        "STT event received",
                        extra={
                            "channel_id": session.channel_id,
                            "sequence": stt_event.sequence,
                            "partial": bool(stt_event.partial),
                            "final": bool(stt_event.final),
                            "text_length": len(stt_event.final or stt_event.partial or ""),
                            "state": session.state.value,
                            "queue_depth": stt_queue.qsize(),
                        },
                    )
                    if stt_event.final:
                        logger.info(
                            "Dispatching final transcript to turn orchestrator",
                            extra={
                                "channel_id": session.channel_id,
                                "text_length": len(stt_event.final),
                                "state": session.state.value,
                            },
                        )
                        logger.info(
                            "CALLER SAID: %s",
                            stt_event.final,
                            extra={
                                "channel_id": session.channel_id,
                                "state": session.state.value,
                            },
                        )
                        await orchestrator.handle_user_transcript(stt_event.final)
                logger.warning(
                    "STT consumer ended normally",
                    extra={"channel_id": session.channel_id},
                )
            except asyncio.CancelledError:
                logger.info(
                    "STT consumer cancelled",
                    extra={"channel_id": session.channel_id},
                )
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "STT consumer error",
                    extra={"channel_id": session.channel_id, "error": str(exc)},
                )

        session.stt_task = asyncio.create_task(stt_consumer())
        session.tasks.append(asyncio.create_task(orchestrator.send_welcome_greeting()))

    async def _teardown(self, channel_id: str) -> None:
        session = await self.registry.remove_session(channel_id)
        if session is None:
            return
        logger.info(
            "Call ended",
            extra={
                "channel_id": channel_id,
                "phone": session.phone,
                "duration_sec": round(session.duration_sec(), 1),
                "final_status": session.final_status,
            },
        )
        # -- local pipeline cleanup (never raises) ---------------------------
        session.cancel_watchdog()
        if session.watchdog_task is not None:
            try:
                await session.watchdog_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            session.watchdog_task = None
        if session.stt_task is not None:
            session.stt_task.cancel()
            try:
                await session.stt_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            session.stt_task = None
        if session.stt_queue is not None:
            try:
                session.stt_queue.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass
        if session.orchestrator is not None:
            try:
                await session.orchestrator.close()
            except Exception:  # noqa: BLE001
                pass
            session.orchestrator = None
        for task in session.tasks:
            task.cancel()
        for task in session.tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - teardown must never raise
                # Log it: a swallowed task exception is how "greeting never
                # played" stayed invisible on a live call.
                logger.error(
                    "Call pipeline task failed during teardown",
                    extra={"channel_id": channel_id, "error": str(exc)},
                )
        session.tasks.clear()
        # -- Asterisk cleanup (best effort; call leg is usually gone already) --
        # Hang up the media leg BEFORE the bridge AND before closing the local
        # transport: the AudioSocket channel is torn down through its TCP leg, so
        # closing our socket first makes Asterisk fail the DELETE and leaves the
        # channel orphaned in Stasis on every call (observed leak, Asterisk
        # 20.21.0). It is not destroyed by the call leg's hangup either.
        if self.ari is not None:
            if session.media_channel_id:
                self._media_channel_ids.discard(session.media_channel_id)
                await self._best_effort_hangup(
                    session.media_channel_id, session.channel_id
                )
            if session.bridge_id:
                try:
                    await self.ari.destroy_bridge(session.bridge_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to destroy bridge %s: %s",
                        session.bridge_id,
                        exc,
                        extra={"channel_id": channel_id},
                    )
        # -- local media transport last: its socket is what keeps the Asterisk
        #    AudioSocket channel alive, so it must outlive the hangup above. ----
        if session.rtp is not None:
            try:
                await session.rtp.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to close media transport: %s",
                    exc,
                    extra={"channel_id": channel_id},
                )
            session.rtp = None

    async def _best_effort_hangup(self, target: str, channel_id: str) -> None:
        """Hang up a channel, warning instead of failing silently.

        A channel that was already destroyed answers 404, which is expected and
        not logged; any other failure means a real leak, so it is reported.
        """
        if self.ari is None:
            return
        try:
            await self.ari.hangup(target)
        except AriError as exc:
            status = getattr(exc, "status_code", None)
            if status not in (404, 422):
                logger.warning(
                    "Failed to hang up channel %s (status=%s): %s",
                    target,
                    status,
                    exc,
                    extra={"channel_id": channel_id},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to hang up channel %s: %s",
                target,
                exc,
                extra={"channel_id": channel_id},
            )

    # ------------------------------------------------------------- reporting

    def get_status(self) -> dict:
        s = get_settings()
        return {
            "enabled": s.enabled,
            "configured": self.is_configured(),
            "connected": bool(self.ari is not None and self.ari.connected.is_set()),
            "app": s.ari_app,
            "ari_base_url": s.ari_base_url,
            "threecx_extension": s.threecx_extension,
            "threecx_pjsip_endpoint": s.threecx_pjsip_endpoint,
            "media_format": s.media_format,
            "active_call_count": self.registry.active_call_count(),
            "active_calls": [session.to_dict() for session in self.registry.all_sessions()],
        }


asterisk_call_manager = AsteriskCallManager()




