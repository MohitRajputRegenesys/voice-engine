"""Asterisk/3CX call session state machine and active-call registry.

Mirrors ``voice_engine.twilio.session`` so the conversation lifecycle is
identical across telephony backends, but keyed by the ARI channel id of the
call leg and extended with the telephony-side statuses (DIALING / RINGING /
ANSWERED) that ARI channel events provide.
"""
import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional


class CallState(str, Enum):
    NEW = "NEW"
    DIALING = "DIALING"
    RINGING = "RINGING"
    ANSWERED = "ANSWERED"
    WELCOME = "WELCOME"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ENDED = "ENDED"


class AsteriskCallSession:
    """Tracks one call leg (the PJSIP channel to/from 3CX) and its pipeline."""

    def __init__(
        self,
        channel_id: str,
        direction: str,
        phone: str = "",
        caller_id: str = "",
    ) -> None:
        self.channel_id: str = channel_id
        self.direction: str = direction  # "outbound" | "inbound"
        self.phone: str = phone
        self.caller_id: str = caller_id
        self.bridge_id: Optional[str] = None
        self.media_channel_id: Optional[str] = None
        self.created_at: float = time.time()
        self.answered_at: Optional[float] = None
        self.state: CallState = CallState.NEW
        self.final_status: Optional[str] = None  # COMPLETED/FAILED/BUSY/NO_ANSWER
        self.turn_number: int = 0
        self.conversation: List[dict] = []
        self.lock: asyncio.Lock = asyncio.Lock()
        self.consecutive_barge_in_frames: int = 0
        self.conversation_started: bool = False

        # Pipeline objects (wired by AsteriskCallManager._wire_pipeline).
        self.rtp = None
        self.stt_provider = None
        self.stt_queue: Optional[asyncio.Queue] = None
        self.stt_task: Optional[asyncio.Task] = None
        self.orchestrator = None
        self.tasks: List[asyncio.Task] = []
        # Guards against a leg that is never answered: ARI's dial timeout does
        # not always destroy the channel, so the manager enforces its own bound.
        self.watchdog_task: Optional[asyncio.Task] = None

    def transition_to(self, new_state: CallState) -> None:
        self.state = new_state

    def mark_answered(self) -> None:
        """Record the answer time and move to ANSWERED (telephony states only)."""
        if self.answered_at is None:
            self.answered_at = time.time()
        if self.state in (CallState.NEW, CallState.DIALING, CallState.RINGING):
            self.transition_to(CallState.ANSWERED)
        # Answering means the dial watchdog is no longer needed.
        self.cancel_watchdog()

    def cancel_watchdog(self) -> None:
        """Cancel the unanswered-dial watchdog (no-op when not armed)."""
        task = self.watchdog_task
        self.watchdog_task = None
        if task is not None and not task.done():
            task.cancel()

    def add_user_message(self, text: str) -> None:
        self.conversation.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.conversation.append({"role": "assistant", "content": text})

    def duration_sec(self) -> float:
        """Elapsed call duration in seconds (from answer when available)."""
        start = self.answered_at if self.answered_at is not None else self.created_at
        return time.time() - start

    def to_dict(self) -> dict:
        media = None
        if self.rtp is not None:
            stats = getattr(self.rtp, "stats", None)
            if stats is not None:
                # Live proof that audio is flowing in both directions.
                media = dict(stats)
        return {
            "channel_id": self.channel_id,
            "direction": self.direction,
            "phone": self.phone,
            "caller_id": self.caller_id,
            "state": self.state.value,
            "final_status": self.final_status,
            "bridge_id": self.bridge_id,
            "media_channel_id": self.media_channel_id,
            "turn_number": self.turn_number,
            "duration_sec": round(self.duration_sec(), 3),
            "created_at": self.created_at,
            "answered_at": self.answered_at,
            "conversation": list(self.conversation),
            "media": media,
        }


class AsteriskCallRegistry:
    """In-memory registry of active calls, keyed by ARI channel id."""

    def __init__(self) -> None:
        self._sessions: Dict[str, AsteriskCallSession] = {}
        self._media_index: Dict[str, str] = {}  # media channel id -> call channel id
        self._lock: asyncio.Lock = asyncio.Lock()

    async def register_outbound(
        self, channel_id: str, phone: str, caller_id: str = ""
    ) -> AsteriskCallSession:
        async with self._lock:
            session = AsteriskCallSession(
                channel_id=channel_id, direction="outbound", phone=phone, caller_id=caller_id
            )
            self._sessions[channel_id] = session
            return session

    async def register_inbound(
        self, channel_id: str, phone: str = "", caller_id: str = ""
    ) -> AsteriskCallSession:
        async with self._lock:
            session = AsteriskCallSession(
                channel_id=channel_id, direction="inbound", phone=phone, caller_id=caller_id
            )
            self._sessions[channel_id] = session
            return session

    async def get_session(self, channel_id: str) -> Optional[AsteriskCallSession]:
        async with self._lock:
            return self._sessions.get(channel_id)

    async def get_by_media_channel(
        self, media_channel_id: str
    ) -> Optional[AsteriskCallSession]:
        async with self._lock:
            call_channel_id = self._media_index.get(media_channel_id)
            return self._sessions.get(call_channel_id) if call_channel_id else None

    async def link_media_channel(self, channel_id: str, media_channel_id: str) -> None:
        async with self._lock:
            if channel_id in self._sessions:
                self._media_index[media_channel_id] = channel_id

    async def remove_session(self, channel_id: str) -> Optional[AsteriskCallSession]:
        async with self._lock:
            session = self._sessions.pop(channel_id, None)
            if session:
                session.transition_to(CallState.ENDED)
                for media_id, call_id in list(self._media_index.items()):
                    if call_id == channel_id:
                        del self._media_index[media_id]
            return session

    def active_call_count(self) -> int:
        return len(self._sessions)

    def all_sessions(self) -> List[AsteriskCallSession]:
        return list(self._sessions.values())


asterisk_call_registry = AsteriskCallRegistry()
