"""Twilio call session state machine and active-call registry."""
import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional


class CallState(str, Enum):
    NEW = "NEW"
    WELCOME = "WELCOME"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ENDED = "ENDED"


class TwilioCallSession:
    """Manages call state, turn lock, conversation history, and lifecycle metrics."""

    def __init__(self, call_sid: str, stream_sid: str) -> None:
        self.call_sid: str = call_sid
        self.stream_sid: str = stream_sid
        self.created_at: float = time.time()
        self.state: CallState = CallState.NEW
        self.turn_number: int = 0
        self.conversation: List[dict] = []
        self.lock: asyncio.Lock = asyncio.Lock()
        self.pending_mark_name: Optional[str] = None
        self.pending_assistant_reply: str = ""
        self.consecutive_barge_in_frames: int = 0

    def transition_to(self, new_state: CallState) -> None:
        """Safely transition call state."""
        self.state = new_state

    def add_user_message(self, text: str) -> None:
        """Append a user transcript to the conversation history."""
        self.conversation.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        """Append an assistant reply to the conversation history."""
        self.conversation.append({"role": "assistant", "content": text})

    def duration_sec(self) -> float:
        """Return total elapsed call duration in seconds."""
        return time.time() - self.created_at


class TwilioCallRegistry:
    """In-memory registry managing active Twilio call sessions and lifecycle cleanup."""

    def __init__(self) -> None:
        self._sessions: Dict[str, TwilioCallSession] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def register_session(self, call_sid: str, stream_sid: str) -> TwilioCallSession:
        """Create and register a new call session keyed by stream_sid."""
        async with self._lock:
            session = TwilioCallSession(call_sid=call_sid, stream_sid=stream_sid)
            self._sessions[stream_sid] = session
            return session

    async def get_session(self, stream_sid: str) -> Optional[TwilioCallSession]:
        """Retrieve an active call session by stream_sid."""
        async with self._lock:
            return self._sessions.get(stream_sid)

    async def remove_session(self, stream_sid: str) -> Optional[TwilioCallSession]:
        """Remove a call session from the registry and mark it ended."""
        async with self._lock:
            session = self._sessions.pop(stream_sid, None)
            if session:
                session.transition_to(CallState.ENDED)
            return session

    def active_call_count(self) -> int:
        """Return the number of currently active calls."""
        return len(self._sessions)


twilio_call_registry = TwilioCallRegistry()