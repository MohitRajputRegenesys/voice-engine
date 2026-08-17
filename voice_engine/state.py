from enum import Enum
from typing import Set, Dict


class SessionState(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR = "ERROR"


# Valid transitions map
_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
    SessionState.IDLE: {SessionState.CONNECTING, SessionState.CLOSED, SessionState.ERROR},
    SessionState.CONNECTING: {SessionState.LISTENING, SessionState.ERROR, SessionState.CLOSING},
    SessionState.LISTENING: {SessionState.PROCESSING, SessionState.CLOSING, SessionState.ERROR},
    SessionState.PROCESSING: {SessionState.SPEAKING, SessionState.LISTENING, SessionState.INTERRUPTED, SessionState.CLOSING, SessionState.ERROR},
    SessionState.SPEAKING: {SessionState.LISTENING, SessionState.INTERRUPTED, SessionState.CLOSING, SessionState.ERROR},
    SessionState.INTERRUPTED: {SessionState.PROCESSING, SessionState.LISTENING, SessionState.CLOSING, SessionState.ERROR},
    SessionState.CLOSING: {SessionState.CLOSED},
    SessionState.CLOSED: set(),
    SessionState.ERROR: {SessionState.CLOSING, SessionState.CLOSED},
}


class InvalidTransition(Exception):
    pass


class StateMachine:
    def __init__(self):
        self.state = SessionState.IDLE

    def transition(self, new_state: SessionState):
        if new_state == self.state:
            return
        allowed = _TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransition(f"Invalid state transition {self.state} -> {new_state}")
        self.state = new_state
