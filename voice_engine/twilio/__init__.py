"""Twilio calling feature for the Voice Engine.

This subpackage adds Twilio voice-calling to the voice engine so a real phone
call can use the engine's own STT/TTS providers and the RAG brain from the
saleKnowledgeBase app backend -- with no code outside this package required.

Endpoints added to the gateway:

* ``WS  /media-stream``        -- Twilio Media Streams bidirectional audio
* ``POST /api/twilio/voice``   -- inbound call TwiML (signature validated)
* ``POST /api/twilio/status``  -- call status webhook (signature validated)
* ``POST /api/calls/outbound`` -- outbound call trigger (API-key protected)

The call path keeps audio in native 8 kHz mu-law end-to-end (Twilio -> Deepgram
STT -> Deepgram Aura TTS -> Twilio), so there are no PCM conversions and no
audio-format mismatch beeps -- the same "zero-beep" design as the reference
calling agent.
"""
from .audio import (
    decode_mulaw_base64,
    encode_mulaw_base64,
    is_ws_open,
    linear16_to_mulaw,
    rms,
    PreBuffer,
)
from .config import get_settings, TwilioSettings
from .hold_music import SAMPLE_RATE, load_hold_music
from .session import (
    CallState,
    TwilioCallRegistry,
    TwilioCallSession,
    twilio_call_registry,
)
from .twiml import format_stream_url, outbound_response, voice_response
from .turn import TwilioTurnOrchestrator

__all__ = [
    "decode_mulaw_base64",
    "encode_mulaw_base64",
    "is_ws_open",
    "linear16_to_mulaw",
    "rms",
    "PreBuffer",
    "get_settings",
    "TwilioSettings",
    "SAMPLE_RATE",
    "load_hold_music",
    "CallState",
    "TwilioCallRegistry",
    "TwilioCallSession",
    "twilio_call_registry",
    "format_stream_url",
    "outbound_response",
    "voice_response",
    "TwilioTurnOrchestrator",
]