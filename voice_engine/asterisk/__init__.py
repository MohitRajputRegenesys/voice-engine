"""3CX / Asterisk calling feature for the Voice Engine (ARI + ExternalMedia RTP).

This subpackage adds 3CX phone calling to the voice engine so a real call can
use the engine's own STT/TTS providers and the RAG brain from the
saleKnowledgeBase app backend -- with no code outside this package required.
The Twilio calling feature (``voice_engine.twilio``) stays fully intact; the
two can run side by side.

Architecture::

    voice-engine --HTTP/WS ARI--> Asterisk --SIP REGISTER ext 900--> 3CX --> Airtel
        ^                              |
        └── STT -> RAG LLM -> TTS <----+-- ExternalMedia RTP (mu-law/alaw/slin16)

Endpoints added to the gateway:

* ``POST /api/threecx/calls/outbound`` (+ ``/api/asterisk/calls/outbound``)
  -- outbound call trigger (API-key protected)
* ``GET  /api/threecx/status`` (+ ``/api/asterisk/status``) -- health/active calls
* ``GET  /api/threecx/calls`` -- active call list
* ``POST /api/threecx/calls/{channel_id}/hangup`` -- hang up an active call

The integration is disabled unless ``ASTERISK_ENABLED=true``; sample Asterisk
configuration (pjsip/http/ari/dialplan) lives in ``asterisk-config/``.
"""
from .ari import AriClient, AriError
from .calls import AsteriskCallManager, asterisk_call_manager
from .config import AsteriskSettings, get_settings, normalize_outbound_number
from .rtp import RtpSession, build_rtp_packet, parse_rtp_packet
from .session import (
    AsteriskCallRegistry,
    AsteriskCallSession,
    CallState,
    asterisk_call_registry,
)
from .turn import AsteriskTurnOrchestrator

__all__ = [
    "AriClient",
    "AriError",
    "AsteriskCallManager",
    "asterisk_call_manager",
    "AsteriskSettings",
    "get_settings",
    "normalize_outbound_number",
    "RtpSession",
    "build_rtp_packet",
    "parse_rtp_packet",
    "AsteriskCallRegistry",
    "AsteriskCallSession",
    "CallState",
    "asterisk_call_registry",
    "AsteriskTurnOrchestrator",
]
