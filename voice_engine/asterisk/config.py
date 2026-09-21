"""3CX / Asterisk call configuration read lazily from environment variables.

Reading happens per call so tests can monkeypatch ``os.environ`` freely and so
the gateway picks up runtime changes without a restart -- the same pattern as
the Twilio feature (``voice_engine.twilio.config``).

The integration is DISABLED unless ``ASTERISK_ENABLED=true`` so development
machines without an Asterisk/3CX deployment keep working unchanged.

Connection map (see asterisk-config/README.md for the Asterisk/3CX side):

    voice-engine --HTTP/WS ARI--> Asterisk --SIP REGISTER ext 900--> 3CX --> Airtel
"""
import os
import re
from dataclasses import dataclass


@dataclass
class AsteriskSettings:
    # -- master switch --------------------------------------------------------
    enabled: bool
    # -- ARI (application connection to the local Asterisk gateway) -----------
    ari_base_url: str
    ari_username: str
    ari_password: str
    ari_app: str
    # -- 3CX identity (Asterisk registers to 3CX as this extension) -----------
    threecx_pjsip_endpoint: str  # PJSIP endpoint name in pjsip.conf
    threecx_domain: str
    threecx_extension: str
    # -- outbound dialing -----------------------------------------------------
    outbound_prefix: str
    caller_id: str
    channel_timeout: int
    # -- external media (AudioSocket/TCP by default; RTP/UDP where supported) --
    media_transport: str  # audiosocket | rtp
    media_bind_ip: str
    media_host: str  # IP of THIS machine as reachable from Asterisk
    media_port: int  # 0 = auto-pick a free UDP port
    media_format: str  # ulaw | alaw | slin16
    media_remote_addr: str  # optional static Asterisk RTP target (else latched)
    media_remote_port: int
    frame_ms: int
    # -- conversation tuning (same semantics as the Twilio block) -------------
    welcome_greeting: str
    prebuffer_ms: int
    barge_in_rms_threshold: float
    barge_in_consecutive_frames: int
    hold_music_enabled: bool
    hold_music_file: str
    hold_music_gain: float
    hold_music_start_delay_ms: int
    # -- phone-path STT tuning ------------------------------------------------
    stt_model: str
    stt_language: str
    silence_endpoint_ms: int
    utterance_end_ms: int
    # -- shared API key for the outbound trigger (same as Twilio's X-API-Key) --
    api_key: str


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def get_settings() -> AsteriskSettings:
    """Build the current Asterisk/3CX settings from the environment."""
    return AsteriskSettings(
        enabled=_env_flag("ASTERISK_ENABLED", False),
        ari_base_url=os.environ.get("ARI_BASE_URL", "http://127.0.0.1:8088"),
        ari_username=os.environ.get("ARI_USERNAME", "voice-engine"),
        ari_password=os.environ.get("ARI_PASSWORD", ""),
        ari_app=os.environ.get("ARI_APP", "voice-engine"),
        threecx_pjsip_endpoint=os.environ.get("THREECX_PJSIP_ENDPOINT", "3cx"),
        threecx_domain=os.environ.get("THREECX_DOMAIN", ""),
        threecx_extension=os.environ.get("THREECX_EXTENSION", "900"),
        outbound_prefix=os.environ.get("ASTERISK_OUTBOUND_PREFIX", ""),
        caller_id=os.environ.get("ASTERISK_CALLER_ID", ""),
        channel_timeout=int(os.environ.get("ASTERISK_CHANNEL_TIMEOUT", "30") or 30),
        media_transport=(os.environ.get("ASTERISK_MEDIA_TRANSPORT", "audiosocket") or "audiosocket").strip().lower(),
        media_bind_ip=os.environ.get("ASTERISK_MEDIA_BIND_IP", "0.0.0.0"),
        media_host=os.environ.get("ASTERISK_MEDIA_HOST", "127.0.0.1"),
        media_port=int(os.environ.get("ASTERISK_MEDIA_PORT", "0") or 0),
        media_format=(os.environ.get("ASTERISK_MEDIA_FORMAT", "ulaw") or "ulaw").strip().lower(),
        media_remote_addr=os.environ.get("ASTERISK_MEDIA_REMOTE_ADDR", ""),
        media_remote_port=int(os.environ.get("ASTERISK_MEDIA_REMOTE_PORT", "0") or 0),
        frame_ms=int(os.environ.get("ASTERISK_FRAME_MS", "20") or 20),
        welcome_greeting=os.environ.get(
            "ASTERISK_WELCOME_GREETING",
            "Hello! Welcome to our sales information service. How can I help you today?",
        ),
        prebuffer_ms=int(os.environ.get("ASTERISK_PREBUFFER_MS", "180") or 180),
        barge_in_rms_threshold=float(
            os.environ.get("ASTERISK_BARGE_IN_RMS_THRESHOLD", "900.0") or 900.0
        ),
        barge_in_consecutive_frames=int(
            os.environ.get("ASTERISK_BARGE_IN_CONSECUTIVE_FRAMES", "4") or 4
        ),
        # AudioSocket hold music is opt-in because queued music frames can
        # contaminate the first TTS frames on the phone bridge.
        hold_music_enabled=_env_flag("ASTERISK_HOLD_MUSIC", False),
        hold_music_file=os.environ.get("ASTERISK_HOLD_MUSIC_FILE", ""),
        hold_music_gain=float(os.environ.get("ASTERISK_HOLD_MUSIC_GAIN", "0.3") or 0.3),
        hold_music_start_delay_ms=int(
            os.environ.get("ASTERISK_HOLD_MUSIC_DELAY_MS", "400") or 400
        ),
        stt_model=os.environ.get("ASTERISK_STT_MODEL", ""),
        stt_language=os.environ.get("ASTERISK_STT_LANGUAGE", ""),
        silence_endpoint_ms=int(os.environ.get("ASTERISK_SILENCE_ENDPOINT_MS", "800") or 800),
        utterance_end_ms=int(os.environ.get("ASTERISK_UTTERANCE_END_MS", "1500") or 1500),
        api_key=os.environ.get("CALLS_API_KEY", "secret-api-key"),
    )


def normalize_outbound_number(phone: str, prefix: str = "") -> str:
    """Reduce a phone number to the digits the 3CX outbound rule expects.

    All non-digits are stripped (``+91 98765 43210`` -> ``919876543210``) and
    ``prefix`` is prepended verbatim (e.g. ``0`` for national format or an
    Airtel trunk prefix). Which final digit string works is decided by the 3CX
    outbound rule for extension 900 -- do NOT add transformations here blindly;
    tune ``ASTERISK_OUTBOUND_PREFIX`` after checking the existing Airtel rule.
    """
    digits = re.sub(r"[^0-9]", "", phone or "")
    if not digits:
        raise ValueError("phone number contains no digits")
    return f"{prefix}{digits}"
