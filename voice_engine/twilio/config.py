"""Twilio call configuration read lazily from environment variables.

Reading happens per call so tests can monkeypatch ``os.environ`` freely and so
the gateway picks up any runtime changes without a restart.
"""
import os
from dataclasses import dataclass


@dataclass
class TwilioSettings:
    account_sid: str
    auth_token: str
    phone_number: str
    base_url: str
    voice_url: str
    api_key: str
    env: str
    prebuffer_ms: int
    barge_in_rms_threshold: float
    barge_in_consecutive_frames: int
    welcome_greeting: str
    hold_music_enabled: bool
    hold_music_file: str
    hold_music_gain: float
    hold_music_start_delay_ms: int


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def get_settings() -> TwilioSettings:
    """Build the current Twilio settings from the environment."""
    return TwilioSettings(
        account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
        phone_number=os.environ.get("TWILIO_PHONE_NUMBER", ""),
        base_url=os.environ.get("TWILIO_BASE_URL", "localhost:8001"),
        voice_url=os.environ.get("TWILIO_VOICE_URL", ""),
        api_key=os.environ.get("CALLS_API_KEY", "secret-api-key"),
        env=os.environ.get("ENV", "development"),
        prebuffer_ms=int(os.environ.get("TWILIO_PREBUFFER_MS", "180")),
        barge_in_rms_threshold=float(
            os.environ.get("TWILIO_BARGE_IN_RMS_THRESHOLD", "900.0")
        ),
        barge_in_consecutive_frames=int(
            os.environ.get("TWILIO_BARGE_IN_CONSECUTIVE_FRAMES", "4")
        ),
        welcome_greeting=os.environ.get(
            "TWILIO_WELCOME_GREETING",
            "Hello! Welcome to our sales information service. How can I help you today?",
        ),
        hold_music_enabled=_env_flag("TWILIO_HOLD_MUSIC", True),
        hold_music_file=os.environ.get("TWILIO_HOLD_MUSIC_FILE", ""),
        hold_music_gain=float(os.environ.get("TWILIO_HOLD_MUSIC_GAIN", "0.3") or 0.3),
        hold_music_start_delay_ms=int(
            os.environ.get("TWILIO_HOLD_MUSIC_DELAY_MS", "400") or 400
        ),
    )