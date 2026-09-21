"""Tests for the 3CX/Asterisk settings loader and number normalisation."""
import pytest

from voice_engine.asterisk.config import get_settings, normalize_outbound_number

ASTERISK_ENV_KEYS = [
    "ASTERISK_ENABLED",
    "ARI_BASE_URL",
    "ARI_USERNAME",
    "ARI_PASSWORD",
    "ARI_APP",
    "THREECX_PJSIP_ENDPOINT",
    "THREECX_DOMAIN",
    "THREECX_EXTENSION",
    "ASTERISK_OUTBOUND_PREFIX",
    "ASTERISK_CALLER_ID",
    "ASTERISK_CHANNEL_TIMEOUT",
    "ASTERISK_MEDIA_BIND_IP",
    "ASTERISK_MEDIA_HOST",
    "ASTERISK_MEDIA_PORT",
    "ASTERISK_MEDIA_FORMAT",
    "ASTERISK_MEDIA_REMOTE_ADDR",
    "ASTERISK_MEDIA_REMOTE_PORT",
    "ASTERISK_FRAME_MS",
    "ASTERISK_PREBUFFER_MS",
    "ASTERISK_BARGE_IN_RMS_THRESHOLD",
    "ASTERISK_BARGE_IN_CONSECUTIVE_FRAMES",
    "ASTERISK_HOLD_MUSIC",
    "ASTERISK_HOLD_MUSIC_FILE",
    "ASTERISK_HOLD_MUSIC_GAIN",
    "ASTERISK_HOLD_MUSIC_DELAY_MS",
    "ASTERISK_STT_MODEL",
    "ASTERISK_STT_LANGUAGE",
    "ASTERISK_SILENCE_ENDPOINT_MS",
    "ASTERISK_UTTERANCE_END_MS",
]


@pytest.fixture
def clean_env(monkeypatch):
    for key in ASTERISK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_disabled_by_default(clean_env):
    s = get_settings()
    assert s.enabled is False  # dev machines without Asterisk are unaffected
    assert s.ari_base_url == "http://127.0.0.1:8088"
    assert s.ari_username == "voice-engine"
    assert s.ari_app == "voice-engine"
    assert s.threecx_pjsip_endpoint == "3cx"
    assert s.threecx_extension == "900"
    assert s.media_format == "ulaw"
    assert s.media_port == 0
    assert s.frame_ms == 20
    assert s.prebuffer_ms == 180
    assert s.hold_music_enabled is False
    assert s.channel_timeout == 30


def test_env_overrides(clean_env, monkeypatch):
    monkeypatch.setenv("ASTERISK_ENABLED", "true")
    monkeypatch.setenv("ARI_BASE_URL", "http://asterisk.lan:8088")
    monkeypatch.setenv("ARI_USERNAME", "svc")
    monkeypatch.setenv("ARI_PASSWORD", "pw")
    monkeypatch.setenv("ARI_APP", "my-app")
    monkeypatch.setenv("THREECX_PJSIP_ENDPOINT", "threecx")
    monkeypatch.setenv("THREECX_EXTENSION", "901")
    monkeypatch.setenv("ASTERISK_MEDIA_FORMAT", "ALAW")  # case-insensitive
    monkeypatch.setenv("ASTERISK_MEDIA_HOST", "10.0.0.5")
    monkeypatch.setenv("ASTERISK_MEDIA_PORT", "16000")
    monkeypatch.setenv("ASTERISK_OUTBOUND_PREFIX", "0")
    monkeypatch.setenv("ASTERISK_HOLD_MUSIC", "false")
    monkeypatch.setenv("ASTERISK_PREBUFFER_MS", "50")
    monkeypatch.setenv("ASTERISK_CHANNEL_TIMEOUT", "45")

    s = get_settings()
    assert s.enabled is True
    assert s.ari_base_url == "http://asterisk.lan:8088"
    assert s.ari_app == "my-app"
    assert s.threecx_pjsip_endpoint == "threecx"
    assert s.threecx_extension == "901"
    assert s.media_format == "alaw"
    assert s.media_port == 16000
    assert s.outbound_prefix == "0"
    assert s.hold_music_enabled is False
    assert s.prebuffer_ms == 50
    assert s.channel_timeout == 45


def test_normalize_outbound_number():
    assert normalize_outbound_number("+919876543210") == "919876543210"
    assert normalize_outbound_number("+91 98765 43210") == "919876543210"
    assert normalize_outbound_number("(020) 123-4567") == "0201234567"
    assert normalize_outbound_number("9876543210", prefix="0") == "09876543210"
    with pytest.raises(ValueError):
        normalize_outbound_number("   ")
    with pytest.raises(ValueError):
        normalize_outbound_number("")
