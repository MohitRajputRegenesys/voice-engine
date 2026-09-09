"""Unit tests for the Twilio TwiML generators and the mu-law audio helpers."""
from voice_engine.twilio.audio import (
    decode_mulaw_base64,
    encode_mulaw_base64,
    rms,
    PreBuffer,
)
from voice_engine.twilio.twiml import (
    format_stream_url,
    outbound_response,
    public_https_url,
    voice_response,
)


# ---------------------------------------------------------------- public_https_url

def test_public_https_url_bare_host():
    assert public_https_url("presoak-excavate-exodus.ngrok-free.dev", "/api/twilio/status") == (
        "https://presoak-excavate-exodus.ngrok-free.dev/api/twilio/status"
    )


def test_public_https_url_already_has_scheme():
    assert public_https_url("https://example.com", "/api/twilio/status") == "https://example.com/api/twilio/status"
    assert public_https_url("http://example.com", "/api/twilio/status") == "https://example.com/api/twilio/status"
    assert public_https_url("wss://example.com", "/api/twilio/status") == "https://example.com/api/twilio/status"


def test_public_https_url_never_double_scheme():
    # regression: previously https:// + https://host produced https://https://host (Twilio 11100)
    assert public_https_url("https://voice.example.com", "/api/twilio/status") != "https://https://voice.example.com/api/twilio/status"


def test_public_https_url_strips_path_and_slash():
    assert public_https_url("https://example.com/api/twilio/voice", "/api/twilio/status") == (
        "https://example.com/api/twilio/status"
    )
    assert public_https_url("https://example.com/", "/api/twilio/status") == "https://example.com/api/twilio/status"


def test_public_https_url_normalizes_path_leading_slash():
    assert public_https_url("host.example", "api/twilio/status") == "https://host.example/api/twilio/status"


# ---------------------------------------------------------------- TwiML

def test_format_stream_url_https():
    assert format_stream_url("https://example.com") == "wss://example.com/media-stream"


def test_format_stream_url_http():
    assert format_stream_url("http://example.com:8001") == "ws://example.com:8001/media-stream"


def test_format_stream_url_bare_host():
    assert format_stream_url("localhost:8001") == "wss://localhost:8001/media-stream"


def test_format_stream_url_wss_passthrough():
    assert format_stream_url("wss://example.com:8001") == "wss://example.com:8001/media-stream"


def test_format_stream_url_preserves_existing_path():
    assert format_stream_url("https://example.com/media-stream") == "wss://example.com/media-stream"


def test_format_stream_url_handles_trailing_slash():
    assert format_stream_url("https://example.com/") == "wss://example.com/media-stream"
    assert format_stream_url("https://example.com/media-stream/") == "wss://example.com/media-stream"


def test_format_stream_url_bare_public_host_no_duplicate_path():
    # ALCALLINGAGENT uses a bare public host (its working BASE_URL format)
    assert format_stream_url("presoak-excavate-exodus.ngrok-free.dev") == (
        "wss://presoak-excavate-exodus.ngrok-free.dev/media-stream"
    )


def test_voice_response_contains_connect_stream_and_call_sid():
    xml = voice_response("https://example.com", call_sid="CA123")
    assert "<Connect>" in xml
    assert "wss://example.com/media-stream" in xml
    assert 'name="call_sid"' in xml
    assert "CA123" in xml


def test_voice_response_without_call_sid():
    xml = voice_response("https://example.com")
    assert "call_sid" not in xml


def test_outbound_response_matches_voice_response():
    assert outbound_response("https://example.com", "CA1") == voice_response(
        "https://example.com", "CA1"
    )


# ---------------------------------------------------------------- mu-law audio

def test_mulaw_base64_roundtrip():
    payload = encode_mulaw_base64(b"\x00\x01\x7f\xff")
    assert decode_mulaw_base64(payload) == b"\x00\x01\x7f\xff"


def test_rms_silence_is_zero():
    assert rms(b"\xff" * 160) == 0.0


def test_rms_loud_mulaw_is_high():
    assert rms(b"\x00" * 160) > 1000.0


def test_prebuffer_buffers_then_flushes_batch():
    p = PreBuffer(target_ms=10, sample_rate=8000)  # 80 bytes target
    assert p.push(b"\x00" * 40) == []  # still buffering
    ready = p.push(b"\x00" * 40)
    assert len(ready) == 1
    assert len(ready[0]) == 80


def test_prebuffer_flush_remaining():
    p = PreBuffer(target_ms=100000, sample_rate=8000)  # huge target
    p.push(b"\x00" * 100)
    chunks = p.flush()
    assert len(chunks) == 1
    assert len(chunks[0]) == 100


def test_prebuffer_reset_clears():
    p = PreBuffer(target_ms=10, sample_rate=8000)
    p.push(b"\x00" * 40)
    p.reset()
    assert p.push(b"\x00" * 40) == []