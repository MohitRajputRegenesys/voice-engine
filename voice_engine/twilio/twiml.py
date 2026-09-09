"""TwiML response generators for Twilio Voice and Media Streams.

Ported from the reference calling agent so inbound/outbound calls connect a
Twilio Media Stream to the engine's ``/media-stream`` WebSocket endpoint.
"""
import logging
from urllib.parse import urlparse

from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

logger = logging.getLogger("voice_engine.twilio.twiml")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


def warn_if_local_host(base_url: str, what: str = "Twilio") -> None:
    """Log a loud warning when a local/private host is used in a Twilio-facing URL.

    Twilio rejects local URLs with error 11100 "Invalid URL" -- this makes that
    misconfiguration obvious in the server logs instead of failing silently.
    """
    value = (base_url or "").strip()
    if not value:
        logger.warning(
            "%s URL host is empty -- Twilio will reject it with error 11100. "
            "Set TWILIO_BASE_URL to a public https host (e.g. your ngrok URL).",
            what,
        )
        return
    if "://" not in value:
        value = "https://" + value
    host = (urlparse(value).hostname or "").lower()
    if host in _LOCAL_HOSTS or host.endswith(".local"):
        logger.warning(
            "%s URL host '%s' is local/private -- Twilio will reject it with "
            "error 11100 'Invalid URL'. Set TWILIO_BASE_URL to a public https "
            "host (e.g. your ngrok URL).",
            what,
            host,
        )


def format_stream_url(base_url: str) -> str:
    """Format and ensure the stream URL uses the wss:// scheme and ends with /media-stream.

    Accepts a bare public host (``your-host.ngrok-free.dev``), a host with port
    (``your-host.ngrok-free.dev:8001``), or a full URL with any scheme, and always
    returns a ``wss://<host>/media-stream`` URL -- the only form Twilio Media
    Streams accepts (a local/private host such as ``localhost`` is rejected by
    Twilio with error 11100 "Invalid URL").
    """
    url = base_url.strip()
    if url.startswith("https://"):
        url = "wss://" + url[8:]
    elif url.startswith("http://"):
        url = "ws://" + url[7:]
    elif not url.startswith("wss://") and not url.startswith("ws://"):
        url = f"wss://{url}"

    url = url.rstrip("/")
    if not url.endswith("/media-stream"):
        url = f"{url}/media-stream"

    return url


def public_https_url(base_url: str, path: str) -> str:
    """Build an absolute ``https://`` URL for a public path from a base URL.

    Accepts a bare host (``your-host.ngrok-free.dev``), a host with port, or a
    full URL (``https://`` / ``http://`` / ``wss://``) -- with or without a
    trailing slash -- and returns ``https://<host><path>``. Mirrors the reference
    calling agent, which prepends ``https://`` to a bare ``BASE_URL``, but also
    tolerates a scheme already being present so the status callback can never be
    mangled into something like ``https://https://host/...`` (invalid for Twilio).
    """
    warn_if_local_host(base_url, "Status callback")
    value = (base_url or "").strip()
    if "://" not in value:
        value = "https://" + value
    host = urlparse(value).netloc.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"https://{host}{path}"


def voice_response(stream_url: str, call_sid: str = "") -> str:
    """Generate TwiML for an inbound call that connects a Twilio Media Stream."""
    warn_if_local_host(stream_url, "Media Stream")
    formatted_url = format_stream_url(stream_url)
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=formatted_url)
    if call_sid:
        stream.parameter(name="call_sid", value=call_sid)
    connect.append(stream)
    response.append(connect)
    return str(response)


def outbound_response(stream_url: str, call_sid: str = "") -> str:
    """Generate TwiML for an outbound call that connects a Twilio Media Stream."""
    return voice_response(stream_url=stream_url, call_sid=call_sid)