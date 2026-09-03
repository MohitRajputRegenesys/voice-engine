"""Provider selection matrix + Deepgram Aura TTS unit tests (fully mocked)."""
import httpx
import pytest

from voice_engine.providers import (
    EdgeTTSProvider,
    DeepgramSTTProvider,
    DeepgramTTSProvider,
    MockSTTProvider,
    MockTTSProvider,
    clean_spoken_text,
    build_stt_provider,
    build_tts_provider,
)


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)
    monkeypatch.delenv("STT_LANGUAGE", raising=False)
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    monkeypatch.delenv("TTS_MODEL", raising=False)


# ---------------------------------------------------------------- STT factory

def test_stt_deepgram_when_key_set(clean_env, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = build_stt_provider()
    assert isinstance(p, DeepgramSTTProvider)
    assert p.model == "nova-2"          # fixed fallback (was invalid nova-2-general)
    assert p.sample_rate == 16000

    monkeypatch.setenv("STT_LANGUAGE", "en-US")
    p = build_stt_provider()
    assert p.language == "en-US"


def test_stt_mock_without_key(clean_env):
    assert isinstance(build_stt_provider(), MockSTTProvider)


# ---------------------------------------------------------------- TTS factory

def test_tts_auto_selects_deepgram_when_key_present(clean_env, monkeypatch):
    """Default behavior: Aura when the shared Deepgram key exists."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = build_tts_provider()
    assert isinstance(p, DeepgramTTSProvider)
    assert p.model == "aura-2-thalia-en"


def test_tts_mock_without_key(clean_env):
    assert isinstance(build_tts_provider(), MockTTSProvider)


def test_tts_edge_opt_in_wins(clean_env, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")   # deepgram available...
    monkeypatch.setenv("TTS_PROVIDER", "edge")           # ...but edge explicitly chosen
    assert isinstance(build_tts_provider(), EdgeTTSProvider)


def test_tts_explicit_deepgram_requires_key(clean_env, monkeypatch):
    """Choosing Aura explicitly without a key must fail loudly, not silently mock."""
    monkeypatch.setenv("TTS_PROVIDER", "deepgram")
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        build_tts_provider()


# ------------------------------------------------------------ media types

def test_media_type_attributes(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    assert MockTTSProvider().media_type == "application/octet-stream"
    assert EdgeTTSProvider().media_type == "audio/mpeg"
    assert DeepgramTTSProvider().media_type == "audio/mpeg"


# --------------------------------------------- DeepgramTTSProvider synthesis

class FakeResponse:
    def __init__(self, status_code=200, content=b"", text=""):
        self.status_code = status_code
        self.content = content
        self.text = text


class FakeAsyncClient:
    """Stands in for httpx.AsyncClient and records posts."""
    calls = []

    def __init__(self, **kwargs):
        FakeAsyncClient.calls = []
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kw):
        FakeAsyncClient.calls.append((url, kw))
        return FakeResponse(content=b"AURA_MP3_BYTES")


@pytest.fixture
def fake_http(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)


def _text_iter(*pieces):
    class _Iter:
        def __aiter__(self):
            self._it = iter(pieces)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    return _Iter()


@pytest.mark.asyncio
async def test_aura_short_text_single_chunk(clean_env, monkeypatch, fake_http):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = DeepgramTTSProvider(model="aura-2-thalia-en")

    chunks = [c async for c in p.synthesize(_text_iter("Hello world"), "resp-1")]

    assert len(chunks) == 1
    assert chunks[0].response_id == "resp-1"
    assert chunks[0].sequence == 0
    assert chunks[0].data == b"AURA_MP3_BYTES"

    url, kw = FakeAsyncClient.calls[0]
    assert url == DeepgramTTSProvider.API_URL
    assert kw["params"] == {"model": "aura-2-thalia-en"}
    assert kw["headers"]["Authorization"] == "Token test-key"
    assert kw["json"] == {"text": "Hello world"}


@pytest.mark.asyncio
async def test_aura_long_text_respects_char_limit(clean_env, monkeypatch, fake_http):
    """Text beyond Aura's 2000-char limit must be split into multiple requests."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = DeepgramTTSProvider()

    long_text = ("lorem ipsum dolor sit amet " * 300)  # ~8100 chars
    chunks = [c async for c in p.synthesize(_text_iter(long_text), "resp-2")]

    assert len(chunks) >= 2                          # split happened
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
    for url, kw in FakeAsyncClient.calls:
        sent = kw["json"]["text"]
        assert len(sent) <= DeepgramTTSProvider.MAX_CHARS


@pytest.mark.asyncio
async def test_aura_skips_empty_input(clean_env, monkeypatch, fake_http):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = DeepgramTTSProvider()
    chunks = [c async for c in p.synthesize(_text_iter("   "), "resp-3")]
    assert chunks == []


def test_aura_fails_fast_without_key(clean_env):
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        DeepgramTTSProvider()


# ------------------------------------------------------- spoken-text cleaning

@pytest.mark.parametrize(
    "raw,expected",
    [
        # The exact case from the bug report: "- ** test **" -> "test".
        ("- ** test **", "test"),
        ("**bold**", "bold"),
        ("__bold__", "bold"),
        ("*italic*", "italic"),
        ("_italic_", "italic"),
        ("`code`", "code"),
        ("~~strike~~", "strike"),
        ("Mixed **bold** and *italic* text", "Mixed bold and italic text"),
        # Plain text is left untouched.
        ("Hello world", "Hello world"),
        # Whitespace-only collapses to empty (so the provider skips it).
        ("   ", ""),
        ("", ""),
        (None, None),
        # Heading / blockquote / bullet prefixes at line start are dropped.
        ("# Heading", "Heading"),
        ("> quoted", "quoted"),
        ("- bullet item", "bullet item"),
        # Lone/unbalanced asterisks are stripped, not spoken.
        ("5 * 3 = 15", "5 3 = 15"),
    ],
)
def test_clean_spoken_text_strips_markdown(raw, expected):
    assert clean_spoken_text(raw) == expected


@pytest.mark.asyncio
async def test_aura_strips_markdown_before_sending(clean_env, monkeypatch, fake_http):
    """Markdown markers in the answer must be removed before hitting the TTS API."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = DeepgramTTSProvider()

    chunks = [c async for c in p.synthesize(_text_iter("**bold** and *italic*"), "resp")]

    assert len(chunks) == 1
    assert chunks[0].data == b"AURA_MP3_BYTES"
    url, kw = FakeAsyncClient.calls[0]
    assert url == DeepgramTTSProvider.API_URL
    # The asterisks themselves must never be sent to Deepgram.
    assert kw["json"]["text"] == "bold and italic"
    assert "*" not in kw["json"]["text"]


@pytest.mark.asyncio
async def test_aura_strips_leading_marker_and_bold(clean_env, monkeypatch, fake_http):
    """The reported input ``- ** test **`` must be spoken as just ``test``."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    p = DeepgramTTSProvider()

    [c async for c in p.synthesize(_text_iter("- ** test **"), "resp")]

    _, kw = FakeAsyncClient.calls[0]
    assert kw["json"]["text"] == "test"


@pytest.mark.asyncio
async def test_mock_tts_strips_markdown(clean_env):
    """MockTTSProvider must also never emit raw asterisks in its fake audio."""
    p = MockTTSProvider()
    # MockTTSProvider cleans each incoming chunk independently (no joining),
    # so every chunk gets its own AUDIO(...) frame.
    chunks = [c async for c in p.synthesize(_text_iter("- ** test **", "extra **hi**"), "resp")]
    texts = [c.data.decode("utf-8") for c in chunks]
    assert texts == ["AUDIO(test)", "AUDIO(extra hi)"]
    assert "*" not in "".join(texts)