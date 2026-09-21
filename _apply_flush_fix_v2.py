import py_compile, os

ROOT = "D:/tg-imp/saleKnowledgeBase/voice-engine"
FILES = {
 "turn": ROOT + "/voice_engine/asterisk/turn.py",
 "calls": ROOT + "/voice_engine/asterisk/calls.py",
 "test": ROOT + "/tests/test_asterisk_turn.py",
}

def load(p):
    raw = open(p, "rb").read()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8").replace("\r\n", "\n"), nl

def save(p, t, nl):
    open(p, "wb").write(t.replace("\n", nl).encode())

def patch(p, edits):
    t, nl = load(p)
    for old, new in edits:
        c = t.count(old)
        assert c == 1, "BAD ANCHOR count=%r :: %r" % (c, old[:80])
        t = t.replace(old, new, 1)
    save(p, t, nl)
    py_compile.compile(p, doraise=True)
    print("patched+compiled", p)

turn_edits = [
 ('        clear_media_func: Optional[Callable[[], Awaitable[None]]] = None,\n        settings: Optional[AsteriskSettings] = None,\n',
  '        clear_media_func: Optional[Callable[[], Awaitable[None]]] = None,\n        flush_media_func: Optional[Callable[[], Awaitable[None]]] = None,\n        settings: Optional[AsteriskSettings] = None,\n'),
 ('        self.clear_media_func = clear_media_func\n        settings = settings or get_settings()\n',
  '        self.clear_media_func = clear_media_func\n        self.flush_media_func = flush_media_func\n        settings = settings or get_settings()\n'),
 ('    async def _wait_for_media_drained(self) -> None:\n        """Wait until the RTP sender has handed every queued frame to Asterisk."""\n',
  '    async def _flush_outbound(self) -> None:\n        """Flush the wire-framer tail partial at turn end.\n\n        push() only emits complete ptime frames; the remainder stays in _pending.\n        Never flushing it makes the next turn first frame stitch from stale\n        mu-law bytes -> Asterisk decodes 0x10 as slin -> full-scale PCM noise\n        (the post-greeting noise). Padding with mu-law silence gives a clean\n        boundary and empties _pending for the next turn.\n        """\n        if self.flush_media_func is None:\n            return\n        try:\n            result = self.flush_media_func()\n            if inspect.isawaitable(result):\n                await result\n        except Exception as exc:  # noqa: BLE001\n            logger.warning("Flushing wire framer failed: %s", exc)\n\n    async def _wait_for_media_drained(self) -> None:\n        """Wait until the RTP sender has handed every queued frame to Asterisk."""\n'),
 ('            for chunk in self.pre_buffer.flush():\n                await self.send_media_func(chunk)\n\n            await self._wait_for_media_drained()\n            if self.session.state == CallState.WELCOME:\n',
  '            for chunk in self.pre_buffer.flush():\n                await self.send_media_func(chunk)\n            await self._flush_outbound()\n\n            await self._wait_for_media_drained()\n            if self.session.state == CallState.WELCOME:\n'),
 ('                for chunk in self.pre_buffer.flush():\n                    await self.send_media_func(chunk)\n\n                full_reply = sanitize_speech_text("".join(self._current_reply_parts))\n',
  '                for chunk in self.pre_buffer.flush():\n                    await self.send_media_func(chunk)\n                await self._flush_outbound()\n\n                full_reply = sanitize_speech_text("".join(self._current_reply_parts))\n'),
]

calls_edits = [
 ('            send_media_func=rtp.queue_mulaw,\n            wait_media_drained_func=rtp.drain,\n',
  '            send_media_func=rtp.queue_mulaw,\n            flush_media_func=rtp.flush_mulaw,\n            wait_media_drained_func=rtp.drain,\n'),
]

print("TURN"); patch(FILES["turn"], turn_edits)
print("CALLS"); patch(FILES["calls"], calls_edits)

# tests
t, nl = load(FILES["test"])
old1 = '        self.chunks = []\n        self.cleared = False\n\n    async def queue_mulaw(self, chunk):\n        self.chunks.append(chunk)\n\n    async def drain(self, timeout=10.0):\n        return None\n\n    def clear(self):\n        self.cleared = True\n'
new1 = '        self.chunks = []\n        self.cleared = False\n        self.flush_count = 0\n\n    async def queue_mulaw(self, chunk):\n        self.chunks.append(chunk)\n\n    async def drain(self, timeout=10.0):\n        return None\n\n    def clear(self):\n        self.cleared = True\n\n    async def flush_mulaw(self):\n        self.flush_count += 1\n'
assert t.count(old1) == 1, "test anchor1 bad"
t = t.replace(old1, new1, 1)
old2 = '        send_media_func=media.queue_mulaw,\n        wait_media_drained_func=media.drain,\n        clear_media_func=media.clear,\n    )\n'
new2 = '        send_media_func=media.queue_mulaw,\n        flush_media_func=media.flush_mulaw,\n        wait_media_drained_func=media.drain,\n        clear_media_func=media.clear,\n    )\n'
assert t.count(old2) == 1, "test anchor2 bad"
t = t.replace(old2, new2, 1)
REG = '\n\n@pytest.mark.asyncio\nasync def test_turn_flushes_wire_framer_at_end_of_every_turn(call_settings):\n    """Regression: the shared WireFramer must be flushed at each turn end so the\n    tail partial mu-law is padded+emitted, not carried into the next turn first\n    frame (the post-greeting noise root cause on AudioSocket)."""\n    session, tts, llm, media, orchestrator = build_harness()\n\n    await orchestrator.send_welcome_greeting()\n    assert media.flush_count == 1            # greeting flushed its final partial\n\n    session.transition_to(CallState.LISTENING)\n    before = media.flush_count\n    await orchestrator.handle_user_transcript("What courses do you offer?")\n    assert media.flush_count == before + 1   # reply flushed its final partial\n    assert session.state == CallState.LISTENING\n    assert session.turn_number == 1\n'
t = t.rstrip() + "\n" + REG
save(FILES["test"], t, nl)
py_compile.compile(FILES["test"], doraise=True)
print("patched+compiled", FILES["test"])
print("ALL DONE")
