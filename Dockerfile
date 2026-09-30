
# ---------------------------------------------------------------------------
# Stage 1 — install the "server" dependency extra into a self-contained venv
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

# gcc is a safety net for any dependency without a cp313 manylinux wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# README.md is required by hatchling: pyproject.toml declares `readme = "README.md"`
COPY pyproject.toml README.md /build/
COPY voice_engine /build/voice_engine

RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && cd /build && /venv/bin/pip install --no-cache-dir ".[server]"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

# Non-root user (same convention as the ai-facilitators backend image)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Dependencies (fastapi, uvicorn[standard], twilio, edge-tts, ...)
COPY --from=builder /venv /venv

WORKDIR /app

# Same layout as the repo, so voice_engine/server.py still finds /app/.env and
# PYTHONPATH=/app makes the source here win over the venv's installed copy
# (which also lets you bind-mount ./voice_engine for live edits).
COPY --chown=appuser:appuser voice_engine /app/voice_engine

# Hold-music track referenced by TWILIO_HOLD_MUSIC_FILE / ASTERISK_HOLD_MUSIC_FILE
COPY --chown=appuser:appuser hold_music.wav /app/hold_music.wav

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

USER appuser

EXPOSE 8001

# Same liveness probe the saleKnowledgeBase app uses for its /api/v1/voice/health
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/').read()" || exit 1

CMD ["uvicorn", "voice_engine.server:app", "--host", "0.0.0.0", "--port", "8001"]
