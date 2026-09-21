# Asterisk + 3CX deployment runbook (voice-engine AI agent)

This folder contains ready-to-adapt **Asterisk** configuration for the
architecture below. The Airtel side already exists inside 3CX — nothing there
changes except adding the outbound rule condition.

```
voice-engine (this repo, port 8001)
     │  ARI: HTTP :8088 + WS /ari/events          ← http.conf / ari.conf
     ▼
Asterisk ── SIP REGISTER as ext 900 ──► 3CX ── outbound rule ──► Airtel ──► PSTN
     ▲  PJSIP endpoint "3cx"                (FQDN digitalregenesys.elastix.com)
     └── ExternalMedia (AudioSocket/TCP, ulaw) ── mixing bridge
```

**Security:** keep 3CX + Asterisk + the voice-engine on the same LAN/private
VPN (WireGuard / site-to-site if the engine is in AWS). Do not expose 5060,
8088 or the RTP range to the internet — 3CX blacklists IPs after repeated
failed registrations.

> **Verified live** against Asterisk 20.21.0 (Docker) + 3CX ext 900 → Airtel:
> registration, outbound dialing, the mixing bridge, ExternalMedia and the full
> STT → LLM → TTS conversation were all exercised on a real phone call.
> The commands below are the exact ones used.

> **Media transport:** the engine defaults to `ASTERISK_MEDIA_TRANSPORT=audiosocket`
> (TCP) — Asterisk 18/20/21 rejects `externalMedia` over UDP RTP with HTTP 501,
> and AudioSocket needs **no** published RTP ports. Set it to `rtp` only if your
> deployment actually supports `unicastRTP`. With AudioSocket the engine never
> sends a UUID frame back (Asterisk fails the channel if you do).

## Files to copy to `/etc/asterisk/`

| File | Purpose |
|------|---------|
| `pjsip.conf` | Transport, the `3cx` endpoint/auth/AOR and the `3cx-reg` registration as **extension 900** |
| `extensions.conf` | `from-3cx` → `Stasis(voice-engine,inbound)` for inbound calls + Milestone-3 echo test |
| `http.conf` | ARI HTTP/WebSocket listener (:8088) |
| `ari.conf` | ARI user for the engine (`ARI_USERNAME`/`ARI_PASSWORD`) |
| `rtp.conf` | RTP port range (10000–20000) |

Replace the three placeholders in `pjsip.conf` with the values from
**3CX Admin → Users → AI Agent → IP Phone → "I will configure the phone
myself"**: Authentication ID, Password, and the 3CX FQDN/port.

Then (examples assume the container is named `asterisk-test`):

```bash
docker exec asterisk-test asterisk -rx "core reload"            # or: systemctl restart asterisk
docker exec asterisk-test asterisk -rx "pjsip show registrations"   # want: 3cx-reg  ->  Registered
docker exec asterisk-test asterisk -rx "ari show users"             # want: voice-engine
docker exec asterisk-test asterisk -rx "ari show apps"              # app appears after the engine starts
docker exec asterisk-test asterisk -rx "http show status"           # want: Server Enabled + Bound to 0.0.0.0:8088
```

ARI is only reachable from the host when the container publishes it:
`ports: ["127.0.0.1:8088:8088"]` (the engine is the only client that needs it).

## Milestone checklist

| # | Milestone | How to verify |
|---|-----------|---------------|
| 1 | Asterisk → 3CX registration | `pjsip show registrations` → **Registered**; 3CX dashboard shows ext 900 online |
| 2 | 3CX → Airtel rings your mobile | Originate `PJSIP/3cx/<mobile>` into the echo test — your phone rings (`channel originate PJSIP/0730825043@3cx extension 0730825043@from-3cx-dialout` style also works) |
| 3 | Two-way audio Asterisk ↔ customer | Answer the call: `Echo()` repeats everything you say (proves 3CX+RTP+codecs) |
| 4 | STT → LLM → TTS over the call | Start the engine with `ASTERISK_ENABLED=true` and call again via the API — greeting plays, RAG answers, barge-in works |
| 5 | `POST /api/threecx/calls/outbound` | API-driven dialing end-to-end (below) |

## 3CX side (one-time)

1. **Extension 900 "AI Agent"** — already created (User role, Aashna Team). DID not needed yet.
2. **Outbound Rule** — add rule `AI Agent - Airtel`:
   - *Calls from extension(s)*: `900`
   - Route: your existing **Airtel** trunk
   - Leave number transformation alone at first — inspect how the existing
     Airtel rule formats digits, then tune `ASTERISK_OUTBOUND_PREFIX` in the
     engine `.env` if needed.
3. **Caller ID** — once calls connect, set the outbound caller ID on the rule
   or trunk (3CX order: outbound rule → extension → trunk → main trunk number).

## Voice-engine configuration

Copy the `ASTERISK_*` / `ARI_*` / `THREECX_*` block from `.env.example` into
`.env`, set `ASTERISK_ENABLED=true`, then:

- `ARI_BASE_URL` — `http://<asterisk-ip>:8088`
- `ARI_USERNAME` / `ARI_PASSWORD` — from `ari.conf`
- `ASTERISK_MEDIA_TRANSPORT` — `audiosocket` (default, TCP) or `rtp` (UDP)
- `ASTERISK_MEDIA_HOST` — **the address of the machine running the voice-engine
  as reachable from Asterisk**. With Asterisk in Docker on the same host use
  `host.docker.internal`; on separate hosts use the engine's LAN/VPN IP.
- `ASTERISK_MEDIA_BIND_IP` — `0.0.0.0` so the container can dial back in
- `ASTERISK_MEDIA_FORMAT` — `ulaw` (default), `alaw`, or `slin16`

Start the gateway:

```bash
uv run uvicorn voice_engine.server:app --port 8001
```

Startup log should show `ARI events websocket connected (app=voice-engine)`;
`GET /api/threecx/status` should return `"connected": true`.

## Driving calls

```bash
# Outbound call (Milestone 5)
curl -X POST http://localhost:8001/api/threecx/calls/outbound \
     -H 'Content-Type: application/json' -H 'X-API-Key: secret-api-key' \
     -d '{"phone": "+919876543210"}'

# Health / active calls
curl http://localhost:8001/api/threecx/status

# Hang up
curl -X POST http://localhost:8001/api/threecx/calls/<channel_id>/hangup \
     -H 'X-API-Key: secret-api-key'
```

Inbound: point the 3CX DID/routing for a trunk number at **extension 900** —
the call lands in `[from-3cx]` → Stasis → the AI answers with the greeting.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| `pjsip show registrations` → Rejected/401 | Wrong Authentication ID / password / port vs the 3CX manual-config screen |
| Registered but outbound INVITE → 403 | 3CX outbound rule doesn't match ext 900, or anti-fraud lockout — check 3CX firewall log |
| `http show status` → "Server Disabled" | `http.conf` not loaded / container was never recreated after enabling it — `docker compose up -d --force-recreate` |
| `ari show users` empty | `ari.conf` missing or not applied — reload and re-check `ari show users` |
| `curl --digest` to :8088 → 401 | Digest auth: confirm `ARI_USERNAME`/`ARI_PASSWORD` match `ari.conf` (section name = username) |
| `ari show apps` empty | Engine not running with `ASTERISK_ENABLED=true`, or `ARI_APP` mismatch |
| Call answers then drops after ~2 s, log shows `Received AudioSocket message other than hangup or audio` | The engine must not send a UUID frame back over AudioSocket (Asterisk sends *its* UUID to us). Fixed in `voice_engine/asterisk/audiosocket.py` — see `announce_uuid`. |
| Call answers but caller hears silence, state stuck at `ANSWERED`, `turn_number` 0 | The greeting must accept telephony states (`ANSWERED`/`RINGING`), not just `NEW` — outbound legs are already answered when the media bridge is built. Fixed in `voice_engine/asterisk/turn.py`. |
| `externalMedia` → HTTP 501 | `externalMedia` over UDP RTP is not implemented in Asterisk 20/21 — use `ASTERISK_MEDIA_TRANSPORT=audiosocket` |
| Call connects, no audio (`rtp` transport) | `ASTERISK_MEDIA_HOST` must be the engine's IP **as Asterisk sees it** (same LAN/VPN, `host.docker.internal` for Docker); open RTP 10000–20000 toward Asterisk |
| One-way audio (`rtp` transport) | `direct_media=no` in `pjsip.conf`; Asterisk RTP range reachable |
| Engine logs nothing at all | `LOG_LEVEL` (default `INFO`); `voice_engine` gets a console handler at import of `server.py` |
