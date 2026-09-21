"""Minimal async client for the Asterisk REST Interface (ARI).

Only the operations the calling feature needs are implemented: channel
create/dial/answer/hangup, mixing bridges, and external media channels.
Events arrive over the ``/ari/events`` WebSocket, which reconnects with
exponential backoff so the gateway survives Asterisk restarts.

Authentication: HTTP requests use HTTP Basic auth, which is what Asterisk
serves when the ari.conf ``password_format`` is ``plain`` (the default). With
crypt-hashed passwords Asterisk switches to digest -- set ``ARI_AUTH=digest``
in the environment to match. The events WebSocket always uses the ``api_key``
query parameter (``username:password``), which works for both schemes.
"""
import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx
import websockets

logger = logging.getLogger("voice_engine.asterisk.ari")


class AriError(RuntimeError):
    """An ARI REST call failed (non-2xx or transport error)."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


Handler = Callable[[dict], Awaitable[None]]


class AriClient:
    """Async ARI client: REST + events websocket with reconnect/backoff."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        app: str,
    ) -> None:
        self.base_url = (base_url or "http://127.0.0.1:8088").rstrip("/")
        self.username = username
        self.password = password
        self.app = app
        self._client: Optional[httpx.AsyncClient] = None
        self._ws = None
        self._events_task: Optional[asyncio.Task] = None
        self._handlers: Dict[str, List[Handler]] = {}
        self._stop = False
        self.connected = asyncio.Event()

    # ------------------------------------------------------------ transport

    def ws_url(self) -> str:
        base = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        creds = f"{quote(self.username, safe='')}:{quote(self.password, safe='')}"
        return f"{base}/ari/events?app={quote(self.app, safe='')}&api_key={creds}"

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            auth = (
                httpx.DigestAuth(self.username, self.password)
                if os.environ.get('ARI_AUTH', 'basic').strip().lower() == 'digest'
                else httpx.BasicAuth(self.username, self.password)
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                auth=auth,
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        self._stop = True
        self.connected.clear()
        if self._events_task is not None:
            self._events_task.cancel()
            try:
                await self._events_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass
            self._events_task = None
        self._ws = None
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # ---------------------------------------------------------------- events

    def on(self, event_type: str, handler: Handler) -> None:
        """Register an async handler for an ARI event type (e.g. StasisStart)."""
        self._handlers.setdefault(event_type, []).append(handler)

    async def start_events(self) -> None:
        """Start the events websocket task (idempotent)."""
        if self._events_task is not None and not self._events_task.done():
            return
        self._events_task = asyncio.create_task(self._events_loop())

    async def _events_loop(self) -> None:
        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(
                    self.ws_url(), max_size=2**23, ping_interval=20
                ) as ws:
                    self._ws = ws
                    self.connected.set()
                    backoff = 1.0
                    logger.info("ARI events websocket connected (app=%s)", self.app)
                    async for raw in ws:
                        if self._stop:
                            break
                        try:
                            event = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(event, dict) and event.get("type"):
                            await self._dispatch(event)
                        if self._stop:
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any failure
                logger.warning("ARI events connection error: %s -- reconnecting", exc)
            finally:
                self._ws = None
                self.connected.clear()
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _dispatch(self, event: dict) -> None:
        for handler in self._handlers.get(event.get("type", ""), []):
            try:
                await handler(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ARI event handler error for %s: %s", event.get("type"), exc
                )

    # ------------------------------------------------------------------ REST

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> Any:
        client = self._http()
        try:
            resp = await client.request(method, path, params=params, json=json_body)
        except httpx.HTTPError as exc:
            raise AriError(f"ARI request failed: {exc}") from exc
        if resp.status_code in (200, 201):
            if not resp.content:
                return {}
            try:
                return resp.json()
            except ValueError:
                return {}
        if resp.status_code == 204:
            return {}
        raise AriError(
            f"ARI {method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}",
            status_code=resp.status_code,
        )

    # -- channels -----------------------------------------------------------

    async def create_channel(
        self,
        *,
        endpoint: str,
        app_args: str = "",
        channel_id: Optional[str] = None,
        caller_id: Optional[str] = None,
    ) -> dict:
        """Create an unanswered channel inside the Stasis app (create+dial flow)."""
        params: Dict[str, Any] = {"endpoint": endpoint, "app": self.app}
        if app_args:
            params["appArgs"] = app_args
        if channel_id:
            params["channelId"] = channel_id
        if caller_id:
            params["callerId"] = caller_id
        return await self._request("POST", "/ari/channels/create", params=params)

    async def originate_channel(
        self,
        *,
        endpoint: str,
        app_args: str = "",
        channel_id: Optional[str] = None,
        caller_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """Classic ARI originate (alternative to create+dial; kept for tooling)."""
        params: Dict[str, Any] = {"endpoint": endpoint, "app": self.app}
        if app_args:
            params["appArgs"] = app_args
        if channel_id:
            params["channelId"] = channel_id
        if caller_id:
            params["callerId"] = caller_id
        if timeout:
            params["timeout"] = timeout
        return await self._request("POST", "/ari/channels", params=params)

    async def dial(self, channel_id: str, timeout: Optional[int] = None) -> None:
        params: Dict[str, Any] = {}
        if timeout:
            params["timeout"] = timeout
        await self._request(
            "POST", f"/ari/channels/{quote(channel_id, safe='')}/dial", params=params
        )

    async def answer(self, channel_id: str) -> None:
        await self._request("POST", f"/ari/channels/{quote(channel_id, safe='')}/answer")

    async def hangup(self, channel_id: str, reason: str = "normal") -> None:
        await self._request(
            "DELETE",
            f"/ari/channels/{quote(channel_id, safe='')}",
            params={"reason": reason},
        )

    async def get_channel(self, channel_id: str) -> dict:
        return await self._request("GET", f"/ari/channels/{quote(channel_id, safe='')}")

    async def set_channel_var(self, channel_id: str, variable: str, value: str) -> None:
        await self._request(
            "POST",
            f"/ari/channels/{quote(channel_id, safe='')}/variable",
            params={"variable": variable, "value": value},
        )

    # -- bridges -------------------------------------------------------------

    async def create_bridge(self, name: str = "", bridge_type: str = "mixing") -> dict:
        params: Dict[str, Any] = {"type": bridge_type}
        if name:
            params["name"] = name
        return await self._request("POST", "/ari/bridges", params=params)

    async def add_channel_to_bridge(
        self, bridge_id: str, channel_id: str, role: Optional[str] = None
    ) -> None:
        params: Dict[str, Any] = {"channel": channel_id}
        if role:
            params["role"] = role
        await self._request(
            "POST", f"/ari/bridges/{quote(bridge_id, safe='')}/addChannel", params=params
        )

    async def remove_channel_from_bridge(self, bridge_id: str, channel_id: str) -> None:
        await self._request(
            "POST",
            f"/ari/bridges/{quote(bridge_id, safe='')}/removeChannel",
            params={"channel": channel_id},
        )

    async def destroy_bridge(self, bridge_id: str) -> None:
        await self._request("DELETE", f"/ari/bridges/{quote(bridge_id, safe='')}")

    # -- external media -------------------------------------------------------

    async def create_external_media(
        self,
        *,
        external_host: str,
        fmt: str = "ulaw",
        channel_id: Optional[str] = None,
        connection_type: str = "client",
        encapsulation: str = "none",
        transport: str = "udp",
        data: Optional[str] = None,
    ) -> dict:
        """Create an externalMedia channel streaming RTP to ``external_host``.

        Uses encapsulation=none / transport=udp -- the combination supported by
        every Asterisk release since ExternalMedia was introduced (16.6+).
        """
        params: Dict[str, Any] = {
            "app": self.app,
            "external_host": external_host,
            "encapsulation": encapsulation,
            "transport": transport,
            "connection_type": connection_type,
            "format": fmt,
        }
        if data:
            params["data"] = data
        if channel_id:
            params["channelId"] = channel_id
        data = await self._request("POST", "/ari/channels/externalMedia", params=params)
        if isinstance(data, dict):
            resolved = data.get("id") or data.get("channelid") or channel_id
            if resolved:
                data = {**data, "id": resolved}
        return data


