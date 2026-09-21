"""3CX (Asterisk ARI) calling API routes.

Same surface as the Twilio calling feature, but calls are originated through
the local Asterisk gateway (ARI), which registers to 3CX as extension 900:

    POST /api/threecx/calls/outbound   (alias /api/asterisk/calls/outbound)
      {"phone": "+919876543210"}   header X-API-Key: <CALLS_API_KEY>
      -> ARI create+dial -> PJSIP/3cx -> 3CX ext 900 -> outbound rule -> Airtel

    GET  /api/threecx/status           (alias /api/asterisk/status)
    GET  /api/threecx/calls
    POST /api/threecx/calls/{channel_id}/hangup

The Twilio calling feature remains available in parallel -- this router does
not touch any of its endpoints.
"""
import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from .ari import AriError
from .calls import asterisk_call_manager
from .config import get_settings

logger = logging.getLogger("voice_engine.asterisk.routes")
router = APIRouter(tags=["threecx"])


class OutboundCallRequest(BaseModel):
    phone: str = Field(..., description="Destination phone number, e.g. +919876543210")
    caller_id: Optional[str] = Field(
        None, description="Optional outbound caller-id override (Asterisk/3CX permitting)"
    )
    customer_id: Optional[str] = Field(
        None, description="Optional app-backend correlation id (logged only)"
    )


class OutboundCallResponse(BaseModel):
    status: str
    channel_id: str
    to: str
    from_extension: str


def _authorize(x_api_key: Optional[str]) -> None:
    settings = get_settings()
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        logger.warning("Unauthorized access attempt to a 3CX call endpoint")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def _ensure_enabled() -> None:
    settings = get_settings()
    if not settings.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Asterisk/3CX integration is disabled (set ASTERISK_ENABLED=true)",
        )
    if not asterisk_call_manager.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Asterisk/3CX integration is not configured (ARI_* environment)",
        )
    if asterisk_call_manager.ari is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Asterisk ARI client is not connected yet",
        )


@router.post("/api/threecx/calls/outbound", response_model=OutboundCallResponse)
@router.post("/api/asterisk/calls/outbound", response_model=OutboundCallResponse)
async def trigger_outbound_call(
    payload: OutboundCallRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> OutboundCallResponse:
    """Initiate an outbound call: NestJS/app -> engine -> ARI -> 3CX -> Airtel."""
    _authorize(x_api_key)
    _ensure_enabled()
    settings = get_settings()
    try:
        session = await asterisk_call_manager.originate_call(
            payload.phone, caller_id=payload.caller_id or ""
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    except AriError as exc:
        logger.error(
            "ARI failed to dial", extra={"phone": payload.phone, "error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ARI dial failed: {exc}",
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ARI dial failed: {exc}",
        )
    logger.info(
        "Outbound 3CX call requested",
        extra={
            "channel_id": session.channel_id,
            "phone": session.phone,
            "customer_id": payload.customer_id,
        },
    )
    return OutboundCallResponse(
        status=session.state.value.lower(),
        channel_id=session.channel_id,
        to=session.phone,
        from_extension=settings.threecx_extension,
    )


@router.get("/api/threecx/status")
@router.get("/api/asterisk/status")
async def threecx_status() -> dict:
    """Integration health: configured/connected flags plus active calls."""
    return asterisk_call_manager.get_status()


@router.get("/api/threecx/calls")
async def list_calls() -> dict:
    """List the currently active 3CX calls with their lifecycle state."""
    status_info = asterisk_call_manager.get_status()
    return {
        "active_call_count": status_info["active_call_count"],
        "active_calls": status_info["active_calls"],
    }


@router.post("/api/threecx/calls/{channel_id}/hangup")
async def hangup_call(
    channel_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> dict:
    """Hang up an active call by ARI channel id."""
    _authorize(x_api_key)
    _ensure_enabled()
    try:
        ok = await asterisk_call_manager.hangup(channel_id)
    except AriError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ARI hangup failed: {exc}",
        )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown channel id (call may have already ended)",
        )
    return {"status": "hangup_requested", "channel_id": channel_id}

