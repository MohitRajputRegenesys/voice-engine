"""Twilio webhook + outbound call API routes (inside the voice engine).

These routes let the voice-engine gateway itself answer inbound calls with TwiML
and trigger outbound calls -- the same surface as the reference calling agent,
but implemented within this package and driven by the engine's own providers.
"""
import logging
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient

from .config import get_settings
from .twiml import outbound_response, public_https_url, voice_response

logger = logging.getLogger("voice_engine.twilio.routes")
router = APIRouter(tags=["twilio"])


class OutboundCallRequest(BaseModel):
    to: str = Field(..., description="Destination phone number in E.164 format (e.g. +1234567890)")
    from_number: Optional[str] = Field(None, alias="from", description="Source phone number in E.164 format")
    custom_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom parameters")


class OutboundCallResponse(BaseModel):
    status: str
    call_sid: str
    to: str
    from_number: str


async def validate_twilio_signature(request: Request) -> bool:
    """Validate the incoming X-Twilio-Signature against the request URL and form body."""
    settings = get_settings()
    if settings.env == "test":
        return True

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        logger.warning("Missing X-Twilio-Signature header")
        return False

    validator = RequestValidator(settings.auth_token)
    url = str(request.url)

    # Twilio sends form data (application/x-www-form-urlencoded)
    form_data = await request.form()
    params = {k: v for k, v in form_data.items()}

    is_valid = validator.validate(url, params, signature)
    if not is_valid and settings.voice_url:
        # Fallback check if the public URL differs from request.url
        is_valid = validator.validate(settings.voice_url, params, signature)

    return is_valid


@router.post("/api/twilio/voice")
async def handle_voice_webhook(request: Request) -> Response:
    """Twilio Voice Webhook: validate signature and return TwiML with Media Stream connection."""
    if not await validate_twilio_signature(request):
        logger.warning("Twilio signature validation failed for /api/twilio/voice endpoint")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Twilio signature",
        )

    form_data = await request.form()
    call_sid = form_data.get("CallSid", "")
    settings = get_settings()

    logger.info("Incoming call webhook received", extra={"call_sid": call_sid})

    twiml_xml = voice_response(stream_url=settings.base_url, call_sid=call_sid)
    return Response(content=twiml_xml, media_type="application/xml")


@router.post("/api/twilio/status")
async def handle_status_webhook(request: Request) -> Response:
    """Twilio Call Status Webhook: validate signature and log call lifecycle transitions."""
    if not await validate_twilio_signature(request):
        logger.warning("Twilio signature validation failed for /api/twilio/status endpoint")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Twilio signature",
        )

    form_data = await request.form()
    logger.info(
        "Twilio call status update",
        extra={
            "call_sid": form_data.get("CallSid", ""),
            "call_status": form_data.get("CallStatus", ""),
            "duration_sec": form_data.get("CallDuration", ""),
        },
    )

    return Response(content="<Response></Response>", media_type="application/xml")


@router.post("/api/calls/outbound", response_model=OutboundCallResponse)
async def trigger_outbound_call(
    payload: OutboundCallRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> OutboundCallResponse:
    """Initiate an outbound call via the Twilio REST API."""
    settings = get_settings()
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        logger.warning("Unauthorized access attempt to outbound call endpoint")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    from_phone = payload.from_number or settings.phone_number
    if not from_phone:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TWILIO_PHONE_NUMBER is not configured",
        )

    twiml_content = outbound_response(stream_url=settings.base_url)
    # The status callback must be a public https:// URL -- same rule as the
    # reference calling agent (TWILIO_BASE_URL must be the public ngrok host).
    status_callback_url = public_https_url(settings.base_url, "/api/twilio/status")

    try:
        twilio_client = TwilioClient(settings.account_sid, settings.auth_token)
        call = twilio_client.calls.create(
            to=payload.to,
            from_=from_phone,
            twiml=twiml_content,
            status_callback=status_callback_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )

        logger.info("Outbound call initiated", extra={"call_sid": call.sid, "to": payload.to})
        return OutboundCallResponse(
            status="initiated",
            call_sid=call.sid,
            to=payload.to,
            from_number=from_phone,
        )

    except Exception as e:
        logger.error("Failed to create Twilio outbound call", extra={"error": str(e), "to": payload.to})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initiate call: {str(e)}",
        )