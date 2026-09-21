"""Place and manage 3CX calls through a running voice-engine gateway.

Prerequisites (see README.md and asterisk-config/README.md):

  * the Asterisk container is up and registered to 3CX as extension 900
    (``docker exec asterisk-test asterisk -rx "pjsip show registrations"``)
  * the gateway is running with ASTERISK_ENABLED=true

Usage:

  uv run python call.py 0730825043              # dial a number
  uv run python call.py 0730825043 --follow     # dial and watch the call state
  uv run python call.py --status                # gateway health + active calls
  uv run python call.py --hangup <channel_id>   # hang up an active call

The phone number may be written with or without a leading ``+``; the gateway
normalises it (digits only, plus any ASTERISK_OUTBOUND_PREFIX).
"""
import argparse
import os
import pathlib
import time

import httpx
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

BASE_URL = os.getenv("VOICE_ENGINE_URL", "http://127.0.0.1:8001").rstrip("/")
API_KEY = os.getenv("CALLS_API_KEY", "secret-api-key")
HEADERS = {"X-API-Key": API_KEY}


def _fail(message: str) -> None:
    """Print a helpful error and exit non-zero."""
    print(f"ERROR: {message}")
    raise SystemExit(1)


def show_status(client: httpx.Client) -> dict:
    """Print gateway health and the active-call list."""
    status = client.get(f"{BASE_URL}/api/threecx/status").json()
    print(
        f"gateway   : enabled={status['enabled']} configured={status['configured']} "
        f"connected={status['connected']}"
    )
    print(
        f"3CX       : extension={status['threecx_extension']} "
        f"endpoint={status['threecx_pjsip_endpoint']} format={status['media_format']}"
    )
    print(f"active    : {status['active_call_count']} call(s)")
    for call in status["active_calls"]:
        media = call.get("media") or {}
        print(
            f"  - {call['channel_id']} state={call['state']} to={call['phone']} "
            f"turns={call['turn_number']} {call['duration_sec']}s "
            f"audio in/out={media.get('rx_packets', 0)}/{media.get('tx_packets', 0)}"
        )
        for message in call.get("conversation", []):
            print(f"      {message['role']:9}: {message['content']}")
    return status


def dial(client: httpx.Client, phone: str) -> str:
    """Originate a call and return its ARI channel id."""
    response = client.post(
        f"{BASE_URL}/api/threecx/calls/outbound",
        headers=HEADERS,
        json={"phone": phone},
        timeout=30.0,
    )
    if response.status_code == 401:
        _fail(f"unauthorised - check CALLS_API_KEY in .env (sent '{API_KEY}')")
    if response.status_code == 503:
        _fail("3CX/Asterisk integration is disabled - set ASTERISK_ENABLED=true and restart")
    if response.status_code == 502:
        _fail(f"Asterisk/ARI rejected the call: {response.json().get('detail')}")
    response.raise_for_status()

    payload = response.json()
    print(f"dialing   : {payload['to']} via extension {payload['from_extension']}")
    print(f"channel   : {payload['channel_id']}")
    return payload["channel_id"]


def follow(client: httpx.Client, channel_id: str) -> None:
    """Poll the call until it ends, printing every state change."""
    print("\nWatching the call (Ctrl+C to stop watching; the call keeps running)...\n")
    previous = None
    try:
        while True:
            calls = client.get(f"{BASE_URL}/api/threecx/calls").json()["active_calls"]
            call = next((c for c in calls if c["channel_id"] == channel_id), None)
            if call is None:
                print("call ended")
                return
            snapshot = (call["state"], call["turn_number"])
            if snapshot != previous:
                media = call.get("media") or {}
                # Audio counters are the proof that the AI media bridge is live.
                print(
                    f"  [{call['duration_sec']:>6.1f}s] state={call['state']:<10} "
                    f"turns={call['turn_number']} "
                    f"audio in/out={media.get('rx_packets', 0)}/{media.get('tx_packets', 0)}"
                )
                for message in call.get("conversation", []):
                    print(f"             {message['role']:9}: {message['content']}")
                previous = snapshot
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopped watching")


def hangup(client: httpx.Client, channel_id: str) -> None:
    """Hang up an active call by channel id."""
    response = client.post(
        f"{BASE_URL}/api/threecx/calls/{channel_id}/hangup", headers=HEADERS, timeout=15.0
    )
    if response.status_code == 404:
        _fail(f"unknown channel id '{channel_id}' (the call may have already ended)")
    response.raise_for_status()
    print(f"hangup    : {channel_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place and manage 3CX calls through the voice-engine gateway."
    )
    parser.add_argument("phone", nargs="?", help="destination number, e.g. +919876543210")
    parser.add_argument("--follow", "-f", action="store_true", help="watch call state")
    parser.add_argument("--status", "-s", action="store_true", help="show gateway health")
    parser.add_argument("--hangup", metavar="CHANNEL_ID", help="hang up an active call")
    args = parser.parse_args()

    if not any((args.phone, args.status, args.hangup)):
        parser.print_help()
        raise SystemExit(0)

    try:
        with httpx.Client(timeout=15.0) as client:
            client.get(f"{BASE_URL}/api/threecx/status")
            if args.status:
                show_status(client)
            if args.hangup:
                hangup(client, args.hangup)
            if args.phone:
                channel_id = dial(client, args.phone)
                if args.follow:
                    follow(client, channel_id)
                else:
                    print("\nThe phone should ring now.")
                    print("Talk, then check the transcript with:")
                    print("  uv run python call.py --status")
                    print(f"  uv run python call.py --hangup {channel_id}")
    except httpx.ConnectError:
        _fail(
            f"cannot reach the gateway at {BASE_URL} - start it first:\n"
            "  uv run uvicorn voice_engine.server:app --host 127.0.0.1 --port 8001"
        )


if __name__ == "__main__":
    main()