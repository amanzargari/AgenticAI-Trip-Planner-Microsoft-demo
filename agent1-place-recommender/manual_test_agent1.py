from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.a2a_utils import call_agent

DEFAULT_PAYLOAD: dict[str, Any] = {
    "city": "Milan, Italy",
    "trip_start": "2026-06-10T09:00:00",
    "trip_end": "2026-06-15T21:00:00",
    "budget": 560,
    "trip_reason": "friends",
    "preferences": ["art", "outdoor", "food", "museums", "nightlife"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Agent1 (Place Recommender) manually with your own payload."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("AGENT1_URL", "http://localhost:8001"),
        help="Agent1 A2A URL (default: AGENT1_URL or http://localhost:8001)",
    )
    parser.add_argument(
        "--payload-file",
        type=Path,
        default=Path(__file__).resolve().parent / "sample-input.json",
        help="Path to JSON payload file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for the A2A task",
    )
    return parser.parse_args()


def _load_payload(payload_file: Path) -> dict[str, Any]:
    if not payload_file.exists():
        return dict(DEFAULT_PAYLOAD)

    with payload_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Payload file must contain a JSON object")

    return payload


async def _run(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return await call_agent(
        url=url,
        data=payload,
        timeout=timeout,
        poll_interval=1.0,
    )


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    args = _parse_args()

    payload = _load_payload(args.payload_file)

    print(f"Using agent URL: {args.url}")
    print(f"Using payload file: {args.payload_file}")
    print("Sending payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    try:
        result = asyncio.run(_run(args.url, payload, args.timeout))
    except Exception as exc:
        print(f"\nAgent1 call failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    print("\nAgent1 response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    candidates = result.get("place_candidates") if isinstance(result, dict) else None
    if isinstance(candidates, list):
        print(f"\nplace_candidates count: {len(candidates)}")


if __name__ == "__main__":
    main()
