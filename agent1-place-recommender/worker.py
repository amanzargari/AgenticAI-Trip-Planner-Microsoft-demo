from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker
from pydantic import BaseModel

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from shared.models import PlaceCandidate
from tools import TOOLS, geocode_city, search_places

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


AGENT1_MAX_TOOL_STEPS = _env_int("AGENT1_MAX_TOOL_STEPS", 8)
AGENT1_MAX_TOOL_CALLS = _env_int("AGENT1_MAX_TOOL_CALLS", 12)
AGENT1_MAX_SEARCH_CALLS = _env_int("AGENT1_MAX_SEARCH_CALLS", 6)
AGENT1_TOOL_TIMEOUT_SEC = _env_float("AGENT1_TOOL_TIMEOUT_SEC", 20.0)
AGENT1_TASK_BUDGET_SEC = _env_float("AGENT1_TASK_BUDGET_SEC", 90.0)
AGENT1_MIN_CANDIDATES_EARLY_EXIT = _env_int("AGENT1_MIN_CANDIDATES_EARLY_EXIT", 12)
AGENT1_FALLBACK_MAX_QUERIES = _env_int("AGENT1_FALLBACK_MAX_QUERIES", 4)


# ── Structured output model ───────────────────────────────────────────────────

class PlaceOutput(BaseModel):
    place_candidates: list[PlaceCandidate]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Place Recommender Agent.
Goal: produce a high-quality, diverse list of places to visit for a trip.

Input (JSON):
- city: destination city
- trip_start: ISO datetime
- trip_end: ISO datetime
- budget: float | null (activity budget in EUR)
- trip_reason: string | null
- preferences: free-text user notes — treat as context, not structured tags.
  Examples: "loves art and history", "hates crowded tourist traps", "outdoor activities preferred".

Tool strategy:
1) Call geocode_city(city) once to anchor location context.
2) Call search_places multiple times with varied intent:
     - preference-focused queries based on liked interests from the preferences note
     - broad fallback query like "top attractions in <city>"
     - optional place_type when helpful, but do not rely on one type only
3) Target 3-6 searches total when possible.
4) If any tool call fails, continue with other searches.
5) Deduplicate and keep the strongest candidates (quality + diversity).

Ranking guidance:
- EXCLUDE place categories the user explicitly hates (e.g. if preferences say "hate beaches",
  omit beach/coastal attractions entirely).
- PREFER places matching liked interests and trip_reason.
- Prefer higher-rated places when ratings exist.
- Keep a mix of categories to avoid repetitive itineraries.
- Return 10-20 items when available; otherwise return as many as found.
- If no usable results are found, return an empty list.
"""


@dataclass
class PlaceRecommenderWorker(Worker[None]):

    def build_message_history(self, history: list[Message]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = ""
            for part in msg.get("parts", []):
                if part["kind"] == "text":
                    content = part["text"]
                elif part["kind"] == "data":
                    content = json.dumps(part["data"])
            messages.append({"role": role, "content": content})
        return messages

    def build_artifacts(self, result: Any) -> list[Artifact]:
        if not isinstance(result, dict):
            result = {"result": result}
        return [make_data_artifact(result)]

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self.storage.update_task(params["id"], state="canceled")

    async def run_task(self, params: TaskSendParams) -> None:
        task_id = params["id"]
        await self.storage.update_task(task_id, state="working")

        try:
            data = extract_message_data(params["message"])
            llm = get_llm_client()
            collected: list[dict[str, Any]] = []
            tool_called = False
            tool_calls_executed = 0
            search_calls_executed = 0
            seen_search_signatures: set[str] = set()
            geocode_cache: dict[str, dict[str, float]] = {}
            deadline = time.monotonic() + max(5.0, AGENT1_TASK_BUDGET_SEC)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            # Phase 1: tool calls — no response_format (Gemini rejects tool_choice=required + response_format together)
            stop_phase_1 = False
            for step in range(max(1, AGENT1_MAX_TOOL_STEPS)):
                if time.monotonic() >= deadline:
                    logger.warning("PlaceRecommender phase-1 deadline reached task_id=%s step=%d", task_id, step + 1)
                    break

                request_timeout = min(
                    max(5.0, deadline - time.monotonic()),
                    max(5.0, AGENT1_TOOL_TIMEOUT_SEC),
                )
                response = await llm.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto" if tool_called else "required",
                    timeout=request_timeout,
                )
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    tool_called = True
                    tool_calls = choice.message.tool_calls or []
                    messages.append({
                        "role": "assistant",
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in tool_calls
                        ],
                    })
                    for tc in tool_calls:
                        if tool_calls_executed >= max(1, AGENT1_MAX_TOOL_CALLS):
                            stop_phase_1 = True
                            break

                        if time.monotonic() >= deadline:
                            stop_phase_1 = True
                            break

                        name = tc.function.name
                        tool_calls_executed += 1

                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as exc:
                            tool_result: Any = {"error": f"Invalid arguments: {exc}"}
                        else:
                            try:
                                if name == "search_places":
                                    normalized = _normalize_search_places_args(args, data)
                                    if not normalized["city"]:
                                        tool_result = {"error": "search_places failed: city is required"}
                                    else:
                                        sig = _search_signature(normalized)
                                        if sig in seen_search_signatures:
                                            tool_result = []
                                        elif search_calls_executed >= max(1, AGENT1_MAX_SEARCH_CALLS):
                                            tool_result = []
                                        else:
                                            seen_search_signatures.add(sig)
                                            search_calls_executed += 1
                                            tool_result = await asyncio.wait_for(
                                                search_places(**normalized),
                                                timeout=max(3.0, AGENT1_TOOL_TIMEOUT_SEC),
                                            )
                                elif name == "geocode_city":
                                    normalized_geo = _normalize_geocode_args(args, data)
                                    city = normalized_geo["city"]
                                    if not city:
                                        tool_result = {"error": "geocode_city failed: city is required"}
                                    elif city in geocode_cache:
                                        tool_result = geocode_cache[city]
                                    else:
                                        geo = await asyncio.wait_for(
                                            geocode_city(**normalized_geo),
                                            timeout=max(3.0, AGENT1_TOOL_TIMEOUT_SEC),
                                        )
                                        geocode_cache[city] = geo
                                        tool_result = geo
                                else:
                                    tool_result = {"error": f"Unknown tool: {name}"}
                            except Exception as exc:
                                tool_result = {"error": f"{name} failed: {exc}"}

                        if name == "search_places" and isinstance(tool_result, list):
                            collected.extend(p for p in tool_result if isinstance(p, dict))

                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": tc.id,
                        })

                    if stop_phase_1:
                        break

                    if len(_coerce_candidates(collected)) >= max(1, AGENT1_MIN_CANDIDATES_EARLY_EXIT):
                        break
                else:
                    break

            # Phase 2: structured output — no tools so response_format works without conflict
            collected_candidates = _coerce_candidates(collected)
            if collected_candidates:
                result_dict = PlaceOutput(place_candidates=collected_candidates).model_dump(mode="json")
            elif time.monotonic() >= deadline:
                logger.warning("PlaceRecommender deadline reached before phase-2 parse task_id=%s", task_id)
                result_dict = await _fallback(data)
            try:
                if not collected_candidates and time.monotonic() < deadline:
                    parse_timeout = min(
                        max(5.0, deadline - time.monotonic()),
                        max(5.0, AGENT1_TOOL_TIMEOUT_SEC),
                    )
                    final = await asyncio.wait_for(
                        llm.beta.chat.completions.parse(
                            model=DEFAULT_MODEL,
                            messages=messages,
                            response_format=PlaceOutput,
                            timeout=parse_timeout,
                        ),
                        timeout=parse_timeout + 1.0,
                    )
                    parsed: Optional[PlaceOutput] = final.choices[0].message.parsed
                    if parsed and parsed.place_candidates:
                        result_dict = parsed.model_dump(mode="json")
                    else:
                        result_dict = await _fallback(data)
            except Exception:
                logger.exception("PlaceRecommender phase-2 parse failed task_id=%s", task_id)
                result_dict = PlaceOutput(place_candidates=collected_candidates).model_dump(mode="json") if collected_candidates else await _fallback(data)
            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=self.build_artifacts(result_dict),
            )

        except Exception:
            logger.exception("PlaceRecommender task crashed task_id=%s", task_id)
            await self.storage.update_task(task_id, state="failed")
            raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_candidates(raw: list[dict[str, Any]], max_items: int = 20) -> list[PlaceCandidate]:
    """Best-effort coercion of raw place dicts into PlaceCandidate, deduped."""
    seen: set[str] = set()
    out: list[PlaceCandidate] = []
    for place in raw:
        key = str(place.get("id") or place.get("name", "")).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            out.append(PlaceCandidate.model_validate(place))
        except Exception:
            pass
        if len(out) >= max_items:
            break
    return out


async def _fallback(data: dict[str, Any]) -> dict[str, Any]:
    city = str(data.get("city") or "").strip()
    if not city:
        return {"place_candidates": []}
    prefs = data.get("preferences") if isinstance(data.get("preferences"), list) else []
    queries = [f"top attractions in {city}", f"things to do in {city}"]
    queries += [f"{p} in {city}" for p in prefs if str(p).strip()]
    gathered: list[dict[str, Any]] = []
    for q in queries[: max(1, AGENT1_FALLBACK_MAX_QUERIES)]:
        try:
            rows = await asyncio.wait_for(
                search_places(query=q, city=city, radius_meters=15_000, place_type=None),
                timeout=max(3.0, AGENT1_TOOL_TIMEOUT_SEC),
            )
            if isinstance(rows, list):
                gathered.extend(p for p in rows if isinstance(p, dict))
        except Exception:
            pass
    return PlaceOutput(place_candidates=_coerce_candidates(gathered)).model_dump(mode="json")


def _normalize_search_places_args(args: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]:
    city = str(args.get("city") or request_data.get("city") or "").strip()
    query = str(args.get("query") or "").strip()
    if not query:
        query = f"top attractions in {city}" if city else "top attractions"

    radius_meters = args.get("radius_meters")
    try:
        radius = int(radius_meters) if radius_meters is not None else 15_000
    except (TypeError, ValueError):
        radius = 15_000
    if radius <= 0:
        radius = 15_000

    place_type = args.get("place_type")
    place_type = str(place_type).strip() if place_type is not None else None
    if not place_type:
        place_type = None

    return {
        "query": query,
        "city": city,
        "radius_meters": radius,
        "place_type": place_type,
    }


def _normalize_geocode_args(args: dict[str, Any], request_data: dict[str, Any]) -> dict[str, str]:
    city = str(args.get("city") or request_data.get("city") or "").strip()
    return {"city": city}


def _search_signature(normalized_args: dict[str, Any]) -> str:
    return "|".join(
        [
            str(normalized_args.get("city") or "").strip().lower(),
            str(normalized_args.get("query") or "").strip().lower(),
            str(normalized_args.get("radius_meters") or ""),
            str(normalized_args.get("place_type") or "").strip().lower(),
        ]
    )
