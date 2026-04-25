from __future__ import annotations

import json
import logging
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
- preferences: list[str]

Tool strategy:
1) Call geocode_city(city) once to anchor location context.
2) Call search_places multiple times with varied intent:
     - preference-focused queries (art, museums, outdoors, nightlife, etc.)
     - broad fallback query like "top attractions in <city>"
     - optional place_type when helpful, but do not rely on one type only
3) Target 3-6 searches total when possible.
4) If any tool call fails, continue with other searches.
5) Deduplicate and keep the strongest candidates (quality + diversity).

Ranking guidance:
- Prefer places matching preferences and trip_reason.
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

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            # Phase 1: tool calls — no response_format (Gemini rejects tool_choice=required + response_format together)
            for _ in range(12):
                response = await llm.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto" if tool_called else "required",
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
                        name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as exc:
                            tool_result: Any = {"error": f"Invalid arguments: {exc}"}
                        else:
                            try:
                                if name == "search_places":
                                    tool_result = await search_places(**args)
                                elif name == "geocode_city":
                                    tool_result = await geocode_city(**args)
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
                else:
                    break

            # Phase 2: structured output — no tools so response_format works without conflict
            try:
                final = await llm.beta.chat.completions.parse(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    response_format=PlaceOutput,
                )
                parsed: Optional[PlaceOutput] = final.choices[0].message.parsed
                if parsed and parsed.place_candidates:
                    result_dict = parsed.model_dump(mode="json")
                elif collected:
                    result_dict = PlaceOutput(place_candidates=_coerce_candidates(collected)).model_dump(mode="json")
                else:
                    result_dict = await _fallback(data)
            except Exception:
                logger.exception("PlaceRecommender phase-2 parse failed task_id=%s", task_id)
                result_dict = (
                    PlaceOutput(place_candidates=_coerce_candidates(collected)).model_dump(mode="json")
                    if collected else await _fallback(data)
                )
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
    for q in queries[:6]:
        try:
            rows = await search_places(query=q, city=city)
            if isinstance(rows, list):
                gathered.extend(p for p in rows if isinstance(p, dict))
        except Exception:
            pass
    return PlaceOutput(place_candidates=_coerce_candidates(gathered)).model_dump(mode="json")
