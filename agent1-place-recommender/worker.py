from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from tools import TOOLS, geocode_city, search_places

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

Output rules (STRICT):
- Return ONLY JSON. No markdown, no prose, no questions.
- Return this exact top-level shape:
{
    "place_candidates": [
        {
            "id": "...",
            "name": "...",
            "location": {"latitude": 0.0, "longitude": 0.0, "address": "..."},
            "estimated_visit_duration_minutes": 60,
            "estimated_cost": null,
            "category": "...",
            "rating": 4.5,
            "summary": "..."
        }
    ]
}
- Return 10-20 items when available; otherwise return as many as found.
- If no usable results are found, return {"place_candidates": []}.
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
            collected_candidates: list[dict[str, Any]] = []

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            for _ in range(12):
                response = await llm.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    tool_calls = choice.message.tool_calls or []
                    messages.append(
                        {
                            "role": "assistant",
                            "content": choice.message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )
                    for tc in tool_calls:
                        name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as exc:
                            tool_result = {
                                "error": f"Invalid arguments for tool {name}: {exc}"
                            }
                        else:
                            try:
                                if name == "search_places":
                                    tool_result = await search_places(**args)
                                elif name == "geocode_city":
                                    tool_result = await geocode_city(**args)
                                else:
                                    tool_result = {"error": f"Unknown tool: {name}"}
                            except Exception as exc:
                                tool_result = {"error": f"Tool {name} failed: {exc}"}

                        if name == "search_places" and isinstance(tool_result, list):
                            collected_candidates.extend(
                                p for p in tool_result if isinstance(p, dict)
                            )

                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tc.id,
                            }
                        )
                else:
                    raw = choice.message.content or "{}"
                    result = _coerce_place_result(
                        _parse_json(raw),
                        fallback_candidates=collected_candidates,
                    )
                    if not result.get("place_candidates"):
                        result["place_candidates"] = await _fallback_search_candidates(data)
                    await self.storage.update_task(
                        task_id,
                        state="completed",
                        new_artifacts=self.build_artifacts(result),
                    )
                    return

            if not collected_candidates:
                collected_candidates = await _fallback_search_candidates(data)

            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=self.build_artifacts(
                    {"place_candidates": _dedupe_places(collected_candidates)}
                ),
            )
            return

        except Exception:
            await self.storage.update_task(task_id, state="failed")
            raise


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"place_candidates": []}


def _coerce_place_result(
    result: Any,
    fallback_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"place_candidates": _dedupe_places(fallback_candidates)}

    place_candidates = result.get("place_candidates")
    if not isinstance(place_candidates, list):
        result["place_candidates"] = _dedupe_places(fallback_candidates)
        return result

    valid_candidates = [p for p in place_candidates if isinstance(p, dict)]
    if valid_candidates:
        result["place_candidates"] = _dedupe_places(valid_candidates)
        return result

    result["place_candidates"] = _dedupe_places(fallback_candidates)
    return result


def _dedupe_places(
    places: list[dict[str, Any]],
    max_items: int = 20,
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for place in places:
        place_id = str(place.get("id") or "").strip()
        if place_id:
            key = f"id:{place_id}"
        else:
            loc = place.get("location", {})
            lat = loc.get("latitude") if isinstance(loc, dict) else None
            lng = loc.get("longitude") if isinstance(loc, dict) else None
            name = str(place.get("name") or "").strip().lower()
            key = f"name:{name}|lat:{lat}|lng:{lng}"

        if key in seen:
            continue

        seen.add(key)
        deduped.append(place)
        if len(deduped) >= max_items:
            break

    return deduped


async def _fallback_search_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    city = str(data.get("city") or "").strip()
    if not city:
        return []

    preferences = data.get("preferences") if isinstance(data.get("preferences"), list) else []
    query_candidates = [
        f"top attractions in {city}",
        f"things to do in {city}",
        f"popular museums in {city}",
    ]
    query_candidates.extend(
        f"{str(pref).strip()} in {city}"
        for pref in preferences
        if str(pref).strip()
    )

    seen_queries: set[str] = set()
    queries: list[str] = []
    for q in query_candidates:
        norm = q.strip().lower()
        if norm and norm not in seen_queries:
            seen_queries.add(norm)
            queries.append(q)

    gathered: list[dict[str, Any]] = []
    for query in queries[:6]:
        try:
            rows = await search_places(query=query, city=city)
        except Exception:
            continue
        if isinstance(rows, list):
            gathered.extend(p for p in rows if isinstance(p, dict))

    return _dedupe_places(gathered)
