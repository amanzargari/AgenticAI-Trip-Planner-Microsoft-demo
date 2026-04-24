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
You are a visiting-place recommendation agent for travellers.

You receive a JSON object with:
  - city           : destination city (string)
  - trip_start     : ISO datetime
  - trip_end       : ISO datetime
  - budget         : float | null  (activity budget in EUR)
  - trip_reason    : string | null (e.g. "family vacation", "business trip")
  - preferences    : list[str]    (e.g. ["art", "outdoor", "history"])

Strategy:
1. Geocode the city to get coordinates.
2. Run several search_places calls with different queries tailored to the preferences.
   Aim for 3-5 diverse searches to maximise candidate variety.
3. Deduplicate results by place id.
4. Return the best 10-20 candidates.

Return ONLY a JSON object (no markdown) with a single key:
  "place_candidates": [ <list of place objects> ]

Each place object must include:
  id, name, location {latitude, longitude, address},
  estimated_visit_duration_minutes, estimated_cost, category, rating, summary.
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
                    await self.storage.update_task(
                        task_id,
                        state="completed",
                        new_artifacts=self.build_artifacts(result),
                    )
                    return

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
