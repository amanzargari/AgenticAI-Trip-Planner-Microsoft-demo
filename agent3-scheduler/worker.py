from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from tools import (
    TOOLS,
    estimate_travel_minutes,
    order_places_by_proximity,
    recommend_restaurant,
)

SYSTEM_PROMPT = """\
You are the Daily Scheduler Agent.
Goal: create a realistic, chronological one-day schedule.

Input (JSON):
- places: list of place objects for one day
- day_start: ISO datetime (e.g. 09:00)
- day_end: ISO datetime (e.g. 21:00)
- food_budget_per_day: float | null
- preferences: list[str]

Scheduling policy:
1) Order places using order_places_by_proximity.
2) Use estimate_travel_minutes between consecutive visits.
3) Keep events chronological, non-overlapping, and within [day_start, day_end].
4) Insert lunch in [12:00, 14:00] and dinner in [19:00, 21:00] when time allows.
5) For each meal slot, call recommend_restaurant and select first result if available.
6) Budget per meal = food_budget_per_day / 2 when budget exists.
7) If no places can fit, return empty events list for that date.

Output rules (STRICT):
- Return ONLY JSON (no markdown or prose).
- Return exactly:
{
    "date": "YYYY-MM-DD",
    "events": [
        {
            "type": "visit",
            "start_time": "ISO datetime",
            "end_time": "ISO datetime",
            "place": { ... }
        },
        {
            "type": "meal",
            "time": "ISO datetime",
            "restaurant": { ... },
            "meal_slot": "lunch"
        }
    ]
}
"""


@dataclass
class SchedulerWorker(Worker[None]):

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

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            for _ in range(20):
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
                        args = json.loads(tc.function.arguments)
                        name = tc.function.name
                        if name == "order_places_by_proximity":
                            tool_result = order_places_by_proximity(**args)
                        elif name == "estimate_travel_minutes":
                            tool_result = estimate_travel_minutes(**args)
                        elif name == "recommend_restaurant":
                            tool_result = await recommend_restaurant(**args)
                        else:
                            tool_result = {"error": f"Unknown tool: {name}"}
                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tc.id,
                            }
                        )
                else:
                    raw = choice.message.content or "{}"
                    result = _parse_json(raw)
                    await self.storage.update_task(
                        task_id,
                        state="completed",
                        new_artifacts=self.build_artifacts(result),
                    )
                    return

            await self.storage.update_task(task_id, state="failed")

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
        return {"date": "", "events": []}
