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
You are a daily trip scheduler. You create a chronological schedule for one day.

You receive a JSON object with:
  - places               : list of place objects to visit that day
  - day_start            : ISO datetime (e.g. "2026-06-10T09:00:00")
  - day_end              : ISO datetime (e.g. "2026-06-10T21:00:00")
  - food_budget_per_day  : float | null  (EUR per person for all meals)
  - preferences          : list[str]

Scheduling rules:
  • Start from day_start (typically 09:00).
  • Use order_places_by_proximity to get an efficient visit order.
  • Use estimate_travel_minutes between consecutive places.
  • Insert LUNCH between 12:00 and 14:00 and DINNER between 19:00 and 21:00.
  • Call recommend_restaurant for each meal slot; pick the first restaurant returned.
  • Each visit: start_time = previous_end + travel_time, end_time = start_time + estimated_visit_duration_minutes.
  • Do not exceed day_end.
  • Budget per meal = food_budget_per_day / 2 if food_budget_per_day is set, else null.

Return ONLY a JSON object (no markdown) with:
{
  "date": "YYYY-MM-DD",
  "events": [
    {
      "type": "visit",
      "start_time": "ISO datetime",
      "end_time": "ISO datetime",
      "place": { <full place object> }
    },
    {
      "type": "meal",
      "time": "ISO datetime",
      "restaurant": { <restaurant object> },
      "meal_slot": "lunch" | "dinner"
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
