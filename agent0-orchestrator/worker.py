from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from tools import TOOLS, cluster_places, recommend_places, schedule_day

SYSTEM_PROMPT = """\
You are the Trip Planner Orchestrator. You coordinate specialist agents to build a complete trip itinerary.

You receive a JSON object with:
  - city         : string
  - trip_start   : ISO datetime
  - trip_end     : ISO datetime
  - budget       : {"total_budget": float, "currency": "EUR"} | null
  - trip_reason  : string | null
  - preferences  : list[str]

Budget split (when provided):
  activity_budget      = 0.70 × total_budget
  food_budget_total    = 0.30 × total_budget
  food_budget_per_day  = food_budget_total / num_days
  num_days             = ceil((trip_end - trip_start) in hours / 24)

You have THREE tools – use them via tool-calls (not pipeline):

  1. recommend_places(city, trip_start, trip_end, activity_budget, trip_reason, preferences)
     → returns {place_candidates: [...]}

  2. cluster_places(trip_start, trip_end, place_candidates)
     → returns {clustered_place_candidates: [[day1_places], [day2_places], ...]}

  3. schedule_day(places, day_start, day_end, food_budget_per_day, preferences)
     → returns a DailySchedule {date, events}
     Call this ONCE PER CLUSTER (i.e., once per day).

Workflow:
  a) Call recommend_places.
  b) Call cluster_places with the candidates.
  c) For each cluster, call schedule_day.
     - day_start = trip_start date + 09:00 (local)
     - day_end   = same date + 21:00
     - For day i (0-indexed): date = trip_start.date + i days

After all schedule_day calls are done, assemble the final itinerary and return ONLY a JSON object:
{
  "city": "...",
  "trip_start": "ISO",
  "trip_end": "ISO",
  "total_budget": float | null,
  "schedules": [ <DailySchedule>, ... ]
}

Do NOT hard-code a pipeline: use the tool results to decide next steps adaptively.
"""


@dataclass
class OrchestratorWorker(Worker[None]):

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

            # Generous iteration budget: recommend + cluster + up to 14 days × schedule_day
            for _ in range(40):
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
                        try:
                            if name == "recommend_places":
                                tool_result = await recommend_places(**args)
                            elif name == "cluster_places":
                                tool_result = await cluster_places(**args)
                            elif name == "schedule_day":
                                tool_result = await schedule_day(**args)
                            else:
                                tool_result = {"error": f"Unknown tool: {name}"}
                        except Exception as exc:
                            tool_result = {"error": str(exc)}

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
        return {"error": "Could not parse orchestrator output", "raw": text}
