from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Annotated, Any, Optional, Union

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker
from pydantic import BaseModel, Field

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from shared.models import MealEvent, VisitEvent
from tools import (
    TOOLS,
    estimate_travel_minutes,
    hydrate_places, 
    order_places_by_proximity,
    recommend_restaurant,
)

logger = logging.getLogger(__name__)


# ── Structured output model ───────────────────────────────────────────────────

# Discriminated union so the LLM knows which event shape to produce
ScheduleEvent = Annotated[Union[VisitEvent, MealEvent], Field(discriminator="type")]


class SchedulerOutput(BaseModel):
    date: str                        # YYYY-MM-DD
    events: list[ScheduleEvent]


# ── System prompt ─────────────────────────────────────────────────────────────

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
1) Call order_places_by_proximity to get an efficient visit order.
2) Use estimate_travel_minutes between consecutive visits.
3) Keep events chronological, non-overlapping, and within [day_start, day_end].
4) Insert lunch in [12:00, 14:00] and dinner in [19:00, 21:00] when time allows.
5) For each meal slot, call recommend_restaurant and select the first result if available.
   If recommend_restaurant returns an error or empty list, skip that meal event.
6) Budget per meal = food_budget_per_day / 2 when budget exists.
7) If no places can fit, return an empty events list.

Important:
- If a tool call returns {"error": ...}, ignore it gracefully and continue.
- Do NOT retry a tool that already returned an error.
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

            day_date = str(data.get("day_start", ""))[:10]

            # Resolve Place IDs → full place objects so the LLM and tools
            # always see a uniform shape, regardless of what the orchestrator sent.
            hydrated = await hydrate_places(data.get("places"))
            if not hydrated:
                logger.warning("Scheduler: no schedulable places after hydration task_id=%s", task_id)
            data["places"] = hydrated

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            # Phase 1: tool calls — no response_format (Gemini rejects tool_choice=required + response_format together)
            tool_called = False
            for step in range(20):
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
                            tool_result: Any = {"error": f"Bad arguments for {name}: {exc}"}
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tc.id,
                            })
                            continue

                        try:
                            if name == "order_places_by_proximity":
                                tool_result = order_places_by_proximity(**args)
                            elif name == "estimate_travel_minutes":
                                tool_result = estimate_travel_minutes(**args)
                            elif name == "recommend_restaurant":
                                tool_result = await recommend_restaurant(**args)
                            else:
                                tool_result = {"error": f"Unknown tool: {name}"}
                        except Exception as exc:
                            logger.warning("Scheduler tool %s failed task_id=%s: %s", name, task_id, exc)
                            tool_result = {"error": f"{name} failed: {exc}", "restaurants": []}

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
                    response_format=SchedulerOutput,
                )
                parsed: Optional[SchedulerOutput] = final.choices[0].message.parsed
                result = parsed.model_dump(mode="json") if parsed else {"date": day_date, "events": []}
            except Exception:
                logger.warning("Scheduler phase-2 parse failed task_id=%s", task_id)
                result = {"date": day_date, "events": []}
            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=self.build_artifacts(result),
            )

        except Exception:
            logger.exception("Scheduler task crashed task_id=%s", task_id)
            await self.storage.update_task(task_id, state="failed")
            raise
