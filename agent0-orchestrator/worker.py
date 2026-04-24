from __future__ import annotations

import json
import logging
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


logger = logging.getLogger(__name__)

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

Important output rules:
    - Never ask follow-up questions.
    - Never return plain text explanations.
    - If there are no candidates or schedules, still return the JSON object above with "schedules": [].

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
            logger.info(
                "Orchestrator task started task_id=%s payload_keys=%s",
                task_id,
                sorted(data.keys()),
            )
            llm = get_llm_client()

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]
            fallback_schedules: list[dict[str, Any]] = []

            # Generous iteration budget: recommend + cluster + up to 14 days × schedule_day
            for step in range(40):
                response = await llm.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                choice = response.choices[0]
                logger.info(
                    "Orchestrator LLM step task_id=%s step=%d finish_reason=%s",
                    task_id,
                    step + 1,
                    choice.finish_reason,
                )

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
                            logger.exception(
                                "Orchestrator tool arguments JSON decode failed task_id=%s tool=%s args=%s",
                                task_id,
                                name,
                                tc.function.arguments,
                            )
                            tool_result = {
                                "error": f"Invalid arguments for tool {name}: {exc}"
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": json.dumps(tool_result),
                                    "tool_call_id": tc.id,
                                }
                            )
                            continue

                        logger.info(
                            "Orchestrator tool call task_id=%s tool=%s arg_keys=%s",
                            task_id,
                            name,
                            sorted(args.keys()),
                        )
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
                            logger.exception(
                                "Orchestrator tool execution failed task_id=%s tool=%s args=%s",
                                task_id,
                                name,
                                args,
                            )
                            tool_result = {"error": str(exc)}

                        if isinstance(tool_result, dict) and tool_result.get("error"):
                            logger.warning(
                                "Orchestrator tool returned error task_id=%s tool=%s error=%s",
                                task_id,
                                name,
                                tool_result.get("error"),
                            )
                        else:
                            logger.info(
                                "Orchestrator tool succeeded task_id=%s tool=%s result_keys=%s",
                                task_id,
                                name,
                                sorted(tool_result.keys()) if isinstance(tool_result, dict) else "non-dict",
                            )
                            if name == "schedule_day" and isinstance(tool_result, dict):
                                fallback_schedules.append(tool_result)

                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tc.id,
                            }
                        )
                else:
                    raw = choice.message.content or "{}"
                    parsed = _parse_json(raw)
                    result = _normalize_orchestrator_result(
                        parsed,
                        request_data=data,
                        fallback_schedules=fallback_schedules,
                    )
                    if isinstance(parsed, dict) and parsed.get("error"):
                        logger.warning(
                            "Orchestrator produced error-shaped result task_id=%s error=%s",
                            task_id,
                            parsed.get("error"),
                        )
                    logger.info(
                        "Orchestrator task completed task_id=%s result_keys=%s",
                        task_id,
                        sorted(result.keys()) if isinstance(result, dict) else "non-dict",
                    )
                    await self.storage.update_task(
                        task_id,
                        state="completed",
                        new_artifacts=self.build_artifacts(result),
                    )
                    return

            logger.error("Orchestrator task exceeded max iterations task_id=%s", task_id)
            fallback_result = _normalize_orchestrator_result(
                {},
                request_data=data,
                fallback_schedules=fallback_schedules,
            )
            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=self.build_artifacts(fallback_result),
            )
            return

        except Exception:
            logger.exception("Orchestrator task crashed task_id=%s", task_id)
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
        logger.warning(
            "Orchestrator final response was not valid JSON; excerpt=%s",
            text[:240],
        )
        return {"error": "Could not parse orchestrator output", "raw": text}


def _normalize_orchestrator_result(
    result: Any,
    request_data: dict[str, Any],
    fallback_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    src = result if isinstance(result, dict) else {}

    fallback_city = str(request_data.get("city") or "")
    fallback_trip_start = str(request_data.get("trip_start") or "")
    fallback_trip_end = str(request_data.get("trip_end") or "")
    fallback_total_budget = _extract_total_budget(request_data)

    schedules = src.get("schedules")
    if isinstance(schedules, list):
        valid_schedules = [s for s in schedules if isinstance(s, dict)]
        schedules = valid_schedules or [s for s in fallback_schedules if isinstance(s, dict)]
    else:
        schedules = [s for s in fallback_schedules if isinstance(s, dict)]

    total_budget = src.get("total_budget")
    if total_budget is None:
        total_budget = fallback_total_budget
    elif not isinstance(total_budget, (int, float)):
        try:
            total_budget = float(total_budget)
        except (TypeError, ValueError):
            total_budget = fallback_total_budget

    return {
        "city": str(src.get("city") or fallback_city),
        "trip_start": str(src.get("trip_start") or fallback_trip_start),
        "trip_end": str(src.get("trip_end") or fallback_trip_end),
        "total_budget": total_budget,
        "schedules": schedules,
    }


def _extract_total_budget(request_data: dict[str, Any]) -> float | None:
    budget = request_data.get("budget")
    if isinstance(budget, dict):
        value = budget.get("total_budget")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None
