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
You are the Trip Planner Orchestrator — a conversational AI that helps users plan and refine travel itineraries.

You handle two types of requests:

━━━ TYPE 1: INITIAL PLANNING ━━━
Input has: city, trip_start, trip_end, budget, trip_reason, preferences.
→ Call recommend_places → cluster_places → schedule_day (once per day).
→ Produce a complete itinerary from scratch.

━━━ TYPE 2: MODIFICATION ━━━
Input has: modification_request (string), current_itinerary (existing plan), plus original trip fields.
→ Understand what the user wants to change:
   • New activities / different vibe → call recommend_places (with updated preferences), then cluster_places, then schedule_day.
   • Restaurants / food change only → call schedule_day again for the affected day(s) with updated preferences.
   • Minor preference tweak within existing places → call schedule_day for affected days.
   • Structural date/budget change → treat as a full re-plan (all three tools).
→ Be smart: preserve unchanged days from current_itinerary whenever possible.
→ Use the modification_request to understand intent (e.g. "more outdoor", "vegetarian food", "add a museum").

Budget policy when budget is provided:
- activity_budget = 0.70 * total_budget
- food_budget_total = 0.30 * total_budget
- num_days = ceil((trip_end - trip_start) in hours / 24)
- food_budget_per_day = food_budget_total / num_days

Available tools:
1) recommend_places(city, trip_start, trip_end, activity_budget, trip_reason, preferences)
     → {"place_candidates": [...]}
2) cluster_places(trip_start, trip_end, place_candidates)
     → {"clustered_place_candidates": [[...], [...], ...]}
3) schedule_day(places, day_start, day_end, food_budget_per_day, preferences)
     → {"date": "YYYY-MM-DD", "events": [...]}

Execution rules:
- Always call recommend_places before cluster_places; cluster_places before schedule_day.
- day_start = 09:00 on each date; day_end = 21:00 on each date.
- Never hallucinate tool results; only use actual tool outputs.
- For modifications, you MAY skip recommend_places/cluster_places if the existing places still work.
- If a tool returns {"error": ...} or {"date": ..., "events": []}, do NOT retry it. Accept the result and continue.
- After all schedule_day calls complete (even if some returned errors/empty), output the final JSON immediately.

Output (STRICT — no markdown, no prose, no backticks):
{
    "city": "...",
    "trip_start": "ISO",
    "trip_end": "ISO",
    "total_budget": float | null,
    "schedules": [
        {
            "date": "YYYY-MM-DD",
            "events": [...]
        }
    ]
}
- Always return this exact shape, even on partial/empty data.
- Set "schedules" to [] only if nothing could be scheduled.
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
            trace_events: list[dict[str, Any]] = []
            schedule_day_failures: dict[str, int] = {}

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
                    task_id, step + 1, choice.finish_reason,
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
                                "Orchestrator tool arguments JSON decode failed task_id=%s tool=%s",
                                task_id, name,
                            )
                            tool_result = {"error": f"Invalid arguments for tool {name}: {exc}"}
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tc.id,
                            })
                            trace_events.append(_make_trace_event(name, {}, tool_result, "error"))
                            continue

                        logger.info(
                            "Orchestrator tool call task_id=%s tool=%s arg_keys=%s",
                            task_id, name, sorted(args.keys()),
                        )

                        skipped = False
                        try:
                            if name == "recommend_places":
                                tool_result = await recommend_places(**args)
                            elif name == "cluster_places":
                                tool_result = await cluster_places(**args)
                            elif name == "schedule_day":
                                day_key = str(args.get("day_start", ""))[:10]
                                if schedule_day_failures.get(day_key, 0) >= 2:
                                    logger.warning(
                                        "Orchestrator skipping schedule_day retry task_id=%s day=%s",
                                        task_id, day_key,
                                    )
                                    tool_result = {"date": day_key, "events": []}
                                    skipped = True
                                else:
                                    tool_result = await schedule_day(**args)
                            else:
                                tool_result = {"error": f"Unknown tool: {name}"}
                        except Exception as exc:
                            logger.exception(
                                "Orchestrator tool execution failed task_id=%s tool=%s",
                                task_id, name,
                            )
                            tool_result = {"error": str(exc)}

                        is_error = isinstance(tool_result, dict) and bool(tool_result.get("error"))
                        if is_error:
                            logger.warning(
                                "Orchestrator tool returned error task_id=%s tool=%s error=%s",
                                task_id, name, tool_result.get("error"),
                            )
                            if name == "schedule_day":
                                day_key = str(args.get("day_start", ""))[:10]
                                schedule_day_failures[day_key] = schedule_day_failures.get(day_key, 0) + 1
                        else:
                            logger.info(
                                "Orchestrator tool succeeded task_id=%s tool=%s",
                                task_id, name,
                            )
                            if name == "schedule_day" and isinstance(tool_result, dict):
                                fallback_schedules.append(tool_result)

                        status = "skipped" if skipped else ("error" if is_error else "success")
                        trace_events.append(_make_trace_event(name, args, tool_result, status))

                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": tc.id,
                        })
                else:
                    raw = choice.message.content or "{}"
                    parsed = _parse_json(raw)
                    result = _normalize_orchestrator_result(
                        parsed,
                        request_data=data,
                        fallback_schedules=fallback_schedules,
                        trace_events=trace_events,
                    )
                    if isinstance(parsed, dict) and parsed.get("error"):
                        logger.warning(
                            "Orchestrator produced error-shaped result task_id=%s error=%s",
                            task_id, parsed.get("error"),
                        )
                    logger.info(
                        "Orchestrator task completed task_id=%s schedules=%d trace_steps=%d",
                        task_id,
                        len(result.get("schedules", [])),
                        len(trace_events),
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
                trace_events=trace_events,
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
    # Strip markdown code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip any text prefix before the first '{' (e.g. "final_response:{...}")
    brace = text.find("{")
    if brace > 0:
        try:
            return json.loads(text[brace:])
        except json.JSONDecodeError:
            pass
    logger.warning(
        "Orchestrator final response was not valid JSON; excerpt=%s",
        text[:240],
    )
    return {"error": "Could not parse orchestrator output", "raw": text}


_AGENT_LABELS: dict[str, str] = {
    "recommend_places": "Place Recommender",
    "cluster_places":   "Clustering",
    "schedule_day":     "Daily Scheduler",
}


def _make_trace_event(
    tool: str,
    args: dict[str, Any],
    result: Any,
    status: str,
) -> dict[str, Any]:
    return {
        "tool":   tool,
        "agent":  _AGENT_LABELS.get(tool, tool),
        "status": status,
        "input":  args,
        "output": result if isinstance(result, dict) else {"value": result},
    }


def _normalize_orchestrator_result(
    result: Any,
    request_data: dict[str, Any],
    fallback_schedules: list[dict[str, Any]],
    trace_events: list[dict[str, Any]] | None = None,
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
        "city":        str(src.get("city") or fallback_city),
        "trip_start":  str(src.get("trip_start") or fallback_trip_start),
        "trip_end":    str(src.get("trip_end") or fallback_trip_end),
        "total_budget": total_budget,
        "schedules":   schedules,
        "trace":       trace_events or [],
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
