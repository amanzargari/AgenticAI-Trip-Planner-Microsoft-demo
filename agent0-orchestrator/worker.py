from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from tools import TOOLS, cluster_places, recommend_places, schedule_day

MODEL = os.getenv("AGENT0_MODEL", DEFAULT_MODEL)

logger = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Trip Planner Orchestrator — a conversational AI that plans and refines travel itineraries.

You handle two request types:

━━━ TYPE 1: INITIAL PLANNING ━━━
Input has: city, trip_start, trip_end, budget, trip_reason, preferences.
→ Call recommend_places → cluster_places → schedule_day (once per cluster/day).
→ After all schedule_day calls complete, output the trip metadata.

━━━ TYPE 2: MODIFICATION ━━━
Input has: modification_request (string), current_itinerary (existing plan), plus original trip fields.
→ Decide which tools to re-run based on what the user wants to change:
   • New activities / different vibe → recommend_places, cluster_places, then schedule_day for each day.
   • Restaurants / food change only → schedule_day again for affected days with updated preferences.
   • Minor tweak within existing places → schedule_day for affected days.
→ After all needed tool calls complete, output the trip metadata.

Budget policy when budget is provided:
- activity_budget = 0.70 * total_budget
- food_budget_total = 0.30 * total_budget
- num_days = ceil((trip_end - trip_start) in hours / 24)
- food_budget_per_day = food_budget_total / num_days

Available tools:
1) recommend_places(city, trip_start, trip_end, activity_budget, trip_reason, preferences)
     → {"place_candidates": [...]}
2) cluster_places(trip_start, trip_end, place_candidates)
     → {"clustered_place_candidates": [[...], ...]}
3) schedule_day(places, day_start, day_end, food_budget_per_day, preferences)
     → {"date": "YYYY-MM-DD", "events": [...]}

IMPORTANT: when calling schedule_day, pass each cluster's place objects from cluster_places output VERBATIM (with id, name, location, etc.). Never pass only IDs or strings — agent3 will return empty events.
IMPORTANT: clusters are pre-sorted by iconicity — cluster 0 contains the most famous/iconic places. Schedule them in order: cluster 0 = day 1, cluster 1 = day 2, etc. so iconic landmarks always appear first.

Execution rules:
- Always call recommend_places before cluster_places; cluster_places before schedule_day.
- day_start = 09:00 on each date; day_end = 21:00 on each date.
- Never hallucinate tool results; only use actual tool outputs.
- If a tool returns {"error": ...} or {"events": []}, do NOT retry it — accept and move on.
- After all schedule_day calls finish (even if some returned errors), output the metadata immediately.

Final output — ONLY these four fields (schedules are assembled automatically from tool results):
{
    "city": "...",
    "trip_start": "ISO datetime",
    "trip_end": "ISO datetime",
    "total_budget": float or null
}
"""


_AGENT_LABELS: dict[str, str] = {
    "recommend_places": "Place Recommender",
    "cluster_places":   "Clustering",
    "schedule_day":     "Daily Scheduler",
}


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
            logger.info("Orchestrator started task_id=%s keys=%s", task_id, sorted(data.keys()))
            llm = get_llm_client()

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            # Schedules collected from schedule_day tool results (source of truth)
            collected_schedules: list[dict[str, Any]] = []
            trace_events: list[dict[str, Any]] = []
            schedule_day_failures: dict[str, int] = {}
            expected_schedule_count: int = 0  # set after cluster_places returns

            is_initial = "modification_request" not in data
            called_tools: set[str] = set()

            for step in range(40):
                # Force tool calls to prevent the LLM from skipping the planning pipeline.
                # With tool_choice="auto" + response_format, models tend to skip tools entirely.
                if is_initial:
                    if "recommend_places" not in called_tools:
                        tool_choice: Any = {"type": "function", "function": {"name": "recommend_places"}}
                    elif "cluster_places" not in called_tools:
                        tool_choice = {"type": "function", "function": {"name": "cluster_places"}}
                    elif expected_schedule_count > 0 and len(collected_schedules) < expected_schedule_count:
                        # Still more days to schedule — force another tool call
                        tool_choice = "required"
                    elif not collected_schedules:
                        # cluster_places not processed yet — keep forcing
                        tool_choice = "required"
                    else:
                        tool_choice = "auto"
                else:
                    tool_choice = "required" if step == 0 else "auto"

                response = await llm.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                )
                choice = response.choices[0]
                logger.info("Orchestrator step=%d finish=%s tc=%s task_id=%s", step + 1, choice.finish_reason, tool_choice, task_id)

                if choice.finish_reason == "tool_calls":
                    tool_calls = choice.message.tool_calls or []
                    messages.append(choice.message.model_dump(exclude_none=True))

                    for tc in tool_calls:
                        name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as exc:
                            tool_result: Any = {"error": f"Invalid arguments for {name}: {exc}"}
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tc.id,
                            })
                            trace_events.append(_trace(name, {}, tool_result, "error"))
                            continue

                        logger.info("Orchestrator tool=%s arg_keys=%s task_id=%s", name, sorted(args.keys()), task_id)
                        called_tools.add(name)

                        skipped = False
                        try:
                            if name == "recommend_places":
                                tool_result = await recommend_places(**args)
                            elif name == "cluster_places":
                                tool_result = await cluster_places(**args)
                                # Track how many days need scheduling so tool_choice stays
                                # "required" until every cluster has a schedule_day call.
                                clusters = tool_result.get("clustered_place_candidates", [])
                                if isinstance(clusters, list):
                                    expected_schedule_count = len(clusters)
                                    logger.info(
                                        "Orchestrator cluster_places → %d clusters task_id=%s",
                                        expected_schedule_count, task_id,
                                    )
                            elif name == "schedule_day":
                                day_key = str(args.get("day_start", ""))[:10]
                                if schedule_day_failures.get(day_key, 0) >= 2:
                                    logger.warning("Orchestrator skipping retry for day=%s task_id=%s", day_key, task_id)
                                    tool_result = {"date": day_key, "events": []}
                                    skipped = True
                                else:
                                    tool_result = await schedule_day(**args)
                            else:
                                tool_result = {"error": f"Unknown tool: {name}"}
                        except Exception as exc:
                            logger.exception("Orchestrator tool=%s failed task_id=%s", name, task_id)
                            tool_result = {"error": str(exc)}

                        is_error = isinstance(tool_result, dict) and bool(tool_result.get("error"))
                        if is_error:
                            if name == "schedule_day":
                                day_key = str(args.get("day_start", ""))[:10]
                                schedule_day_failures[day_key] = schedule_day_failures.get(day_key, 0) + 1
                                logger.warning("schedule_day error day=%s failures=%d task_id=%s",
                                               day_key, schedule_day_failures[day_key], task_id)
                        elif name == "schedule_day" and isinstance(tool_result, dict):
                            collected_schedules.append(tool_result)

                        status = "skipped" if skipped else ("error" if is_error else "success")
                        trace_events.append(_trace(name, args, tool_result, status))

                        # Push partial result after every tool so UI can stream day-by-day
                        try:
                            _pm = _meta_from_request(data)
                            await self.storage.update_task(
                                task_id,
                                state="working",
                                new_artifacts=[make_data_artifact({
                                    "partial": True,
                                    **_pm,
                                    "schedules": list(collected_schedules),
                                    "trace": list(trace_events),
                                })],
                            )
                        except Exception:
                            pass

                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": tc.id,
                        })

                else:
                    # LLM stopped — metadata always comes from the request, not LLM text.
                    meta = _meta_from_request(data)

                    # Schedules come from tool results, not from the LLM output.
                    # For modifications: if no new tool calls were made, preserve existing schedules.
                    schedules = collected_schedules or _existing_schedules(data)

                    result = {**meta, "schedules": schedules, "trace": trace_events}
                    logger.info(
                        "Orchestrator completed task_id=%s schedules=%d trace=%d",
                        task_id, len(schedules), len(trace_events),
                    )
                    await self.storage.update_task(
                        task_id,
                        state="completed",
                        new_artifacts=self.build_artifacts(result),
                    )
                    return

            # Exceeded max iterations — use what we have
            logger.error("Orchestrator exceeded iterations task_id=%s", task_id)
            meta = _meta_from_request(data)
            schedules = collected_schedules or _existing_schedules(data)
            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=self.build_artifacts({**meta, "schedules": schedules, "trace": trace_events}),
            )

        except Exception:
            logger.exception("Orchestrator crashed task_id=%s", task_id)
            await self.storage.update_task(task_id, state="failed")
            raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trace(tool: str, args: dict, result: Any, status: str) -> dict[str, Any]:
    return {
        "tool":   tool,
        "agent":  _AGENT_LABELS.get(tool, tool),
        "status": status,
        "input":  args,
        "output": result if isinstance(result, dict) else {"value": result},
    }


def _meta_from_request(data: dict[str, Any]) -> dict[str, Any]:
    budget = data.get("budget")
    total = None
    if isinstance(budget, dict):
        try:
            total = float(budget.get("total_budget", 0) or 0) or None
        except (TypeError, ValueError):
            total = None
    return {
        "city":         str(data.get("city") or ""),
        "trip_start":   str(data.get("trip_start") or ""),
        "trip_end":     str(data.get("trip_end") or ""),
        "total_budget": total,
    }


def _existing_schedules(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return schedules from current_itinerary when in modification mode."""
    itin = data.get("current_itinerary")
    if isinstance(itin, dict):
        schedules = itin.get("schedules")
        if isinstance(schedules, list):
            return [s for s in schedules if isinstance(s, dict)]
    return []
