from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker
from pydantic import BaseModel

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from shared.models import MealEvent, PlaceCandidate, RestaurantCandidate, VisitEvent
from tools import (
    TOOLS,
    estimate_travel_minutes,
    hydrate_places,
    order_places_by_proximity,
    recommend_restaurant,
)

logger = logging.getLogger(__name__)


# ── Simplified internal output model ─────────────────────────────────────────
# Flat structure avoids discriminated-union JSON schema issues with Gemini/OpenRouter.
# We reconstruct proper VisitEvent/MealEvent objects in Python after parsing.

class _EventEntry(BaseModel):
    event_type: str = ""       # "visit" or "meal"
    # visit fields
    place_id: str = ""
    place_name: str = ""
    start_time: str = ""       # ISO datetime "YYYY-MM-DDTHH:MM:SS"
    end_time: str = ""
    # meal fields
    meal_slot: str = ""        # "breakfast", "lunch", or "dinner"
    meal_time: str = ""        # ISO datetime "YYYY-MM-DDTHH:MM:SS"
    restaurant_id: str = ""
    restaurant_name: str = ""


class _SchedulerOutput(BaseModel):
    date: str = ""
    events: list[_EventEntry] = []


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Daily Scheduler Agent.
Goal: create a realistic schedule that fills the ENTIRE day from day_start to day_end.

Input (JSON):
- places: list of place objects, each with estimated_visit_duration_minutes
- day_start: ISO datetime — when the day begins
- day_end: ISO datetime — when the day ends
- food_budget_per_day: float | null
- preferences: free-text user notes (likes, dislikes, dietary needs, desired meal times, etc.)

Step-by-step workflow:

1) Call order_places_by_proximity with ALL input places to get an efficient visit order.

2) Read the preferences note carefully:
   - If the user mentions HATING a type of place (e.g. "hate museums", "dislike churches"),
     mentally note those categories to EXCLUDE from the schedule.
   - If the user mentions LIKING something, give those places higher priority in the schedule.
   - If the user mentions preferred meal times (e.g. "early dinner at 18:00"), use those.

3) Call estimate_travel_minutes between each pair of consecutive places to know transfer times.

4) Determine meal times based on preferences and the visit schedule:
   - BREAKFAST: at or shortly after day_start (adjust if preferences mention a specific time)
   - LUNCH: at a natural mid-day break (typically when current time reaches ~12:00-13:00)
   - DINNER: at a natural evening break (typically when current time reaches ~18:00-20:00)
   For each meal, call recommend_restaurant once with the appropriate time_of_day value.
   Use the location of the nearest upcoming place as search_center.
   Budget per meal = food_budget_per_day / 3 when budget is provided, else null.
   If recommend_restaurant returns an error or empty list, skip that meal — do not retry.

5) Fill the ENTIRE day:
   - Include ALL input places (skip only ones explicitly hated in preferences).
   - Use each place's estimated_visit_duration_minutes as the visit duration.
   - Add travel time between consecutive visits.
   - Keep scheduling until you reach day_end or run out of places.
   - Do NOT stop after 2-3 events — pack the whole day.

Important:
- Events must be chronological and non-overlapping.
- Prefer liked-category places earlier in the day.
- If a tool returns {"error": ...}, ignore it and continue — do NOT retry it.
"""

PHASE2_INSTRUCTION = """\
Based on all the tool results above, produce the complete day schedule as structured JSON.

For each scheduled event output an object with:
  - event_type: "visit" for a place visit, or "meal" for a meal break
  - For visits: place_id (from the input places), place_name, \
start_time ("YYYY-MM-DDTHH:MM:SS"), end_time ("YYYY-MM-DDTHH:MM:SS")
  - For meals: meal_slot ("breakfast", "lunch", or "dinner"), \
meal_time ("YYYY-MM-DDTHH:MM:SS"), restaurant_id, restaurant_name

Rules:
- Include ALL places from order_places_by_proximity that are not explicitly excluded by preferences.
- Use the exact estimated_visit_duration_minutes for each place's duration.
- Use travel times from estimate_travel_minutes to space visits.
- Insert breakfast near day_start, lunch around mid-day, dinner in the evening — using restaurants from recommend_restaurant results.
- List events in chronological order and fill the full day.
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

            # Resolve Place IDs → full place objects
            hydrated = await hydrate_places(data.get("places"))
            if not hydrated:
                logger.warning("Scheduler: no schedulable places task_id=%s", task_id)
            data["places"] = hydrated

            # Index places by id for later reconstruction
            places_by_id: dict[str, dict] = {
                p["id"]: p for p in hydrated if isinstance(p, dict) and p.get("id")
            }

            # Join list preferences into a single free-text note for the LLM
            prefs_raw = data.get("preferences") or []
            if isinstance(prefs_raw, list):
                prefs_note = ". ".join(str(p).strip() for p in prefs_raw if str(p).strip())
            else:
                prefs_note = str(prefs_raw).strip()
            data["preferences"] = prefs_note

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            # ── Phase 1: tool calls ───────────────────────────────────────────
            restaurants_collected: list[dict] = []
            tool_called = False
            for _step in range(25):
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
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
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
                                # Collect for later reconstruction
                                if isinstance(tool_result, list):
                                    restaurants_collected.extend(
                                        r for r in tool_result
                                        if isinstance(r, dict) and r.get("id")
                                    )
                            else:
                                tool_result = {"error": f"Unknown tool: {name}"}
                        except Exception as exc:
                            logger.warning(
                                "Scheduler tool %s failed task_id=%s: %s", name, task_id, exc
                            )
                            tool_result = {"error": f"{name} failed: {exc}"}

                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": tc.id,
                        })
                else:
                    break

            # ── Phase 2: structured output with simplified flat model ──────────
            messages.append({"role": "user", "content": PHASE2_INSTRUCTION})

            result_dict: dict[str, Any] = {"date": day_date, "events": []}
            try:
                final = await llm.beta.chat.completions.parse(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    response_format=_SchedulerOutput,
                )
                parsed: Optional[_SchedulerOutput] = final.choices[0].message.parsed
                if parsed:
                    events = _reconstruct_events(
                        parsed.events, places_by_id, restaurants_collected, day_date
                    )
                    result_dict = {"date": parsed.date or day_date, "events": events}
                    if not events:
                        logger.warning(
                            "Scheduler: parsed output had %d entries but 0 reconstructed task_id=%s",
                            len(parsed.events), task_id,
                        )
                else:
                    logger.warning("Scheduler phase-2 returned no parsed output task_id=%s", task_id)
            except Exception as exc:
                logger.warning("Scheduler phase-2 parse failed task_id=%s: %s", task_id, exc)

            # Fallback: build minimal schedule from available data when events still empty
            if not result_dict.get("events"):
                logger.info("Scheduler: using fallback schedule builder task_id=%s", task_id)
                result_dict = _fallback_schedule(
                    places_by_id, restaurants_collected, day_date, data
                )

            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=self.build_artifacts(result_dict),
            )

        except Exception:
            logger.exception("Scheduler task crashed task_id=%s", task_id)
            await self.storage.update_task(task_id, state="failed")
            raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_date(time_str: str, day_date: str) -> str:
    """Prepend the date when only a time portion (HH:MM or HH:MM:SS) is given."""
    s = time_str.strip()
    if not s:
        return f"{day_date}T00:00:00"
    if "T" in s:
        return s
    return f"{day_date}T{s}"


def _reconstruct_events(
    entries: list[_EventEntry],
    places_by_id: dict[str, dict],
    restaurants_collected: list[dict],
    day_date: str,
) -> list[dict]:
    """Convert flat _EventEntry list into proper VisitEvent/MealEvent dicts."""
    restaurants_by_id: dict[str, dict] = {
        r["id"]: r for r in restaurants_collected if r.get("id")
    }
    restaurants_by_name: dict[str, dict] = {
        r.get("name", ""): r for r in restaurants_collected if r.get("name")
    }

    events: list[dict] = []
    for e in entries:
        etype = (e.event_type or "").strip().lower()

        if etype == "visit":
            place = places_by_id.get(e.place_id)
            if not place:
                # Fallback: match by name
                place = next(
                    (p for p in places_by_id.values() if p.get("name") == e.place_name),
                    None,
                )
            if not place or not e.start_time or not e.end_time:
                continue
            try:
                visit = VisitEvent(
                    start_time=_ensure_date(e.start_time, day_date),
                    end_time=_ensure_date(e.end_time, day_date),
                    place=PlaceCandidate.model_validate(place),
                )
                events.append(visit.model_dump(mode="json"))
            except Exception as exc:
                logger.debug("Scheduler: skip visit entry: %s", exc)

        elif etype == "meal":
            restaurant = restaurants_by_id.get(e.restaurant_id) or restaurants_by_name.get(
                e.restaurant_name
            )
            if not restaurant or not e.meal_slot or not e.meal_time:
                continue
            valid_slots = {"breakfast", "lunch", "dinner"}
            if e.meal_slot.lower() not in valid_slots:
                continue
            try:
                meal = MealEvent(
                    time=_ensure_date(e.meal_time, day_date),
                    restaurant=RestaurantCandidate.model_validate(restaurant),
                    meal_slot=e.meal_slot.lower(),  # type: ignore[arg-type]
                )
                events.append(meal.model_dump(mode="json"))
            except Exception as exc:
                logger.debug("Scheduler: skip meal entry: %s", exc)

    def _sort_key(ev: dict) -> str:
        return str(ev.get("start_time") or ev.get("time") or "")

    return sorted(events, key=_sort_key)


def _fallback_schedule(
    places_by_id: dict[str, dict],
    restaurants_collected: list[dict],
    day_date: str,
    data: dict,
) -> dict[str, Any]:
    """Deterministic schedule built from place durations when LLM output is unusable."""
    day_start_str = str(data.get("day_start") or f"{day_date}T09:00:00")
    day_end_str = str(data.get("day_end") or f"{day_date}T21:00:00")

    try:
        current = datetime.fromisoformat(day_start_str)
        day_end = datetime.fromisoformat(day_end_str)
    except ValueError:
        return {"date": day_date, "events": []}

    events: list[dict] = []

    # Assign collected restaurants to breakfast/lunch/dinner in collection order
    meal_slots = ["breakfast", "lunch", "dinner"]
    slot_restaurants = {
        slot: restaurants_collected[i]
        for i, slot in enumerate(meal_slots)
        if i < len(restaurants_collected)
    }

    total_mins = (day_end - current).total_seconds() / 60
    places = list(places_by_id.values())

    # Rough meal timing checkpoints relative to day span
    lunch_checkpoint = current + timedelta(minutes=total_mins * 0.40)
    dinner_checkpoint = current + timedelta(minutes=total_mins * 0.70)

    def _add_meal(slot: str, at: datetime) -> None:
        r = slot_restaurants.get(slot)
        if not r:
            return
        try:
            meal = MealEvent(
                time=at.isoformat(),
                restaurant=RestaurantCandidate.model_validate(r),
                meal_slot=slot,  # type: ignore[arg-type]
            )
            events.append(meal.model_dump(mode="json"))
        except Exception:
            pass

    # Breakfast shortly after day_start
    _add_meal("breakfast", current)
    current += timedelta(minutes=45)

    lunch_added = dinner_added = False
    for place in places:
        duration = int(place.get("estimated_visit_duration_minutes") or 60)
        end = current + timedelta(minutes=duration)
        if end > day_end:
            break

        # Insert lunch before this visit if checkpoint reached
        if not lunch_added and current >= lunch_checkpoint:
            _add_meal("lunch", current)
            current += timedelta(minutes=60)
            lunch_added = True
            end = current + timedelta(minutes=duration)
            if end > day_end:
                break

        # Insert dinner before this visit if checkpoint reached
        if not dinner_added and current >= dinner_checkpoint:
            _add_meal("dinner", current)
            current += timedelta(minutes=75)
            dinner_added = True
            end = current + timedelta(minutes=duration)
            if end > day_end:
                break

        try:
            visit = VisitEvent(
                start_time=current.isoformat(),
                end_time=end.isoformat(),
                place=PlaceCandidate.model_validate(place),
            )
            events.append(visit.model_dump(mode="json"))
        except Exception:
            pass

        current = end + timedelta(minutes=15)  # travel buffer

    # Add remaining meals if not yet inserted
    if not lunch_added:
        _add_meal("lunch", min(current, day_end - timedelta(minutes=60)))
    if not dinner_added:
        _add_meal("dinner", min(current + timedelta(minutes=30), day_end - timedelta(minutes=75)))

    def _sort_key(ev: dict) -> str:
        return str(ev.get("start_time") or ev.get("time") or "")

    return {"date": day_date, "events": sorted(events, key=_sort_key)}
