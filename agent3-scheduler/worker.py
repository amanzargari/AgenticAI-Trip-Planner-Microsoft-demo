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

_MEAL_SLOTS = ("breakfast", "lunch", "dinner")
_MEAL_DURATION_MIN = {"breakfast": 45, "lunch": 60, "dinner": 75}


# ── Simplified internal output model ─────────────────────────────────────────
# Flat structure avoids discriminated-union JSON schema issues with Gemini/OpenRouter.
# Python rebuilds proper VisitEvent/MealEvent after parsing.

class _EventEntry(BaseModel):
    event_type: str = ""      # "visit" or "meal"
    # visit fields
    place_id: str = ""
    place_name: str = ""
    start_time: str = ""      # ISO datetime "YYYY-MM-DDTHH:MM:SS"
    end_time: str = ""
    # meal fields
    meal_slot: str = ""       # "breakfast", "lunch", or "dinner"
    meal_time: str = ""       # ISO datetime "YYYY-MM-DDTHH:MM:SS"
    restaurant_id: str = ""
    restaurant_name: str = ""


class _SchedulerOutput(BaseModel):
    date: str = ""
    events: list[_EventEntry] = []


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Daily Scheduler Agent.
Goal: fill the COMPLETE day from day_start to day_end — no large idle gaps, no early stops.

Input (JSON):
- places: place objects, each has estimated_visit_duration_minutes
- day_start / day_end: ISO datetime boundaries for this day
- food_budget_per_day: float | null
- preferences: free-text user notes (likes, dislikes, dietary needs, preferred meal times)

Mandatory workflow — follow every step:

STEP 1 — Order places:
  Call order_places_by_proximity with ALL input places.

STEP 2 — Apply preferences:
  - EXCLUDE places whose category the user explicitly hates (e.g. "hate museums" → skip museums).
  - Schedule liked-category places earlier in the day.
  - Note preferred meal times if mentioned (they override defaults).

STEP 3 — Travel times:
  Call estimate_travel_minutes between each pair of consecutive places.

STEP 4 — Recommend restaurants (ALL 3 meals, no exceptions):
  Call recommend_restaurant for breakfast, then lunch, then dinner.
  - time_of_day = "breakfast" / "lunch" / "dinner"
  - search_center = location of the nearest place at that time of day
  - budget_per_meal_per_person = food_budget_per_day / 3 (null if no budget)
  ★ CRITICAL: if the call fails OR returns an empty list, you MUST still include a meal
    event for that slot in the output. Use:
      restaurant_id = "placeholder_breakfast" / "placeholder_lunch" / "placeholder_dinner"
      restaurant_name = "Breakfast — find a good spot locally" / "Lunch — …" / "Dinner — …"
    Never skip a meal slot under any circumstance.

STEP 5 — Fill the ENTIRE day:
  - Include ALL places (except hated ones).
  - Use each place's estimated_visit_duration_minutes as the visit duration.
  - If all places are scheduled but day_end has not been reached, proportionally EXTEND
    visit durations so the schedule fills to day_end.
  - Insert meals at natural break points (breakfast near day_start, lunch mid-day,
    dinner in the evening). Adjust to user's preferred times when stated.
  - Only travel time should separate consecutive events — no large idle blocks.

Output MUST contain exactly 3 meal events (breakfast + lunch + dinner) and ALL place visits.
"""

PHASE2_INSTRUCTION = (
    "Produce the complete structured day schedule now.\n\n"
    "NON-NEGOTIABLE requirements:\n"
    "1. ALL input places appear as visit events (skip only those matching an explicit 'hate' "
    "in preferences).\n"
    "2. EXACTLY 3 meal events — breakfast, lunch, AND dinner — no skipping.\n"
    "   • Found a restaurant → use its id and name.\n"
    "   • No restaurant / tool failed → use restaurant_id='placeholder_<slot>' and "
    "restaurant_name='<Slot> — find a good spot locally'.\n"
    "3. Visit durations use estimated_visit_duration_minutes. If places end before day_end, "
    "EXTEND the last visits proportionally so the schedule reaches day_end.\n"
    "4. Events are chronological, non-overlapping, and span the full day.\n\n"
    "Field format:\n"
    "  visit → event_type='visit', place_id, place_name, "
    "start_time (YYYY-MM-DDTHH:MM:SS), end_time (YYYY-MM-DDTHH:MM:SS)\n"
    "  meal  → event_type='meal', meal_slot (breakfast/lunch/dinner), "
    "meal_time (YYYY-MM-DDTHH:MM:SS), restaurant_id, restaurant_name\n"
)


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

            places_by_id: dict[str, dict] = {
                p["id"]: p for p in hydrated if isinstance(p, dict) and p.get("id")
            }

            # Join list preferences into a single free-text sentence for the LLM
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
            # Track restaurants per meal slot so the ensure-meals step can use them.
            restaurants_collected: list[dict] = []
            restaurants_by_slot: dict[str, list[dict]] = {s: [] for s in _MEAL_SLOTS}
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
                                if isinstance(tool_result, list):
                                    slot = str(args.get("time_of_day", "")).lower()
                                    for r in tool_result:
                                        if isinstance(r, dict) and r.get("id"):
                                            restaurants_collected.append(r)
                                            if slot in restaurants_by_slot:
                                                restaurants_by_slot[slot].append(r)
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

            # ── Phase 2: structured output ────────────────────────────────────
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
                    logger.info(
                        "Scheduler phase-2 produced %d events task_id=%s", len(events), task_id
                    )
                else:
                    logger.warning("Scheduler phase-2 returned no parsed object task_id=%s", task_id)
            except Exception as exc:
                logger.warning("Scheduler phase-2 parse failed task_id=%s: %s", task_id, exc)

            # Use deterministic fallback when LLM produced nothing
            if not result_dict.get("events"):
                logger.info("Scheduler: using fallback builder task_id=%s", task_id)
                result_dict = _fallback_schedule(
                    places_by_id, restaurants_by_slot, day_date, data
                )

            # ── Guarantee all 3 meal slots are present ────────────────────────
            day_start_str = str(data.get("day_start") or f"{day_date}T09:00:00")
            day_end_str = str(data.get("day_end") or f"{day_date}T21:00:00")
            result_dict["events"] = _ensure_three_meals(
                result_dict.get("events") or [],
                places_by_id,
                restaurants_by_slot,
                day_date,
                day_start_str,
                day_end_str,
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
    s = time_str.strip()
    if not s:
        return f"{day_date}T00:00:00"
    if "T" in s:
        return s
    return f"{day_date}T{s}"


def _centroid_location(places_by_id: dict[str, dict]) -> Optional[dict]:
    lats, lngs = [], []
    for p in places_by_id.values():
        loc = p.get("location") or {}
        try:
            lats.append(float(loc["latitude"]))
            lngs.append(float(loc["longitude"]))
        except (KeyError, TypeError, ValueError):
            pass
    if not lats:
        return None
    return {"latitude": sum(lats) / len(lats), "longitude": sum(lngs) / len(lngs), "address": None}


def _make_placeholder_restaurant(slot: str, location: dict) -> dict:
    labels = {
        "breakfast": "Breakfast — find a good spot locally",
        "lunch": "Lunch — find a good spot locally",
        "dinner": "Dinner — find a good spot locally",
    }
    return {
        "id": f"placeholder_{slot}",
        "name": labels.get(slot, f"{slot.capitalize()} — find a good spot locally"),
        "location": location,
        "price_level": None,
        "cuisines": None,
        "rating": None,
        "summary": None,
    }


def _reconstruct_events(
    entries: list[_EventEntry],
    places_by_id: dict[str, dict],
    restaurants_collected: list[dict],
    day_date: str,
) -> list[dict]:
    restaurants_by_id: dict[str, dict] = {
        r["id"]: r for r in restaurants_collected if r.get("id")
    }
    restaurants_by_name: dict[str, dict] = {
        r.get("name", ""): r for r in restaurants_collected if r.get("name")
    }
    centroid = _centroid_location(places_by_id)

    events: list[dict] = []
    for e in entries:
        etype = (e.event_type or "").strip().lower()

        if etype == "visit":
            place = places_by_id.get(e.place_id) or next(
                (p for p in places_by_id.values() if p.get("name") == e.place_name), None
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
            slot = (e.meal_slot or "").lower()
            if slot not in _MEAL_SLOTS or not e.meal_time:
                continue

            # Resolve restaurant: real → placeholder
            restaurant = (
                restaurants_by_id.get(e.restaurant_id)
                or restaurants_by_name.get(e.restaurant_name)
            )
            if not restaurant and e.restaurant_id.startswith("placeholder_") and centroid:
                restaurant = _make_placeholder_restaurant(slot, centroid)
            if not restaurant and centroid:
                restaurant = _make_placeholder_restaurant(slot, centroid)
            if not restaurant:
                continue

            try:
                meal = MealEvent(
                    time=_ensure_date(e.meal_time, day_date),
                    restaurant=RestaurantCandidate.model_validate(restaurant),
                    meal_slot=slot,  # type: ignore[arg-type]
                )
                events.append(meal.model_dump(mode="json"))
            except Exception as exc:
                logger.debug("Scheduler: skip meal entry: %s", exc)

    return sorted(events, key=lambda ev: str(ev.get("start_time") or ev.get("time") or ""))


def _find_free_window(
    events: list[dict],
    ideal_time: datetime,
    duration_min: int,
    day_start: datetime,
    day_end: datetime,
) -> Optional[datetime]:
    """Return the nearest free time slot of `duration_min` minutes close to `ideal_time`."""
    occupied: list[tuple[datetime, datetime]] = []
    for ev in events:
        try:
            if ev.get("type") == "visit":
                s = datetime.fromisoformat(str(ev["start_time"]))
                e_end = datetime.fromisoformat(str(ev["end_time"]))
            elif ev.get("type") == "meal":
                s = datetime.fromisoformat(str(ev["time"]))
                e_end = s + timedelta(minutes=duration_min)
            else:
                continue
            occupied.append((s, e_end))
        except (ValueError, KeyError, TypeError):
            pass
    occupied.sort()

    # Search outward from ideal_time in 15-min steps
    for offset_min in range(0, 5 * 60 + 1, 15):
        for delta in ([0] if offset_min == 0 else [offset_min, -offset_min]):
            t = ideal_time + timedelta(minutes=delta)
            t_end = t + timedelta(minutes=duration_min)
            if t < day_start or t_end > day_end:
                continue
            if all(t_end <= s or t >= e for s, e in occupied):
                return t
    return None


def _ensure_three_meals(
    events: list[dict],
    places_by_id: dict[str, dict],
    restaurants_by_slot: dict[str, list[dict]],
    day_date: str,
    day_start_str: str,
    day_end_str: str,
) -> list[dict]:
    """Guarantee exactly 3 meal slots are present; insert any that are missing."""
    present_slots = {ev.get("meal_slot") for ev in events if ev.get("type") == "meal"}
    if present_slots >= set(_MEAL_SLOTS):
        return events

    centroid = _centroid_location(places_by_id)
    if not centroid:
        return events

    try:
        day_start = datetime.fromisoformat(day_start_str)
        day_end = datetime.fromisoformat(day_end_str)
        span_min = (day_end - day_start).total_seconds() / 60
    except ValueError:
        return events

    # Default ideal times (fraction of day span)
    slot_fractions = {"breakfast": 0.05, "lunch": 0.42, "dinner": 0.75}

    for slot in _MEAL_SLOTS:
        if slot in present_slots:
            continue

        duration = _MEAL_DURATION_MIN[slot]
        ideal = day_start + timedelta(minutes=span_min * slot_fractions[slot])
        free_at = _find_free_window(events, ideal, duration, day_start, day_end)
        if free_at is None:
            continue

        # Use real restaurant if available, else placeholder
        restaurant_data = (restaurants_by_slot.get(slot) or [None])[0]
        if not restaurant_data:
            restaurant_data = _make_placeholder_restaurant(slot, centroid)

        try:
            meal = MealEvent(
                time=free_at.isoformat(),
                restaurant=RestaurantCandidate.model_validate(restaurant_data),
                meal_slot=slot,  # type: ignore[arg-type]
            )
            events.append(meal.model_dump(mode="json"))
            present_slots.add(slot)
        except Exception as exc:
            logger.debug("Scheduler _ensure_three_meals: %s", exc)

    return sorted(events, key=lambda ev: str(ev.get("start_time") or ev.get("time") or ""))


def _fallback_schedule(
    places_by_id: dict[str, dict],
    restaurants_by_slot: dict[str, list[dict]],
    day_date: str,
    data: dict,
) -> dict[str, Any]:
    """Deterministic schedule when LLM output is unusable. Scales visit durations to fill the day."""
    day_start_str = str(data.get("day_start") or f"{day_date}T09:00:00")
    day_end_str = str(data.get("day_end") or f"{day_date}T21:00:00")

    try:
        day_start = datetime.fromisoformat(day_start_str)
        day_end = datetime.fromisoformat(day_end_str)
    except ValueError:
        return {"date": day_date, "events": []}

    places = list(places_by_id.values())
    n = len(places)
    if not n:
        return {"date": day_date, "events": []}

    # How much time is available for place visits
    total_meal_min = sum(_MEAL_DURATION_MIN.values())   # 180 min
    total_travel_min = max(0, n - 1) * 15
    day_span_min = (day_end - day_start).total_seconds() / 60
    available_for_visits = max(30.0, day_span_min - total_meal_min - total_travel_min)

    total_original = sum(int(p.get("estimated_visit_duration_minutes") or 60) for p in places)
    # Scale up so visits fill the available window; cap at 4× to avoid absurd durations
    scale = min(4.0, available_for_visits / max(1, total_original))
    scale = max(1.0, scale)

    centroid = _centroid_location(places_by_id)

    def _best_restaurant(slot: str) -> Optional[dict]:
        candidates = restaurants_by_slot.get(slot) or []
        if candidates:
            return candidates[0]
        return _make_placeholder_restaurant(slot, centroid) if centroid else None

    events: list[dict] = []
    current = day_start

    # Breakfast right at day_start
    breakfast_r = _best_restaurant("breakfast")
    if breakfast_r:
        try:
            meal = MealEvent(
                time=current.isoformat(),
                restaurant=RestaurantCandidate.model_validate(breakfast_r),
                meal_slot="breakfast",
            )
            events.append(meal.model_dump(mode="json"))
        except Exception:
            pass
    current += timedelta(minutes=_MEAL_DURATION_MIN["breakfast"])

    lunch_checkpoint = day_start + timedelta(minutes=day_span_min * 0.40)
    dinner_checkpoint = day_start + timedelta(minutes=day_span_min * 0.72)
    lunch_added = dinner_added = False

    for place in places:
        orig_dur = int(place.get("estimated_visit_duration_minutes") or 60)
        duration = int(orig_dur * scale)

        # Lunch break before this visit if checkpoint reached
        if not lunch_added and current >= lunch_checkpoint:
            lunch_r = _best_restaurant("lunch")
            if lunch_r:
                try:
                    meal = MealEvent(
                        time=current.isoformat(),
                        restaurant=RestaurantCandidate.model_validate(lunch_r),
                        meal_slot="lunch",
                    )
                    events.append(meal.model_dump(mode="json"))
                except Exception:
                    pass
            current += timedelta(minutes=_MEAL_DURATION_MIN["lunch"])
            lunch_added = True

        # Dinner break before this visit if checkpoint reached
        if not dinner_added and current >= dinner_checkpoint:
            dinner_r = _best_restaurant("dinner")
            if dinner_r:
                try:
                    meal = MealEvent(
                        time=current.isoformat(),
                        restaurant=RestaurantCandidate.model_validate(dinner_r),
                        meal_slot="dinner",
                    )
                    events.append(meal.model_dump(mode="json"))
                except Exception:
                    pass
            current += timedelta(minutes=_MEAL_DURATION_MIN["dinner"])
            dinner_added = True

        end = current + timedelta(minutes=duration)
        # Clamp to day_end
        if end > day_end:
            end = day_end
        if current >= day_end:
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

    # Insert any meals not yet placed
    if not lunch_added:
        lunch_r = _best_restaurant("lunch")
        if lunch_r:
            lunch_time = min(current, day_end - timedelta(minutes=_MEAL_DURATION_MIN["lunch"]))
            try:
                meal = MealEvent(
                    time=lunch_time.isoformat(),
                    restaurant=RestaurantCandidate.model_validate(lunch_r),
                    meal_slot="lunch",
                )
                events.append(meal.model_dump(mode="json"))
            except Exception:
                pass

    if not dinner_added:
        dinner_r = _best_restaurant("dinner")
        if dinner_r:
            dinner_time = max(
                current,
                day_end - timedelta(minutes=_MEAL_DURATION_MIN["dinner"] + 15),
            )
            dinner_time = min(dinner_time, day_end - timedelta(minutes=_MEAL_DURATION_MIN["dinner"]))
            try:
                meal = MealEvent(
                    time=dinner_time.isoformat(),
                    restaurant=RestaurantCandidate.model_validate(dinner_r),
                    meal_slot="dinner",
                )
                events.append(meal.model_dump(mode="json"))
            except Exception:
                pass

    return {
        "date": day_date,
        "events": sorted(events, key=lambda ev: str(ev.get("start_time") or ev.get("time") or "")),
    }
