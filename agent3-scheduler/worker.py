from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker
from pydantic import BaseModel

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client

MODEL = os.getenv("AGENT3_MODEL", DEFAULT_MODEL)
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
Goal: build a realistic, comfortable day that fills from day_start to day_end with NO overflow.

Input (JSON):
- places: candidate places for this day, each with estimated_visit_duration_minutes
- day_start / day_end: ISO datetime boundaries (typically 09:00 – 21:00 = 720 min)
- food_budget_per_day: float | null
- preferences: free-text user notes (likes, dislikes, dietary needs, preferred meal times)

Mandatory workflow — follow every step in order:

STEP 1 — Order places by proximity:
  Call order_places_by_proximity with ALL input places.

STEP 2 — Calculate how many places fit:
  • day_span  = (day_end − day_start) in minutes
  • meal_time = 45 + 60 + 75 = 180 min (breakfast + lunch + dinner, fixed)
  • budget    = day_span − meal_time          (e.g. 720 − 180 = 540 min for visits + travel)
  • Walk through places in proximity order, accumulating:
      used += place.estimated_visit_duration_minutes + 15 (travel buffer)
    Stop adding when used > budget.
  • Only schedule the places that fit. Drop the rest — it is CORRECT and expected to skip
    places when time runs out. A realistic day visits 2–5 places, not 8.
  • EXCLUDE any place whose category the user explicitly hates.
  • If time allows after fitting all non-excluded places, EXTEND visit durations proportionally
    so the schedule reaches day_end with no large idle gap.

STEP 3 — Travel times:
  Call estimate_travel_minutes between each consecutive pair of SELECTED places.

STEP 4 — Recommend restaurants (ALL 3 meals, no exceptions):
  Call recommend_restaurant for breakfast, then lunch, then dinner.
  - time_of_day = "breakfast" / "lunch" / "dinner"
  - search_center = location of the nearest selected place at that time
  - budget_per_meal_per_person = food_budget_per_day / 3 (null if no budget)
  ★ CRITICAL: if the call fails or returns empty, still include that meal event. Use:
      restaurant_id   = "placeholder_breakfast" / "placeholder_lunch" / "placeholder_dinner"
      restaurant_name = "Breakfast — find a good spot locally" / etc.
    Never skip a meal slot under any circumstance.

STEP 5 — Assemble the schedule:
  • All events MUST start at or after day_start and END at or before day_end. No overflow.
  • Breakfast near day_start, lunch mid-day, dinner in the evening.
  • Visits between meals; travel buffers between consecutive visits.
  • Preferred meal times from preferences override defaults.
  • Result: 3 meal events + N visit events (N = however many fit), all within [day_start, day_end].
"""

PHASE2_INSTRUCTION = (
    "Produce the complete structured day schedule now.\n\n"
    "NON-NEGOTIABLE requirements:\n"
    "1. Include ONLY the places that fit within the day time budget (step 2 above).\n"
    "   Skipping overflow places is CORRECT. A day with 3 great visits is better than 7 rushed ones.\n"
    "2. EXACTLY 3 meal events — breakfast, lunch, AND dinner — no skipping.\n"
    "   • Found a restaurant → use its id and name.\n"
    "   • No restaurant / tool failed → use restaurant_id='placeholder_<slot>' and "
    "restaurant_name='<Slot> — find a good spot locally'.\n"
    "3. Every event MUST end on or before day_end. Never schedule past day_end.\n"
    "4. If selected places finish before day_end, extend their visit durations proportionally "
    "so the day runs all the way to day_end.\n"
    "5. Events are chronological and non-overlapping.\n\n"
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

            # Pre-filter: only pass the LLM as many places as can realistically fit
            day_start_str = str(data.get("day_start") or f"{day_date}T09:00:00")
            day_end_str = str(data.get("day_end") or f"{day_date}T21:00:00")
            try:
                _ds = datetime.fromisoformat(day_start_str)
                _de = datetime.fromisoformat(day_end_str)
                day_span_min = (_de - _ds).total_seconds() / 60
            except ValueError:
                day_span_min = 720.0
            feasible = _select_feasible_places(hydrated, day_span_min)
            logger.info(
                "Scheduler: %d candidates → %d feasible task_id=%s",
                len(hydrated), len(feasible), task_id,
            )
            data["places"] = feasible

            places_by_id: dict[str, dict] = {
                p["id"]: p for p in feasible if isinstance(p, dict) and p.get("id")
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
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto" if tool_called else "required",
                )
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    tool_called = True
                    tool_calls = choice.message.tool_calls or []
                    messages.append(choice.message.model_dump(exclude_none=True))
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
                    model=MODEL,
                    messages=messages,
                    response_format=_SchedulerOutput,
                )
                parsed: Optional[_SchedulerOutput] = final.choices[0].message.parsed
                if parsed:
                    events = _reconstruct_events(
                        parsed.events, places_by_id, restaurants_collected, day_date,
                        restaurants_by_slot,
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
    restaurants_by_slot: Optional[dict[str, list[dict]]] = None,
) -> list[dict]:
    restaurants_by_id: dict[str, dict] = {
        r["id"]: r for r in restaurants_collected if r.get("id")
    }
    restaurants_by_name: dict[str, dict] = {
        r.get("name", ""): r for r in restaurants_collected if r.get("name")
    }
    centroid = _centroid_location(places_by_id)

    events: list[dict] = []
    seen_meal_slots: set[str] = set()        # one meal per slot
    used_restaurant_ids: set[str] = set()    # no restaurant twice in the same day

    for e in entries:
        etype = (e.event_type or "").strip().lower()

        if etype == "visit":
            place = places_by_id.get(e.place_id) or next(
                (p for p in places_by_id.values() if p.get("name") == e.place_name), None
            )
            if not place or not e.start_time or not e.end_time:
                continue
            try:
                s = _ensure_date(e.start_time, day_date)
                en = _ensure_date(e.end_time, day_date)
                # Reject zero-duration or reversed visits (LLM time overflow artefacts)
                if s >= en:
                    logger.debug("Scheduler: skip zero/negative duration visit %s", e.place_name)
                    continue
                visit = VisitEvent(
                    start_time=s,
                    end_time=en,
                    place=PlaceCandidate.model_validate(place),
                )
                events.append(visit.model_dump(mode="json"))
            except Exception as exc:
                logger.debug("Scheduler: skip visit entry: %s", exc)

        elif etype == "meal":
            slot = (e.meal_slot or "").lower()
            if slot not in _MEAL_SLOTS or not e.meal_time:
                continue
            # Skip duplicate slot — LLM sometimes emits the same slot twice
            if slot in seen_meal_slots:
                continue

            # Resolve restaurant: real → next available for slot → placeholder
            restaurant = (
                restaurants_by_id.get(e.restaurant_id)
                or restaurants_by_name.get(e.restaurant_name)
            )
            # If this restaurant was already used for another slot, pick a fresh one
            if restaurant and restaurant.get("id") in used_restaurant_ids:
                restaurant = None
            if not restaurant and restaurants_by_slot:
                for candidate in (restaurants_by_slot.get(slot) or []):
                    if candidate.get("id") not in used_restaurant_ids:
                        restaurant = candidate
                        break
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
                seen_meal_slots.add(slot)
                rid = restaurant.get("id", "")
                if rid:
                    used_restaurant_ids.add(rid)
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
    """Guarantee exactly 3 meal slots; insert any missing ones, never skipping."""
    present_slots = {ev.get("meal_slot") for ev in events if ev.get("type") == "meal"}
    if present_slots >= set(_MEAL_SLOTS):
        return events

    centroid = _centroid_location(places_by_id)
    if not centroid:
        centroid = {"latitude": 0.0, "longitude": 0.0, "address": None}

    try:
        day_start = datetime.fromisoformat(day_start_str)
        day_end = datetime.fromisoformat(day_end_str)
        span_min = (day_end - day_start).total_seconds() / 60
    except ValueError:
        return events

    # Collect restaurant IDs already used in existing meal events
    used_restaurant_ids: set[str] = {
        ev.get("restaurant", {}).get("id", "")
        for ev in events
        if ev.get("type") == "meal"
    }

    slot_fractions = {"breakfast": 0.05, "lunch": 0.42, "dinner": 0.75}

    for slot in _MEAL_SLOTS:
        if slot in present_slots:
            continue

        duration = _MEAL_DURATION_MIN[slot]
        ideal = day_start + timedelta(minutes=span_min * slot_fractions[slot])
        free_at = _find_free_window(events, ideal, duration, day_start, day_end)

        # If no clean window found, force-insert at ideal time (clamped to day bounds)
        if free_at is None:
            free_at = max(day_start, min(ideal, day_end - timedelta(minutes=duration)))

        # Pick a restaurant not already used; fall back to placeholder
        restaurant_data: Optional[dict] = None
        for candidate in (restaurants_by_slot.get(slot) or []):
            if candidate.get("id") not in used_restaurant_ids:
                restaurant_data = candidate
                break
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
            rid = restaurant_data.get("id", "")
            if rid:
                used_restaurant_ids.add(rid)
        except Exception as exc:
            logger.debug("Scheduler _ensure_three_meals: %s", exc)

    return sorted(events, key=lambda ev: str(ev.get("start_time") or ev.get("time") or ""))


def _select_feasible_places(places: list[dict], day_span_min: float) -> list[dict]:
    """Return only the places that actually fit within the day's visit time budget.

    Budget = day_span − fixed_meal_time (180 min).
    Each place consumes its estimated_visit_duration_minutes + 15 min travel.
    Places are taken in order (assumed proximity-sorted) until the budget runs out.
    """
    total_meal_min = sum(_MEAL_DURATION_MIN.values())  # 180
    budget = max(60.0, day_span_min - total_meal_min)

    selected: list[dict] = []
    used = 0.0
    for p in places:
        dur = int(p.get("estimated_visit_duration_minutes") or 60)
        travel = 15
        if used + dur + travel <= budget or not selected:
            selected.append(p)
            used += dur + travel
        # else: time budget exhausted — skip remaining places
    return selected


def _fallback_schedule(
    places_by_id: dict[str, dict],
    restaurants_by_slot: dict[str, list[dict]],
    day_date: str,
    data: dict,
) -> dict[str, Any]:
    """Deterministic schedule when LLM output is unusable."""
    day_start_str = str(data.get("day_start") or f"{day_date}T09:00:00")
    day_end_str = str(data.get("day_end") or f"{day_date}T21:00:00")

    try:
        day_start = datetime.fromisoformat(day_start_str)
        day_end = datetime.fromisoformat(day_end_str)
    except ValueError:
        return {"date": day_date, "events": []}

    day_span_min = (day_end - day_start).total_seconds() / 60
    all_places = list(places_by_id.values())
    if not all_places:
        return {"date": day_date, "events": []}

    # Only keep places that realistically fit within the day
    places = _select_feasible_places(all_places, day_span_min)
    n = len(places)

    # Scale visit durations up to fill the remaining time after meals + travel
    total_meal_min = sum(_MEAL_DURATION_MIN.values())
    total_travel_min = max(0, n - 1) * 15
    available_for_visits = max(30.0, day_span_min - total_meal_min - total_travel_min)
    total_original = sum(int(p.get("estimated_visit_duration_minutes") or 60) for p in places)
    scale = min(4.0, available_for_visits / max(1, total_original))
    scale = max(1.0, scale)

    centroid = _centroid_location(places_by_id)
    _used_in_fallback: set[str] = set()

    def _best_restaurant(slot: str) -> Optional[dict]:
        for candidate in (restaurants_by_slot.get(slot) or []):
            rid = candidate.get("id", "")
            if rid not in _used_in_fallback:
                if rid:
                    _used_in_fallback.add(rid)
                return candidate
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
    dinner_checkpoint = day_start + timedelta(minutes=day_span_min * 0.70)
    lunch_added = dinner_added = False

    for place in places:
        if current >= day_end:
            break

        # Lunch break if we've crossed the mid-day checkpoint
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

        # Dinner break if we've crossed the evening checkpoint
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

        if current >= day_end:
            break

        orig_dur = int(place.get("estimated_visit_duration_minutes") or 60)
        duration = int(orig_dur * scale)
        end = min(current + timedelta(minutes=duration), day_end)
        if end <= current:
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

        current = end + timedelta(minutes=15)

    # Insert any meals not yet placed (clamped to fit before day_end)
    if not lunch_added:
        lunch_r = _best_restaurant("lunch")
        if lunch_r:
            t = min(current, day_end - timedelta(minutes=_MEAL_DURATION_MIN["lunch"]))
            t = max(t, day_start)
            try:
                events.append(MealEvent(
                    time=t.isoformat(),
                    restaurant=RestaurantCandidate.model_validate(lunch_r),
                    meal_slot="lunch",
                ).model_dump(mode="json"))
            except Exception:
                pass

    if not dinner_added:
        dinner_r = _best_restaurant("dinner")
        if dinner_r:
            t = day_end - timedelta(minutes=_MEAL_DURATION_MIN["dinner"])
            t = max(t, day_start)
            try:
                events.append(MealEvent(
                    time=t.isoformat(),
                    restaurant=RestaurantCandidate.model_validate(dinner_r),
                    meal_slot="dinner",
                ).model_dump(mode="json"))
            except Exception:
                pass

    return {
        "date": day_date,
        "events": sorted(events, key=lambda ev: str(ev.get("start_time") or ev.get("time") or "")),
    }
