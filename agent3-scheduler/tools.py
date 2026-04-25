"""Scheduling tools for the Daily Scheduler agent.

Provides pure-Python helpers that the LLM can call via tool-use.
The agent also calls Agent 4 (Food Recommender) via A2A.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any
import asyncio
import httpx

from shared.a2a_utils import call_agent

AGENT4_URL: str = os.getenv("AGENT4_URL", "http://localhost:8004")


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance in kilometres between two lat/lng points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def estimate_travel_minutes(
    from_lat: float, from_lng: float, to_lat: float, to_lng: float
) -> int:
    """Estimate travel time in minutes assuming ~4 km/h walking + overhead."""
    dist = haversine_km(from_lat, from_lng, to_lat, to_lng)
    return max(5, int(dist / 4.0 * 60) + 10)  # walking + overhead


def order_places_by_proximity(
    places: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Greedily order places to minimise total walking distance (nearest-neighbour TSP)."""
    clean_places = _coerce_places(places)
    if len(clean_places) <= 1:
        return list(clean_places)

    remaining = list(clean_places)
    ordered = [remaining.pop(0)]

    while remaining:
        last = ordered[-1]["location"]
        nearest = min(
            remaining,
            key=lambda p: haversine_km(
                last["latitude"], last["longitude"],
                p["location"]["latitude"], p["location"]["longitude"],
            ),
        )
        ordered.append(nearest)
        remaining.remove(nearest)

    return ordered


def _coerce_places(raw: Any) -> list[dict[str, Any]]:
    """Normalize potentially stringified/invalid tool input into place dicts."""
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []

    if isinstance(value, dict):
        nested = None
        for key in ("places", "place_candidates", "clustered_place_candidates"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                nested = candidate
                break
        if nested is None:
            return []
        value = nested

    if not isinstance(value, list):
        return []

    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue

        if isinstance(item, list):
            out.extend(_coerce_places(item))
            continue

        if not isinstance(item, dict):
            continue

        loc = item.get("location")
        if not isinstance(loc, dict):
            continue

        try:
            lat = float(loc.get("latitude"))
            lng = float(loc.get("longitude"))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(lat) or not math.isfinite(lng):
            continue

        normalized = dict(item)
        normalized_loc = dict(loc)
        normalized_loc["latitude"] = lat
        normalized_loc["longitude"] = lng
        normalized["location"] = normalized_loc
        out.append(normalized)

    return out


async def recommend_restaurant(
    time_of_day: str,
    search_center: dict[str, float],
    search_radius_meters: int = 500,
    budget_per_meal_per_person: float | None = None,
    preferences: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Call Agent 4 (Food Recommender) via A2A and return restaurant candidates."""
    payload: dict[str, Any] = {
        "time_of_day": time_of_day,
        "search_center": search_center,
        "search_radius_meters": search_radius_meters,
        "budget_per_meal_per_person": budget_per_meal_per_person,
        "preferences": preferences or [],
    }
    if budget_per_meal_per_person is not None:
        max_price = min(3, max(0, int(budget_per_meal_per_person / 15)))
        payload["max_price_level"] = max_price

    result = await call_agent(AGENT4_URL, payload)
    return result.get("restaurants", [])

_PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


def _category_from_types(types: list[str]) -> str:
    priority = [
        "museum", "art_gallery", "tourist_attraction", "park", "zoo",
        "amusement_park", "aquarium", "church", "mosque", "synagogue",
        "stadium", "shopping_mall", "spa", "night_club", "library",
        "natural_feature", "landmark",
    ]
    for t in priority:
        if t in types:
            return t.replace("_", " ")
    return types[0].replace("_", " ") if types else "attraction"


def _duration_from_types(types: list[str]) -> int:
    long_visits = {"museum", "art_gallery", "zoo", "amusement_park", "aquarium"}
    short_visits = {"church", "mosque", "synagogue", "natural_feature", "landmark"}
    for t in types:
        if t in long_visits:
            return 120
        if t in short_visits:
            return 45
    return 60


async def _fetch_place_details(place_id: str, client: httpx.AsyncClient) -> dict[str, Any] | None:
    """Resolve a Google Place ID to a full place object. Returns None on any failure."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None
    params = {
        "place_id": place_id,
        "fields": "place_id,name,geometry/location,formatted_address,types,rating",
        "key": api_key,
    }
    try:
        resp = await client.get(_PLACE_DETAILS_URL, params=params, timeout=15.0)
        resp.raise_for_status()
        raw = resp.json()
    except Exception:
        return None

    if str(raw.get("status") or "") != "OK":
        return None

    result = raw.get("result") or {}
    loc = (result.get("geometry") or {}).get("location") or {}
    if "lat" not in loc or "lng" not in loc:
        return None

    types = result.get("types") or []
    return {
        "id": result.get("place_id") or place_id,
        "name": result.get("name") or "",
        "location": {
            "latitude": loc["lat"],
            "longitude": loc["lng"],
            "address": result.get("formatted_address") or "",
        },
        "estimated_visit_duration_minutes": _duration_from_types(types),
        "category": _category_from_types(types),
        "rating": result.get("rating"),
    }


async def hydrate_places(raw: Any) -> list[dict[str, Any]]:
    """Turn a mixed list of Place IDs (strings) and/or place dicts into full place objects.

    - dicts: passed through _coerce_places (existing validation preserved).
    - strings: treated as Google Place IDs and resolved via the Places Details API.
    - stringified JSON objects: parsed and treated as dicts.
    """
    if isinstance(raw, dict):
        for key in ("places", "place_candidates", "clustered_place_candidates"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return []

    ids: list[str] = []
    dicts: list[Any] = []
    for item in raw:
        if isinstance(item, dict):
            dicts.append(item)
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    dicts.append(parsed)
                    continue
            except json.JSONDecodeError:
                pass
            ids.append(item)

    coerced = _coerce_places(dicts)

    if ids:
        async with httpx.AsyncClient() as client:
            hydrated = await asyncio.gather(
                *(_fetch_place_details(pid, client) for pid in ids)
            )
        coerced.extend(p for p in hydrated if p is not None)

    return coerced

# ── LLM tool schemas ──────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "order_places_by_proximity",
            "strict": True,
            "description": (
                "Reorder a list of places using nearest-neighbour to minimise "
                "total walking distance. Call this first to get an efficient visit order."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "places": {
                        "type": "array",
                        "description": "List of place objects with location.latitude and location.longitude.",
                        "items": {},
                    }
                },
                "required": ["places"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_travel_minutes",
            "strict": True,
            "description": "Estimate walking travel time in minutes between two lat/lng points.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "from_lat": {"type": "number"},
                    "from_lng": {"type": "number"},
                    "to_lat": {"type": "number"},
                    "to_lng": {"type": "number"},
                },
                "required": ["from_lat", "from_lng", "to_lat", "to_lng"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_restaurant",
            "strict": True,
            "description": (
                "Ask the Food Recommender agent for restaurant suggestions near a location. "
                "Call once for each meal: breakfast, lunch, and dinner."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "time_of_day": {
                        "type": "string",
                        "description": "'breakfast', 'lunch', or 'dinner'.",
                    },
                    "search_center": {
                        "type": "object",
                        "additionalProperties": False,
                        "description": "Center of search area {latitude, longitude}.",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["latitude", "longitude"],
                    },
                    "search_radius_meters": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "description": "Search radius in metres (default 500).",
                    },
                    "budget_per_meal_per_person": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                        "description": "Budget per meal per person in EUR (food_budget_per_day / 3; null = no limit).",
                    },
                    "preferences": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "description": "Dietary restrictions or cuisine preferences from user notes.",
                    },
                },
                "required": ["time_of_day", "search_center", "search_radius_meters", "budget_per_meal_per_person", "preferences"],
            },
        },
    },
]


