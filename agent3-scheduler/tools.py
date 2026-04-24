"""Scheduling tools for the Daily Scheduler agent.

Provides pure-Python helpers that the LLM can call via tool-use.
The agent also calls Agent 4 (Food Recommender) via A2A.
"""
from __future__ import annotations

import math
import os
from typing import Any

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
    if len(places) <= 1:
        return list(places)

    remaining = list(places)
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


# ── LLM tool schemas ──────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "order_places_by_proximity",
            "description": (
                "Reorder a list of places using nearest-neighbour to minimise "
                "total walking distance. Call this first to get an efficient visit order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "places": {
                        "type": "array",
                        "description": "List of place objects with location.latitude and location.longitude.",
                        "items": {"type": "object"},
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
            "description": "Estimate walking travel time in minutes between two lat/lng points.",
            "parameters": {
                "type": "object",
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
            "description": (
                "Ask the Food Recommender agent for restaurant suggestions near a location. "
                "Call once for lunch and once for dinner."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_of_day": {
                        "type": "string",
                        "description": "'breakfast', 'lunch', or 'dinner'.",
                    },
                    "search_center": {
                        "type": "object",
                        "description": "Center of search area {latitude, longitude}.",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["latitude", "longitude"],
                    },
                    "search_radius_meters": {
                        "type": "integer",
                        "description": "Search radius in metres.",
                        "default": 500,
                    },
                    "budget_per_meal_per_person": {
                        "type": "number",
                        "description": "Budget per meal per person in EUR (null = no limit).",
                    },
                    "preferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dietary or cuisine preferences.",
                    },
                },
                "required": ["time_of_day", "search_center"],
            },
        },
    },
]
