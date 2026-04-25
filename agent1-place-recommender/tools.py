"""Google Places tools for the Place Recommender agent."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import googlemaps
import httpx


_PLACES_BASE = "https://maps.googleapis.com/maps/api/place"


def _gmaps() -> googlemaps.Client:
    return googlemaps.Client(key=os.environ["GOOGLE_MAPS_API_KEY"])


async def geocode_city(city: str) -> dict[str, float]:
    """Return the lat/lng centre of *city*."""
    client = _gmaps()
    results = await asyncio.to_thread(client.geocode, city)
    if not results:
        raise ValueError(f"Could not geocode city: {city!r}")
    loc = results[0]["geometry"]["location"]
    return {"latitude": loc["lat"], "longitude": loc["lng"]}


async def search_places(
    query: str,
    city: str,
    radius_meters: int = 15_000,
    place_type: str | None = None,
) -> list[dict[str, Any]]:
    """Text-search for places of interest in *city* using the Google Places API."""
    key = os.environ["GOOGLE_MAPS_API_KEY"]

    normalized_query = (query or "").strip()
    if not normalized_query:
        normalized_query = f"top attractions in {city}"
    elif city and city.lower() not in normalized_query.lower():
        normalized_query = f"{normalized_query} in {city}"

    # Get city centre for location bias
    location: tuple[float, float] | None = None
    try:
        geocode_results = await asyncio.to_thread(_gmaps().geocode, city)
        if geocode_results:
            loc = geocode_results[0]["geometry"]["location"]
            location = (loc["lat"], loc["lng"])
    except Exception:
        # Some keys/projects have Places enabled but Geocoding disabled.
        # In that case, continue with unbiased text search instead of failing.
        location = None

    params: dict[str, Any] = {
        "query": normalized_query,
        "language": "en",
        "key": key,
    }
    if location:
        params["location"] = f"{location[0]},{location[1]}"
        params["radius"] = radius_meters
    if place_type:
        params["type"] = place_type

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{_PLACES_BASE}/textsearch/json", params=params)
        resp.raise_for_status()
        raw = resp.json()

    status = str(raw.get("status") or "")
    if status not in {"OK", "ZERO_RESULTS"}:
        details = raw.get("error_message") or status or "unknown error"
        raise RuntimeError(f"Google Places text search failed: {details}")

    results: list[dict[str, Any]] = []

    for place in raw.get("results", []):
        geo = place.get("geometry", {}).get("location", {})
        types = place.get("types", [])
        category = _primary_category(types)
        duration = _estimate_duration(types)

        results.append(
            {
                "id": place.get("place_id", ""),
                "name": place.get("name", ""),
                "location": {
                    "latitude": geo.get("lat", 0.0),
                    "longitude": geo.get("lng", 0.0),
                    "address": place.get("formatted_address", ""),
                },
                "estimated_visit_duration_minutes": duration,
                "estimated_cost": None,
                "category": category,
                "rating": place.get("rating"),
                "summary": (
                    f"{category.title()} – Rating: {place.get('rating', 'N/A')}"
                ),
            }
        )

    return results[:15]


def _primary_category(types: list[str]) -> str:
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


def _estimate_duration(types: list[str]) -> int:
    long_visits = {"museum", "art_gallery", "zoo", "amusement_park", "aquarium"}
    short_visits = {"church", "mosque", "synagogue", "natural_feature", "landmark"}
    for t in types:
        if t in long_visits:
            return 120
        if t in short_visits:
            return 45
    return 60


# ── LLM tool schemas ──────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_places",
            "strict": True,
            "description": (
                "Search for tourist attractions, landmarks, and points of interest "
                "in a city using Google Places Text Search. "
                "Call multiple times with different queries to explore diverse options."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text search query, e.g. "
                            "'museums in Paris', 'parks in Rome', 'historic sites in Athens'."
                        ),
                    },
                    "city": {
                        "type": "string",
                        "description": "City name used as a location bias for the search.",
                    },
                    "radius_meters": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "description": "Search radius in metres around city centre (default 15000).",
                    },
                    "place_type": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": (
                            "Optional Google Places type filter, e.g. "
                            "'museum', 'park', 'tourist_attraction'."
                        ),
                    },
                },
                "required": ["query", "city", "radius_meters", "place_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "geocode_city",
            "strict": True,
            "description": "Return the latitude and longitude of a city centre.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "city": {"type": "string", "description": "City name to geocode."}
                },
                "required": ["city"],
            },
        },
    },
]
