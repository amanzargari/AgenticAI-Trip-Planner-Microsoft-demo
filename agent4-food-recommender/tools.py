"""Google Places tools for the Food Recommender agent."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import googlemaps


def _gmaps() -> googlemaps.Client:
    return googlemaps.Client(key=os.environ["GOOGLE_MAPS_API_KEY"])


async def search_restaurants(
    latitude: float,
    longitude: float,
    radius_meters: int = 500,
    cuisine_type: str | None = None,
    max_price_level: int | None = None,
) -> list[dict[str, Any]]:
    """Search for restaurants near *location* using the Google Places Nearby Search API."""
    client = _gmaps()

    kwargs: dict[str, Any] = {
        "location": (latitude, longitude),
        "radius": radius_meters,
        "type": "restaurant",
        "language": "en",
    }
    if cuisine_type:
        kwargs["keyword"] = cuisine_type

    raw = await asyncio.to_thread(client.places_nearby, **kwargs)
    results: list[dict[str, Any]] = []

    for place in raw.get("results", []):
        price_level = place.get("price_level")
        if max_price_level is not None and price_level is not None and price_level > max_price_level:
            continue

        geo = place.get("geometry", {}).get("location", {})
        price_symbol = "€" * price_level if price_level else "N/A"
        results.append(
            {
                "id": place.get("place_id", ""),
                "name": place.get("name", ""),
                "location": {
                    "latitude": geo.get("lat", 0.0),
                    "longitude": geo.get("lng", 0.0),
                    "address": place.get("vicinity", ""),
                },
                "price_level": price_level,
                "cuisines": _extract_cuisines(place.get("types", [])),
                "rating": place.get("rating"),
                "summary": (
                    f"Rating: {place.get('rating', 'N/A')}, Price: {price_symbol}"
                ),
            }
        )

    return results[:10]


def _extract_cuisines(types: list[str]) -> list[str]:
    food_keywords = {
        "italian_restaurant", "chinese_restaurant", "french_restaurant",
        "japanese_restaurant", "mexican_restaurant", "indian_restaurant",
        "thai_restaurant", "mediterranean_restaurant", "american_restaurant",
        "pizza_restaurant", "seafood_restaurant", "steak_house",
        "cafe", "bakery", "bar", "fast_food_restaurant",
    }
    cuisines = [
        t.replace("_restaurant", "").replace("_", " ")
        for t in types
        if t in food_keywords
    ]
    return cuisines if cuisines else ["restaurant"]


# ── LLM tool schema ───────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": (
                "Search for restaurants near a geographic location using Google Places. "
                "Call multiple times with different cuisine_type values to explore options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the search center.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the search center.",
                    },
                    "radius_meters": {
                        "type": "integer",
                        "description": "Search radius in metres (default 500).",
                        "default": 500,
                    },
                    "cuisine_type": {
                        "type": "string",
                        "description": (
                            "Optional cuisine keyword, e.g. 'Italian', 'Japanese', 'vegan'."
                        ),
                    },
                    "max_price_level": {
                        "type": "integer",
                        "description": (
                            "Max Google price level 0-4 "
                            "(0=free, 1=cheap, 2=moderate, 3=expensive, 4=very expensive)."
                        ),
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    }
]
