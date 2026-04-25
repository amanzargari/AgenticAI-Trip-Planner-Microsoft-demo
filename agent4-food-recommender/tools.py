"""Google Places tools for the Food Recommender agent."""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

_PLACES_BASE = "https://maps.googleapis.com/maps/api/place"

_BUDGET_TO_PRICE_LEVEL = [
    (15.0, 1),
    (35.0, 2),
    (70.0, 3),
]

_SKIP_TYPES = {
    "food", "point_of_interest", "establishment",
    "restaurant", "store", "health", "premise",
}

_CUISINE_MAP: dict[str, str] = {
    "italian_restaurant":        "Italian",
    "pizza_restaurant":          "Pizza",
    "cafe":                      "Café",
    "bakery":                    "Bakery",
    "bar":                       "Bar",
    "fast_food_restaurant":      "Fast Food",
    "meal_takeaway":             "Takeaway",
    "meal_delivery":             "Delivery",
    "japanese_restaurant":       "Japanese",
    "chinese_restaurant":        "Chinese",
    "american_restaurant":       "American",
    "french_restaurant":         "French",
    "mediterranean_restaurant":  "Mediterranean",
    "seafood_restaurant":        "Seafood",
    "steak_house":               "Steakhouse",
    "vegetarian_restaurant":     "Vegetarian",
    "vegan_restaurant":          "Vegan",
    "sushi_restaurant":          "Sushi",
    "indian_restaurant":         "Indian",
    "mexican_restaurant":        "Mexican",
    "greek_restaurant":          "Greek",
    "middle_eastern_restaurant": "Middle Eastern",
    "brunch_restaurant":         "Brunch",
    "breakfast_restaurant":      "Breakfast",
    "wine_bar":                  "Wine Bar",
    "thai_restaurant":           "Thai",
    "spanish_restaurant":        "Spanish",
    "turkish_restaurant":        "Turkish",
}

_SLOT_KEYWORDS: dict[str, str] = {
    "breakfast": "breakfast cafe",
    "lunch":     "lunch restaurant",
    "dinner":    "dinner restaurant",
}


def _budget_to_max_price(budget: float) -> int | None:
    for threshold, level in _BUDGET_TO_PRICE_LEVEL:
        if budget <= threshold:
            return level
    return None


def _parse_cuisines(types: list[str]) -> list[str]:
    result = []
    for t in types:
        if t in _SKIP_TYPES:
            continue
        label = _CUISINE_MAP.get(t)
        if label:
            result.append(label)
        else:
            result.append(t.replace("_", " ").title())
    return result[:3]


async def search_restaurants(
    latitude: float,
    longitude: float,
    radius_meters: int = 500,
    meal_slot: str = "lunch",
    budget_per_person: float | None = None,
    cuisine_type: str | None = None,
) -> list[dict[str, Any]]:
    """Search for restaurants using the Google Places Nearby Search API."""
    api_key = os.environ["GOOGLE_MAPS_API_KEY"]

    keyword = cuisine_type or _SLOT_KEYWORDS.get(meal_slot, "restaurant")

    params: dict[str, Any] = {
        "location": f"{latitude},{longitude}",
        "radius":   radius_meters,
        "type":     "restaurant",
        "keyword":  keyword,
        "key":      api_key,
    }

    if budget_per_person is not None:
        max_price = _budget_to_max_price(budget_per_person)
        if max_price is not None:
            params["maxprice"] = max_price

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{_PLACES_BASE}/nearbysearch/json", params=params)
        resp.raise_for_status()
        data = resp.json()

    candidates: list[dict[str, Any]] = []
    for place in data.get("results", [])[:6]:
        geo      = place.get("geometry", {}).get("location", {})
        types    = place.get("types", [])
        cuisines = _parse_cuisines(types)

        raw_price   = place.get("price_level")
        price_level = int(raw_price) if raw_price is not None else None

        candidates.append({
            "id":       place.get("place_id", str(uuid.uuid4())),
            "name":     place.get("name", "Unknown"),
            "location": {
                "latitude":  geo.get("lat", 0.0),
                "longitude": geo.get("lng", 0.0),
                "address":   place.get("vicinity"),
            },
            "price_level": price_level,
            "cuisines":    cuisines or None,
            "rating":      place.get("rating"),
            "summary":     place.get("editorial_summary", {}).get("overview"),
        })

    return candidates


# ── LLM tool schema ───────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "strict": True,
            "description": (
                "Search for restaurants near a location using Google Places Nearby Search. "
                "Call multiple times with different meal_slot or cuisine_type to find diverse options."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
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
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "description": "Search radius in metres (default 500).",
                    },
                    "meal_slot": {
                        "anyOf": [
                            {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
                            {"type": "null"},
                        ],
                        "description": "Meal slot being searched — sets default keyword (breakfast/lunch/dinner).",
                    },
                    "budget_per_person": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                        "description": (
                            "Per-person budget in EUR. Automatically maps to a price level: "
                            "≤€15→cheap (1), ≤€35→moderate (2), ≤€70→expensive (3)."
                        ),
                    },
                    "cuisine_type": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": (
                            "Optional cuisine keyword overriding the slot default, "
                            "e.g. 'Italian', 'Japanese', 'vegan', 'vegetarian'."
                        ),
                    },
                },
                "required": ["latitude", "longitude", "radius_meters", "meal_slot", "budget_per_person", "cuisine_type"],
            },
        },
    }
]
