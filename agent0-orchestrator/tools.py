"""A2A tool wrappers for the Orchestrator agent.

Each function calls a downstream agent via the A2A protocol.
The LLM decides which to call and in what order via tool-use.
"""
from __future__ import annotations

import os
from typing import Any

from shared.a2a_utils import call_agent

AGENT1_URL: str = os.getenv("AGENT1_URL", "http://localhost:8001")
AGENT2_URL: str = os.getenv("AGENT2_URL", "http://localhost:8002")
AGENT3_URL: str = os.getenv("AGENT3_URL", "http://localhost:8003")


async def recommend_places(
    city: str,
    trip_start: str,
    trip_end: str,
    activity_budget: float | None = None,
    trip_reason: str | None = None,
    preferences: list[str] | None = None,
) -> dict[str, Any]:
    """Call Agent 1 – Place Recommender – and return place_candidates."""
    return await call_agent(
        AGENT1_URL,
        {
            "city": city,
            "trip_start": trip_start,
            "trip_end": trip_end,
            "budget": activity_budget,
            "trip_reason": trip_reason,
            "preferences": preferences or [],
        },
    )


async def cluster_places(
    trip_start: str,
    trip_end: str,
    place_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call Agent 2 – Clustering – and return clustered_place_candidates."""
    return await call_agent(
        AGENT2_URL,
        {
            "trip_start": trip_start,
            "trip_end": trip_end,
            "place_candidates": place_candidates,
        },
    )


async def schedule_day(
    places: list[dict[str, Any]],
    day_start: str,
    day_end: str,
    food_budget_per_day: float | None = None,
    preferences: list[str] | None = None,
) -> dict[str, Any]:
    """Call Agent 3 – Daily Scheduler – and return a DailySchedule."""
    return await call_agent(
        AGENT3_URL,
        {
            "places": places,
            "day_start": day_start,
            "day_end": day_end,
            "food_budget_per_day": food_budget_per_day,
            "preferences": preferences or [],
        },
    )


# ── LLM tool schemas ──────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "recommend_places",
            "strict": True,
            "description": (
                "Ask the Place Recommender agent (Agent 1) for a list of tourist "
                "attractions and points of interest for the trip city. "
                "Returns a 'place_candidates' list."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Destination city name.",
                    },
                    "trip_start": {
                        "type": "string",
                        "description": "Trip start as ISO datetime string.",
                    },
                    "trip_end": {
                        "type": "string",
                        "description": "Trip end as ISO datetime string.",
                    },
                    "activity_budget": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                        "description": "Activity budget in EUR (70% of total). Null if unset.",
                    },
                    "trip_reason": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Reason for the trip, e.g. 'family vacation'.",
                    },
                    "preferences": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "description": "User preference tags, e.g. ['art', 'outdoor'].",
                    },
                },
                "required": ["city", "trip_start", "trip_end", "activity_budget", "trip_reason", "preferences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_places",
            "strict": True,
            "description": (
                "Ask the Clustering agent (Agent 2) to group place_candidates "
                "into one geographic cluster per trip day. "
                "Returns 'clustered_place_candidates'."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "trip_start": {
                        "type": "string",
                        "description": "Trip start ISO datetime (used to compute num_days).",
                    },
                    "trip_end": {
                        "type": "string",
                        "description": "Trip end ISO datetime.",
                    },
                    "place_candidates": {
                        "type": "array",
                        "items": {},
                        "description": "Full list of place objects returned by recommend_places.",
                    },
                },
                "required": ["trip_start", "trip_end", "place_candidates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_day",
            "strict": True,
            "description": (
                "Ask the Daily Scheduler agent (Agent 3) to build a chronological "
                "schedule for one day from a cluster of places. "
                "Call once per cluster returned by cluster_places."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "places": {
                        "type": "array",
                        "items": {},
                        "description": "Places for this specific day (one cluster).",
                    },
                    "day_start": {
                        "type": "string",
                        "description": "Start of the day as ISO datetime (e.g. '2026-06-10T09:00:00').",
                    },
                    "day_end": {
                        "type": "string",
                        "description": "End of the day as ISO datetime (e.g. '2026-06-10T21:00:00').",
                    },
                    "food_budget_per_day": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                        "description": "Food budget in EUR for this day. Null if unset.",
                    },
                    "preferences": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "description": "User dietary/cuisine preferences.",
                    },
                },
                "required": ["places", "day_start", "day_end", "food_budget_per_day", "preferences"],
            },
        },
    },
]
