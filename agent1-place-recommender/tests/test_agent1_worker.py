"""Regression tests for agent1 worker helper behavior."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from unittest.mock import AsyncMock, patch

import pytest


_AGENT_DIR = pathlib.Path(__file__).parent.parent
_ROOT_DIR = _AGENT_DIR.parent

for mod in list(sys.modules):
    if mod in ("tools", "worker") or mod.startswith("tools."):
        del sys.modules[mod]

for _d in (str(_AGENT_DIR), str(_ROOT_DIR)):
    if _d in sys.path:
        sys.path.remove(_d)
    sys.path.insert(0, _d)

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

_TOOLS_SPEC = importlib.util.spec_from_file_location("tools", _AGENT_DIR / "tools.py")
_tools = importlib.util.module_from_spec(_TOOLS_SPEC)
sys.modules["tools"] = _tools
_TOOLS_SPEC.loader.exec_module(_tools)

_MOD_NAME = "agent1_worker"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "worker.py")
_worker = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _worker
_spec.loader.exec_module(_worker)


def _place(
    pid: str,
    *,
    name: str = "Some Place",
    lat: float = 45.4642,
    lng: float = 9.19,
) -> dict:
    return {
        "id": pid,
        "name": name,
        "location": {"latitude": lat, "longitude": lng, "address": ""},
        "estimated_visit_duration_minutes": 60,
        "estimated_cost": None,
        "category": "attraction",
        "rating": 4.2,
        "summary": "",
    }


def test_coerce_candidates_dedupes_and_skips_invalid() -> None:
    p1 = _place("louvre", name="Louvre")
    p1_dup = _place("louvre", name="Louvre Museum")
    invalid = {"id": "bad", "name": "Missing location"}

    out = _worker._coerce_candidates([p1, p1_dup, invalid])

    assert len(out) == 1
    assert out[0].id == "louvre"


def test_normalize_search_args_uses_request_city_and_defaults() -> None:
    normalized = _worker._normalize_search_places_args(
        {
            "query": "museums",
            "city": "",
            "radius_meters": None,
            "place_type": "",
        },
        {"city": "Milan, Italy"},
    )

    assert normalized["city"] == "Milan, Italy"
    assert normalized["query"] == "museums"
    assert normalized["radius_meters"] == 15_000
    assert normalized["place_type"] is None


def test_normalize_search_args_sanitizes_bad_radius() -> None:
    normalized = _worker._normalize_search_places_args(
        {
            "query": "",
            "city": "Paris",
            "radius_meters": "bad",
            "place_type": "museum",
        },
        {},
    )

    assert normalized["query"] == "top attractions in Paris"
    assert normalized["radius_meters"] == 15_000
    assert normalized["place_type"] == "museum"


@pytest.mark.asyncio
async def test_fallback_returns_candidates_when_search_works() -> None:
    sample = [_place("p1", name="Duomo")]
    with patch.object(_worker, "search_places", new=AsyncMock(return_value=sample)):
        result = await _worker._fallback(
            {
                "city": "Milan, Italy",
                "preferences": ["art", "museums"],
            }
        )

    assert len(result["place_candidates"]) >= 1
    assert result["place_candidates"][0]["id"] == "p1"


@pytest.mark.asyncio
async def test_fallback_handles_search_errors() -> None:
    with patch.object(
        _worker,
        "search_places",
        new=AsyncMock(side_effect=RuntimeError("api down")),
    ):
        result = await _worker._fallback(
            {
                "city": "Milan, Italy",
                "preferences": ["art", "museums"],
            }
        )

    assert result == {"place_candidates": []}
