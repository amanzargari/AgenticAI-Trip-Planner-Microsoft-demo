"""Regression tests for agent1 worker safety fallbacks."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from unittest.mock import AsyncMock, patch

import pytest


_AGENT_DIR = pathlib.Path(__file__).parent.parent
_ROOT_DIR = _AGENT_DIR.parent
for _d in (str(_ROOT_DIR),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

_MOD_NAME = "agent1_worker"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "worker.py")
_worker = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _worker
_spec.loader.exec_module(_worker)


def _place(
    pid: str | None,
    *,
    name: str = "Some Place",
    lat: float = 45.4642,
    lng: float = 9.19,
) -> dict:
    data = {
        "name": name,
        "location": {"latitude": lat, "longitude": lng, "address": ""},
        "estimated_visit_duration_minutes": 60,
        "estimated_cost": None,
        "category": "attraction",
        "rating": 4.2,
        "summary": "",
    }
    if pid is not None:
        data["id"] = pid
    return data


def test_dedupe_places_by_id_and_fallback_key() -> None:
    p1 = _place("louvre", name="Louvre")
    p1_dup = _place("louvre", name="Louvre Museum")
    p2 = _place(None, name="Duomo", lat=45.4641, lng=9.1919)
    p2_dup = _place(None, name="Duomo", lat=45.4641, lng=9.1919)

    deduped = _worker._dedupe_places([p1, p1_dup, p2, p2_dup])

    assert len(deduped) == 2
    assert deduped[0]["id"] == "louvre"


def test_coerce_place_result_uses_fallback_when_missing_candidates() -> None:
    fallback = [_place("p1", name="Fallback")]
    result = _worker._coerce_place_result({"foo": "bar"}, fallback)

    assert "place_candidates" in result
    assert len(result["place_candidates"]) == 1
    assert result["place_candidates"][0]["id"] == "p1"


def test_coerce_place_result_keeps_valid_candidates() -> None:
    result = _worker._coerce_place_result(
        {"place_candidates": [_place("a"), _place("a"), "bad"]},
        fallback_candidates=[_place("fallback")],
    )

    assert len(result["place_candidates"]) == 1
    assert result["place_candidates"][0]["id"] == "a"


@pytest.mark.asyncio
async def test_fallback_search_candidates_returns_data_when_search_works() -> None:
    sample = [_place("p1", name="Duomo")]
    with patch.object(_worker, "search_places", new=AsyncMock(return_value=sample)):
        rows = await _worker._fallback_search_candidates(
            {
                "city": "Milan, Italy",
                "preferences": ["art", "museums"],
            }
        )

    assert len(rows) >= 1
    assert rows[0]["id"] == "p1"


@pytest.mark.asyncio
async def test_fallback_search_candidates_handles_search_errors() -> None:
    with patch.object(
        _worker,
        "search_places",
        new=AsyncMock(side_effect=RuntimeError("api down")),
    ):
        rows = await _worker._fallback_search_candidates(
            {
                "city": "Milan, Italy",
                "preferences": ["art", "museums"],
            }
        )

    assert rows == []
