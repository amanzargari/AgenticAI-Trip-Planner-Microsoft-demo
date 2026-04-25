"""Tests for orchestrator worker helper functions."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys


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

_MOD_NAME = "agent0_worker"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "worker.py")
_worker = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _worker
_spec.loader.exec_module(_worker)


def test_meta_from_request_parses_budget() -> None:
    data = {
        "city": "Milan, Italy",
        "trip_start": "2026-06-10T09:00:00",
        "trip_end": "2026-06-15T21:00:00",
        "budget": {"total_budget": "800", "currency": "EUR"},
    }

    meta = _worker._meta_from_request(data)

    assert meta["city"] == "Milan, Italy"
    assert meta["trip_start"] == "2026-06-10T09:00:00"
    assert meta["trip_end"] == "2026-06-15T21:00:00"
    assert meta["total_budget"] == 800.0


def test_meta_from_request_handles_missing_or_invalid_budget() -> None:
    assert _worker._meta_from_request({"city": "Milan"})["total_budget"] is None
    assert _worker._meta_from_request({"budget": {"total_budget": "bad"}})["total_budget"] is None


def test_existing_schedules_returns_current_itinerary_schedules() -> None:
    data = {
        "current_itinerary": {
            "schedules": [
                {"date": "2026-06-10", "events": []},
                "bad",
            ]
        }
    }

    schedules = _worker._existing_schedules(data)

    assert schedules == [{"date": "2026-06-10", "events": []}]


def test_trace_structure_includes_tool_agent_status() -> None:
    event = _worker._trace(
        tool="recommend_places",
        args={"city": "Milan"},
        result={"place_candidates": []},
        status="success",
    )

    assert event["tool"] == "recommend_places"
    assert event["agent"] == "Place Recommender"
    assert event["status"] == "success"
    assert event["input"]["city"] == "Milan"
