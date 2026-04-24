"""Tests for orchestrator worker output normalization helpers."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys


_AGENT_DIR = pathlib.Path(__file__).parent.parent
_ROOT_DIR = _AGENT_DIR.parent
for _d in (str(_ROOT_DIR),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

_MOD_NAME = "agent0_worker"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "worker.py")
_worker = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _worker
_spec.loader.exec_module(_worker)


def test_normalize_orchestrator_result_fallbacks_to_request_data() -> None:
    request = {
        "city": "Milan, Italy",
        "trip_start": "2026-06-10T09:00:00",
        "trip_end": "2026-06-15T21:00:00",
        "budget": {"total_budget": 800, "currency": "EUR"},
    }
    fallback_schedules = [{"date": "2026-06-10", "events": []}]

    result = _worker._normalize_orchestrator_result(
        {"error": "Could not parse orchestrator output"},
        request_data=request,
        fallback_schedules=fallback_schedules,
    )

    assert result["city"] == "Milan, Italy"
    assert result["trip_start"] == "2026-06-10T09:00:00"
    assert result["trip_end"] == "2026-06-15T21:00:00"
    assert result["total_budget"] == 800.0
    assert result["schedules"] == fallback_schedules


def test_normalize_orchestrator_result_preserves_valid_json_payload() -> None:
    request = {"city": "Milan, Italy", "trip_start": "s", "trip_end": "e"}
    payload = {
        "city": "Rome, Italy",
        "trip_start": "2026-07-01T09:00:00",
        "trip_end": "2026-07-02T21:00:00",
        "total_budget": 500,
        "schedules": [{"date": "2026-07-01", "events": []}],
    }

    result = _worker._normalize_orchestrator_result(
        payload,
        request_data=request,
        fallback_schedules=[],
    )

    assert result["city"] == "Rome, Italy"
    assert result["total_budget"] == 500
    assert len(result["schedules"]) == 1


def test_extract_total_budget_handles_missing_and_invalid() -> None:
    assert _worker._extract_total_budget({}) is None
    assert _worker._extract_total_budget({"budget": {"total_budget": "x"}}) is None
    assert _worker._extract_total_budget({"budget": {"total_budget": "123.5"}}) == 123.5
