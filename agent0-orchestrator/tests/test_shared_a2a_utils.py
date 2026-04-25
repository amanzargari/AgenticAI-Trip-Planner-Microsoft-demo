"""Regression tests for shared A2A artifact payload extraction."""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

_ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from shared import a2a_utils as _a2a


def test_extract_artifact_payload_prefers_non_partial_trace() -> None:
    artifacts = [
        {"parts": [{"kind": "data", "data": {"partial_trace": [{"tool": "recommend_places"}]}}]},
        {"parts": [{"kind": "data", "data": {"city": "Milan, Italy", "schedules": []}}]},
    ]

    payload = _a2a._extract_artifact_payload(artifacts, task_id="t1", url="http://agent")

    assert payload == {"city": "Milan, Italy", "schedules": []}


def test_extract_artifact_payload_skips_latest_partial_trace_if_older_final_exists() -> None:
    artifacts = [
        {"parts": [{"kind": "data", "data": {"city": "Milan, Italy", "schedules": [{"date": "2026-06-10", "events": []}]}}]},
        {"parts": [{"kind": "data", "data": {"partial_trace": [{"tool": "schedule_day"}]}}]},
    ]

    payload = _a2a._extract_artifact_payload(artifacts, task_id="t2", url="http://agent")

    assert payload is not None
    assert payload.get("city") == "Milan, Italy"
    assert isinstance(payload.get("schedules"), list)


def test_extract_artifact_payload_returns_partial_trace_when_it_is_only_payload() -> None:
    artifacts = [
        {"parts": [{"kind": "data", "data": {"partial_trace": [{"tool": "cluster_places"}]}}]},
    ]

    payload = _a2a._extract_artifact_payload(artifacts, task_id="t3", url="http://agent")

    assert payload == {"partial_trace": [{"tool": "cluster_places"}]}


def test_extract_artifact_payload_wraps_non_dict_data() -> None:
    artifacts = [
        {"parts": [{"kind": "data", "data": [1, 2, 3]}]},
    ]

    payload = _a2a._extract_artifact_payload(artifacts, task_id="t4", url="http://agent")

    assert payload == {"value": [1, 2, 3]}


def test_extract_artifact_payload_raises_on_non_json_text() -> None:
    artifacts = [
        {"parts": [{"kind": "text", "text": "not-json"}]},
    ]

    with pytest.raises(RuntimeError, match="non-JSON text artifact"):
        _a2a._extract_artifact_payload(artifacts, task_id="t5", url="http://agent")
