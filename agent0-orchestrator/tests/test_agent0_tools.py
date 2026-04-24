"""Tests for agent0 orchestrator tools. All A2A calls mocked."""
from __future__ import annotations
import importlib.util, pathlib, sys, os
from unittest.mock import AsyncMock, patch
import pytest

_AGENT_DIR = pathlib.Path(__file__).parent.parent
_ROOT_DIR  = _AGENT_DIR.parent
for _d in (str(_ROOT_DIR),):
    if _d not in sys.path: sys.path.insert(0, _d)

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("AGENT1_URL", "http://localhost:8001")
os.environ.setdefault("AGENT2_URL", "http://localhost:8002")
os.environ.setdefault("AGENT3_URL", "http://localhost:8003")

_MOD_NAME = "agent0_tools"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "tools.py")
_tools = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _tools
_spec.loader.exec_module(_tools)

MOCK_PLACE = {"id":"p1","name":"Eiffel Tower","location":{"latitude":48.8584,"longitude":2.2945,"address":"Paris"},"estimated_visit_duration_minutes":90,"category":"tourist_attraction","rating":4.7,"summary":""}
MOCK_RESTAURANT = {"id":"r1","name":"Café Flore","location":{"latitude":48.854,"longitude":2.333,"address":"Paris"},"price_level":2,"cuisines":["french"],"rating":4.3,"summary":""}
MOCK_CANDIDATES  = {"place_candidates": [MOCK_PLACE]}
MOCK_CLUSTERS    = {"clustered_place_candidates": [[MOCK_PLACE]]}
MOCK_SCHEDULE    = {"date":"2026-06-10","events":[{"type":"visit","start_time":"2026-06-10T09:00:00","end_time":"2026-06-10T10:30:00","place":MOCK_PLACE},{"type":"meal","time":"2026-06-10T12:30:00","restaurant":MOCK_RESTAURANT,"meal_slot":"lunch"}]}

@pytest.mark.asyncio
async def test_recommend_places_calls_agent1():
    with patch(f"{_MOD_NAME}.call_agent", new=AsyncMock(return_value=MOCK_CANDIDATES)):
        result = await _tools.recommend_places(city="Paris",trip_start="2026-06-10T09:00:00",trip_end="2026-06-13T21:00:00")
        assert "place_candidates" in result

@pytest.mark.asyncio
async def test_cluster_places_calls_agent2():
    with patch(f"{_MOD_NAME}.call_agent", new=AsyncMock(return_value=MOCK_CLUSTERS)):
        result = await _tools.cluster_places(trip_start="2026-06-10T09:00:00",trip_end="2026-06-13T21:00:00",place_candidates=[MOCK_PLACE])
        assert "clustered_place_candidates" in result

@pytest.mark.asyncio
async def test_schedule_day_calls_agent3():
    with patch(f"{_MOD_NAME}.call_agent", new=AsyncMock(return_value=MOCK_SCHEDULE)):
        result = await _tools.schedule_day(places=[MOCK_PLACE],day_start="2026-06-10T09:00:00",day_end="2026-06-10T21:00:00")
        assert result["date"] == "2026-06-10"

def test_tools_schema_completeness():
    assert {t["function"]["name"] for t in _tools.TOOLS} == {"recommend_places","cluster_places","schedule_day"}

def test_tools_required_fields():
    for t in _tools.TOOLS:
        assert "required" in t["function"]["parameters"]

def test_recommend_places_schema():
    t = next(t for t in _tools.TOOLS if t["function"]["name"]=="recommend_places")
    assert set(t["function"]["parameters"]["required"]) == {"city","trip_start","trip_end"}

def test_schedule_day_schema():
    t = next(t for t in _tools.TOOLS if t["function"]["name"]=="schedule_day")
    assert set(t["function"]["parameters"]["required"]) == {"places","day_start","day_end"}
