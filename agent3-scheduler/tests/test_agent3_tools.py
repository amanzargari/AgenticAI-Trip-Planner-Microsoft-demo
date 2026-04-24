"""Tests for agent3 scheduling tools."""
from __future__ import annotations
import importlib.util, pathlib, sys, os
from unittest.mock import AsyncMock, patch
import pytest

_AGENT_DIR = pathlib.Path(__file__).parent.parent
_ROOT_DIR  = _AGENT_DIR.parent
for _d in (str(_ROOT_DIR),):
    if _d not in sys.path: sys.path.insert(0, _d)

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("AGENT4_URL", "http://localhost:8004")

_MOD_NAME = "agent3_tools"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "tools.py")
_tools = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _tools
_spec.loader.exec_module(_tools)

def _p(pid,lat,lng,dur=60):
    return {"id":pid,"name":f"P{pid}","location":{"latitude":lat,"longitude":lng,"address":""},"estimated_visit_duration_minutes":dur,"category":"attraction","rating":4.0}

def test_haversine_same():
    assert _tools.haversine_km(48.86,2.35,48.86,2.35)==pytest.approx(0.0,abs=1e-6)

def test_haversine_known():
    d = _tools.haversine_km(48.8584,2.2945,48.8530,2.3499)
    assert 3.0<d<4.5

def test_travel_minutes_floor():
    assert _tools.estimate_travel_minutes(48.86,2.35,48.86,2.35)>=5

def test_travel_minutes_range():
    m = _tools.estimate_travel_minutes(48.8584,2.2945,48.8530,2.3499)
    assert 40<m<90

def test_order_single():
    p = [_p("a",48.86,2.35)]
    assert _tools.order_places_by_proximity(p)==p

def test_order_greedy():
    p1,p2,p3 = _p("p1",48.860,2.350),_p("p2",48.861,2.351),_p("p3",48.880,2.400)
    ordered = _tools.order_places_by_proximity([p1,p3,p2])
    ids = [p["id"] for p in ordered]
    assert ids==["p1","p2","p3"]

@pytest.mark.asyncio
async def test_recommend_restaurant_calls_agent4():
    mock = {"restaurants":[{"id":"r1","name":"Bistro","location":{"latitude":48.86,"longitude":2.35,"address":""},"price_level":2,"cuisines":["french"],"rating":4.4,"summary":""}]}
    with patch(f"{_MOD_NAME}.call_agent", new=AsyncMock(return_value=mock)):
        res = await _tools.recommend_restaurant(time_of_day="lunch",search_center={"latitude":48.86,"longitude":2.35},search_radius_meters=500,budget_per_meal_per_person=25.0,preferences=["French"])
        assert res[0]["name"]=="Bistro"

def test_tools_names():
    names = [t["function"]["name"] for t in _tools.TOOLS]
    assert "order_places_by_proximity" in names
    assert "estimate_travel_minutes" in names
    assert "recommend_restaurant" in names
