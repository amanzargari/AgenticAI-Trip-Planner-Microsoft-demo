"""Tests for agent4 tools – Google Places calls mocked."""
from __future__ import annotations
import importlib.util, pathlib, sys, os
from unittest.mock import MagicMock, patch
import pytest

_AGENT_DIR = pathlib.Path(__file__).parent.parent
_ROOT_DIR  = _AGENT_DIR.parent
for _d in (str(_ROOT_DIR),):
    if _d not in sys.path: sys.path.insert(0, _d)

os.environ.setdefault("GOOGLE_MAPS_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

_MOD_NAME = "agent4_tools"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "tools.py")
_tools = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _tools
_spec.loader.exec_module(_tools)

MOCK = {"results":[{"place_id":"abc","name":"La Belle Époque","geometry":{"location":{"lat":48.8601,"lng":2.3477}},"vicinity":"10 Rue Rivoli","rating":4.6,"price_level":2,"types":["french_restaurant","restaurant"]},{"place_id":"def","name":"Sushi Sakura","geometry":{"location":{"lat":48.8611,"lng":2.3490}},"vicinity":"22 Rue Denis","rating":4.3,"price_level":3,"types":["japanese_restaurant","restaurant"]}],"status":"OK"}

@pytest.mark.asyncio
async def test_returns_list():
    with patch(f"{_MOD_NAME}.googlemaps.Client") as MC:
        MC.return_value.places_nearby.return_value = MOCK
        r = await _tools.search_restaurants(latitude=48.86,longitude=2.35)
        assert len(r)==2 and r[0]["name"]=="La Belle Époque"

@pytest.mark.asyncio
async def test_price_filter():
    with patch(f"{_MOD_NAME}.googlemaps.Client") as MC:
        MC.return_value.places_nearby.return_value = MOCK
        r = await _tools.search_restaurants(latitude=48.86,longitude=2.35,max_price_level=2)
        assert len(r)==1 and r[0]["name"]=="La Belle Époque"

@pytest.mark.asyncio
async def test_cuisine_keyword():
    with patch(f"{_MOD_NAME}.googlemaps.Client") as MC:
        MC.return_value.places_nearby.return_value = {"results":[],"status":"OK"}
        await _tools.search_restaurants(latitude=48.86,longitude=2.35,cuisine_type="Japanese")
        assert MC.return_value.places_nearby.call_args[1]["keyword"]=="Japanese"

def test_extract_french():
    assert "french" in _tools._extract_cuisines(["french_restaurant","restaurant"])

def test_extract_fallback():
    assert _tools._extract_cuisines(["point_of_interest"])==["restaurant"]

def test_schema():
    fn = _tools.TOOLS[0]["function"]
    assert fn["name"]=="search_restaurants"
    assert set(fn["parameters"]["required"])=={"latitude","longitude"}
