"""Tests for agent1 tools – Google API calls mocked."""
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

_MOD_NAME = "agent1_tools"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "tools.py")
_tools = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _tools
_spec.loader.exec_module(_tools)

MOCK_GEOCODE = [{"geometry":{"location":{"lat":48.8566,"lng":2.3522}}}]
MOCK_PLACES  = {"results":[{"place_id":"louvre","name":"Musée du Louvre","formatted_address":"Paris","geometry":{"location":{"lat":48.8606,"lng":2.3376}},"rating":4.7,"types":["museum","tourist_attraction"]},{"place_id":"eiffel","name":"Tour Eiffel","formatted_address":"Paris","geometry":{"location":{"lat":48.8584,"lng":2.2945}},"rating":4.6,"types":["tourist_attraction","landmark"]}],"status":"OK"}

@pytest.mark.asyncio
async def test_geocode_city_returns_coords():
    with patch(f"{_MOD_NAME}.googlemaps.Client") as MC:
        MC.return_value.geocode.return_value = MOCK_GEOCODE
        coords = await _tools.geocode_city("Paris")
        assert coords["latitude"] == pytest.approx(48.8566)

@pytest.mark.asyncio
async def test_geocode_city_unknown_raises():
    with patch(f"{_MOD_NAME}.googlemaps.Client") as MC:
        MC.return_value.geocode.return_value = []
        with pytest.raises(ValueError, match="Could not geocode"):
            await _tools.geocode_city("NowhereCity12345")

@pytest.mark.asyncio
async def test_search_places_returns_candidates():
    with patch(f"{_MOD_NAME}.googlemaps.Client") as MC:
        MC.return_value.geocode.return_value = MOCK_GEOCODE
        MC.return_value.places.return_value = MOCK_PLACES
        results = await _tools.search_places(query="museums in Paris", city="Paris")
        assert len(results) == 2
        assert results[0]["name"] == "Musée du Louvre"
        assert results[0]["estimated_visit_duration_minutes"] == 120

@pytest.mark.asyncio
async def test_search_places_caps_at_15():
    many = {"results":[{"place_id":f"p{i}","name":f"P{i}","formatted_address":"","geometry":{"location":{"lat":48.0+i*0.01,"lng":2.0}},"rating":4.0,"types":["tourist_attraction"]} for i in range(25)],"status":"OK"}
    with patch(f"{_MOD_NAME}.googlemaps.Client") as MC:
        MC.return_value.geocode.return_value = MOCK_GEOCODE
        MC.return_value.places.return_value = many
        results = await _tools.search_places(query="stuff", city="Paris")
        assert len(results) <= 15

def test_primary_category_priority():
    assert _tools._primary_category(["museum","tourist_attraction"]) == "museum"
    assert _tools._primary_category([]) == "attraction"

def test_estimate_duration_museum():
    assert _tools._estimate_duration(["museum"]) == 120

def test_estimate_duration_church():
    assert _tools._estimate_duration(["church"]) == 45

def test_tools_schema_valid():
    names = [t["function"]["name"] for t in _tools.TOOLS]
    assert "search_places" in names and "geocode_city" in names
