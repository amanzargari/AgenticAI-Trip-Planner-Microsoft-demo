"""Tests for agent2 geographic clustering tools."""
from __future__ import annotations
import importlib.util, pathlib, sys, os
import pytest

_AGENT_DIR = pathlib.Path(__file__).parent.parent
_ROOT_DIR  = _AGENT_DIR.parent
for _d in (str(_ROOT_DIR),):
    if _d not in sys.path: sys.path.insert(0, _d)

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

_MOD_NAME = "agent2_tools"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENT_DIR / "tools.py")
_tools = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _tools
_spec.loader.exec_module(_tools)

def _p(pid, lat, lng):
    return {"id":pid,"name":f"P{pid}","location":{"latitude":lat,"longitude":lng,"address":""},"estimated_visit_duration_minutes":60,"category":"attraction","rating":4.0}

NORTH = [_p(f"n{i}", 48.86+i*0.002, 2.33) for i in range(4)]
SOUTH = [_p(f"s{i}", 48.84+i*0.002, 2.35) for i in range(4)]

def test_two_clusters():
    c = _tools.cluster_places(NORTH+SOUTH, 2)
    assert len(c)==2 and sum(len(x) for x in c)==8

def test_single_cluster():
    c = _tools.cluster_places(NORTH, 1)
    assert len(c)==1 and len(c[0])==4

def test_empty_input():
    assert _tools.cluster_places([], 3)==[]

def test_fewer_places_than_clusters():
    two = [_p("a",48.86,2.33), _p("b",48.84,2.35)]
    c = _tools.cluster_places(two, 5)
    assert len(c)<=2 and sum(len(x) for x in c)==2

def test_geographic_coherence():
    c = _tools.cluster_places(NORTH+SOUTH, 2)
    lats = [sum(p["location"]["latitude"] for p in x)/len(x) for x in c]
    assert abs(lats[0]-lats[1])>0.01

def test_identical_coordinates_do_not_crash():
    same = [_p(f"x{i}", 48.8566, 2.3522) for i in range(4)]
    c = _tools.cluster_places(same, 2)
    assert len(c) == 2 and sum(len(x) for x in c) == 4

def test_non_positive_num_clusters_is_sanitized():
    two = [_p("a", 48.86, 2.33), _p("b", 48.84, 2.35)]
    c0 = _tools.cluster_places(two, 0)
    cneg = _tools.cluster_places(two, -3)
    assert len(c0) == 1 and sum(len(x) for x in c0) == 2
    assert len(cneg) == 1 and sum(len(x) for x in cneg) == 2

def test_num_days_three():
    assert _tools.num_days_from_dates("2026-06-01T09:00:00","2026-06-04T09:00:00")==3

def test_num_days_partial():
    assert _tools.num_days_from_dates("2026-06-01T08:00:00","2026-06-02T09:00:00")==2

def test_num_days_one():
    assert _tools.num_days_from_dates("2026-06-01T09:00:00","2026-06-01T23:00:00")==1

def test_schema():
    fn = _tools.TOOLS[0]["function"]
    assert fn["name"]=="cluster_places"
    assert "places" in fn["parameters"]["properties"]
