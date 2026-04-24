"""Web UI backend – thin FastAPI layer that forwards requests to the Orchestrator."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared.a2a_utils import call_agent

ORCHESTRATOR_URL: str = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)

app = FastAPI(title="Trip Planner UI", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request/response models ───────────────────────────────────────────────────

class TripRequest(BaseModel):
    city: str
    trip_start: str          # ISO datetime
    trip_end: str            # ISO datetime
    total_budget: float | None = None
    trip_reason: str | None = None
    preferences: list[str] = []


class ChatRequest(BaseModel):
    message: str
    current_itinerary: dict | None = None
    original_params: dict | None = None


class AgentStatusResponse(BaseModel):
    name: str
    url: str
    status: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/plan")
async def plan_trip(req: TripRequest) -> JSONResponse:
    """Forward an initial trip planning request to the Orchestrator."""
    start_time = time.perf_counter()
    logger.info(
        "Plan request received city=%s trip_start=%s trip_end=%s has_budget=%s preference_count=%d",
        req.city, req.trip_start, req.trip_end,
        req.total_budget is not None, len(req.preferences),
    )

    budget_payload = (
        {"total_budget": req.total_budget, "currency": "EUR"}
        if req.total_budget
        else None
    )
    payload: dict[str, Any] = {
        "city":        req.city,
        "trip_start":  req.trip_start,
        "trip_end":    req.trip_end,
        "budget":      budget_payload,
        "trip_reason": req.trip_reason,
        "preferences": req.preferences,
    }

    try:
        result = await call_agent(ORCHESTRATOR_URL, payload, timeout=600.0, poll_interval=3.0)
        elapsed = time.perf_counter() - start_time
        schedules = len(result.get("schedules", [])) if isinstance(result, dict) else "unknown"
        logger.info(
            "Plan request succeeded city=%s elapsed_sec=%.2f schedules=%s",
            req.city, elapsed, schedules,
        )
        return JSONResponse(content=result)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except Exception as exc:
        logger.exception("Plan request failed city=%s", req.city)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    """Handle a follow-up modification request against an existing itinerary."""
    if not req.current_itinerary or not req.original_params:
        raise HTTPException(
            status_code=400,
            detail="current_itinerary and original_params are required for chat modifications.",
        )

    start_time = time.perf_counter()
    logger.info("Chat modification request: %s", req.message[:120])

    payload: dict[str, Any] = {
        **req.original_params,
        "modification_request": req.message,
        "current_itinerary":    req.current_itinerary,
    }

    try:
        result = await call_agent(ORCHESTRATOR_URL, payload, timeout=600.0, poll_interval=3.0)
        elapsed = time.perf_counter() - start_time
        logger.info("Chat modification succeeded elapsed_sec=%.2f", elapsed)
        return JSONResponse(content={"itinerary": result})
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except Exception as exc:
        logger.exception("Chat modification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/agents/status")
async def agents_status() -> list[AgentStatusResponse]:
    """Check the health of all agent services."""
    agents = [
        ("Orchestrator",     ORCHESTRATOR_URL),
        ("Place Recommender", os.getenv("AGENT1_URL", "http://localhost:8001")),
        ("Clustering",        os.getenv("AGENT2_URL", "http://localhost:8002")),
        ("Daily Scheduler",   os.getenv("AGENT3_URL", "http://localhost:8003")),
        ("Food Recommender",  os.getenv("AGENT4_URL", "http://localhost:8004")),
    ]
    statuses: list[AgentStatusResponse] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in agents:
            try:
                resp = await client.get(f"{url}/.well-known/agent-card.json")
                status = "online" if resp.status_code == 200 else f"error {resp.status_code}"
            except Exception:
                status = "offline"
            statuses.append(AgentStatusResponse(name=name, url=url, status=status))
    return statuses
