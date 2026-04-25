"""Web UI backend – thin FastAPI layer that forwards requests to the Orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fasta2a.client import A2AClient
from pydantic import BaseModel

from shared.a2a_utils import call_agent, submit_agent_task

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

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


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


@app.post("/api/plan/submit")
async def plan_trip_submit(req: TripRequest) -> JSONResponse:
    """Submit trip planning task and immediately return task_id for SSE streaming."""
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
        task_id = await submit_agent_task(ORCHESTRATOR_URL, payload)
        return JSONResponse({"task_id": task_id})
    except Exception as exc:
        logger.exception("plan/submit failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/plan/stream/{task_id}")
async def plan_trip_stream(task_id: str) -> StreamingResponse:
    """SSE stream: emits day schedules as they complete, then a final done event."""

    async def _generate():
        seen_dates: set[str] = set()
        last_trace_count = 0
        deadline = asyncio.get_event_loop().time() + 660.0

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as http:
            client = A2AClient(ORCHESTRATOR_URL, http_client=http)

            while True:
                if asyncio.get_event_loop().time() > deadline:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'timeout'})}\n\n"
                    return

                try:
                    resp = await client.get_task(task_id)
                    if resp.get("error"):
                        yield f"data: {json.dumps({'type': 'error', 'message': str(resp['error'])})}\n\n"
                        return
                    task = resp["result"]
                except Exception as exc:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                    return

                state = task.get("status", {}).get("state", "unknown")
                artifacts = task.get("artifacts", [])

                # Scan artifacts for the latest partial result
                partial: dict[str, Any] | None = None
                for artifact in reversed(artifacts):
                    for part in reversed(artifact.get("parts", [])):
                        if part.get("kind") == "data":
                            d = part.get("data", {})
                            if isinstance(d, dict) and d.get("partial") is True:
                                partial = d
                                break
                    if partial:
                        break

                if partial:
                    for schedule in (partial.get("schedules") or []):
                        date = schedule.get("date", "")
                        if date and date not in seen_dates:
                            seen_dates.add(date)
                            yield f"data: {json.dumps({'type': 'day', 'schedule': schedule})}\n\n"

                    trace = partial.get("trace") or []
                    if len(trace) > last_trace_count:
                        new_events = trace[last_trace_count:]
                        last_trace_count = len(trace)
                        yield f"data: {json.dumps({'type': 'trace', 'events': new_events})}\n\n"

                if state in ("completed", "failed", "canceled"):
                    if state == "completed":
                        # Find the final non-partial artifact (most recent without partial flag)
                        final: dict[str, Any] | None = None
                        for artifact in reversed(artifacts):
                            for part in reversed(artifact.get("parts", [])):
                                if part.get("kind") == "data":
                                    d = part.get("data", {})
                                    if isinstance(d, dict) and not d.get("partial"):
                                        final = d
                                        break
                            if final:
                                break
                        if final:
                            yield f"data: {json.dumps({'type': 'done', 'itinerary': final})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'message': 'No final artifact returned'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Task ended with state: {state}'})}\n\n"
                    return

                await asyncio.sleep(2.0)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
