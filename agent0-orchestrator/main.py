from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Skill
from fasta2a.storage import InMemoryStorage

from worker import OrchestratorWorker

PORT = int(os.getenv("PORT", "8000"))
SERVICE_URL = os.getenv("ORCHESTRATOR_URL", f"http://localhost:{PORT}")

_broker = InMemoryBroker()
_storage: InMemoryStorage[None] = InMemoryStorage()
_worker = OrchestratorWorker(broker=_broker, storage=_storage)


@asynccontextmanager
async def _lifespan(app: FastA2A):
    async with app.task_manager:
        async with _worker.run():
            yield


app = FastA2A(
    storage=_storage,
    broker=_broker,
    name="Trip Planner Orchestrator",
    url=SERVICE_URL,
    description=(
        "Main entry point for trip planning. Accepts a trip request and orchestrates "
        "the Place Recommender, Clustering, and Daily Scheduler agents via tool-calling "
        "to produce a complete multi-day itinerary."
    ),
    skills=[
        Skill(
            id="trip-planning",
            name="Trip Planning",
            description=(
                "Generate a complete day-by-day trip itinerary for a city "
                "given dates, budget, and preferences."
            ),
            tags=["trip", "planning", "orchestration", "itinerary"],
            examples=[
                "Plan a 3-day trip to Paris with a €500 budget focused on art and history.",
                "Create an itinerary for Rome from 2026-06-10 to 2026-06-13.",
            ],
            input_modes=["application/json"],
            output_modes=["application/json"],
        )
    ],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
