from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Skill
from fasta2a.storage import InMemoryStorage

from worker import SchedulerWorker

PORT = int(os.getenv("PORT", "8003"))
SERVICE_URL = os.getenv("AGENT3_URL", f"http://localhost:{PORT}")

_broker = InMemoryBroker()
_storage: InMemoryStorage[None] = InMemoryStorage()
_worker = SchedulerWorker(broker=_broker, storage=_storage)


@asynccontextmanager
async def _lifespan(app: FastA2A):
    async with app.task_manager:
        async with _worker.run():
            yield


app = FastA2A(
    storage=_storage,
    broker=_broker,
    name="Daily Scheduler Agent",
    url=SERVICE_URL,
    description=(
        "Creates a chronological daily schedule from a cluster of places. "
        "Orders visits by proximity, estimates travel times, and inserts "
        "meal breaks by calling the Food Recommender agent."
    ),
    skills=[
        Skill(
            id="daily-scheduling",
            name="Daily Scheduling",
            description="Build a full day itinerary from a list of places including meal breaks.",
            tags=["scheduling", "planning", "itinerary"],
            input_modes=["application/json"],
            output_modes=["application/json"],
        )
    ],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
