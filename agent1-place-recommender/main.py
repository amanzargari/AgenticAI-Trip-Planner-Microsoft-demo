from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Skill
from fasta2a.storage import InMemoryStorage

from worker import PlaceRecommenderWorker

PORT = int(os.getenv("PORT", "8001"))
SERVICE_URL = os.getenv("AGENT1_URL", f"http://localhost:{PORT}")

_broker = InMemoryBroker()
_storage: InMemoryStorage[None] = InMemoryStorage()
_worker = PlaceRecommenderWorker(broker=_broker, storage=_storage)


@asynccontextmanager
async def _lifespan(app: FastA2A):
    async with app.task_manager:
        async with _worker.run():
            yield


app = FastA2A(
    storage=_storage,
    broker=_broker,
    name="Place Recommender Agent",
    url=SERVICE_URL,
    description=(
        "Generates a list of candidate tourist attractions and places of interest "
        "for a city trip, using the Google Places Text Search API and LLM reasoning."
    ),
    skills=[
        Skill(
            id="place-recommendation",
            name="Place Recommendation",
            description="Discover places to visit for a city trip based on preferences and budget.",
            tags=["places", "tourism", "recommendation", "attractions"],
            input_modes=["application/json"],
            output_modes=["application/json"],
        )
    ],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
