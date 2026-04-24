from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Skill
from fasta2a.storage import InMemoryStorage

from worker import FoodRecommenderWorker

PORT = int(os.getenv("PORT", "8004"))
SERVICE_URL = os.getenv("AGENT4_URL", f"http://localhost:{PORT}")

_broker = InMemoryBroker()
_storage: InMemoryStorage[None] = InMemoryStorage()
_worker = FoodRecommenderWorker(broker=_broker, storage=_storage)


@asynccontextmanager
async def _lifespan(app: FastA2A):
    async with app.task_manager:
        async with _worker.run():
            yield


app = FastA2A(
    storage=_storage,
    broker=_broker,
    name="Food Recommender Agent",
    url=SERVICE_URL,
    description=(
        "Recommends restaurants for a specific meal context "
        "(location, time of day, budget, dietary preferences) "
        "using the Google Places Nearby Search API."
    ),
    skills=[
        Skill(
            id="food-recommendation",
            name="Food Recommendation",
            description="Find nearby restaurants matching a meal context and budget.",
            tags=["food", "restaurant", "recommendation", "places"],
            input_modes=["application/json"],
            output_modes=["application/json"],
        )
    ],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
