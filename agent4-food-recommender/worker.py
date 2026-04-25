from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker
from pydantic import BaseModel

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client

MODEL = os.getenv("AGENT4_MODEL", DEFAULT_MODEL)
from shared.models import RestaurantCandidate
from tools import TOOLS, search_restaurants

logger = logging.getLogger(__name__)


# ── Structured output model ───────────────────────────────────────────────────

class FoodOutput(BaseModel):
    restaurants: list[RestaurantCandidate]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Food Recommender Agent.
Goal: suggest practical, high-quality meal options near the requested location and budget.

Input (JSON):
- time_of_day: breakfast | lunch | dinner | ISO time string
- search_center: {"latitude": float, "longitude": float}
- search_radius_meters: int (default 500)
- budget_per_meal_per_person: float | null
- preferences: list[str]

Tool strategy:
1) Call search_restaurants with the matching meal_slot (breakfast/lunch/dinner).
   Pass budget_per_person if provided so results are filtered by price level.
2) If preferences include cuisine types (e.g. "vegetarian", "italian", "japanese"),
   run additional searches with cuisine_type set to that keyword.
3) Deduplicate overlapping results; keep the best options by rating.
4) If a tool call fails, continue with remaining searches.

Return up to 5 restaurants. If none available, return an empty list.
"""


@dataclass
class FoodRecommenderWorker(Worker[None]):

    def build_message_history(self, history: list[Message]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = ""
            for part in msg.get("parts", []):
                if part["kind"] == "text":
                    content = part["text"]
                elif part["kind"] == "data":
                    content = json.dumps(part["data"])
            messages.append({"role": role, "content": content})
        return messages

    def build_artifacts(self, result: Any) -> list[Artifact]:
        if not isinstance(result, dict):
            result = {"result": result}
        return [make_data_artifact(result)]

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self.storage.update_task(params["id"], state="canceled")

    async def run_task(self, params: TaskSendParams) -> None:
        task_id = params["id"]
        await self.storage.update_task(task_id, state="working")

        try:
            data = extract_message_data(params["message"])
            llm = get_llm_client()

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            # Phase 1: tool calls — no response_format (Gemini rejects tool_choice=required + response_format together)
            tool_called = False
            for _ in range(8):
                response = await llm.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto" if tool_called else "required",
                )
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    tool_called = True
                    tool_calls = choice.message.tool_calls or []
                    messages.append(choice.message.model_dump(exclude_none=True))
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as exc:
                            tool_result: Any = {"error": str(exc)}
                        else:
                            try:
                                if tc.function.name == "search_restaurants":
                                    tool_result = await search_restaurants(**args)
                                else:
                                    tool_result = {"error": f"Unknown tool: {tc.function.name}"}
                            except Exception as exc:
                                tool_result = {"error": str(exc), "restaurants": []}
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": tc.id,
                        })
                else:
                    break

            # Phase 2: structured output — no tools so response_format works without conflict
            try:
                final = await llm.beta.chat.completions.parse(
                    model=MODEL,
                    messages=messages,
                    response_format=FoodOutput,
                )
                parsed: Optional[FoodOutput] = final.choices[0].message.parsed
                result = parsed.model_dump(mode="json") if parsed else {"restaurants": []}
            except Exception:
                logger.warning("FoodRecommender phase-2 parse failed task_id=%s", task_id)
                result = {"restaurants": []}
            await self.storage.update_task(task_id, state="completed",
                                           new_artifacts=self.build_artifacts(result))

        except Exception:
            logger.exception("FoodRecommender task crashed task_id=%s", task_id)
            await self.storage.update_task(task_id, state="failed")
            raise
