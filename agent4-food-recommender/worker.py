from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from tools import TOOLS, search_restaurants

SYSTEM_PROMPT = """\
You are the Food Recommender Agent.
Goal: suggest practical meal options near the requested location and budget.

Input (JSON):
- time_of_day: breakfast | lunch | dinner | ISO time string
- search_center: {"latitude": float, "longitude": float}
- search_radius_meters: int
- budget_per_meal_per_person: float | null
- preferences: list[str]

Tool strategy:
1) Call search_restaurants at least once.
2) If preferences exist, run additional searches with cuisine-specific keywords.
3) Deduplicate overlapping restaurants and keep best options by relevance/rating.
4) If a tool call fails, continue with remaining searches.

Output rules (STRICT):
- Return ONLY JSON (no markdown, no prose).
- Return exactly:
{
    "restaurants": [
        {
            "id": "...",
            "name": "...",
            "location": {"latitude": 0.0, "longitude": 0.0, "address": "..."},
            "price_level": 2,
            "cuisines": ["italian"],
            "rating": 4.4,
            "summary": "..."
        }
    ]
}
- Return up to 5 items; if none are available return {"restaurants": []}.
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

            for _ in range(8):
                response = await llm.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    tool_calls = choice.message.tool_calls or []
                    messages.append(
                        {
                            "role": "assistant",
                            "content": choice.message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )
                    for tc in tool_calls:
                        args = json.loads(tc.function.arguments)
                        if tc.function.name == "search_restaurants":
                            tool_result = await search_restaurants(**args)
                        else:
                            tool_result = {"error": f"Unknown tool: {tc.function.name}"}
                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tc.id,
                            }
                        )
                else:
                    raw = choice.message.content or "{}"
                    result = _parse_json(raw)
                    await self.storage.update_task(
                        task_id,
                        state="completed",
                        new_artifacts=self.build_artifacts(result),
                    )
                    return

            await self.storage.update_task(task_id, state="failed")

        except Exception:
            await self.storage.update_task(task_id, state="failed")
            raise


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"restaurants": []}
