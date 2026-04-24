from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from tools import TOOLS, cluster_places, num_days_from_dates

SYSTEM_PROMPT = """\
You are a geographic clustering agent for trip planning.

You receive a JSON object with:
  - trip_start         : ISO datetime
  - trip_end           : ISO datetime
  - place_candidates   : list of place objects

Steps:
1. Compute the number of trip days from trip_start and trip_end (use ceiling division).
2. Call cluster_places with all place_candidates and num_clusters = number of days.
3. Return the result immediately.

Return ONLY a JSON object (no markdown) with a single key:
  "clustered_place_candidates": [ [<places for day 1>], [<places for day 2>], ... ]

Each inner list represents places intended for one day.
Keep all place fields intact in the output.
"""


@dataclass
class ClusteringWorker(Worker[None]):

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

            # Fast path: compute directly without LLM if data is well-structured
            if (
                "place_candidates" in data
                and "trip_start" in data
                and "trip_end" in data
            ):
                try:
                    n_days = num_days_from_dates(data["trip_start"], data["trip_end"])
                    clusters = cluster_places(data["place_candidates"], n_days)
                    result = {"clustered_place_candidates": clusters}
                    await self.storage.update_task(
                        task_id,
                        state="completed",
                        new_artifacts=self.build_artifacts(result),
                    )
                    return
                except Exception:
                    pass  # fall through to LLM path

            # LLM path for edge cases / natural-language input
            llm = get_llm_client()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            for _ in range(6):
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
                        if tc.function.name == "cluster_places":
                            tool_result = cluster_places(**args)
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
        return {"clustered_place_candidates": []}
