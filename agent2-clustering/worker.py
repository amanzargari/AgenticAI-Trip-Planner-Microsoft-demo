from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from shared.a2a_utils import extract_message_data, make_data_artifact
from shared.llm import DEFAULT_MODEL, get_llm_client
from tools import TOOLS, cluster_places, num_days_from_dates

MODEL = os.getenv("AGENT2_MODEL", DEFAULT_MODEL)


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Clustering Agent.
Goal: split place candidates into day-wise geographic clusters.

Input (JSON):
- trip_start: ISO datetime
- trip_end: ISO datetime
- place_candidates: list[place]

Rules:
1) Compute trip day count using ceiling((trip_end - trip_start) / 24h).
2) Call cluster_places(places, num_clusters=day_count).
3) Preserve place objects exactly; do not remove fields.
4) If place_candidates is empty, return an empty cluster list.

Output rules (STRICT):
- Return ONLY JSON (no markdown or prose).
- Return exactly:
{
    "clustered_place_candidates": [
        [<place>, <place>],
        [<place>]
    ]
}
- Each inner list corresponds to one day cluster.
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

        data: dict[str, Any] = {}
        try:
            data = extract_message_data(params["message"])

            # Deterministic fast path for orchestrator payloads.
            # This avoids unnecessary LLM/tool loops and prevents agent failures
            # from bubbling up when clustering input is degenerate.
            places = data.get("place_candidates")
            if isinstance(places, list):
                try:
                    n_days = num_days_from_dates(
                        str(data.get("trip_start") or ""),
                        str(data.get("trip_end") or ""),
                    )
                except Exception:
                    n_days = 1

                try:
                    clusters = cluster_places(places, n_days)
                except Exception:
                    logger.exception("Clustering fast-path crashed task_id=%s", task_id)
                    clusters = [places] if places else []

                result = {"clustered_place_candidates": clusters}
                await self.storage.update_task(
                    task_id,
                    state="completed",
                    new_artifacts=self.build_artifacts(result),
                )
                return

            # LLM path for edge cases / natural-language input
            llm = get_llm_client()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(data)},
            ]

            tool_called_2 = False
            for _ in range(6):
                response = await llm.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto" if tool_called_2 else "required",
                )
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    tool_called_2 = True
                    tool_calls = choice.message.tool_calls or []
                    messages.append(choice.message.model_dump(exclude_none=True))
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as exc:
                            tool_result = {"error": f"Invalid arguments: {exc}"}
                        else:
                            if tc.function.name == "cluster_places":
                                try:
                                    tool_result = cluster_places(**args)
                                except Exception as exc:
                                    logger.exception("LLM tool cluster_places failed task_id=%s", task_id)
                                    tool_result = {"error": f"cluster_places failed: {exc}"}
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
            logger.exception("Clustering worker crashed task_id=%s", task_id)
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
