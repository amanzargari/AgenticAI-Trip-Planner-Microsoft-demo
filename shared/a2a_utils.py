from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import httpx
from fasta2a.client import A2AClient
from fasta2a.schema import Artifact, Message


async def call_agent(
    url: str,
    data: dict[str, Any],
    context_id: str | None = None,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Send a task to an A2A agent and block until it completes, returning the artifact data."""
    cid = context_id or str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as http_client:
        client = A2AClient(url, http_client=http_client)

        message: Message = {
            "role": "user",
            "kind": "message",
            "message_id": str(uuid.uuid4()),
            "context_id": cid,
            "parts": [{"kind": "data", "data": data}],
        }

        response = await client.send_message(message)

        if response.get("error"):
            raise RuntimeError(f"Agent error on send: {response['error']}")

        task = response["result"]
        task_id = task["id"]

        deadline = asyncio.get_event_loop().time() + timeout
        while task["status"]["state"] not in ("completed", "failed", "canceled"):
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Agent task {task_id} timed out after {timeout}s")
            await asyncio.sleep(poll_interval)
            get_resp = await client.get_task(task_id)
            task = get_resp["result"]

        if task["status"]["state"] != "completed":
            raise RuntimeError(
                f"Agent task {task_id} ended with state: {task['status']['state']}"
            )

        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part["kind"] == "data":
                    return part["data"]
                if part["kind"] == "text":
                    return json.loads(part["text"])

        raise RuntimeError("Agent completed but returned no artifacts")


def make_data_artifact(data: dict[str, Any], name: str = "result") -> Artifact:
    return {
        "artifact_id": str(uuid.uuid4()),
        "name": name,
        "parts": [{"kind": "data", "data": data}],
    }


def extract_message_data(message: Message) -> dict[str, Any]:
    """Pull the first data or text part from an A2A message into a plain dict."""
    for part in message.get("parts", []):
        if part["kind"] == "data":
            return part["data"]
        if part["kind"] == "text":
            try:
                return json.loads(part["text"])
            except json.JSONDecodeError:
                return {"text": part["text"]}
    return {}
