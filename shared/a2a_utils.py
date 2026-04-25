from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import httpx
from fasta2a.client import A2AClient
from fasta2a.schema import Artifact, Message


logger = logging.getLogger(__name__)


async def call_agent(
    url: str,
    data: dict[str, Any],
    context_id: str | None = None,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Send a task to an A2A agent and block until it completes, returning the artifact data."""
    cid = context_id or str(uuid.uuid4())
    logger.info(
        "A2A call start: url=%s context_id=%s payload_keys=%s",
        url,
        cid,
        sorted(data.keys()),
    )

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
            logger.error("A2A send_message failed: url=%s error=%s", url, response["error"])
            raise RuntimeError(f"Agent error on send: {response['error']}")

        task = response["result"]
        task_id = task["id"]
        status = task.get("status", {})
        last_state = status.get("state")
        logger.info(
            "A2A task accepted: url=%s task_id=%s state=%s",
            url,
            task_id,
            last_state,
        )

        deadline = asyncio.get_event_loop().time() + timeout
        while task["status"]["state"] not in ("completed", "failed", "canceled"):
            if asyncio.get_event_loop().time() > deadline:
                logger.error(
                    "A2A task timeout: url=%s task_id=%s last_state=%s status=%s",
                    url,
                    task_id,
                    task.get("status", {}).get("state"),
                    task.get("status", {}),
                )
                raise TimeoutError(
                    f"Agent task {task_id} timed out after {timeout}s "
                    f"(last_state={task.get('status', {}).get('state')})"
                )
            await asyncio.sleep(poll_interval)
            get_resp = await client.get_task(task_id)
            if get_resp.get("error"):
                logger.error(
                    "A2A get_task failed: url=%s task_id=%s error=%s",
                    url,
                    task_id,
                    get_resp["error"],
                )
                raise RuntimeError(f"Agent error on get_task: {get_resp['error']}")
            task = get_resp["result"]
            current_state = task.get("status", {}).get("state")
            if current_state != last_state:
                logger.info(
                    "A2A task state change: url=%s task_id=%s from=%s to=%s",
                    url,
                    task_id,
                    last_state,
                    current_state,
                )
                last_state = current_state

        final_state = task.get("status", {}).get("state")
        if final_state != "completed":
            status = task.get("status", {})
            status_message = status.get("message") or status.get("reason")
            logger.error(
                "A2A task ended unsuccessfully: url=%s task_id=%s state=%s status=%s",
                url,
                task_id,
                final_state,
                status,
            )
            detail = f"Agent task {task_id} ended with state: {final_state}"
            if status_message:
                detail += f" ({status_message})"
            raise RuntimeError(
                detail
            )

        artifacts = task.get("artifacts", [])
        logger.info(
            "A2A task completed: url=%s task_id=%s artifacts=%d",
            url,
            task_id,
            len(artifacts),
        )

        payload = _extract_artifact_payload(
            artifacts=artifacts,
            task_id=task_id,
            url=url,
        )
        if payload is not None:
            return payload

        logger.error(
            "A2A task completed without artifacts: url=%s task_id=%s status=%s",
            url,
            task_id,
            task.get("status", {}),
        )
        raise RuntimeError("Agent completed but returned no artifacts")


def _extract_artifact_payload(
    artifacts: list[dict[str, Any]],
    task_id: str,
    url: str,
) -> dict[str, Any] | None:
    """Pick the most relevant artifact payload (prefer latest non-partial result)."""
    partial_trace_payload: dict[str, Any] | None = None

    for artifact in reversed(artifacts):
        parts = artifact.get("parts", []) if isinstance(artifact, dict) else []
        for part in reversed(parts):
            if not isinstance(part, dict):
                continue

            kind = part.get("kind")
            if kind == "data":
                raw = part.get("data", {})
                payload = raw if isinstance(raw, dict) else {"value": raw}
                if _is_partial_trace_payload(payload):
                    if partial_trace_payload is None:
                        partial_trace_payload = payload
                    continue
                return payload

            if kind == "text":
                text_payload = part.get("text", "")
                try:
                    decoded = json.loads(text_payload)
                except json.JSONDecodeError as exc:
                    logger.exception(
                        "A2A text artifact JSON decode failed: url=%s task_id=%s excerpt=%s",
                        url,
                        task_id,
                        _short(text_payload),
                    )
                    raise RuntimeError(
                        f"Agent task {task_id} returned non-JSON text artifact"
                    ) from exc

                payload = decoded if isinstance(decoded, dict) else {"value": decoded}
                if _is_partial_trace_payload(payload):
                    if partial_trace_payload is None:
                        partial_trace_payload = payload
                    continue
                return payload

    return partial_trace_payload


def _is_partial_trace_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload, dict) and set(payload.keys()) == {"partial_trace"}


def _short(value: Any, max_len: int = 240) -> str:
    text = str(value)
    return text if len(text) <= max_len else f"{text[:max_len]}…"


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
