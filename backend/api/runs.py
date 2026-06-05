"""
API router for workflow run management and server-sent event streaming.

What it does:
    Lets clients start a workflow run against a stored schema, stream
    real-time execution events via SSE (backed by ``asyncio.Queue``), and
    list historical run records.

Entities in it:
    - router: FastAPI APIRouter mounted at ``/api/runs``.
    - init: Dependency-injection entry point called at application startup.

How used by other modules:
    ``backend.main`` calls ``init(executor, schema_persistence)`` during
    lifespan startup and includes the returned router in the FastAPI app.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from backend.orchestration.executor import WorkflowExecutor
from backend.schema.persistence import SchemaPersistence

router = APIRouter(prefix="/api/runs", tags=["runs"])

_executor: WorkflowExecutor | None = None
_schema_persistence: SchemaPersistence | None = None
_event_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}


def init(
    executor: WorkflowExecutor,
    schema_persistence: SchemaPersistence,
) -> APIRouter:
    """Wire backend dependencies and return the configured router.

    Args:
        executor: WorkflowExecutor for running schemas.
        schema_persistence: SchemaPersistence for loading schemas by ID.

    Returns:
        The fully configured APIRouter.
    """
    global _executor, _schema_persistence  # noqa: PLW0603
    _executor = executor
    _schema_persistence = schema_persistence
    return router


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{schema_id}/start")
async def start_run(schema_id: str) -> dict[str, str]:
    """Start a new workflow run for *schema_id*.

    The run executes in the background.  Clients should connect to
    ``GET /{schema_id}/stream`` immediately after to receive events.

    Args:
        schema_id: ID of the schema to run.

    Returns:
        ``{"schema_id": …, "status": "started"}``.

    Raises:
        HTTPException 404: If the schema does not exist.
    """
    try:
        schema = _schema_persistence.load_schema(schema_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"Schema '{schema_id}' not found")

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    _event_queues[schema_id] = queue

    async def event_callback(event: dict[str, Any]) -> None:
        """Push every executor event into the SSE queue."""
        await queue.put(event)

    async def run_task() -> None:
        """Background coroutine that drives the executor and signals done."""
        try:
            await _executor.execute_workflow(schema, event_callback)
        except Exception as exc:
            await queue.put({"type": "workflow_error", "error": str(exc)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_task())

    return {"schema_id": schema_id, "status": "started"}


@router.get("/{schema_id}/stream")
async def stream_events(schema_id: str) -> EventSourceResponse:
    """Stream execution events for the active run on *schema_id* via SSE.

    The connection stays open until a terminal event (``workflow_completed``,
    ``workflow_error``, ``workflow_timeout``) is received, after which the
    server closes the stream.

    Args:
        schema_id: Schema whose active run to stream.

    Returns:
        An ``EventSourceResponse`` yielding JSON-encoded event dicts.

    Raises:
        HTTPException 404: If there is no active run queue for *schema_id*.
    """
    queue = _event_queues.get(schema_id)
    if queue is None:
        raise HTTPException(
            404, detail=f"No active run for schema '{schema_id}'"
        )

    async def event_generator():
        """Yield SSE data frames until the run completes."""
        while True:
            event = await queue.get()
            if event is None:
                break
            yield {"data": json.dumps(event, default=str)}

    return EventSourceResponse(event_generator())


@router.get("/")
async def list_records() -> list[dict[str, Any]]:
    """List all stored run records.

    Returns:
        A JSON array of run-record summaries (newest first).
    """
    records = _executor.list_run_records()
    return [
        {
            "run_id": record.run_id,
            "schema_id": record.schema_id,
            "status": record.status,
            "started_at": record.started_at.isoformat(),
            "completed_at": (
                record.completed_at.isoformat()
                if record.completed_at
                else None
            ),
            "errors": record.errors,
        }
        for record in records
    ]
