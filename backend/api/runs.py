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
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.orchestration.executor import WorkflowExecutor
from backend.run_log_handler import (
    activate_run,
    deactivate_run,
    get_log_file_path,
    get_log_queue,
)
from backend.schema.persistence import SchemaPersistence

router = APIRouter(prefix="/api/runs", tags=["runs"])

_executor: WorkflowExecutor | None = None
_schema_persistence: SchemaPersistence | None = None

# Keyed by run_id so each individual run has its own event queue.
_event_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}


class StartRunRequest(BaseModel):
    """Request body for starting a workflow run.

    Attributes:
        schema_id: Identifier of the schema to execute.
    """

    schema_id: str


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

@router.post("/")
@router.post("")
async def start_run(request: StartRunRequest) -> dict[str, Any]:
    """Start a new workflow run for the schema identified in the request body.

    The run executes in the background.  Clients should connect to
    ``GET /{run_id}/events`` immediately after to receive events.

    Args:
        request: StartRunRequest containing the schema_id to execute.

    Returns:
        A RunRecord-shaped dict containing run_id, schema_id, schema_name,
        status, started_at, finished_at, and error_message.

    Raises:
        HTTPException 404: If the schema does not exist.
    """
    try:
        schema = _schema_persistence.load_schema(request.schema_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"Schema '{request.schema_id}' not found")

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    _event_queues[run_id] = queue

    async def event_callback(raw_event: dict[str, Any]) -> None:
        """Normalise and push every executor event into the SSE queue."""
        normalised = _normalise_event(raw_event, run_id)
        await queue.put(normalised)

    async def run_task() -> None:
        """Background coroutine that drives the executor and signals completion."""
        activate_run(run_id)
        try:
            await _executor.execute_workflow(schema, event_callback, run_id=run_id)
        except Exception as execution_error:
            await queue.put(_make_event(
                run_id=run_id,
                event_type="run_error",
                node_id=None,
                data={"error": str(execution_error)},
            ))
        finally:
            deactivate_run(run_id)
            await queue.put(None)
            _event_queues.pop(run_id, None)

    asyncio.create_task(run_task())

    return {
        "run_id": run_id,
        "schema_id": schema.schema_id,
        "schema_name": schema.name,
        "status": "running",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "error_message": None,
    }


@router.get("/{run_id}/events")
async def stream_events(run_id: str) -> EventSourceResponse:
    """Stream execution events for a run via SSE.

    If the run is still active, streams from the live queue.
    If the run already completed (queue drained), replays stored events
    from the RunRecord.

    Args:
        run_id: Unique identifier of the run to stream.

    Returns:
        An ``EventSourceResponse`` yielding JSON-encoded event dicts.

    Raises:
        HTTPException 404: If run_id is unknown.
    """
    queue = _event_queues.get(run_id)

    async def event_generator():
        """Yield SSE data frames from live queue or stored record."""
        if queue is not None:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield {"data": json.dumps(event, default=str)}
        else:
            record = _executor.get_run_record(run_id) if _executor else None
            if record is None:
                return
            for raw_event in record.events:
                normalised = _normalise_event(raw_event, run_id)
                yield {"data": json.dumps(normalised, default=str)}

    if queue is None:
        try:
            _executor.get_run_record(run_id)
        except (KeyError, AttributeError):
            raise HTTPException(
                404, detail=f"No run found for run_id '{run_id}'"
            )

    return EventSourceResponse(event_generator())


@router.get("/{run_id}/logs/stream")
async def stream_run_logs(run_id: str) -> EventSourceResponse:
    """Stream log lines for run_id as Server-Sent Events.

    Emits named events:
        - "log": one per log record, data is the formatted log line.
        - "status": emitted once when the run finishes, data is "done".

    If the run already completed, replays log lines from the on-disk file.
    """
    log_queue = get_log_queue(run_id)
    log_file = get_log_file_path(run_id)

    if log_queue is None and log_file is None:
        raise HTTPException(
            404, detail=f"No logs available for run_id '{run_id}'"
        )

    async def log_generator():
        if log_queue is not None:
            while True:
                line = await log_queue.get()
                if line is None:
                    yield {"event": "status", "data": "done"}
                    return
                yield {"event": "log", "data": line}
        elif log_file is not None:
            with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    stripped = raw_line.rstrip("\n")
                    if stripped:
                        yield {"event": "log", "data": stripped}
            yield {"event": "status", "data": "done"}

    return EventSourceResponse(log_generator())


@router.get("/")
@router.get("")
async def list_records() -> list[dict[str, Any]]:
    """List all stored run records.

    Returns:
        A JSON array of run-record summaries (newest first), each shaped as
        ``{run_id, schema_id, schema_name, status, started_at, finished_at,
        error_message}``.
    """
    records = _executor.list_run_records()
    return [
        {
            "run_id": record.run_id,
            "schema_id": record.schema_id,
            "schema_name": record.schema_name,
            "status": record.status,
            "started_at": record.started_at.isoformat(),
            "finished_at": (
                record.completed_at.isoformat()
                if record.completed_at
                else None
            ),
            "error_message": record.errors[0] if record.errors else None,
        }
        for record in records
    ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_event(
    run_id: str,
    event_type: str,
    node_id: str | None,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Construct a normalised SSE event dict.

    Args:
        run_id: The run this event belongs to.
        event_type: Semantic event type string.
        node_id: Identifier of the node this event is about (or None).
        data: Arbitrary payload for this event.

    Returns:
        A dict with event_id, run_id, node_id, event_type, timestamp, data.
    """
    return {
        "event_id": str(uuid.uuid4()),
        "run_id": run_id,
        "node_id": node_id or "",
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


def _normalise_event(
    raw_event: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Translate a raw executor event into the canonical frontend shape.

    The executor emits events with a ``type`` key and various fields.  This
    function translates each known type into the ``{event_id, run_id,
    node_id, event_type, timestamp, data}`` shape the frontend expects.

    Args:
        raw_event: The raw event dict from WorkflowExecutor.
        run_id: The run identifier to embed.

    Returns:
        A normalised event dict ready for SSE serialisation.
    """
    raw_type = raw_event.get("type", "unknown")
    node_id = raw_event.get("node_id", "")
    data: dict[str, Any] = {
        key: value
        for key, value in raw_event.items()
        if key not in {"type", "node_id", "run_id"}
    }

    type_mapping: dict[str, str] = {
        "workflow_started": "run_start",
        "workflow_completed": "run_complete",
        "workflow_error": "run_error",
        "workflow_timeout": "run_error",
        "stage_started": "node_start",
        "stage_completed": "node_complete",
        "node_started": "node_start",
        "node_completed": "node_complete",
        "node_error": "node_error",
        "node_input": "node_input",
        "node_output": "node_output",
        "tool_call": "tool_call",
        "tool_result": "tool_result",
    }

    event_type = type_mapping.get(raw_type, raw_type)

    return _make_event(
        run_id=run_id,
        event_type=event_type,
        node_id=node_id,
        data=data,
    )
