"""
Per-run log handler that captures Python logging output for live streaming.

What it does:
    Provides a logging.Handler subclass that routes log records into an
    asyncio.Queue keyed by run_id. The GUI's log stream SSE endpoint
    consumes from this queue. A disk-backed log file is also written so
    completed runs can replay their logs after the queue is drained.

Entities in it:
    - RunLogHandler: logging.Handler that publishes formatted lines to the
      active run's queue.
    - activate_run: Start capturing logs for a given run_id.
    - deactivate_run: Stop capturing and signal stream termination.
    - get_log_queue: Retrieve the queue for a run (used by SSE endpoint).
    - get_log_file_path: Retrieve the on-disk log path for replay.

How used by other modules:
    - backend.api.runs calls activate_run/deactivate_run around execution.
    - backend.api.runs exposes an SSE endpoint that reads from the queue.
    - All backend modules that call logging.info/warning/error during a run
      automatically feed the terminal.
"""

import asyncio
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

CURRENT_RUN_ID: ContextVar[Optional[str]] = ContextVar(
    "price_monitor_current_run_id", default=None
)

_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "run_logs"

_run_queues: dict[str, asyncio.Queue[str | None]] = {}
_run_log_files: dict[str, Path] = {}


class RunLogHandler(logging.Handler):
    """Routes log records to the active run's queue and log file."""

    def __init__(self, level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self.setFormatter(logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
        ))

    def emit(self, record: logging.LogRecord) -> None:
        run_id = CURRENT_RUN_ID.get()
        if run_id is None:
            return
        queue = _run_queues.get(run_id)
        if queue is None:
            return
        try:
            formatted = self.format(record)
            queue.put_nowait(formatted)
            log_path = _run_log_files.get(run_id)
            if log_path:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(formatted + "\n")
        except Exception:
            self.handleError(record)


_handler_instance: RunLogHandler | None = None


def install_handler() -> None:
    """Attach the RunLogHandler to the root logger. Call once at startup."""
    global _handler_instance
    if _handler_instance is not None:
        return
    _handler_instance = RunLogHandler()
    root = logging.getLogger()
    root.addHandler(_handler_instance)
    if root.level == logging.NOTSET or root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)


def activate_run(run_id: str) -> None:
    """Start capturing logs for run_id. Creates queue and log file."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / f"{run_id}.log"
    _run_log_files[run_id] = log_path
    _run_queues[run_id] = asyncio.Queue()
    CURRENT_RUN_ID.set(run_id)


def deactivate_run(run_id: str) -> None:
    """Signal end-of-stream for run_id and remove the queue.

    After this call, subsequent SSE connections will fall through to the
    on-disk log file for replay instead of blocking on an empty queue.
    """
    queue = _run_queues.pop(run_id, None)
    if queue:
        queue.put_nowait(None)
    CURRENT_RUN_ID.set(None)


def get_log_queue(run_id: str) -> Optional[asyncio.Queue[str | None]]:
    """Return the log queue for run_id, or None if not active."""
    return _run_queues.get(run_id)


def get_log_file_path(run_id: str) -> Optional[Path]:
    """Return the on-disk log file for run_id (for replay of completed runs)."""
    path = _run_log_files.get(run_id)
    if path and path.exists():
        return path
    candidate = _LOG_DIR / f"{run_id}.log"
    if candidate.exists():
        return candidate
    return None
