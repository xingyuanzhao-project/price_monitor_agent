"""
Artifact storage for large tool results.

When a tool returns data too large to inline in agent context (e.g. 10 MB of
trading history), the artifact store writes it to disk and returns a path.
The context harness then gives the agent a pointer instead of the raw data.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

ARTIFACT_SIZE_THRESHOLD = 50_000  # bytes — above this, data becomes an artifact


class ArtifactStore:
    """Writes large tool results to disk and returns file paths."""

    def __init__(self, run_id: str, base_dir: str = "data/artifacts") -> None:
        self._run_dir = Path(base_dir) / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def is_large(self, content: Any) -> bool:
        """Check whether content exceeds the inline threshold."""
        return self._estimate_size(content) > ARTIFACT_SIZE_THRESHOLD

    def write(self, data_type: str, content: Any) -> str:
        """Write content to an artifact file and return the path."""
        artifact_id = uuid.uuid4().hex[:12]
        filename = f"{data_type}_{artifact_id}.json"
        filepath = self._run_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, default=str)
        return str(filepath)

    def read(self, path: str) -> Any:
        """Read an artifact back from disk."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _estimate_size(content: Any) -> int:
        if isinstance(content, str):
            return len(content.encode("utf-8"))
        if isinstance(content, (bytes, bytearray)):
            return len(content)
        return len(json.dumps(content, default=str).encode("utf-8"))
