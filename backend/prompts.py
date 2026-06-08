"""
Prompt loader for group planner templates.

Entities:
    - load_prompt_template: Load a prompt template JSON file with {variable} placeholders.
    - PROMPTS_DIR: Resolved path to the prompts/ directory.

How used by other modules:
    - backend.agent.group calls load_prompt_template("group_planner_structured.json")
      and load_prompt_template("group_planner_default.json"), then formats with
      runtime values ({structure}, {shared_state}, {min_agents}, {max_agents}).
"""

import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_template(filename: str) -> str:
    """Load a prompt template and return the system_prompt string.

    Args:
        filename: JSON filename relative to prompts/ (e.g. "group_planner_structured.json").

    Returns:
        The raw system_prompt string (contains {placeholders} for .format()).

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        KeyError: If the JSON does not contain a "system_prompt" key.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt template not found: {filepath}")
    data = json.loads(filepath.read_text(encoding="utf-8"))
    if "system_prompt" not in data:
        raise KeyError(
            f"Prompt file '{filename}' missing required 'system_prompt' key."
        )
    return data["system_prompt"]
