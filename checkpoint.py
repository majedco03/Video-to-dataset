"""Checkpoint helpers for resumable pipeline runs.

A checkpoint file is written into the output folder after each step completes.
On the next run the runner reads it and skips already-finished steps.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set

CHECKPOINT_FILENAME = ".pipeline_checkpoint.json"


def checkpoint_path(output_root: str) -> str:
    return os.path.join(output_root, CHECKPOINT_FILENAME)


def load_checkpoint(output_root: str) -> Dict[str, Any]:
    path = checkpoint_path(output_root)
    if not os.path.isfile(path):
        return {"completed_steps": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"completed_steps": []}
        return data
    except (OSError, json.JSONDecodeError):
        return {"completed_steps": []}


def completed_step_names(output_root: str) -> Set[str]:
    data = load_checkpoint(output_root)
    names = data.get("completed_steps", [])
    return set(names) if isinstance(names, list) else set()


def mark_step_complete(output_root: str, step_name: str) -> None:
    data = load_checkpoint(output_root)
    completed: List[str] = data.get("completed_steps", [])
    if not isinstance(completed, list):
        completed = []
    if step_name not in completed:
        completed.append(step_name)
    data["completed_steps"] = completed
    os.makedirs(output_root, exist_ok=True)
    path = checkpoint_path(output_root)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clear_checkpoint(output_root: str) -> None:
    path = checkpoint_path(output_root)
    if os.path.isfile(path):
        os.remove(path)
