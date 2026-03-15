"""Local storage helpers for saved pipeline profiles."""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
from typing import Any, Dict

PROFILE_STORAGE_DIR = os.path.join(os.path.expanduser("~"), ".video_to_dataset")
PROFILE_STORAGE_FILE = os.path.join(PROFILE_STORAGE_DIR, "pipeline_profiles.json")
PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")


def validate_profile_name(name: str) -> bool:
    """Keep saved profile names readable and filesystem-friendly."""

    return bool(PROFILE_NAME_PATTERN.fullmatch(name.strip()))


def _default_store() -> Dict[str, Any]:
    return {"version": 1, "profiles": {}}


def load_profile_store() -> Dict[str, Any]:
    """Read the local profile store, returning an empty one when needed."""

    if not os.path.isfile(PROFILE_STORAGE_FILE):
        return _default_store()

    try:
        with open(PROFILE_STORAGE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return _default_store()

    if not isinstance(data, dict):
        return _default_store()

    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        data["profiles"] = {}
    return data


def save_profile_store(data: Dict[str, Any]) -> None:
    """Write the profile store to the user's device."""

    os.makedirs(PROFILE_STORAGE_DIR, exist_ok=True)
    with open(PROFILE_STORAGE_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def list_pipeline_profiles() -> Dict[str, Dict[str, Any]]:
    """Return all locally saved pipeline profiles."""

    store = load_profile_store()
    profiles = store.get("profiles", {})
    return profiles if isinstance(profiles, dict) else {}


def load_pipeline_profile(name: str) -> Dict[str, Any] | None:
    """Load one saved profile by name."""

    profiles = list_pipeline_profiles()
    entry = profiles.get(name)
    if not isinstance(entry, dict):
        return None
    options = entry.get("options")
    return options if isinstance(options, dict) else None


def save_pipeline_profile(name: str, options: Dict[str, Any]) -> None:
    """Save or update one named profile."""

    store = load_profile_store()
    profiles = store.setdefault("profiles", {})
    profiles[name] = {
        "name": name,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "options": options,
    }
    save_profile_store(store)
