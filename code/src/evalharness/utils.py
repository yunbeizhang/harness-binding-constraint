"""Small utilities shared across EvalHarness modules."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON or TOML mapping file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    raw = file_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(raw)
    if suffix == ".toml":
        return tomllib.loads(raw)
    raise ValueError(f"Unsupported config format: {file_path}")


def deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Deep merge child into parent without mutating either input."""
    merged = dict(parent)
    for key, value in child.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def approx_token_count(text: str) -> int:
    """A cheap token estimator that is good enough for harness comparisons."""
    return max(1, len(text) // 4)


def truncate_text(text: str, limit: int) -> str:
    """Truncate long strings for prompt inclusion and logging."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def slugify(text: str) -> str:
    """Convert arbitrary labels into filesystem-safe path fragments."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "item"


def json_dumps_compact(payload: Any) -> str:
    """Stable compact JSON for prompts and signatures."""
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def safe_jsonable(obj: Any) -> Any:
    """Convert nested objects into JSON-safe data."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_jsonable(v) for v in obj]
    if hasattr(obj, "value"):
        return getattr(obj, "value")
    if hasattr(obj, "__dict__"):
        return {k: safe_jsonable(v) for k, v in vars(obj).items()}
    return obj
