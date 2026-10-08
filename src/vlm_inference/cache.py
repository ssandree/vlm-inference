"""Persistent inference cache for feasibility dedup / resume."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_cache_path(path: str | Path, *, cwd: Path | None = None) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (cwd or Path.cwd()) / p
    return p.resolve()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring corrupt cache %s: %s", path, exc)
        return {}
    if isinstance(data, dict) and isinstance(data.get("entries"), dict):
        return {str(k): v for k, v in data["entries"].items() if isinstance(v, dict)}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def ingest_results_json(path: Path, cache: dict[str, dict[str, Any]]) -> int:
    """Merge completed records from a previous results JSON into the cache."""
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    records: list[dict[str, Any]]
    if isinstance(data, list):
        records = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        records = [r for r in data["results"] if isinstance(r, dict)]
    else:
        return 0
    added = 0
    for rec in records:
        key = rec.get("cache_key")
        if not key or rec.get("error"):
            continue
        if rec.get("caption") in (None, ""):
            continue
        if str(key) in cache:
            continue
        cache[str(key)] = rec
        added += 1
    return added


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "entries": cache}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cache_reusable(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    if entry.get("error"):
        return False
    caption = entry.get("caption") or entry.get("response")
    return bool(caption)
