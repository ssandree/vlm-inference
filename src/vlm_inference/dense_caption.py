"""Minimal, lossless dense-caption candidate validation (no action taxonomy)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_TIMESTAMP = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")


def parse_dense_captions(raw: str, duration_sec: float) -> dict:
    """Keep valid two-field events; preserve invalid rows without clamping/repair.

    Sorting is temporal only: overlapping captions and similar descriptions of
    different actors are never deduplicated or merged by heuristics.
    """
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("duration_sec must be positive and finite")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        document = json.loads(text)
    except (ValueError, TypeError) as exc:
        return {"status": "invalid_json", "events": [], "rejected_events": [], "errors": [str(exc)]}
    if not isinstance(document, list):
        return {"status": "invalid_schema", "events": [], "rejected_events": [], "errors": ["Expected a JSON array"]}
    accepted, rejected = [], []
    for index, row in enumerate(document):
        errors = []
        if not isinstance(row, dict):
            rejected.append({"index": index, "original": row, "errors": ["Expected an object"]})
            continue
        if set(row) != {"timestamp", "sentence"}:
            errors.append("Each event must have exactly timestamp and sentence")
        match = _TIMESTAMP.fullmatch(row.get("timestamp", "")) if isinstance(row.get("timestamp"), str) else None
        start, end = None, None
        if match is None:
            errors.append('timestamp must be "start_sec - end_sec"')
        else:
            start, end = map(float, match.groups())
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration_sec):
                errors.append("Timestamp outside video or reversed/zero-length; not clamped")
        if not isinstance(row.get("sentence"), str) or not row["sentence"].strip():
            errors.append("sentence must be nonempty text")
        # A mechanical punctuation count cannot establish a factual single sentence.
        # Leave prose and semantic verification to the human reviewer.
        if errors:
            rejected.append({"index": index, "original": row, "errors": errors})
        else:
            accepted.append((start, end, index, dict(row)))
    accepted.sort(key=lambda item: item[:3])
    return {"status": "partial" if rejected else "ok", "events": [x[3] for x in accepted],
            "rejected_events": rejected, "errors": []}


def write_new_json(path: Path, value) -> None:
    """Exclusive create protects edited candidates and prior run artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
