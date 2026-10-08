#!/usr/bin/env python3
"""Collect video, condition, and description from a full-frame sweep.

Reads ``results.json`` written by ``scripts/run_fullframe_motion_multiroi.py``
and writes one text file. Models are section headers because several presets
share the same video and condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RESULTS = Path("outputs/fullframe_proposed_roi_5model/results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Sweep results.json (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output text file (default: descriptions.txt next to results.json)",
    )
    return parser.parse_args()


def render_descriptions(data: dict) -> str:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    runs = data.get("runs") or []
    if not isinstance(runs, list):
        raise ValueError("results JSON must contain a 'runs' list")

    lookup: dict[tuple[str, str, str], dict] = {}
    models: list[str] = [str(name) for name in (meta.get("models") or [])]
    videos: list[str] = [str(name) for name in (meta.get("videos") or [])]
    conditions: list[str] = [str(name) for name in (meta.get("conditions") or [])]
    for row in runs:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or "")
        video_id = str(row.get("video_id") or "")
        condition = str(row.get("condition") or "")
        if not model or not video_id or not condition:
            continue
        lookup[(model, video_id, condition)] = row
        if model not in models:
            models.append(model)
        if video_id not in videos:
            videos.append(video_id)
        if condition not in conditions:
            conditions.append(condition)

    blocks: list[str] = []
    for model in models:
        chunks: list[str] = []
        for video_id in videos:
            for condition in conditions:
                row = lookup.get((model, video_id, condition))
                if row is None:
                    continue
                description = str(row.get("generated_description") or "").strip()
                if not description:
                    error = str(row.get("error") or "").strip()
                    description = f"ERROR: {error}" if error else ""
                name = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
                chunks.append(
                    "\n".join(
                        [
                            f"video: {name}",
                            f"condition: {condition}",
                            f"description: {description}",
                        ]
                    )
                )
        if not chunks:
            continue
        blocks.append(f"# {model}\n\n" + "\n\n".join(chunks))
    return "\n\n".join(blocks).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if not args.results.is_file():
        raise SystemExit(f"results not found: {args.results}")
    data = json.loads(args.results.read_text(encoding="utf-8"))
    text = render_descriptions(data)
    dst = args.output or args.results.with_name("descriptions.txt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    n_videos = text.count("\nvideo: ") + (1 if text.startswith("video: ") else 0)
    print(f"wrote {dst} ({n_videos} rows)")


if __name__ == "__main__":
    main()
