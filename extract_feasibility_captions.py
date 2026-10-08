#!/usr/bin/env python3
"""Extract per-video captions from a feasibility results JSON."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

# Removed from the active experiment set; never emit even if present in old results.
EXCLUDED_METHODS = frozenset({"full_2fps"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a feasibility_captions_*.json dump from results JSON."
    )
    parser.add_argument(
        "results",
        type=Path,
        help="feasibility_results_*.json path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output captions JSON (default: outputs/feasibility_captions_<stamp>.json)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method order to include (default: methods list in results, minus excluded)",
    )
    parser.add_argument(
        "--include-null",
        action="store_true",
        help="Keep methods with missing captions as null (default: drop all-null methods)",
    )
    return parser.parse_args()


def _default_output(results: Path) -> Path:
    name = results.name
    if name.startswith("feasibility_results_") and name.endswith(".json"):
        stamp = name[len("feasibility_results_") : -len(".json")]
        return results.parent / f"feasibility_captions_{stamp}.json"
    return results.with_name(results.stem + "_captions.json")


def extract_captions(
    data: dict,
    *,
    methods: list[str] | None,
    include_null: bool,
) -> dict:
    results = data.get("results") or []
    if not isinstance(results, list):
        raise ValueError("results JSON must contain a 'results' list")

    declared = [str(m) for m in (data.get("methods") or [])]
    method_order = list(methods) if methods else declared
    method_order = [m for m in method_order if m not in EXCLUDED_METHODS]

    # Preserve video order from results.
    video_order: list[str] = []
    by_video: dict[str, dict[str, str | None]] = OrderedDict()
    for row in results:
        if not isinstance(row, dict):
            continue
        video_id = str(row.get("video_id") or "")
        method = str(row.get("method") or "")
        if not video_id or not method or method in EXCLUDED_METHODS:
            continue
        if video_id not in by_video:
            by_video[video_id] = {}
            video_order.append(video_id)
        caption = row.get("caption")
        if isinstance(caption, str) and caption.strip():
            by_video[video_id][method] = caption.strip()
        elif method not in by_video[video_id]:
            by_video[video_id][method] = None

        if method not in method_order:
            method_order.append(method)

    if not include_null:
        keep: list[str] = []
        for method in method_order:
            if any(by_video[v].get(method) for v in video_order):
                keep.append(method)
        method_order = keep

    videos = []
    for video_id in video_order:
        captions = OrderedDict(
            (method, by_video[video_id].get(method)) for method in method_order
        )
        videos.append({"video_id": video_id, "captions": captions})

    return {
        "source": Path(data.get("output_file") or "").name
        or None,
        "methods": method_order,
        "videos": videos,
    }


def main() -> None:
    args = parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    # Prefer the input filename as source label.
    out = extract_captions(
        data, methods=args.methods, include_null=args.include_null
    )
    out["source"] = args.results.name

    dst = args.output or _default_output(args.results)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dst} ({len(out['videos'])} videos, methods={out['methods']})")


if __name__ == "__main__":
    main()
