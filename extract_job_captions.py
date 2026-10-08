#!/usr/bin/env python3
"""Extract job and caption fields from a feasibility results JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Output JSON (default: <results_stem>_job_captions.json next to the input)",
    )
    return parser.parse_args()


def extract_job_captions(data: dict) -> list[dict[str, str | None]]:
    results = data.get("results") or []
    if not isinstance(results, list):
        raise ValueError("results JSON must contain a 'results' list")
    rows: list[dict[str, str | None]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        job = row.get("job")
        if not job:
            continue
        caption = row.get("caption")
        if isinstance(caption, str):
            caption = caption.strip() or None
        else:
            caption = None
        rows.append({"job": str(job), "caption": caption})
    return rows


def main() -> None:
    args = parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = extract_job_captions(data)
    dst = args.output or args.results.with_name(args.results.stem + "_job_captions.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dst} ({len(rows)} jobs)")


if __name__ == "__main__":
    main()
