#!/usr/bin/env python3
"""Export stable comparison artifacts from a global_roi_cross feasibility run."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "global_roi_cross"
CONFIG = ROOT / "configs" / "feasibility_global_roi_cross.yaml"
GT = ROOT / "configs" / "video9_gt_event.txt"
METHODS = (
    "global_roi_video_start_roi_fixed_1fps",
    "global_roi_video_start_roi_fixed_2fps",
    "global_roi_video_start_roi_accumulated_motion",
    "global_roi_video_start_roi_centroid_2fps",
    "global_roi_start_only_roi_fixed_1fps",
    "global_roi_start_only_roi_fixed_2fps",
    "global_roi_start_only_roi_accumulated_motion",
    "global_roi_start_only_roi_centroid_2fps",
)


def load_rows() -> tuple[Path, list[dict]]:
    paths = sorted(OUT.glob("feasibility_results_global_roi_cross_*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError("no global_roi_cross feasibility result found")
    path = paths[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("results", payload) if not row.get("alias_of")]
    return path, rows


def parse_gt() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for block in re.split(r"(?=^VIRAT_)", GT.read_text(encoding="utf-8"), flags=re.M):
        match = re.match(r"(VIRAT_\S+) Event", block)
        if match:
            result[match.group(1)] = re.findall(r"^\d+\.\s*(.+)", block, flags=re.M)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path, rows = load_rows()
    if len(rows) != 72:
        raise ValueError(f"expected 72 raw rows, found {len(rows)}")
    methods = {row["method"] for row in rows}
    if methods != set(METHODS):
        raise ValueError(f"unexpected methods: {sorted(methods)}")

    shutil.copyfile(CONFIG, OUT / "config.yaml")
    dry_runs = sorted(OUT.glob("feasibility_results_global_roi_cross_*.json"), key=lambda p: p.stat().st_mtime)
    dry_run = next((p for p in dry_runs if p != raw_path), None)
    if dry_run:
        shutil.copyfile(dry_run, OUT / "dry_run.json")

    frame_rows = []
    for row in rows:
        frame_rows.append({
            "video_id": row["video_id"],
            "method": row["method"],
            "global_insert": row.get("global_insert"),
            "roi_sampling": row.get("roi_sampling"),
            "num_global_frames": row.get("num_global_images", 0),
            "num_roi_frames": row.get("num_roi_images", 0),
            "total_frames": row.get("num_total_images", 0),
            "roi_timestamps": json.dumps(row.get("roi_timestamps", {}), ensure_ascii=False),
            "global_timestamps": json.dumps(row.get("global_timestamps", []), ensure_ascii=False),
            "success": row.get("success", False),
            "error": row.get("error"),
        })
    with (OUT / "frame_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    (OUT / "frame_counts.json").write_text(json.dumps(frame_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    captions = {video: {} for video in sorted({row["video_id"] for row in rows})}
    jobs = {}
    summary = []
    for row in rows:
        caption = row.get("caption")
        captions[row["video_id"]][row["method"]] = caption
        jobs[row["job"]] = caption
        token_stats = "actual_processor" if row.get("num_visual_tokens") is not None else "estimated_from_input"
        summary.append({
            "video_id": row["video_id"],
            "method": row["method"],
            "job": row["job"],
            "global_insert": row.get("global_insert"),
            "roi_sampling": row.get("roi_sampling"),
            "roi_timestamps": row.get("roi_timestamps", {}),
            "global_timestamps": row.get("global_timestamps", []),
            "num_global_frames": row.get("num_global_images"),
            "num_roi_frames": row.get("num_roi_images"),
            "total_frames": row.get("num_total_images"),
            "estimated_visual_tokens": row.get("num_visual_tokens"),
            "visual_token_source": token_stats,
            "gt_event_recall": None,
            "hallucinated_event_count": None,
            "caption": caption,
            "success": row.get("success", False),
            "error": row.get("error"),
        })
    (OUT / "captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "job_captions.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = []
    for video, method_caps in captions.items():
        lines.extend([f"## {video}", ""])
        for method in METHODS:
            lines.extend([f"### {method}", method_caps.get(method) or "", ""])
    (OUT / "captions.md").write_text("\n".join(lines), encoding="utf-8")

    gt = parse_gt()
    eval_rows = []
    for row in summary:
        events = gt[row["video_id"]]
        eval_rows.append({
            "video_id": row["video_id"],
            "method": row["method"],
            "caption": row["caption"],
            "gt_events": [{"event_id": i, "event": event, "captured": None, "rationale": "Manual semantic adjudication required; no automatic metric substituted."} for i, event in enumerate(events, 1)],
            "gt_event_recall": None,
            "hallucinated_event_count": None,
            "hallucination_notes": "Manual semantic adjudication required.",
        })
    (OUT / "gt_eval.json").write_text(json.dumps({"rubric": "Binary semantic core-event matching; no partial credit.", "n_gt_events": sum(map(len, gt.values())), "rows": eval_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"exported {len(summary)} rows from {raw_path}")


if __name__ == "__main__":
    main()