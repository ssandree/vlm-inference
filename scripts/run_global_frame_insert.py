#!/usr/bin/env python3
"""Run the 3-way Global full-frame insertion experiment.

Conditions (ROI 1fps crops stay fixed; only Global timestamps change):
  video_start  one Full-frame at video start (nearest fps_1 frame to t=0)
  start_only   Full-frame at each ROI start
  start_end    Full-frame at each ROI start and end

Default is dry-run (input validation, no VLM). Pass --inference for captions.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_inference.config import load_config
from vlm_inference.global_insert import (
    GLOBAL_INSERT_START_END,
    GLOBAL_INSERT_START_ONLY,
    GLOBAL_INSERT_VIDEO_START,
    ROI_GLOBAL_INSERT_METHODS,
)

MODES = (
    GLOBAL_INSERT_VIDEO_START,
    GLOBAL_INSERT_START_ONLY,
    GLOBAL_INSERT_START_END,
)
METHOD_BY_MODE = {
    GLOBAL_INSERT_VIDEO_START: "global_roi_video_start_proposed_multi_roi",
    GLOBAL_INSERT_START_ONLY: "global_roi_start_only_proposed_multi_roi",
    GLOBAL_INSERT_START_END: "global_roi_start_end_proposed_multi_roi",
}
MODE_LABEL = {
    GLOBAL_INSERT_VIDEO_START: "video start (1 full-frame at t=0)",
    GLOBAL_INSERT_START_ONLY: "ROI start only",
    GLOBAL_INSERT_START_END: "ROI start and end",
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/feasibility_global_frame.yaml")
    p.add_argument("--output-dir", default="outputs/global_frame_insert")
    p.add_argument(
        "--modes",
        nargs="*",
        default=None,
        choices=list(MODES),
        help="Subset of conditions (default: all three).",
    )
    p.add_argument(
        "--inference",
        action="store_true",
        help="Run caption inference after dry-run validation.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing inference cache.",
    )
    return p.parse_args()


def selected_methods(modes: list[str]) -> list[str]:
    return [METHOD_BY_MODE[mode] for mode in modes]


def write_run_config(
    *,
    base: dict,
    output_dir: Path,
    methods: list[str],
) -> Path:
    raw = json.loads(json.dumps(base))
    raw["experiments"] = methods
    raw["output"] = {
        "dir": str(output_dir),
        "file": "feasibility_results_global_frame.json",
    }
    raw.setdefault("feasibility", {})
    raw["feasibility"]["cache_file"] = str(output_dir / "feasibility_cache.json")
    prompts = raw["feasibility"].setdefault("method_prompts", {})
    for method in methods:
        prompts.setdefault(method, raw.get("prompt_file") or "prompts/frames_roi.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def run_inference(config: Path, *, dry_run: bool, no_resume: bool) -> Path:
    cfg = load_config(config)
    log_name = "dry_run.log" if dry_run else "inference.log"
    log = Path(cfg.output.dir) / log_name
    cmd = [sys.executable, str(ROOT / "run_inference.py"), "--config", str(config)]
    if dry_run:
        cmd.append("--dry-run")
    if no_resume:
        cmd.append("--no-resume")
    with log.open("w") as fh:
        subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=True)
    stem = Path(cfg.output.path).stem
    candidates = sorted(
        Path(cfg.output.dir).glob(f"{stem}_*.json"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    if not candidates:
        raise FileNotFoundError(f"no results JSON under {cfg.output.dir}")
    payload = json.loads(candidates[-1].read_text())
    if payload.get("dry_run") != dry_run:
        raise RuntimeError(
            f"{candidates[-1]} dry_run={payload.get('dry_run')} expected {dry_run}"
        )
    return candidates[-1]


def mode_for_method(method: str) -> str:
    mode = ROI_GLOBAL_INSERT_METHODS.get(method)
    if mode not in MODES:
        raise ValueError(f"unexpected method {method}")
    return mode


def validate_rows(rows: list[dict], *, modes: list[str], videos: list[str]) -> None:
    expected_methods = set(selected_methods(modes))
    by_video: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("error"):
            raise RuntimeError(f"{row['job']}: {row['error']}")
        method = row["method"]
        if method not in expected_methods:
            continue
        by_video[row["video_id"]][method] = row

    missing_videos = [v for v in videos if v not in by_video]
    if missing_videos:
        raise RuntimeError(f"missing videos: {missing_videos}")

    for video_id in videos:
        per_method = by_video[video_id]
        missing = [m for m in selected_methods(modes) if m not in per_method]
        if missing:
            raise RuntimeError(f"{video_id}: missing methods {missing}")

        video_start = per_method.get(METHOD_BY_MODE[GLOBAL_INSERT_VIDEO_START])
        if video_start is not None:
            n_global = int(video_start["num_global_images"])
            if n_global != 1:
                raise RuntimeError(
                    f"{video_id} video_start: expected 1 Global frame, got {n_global}"
                )

        roi_counts = {
            method: int(row["num_roi_images"]) for method, row in per_method.items()
        }
        if len(set(roi_counts.values())) != 1:
            raise RuntimeError(f"{video_id}: ROI counts differ across conditions: {roi_counts}")
        roi_ts = {
            method: row.get("roi_timestamps") for method, row in per_method.items()
        }
        first_ts = next(iter(roi_ts.values()))
        if any(ts != first_ts for ts in roi_ts.values()):
            raise RuntimeError(f"{video_id}: ROI timestamps differ across conditions")


def frame_count_rows(rows: list[dict], videos: list[str], modes: list[str]) -> list[dict]:
    by_video: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        method = row.get("method")
        if method in METHOD_BY_MODE.values():
            by_video[row["video_id"]][mode_for_method(method)] = row

    out = []
    for video_id in videos:
        rec = {"video_id": video_id}
        for mode in modes:
            row = by_video[video_id][mode]
            rec[f"{mode}_n_global"] = int(row["num_global_images"])
            rec[f"{mode}_n_roi"] = int(row["num_roi_images"])
            rec[f"{mode}_global_t"] = row.get("global_timestamps")
        rec["n_roi_tubes"] = int(next(iter(by_video[video_id].values()))["num_roi_tubes"])
        out.append(rec)
    return out


def write_captions_md(path: Path, rows: list[dict], videos: list[str], modes: list[str]) -> None:
    lines = ["# Global frame insertion captions", ""]
    for video_id in videos:
        lines += [f"## {video_id}", ""]
        for mode in modes:
            method = METHOD_BY_MODE[mode]
            match = next(
                r for r in rows if r["video_id"] == video_id and r["method"] == method
            )
            caption = (match.get("caption") or "").strip()
            n_global = match["num_global_images"]
            ts = match.get("global_timestamps") or []
            lines += [
                f"### {mode} — {MODE_LABEL[mode]}",
                "",
                f"Global frames: {n_global} at t={ts}",
                "",
                caption,
                "",
            ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    modes = list(args.modes) if args.modes else list(MODES)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(args.config).read_text())
    methods = selected_methods(modes)
    config = write_run_config(base=base, output_dir=out, methods=methods)
    cfg = load_config(config)
    videos = list(cfg.feasibility.videos)

    print(f"Dry-run modes={modes}", flush=True)
    dry_path = run_inference(config, dry_run=True, no_resume=args.no_resume)
    dry_payload = json.loads(dry_path.read_text())
    dry_rows = dry_payload["results"]
    validate_rows(dry_rows, modes=modes, videos=videos)
    write_json(out / "dry_run.json", dry_rows)
    counts = frame_count_rows(dry_rows, videos, modes)
    write_json(out / "frame_counts.json", counts)
    fieldnames = list(counts[0].keys()) if counts else []
    with (out / "frame_counts.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(counts)
    print(json.dumps(counts, indent=2, ensure_ascii=False), flush=True)
    print(f"Dry-run OK: {len(videos)} videos × {len(modes)} conditions", flush=True)

    if not args.inference:
        print("Dry-run only. Re-run with --inference to generate captions.", flush=True)
        return

    print(f"Inference modes={modes}", flush=True)
    inf_path = run_inference(config, dry_run=False, no_resume=args.no_resume)
    inf_payload = json.loads(inf_path.read_text())
    inf_rows = inf_payload["results"]
    validate_rows(inf_rows, modes=modes, videos=videos)
    planned = {
        (r["video_id"], r["method"]): r
        for r in dry_rows
        if r["method"] in METHOD_BY_MODE.values()
    }
    for row in inf_rows:
        if row["method"] not in METHOD_BY_MODE.values():
            continue
        key = (row["video_id"], row["method"])
        for field in ("num_roi_images", "num_global_images", "roi_timestamps", "global_timestamps"):
            if row.get(field) != planned[key].get(field):
                raise RuntimeError(f"{row['job']}: {field} differs from dry-run")
        if not row.get("success") or not (row.get("caption") or "").strip():
            raise RuntimeError(f"{row['job']}: missing caption")
    write_json(out / "captions.json", inf_rows)
    write_captions_md(out / "captions.md", inf_rows, videos, modes)
    expected = len(videos) * len(modes)
    print(
        f"Completed: {expected}/{expected} captions; "
        f"dry-run/inference Global/ROI counts match. Results: {inf_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
