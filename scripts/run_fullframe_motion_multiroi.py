#!/usr/bin/env python3
"""Full-frame vs Motion Multi-ROI feasibility inference.

Dry-run checks source videos and saved Stage3 ROI tubes, then runs Qwen3-VL
on every available model size. One failed run is recorded and the sweep
continues. Results are rewritten after every run.

Example:
  python scripts/run_fullframe_motion_multiroi.py --dry-run
  python scripts/run_fullframe_motion_multiroi.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_inference.config import apply_model_override  # noqa: E402
from vlm_inference.fullframe_motion_multiroi import (  # noqa: E402
    ALL_CONDITIONS,
    CONDITION_ANCHOR,
    CONDITION_FULL,
    CONDITION_FULL_2FPS,
    CONDITION_MOTION,
    CONDITION_MOTION_ROI,
    CONDITION_PEAK,
    CONDITIONS,
    DEFAULT_TRACKS_ROOT,
    DEFAULT_VIDEO_ROOTS,
    DEFAULT_VIDEOS,
    discover_model_sizes,
    discover_videos,
    dry_run_payload,
    empty_run,
    format_dry_run,
    format_exception,
    format_peak_anchor_audit,
    format_start_anchor_audit,
    generation_payload,
    is_success,
    MOTION_SCORE_SOURCE,
    load_prompt_text,
    load_saved_runs,
    load_sweep_request,
    materialize_video,
    peak_anchor_audit,
    run_key,
    save_run_file,
    start_anchor_audit,
    write_outputs,
)

logger = logging.getLogger("fullframe_motion_multiroi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fullframe_motion_multiroi.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--full-prompt", default=None, help="Prompt file for the full-frame condition")
    parser.add_argument("--motion-prompt", default=None, help="Prompt file for the Motion Multi-ROI condition")
    parser.add_argument("--tracks-root", default=DEFAULT_TRACKS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of size presets, e.g. 8b 30b")
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Video ids. Short VIRAT stems are prefixed with VIRAT_S_. Default is the original 8-video set.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Subset of full_frame, full_frame_2fps, motion_roi, motion_multi_roi, motion_multi_roi_start_anchor, motion_multi_roi_peak_anchor",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def _condition_parts(prepared, condition: str):
    if condition == CONDITION_FULL:
        return prepared.full_sample, prepared.plan.full_frame_image_count(), prepared.full_pixels
    if condition == CONDITION_FULL_2FPS:
        return (
            prepared.full_2fps_sample,
            prepared.plan.full_2fps_frame_image_count(),
            prepared.full_2fps_pixels,
        )
    if condition == CONDITION_MOTION:
        return (
            prepared.motion_sample,
            prepared.plan.motion_frame_image_count(),
            prepared.motion_pixels,
        )
    if condition == CONDITION_MOTION_ROI:
        return (
            prepared.motion_roi_sample,
            prepared.motion_roi_frame_image_count,
            prepared.motion_roi_pixels,
        )
    if condition == CONDITION_ANCHOR:
        return prepared.anchor_sample, prepared.anchor_frame_image_count, prepared.anchor_pixels
    if condition == CONDITION_PEAK:
        return prepared.peak_sample, prepared.peak_frame_image_count, prepared.peak_pixels
    raise ValueError(condition)


def _ordered(runs: list[dict], videos: list[str], models: list[str], conditions: list[str]) -> list[dict]:
    lookup = {run_key(row): row for row in runs}
    ordered = []
    for model in models:
        for video_id in videos:
            for condition in conditions:
                row = lookup.get((model, video_id, condition))
                if row is not None:
                    ordered.append(row)
    return ordered


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    request = load_sweep_request(args.config)
    cfg = request.cfg
    output_dir = Path(args.output_dir or cfg.output.dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    cfg.output.dir = str(output_dir)
    cfg.config_dir = Path(args.config).resolve().parent

    preset_names = list(args.models) if args.models else request.models
    models = discover_model_sizes(presets=preset_names)
    if preset_names:
        by_name = {item["name"]: item for item in models}
        missing = [name for name in preset_names if name not in by_name]
        if missing:
            logger.error("Unavailable model preset(s): %s", ", ".join(missing))
            return 1
        models = [by_name[name] for name in preset_names]
    plans = discover_videos(
        _video_ids(args.videos if args.videos else request.videos),
        video_roots=DEFAULT_VIDEO_ROOTS,
        tracks_root=args.tracks_root,
    )
    conditions = list(args.conditions or request.conditions or CONDITIONS)
    unknown = [name for name in conditions if name not in ALL_CONDITIONS]
    if unknown:
        logger.error("Unknown condition(s): %s", ", ".join(unknown))
        return 1
    payload = dry_run_payload(plans, models)
    payload["conditions"] = conditions
    prompt_paths = {
        condition: _prompt_path(condition, args, request.prompts, cfg.prompt_file)
        for condition in conditions
    }
    payload["prompts"] = prompt_paths
    anchor_audit = None
    if CONDITION_ANCHOR in conditions:
        anchor_audit = start_anchor_audit(plans, video=cfg.video, image=cfg.image)
        payload["start_anchor"] = anchor_audit
    peak_audit = None
    if CONDITION_PEAK in conditions:
        peak_audit = peak_anchor_audit(plans, video=cfg.video, image=cfg.image)
        payload["peak_anchor"] = peak_audit
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_existing_dry_run = (
        set(conditions) == {CONDITION_PEAK} and (output_dir / "dry_run.json").is_file()
    )
    dry_path = output_dir / ("peak_anchor_dry_run.json" if keep_existing_dry_run else "dry_run.json")
    dry_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(format_dry_run(payload))
    if anchor_audit is not None:
        print()
        print(format_start_anchor_audit(anchor_audit))
    if peak_audit is not None:
        print()
        print(format_peak_anchor_audit(peak_audit))
    print(f"\nDry-run written to {dry_path}")
    if args.dry_run:
        videos_ok = payload["videos_ok"] == payload["videos_total"]
        anchor_ok = anchor_audit is None or bool(anchor_audit["ok"])
        peak_ok = peak_audit is None or bool(peak_audit["ok"])
        return 0 if videos_ok and models and anchor_ok and peak_ok else 1
    if not models:
        logger.error("No Qwen3-VL checkpoints are available")
        return 1
    if payload["videos_ok"] != payload["videos_total"]:
        logger.error("Dry-run found missing inputs; inference was not started")
        return 1

    logger.info("Materializing 1 fps frames under %s", output_dir / "inputs")
    prepared = [
        materialize_video(plan, output_dir / "inputs", video=cfg.video)
        for plan in plans
    ]
    for item in prepared:
        if item.errors:
            logger.error("%s materialize: %s", item.plan.video_id, "; ".join(item.errors))

    missing_prompts = [condition for condition, path in prompt_paths.items() if not path]
    if missing_prompts:
        logger.error("Missing prompt file for: %s", ", ".join(missing_prompts))
        return 1
    prompts = {
        condition: load_prompt_text(path, base=ROOT)
        for condition, path in prompt_paths.items()
    }
    video_ids = [plan.video_id for plan in plans]
    model_names = [item["name"] for item in models]
    meta = {
        "prompt_file": cfg.prompt_file,
        "prompts": prompt_paths,
        "prompt": prompts,
        "sampling_fps": 1.0,
        "generation": generation_payload(cfg.generation),
        "video": {
            "fps": cfg.video.fps,
            "min_pixels": cfg.video.min_pixels,
            "max_pixels": cfg.video.max_pixels,
            "total_pixels": cfg.video.total_pixels,
        },
        "image": {
            "min_pixels": cfg.image.min_pixels,
            "max_pixels": cfg.image.max_pixels,
        },
        "models": model_names,
        "videos": video_ids,
        "tracks_root": str(args.tracks_root),
        "conditions": list(conditions),
    }
    runs = load_saved_runs(output_dir)

    def flush() -> None:
        report_meta = _report_meta(meta, runs, output_dir)
        report_models = list(report_meta["models"])
        report_conditions = list(report_meta["conditions"])
        write_outputs(
            output_dir,
            runs=_ordered(runs, video_ids, report_models, report_conditions),
            meta=report_meta,
        )

    from vlm_inference.model import Qwen3VLCaptioner

    for spec in models:
        apply_model_override(cfg, spec["name"])
        logger.info("Loading model %s from %s", spec["name"], cfg.model.path)
        try:
            captioner = Qwen3VLCaptioner(cfg)
        except Exception as exc:  # noqa: BLE001
            message = format_exception(exc)
            logger.error("Model load failed for %s: %s", spec["name"], exc)
            for item in prepared:
                for condition in conditions:
                    count, pixels = _frame_stats(item, condition)
                    failed = empty_run(
                        video_id=item.plan.video_id,
                        model=spec["name"],
                        condition=condition,
                        frame_image_count=count,
                        total_pixels=pixels,
                        error=message,
                    )
                    failed["model_path"] = cfg.model.path
                    failed["dtype"] = cfg.model.dtype
                    failed["load_in_8bit"] = bool(cfg.model.load_in_8bit)
                    _store(runs, failed, output_dir)
            flush()
            continue

        for item in prepared:
            for condition in conditions:
                key = (spec["name"], item.plan.video_id, condition)
                existing = next((row for row in runs if run_key(row) == key), None)
                if existing is not None and is_success(existing):
                    logger.info("Skip completed %s %s %s", *key)
                    continue
                sample, frame_count, pixels = _condition_parts(item, condition)
                record = empty_run(
                    video_id=item.plan.video_id,
                    model=spec["name"],
                    condition=condition,
                    frame_image_count=frame_count,
                    total_pixels=pixels or None,
                )
                record["model_path"] = cfg.model.path
                record["dtype"] = cfg.model.dtype
                record["load_in_8bit"] = bool(cfg.model.load_in_8bit)
                record["prompt_file"] = prompt_paths[condition]
                if condition == CONDITION_PEAK:
                    record["peak_anchors"] = item.peak_anchors
                    record["motion_score_source"] = MOTION_SCORE_SOURCE
                peak_failed = condition == CONDITION_PEAK and bool(item.peak_errors)
                full_2fps_failed = condition == CONDITION_FULL_2FPS and bool(item.full_2fps_errors)
                if item.errors or peak_failed or full_2fps_failed or sample is None:
                    record["error"] = (
                        "; ".join(
                            [
                                *item.errors,
                                *item.peak_errors,
                                *(item.full_2fps_errors if full_2fps_failed else []),
                            ]
                        )
                        or "input was not prepared"
                    )
                    _store(runs, record, output_dir)
                    flush()
                    continue
                logger.info("Inference %s %s %s (%d frames)", *key, frame_count)
                started = time.perf_counter()
                try:
                    generated = captioner.generate(sample, prompts[condition])
                    record["inference_time_sec"] = round(time.perf_counter() - started, 3)
                    record["generated_description"] = generated.text
                    record["visual_tokens"] = generated.num_visual_tokens
                    record["num_text_tokens"] = generated.num_text_tokens
                    record["num_total_input_tokens"] = generated.num_total_input_tokens
                except Exception as exc:  # noqa: BLE001
                    record["inference_time_sec"] = round(time.perf_counter() - started, 3)
                    record["error"] = format_exception(exc)
                    logger.error("Run failed %s %s %s: %s", *key, exc)
                _store(runs, record, output_dir)
                flush()
                logger.info("Saved %s %s %s", *key)

        del captioner
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    flush()
    print(f"Results: {output_dir / 'results.json'}")
    print(f"CSV: {output_dir / 'results.csv'}")
    print(f"Report: {output_dir / 'report.md'}")
    failed = [
        row for row in runs if row.get("condition") in conditions and not is_success(row)
    ]
    return 1 if failed else 0


def _frame_stats(item, condition: str) -> tuple[int, int]:
    if condition == CONDITION_FULL:
        return item.plan.full_frame_image_count(), item.full_pixels
    if condition == CONDITION_FULL_2FPS:
        return item.plan.full_2fps_frame_image_count(), item.full_2fps_pixels
    if condition == CONDITION_MOTION_ROI:
        return item.motion_roi_frame_image_count, item.motion_roi_pixels
    if condition == CONDITION_ANCHOR:
        return item.anchor_frame_image_count, item.anchor_pixels
    if condition == CONDITION_PEAK:
        return item.peak_frame_image_count, item.peak_pixels
    return item.plan.motion_frame_image_count(), item.motion_pixels


def _prompt_path(condition: str, args, prompts: dict[str, str], fallback: str | None) -> str | None:
    if condition in {CONDITION_FULL, CONDITION_FULL_2FPS}:
        if args.full_prompt:
            return args.full_prompt
        if condition in prompts:
            return prompts[condition]
        if CONDITION_FULL in prompts:
            return prompts[CONDITION_FULL]
        return fallback
    if args.motion_prompt:
        return args.motion_prompt
    if condition in prompts:
        return prompts[condition]
    return fallback


def _video_ids(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(DEFAULT_VIDEOS)
    ids = []
    for item in raw:
        text = item.strip()
        if not text:
            continue
        if not text.startswith("VIRAT_"):
            text = f"VIRAT_S_{text}"
        ids.append(text)
    return ids


def _report_meta(meta: dict, runs: list[dict], output_dir: Path) -> dict:
    """Keep previously saved conditions when appending one new condition."""
    previous: dict = {}
    path = output_dir / "results.json"
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("meta"), dict):
                previous = loaded["meta"]
        except json.JSONDecodeError:
            previous = {}
    present = {str(row.get("condition")) for row in runs}
    requested = {str(name) for name in meta.get("conditions") or []}
    conditions = [name for name in ALL_CONDITIONS if name in present or name in requested]
    prompts = dict(previous.get("prompts") or {})
    prompts.update(meta.get("prompts") or {})
    prompt_text = dict(previous.get("prompt") or {})
    prompt_text.update(meta.get("prompt") or {})
    models: list[str] = []
    for name in [*(previous.get("models") or []), *(meta.get("models") or [])]:
        if name not in models:
            models.append(str(name))
    merged = dict(meta)
    merged["conditions"] = conditions
    merged["prompts"] = {name: prompts[name] for name in conditions if name in prompts}
    merged["prompt"] = {name: prompt_text[name] for name in conditions if name in prompt_text}
    merged["models"] = models or list(meta.get("models") or [])
    return merged


def _store(runs: list[dict], record: dict, output_dir: Path) -> None:
    key = run_key(record)
    save_run_file(output_dir, record)
    for index, row in enumerate(runs):
        if run_key(row) == key:
            runs[index] = record
            return
    runs.append(record)


if __name__ == "__main__":
    raise SystemExit(main())
