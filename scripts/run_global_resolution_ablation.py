#!/usr/bin/env python3
"""Global-resolution ablation dry-run (optional later inference) on VIRAT9.

Keeps the existing start_only + accumulated-motion ROI pipeline fixed and
varies only Global spatial resolution / Global visual-token budget.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_inference.config import load_config
from vlm_inference.feasibility import collect_job_io_metadata, collect_method_inputs
from vlm_inference.global_resolution import (
    DEFAULT_METHOD,
    GLOBAL_SCALES,
    format_scale,
    resolution_audit,
    summarize_token_stats,
    verify_invariants,
    write_json,
)
from vlm_inference.runner import run_inference

logger = logging.getLogger(__name__)


def _load_raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_condition_config(
    *,
    base: dict,
    output_root: Path,
    scale: float,
    cache_parent: Path,
) -> Path:
    raw = json.loads(json.dumps(base))
    tag = format_scale(scale)
    condition_dir = output_root / f"scale_{tag}"
    raw["experiment"]["global_scale"] = float(scale)
    raw["output"] = {
        "dir": str(cache_parent),
        "file": str(Path(f"scale_{tag}") / "results.json"),
    }
    raw["feasibility"]["cache_file"] = str(condition_dir / "inference_cache.json")
    config_path = condition_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def _audit_rows(cfg, videos: list[str], method: str) -> list[dict]:
    rows: list[dict] = []
    for video_id in videos:
        inputs = collect_method_inputs(video_id, method, cfg)
        if inputs.errors:
            raise RuntimeError(
                f"{video_id} / {method} scale={cfg.experiment.global_scale}: "
                + "; ".join(inputs.errors)
            )
        rows.append(
            resolution_audit(
                inputs, cfg=cfg, extra_io=collect_job_io_metadata(inputs)
            )
        )
    return rows


def _roi_sampling_ready(root: Path, videos: list[str]) -> bool:
    return all((root / video_id / "metadata.json").is_file() for video_id in videos)


def _ensure_shared_roi_crops(output_root: Path, videos: list[str], cfg) -> str:
    dest = output_root / "accumulated_motion"
    existing = ROOT / "outputs" / "tau_sweep" / "tau_8" / "accumulated_motion"
    if _roi_sampling_ready(dest, videos):
        return str(dest)
    if _roi_sampling_ready(existing, videos):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(
                f"{dest} exists but is incomplete; not overwriting existing experiment files"
            )
        dest.symlink_to(existing.resolve())
        logger.info("Reusing existing tau_8 accumulated_motion crops via %s", dest)
        return str(dest)
    from vlm_inference.motion_adaptive import prepare_motion_adaptive_sampling

    logger.info("Preparing accumulated-motion ROI sampling under %s", output_root)
    prepare_motion_adaptive_sampling(cfg, overwrite=False)
    if not _roi_sampling_ready(dest, videos):
        raise FileNotFoundError(
            f"ROI sampling metadata missing under {dest} and tracks were not available"
        )
    return str(dest)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_inference_results(scale_dir: Path) -> Path | None:
    candidates = sorted(
        scale_dir.glob("results_*.json"), key=lambda p: p.stat().st_mtime_ns
    )
    for path in reversed(candidates):
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("dry_run") is False:
            return path
    return None


def _result_rows(path: Path) -> list[dict]:
    payload = _load_json(path)
    rows = payload.get("results") if isinstance(payload, dict) else payload
    return [row for row in (rows or []) if isinstance(row, dict)]


def _dump_raw_predictions(scale_dir: Path, results_path: Path) -> Path:
    pred_dir = scale_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    captions: dict[str, str | None] = {}
    for row in _result_rows(results_path):
        video_id = str(row.get("video_id") or "")
        if not video_id:
            continue
        write_json(pred_dir / f"{video_id}.json", row)
        captions[video_id] = row.get("caption")
    captions_path = scale_dir / "captions.json"
    write_json(captions_path, captions)
    return captions_path


def _audit_by_video(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = _load_json(path)
    if not isinstance(rows, list):
        return {}
    return {str(row["video_id"]): row for row in rows if isinstance(row, dict)}


def _scale_complete(scale_dir: Path, videos: list[str]) -> bool:
    results_path = _latest_inference_results(scale_dir)
    if results_path is None:
        return False
    by_video = {str(row.get("video_id")): row for row in _result_rows(results_path)}
    for video_id in videos:
        row = by_video.get(video_id) or {}
        if row.get("error") or not row.get("caption"):
            return False
    return True


def _write_summaries(output_root: Path, videos: list[str]) -> tuple[Path, Path]:
    scale1_tokens: dict[str, int] = {}
    rows: list[dict] = []
    aggregates: dict[str, dict] = {}

    for scale in GLOBAL_SCALES:
        tag = format_scale(scale)
        scale_dir = output_root / f"scale_{tag}"
        audit = _audit_by_video(scale_dir / "dry_run_audit.json")
        results_path = _latest_inference_results(scale_dir)
        result_by_video = {
            str(row.get("video_id")): row
            for row in (_result_rows(results_path) if results_path else [])
        }
        n_ok = n_fail = n_oom = 0
        token_g = token_r = token_t = 0
        runtime_sum = 0.0
        runtime_n = 0
        for video_id in videos:
            audit_row = audit.get(video_id) or {}
            result = result_by_video.get(video_id) or {}
            error = result.get("error")
            oom = bool(result.get("oom")) or (
                "out of memory" in str(error or "").lower()
            )
            success = bool(result.get("caption")) and not error
            if success:
                n_ok += 1
            elif error or video_id in result_by_video:
                n_fail += 1
                if oom:
                    n_oom += 1
            latency = (result.get("roi_wise") or {}).get("total_latency_s")
            if latency is not None:
                runtime_sum += float(latency)
                runtime_n += 1
            g_tokens = int(audit_row.get("global_visual_tokens") or 0)
            r_tokens = int(audit_row.get("roi_visual_tokens") or 0)
            t_tokens = int(audit_row.get("total_visual_tokens") or 0)
            token_g += g_tokens
            token_r += r_tokens
            token_t += t_tokens
            if abs(float(scale) - 1.0) <= 1e-12:
                scale1_tokens[video_id] = t_tokens
            base_tokens = scale1_tokens.get(video_id)
            ratio = (t_tokens / base_tokens) if base_tokens else None
            row = {
                "video_id": video_id,
                "global_scale": float(scale),
                "global_resolution": audit_row.get("resized_global_resolution"),
                "vlm_global_resolution": audit_row.get("vlm_global_resolution"),
                "n_global_images": audit_row.get("num_global_images"),
                "n_roi_images": audit_row.get("num_roi_images"),
                "global_visual_tokens": g_tokens,
                "roi_visual_tokens": r_tokens,
                "total_visual_tokens": t_tokens,
                "token_ratio_vs_scale1": ratio,
                "total_input_pixels": audit_row.get("total_input_pixels"),
                "inference_time": latency,
                "success": success,
                "oom": oom,
                "error": error,
                "caption": result.get("caption"),
                "results_path": str(results_path) if results_path else None,
            }
            rows.append(row)

        agg = {
            "global_scale": float(scale),
            "n_videos": len(videos),
            "n_success": n_ok,
            "n_failed": n_fail,
            "n_oom": n_oom,
            "global_visual_tokens_sum": token_g,
            "roi_visual_tokens_sum": token_r,
            "total_visual_tokens_sum": token_t,
            "n_global_images_sum": int(
                sum(int((audit.get(v) or {}).get("num_global_images") or 0) for v in videos)
            ),
            "n_roi_images_sum": int(
                sum(int((audit.get(v) or {}).get("num_roi_images") or 0) for v in videos)
            ),
            "total_input_pixels_sum": int(
                sum(int((audit.get(v) or {}).get("total_input_pixels") or 0) for v in videos)
            ),
            "inference_time_sum": runtime_sum,
            "inference_time_mean": (runtime_sum / runtime_n) if runtime_n else None,
            "results_path": str(results_path) if results_path else None,
        }
        if abs(float(scale) - 1.0) <= 1e-12:
            agg["token_ratio_vs_scale1"] = 1.0
            agg["token_reduction_vs_scale1"] = 0.0
        else:
            base_sum = aggregates.get(format_scale(1.0), {}).get("total_visual_tokens_sum")
            ratio = (token_t / base_sum) if base_sum else None
            agg["token_ratio_vs_scale1"] = ratio
            agg["token_reduction_vs_scale1"] = (1.0 - ratio) if ratio is not None else None
        aggregates[tag] = agg
        write_json(
            scale_dir / "summary.json",
            {
                "rows": [
                    row
                    for row in rows
                    if abs(float(row["global_scale"]) - float(scale)) <= 1e-12
                ],
                "aggregate": agg,
            },
        )

    summary_json = output_root / "inference_summary.json"
    summary_csv = output_root / "inference_summary.csv"
    n_success = sum(1 for row in rows if row.get("success"))
    n_fail = sum(1 for row in rows if row.get("error") or (row.get("video_id") and not row.get("success") and row.get("results_path")))
    n_oom = sum(1 for row in rows if row.get("oom"))
    write_json(
        summary_json,
        {
            "method": DEFAULT_METHOD,
            "videos": videos,
            "scales": [float(s) for s in GLOBAL_SCALES],
            "n_jobs": len(videos) * len(GLOBAL_SCALES),
            "n_success": n_success,
            "n_failed": n_fail,
            "n_oom": n_oom,
            "rows": rows,
            "aggregates": aggregates,
        },
    )
    fieldnames = [
        "video_id",
        "global_scale",
        "global_resolution",
        "vlm_global_resolution",
        "n_global_images",
        "n_roi_images",
        "global_visual_tokens",
        "roi_visual_tokens",
        "total_visual_tokens",
        "token_ratio_vs_scale1",
        "total_input_pixels",
        "inference_time",
        "success",
        "oom",
        "error",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("global_resolution", "vlm_global_resolution"):
                if isinstance(out.get(key), list):
                    out[key] = "x".join(str(v) for v in out[key])
            writer.writerow(out)
    write_json(output_root / "aggregate_summary.json", aggregates)
    return summary_json, summary_csv


def _invariants_passed(output_root: Path) -> bool:
    path = output_root / "invariants.json"
    if not path.is_file():
        return False
    try:
        return bool(_load_json(path).get("passed"))
    except json.JSONDecodeError:
        return False


def _finalize_scale(scale_dir: Path, videos: list[str], output_root: Path) -> None:
    results_path = _latest_inference_results(scale_dir)
    if results_path is None:
        logger.warning("No inference results yet in %s", scale_dir)
        _write_summaries(output_root, videos)
        return
    _dump_raw_predictions(scale_dir, results_path)
    _write_summaries(output_root, videos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/feasibility_global_resolution_ablation.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/global_resolution_ablation",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run VLM inference after dry-run invariants pass.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = _load_raw(Path(args.config))
    base["output"] = {"dir": str(output_root), "file": "results.json"}
    base["feasibility"]["cache_file"] = str(output_root / "inference_cache.json")
    base["experiments"] = [DEFAULT_METHOD]
    base.setdefault("feasibility", {}).setdefault("method_prompts", {})[
        DEFAULT_METHOD
    ] = base.get("prompt_file") or "prompts/frames_roi.txt"

    prepare_cfg = load_config(args.config)
    prepare_cfg.output.dir = str(output_root)
    prepare_cfg.output.file = "results.json"
    prepare_cfg.feasibility.cache_file = str(output_root / "inference_cache.json")
    prepare_cfg.feasibility.methods = [DEFAULT_METHOD]
    prepare_cfg.experiment.global_scale = 1.0
    prepare_cfg.experiment.roi_sampling_overwrite = False

    videos = list(prepare_cfg.feasibility.videos)
    _ensure_shared_roi_crops(output_root, videos, prepare_cfg)

    config_paths: list[Path] = []
    skip_dry_run = _invariants_passed(output_root) and all(
        (output_root / f"scale_{format_scale(scale)}" / "config.yaml").is_file()
        and (output_root / f"scale_{format_scale(scale)}" / "dry_run_audit.json").is_file()
        for scale in GLOBAL_SCALES
    )
    if skip_dry_run:
        logger.info("Reusing passed dry-run invariants under %s", output_root)
        for scale in GLOBAL_SCALES:
            config_paths.append(
                _write_condition_config(
                    base=base,
                    output_root=output_root,
                    scale=scale,
                    cache_parent=output_root,
                )
            )
    else:
        logger.info("Collecting unscaled baseline (global_scale=1.0, no extra resize)")
        baseline_rows = _audit_rows(prepare_cfg, videos, DEFAULT_METHOD)
        write_json(output_root / "baseline_unscaled.json", baseline_rows)

        rows_by_scale: dict[float, list[dict]] = {}
        for scale in GLOBAL_SCALES:
            tag = format_scale(scale)
            print(f"Dry-run global_scale={tag}", flush=True)
            config_path = _write_condition_config(
                base=base,
                output_root=output_root,
                scale=scale,
                cache_parent=output_root,
            )
            config_paths.append(config_path)
            cfg = load_config(config_path)
            rows = _audit_rows(cfg, list(cfg.feasibility.videos), DEFAULT_METHOD)
            rows_by_scale[float(scale)] = rows
            write_json(output_root / f"scale_{tag}" / "dry_run_audit.json", rows)
            run_inference(cfg, dry_run=True)

        token_stats = summarize_token_stats(rows_by_scale)
        write_json(output_root / "token_stats.json", token_stats)
        invariant_report = verify_invariants(
            rows_by_scale,
            expected_videos=videos,
            baseline_rows=baseline_rows,
        )
        write_json(output_root / "invariants.json", invariant_report)
        write_json(
            output_root / "dry_run_report.json",
            {
                "method": DEFAULT_METHOD,
                "videos": videos,
                "scales": [float(s) for s in GLOBAL_SCALES],
                "n_videos": len(videos),
                "n_conditions": len(GLOBAL_SCALES),
                "n_inferences_if_run": len(videos) * len(GLOBAL_SCALES),
                "token_stats": token_stats,
                "invariants": invariant_report,
                "rows_by_scale": {
                    format_scale(scale): rows_by_scale[scale] for scale in GLOBAL_SCALES
                },
            },
        )
        print(json.dumps(invariant_report, indent=2), flush=True)
        print(
            f"Dry-run passed: {len(videos)} videos × {len(GLOBAL_SCALES)} scales = "
            f"{len(videos) * len(GLOBAL_SCALES)} jobs.",
            flush=True,
        )
        if not invariant_report.get("passed"):
            raise SystemExit("Dry-run invariants failed; refusing inference.")

    if not args.inference:
        print("VLM inference was not run.", flush=True)
        return

    status = {
        "n_jobs": len(videos) * len(GLOBAL_SCALES),
        "scales": {},
    }
    for scale, config_path in zip(GLOBAL_SCALES, config_paths):
        tag = format_scale(scale)
        scale_dir = output_root / f"scale_{tag}"
        print(f"Inference global_scale={tag}", flush=True)
        if _scale_complete(scale_dir, videos):
            logger.info("Skipping completed scale %s", tag)
            status["scales"][tag] = {"skipped": True, "complete": True}
        else:
            results_path = run_inference(load_config(config_path), dry_run=False)
            status["scales"][tag] = {
                "skipped": False,
                "results_path": str(results_path),
                "complete": _scale_complete(scale_dir, videos),
            }
        _finalize_scale(scale_dir, videos, output_root)
        write_json(output_root / "run_status.json", status)

    summary_json, summary_csv = _write_summaries(output_root, videos)
    print(f"Wrote {summary_json}", flush=True)
    print(f"Wrote {summary_csv}", flush=True)
    print("Completed Global-resolution ablation inference.", flush=True)


if __name__ == "__main__":
    main()
