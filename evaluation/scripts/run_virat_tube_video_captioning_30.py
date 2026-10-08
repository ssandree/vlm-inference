#!/usr/bin/env python3
"""VIRAT sample30 tube-video captioning. Requires Task 2 audit PASS.

Does not rewrite sample30 image-packed outputs. Loads Qwen3-VL once.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
VLM_INF_SRC = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/src")
for path in (str(EVAL_ROOT), str(VLM_INF_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vlm_inference.config import load_config
from vlm_inference.gt_action_cls import seed_everything
from vlm_inference.model import Qwen3VLCaptioner
from vlm_inference.motion_adaptive import DEFAULT_TAU
from vlm_inference.sovabench import GLOBAL_DEDUP_EPS_SEC
from vlm_inference.sovabench_temporal import install_recorded_path_aliases
from vlm_inference.video_io import inspect_video, install_av_video_reader, resolve_video_file

from vantage_vqa.tube_video import (
    bind_existing_roi_jpegs,
    image_packed_sample,
    restore_tube_packing,
    summarize_messages,
    tube_video_sample,
)
from vantage_vqa.visual import (
    CONDITION_ROI_CENTROID,
    CONDITION_ROI_FIXED_1FPS,
    build_video_condition_visual,
    load_video_motion_records,
    load_video_rois,
)

logger = logging.getLogger(__name__)

AUDIT_MOD = EVAL_ROOT / "scripts" / "run_virat_tube_video_audit.py"
AUDIT_DIR = EVAL_ROOT / "experiments" / "virat" / "tube_video_audit"
OUT_DIR = EVAL_ROOT / "experiments" / "virat" / "tube_video_captioning_30"
SAMPLE30_OUT = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/outputs/sample30_30b_start_centroid")
SAMPLE30_RESULTS = (
    SAMPLE30_OUT / "feasibility_results_sample30_30b_start_centroid_0914125816.json"
)
CAPTION_SRC = "global_roi_start_only_roi_centroid_2fps"

TUBE_CONDITIONS = (
    (CAPTION_SRC, f"{CAPTION_SRC}_tube_video"),
    (CONDITION_ROI_FIXED_1FPS, "roi_fixed_1fps_tube_video"),
    (CONDITION_ROI_CENTROID, "roi_centroid_tube_video"),
)


def _load_audit():
    spec = importlib.util.spec_from_file_location("virat_tube_video_audit", AUDIT_MOD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def require_task2_pass() -> dict[str, Any]:
    summary_path = AUDIT_DIR / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Task 2 summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed") or int(summary.get("n_failed") or 0) != 0:
        write_json(
            OUT_DIR / "aborted.json",
            {
                "aborted": True,
                "reason": "Task 2 structural/evidence validation did not pass",
                "task2_summary": str(summary_path),
                "n_failed": summary.get("n_failed"),
            },
        )
        raise SystemExit("Task 2 did not PASS; inference not started")
    return summary


def assert_tube_construction(messages, packing) -> list[str]:
    failures = []
    summary = summarize_messages(messages)
    if summary["n_image"] != len(packing.global_frames):
        failures.append(
            f"assert_global_not_image:{summary['n_image']}!={len(packing.global_frames)}"
        )
    if summary["n_video"] != len(packing.tubes):
        failures.append(
            f"assert_tube_not_independent_video:{summary['n_video']}!={len(packing.tubes)}"
        )
    video_items = [item for item in summary["items"] if item.get("type") == "video"]
    image_items = [item for item in summary["items"] if item.get("type") == "image"]
    if any(item.get("type") == "video" for item in image_items):
        failures.append("assert_global_is_video")
    for (roi_id, frames), item in zip(packing.tubes, video_items):
        if not item.get("is_frame_list"):
            failures.append(f"assert_tube_{roi_id}_not_frame_list")
        if int(item.get("n_frames") or 0) != len(frames):
            failures.append(f"assert_tube_{roi_id}_frame_count")
    return failures


def load_image_packed_captions() -> dict[str, dict[str, Any]]:
    payload = json.loads(SAMPLE30_RESULTS.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("results") or []:
        if row.get("method") != CAPTION_SRC:
            continue
        out[str(row["video_id"])] = {
            "caption": row.get("caption") or "",
            "visual_tokens": row.get("num_visual_tokens"),
            "processed_pixels": row.get("total_input_pixels"),
            "num_global_images": row.get("num_global_images"),
            "num_roi_tubes": row.get("num_roi_tubes"),
            "num_roi_frames": row.get("num_roi_images"),
            "success": bool(row.get("success")),
            "vlm_time": ((row.get("roi_wise") or {}).get("total_latency_s")),
        }
    return out


def load_task2_costs() -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads((AUDIT_DIR / "rows.json").read_text(encoding="utf-8"))["rows"]
    out = {}
    for row in rows:
        out[(row["video_id"], row["condition"])] = row
    return out


def packing_debug(packing) -> dict[str, Any]:
    return {
        "num_global_images": len(packing.global_frames),
        "num_roi_tubes": len(packing.tubes),
        "num_roi_frames": sum(len(frames) for _roi, frames in packing.tubes),
        "tube_frame_counts": {str(roi_id): len(frames) for roi_id, frames in packing.tubes},
        "tube_timestamps": {
            str(roi_id): [frame.timestamp for frame in frames]
            for roi_id, frames in packing.tubes
        },
        "global_timestamps": [frame.timestamp for frame in packing.global_frames],
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict[str, Any]], *, model_load_count: int) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    conditions = {}
    for condition, subset in by_cond.items():
        ok = [row for row in subset if not row.get("error")]
        tokens = [float(row["visual_tokens"]) for row in ok if row.get("visual_tokens") is not None]
        pixels = [float(row["processed_pixels"]) for row in ok if row.get("processed_pixels") is not None]
        vlm = [float(row["vlm_time"]) for row in ok]
        total = [float(row["total_time"]) for row in ok]
        packed_red = [
            float(row["token_reduction_vs_image_packed"])
            for row in ok
            if row.get("token_reduction_vs_image_packed") is not None
        ]
        exact = sum(1 for row in ok if row.get("caption_exact_match_image_packed"))
        comparable = sum(
            1 for row in ok if row.get("image_packed_caption") not in (None, "")
        )
        conditions[condition] = {
            "completed": len(ok),
            "failed": sum(1 for row in subset if row.get("error")),
            "n": len(subset),
            "mean_global_images": _mean([row["num_global_images"] for row in ok]),
            "mean_roi_tubes": _mean([row["num_roi_tubes"] for row in ok]),
            "mean_roi_frames": _mean([row["num_roi_frames"] for row in ok]),
            "mean_visual_tokens": _mean(tokens),
            "median_visual_tokens": _median(tokens),
            "mean_processed_pixels": _mean(pixels),
            "mean_vlm_time": _mean(vlm),
            "mean_total_time": _mean(total),
            "mean_token_reduction_vs_image_packed": _mean(packed_red),
            "caption_exact_match_count": exact,
            "caption_exact_match_comparable": comparable,
        }
    return {
        "n_videos": 30,
        "n_rows": len(rows),
        "model_load_count": model_load_count,
        "generation_ran": True,
        "conditions": conditions,
        "existing_files_untouched": [
            str(SAMPLE30_OUT),
            str(SAMPLE30_RESULTS),
            str(AUDIT_DIR / "summary.json"),
        ],
        "output_dir": str(OUT_DIR),
    }


def cost_table(rows: list[dict[str, Any]], packed_captions, task2) -> dict[str, Any]:
    """Read-only Full/image-packed vs tube-video costs. No new Full inference."""
    note = (
        "sample30_30b_start_centroid has no Full-frame captioning method; "
        "Full baseline captions/cost for this 30-video 30B experiment were not found."
    )
    packed_caption_tokens = [
        float(v["visual_tokens"])
        for v in packed_captions.values()
        if v.get("visual_tokens") is not None
    ]
    packed_pixels = [
        float(v["processed_pixels"])
        for v in packed_captions.values()
        if v.get("processed_pixels") is not None
    ]
    table = [
        {
            "representation": "full_baseline",
            "source": None,
            "n": 0,
            "mean_visual_tokens": None,
            "mean_processed_pixels": None,
            "available": False,
            "note": note,
        },
        {
            "representation": "image_packed_roi",
            "source_condition": CAPTION_SRC,
            "n": len(packed_caption_tokens),
            "mean_visual_tokens": _mean(packed_caption_tokens),
            "mean_processed_pixels": _mean(packed_pixels),
            "available": True,
            "note": "Existing sample30 image-packed captioning results (not re-run).",
        },
    ]
    for src, tube_name in TUBE_CONDITIONS:
        subset = [row for row in rows if row["condition"] == tube_name and not row.get("error")]
        tokens = [float(row["visual_tokens"]) for row in subset if row.get("visual_tokens") is not None]
        pixels = [float(row["processed_pixels"]) for row in subset if row.get("processed_pixels") is not None]
        packed_tokens = []
        if src == CAPTION_SRC:
            packed_tokens = packed_caption_tokens
        else:
            packed_tokens = [
                float(task2[(row["video_id"], src)]["image_packed_message"]["visual_tokens"])
                for row in subset
                if (row["video_id"], src) in task2
                and task2[(row["video_id"], src)].get("image_packed_message", {}).get("visual_tokens")
            ]
        mean_new = _mean(tokens)
        mean_old = _mean(packed_tokens)
        table.append(
            {
                "representation": "tube_video_roi",
                "condition": tube_name,
                "source_roi_condition": src,
                "n": len(subset),
                "mean_visual_tokens": mean_new,
                "mean_processed_pixels": _mean(pixels),
                "image_packed_mean_visual_tokens": mean_old,
                "token_ratio_vs_image_packed": (
                    float(mean_new / mean_old) if mean_new and mean_old else None
                ),
                "token_reduction_vs_image_packed": (
                    float((mean_old - mean_new) / mean_old) if mean_new and mean_old else None
                ),
                "available": True,
            }
        )
    return {"full_baseline_found": False, "rows": table, "note": note}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE30_OUT.resolve() == OUT_DIR.resolve():
        raise SystemExit("refusing to write into sample30 outputs")
    task2 = require_task2_pass()
    logger.info("Task 2 passed n_passed=%s n_failed=%s", task2["n_passed"], task2["n_failed"])

    seed_everything(0)
    install_av_video_reader()
    install_recorded_path_aliases()
    audit = _load_audit()
    cfg = load_config(EVAL_ROOT / "configs" / "virat_tube_video_audit.yaml")
    cfg.output.dir = str(OUT_DIR)
    cfg.experiment.save_vlm_inputs = False
    videos = list(cfg.feasibility.videos)
    if len(videos) != 30:
        raise SystemExit(f"expected 30 videos, got {len(videos)}")
    tracks_root = Path(cfg.feasibility.tracks_root)
    video_roots = [Path(p) for p in cfg.feasibility.datasets.all_video_roots()]
    jpeg_quality = int(cfg.feasibility.jpeg_quality)
    prompt = Path(cfg.prompt_file).read_text(encoding="utf-8")
    packed_captions = load_image_packed_captions()
    task2_costs = load_task2_costs()
    audit_frames = AUDIT_DIR / "frames"

    logger.info("Loading Qwen3-VL once from %s", cfg.model.path)
    captioner = Qwen3VLCaptioner(cfg)
    model_load_count = 1

    jsonl_path = OUT_DIR / "predictions.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    rows: list[dict[str, Any]] = []
    decoded_by_video: dict[str, dict[int, Any]] = defaultdict(dict)

    for video_id in videos:
        video_path = resolve_video_file(video_id, video_roots)
        info = inspect_video(video_path)
        duration = float(max(int(info.frame_count) - 1, 0)) / (
            float(info.native_fps) if info.native_fps > 0 else 30.0
        )
        rois = load_video_rois(video_id, tracks_root=tracks_root, video_roots=video_roots)
        records = load_video_motion_records(
            video_id,
            tracks_root=tracks_root,
            video_roots=video_roots,
            motion_cfg=cfg.experiment.centroid_motion,
        )
        for src, tube_name in TUBE_CONDITIONS:
            t_build = time.perf_counter()
            try:
                if src == CAPTION_SRC:
                    manifest = audit.load_caption_manifest(video_id)
                    packing = audit.packing_from_caption_manifest(manifest)
                    _image_sample, tube_sample = audit.caption_samples(
                        video_id,
                        packing,
                        info=info,
                        tracks_root=tracks_root,
                        video_roots=video_roots,
                    )
                    processed = audit.jpeg_pixels(
                        [frame.path for frame in packing.global_frames]
                        + [frame.path for _roi, frames in packing.tubes for frame in frames]
                    )
                    image_sample = _image_sample
                else:
                    visual = build_video_condition_visual(
                        video_id=video_id,
                        video_path=video_path,
                        condition=src,
                        info=info,
                        rois=rois,
                        records=records,
                        video_roots=video_roots,
                        frames_root=audit_frames,
                        jpeg_quality=jpeg_quality,
                        decoded_frames=decoded_by_video[video_id],
                        tau=DEFAULT_TAU,
                        dedup_eps=float(
                            cfg.experiment.global_insert_dedup_eps or GLOBAL_DEDUP_EPS_SEC
                        ),
                        skip_frames=True,
                    )
                    visual = bind_existing_roi_jpegs(visual, audit_frames)
                    packing = restore_tube_packing(visual)
                    image_sample = image_packed_sample(visual)
                    tube_sample = tube_video_sample(visual, packing)
                    processed = int(visual.processed_pixels)
                build_time = time.perf_counter() - t_build
                messages = captioner._build_messages(tube_sample, prompt)
                failures = assert_tube_construction(messages, packing)
                if failures:
                    raise RuntimeError("pre_generate_assert:" + ",".join(failures))
                t0 = time.perf_counter()
                result = captioner.generate(tube_sample, prompt)
                vlm_time = time.perf_counter() - t0
                packed = packed_captions.get(video_id) if src == CAPTION_SRC else None
                task2_row = task2_costs.get((video_id, src))
                packed_tokens = None
                packed_caption = None
                if packed:
                    packed_tokens = packed.get("visual_tokens")
                    packed_caption = packed.get("caption")
                elif task2_row:
                    packed_tokens = (task2_row.get("image_packed_message") or {}).get(
                        "visual_tokens"
                    )
                reduction = None
                if packed_tokens and result.num_visual_tokens:
                    reduction = (float(packed_tokens) - float(result.num_visual_tokens)) / float(
                        packed_tokens
                    )
                debug = packing_debug(packing)
                rec = {
                    "video_id": video_id,
                    "condition": tube_name,
                    "source_condition": src,
                    "caption": result.text,
                    "status": "ok",
                    "error": None,
                    **debug,
                    "visual_tokens": result.num_visual_tokens,
                    "processed_pixels": processed,
                    "input_build_time": build_time,
                    "vlm_time": vlm_time,
                    "total_time": build_time + vlm_time,
                    "image_packed_caption": packed_caption,
                    "image_packed_visual_tokens": packed_tokens,
                    "token_reduction_vs_image_packed": reduction,
                    "caption_exact_match_image_packed": (
                        bool(packed_caption) and packed_caption.strip() == result.text.strip()
                    ),
                    "model": cfg.model.path,
                }
            except Exception as exc:
                logger.exception("inference failed %s %s", video_id, tube_name)
                rec = {
                    "video_id": video_id,
                    "condition": tube_name,
                    "source_condition": src,
                    "caption": "",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "num_global_images": None,
                    "num_roi_tubes": None,
                    "num_roi_frames": None,
                    "tube_frame_counts": {},
                    "tube_timestamps": {},
                    "global_timestamps": [],
                    "visual_tokens": None,
                    "processed_pixels": None,
                    "input_build_time": 0.0,
                    "vlm_time": 0.0,
                    "total_time": 0.0,
                    "image_packed_caption": (packed_captions.get(video_id) or {}).get("caption")
                    if src == CAPTION_SRC
                    else None,
                    "image_packed_visual_tokens": None,
                    "token_reduction_vs_image_packed": None,
                    "caption_exact_match_image_packed": False,
                    "model": cfg.model.path,
                }
            rows.append(rec)
            append_jsonl(jsonl_path, rec)
            logger.info(
                "%s %s status=%s tokens=%s vlm=%.2fs",
                video_id,
                tube_name,
                rec["status"],
                rec.get("visual_tokens"),
                rec.get("vlm_time") or 0.0,
            )

    summary = summarize_rows(rows, model_load_count=model_load_count)
    comparison = []
    for video_id in videos:
        packed = packed_captions.get(video_id) or {}
        by_cond = {row["condition"]: row for row in rows if row["video_id"] == video_id}
        comparison.append(
            {
                "video_id": video_id,
                "image_packed_caption": packed.get("caption"),
                "image_packed_visual_tokens": packed.get("visual_tokens"),
                "tube_captions": {
                    name: {
                        "caption": (by_cond.get(name) or {}).get("caption"),
                        "visual_tokens": (by_cond.get(name) or {}).get("visual_tokens"),
                        "exact_match_image_packed": (by_cond.get(name) or {}).get(
                            "caption_exact_match_image_packed"
                        ),
                        "error": (by_cond.get(name) or {}).get("error"),
                    }
                    for _src, name in TUBE_CONDITIONS
                },
            }
        )
    costs = cost_table(rows, packed_captions, task2_costs)
    summary["cost_comparison"] = costs
    summary["outputs"] = {
        "predictions_json": str(OUT_DIR / "predictions.json"),
        "predictions_jsonl": str(jsonl_path),
        "summary": str(OUT_DIR / "summary.json"),
        "comparison": str(OUT_DIR / "caption_comparison.json"),
        "cost_table": str(OUT_DIR / "cost_comparison.json"),
    }
    write_json(OUT_DIR / "predictions.json", {"model_load_count": model_load_count, "rows": rows})
    write_json(OUT_DIR / "caption_comparison.json", {"n": len(comparison), "rows": comparison})
    write_json(OUT_DIR / "cost_comparison.json", costs)
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))
    print(json.dumps(summary["conditions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
