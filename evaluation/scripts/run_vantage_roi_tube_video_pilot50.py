#!/usr/bin/env python3
"""Validate then run ROI tube-as-video on VANTAGE VQA pilot50.

Phase 1: processor-only validation for roi_fixed_1fps and roi_centroid (50x2).
Phase 2: only if Phase 1 has zero structural/evidence failures; 100 inferences,
one Qwen load. Does not rewrite the existing 50x5 image-packed results.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
VLM_INF_SRC = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/src")
for path in (str(EVAL_ROOT), str(VLM_INF_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from transformers import AutoProcessor

from vlm_inference.config import load_config
from vlm_inference.gt_action_cls import seed_everything
from vlm_inference.model import Qwen3VLCaptioner

from vantage_vqa.prompt import extract_answer, format_vantage_vqa_prompt
from vantage_vqa.runner import (
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_VIDEO_ROOT,
    load_pilot_items,
    prepare_unique_videos,
    video_duration_sec,
)
from vantage_vqa.tube_video import (
    bind_existing_roi_jpegs,
    image_packed_sample,
    inspect_processor_output,
    restore_tube_packing,
    summarize_messages,
    tube_video_sample,
    validate_tube_representation,
)
from vantage_vqa.visual import ROI_CONDITIONS

logger = logging.getLogger(__name__)

OUT_DIR = DEFAULT_OUTPUT / "tube_video"
EXISTING_JSONL = DEFAULT_OUTPUT / "predictions.jsonl"
EXISTING_SUMMARY = DEFAULT_OUTPUT / "full_pilot_summary.json"
FULL_MEANS = {
    "full_2fps": 6648.0,
    "full_1fps": 3441.6,
    "full_05fps": 1833.6,
}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def load_existing_rows() -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for line in EXISTING_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(int(row["index"]), str(row["condition"]))] = row
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tube_debug(packing) -> dict[str, Any]:
    return {
        "num_global_images": len(packing.global_frames),
        "num_roi_tubes": len(packing.tubes),
        "num_roi_frames": sum(len(frames) for _roi, frames in packing.tubes),
        "global_timestamps": [frame.timestamp for frame in packing.global_frames],
        "tubes": [
            {
                "roi_id": roi_id,
                "n_frames": len(frames),
                "timestamps": [frame.timestamp for frame in frames],
                "native_frames": [frame.native_frame for frame in frames],
            }
            for roi_id, frames in packing.tubes
        ],
    }


def condition_phase1_stats(
    rows: list[dict[str, Any]],
    *,
    existing: dict[tuple[int, str], dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    subset = [row for row in rows if row["condition"] == condition]
    tokens = [float(row["visual_tokens"]) for row in subset if row.get("visual_tokens") is not None]
    packed_tokens = []
    for row in subset:
        old = existing.get((int(row["index"]), condition))
        if old and old.get("visual_tokens") is not None and row.get("visual_tokens") is not None:
            packed_tokens.append((float(old["visual_tokens"]), float(row["visual_tokens"])))
    reductions = [
        (old - new) / old for old, new in packed_tokens if old > 0
    ]
    mean_new = _mean(tokens)
    return {
        "n": len(subset),
        "n_failures": sum(1 for row in subset if row["failures"]),
        "mean_global_images": _mean([row["num_global_images"] for row in subset]),
        "mean_roi_tubes": _mean([row["num_roi_tubes"] for row in subset]),
        "mean_roi_frames": _mean([row["num_roi_frames"] for row in subset]),
        "mean_visual_tokens": mean_new,
        "median_visual_tokens": _median(tokens),
        "min_visual_tokens": min(tokens) if tokens else None,
        "max_visual_tokens": max(tokens) if tokens else None,
        "mean_processed_pixels": _mean([row["processed_pixels"] for row in subset]),
        "image_packed_mean_visual_tokens": _mean([old for old, _new in packed_tokens]),
        "token_reduction_vs_image_packed": _mean(reductions),
        "token_ratio_vs_image_packed": (
            mean_new / _mean([old for old, _new in packed_tokens])
            if packed_tokens and _mean([old for old, _new in packed_tokens])
            else None
        ),
        "token_ratio_vs_full_2fps": (mean_new / FULL_MEANS["full_2fps"]) if mean_new else None,
        "token_ratio_vs_full_1fps": (mean_new / FULL_MEANS["full_1fps"]) if mean_new else None,
        "token_ratio_vs_full_05fps": (mean_new / FULL_MEANS["full_05fps"]) if mean_new else None,
        "failures": [row for row in subset if row["failures"]],
    }


def run_phase1(items, prepared, cfg, existing) -> dict[str, Any]:
    frames_root = prepared["frames_root"]
    infos = prepared["infos"]
    cache = prepared["visual_cache"]
    logger.info("Loading AutoProcessor only (no Qwen weights) from %s", cfg.model.path)
    processor = AutoProcessor.from_pretrained(cfg.model.path)
    builder = object.__new__(Qwen3VLCaptioner)
    builder.cfg = cfg

    visual_store: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for item in items:
        for condition in ROI_CONDITIONS:
            key = (str(item["video_id"]), condition)
            failures: list[str] = []
            try:
                if key not in visual_store:
                    t0 = time.perf_counter()
                    visual = cache.require(str(item["video_id"]), condition)
                    visual = bind_existing_roi_jpegs(visual, frames_root)
                    packing = restore_tube_packing(visual)
                    packed_sample = image_packed_sample(visual)
                    tube_sample = tube_video_sample(visual, packing)
                    duration = video_duration_sec(infos[item["video_id"]])
                    prompt = format_vantage_vqa_prompt(item["question"], item["options"])
                    messages = Qwen3VLCaptioner._build_messages(builder, tube_sample, prompt)
                    audit = inspect_processor_output(processor, messages)
                    build_time = time.perf_counter() - t0
                    visual_store[key] = {
                        "visual": visual,
                        "packing": packing,
                        "packed_sample": packed_sample,
                        "tube_sample": tube_sample,
                        "duration": duration,
                        "audit": audit,
                        "messages_template_ok": summarize_messages(messages),
                        "input_build_time": build_time,
                    }
                    failures.extend(
                        validate_tube_representation(
                            packing=packing,
                            packed_sample=packed_sample,
                            tube_sample=tube_sample,
                            messages=messages,
                            audit=audit,
                            duration_sec=duration,
                        )
                    )
                    audit_tokens = (audit.get("token_stats") or {}).get("num_visual_tokens")
                    processed = int(visual.processed_pixels)
                    debug = tube_debug(packing)
                    msg = summarize_messages(messages)
                    input_build_time = build_time
                else:
                    stored = visual_store[key]
                    packing = stored["packing"]
                    visual = stored["visual"]
                    prompt = format_vantage_vqa_prompt(item["question"], item["options"])
                    messages = Qwen3VLCaptioner._build_messages(
                        builder, stored["tube_sample"], prompt
                    )
                    failures.extend(
                        validate_tube_representation(
                            packing=packing,
                            packed_sample=stored["packed_sample"],
                            tube_sample=stored["tube_sample"],
                            messages=messages,
                            audit=stored["audit"],
                            duration_sec=stored["duration"],
                        )
                    )
                    audit = stored["audit"]
                    audit_tokens = (audit.get("token_stats") or {}).get("num_visual_tokens")
                    processed = int(visual.processed_pixels)
                    debug = tube_debug(packing)
                    msg = summarize_messages(messages)
                    input_build_time = 0.0
            except Exception as exc:
                failures.append(f"exception:{type(exc).__name__}:{exc}")
                logger.exception("phase1 failed %s %s", item["index"], condition)
                audit = {}
                audit_tokens = None
                processed = 0
                debug = {
                    "num_global_images": 0,
                    "num_roi_tubes": 0,
                    "num_roi_frames": 0,
                    "global_timestamps": [],
                    "tubes": [],
                }
                msg = {"n_image": 0, "n_video": 0, "items": []}
                input_build_time = 0.0
                traceback.print_exc()

            row = {
                "index": int(item["index"]),
                "video_id": item["video_id"],
                "q_uid": item["q_uid"],
                "condition": condition,
                "passed": not failures,
                "failures": failures,
                "num_global_images": debug["num_global_images"],
                "num_roi_tubes": debug["num_roi_tubes"],
                "num_roi_frames": debug["num_roi_frames"],
                "visual_tokens": audit_tokens,
                "processed_pixels": processed,
                "n_message_images": msg.get("n_image"),
                "n_message_videos": msg.get("n_video"),
                "image_grid_thw": audit.get("image_grid_thw"),
                "video_grid_thw": audit.get("video_grid_thw"),
                "n_image_pad": audit.get("n_image_pad"),
                "n_video_pad": audit.get("n_video_pad"),
                "temporal_pack": audit.get("temporal_pack"),
                "input_build_time": input_build_time,
                "tubes": debug["tubes"],
                "global_timestamps": debug["global_timestamps"],
            }
            rows.append(row)
            status = "PASS" if not failures else "FAIL"
            logger.info(
                "phase1 %s %s %s tokens=%s tubes=%s frames=%s failures=%s",
                item["index"],
                condition,
                status,
                audit_tokens,
                debug["num_roi_tubes"],
                debug["num_roi_frames"],
                failures,
            )

    report = {
        "phase": 1,
        "n_total": len(rows),
        "n_passed": sum(1 for row in rows if row["passed"]),
        "n_failed": sum(1 for row in rows if not row["passed"]),
        "passed": all(row["passed"] for row in rows) and len(rows) == 100,
        "conditions": {
            condition: condition_phase1_stats(rows, existing=existing, condition=condition)
            for condition in ROI_CONDITIONS
        },
        "rows": rows,
    }
    return report, visual_store


def run_phase2(items, visual_store, cfg, existing) -> dict[str, Any]:
    logger.info("Loading Qwen3-VL once from %s", cfg.model.path)
    captioner = Qwen3VLCaptioner(cfg)
    model_load_count = 1
    rows: list[dict[str, Any]] = []
    for item in items:
        for condition in ROI_CONDITIONS:
            stored = visual_store[(str(item["video_id"]), condition)]
            packing = stored["packing"]
            visual = stored["visual"]
            tube_sample = stored["tube_sample"]
            debug = tube_debug(packing)
            prompt = format_vantage_vqa_prompt(item["question"], item["options"])
            first = not any(
                row["video_id"] == item["video_id"] and row["condition"] == condition
                for row in rows
            )
            input_build_time = float(stored["input_build_time"] if first else 0.0)
            try:
                t0 = time.perf_counter()
                result = captioner.generate(tube_sample, prompt)
                vlm_time = time.perf_counter() - t0
                raw = result.text
                letter = extract_answer(raw)
                rec = {
                    "index": int(item["index"]),
                    "video_id": item["video_id"],
                    "q_uid": item["q_uid"],
                    "question": item["question"],
                    "options": list(item["options"]),
                    "condition": condition,
                    "prediction_raw": raw,
                    "prediction_letter": letter,
                    "parse_success": letter is not None,
                    "num_global_images": debug["num_global_images"],
                    "num_roi_tubes": debug["num_roi_tubes"],
                    "num_roi_frames": debug["num_roi_frames"],
                    "visual_tokens": result.num_visual_tokens,
                    "processed_pixels": int(visual.processed_pixels),
                    "input_build_time": input_build_time,
                    "vlm_time": vlm_time,
                    "total_time": input_build_time + vlm_time,
                    "tubes": debug["tubes"],
                    "global_timestamps": debug["global_timestamps"],
                    "error": None,
                }
            except Exception as exc:
                logger.exception("phase2 failed %s %s", item["index"], condition)
                rec = {
                    "index": int(item["index"]),
                    "video_id": item["video_id"],
                    "q_uid": item["q_uid"],
                    "question": item["question"],
                    "options": list(item["options"]),
                    "condition": condition,
                    "prediction_raw": "",
                    "prediction_letter": None,
                    "parse_success": False,
                    "num_global_images": debug["num_global_images"],
                    "num_roi_tubes": debug["num_roi_tubes"],
                    "num_roi_frames": debug["num_roi_frames"],
                    "visual_tokens": (stored["audit"].get("token_stats") or {}).get(
                        "num_visual_tokens"
                    ),
                    "processed_pixels": int(visual.processed_pixels),
                    "input_build_time": input_build_time,
                    "vlm_time": 0.0,
                    "total_time": input_build_time,
                    "tubes": debug["tubes"],
                    "global_timestamps": debug["global_timestamps"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(rec)
            logger.info(
                "phase2 %s %s letter=%s tokens=%s vlm=%.2fs",
                rec["index"],
                condition,
                rec["prediction_letter"],
                rec["visual_tokens"],
                rec["vlm_time"],
            )
    return {
        "model_load_count": model_load_count,
        "rows": rows,
        "existing": existing,
    }


def summarize_phase2(phase1: dict[str, Any], phase2: dict[str, Any], existing) -> dict[str, Any]:
    rows = phase2["rows"]
    ok = [row for row in rows if row.get("error") is None]
    conditions = {}
    for condition in ROI_CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        ok_c = [row for row in subset if row.get("error") is None]
        tokens = [float(row["visual_tokens"]) for row in ok_c if row.get("visual_tokens") is not None]
        letters = Counter(row.get("prediction_letter") for row in ok_c)
        packed_pairs = []
        agree = 0
        comparable = 0
        for row in ok_c:
            old = existing.get((int(row["index"]), condition))
            if old and old.get("visual_tokens") and row.get("visual_tokens"):
                packed_pairs.append((float(old["visual_tokens"]), float(row["visual_tokens"])))
            if old and old.get("prediction_letter") and row.get("prediction_letter"):
                comparable += 1
                if old["prediction_letter"] == row["prediction_letter"]:
                    agree += 1
        mean_new = _mean(tokens)
        old_mean = _mean([old for old, _new in packed_pairs])
        conditions[condition] = {
            "completed": len(ok_c),
            "n": len(subset),
            "errors": sum(1 for row in subset if row.get("error")),
            "skipped": 0,
            "parse_failures": sum(1 for row in ok_c if not row.get("parse_success")),
            "mean_global_images": _mean([row["num_global_images"] for row in ok_c]),
            "mean_roi_tubes": _mean([row["num_roi_tubes"] for row in ok_c]),
            "mean_roi_frames": _mean([row["num_roi_frames"] for row in ok_c]),
            "mean_visual_tokens": mean_new,
            "median_visual_tokens": _median(tokens),
            "mean_processed_pixels": _mean([row["processed_pixels"] for row in ok_c]),
            "mean_vlm_time": _mean([row["vlm_time"] for row in ok_c]),
            "prediction_distribution": {
                "A": letters.get("A", 0),
                "B": letters.get("B", 0),
                "C": letters.get("C", 0),
                "D": letters.get("D", 0),
            },
            "token_reduction_vs_image_packed": _mean(
                [(old - new) / old for old, new in packed_pairs if old > 0]
            ),
            "prediction_agreement_vs_image_packed": (
                agree / comparable if comparable else None
            ),
            "n_agreement_comparable": comparable,
            "token_ratio_vs_full_2fps": (mean_new / FULL_MEANS["full_2fps"]) if mean_new else None,
            "token_ratio_vs_full_1fps": (mean_new / FULL_MEANS["full_1fps"]) if mean_new else None,
            "token_ratio_vs_full_05fps": (mean_new / FULL_MEANS["full_05fps"]) if mean_new else None,
            "image_packed_mean_visual_tokens": old_mean,
        }
    return {
        "phase1_passed": bool(phase1["passed"]),
        "phase1_n_passed": phase1["n_passed"],
        "phase1_n_failed": phase1["n_failed"],
        "completed": len(ok),
        "n_total": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "skipped": 0,
        "parse_failures": sum(1 for row in ok if not row.get("parse_success")),
        "model_load_count": phase2["model_load_count"],
        "conditions": conditions,
        "output_files": {
            "phase1": str(OUT_DIR / "phase1_validation.json"),
            "predictions_json": str(OUT_DIR / "predictions.json"),
            "predictions_jsonl": str(OUT_DIR / "predictions.jsonl"),
            "summary": str(OUT_DIR / "summary.json"),
        },
        "existing_files_untouched": {
            "predictions_json": str(DEFAULT_OUTPUT / "predictions.json"),
            "predictions_jsonl": str(EXISTING_JSONL),
            "full_pilot_summary": str(EXISTING_SUMMARY),
        },
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if EXISTING_JSONL.resolve() == (OUT_DIR / "predictions.jsonl").resolve():
        raise SystemExit("refusing to write into the existing image-packed jsonl")
    seed_everything(0)
    cfg = load_config(DEFAULT_CONFIG)
    items = load_pilot_items(DEFAULT_MANIFEST)
    if len(items) != 50:
        raise SystemExit(f"expected 50 pilot items, got {len(items)}")
    existing = load_existing_rows()
    prepared = prepare_unique_videos(
        items,
        output_dir=DEFAULT_OUTPUT,
        video_root=DEFAULT_VIDEO_ROOT,
        cfg=cfg,
        jpeg_quality=95,
        skip_frames=True,
        skip_motion=False,
    )
    phase1, visual_store = run_phase1(items, prepared, cfg, existing)
    write_json(OUT_DIR / "phase1_validation.json", phase1)
    logger.info(
        "phase1 passed=%s %s/%s",
        phase1["passed"],
        phase1["n_passed"],
        phase1["n_total"],
    )
    if not phase1["passed"]:
        write_json(
            OUT_DIR / "summary.json",
            {
                "phase1_passed": False,
                "phase2_ran": False,
                "phase1_n_failed": phase1["n_failed"],
                "conditions": phase1["conditions"],
                "output_files": {
                    "phase1": str(OUT_DIR / "phase1_validation.json"),
                    "summary": str(OUT_DIR / "summary.json"),
                },
            },
        )
        print(json.dumps({"phase1_passed": False, "n_failed": phase1["n_failed"]}, indent=2))
        return 1

    phase2 = run_phase2(items, visual_store, cfg, existing)
    write_json(
        OUT_DIR / "predictions.json",
        {
            "model_load_count": phase2["model_load_count"],
            "n_records": len(phase2["rows"]),
            "rows": phase2["rows"],
        },
    )
    jsonl = OUT_DIR / "predictions.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in phase2["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_phase2(phase1, phase2, existing)
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))
    print(json.dumps(summary["conditions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
