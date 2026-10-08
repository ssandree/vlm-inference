#!/usr/bin/env python3
"""Prototype: roi_fixed_1fps Global IMAGE + per-tube VIDEO, index 636 only.

Reuses existing motion/ROI/JPEG cache. Does not run the 50x5 pilot, does not
change centroid, and does not overwrite predictions.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
VLM_INF_SRC = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/src")
for path in (str(EVAL_ROOT), str(VLM_INF_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vlm_inference.config import apply_output_dir_override, load_config
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
)
from vantage_vqa.tube_video import (
    _basename,
    bind_existing_roi_jpegs,
    evidence_rows,
    image_packed_sample,
    inspect_processor,
    restore_tube_packing,
    same_jpeg_evidence,
    summarize_messages,
    tube_video_sample,
)
from vantage_vqa.visual import CONDITION_ROI_FIXED_1FPS

TARGET_INDEX = 636
CONDITION = CONDITION_ROI_FIXED_1FPS


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    seed_everything(0)
    cfg = load_config(DEFAULT_CONFIG)
    apply_output_dir_override(cfg, DEFAULT_OUTPUT)
    items = load_pilot_items(DEFAULT_MANIFEST)
    item = next((row for row in items if int(row["index"]) == TARGET_INDEX), None)
    if item is None:
        raise SystemExit(f"index {TARGET_INDEX} not in manifest")
    if str(item["video_id"]) != "concat_wh_52_910_0":
        raise SystemExit(f"unexpected video_id for {TARGET_INDEX}: {item['video_id']}")

    prepared = prepare_unique_videos(
        [item],
        output_dir=DEFAULT_OUTPUT,
        video_root=DEFAULT_VIDEO_ROOT,
        cfg=cfg,
        jpeg_quality=95,
        skip_frames=True,
        skip_motion=False,
    )
    visual = prepared["visual_cache"].require(str(item["video_id"]), CONDITION)
    visual = bind_existing_roi_jpegs(visual, DEFAULT_OUTPUT / "frames")
    packing = restore_tube_packing(visual)
    packed_sample = image_packed_sample(visual)
    tube_sample = tube_video_sample(visual, packing)
    evidence = same_jpeg_evidence(packed_sample, tube_sample, packing)
    if not evidence["same_basenames_multiset"] or not evidence["same_order_as_job"]:
        raise SystemExit(f"ROI evidence mismatch: {evidence}")

    logging.info("Loading Qwen3-VL once from %s", cfg.model.path)
    captioner = Qwen3VLCaptioner(cfg)
    prompt = format_vantage_vqa_prompt(item["question"], item["options"])

    original_mode = cfg.mode
    cfg.mode = "experiment"
    packed_messages = captioner._build_messages(packed_sample, prompt)
    packed_audit = inspect_processor(captioner, packed_messages)
    t0 = time.perf_counter()
    packed_result = captioner.generate(packed_sample, prompt)
    packed_vlm = time.perf_counter() - t0
    cfg.mode = original_mode

    tube_messages = captioner._build_messages(tube_sample, prompt)
    tube_audit = inspect_processor(captioner, tube_messages)
    t1 = time.perf_counter()
    tube_result = captioner.generate(tube_sample, prompt)
    tube_vlm = time.perf_counter() - t1

    report = {
        "index": TARGET_INDEX,
        "video_id": item["video_id"],
        "condition": CONDITION,
        "model_load_count": 1,
        "question": item["question"],
        "options": item["options"],
        "n_global_images": len(packing.global_frames),
        "n_roi_tubes": len(packing.tubes),
        "tubes": [
            {
                "roi_id": roi_id,
                "n_frames": len(frames),
                "timestamps": [frame.timestamp for frame in frames],
                "native_frames": [frame.native_frame for frame in frames],
                "basenames": [_basename(frame.path) for frame in frames],
            }
            for roi_id, frames in packing.tubes
        ],
        "evidence_rows": evidence_rows(packing),
        "same_jpeg_evidence": evidence,
        "processed_pixels": int(visual.processed_pixels),
        "input_image_resolutions": visual.input_image_resolutions,
        "image_packed": {
            "packing": "independent_images",
            "message": summarize_messages(packed_messages),
            "processor": packed_audit,
            "num_input_images": int(visual.num_input_images),
            "visual_tokens": packed_result.num_visual_tokens,
            "processed_pixels": int(visual.processed_pixels),
            "prediction_raw": packed_result.text,
            "prediction_letter": extract_answer(packed_result.text),
            "vlm_time": packed_vlm,
        },
        "tube_video": {
            "packing": "global_image_plus_per_tube_video",
            "message": summarize_messages(tube_messages),
            "processor": tube_audit,
            "num_input_images": int(visual.num_input_images),
            "visual_tokens": tube_result.num_visual_tokens,
            "processed_pixels": int(visual.processed_pixels),
            "prediction_raw": tube_result.text,
            "prediction_letter": extract_answer(tube_result.text),
            "vlm_time": tube_vlm,
        },
        "baseline_roi_fixed_636": {
            "num_input_images": 10,
            "visual_tokens": 2412,
            "prediction_letter": "B",
        },
    }
    out = DEFAULT_OUTPUT / "roi_tube_video_proto636.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "wrote": str(out),
            "n_global": report["n_global_images"],
            "n_tubes": report["n_roi_tubes"],
            "same_jpeg": evidence,
            "image_packed": {
                "n_image": report["image_packed"]["message"]["n_image"],
                "n_video": report["image_packed"]["message"]["n_video"],
                "tokens": packed_result.num_visual_tokens,
                "letter": report["image_packed"]["prediction_letter"],
                "vlm_time": packed_vlm,
            },
            "tube_video": {
                "n_image": report["tube_video"]["message"]["n_image"],
                "n_video": report["tube_video"]["message"]["n_video"],
                "tokens": tube_result.num_visual_tokens,
                "letter": report["tube_video"]["prediction_letter"],
                "vlm_time": tube_vlm,
                "video_grid_thw": tube_audit.get("video_grid_thw"),
                "temporal_pack": tube_audit.get("temporal_pack"),
            },
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
