#!/usr/bin/env python3
"""VIRAT sample30 processor/token audit: image-packed vs per-tube VIDEO.

Does not run VLM generate. Does not rewrite sample30 captioning outputs.
Reuses existing motion tracks and, for the captioning condition, existing JPEGs.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

EVAL_ROOT = Path(__file__).resolve().parents[1]
VLM_INF_SRC = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/src")
for path in (str(EVAL_ROOT), str(VLM_INF_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from transformers import AutoProcessor

from vlm_inference.config import load_config
from vlm_inference.gt_action_cls import seed_everything
from vlm_inference.gt_multilabel_cls import measured_image_resolutions
from vlm_inference.inputs import Sample
from vlm_inference.model import Qwen3VLCaptioner, count_processor_tokens
from vlm_inference.motion_adaptive import DEFAULT_TAU
from vlm_inference.roi_margin import load_tube_bboxes, tracks_json_path
from vlm_inference.sovabench import GLOBAL_DEDUP_EPS_SEC
from vlm_inference.sovabench_spatial import _motion_sample
from vlm_inference.sovabench_temporal import install_recorded_path_aliases
from vlm_inference.video_io import inspect_video, install_av_video_reader, resolve_video_file

from vantage_vqa.tube_video import (
    PackedFrame,
    TubeVideoPacking,
    bind_existing_roi_jpegs,
    image_packed_sample,
    inspect_processor_output,
    restore_tube_packing,
    same_jpeg_evidence,
    summarize_messages,
    tube_video_sample,
    validate_tube_representation,
)
from vantage_vqa.visual import (
    CONDITION_ROI_CENTROID,
    CONDITION_ROI_FIXED_1FPS,
    build_video_condition_visual,
    load_video_motion_records,
    load_video_rois,
)

logger = logging.getLogger(__name__)

CAPTION_CONDITION = "global_roi_start_only_roi_centroid_2fps"
ROI_CONDITIONS = (CONDITION_ROI_FIXED_1FPS, CONDITION_ROI_CENTROID)
SAMPLE30_CFG = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/configs/sample30_30b_start_centroid.yaml")
SAMPLE30_OUT = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/outputs/sample30_30b_start_centroid")
SAMPLE30_YAML_TRACKS = Path(
    "/home/ssandree/motion-detector/outputs/stage3/3_roi_tube/sample30_tg0.35_tm0.35"
)
PATH_ALIASES = (
    ("/home/ssandree/vlm-inference/", "/home/ssandree/Motion-guided-ROI/vlm-inference/"),
    ("/home/ssandree/motion-detector/", "/home/ssandree/Motion-guided-ROI/motion-detector/"),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remap_path(raw: str) -> str:
    text = raw
    if text.startswith("file://"):
        parsed = urlparse(text)
        local = unquote(parsed.path)
        remapped = remap_path(local)
        return Path(remapped).resolve().as_uri()
    for old, new in PATH_ALIASES:
        if text.startswith(old):
            candidate = new + text[len(old) :]
            if Path(candidate).exists():
                return candidate
    return text


def _as_uri(path: str | Path) -> str:
    raw = remap_path(str(path))
    if raw.startswith("file://"):
        return raw
    p = Path(raw)
    if p.is_file():
        return p.resolve().as_uri()
    return raw


def _basename(path: str) -> str:
    if path.startswith("file://"):
        parsed = urlparse(path)
        return Path(unquote(parsed.path)).name
    return Path(path).name


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def jpeg_pixels(paths: list[str]) -> int:
    total = 0
    for path in paths:
        local = remap_path(path)
        if local.startswith("file://"):
            local = unquote(urlparse(local).path)
        from PIL import Image

        with Image.open(local) as img:
            w, h = img.size
        total += int(w) * int(h)
    return total


def load_caption_manifest(video_id: str) -> dict[str, Any]:
    path = SAMPLE30_OUT / "vlm_inputs" / "start_only" / video_id / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for frame in payload["frames"]:
        frame["source"] = remap_path(frame["source"])
        if not Path(frame["source"]).is_file():
            copied = path.parent / "frames" / frame["filename"]
            if copied.is_file():
                frame["source"] = str(copied)
    return payload


def packing_from_caption_manifest(manifest: dict[str, Any]) -> TubeVideoPacking:
    globals_: list[PackedFrame] = []
    tubes: dict[int, list[PackedFrame]] = {}
    for frame in manifest["frames"]:
        path = _as_uri(frame["source"])
        packed = PackedFrame(
            path=path,
            timestamp=float(frame["timestamp"]),
            native_frame=int(frame.get("index") or 0),
            source="global" if frame["kind"] == "global" else f"roi:{int(frame['roi_id'])}",
        )
        if frame["kind"] == "global":
            globals_.append(packed)
        else:
            tubes.setdefault(int(frame["roi_id"]), []).append(packed)
    ordered = []
    for roi_id in sorted(tubes):
        frames = sorted(tubes[roi_id], key=lambda item: (item.timestamp, item.native_frame))
        ordered.append((roi_id, frames))
    globals_.sort(key=lambda item: (item.timestamp, item.native_frame))
    return TubeVideoPacking(global_frames=globals_, tubes=ordered)


def caption_samples(
    video_id: str,
    packing: TubeVideoPacking,
    *,
    info,
    tracks_root: Path,
    video_roots: list[Path],
) -> tuple[Sample, Sample]:
    all_paths = [frame.path for frame in packing.global_frames]
    for _roi_id, frames in packing.tubes:
        all_paths.extend(frame.path for frame in frames)
    tracks_path = tracks_json_path(tracks_root, video_id)
    tubes = load_tube_bboxes(
        tracks_path, frame_width=int(info.width), frame_height=int(info.height)
    )
    bboxes = {int(tid): tuple(int(v) for v in tube.bbox_px) for tid, tube in tubes.items()}
    image_sample = Sample(
        job_name=f"{video_id}__{CAPTION_CONDITION}",
        sample_id=video_id,
        input_type="frames",
        paths=all_paths,
        frame_width=int(info.width),
        frame_height=int(info.height),
        roi_bboxes_px=bboxes or None,
    )
    tube_sample = Sample(
        job_name=f"{video_id}__{CAPTION_CONDITION}__tube_video",
        sample_id=video_id,
        input_type="global_then_tube_videos",
        paths=all_paths,
        frame_width=int(info.width),
        frame_height=int(info.height),
        roi_bboxes_px=bboxes or None,
        global_image_paths=[frame.path for frame in packing.global_frames],
        tube_video_paths=[
            (roi_id, [frame.path for frame in frames]) for roi_id, frames in packing.tubes
        ],
    )
    return image_sample, tube_sample


def existing_modality_report(cfg) -> dict[str, Any]:
    return {
        "captioning_experiment": {
            "config": str(SAMPLE30_CFG),
            "output_dir": str(SAMPLE30_OUT),
            "condition": CAPTION_CONDITION,
            "mode": "experiment",
            "n_videos": 30,
            "full_condition_present": False,
            "qwen_message_construction": (
                "Qwen3VLCaptioner._build_messages: mode==experiment and "
                "input_type in {frames, images} -> _build_experiment_image_content. "
                "Every Global and ROI JPEG is an independent type=image item, with "
                "text labels. ROI tube identity is kept in filenames and [ROI#] text, "
                "not as Qwen video groups."
            ),
            "sparse_global": (
                "global_insert=start_only, global_insert_source=fps_1: one full-frame "
                "IMAGE at each ROI tube start, passed as type=image."
            ),
            "roi_sampled_frames": "independent IMAGE (not VIDEO)",
            "tube_identity_until_qwen_message": (
                "Yes in filenames (tubeNN_...) and interleaved text labels; "
                "No as a Qwen video object. All ROI crops remain separate images."
            ),
        },
        "sovabench_evaluation_contrast": {
            "example_config": "/home/ssandree/Motion-guided-ROI/vlm-inference/configs/sovabench_pilot121.yaml",
            "mode": "default",
            "roi_sample_constructor": "sovabench_spatial._motion_sample input_type=frames",
            "qwen_message_construction": (
                "When mode!=experiment, input_type=frames becomes a single type=video "
                "frame-list of ALL paths (Global crops mixed with every tube)."
            ),
        },
        "native_full_video_jobs": {
            "inputs.resolve_job type=video": "input_type=video -> Qwen type=video (mp4)",
            "sample30_has_full_job": False,
        },
        "token_accounting": f"{count_processor_tokens.__module__}.{count_processor_tokens.__name__}",
        "new_representation": {
            "name": "global_then_tube_videos",
            "global": "IMAGE",
            "each_roi_tube": "separate VIDEO frame-list, timestamp order preserved, tubes not mixed",
        },
        "tracks_root": {
            "sample30_yaml": str(SAMPLE30_YAML_TRACKS),
            "sample30_yaml_exists": SAMPLE30_YAML_TRACKS.is_dir(),
            "relocated": str(cfg.feasibility.tracks_root),
            "relocated_exists": Path(cfg.feasibility.tracks_root).is_dir(),
            "note": (
                "sample30_30b_start_centroid.yaml points at /home/ssandree/motion-detector/... "
                "which is absent here; identical Stage3 dump lives under Motion-guided-ROI/."
            ),
        },
        "video_pixel_budget_from_sample30": {
            "fps": cfg.video.fps,
            "min_pixels": cfg.video.min_pixels,
            "max_pixels": cfg.video.max_pixels,
            "total_pixels": cfg.video.total_pixels,
        },
    }


def odd_pad_stats(packing: TubeVideoPacking, audit: dict[str, Any]) -> dict[str, Any]:
    packs = audit.get("temporal_pack") or []
    extra_frames = 0
    odd_tubes = 0
    tps_ok = True
    details = []
    for (roi_id, frames), pack in zip(packing.tubes, packs):
        n_orig = len(frames)
        n_fetch = int(pack.get("n_frames") or 0)
        extra = max(n_fetch - n_orig, 0)
        odd = n_orig % 2 == 1
        if odd:
            odd_tubes += 1
        extra_frames += extra
        if int(pack.get("temporal_patch_size") or 0) != 2:
            tps_ok = False
        details.append(
            {
                "roi_id": roi_id,
                "n_sampled": n_orig,
                "n_fetch_video": n_fetch,
                "odd_sampled": odd,
                "padded_frames": extra,
                "temporal_patch_size": pack.get("temporal_patch_size"),
                "grid_t_equals_nframes_div_tps": pack.get("grid_t_equals_nframes_div_tps"),
            }
        )
    return {
        "n_odd_tubes": odd_tubes,
        "n_padded_frames": extra_frames,
        "temporal_patch_size_2": tps_ok and bool(packing.tubes),
        "tubes": details,
    }


def audit_pair(
    *,
    video_id: str,
    condition: str,
    packing: TubeVideoPacking,
    image_sample: Sample,
    tube_sample: Sample,
    builder,
    processor,
    prompt: str,
    duration_sec: float,
    processed_pixels: int,
) -> dict[str, Any]:
    image_messages = Qwen3VLCaptioner._build_messages(builder, image_sample, prompt)
    tube_messages = Qwen3VLCaptioner._build_messages(builder, tube_sample, prompt)
    image_audit = inspect_processor_output(processor, image_messages)
    tube_audit = inspect_processor_output(processor, tube_messages)
    failures = validate_tube_representation(
        packing=packing,
        packed_sample=image_sample,
        tube_sample=tube_sample,
        messages=tube_messages,
        audit=tube_audit,
        duration_sec=duration_sec,
    )
    image_summary = summarize_messages(image_messages)
    tube_summary = summarize_messages(tube_messages)
    evidence = same_jpeg_evidence(image_sample, tube_sample, packing)
    if not evidence["same_basenames_multiset"]:
        failures.append("jpeg_evidence_mismatch")
    image_types = [item["type"] for item in image_summary["items"] if item["type"] in {"image", "video"}]
    if condition == CAPTION_CONDITION:
        if image_summary["n_video"] != 0:
            failures.append("captioning_image_packed_has_video")
        if image_summary["n_image"] != len(image_sample.paths):
            failures.append(
                f"captioning_image_count:{image_summary['n_image']}!={len(image_sample.paths)}"
            )
    pad = odd_pad_stats(packing, tube_audit)
    image_tokens = (image_audit.get("token_stats") or {}).get("num_visual_tokens")
    tube_tokens = (tube_audit.get("token_stats") or {}).get("num_visual_tokens")
    reduction = None
    if image_tokens and tube_tokens is not None and image_tokens > 0:
        reduction = (float(image_tokens) - float(tube_tokens)) / float(image_tokens)
    row = {
        "video_id": video_id,
        "condition": condition,
        "passed": not failures,
        "failures": failures,
        "num_global_images": len(packing.global_frames),
        "num_roi_tubes": len(packing.tubes),
        "num_roi_frames": sum(len(frames) for _roi, frames in packing.tubes),
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
        "global_timestamps": [frame.timestamp for frame in packing.global_frames],
        "each_roi_frame_in_one_tube": "roi_frame_in_multiple_tubes" not in failures,
        "tube_timestamps_monotonic": not any(
            item.startswith("tube_") and item.endswith("_timestamps_not_monotonic")
            for item in failures
        ),
        "same_jpeg_evidence": evidence["same_basenames_multiset"],
        "image_packed_message": {
            "n_image": image_summary["n_image"],
            "n_video": image_summary["n_video"],
            "visual_item_types": image_types,
            "image_grid_thw": image_audit.get("image_grid_thw"),
            "video_grid_thw": image_audit.get("video_grid_thw"),
            "n_image_pad": image_audit.get("n_image_pad"),
            "n_video_pad": image_audit.get("n_video_pad"),
            "visual_tokens": image_tokens,
        },
        "tube_video_message": {
            "n_image": tube_summary["n_image"],
            "n_video": tube_summary["n_video"],
            "image_grid_thw": tube_audit.get("image_grid_thw"),
            "video_grid_thw": tube_audit.get("video_grid_thw"),
            "n_image_pad": tube_audit.get("n_image_pad"),
            "n_video_pad": tube_audit.get("n_video_pad"),
            "visual_tokens": tube_tokens,
            "temporal_pack": tube_audit.get("temporal_pack"),
        },
        "processed_pixels_jpeg": processed_pixels,
        "token_reduction_vs_image_packed": reduction,
        "odd_frame_padding": pad,
        "token_accounting": f"{count_processor_tokens.__module__}.{count_processor_tokens.__name__}",
    }
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    out = {}
    for condition, subset in by_cond.items():
        tubes = [float(row["num_roi_tubes"]) for row in subset]
        frames = [float(row["num_roi_frames"]) for row in subset]
        packed = [float(row["image_packed_message"]["visual_tokens"]) for row in subset]
        tube = [float(row["tube_video_message"]["visual_tokens"]) for row in subset]
        pixels = [float(row["processed_pixels_jpeg"]) for row in subset]
        reductions = [
            float(row["token_reduction_vs_image_packed"])
            for row in subset
            if row.get("token_reduction_vs_image_packed") is not None
        ]
        pad_frames = [float(row["odd_frame_padding"]["n_padded_frames"]) for row in subset]
        odd_tubes = [float(row["odd_frame_padding"]["n_odd_tubes"]) for row in subset]
        by_n_tubes: dict[str, list[float]] = defaultdict(list)
        for row in subset:
            if row.get("token_reduction_vs_image_packed") is not None:
                by_n_tubes[str(int(row["num_roi_tubes"]))].append(
                    float(row["token_reduction_vs_image_packed"])
                )
        out[condition] = {
            "n": len(subset),
            "n_passed": sum(1 for row in subset if row["passed"]),
            "n_failed": sum(1 for row in subset if not row["passed"]),
            "mean_roi_tubes": _mean(tubes),
            "median_roi_tubes": _median(tubes),
            "min_roi_tubes": min(tubes) if tubes else None,
            "max_roi_tubes": max(tubes) if tubes else None,
            "mean_roi_sampled_frames": _mean(frames),
            "mean_image_packed_visual_tokens": _mean(packed),
            "median_image_packed_visual_tokens": _median(packed),
            "min_image_packed_visual_tokens": min(packed) if packed else None,
            "max_image_packed_visual_tokens": max(packed) if packed else None,
            "mean_tube_video_visual_tokens": _mean(tube),
            "median_tube_video_visual_tokens": _median(tube),
            "min_tube_video_visual_tokens": min(tube) if tube else None,
            "max_tube_video_visual_tokens": max(tube) if tube else None,
            "mean_token_reduction": _mean(reductions),
            "mean_processed_pixels": _mean(pixels),
            "processed_pixels_equal_by_construction": True,
            "mean_odd_tubes": _mean(odd_tubes),
            "mean_padded_frames": _mean(pad_frames),
            "token_reduction_by_n_tubes": {
                key: {"n": len(vals), "mean_token_reduction": _mean(vals)}
                for key, vals in sorted(by_n_tubes.items(), key=lambda kv: int(kv[0]))
            },
            "failures": [
                {"video_id": row["video_id"], "failures": row["failures"]}
                for row in subset
                if row["failures"]
            ],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all 30")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    seed_everything(0)
    install_av_video_reader()
    install_recorded_path_aliases()
    cfg_path = EVAL_ROOT / "configs" / "virat_tube_video_audit.yaml"
    cfg = load_config(cfg_path)
    out_dir = EVAL_ROOT / "experiments" / "virat" / "tube_video_audit"
    cfg.output.dir = str(out_dir)
    if SAMPLE30_OUT.resolve() == out_dir.resolve():
        raise SystemExit("refusing to write into sample30 captioning outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = out_dir / "frames"
    videos = list(cfg.feasibility.videos)
    if len(videos) != 30:
        raise SystemExit(f"expected 30 videos, got {len(videos)}")
    if args.max_videos:
        videos = videos[: int(args.max_videos)]
    tracks_root = Path(cfg.feasibility.tracks_root)
    video_roots = [Path(p) for p in cfg.feasibility.datasets.all_video_roots()]
    jpeg_quality = int(cfg.feasibility.jpeg_quality)
    prompt = Path(cfg.prompt_file).read_text(encoding="utf-8")

    modality = existing_modality_report(cfg)
    write_json(out_dir / "existing_modality.json", modality)

    logger.info("Loading AutoProcessor only from %s", cfg.model.path)
    processor = AutoProcessor.from_pretrained(cfg.model.path)
    builder = object.__new__(Qwen3VLCaptioner)
    builder.cfg = cfg

    rows: list[dict[str, Any]] = []
    decoded_by_video: dict[str, dict[int, Any]] = defaultdict(dict)

    for video_id in videos:
        video_path = resolve_video_file(video_id, video_roots)
        info = inspect_video(video_path)
        duration = float(max(int(info.frame_count) - 1, 0)) / (
            float(info.native_fps) if info.native_fps > 0 else 30.0
        )
        logger.info("captioning condition %s", video_id)
        manifest = load_caption_manifest(video_id)
        packing = packing_from_caption_manifest(manifest)
        image_sample, tube_sample = caption_samples(
            video_id,
            packing,
            info=info,
            tracks_root=tracks_root,
            video_roots=video_roots,
        )
        pixels = jpeg_pixels(
            [frame.path for frame in packing.global_frames]
            + [frame.path for _roi, frames in packing.tubes for frame in frames]
        )
        try:
            rows.append(
                audit_pair(
                    video_id=video_id,
                    condition=CAPTION_CONDITION,
                    packing=packing,
                    image_sample=image_sample,
                    tube_sample=tube_sample,
                    builder=builder,
                    processor=processor,
                    prompt=prompt,
                    duration_sec=duration,
                    processed_pixels=pixels,
                )
            )
        except Exception as exc:
            logger.exception("caption audit failed %s", video_id)
            rows.append(
                {
                    "video_id": video_id,
                    "condition": CAPTION_CONDITION,
                    "passed": False,
                    "failures": [f"exception:{type(exc).__name__}:{exc}"],
                    "traceback": traceback.format_exc(),
                }
            )

        rois = load_video_rois(video_id, tracks_root=tracks_root, video_roots=video_roots)
        records = load_video_motion_records(
            video_id,
            tracks_root=tracks_root,
            video_roots=video_roots,
            motion_cfg=cfg.experiment.centroid_motion,
        )
        for condition in ROI_CONDITIONS:
            logger.info("temporal condition %s / %s", video_id, condition)
            try:
                visual = build_video_condition_visual(
                    video_id=video_id,
                    video_path=video_path,
                    condition=condition,
                    info=info,
                    rois=rois,
                    records=records,
                    video_roots=video_roots,
                    frames_root=frames_root,
                    jpeg_quality=jpeg_quality,
                    decoded_frames=decoded_by_video[video_id],
                    tau=DEFAULT_TAU,
                    dedup_eps=float(
                        cfg.experiment.global_insert_dedup_eps or GLOBAL_DEDUP_EPS_SEC
                    ),
                    skip_frames=False,
                )
                visual = bind_existing_roi_jpegs(visual, frames_root)
                packing = restore_tube_packing(visual)
                image_sample = image_packed_sample(visual)
                # Force experiment-mode independent IMAGE packing (VIRAT captioning).
                image_sample.input_type = "frames"
                tube_sample = tube_video_sample(visual, packing)
                pixels = int(visual.processed_pixels) or jpeg_pixels(list(visual.frame_paths))
                rows.append(
                    audit_pair(
                        video_id=video_id,
                        condition=condition,
                        packing=packing,
                        image_sample=image_sample,
                        tube_sample=tube_sample,
                        builder=builder,
                        processor=processor,
                        prompt=prompt,
                        duration_sec=duration,
                        processed_pixels=pixels,
                    )
                )
            except Exception as exc:
                logger.exception("temporal audit failed %s %s", video_id, condition)
                rows.append(
                    {
                        "video_id": video_id,
                        "condition": condition,
                        "passed": False,
                        "failures": [f"exception:{type(exc).__name__}:{exc}"],
                        "traceback": traceback.format_exc(),
                    }
                )

    summary = {
        "n_videos": len(videos),
        "n_rows": len(rows),
        "n_passed": sum(1 for row in rows if row.get("passed")),
        "n_failed": sum(1 for row in rows if not row.get("passed")),
        "passed": all(row.get("passed") for row in rows)
        and len(rows) == len(videos) * (1 + len(ROI_CONDITIONS)),
        "generation_ran": False,
        "existing_captioning_outputs_untouched": str(SAMPLE30_OUT),
        "token_accounting": f"{count_processor_tokens.__module__}.{count_processor_tokens.__name__}",
        "conditions": aggregate(rows),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "rows.json", {"n_records": len(rows), "rows": rows})
    jsonl = out_dir / "rows.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))
    print(json.dumps(summary["conditions"], indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
