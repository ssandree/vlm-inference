"""VANTAGE VQA pilot runner: reuse existing motion/ROI/Qwen implementations.

Does not modify research algorithms. Loads Qwen at most once per process.
Caches visual inputs by (video_id, condition). Motion is computed once per
unique video via the existing motion-detector pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from vlm_inference.config import AppConfig, load_config
from vlm_inference.gt_action_cls import seed_everything
from vlm_inference.model import Qwen3VLCaptioner, count_processor_tokens
from vlm_inference.motion_adaptive import DEFAULT_TAU
from vlm_inference.sovabench import GLOBAL_DEDUP_EPS_SEC
from vlm_inference.sovabench_spatial import _motion_sample, _plain_sample
from vlm_inference.video_io import inspect_video

from .prompt import extract_answer, format_vantage_vqa_prompt
from .visual import (
    ROI_CONDITIONS,
    USER_CONDITIONS,
    VisualInput,
    assert_visual_builder_isolated,
    build_video_condition_visual,
    is_full_condition,
    load_video_motion_records,
    load_video_rois,
    visual_builder_forbidden_params,
)

logger = logging.getLogger(__name__)

EVAL_ROOT = Path(__file__).resolve().parents[1]
MOTION_WORKER = EVAL_ROOT / "scripts" / "compute_vantage_motion.py"
DEFAULT_MANIFEST = EVAL_ROOT / "experiments" / "vantage" / "vantage_vqa_pilot50.json"
DEFAULT_CONFIG = EVAL_ROOT / "configs" / "vantage_vqa_pilot50.yaml"
DEFAULT_OUTPUT = EVAL_ROOT / "experiments" / "vantage" / "vqa_pilot50"
DEFAULT_VIDEO_ROOT = Path("/data/datasets/VANTAGE-Bench/data/vqa/videos")
MOTION_PYTHON = Path("/home/ssandree/miniconda3/bin/python3")
OPENCV_CUDA_PREFIX = Path.home() / ".local" / "opencv-cuda"
MOTION_SRC = Path("/home/ssandree/Motion-guided-ROI/motion-detector/src")
VLM_INF_SRC = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/src")
TOKEN_ACCOUNTING = f"{count_processor_tokens.__module__}.{count_processor_tokens.__name__}"


def load_pilot_items(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"manifest has no items: {path}")
    return items


def video_duration_sec(info) -> float:
    fps = float(info.native_fps) if info.native_fps > 0 else 30.0
    return float(max(int(info.frame_count) - 1, 0)) / fps


def motion_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    prefix = OPENCV_CUDA_PREFIX
    site = prefix / "lib" / "python" / "site-packages"
    lib = prefix / "lib"
    py_path = [str(MOTION_SRC), str(site)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_path.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_path)
    env["OPENCV_CUDA_PREFIX"] = str(prefix)
    ld = [str(lib), "/home/ssandree/miniconda3/lib", "/usr/local/cuda/lib64"]
    if env.get("LD_LIBRARY_PATH"):
        ld.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld)
    return env


def ensure_video_motion(
    video_id: str,
    *,
    video_root: Path,
    cache_root: Path,
    python_bin: Path = MOTION_PYTHON,
) -> dict[str, Any]:
    tracks = cache_root / "tracks" / f"{video_id}_roi_tracks.json"
    marker = cache_root / "videos" / video_id / "motion_meta.json"
    if tracks.is_file() and marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["status"] = "cache_hit"
        payload["elapsed_sec"] = 0.0
        return payload
    cmd = [
        str(python_bin),
        str(MOTION_WORKER),
        "--video-id",
        video_id,
        "--video-root",
        str(video_root),
        "--cache-root",
        str(cache_root),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=motion_subprocess_env(),
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"motion failed for {video_id} (exit {proc.returncode}):\n"
            f"{proc.stderr[-4000:]}\n{proc.stdout[-2000:]}"
        )
    line = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(line)
    payload["elapsed_sec"] = float(payload.get("elapsed_sec") or elapsed)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


class VisualCache:
    """Cache visual representations by (video_id, condition)."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], VisualInput] = {}

    def get(self, video_id: str, condition: str) -> VisualInput | None:
        return self._store.get((video_id, condition))

    def put(self, visual: VisualInput) -> VisualInput:
        key = (visual.video_id, visual.condition)
        existing = self._store.get(key)
        if existing is not None:
            cloned = VisualInput(**{**existing.__dict__, "cache_hit": True, "input_build_time": 0.0})
            return cloned
        self._store[key] = visual
        return visual

    def require(self, video_id: str, condition: str) -> VisualInput:
        visual = self.get(video_id, condition)
        if visual is None:
            raise KeyError(f"missing visual cache for {(video_id, condition)}")
        return visual

    def stored_items(self) -> list[VisualInput]:
        return list(self._store.values())


def prepare_unique_videos(
    items: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    video_root: Path,
    cfg: AppConfig,
    jpeg_quality: int = 95,
    skip_frames: bool = False,
    skip_motion: bool = False,
) -> dict[str, Any]:
    assert_visual_builder_isolated()
    unique_ids = sorted({str(row["video_id"]) for row in items})
    frames_root = output_dir / "frames"
    motion_root = output_dir / "motion_cache"
    tracks_root = motion_root / "tracks"
    video_roots = [video_root]
    infos = {}
    motion_meta: dict[str, Any] = {}
    rois_by_video: dict[str, list] = {}
    records_by_video: dict[str, dict] = {}
    decoded_by_video: dict[str, dict[int, Any]] = defaultdict(dict)
    visual_cache = VisualCache()

    for video_id in unique_ids:
        path = video_root / f"{video_id}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        infos[video_id] = inspect_video(path)

    if not skip_motion:
        for video_id in unique_ids:
            logger.info("Motion analysis %s", video_id)
            motion_meta[video_id] = ensure_video_motion(
                video_id, video_root=video_root, cache_root=motion_root
            )
            rois_by_video[video_id] = load_video_rois(
                video_id, tracks_root=tracks_root, video_roots=video_roots
            )
            records_by_video[video_id] = load_video_motion_records(
                video_id,
                tracks_root=tracks_root,
                video_roots=video_roots,
                motion_cfg=cfg.experiment.centroid_motion,
            )
    else:
        for video_id in unique_ids:
            motion_meta[video_id] = {"status": "skipped", "elapsed_sec": 0.0, "n_tracks": 0}
            rois_by_video[video_id] = []
            records_by_video[video_id] = {}

    for video_id in unique_ids:
        for condition in USER_CONDITIONS:
            if skip_motion and condition in ROI_CONDITIONS:
                continue
            logger.info("Build visual %s / %s", video_id, condition)
            visual = build_video_condition_visual(
                video_id=video_id,
                video_path=video_root / f"{video_id}.mp4",
                condition=condition,
                info=infos[video_id],
                rois=rois_by_video.get(video_id) or [],
                records=records_by_video.get(video_id) or {},
                video_roots=video_roots,
                frames_root=frames_root,
                jpeg_quality=jpeg_quality,
                decoded_frames=decoded_by_video[video_id],
                tau=DEFAULT_TAU,
                dedup_eps=float(cfg.experiment.global_insert_dedup_eps or GLOBAL_DEDUP_EPS_SEC),
                skip_frames=skip_frames,
            )
            visual_cache.put(visual)

    return {
        "unique_video_ids": unique_ids,
        "infos": infos,
        "motion_meta": motion_meta,
        "rois_by_video": {vid: len(rois) for vid, rois in rois_by_video.items()},
        "visual_cache": visual_cache,
        "tracks_root": tracks_root,
        "frames_root": frames_root,
    }


def _stamp_ok(timestamps: Sequence[float], duration: float) -> bool:
    if not timestamps:
        return True
    lo = min(timestamps)
    hi = max(timestamps)
    return lo >= -1e-6 and hi <= float(duration) + 1e-3


def dry_run_validate(
    items: Sequence[dict[str, Any]],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    cache: VisualCache = prepared["visual_cache"]
    infos = prepared["infos"]
    indices = [int(row["index"]) for row in items]
    video_ids = [str(row["video_id"]) for row in items]
    if len(items) != 50:
        issues.append(f"expected 50 QA, got {len(items)}")
    if len(set(indices)) != len(indices):
        issues.append("duplicate QA index")
    if len(set(video_ids)) != 47:
        issues.append(f"expected 47 unique videos, got {len(set(video_ids))}")
    for row in items:
        if not Path(row["video_path"]).is_file():
            issues.append(f"missing video {row['video_path']}")
        if len(row.get("options") or []) != 4:
            issues.append(f"options!=4 index={row.get('index')}")

    per_condition: dict[str, Any] = {}
    reuse_ok = True
    for condition in USER_CONDITIONS:
        visuals = [cache.get(str(row["video_id"]), condition) for row in items]
        if any(v is None for v in visuals):
            issues.append(f"{condition}: missing visual for some QA")
            continue
        if len(visuals) != 50:
            issues.append(f"{condition}: expected 50 visuals, got {len(visuals)}")
        ids = [int(row["index"]) for row in items]
        if ids != indices:
            issues.append(f"{condition}: QA index set mismatch")
        empty = [int(row["index"]) for row, vis in zip(items, visuals) if vis.num_input_images <= 0]
        if empty:
            issues.append(f"{condition}: empty visual input at indices {empty[:8]}")
        oob = []
        for row, vis in zip(items, visuals):
            duration = video_duration_sec(infos[str(row["video_id"])])
            if not _stamp_ok(vis.sampled_timestamps, duration):
                oob.append(int(row["index"]))
        if oob:
            issues.append(f"{condition}: timestamps outside duration at {oob[:8]}")
        by_video: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row, vis in zip(items, visuals):
            by_video[str(row["video_id"])].append(vis.identity)
        reused = 0
        for vid, idents in by_video.items():
            if len(idents) > 1:
                reused += len(idents) - 1
                if len(set(idents)) != 1:
                    reuse_ok = False
                    issues.append(f"{condition}: {vid} reused QA have different visuals")
        per_condition[condition] = {
            "n_qa": len(visuals),
            "n_unique_videos": len({str(row["video_id"]) for row in items}),
            "mean_images": sum(v.num_input_images for v in visuals) / max(len(visuals), 1),
            "n_empty": len(empty) if visuals else 50,
            "repeated_video_qa": reused,
            "non_empty": all(v.num_input_images > 0 for v in visuals),
        }

    leaked = list(visual_builder_forbidden_params())
    isolation_ok = True
    try:
        assert_visual_builder_isolated()
    except AssertionError as exc:
        isolation_ok = False
        issues.append(str(exc))

    visual_inventory = []
    for vis in cache.stored_items():
        visual_inventory.append(
            {
                "video_id": vis.video_id,
                "condition": vis.condition,
                "num_input_images": vis.num_input_images,
                "processed_pixels": vis.processed_pixels,
                "input_build_time": vis.input_build_time,
                "n_rois": vis.n_rois,
                "n_timestamps": len(vis.sampled_timestamps),
            }
        )

    report = {
        "passed": not issues,
        "issues": issues,
        "n_qa": len(items),
        "n_unique_videos": len(set(video_ids)),
        "qa_indices": indices,
        "identical_qa_index_set": True,
        "options_len_4": all(len(row.get("options") or []) == 4 for row in items),
        "all_videos_exist": all(Path(row["video_path"]).is_file() for row in items),
        "visual_builder_isolated": isolation_ok,
        "visual_builder_forbidden_params": leaked,
        "token_accounting": TOKEN_ACCOUNTING,
        "repeated_video_reuse_identical": reuse_ok,
        "motion_meta": prepared["motion_meta"],
        "n_rois_by_video": prepared["rois_by_video"],
        "conditions": per_condition,
        "visual_inventory": visual_inventory,
    }
    return report


def prediction_record(
    item: dict[str, Any],
    *,
    condition: str,
    visual: VisualInput,
    prediction_raw: str,
    prediction_letter: str | None,
    parse_success: bool,
    visual_tokens: int | None,
    motion_time: float,
    input_build_time: float,
    vlm_time: float,
    cache_hit_visual: bool,
    cache_hit_motion: bool,
    model_load_count: int,
) -> dict[str, Any]:
    return {
        "index": int(item["index"]),
        "video_id": item["video_id"],
        "q_uid": item["q_uid"],
        "question": item["question"],
        "options": list(item["options"]),
        "pool200_rank": item.get("pool200_rank"),
        "condition": condition,
        "prediction_raw": prediction_raw,
        "prediction_letter": prediction_letter,
        "parse_success": bool(parse_success),
        "num_input_images": int(visual.num_input_images),
        "visual_tokens": visual_tokens,
        "processed_pixels": int(visual.processed_pixels),
        "motion_time": float(motion_time),
        "input_build_time": float(input_build_time),
        "vlm_time": float(vlm_time),
        "total_time": float(motion_time) + float(input_build_time) + float(vlm_time),
        "cache_hit_visual": bool(cache_hit_visual),
        "cache_hit_motion": bool(cache_hit_motion),
        "sampled_timestamps": list(visual.sampled_timestamps),
        "global_timestamps": list(visual.global_timestamps),
        "roi_timestamps": {str(k): v for k, v in visual.roi_timestamps.items()},
        "source_types": list(visual.source_types),
        "n_rois": int(visual.n_rois),
        "token_accounting": TOKEN_ACCOUNTING,
        "model_load_count": int(model_load_count),
        "visual_identity_sha1": hashlib.sha1(
            json.dumps(visual.identity, default=str).encode("utf-8")
        ).hexdigest()[:16],
    }


def run_qwen_on_visual(
    captioner: Qwen3VLCaptioner,
    cfg: AppConfig,
    item: dict[str, Any],
    visual: VisualInput,
    condition: str,
) -> tuple[str, int | None, float]:
    prompt = format_vantage_vqa_prompt(item["question"], item["options"])
    original_mode = cfg.mode
    try:
        if is_full_condition(condition):
            cfg.mode = "default"
            sample = _plain_sample(visual.job)
        else:
            cfg.mode = "experiment"
            sample = _motion_sample(visual.job)
        t0 = time.perf_counter()
        result = captioner.generate(sample, prompt)
        vlm_time = time.perf_counter() - t0
    finally:
        cfg.mode = original_mode
    return result.text, result.num_visual_tokens, vlm_time


def smoke_test_first_qa(
    items: Sequence[dict[str, Any]],
    prepared: dict[str, Any],
    cfg: AppConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    item = items[0]
    cache: VisualCache = prepared["visual_cache"]
    motion_meta = prepared["motion_meta"]
    logger.info("Loading Qwen3-VL once from %s", cfg.model.path)
    captioner = Qwen3VLCaptioner(cfg)
    model_load_count = 1
    rows: list[dict[str, Any]] = []
    for condition in USER_CONDITIONS:
        visual = cache.require(str(item["video_id"]), condition)
        motion_payload = motion_meta.get(item["video_id"]) or {}
        shared_motion = float(motion_payload.get("elapsed_sec") or 0.0)
        cache_hit_motion = str(motion_payload.get("status")) == "cache_hit"
        # Charge shared motion once per video, on the first condition only.
        motion_time = 0.0 if (cache_hit_motion or condition != USER_CONDITIONS[0]) else shared_motion
        raw, tokens, vlm_time = run_qwen_on_visual(captioner, cfg, item, visual, condition)
        letter = extract_answer(raw)
        rows.append(
            prediction_record(
                item,
                condition=condition,
                visual=visual,
                prediction_raw=raw,
                prediction_letter=letter,
                parse_success=letter is not None,
                visual_tokens=tokens,
                motion_time=motion_time,
                input_build_time=float(visual.input_build_time),
                vlm_time=vlm_time,
                cache_hit_visual=True,
                cache_hit_motion=cache_hit_motion or condition != USER_CONDITIONS[0],
                model_load_count=model_load_count,
            )
        )
        logger.info(
            "smoke %s letter=%s images=%s tokens=%s pixels=%s vlm=%.2fs",
            condition,
            letter,
            visual.num_input_images,
            tokens,
            visual.processed_pixels,
            vlm_time,
        )
    payload = {
        "qa": {
            "index": item["index"],
            "video_id": item["video_id"],
            "q_uid": item["q_uid"],
            "question": item["question"],
            "options": item["options"],
        },
        "model_load_count": model_load_count,
        "token_accounting": TOKEN_ACCOUNTING,
        "rows": rows,
    }
    path = output_dir / "smoke_test_first_qa.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def error_record(
    item: dict[str, Any],
    *,
    condition: str,
    visual: VisualInput | None,
    error: BaseException | str,
    motion_time: float,
    input_build_time: float,
    cache_hit_visual: bool,
    cache_hit_motion: bool,
    model_load_count: int,
) -> dict[str, Any]:
    rec = {
        "index": int(item["index"]),
        "video_id": item["video_id"],
        "q_uid": item["q_uid"],
        "question": item["question"],
        "options": list(item["options"]),
        "pool200_rank": item.get("pool200_rank"),
        "condition": condition,
        "prediction_raw": "",
        "prediction_letter": None,
        "parse_success": False,
        "num_input_images": int(visual.num_input_images) if visual is not None else 0,
        "visual_tokens": None,
        "processed_pixels": int(visual.processed_pixels) if visual is not None else 0,
        "motion_time": float(motion_time),
        "input_build_time": float(input_build_time),
        "vlm_time": 0.0,
        "total_time": float(motion_time) + float(input_build_time),
        "cache_hit_visual": bool(cache_hit_visual),
        "cache_hit_motion": bool(cache_hit_motion),
        "sampled_timestamps": list(visual.sampled_timestamps) if visual is not None else [],
        "global_timestamps": list(visual.global_timestamps) if visual is not None else [],
        "roi_timestamps": (
            {str(k): v for k, v in visual.roi_timestamps.items()} if visual is not None else {}
        ),
        "source_types": list(visual.source_types) if visual is not None else [],
        "n_rois": int(visual.n_rois) if visual is not None else 0,
        "token_accounting": TOKEN_ACCOUNTING,
        "model_load_count": int(model_load_count),
        "visual_identity_sha1": None,
        "error": str(error),
    }
    if visual is not None:
        rec["visual_identity_sha1"] = hashlib.sha1(
            json.dumps(visual.identity, default=str).encode("utf-8")
        ).hexdigest()[:16]
    return rec


def _mean(values: Sequence[float | int | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def summarize_full_results(
    rows: Sequence[dict[str, Any]],
    *,
    model_load_count: int,
) -> dict[str, Any]:
    n_total = len(USER_CONDITIONS) * 50
    n_error = sum(1 for row in rows if row.get("error"))
    n_ok = sum(1 for row in rows if not row.get("error"))
    n_parse_fail = sum(
        1 for row in rows if not row.get("error") and not row.get("parse_success")
    )
    per_condition: dict[str, Any] = {}
    for condition in USER_CONDITIONS:
        subset = [row for row in rows if row.get("condition") == condition]
        ok = [row for row in subset if not row.get("error")]
        letters = [
            str(row.get("prediction_letter"))
            for row in ok
            if row.get("parse_success") and row.get("prediction_letter") in {"A", "B", "C", "D"}
        ]
        dist = {letter: 0 for letter in ("A", "B", "C", "D")}
        dist.update(Counter(letters))
        per_condition[condition] = {
            "completed": len(ok),
            "n": 50,
            "errors": sum(1 for row in subset if row.get("error")),
            "parse_failures": sum(1 for row in ok if not row.get("parse_success")),
            "mean_images": _mean([row.get("num_input_images") for row in ok]),
            "mean_visual_tokens": _mean([row.get("visual_tokens") for row in ok]),
            "mean_processed_pixels": _mean([row.get("processed_pixels") for row in ok]),
            "mean_vlm_time": _mean([row.get("vlm_time") for row in ok]),
            "prediction_distribution": dist,
        }
    return {
        "completed": n_ok,
        "n_total": n_total,
        "n_records": len(rows),
        "errors": n_error,
        "skipped": 0,
        "parse_failures": n_parse_fail,
        "model_load_count": int(model_load_count),
        "token_accounting": TOKEN_ACCOUNTING,
        "conditions": per_condition,
    }


def run_full_pilot(
    items: Sequence[dict[str, Any]],
    prepared: dict[str, Any],
    cfg: AppConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    cache: VisualCache = prepared["visual_cache"]
    motion_meta = prepared["motion_meta"]
    logger.info("Loading Qwen3-VL once from %s", cfg.model.path)
    captioner = Qwen3VLCaptioner(cfg)
    model_load_count = 1

    jsonl_path = output_dir / "predictions.jsonl"
    json_path = output_dir / "predictions.json"
    summary_path = output_dir / "full_pilot_summary.json"
    jsonl_path.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    seen_visual: set[tuple[str, str]] = set()
    charged_motion: set[str] = set()
    n_expected = len(items) * len(USER_CONDITIONS)
    n_done = 0

    for item in items:
        video_id = str(item["video_id"])
        motion_payload = motion_meta.get(video_id) or {}
        motion_was_cached = str(motion_payload.get("status")) == "cache_hit"
        for condition in USER_CONDITIONS:
            n_done += 1
            visual: VisualInput | None = None
            cache_hit_visual = False
            cache_hit_motion = motion_was_cached or video_id in charged_motion
            motion_time = 0.0
            input_build_time = 0.0
            try:
                visual = cache.require(video_id, condition)
                vis_key = (video_id, condition)
                cache_hit_visual = vis_key in seen_visual
                if cache_hit_visual:
                    input_build_time = 0.0
                else:
                    input_build_time = float(visual.input_build_time)
                    seen_visual.add(vis_key)
                if cache_hit_motion:
                    motion_time = 0.0
                else:
                    motion_time = float(motion_payload.get("elapsed_sec") or 0.0)
                    charged_motion.add(video_id)
                raw, tokens, vlm_time = run_qwen_on_visual(
                    captioner, cfg, item, visual, condition
                )
                letter = extract_answer(raw)
                rec = prediction_record(
                    item,
                    condition=condition,
                    visual=visual,
                    prediction_raw=raw,
                    prediction_letter=letter,
                    parse_success=letter is not None,
                    visual_tokens=tokens,
                    motion_time=motion_time,
                    input_build_time=input_build_time,
                    vlm_time=vlm_time,
                    cache_hit_visual=cache_hit_visual,
                    cache_hit_motion=cache_hit_motion,
                    model_load_count=model_load_count,
                )
            except Exception as exc:
                logger.exception(
                    "full %s/%s failed index=%s condition=%s",
                    n_done,
                    n_expected,
                    item.get("index"),
                    condition,
                )
                rec = error_record(
                    item,
                    condition=condition,
                    visual=visual,
                    error=f"{exc}\n{traceback.format_exc()}",
                    motion_time=motion_time,
                    input_build_time=input_build_time,
                    cache_hit_visual=cache_hit_visual,
                    cache_hit_motion=cache_hit_motion,
                    model_load_count=model_load_count,
                )
            rows.append(rec)
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 5 == 0 or n_done == n_expected:
                write_json(json_path, {"model_load_count": model_load_count, "rows": rows})
                write_json(
                    summary_path,
                    summarize_full_results(rows, model_load_count=model_load_count),
                )
            logger.info(
                "full %s/%s index=%s condition=%s letter=%s images=%s tokens=%s vlm=%.2fs%s",
                n_done,
                n_expected,
                rec.get("index"),
                condition,
                rec.get("prediction_letter"),
                rec.get("num_input_images"),
                rec.get("visual_tokens"),
                float(rec.get("vlm_time") or 0.0),
                " ERROR" if rec.get("error") else "",
            )

    payload = {
        "model_load_count": model_load_count,
        "token_accounting": TOKEN_ACCOUNTING,
        "n_records": len(rows),
        "rows": rows,
    }
    write_json(json_path, payload)
    summary = summarize_full_results(rows, model_load_count=model_load_count)
    summary["predictions_json"] = str(json_path)
    summary["predictions_jsonl"] = str(jsonl_path)
    write_json(summary_path, summary)
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
