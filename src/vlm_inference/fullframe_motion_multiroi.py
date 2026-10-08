"""Full-frame vs Motion Multi-ROI feasibility on selected VIRAT videos.

Reads existing Stage3 ROI tubes and source videos. Does not rewrite Stage1–3
outputs, rankings, ROI-tube generation, or previous experiment directories.

Full-frame: uniform 1 fps frames from the source video, packed as one video.
Motion Multi-ROI: start-only sparse Global images plus one 1 fps video per
existing ROI tube (``Sample.input_type = global_then_tube_videos``).
"""

from __future__ import annotations

import csv
import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from .config import (
    AppConfig,
    GenerationConfig,
    ImageConfig,
    OutputConfig,
    VideoConfig,
    load_model_catalog,
    resolve_model_spec,
)
from .global_resolution import vlm_smart_resize
from .inputs import TubeGlobalRoiBlock, TubeInputBlock
from .model import build_tube_global_anchor_content
from .inputs import Sample
from .motion_adaptive import (
    _events_by_id,
    load_timeline,
    load_tracks_payload,
    select_fixed_or_adaptive,
    tube_motion_series,
)
from .prompts import _read_text, resolve_path
from .roi_margin import _crop_rgb, _save_jpeg_rgb, load_tube_bboxes, tracks_json_path
from .video_io import extract_video_frames, inspect_video

logger = logging.getLogger(__name__)

CONDITION_FULL = "full_frame"
CONDITION_FULL_2FPS = "full_frame_2fps"
CONDITION_MOTION = "motion_multi_roi"
CONDITION_MOTION_ROI = "motion_roi"
CONDITION_ANCHOR = "motion_multi_roi_start_anchor"
CONDITION_PEAK = "motion_multi_roi_peak_anchor"
CONDITIONS = (CONDITION_FULL, CONDITION_MOTION)
ALL_CONDITIONS = (
    CONDITION_FULL,
    CONDITION_FULL_2FPS,
    CONDITION_MOTION,
    CONDITION_MOTION_ROI,
    CONDITION_ANCHOR,
    CONDITION_PEAK,
)
CONDITION_LABELS = {
    CONDITION_FULL: "Full-frame",
    CONDITION_FULL_2FPS: "Full-frame 2 fps",
    CONDITION_MOTION: "Motion Multi-ROI",
    CONDITION_MOTION_ROI: "Motion ROI",
    CONDITION_ANCHOR: "Start-Anchor Multi-ROI",
    CONDITION_PEAK: "Peak-Anchor Multi-ROI",
}
# Existing per-frame score inside a tube. Not a new metric.
MOTION_SCORE_SOURCE = (
    "tube_motion_series mean of stored score_key values over each member cell"
)
# 2x linear enlargement of the VLM pixel budget versus the ROI sequence frame.
ANCHOR_AREA_SCALE = 4
IMAGE_PATCH_SIZE = 16

DEFAULT_VIDEOS = (
    "VIRAT_S_010001_03_000537_000563",
    "VIRAT_S_010003_03_000219_000259",
    "VIRAT_S_010003_10_000901_000934",
    "VIRAT_S_010003_06_000526_000560",
    "VIRAT_S_010106_02_000656_000692",
    "VIRAT_S_010113_05_000776_000805",
    "VIRAT_S_040000_01_000042_000099",
    "VIRAT_S_040005_08_001225_001276",
)
DEFAULT_VIDEO_ROOTS = (
    "/data/datasets/VIRAT/videos-00",
    "/data/datasets/VIRAT/videos-01",
    "/data/datasets/VIRAT/videos-04",
    "/data/datasets/VIRAT/videos-05",
    "/data/datasets/virat_9vid",
)
DEFAULT_TRACKS_ROOT = (
    "/home/ssandree/Motion-guided-ROI/motion-detector/outputs/virat_all_light_uf"
)
SAMPLING_FPS = 1.0
FULL_FRAME_2FPS = 2.0
GLOBAL_DEDUP_EPS_SEC = 0.5
JPEG_QUALITY = 95

# Distinct parameter sizes. FP8 presets are the same size as the BF16 sibling.
SIZE_PRESETS = ("8b", "30b")
# Precision sweep: BF16, INT8 (bitsandbytes on the BF16 checkpoint), and official FP8.
PRECISION_PRESETS = ("8b", "8b-int8", "8b-fp8", "30b", "30b-fp8")

CSV_FIELDS = (
    "video_id",
    "model",
    "condition",
    "generated_description",
    "frame_image_count",
    "visual_tokens",
    "total_pixels",
    "inference_time_sec",
    "error",
)


@dataclass(frozen=True)
class SampledStamp:
    timestamp: float
    native_frame: int


@dataclass(frozen=True)
class Stage3Tube:
    roi_id: int
    t0_sec: float
    t1_sec: float
    bbox_px: tuple[int, int, int, int]


def resolve_video_path(video_id: str, video_roots: Sequence[str | Path]) -> Path:
    roots = [Path(root) for root in video_roots]
    for root in roots:
        path = root / f"{video_id}.mp4"
        if path.is_file():
            return path
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"missing video {video_id}.mp4 in: {searched}")


def load_stage3_tubes(
    *,
    video_id: str,
    tracks_root: Path,
    video_roots: Sequence[str | Path],
) -> list[Stage3Tube]:
    tracks_path = tracks_json_path(Path(tracks_root), video_id)
    if not tracks_path.is_file():
        raise FileNotFoundError(f"missing Stage3 ROI tracks: {tracks_path}")
    payload = load_tracks_payload(tracks_path)
    fusion = payload.get("fusion_npz")
    if not fusion:
        raise ValueError(f"{tracks_path}: missing fusion_npz")
    timeline = load_timeline(fusion)
    timestamps = timeline["timestamp_sec_curr"]
    if len(timestamps) == 0:
        raise ValueError(f"{fusion}: empty timestamp_sec_curr")
    video_path = resolve_video_path(video_id, video_roots)
    info = inspect_video(video_path)
    bboxes = load_tube_bboxes(
        tracks_path, frame_width=info.width, frame_height=info.height
    )
    tubes: list[Stage3Tube] = []
    last = len(timestamps) - 1
    for tube in payload.get("tracks") or []:
        roi_id = int(tube.get("tube_id") or 0)
        if roi_id <= 0 or roi_id not in bboxes:
            continue
        t0 = int(tube.get("t0", 0))
        t1 = int(tube.get("t1", t0))
        if t1 < t0:
            t0, t1 = t1, t0
        t0 = min(max(t0, 0), last)
        t1 = min(max(t1, 0), last)
        tubes.append(
            Stage3Tube(
                roi_id=roi_id,
                t0_sec=float(timestamps[t0]),
                t1_sec=float(timestamps[t1]),
                bbox_px=tuple(int(v) for v in bboxes[roi_id].bbox_px),
            )
        )
    tubes.sort(key=lambda tube: tube.roi_id)
    if not tubes:
        raise ValueError(f"{tracks_path}: no usable Stage3 ROI tubes")
    return tubes


@dataclass(frozen=True)
class TubePlan:
    roi_id: int
    t0_sec: float
    t1_sec: float
    bbox_px: tuple[int, int, int, int]
    stamps: tuple[SampledStamp, ...]


@dataclass
class VideoPlan:
    video_id: str
    video_path: str | None = None
    tracks_path: str | None = None
    fusion_path: str | None = None
    width: int = 0
    height: int = 0
    native_fps: float = 0.0
    frame_count: int = 0
    full_stamps: tuple[SampledStamp, ...] = ()
    full_2fps_stamps: tuple[SampledStamp, ...] = ()
    global_stamps: tuple[SampledStamp, ...] = ()
    tubes: tuple[TubePlan, ...] = ()
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.video_path) and bool(self.tubes) and bool(self.full_stamps)

    def full_frame_image_count(self) -> int:
        return len(self.full_stamps)

    def full_2fps_frame_image_count(self) -> int:
        return len(self.full_2fps_stamps)

    def full_2fps_pixels(self) -> int:
        return int(self.width) * int(self.height) * len(self.full_2fps_stamps)

    def motion_frame_image_count(self) -> int:
        return len(self.global_stamps) + sum(len(tube.stamps) for tube in self.tubes)

    def full_pixels(self) -> int:
        return int(self.width) * int(self.height) * len(self.full_stamps)

    def motion_pixels(self) -> int:
        global_px = int(self.width) * int(self.height) * len(self.global_stamps)
        roi_px = 0
        for tube in self.tubes:
            x0, y0, x1, y1 = tube.bbox_px
            roi_px += max(x1 - x0, 0) * max(y1 - y0, 0) * len(tube.stamps)
        return global_px + roi_px


@dataclass
class PreparedVideo:
    plan: VideoPlan
    full_sample: Sample | None = None
    full_2fps_sample: Sample | None = None
    motion_sample: Sample | None = None
    motion_roi_sample: Sample | None = None
    anchor_sample: Sample | None = None
    peak_sample: Sample | None = None
    full_pixels: int = 0
    full_2fps_pixels: int = 0
    full_2fps_errors: list[str] = field(default_factory=list)
    motion_pixels: int = 0
    motion_roi_pixels: int = 0
    motion_roi_frame_image_count: int = 0
    anchor_pixels: int = 0
    anchor_frame_image_count: int = 0
    peak_pixels: int = 0
    peak_frame_image_count: int = 0
    peak_anchors: list[dict[str, Any]] = field(default_factory=list)
    peak_errors: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SweepRequest:
    """Experiment config. None fields keep the runner's historical defaults."""

    cfg: AppConfig
    models: list[str] | None = None
    videos: list[str] | None = None
    conditions: list[str] | None = None
    prompts: dict[str, str] = field(default_factory=dict)


def load_sweep_request(config_path: str | Path) -> SweepRequest:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    prompts_raw = raw.get("prompts") or {}
    if not isinstance(prompts_raw, dict):
        raise ValueError(f"{path}: prompts must be a mapping of condition -> prompt file")
    videos = _string_list(raw.get("videos"), f"{path}: videos")
    videos_file = raw.get("videos_file")
    if videos_file:
        file_path = Path(str(videos_file))
        if not file_path.is_absolute():
            file_path = path.parent / file_path
        listed = _video_ids_from_file(file_path)
        videos = [*videos, *listed] if videos else listed
    return SweepRequest(
        cfg=load_sweep_config(path),
        models=_string_list(raw.get("models"), f"{path}: models"),
        videos=videos,
        conditions=_string_list(raw.get("conditions"), f"{path}: conditions"),
        prompts={str(key): str(value) for key, value in prompts_raw.items()},
    )


def _video_ids_from_file(path: Path) -> list[str]:
    """Read a video list. Each line is an id or a path; blank lines and # comments are skipped."""
    if not path.is_file():
        raise FileNotFoundError(f"video list not found: {path}")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        stem = Path(text).stem
        if not stem.startswith("VIRAT_"):
            stem = f"VIRAT_S_{stem}"
        ids.append(stem)
    if not ids:
        raise ValueError(f"video list is empty: {path}")
    return ids


def _string_list(value: Any, ctx: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{ctx} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def load_sweep_config(config_path: str | Path) -> AppConfig:
    """Load prompt, generation, and pixel settings without a legacy jobs list."""
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    gen_raw = raw.get("generation") or {}
    video_raw = raw.get("video") or {}
    image_raw = raw.get("image") or {}
    out_raw = raw.get("output") or {}
    return AppConfig(
        model=resolve_model_spec(raw.get("model") or "8b"),
        generation=GenerationConfig(
            max_new_tokens=int(gen_raw.get("max_new_tokens", 256)),
            temperature=float(gen_raw.get("temperature", 0.7)),
            top_p=float(gen_raw.get("top_p", 0.8)),
            top_k=int(gen_raw.get("top_k", 20)),
            repetition_penalty=float(gen_raw.get("repetition_penalty", 1.2)),
            do_sample=bool(gen_raw.get("do_sample", False)),
        ),
        video=VideoConfig(
            fps=video_raw.get("fps", 1.0),
            nframes=video_raw.get("nframes"),
            min_pixels=video_raw.get("min_pixels", 4 * 32 * 32),
            max_pixels=video_raw.get("max_pixels", 256 * 32 * 32),
            total_pixels=video_raw.get("total_pixels", 20480 * 32 * 32),
            sample_fps=video_raw.get("sample_fps"),
        ),
        image=ImageConfig(
            min_pixels=image_raw.get("min_pixels", 4 * 32 * 32),
            max_pixels=image_raw.get("max_pixels", 256 * 32 * 32),
        ),
        output=OutputConfig(
            dir=str(out_raw.get("dir", "outputs/fullframe_motion_multiroi")),
            file=str(out_raw.get("file", "results.json")),
        ),
        mode="default",
        jobs=[],
        prompt=raw.get("prompt"),
        prompt_file=raw.get("prompt_file"),
        config_dir=path.parent,
    )


def discover_model_sizes(
    catalog_path: str | Path | None = None,
    presets: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Return available catalog presets in the requested order.

    The default list is one BF16 checkpoint per size. A precision sweep passes
    ``PRECISION_PRESETS`` or the names from the experiment YAML.
    """
    catalog = load_model_catalog(catalog_path)
    found: list[dict[str, str]] = []
    for name in presets or SIZE_PRESETS:
        spec = catalog.get(name)
        if not spec:
            continue
        path = Path(str(spec["path"]))
        if not (path / "config.json").is_file():
            continue
        weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        if not weights:
            continue
        found.append(
            {
                "name": name,
                "path": str(path),
                "dtype": str(spec.get("dtype", "bfloat16")),
                "load_in_8bit": bool(spec.get("load_in_8bit", False)),
            }
        )
    return found


def nearest_full_stamp(target_t: float, pool: Sequence[SampledStamp]) -> SampledStamp:
    """Same nearest-frame rule as start_only, for one ROI-tube start."""
    return min(
        pool,
        key=lambda frame: (
            abs(float(frame.timestamp) - float(target_t)),
            float(frame.timestamp),
            int(frame.native_frame),
        ),
    )


def effective_roi_frame_max_pixels(n_frames: int, video: VideoConfig) -> int:
    """Pixel cap Qwen applies to one ROI-sequence frame. Does not change sampling."""
    from qwen_vl_utils.vision_process import (
        FRAME_FACTOR,
        SPATIAL_MERGE_SIZE,
        VIDEO_MAX_TOKEN_NUM,
        ceil_by_factor,
    )

    factor = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE
    nframes = ceil_by_factor(max(int(n_frames), 1), FRAME_FACTOR)
    video_frame_max = int(VIDEO_MAX_TOKEN_NUM) * factor * factor
    min_pixels = int(video.min_pixels if video.min_pixels is not None else 4 * factor * factor)
    total = (
        int(video.total_pixels)
        if video.total_pixels is not None
        else int(128000 * factor * factor * 0.9)
    )
    dynamic = max(
        min(video_frame_max, total / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    supposed = int(video.max_pixels) if video.max_pixels is not None else int(dynamic)
    return int(min(supposed, dynamic))


def anchor_pixel_budget(
    crop_width: int,
    crop_height: int,
    *,
    roi_max_pixels: int,
    roi_min_pixels: int,
) -> dict[str, int]:
    """Force the anchor image above the ROI sequence frame's VLM resolution.

    The source crop is the tube's first ROI frame. Sequence frames keep
    ``roi_min_pixels`` / ``roi_max_pixels``. The anchor raises min/max pixels
    so ``smart_resize`` delivers more pixels from that same crop.
    """
    roi_w, roi_h, _roi_tokens = vlm_smart_resize(
        int(crop_width),
        int(crop_height),
        min_pixels=int(roi_min_pixels),
        max_pixels=int(roi_max_pixels),
    )
    roi_pixels = int(roi_w) * int(roi_h)
    target = max(roi_pixels * ANCHOR_AREA_SCALE, roi_pixels + 32 * 32)
    # Qwen image cap: IMAGE_MAX_TOKEN_NUM * 32 * 32.
    image_abs_max = 16384 * 32 * 32
    anchor_max = min(image_abs_max, max(target, int(roi_max_pixels) * ANCHOR_AREA_SCALE))
    anchor_min = min(anchor_max, max(int(roi_min_pixels), target))
    anchor_w, anchor_h, anchor_tokens = vlm_smart_resize(
        int(crop_width),
        int(crop_height),
        min_pixels=anchor_min,
        max_pixels=anchor_max,
    )
    anchor_pixels = int(anchor_w) * int(anchor_h)
    if anchor_pixels <= roi_pixels:
        raise ValueError(
            f"anchor resolution {anchor_w}x{anchor_h} does not exceed "
            f"ROI sequence {roi_w}x{roi_h}"
        )
    return {
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "roi_min_pixels": int(roi_min_pixels),
        "roi_max_pixels": int(roi_max_pixels),
        "roi_resized_width": int(roi_w),
        "roi_resized_height": int(roi_h),
        "roi_resized_pixels": roi_pixels,
        "anchor_min_pixels": int(anchor_min),
        "anchor_max_pixels": int(anchor_max),
        "anchor_resized_width": int(anchor_w),
        "anchor_resized_height": int(anchor_h),
        "anchor_resized_pixels": anchor_pixels,
        "anchor_visual_tokens": int(anchor_tokens),
    }


def uniform_stamps(
    info, *, start_time: float, end_time: float, fps: float = SAMPLING_FPS
) -> tuple[SampledStamp, ...]:
    selected = select_fixed_or_adaptive(
        t0=float(start_time),
        t1=float(end_time),
        fps=float(fps),
        native_fps=float(info.native_fps),
        frame_count=int(info.frame_count),
    )
    return tuple(
        SampledStamp(timestamp=float(stamp.timestamp), native_frame=int(stamp.native_frame))
        for stamp in selected
    )


def discover_video(
    video_id: str,
    *,
    video_roots: Sequence[str | Path],
    tracks_root: str | Path,
) -> VideoPlan:
    """Locate the source video and the saved Stage3 tube. Read-only."""
    plan = VideoPlan(video_id=video_id)
    try:
        video_path = resolve_video_path(video_id, video_roots)
    except FileNotFoundError as exc:
        plan.errors.append(str(exc))
        return plan
    plan.video_path = str(video_path)

    tracks_path = tracks_json_path(Path(tracks_root), video_id)
    plan.tracks_path = str(tracks_path)
    if not tracks_path.is_file():
        plan.errors.append(f"missing Stage3 ROI tracks: {tracks_path}")
        return plan

    try:
        info = inspect_video(video_path)
    except Exception as exc:  # noqa: BLE001
        plan.errors.append(f"video inspect failed: {type(exc).__name__}: {exc}")
        return plan
    plan.width = int(info.width)
    plan.height = int(info.height)
    plan.native_fps = float(info.native_fps)
    plan.frame_count = int(info.frame_count)

    try:
        rois = load_stage3_tubes(
            video_id=video_id,
            tracks_root=Path(tracks_root),
            video_roots=video_roots,
        )
    except Exception as exc:  # noqa: BLE001
        plan.errors.append(f"Stage3 ROI load failed: {type(exc).__name__}: {exc}")
        return plan

    try:
        payload = load_tracks_payload(tracks_path)
        fusion = payload.get("fusion_npz")
        plan.fusion_path = str(fusion) if fusion else None
        if not fusion or not Path(str(fusion)).is_file():
            plan.errors.append(f"missing Stage3 fusion_npz: {fusion}")
    except Exception as exc:  # noqa: BLE001
        plan.errors.append(f"tracks read failed: {type(exc).__name__}: {exc}")

    end_time = info.timestamp_of(info.frame_count - 1)
    plan.full_stamps = uniform_stamps(info, start_time=0.0, end_time=end_time)
    if not plan.full_stamps:
        plan.errors.append("full-frame 1 fps sampling produced no frames")
    plan.full_2fps_stamps = uniform_stamps(
        info, start_time=0.0, end_time=end_time, fps=FULL_FRAME_2FPS
    )
    if not plan.full_2fps_stamps:
        plan.errors.append("full-frame 2 fps sampling produced no frames")

    tubes: list[TubePlan] = []
    anchors: list[SampledStamp] = []
    for roi in sorted(rois, key=lambda item: (float(item.t0_sec), int(item.roi_id))):
        stamps = uniform_stamps(info, start_time=float(roi.t0_sec), end_time=float(roi.t1_sec))
        if not stamps:
            plan.errors.append(f"ROI tube {roi.roi_id} produced no 1 fps frames")
            continue
        tubes.append(
            TubePlan(
                roi_id=int(roi.roi_id),
                t0_sec=float(roi.t0_sec),
                t1_sec=float(roi.t1_sec),
                bbox_px=tuple(int(v) for v in roi.bbox_px),
                stamps=stamps,
            )
        )
        anchors.append(
            SampledStamp(
                timestamp=float(roi.t0_sec),
                native_frame=info.nearest_frame_index(float(roi.t0_sec)),
            )
        )
    plan.tubes = tuple(tubes)
    if not plan.tubes:
        plan.errors.append(f"no usable Stage3 ROI tubes in {tracks_path}")
    plan.global_stamps = start_only_global_stamps(
        anchors, plan.full_stamps, dedup_eps=GLOBAL_DEDUP_EPS_SEC
    )
    if plan.tubes and not plan.global_stamps:
        plan.errors.append("start-only Global selection produced no frames")
    return plan


def start_only_global_stamps(
    roi_anchor_stamps: Sequence[SampledStamp],
    pool: Sequence[SampledStamp],
    *,
    dedup_eps: float = GLOBAL_DEDUP_EPS_SEC,
) -> tuple[SampledStamp, ...]:
    """Nearest 1 fps pool frame for each ROI start, with 0.5 s dedup."""
    if not pool or not roi_anchor_stamps:
        return ()
    selected: list[SampledStamp] = []
    for anchor in sorted(roi_anchor_stamps, key=lambda s: (s.timestamp, s.native_frame)):
        nearest = min(
            pool,
            key=lambda frame: (
                abs(float(frame.timestamp) - float(anchor.timestamp)),
                float(frame.timestamp),
                int(frame.native_frame),
            ),
        )
        if selected and abs(float(nearest.timestamp) - float(selected[-1].timestamp)) <= float(
            dedup_eps
        ):
            continue
        if selected and int(nearest.native_frame) == int(selected[-1].native_frame):
            continue
        selected.append(nearest)
    return tuple(selected)


def discover_videos(
    video_ids: Sequence[str],
    *,
    video_roots: Sequence[str | Path] = DEFAULT_VIDEO_ROOTS,
    tracks_root: str | Path = DEFAULT_TRACKS_ROOT,
) -> list[VideoPlan]:
    return [
        discover_video(video_id, video_roots=video_roots, tracks_root=tracks_root)
        for video_id in video_ids
    ]


def dry_run_payload(plans: Sequence[VideoPlan], models: Sequence[dict[str, str]]) -> dict[str, Any]:
    rows = []
    for plan in plans:
        rows.append(
            {
                "video_id": plan.video_id,
                "ok": plan.ok,
                "video_path": plan.video_path,
                "tracks_path": plan.tracks_path,
                "fusion_path": plan.fusion_path,
                "width": plan.width,
                "height": plan.height,
                "native_fps": plan.native_fps,
                "source_frames": plan.frame_count,
                "num_roi_tubes": len(plan.tubes),
                "roi_ids": [tube.roi_id for tube in plan.tubes],
                "full_frame_count": plan.full_frame_image_count(),
                "full_pixels": plan.full_pixels(),
                "full_2fps_frame_count": plan.full_2fps_frame_image_count(),
                "full_2fps_pixels": plan.full_2fps_pixels(),
                "motion_roi_frame_image_count": motion_roi_counts(plan)[0] if plan.tubes else 0,
                "motion_roi_pixels": motion_roi_counts(plan)[1] if plan.tubes else 0,
                "global_image_count": len(plan.global_stamps),
                "roi_frame_count": sum(len(tube.stamps) for tube in plan.tubes),
                "motion_frame_image_count": plan.motion_frame_image_count(),
                "motion_pixels": plan.motion_pixels(),
                "errors": list(plan.errors),
            }
        )
    return {
        "models": list(models),
        "conditions": list(CONDITIONS),
        "sampling_fps": SAMPLING_FPS,
        "global_insert": "start_only",
        "global_dedup_sec": GLOBAL_DEDUP_EPS_SEC,
        "input_packing": {
            CONDITION_FULL: "frames_as_one_video",
            CONDITION_FULL_2FPS: "frames_as_one_video_2fps",
            CONDITION_MOTION: "global_then_tube_videos",
            CONDITION_MOTION_ROI: "per_tube_global_then_roi_video",
            CONDITION_ANCHOR: "per_tube_global_anchor_then_roi_video",
            CONDITION_PEAK: "per_tube_global_peak_anchor_then_roi_video",
        },
        "videos_ok": sum(1 for plan in plans if plan.ok),
        "videos_total": len(plans),
        "videos": rows,
    }


def start_anchor_audit(
    plans: Sequence[VideoPlan],
    *,
    video: VideoConfig,
    image: ImageConfig,
) -> dict[str, Any]:
    """Input order and VLM resolution for Start-Anchor Multi-ROI. No Stage3 writes."""
    rows: list[dict[str, Any]] = []
    all_ok = True
    roi_min = int(video.min_pixels or image.min_pixels or 4096)
    for plan in plans:
        if not plan.ok:
            all_ok = False
            rows.append({"video_id": plan.video_id, "ok": False, "errors": list(plan.errors), "tubes": []})
            continue
        blocks: list[TubeInputBlock] = []
        tubes: list[dict[str, Any]] = []
        for tube in plan.tubes:
            global_stamp = nearest_full_stamp(tube.t0_sec, plan.full_stamps)
            first = tube.stamps[0]
            x0, y0, x1, y1 = tube.bbox_px
            roi_max = effective_roi_frame_max_pixels(len(tube.stamps), video)
            budget = anchor_pixel_budget(
                x1 - x0,
                y1 - y0,
                roi_max_pixels=roi_max,
                roi_min_pixels=roi_min,
            )
            global_name = (
                f"global_f{global_stamp.native_frame:06d}_t{global_stamp.timestamp:.3f}.jpg"
            )
            anchor_name = (
                f"tube{tube.roi_id:02d}_0000_f{first.native_frame:06d}_t{first.timestamp:.3f}.jpg"
            )
            sequence_names = [
                f"tube{tube.roi_id:02d}_{index:04d}_f{stamp.native_frame:06d}_t{stamp.timestamp:.3f}.jpg"
                for index, stamp in enumerate(tube.stamps)
            ]
            if sequence_names[0] != anchor_name:
                raise RuntimeError(f"{plan.video_id} tube {tube.roi_id}: anchor is not sequence frame 0")
            blocks.append(
                TubeInputBlock(
                    roi_id=int(tube.roi_id),
                    global_image_path=global_name,
                    anchor_image_path=anchor_name,
                    sequence_paths=sequence_names,
                    anchor_min_pixels=budget["anchor_min_pixels"],
                    anchor_max_pixels=budget["anchor_max_pixels"],
                )
            )
            exceeds = budget["anchor_resized_pixels"] > budget["roi_resized_pixels"]
            if not exceeds:
                all_ok = False
            tubes.append(
                {
                    "roi_id": int(tube.roi_id),
                    "t0_sec": tube.t0_sec,
                    "order": ["[Global]", "[ROI Anchor]", f"[ROI{int(tube.roi_id)}]", "roi_sequence"],
                    "global_native_frame": int(global_stamp.native_frame),
                    "global_timestamp": float(global_stamp.timestamp),
                    "anchor_native_frame": int(first.native_frame),
                    "anchor_is_first_sequence_frame": True,
                    "sequence_frame_count": len(tube.stamps),
                    "sequence_native_frames": [int(stamp.native_frame) for stamp in tube.stamps],
                    "resolution": budget,
                    "anchor_exceeds_sequence": exceeds,
                }
            )
        content = build_tube_global_anchor_content(
            blocks,
            global_image=image,
            video=video,
        )
        rows.append(
            {
                "video_id": plan.video_id,
                "ok": all(tube["anchor_exceeds_sequence"] for tube in tubes),
                "existing_motion_order": {
                    "globals_then_all_tube_videos": True,
                    "deduped_global_frames": [
                        int(stamp.native_frame) for stamp in plan.global_stamps
                    ],
                    "tube_ids": [int(tube.roi_id) for tube in plan.tubes],
                },
                "start_anchor_order": _content_order(content),
                "frame_image_count": {
                    CONDITION_MOTION: len(plan.global_stamps) + sum(len(tube.stamps) for tube in plan.tubes),
                    CONDITION_ANCHOR: sum(2 + len(tube.stamps) for tube in plan.tubes),
                },
                "native_pixels": {
                    CONDITION_MOTION: plan.motion_pixels(),
                    CONDITION_ANCHOR: start_anchor_counts(plan)[1],
                },
                "tubes": tubes,
            }
        )
        if not rows[-1]["ok"]:
            all_ok = False
    return {
        "condition": CONDITION_ANCHOR,
        "structure": "per tube: [Global] image -> [ROI Anchor] image -> [ROI#] -> ROI 1 fps video",
        "anchor_source": "first frame of the unchanged ROI sequence, same lifetime crop",
        "anchor_area_scale": ANCHOR_AREA_SCALE,
        "ok": all_ok,
        "videos": rows,
    }


def _content_order(content: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[dict[str, Any]] = []
    for item in content:
        kind = item.get("type")
        if kind == "text":
            order.append({"type": "text", "text": item.get("text")})
        elif kind == "image":
            order.append(
                {
                    "type": "image",
                    "image": item.get("image"),
                    "min_pixels": item.get("min_pixels"),
                    "max_pixels": item.get("max_pixels"),
                }
            )
        elif kind == "video":
            frames = list(item.get("video") or [])
            order.append(
                {
                    "type": "video",
                    "n_frames": len(frames),
                    "first_frame": frames[0] if frames else None,
                    "min_pixels": item.get("min_pixels"),
                    "max_pixels": item.get("max_pixels"),
                    "total_pixels": item.get("total_pixels"),
                }
            )
    return order


def format_start_anchor_audit(audit: dict[str, Any]) -> str:
    lines = [
        f"start-anchor ok: {audit['ok']}",
        audit["structure"],
        f"anchor source: {audit['anchor_source']}",
        "",
    ]
    for video in audit["videos"]:
        status = "OK" if video.get("ok") else "FAIL"
        lines.append(f"[{status}] {video['video_id']}")
        if video.get("errors"):
            for err in video["errors"]:
                lines.append(f"  error: {err}")
            continue
        existing = video["existing_motion_order"]
        lines.append(
            "  existing Multi-ROI: deduped globals "
            f"{existing['deduped_global_frames']} then tube videos {existing['tube_ids']}"
        )
        counts = video.get("frame_image_count") or {}
        pixels = video.get("native_pixels") or {}
        if counts:
            lines.append(
                "  frames/images: "
                f"Multi-ROI {counts.get(CONDITION_MOTION)} "
                f"({pixels.get(CONDITION_MOTION)} px), "
                f"Start-Anchor {counts.get(CONDITION_ANCHOR)} "
                f"({pixels.get(CONDITION_ANCHOR)} px)"
            )
        for tube in video["tubes"]:
            res = tube["resolution"]
            lines.append(
                f"  tube {tube['roi_id']}: [Global] f{tube['global_native_frame']:06d} "
                f"-> [ROI Anchor] f{tube['anchor_native_frame']:06d} "
                f"{res['crop_width']}x{res['crop_height']} "
                f"vlm {res['anchor_resized_width']}x{res['anchor_resized_height']} "
                f"({res['anchor_resized_pixels']} px) "
                f"-> [ROI{tube['roi_id']}] sequence {tube['sequence_frame_count']} frames "
                f"vlm {res['roi_resized_width']}x{res['roi_resized_height']} "
                f"({res['roi_resized_pixels']} px)"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def format_dry_run(payload: dict[str, Any]) -> str:
    lines = [
        f"models: {', '.join(item['name'] for item in payload['models']) or '(none)'}",
        f"videos ok: {payload['videos_ok']}/{payload['videos_total']}",
        "",
    ]
    for row in payload["videos"]:
        status = "OK" if row["ok"] else "FAIL"
        lines.append(
            f"[{status}] {row['video_id']}  tubes={row['num_roi_tubes']} "
            f"full_frames={row['full_frame_count']} "
            f"full_2fps={row.get('full_2fps_frame_count', '')} "
            f"motion_roi_frames={row.get('motion_roi_frame_image_count', '')} "
            f"globals={row['global_image_count']} roi_frames={row['roi_frame_count']}"
        )
        lines.append(f"  video:  {row['video_path']}")
        lines.append(f"  tracks: {row['tracks_path']}")
        lines.append(f"  fusion: {row['fusion_path']}")
        for err in row["errors"]:
            lines.append(f"  error: {err}")
    return "\n".join(lines)


def _as_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _save_rgb(path: Path, rgb) -> int:
    _save_jpeg_rgb(path, rgb, quality=JPEG_QUALITY)
    height, width = rgb.shape[:2]
    return int(width) * int(height)


def materialize_video(
    plan: VideoPlan,
    cache_dir: Path,
    video: VideoConfig | None = None,
) -> PreparedVideo:
    """Write VLM JPEGs under the experiment cache. Does not touch Stage3 files."""
    prepared = PreparedVideo(plan=plan)
    if not plan.ok or plan.video_path is None:
        prepared.errors.extend(plan.errors or ["video plan is not runnable"])
        return prepared

    video_dir = cache_dir / plan.video_id
    manifest_path = video_dir / "manifest.json"
    expected = {
        "video_id": plan.video_id,
        "full_frames": [stamp.native_frame for stamp in plan.full_stamps],
        "global_frames": [stamp.native_frame for stamp in plan.global_stamps],
        "tubes": [
            {
                "roi_id": tube.roi_id,
                "bbox_px": list(tube.bbox_px),
                "frames": [stamp.native_frame for stamp in tube.stamps],
            }
            for tube in plan.tubes
        ],
    }
    if manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None
        if cached == expected and _manifest_files_exist(video_dir, expected):
            return _prepared_from_manifest(plan, video_dir, expected, video=video)

    needed = {stamp.native_frame for stamp in plan.full_stamps}
    needed.update(stamp.native_frame for stamp in plan.global_stamps)
    for tube in plan.tubes:
        needed.update(stamp.native_frame for stamp in tube.stamps)
    try:
        decoded = extract_video_frames(Path(plan.video_path), needed)
    except Exception as exc:  # noqa: BLE001
        prepared.errors.append(f"frame decode failed: {type(exc).__name__}: {exc}")
        return prepared

    full_dir = video_dir / "full_frame"
    global_dir = video_dir / "motion_multi_roi" / "global"
    full_paths: list[str] = []
    full_pixels = 0
    try:
        for index, stamp in enumerate(plan.full_stamps):
            path = full_dir / f"full_{index:04d}_f{stamp.native_frame:06d}_t{stamp.timestamp:.3f}.jpg"
            full_pixels += _save_rgb(path, decoded[stamp.native_frame])
            full_paths.append(_as_uri(path))
        global_paths: list[str] = []
        for index, stamp in enumerate(plan.global_stamps):
            path = global_dir / f"global_{index:04d}_f{stamp.native_frame:06d}_t{stamp.timestamp:.3f}.jpg"
            _save_rgb(path, decoded[stamp.native_frame])
            global_paths.append(_as_uri(path))
        tube_videos: list[tuple[int, list[str]]] = []
        motion_pixels = plan.width * plan.height * len(plan.global_stamps)
        for tube in plan.tubes:
            tube_dir = video_dir / "motion_multi_roi" / f"tube{tube.roi_id:02d}"
            paths: list[str] = []
            for index, stamp in enumerate(tube.stamps):
                crop = _crop_rgb(decoded[stamp.native_frame], tube.bbox_px)
                path = tube_dir / (
                    f"tube{tube.roi_id:02d}_{index:04d}_f{stamp.native_frame:06d}_t{stamp.timestamp:.3f}.jpg"
                )
                motion_pixels += _save_rgb(path, crop)
                paths.append(_as_uri(path))
            tube_videos.append((tube.roi_id, paths))
    except Exception as exc:  # noqa: BLE001
        prepared.errors.append(f"jpeg materialize failed: {type(exc).__name__}: {exc}")
        return prepared

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    prepared.full_sample = _full_sample(plan.video_id, full_paths)
    prepared.motion_sample = _motion_sample(plan.video_id, global_paths, tube_videos)
    _attach_full_2fps(prepared, video_dir)
    prepared.full_pixels = full_pixels
    prepared.motion_pixels = motion_pixels
    _attach_motion_roi(prepared, video_dir)
    _attach_start_anchor(prepared, video_dir, video)
    _attach_peak_anchor(prepared, video_dir, video)
    return prepared


def _manifest_files_exist(video_dir: Path, expected: dict[str, Any]) -> bool:
    full_dir = video_dir / "full_frame"
    if len(list(full_dir.glob("full_*.jpg"))) != len(expected["full_frames"]):
        return False
    global_dir = video_dir / "motion_multi_roi" / "global"
    if len(list(global_dir.glob("global_*.jpg"))) != len(expected["global_frames"]):
        return False
    for tube in expected["tubes"]:
        tube_dir = video_dir / "motion_multi_roi" / f"tube{int(tube['roi_id']):02d}"
        if len(list(tube_dir.glob("tube*.jpg"))) != len(tube["frames"]):
            return False
    return True


def _prepared_from_manifest(
    plan: VideoPlan,
    video_dir: Path,
    expected: dict[str, Any],
    video: VideoConfig | None = None,
) -> PreparedVideo:
    full_paths = [_as_uri(path) for path in sorted((video_dir / "full_frame").glob("full_*.jpg"))]
    global_paths = [
        _as_uri(path)
        for path in sorted((video_dir / "motion_multi_roi" / "global").glob("global_*.jpg"))
    ]
    tube_videos: list[tuple[int, list[str]]] = []
    motion_pixels = 0
    for tube in plan.tubes:
        paths = sorted((video_dir / "motion_multi_roi" / f"tube{tube.roi_id:02d}").glob("tube*.jpg"))
        tube_videos.append((tube.roi_id, [_as_uri(path) for path in paths]))
        x0, y0, x1, y1 = tube.bbox_px
        motion_pixels += max(x1 - x0, 0) * max(y1 - y0, 0) * len(paths)
    motion_pixels += plan.width * plan.height * len(global_paths)
    prepared = PreparedVideo(
        plan=plan,
        full_sample=_full_sample(plan.video_id, full_paths),
        motion_sample=_motion_sample(plan.video_id, global_paths, tube_videos),
        full_pixels=plan.width * plan.height * len(full_paths),
        motion_pixels=motion_pixels,
    )
    if len(full_paths) != len(expected["full_frames"]):
        prepared.errors.append("cached full-frame count does not match the plan")
    else:
        _attach_motion_roi(prepared, video_dir)
        _attach_start_anchor(prepared, video_dir, video)
        _attach_peak_anchor(prepared, video_dir, video)
        _attach_full_2fps(prepared, video_dir)
    return prepared


def _full_sample(video_id: str, paths: list[str], condition: str = CONDITION_FULL) -> Sample:
    return Sample(
        job_name=f"{video_id}__{condition}",
        sample_id=video_id,
        input_type="frames",
        paths=list(paths),
    )


def _attach_full_2fps(prepared: PreparedVideo, video_dir: Path) -> None:
    """Write a separate 2 fps full-frame sequence. Does not replace the 1 fps frames."""
    plan = prepared.plan
    stamps = plan.full_2fps_stamps
    dest = video_dir / "full_frame_2fps"
    paths: list[Path] = []
    missing: list[tuple[SampledStamp, Path]] = []
    for index, stamp in enumerate(stamps):
        path = dest / f"full_{index:04d}_f{stamp.native_frame:06d}_t{stamp.timestamp:.3f}.jpg"
        paths.append(path)
        if not path.is_file():
            missing.append((stamp, path))
    if missing:
        if plan.video_path is None:
            prepared.full_2fps_errors.append("full-frame 2 fps source video is missing")
            return
        try:
            decoded = extract_video_frames(
                Path(plan.video_path), {stamp.native_frame for stamp, _ in missing}
            )
            for stamp, path in missing:
                _save_rgb(path, decoded[stamp.native_frame])
        except Exception as exc:  # noqa: BLE001
            prepared.full_2fps_errors.append(
                f"full-frame 2 fps materialize failed: {type(exc).__name__}: {exc}"
            )
            return
    prepared.full_2fps_pixels = int(plan.width) * int(plan.height) * len(stamps)
    prepared.full_2fps_sample = _full_sample(
        plan.video_id, [_as_uri(path) for path in paths], CONDITION_FULL_2FPS
    )


def motion_roi_counts(plan: VideoPlan) -> tuple[int, int]:
    """One full-frame Global per tube, then that tube's ROI sequence. No anchor."""
    frames = 0
    pixels = 0
    for tube in plan.tubes:
        x0, y0, x1, y1 = tube.bbox_px
        crop = max(x1 - x0, 0) * max(y1 - y0, 0)
        frames += 1 + len(tube.stamps)
        pixels += int(plan.width) * int(plan.height) + crop * len(tube.stamps)
    return frames, pixels


def _attach_motion_roi(prepared: PreparedVideo, video_dir: Path) -> None:
    plan = prepared.plan
    frames, pixels = motion_roi_counts(plan)
    prepared.motion_roi_frame_image_count = frames
    prepared.motion_roi_pixels = pixels
    try:
        prepared.motion_roi_sample = build_motion_roi_sample(plan, video_dir)
    except Exception as exc:  # noqa: BLE001
        prepared.errors.append(f"motion-roi sample failed: {type(exc).__name__}: {exc}")


def build_motion_roi_sample(plan: VideoPlan, video_dir: Path) -> Sample:
    """Per tube: [Global] at the ROI start, then [ROI#] and the existing ROI video."""
    full_by_frame: dict[int, Path] = {}
    for path in (video_dir / "full_frame").glob("full_*.jpg"):
        frame = _native_frame_from_name(path)
        if frame is not None:
            full_by_frame[frame] = path
    blocks: list[TubeGlobalRoiBlock] = []
    for tube in plan.tubes:
        global_stamp = nearest_full_stamp(tube.t0_sec, plan.full_stamps)
        global_path = full_by_frame.get(int(global_stamp.native_frame))
        if global_path is None:
            raise FileNotFoundError(
                f"{plan.video_id} tube {tube.roi_id}: missing global frame {global_stamp.native_frame}"
            )
        sequence = sorted((video_dir / "motion_multi_roi" / f"tube{tube.roi_id:02d}").glob("tube*.jpg"))
        if [_native_frame_from_name(path) for path in sequence] != [
            int(stamp.native_frame) for stamp in tube.stamps
        ]:
            raise ValueError(
                f"{plan.video_id} tube {tube.roi_id}: ROI sequence files differ from the saved tube stamps"
            )
        blocks.append(
            TubeGlobalRoiBlock(
                roi_id=int(tube.roi_id),
                global_image_path=_as_uri(global_path),
                sequence_paths=[_as_uri(path) for path in sequence],
            )
        )
    return Sample(
        job_name=f"{plan.video_id}__{CONDITION_MOTION_ROI}",
        sample_id=plan.video_id,
        input_type="tube_global_roi_videos",
        paths=[],
        tube_global_roi_blocks=blocks,
    )


def start_anchor_counts(plan: VideoPlan) -> tuple[int, int]:
    """Global + extra anchor image + unchanged ROI sequence, in native pixels."""
    frames = 0
    pixels = 0
    for tube in plan.tubes:
        x0, y0, x1, y1 = tube.bbox_px
        crop = max(x1 - x0, 0) * max(y1 - y0, 0)
        frames += 2 + len(tube.stamps)
        pixels += int(plan.width) * int(plan.height) + crop * (1 + len(tube.stamps))
    return frames, pixels


def _native_frame_from_name(path: Path) -> int | None:
    import re

    match = re.search(r"_f(\d+)_", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _attach_start_anchor(
    prepared: PreparedVideo,
    video_dir: Path,
    video: VideoConfig | None = None,
) -> None:
    """Point the new condition at existing crops. Does not rewrite ROI JPEGs."""
    plan = prepared.plan
    frames, pixels = start_anchor_counts(plan)
    prepared.anchor_frame_image_count = frames
    prepared.anchor_pixels = pixels
    try:
        prepared.anchor_sample = build_start_anchor_sample(plan, video_dir, video)
    except Exception as exc:  # noqa: BLE001
        prepared.errors.append(f"start-anchor sample failed: {type(exc).__name__}: {exc}")


def build_start_anchor_sample(plan: VideoPlan, video_dir: Path, video: VideoConfig | None = None) -> Sample:
    """Per tube: Global, enlarged first-crop anchor, then the existing ROI video."""
    if video is None:
        video = VideoConfig(
            fps=1.0,
            min_pixels=4096,
            max_pixels=256 * 32 * 32,
            total_pixels=20480 * 32 * 32,
        )
    full_by_frame: dict[int, Path] = {}
    for path in (video_dir / "full_frame").glob("full_*.jpg"):
        frame = _native_frame_from_name(path)
        if frame is not None:
            full_by_frame[frame] = path
    roi_min = int(video.min_pixels or 4096)
    blocks: list[TubeInputBlock] = []
    for tube in plan.tubes:
        global_stamp = nearest_full_stamp(tube.t0_sec, plan.full_stamps)
        global_path = full_by_frame.get(int(global_stamp.native_frame))
        if global_path is None:
            raise FileNotFoundError(
                f"{plan.video_id} tube {tube.roi_id}: missing global frame {global_stamp.native_frame}"
            )
        sequence = sorted((video_dir / "motion_multi_roi" / f"tube{tube.roi_id:02d}").glob("tube*.jpg"))
        if [ _native_frame_from_name(path) for path in sequence ] != [
            int(stamp.native_frame) for stamp in tube.stamps
        ]:
            raise ValueError(
                f"{plan.video_id} tube {tube.roi_id}: ROI sequence files differ from the saved tube stamps"
            )
        x0, y0, x1, y1 = tube.bbox_px
        budget = anchor_pixel_budget(
            x1 - x0,
            y1 - y0,
            roi_max_pixels=effective_roi_frame_max_pixels(len(sequence), video),
            roi_min_pixels=roi_min,
        )
        blocks.append(
            TubeInputBlock(
                roi_id=int(tube.roi_id),
                global_image_path=_as_uri(global_path),
                anchor_image_path=_as_uri(sequence[0]),
                sequence_paths=[_as_uri(path) for path in sequence],
                anchor_min_pixels=budget["anchor_min_pixels"],
                anchor_max_pixels=budget["anchor_max_pixels"],
            )
        )
    return Sample(
        job_name=f"{plan.video_id}__{CONDITION_ANCHOR}",
        sample_id=plan.video_id,
        input_type="tube_global_anchor_videos",
        paths=[],
        tube_input_blocks=blocks,
    )


def _member_events_from_score(tube: dict[str, Any], mag) -> dict[int, dict[str, Any]]:
    """Rebuild the unsaved event magnitudes from members plus the stored score map."""
    import numpy as np

    events: dict[int, dict[str, Any]] = {}
    for member in tube.get("members") or []:
        y = int(member["y"])
        x = int(member["x"])
        per_frame = []
        for frame_index in range(int(member["t0"]), int(member["t1"]) + 1):
            value = 0.0
            if 0 <= frame_index < mag.shape[0] and 0 <= y < mag.shape[1] and 0 <= x < mag.shape[2]:
                raw = float(mag[frame_index, y, x])
                if np.isfinite(raw):
                    value = raw
            per_frame.append({"frame_index": int(frame_index), "magnitudes": [value]})
        event_id = int(member["event_id"])
        events[event_id] = {"event_id": event_id, "per_frame": per_frame}
    return events


def select_tube_motion_peaks(plan: VideoPlan) -> list[dict[str, Any]]:
    """Pick each tube's stored max motion frame. Does not recompute Stage1–3."""
    import numpy as np

    if not plan.tracks_path:
        raise FileNotFoundError(f"{plan.video_id}: missing tracks path")
    payload = load_tracks_payload(Path(plan.tracks_path))
    fusion = payload.get("fusion_npz")
    if not fusion or not Path(str(fusion)).is_file():
        raise FileNotFoundError(f"{plan.video_id}: missing fusion timeline {fusion}")
    score_key = str(payload.get("score_key") or "MU_fused")
    with np.load(fusion) as data:
        timeline = {
            "timestamp_sec_curr": np.asarray(data["timestamp_sec_curr"], dtype=np.float64),
            "frame_idx_curr": np.asarray(data["frame_idx_curr"], dtype=np.int32),
        }
        mag = np.asarray(data[score_key], dtype=np.float64)
    saved_events = _events_by_id(payload)
    tubes = {int(tube["tube_id"]): tube for tube in (payload.get("tracks") or [])}
    selected: list[dict[str, Any]] = []
    for tube_plan in plan.tubes:
        tube = tubes.get(int(tube_plan.roi_id))
        if tube is None:
            raise KeyError(f"{plan.video_id}: ROI {tube_plan.roi_id} is not in the saved tracks")
        events = saved_events or _member_events_from_score(tube, mag)
        _mean, _total, series = tube_motion_series(tube, events, timeline)
        if not series:
            raise ValueError(f"{plan.video_id} tube {tube_plan.roi_id}: empty motion series")
        t0 = int(tube.get("t0", series[0].stage3_index))
        t1 = int(tube.get("t1", series[-1].stage3_index))
        if t1 < t0:
            t0, t1 = t1, t0
        inside = [sample for sample in series if t0 <= int(sample.stage3_index) <= t1]
        if not inside:
            raise ValueError(f"{plan.video_id} tube {tube_plan.roi_id}: no motion frame inside the tube")
        peak = max(inside, key=lambda sample: (float(sample.magnitude), -int(sample.stage3_index)))
        start = tube_plan.stamps[0]
        selected.append(
            {
                "roi_id": int(tube_plan.roi_id),
                "tube_start_frame": int(start.native_frame),
                "tube_start_timestamp": float(start.timestamp),
                "peak_frame": int(peak.native_frame),
                "peak_timestamp": float(peak.timestamp),
                "peak_stage3_index": int(peak.stage3_index),
                "motion_score": float(peak.magnitude),
                "motion_score_source": MOTION_SCORE_SOURCE,
                "score_key": score_key,
                "same_as_start_frame": int(peak.native_frame) == int(start.native_frame),
            }
        )
    return selected


def _attach_peak_anchor(
    prepared: PreparedVideo,
    video_dir: Path,
    video: VideoConfig | None = None,
) -> None:
    """Save only new peak crops. Does not rewrite sequence JPEGs or start-anchor inputs."""
    plan = prepared.plan
    frames, pixels = start_anchor_counts(plan)
    prepared.peak_frame_image_count = frames
    prepared.peak_pixels = pixels
    try:
        peaks = select_tube_motion_peaks(plan)
        prepared.peak_sample = build_peak_anchor_sample(plan, video_dir, peaks, video)
        prepared.peak_anchors = peaks
    except Exception as exc:  # noqa: BLE001
        prepared.peak_errors.append(f"peak-anchor sample failed: {type(exc).__name__}: {exc}")


def build_peak_anchor_sample(
    plan: VideoPlan,
    video_dir: Path,
    peaks: Sequence[dict[str, Any]],
    video: VideoConfig | None = None,
) -> Sample:
    """Per tube: Global, enlarged motion-peak crop, [ROI#], then the existing ROI video."""
    if video is None:
        video = VideoConfig(
            fps=1.0,
            min_pixels=4096,
            max_pixels=256 * 32 * 32,
            total_pixels=20480 * 32 * 32,
        )
    by_roi = {int(row["roi_id"]): row for row in peaks}
    full_by_frame: dict[int, Path] = {}
    for path in (video_dir / "full_frame").glob("full_*.jpg"):
        frame = _native_frame_from_name(path)
        if frame is not None:
            full_by_frame[frame] = path
    roi_min = int(video.min_pixels or 4096)
    blocks: list[TubeInputBlock] = []
    for tube in plan.tubes:
        peak = by_roi.get(int(tube.roi_id))
        if peak is None:
            raise KeyError(f"{plan.video_id} tube {tube.roi_id}: missing peak selection")
        global_stamp = nearest_full_stamp(tube.t0_sec, plan.full_stamps)
        global_path = full_by_frame.get(int(global_stamp.native_frame))
        if global_path is None:
            raise FileNotFoundError(
                f"{plan.video_id} tube {tube.roi_id}: missing global frame {global_stamp.native_frame}"
            )
        sequence = sorted((video_dir / "motion_multi_roi" / f"tube{tube.roi_id:02d}").glob("tube*.jpg"))
        sequence_frames = [_native_frame_from_name(path) for path in sequence]
        if sequence_frames != [int(stamp.native_frame) for stamp in tube.stamps]:
            raise ValueError(
                f"{plan.video_id} tube {tube.roi_id}: ROI sequence files differ from the saved tube stamps"
            )
        anchor_path = _peak_anchor_image(plan, video_dir, tube, sequence, peak)
        x0, y0, x1, y1 = tube.bbox_px
        budget = anchor_pixel_budget(
            x1 - x0,
            y1 - y0,
            roi_max_pixels=effective_roi_frame_max_pixels(len(sequence), video),
            roi_min_pixels=roi_min,
        )
        peak["anchor_resolution"] = {
            "crop_width": budget["crop_width"],
            "crop_height": budget["crop_height"],
            "vlm_width": budget["anchor_resized_width"],
            "vlm_height": budget["anchor_resized_height"],
            "vlm_pixels": budget["anchor_resized_pixels"],
            "sequence_vlm_width": budget["roi_resized_width"],
            "sequence_vlm_height": budget["roi_resized_height"],
            "sequence_vlm_pixels": budget["roi_resized_pixels"],
        }
        blocks.append(
            TubeInputBlock(
                roi_id=int(tube.roi_id),
                global_image_path=_as_uri(global_path),
                anchor_image_path=_as_uri(anchor_path),
                sequence_paths=[_as_uri(path) for path in sequence],
                anchor_min_pixels=budget["anchor_min_pixels"],
                anchor_max_pixels=budget["anchor_max_pixels"],
            )
        )
    return Sample(
        job_name=f"{plan.video_id}__{CONDITION_PEAK}",
        sample_id=plan.video_id,
        input_type="tube_global_anchor_videos",
        paths=[],
        tube_input_blocks=blocks,
    )


def _peak_anchor_image(
    plan: VideoPlan,
    video_dir: Path,
    tube: TubePlan,
    sequence: Sequence[Path],
    peak: dict[str, Any],
) -> Path:
    """Reuse a sequence crop when the peak frame is already sampled. Otherwise write one new crop."""
    peak_frame = int(peak["peak_frame"])
    for path in sequence:
        if _native_frame_from_name(path) == peak_frame:
            return path
    dest = video_dir / "motion_multi_roi_peak_anchor" / (
        f"tube{tube.roi_id:02d}_anchor_f{peak_frame:06d}_t{float(peak['peak_timestamp']):.3f}.jpg"
    )
    if dest.is_file():
        return dest
    if plan.video_path is None:
        raise FileNotFoundError(f"{plan.video_id}: missing source video for peak crop")
    decoded = extract_video_frames(Path(plan.video_path), {peak_frame})
    frame = decoded.get(peak_frame)
    if frame is None:
        raise KeyError(f"{plan.video_id} tube {tube.roi_id}: could not decode peak frame {peak_frame}")
    _save_rgb(dest, _crop_rgb(frame, tube.bbox_px))
    return dest


def peak_anchor_audit(
    plans: Sequence[VideoPlan],
    *,
    video: VideoConfig,
    image: ImageConfig,
) -> dict[str, Any]:
    """Peak-frame selection and the same anchor pixel budget as start-anchor. No writes."""
    rows: list[dict[str, Any]] = []
    all_ok = True
    roi_min = int(video.min_pixels or image.min_pixels or 4096)
    for plan in plans:
        if not plan.ok:
            all_ok = False
            rows.append({"video_id": plan.video_id, "ok": False, "errors": list(plan.errors), "tubes": []})
            continue
        try:
            peaks = select_tube_motion_peaks(plan)
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            rows.append(
                {
                    "video_id": plan.video_id,
                    "ok": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "tubes": [],
                }
            )
            continue
        tubes: list[dict[str, Any]] = []
        video_ok = True
        by_roi = {int(row["roi_id"]): row for row in peaks}
        for tube in plan.tubes:
            peak = by_roi[int(tube.roi_id)]
            x0, y0, x1, y1 = tube.bbox_px
            budget = anchor_pixel_budget(
                x1 - x0,
                y1 - y0,
                roi_max_pixels=effective_roi_frame_max_pixels(len(tube.stamps), video),
                roi_min_pixels=roi_min,
            )
            start_frame = int(tube.stamps[0].native_frame)
            in_range = float(tube.t0_sec) - 1e-6 <= float(peak["peak_timestamp"]) <= float(tube.t1_sec) + 1e-6
            if not in_range or int(peak["tube_start_frame"]) != start_frame:
                video_ok = False
            row = dict(peak)
            row["anchor_resolution"] = {
                "crop_width": budget["crop_width"],
                "crop_height": budget["crop_height"],
                "vlm_width": budget["anchor_resized_width"],
                "vlm_height": budget["anchor_resized_height"],
                "vlm_pixels": budget["anchor_resized_pixels"],
                "sequence_vlm_width": budget["roi_resized_width"],
                "sequence_vlm_height": budget["roi_resized_height"],
                "sequence_vlm_pixels": budget["roi_resized_pixels"],
            }
            row["inside_tube"] = in_range
            tubes.append(row)
        if not video_ok:
            all_ok = False
        rows.append({"video_id": plan.video_id, "ok": video_ok, "tubes": tubes})
    return {
        "condition": CONDITION_PEAK,
        "structure": "per tube: [Global] image -> [ROI Anchor] peak crop -> [ROI#] -> unchanged ROI 1 fps video",
        "motion_score_source": MOTION_SCORE_SOURCE,
        "ok": all_ok,
        "videos": rows,
    }


def format_peak_anchor_audit(audit: dict[str, Any]) -> str:
    lines = [
        f"peak-anchor ok: {audit['ok']}",
        audit["structure"],
        f"motion score: {audit['motion_score_source']}",
        "video_id / ROI id / tube start frame / selected peak frame / peak timestamp / motion score / anchor resolution",
        "",
    ]
    for video in audit["videos"]:
        if video.get("errors"):
            lines.append(f"[FAIL] {video['video_id']}")
            for err in video["errors"]:
                lines.append(f"  error: {err}")
            continue
        for tube in video["tubes"]:
            res = tube["anchor_resolution"]
            same = " SAME_AS_START" if tube["same_as_start_frame"] else ""
            lines.append(
                f"{video['video_id']} / ROI {tube['roi_id']} / "
                f"start f{tube['tube_start_frame']:06d} / "
                f"peak f{tube['peak_frame']:06d} / "
                f"t={tube['peak_timestamp']:.3f} / "
                f"score={tube['motion_score']:.6f} / "
                f"crop {res['crop_width']}x{res['crop_height']} "
                f"vlm {res['vlm_width']}x{res['vlm_height']} ({res['vlm_pixels']} px)"
                f"{same}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _motion_sample(
    video_id: str,
    global_paths: list[str],
    tube_videos: list[tuple[int, list[str]]],
) -> Sample:
    return Sample(
        job_name=f"{video_id}__{CONDITION_MOTION}",
        sample_id=video_id,
        input_type="global_then_tube_videos",
        paths=[],
        global_image_paths=list(global_paths),
        tube_video_paths=list(tube_videos),
    )


def run_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record["model"]), str(record["video_id"]), str(record["condition"]))


def is_success(record: dict[str, Any]) -> bool:
    return not record.get("error") and bool(str(record.get("generated_description") or "").strip())


def load_saved_runs(output_dir: Path) -> list[dict[str, Any]]:
    results_path = output_dir / "results.json"
    if results_path.is_file():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            rows = payload.get("runs") if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        except json.JSONDecodeError:
            logger.warning("Could not parse %s; scanning per-run files", results_path)
    runs_dir = output_dir / "runs"
    rows = []
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("*.json")):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return rows


def shared_prompt(cfg: AppConfig) -> str:
    if cfg.prompt_file:
        return load_prompt_text(cfg.prompt_file, base=cfg.config_dir)
    if cfg.prompt:
        return cfg.prompt.strip()
    raise ValueError("config requires prompt_file")


def load_prompt_text(path: str | Path, *, base: Path | None = None) -> str:
    return _read_text(resolve_path(base or Path.cwd(), str(path)))


def generation_payload(gen: GenerationConfig) -> dict[str, Any]:
    return {
        "max_new_tokens": gen.max_new_tokens,
        "do_sample": gen.do_sample,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "top_k": gen.top_k,
        "repetition_penalty": gen.repetition_penalty,
    }


def empty_run(
    *,
    video_id: str,
    model: str,
    condition: str,
    frame_image_count: int | None = None,
    total_pixels: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "model": model,
        "model_path": None,
        "condition": condition,
        "generated_description": None,
        "frame_image_count": frame_image_count,
        "visual_tokens": None,
        "total_pixels": total_pixels,
        "inference_time_sec": None,
        "error": error,
    }


def write_outputs(
    output_dir: Path,
    *,
    runs: Sequence[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "runs": list(runs)}
    _atomic_text(output_dir / "results.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(output_dir / "results.csv", _csv_text(runs))
    _atomic_text(output_dir / "report.md", render_report(runs, meta))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _csv_text(runs: Sequence[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
    writer.writeheader()
    for run in runs:
        writer.writerow({key: run.get(key) for key in CSV_FIELDS})
    return buffer.getvalue()


def _listed_conditions(meta: dict[str, Any]) -> list[str]:
    listed = [str(name) for name in (meta.get("conditions") or CONDITIONS)]
    return [name for name in listed if name in CONDITION_LABELS]


def _prompt_lines(meta: dict[str, Any]) -> list[str]:
    prompts = meta.get("prompts")
    if isinstance(prompts, dict) and prompts:
        lines = ["- Prompts:"]
        for condition in _listed_conditions(meta):
            path = prompts.get(condition)
            if path:
                lines.append(f"  - {CONDITION_LABELS[condition]}: `{path}`")
        return lines
    return [f"- Prompt: `{meta.get('prompt_file')}`"]


def render_report(runs: Sequence[dict[str, Any]], meta: dict[str, Any]) -> str:
    """Side-by-side record. Does not rank or prefer a condition."""
    lines = [
        "# Full-frame vs Motion Multi-ROI",
        "",
        "같은 generation setting으로 생성한 결과를 video별로 나란히 기록한다. "
        "어느 condition이 더 좋은지는 판정하지 않는다.",
        "",
        "## Setup",
        "",
        *_prompt_lines(meta),
        f"- Sampling: full-frame uniform {meta.get('sampling_fps')} fps; "
        "ROI sequence는 저장된 Stage3 tube의 1 fps crop.",
        "- Motion ROI packing: each tube is `[Global]` once at the ROI start, then `[ROI#]` and that ROI video. No anchor.",
        f"- Models: {', '.join(meta.get('models') or [])}",
        "",
        "Generation:",
        "",
        "```json",
        json.dumps(meta.get("generation") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| video_id | model | condition | frame/image count | visual tokens | total pixels | inference time (s) | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    by_video: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_video.setdefault(str(run["video_id"]), []).append(run)
        status = "ok" if is_success(run) else "error"
        lines.append(
            "| {video_id} | {model} | {condition} | {frames} | {tokens} | {pixels} | {time} | {status} |".format(
                video_id=run.get("video_id"),
                model=run.get("model"),
                condition=CONDITION_LABELS.get(str(run.get("condition")), run.get("condition")),
                frames=_fmt(run.get("frame_image_count")),
                tokens=_fmt(run.get("visual_tokens")),
                pixels=_fmt(run.get("total_pixels")),
                time=_fmt_time(run.get("inference_time_sec")),
                status=status,
            )
        )
    lines.extend(["", "## Descriptions", ""])
    for video_id in meta.get("videos") or list(by_video):
        lines.append(f"### {video_id}")
        lines.append("")
        video_runs = by_video.get(video_id, [])
        models = []
        for run in video_runs:
            if run["model"] not in models:
                models.append(run["model"])
        for model in models:
            lines.append(f"#### {model}")
            lines.append("")
            for condition in _listed_conditions(meta):
                match = next(
                    (
                        run
                        for run in video_runs
                        if run["model"] == model and run["condition"] == condition
                    ),
                    None,
                )
                lines.append(f"**{CONDITION_LABELS[condition]}**")
                lines.append("")
                if match is None:
                    lines.append("_아직 실행되지 않음_")
                    lines.append("")
                    continue
                if match.get("error"):
                    lines.append(f"Error: `{match['error']}`")
                    lines.append("")
                description = str(match.get("generated_description") or "").strip()
                if description:
                    lines.append(description)
                    lines.append("")
                else:
                    lines.append("_생성된 description 없음_")
                    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def _fmt_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def save_run_file(output_dir: Path, record: dict[str, Any]) -> None:
    model, video_id, condition = run_key(record)
    safe_model = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
    path = output_dir / "runs" / f"{safe_model}__{video_id}__{condition}.json"
    _atomic_text(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
