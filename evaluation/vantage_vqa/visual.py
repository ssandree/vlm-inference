"""Video-level VANTAGE visual inputs via existing SOVABench builders.

This module must not accept VQA question, options, predicted answers, or GT.
Visual construction depends only on the original video and existing motion/ROI
sampling implementations.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from vlm_inference.gt_action_cls import ProposedROI, load_proposed_rois
from vlm_inference.motion_adaptive import DEFAULT_TAU, RoiMotionRecord
from vlm_inference.sovabench import (
    GLOBAL_DEDUP_EPS_SEC,
    SovaSample,
    SpatialJob,
    attach_native_times,
)
from vlm_inference.sovabench_comparison import (
    CONDITION_FULL_1,
    CONDITION_FULL_2,
    build_full_jobs_at_fps,
    is_full_frame as _existing_is_full_frame,
    materialize_full_frames,
)
from vlm_inference.sovabench_spatial import materialize_job_frames
from vlm_inference.sovabench_temporal import (
    CONDITION_CENTROID,
    CONDITION_FIXED_1,
    build_temporal_jobs,
    records_by_video_from_tracks,
)
from vlm_inference.video_io import inspect_video

CONDITION_FULL_2FPS = "full_2fps"
CONDITION_FULL_1FPS = "full_1fps"
CONDITION_FULL_05FPS = "full_05fps"
CONDITION_ROI_FIXED_1FPS = "roi_fixed_1fps"
CONDITION_ROI_CENTROID = "roi_centroid"

USER_CONDITIONS = (
    CONDITION_FULL_2FPS,
    CONDITION_FULL_1FPS,
    CONDITION_FULL_05FPS,
    CONDITION_ROI_FIXED_1FPS,
    CONDITION_ROI_CENTROID,
)

FULL_CONDITIONS = (
    CONDITION_FULL_2FPS,
    CONDITION_FULL_1FPS,
    CONDITION_FULL_05FPS,
)
ROI_CONDITIONS = (CONDITION_ROI_FIXED_1FPS, CONDITION_ROI_CENTROID)

_FULL_FPS = {
    CONDITION_FULL_2FPS: 2.0,
    CONDITION_FULL_1FPS: 1.0,
    CONDITION_FULL_05FPS: 0.5,
}

# Existing sampler names used internally; user-facing roi_centroid maps here.
_INTERNAL_CONDITION = {
    CONDITION_FULL_2FPS: CONDITION_FULL_2,
    CONDITION_FULL_1FPS: CONDITION_FULL_1,
    CONDITION_FULL_05FPS: CONDITION_FULL_05FPS,
    CONDITION_ROI_FIXED_1FPS: CONDITION_FIXED_1,
    CONDITION_ROI_CENTROID: CONDITION_CENTROID,
}


@dataclass
class VisualInput:
    video_id: str
    condition: str
    frame_paths: list[str]
    sampled_timestamps: list[float]
    sampled_native_frames: list[int]
    global_timestamps: list[float]
    global_native_frames: list[int]
    roi_timestamps: dict[int, list[float]]
    source_types: list[str]
    n_rois: int
    num_input_images: int
    processed_pixels: int
    input_image_resolutions: list[list[int]]
    input_build_time: float
    cache_hit: bool
    identity: tuple[Any, ...]
    job: SpatialJob | None = field(default=None, repr=False)


def visual_builder_forbidden_params() -> tuple[str, ...]:
    """Names that must never appear on the visual-input builder."""
    return (
        "question",
        "options",
        "prediction",
        "answer",
        "gt",
        "gt_option",
        "q_uid",
        "prompt",
    )


def assert_visual_builder_isolated() -> None:
    sig = inspect.signature(build_video_condition_visual)
    forbidden = set(visual_builder_forbidden_params())
    leaked = forbidden.intersection(sig.parameters)
    if leaked:
        raise AssertionError(f"visual builder accepts QA/GT params: {sorted(leaked)}")


def is_full_condition(condition: str) -> bool:
    if condition in FULL_CONDITIONS:
        return True
    return bool(_existing_is_full_frame(condition))


def video_level_sample(video_id: str, info) -> SovaSample:
    """Full-video interval only. No VQA fields."""
    last = max(int(info.frame_count) - 1, 0)
    raw = SovaSample(
        sample_id=video_id,
        video_id=video_id,
        activity="",
        start_frame=0,
        end_frame=last,
        start_time=0.0,
        end_time=0.0,
        subdim="",
        dim="",
    )
    return attach_native_times(
        raw,
        native_fps=float(info.native_fps),
        frame_count=int(info.frame_count),
    )


def load_video_rois(
    video_id: str,
    *,
    tracks_root: Path,
    video_roots: Sequence[str | Path],
) -> list[ProposedROI]:
    try:
        return load_proposed_rois(
            video_id=video_id, tracks_root=tracks_root, video_roots=video_roots
        )
    except (FileNotFoundError, ValueError):
        return []


def load_video_motion_records(
    video_id: str,
    *,
    tracks_root: Path,
    video_roots: Sequence[str | Path],
    motion_cfg=None,
) -> dict[int, RoiMotionRecord]:
    packed = records_by_video_from_tracks(
        [video_id],
        tracks_root=tracks_root,
        video_roots=video_roots,
        motion_cfg=motion_cfg,
        need_centroid=True,
    )
    return packed.get(video_id) or {}


def _source_types(job: SpatialJob, condition: str) -> list[str]:
    if is_full_condition(condition):
        return ["full"] * len(job.stamps)
    types: list[str] = ["global"] * len(job.global_stamps)
    for roi in job.rois:
        n = len(job.roi_stamps.get(int(roi.roi_id), ()))
        types.extend([f"roi:{int(roi.roi_id)}"] * n)
    return types


def _identity(job: SpatialJob, condition: str) -> tuple[Any, ...]:
    roi_keys = tuple(
        (int(roi_id), tuple(float(s.timestamp) for s in stamps))
        for roi_id, stamps in sorted(job.roi_stamps.items())
    )
    return (
        condition,
        job.sample.video_id,
        tuple(float(s.timestamp) for s in job.stamps),
        tuple(int(s.native_frame) for s in job.stamps),
        tuple(float(s.timestamp) for s in job.global_stamps),
        tuple(int(s.native_frame) for s in job.global_stamps),
        roi_keys,
        tuple(int(roi.roi_id) for roi in job.rois),
        tuple(job.frame_paths),
    )


def plan_video_job(
    *,
    sample: SovaSample,
    condition: str,
    info,
    rois: Sequence[ProposedROI],
    records: dict[int, RoiMotionRecord],
    tau: float = DEFAULT_TAU,
    dedup_eps: float = GLOBAL_DEDUP_EPS_SEC,
) -> SpatialJob:
    sizes = {sample.video_id: (int(info.width), int(info.height))}
    native = {sample.video_id: (float(info.native_fps), int(info.frame_count))}
    internal = _INTERNAL_CONDITION[condition]
    if condition in FULL_CONDITIONS:
        jobs = build_full_jobs_at_fps(
            [sample],
            video_sizes=sizes,
            native=native,
            sampling_fps=float(_FULL_FPS[condition]),
            condition=internal,
        )
        return jobs[0]
    packed = build_temporal_jobs(
        [sample],
        {sample.video_id: list(rois)},
        {sample.video_id: records},
        video_sizes=sizes,
        native=native,
        conditions=(internal,),
        tau=tau,
        dedup_eps=dedup_eps,
    )
    return packed[internal][0]


def materialize_video_job(
    job: SpatialJob,
    *,
    condition: str,
    video_roots: Sequence[str | Path],
    frames_root: Path,
    jpeg_quality: int,
    decoded_frames: dict[int, Any] | None,
) -> SpatialJob:
    if is_full_condition(condition):
        return materialize_full_frames(
            job,
            video_roots=video_roots,
            frames_root=frames_root,
            jpeg_quality=jpeg_quality,
            decoded_frames=decoded_frames,
        )
    return materialize_job_frames(
        job,
        video_roots=video_roots,
        frames_root=frames_root,
        jpeg_quality=jpeg_quality,
        decoded_frames=decoded_frames,
    )


def build_video_condition_visual(
    *,
    video_id: str,
    video_path: Path,
    condition: str,
    info,
    rois: Sequence[ProposedROI],
    records: dict[int, RoiMotionRecord],
    video_roots: Sequence[str | Path],
    frames_root: Path,
    jpeg_quality: int = 95,
    decoded_frames: dict[int, Any] | None = None,
    tau: float = DEFAULT_TAU,
    dedup_eps: float = GLOBAL_DEDUP_EPS_SEC,
    skip_frames: bool = False,
) -> VisualInput:
    """Build one (video_id, condition) visual representation from the video only."""
    if condition not in USER_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    sample = video_level_sample(video_id, info)
    t0 = time.perf_counter()
    job = plan_video_job(
        sample=sample,
        condition=condition,
        info=info,
        rois=rois,
        records=records,
        tau=tau,
        dedup_eps=dedup_eps,
    )
    if not skip_frames:
        job = materialize_video_job(
            job,
            condition=condition,
            video_roots=video_roots,
            frames_root=frames_root,
            jpeg_quality=jpeg_quality,
            decoded_frames=decoded_frames,
        )
    build_time = float(job.preprocess_elapsed or (time.perf_counter() - t0))
    stamps = list(job.stamps)
    if not is_full_condition(condition):
        stamps = list(job.global_stamps)
        for roi in job.rois:
            stamps.extend(job.roi_stamps.get(int(roi.roi_id), ()))
    visual = VisualInput(
        video_id=video_id,
        condition=condition,
        frame_paths=list(job.frame_paths),
        sampled_timestamps=[float(s.timestamp) for s in stamps],
        sampled_native_frames=[int(s.native_frame) for s in stamps],
        global_timestamps=[float(s.timestamp) for s in job.global_stamps],
        global_native_frames=[int(s.native_frame) for s in job.global_stamps],
        roi_timestamps={
            int(roi_id): [float(s.timestamp) for s in packed]
            for roi_id, packed in job.roi_stamps.items()
        },
        source_types=_source_types(job, condition),
        n_rois=len(job.rois),
        num_input_images=len(job.input_image_resolutions) or len(job.frame_paths) or len(stamps),
        processed_pixels=int(job.total_input_pixels),
        input_image_resolutions=[list(row) for row in job.input_image_resolutions],
        input_build_time=float(build_time),
        cache_hit=False,
        identity=_identity(job, condition),
        job=job,
    )
    return visual
