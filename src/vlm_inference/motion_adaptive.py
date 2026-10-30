"""ROI motion-adaptive temporal sampling.

Does not change Global frames or Stage3 ROI tube geometry. Only the timestamps
at which existing tubes are cropped are selected by a sampling policy.

Motion scores come from Stage3 ``*_roi_tracks.json`` member-event
``per_frame.magnitudes`` (same values as ``*_motion_events.json``).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .magnitude_config import MagnitudeAdaptiveConfig
from .metadata_format import (
    FrameRef,
    VideoBundleMetadata,
    active_rois,
    build_bundle,
    has_metadata_json,
    load_video_bundle_metadata,
    ordered_frame_items,
)
from .roi_margin import (
    _crop_rgb,
    _save_jpeg_rgb,
    load_tube_bboxes,
    tracks_json_path,
)
from .video_io import extract_video_frames, inspect_video, resolve_video_file

logger = logging.getLogger(__name__)

ROI_SPATIAL = "spatial_occupancy"

ROI_PIECEWISE = "piecewise_fps"

ROI_FIXED_1FPS = "fixed_1fps"
ROI_FIXED_2FPS = "fixed_2fps"
ROI_ADAPTIVE_012 = "adaptive_0.5_1_2"
ROI_ADAPTIVE_014 = "adaptive_0.5_1_4"
ROI_ACCUMULATED = "accumulated_motion"
ROI_ACCUMULATED_CAP2FPS = "accumulated_motion_cap2fps"
ROI_BURST_2FPS = "burst_2fps"
ROI_CENTROID_2FPS = "centroid_2fps"
ROI_MAGNITUDE_ADAPTIVE = "motion_magnitude_adaptive"

VALID_ROI_SAMPLING_MODES = frozenset(
    {
        ROI_SPATIAL,
        ROI_PIECEWISE,
        ROI_FIXED_1FPS,
        ROI_FIXED_2FPS,
        ROI_ADAPTIVE_012,
        ROI_ADAPTIVE_014,
        ROI_ACCUMULATED,
        ROI_ACCUMULATED_CAP2FPS,
        ROI_BURST_2FPS,
        ROI_CENTROID_2FPS,
        ROI_MAGNITUDE_ADAPTIVE,
    }
)

FIXED_FPS = {
    ROI_FIXED_1FPS: 1.0,
    ROI_FIXED_2FPS: 2.0,
}
ADAPTIVE_FPS = {
    ROI_ADAPTIVE_012: (0.5, 1.0, 2.0),
    ROI_ADAPTIVE_014: (0.5, 1.0, 4.0),
}

ROI_SAMPLING_METHODS: dict[str, str] = {
    "global_roi_start_box_roi_centroid_2fps": ROI_CENTROID_2FPS,
    "global_roi_start_box_roi_fixed_2fps": ROI_FIXED_2FPS,
    "global_0.25fps_roi_fixed_1fps": ROI_FIXED_1FPS,
    "global_0.25fps_roi_fixed_2fps": ROI_FIXED_2FPS,
    "global_0.25fps_roi_adaptive_0.5_1_2": ROI_ADAPTIVE_012,
    "global_0.25fps_roi_adaptive_0.5_1_4": ROI_ADAPTIVE_014,
    "global_0.25fps_roi_accumulated_motion": ROI_ACCUMULATED,
    "global_0.25fps_roi_accumulated_motion_cap2fps": ROI_ACCUMULATED_CAP2FPS,
    "global_0.25fps_roi_burst_2fps": ROI_BURST_2FPS,
    "global_roi_start_only_roi_fixed_1fps": ROI_FIXED_1FPS,
    "global_roi_start_only_roi_fixed_2fps": ROI_FIXED_2FPS,
    "global_roi_start_only_roi_accumulated_motion": ROI_ACCUMULATED,
    "global_roi_start_only_roi_centroid_2fps": ROI_CENTROID_2FPS,
    "global_roi_video_start_roi_fixed_1fps": ROI_FIXED_1FPS,
    "global_roi_video_start_roi_fixed_2fps": ROI_FIXED_2FPS,
    "global_roi_video_start_roi_accumulated_motion": ROI_ACCUMULATED,
    "global_roi_video_start_roi_centroid_2fps": ROI_CENTROID_2FPS,
    "global_roi_start_only_roi_motion_magnitude_adaptive": ROI_MAGNITUDE_ADAPTIVE,
}

DEFAULT_TAU = 8.0
ACCUMULATED_CAP2FPS_MIN_INTERVAL_SEC = 0.5
BURST_BG_FPS = 1.0
BURST_PEAK_FPS = 2.0
BURST_PERCENTILE = 75.0
BURST_MAX_DURATION = 2.0
BURST_PAD_SEC = 0.25
GROUP_LOW = "low"
GROUP_MID = "mid"
GROUP_HIGH = "high"


@dataclass(frozen=True)
class MotionSample:
    stage3_index: int
    timestamp: float
    native_frame: int
    magnitude: float


@dataclass(frozen=True)
class SelectedStamp:
    timestamp: float
    native_frame: int
    accumulated: float | None = None
    stage3_index: int | None = None


@dataclass(frozen=True)
class BurstWindow:
    t0: float
    t1: float
    duration: float
    peak_timestamp: float
    peak_magnitude: float


@dataclass
class RoiMotionRecord:
    video_id: str
    roi_id: int
    tube_id: int
    t0_stage3: int
    t1_stage3: int
    t0_sec: float
    t1_sec: float
    motion_score: float
    motion_sum: float
    n_motion_frames: int
    percentile: float | None = None
    group: str | None = None
    assigned_fps: float | None = None
    assigned_mode: str = ""
    bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    native_fps: float = 30.0
    frame_count: int = 0
    series: tuple[MotionSample, ...] = ()
    selected: tuple[SelectedStamp, ...] = ()
    burst_windows: tuple[BurstWindow, ...] = ()
    spatial_analysis: dict[str, Any] | None = None
    centroid_analysis: dict[str, Any] | None = None


def parse_roi_sampling_mode(value: str | None) -> str | None:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if not mode or mode in {"none", "null", "off"}:
        return None
    if mode not in VALID_ROI_SAMPLING_MODES:
        raise ValueError(
            f"roi_sampling must be one of {sorted(VALID_ROI_SAMPLING_MODES)}, "
            f"got '{value}'"
        )
    return mode


def sampling_mode_for_method(method: str, configured: str | None) -> str | None:
    named = ROI_SAMPLING_METHODS.get(method)
    if named is not None:
        return named
    return parse_roi_sampling_mode(configured)


def filter_methods_for_roi_sampling(methods: list[str], mode: str | None) -> list[str]:
    parsed = parse_roi_sampling_mode(mode)
    if parsed is None:
        return list(methods)
    matching = [m for m in methods if ROI_SAMPLING_METHODS.get(m) == parsed]
    return matching if matching else list(methods)


def format_timestamp(seconds: float) -> str:
    value = round(float(seconds), 1)
    if abs(value) < 1e-9:
        value = 0.0
    sign = "-" if value < 0 else ""
    body = f"{abs(value):.1f}"
    integer, frac = body.split(".")
    return f"{sign}{int(integer):02d}.{frac}"


def roi_frame_name(roi_id: int, seq: int, timestamp_sec: float) -> str:
    return f"tube{int(roi_id):02d}_frame_{seq:06d}_t{format_timestamp(timestamp_sec)}.jpg"


def condition_root(output_dir: Path | str, mode: str) -> Path:
    return Path(output_dir) / str(mode)


def sampling_video_dir(output_dir: Path | str, mode: str, video_id: str) -> Path:
    return condition_root(output_dir, mode) / video_id


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _rank_percentile(score: float, all_scores: list[float]) -> float:
    if not all_scores:
        return 0.0
    n = sum(1 for s in all_scores if float(s) <= float(score))
    return 100.0 * float(n) / float(len(all_scores))


def assign_group(score: float, p33: float, p66: float) -> str:
    if float(score) <= float(p33):
        return GROUP_LOW
    if float(score) <= float(p66):
        return GROUP_MID
    return GROUP_HIGH


def fps_for_record(mode: str, record: RoiMotionRecord) -> float | None:
    if mode in FIXED_FPS:
        return float(FIXED_FPS[mode])
    if mode in ADAPTIVE_FPS:
        low, mid, high = ADAPTIVE_FPS[mode]
        if record.group == GROUP_LOW:
            return float(low)
        if record.group == GROUP_MID:
            return float(mid)
        return float(high)
    return None


def target_timestamps(t0: float, t1: float, fps: float) -> list[float]:
    if t1 < t0 or fps <= 0:
        return [float(t0)]
    n = int(np.floor((float(t1) - float(t0)) * float(fps) + 1e-9)) + 1
    out = [float(t0) + float(i) / float(fps) for i in range(max(n, 1))]
    if not out or out[-1] < float(t1) - 1e-6:
        out.append(float(t1))
    return [t for t in out if t <= float(t1) + 1e-6]


def select_fixed_or_adaptive(
    *,
    t0: float,
    t1: float,
    fps: float,
    native_fps: float,
    frame_count: int,
) -> list[SelectedStamp]:
    fps_n = float(native_fps) if native_fps > 0 else 30.0
    n = max(int(frame_count), 1)
    selected: list[SelectedStamp] = []
    seen: set[int] = set()
    for target in target_timestamps(t0, t1, fps):
        idx = int(round(float(target) * fps_n))
        idx = min(max(idx, 0), n - 1)
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(
            SelectedStamp(
                timestamp=float(idx) / fps_n,
                native_frame=idx,
                accumulated=None,
                stage3_index=None,
            )
        )
    return selected


def validate_piecewise(default_fps: float, intervals: list[dict[str, float]]) -> list[dict[str, float]]:
    """Validate absolute video-time intervals; touching intervals are allowed."""
    if not math.isfinite(default_fps) or default_fps <= 0:
        raise ValueError("roi_sampling_default_fps must be finite and positive")
    rows = []
    for raw in intervals:
        row = {key: float(raw[key]) for key in ("start_sec", "end_sec", "fps")}
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("ROI sampling interval values must be finite")
        if row["start_sec"] < 0 or row["end_sec"] <= row["start_sec"] or row["fps"] <= 0:
            raise ValueError("ROI intervals require 0 <= start_sec < end_sec and fps > 0")
        rows.append(row)
    rows.sort(key=lambda r: r["start_sec"])
    for left, right in zip(rows, rows[1:]):
        if left["end_sec"] > right["start_sec"]:
            raise ValueError("ROI sampling intervals must not overlap")
    return rows


def select_piecewise(rec: RoiMotionRecord, *, default_fps: float,
                     intervals: list[dict[str, float]]) -> list[SelectedStamp]:
    """Split a tube at interval boundaries and sample each segment from its start.

    Include segment endpoints, snap to native frames, and deduplicate boundaries.
    All times are seconds from the start of the video, not from the ROI start.
    """
    intervals = validate_piecewise(default_fps, intervals)
    boundaries = {rec.t0_sec, rec.t1_sec}
    for row in intervals:
        boundaries.update(t for t in (row["start_sec"], row["end_sec"])
                          if rec.t0_sec < t < rec.t1_sec)
    bounds = sorted(boundaries)
    segments = list(zip(bounds, bounds[1:])) or [(rec.t0_sec, rec.t1_sec)]
    selected = {}
    for lo, hi in segments:
        midpoint = (lo + hi) / 2
        fps = next((r["fps"] for r in intervals
                    if r["start_sec"] <= midpoint < r["end_sec"]), default_fps)
        for stamp in select_fixed_or_adaptive(t0=lo, t1=hi, fps=fps,
                native_fps=rec.native_fps, frame_count=rec.frame_count):
            selected[stamp.native_frame] = stamp
    return sorted(selected.values(), key=lambda stamp: stamp.native_frame)


def select_accumulated(
    series: Sequence[MotionSample],
    *,
    tau: float,
) -> list[SelectedStamp]:
    samples = list(series)
    if not samples:
        return []
    selected: list[SelectedStamp] = []
    acc = 0.0
    last_native: int | None = None
    for i, sample in enumerate(samples):
        if i == 0:
            selected.append(
                SelectedStamp(
                    timestamp=float(sample.timestamp),
                    native_frame=int(sample.native_frame),
                    accumulated=0.0,
                    stage3_index=int(sample.stage3_index),
                )
            )
            last_native = int(sample.native_frame)
            acc = 0.0
            continue
        acc += float(sample.magnitude)
        is_last = i == len(samples) - 1
        if acc >= float(tau) or is_last:
            if last_native == int(sample.native_frame):
                continue
            selected.append(
                SelectedStamp(
                    timestamp=float(sample.timestamp),
                    native_frame=int(sample.native_frame),
                    accumulated=float(acc),
                    stage3_index=int(sample.stage3_index),
                )
            )
            last_native = int(sample.native_frame)
            acc = 0.0
    return selected


def select_accumulated_cap2fps(
    series: Sequence[MotionSample], *, tau: float,
    min_interval_sec: float = ACCUMULATED_CAP2FPS_MIN_INTERVAL_SEC,
) -> list[SelectedStamp]:
    """Accumulated motion with at least 0.5 seconds between selections."""
    samples = list(series)
    if not samples:
        return []
    selected = [SelectedStamp(
        timestamp=float(samples[0].timestamp), native_frame=int(samples[0].native_frame),
        accumulated=0.0, stage3_index=int(samples[0].stage3_index),
    )]
    last_timestamp = float(samples[0].timestamp)
    acc = 0.0
    for i, sample in enumerate(samples[1:], start=1):
        acc += float(sample.magnitude)
        if not (acc >= float(tau) or i == len(samples) - 1):
            continue
        if float(sample.timestamp) - last_timestamp < float(min_interval_sec) - 1e-9:
            continue
        selected.append(SelectedStamp(
            timestamp=float(sample.timestamp), native_frame=int(sample.native_frame),
            accumulated=float(acc), stage3_index=int(sample.stage3_index),
        ))
        last_timestamp = float(sample.timestamp)
        acc = 0.0
    return selected


def detect_burst_windows(
    series: Sequence[MotionSample],
    *,
    percentile: float = BURST_PERCENTILE,
    max_duration: float = BURST_MAX_DURATION,
) -> list[BurstWindow]:
    """Consecutive mag > tube percentile, kept only if duration <= max_duration.

    Long high-magnitude plateaus (trees, traffic) stay on the 1 fps background.
    Short peaks (door, alighting) are the densify targets.
    """
    samples = list(series)
    if not samples:
        return []
    mags = [float(s.magnitude) for s in samples]
    thr = float(np.percentile(np.asarray(mags, dtype=np.float64), percentile))
    if len(samples) > 1:
        dt = (float(samples[-1].timestamp) - float(samples[0].timestamp)) / (
            len(samples) - 1
        )
    else:
        dt = 0.2
    windows: list[BurstWindow] = []
    i = 0
    n = len(samples)
    while i < n:
        if mags[i] <= thr:
            i += 1
            continue
        j = i
        peak_i = i
        while j + 1 < n and mags[j + 1] > thr:
            j += 1
            if mags[j] > mags[peak_i]:
                peak_i = j
        t0 = float(samples[i].timestamp)
        t1 = float(samples[j].timestamp)
        duration = t1 - t0 if j > i else float(dt)
        if duration <= float(max_duration) + 1e-9:
            windows.append(
                BurstWindow(
                    t0=t0,
                    t1=t1,
                    duration=float(duration),
                    peak_timestamp=float(samples[peak_i].timestamp),
                    peak_magnitude=float(mags[peak_i]),
                )
            )
        i = j + 1
    return windows


def _snap_stamp(
    timestamp: float,
    *,
    native_fps: float,
    frame_count: int,
    seen: set[int],
) -> SelectedStamp | None:
    fps_n = float(native_fps) if native_fps > 0 else 30.0
    n = max(int(frame_count), 1)
    idx = int(round(float(timestamp) * fps_n))
    idx = min(max(idx, 0), n - 1)
    if idx in seen:
        return None
    seen.add(idx)
    return SelectedStamp(
        timestamp=float(idx) / fps_n,
        native_frame=idx,
        accumulated=None,
        stage3_index=None,
    )


def select_burst_2fps(
    rec: RoiMotionRecord,
    *,
    percentile: float = BURST_PERCENTILE,
    max_duration: float = BURST_MAX_DURATION,
    pad_sec: float = BURST_PAD_SEC,
) -> tuple[list[SelectedStamp], list[BurstWindow]]:
    """1 fps over the tube; 2 fps (plus the peak frame) inside short bursts."""
    windows = detect_burst_windows(
        rec.series, percentile=percentile, max_duration=max_duration
    )
    padded = [
        (float(w.t0) - float(pad_sec), float(w.t1) + float(pad_sec), w) for w in windows
    ]
    seen: set[int] = set()
    selected: list[SelectedStamp] = []
    for target in target_timestamps(rec.t0_sec, rec.t1_sec, BURST_BG_FPS):
        stamp = _snap_stamp(
            target,
            native_fps=rec.native_fps,
            frame_count=rec.frame_count,
            seen=seen,
        )
        if stamp is not None:
            selected.append(stamp)
    extra_targets: list[float] = []
    for target in target_timestamps(rec.t0_sec, rec.t1_sec, BURST_PEAK_FPS):
        if any(lo <= target <= hi for lo, hi, _ in padded):
            extra_targets.append(float(target))
    extra_targets.extend(float(w.peak_timestamp) for w in windows)
    for target in extra_targets:
        stamp = _snap_stamp(
            target,
            native_fps=rec.native_fps,
            frame_count=rec.frame_count,
            seen=seen,
        )
        if stamp is not None:
            selected.append(stamp)
    selected.sort(key=lambda s: (float(s.timestamp), int(s.native_frame)))
    return selected, windows


def _magnitude_cuts(
    series: Sequence[MotionSample],
    cfg: MagnitudeAdaptiveConfig,
) -> tuple[float, float]:
    if cfg.threshold_source == "absolute":
        return float(cfg.low_threshold), float(cfg.high_threshold)
    positive = [float(s.magnitude) for s in series if float(s.magnitude) > 0]
    if len(positive) < 3:
        return float("inf"), float("inf")
    return _percentile(positive, cfg.low_quantile), _percentile(positive, cfg.high_quantile)


def fps_for_magnitude(magnitude: float, low: float, high: float, cfg: MagnitudeAdaptiveConfig) -> float:
    if float(magnitude) < float(low):
        return float(cfg.min_fps)
    if float(magnitude) < float(high):
        return float(cfg.mid_fps)
    return float(cfg.max_fps)


def select_motion_magnitude_adaptive(
    rec: RoiMotionRecord,
    cfg: MagnitudeAdaptiveConfig | None = None,
) -> list[SelectedStamp]:
    """Keep a nested 0.5 / 1 / 2 fps grid over the whole tube.

    The 0.5 fps grid is always kept, including both tube endpoints.
    1 fps points are kept where the nearest Stage3 magnitude is at least the
    low cut. 2 fps points are kept only at or above the high cut.
    Short magnitude spikes therefore cannot exceed ``max_fps``.
    Native frames are deduplicated. Flow is not recomputed.
    """
    cfg = cfg or MagnitudeAdaptiveConfig()
    samples = list(rec.series)
    if not samples:
        return []
    low, high = _magnitude_cuts(samples, cfg)
    t0 = float(samples[0].timestamp)
    t1 = float(samples[-1].timestamp)

    def magnitude_at(timestamp: float) -> float:
        nearest = min(samples, key=lambda sample: abs(float(sample.timestamp) - float(timestamp)))
        return float(nearest.magnitude)

    selected: dict[int, SelectedStamp] = {}
    for rate in (float(cfg.min_fps), float(cfg.mid_fps), float(cfg.max_fps)):
        for stamp in select_fixed_or_adaptive(
            t0=t0,
            t1=t1,
            fps=rate,
            native_fps=rec.native_fps,
            frame_count=rec.frame_count,
        ):
            allowed = fps_for_magnitude(magnitude_at(stamp.timestamp), low, high, cfg)
            if allowed + 1e-9 >= rate:
                selected.setdefault(int(stamp.native_frame), stamp)
    return sorted(selected.values(), key=lambda stamp: (float(stamp.timestamp), int(stamp.native_frame)))


def load_tracks_payload(tracks_path: Path) -> dict[str, Any]:
    return json.loads(Path(tracks_path).read_text(encoding="utf-8"))


def load_timeline(fusion_npz: Path | str) -> dict[str, np.ndarray]:
    path = Path(fusion_npz)
    with np.load(path) as data:
        if "timestamp_sec_curr" not in data.files or "frame_idx_curr" not in data.files:
            raise KeyError(f"{path}: missing timestamp_sec_curr / frame_idx_curr")
        return {
            "timestamp_sec_curr": np.asarray(data["timestamp_sec_curr"], dtype=np.float64),
            "frame_idx_curr": np.asarray(data["frame_idx_curr"], dtype=np.int32),
        }


def _events_by_id(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    events = payload.get("events") or []
    return {int(ev["event_id"]): ev for ev in events if "event_id" in ev}


def _member_score_lists(
    tube: dict[str, Any],
    score_map: np.ndarray,
    t0: int,
    t1: int,
) -> dict[int, list[float]]:
    """Mean inputs from stored ``MU`` at member cells. No new optical flow."""
    volume = np.asarray(score_map)
    found: dict[int, list[float]] = {}
    for member in tube.get("members") or []:
        if not all(key in member for key in ("y", "x", "t0", "t1")):
            continue
        y, x = int(member["y"]), int(member["x"])
        start, end = int(member["t0"]), int(member["t1"])
        if end < start:
            start, end = end, start
        start = max(start, t0)
        end = min(end, t1)
        if y < 0 or x < 0 or y >= volume.shape[1] or x >= volume.shape[2]:
            continue
        for frame_index in range(start, end + 1):
            if frame_index < 0 or frame_index >= volume.shape[0]:
                continue
            value = float(volume[frame_index, y, x])
            if math.isfinite(value):
                found.setdefault(frame_index, []).append(value)
    return found


def tube_motion_series(
    tube: dict[str, Any],
    events_by_id: dict[int, dict[str, Any]],
    timeline: dict[str, np.ndarray],
    score_map: np.ndarray | None = None,
) -> tuple[float, float, tuple[MotionSample, ...]]:
    """Per-stage3-frame mean cell magnitude from member events; 0 if none.

    Light tracks omit ``per_frame.magnitudes``. When ``score_map`` is the saved
    Stage2 ``MU`` volume, those member cells are read from it instead.
    """
    ts = timeline["timestamp_sec_curr"]
    native = timeline["frame_idx_curr"]
    n = int(ts.shape[0])
    t0 = max(0, min(int(tube.get("t0", 0)), n - 1)) if n else 0
    t1 = max(0, min(int(tube.get("t1", t0)), n - 1)) if n else 0
    if t1 < t0:
        t0, t1 = t1, t0

    mag_lists: dict[int, list[float]] = {}
    for member in tube.get("members") or []:
        event = events_by_id.get(int(member["event_id"]))
        if event is None:
            continue
        for pf in event.get("per_frame") or []:
            fi = int(pf["frame_index"])
            if fi < t0 or fi > t1:
                continue
            mags = pf.get("magnitudes") or []
            if not mags:
                continue
            mag_lists.setdefault(fi, []).extend(float(x) for x in mags)
    if not mag_lists and score_map is not None:
        mag_lists = _member_score_lists(tube, score_map, t0, t1)

    series: list[MotionSample] = []
    observed: list[float] = []
    for fi in range(t0, t1 + 1):
        vals = mag_lists.get(fi) or []
        mag = float(np.mean(vals)) if vals else 0.0
        if vals:
            observed.append(mag)
        series.append(
            MotionSample(
                stage3_index=fi,
                timestamp=float(ts[fi]),
                native_frame=int(native[fi]),
                magnitude=mag,
            )
        )
    score = float(np.mean(observed)) if observed else 0.0
    total = float(np.sum(observed)) if observed else 0.0
    return score, total, tuple(series)


def proposed_roi_ids(proposed_dir: Path) -> list[int]:
    if has_metadata_json(proposed_dir):
        bundle = load_video_bundle_metadata(proposed_dir, require_files=False)
        return [int(roi.roi_id) for roi in active_rois(bundle)]
    ids: set[int] = set()
    for path in proposed_dir.rglob("tube*.jpg"):
        stem = path.stem
        if stem.startswith("tube"):
            try:
                ids.add(int(stem[4:6]))
            except ValueError:
                continue
    return sorted(ids)


def collect_eval_roi_records(
    *,
    videos: Iterable[str],
    proposed_root: Path,
    tracks_root: Path,
    video_root: Path,
    extra_video_roots: Sequence[str | Path] | None = None,
) -> list[RoiMotionRecord]:
    records: list[RoiMotionRecord] = []
    for video_id in videos:
        proposed_dir = Path(proposed_root) / video_id
        tracks_path = tracks_json_path(Path(tracks_root), video_id)
        if not tracks_path.is_file():
            raise FileNotFoundError(f"missing tracks JSON: {tracks_path}")
        payload = load_tracks_payload(tracks_path)
        timeline = load_timeline(payload["fusion_npz"])
        events = _events_by_id(payload)
        score_map = None if _events_have_magnitudes(events) else load_score_map(payload)
        tubes = {int(t["tube_id"]): t for t in (payload.get("tracks") or [])}
        video_path = Path(video_root) / f"{video_id}.mp4"
        if not video_path.is_file():
            video_path = resolve_video_file(
                video_id, [video_root, *(extra_video_roots or [])]
            )
        info = inspect_video(video_path)
        bboxes = load_tube_bboxes(
            tracks_path, frame_width=info.width, frame_height=info.height
        )
        roi_ids = proposed_roi_ids(proposed_dir)
        if not roi_ids:
            roi_ids = sorted(tubes)
            if roi_ids:
                logger.info(
                    "%s: proposed ROI folder missing; using %d track tubes",
                    video_id,
                    len(roi_ids),
                )
        for roi_id in roi_ids:
            tube = tubes.get(int(roi_id))
            if tube is None:
                logger.warning("%s: proposed ROI %s not in tracks; skipped", video_id, roi_id)
                continue
            score, total, series = tube_motion_series(
                tube, events, timeline, score_map=score_map
            )
            bbox = bboxes.get(int(roi_id))
            records.append(
                RoiMotionRecord(
                    video_id=video_id,
                    roi_id=int(roi_id),
                    tube_id=int(tube["tube_id"]),
                    t0_stage3=int(tube.get("t0", 0)),
                    t1_stage3=int(tube.get("t1", 0)),
                    t0_sec=float(series[0].timestamp) if series else 0.0,
                    t1_sec=float(series[-1].timestamp) if series else 0.0,
                    motion_score=score,
                    motion_sum=total,
                    n_motion_frames=sum(1 for s in series if s.magnitude > 0),
                    bbox_px=bbox.bbox_px if bbox else (0, 0, 0, 0),
                    native_fps=float(info.native_fps),
                    frame_count=int(info.frame_count),
                    series=series,
                )
            )
    return records


def apply_percentiles(records: list[RoiMotionRecord]) -> tuple[float, float]:
    scores = [float(r.motion_score) for r in records]
    p33 = _percentile(scores, 100.0 / 3.0)
    p66 = _percentile(scores, 200.0 / 3.0)
    for rec in records:
        rec.percentile = _rank_percentile(rec.motion_score, scores)
        rec.group = assign_group(rec.motion_score, p33, p66)
    return p33, p66


def load_score_map(payload: dict[str, Any]) -> np.ndarray | None:
    """Saved fused magnitude volume, when the tracks JSON points at one."""
    path = payload.get("fusion_npz")
    if not path or not Path(str(path)).is_file():
        return None
    key = str(payload.get("score_key") or "MU_fused")
    with np.load(path) as data:
        if key not in data.files:
            return None
        return np.asarray(data[key], dtype=np.float64)


def _events_have_magnitudes(events_by_id: dict[int, dict[str, Any]]) -> bool:
    for event in events_by_id.values():
        for frame in event.get("per_frame") or []:
            if frame.get("magnitudes"):
                return True
    return False


def assign_policy(
    records: list[RoiMotionRecord],
    *,
    mode: str,
    tau: float,
    default_fps: float = 1.0,
    intervals: list[dict[str, float]] | None = None,
    magnitude_cfg: MagnitudeAdaptiveConfig | None = None,
) -> list[RoiMotionRecord]:
    mode = parse_roi_sampling_mode(mode) or mode
    for rec in records:
        rec.assigned_mode = mode
        rec.burst_windows = ()
        if mode == ROI_SPATIAL:
            if rec.spatial_analysis is None:
                raise ValueError("spatial analysis must be attached before sampling")
            rec.assigned_fps = None
            rec.selected = tuple(select_piecewise(
                rec, default_fps=rec.spatial_analysis["config"]["base_fps"],
                intervals=rec.spatial_analysis["high_fps_intervals"]))
        elif mode == ROI_CENTROID_2FPS:
            if rec.centroid_analysis is None:
                raise ValueError("centroid analysis must be attached before sampling")
            rec.assigned_fps = None
            rec.selected = tuple(select_piecewise(
                rec, default_fps=rec.centroid_analysis["config"]["base_fps"],
                intervals=rec.centroid_analysis["high_fps_intervals"]))
            rec.burst_windows = tuple(
                BurstWindow(
                    t0=float(row["start_sec"]),
                    t1=float(row["end_sec"]),
                    duration=float(row["end_sec"]) - float(row["start_sec"]),
                    peak_timestamp=float(row["start_sec"]),
                    peak_magnitude=float(
                        rec.centroid_analysis["config"].get(
                            "centroid_displacement_threshold"
                        )
                        or 0.0
                    ),
                )
                for row in rec.centroid_analysis["high_fps_intervals"]
            )
        elif mode == ROI_PIECEWISE:
            rec.assigned_fps = None
            rec.selected = tuple(select_piecewise(rec, default_fps=default_fps, intervals=intervals or []))
        elif mode == ROI_ACCUMULATED:
            rec.assigned_fps = None
            rec.selected = tuple(select_accumulated(rec.series, tau=tau))
        elif mode == ROI_ACCUMULATED_CAP2FPS:
            rec.assigned_fps = 2.0
            rec.selected = tuple(select_accumulated_cap2fps(rec.series, tau=tau))
        elif mode == ROI_BURST_2FPS:
            selected, windows = select_burst_2fps(rec)
            rec.selected = tuple(selected)
            rec.burst_windows = tuple(windows)
            dur = float(rec.t1_sec) - float(rec.t0_sec)
            if dur > 1e-6 and len(selected) > 1:
                rec.assigned_fps = (len(selected) - 1) / dur
            else:
                rec.assigned_fps = float(BURST_BG_FPS)
        elif mode == ROI_MAGNITUDE_ADAPTIVE:
            cfg_mag = magnitude_cfg or MagnitudeAdaptiveConfig()
            selected = select_motion_magnitude_adaptive(rec, cfg_mag)
            rec.selected = tuple(selected)
            dur = float(rec.t1_sec) - float(rec.t0_sec)
            if dur > 1e-6 and len(selected) > 1:
                rec.assigned_fps = (len(selected) - 1) / dur
            else:
                rec.assigned_fps = float(cfg_mag.min_fps)
        else:
            fps = fps_for_record(mode, rec)
            rec.assigned_fps = fps
            rec.selected = tuple(
                select_fixed_or_adaptive(
                    t0=rec.t0_sec,
                    t1=rec.t1_sec,
                    fps=float(fps or 1.0),
                    native_fps=rec.native_fps,
                    frame_count=rec.frame_count,
                )
            )
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record_public(rec: RoiMotionRecord) -> dict[str, Any]:
    row: dict[str, Any] = {
        "video_id": rec.video_id,
        "roi_id": rec.roi_id,
        "tube_id": rec.tube_id,
        "t0_sec": rec.t0_sec,
        "t1_sec": rec.t1_sec,
        "motion_score": rec.motion_score,
        "motion_sum": rec.motion_sum,
        "n_motion_frames": rec.n_motion_frames,
        "percentile": rec.percentile,
        "group": rec.group,
        "assigned_fps": rec.assigned_fps,
        "assigned_mode": rec.assigned_mode,
        "n_selected": len(rec.selected),
        "selected_timestamps": [s.timestamp for s in rec.selected],
    }
    if rec.spatial_analysis is not None:
        row["high_fps_intervals"] = rec.spatial_analysis["high_fps_intervals"]
        row["spatial_occupancy_config"] = rec.spatial_analysis["config"]
    if rec.centroid_analysis is not None:
        row["high_fps_intervals"] = rec.centroid_analysis["high_fps_intervals"]
        row["centroid_motion_config"] = rec.centroid_analysis["config"]
        row["trigger_timestamps"] = rec.centroid_analysis.get("trigger_timestamps")
        row["n_1fps_equivalent"] = rec.centroid_analysis.get("n_1fps_equivalent")
        row["n_additional"] = rec.centroid_analysis.get("n_additional")
    if rec.assigned_mode == ROI_BURST_2FPS:
        row["burst_percentile"] = BURST_PERCENTILE
        row["burst_max_duration"] = BURST_MAX_DURATION
        row["n_bursts"] = len(rec.burst_windows)
        row["bursts"] = [
            {
                "t0": w.t0,
                "t1": w.t1,
                "duration": w.duration,
                "peak_timestamp": w.peak_timestamp,
                "peak_magnitude": w.peak_magnitude,
            }
            for w in rec.burst_windows
        ]
    if rec.assigned_mode == ROI_CENTROID_2FPS:
        row["n_centroid_windows"] = len(rec.burst_windows)
    if rec.assigned_mode in {ROI_ACCUMULATED, ROI_ACCUMULATED_CAP2FPS}:
        row["selected_accumulated"] = [s.accumulated for s in rec.selected]
        row["series"] = [
            {
                "stage3_index": s.stage3_index,
                "timestamp": s.timestamp,
                "magnitude": s.magnitude,
            }
            for s in rec.series
        ]
    return row


def _assignments_payload(
    records: list[RoiMotionRecord],
    *,
    mode: str,
    p33: float,
    p66: float,
    tau: float,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "tau": tau if mode in {ROI_ACCUMULATED, ROI_ACCUMULATED_CAP2FPS} else None,
        "centroid_motion": (
            records[0].centroid_analysis["config"]
            if mode == ROI_CENTROID_2FPS and records and records[0].centroid_analysis
            else None
        ),
        "minimum_selected_interval_sec": (
            ACCUMULATED_CAP2FPS_MIN_INTERVAL_SEC if mode == ROI_ACCUMULATED_CAP2FPS else None
        ),
        "score_definition": (
            "mean over stage3 frames of mean member-event cell magnitudes "
            "(tracks.json events[].per_frame.magnitudes); empty frames ignored in the mean"
        ),
        "percentile_thresholds": {"p33": p33, "p66": p66, "n_rois": len(records)},
        "rois": [_record_public(r) for r in records],
    }


def _write_assignments_csv(path: Path, records: list[RoiMotionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "roi_id",
        "motion_score",
        "percentile",
        "group",
        "assigned_fps",
        "assigned_mode",
        "n_selected",
        "t0_sec",
        "t1_sec",
        "n_bursts",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "video_id": rec.video_id,
                    "roi_id": rec.roi_id,
                    "motion_score": f"{rec.motion_score:.6f}",
                    "percentile": "" if rec.percentile is None else f"{rec.percentile:.2f}",
                    "group": rec.group or "",
                    "assigned_fps": "" if rec.assigned_fps is None else rec.assigned_fps,
                    "assigned_mode": rec.assigned_mode,
                    "n_selected": len(rec.selected),
                    "t0_sec": f"{rec.t0_sec:.3f}",
                    "t1_sec": f"{rec.t1_sec:.3f}",
                    "n_bursts": len(rec.burst_windows),
                }
            )


def _write_summary(
    path_json: Path,
    path_csv: Path,
    records: list[RoiMotionRecord],
    *,
    mode: str,
) -> dict[str, Any]:
    by_video: dict[str, list[RoiMotionRecord]] = {}
    for rec in records:
        by_video.setdefault(rec.video_id, []).append(rec)
    videos = []
    for video_id, rows in by_video.items():
        n_roi_frames = sum(len(r.selected) for r in rows)
        videos.append(
            {
                "video_id": video_id,
                "num_rois": len(rows),
                "num_roi_frames": n_roi_frames,
                "rois": [
                    {
                        "roi_id": r.roi_id,
                        "assigned_fps": r.assigned_fps,
                        "assigned_mode": r.assigned_mode,
                        "group": r.group,
                        "motion_score": r.motion_score,
                        "num_frames": len(r.selected),
                    }
                    for r in rows
                ],
            }
        )
    payload = {
        "mode": mode,
        "num_videos": len(videos),
        "num_rois": len(records),
        "num_roi_frames": sum(v["num_roi_frames"] for v in videos),
        "videos": videos,
    }
    _write_json(path_json, payload)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "video_id",
                "roi_id",
                "assigned_fps",
                "assigned_mode",
                "group",
                "num_roi_frames",
                "video_num_roi_frames",
                "video_num_rois",
            ]
        )
        for video in videos:
            for roi in video["rois"]:
                writer.writerow(
                    [
                        video["video_id"],
                        roi["roi_id"],
                        roi["assigned_fps"] if roi["assigned_fps"] is not None else "",
                        roi["assigned_mode"],
                        roi["group"] or "",
                        roi["num_frames"],
                        video["num_roi_frames"],
                        video["num_rois"],
                    ]
                )
    return payload


def _write_roi_metadata(
    video_dir: Path,
    *,
    video_id: str,
    records: list[RoiMotionRecord],
    saved: dict[int, list[FrameRef]],
) -> None:
    rois = []
    for rec in records:
        frames = saved.get(rec.roi_id) or []
        rois.append(
            {
                "roi_id": rec.roi_id,
                "time_range": [
                    float(frames[0].timestamp) if frames else rec.t0_sec,
                    float(frames[-1].timestamp) if frames else rec.t1_sec,
                ],
                "source_region": "motion-adaptive sample",
                "normalized_bbox": [0.0, 0.0, 1.0, 1.0],
                "num_frames": len(frames),
                "frames": [
                    {
                        "frame_index": fr.frame_index,
                        "timestamp": fr.timestamp,
                        "file": fr.file,
                    }
                    for fr in frames
                ],
            }
        )
    _write_json(
        video_dir / "metadata.json",
        {
            "video_id": video_id,
            "global": {"coverage": "full scene", "num_frames": 0, "frames": []},
            "rois": rois,
        },
    )


def _tube_board_label(rec: RoiMotionRecord) -> str:
    if rec.assigned_mode == ROI_ACCUMULATED_CAP2FPS:
        return f"Tube {rec.roi_id:02d}  accum cap2fps"
    if rec.assigned_mode == ROI_ACCUMULATED:
        return f"Tube {rec.roi_id:02d}  accumulated motion"
    if rec.assigned_mode == ROI_BURST_2FPS:
        n_burst = len(rec.burst_windows)
        return f"Tube {rec.roi_id:02d}  1fps+burst2 ({n_burst})"
    if rec.assigned_mode == ROI_CENTROID_2FPS:
        n_win = len(rec.burst_windows)
        return f"Tube {rec.roi_id:02d}  1fps+centroid2 ({n_win})"
    fps = rec.assigned_fps
    fps_txt = f"{fps:g}fps" if fps is not None else rec.assigned_mode
    group = f" {rec.group}" if rec.group else ""
    return f"Tube {rec.roi_id:02d}  {fps_txt}{group}"


def _render_boards(
    *,
    video_id: str,
    method: str,
    items: list[tuple[str, Path]],
    out_dir: Path,
    tube_labels: dict[int, str],
) -> None:
    from argparse import Namespace

    from .export_vlm_inputs import _viz_module

    viz = _viz_module()
    args = Namespace(
        thumb_h=72,
        frame_gap=8,
        row_gap=10,
        section_gap=36,
        label_w=210,
        pad=24,
        canvas_w=2400,
        section_pad=12,
    )
    grouped = viz.render_grouped_canvas(
        items, video_id=video_id, method=method, args=args, tube_labels=tube_labels
    )
    order = viz.render_order_canvas(items, video_id=video_id, method=method, args=args)
    viz.cv2.imwrite(str(out_dir / "board_grouped.png"), grouped)
    viz.cv2.imwrite(str(out_dir / "board_order.png"), order)
    boards = out_dir.parent / "boards"
    boards.mkdir(parents=True, exist_ok=True)
    viz.cv2.imwrite(str(boards / f"{video_id}__grouped.png"), grouped)
    viz.cv2.imwrite(str(boards / f"{video_id}__order.png"), order)


def _materialize_video_crops(
    *,
    records: list[RoiMotionRecord],
    video_path: Path,
    video_dir: Path,
    jpeg_quality: int,
    overwrite: bool,
    decoded: dict[int, Any] | None = None,
) -> dict[int, list[FrameRef]]:
    wanted: set[int] = set()
    for rec in records:
        for stamp in rec.selected:
            wanted.add(int(stamp.native_frame))
    if decoded is None:
        decoded = extract_video_frames(video_path, wanted)
    frames_dir = video_dir / "frames"
    if overwrite and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[int, list[FrameRef]] = {}
    for rec in records:
        roi_dir = video_dir / f"roi_{rec.roi_id:02d}"
        roi_dir.mkdir(parents=True, exist_ok=True)
        refs: list[FrameRef] = []
        for seq, stamp in enumerate(rec.selected, start=1):
            name = roi_frame_name(rec.roi_id, seq, stamp.timestamp)
            dest = roi_dir / name
            flat = frames_dir / name
            if overwrite or not dest.is_file():
                rgb = decoded[int(stamp.native_frame)]
                crop = _crop_rgb(rgb, rec.bbox_px)
                _save_jpeg_rgb(dest, crop, quality=jpeg_quality)
            if overwrite and flat.exists():
                flat.unlink()
            if not flat.exists():
                try:
                    flat.hardlink_to(dest)
                except OSError:
                    shutil.copy2(dest, flat)
            rel = dest.relative_to(video_dir).as_posix()
            refs.append(
                FrameRef(
                    frame_index=seq,
                    timestamp=float(stamp.timestamp),
                    file=rel,
                    path=dest.resolve(),
                )
            )
        saved[rec.roi_id] = refs
    return saved


def _prepare_spatial_analysis(records: list[RoiMotionRecord], cfg) -> None:
    """Adapter: Stage3 cells -> independent detector -> per-ROI intervals."""
    from .spatial_occupancy import (
        active_sets_from_events, analyze_spatial_occupancy,
        plot_spatial_occupancy, write_spatial_debug,
    )
    for video_id in dict.fromkeys(r.video_id for r in records):
        payload = load_tracks_payload(tracks_json_path(Path(cfg.feasibility.tracks_root), video_id))
        events = _events_by_id(payload)
        tubes = {int(t['tube_id']): t for t in payload['tracks']}
        cell_px = int(payload.get('cell_px') or 64)
        if cell_px <= 0:
            raise ValueError('Stage3 cell_px must be positive')
        for rec in (r for r in records if r.video_id == video_id):
            indices = [s.stage3_index for s in rec.series]
            active = active_sets_from_events(tubes[rec.tube_id], events, indices)
            # Exactly the fixed crop bbox, including clipped cells on video edges.
            bbox = [v / cell_px for v in rec.bbox_px]
            result = analyze_spatial_occupancy(
                [s.timestamp for s in rec.series], active, bbox, cfg.experiment.spatial_occupancy)
            result.update(video_id=video_id, roi_id=rec.roi_id,
                          fixed_bbox_px=list(rec.bbox_px), cell_px=cell_px,
                          stage3_indices=indices)
            rec.spatial_analysis = result
            directory = sampling_video_dir(cfg.output.dir, ROI_SPATIAL, video_id) / 'debug'
            stem = f'roi_{rec.roi_id:02d}_spatial'
            write_spatial_debug(result, directory, stem)
            plot_spatial_occupancy(result, directory / f'{stem}.png',
                                   title=f'{video_id} / ROI {rec.roi_id}')


def _prepare_centroid_analysis(records: list[RoiMotionRecord], cfg) -> None:
    """Adapter: Stage3 cells -> centroid displacement detector -> 2fps intervals."""
    from .centroid_motion import analyze_centroid_motion
    from .spatial_occupancy import active_sets_from_events, occupancy_features

    for video_id in dict.fromkeys(r.video_id for r in records):
        payload = load_tracks_payload(tracks_json_path(Path(cfg.feasibility.tracks_root), video_id))
        events = _events_by_id(payload)
        tubes = {int(t["tube_id"]): t for t in payload["tracks"]}
        cell_px = int(payload.get("cell_px") or 64)
        if cell_px <= 0:
            raise ValueError("Stage3 cell_px must be positive")
        for rec in (r for r in records if r.video_id == video_id):
            indices = [s.stage3_index for s in rec.series]
            active = active_sets_from_events(tubes[rec.tube_id], events, indices)
            bbox_grid = [v / cell_px for v in rec.bbox_px]
            counts, centroid, *_ = occupancy_features(active, bbox_grid)
            result = analyze_centroid_motion(
                [s.timestamp for s in rec.series],
                centroid,
                rec.bbox_px,
                cfg.experiment.centroid_motion,
                t0=rec.t0_sec,
                t1=rec.t1_sec,
            )
            result.update(
                video_id=video_id,
                roi_id=rec.roi_id,
                cell_px=cell_px,
                n_active=[int(v) for v in counts],
                stage3_indices=indices,
            )
            rec.centroid_analysis = result


def _write_centroid_sampling_debug(records: list[RoiMotionRecord], cfg) -> None:
    from .centroid_motion import write_centroid_debug

    for rec in records:
        if rec.centroid_analysis is None:
            continue
        n_1fps = len(
            select_fixed_or_adaptive(
                t0=rec.t0_sec,
                t1=rec.t1_sec,
                fps=1.0,
                native_fps=rec.native_fps,
                frame_count=rec.frame_count,
            )
        )
        rec.centroid_analysis["n_selected"] = len(rec.selected)
        rec.centroid_analysis["n_1fps_equivalent"] = n_1fps
        rec.centroid_analysis["n_additional"] = len(rec.selected) - n_1fps
        rec.centroid_analysis["selected_timestamps"] = [
            float(s.timestamp) for s in rec.selected
        ]
        directory = (
            sampling_video_dir(cfg.output.dir, ROI_CENTROID_2FPS, rec.video_id) / "debug"
        )
        write_centroid_debug(
            rec.centroid_analysis, directory, f"roi_{rec.roi_id:02d}_centroid"
        )


def sampling_outputs_ready(cfg, modes: list[str] | None = None) -> bool:
    """True when every video already has metadata.json for the needed modes."""
    feas = cfg.feasibility
    if feas is None:
        return False
    needed: list[str] = []
    if modes:
        needed = [parse_roi_sampling_mode(m) or m for m in modes]
    else:
        for method in feas.methods:
            mode = sampling_mode_for_method(method, cfg.experiment.roi_sampling)
            if mode and mode not in needed:
                needed.append(mode)
        if not needed and cfg.experiment.roi_sampling:
            needed = [cfg.experiment.roi_sampling]
    if not needed:
        return True
    out_root = Path(cfg.output.dir)
    return all(
        (sampling_video_dir(out_root, mode, video_id) / "metadata.json").is_file()
        for mode in needed
        for video_id in feas.videos
    )


def prepare_motion_adaptive_sampling(
    cfg,
    *,
    modes: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score all eval ROIs, assign fps, crop, write boards/summaries."""
    feas = cfg.feasibility
    if feas is None:
        raise ValueError("feasibility config is required")
    exp = cfg.experiment
    needed: list[str] = []
    if modes:
        needed = [parse_roi_sampling_mode(m) or m for m in modes]
    else:
        for method in feas.methods:
            mode = sampling_mode_for_method(method, exp.roi_sampling)
            if mode and mode not in needed:
                needed.append(mode)
        if not needed and exp.roi_sampling:
            needed = [exp.roi_sampling]
    if not needed:
        return {}

    if ROI_PIECEWISE in needed:
        validate_piecewise(exp.roi_sampling_default_fps, exp.roi_sampling_intervals)
        # Regenerate to prevent stale crops after a schedule edit in the same output directory.
        overwrite = True

    if ROI_SPATIAL in needed:
        overwrite = True  # Parameters and active-block input may have changed.
    if ROI_CENTROID_2FPS in needed:
        overwrite = True

    tau = float(exp.roi_sampling_tau)
    records = collect_eval_roi_records(
        videos=feas.videos,
        proposed_root=Path(feas.datasets.proposed_roi_w_fps),
        tracks_root=Path(feas.tracks_root),
        video_root=Path(feas.datasets.video_root),
        extra_video_roots=feas.datasets.video_roots,
    )
    p33, p66 = apply_percentiles(records)
    logger.info(
        "ROI motion scores n=%d p33=%.4f p66=%.4f tau=%.3f modes=%s",
        len(records),
        p33,
        p66,
        tau,
        needed,
    )

    out_root = Path(cfg.output.dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {"p33": p33, "p66": p66, "tau": tau, "conditions": {}}

    mode_rows: dict[str, list[RoiMotionRecord]] = {}
    for mode in needed:
        clones: list[RoiMotionRecord] = []
        for rec in records:
            clones.append(
                RoiMotionRecord(
                    video_id=rec.video_id,
                    roi_id=rec.roi_id,
                    tube_id=rec.tube_id,
                    t0_stage3=rec.t0_stage3,
                    t1_stage3=rec.t1_stage3,
                    t0_sec=rec.t0_sec,
                    t1_sec=rec.t1_sec,
                    motion_score=rec.motion_score,
                    motion_sum=rec.motion_sum,
                    n_motion_frames=rec.n_motion_frames,
                    percentile=rec.percentile,
                    group=rec.group,
                    bbox_px=rec.bbox_px,
                    native_fps=rec.native_fps,
                    frame_count=rec.frame_count,
                    series=rec.series,
                )
            )
        if mode == ROI_SPATIAL:
            _prepare_spatial_analysis(clones, cfg)
        if mode == ROI_CENTROID_2FPS:
            _prepare_centroid_analysis(clones, cfg)
        assign_policy(
            clones,
            mode=mode,
            tau=tau,
            default_fps=exp.roi_sampling_default_fps,
            intervals=exp.roi_sampling_intervals,
            magnitude_cfg=getattr(exp, "magnitude_adaptive", None),
        )
        if mode == ROI_CENTROID_2FPS:
            _write_centroid_sampling_debug(clones, cfg)
        mode_rows[mode] = clones
        cond_dir = condition_root(out_root, mode)
        cond_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            cond_dir / "assignments.json",
            _assignments_payload(clones, mode=mode, p33=p33, p66=p66, tau=tau),
        )
        _write_assignments_csv(cond_dir / "assignments.csv", clones)
        summaries["conditions"][mode] = _write_summary(
            cond_dir / "summary.json", cond_dir / "summary.csv", clones, mode=mode
        )

    for video_id in feas.videos:
        video_ready = all(
            (sampling_video_dir(out_root, mode, video_id) / "metadata.json").is_file()
            for mode in needed
        )
        if video_ready and not overwrite:
            logger.info("%s: sampling already exists, skip decode", video_id)
            continue
        wanted: set[int] = set()
        for clones in mode_rows.values():
            for rec in clones:
                if rec.video_id != video_id:
                    continue
                for stamp in rec.selected:
                    wanted.add(int(stamp.native_frame))
        video_path = Path(feas.datasets.video_root) / f"{video_id}.mp4"
        if not video_path.is_file():
            video_path = resolve_video_file(
                video_id, feas.datasets.all_video_roots()
            )
        logger.info("%s: decoding %d unique frames", video_id, len(wanted))
        decoded = extract_video_frames(video_path, wanted)
        proposed_dir = Path(feas.datasets.proposed_roi_w_fps) / video_id
        proposed = None
        if has_metadata_json(proposed_dir):
            try:
                proposed = load_video_bundle_metadata(proposed_dir, require_files=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: proposed metadata unusable (%s)", video_id, exc)

        for mode, clones in mode_rows.items():
            rows = [r for r in clones if r.video_id == video_id]
            if not rows:
                continue
            video_dir = sampling_video_dir(out_root, mode, video_id)
            video_dir.mkdir(parents=True, exist_ok=True)
            saved = _materialize_video_crops(
                records=rows,
                video_path=video_path,
                video_dir=video_dir,
                jpeg_quality=int(feas.jpeg_quality),
                overwrite=overwrite,
                decoded=decoded,
            )
            _write_roi_metadata(video_dir, video_id=video_id, records=rows, saved=saved)
            _write_json(
                video_dir / "roi_log.json",
                {
                    "video_id": video_id,
                    "mode": mode,
                    "default_fps": exp.roi_sampling_default_fps if mode == ROI_PIECEWISE else None,
                    "intervals": exp.roi_sampling_intervals if mode == ROI_PIECEWISE else None,
                    "tau": tau if mode in {ROI_ACCUMULATED, ROI_ACCUMULATED_CAP2FPS} else None,
                    "centroid_motion": (
                        exp.centroid_motion.__dict__ if mode == ROI_CENTROID_2FPS else None
                    ),
                    "rois": [_record_public(r) for r in rows],
                },
            )
            sampled = load_video_bundle_metadata(video_dir, require_files=True)
            bundle = (
                build_bundle(
                    global_metadata=proposed, roi_metadata=sampled, video_id=video_id
                )
                if proposed is not None
                else sampled
            )
            items = [
                (item.text, item.image_path)
                for item in ordered_frame_items(bundle, input_order=exp.input_order)
                if item.image_path is not None
            ]
            labels = {r.roi_id: _tube_board_label(r) for r in rows}
            method = next(
                (m for m, md in ROI_SAMPLING_METHODS.items() if md == mode), mode
            )
            try:
                _render_boards(
                    video_id=video_id,
                    method=method,
                    items=items,
                    out_dir=video_dir,
                    tube_labels=labels,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("board failed for %s / %s: %s", video_id, mode, exc)
            logger.info(
                "%s / %s: rois=%d frames=%d",
                video_id,
                mode,
                len(rows),
                sum(len(r.selected) for r in rows),
            )

    _write_json(out_root / "assignments.json", summaries)
    logger.info("Wrote motion-adaptive sampling under %s", out_root)
    return summaries


def apply_sampled_rois(
    bundle: VideoBundleMetadata,
    sampled_dir: Path,
    *,
    video_id: str,
) -> VideoBundleMetadata:
    sampled = load_video_bundle_metadata(sampled_dir, require_files=True)
    return build_bundle(global_metadata=bundle, roi_metadata=sampled, video_id=video_id)


def load_roi_sampling_labels(sampled_dir: Path) -> dict[int, str]:
    log_path = Path(sampled_dir) / "roi_log.json"
    if not log_path.is_file():
        return {}
    data = json.loads(log_path.read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    for row in data.get("rois") or []:
        rec = RoiMotionRecord(
            video_id=str(data.get("video_id") or ""),
            roi_id=int(row["roi_id"]),
            tube_id=int(row.get("tube_id") or row["roi_id"]),
            t0_stage3=0,
            t1_stage3=0,
            t0_sec=float(row.get("t0_sec") or 0),
            t1_sec=float(row.get("t1_sec") or 0),
            motion_score=float(row.get("motion_score") or 0),
            motion_sum=float(row.get("motion_sum") or 0),
            n_motion_frames=int(row.get("n_motion_frames") or 0),
            percentile=row.get("percentile"),
            group=row.get("group"),
            assigned_fps=row.get("assigned_fps"),
            assigned_mode=str(row.get("assigned_mode") or data.get("mode") or ""),
        )
        n_win = int(row.get("n_centroid_windows") or row.get("n_bursts") or 0)
        if n_win:
            rec.burst_windows = tuple(
                BurstWindow(t0=0.0, t1=0.0, duration=0.0, peak_timestamp=0.0, peak_magnitude=0.0)
                for _ in range(n_win)
            )
        labels[rec.roi_id] = _tube_board_label(rec)
    return labels
