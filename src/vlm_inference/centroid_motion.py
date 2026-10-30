"""ROI centroid displacement for local 2fps densify.

Equal-weight active-block centroid inside the fixed ROI. Consecutive valid
centroids yield a dimensionless displacement (pixel distance / ROI diagonal).
Missing centroids are not interpolated and are not treated as the origin.

If D_t >= tau_c, the window [t - pad, t + pad] (clipped to the tube) is
sampled at high_fps. The rest of the tube stays at base_fps. The default pad
is 0.5 s, so the default window is [t - 0.5, t + 0.5]. Overlapping or touching
windows are unioned. There is no temporal smoothing, hysteresis, or min-duration.
This centroid is the spatial center of active cells, not the temporal center
of the magnitude series.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CentroidMotionConfig:
    centroid_displacement_threshold: float = 0.02
    pad_sec: float = 0.5
    base_fps: float = 1.0
    high_fps: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.centroid_displacement_threshold,
            self.pad_sec,
            self.base_fps,
            self.high_fps,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("centroid_motion parameters must be finite")
        if self.centroid_displacement_threshold < 0:
            raise ValueError("centroid_displacement_threshold must be >= 0")
        if self.pad_sec < 0:
            raise ValueError("pad_sec must be >= 0")
        if not 0 < self.base_fps < self.high_fps:
            raise ValueError("require 0 < base_fps < high_fps")


def roi_diagonal_px(bbox_px: Sequence[float]) -> float:
    x0, y0, x1, y1 = (float(v) for v in bbox_px)
    return math.hypot(x1 - x0, y1 - y0)


def centroid_displacement(
    prev: Sequence[float],
    curr: Sequence[float],
    bbox_px: Sequence[float],
) -> float | None:
    """Pixel displacement / ROI diagonal. No delta_t. None if either centroid is invalid."""
    if not all(math.isfinite(float(v)) for v in (*prev, *curr)):
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox_px)
    width = x1 - x0
    height = y1 - y0
    diag = math.hypot(width, height)
    if diag <= 0:
        return None
    dpx = (float(curr[0]) - float(prev[0])) * width
    dpy = (float(curr[1]) - float(prev[1])) * height
    return math.hypot(dpx, dpy) / diag


def union_intervals(
    intervals: Iterable[dict[str, float]],
    *,
    fps: float,
) -> list[dict[str, float]]:
    """Union overlapping or touching intervals. Gaps stay separate."""
    rows = sorted(
        (
            {
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "fps": float(row.get("fps", fps)),
            }
            for row in intervals
            if float(row["end_sec"]) > float(row["start_sec"])
        ),
        key=lambda row: row["start_sec"],
    )
    merged: list[dict[str, float]] = []
    for row in rows:
        if merged and row["start_sec"] <= merged[-1]["end_sec"] + 1e-9:
            if row["end_sec"] > merged[-1]["end_sec"]:
                merged[-1]["end_sec"] = row["end_sec"]
            continue
        merged.append({"start_sec": row["start_sec"], "end_sec": row["end_sec"], "fps": float(fps)})
    return merged


def trigger_windows(
    timestamps: Sequence[float],
    *,
    pad_sec: float,
    fps: float,
    clip_t0: float,
    clip_t1: float,
) -> list[dict[str, float]]:
    windows = []
    for t in timestamps:
        start = max(float(clip_t0), float(t) - float(pad_sec))
        end = min(float(clip_t1), float(t) + float(pad_sec))
        if end > start:
            windows.append({"start_sec": start, "end_sec": end, "fps": float(fps)})
    return union_intervals(windows, fps=fps)


def _in_high_fps(timestamp: float, intervals: Sequence[dict[str, float]]) -> bool:
    t = float(timestamp)
    return any(row["start_sec"] <= t <= row["end_sec"] for row in intervals)


def analyze_centroid_motion(
    timestamps: Sequence[float],
    centroids: Sequence[Sequence[float]] | np.ndarray,
    bbox_px: Sequence[float],
    cfg: CentroidMotionConfig | None = None,
    *,
    t0: float | None = None,
    t1: float | None = None,
) -> dict:
    cfg = cfg or CentroidMotionConfig()
    ts = np.asarray(timestamps, dtype=float)
    xy = np.asarray(centroids, dtype=float)
    if ts.ndim != 1 or len(ts) == 0 or xy.shape != (len(ts), 2):
        raise ValueError("timestamps and centroids must be nonempty and aligned")
    if not np.all(np.isfinite(ts)) or (len(ts) > 1 and np.any(np.diff(ts) <= 0)):
        raise ValueError("timestamps must be finite and strictly increasing")

    clip_t0 = max(0.0, float(ts[0] if t0 is None else t0))
    clip_t1 = float(ts[-1] if t1 is None else t1)
    tau_c = float(cfg.centroid_displacement_threshold)

    frames = []
    trigger_timestamps: list[float] = []
    prev_i: int | None = None
    for i, t in enumerate(ts):
        cx, cy = xy[i]
        valid = math.isfinite(float(cx)) and math.isfinite(float(cy))
        prev_ts = float(ts[prev_i]) if prev_i is not None else None
        displacement = None
        is_trigger = False
        if valid and prev_i is not None:
            displacement = centroid_displacement(xy[prev_i], xy[i], bbox_px)
            is_trigger = displacement is not None and displacement >= tau_c
        if is_trigger:
            trigger_timestamps.append(float(t))
        frames.append(
            {
                "timestamp": float(t),
                "centroid_x": float(cx) if valid else None,
                "centroid_y": float(cy) if valid else None,
                "prev_valid_centroid_timestamp": prev_ts,
                "displacement": displacement,
                "is_trigger": is_trigger,
                "selected_fps": cfg.base_fps,
            }
        )
        if valid:
            prev_i = i

    intervals = trigger_windows(
        trigger_timestamps,
        pad_sec=cfg.pad_sec,
        fps=cfg.high_fps,
        clip_t0=clip_t0,
        clip_t1=clip_t1,
    )
    for row in frames:
        if _in_high_fps(row["timestamp"], intervals):
            row["selected_fps"] = cfg.high_fps

    return {
        "config": asdict(cfg),
        "bbox_px": [float(v) for v in bbox_px],
        "roi_diagonal_px": roi_diagonal_px(bbox_px),
        "centroid_displacement_threshold": tau_c,
        "trigger_timestamps": trigger_timestamps,
        "high_fps_intervals": intervals,
        "frames": frames,
    }


def write_centroid_debug(result: dict, directory, stem: str) -> None:
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    frames = result.get("frames") or []
    if not frames:
        return
    with (directory / f"{stem}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(frames[0].keys()))
        writer.writeheader()
        writer.writerows(frames)
