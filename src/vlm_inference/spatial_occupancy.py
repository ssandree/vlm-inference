"""Spatial occupancy change detection, independent of video decoding/sampling.

Cells are integer (row, column) indices. bbox is a FIXED (x0,y0,x1,y1)
rectangle in the same grid coordinates (fractional edges are allowed).
No optical-flow vectors or per-frame active bounding boxes are used.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SpatialOccupancyConfig:
    delta_sec: float = 0.6
    n_min: int = 3
    tau_high: float = 0.25
    tau_low: float = 0.15
    base_fps: float = 1.0
    high_fps: float = 2.0
    max_missing_frames: int = 1

    def __post_init__(self):
        values = (self.delta_sec, self.tau_high, self.tau_low, self.base_fps, self.high_fps)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('spatial occupancy parameters must be finite')
        if self.delta_sec <= 0 or not 0 <= self.tau_low < self.tau_high <= 1:
            raise ValueError('require delta_sec > 0 and 0 <= tau_low < tau_high <= 1')
        if not 0 < self.base_fps < self.high_fps:
            raise ValueError('require 0 < base_fps < high_fps')
        if not isinstance(self.n_min, int) or self.n_min < 1:
            raise ValueError('n_min must be a positive integer')
        if not isinstance(self.max_missing_frames, int) or self.max_missing_frames < 0:
            raise ValueError('max_missing_frames must be a nonnegative integer')


def active_sets_from_events(tube: dict, events_by_id: dict, frame_indices: Sequence[int]):
    """Union member-event cells, without double-counting shared blocks.

    Prefers Stage3 ``per_frame`` cells when the event payload is present.
    Light-UF tracks store compact block events on the member itself
    (``y``, ``x``, ``t0``, ``t1``) and omit the top-level events list.
    """
    active = {int(i): set() for i in frame_indices}
    for member in tube.get('members', []):
        event = None
        if member.get('event_id') is not None:
            event = events_by_id.get(int(member['event_id']))
        wrote = False
        if event:
            for frame in event.get('per_frame', []):
                index = int(frame['frame_index'])
                if index in active:
                    active[index].update((int(y), int(x)) for y, x in frame.get('cells', []))
                    wrote = True
        if wrote:
            continue
        if not all(k in member for k in ('y', 'x', 't0', 't1')):
            continue
        y, x = int(member['y']), int(member['x'])
        t0, t1 = int(member['t0']), int(member['t1'])
        if t1 < t0:
            t0, t1 = t1, t0
        for index in range(t0, t1 + 1):
            if index in active:
                active[index].add((y, x))
    return [active[int(i)] for i in frame_indices]


def occupancy_features(active_sets: Sequence[Iterable[tuple[int, int]]], fixed_bbox):
    x0, y0, x1, y1 = map(float, fixed_bbox)
    if not all(map(math.isfinite, (x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
        raise ValueError('fixed_bbox must be a finite, nonempty rectangle')
    # Count every grid cell intersecting the actual, clipped ROI rectangle.
    total = (math.ceil(x1) - math.floor(x0)) * (math.ceil(y1) - math.floor(y0))
    n = len(active_sets)
    counts = np.zeros(n, dtype=int)
    centroid = np.full((n, 2), np.nan)
    histogram = np.zeros((n, 9))
    for i, cells in enumerate(active_sets):
        centers = []
        for y, x in set(cells):
            left, right = max(x0, x), min(x1, x + 1)
            top, bottom = max(y0, y), min(y1, y + 1)
            if right <= left or bottom <= top:
                continue
            centers.append((((left + right) / 2 - x0) / (x1 - x0),
                            ((top + bottom) / 2 - y0) / (y1 - y0)))
        counts[i] = len(centers)
        if centers:
            xy = np.asarray(centers)
            centroid[i] = xy.mean(axis=0)
            bins = np.minimum((xy * 3).astype(int), 2)
            histogram[i] = np.bincount(bins[:, 1] * 3 + bins[:, 0], minlength=9) / len(xy)
    return counts, centroid, histogram, counts / total, total


def interpolate_short_missing(times, values, max_missing_frames):
    """Fill bounded short NaN runs only; never extrapolate leading/trailing gaps."""
    out = np.array(values, dtype=float, copy=True)
    valid = np.all(np.isfinite(out), axis=1)
    imputed = np.zeros(len(out), dtype=bool)
    i = 0
    while i < len(out):
        if valid[i]:
            i += 1
            continue
        end = i
        while end < len(out) and not valid[end]:
            end += 1
        if i > 0 and end < len(out) and end - i <= max_missing_frames:
            for j in range(i, end):
                alpha = (times[j] - times[i - 1]) / (times[end] - times[i - 1])
                out[j] = (1 - alpha) * out[i - 1] + alpha * out[end]
                imputed[j] = True
        i = end
    return out, imputed


def smooth_three_frames(values):
    """Trailing three-frame mean; retain invalid current rows (no gap bridging)."""
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        if not np.all(np.isfinite(values[i])):
            continue
        start = max(0, i - 2)
        # Do not smooth across an unfilled gap.
        while start < i and not np.all(np.isfinite(values[start:i + 1])):
            start += 1
        out[i] = values[start:i + 1].mean(axis=0)
    return out


def _at(times, values, target):
    """Interpolate at exact lag time; do not interpolate across invalid endpoints."""
    if target < times[0] - 1e-8:
        return np.full_like(values[0], np.nan, dtype=float)
    idx = int(np.searchsorted(times, target))
    if idx < len(times) and abs(times[idx] - target) <= 1e-8:
        return values[idx]
    if idx > 0 and abs(times[idx - 1] - target) <= 1e-8:
        return values[idx - 1]
    if idx == 0 or idx == len(times):
        return np.full_like(values[0], np.nan, dtype=float)
    alpha = (target - times[idx - 1]) / (times[idx] - times[idx - 1])
    return (1 - alpha) * values[idx - 1] + alpha * values[idx]


def hysteresis_intervals(times, scores, cfg: SpatialOccupancyConfig):
    """State applies on [timestamp_i, timestamp_i+1); no padding."""
    high = False
    start = None
    intervals, fps = [], []
    for t, score in zip(times, scores):
        if math.isfinite(score):
            if not high and score >= cfg.tau_high:
                high, start = True, float(t)
            elif high and score <= cfg.tau_low:
                if t > start:
                    intervals.append(dict(start_sec=start, end_sec=float(t), fps=cfg.high_fps))
                high, start = False, None
        fps.append(cfg.high_fps if high else cfg.base_fps)
    if high and times[-1] > start:
        intervals.append(dict(start_sec=start, end_sec=float(times[-1]), fps=cfg.high_fps))
    return fps, intervals


def analyze_spatial_occupancy(timestamps, active_sets, fixed_bbox,
                              cfg: SpatialOccupancyConfig | None = None):
    cfg = cfg or SpatialOccupancyConfig()
    ts = np.asarray(timestamps, dtype=float)
    if ts.ndim != 1 or not len(ts) or len(ts) != len(active_sets):
        raise ValueError('timestamps and active_sets must be nonempty and aligned')
    if not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    n, raw_c, raw_p, raw_r, total = occupancy_features(active_sets, fixed_bbox)
    filled_c, imputed = interpolate_short_missing(ts, raw_c, cfg.max_missing_frames)
    c = smooth_three_frames(filled_c)
    p = smooth_three_frames(raw_p)
    ratio = smooth_three_frames(raw_r)
    # Reliability uses observed counts, not interpolated counts or fabricated activity.
    reliability = np.minimum(1.0, n / cfg.n_min)
    rows, scores = [], []
    for i, t in enumerate(ts):
        previous = float(t - cfg.delta_sec)
        dc = dp = dr = score = None
        if previous >= ts[0] - 1e-8:
            pc = _at(ts, c, previous)
            dc = 0.0
            if np.all(np.isfinite(c[i])) and np.all(np.isfinite(pc)):
                dc = float(np.linalg.norm(c[i] - pc) / math.sqrt(2))
                dc *= float(reliability[i] * _at(ts, reliability, previous))
            dp = float(0.5 * np.abs(p[i] - _at(ts, p, previous)).sum())
            dr = float(abs(ratio[i] - _at(ts, ratio, previous)))
            score = 0.4 * dc + 0.5 * dp + 0.1 * dr
        scores.append(float('nan') if score is None else score)
        rows.append(dict(timestamp=float(t), n_active=int(n[i]),
            centroid_x=float(c[i, 0]) if np.isfinite(c[i, 0]) else None,
            centroid_y=float(c[i, 1]) if np.isfinite(c[i, 1]) else None,
            centroid_raw_x=float(raw_c[i, 0]) if n[i] else None,
            centroid_raw_y=float(raw_c[i, 1]) if n[i] else None,
            centroid_observed=bool(n[i]), centroid_interpolated=bool(imputed[i]),
            occupancy_histogram=p[i].tolist(), occupancy_histogram_raw=raw_p[i].tolist(),
            active_ratio=float(ratio[i]), active_ratio_raw=float(raw_r[i]),
            reliability=float(reliability[i]), Dc=dc, Dp=dp, Dr=dr, S=score))
    fps, intervals = hysteresis_intervals(ts, scores, cfg)
    for row, rate in zip(rows, fps):
        row['selected_fps'] = rate
    return dict(config=asdict(cfg), fixed_bbox_grid=list(map(float, fixed_bbox)),
                total_roi_blocks=total, smoothing='trailing_3_frames',
                high_fps_intervals=intervals, frames=rows)


def write_spatial_debug(result, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{stem}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    fields = [k for k in result['frames'][0] if not k.startswith('occupancy_histogram')]
    with (directory / f'{stem}.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['frames'])


def plot_spatial_occupancy(result, path, title='ROI spatial occupancy'):
    """Export S(t), thresholds and sampling state. Accepts result dict or JSON path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if isinstance(result, (str, Path)):
        result = json.loads(Path(result).read_text())
    rows, cfg = result['frames'], result['config']
    ts = [r['timestamp'] for r in rows]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 4.8),
                             gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(ts, [r['S'] if r['S'] is not None else np.nan for r in rows], label='S(t)')
    for name, color in [('tau_high', 'tab:red'), ('tau_low', 'tab:green')]:
        axes[0].axhline(cfg[name], color=color, linestyle='--', label=f'{name}={cfg[name]:g}')
    for interval in result['high_fps_intervals']:
        for ax in axes:
            ax.axvspan(interval['start_sec'], interval['end_sec'], color='orange', alpha=0.18)
    axes[0].set(ylabel='Spatial change score', title=title, ylim=(-0.02, 1.02))
    axes[0].legend(loc='upper right')
    axes[1].step(ts, [r['selected_fps'] for r in rows], where='post')
    axes[1].set(xlabel='Video timestamp (s)', ylabel='ROI fps',
                yticks=[cfg['base_fps'], cfg['high_fps']],
                ylim=(cfg['base_fps'] - 0.2, cfg['high_fps'] + 0.2))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
