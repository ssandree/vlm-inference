"""Experiment-mode frame labeling: Global/ROI + timestamp interleaved with images."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import ExperimentConfig

INPUT_ORDER_TIMESTAMP = "timestamp"
INPUT_ORDER_GROUPED = "grouped"
INPUT_ORDER_GLOBAL_ANCHOR = "global_anchor"
INPUT_ORDER_TUBE_STORY = "tube_story"
VALID_INPUT_ORDERS = frozenset(
    {
        INPUT_ORDER_TIMESTAMP,
        INPUT_ORDER_GROUPED,
        INPUT_ORDER_GLOBAL_ANCHOR,
        INPUT_ORDER_TUBE_STORY,
    }
)

# Bump when the VLM interleaved text structure changes (invalidates feasibility cache).
INPUT_FORMAT_VERSION = "dataset_global_anchor_group_roi_v1"

BBOX_DEFINITION = (
    "All ROI locations refer to the original full-frame coordinate system.\n"
    "normalized_bbox = [x_min, y_min, x_max, y_max], normalized to [0,1]."
)

_GRID_ROWS = ("top", "center", "bottom")
_GRID_COLS = ("left", "center", "right")


@dataclass(frozen=True)
class LabeledFrame:
    path: str  # file URI or local path suitable for processor
    label: str  # e.g. "[Global] t=12.0s"
    time_s: float
    kind: str  # "global" | "roi"
    roi_id: int | None  # 1-based for ROI, None for global
    frame_idx: int | None


@dataclass(frozen=True)
class RoiSpatialMeta:
    """Per-tube spatial location in the original full frame."""

    roi_id: int
    normalized_bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in [0, 1]
    source_region: str


def _basename(path: str) -> str:
    if path.startswith("file://"):
        parsed = urlparse(path)
        return Path(unquote(parsed.path)).name
    return Path(path).name


def _format_time(raw: str) -> str:
    """Keep filename time token as-is for label fidelity."""
    return raw


def _format_time_s(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_labeled_frame(path: str, cfg: ExperimentConfig) -> LabeledFrame | None:
    name = _basename(path)
    stem = Path(name).stem

    g = re.fullmatch(cfg.global_pattern, stem)
    if g:
        gd = g.groupdict()
        raw_t = gd["time"]
        return LabeledFrame(
            path=path,
            label=cfg.global_label.format(time=_format_time(raw_t)),
            time_s=float(raw_t),
            kind="global",
            roi_id=None,
            frame_idx=int(gd["frame"]) if gd.get("frame") is not None else None,
        )

    r = re.fullmatch(cfg.roi_pattern, stem)
    if r:
        rd = r.groupdict()
        raw_t = rd["time"]
        roi_id = int(rd["roi"])
        return LabeledFrame(
            path=path,
            label=cfg.roi_label.format(roi=roi_id, time=_format_time(raw_t)),
            time_s=float(raw_t),
            kind="roi",
            roi_id=roi_id,
            frame_idx=int(rd["frame"]) if rd.get("frame") is not None else None,
        )

    return None


def build_labeled_frames(paths: list[str], cfg: ExperimentConfig) -> list[LabeledFrame]:
    frames: list[LabeledFrame] = []
    unmatched: list[str] = []
    for path in paths:
        parsed = parse_labeled_frame(path, cfg)
        if parsed is None:
            unmatched.append(_basename(path))
        else:
            frames.append(parsed)

    if unmatched:
        preview = ", ".join(unmatched[:5])
        more = f" (+{len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
        raise ValueError(
            "experiment mode: filename(s) did not match global/roi patterns: "
            f"{preview}{more}"
        )

    # time asc; Global before ROI at same time; then ROI id; then frame idx
    frames.sort(
        key=lambda f: (
            f.time_s,
            0 if f.kind == "global" else 1,
            f.roi_id or 0,
            f.frame_idx if f.frame_idx is not None else 0,
        )
    )
    return frames


def normalize_bbox_px(
    bbox_px: tuple[int, int, int, int] | list[int],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"invalid frame size for bbox normalize: ({frame_width}, {frame_height})"
        )
    x0, y0, x1, y1 = (int(v) for v in bbox_px)
    return (
        x0 / float(frame_width),
        y0 / float(frame_height),
        x1 / float(frame_width),
        y1 / float(frame_height),
    )


def _grid_cell_label(row: int, col: int) -> str:
    row_name = _GRID_ROWS[row]
    col_name = _GRID_COLS[col]
    if row == 1 and col == 1:
        return "center"
    if row == 1:
        return f"center-{col_name}"
    if col == 1:
        return f"{row_name}-center"
    return f"{row_name}-{col_name}"


def source_region_label(
    normalized_bbox: tuple[float, float, float, float],
    *,
    overlap_thresh: float = 0.05,
) -> str:
    """Map a normalized bbox onto a 3x3 spatial label (with coverage span)."""
    x0, y0, x1, y1 = (float(v) for v in normalized_bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty normalized bbox: {normalized_bbox}")

    bbox_area = max((x1 - x0) * (y1 - y0), 1e-12)
    covered: list[tuple[int, int]] = []
    for row in range(3):
        for col in range(3):
            cx0, cy0 = col / 3.0, row / 3.0
            cx1, cy1 = (col + 1) / 3.0, (row + 1) / 3.0
            ix0 = max(x0, cx0)
            iy0 = max(y0, cy0)
            ix1 = min(x1, cx1)
            iy1 = min(y1, cy1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            if inter / bbox_area >= overlap_thresh:
                covered.append((row, col))

    if not covered:
        cx = min(max((x0 + x1) * 0.5, 0.0), 0.999999)
        cy = min(max((y0 + y1) * 0.5, 0.0), 0.999999)
        covered = [(min(2, int(cy * 3)), min(2, int(cx * 3)))]

    if len(covered) == 1:
        return _grid_cell_label(*covered[0])
    first, last = covered[0], covered[-1]
    if first == last:
        return _grid_cell_label(*first)
    return f"{_grid_cell_label(*first)} to {_grid_cell_label(*last)}"


def build_roi_spatial_meta(
    roi_bboxes_px: dict[int, tuple[int, int, int, int] | list[int]],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[int, RoiSpatialMeta]:
    out: dict[int, RoiSpatialMeta] = {}
    for roi_id, bbox in sorted(roi_bboxes_px.items()):
        norm = normalize_bbox_px(
            bbox, frame_width=frame_width, frame_height=frame_height
        )
        out[int(roi_id)] = RoiSpatialMeta(
            roi_id=int(roi_id),
            normalized_bbox=norm,
            source_region=source_region_label(norm),
        )
    return out


def _format_normalized_bbox(bbox: tuple[float, float, float, float]) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in bbox) + "]"


def format_global_section_header(num_frames: int) -> str:
    return (
        "[GLOBAL FRAME]\n"
        "coverage: full scene\n"
        f"num_frames: {int(num_frames)}"
    )


def format_roi_section_header(
    meta: RoiSpatialMeta,
    *,
    start_time_s: float,
    end_time_s: float,
    num_frames: int,
) -> str:
    return (
        f"[ROI {meta.roi_id}]\n"
        f"time_range: [{_format_time_s(start_time_s)}s, {_format_time_s(end_time_s)}s]\n"
        f"normalized_bbox: {_format_normalized_bbox(meta.normalized_bbox)}\n"
        f"num_frames: {int(num_frames)}"
    )


def format_global_sequence_begin() -> str:
    return "[Global sequence begins]"


def format_global_sequence_end() -> str:
    return "[Global sequence ends]"


def format_roi_sequence_begin(roi_id: int) -> str:
    return f"[ROI{int(roi_id)} sequence begins]"


def format_roi_sequence_end(roi_id: int) -> str:
    return f"[ROI{int(roi_id)} sequence ends]"


def format_roi_sequence_meta(
    meta: RoiSpatialMeta,
    *,
    start_time_s: float,
    end_time_s: float,
) -> str:
    """Compact ROI meta shown inside a sequence (no location-anchor label)."""
    return (
        f"time_range: [{_format_time_s(start_time_s)}s, {_format_time_s(end_time_s)}s]\n"
        f"normalized_bbox: {_format_normalized_bbox(meta.normalized_bbox)}"
    )


def group_frames_for_prompt(frames: list[LabeledFrame]) -> tuple[
    list[LabeledFrame],
    list[tuple[int, list[LabeledFrame]]],
]:
    """Split into Global frames and ROI tubes (id asc), each sorted by time.

    Used for spatial section headers and for ``grouped`` VLM image order.
    """
    globals_ = sorted(
        [f for f in frames if f.kind == "global"],
        key=lambda f: (f.time_s, f.frame_idx if f.frame_idx is not None else 0),
    )
    by_roi: dict[int, list[LabeledFrame]] = {}
    for frame in frames:
        if frame.kind != "roi" or frame.roi_id is None:
            continue
        by_roi.setdefault(frame.roi_id, []).append(frame)
    roi_groups: list[tuple[int, list[LabeledFrame]]] = []
    for roi_id in sorted(by_roi):
        group = sorted(
            by_roi[roi_id],
            key=lambda f: (f.time_s, f.frame_idx if f.frame_idx is not None else 0),
        )
        roi_groups.append((roi_id, group))
    return globals_, roi_groups


def interleave_frames_by_time(frames: list[LabeledFrame]) -> list[LabeledFrame]:
    """Order Global/ROI frames by timestamp for the VLM message.

    Tie-break: Global before ROI at the same time, then ROI id, then frame idx.
    """
    return sorted(
        frames,
        key=lambda f: (
            f.time_s,
            0 if f.kind == "global" else 1,
            f.roi_id or 0,
            f.frame_idx if f.frame_idx is not None else 0,
        ),
    )


def group_frames_for_vlm(frames: list[LabeledFrame]) -> list[LabeledFrame]:
    """Order as Global sequence, then each ROI tube sequence (id asc).

    Within each sequence, timestamps ascend. Global frames appear once only.
    """
    globals_, roi_groups = group_frames_for_prompt(frames)
    ordered: list[LabeledFrame] = list(globals_)
    for _, group in roi_groups:
        ordered.extend(group)
    return ordered


def tube_story_frames(frames: list[LabeledFrame]) -> list[LabeledFrame]:
    """Order as one Global establishing shot, then that tube's full ROI sequence.

    Tubes are sorted by first crop time, then roi_id. Globals are consumed in
    timestamp order, one per tube (start_only: Global at tube start). Leftover
    Globals, if any, are appended after all tubes.
    """
    globals_, roi_groups = group_frames_for_prompt(frames)
    if not roi_groups:
        return list(globals_)
    roi_groups_sorted = sorted(
        roi_groups,
        key=lambda item: (
            item[1][0].time_s if item[1] else float("inf"),
            item[0],
        ),
    )
    unused = list(globals_)
    ordered: list[LabeledFrame] = []
    for _, group in roi_groups_sorted:
        if unused:
            ordered.append(unused.pop(0))
        ordered.extend(group)
    ordered.extend(unused)
    return ordered


def _roi_frames_in_interval(
    roi_groups: list[tuple[int, list[LabeledFrame]]],
    t_lo: float,
    t_hi: float,
) -> list[LabeledFrame]:
    """ROI frames in [t_lo, t_hi), grouped by roi_id then time."""
    out: list[LabeledFrame] = []
    for _, group in roi_groups:
        out.extend(frame for frame in group if t_lo <= frame.time_s < t_hi)
    return out


def anchor_frames_by_global(frames: list[LabeledFrame]) -> list[LabeledFrame]:
    """Order as Global anchors with ROI tubes grouped inside each interval.

    For Global timestamps G_0 < G_1 < ... < G_n:
      [ROIs with t < G_0] grouped by roi_id (edge case), then for each i:
        G_i → ROI1 in [G_i, G_{i+1}) → ROI2 in [G_i, G_{i+1}) → ...
    Last interval: t >= G_n. Within each ROI, timestamps ascend.
    """
    globals_, roi_groups = group_frames_for_prompt(frames)
    if not globals_:
        return _roi_frames_in_interval(roi_groups, float("-inf"), float("inf"))

    ordered: list[LabeledFrame] = []
    ordered.extend(
        _roi_frames_in_interval(roi_groups, float("-inf"), globals_[0].time_s)
    )
    for g_i, g_frame in enumerate(globals_):
        ordered.append(g_frame)
        t_hi = (
            globals_[g_i + 1].time_s
            if g_i + 1 < len(globals_)
            else float("inf")
        )
        ordered.extend(_roi_frames_in_interval(roi_groups, g_frame.time_s, t_hi))
    return ordered


def order_frames_for_vlm(
    frames: list[LabeledFrame],
    input_order: str = INPUT_ORDER_TIMESTAMP,
) -> list[LabeledFrame]:
    order = str(input_order).strip().lower()
    if order not in VALID_INPUT_ORDERS:
        raise ValueError(
            f"input_order must be one of {sorted(VALID_INPUT_ORDERS)}, got '{input_order}'"
        )
    if order == INPUT_ORDER_GROUPED:
        return group_frames_for_vlm(frames)
    if order == INPUT_ORDER_GLOBAL_ANCHOR:
        return anchor_frames_by_global(frames)
    if order == INPUT_ORDER_TUBE_STORY:
        return tube_story_frames(frames)
    return interleave_frames_by_time(frames)


def format_input_order_dump(
    frames: list[LabeledFrame],
    *,
    input_order: str = INPUT_ORDER_TIMESTAMP,
) -> str:
    """Human-readable dump of the VLM image order (pre-call)."""
    order = str(input_order).strip().lower()
    lines: list[str] = [f"# input_order={order}"]
    for frame in order_frames_for_vlm(frames, order):
        path = str(frame.path or "").strip()
        if path.startswith("file://"):
            path = path[len("file://") :]
        if path:
            lines.append(f"{frame.label}  ->  {path}")
        else:
            lines.append(frame.label)
    return "\n".join(lines).rstrip() + "\n"


def _require_roi_spatial(
    roi_groups: list[tuple[int, list[LabeledFrame]]],
    spatial_by_roi: dict[int, RoiSpatialMeta] | None,
) -> dict[int, RoiSpatialMeta]:
    if not spatial_by_roi:
        missing = [roi_id for roi_id, _ in roi_groups]
        raise ValueError(
            "experiment mode: ROI spatial metadata required for tube(s) "
            f"{missing}, but none was provided"
        )
    missing = [roi_id for roi_id, _ in roi_groups if roi_id not in spatial_by_roi]
    if missing:
        raise ValueError(
            "experiment mode: missing ROI spatial metadata for tube(s) "
            f"{missing}"
        )
    return spatial_by_roi


def build_experiment_text_and_image_sequence(
    frames: list[LabeledFrame],
    spatial_by_roi: dict[int, RoiSpatialMeta] | None,
    *,
    input_order: str = INPUT_ORDER_TIMESTAMP,
) -> list[tuple[str, str | None]]:
    """Return ordered (text, image_path|None) items for the experiment message.

    Emits only per-frame labels + images (no spatial text metadata).
    ``spatial_by_roi`` is accepted for call-site compatibility but unused.
    """
    order = str(input_order).strip().lower()
    if order not in VALID_INPUT_ORDERS:
        raise ValueError(
            f"input_order must be one of {sorted(VALID_INPUT_ORDERS)}, got '{input_order}'"
        )
    _ = spatial_by_roi
    return [(frame.label, frame.path) for frame in order_frames_for_vlm(frames, order)]
