"""Separate vs Union ROI representation.

Does not change ROI detection, clustering, sampling timestamps, fps, or
Global frames. Only the spatial packing of the *same* proposed tubes changes.

Temporal policy (solo vs overlap)
---------------------------------
An ROI is **active** at timestamp ``t`` when either:

1. ``t`` lies in that ROI's existing metadata ``time_range`` ``[t0, t1]``
   (inclusive, with a 1e-3 s tolerance), or
2. the ROI has a sampled frame at ``t`` (sample implies active, so rounding
   between ``time_range`` and the 1 fps filename cannot drop a source crop).

Sampling timestamps are the sorted unique timestamps already present in the
proposed ROI sequences. They are never resampled.

At each timestamp ``t``:

* **overlap** (2+ active ROIs): crop
  ``union_bbox = [min(x1), min(y1), max(x2), max(y2)]`` of the active ROIs'
  lifetime AABBs (the same boxes used by the existing proposed crops).
* **solo** (exactly one active ROI): crop that ROI's lifetime AABB. The
  original JPEG is reused so pixels match ``separate_roi`` at that ``t``.

A video with a single proposed ROI is a control: union is identical to
separate (all stamps are solo).
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

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
    _link_or_copy,
    _save_jpeg_rgb,
    load_tube_bboxes,
    tracks_json_path,
)
from .single_motion import clip_bbox, union_bboxes
from .video_io import extract_video_frames, inspect_video

logger = logging.getLogger(__name__)

ROI_REPR_SEPARATE = "separate_roi"
ROI_REPR_UNION = "union_roi"
VALID_ROI_REPR_MODES = frozenset({ROI_REPR_SEPARATE, ROI_REPR_UNION})

ROI_REPR_METHODS: dict[str, str] = {
    "global_0.25fps_roi_separate": ROI_REPR_SEPARATE,
    "global_0.25fps_roi_union": ROI_REPR_UNION,
}

TIME_EPS = 1e-3
UNION_ROI_ID = 1

# BGR for OpenCV overlays (viz only; never fed to the VLM).
COLOR_ROI = (
    (80, 220, 80),
    (255, 180, 50),
    (50, 200, 255),
    (220, 80, 220),
)
COLOR_UNION = (0, 0, 255)


@dataclass(frozen=True)
class UnionStamp:
    timestamp: float
    mode: str  # overlap | solo
    active_roi_ids: tuple[int, ...]
    crop_bbox_px: tuple[int, int, int, int]
    source_bboxes_px: dict[int, tuple[int, int, int, int]]
    source_path: Path | None
    native_frame: int


@dataclass
class ReprPlan:
    video_id: str
    mode: str
    source_roi_ids: tuple[int, ...]
    stamps: tuple[UnionStamp, ...] = ()
    n_overlap: int = 0
    n_solo: int = 0
    frame_width: int = 0
    frame_height: int = 0
    source_bboxes_px: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    enclosing_bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)


def parse_roi_repr_mode(value: str | None) -> str | None:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if not mode or mode in {"none", "null", "off"}:
        return None
    if mode not in VALID_ROI_REPR_MODES:
        raise ValueError(
            f"roi_repr must be one of {sorted(VALID_ROI_REPR_MODES)}, got '{value}'"
        )
    return mode


def repr_mode_for_method(method: str, configured: str | None) -> str | None:
    named = ROI_REPR_METHODS.get(method)
    if named is not None:
        return named
    return parse_roi_repr_mode(configured)


def filter_methods_for_roi_repr(methods: list[str], mode: str | None) -> list[str]:
    parsed = parse_roi_repr_mode(mode)
    if parsed is None:
        return list(methods)
    matching = [m for m in methods if ROI_REPR_METHODS.get(m) == parsed]
    return matching if matching else list(methods)


def condition_root(output_dir: Path | str, mode: str) -> Path:
    return Path(output_dir) / str(mode)


def repr_video_dir(output_dir: Path | str, mode: str, video_id: str) -> Path:
    return condition_root(output_dir, mode) / video_id


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


def _in_time_range(t0: float, t1: float, t: float) -> bool:
    lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
    return (lo - TIME_EPS) <= float(t) <= (hi + TIME_EPS)


def _normalize_bbox_px(
    bbox_px: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    x0, y0, x1, y1 = (int(v) for v in bbox_px)
    w = max(int(frame_width), 1)
    h = max(int(frame_height), 1)
    return [x0 / w, y0 / h, x1 / w, y1 / h]


def _denorm_bbox(
    norm: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in norm)
    return clip_bbox(
        (
            int(round(x0 * frame_width)),
            int(round(y0 * frame_height)),
            int(round(x1 * frame_width)),
            int(round(y1 * frame_height)),
        ),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _jpeg_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_source_bboxes(
    *,
    video_id: str,
    tracks_root: Path,
    video_root: Path,
    bundle: VideoBundleMetadata,
) -> tuple[dict[int, tuple[int, int, int, int]], int, int]:
    """Pixel AABBs of the existing proposed ROIs (metadata normalized_bbox).

    Tracks lifetime AABBs are a fallback only. Union must wrap the same boxes
    that ``separate_roi`` already shows.
    """
    video_path = Path(video_root) / f"{video_id}.mp4"
    info = inspect_video(video_path)
    tubes: dict[int, Any] = {}
    tracks_path = tracks_json_path(Path(tracks_root), video_id)
    if tracks_path.is_file():
        try:
            tubes = load_tube_bboxes(
                tracks_path, frame_width=info.width, frame_height=info.height
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: tracks bboxes unavailable (%s)", video_id, exc)
    out: dict[int, tuple[int, int, int, int]] = {}
    for roi in active_rois(bundle):
        rid = int(roi.roi_id)
        if roi.normalized_bbox and any(float(v) != 0 for v in roi.normalized_bbox):
            out[rid] = _denorm_bbox(
                roi.normalized_bbox,
                frame_width=info.width,
                frame_height=info.height,
            )
        elif rid in tubes:
            out[rid] = tuple(int(v) for v in tubes[rid].bbox_px)
    return out, int(info.width), int(info.height)


def plan_union_stamps(
    bundle: VideoBundleMetadata,
    *,
    source_bboxes: dict[int, tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    native_fps: float,
    frame_count: int,
) -> list[UnionStamp]:
    rois = list(active_rois(bundle))
    by_t: dict[float, list[tuple[int, Path]]] = {}
    for roi in rois:
        for fr in roi.frames:
            by_t.setdefault(float(fr.timestamp), []).append((int(roi.roi_id), Path(fr.path)))

    stamps: list[UnionStamp] = []
    n = max(int(frame_count), 1)
    fps_n = float(native_fps) if native_fps > 0 else 30.0
    for t in sorted(by_t):
        sampled_ids = {rid for rid, _ in by_t[t]}
        temporal_ids = {
            int(roi.roi_id)
            for roi in rois
            if _in_time_range(float(roi.time_range[0]), float(roi.time_range[1]), t)
        }
        active = tuple(sorted(sampled_ids | temporal_ids))
        boxes = [source_bboxes[rid] for rid in active if rid in source_bboxes]
        if not boxes:
            continue
        if len(active) >= 2:
            crop = union_bboxes(
                boxes, frame_width=frame_width, frame_height=frame_height
            )
            mode = "overlap"
            source_path = None
        else:
            crop = boxes[0]
            mode = "solo"
            source_path = by_t[t][0][1] if by_t[t] else None
        native = int(round(float(t) * fps_n))
        native = min(max(native, 0), n - 1)
        stamps.append(
            UnionStamp(
                timestamp=float(t),
                mode=mode,
                active_roi_ids=active,
                crop_bbox_px=tuple(int(v) for v in crop),
                source_bboxes_px={
                    rid: source_bboxes[rid] for rid in active if rid in source_bboxes
                },
                source_path=source_path,
                native_frame=native,
            )
        )
    return stamps


def _tube_board_label(*, mode: str, roi_id: int) -> str:
    if mode == ROI_REPR_UNION:
        return f"Tube {roi_id:02d}  union"
    return f"Tube {roi_id:02d}  ROI{roi_id}"


def load_roi_repr_labels(repr_dir: Path) -> dict[int, str]:
    log_path = Path(repr_dir) / "roi_log.json"
    if not log_path.is_file():
        return {}
    data = json.loads(log_path.read_text(encoding="utf-8"))
    mode = str(data.get("mode") or "")
    labels: dict[int, str] = {}
    for row in data.get("rois") or []:
        rid = int(row["roi_id"])
        labels[rid] = _tube_board_label(mode=mode, roi_id=rid)
    return labels


def load_bbox_preview_section(repr_dir: Path) -> list[tuple[str, list[Path]]]:
    preview = Path(repr_dir) / "bbox_preview"
    if not preview.is_dir():
        return []
    paths = sorted(p for p in preview.glob("*.jpg") if p.is_file())
    if not paths:
        return []
    return [("Full-frame  ROI1 / ROI2 / union bbox", paths)]


def apply_roi_repr(
    bundle: VideoBundleMetadata,
    repr_dir: Path,
    *,
    video_id: str,
) -> VideoBundleMetadata:
    sampled = load_video_bundle_metadata(repr_dir, require_files=True)
    return build_bundle(global_metadata=bundle, roi_metadata=sampled, video_id=video_id)


def _copy_separate_rois(
    *,
    bundle: VideoBundleMetadata,
    video_dir: Path,
    overwrite: bool,
) -> dict[int, list[FrameRef]]:
    saved: dict[int, list[FrameRef]] = {}
    for roi in active_rois(bundle):
        roi_dir = video_dir / f"roi_{int(roi.roi_id):02d}"
        if overwrite and roi_dir.exists():
            shutil.rmtree(roi_dir)
        roi_dir.mkdir(parents=True, exist_ok=True)
        refs: list[FrameRef] = []
        for seq, fr in enumerate(roi.frames, start=1):
            name = Path(fr.file).name
            dest = roi_dir / name
            _link_or_copy(Path(fr.path), dest)
            rel = f"roi_{int(roi.roi_id):02d}/{name}"
            refs.append(
                FrameRef(
                    frame_index=seq,
                    timestamp=float(fr.timestamp),
                    file=rel,
                    path=dest.resolve(),
                )
            )
        saved[int(roi.roi_id)] = refs
    return saved


def _materialize_union_crops(
    *,
    plan: ReprPlan,
    video_path: Path,
    video_dir: Path,
    jpeg_quality: int,
    overwrite: bool,
) -> list[FrameRef]:
    roi_dir = video_dir / f"roi_{UNION_ROI_ID:02d}"
    if overwrite and roi_dir.exists():
        shutil.rmtree(roi_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)

    need_idx = {
        int(s.native_frame)
        for s in plan.stamps
        if s.mode == "overlap" or s.source_path is None or not Path(s.source_path).is_file()
    }
    decoded = extract_video_frames(video_path, need_idx) if need_idx else {}

    refs: list[FrameRef] = []
    for seq, stamp in enumerate(plan.stamps, start=1):
        name = roi_frame_name(UNION_ROI_ID, seq, stamp.timestamp)
        dest = roi_dir / name
        if stamp.mode == "solo" and stamp.source_path is not None and stamp.source_path.is_file():
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            _link_or_copy(stamp.source_path, dest)
        else:
            crop = _crop_rgb(decoded[int(stamp.native_frame)], stamp.crop_bbox_px)
            _save_jpeg_rgb(dest, crop, quality=jpeg_quality)
        rel = f"roi_{UNION_ROI_ID:02d}/{name}"
        refs.append(
            FrameRef(
                frame_index=seq,
                timestamp=float(stamp.timestamp),
                file=rel,
                path=dest.resolve(),
            )
        )
    return refs


def _write_separate_metadata(
    *,
    video_dir: Path,
    video_id: str,
    bundle: VideoBundleMetadata,
    saved: dict[int, list[FrameRef]],
    source_bboxes: dict[int, tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
) -> None:
    rois: list[dict[str, Any]] = []
    for roi in active_rois(bundle):
        frames = saved.get(int(roi.roi_id)) or []
        bbox = source_bboxes.get(int(roi.roi_id), (0, 0, 0, 0))
        rois.append(
            {
                "roi_id": int(roi.roi_id),
                "time_range": [float(roi.time_range[0]), float(roi.time_range[1])],
                "source_region": roi.source_region,
                "normalized_bbox": _normalize_bbox_px(
                    bbox, frame_width=frame_width, frame_height=frame_height
                ),
                "bbox_px": list(bbox),
                "num_frames": len(frames),
                "frames": [
                    {"timestamp": float(fr.timestamp), "file": fr.file} for fr in frames
                ],
            }
        )
    _write_json(
        video_dir / "metadata.json",
        {
            "video_id": video_id,
            "roi_repr": ROI_REPR_SEPARATE,
            "global": {"coverage": "full scene", "num_frames": 0, "frames": []},
            "rois": rois,
        },
    )


def _write_union_metadata(
    *,
    video_dir: Path,
    plan: ReprPlan,
    refs: list[FrameRef],
) -> None:
    frames: list[dict[str, Any]] = []
    for stamp, fr in zip(plan.stamps, refs):
        frames.append(
            {
                "timestamp": float(stamp.timestamp),
                "file": fr.file,
                "mode": stamp.mode,
                "active_roi_ids": list(stamp.active_roi_ids),
                "crop_bbox_px": list(stamp.crop_bbox_px),
                "source_bboxes_px": {
                    str(k): list(v) for k, v in sorted(stamp.source_bboxes_px.items())
                },
            }
        )
    t0 = float(refs[0].timestamp) if refs else 0.0
    t1 = float(refs[-1].timestamp) if refs else 0.0
    source_ids = ",".join(str(i) for i in plan.source_roi_ids) or "none"
    _write_json(
        video_dir / "metadata.json",
        {
            "video_id": plan.video_id,
            "roi_repr": ROI_REPR_UNION,
            "global": {"coverage": "full scene", "num_frames": 0, "frames": []},
            "rois": [
                {
                    "roi_id": UNION_ROI_ID,
                    "time_range": [t0, t1],
                    "source_region": (
                        "union of proposed ROI tubes; overlap uses enclosing bbox, "
                        f"solo uses the single active ROI bbox (source ROI ids: {source_ids})"
                    ),
                    "normalized_bbox": _normalize_bbox_px(
                        plan.enclosing_bbox_px,
                        frame_width=plan.frame_width,
                        frame_height=plan.frame_height,
                    ),
                    "bbox_px": list(plan.enclosing_bbox_px),
                    "source_roi_ids": list(plan.source_roi_ids),
                    "source_bboxes_px": {
                        str(k): list(v)
                        for k, v in sorted(plan.source_bboxes_px.items())
                    },
                    "num_frames": len(frames),
                    "frames": frames,
                }
            ],
        },
    )


def _stamp_crop_sizes(refs: Iterable[FrameRef]) -> list[dict[str, Any]]:
    rows = []
    for fr in refs:
        w, h = _jpeg_size(fr.path)
        rows.append(
            {
                "timestamp": float(fr.timestamp),
                "file": fr.file,
                "width": w,
                "height": h,
                "crop_pixels": int(w) * int(h),
            }
        )
    return rows


def _write_separate_log(
    *,
    video_dir: Path,
    video_id: str,
    bundle: VideoBundleMetadata,
    saved: dict[int, list[FrameRef]],
    source_bboxes: dict[int, tuple[int, int, int, int]],
) -> dict[str, Any]:
    rois_payload: list[dict[str, Any]] = []
    for roi in active_rois(bundle):
        refs = saved.get(int(roi.roi_id)) or []
        sizes = _stamp_crop_sizes(refs)
        rois_payload.append(
            {
                "roi_id": int(roi.roi_id),
                "time_range": [float(roi.time_range[0]), float(roi.time_range[1])],
                "bbox_px": list(source_bboxes.get(int(roi.roi_id), (0, 0, 0, 0))),
                "num_frames": len(refs),
                "timestamps": [float(fr.timestamp) for fr in refs],
                "crops": sizes,
            }
        )
    n_roi = sum(int(r["num_frames"]) for r in rois_payload)
    payload = {
        "video_id": video_id,
        "mode": ROI_REPR_SEPARATE,
        "n_source_rois": len(rois_payload),
        "n_roi_images": n_roi,
        "n_overlap": 0,
        "n_solo": n_roi,
        "rois": rois_payload,
    }
    _write_json(video_dir / "roi_log.json", payload)
    return payload


def _write_union_log(
    *,
    video_dir: Path,
    plan: ReprPlan,
    refs: list[FrameRef],
) -> dict[str, Any]:
    sizes = _stamp_crop_sizes(refs)
    size_by_t = {row["timestamp"]: row for row in sizes}
    stamps_payload = []
    for stamp in plan.stamps:
        size = size_by_t.get(float(stamp.timestamp)) or {}
        stamps_payload.append(
            {
                "timestamp": float(stamp.timestamp),
                "mode": stamp.mode,
                "active_roi_ids": list(stamp.active_roi_ids),
                "crop_bbox_px": list(stamp.crop_bbox_px),
                "source_bboxes_px": {
                    str(k): list(v) for k, v in sorted(stamp.source_bboxes_px.items())
                },
                "native_frame": int(stamp.native_frame),
                "width": size.get("width"),
                "height": size.get("height"),
                "crop_pixels": size.get("crop_pixels"),
            }
        )
    payload = {
        "video_id": plan.video_id,
        "mode": ROI_REPR_UNION,
        "temporal_policy": (
            "active if t in metadata time_range or the ROI has a sample at t; "
            "overlap -> union of lifetime AABBs; solo -> that ROI's AABB "
            "(original crop reused)"
        ),
        "n_source_rois": len(plan.source_roi_ids),
        "source_roi_ids": list(plan.source_roi_ids),
        "source_bboxes_px": {
            str(k): list(v) for k, v in sorted(plan.source_bboxes_px.items())
        },
        "enclosing_bbox_px": list(plan.enclosing_bbox_px),
        "n_roi_images": len(refs),
        "n_overlap": plan.n_overlap,
        "n_solo": plan.n_solo,
        "union_is_identity": len(plan.source_roi_ids) <= 1,
        "stamps": stamps_payload,
        "rois": [
            {
                "roi_id": UNION_ROI_ID,
                "source_roi_ids": list(plan.source_roi_ids),
            }
        ],
    }
    _write_json(video_dir / "roi_log.json", payload)
    return payload


def _annotate_full_frames(
    *,
    proposed_dir: Path,
    video_dir: Path,
    plan: ReprPlan,
) -> list[Path]:
    import cv2

    global_dir = Path(proposed_dir) / "global"
    sources = sorted(global_dir.glob("global_frame_*.jpg")) if global_dir.is_dir() else []
    if not sources:
        sources = sorted(Path(proposed_dir).glob("global_frame_*.jpg"))
    out_dir = video_dir / "bbox_preview"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for src in sources:
        img = cv2.imread(str(src))
        if img is None:
            continue
        # Union box first (outer), then each source ROI.
        if any(plan.source_bboxes_px.values()):
            boxes = list(plan.source_bboxes_px.values())
            union = union_bboxes(
                boxes,
                frame_width=plan.frame_width,
                frame_height=plan.frame_height,
            )
            x0, y0, x1, y1 = union
            cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_UNION, 3)
        for i, (rid, bbox) in enumerate(sorted(plan.source_bboxes_px.items())):
            x0, y0, x1, y1 = bbox
            color = COLOR_ROI[i % len(COLOR_ROI)]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                img,
                f"ROI{rid}",
                (x0 + 4, max(y0 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        legend_y = 22
        cv2.putText(
            img,
            "green/cyan=source ROI   red=union bbox  (viz only)",
            (8, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        dest = out_dir / src.name
        cv2.imwrite(str(dest), img)
        written.append(dest)
    return written


def _render_boards(
    *,
    video_id: str,
    method: str,
    items: list[tuple[str, Path]],
    out_dir: Path,
    tube_labels: dict[int, str],
    extra_sections: list[tuple[str, list[Path]]] | None = None,
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
        items,
        video_id=video_id,
        method=method,
        args=args,
        tube_labels=tube_labels,
        extra_sections=extra_sections,
    )
    order = viz.render_order_canvas(items, video_id=video_id, method=method, args=args)
    viz.cv2.imwrite(str(out_dir / "board_grouped.png"), grouped)
    viz.cv2.imwrite(str(out_dir / "board_order.png"), order)
    boards = out_dir.parent / "boards"
    boards.mkdir(parents=True, exist_ok=True)
    viz.cv2.imwrite(str(boards / f"{video_id}__grouped.png"), grouped)
    viz.cv2.imwrite(str(boards / f"{video_id}__order.png"), order)
    if extra_sections:
        preview = viz.render_sections(
            extra_sections,
            title=f"{video_id}  |  {method}  |  bbox preview",
            canvas_w=int(args.canvas_w),
            label_w=int(args.label_w),
            pad=int(args.pad),
            thumb_h=int(args.thumb_h),
            frame_gap=int(args.frame_gap),
            row_gap=int(args.row_gap),
            section_gap=int(args.section_gap),
            section_pad=int(args.section_pad),
        )
        viz.cv2.imwrite(str(out_dir / "board_bbox_preview.png"), preview)
        viz.cv2.imwrite(str(boards / f"{video_id}__bbox_preview.png"), preview)


def _video_summary_row(
    *,
    video_id: str,
    mode: str,
    log: dict[str, Any],
    n_global: int,
) -> dict[str, Any]:
    crops: list[dict[str, Any]] = []
    if mode == ROI_REPR_SEPARATE:
        for roi in log.get("rois") or []:
            crops.extend(roi.get("crops") or [])
        timestamps = []
        for roi in log.get("rois") or []:
            timestamps.extend(roi.get("timestamps") or [])
        n_overlap = 0
        n_solo = int(log.get("n_roi_images") or 0)
    else:
        stamps = log.get("stamps") or []
        timestamps = [s.get("timestamp") for s in stamps]
        n_overlap = int(log.get("n_overlap") or 0)
        n_solo = int(log.get("n_solo") or 0)
        for s in stamps:
            if s.get("width") is not None:
                crops.append(s)
    widths = [int(c["width"]) for c in crops if c.get("width")]
    heights = [int(c["height"]) for c in crops if c.get("height")]
    pixels = [int(c["crop_pixels"]) for c in crops if c.get("crop_pixels")]
    return {
        "video_id": video_id,
        "mode": mode,
        "n_source_rois": int(log.get("n_source_rois") or 0),
        "n_global": n_global,
        "n_roi_images": int(log.get("n_roi_images") or 0),
        "n_overlap": n_overlap,
        "n_solo": n_solo,
        "union_is_identity": bool(log.get("union_is_identity", False)),
        "crop_width_mean": float(sum(widths) / len(widths)) if widths else 0.0,
        "crop_height_mean": float(sum(heights) / len(heights)) if heights else 0.0,
        "crop_pixels_mean": float(sum(pixels) / len(pixels)) if pixels else 0.0,
        "timestamps": timestamps,
    }


def _write_condition_summary(cond_dir: Path, rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "num_videos": len(rows),
        "num_roi_images": sum(int(r["n_roi_images"]) for r in rows),
        "num_global_images": sum(int(r["n_global"]) for r in rows),
        "videos": rows,
    }
    _write_json(cond_dir / "summary.json", payload)
    csv_path = cond_dir / "summary.csv"
    fieldnames = [
        "video_id",
        "mode",
        "n_source_rois",
        "n_global",
        "n_roi_images",
        "n_overlap",
        "n_solo",
        "union_is_identity",
        "crop_width_mean",
        "crop_height_mean",
        "crop_pixels_mean",
        "n_timestamps",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": row["video_id"],
                    "mode": row["mode"],
                    "n_source_rois": row["n_source_rois"],
                    "n_global": row["n_global"],
                    "n_roi_images": row["n_roi_images"],
                    "n_overlap": row["n_overlap"],
                    "n_solo": row["n_solo"],
                    "union_is_identity": row["union_is_identity"],
                    "crop_width_mean": f"{row['crop_width_mean']:.1f}",
                    "crop_height_mean": f"{row['crop_height_mean']:.1f}",
                    "crop_pixels_mean": f"{row['crop_pixels_mean']:.1f}",
                    "n_timestamps": len(row.get("timestamps") or []),
                }
            )
    return payload


def prepare_roi_repr(
    cfg,
    *,
    modes: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy separate ROI sequences and/or recrop temporally-aware union ROIs."""
    feas = cfg.feasibility
    if feas is None:
        raise ValueError("feasibility config is required")
    exp = cfg.experiment
    needed: list[str] = []
    if modes:
        needed = [parse_roi_repr_mode(m) or m for m in modes]
    else:
        for method in feas.methods:
            mode = repr_mode_for_method(method, exp.roi_repr)
            if mode and mode not in needed:
                needed.append(mode)
        if not needed and exp.roi_repr:
            needed = [exp.roi_repr]
    if not needed:
        return {}

    out_root = Path(cfg.output.dir)
    out_root.mkdir(parents=True, exist_ok=True)
    proposed_root = Path(feas.datasets.proposed_roi_w_fps)
    video_root = Path(feas.datasets.video_root)
    tracks_root = Path(feas.tracks_root)
    summaries: dict[str, Any] = {"conditions": {}}
    per_mode_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in needed}

    for video_id in feas.videos:
        proposed_dir = proposed_root / video_id
        proposed = load_video_bundle_metadata(proposed_dir, require_files=True)
        source_bboxes, frame_w, frame_h = load_source_bboxes(
            video_id=video_id,
            tracks_root=tracks_root,
            video_root=video_root,
            bundle=proposed,
        )
        video_path = video_root / f"{video_id}.mp4"
        info = inspect_video(video_path)
        source_ids = tuple(int(r.roi_id) for r in active_rois(proposed))
        n_global = len(proposed.global_section.frames)

        for mode in needed:
            video_dir = repr_video_dir(out_root, mode, video_id)
            ready = (video_dir / "metadata.json").is_file()
            if ready and not overwrite:
                log_path = video_dir / "roi_log.json"
                log = (
                    json.loads(log_path.read_text(encoding="utf-8"))
                    if log_path.is_file()
                    else {"n_source_rois": len(source_ids), "n_roi_images": 0}
                )
                per_mode_rows[mode].append(
                    _video_summary_row(
                        video_id=video_id, mode=mode, log=log, n_global=n_global
                    )
                )
                logger.info("%s / %s: already exists, skip", video_id, mode)
                continue

            video_dir.mkdir(parents=True, exist_ok=True)
            extra_sections: list[tuple[str, list[Path]]] | None = None
            method = next((m for m, md in ROI_REPR_METHODS.items() if md == mode), mode)

            if mode == ROI_REPR_SEPARATE:
                saved = _copy_separate_rois(
                    bundle=proposed, video_dir=video_dir, overwrite=overwrite
                )
                _write_separate_metadata(
                    video_dir=video_dir,
                    video_id=video_id,
                    bundle=proposed,
                    saved=saved,
                    source_bboxes=source_bboxes,
                    frame_width=frame_w,
                    frame_height=frame_h,
                )
                log = _write_separate_log(
                    video_dir=video_dir,
                    video_id=video_id,
                    bundle=proposed,
                    saved=saved,
                    source_bboxes=source_bboxes,
                )
                labels = {
                    int(r.roi_id): _tube_board_label(mode=mode, roi_id=int(r.roi_id))
                    for r in active_rois(proposed)
                }
            else:
                stamps = plan_union_stamps(
                    proposed,
                    source_bboxes=source_bboxes,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    native_fps=float(info.native_fps),
                    frame_count=int(info.frame_count),
                )
                enclosing = (
                    union_bboxes(
                        list(source_bboxes.values()),
                        frame_width=frame_w,
                        frame_height=frame_h,
                    )
                    if source_bboxes
                    else (0, 0, 0, 0)
                )
                plan = ReprPlan(
                    video_id=video_id,
                    mode=mode,
                    source_roi_ids=source_ids,
                    stamps=tuple(stamps),
                    n_overlap=sum(1 for s in stamps if s.mode == "overlap"),
                    n_solo=sum(1 for s in stamps if s.mode == "solo"),
                    frame_width=frame_w,
                    frame_height=frame_h,
                    source_bboxes_px=source_bboxes,
                    enclosing_bbox_px=enclosing,
                )
                refs = _materialize_union_crops(
                    plan=plan,
                    video_path=video_path,
                    video_dir=video_dir,
                    jpeg_quality=int(feas.jpeg_quality),
                    overwrite=overwrite,
                )
                _write_union_metadata(video_dir=video_dir, plan=plan, refs=refs)
                log = _write_union_log(video_dir=video_dir, plan=plan, refs=refs)
                preview = _annotate_full_frames(
                    proposed_dir=proposed_dir, video_dir=video_dir, plan=plan
                )
                extra_sections = (
                    [("Full-frame  ROI1 / ROI2 / union bbox", preview)] if preview else None
                )
                labels = {UNION_ROI_ID: _tube_board_label(mode=mode, roi_id=UNION_ROI_ID)}

            sampled = load_video_bundle_metadata(video_dir, require_files=True)
            composed = build_bundle(
                global_metadata=proposed, roi_metadata=sampled, video_id=video_id
            )
            items = [
                (item.text, item.image_path)
                for item in ordered_frame_items(composed, input_order=exp.input_order)
                if item.image_path is not None
            ]
            try:
                _render_boards(
                    video_id=video_id,
                    method=method,
                    items=items,
                    out_dir=video_dir,
                    tube_labels=labels,
                    extra_sections=extra_sections,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("board failed for %s / %s: %s", video_id, mode, exc)

            per_mode_rows[mode].append(
                _video_summary_row(
                    video_id=video_id, mode=mode, log=log, n_global=n_global
                )
            )
            logger.info(
                "%s / %s: source_rois=%s roi_images=%s overlap=%s solo=%s",
                video_id,
                mode,
                log.get("n_source_rois"),
                log.get("n_roi_images"),
                log.get("n_overlap"),
                log.get("n_solo"),
            )

    for mode, rows in per_mode_rows.items():
        summaries["conditions"][mode] = _write_condition_summary(
            condition_root(out_root, mode), rows, mode
        )
    _write_json(out_root / "summary.json", summaries)
    logger.info("Wrote ROI representation outputs under %s", out_root)
    return summaries
