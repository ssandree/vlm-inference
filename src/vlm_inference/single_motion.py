"""Single-motion ROI control: union of all proposed tube lifetime bboxes.

Does not re-run Graph→Chebyshev. Spatial bbox is the enclosing box of every
proposed tube lifetime AABB. Temporal samples are the sorted unique timestamps
already present in ``proposed_ROI``. Crops are taken from the original mp4.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FeasibilityConfig
from .inputs import list_image_files
from .roi_margin import (
    _crop_rgb,
    _link_or_copy,
    _save_jpeg_rgb,
    list_roi_samples,
    load_tube_bboxes,
    tracks_json_path,
)
from .video_io import extract_video_frames, inspect_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleMotionPlan:
    bbox_px: tuple[int, int, int, int]
    area_ratio: float
    source_tube_ids: tuple[int, ...]
    timestamps: tuple[float, ...]
    time_tokens: tuple[str, ...]


def clip_bbox(
    bbox: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0 = min(max(x0, 0), frame_width)
    x1 = min(max(x1, 0), frame_width)
    y0 = min(max(y0, 0), frame_height)
    y1 = min(max(y1, 0), frame_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"empty bbox after clip bbox={bbox} frame=({frame_width},{frame_height})"
        )
    return (x0, y0, x1, y1)


def union_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    if not bboxes:
        raise ValueError("cannot union empty bbox list")
    x0 = min(int(b[0]) for b in bboxes)
    y0 = min(int(b[1]) for b in bboxes)
    x1 = max(int(b[2]) for b in bboxes)
    y1 = max(int(b[3]) for b in bboxes)
    return clip_bbox(
        (x0, y0, x1, y1), frame_width=frame_width, frame_height=frame_height
    )


def plan_single_motion(
    video_id: str,
    feas: FeasibilityConfig,
    *,
    roi_frames: str = "default",
) -> SingleMotionPlan:
    ds = feas.datasets
    video_path = Path(ds.video_root) / f"{video_id}.mp4"
    roi_dir = Path(ds.proposed_roi) / video_id
    tracks_path = tracks_json_path(Path(feas.tracks_root), video_id)
    if not video_path.is_file():
        raise FileNotFoundError(f"original video not found: {video_path}")
    if not roi_dir.is_dir():
        raise FileNotFoundError(f"proposed ROI dir not found: {roi_dir}")
    if not tracks_path.is_file():
        raise FileNotFoundError(f"ROI tracks JSON not found: {tracks_path}")

    info = inspect_video(video_path)
    tubes = load_tube_bboxes(
        tracks_path, frame_width=info.width, frame_height=info.height
    )
    samples = list_roi_samples(roi_dir, tubes)
    frame_set = str(roi_frames).strip().lower()
    if frame_set == "fps_1":
        samples = [s for s in samples if "_fps_1" in s.path.stem]
    else:
        samples = [s for s in samples if "_fps_1" not in s.path.stem]
    if not samples:
        raise FileNotFoundError(
            f"no proposed ROI crops in {roi_dir} for roi_frames={frame_set}"
        )

    source_ids = tuple(sorted({s.tube_id for s in samples}))
    bboxes = [tubes[tid].bbox_px for tid in source_ids]
    union = union_bboxes(
        bboxes, frame_width=info.width, frame_height=info.height
    )
    frame_area = max(1, info.width * info.height)
    area_ratio = float((union[2] - union[0]) * (union[3] - union[1])) / float(
        frame_area
    )

    by_time: dict[float, str] = {}
    for sample in samples:
        key = round(float(sample.time_s), 1)
        # Keep optional `_fps_1` in the token so regenerated names stay distinct.
        token = sample.path.stem.rsplit("_t", 1)[-1]
        by_time.setdefault(key, token)
    times = tuple(sorted(by_time))
    tokens = tuple(by_time[t] for t in times)
    return SingleMotionPlan(
        bbox_px=union,
        area_ratio=area_ratio,
        source_tube_ids=source_ids,
        timestamps=times,
        time_tokens=tokens,
    )


def single_roi_name(seq: int, time_token: str) -> str:
    return f"tube01_frame_{seq:06d}_t{time_token}.jpg"


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


def _write_single_motion_metadata(
    *,
    video_id: str,
    combo_dir: Path,
    roi_dir: Path,
    plan: SingleMotionPlan,
    planned_rel: list[tuple[float, str]],
    frame_width: int,
    frame_height: int,
    roi_frames: str,
) -> None:
    """Write metadata.json for single-motion ROI (+ combo) bundles."""
    global_frames: list[dict[str, Any]] = []
    gdir = combo_dir / "global"
    if gdir.is_dir():
        for path in sorted(gdir.glob("global_frame_*.jpg")):
            # global_frame_000001_t00.0.jpg
            stem = path.stem
            try:
                ts = float(stem.rsplit("_t", 1)[-1])
            except ValueError:
                continue
            global_frames.append(
                {"timestamp": ts, "file": f"global/{path.name}"}
            )

    roi_frame_entries = [
        {"timestamp": float(ts), "file": rel} for ts, rel in planned_rel
    ]
    norm = _normalize_bbox_px(
        plan.bbox_px, frame_width=frame_width, frame_height=frame_height
    )
    t0 = float(planned_rel[0][0]) if planned_rel else 0.0
    t1 = float(planned_rel[-1][0]) if planned_rel else 0.0
    roi_obj: dict[str, Any] = {
        "roi_id": 1,
        "time_range": [t0, t1],
        "source_region": "center",
        "normalized_bbox": norm,
        "num_frames": len(roi_frame_entries),
        "frames": list(roi_frame_entries),
    }
    if str(roi_frames).strip().lower() == "fps_1":
        roi_obj["num_frames_fps_1"] = len(roi_frame_entries)
        roi_obj["frames_fps_1"] = list(roi_frame_entries)

    combo_meta: dict[str, Any] = {
        "video_id": video_id,
        "global": {
            "coverage": "full scene",
            "num_frames": len(global_frames),
            "frames": global_frames,
        },
        "rois": [roi_obj],
    }
    if str(roi_frames).strip().lower() == "fps_1":
        combo_meta["roi_fps_1"] = 1.0

    roi_only_meta: dict[str, Any] = {
        "video_id": video_id,
        "global": None,
        "rois": [roi_obj],
    }
    if str(roi_frames).strip().lower() == "fps_1":
        roi_only_meta["roi_fps_1"] = 1.0

    combo_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    (combo_dir / "metadata.json").write_text(
        json.dumps(combo_meta, indent=2) + "\n", encoding="utf-8"
    )
    (roi_dir / "metadata.json").write_text(
        json.dumps(roi_only_meta, indent=2) + "\n", encoding="utf-8"
    )


def prepare_video_single_motion(
    video_id: str,
    feas: FeasibilityConfig,
    *,
    overwrite: bool = False,
    roi_frames: str = "default",
) -> dict[str, Any]:
    ds = feas.datasets
    video_path = Path(ds.video_root) / f"{video_id}.mp4"
    roi_dir = Path(ds.single_motion_roi) / video_id
    combo_dir = Path(ds.single_motion_roi_w_fps) / video_id
    global_dir = Path(ds.fps_0_25) / video_id
    frame_set = str(roi_frames).strip().lower()

    # Nested metadata bundles already prepared — do not flatten/rebuild.
    if (
        not overwrite
        and (combo_dir / "metadata.json").is_file()
        and (roi_dir / "metadata.json").is_file()
    ):
        from .metadata_format import has_metadata_json, load_video_bundle_metadata

        if has_metadata_json(combo_dir):
            try:
                bundle = load_video_bundle_metadata(
                    combo_dir, require_files=True, roi_frames=frame_set
                )
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "%s single_motion cache incompatible with roi_frames=%s (%s) — rebuild",
                    video_id,
                    frame_set,
                    exc,
                )
            else:
                n_global = len(bundle.global_section.frames)
                n_roi = sum(len(r.frames) for r in bundle.rois)
                logger.info(
                    "%s single_motion cache already present (g=%d roi=%d roi_frames=%s) — skip",
                    video_id,
                    n_global,
                    n_roi,
                    frame_set,
                )
                return {
                    "video_id": video_id,
                    "skipped_existing": True,
                    "roi_frames": frame_set,
                    "num_combo_global": n_global,
                    "num_combo_roi": n_roi,
                    "num_combo_total": n_global + n_roi,
                    "num_write": 0,
                    "num_skip": n_roi,
                }

    plan = plan_single_motion(video_id, feas, roi_frames=frame_set)
    info = inspect_video(video_path)

    dest_dir = roi_dir / "roi_01"
    dest_dir.mkdir(parents=True, exist_ok=True)
    planned: list[tuple[Path, int, float, str]] = []
    for seq, (ts, token) in enumerate(zip(plan.timestamps, plan.time_tokens), start=1):
        # Prefer the source token (may already include `_fps_1`).
        name = single_roi_name(seq, token)
        dest = dest_dir / name
        planned.append((dest, info.nearest_frame_index(ts), float(ts), f"roi_01/{name}"))

    need_idx: set[int] = set()
    n_skip = 0
    for dest, idx, _, _ in planned:
        if dest.exists() and not overwrite:
            n_skip += 1
            continue
        need_idx.add(idx)
    frames = extract_video_frames(video_path, need_idx) if need_idx else {}

    n_write = 0
    for dest, idx, _, _ in planned:
        if dest.exists() and not overwrite:
            continue
        crop = _crop_rgb(frames[idx], plan.bbox_px)
        _save_jpeg_rgb(dest, crop, quality=feas.jpeg_quality)
        n_write += 1

    combo_dir.mkdir(parents=True, exist_ok=True)
    n_global = 0
    global_srcs = list_image_files(global_dir)
    global_subdir = global_dir / "global"
    if global_subdir.is_dir():
        global_srcs = list_image_files(global_subdir)
    combo_global = combo_dir / "global" if global_subdir.is_dir() else combo_dir
    for src in global_srcs:
        _link_or_copy(src, combo_global / src.name)
        n_global += 1

    # Sync only the planned ROI crops into the combo bundle.
    n_roi = 0
    planned_rels: list[tuple[float, str]] = []
    for dest, _, ts, rel in planned:
        combo_dest = combo_dir / rel
        _link_or_copy(dest, combo_dest)
        planned_rels.append((ts, rel))
        n_roi += 1

    _write_single_motion_metadata(
        video_id=video_id,
        combo_dir=combo_dir,
        roi_dir=roi_dir,
        plan=plan,
        planned_rel=planned_rels,
        frame_width=info.width,
        frame_height=info.height,
        roi_frames=frame_set,
    )

    row = {
        "video_id": video_id,
        "roi_frames": frame_set,
        "bbox_px": list(plan.bbox_px),
        "area_ratio": plan.area_ratio,
        "num_source_tubes": len(plan.source_tube_ids),
        "source_tube_ids": list(plan.source_tube_ids),
        "num_roi_timestamps": len(plan.timestamps),
        "num_write": n_write,
        "num_skip": n_skip,
        "num_combo_global": n_global,
        "num_combo_roi": n_roi,
        "num_combo_total": n_global + n_roi,
    }
    logger.info(
        "%s single_motion bbox=%s tubes=%s timestamps=%d write=%d skip=%d combo=%d roi_frames=%s",
        video_id,
        plan.bbox_px,
        list(plan.source_tube_ids),
        len(plan.timestamps),
        n_write,
        n_skip,
        n_global + n_roi,
        frame_set,
    )
    return row


def prepare_single_motion_cache(
    feas: FeasibilityConfig,
    *,
    overwrite: bool = False,
    video_ids: list[str] | None = None,
    roi_frames: str = "default",
) -> list[dict[str, Any]]:
    ids = video_ids or list(feas.videos)
    return [
        prepare_video_single_motion(
            video_id, feas, overwrite=overwrite, roi_frames=roi_frames
        )
        for video_id in ids
    ]
