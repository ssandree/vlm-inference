"""Recrop proposed ROI tubes from original video with spatial bbox margin.

Does not change tube IDs, timestamps, or temporal sampling. Existing
``proposed_ROI`` JPEGs are never overwritten. Crops are taken from the original
mp4 using the Graph→Chebyshev lifetime AABB in ``*_roi_tracks.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .config import FeasibilityConfig
from .inputs import list_image_files
from .video_io import extract_video_frames, inspect_video

logger = logging.getLogger(__name__)

MarginVariant = Literal["margin_25", "margin_50", "always_margin"]

MARGIN_VARIANTS: dict[str, float | None] = {
    "margin_25": 0.25,
    "margin_50": 0.50,
    "always_margin": None,  # always expand
}

SCORE_KEYS = ("importance", "importance_score", "score")


@dataclass(frozen=True)
class TubeBBox:
    tube_id: int
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1 original lifetime AABB
    area_ratio: float
    num_cells: int
    extra_score: float | None


@dataclass(frozen=True)
class RoiSample:
    path: Path
    tube_id: int
    frame_idx_name: int | None
    time_s: float
    bbox_px: tuple[int, int, int, int]
    area_ratio: float


def tracks_json_path(tracks_root: Path, video_id: str) -> Path:
    return tracks_root / f"{video_id}_roi_tracks.json"


def grid_bbox_to_pixels(
    bbox: tuple[int, int, int, int],
    *,
    unit_pixel_size: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(x0 * unit_pixel_size),
        int(y0 * unit_pixel_size),
        int(min(frame_width, x1 * unit_pixel_size)),
        int(min(frame_height, y1 * unit_pixel_size)),
    )


def load_tube_bboxes(
    tracks_path: Path,
    *,
    frame_width: int,
    frame_height: int,
) -> dict[int, TubeBBox]:
    return _load_tube_bboxes_cached(str(Path(tracks_path).resolve()), int(frame_width), int(frame_height))


@lru_cache(maxsize=16)
def _load_tube_bboxes_cached(
    tracks_path: str, frame_width: int, frame_height: int
) -> dict[int, TubeBBox]:
    data = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    variant = str(data.get("variant") or "")
    allowed = {
        "graph_then_constrained_chebyshev",
        "coarse_gc_compose_v2c",
        "stage3_v2_coarse_gc_compose",
        "stage3_light_uf_compose",
        "stage3_v2_light_uf_compose",
    }
    if variant and variant not in allowed:
        raise ValueError(
            f"{tracks_path}: unexpected Stage3 variant {variant!r} (allowed {sorted(allowed)})"
        )
    cell_px = int(data.get("cell_px") or 64)
    tracks = data.get("tracks") or data.get("tubes") or []
    frame_area = max(1, int(frame_width) * int(frame_height))
    out: dict[int, TubeBBox] = {}
    for tube in tracks:
        bbox_grid = tube.get("spatial_bbox_grid")
        if not bbox_grid:
            continue
        tube_id = int(tube.get("tube_id") or 0)
        if tube_id <= 0:
            continue
        px = grid_bbox_to_pixels(
            tuple(int(v) for v in bbox_grid),
            unit_pixel_size=cell_px,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        x0, y0, x1, y1 = px
        if x1 <= x0 or y1 <= y0:
            continue
        area_ratio = float((x1 - x0) * (y1 - y0)) / float(frame_area)
        extra: float | None = None
        for key in SCORE_KEYS:
            if key in tube and tube[key] is not None:
                extra = float(tube[key])
                break
        out[tube_id] = TubeBBox(
            tube_id=tube_id,
            bbox_px=px,
            area_ratio=area_ratio,
            num_cells=int(tube.get("num_cells") or 0),
            extra_score=extra,
        )
    if not out:
        raise ValueError(f"{tracks_path}: no usable ROI tubes")
    return out


def select_top1_tube_id(tubes: dict[int, TubeBBox], roi_files: list[Path]) -> int:
    """Pick one proposed ROI tube.

    Use an existing importance/score field when present; otherwise keep the
    original ROI ordering and use tube01. Do not invent a new ranking.
    """
    present = sorted(
        {
            int(p.name.split("_", 1)[0].replace("tube", ""))
            for p in roi_files
            if p.name.startswith("tube")
        }
    )
    if not present:
        raise ValueError("no ROI tube files found for top-1 selection")

    scored = [tubes[i] for i in present if i in tubes and tubes[i].extra_score is not None]
    if scored:
        scored.sort(key=lambda t: (-float(t.extra_score), t.tube_id))
        return int(scored[0].tube_id)
    if 1 in present:
        return 1
    return int(present[0])


def apply_spatial_margin(
    bbox: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
    area_ratio: float,
    area_threshold: float | None,
) -> tuple[tuple[int, int, int, int], bool]:
    """Expand left/right by bbox width and top/bottom by bbox height, then clip."""
    x0, y0, x1, y1 = (int(v) for v in bbox)
    apply = area_threshold is None or area_ratio <= float(area_threshold)
    if apply:
        bw = x1 - x0
        bh = y1 - y0
        x0 -= bw
        x1 += bw
        y0 -= bh
        y1 += bh
    x0 = min(max(x0, 0), frame_width)
    x1 = min(max(x1, 0), frame_width)
    y0 = min(max(y0, 0), frame_height)
    y1 = min(max(y1, 0), frame_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"empty bbox after margin/clip original={bbox} "
            f"frame=({frame_width},{frame_height})"
        )
    return (x0, y0, x1, y1), apply


def _link_or_copy(src: Path, dst: Path) -> None:
    import os
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    src = src.resolve()
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def _save_jpeg_rgb(path: Path, rgb, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    image.save(path, format="JPEG", quality=int(quality), subsampling=0, optimize=False)


def _crop_rgb(frame, bbox: tuple[int, int, int, int]):
    x0, y0, x1, y1 = bbox
    h, w = frame.shape[:2]
    x0 = min(max(int(x0), 0), w)
    x1 = min(max(int(x1), 0), w)
    y0 = min(max(int(y0), 0), h)
    y1 = min(max(int(y1), 0), h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop bbox={(x0, y0, x1, y1)} frame={(w, h)}")
    return frame[y0:y1, x0:x1]


def parse_roi_filename(path: Path) -> tuple[int, int | None, float]:
    from .experiment import parse_labeled_frame
    from .config import ExperimentConfig

    parsed = parse_labeled_frame(str(path), ExperimentConfig())
    if parsed is None or parsed.kind != "roi" or parsed.roi_id is None:
        raise ValueError(f"not an ROI filename: {path.name}")
    return parsed.roi_id, parsed.frame_idx, parsed.time_s


def list_roi_crop_files(roi_dir: Path) -> list[Path]:
    """ROI JPEGs under a video bundle: flat ``tube*.jpg`` and/or ``roi_*/tube*.jpg``."""
    root = Path(roi_dir).expanduser()
    if not root.is_dir():
        return []
    found: list[Path] = []
    for path in list_image_files(root):
        if path.name.startswith("tube"):
            found.append(path)
    for sub in sorted(root.glob("roi_*")):
        if not sub.is_dir():
            continue
        for path in list_image_files(sub):
            if path.name.startswith("tube"):
                found.append(path)
    # Stable order: tube id, then filename.
    return sorted(found, key=lambda p: (p.name, str(p)))


def list_roi_samples(
    roi_dir: Path,
    tubes: dict[int, TubeBBox],
    *,
    tube_id: int | None = None,
) -> list[RoiSample]:
    samples: list[RoiSample] = []
    for path in list_roi_crop_files(roi_dir):
        tid, frame_idx, time_s = parse_roi_filename(path)
        if tube_id is not None and tid != tube_id:
            continue
        if tid not in tubes:
            raise KeyError(f"{path.name}: tube{tid:02d} missing from tracks JSON")
        tube = tubes[tid]
        samples.append(
            RoiSample(
                path=path,
                tube_id=tid,
                frame_idx_name=frame_idx,
                time_s=time_s,
                bbox_px=tube.bbox_px,
                area_ratio=tube.area_ratio,
            )
        )
    return samples


def margin_dir(margin_root: Path, variant: str, video_id: str) -> Path:
    return margin_root / variant / video_id


def prepare_video_margin_cache(
    video_id: str,
    feas: FeasibilityConfig,
    *,
    variants: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write margin ROI JPEGs for one video. Existing original ROI is untouched."""
    wanted = variants or list(MARGIN_VARIANTS)
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
    originals = list_roi_samples(roi_dir, tubes)
    if not originals:
        raise FileNotFoundError(f"no ROI crops in {roi_dir}")

    # Skip rebuild when nested margin bundles already exist (metadata.json present).
    if not overwrite:
        ready = True
        for variant in wanted:
            dest_root = margin_dir(Path(ds.margin_root), variant, video_id)
            if not (dest_root / "metadata.json").is_file():
                ready = False
                break
            if not list_roi_crop_files(dest_root):
                ready = False
                break
        if ready:
            logger.info(
                "%s margin cache already present under %s — skip",
                video_id,
                ds.margin_root,
            )
            return {
                "video_id": video_id,
                "num_roi": len(originals),
                "skipped_existing": True,
                "variants": {
                    v: {
                        "num_margin_applied": 0,
                        "num_margin_not_applied": 0,
                        "num_write": 0,
                        "num_skip": len(originals),
                    }
                    for v in wanted
                },
            }

    planned: list[tuple[str, Path, int, tuple[int, int, int, int], bool, Path]] = []
    stats = {
        variant: {"num_margin_applied": 0, "num_margin_not_applied": 0, "num_write": 0, "num_skip": 0}
        for variant in wanted
    }

    for sample in originals:
        native_idx = info.nearest_frame_index(sample.time_s)
        try:
            rel = sample.path.relative_to(roi_dir)
        except ValueError:
            rel = Path(sample.path.name)
        for variant in wanted:
            dest = margin_dir(Path(ds.margin_root), variant, video_id) / rel
            expanded, applied = apply_spatial_margin(
                sample.bbox_px,
                frame_width=info.width,
                frame_height=info.height,
                area_ratio=sample.area_ratio,
                area_threshold=MARGIN_VARIANTS[variant],
            )
            if applied:
                stats[variant]["num_margin_applied"] += 1
            else:
                stats[variant]["num_margin_not_applied"] += 1
            planned.append(
                (variant, dest, native_idx, expanded, applied, sample.path)
            )

    need_idx: set[int] = set()
    for variant, dest, native_idx, _bbox, applied, src in planned:
        if dest.exists() and not overwrite:
            stats[variant]["num_skip"] += 1
            continue
        if applied:
            need_idx.add(native_idx)
        # unapplied: hardlink original JPEG, no decode needed

    frames = extract_video_frames(video_path, need_idx) if need_idx else {}

    for variant, dest, native_idx, bbox, applied, src in planned:
        if dest.exists() and not overwrite:
            continue
        if not applied:
            _link_or_copy(src, dest)
            stats[variant]["num_write"] += 1
            continue
        crop = _crop_rgb(frames[native_idx], bbox)
        _save_jpeg_rgb(dest, crop, quality=feas.jpeg_quality)
        stats[variant]["num_write"] += 1

    logger.info("%s margin cache: %s", video_id, stats)
    return {"video_id": video_id, "num_roi": len(originals), "variants": stats}


def prepare_margin_cache(
    feas: FeasibilityConfig,
    *,
    variants: list[str] | None = None,
    overwrite: bool = False,
    video_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    ids = video_ids or list(feas.videos)
    rows = []
    for video_id in ids:
        rows.append(
            prepare_video_margin_cache(
                video_id, feas, variants=variants, overwrite=overwrite
            )
        )
    return rows
