"""Full-frame insertion for Global timestamp ablation.

Keeps ROI crop sampling unchanged. Replaces the Global (full-frame) set with
frames taken from a sampled full-frame pool at video start and/or ROI
start / mid / end times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .experiment import parse_labeled_frame
from .inputs import list_image_files
from .metadata_format import (
    FrameRef,
    GlobalSection,
    VideoBundleMetadata,
    active_rois,
    has_metadata_json,
    load_video_bundle_metadata,
)

logger = logging.getLogger(__name__)

GLOBAL_INSERT_VIDEO_START = "video_start"
GLOBAL_INSERT_START_ONLY = "start_only"
GLOBAL_INSERT_START_END = "start_end"
GLOBAL_INSERT_START_MID_END = "start_mid_end"
VALID_GLOBAL_INSERT_MODES = frozenset(
    {
        GLOBAL_INSERT_VIDEO_START,
        GLOBAL_INSERT_START_ONLY,
        GLOBAL_INSERT_START_END,
        GLOBAL_INSERT_START_MID_END,
    }
)

# Method name -> insert mode. ROI crops come from proposed_ROI_w_fps.
ROI_GLOBAL_INSERT_METHODS: dict[str, str] = {
    "global_roi_video_start_proposed_multi_roi": GLOBAL_INSERT_VIDEO_START,
    "global_roi_start_only_proposed_multi_roi": GLOBAL_INSERT_START_ONLY,
    "global_roi_start_end_proposed_multi_roi": GLOBAL_INSERT_START_END,
    "global_roi_start_mid_end_proposed_multi_roi": GLOBAL_INSERT_START_MID_END,
    "global_roi_start_only_roi_fixed_1fps": GLOBAL_INSERT_START_ONLY,
    "global_roi_start_only_roi_fixed_2fps": GLOBAL_INSERT_START_ONLY,
    "global_roi_start_only_roi_accumulated_motion": GLOBAL_INSERT_START_ONLY,
    "global_roi_start_only_roi_centroid_2fps": GLOBAL_INSERT_START_ONLY,
    "global_roi_start_only_roi_motion_magnitude_adaptive": GLOBAL_INSERT_START_ONLY,
    "global_roi_video_start_roi_fixed_1fps": GLOBAL_INSERT_VIDEO_START,
    "global_roi_video_start_roi_fixed_2fps": GLOBAL_INSERT_VIDEO_START,
    "global_roi_video_start_roi_accumulated_motion": GLOBAL_INSERT_VIDEO_START,
    "global_roi_video_start_roi_centroid_2fps": GLOBAL_INSERT_VIDEO_START,
    "global_roi_per_tube_merge": GLOBAL_INSERT_START_ONLY,
}

ROI_START_BOX_METHODS = frozenset({
    "global_roi_start_box_roi_centroid_2fps",
    "global_roi_start_box_roi_fixed_2fps",
})
ROI_GLOBAL_INSERT_METHODS.update({m: GLOBAL_INSERT_START_ONLY for m in ROI_START_BOX_METHODS})

DEFAULT_DEDUP_EPS = 0.5


@dataclass(frozen=True)
class GlobalInsertRequest:
    roi_id: int
    role: str  # video_start | start | mid | end
    target_t: float


@dataclass(frozen=True)
class GlobalInsertHit:
    request: GlobalInsertRequest
    selected: FrameRef
    kept: bool
    skip_reason: str | None = None


@dataclass(frozen=True)
class GlobalInsertPlan:
    mode: str
    source_dir: str
    dedup_eps: float
    frames: tuple[FrameRef, ...]
    hits: tuple[GlobalInsertHit, ...]


def parse_global_insert_mode(value: str | None) -> str | None:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if not mode or mode in {"none", "null", "off"}:
        return None
    if mode not in VALID_GLOBAL_INSERT_MODES:
        raise ValueError(
            f"global_insert must be one of {sorted(VALID_GLOBAL_INSERT_MODES)}, "
            f"got '{value}'"
        )
    return mode


def condition_dirname(mode: str) -> str:
    parsed = parse_global_insert_mode(mode)
    return parsed or str(mode)


def roi_anchor_requests(bundle: VideoBundleMetadata, mode: str) -> list[GlobalInsertRequest]:
    mode = parse_global_insert_mode(mode)
    if mode is None:
        return []
    if mode == GLOBAL_INSERT_VIDEO_START:
        return [GlobalInsertRequest(roi_id=0, role="video_start", target_t=0.0)]

    requests: list[GlobalInsertRequest] = []
    for roi in active_rois(bundle):
        start = float(roi.time_range[0])
        end = float(roi.time_range[1])
        if mode == GLOBAL_INSERT_START_ONLY:
            roles = (("start", start),)
        elif mode == GLOBAL_INSERT_START_END:
            roles = (("start", start), ("end", end))
        else:
            mid = (start + end) / 2.0
            roles = (("start", start), ("mid", mid), ("end", end))
        for role, target_t in roles:
            requests.append(
                GlobalInsertRequest(roi_id=int(roi.roi_id), role=role, target_t=target_t)
            )
    requests.sort(key=lambda r: (r.target_t, r.roi_id, r.role))
    return requests


def nearest_full_frame(candidates: Sequence[FrameRef], target_t: float) -> FrameRef:
    if not candidates:
        raise ValueError("no full-frame candidates to snap Global timestamps onto")
    return min(
        candidates,
        key=lambda frame: (
            abs(float(frame.timestamp) - float(target_t)),
            float(frame.timestamp),
            int(frame.frame_index),
        ),
    )


def _already_kept(
    *,
    selected: FrameRef,
    kept_frames: list[FrameRef],
    dedup_eps: float,
) -> str | None:
    selected_path = str(selected.path.resolve())
    for prev in kept_frames:
        if str(prev.path.resolve()) == selected_path:
            return f"same file as t={prev.timestamp}s"
        if abs(float(prev.timestamp) - float(selected.timestamp)) <= float(dedup_eps):
            return (
                f"timestamp {selected.timestamp}s within {dedup_eps}s of "
                f"kept t={prev.timestamp}s"
            )
    return None


def plan_global_insert(
    bundle: VideoBundleMetadata,
    *,
    mode: str,
    candidates: list[FrameRef],
    source_dir: Path | str,
    dedup_eps: float = DEFAULT_DEDUP_EPS,
) -> GlobalInsertPlan:
    mode = parse_global_insert_mode(mode)
    if mode is None:
        raise ValueError("plan_global_insert requires a global_insert mode")
    if not candidates:
        raise ValueError(f"no full-frame candidates under {source_dir}")

    hits: list[GlobalInsertHit] = []
    kept: list[FrameRef] = []
    for request in roi_anchor_requests(bundle, mode):
        selected = nearest_full_frame(candidates, request.target_t)
        reason = _already_kept(
            selected=selected, kept_frames=kept, dedup_eps=dedup_eps
        )
        if reason is None:
            kept.append(selected)
            hits.append(GlobalInsertHit(request=request, selected=selected, kept=True))
        else:
            hits.append(
                GlobalInsertHit(
                    request=request,
                    selected=selected,
                    kept=False,
                    skip_reason=reason,
                )
            )

    kept_sorted = tuple(
        sorted(kept, key=lambda frame: (float(frame.timestamp), int(frame.frame_index)))
    )
    n_skip = sum(1 for hit in hits if not hit.kept)
    logger.info(
        "global_insert mode=%s video=%s requests=%d kept=%d skipped=%d source=%s",
        mode,
        bundle.video_id,
        len(hits),
        len(kept_sorted),
        n_skip,
        source_dir,
    )
    return GlobalInsertPlan(
        mode=mode,
        source_dir=str(Path(source_dir)),
        dedup_eps=float(dedup_eps),
        frames=kept_sorted,
        hits=tuple(hits),
    )


def apply_global_insert_plan(
    bundle: VideoBundleMetadata,
    plan: GlobalInsertPlan,
) -> VideoBundleMetadata:
    warning = (
        f"global_insert={plan.mode}: kept {len(plan.frames)} Full-frame(s) "
        f"from {plan.source_dir} (requests={len(plan.hits)}, "
        f"dedup_eps={plan.dedup_eps})"
    )
    logger.info(warning)
    return replace(
        bundle,
        global_section=GlobalSection(
            coverage=bundle.global_section.coverage or "full scene",
            num_frames=len(plan.frames),
            frames=plan.frames,
        ),
        warnings=bundle.warnings + (warning,),
    )


def apply_global_insert(
    bundle: VideoBundleMetadata,
    *,
    mode: str,
    candidates: list[FrameRef],
    source_dir: Path | str,
    dedup_eps: float = DEFAULT_DEDUP_EPS,
) -> tuple[VideoBundleMetadata, GlobalInsertPlan]:
    plan = plan_global_insert(
        bundle,
        mode=mode,
        candidates=candidates,
        source_dir=source_dir,
        dedup_eps=dedup_eps,
    )
    if not plan.frames:
        raise ValueError(
            f"global_insert={mode}: no Full-frames kept for {bundle.video_id}"
        )
    return apply_global_insert_plan(bundle, plan), plan


def full_frame_source_dir(
    video_id: str,
    *,
    datasets,
    source: str = "fps_1",
) -> Path:
    key = str(source).strip().lower().replace("-", "_")
    mapping = {
        "fps_1": datasets.fps_1,
        "fps1": datasets.fps_1,
        "fps_2": datasets.fps_2,
        "fps2": datasets.fps_2,
        "fps_0_25": datasets.fps_0_25,
        "fps0_25": datasets.fps_0_25,
        "fps_0.25": datasets.fps_0_25,
    }
    if key not in mapping:
        raise ValueError(
            f"global_insert_source must be one of fps_1 / fps_2 / fps_0_25, got '{source}'"
        )
    return Path(mapping[key]) / video_id


def _global_image_paths(directory: Path) -> list[Path]:
    root = Path(directory)
    nested = root / "global"
    if nested.is_dir():
        nested_files = sorted(
            path for path in nested.glob("global_frame_*.jpg") if path.is_file()
        )
        if nested_files:
            return nested_files
    flat = [
        path
        for path in list_image_files(root)
        if path.name.startswith("global_frame_")
    ]
    return sorted(flat)


def load_full_frame_candidates(
    directory: Path | str,
    *,
    video_id: str | None = None,
    require_files: bool = True,
) -> list[FrameRef]:
    """Load sampled Full-frames (metadata.json when present, else filenames)."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"full-frame directory not found: {root}")

    if has_metadata_json(root):
        bundle = load_video_bundle_metadata(root, require_files=require_files)
        frames = list(bundle.global_section.frames)
        if frames:
            return frames
        logger.warning("metadata.json under %s has no global frames; trying files", root)

    from .config import ExperimentConfig

    cfg = ExperimentConfig()
    frames: list[FrameRef] = []
    for i, path in enumerate(_global_image_paths(root), start=1):
        parsed = parse_labeled_frame(str(path), cfg)
        if parsed is None or parsed.kind != "global":
            continue
        rel = path.resolve().relative_to(root)
        frames.append(
            FrameRef(
                frame_index=int(parsed.frame_idx) if parsed.frame_idx is not None else i,
                timestamp=float(parsed.time_s),
                file=str(rel).replace("\\", "/"),
                path=path.resolve(),
            )
        )
    if not frames:
        raise FileNotFoundError(f"no global_frame_*.jpg under {root} (video_id={video_id})")
    frames.sort(key=lambda frame: (float(frame.timestamp), int(frame.frame_index)))
    return frames


def ensure_full_frame_pool(
    video_id: str,
    *,
    datasets,
    source: str = "fps_1",
    video_path: Path,
    cache_root: Path,
    jpeg_quality: int = 95,
) -> Path:
    """Return a 1fps/2fps full-frame directory, extracting from the mp4 if needed."""
    existing = full_frame_source_dir(video_id, datasets=datasets, source=source)
    try:
        if existing.is_dir() and _global_image_paths(existing):
            return existing
    except FileNotFoundError:
        pass

    key = str(source).strip().lower().replace("-", "_")
    fps_map = {"fps_1": 1.0, "fps1": 1.0, "fps_2": 2.0, "fps2": 2.0, "fps_0_25": 0.25, "fps0_25": 0.25}
    if key not in fps_map:
        raise ValueError(f"unsupported full-frame source '{source}'")
    sample_fps = fps_map[key]
    out = Path(cache_root) / "full_frames" / key / video_id
    if out.is_dir() and _global_image_paths(out):
        return out

    from .roi_margin import _save_jpeg_rgb
    from .video_io import extract_video_frames, inspect_video

    info = inspect_video(video_path)
    duration = (max(int(info.frame_count), 1) - 1) / float(info.native_fps or 30.0)
    times: list[float] = []
    t = 0.0
    step = 1.0 / float(sample_fps)
    while t <= duration + 1e-6:
        times.append(round(t, 3))
        t += step
    indices = {info.nearest_frame_index(stamp) for stamp in times}
    decoded = extract_video_frames(Path(video_path), indices)
    out.mkdir(parents=True, exist_ok=True)
    for i, stamp in enumerate(times, start=1):
        idx = info.nearest_frame_index(stamp)
        dest = out / f"global_frame_{i:06d}_t{stamp:04.1f}.jpg"
        if dest.is_file():
            continue
        _save_jpeg_rgb(dest, decoded[idx], quality=int(jpeg_quality))
    logger.info(
        "Extracted %d %s full-frames for %s -> %s",
        len(times),
        key,
        video_id,
        out,
    )
    return out


def draw_roi_start_boxes(
    bundle: VideoBundleMetadata, plan: GlobalInsertPlan, output_dir: Path,
) -> VideoBundleMetadata:
    """Overlay original crop bboxes; retain insertion times and every ROI frame.

    Include all temporally active ROIs and all anchors assigned to this kept
    frame (including snapped/deduplicated starts). PNG avoids changing pixels
    outside rectangle strokes through JPEG re-encoding.
    """
    import json
    from PIL import Image, ImageDraw

    if plan.mode != GLOBAL_INSERT_START_ONLY:
        raise ValueError("ROI boxes require start_only insertion")
    rois = list(active_rois(bundle))
    destination = Path(output_dir) / bundle.video_id
    destination.mkdir(parents=True, exist_ok=True)
    frames = []
    manifest = []
    for frame in plan.frames:
        anchor_ids = {
            hit.request.roi_id for hit in plan.hits
            if nearest_full_frame(plan.frames, hit.selected.timestamp) == frame
        }
        selected_rois = [r for r in rois if r.roi_id in anchor_ids
                         or r.time_range[0] <= frame.timestamp <= r.time_range[1]]
        if not selected_rois:
            raise ValueError(f"No ROI box for {bundle.video_id} t={frame.timestamp}")
        with Image.open(frame.path) as source:
            canvas = source.convert("RGB")
        width, height = canvas.size
        stroke = max(4, round(min(width, height) / 180))
        draw = ImageDraw.Draw(canvas)
        boxes = []
        for roi in selected_rois:
            x0, y0, x1, y1 = roi.normalized_bbox
            rect = [max(0, min(width - 1, round(x0 * width))),
                    max(0, min(height - 1, round(y0 * height))),
                    max(0, min(width - 1, round(x1 * width))),
                    max(0, min(height - 1, round(y1 * height)))]
            draw.rectangle(rect, outline=(255, 255, 0), width=stroke + 4)
            draw.rectangle(rect, outline=(255, 0, 0), width=stroke)
            boxes.append({"roi_id": roi.roi_id, "bbox_px": rect})
        path = destination / (frame.path.stem + ".png")
        canvas.save(path)
        frames.append(replace(frame, path=path.resolve(), file=str(path.resolve())))
        manifest.append({"timestamp": frame.timestamp, "source": str(frame.path),
                         "image": str(path.resolve()), "boxes": boxes})
    (destination / "boxes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return replace(bundle, global_section=replace(bundle.global_section, frames=tuple(frames)))
