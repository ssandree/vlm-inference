"""VIRAT 9-video feasibility jobs: input assembly, stats, cache keys."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image

from .config import AppConfig, FeasibilityConfig
from .experiment import INPUT_FORMAT_VERSION, LabeledFrame, build_labeled_frames
from .inputs import list_image_files
from .metadata_format import (
    METADATA_INPUT_FORMAT_VERSION,
    VideoBundleMetadata,
    active_rois,
    build_bundle,
    has_metadata_json,
    iter_bundle_image_uris,
    keep_first_global_frame,
    load_video_bundle_metadata,
)
from .roi_margin import load_tube_bboxes, tracks_json_path
from .single_motion import plan_single_motion
from .video_io import inspect_video, resolve_video_file
from .global_insert import (
    ROI_GLOBAL_INSERT_METHODS,
    ROI_START_BOX_METHODS,
    draw_roi_start_boxes,
    apply_global_insert,
    ensure_full_frame_pool,
    full_frame_source_dir,
    load_full_frame_candidates,
    parse_global_insert_mode,
)
from .motion_adaptive import (
    apply_sampled_rois,
    load_roi_sampling_labels,
    sampling_mode_for_method,
    sampling_video_dir,
)
from .roi_union import (
    apply_roi_repr,
    load_bbox_preview_section,
    load_roi_repr_labels,
    repr_mode_for_method,
    repr_video_dir,
)

logger = logging.getLogger(__name__)

CANONICAL_METHODS = (
    "full_2fps",
    "full_1fps",
    "full_0.25fps",
    "global_0.25fps_single_motion_roi",
    "global_first_0.25fps_single_motion_roi",
    "single_motion_roi_original",
    "global_0.25fps_proposed_multi_roi",
    "global_first_0.25fps_proposed_multi_roi",
    "global_0.25fps_proposed_multi_roi_no_bbox",
    "global_0.25fps_proposed_multi_roi_margin_10",
    "proposed_multi_roi_only_margin_10",
    "global_roi_video_start_proposed_multi_roi",
    "global_roi_start_only_proposed_multi_roi",
    "global_roi_start_end_proposed_multi_roi",
    "global_roi_start_mid_end_proposed_multi_roi",
    "global_roi_start_box_roi_centroid_2fps",
    "global_roi_start_box_roi_fixed_2fps",
    "global_roi_start_only_roi_fixed_1fps",
    "global_roi_start_only_roi_fixed_2fps",
    "global_roi_start_only_roi_accumulated_motion",
    "global_roi_start_only_roi_centroid_2fps",
    "global_roi_start_only_roi_motion_magnitude_adaptive",
    "global_roi_video_start_roi_fixed_1fps",
    "global_roi_video_start_roi_fixed_2fps",
    "global_roi_video_start_roi_accumulated_motion",
    "global_roi_video_start_roi_centroid_2fps",
    "global_roi_per_tube_merge",
    "global_0.25fps_roi_fixed_1fps",
    "global_0.25fps_roi_fixed_2fps",
    "global_0.25fps_roi_adaptive_0.5_1_2",
    "global_0.25fps_roi_adaptive_0.5_1_4",
    "global_0.25fps_roi_accumulated_motion",
    "global_0.25fps_roi_accumulated_motion_cap2fps",
    "global_0.25fps_roi_burst_2fps",
    "global_0.25fps_roi_separate",
    "global_0.25fps_roi_union",
)

# Global 0.25fps + proposed multi-ROI tubes (proposed_ROI_w_fps).
PROPOSED_W_FPS_METHODS = frozenset(
    {
        "global_0.25fps_proposed_multi_roi",
        "global_first_0.25fps_proposed_multi_roi",
        "global_0.25fps_proposed_multi_roi_no_bbox",
        "global_0.25fps_proposed_multi_roi_margin_10",
        "global_roi_video_start_proposed_multi_roi",
        "global_roi_start_only_proposed_multi_roi",
        "global_roi_start_end_proposed_multi_roi",
        "global_roi_start_mid_end_proposed_multi_roi",
        "global_roi_start_box_roi_centroid_2fps",
        "global_roi_start_box_roi_fixed_2fps",
        "global_roi_start_only_roi_fixed_1fps",
        "global_roi_start_only_roi_fixed_2fps",
        "global_roi_start_only_roi_accumulated_motion",
        "global_roi_start_only_roi_centroid_2fps",
        "global_roi_start_only_roi_motion_magnitude_adaptive",
        "global_roi_video_start_roi_fixed_1fps",
        "global_roi_video_start_roi_fixed_2fps",
        "global_roi_video_start_roi_accumulated_motion",
        "global_roi_video_start_roi_centroid_2fps",
        "global_roi_per_tube_merge",
        "global_0.25fps_roi_fixed_1fps",
        "global_0.25fps_roi_fixed_2fps",
        "global_0.25fps_roi_adaptive_0.5_1_2",
        "global_0.25fps_roi_adaptive_0.5_1_4",
        "global_0.25fps_roi_accumulated_motion",
    "global_0.25fps_roi_accumulated_motion_cap2fps",
        "global_0.25fps_roi_burst_2fps",
        "global_0.25fps_roi_separate",
        "global_0.25fps_roi_union",
    }
)

# Same ROI sources as the full-Global counterparts; only the first 0.25fps
# Global frame is kept (see keep_first_global_frame).
FIRST_GLOBAL_METHODS = frozenset(
    {
        "global_first_0.25fps_single_motion_roi",
        "global_first_0.25fps_proposed_multi_roi",
    }
)

SINGLE_MOTION_W_FPS_METHODS = frozenset(
    {
        "global_0.25fps_single_motion_roi",
        "global_first_0.25fps_single_motion_roi",
    }
)

# Proposed multi-ROI crops only (no Global frames).
PROPOSED_ROI_ONLY_METHODS = frozenset(
    {
        "proposed_multi_roi_only_margin_10",
    }
)

# Single-motion union ROI only (single_motion_ROI), no global frames.
SINGLE_MOTION_ROI_ONLY_METHODS = frozenset({"single_motion_roi_original"})

PROPOSED_MULTI_METHODS = PROPOSED_W_FPS_METHODS | PROPOSED_ROI_ONLY_METHODS

# Legacy alias (single-ROI caption copy is disabled; inputs differ again).
SINGLE_ROI_EQUIV_METHODS = frozenset(
    {
        "global_0.25fps_proposed_multi_roi",
        "single_motion_roi_original",
        "global_0.25fps_single_motion_roi",
    }
)


def _as_file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _local_path(path: str | Path) -> Path:
    text = str(path)
    if text.startswith("file://"):
        parsed = urlparse(text)
        return Path(unquote(parsed.path))
    return Path(text)


@dataclass
class MethodInputs:
    video_id: str
    method: str
    frames: list[LabeledFrame]
    input_dirs: list[str]
    num_global_images: int
    num_roi_images: int
    num_roi_tubes: int
    num_source_tubes: int | None = None
    single_motion_bbox: list[int] | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    roi_bboxes_px: dict[int, tuple[int, int, int, int]] | None = None
    bundle: VideoBundleMetadata | None = None
    bundle_dir: str | None = None
    roi_area_ratios: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    global_insert_mode: str | None = None
    global_insert_plan: Any | None = None
    roi_sampling_mode: str | None = None
    roi_sampling_labels: dict[int, str] | None = None
    roi_repr_mode: str | None = None
    roi_repr_labels: dict[int, str] | None = None
    roi_repr_extra_sections: list[tuple[str, list[Path]]] | None = None
    global_scale: float = 1.0
    original_global_resolution: tuple[int, int] | None = None

    @property
    def num_total_images(self) -> int:
        return self.num_global_images + self.num_roi_images

    @property
    def input_paths(self) -> list[str]:
        return [f.path for f in self.frames]


@dataclass
class ImageStats:
    image_sizes: list[dict[str, Any]]
    total_input_pixels: int


def _labeled_frames_from_bundle(bundle: VideoBundleMetadata) -> list[LabeledFrame]:
    """Project metadata frames into LabeledFrame for stats/cache (order preserved)."""
    frames: list[LabeledFrame] = []
    for fr in bundle.global_section.frames:
        frames.append(
            LabeledFrame(
                path=fr.path.resolve().as_uri(),
                label=f"[Global] t={fr.timestamp}s",
                time_s=float(fr.timestamp),
                kind="global",
                roi_id=None,
                frame_idx=int(fr.frame_index),
            )
        )
    for roi in active_rois(bundle):
        for fr in roi.frames:
            frames.append(
                LabeledFrame(
                    path=fr.path.resolve().as_uri(),
                    label=f"[ROI{roi.roi_id}] t={fr.timestamp}s",
                    time_s=float(fr.timestamp),
                    kind="roi",
                    roi_id=int(roi.roi_id),
                    frame_idx=int(fr.frame_index),
                )
            )
    return frames


def _try_load_bundle(
    directory: Path,
    *,
    label: str,
    errors: list[str],
    warnings: list[str],
    roi_frames: str = "default",
) -> VideoBundleMetadata | None:
    """Load metadata.json when present. Returns None if absent (not an error)."""
    if not has_metadata_json(directory):
        return None
    try:
        bundle = load_video_bundle_metadata(
            directory, require_files=True, roi_frames=roi_frames
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid {label} metadata.json under {directory}: {exc}")
        return None
    warnings.extend(bundle.warnings)
    return bundle


def _load_bundle_or_none(
    directory: Path,
    *,
    label: str,
    errors: list[str],
    warnings: list[str],
    roi_frames: str = "default",
) -> tuple[VideoBundleMetadata | None, bool]:
    """Return (bundle, used_metadata_path).

    ``used_metadata_path`` is True when metadata.json exists (even if invalid),
    so callers do not fall back to flat filename listing.
    """
    if not has_metadata_json(directory):
        return None, False
    bundle = _try_load_bundle(
        directory,
        label=label,
        errors=errors,
        warnings=warnings,
        roi_frames=roi_frames,
    )
    return bundle, True


def _require_dir(path: Path, label: str) -> str | None:
    if not path.is_dir():
        return f"missing {label}: {path}"
    return None


def _require_images(path: Path, label: str) -> str | None:
    missing = _require_dir(path, label)
    if missing:
        return missing
    if not list_image_files(path):
        return f"empty {label}: {path}"
    return None


def _require_roi_crops(path: Path, label: str) -> str | None:
    """Require ROI tube JPEGs under flat or ``roi_*/`` layout."""
    from .roi_margin import list_roi_crop_files

    missing = _require_dir(path, label)
    if missing:
        return missing
    if not list_roi_crop_files(path):
        return f"empty {label}: {path}"
    return None


def _dataset_video_path(video_id: str, datasets) -> Path:
    return resolve_video_file(video_id, datasets.all_video_roots())


def method_input_dirs(
    video_id: str, method: str, feas: FeasibilityConfig
) -> list[str]:
    ds = feas.datasets
    if method == "full_2fps":
        return [str(Path(ds.fps_2) / video_id)]
    if method == "full_1fps":
        return [str(Path(ds.fps_1) / video_id)]
    if method == "full_0.25fps":
        return [str(Path(ds.fps_0_25) / video_id)]
    if method in SINGLE_MOTION_W_FPS_METHODS:
        return [str(Path(ds.single_motion_roi_w_fps) / video_id)]
    if method in SINGLE_MOTION_ROI_ONLY_METHODS:
        return [str(Path(ds.single_motion_roi) / video_id)]
    if method in PROPOSED_ROI_ONLY_METHODS:
        return [str(Path(ds.proposed_roi) / video_id)]
    if method in PROPOSED_W_FPS_METHODS:
        return [str(Path(ds.proposed_roi_w_fps) / video_id)]
    raise ValueError(f"unknown feasibility method: {method}")


def insert_mode_for_method(method: str, configured: str | None) -> str | None:
    """Dedicated ablation methods win; otherwise apply config/CLI to Global+ROI."""
    named = ROI_GLOBAL_INSERT_METHODS.get(method)
    if named is not None:
        return named
    if method in FIRST_GLOBAL_METHODS:
        return None
    if method in PROPOSED_W_FPS_METHODS or method in SINGLE_MOTION_W_FPS_METHODS:
        return parse_global_insert_mode(configured)
    return None


def filter_methods_for_global_insert(methods: list[str], mode: str | None) -> list[str]:
    """If CLI/config names a condition, keep matching dedicated methods when present."""
    parsed = parse_global_insert_mode(mode)
    if parsed is None:
        return list(methods)
    matching = [m for m in methods if ROI_GLOBAL_INSERT_METHODS.get(m) == parsed]
    return matching if matching else list(methods)


def _roi_meta_for_video(video_id: str, feas: FeasibilityConfig):
    ds = feas.datasets
    video_path = _dataset_video_path(video_id, ds)
    tracks_path = tracks_json_path(Path(feas.tracks_root), video_id)
    info = inspect_video(video_path)
    tubes = load_tube_bboxes(
        tracks_path, frame_width=info.width, frame_height=info.height
    )
    return info, tubes


def _area_ratios(roi_frames: list[LabeledFrame], tubes: dict) -> list[float]:
    ratios: list[float] = []
    for frame in roi_frames:
        if frame.roi_id is None or frame.roi_id not in tubes:
            continue
        ratios.append(float(tubes[frame.roi_id].area_ratio))
    return ratios


def _flat_image_uris(directory: Path) -> list[str]:
    """Legacy flat-directory listing (filename-parsing fallback only)."""
    return [_as_file_uri(p) for p in list_image_files(directory)]


def _set_bundle(
    *,
    bundle: VideoBundleMetadata,
    bundle_dir: Path | None,
    input_order: str = "timestamp",
) -> tuple[VideoBundleMetadata, str | None, list[str]]:
    paths = iter_bundle_image_uris(bundle, input_order=input_order)
    return bundle, (str(bundle_dir.resolve()) if bundle_dir else None), paths


def collect_method_inputs(
    video_id: str,
    method: str,
    cfg: AppConfig,
    *,
    allow_missing_inputs: bool = False,
) -> MethodInputs:
    feas = cfg.feasibility
    if feas is None:
        raise ValueError("feasibility config is required")
    ds = feas.datasets
    input_order = cfg.experiment.input_order
    roi_frames = cfg.experiment.roi_frames
    errors: list[str] = []
    warnings: list[str] = []
    paths: list[str] = []
    input_dirs = method_input_dirs(video_id, method, feas)
    num_source_tubes: int | None = None
    single_bbox: list[int] | None = None
    bundle_dir: str | None = None
    bundle: VideoBundleMetadata | None = None

    try:
        video_path = _dataset_video_path(video_id, ds)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        video_path = Path(ds.video_root) / f"{video_id}.mp4"

    if method == "full_2fps":
        directory = Path(ds.fps_2) / video_id
        loaded, used_meta = _load_bundle_or_none(
            directory,
            label="full_2fps",
            errors=errors,
            warnings=warnings,
            roi_frames=roi_frames,
        )
        if used_meta:
            if loaded is not None:
                # Keep global frames only; empty placeholder ROIs are already skipped.
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=build_bundle(global_metadata=loaded, roi_metadata=None),
                    bundle_dir=directory,
                    input_order=input_order,
                )
        else:
            err = _require_images(directory, "full_2fps")
            if err:
                errors.append(err)
            else:
                paths = _flat_image_uris(directory)

    elif method == "full_1fps":
        directory = Path(ds.fps_1) / video_id
        loaded, used_meta = _load_bundle_or_none(
            directory,
            label="full_1fps",
            errors=errors,
            warnings=warnings,
            roi_frames=roi_frames,
        )
        if used_meta:
            if loaded is not None:
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=build_bundle(global_metadata=loaded, roi_metadata=None),
                    bundle_dir=directory,
                    input_order=input_order,
                )
        else:
            err = _require_images(directory, "full_1fps")
            if err:
                errors.append(err)
            else:
                paths = _flat_image_uris(directory)

    elif method == "full_0.25fps":
        directory = Path(ds.fps_0_25) / video_id
        loaded, used_meta = _load_bundle_or_none(
            directory,
            label="full_0.25fps",
            errors=errors,
            warnings=warnings,
            roi_frames=roi_frames,
        )
        if used_meta:
            if loaded is not None:
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=build_bundle(global_metadata=loaded, roi_metadata=None),
                    bundle_dir=directory,
                    input_order=input_order,
                )
        else:
            err = _require_images(directory, "full_0.25fps")
            if err:
                errors.append(err)
            else:
                paths = _flat_image_uris(directory)

    elif method in SINGLE_MOTION_W_FPS_METHODS:
        combo_dir = Path(ds.single_motion_roi_w_fps) / video_id
        roi_dir = Path(ds.single_motion_roi) / video_id
        global_dir = Path(ds.fps_0_25) / video_id
        loaded, used_meta = _load_bundle_or_none(
            combo_dir,
            label="single_motion_ROI_w_fps",
            errors=errors,
            warnings=warnings,
            roi_frames=roi_frames,
        )
        if used_meta:
            if loaded is not None:
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=loaded, bundle_dir=combo_dir, input_order=input_order
                )
        elif has_metadata_json(global_dir) and has_metadata_json(roi_dir):
            g_bundle, g_used = _load_bundle_or_none(
                global_dir,
                label="0.25fps global",
                errors=errors,
                warnings=warnings,
                roi_frames=roi_frames,
            )
            r_bundle, r_used = _load_bundle_or_none(
                roi_dir,
                label="single_motion_ROI",
                errors=errors,
                warnings=warnings,
                roi_frames=roi_frames,
            )
            if g_bundle is not None and r_bundle is not None:
                composed = build_bundle(
                    global_metadata=g_bundle, roi_metadata=r_bundle, video_id=video_id
                )
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=composed, bundle_dir=None, input_order=input_order
                )
                warnings.append(
                    "composed single_motion bundle from "
                    f"{global_dir}/metadata.json + {roi_dir}/metadata.json"
                )
        else:
            err = _require_images(combo_dir, "single_motion_ROI_w_fps")
            if err:
                if allow_missing_inputs:
                    warnings.append(err + " (will be created by --prepare-single-motion)")
                    g_ok = list_image_files(global_dir)
                    r_ok = list_image_files(roi_dir)
                    if g_ok and r_ok:
                        paths = [_as_file_uri(p) for p in g_ok]
                        paths.extend(
                            _as_file_uri(p) for p in r_ok if p.name.startswith("tube")
                        )
                    elif g_ok:
                        paths = [_as_file_uri(p) for p in g_ok]
                    else:
                        errors.append(err)
                else:
                    errors.append(err)
            else:
                paths = _flat_image_uris(combo_dir)
        try:
            plan = plan_single_motion(
                video_id, feas, roi_frames=cfg.experiment.roi_frames
            )
            num_source_tubes = len(plan.source_tube_ids)
            single_bbox = list(plan.bbox_px)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"single-motion plan unavailable: {exc}")

    elif method in SINGLE_MOTION_ROI_ONLY_METHODS:
        directory = Path(ds.single_motion_roi) / video_id
        loaded, used_meta = _load_bundle_or_none(
            directory,
            label="single_motion_ROI",
            errors=errors,
            warnings=warnings,
            roi_frames=roi_frames,
        )
        if used_meta:
            if loaded is not None:
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=loaded, bundle_dir=directory, input_order=input_order
                )
        else:
            err = _require_roi_crops(directory, "single_motion_ROI")
            if err:
                errors.append(err)
            else:
                from .roi_margin import list_roi_crop_files

                paths = [_as_file_uri(p) for p in list_roi_crop_files(directory)]
        try:
            plan = plan_single_motion(
                video_id, feas, roi_frames=cfg.experiment.roi_frames
            )
            num_source_tubes = len(plan.source_tube_ids)
            single_bbox = list(plan.bbox_px)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"single-motion plan unavailable: {exc}")

    elif method in PROPOSED_ROI_ONLY_METHODS:
        directory = Path(ds.proposed_roi) / video_id
        loaded, used_meta = _load_bundle_or_none(
            directory,
            label="proposed_ROI",
            errors=errors,
            warnings=warnings,
            roi_frames=roi_frames,
        )
        if used_meta:
            if loaded is not None:
                # Keep ROI tubes only even if a global section is present.
                rois_only = build_bundle(
                    global_metadata=None, roi_metadata=loaded, video_id=video_id
                )
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=rois_only, bundle_dir=directory, input_order=input_order
                )
        else:
            err = _require_roi_crops(directory, "proposed_ROI")
            if err:
                errors.append(err)
            else:
                from .roi_margin import list_roi_crop_files

                paths = [_as_file_uri(p) for p in list_roi_crop_files(directory)]

    elif method in PROPOSED_W_FPS_METHODS:
        from .motion_adaptive import sampling_mode_for_method, sampling_video_dir

        directory = Path(ds.proposed_roi_w_fps) / video_id
        loaded, used_meta = _load_bundle_or_none(
            directory,
            label="proposed_ROI_w_fps",
            errors=errors,
            warnings=warnings,
            roi_frames=roi_frames,
        )
        if used_meta:
            if loaded is not None:
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=loaded, bundle_dir=directory, input_order=input_order
                )
        else:
            sampling_mode = sampling_mode_for_method(method, cfg.experiment.roi_sampling)
            sampled_dir = (
                sampling_video_dir(cfg.output.dir, sampling_mode, video_id)
                if sampling_mode
                else None
            )
            if sampled_dir is not None and has_metadata_json(sampled_dir):
                sampled, _used = _load_bundle_or_none(
                    sampled_dir,
                    label="sampled ROI",
                    errors=errors,
                    warnings=warnings,
                    roi_frames=roi_frames,
                )
                if sampled is not None:
                    bundle, bundle_dir, paths = _set_bundle(
                        bundle=sampled, bundle_dir=sampled_dir, input_order=input_order
                    )
                    warnings.append(
                        f"proposed_ROI_w_fps missing; using sampled ROI bundle {sampled_dir}"
                    )
            else:
                err = _require_images(directory, "proposed_ROI_w_fps")
                if err:
                    errors.append(err)
                else:
                    paths = _flat_image_uris(directory)

    else:
        errors.append(f"unknown method: {method}")

    if method in FIRST_GLOBAL_METHODS and not errors:
        if bundle is None:
            errors.append(f"{method}: first-global requires a metadata bundle")
        elif not bundle.global_section.frames:
            errors.append(f"{method}: no Global frames to keep")
        else:
            n_before = len(bundle.global_section.frames)
            bundle = keep_first_global_frame(bundle)
            bundle, bundle_dir, paths = _set_bundle(
                bundle=bundle,
                bundle_dir=Path(bundle_dir) if bundle_dir else None,
                input_order=input_order,
            )
            if n_before > 1:
                warnings.append(
                    f"kept first of {n_before} Global 0.25fps frames; ROI tubes unchanged"
                )

    insert_mode = insert_mode_for_method(method, cfg.experiment.global_insert)
    insert_plan = None
    if insert_mode and not errors:
        if bundle is None:
            errors.append(f"{method}: global_insert={insert_mode} requires a metadata bundle")
        else:
            try:
                source_dir = ensure_full_frame_pool(
                    video_id,
                    datasets=ds,
                    source=cfg.experiment.global_insert_source,
                    video_path=video_path,
                    cache_root=Path(cfg.output.dir),
                    jpeg_quality=int(feas.jpeg_quality),
                )
                candidates = load_full_frame_candidates(source_dir, video_id=video_id)
                bundle, insert_plan = apply_global_insert(
                    bundle,
                    mode=insert_mode,
                    candidates=candidates,
                    source_dir=source_dir,
                    dedup_eps=cfg.experiment.global_insert_dedup_eps,
                )
                if method in ROI_START_BOX_METHODS:
                    bundle = draw_roi_start_boxes(
                        bundle, insert_plan, Path(cfg.output.dir) / "global_boxes"
                    )
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=bundle,
                    bundle_dir=Path(bundle_dir) if bundle_dir else None,
                    input_order=input_order,
                )
                src_str = str(source_dir)
                if src_str not in input_dirs:
                    input_dirs.append(src_str)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method}: global_insert={insert_mode} failed: {exc}")

    sampling_mode = sampling_mode_for_method(method, cfg.experiment.roi_sampling)
    sampling_labels: dict[int, str] | None = None
    if sampling_mode and not errors:
        sampled_dir = sampling_video_dir(cfg.output.dir, sampling_mode, video_id)
        if bundle is None:
            errors.append(f"{method}: roi_sampling={sampling_mode} requires a metadata bundle")
        elif not has_metadata_json(sampled_dir):
            errors.append(
                f"{method}: missing sampled ROI bundle {sampled_dir} "
                "(run motion-adaptive prepare / --dry-run first)"
            )
        else:
            try:
                bundle = apply_sampled_rois(bundle, sampled_dir, video_id=video_id)
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=bundle,
                    bundle_dir=sampled_dir,
                    input_order=input_order,
                )
                sampling_labels = load_roi_sampling_labels(sampled_dir)
                sampled_str = str(sampled_dir)
                if sampled_str not in input_dirs:
                    input_dirs.append(sampled_str)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method}: roi_sampling={sampling_mode} failed: {exc}")

    repr_mode = repr_mode_for_method(method, cfg.experiment.roi_repr)
    repr_labels: dict[int, str] | None = None
    repr_extra: list[tuple[str, list[Path]]] | None = None
    if repr_mode and not errors:
        sampled_dir = repr_video_dir(cfg.output.dir, repr_mode, video_id)
        if bundle is None:
            errors.append(f"{method}: roi_repr={repr_mode} requires a metadata bundle")
        elif not has_metadata_json(sampled_dir):
            errors.append(
                f"{method}: missing ROI repr bundle {sampled_dir} "
                "(run roi-union prepare / --dry-run first)"
            )
        else:
            try:
                bundle = apply_roi_repr(bundle, sampled_dir, video_id=video_id)
                bundle, bundle_dir, paths = _set_bundle(
                    bundle=bundle,
                    bundle_dir=sampled_dir,
                    input_order=input_order,
                )
                repr_labels = load_roi_repr_labels(sampled_dir)
                extra = load_bbox_preview_section(sampled_dir)
                repr_extra = extra or None
                sampled_str = str(sampled_dir)
                if sampled_str not in input_dirs:
                    input_dirs.append(sampled_str)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method}: roi_repr={repr_mode} failed: {exc}")

    frames: list[LabeledFrame] = []
    if not errors:
        if bundle is not None:
            frames = _labeled_frames_from_bundle(bundle)
            if not paths:
                paths = [f.path for f in frames]
        elif paths:
            try:
                frames = build_labeled_frames(paths, cfg.experiment)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

    num_global = sum(1 for f in frames if f.kind == "global")
    roi_frames = [f for f in frames if f.kind == "roi"]
    num_roi = len(roi_frames)
    tube_ids = sorted({f.roi_id for f in roi_frames if f.roi_id is not None})
    if bundle is not None:
        rois = active_rois(bundle)
        tube_ids = sorted({roi.roi_id for roi in rois})
        num_global = len(bundle.global_section.frames)
        num_roi = sum(len(roi.frames) for roi in rois)
    ratios: list[float] = []
    frame_width: int | None = None
    frame_height: int | None = None
    roi_bboxes_px: dict[int, tuple[int, int, int, int]] | None = None
    if tube_ids and bundle is None:
        try:
            info, tubes = _roi_meta_for_video(video_id, feas)
            frame_width = int(info.width)
            frame_height = int(info.height)
            if (
                method
                in (
                    "global_0.25fps_single_motion_roi",
                    "global_first_0.25fps_single_motion_roi",
                    "single_motion_roi_original",
                )
                and single_bbox is not None
            ):
                plan = plan_single_motion(
                    video_id, feas, roi_frames=cfg.experiment.roi_frames
                )
                ratios = [float(plan.area_ratio)] * len(roi_frames)
                union = tuple(int(v) for v in single_bbox)
                roi_bboxes_px = {tid: union for tid in tube_ids}
            else:
                ratios = _area_ratios(roi_frames, tubes)
                if method in PROPOSED_MULTI_METHODS:
                    num_source_tubes = len(tube_ids)
                missing = [tid for tid in tube_ids if tid not in tubes]
                if missing:
                    warnings.append(f"missing track bboxes for tube(s) {missing}")
                roi_bboxes_px = {
                    tid: tubes[tid].bbox_px for tid in tube_ids if tid in tubes
                }
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"ROI spatial/area metadata unavailable: {exc}")
    elif bundle is not None and method in PROPOSED_MULTI_METHODS:
        num_source_tubes = len(tube_ids)

    inputs = MethodInputs(
        video_id=video_id,
        method=method,
        frames=frames,
        input_dirs=input_dirs,
        num_global_images=num_global,
        num_roi_images=num_roi,
        num_roi_tubes=len(tube_ids),
        num_source_tubes=num_source_tubes,
        single_motion_bbox=single_bbox,
        frame_width=frame_width,
        frame_height=frame_height,
        roi_bboxes_px=roi_bboxes_px,
        bundle=bundle,
        bundle_dir=bundle_dir,
        roi_area_ratios=ratios,
        errors=errors,
        warnings=warnings,
        global_insert_mode=insert_mode,
        global_insert_plan=insert_plan,
        roi_sampling_mode=sampling_mode,
        roi_sampling_labels=sampling_labels,
        roi_repr_mode=repr_mode,
        roi_repr_labels=repr_labels,
        roi_repr_extra_sections=repr_extra,
        global_scale=float(cfg.experiment.global_scale),
    )
    if not inputs.errors:
        from .global_resolution import apply_global_resolution, format_scale

        inputs = apply_global_resolution(
            inputs,
            scale=float(cfg.experiment.global_scale),
            image_cfg=cfg.image,
            cache_dir=Path(cfg.output.dir)
            / "scaled_globals"
            / format_scale(cfg.experiment.global_scale),
            jpeg_quality=int(feas.jpeg_quality),
        )
    return inputs


def collect_image_stats(inputs: MethodInputs) -> ImageStats:
    sizes: list[dict[str, Any]] = []
    total = 0
    for frame in inputs.frames:
        path = _local_path(frame.path)
        with Image.open(path) as im:
            width, height = im.size
        sizes.append(
            {
                "path": str(path),
                "width": int(width),
                "height": int(height),
                "kind": frame.kind,
                "roi_id": frame.roi_id,
                "time_s": frame.time_s,
            }
        )
        total += int(width) * int(height)
    return ImageStats(image_sizes=sizes, total_input_pixels=total)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = str(path.resolve()) if path.exists() else str(path)
    return _file_fingerprint_cached(resolved, path.name)


@lru_cache(maxsize=8192)
def _file_fingerprint_cached(resolved: str, name: str) -> dict[str, Any]:
    path = Path(resolved)
    if not path.is_file():
        return {"name": name, "size": -1, "sha256": None}
    data = path.read_bytes()
    return {
        "name": name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def cache_key_payload(
    *,
    video_id: str,
    inputs: MethodInputs,
    prompt: str,
    cfg: AppConfig,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for frame in inputs.frames:
        path = _local_path(frame.path)
        fp = _file_fingerprint(path)
        files.append({"name": fp["name"], "size": fp["size"], "sha256": fp["sha256"]})
    payload: dict[str, Any] = {
        "video_id": video_id,
        "input_files": files,
        "prompt": prompt,
        "input_format_version": (
            METADATA_INPUT_FORMAT_VERSION
            if inputs.bundle is not None or inputs.bundle_dir
            else INPUT_FORMAT_VERSION
        ),
        "input_order": cfg.experiment.input_order,
        "roi_frames": cfg.experiment.roi_frames,
        "global_insert": getattr(inputs, "global_insert_mode", None)
        or cfg.experiment.global_insert,
        "global_insert_source": cfg.experiment.global_insert_source,
        "global_insert_dedup_eps": cfg.experiment.global_insert_dedup_eps,
        "roi_sampling": getattr(inputs, "roi_sampling_mode", None)
        or cfg.experiment.roi_sampling,
        "roi_sampling_tau": cfg.experiment.roi_sampling_tau,
        **({
            "roi_sampling_default_fps": cfg.experiment.roi_sampling_default_fps,
            "roi_sampling_intervals": cfg.experiment.roi_sampling_intervals,
        } if inputs.roi_sampling_mode == "piecewise_fps" else {}),
        **({"spatial_occupancy": vars(cfg.experiment.spatial_occupancy)}
           if inputs.roi_sampling_mode == "spatial_occupancy" else {}),
        **({"centroid_motion": vars(cfg.experiment.centroid_motion)}
           if inputs.roi_sampling_mode == "centroid_2fps" else {}),
        "roi_repr": getattr(inputs, "roi_repr_mode", None) or cfg.experiment.roi_repr,
        "bundle_dir": inputs.bundle_dir,
        "bundle_sources": list(inputs.bundle.sources) if inputs.bundle else [],
        "model": {
            "path": cfg.model.path,
            "dtype": cfg.model.dtype,
            "load_in_8bit": cfg.model.load_in_8bit,
        },
        "generation": {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "top_k": cfg.generation.top_k,
            "repetition_penalty": cfg.generation.repetition_penalty,
            "do_sample": cfg.generation.do_sample,
        },
        "image": {
            "min_pixels": cfg.image.min_pixels,
            "max_pixels": cfg.image.max_pixels,
        },
        "mode": cfg.mode,
    }
    scale = float(getattr(inputs, "global_scale", cfg.experiment.global_scale) or 1.0)
    if abs(scale - 1.0) > 1e-12:
        from .global_resolution import global_pixel_budget

        g_min, g_max = global_pixel_budget(cfg.image, scale)
        payload["global_scale"] = scale
        payload["global_image"] = {"min_pixels": g_min, "max_pixels": g_max}
    return payload


def make_cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ratio_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "roi_area_ratio_mean": float(sum(values) / len(values)),
        "roi_area_ratio_min": float(min(values)),
        "roi_area_ratio_max": float(max(values)),
    }


def collect_job_io_metadata(inputs: MethodInputs) -> dict[str, Any]:
    """Per-job IO metadata for ablation result JSON (does not affect inference)."""
    global_timestamps: list[float] = []
    roi_selected_frames: dict[str, int] = {}
    roi_timestamps: dict[str, list[float]] = {}
    assigned_fps: dict[str, float | None] = {}
    roi_groups: dict[str, str | None] = {}

    if inputs.bundle is not None:
        global_timestamps = [
            float(fr.timestamp) for fr in inputs.bundle.global_section.frames
        ]
        for roi in active_rois(inputs.bundle):
            rid = str(int(roi.roi_id))
            roi_selected_frames[rid] = len(roi.frames)
            roi_timestamps[rid] = [float(fr.timestamp) for fr in roi.frames]
    else:
        for frame in inputs.frames:
            if frame.kind == "global":
                global_timestamps.append(float(frame.time_s))
            elif frame.kind == "roi" and frame.roi_id is not None:
                rid = str(int(frame.roi_id))
                roi_selected_frames[rid] = roi_selected_frames.get(rid, 0) + 1
                roi_timestamps.setdefault(rid, []).append(float(frame.time_s))

    if inputs.global_insert_plan is not None:
        plan = inputs.global_insert_plan
        if getattr(plan, "frames", None):
            global_timestamps = [float(fr.timestamp) for fr in plan.frames]

    log_path = Path(inputs.bundle_dir) / "roi_log.json" if inputs.bundle_dir else None
    if log_path is not None and log_path.is_file():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = {}
        for row in log.get("rois") or []:
            if "roi_id" not in row:
                continue
            rid = str(int(row["roi_id"]))
            if row.get("assigned_fps") is not None:
                assigned_fps[rid] = float(row["assigned_fps"])
            elif row.get("assigned_mode") in {"accumulated_motion", "accumulated_motion_cap2fps"}:
                assigned_fps[rid] = None
            if row.get("group") is not None:
                roi_groups[rid] = str(row["group"])
            if row.get("n_selected") is not None and rid not in roi_selected_frames:
                roi_selected_frames[rid] = int(row["n_selected"])

    return {
        "total_images": inputs.num_total_images,
        "roi_selected_frames": roi_selected_frames,
        "roi_timestamps": roi_timestamps,
        "global_timestamps": global_timestamps,
        "assigned_fps": assigned_fps,
        "roi_groups": roi_groups,
        "sampling_mode": inputs.roi_sampling_mode,
        "roi_sampling_tau": None,
    }


def build_result_record(
    *,
    inputs: MethodInputs,
    prompt: str,
    cfg: AppConfig,
    image_stats: ImageStats | None,
    caption: str | None = None,
    error: str | None = None,
    cache_key: str,
    cache_hit: bool = False,
    alias_of: str | None = None,
    copied_from: str | None = None,
    token_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "video_id": inputs.video_id,
        "method": inputs.method,
        "job": f"{inputs.video_id}__{inputs.method}",
        "caption": caption,
        "num_global_images": inputs.num_global_images,
        "num_roi_images": inputs.num_roi_images,
        "num_roi_tubes": inputs.num_roi_tubes,
        "num_total_images": inputs.num_total_images,
        "input_dirs": inputs.input_dirs,
        "total_input_pixels": image_stats.total_input_pixels if image_stats else 0,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "model": cfg.model.path,
        "error": error,
        "global_insert": inputs.global_insert_mode,
        "roi_sampling": inputs.roi_sampling_mode,
        "roi_repr": inputs.roi_repr_mode,
        "success": error is None and bool(caption),
    }
    record.update(collect_job_io_metadata(inputs))
    record["roi_sampling_tau"] = (
        cfg.experiment.roi_sampling_tau if inputs.roi_sampling_mode else None
    )
    if inputs.roi_sampling_mode == "spatial_occupancy":
        record["spatial_occupancy"] = vars(cfg.experiment.spatial_occupancy).copy()
    if inputs.roi_sampling_mode == "centroid_2fps":
        record["centroid_motion"] = vars(cfg.experiment.centroid_motion).copy()
    if inputs.num_source_tubes is not None:
        record["num_source_tubes"] = inputs.num_source_tubes
    if inputs.single_motion_bbox is not None:
        record["single_motion_bbox"] = inputs.single_motion_bbox
    ratios = ratio_summary(inputs.roi_area_ratios)
    if ratios:
        record.update(ratios)
    if alias_of:
        record["alias_of"] = alias_of
    if copied_from:
        record["copied_from"] = copied_from
    if token_stats:
        record.update(token_stats)
    if not inputs.errors and inputs.frames:
        try:
            from .global_resolution import resolution_audit

            audit = resolution_audit(
                inputs, cfg=cfg, extra_io=collect_job_io_metadata(inputs)
            )
            for key in (
                "global_scale",
                "original_global_resolution",
                "resized_global_resolution",
                "vlm_global_resolution",
                "global_visual_tokens",
                "roi_visual_tokens",
                "total_visual_tokens",
                "vlm_input_pixels",
                "roi_bbox",
                "roi_images",
                "global_frames",
                "global_image_opts",
                "roi_image_opts",
                "inference_config",
            ):
                if key in audit:
                    record[key] = audit[key]
            if not record.get("total_input_pixels") and audit.get("total_input_pixels"):
                record["total_input_pixels"] = audit["total_input_pixels"]
            record["num_roi_images"] = audit.get("num_roi_images", record.get("num_roi_images"))
            record["num_global_images"] = audit.get(
                "num_global_images", record.get("num_global_images")
            )
        except Exception as exc:  # noqa: BLE001
            record.setdefault("warnings", [])
            if isinstance(record["warnings"], list):
                record["warnings"].append(f"resolution audit failed: {exc}")
    return record


def executable_methods(feas: FeasibilityConfig) -> list[str]:
    aliases = dict(feas.aliases)
    out: list[str] = []
    seen: set[str] = set()
    for method in feas.methods:
        canonical = aliases.get(method, method)
        if canonical in aliases:
            canonical = aliases[canonical]
        if canonical not in CANONICAL_METHODS:
            raise ValueError(f"unknown feasibility method: {method}")
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def alias_records_for(method: str, feas: FeasibilityConfig) -> list[str]:
    return [alias for alias, target in feas.aliases.items() if target == method]


def dry_run_row(inputs: MethodInputs) -> dict[str, Any]:
    return {
        "video_id": inputs.video_id,
        "method": inputs.method,
        "input_directory": " + ".join(inputs.input_dirs),
        "num_global": inputs.num_global_images,
        "num_roi": inputs.num_roi_images,
        "num_tubes": inputs.num_roi_tubes,
        "num_total": inputs.num_total_images,
        "num_source_tubes": inputs.num_source_tubes,
        "global_insert": inputs.global_insert_mode,
        "roi_sampling": inputs.roi_sampling_mode,
        "roi_repr": inputs.roi_repr_mode,
        "errors": inputs.errors,
        "warnings": inputs.warnings,
    }


def print_dry_run_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "video_id",
        "method",
        "input_directory",
        "num_global",
        "num_roi",
        "num_tubes",
        "num_total",
    )
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    def fmt(row: dict[str, Any]) -> str:
        return "  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)

    print(fmt({h: h for h in headers}))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(fmt(row), flush=True)
        status = []
        if row.get("errors"):
            status.append("ERROR: " + " | ".join(row["errors"]))
        if row.get("warnings"):
            status.append("WARN: " + " | ".join(row["warnings"]))
        if row.get("num_source_tubes") is not None and row["method"] == (
            "global_0.25fps_single_motion_roi"
        ):
            status.append(f"source_tubes={row['num_source_tubes']}")
        for line in status:
            print(f"  {line}", flush=True)
