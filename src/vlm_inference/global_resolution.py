"""Global-frame spatial-resolution ablation for the existing ROI+VLM pipeline.

Does not change ROI generation, clustering, bbox, sampling, Global timestamps,
prompts, or evaluation. The only intended condition difference is the Global
frame's effective VLM resolution / visual-token cost.

Qwen3-VL `fetch_image` (via `process_vision_info`) applies `smart_resize` with
per-image `min_pixels` / `max_pixels` *after* JPEGs are loaded. Config image
`max_pixels` (262144) already caps a 1920×1080 Global to the same visual grid
as 960×540, so shrinking the JPEG alone would not reduce Global tokens at
scale=0.5. This module therefore:

1. Resizes Global source JPEGs by `global_scale` on width and height.
2. Scales Global-only `max_pixels` by `scale**2` (clamped to `min_pixels`)
   so `fetch_image` actually emits fewer Global visual tokens.
3. Leaves ROI crops, ROI `min_pixels`/`max_pixels`, timestamps, and bboxes
   untouched.

Scale 1.0 is a no-op: original Global files and original image opts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from PIL import Image

from .config import AppConfig, ImageConfig
from .experiment import LabeledFrame
from .metadata_format import FrameRef, active_rois

logger = logging.getLogger(__name__)

GLOBAL_SCALES = (1.0, 0.5, 0.25, 0.125)
DEFAULT_METHOD = "global_roi_start_only_roi_accumulated_motion"
IMAGE_PATCH_SIZE = 16  # matches model.py process_vision_info(image_patch_size=16)
SPATIAL_MERGE_SIZE = 2


def format_scale(scale: float) -> str:
    text = f"{float(scale):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def is_identity_scale(scale: float) -> bool:
    return abs(float(scale) - 1.0) <= 1e-12


def is_global_content_text(text: str) -> bool:
    return str(text).lstrip().startswith("[Global]")


def scaled_wh(width: int, height: int, scale: float) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid resolution {(width, height)}")
    if is_identity_scale(scale):
        return int(width), int(height)
    if float(scale) <= 0:
        raise ValueError(f"global_scale must be positive, got {scale}")
    return (
        max(1, int(round(int(width) * float(scale)))),
        max(1, int(round(int(height) * float(scale)))),
    )


def global_pixel_budget(image: ImageConfig, scale: float) -> tuple[int, int]:
    """ROI keeps the original budget; Global max_pixels scales with area."""
    min_pixels = int(image.min_pixels or 4096)
    max_pixels = int(image.max_pixels or 262144)
    if max_pixels < min_pixels:
        raise ValueError("image.max_pixels must be >= image.min_pixels")
    if is_identity_scale(scale):
        return min_pixels, max_pixels
    scaled_max = max(min_pixels, int(round(max_pixels * float(scale) * float(scale))))
    return min_pixels, scaled_max


def global_image_config(image: ImageConfig, scale: float) -> ImageConfig:
    if is_identity_scale(scale):
        return ImageConfig(min_pixels=image.min_pixels, max_pixels=image.max_pixels)
    min_pixels, max_pixels = global_pixel_budget(image, scale)
    return ImageConfig(min_pixels=min_pixels, max_pixels=max_pixels)


def image_opts_for_content_text(
    text: str, image: ImageConfig, scale: float
) -> ImageConfig:
    if is_global_content_text(text):
        return global_image_config(image, scale)
    return image


def vlm_smart_resize(
    width: int,
    height: int,
    *,
    min_pixels: int,
    max_pixels: int,
    patch_size: int = IMAGE_PATCH_SIZE,
) -> tuple[int, int, int]:
    """Return (resized_width, resized_height, visual_tokens) after Qwen fetch_image."""
    from qwen_vl_utils.vision_process import smart_resize

    if width <= 0 or height <= 0:
        return 0, 0, 0
    factor = int(patch_size) * SPATIAL_MERGE_SIZE
    resized_h, resized_w = smart_resize(
        int(height),
        int(width),
        factor=factor,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    tokens = int((int(resized_h) // factor) * (int(resized_w) // factor))
    return int(resized_w), int(resized_h), tokens


def _local_path(path: str | Path) -> Path:
    text = str(path)
    if text.startswith("file://"):
        parsed = urlparse(text)
        return Path(unquote(parsed.path))
    return Path(text)


def _image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(_local_path(path)) as image:
        width, height = image.size
    return int(width), int(height)


def _file_sha256(path: str | Path) -> str | None:
    file_path = _local_path(path)
    if not file_path.is_file():
        return None
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def resize_global_jpeg(
    src: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    quality: int,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        out = rgb.resize((int(width), int(height)), Image.Resampling.BICUBIC)
        out.save(
            dest,
            format="JPEG",
            quality=int(quality),
            subsampling=0,
            optimize=False,
        )


def _scale_global_file(
    src: Path,
    *,
    scale: float,
    cache_dir: Path,
    video_id: str,
    quality: int,
) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    src = src.resolve()
    original = _image_size(src)
    resized = scaled_wh(original[0], original[1], scale)
    if is_identity_scale(scale):
        return src, original, original
    dest = Path(cache_dir) / video_id / src.name
    if (
        dest.is_file()
        and _image_size(dest) == resized
    ):
        return dest.resolve(), original, resized
    resize_global_jpeg(
        src, dest, width=resized[0], height=resized[1], quality=quality
    )
    return dest.resolve(), original, resized


def apply_global_resolution(
    inputs: Any,
    *,
    scale: float,
    image_cfg: ImageConfig,
    cache_dir: Path,
    jpeg_quality: int,
) -> Any:
    """Rewrite Global frame files only. ROI sections/frames are reused as-is."""
    from .feasibility import MethodInputs, _labeled_frames_from_bundle

    if not isinstance(inputs, MethodInputs):
        raise TypeError("apply_global_resolution expects MethodInputs")
    scale = float(scale)
    out = replace(
        inputs,
        frames=list(inputs.frames),
        input_dirs=list(inputs.input_dirs),
        errors=list(inputs.errors),
        warnings=list(inputs.warnings),
        roi_area_ratios=list(inputs.roi_area_ratios),
        global_scale=scale,
    )
    if out.errors:
        return out

    original_res = out.original_global_resolution
    if original_res is None:
        for frame in out.frames:
            if frame.kind == "global":
                original_res = _image_size(frame.path)
                break
        if original_res is None and out.bundle is not None:
            for frame in out.bundle.global_section.frames:
                original_res = _image_size(frame.path)
                break
    out.original_global_resolution = original_res
    if is_identity_scale(scale):
        return out

    cache_dir = Path(cache_dir)
    if out.bundle is not None:
        scaled_refs: list[FrameRef] = []
        for frame in out.bundle.global_section.frames:
            dest, _orig, _resized = _scale_global_file(
                frame.path,
                scale=scale,
                cache_dir=cache_dir,
                video_id=out.video_id,
                quality=jpeg_quality,
            )
            scaled_refs.append(replace(frame, path=dest, file=dest.name))
        out.bundle = replace(
            out.bundle,
            global_section=replace(
                out.bundle.global_section,
                frames=tuple(scaled_refs),
                num_frames=len(scaled_refs),
            ),
        )
        out.frames = _labeled_frames_from_bundle(out.bundle)
    else:
        new_frames: list[LabeledFrame] = []
        for frame in out.frames:
            if frame.kind != "global":
                new_frames.append(frame)
                continue
            dest, _orig, _resized = _scale_global_file(
                _local_path(frame.path),
                scale=scale,
                cache_dir=cache_dir,
                video_id=out.video_id,
                quality=jpeg_quality,
            )
            new_frames.append(replace(frame, path=dest.as_uri()))
        out.frames = new_frames

    extra = str(cache_dir)
    if extra not in out.input_dirs:
        out.input_dirs.append(extra)
    return out


def roi_bbox_payload(inputs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if getattr(inputs, "bundle", None) is not None:
        for roi in active_rois(inputs.bundle):
            payload[str(int(roi.roi_id))] = {
                "normalized_bbox": [float(v) for v in roi.normalized_bbox],
                "time_range": [float(v) for v in roi.time_range],
                "source_region": str(roi.source_region),
            }
    bboxes = getattr(inputs, "roi_bboxes_px", None) or {}
    for roi_id, bbox in bboxes.items():
        key = str(int(roi_id))
        payload.setdefault(key, {})["bbox_px"] = [int(v) for v in bbox]
    return payload


def roi_identity_payload(inputs: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for frame in getattr(inputs, "frames", []) or []:
        if getattr(frame, "kind", None) != "roi":
            continue
        path = _local_path(frame.path)
        width, height = _image_size(path) if path.is_file() else (0, 0)
        rows.append(
            {
                "roi_id": frame.roi_id,
                "time_s": float(frame.time_s),
                "path": str(path.resolve()) if path.exists() else str(path),
                "sha256": _file_sha256(path),
                "width": width,
                "height": height,
            }
        )
    return {
        "roi_bboxes": roi_bbox_payload(inputs),
        "roi_images": rows,
    }


@dataclass(frozen=True)
class FrameVisionCost:
    kind: str
    path: str
    source_width: int
    source_height: int
    vlm_width: int
    vlm_height: int
    visual_tokens: int
    min_pixels: int
    max_pixels: int
    time_s: float
    roi_id: int | None


def frame_vision_cost(
    frame: LabeledFrame,
    *,
    image: ImageConfig,
    scale: float,
) -> FrameVisionCost:
    path = _local_path(frame.path)
    width, height = _image_size(path)
    opts = image_opts_for_content_text(frame.label, image, scale)
    min_pixels = int(opts.min_pixels or 4096)
    max_pixels = int(opts.max_pixels or 262144)
    vlm_w, vlm_h, tokens = vlm_smart_resize(
        width, height, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return FrameVisionCost(
        kind=str(frame.kind),
        path=str(path.resolve()) if path.exists() else str(path),
        source_width=width,
        source_height=height,
        vlm_width=vlm_w,
        vlm_height=vlm_h,
        visual_tokens=tokens,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        time_s=float(frame.time_s),
        roi_id=frame.roi_id,
    )


def resolution_audit(
    inputs: Any,
    *,
    cfg: AppConfig,
    extra_io: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scale = float(getattr(inputs, "global_scale", cfg.experiment.global_scale) or 1.0)
    global_costs: list[FrameVisionCost] = []
    roi_costs: list[FrameVisionCost] = []
    for frame in getattr(inputs, "frames", []) or []:
        cost = frame_vision_cost(frame, image=cfg.image, scale=scale)
        if frame.kind == "global":
            global_costs.append(cost)
        elif frame.kind == "roi":
            roi_costs.append(cost)

    original_global = None
    resized_global = None
    vlm_global = None
    stored = getattr(inputs, "original_global_resolution", None)
    if stored is not None:
        original_global = [int(stored[0]), int(stored[1])]
    if global_costs:
        resized_global = [global_costs[0].source_width, global_costs[0].source_height]
        vlm_global = [global_costs[0].vlm_width, global_costs[0].vlm_height]
        if original_global is None:
            original_global = list(resized_global) if is_identity_scale(scale) else None

    global_tokens = int(sum(item.visual_tokens for item in global_costs))
    roi_tokens = int(sum(item.visual_tokens for item in roi_costs))
    file_pixels = int(
        sum(item.source_width * item.source_height for item in (*global_costs, *roi_costs))
    )
    vlm_pixels = int(
        sum(item.vlm_width * item.vlm_height for item in (*global_costs, *roi_costs))
    )
    io_meta = extra_io or {}
    g_min, g_max = global_pixel_budget(cfg.image, scale)
    identity = roi_identity_payload(inputs)
    return {
        "video_id": inputs.video_id,
        "method": inputs.method,
        "global_scale": scale,
        "original_global_resolution": original_global,
        "resized_global_resolution": resized_global,
        "vlm_global_resolution": vlm_global,
        "num_global_images": int(inputs.num_global_images),
        "num_roi_images": int(inputs.num_roi_images),
        "total_input_images": int(inputs.num_total_images),
        "global_visual_tokens": global_tokens,
        "roi_visual_tokens": roi_tokens,
        "total_visual_tokens": global_tokens + roi_tokens,
        "total_input_pixels": file_pixels,
        "vlm_input_pixels": vlm_pixels,
        "global_timestamps": list(io_meta.get("global_timestamps") or []),
        "roi_timestamps": io_meta.get("roi_timestamps") or {},
        "roi_bbox": identity["roi_bboxes"],
        "roi_images": identity["roi_images"],
        "global_image_opts": {"min_pixels": g_min, "max_pixels": g_max},
        "roi_image_opts": {
            "min_pixels": cfg.image.min_pixels,
            "max_pixels": cfg.image.max_pixels,
        },
        "global_frames": [asdict(item) for item in global_costs],
        "roi_frame_costs": [asdict(item) for item in roi_costs],
        "inference_config": {
            "model": cfg.model.path,
            "model_name": cfg.model.name,
            "dtype": cfg.model.dtype,
            "load_in_8bit": cfg.model.load_in_8bit,
            "generation": asdict(cfg.generation),
            "image": asdict(cfg.image),
            "video": asdict(cfg.video),
            "mode": cfg.mode,
            "input_order": cfg.experiment.input_order,
            "global_insert": cfg.experiment.global_insert,
            "global_insert_source": cfg.experiment.global_insert_source,
            "global_insert_dedup_eps": cfg.experiment.global_insert_dedup_eps,
            "roi_sampling": getattr(inputs, "roi_sampling_mode", None)
            or cfg.experiment.roi_sampling,
            "roi_sampling_tau": cfg.experiment.roi_sampling_tau,
            "prompt_file": cfg.prompt_file,
            "jpeg_quality": None
            if cfg.feasibility is None
            else cfg.feasibility.jpeg_quality,
        },
    }


class InvariantError(AssertionError):
    """Raised when a global-resolution ablation invariant fails."""


def _sorted_roi_images(row: dict[str, Any]) -> list[tuple[Any, ...]]:
    images = row.get("roi_images") or []
    return sorted(
        (
            item.get("roi_id"),
            round(float(item.get("time_s") or 0.0), 6),
            item.get("sha256"),
            int(item.get("width") or 0),
            int(item.get("height") or 0),
            item.get("path"),
        )
        for item in images
    )


def _timestamps(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if isinstance(value, dict):
        return {
            str(roi_id): [round(float(t), 6) for t in times]
            for roi_id, times in sorted(value.items())
        }
    return [round(float(t), 6) for t in (value or [])]


def verify_invariants(
    rows_by_scale: dict[float, Sequence[dict[str, Any]]],
    *,
    expected_videos: Sequence[str] | None = None,
    baseline_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check A–G. Returns a JSON-serializable report; raises on failure."""
    scales = tuple(sorted((float(s) for s in rows_by_scale), reverse=True))
    expected = tuple(float(s) for s in GLOBAL_SCALES)
    if set(scales) != set(expected) or len(scales) != len(expected):
        raise InvariantError(f"expected scales {GLOBAL_SCALES}, got {tuple(rows_by_scale)}")

    by_video: dict[str, dict[float, dict[str, Any]]] = {}
    for scale, rows in rows_by_scale.items():
        for row in rows:
            video_id = str(row["video_id"])
            by_video.setdefault(video_id, {})[float(scale)] = row

    videos = sorted(by_video)
    if expected_videos is not None:
        expected = list(expected_videos)
        if videos != sorted(expected):
            raise InvariantError(
                f"video set mismatch: got {videos}, expected {sorted(expected)}"
            )
        if len(videos) != len(expected):
            raise InvariantError("unexpected video count")

    checks = {
        "A_global_timestamps": True,
        "B_roi_timestamps": True,
        "C_roi_bbox": True,
        "D_roi_images": True,
        "E_only_global_resolution_changes": True,
        "F_global_tokens_decrease": True,
        "G_scale_1_matches_baseline": True,
    }
    details: list[str] = []

    for video_id, scaled in by_video.items():
        missing = [s for s in GLOBAL_SCALES if s not in scaled]
        if missing:
            raise InvariantError(f"{video_id}: missing scales {missing}")
        baseline = scaled[1.0]
        for scale in GLOBAL_SCALES:
            row = scaled[scale]
            if _timestamps(row, "global_timestamps") != _timestamps(
                baseline, "global_timestamps"
            ):
                checks["A_global_timestamps"] = False
                details.append(f"{video_id} scale={scale}: Global timestamps differ")
            if _timestamps(row, "roi_timestamps") != _timestamps(
                baseline, "roi_timestamps"
            ):
                checks["B_roi_timestamps"] = False
                details.append(f"{video_id} scale={scale}: ROI timestamps differ")
            if (row.get("roi_bbox") or {}) != (baseline.get("roi_bbox") or {}):
                checks["C_roi_bbox"] = False
                details.append(f"{video_id} scale={scale}: ROI bbox differs")
            base_roi = _sorted_roi_images(baseline)
            row_roi = _sorted_roi_images(row)
            row_roi_no_path = [item[:-1] for item in row_roi]
            base_roi_no_path = [item[:-1] for item in base_roi]
            if row_roi_no_path != base_roi_no_path or [
                item[-1] for item in row_roi
            ] != [item[-1] for item in base_roi]:
                checks["D_roi_images"] = False
                details.append(
                    f"{video_id} scale={scale}: ROI image identity/resolution differs"
                )
            if int(row.get("num_roi_images") or 0) != int(
                baseline.get("num_roi_images") or 0
            ):
                checks["E_only_global_resolution_changes"] = False
                details.append(f"{video_id} scale={scale}: ROI image count changed")
            if int(row.get("num_global_images") or 0) != int(
                baseline.get("num_global_images") or 0
            ):
                checks["E_only_global_resolution_changes"] = False
                details.append(f"{video_id} scale={scale}: Global image count changed")
            if int(row.get("roi_visual_tokens") or 0) != int(
                baseline.get("roi_visual_tokens") or 0
            ):
                checks["E_only_global_resolution_changes"] = False
                details.append(f"{video_id} scale={scale}: ROI visual tokens changed")

        token_series = [int(scaled[s]["global_visual_tokens"]) for s in GLOBAL_SCALES]
        if any(
            token_series[i] <= token_series[i + 1]
            for i in range(len(token_series) - 1)
        ):
            checks["F_global_tokens_decrease"] = False
            details.append(
                f"{video_id}: Global visual tokens did not strictly decrease: {token_series}"
            )

        if is_identity_scale(1.0):
            resized = baseline.get("resized_global_resolution")
            original = baseline.get("original_global_resolution")
            if resized != original:
                checks["G_scale_1_matches_baseline"] = False
                details.append(
                    f"{video_id}: scale=1.0 resized Global resolution {resized} "
                    f"!= original {original}"
                )
            g_opts = baseline.get("global_image_opts") or {}
            r_opts = baseline.get("roi_image_opts") or {}
            if g_opts.get("max_pixels") != r_opts.get("max_pixels") or g_opts.get(
                "min_pixels"
            ) != r_opts.get("min_pixels"):
                checks["G_scale_1_matches_baseline"] = False
                details.append(
                    f"{video_id}: scale=1.0 Global image opts differ from pipeline image opts"
                )

    if baseline_rows is not None:
        baseline_by_video = {str(row["video_id"]): row for row in baseline_rows}
        for video_id, scaled in by_video.items():
            row = scaled[1.0]
            ref = baseline_by_video.get(video_id)
            if ref is None:
                checks["G_scale_1_matches_baseline"] = False
                details.append(f"{video_id}: missing unscaled baseline row")
                continue
            for key in (
                "global_timestamps",
                "roi_timestamps",
                "roi_bbox",
                "num_global_images",
                "num_roi_images",
                "global_visual_tokens",
                "roi_visual_tokens",
            ):
                left = row.get(key)
                right = ref.get(key)
                if key in {"global_timestamps", "roi_timestamps"}:
                    left = _timestamps(row, key)
                    right = _timestamps(ref, key)
                if left != right:
                    checks["G_scale_1_matches_baseline"] = False
                    details.append(
                        f"{video_id}: scale=1.0 {key} differs from unscaled pipeline"
                    )
            if _sorted_roi_images(row) != _sorted_roi_images(ref):
                checks["G_scale_1_matches_baseline"] = False
                details.append(
                    f"{video_id}: scale=1.0 ROI files differ from unscaled pipeline"
                )
            scale_paths = [
                item.get("path")
                for item in (row.get("global_frames") or [])
            ]
            ref_paths = [
                item.get("path")
                for item in (ref.get("global_frames") or [])
            ]
            if scale_paths != ref_paths:
                checks["G_scale_1_matches_baseline"] = False
                details.append(
                    f"{video_id}: scale=1.0 Global paths differ from unscaled pipeline"
                )

    failed = [name for name, ok in checks.items() if not ok]
    report = {
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "details": details,
        "n_videos": len(videos),
        "n_conditions": len(GLOBAL_SCALES),
        "n_jobs": len(videos) * len(GLOBAL_SCALES),
        "videos": videos,
        "scales": [float(s) for s in GLOBAL_SCALES],
    }
    if failed:
        raise InvariantError(
            "global-resolution invariants failed: "
            + "; ".join(failed)
            + ((" | " + " ; ".join(details[:12])) if details else "")
        )
    return report


def summarize_token_stats(
    rows_by_scale: dict[float, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    per_scale: dict[str, Any] = {}
    for scale in GLOBAL_SCALES:
        rows = list(rows_by_scale[scale])
        def _sum(key: str) -> int:
            return int(sum(int(row.get(key) or 0) for row in rows))

        per_scale[format_scale(scale)] = {
            "global_scale": float(scale),
            "n_videos": len(rows),
            "global_visual_tokens_sum": _sum("global_visual_tokens"),
            "roi_visual_tokens_sum": _sum("roi_visual_tokens"),
            "total_visual_tokens_sum": _sum("total_visual_tokens"),
            "total_input_pixels_sum": _sum("total_input_pixels"),
            "vlm_input_pixels_sum": _sum("vlm_input_pixels"),
            "num_global_images_sum": _sum("num_global_images"),
            "num_roi_images_sum": _sum("num_roi_images"),
            "per_video": [
                {
                    "video_id": row["video_id"],
                    "original_global_resolution": row.get("original_global_resolution"),
                    "resized_global_resolution": row.get("resized_global_resolution"),
                    "vlm_global_resolution": row.get("vlm_global_resolution"),
                    "num_global_images": row.get("num_global_images"),
                    "num_roi_images": row.get("num_roi_images"),
                    "global_visual_tokens": row.get("global_visual_tokens"),
                    "roi_visual_tokens": row.get("roi_visual_tokens"),
                    "total_visual_tokens": row.get("total_visual_tokens"),
                    "total_input_pixels": row.get("total_input_pixels"),
                    "vlm_input_pixels": row.get("vlm_input_pixels"),
                }
                for row in rows
            ],
        }
    return per_scale


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
