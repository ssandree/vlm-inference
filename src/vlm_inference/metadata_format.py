"""Parse motion-detector `{VIDEO_ID}/metadata.json` into VLM multimodal context.

This module is intentionally independent of any specific VLM processor. It
produces an ordered intermediate representation of (text, optional image path)
items; callers convert that IR into model-specific message content.

``metadata.json`` is the source of truth whenever present. Image filenames are
never parsed here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

# Bump when the formatted text structure changes (feasibility cache invalidation).
METADATA_INPUT_FORMAT_VERSION = "metadata_dataset_global_anchor_group_roi_v1"

BBOX_DEFINITION = (
    "ROI spatial metadata:\n"
    "normalized_bbox = [x_min, y_min, x_max, y_max] in the original full-frame "
    "coordinate system, normalized to [0,1]."
)


class MetadataFormatError(ValueError):
    """Raised when metadata.json is missing required fields or is malformed."""


@dataclass(frozen=True)
class FrameRef:
    frame_index: int
    timestamp: float
    file: str  # relative path as recorded in metadata.json
    path: Path  # resolved absolute path under the video bundle root


@dataclass(frozen=True)
class GlobalSection:
    coverage: str
    num_frames: int
    frames: tuple[FrameRef, ...]


@dataclass(frozen=True)
class RoiSection:
    roi_id: int
    time_range: tuple[float, float]
    source_region: str
    normalized_bbox: tuple[float, float, float, float]
    num_frames: int
    frames: tuple[FrameRef, ...]


@dataclass(frozen=True)
class VideoBundleMetadata:
    """Parsed / composed video bundle plus resolved frame paths."""

    video_id: str
    root: Path
    global_section: GlobalSection
    rois: tuple[RoiSection, ...]
    warnings: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class FormattedItem:
    """Intermediate representation: text block, optionally followed by an image."""

    text: str
    image_path: Path | None = None


@dataclass
class FormatResult:
    items: list[FormattedItem]
    metadata: VideoBundleMetadata
    warnings: list[str] = field(default_factory=list)

    @property
    def image_paths(self) -> list[Path]:
        return [item.image_path for item in self.items if item.image_path is not None]


def metadata_json_path(bundle_dir: Path | str) -> Path:
    return Path(bundle_dir).expanduser().resolve() / "metadata.json"


def has_metadata_json(bundle_dir: Path | str) -> bool:
    return metadata_json_path(bundle_dir).is_file()


def _require_mapping(value: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetadataFormatError(f"{ctx}: expected object, got {type(value).__name__}")
    return value


def _require_list(value: Any, ctx: str) -> list[Any]:
    if not isinstance(value, list):
        raise MetadataFormatError(f"{ctx}: expected list, got {type(value).__name__}")
    return value


def _require_key(data: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise MetadataFormatError(f"{ctx}: missing required field '{key}'")
    return data[key]


def _as_float(value: Any, ctx: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MetadataFormatError(f"{ctx}: expected number, got {value!r}") from exc


def _as_int(value: Any, ctx: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MetadataFormatError(f"{ctx}: expected int, got {value!r}") from exc


def _format_num(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    if "." not in text:
        text = f"{text}.0"
    return text


def _format_bbox(bbox: Sequence[float]) -> str:
    return "[" + ", ".join(_format_num(v) for v in bbox) + "]"


def _resolve_frame(
    *,
    root: Path,
    raw: Any,
    ctx: str,
    require_files: bool,
    default_frame_index: int,
) -> FrameRef:
    frame = _require_mapping(raw, ctx)
    if "frame_index" in frame and frame["frame_index"] is not None:
        frame_index = _as_int(frame["frame_index"], f"{ctx}.frame_index")
    else:
        frame_index = int(default_frame_index)
    timestamp = _as_float(_require_key(frame, "timestamp", ctx), f"{ctx}.timestamp")
    rel = _require_key(frame, "file", ctx)
    if not isinstance(rel, str) or not rel.strip():
        raise MetadataFormatError(f"{ctx}.file: expected non-empty string")
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise MetadataFormatError(
            f"{ctx}.file: expected path relative to bundle root, got absolute {rel!r}"
        )
    resolved = (root / rel_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MetadataFormatError(
            f"{ctx}.file: escapes bundle root ({rel!r} -> {resolved})"
        ) from exc
    if require_files and not resolved.is_file():
        raise MetadataFormatError(f"{ctx}: image file not found: {resolved}")
    return FrameRef(
        frame_index=frame_index,
        timestamp=timestamp,
        file=rel.replace("\\", "/"),
        path=resolved,
    )


def _is_empty_roi(
    *,
    frames: Sequence[FrameRef],
    declared_num_frames: int | None,
) -> bool:
    if not frames or len(frames) == 0:
        return True
    if declared_num_frames is not None and int(declared_num_frames) == 0:
        return True
    return False


def _parse_global_section(
    data: dict[str, Any],
    *,
    root: Path,
    ctx: str,
    require_files: bool,
    warnings: list[str],
) -> GlobalSection:
    if "global" not in data or data["global"] is None:
        return GlobalSection(coverage="full scene", num_frames=0, frames=())

    global_raw = _require_mapping(data["global"], f"{ctx}:global")
    g_frames_raw = _require_list(
        global_raw.get("frames", []), f"{ctx}:global.frames"
    )
    g_frames = tuple(
        _resolve_frame(
            root=root,
            raw=item,
            ctx=f"{ctx}:global.frames[{i}]",
            require_files=require_files,
            default_frame_index=i + 1,
        )
        for i, item in enumerate(g_frames_raw)
    )
    actual = len(g_frames)
    if "num_frames" in global_raw and global_raw["num_frames"] is not None:
        declared = _as_int(global_raw["num_frames"], f"{ctx}:global.num_frames")
        if declared != actual:
            warnings.append(
                f"global.num_frames={declared} but len(frames)={actual}; "
                "using len(frames)"
            )
    coverage = str(global_raw.get("coverage", "full scene"))
    return GlobalSection(coverage=coverage, num_frames=actual, frames=g_frames)


def _parse_roi_sections(
    data: dict[str, Any],
    *,
    root: Path,
    ctx: str,
    require_files: bool,
    warnings: list[str],
    roi_frames: str = "default",
) -> tuple[RoiSection, ...]:
    from .config import VALID_ROI_FRAME_SETS

    frame_set = str(roi_frames).strip().lower()
    if frame_set not in VALID_ROI_FRAME_SETS:
        raise MetadataFormatError(
            f"roi_frames must be one of {sorted(VALID_ROI_FRAME_SETS)}, "
            f"got '{roi_frames}'"
        )
    frames_key = "frames_fps_1" if frame_set == "fps_1" else "frames"
    num_key = "num_frames_fps_1" if frame_set == "fps_1" else "num_frames"

    rois_raw = _require_list(data.get("rois", []), f"{ctx}:rois")
    rois: list[RoiSection] = []
    for i, roi_item in enumerate(rois_raw):
        roi_ctx = f"{ctx}:rois[{i}]"
        roi = _require_mapping(roi_item, roi_ctx)
        roi_id = _as_int(_require_key(roi, "roi_id", roi_ctx), f"{roi_ctx}.roi_id")
        tr = _require_list(_require_key(roi, "time_range", roi_ctx), f"{roi_ctx}.time_range")
        if len(tr) != 2:
            raise MetadataFormatError(f"{roi_ctx}.time_range: expected [start, end]")
        time_range = (
            _as_float(tr[0], f"{roi_ctx}.time_range[0]"),
            _as_float(tr[1], f"{roi_ctx}.time_range[1]"),
        )
        source_region = str(_require_key(roi, "source_region", roi_ctx))
        bbox_raw = _require_list(
            _require_key(roi, "normalized_bbox", roi_ctx), f"{roi_ctx}.normalized_bbox"
        )
        if len(bbox_raw) != 4:
            raise MetadataFormatError(
                f"{roi_ctx}.normalized_bbox: expected [x_min, y_min, x_max, y_max]"
            )
        bbox = tuple(
            _as_float(v, f"{roi_ctx}.normalized_bbox[{j}]")
            for j, v in enumerate(bbox_raw)
        )
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise MetadataFormatError(
                f"{roi_ctx}.normalized_bbox: max < min in {list(bbox)}"
            )

        if frame_set == "fps_1" and frames_key not in roi:
            raise MetadataFormatError(
                f"{roi_ctx}: missing '{frames_key}' (roi_frames=fps_1)"
            )

        frames_raw = roi.get(frames_key, [])
        if frames_raw is None:
            frames_raw = []
        frames_raw = _require_list(frames_raw, f"{roi_ctx}.{frames_key}")
        frames = tuple(
            _resolve_frame(
                root=root,
                raw=item,
                ctx=f"{roi_ctx}.{frames_key}[{j}]",
                require_files=require_files,
                default_frame_index=j + 1,
            )
            for j, item in enumerate(frames_raw)
        )
        # Keep temporal order for sequence headers / dumps.
        frames = tuple(
            sorted(frames, key=lambda f: (float(f.timestamp), int(f.frame_index)))
        )
        declared: int | None = None
        if num_key in roi and roi[num_key] is not None:
            declared = _as_int(roi[num_key], f"{roi_ctx}.{num_key}")

        if _is_empty_roi(frames=frames, declared_num_frames=declared):
            warnings.append(
                f"skipping empty ROI roi_id={roi_id} frame_set={frame_set} "
                f"(declared_num_frames={declared}, len(frames)={len(frames)})"
            )
            continue

        actual = len(frames)
        if declared is not None and declared != actual:
            warnings.append(
                f"rois[{i}] (roi_id={roi_id}): {num_key}={declared} but "
                f"len({frames_key})={actual}; using len({frames_key})"
            )

        if frames:
            time_range = (float(frames[0].timestamp), float(frames[-1].timestamp))

        rois.append(
            RoiSection(
                roi_id=roi_id,
                time_range=time_range,
                source_region=source_region,
                normalized_bbox=bbox,  # type: ignore[arg-type]
                num_frames=actual,
                frames=frames,
            )
        )
    return tuple(rois)


def load_video_bundle_metadata(
    bundle_dir: Path | str,
    *,
    require_files: bool = True,
    roi_frames: str = "default",
) -> VideoBundleMetadata:
    """Parse and validate `{bundle_dir}/metadata.json`.

    Frame timestamps / ROI ids / spatial fields come only from metadata — never
    from image filenames. Empty ROIs are dropped. ``num_frames`` is validated
    against ``len(frames)`` and the actual count is kept.

    ``roi_frames``:
      - ``default``: use each ROI's ``frames``
      - ``fps_1``: use each ROI's ``frames_fps_1`` (*_fps_1.jpg @ 1 fps)
    """
    root = Path(bundle_dir).expanduser().resolve()
    meta_path = root / "metadata.json"
    if not meta_path.is_file():
        raise MetadataFormatError(f"metadata.json not found under {root}")

    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataFormatError(
            f"malformed JSON in {meta_path}: {exc.msg} (line {exc.lineno})"
        ) from exc

    data = _require_mapping(raw, str(meta_path))
    video_id = str(_require_key(data, "video_id", str(meta_path)))
    warnings: list[str] = []
    ctx = str(meta_path)

    global_section = _parse_global_section(
        data,
        root=root,
        ctx=ctx,
        require_files=require_files,
        warnings=warnings,
    )
    rois = _parse_roi_sections(
        data,
        root=root,
        ctx=ctx,
        require_files=require_files,
        warnings=warnings,
        roi_frames=roi_frames,
    )

    for warning in warnings:
        logger.warning("%s: %s", meta_path, warning)

    return VideoBundleMetadata(
        video_id=video_id,
        root=root,
        global_section=global_section,
        rois=rois,
        warnings=tuple(warnings),
        sources=(str(meta_path),),
    )


def build_bundle(
    *,
    global_metadata: VideoBundleMetadata | None = None,
    roi_metadata: VideoBundleMetadata | None = None,
    video_id: str | None = None,
) -> VideoBundleMetadata:
    """Compose a logical inference bundle from separate global / ROI sources.

    Does not invent a global section from ROI-only metadata. Callers must pass
    an explicit ``global_metadata`` when global frames are required.
    """
    if global_metadata is None and roi_metadata is None:
        raise MetadataFormatError(
            "build_bundle requires global_metadata and/or roi_metadata"
        )

    warnings: list[str] = []
    sources: list[str] = []

    if global_metadata is not None:
        warnings.extend(global_metadata.warnings)
        sources.extend(global_metadata.sources)
        global_section = global_metadata.global_section
        root = global_metadata.root
        vid = global_metadata.video_id
    else:
        global_section = GlobalSection(coverage="full scene", num_frames=0, frames=())
        assert roi_metadata is not None
        root = roi_metadata.root
        vid = roi_metadata.video_id

    if roi_metadata is not None:
        warnings.extend(roi_metadata.warnings)
        sources.extend(roi_metadata.sources)
        rois = tuple(
            roi
            for roi in roi_metadata.rois
            if not _is_empty_roi(frames=roi.frames, declared_num_frames=roi.num_frames)
        )
        skipped = len(roi_metadata.rois) - len(rois)
        if skipped:
            warnings.append(f"build_bundle: skipped {skipped} empty ROI section(s)")
        if video_id is None and global_metadata is not None:
            if roi_metadata.video_id != global_metadata.video_id:
                warnings.append(
                    "build_bundle: video_id mismatch "
                    f"global={global_metadata.video_id!r} roi={roi_metadata.video_id!r}"
                )
    else:
        rois = ()

    return VideoBundleMetadata(
        video_id=video_id or vid,
        root=root,
        global_section=global_section,
        rois=rois,
        warnings=tuple(warnings),
        sources=tuple(sources),
    )


def keep_first_global_frame(bundle: VideoBundleMetadata) -> VideoBundleMetadata:
    """Keep only the earliest Global frame; ROI tubes are unchanged."""
    frames = tuple(_sorted_global_frames(bundle))
    if len(frames) <= 1:
        return bundle
    first = frames[0]
    return replace(
        bundle,
        global_section=GlobalSection(
            coverage=bundle.global_section.coverage,
            num_frames=1,
            frames=(first,),
        ),
        warnings=bundle.warnings
        + (
            "keep_first_global_frame: kept t="
            f"{first.timestamp}s, dropped {len(frames) - 1} later Global frame(s)",
        ),
    )


def active_rois(metadata: VideoBundleMetadata) -> tuple[RoiSection, ...]:
    return tuple(
        roi
        for roi in sorted(metadata.rois, key=lambda r: int(r.roi_id))
        if not _is_empty_roi(frames=roi.frames, declared_num_frames=roi.num_frames)
    )


def _sorted_global_frames(metadata: VideoBundleMetadata):
    return sorted(
        metadata.global_section.frames,
        key=lambda frame: (float(frame.timestamp), int(frame.frame_index)),
    )


def _sorted_roi_frames(roi):
    return sorted(
        roi.frames,
        key=lambda frame: (float(frame.timestamp), int(frame.frame_index)),
    )


def _interleaved_frame_items(
    metadata: VideoBundleMetadata,
) -> list[FormattedItem]:
    """Global/ROI frame tokens sorted by timestamp (Global before ROI on ties)."""
    tagged: list[tuple[float, int, int, int, FormattedItem]] = []

    for frame in metadata.global_section.frames:
        tagged.append(
            (
                float(frame.timestamp),
                0,  # global before roi
                0,
                int(frame.frame_index),
                FormattedItem(
                    text=f"[Global] t={_format_num(frame.timestamp)}s",
                    image_path=frame.path,
                ),
            )
        )

    for roi in active_rois(metadata):
        for frame in roi.frames:
            tagged.append(
                (
                    float(frame.timestamp),
                    1,
                    int(roi.roi_id),
                    int(frame.frame_index),
                    FormattedItem(
                        text=f"[ROI{roi.roi_id}] t={_format_num(frame.timestamp)}s",
                        image_path=frame.path,
                    ),
                )
            )

    tagged.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [row[4] for row in tagged]


def _grouped_frame_items(
    metadata: VideoBundleMetadata,
) -> list[FormattedItem]:
    """Global sequence first, then each ROI tube sequence (roi_id asc)."""
    items: list[FormattedItem] = []
    for frame in _sorted_global_frames(metadata):
        items.append(
            FormattedItem(
                text=f"[Global] t={_format_num(frame.timestamp)}s",
                image_path=frame.path,
            )
        )
    for roi in active_rois(metadata):
        for frame in _sorted_roi_frames(roi):
            items.append(
                FormattedItem(
                    text=f"[ROI{roi.roi_id}] t={_format_num(frame.timestamp)}s",
                    image_path=frame.path,
                )
            )
    return items


def _roi_items_in_interval(
    rois: Sequence[RoiSection],
    t_lo: float,
    t_hi: float,
) -> list[FormattedItem]:
    """ROI frames in [t_lo, t_hi), grouped by roi_id then time."""
    items: list[FormattedItem] = []
    for roi in rois:
        for frame in _sorted_roi_frames(roi):
            t = float(frame.timestamp)
            if t_lo <= t < t_hi:
                items.append(
                    FormattedItem(
                        text=f"[ROI{roi.roi_id}] t={_format_num(frame.timestamp)}s",
                        image_path=frame.path,
                    )
                )
    return items


def _roi_start_t(roi: RoiSection) -> float:
    """Tube start used to pair with start_only Globals (roi.time_range[0])."""
    return float(roi.time_range[0])


def _global_item(frame: FrameRef) -> FormattedItem:
    return FormattedItem(
        text=f"[Global] t={_format_num(frame.timestamp)}s",
        image_path=frame.path,
    )


def _roi_item(roi: RoiSection, frame: FrameRef) -> FormattedItem:
    return FormattedItem(
        text=f"[ROI{roi.roi_id}] t={_format_num(frame.timestamp)}s",
        image_path=frame.path,
    )


def _tube_story_frame_items(
    metadata: VideoBundleMetadata,
) -> list[FormattedItem]:
    """One Global establishing shot, then that tube's full ROI sequence.

    Tubes are sorted by first crop time, then roi_id. Globals are consumed in
    timestamp order, one per tube. Leftover Globals are appended at the end.
    """
    unused = list(_sorted_global_frames(metadata))
    rois = sorted(
        active_rois(metadata),
        key=lambda roi: (_roi_start_t(roi), int(roi.roi_id)),
    )
    items: list[FormattedItem] = []
    for roi in rois:
        if unused:
            items.append(_global_item(unused.pop(0)))
        for frame in _sorted_roi_frames(roi):
            items.append(_roi_item(roi, frame))
    for frame in unused:
        items.append(_global_item(frame))
    return items


def _global_anchor_frame_items(
    metadata: VideoBundleMetadata,
) -> list[FormattedItem]:
    """Each Global as temporal anchor; ROIs in the interval grouped by roi_id.

    For Global timestamps G_0 < G_1 < ... < G_n:
      [ROIs with t < G_0] grouped by roi_id (edge case), then for each i:
        G_i → ROI1 in [G_i, G_{i+1}) → ROI2 in [G_i, G_{i+1}) → ...
    Last interval is t >= G_n. Within each ROI, timestamps ascend.
    """
    globals_ = _sorted_global_frames(metadata)
    rois = active_rois(metadata)

    if not globals_:
        return _roi_items_in_interval(rois, float("-inf"), float("inf"))

    items: list[FormattedItem] = []
    items.extend(
        _roi_items_in_interval(rois, float("-inf"), float(globals_[0].timestamp))
    )
    for g_i, g_frame in enumerate(globals_):
        items.append(
            FormattedItem(
                text=f"[Global] t={_format_num(g_frame.timestamp)}s",
                image_path=g_frame.path,
            )
        )
        t_hi = (
            float(globals_[g_i + 1].timestamp)
            if g_i + 1 < len(globals_)
            else float("inf")
        )
        items.extend(
            _roi_items_in_interval(rois, float(g_frame.timestamp), t_hi)
        )
    return items


def ordered_frame_items(
    metadata: VideoBundleMetadata,
    *,
    input_order: str = "timestamp",
) -> list[FormattedItem]:
    from .experiment import (
        INPUT_ORDER_GLOBAL_ANCHOR,
        INPUT_ORDER_GROUPED,
        INPUT_ORDER_TUBE_STORY,
        VALID_INPUT_ORDERS,
    )

    order = str(input_order).strip().lower()
    if order not in VALID_INPUT_ORDERS:
        raise ValueError(
            f"input_order must be one of {sorted(VALID_INPUT_ORDERS)}, got '{input_order}'"
        )
    if order == INPUT_ORDER_GROUPED:
        return _grouped_frame_items(metadata)
    if order == INPUT_ORDER_GLOBAL_ANCHOR:
        return _global_anchor_frame_items(metadata)
    if order == INPUT_ORDER_TUBE_STORY:
        return _tube_story_frame_items(metadata)
    return _interleaved_frame_items(metadata)


def format_input_order_dump_from_bundle(
    metadata: VideoBundleMetadata,
    *,
    input_order: str = "timestamp",
) -> str:
    """Dump VLM image order for a metadata bundle (pre-call)."""
    from .experiment import VALID_INPUT_ORDERS

    order = str(input_order).strip().lower()
    if order not in VALID_INPUT_ORDERS:
        raise ValueError(
            f"input_order must be one of {sorted(VALID_INPUT_ORDERS)}, got '{input_order}'"
        )
    lines: list[str] = [f"# input_order={order}", f"# video_id={metadata.video_id}"]
    for item in ordered_frame_items(metadata, input_order=order):
        if item.image_path is None:
            lines.append(item.text)
        else:
            lines.append(f"{item.text}  ->  {item.image_path}")
    return "\n".join(lines).rstrip() + "\n"


def format_video_bundle(
    metadata: VideoBundleMetadata,
    *,
    include_bbox_definition: bool | None = None,
    input_order: str = "timestamp",
) -> FormatResult:
    """Build ordered (text, image_path?) IR from parsed / composed metadata.

    Emits only per-frame labels (``[Global] t=…`` / ``[ROI#] t=…``) plus images.
    No ``time_range`` / ``normalized_bbox`` text and no sequence begin/end markers.
    ``include_bbox_definition`` is accepted for API compatibility but ignored
    (defaults to off).
    """
    from .experiment import VALID_INPUT_ORDERS

    order = str(input_order).strip().lower()
    if order not in VALID_INPUT_ORDERS:
        raise ValueError(
            f"input_order must be one of {sorted(VALID_INPUT_ORDERS)}, got '{input_order}'"
        )

    # Spatial text metadata was removed; keep the flag only for call-site compat.
    _ = include_bbox_definition

    items = list(ordered_frame_items(metadata, input_order=order))
    return FormatResult(
        items=items, metadata=metadata, warnings=list(metadata.warnings)
    )


def load_and_format_video_bundle(
    bundle_dir: Path | str,
    *,
    require_files: bool = True,
    input_order: str = "timestamp",
    roi_frames: str = "default",
) -> FormatResult:
    metadata = load_video_bundle_metadata(
        bundle_dir, require_files=require_files, roi_frames=roi_frames
    )
    return format_video_bundle(metadata, input_order=input_order)


def formatted_items_to_qwen_content(
    items: Iterable[FormattedItem],
    *,
    image_opts: Any | None = None,
    as_file_uri: bool = True,
    global_image_opts: Any | None = None,
) -> list[dict[str, Any]]:
    """Convert IR items into Qwen-VL style multimodal content dicts."""
    content: list[dict[str, Any]] = []
    for item in items:
        content.append({"type": "text", "text": item.text})
        if item.image_path is None:
            continue
        path = item.image_path.resolve()
        image_ref: str = path.as_uri() if as_file_uri else str(path)
        entry: dict[str, Any] = {"type": "image", "image": image_ref}
        opts = image_opts
        if global_image_opts is not None and str(item.text).lstrip().startswith("[Global]"):
            opts = global_image_opts
        if opts is not None:
            if getattr(opts, "min_pixels", None) is not None:
                entry["min_pixels"] = opts.min_pixels
            if getattr(opts, "max_pixels", None) is not None:
                entry["max_pixels"] = opts.max_pixels
        content.append(entry)
    return content


def iter_bundle_image_uris(
    metadata: VideoBundleMetadata,
    *,
    input_order: str = "timestamp",
) -> list[str]:
    """Stable list of image file URIs in VLM prompt order."""
    return [
        item.image_path.resolve().as_uri()
        for item in ordered_frame_items(metadata, input_order=input_order)
        if item.image_path is not None
    ]


def debug_format_summary(result: FormatResult) -> str:
    """Human-readable dump of text headers + image paths (pre-VLM)."""
    lines: list[str] = []
    lines.append(f"video_id={result.metadata.video_id}")
    lines.append(f"sources={list(result.metadata.sources)}")
    lines.append(
        f"global_frames={len(result.metadata.global_section.frames)} "
        f"roi_tubes={len(active_rois(result.metadata))} "
        f"images={len(result.image_paths)}"
    )
    for item in result.items:
        if item.image_path is None:
            lines.append("--- text ---")
            lines.append(item.text)
        else:
            lines.append(f"{item.text}  ->  {item.image_path}")
    return "\n".join(lines)
