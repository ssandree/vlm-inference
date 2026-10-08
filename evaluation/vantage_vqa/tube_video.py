"""Prototype: pack existing ROI Fixed JPEGs as Global IMAGE + one VIDEO per tube.

Does not change motion, tube generation, timestamps, crops, or the image-packed
experiment path. Reconstructs tube membership from the existing SpatialJob.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from vlm_inference.gt_multilabel_cls import measured_image_resolutions
from vlm_inference.inputs import Sample
from vlm_inference.sovabench import fill_job_cost
from vlm_inference.sovabench_spatial import _global_name, _roi_name, _safe_dirname

from .visual import VisualInput

INPUT_TYPE_TUBE_VIDEOS = "global_then_tube_videos"


@dataclass(frozen=True)
class PackedFrame:
    path: str
    timestamp: float
    native_frame: int
    source: str


@dataclass(frozen=True)
class TubeVideoPacking:
    global_frames: list[PackedFrame]
    tubes: list[tuple[int, list[PackedFrame]]]


def _basename(path: str) -> str:
    if path.startswith("file://"):
        parsed = urlparse(path)
        return Path(unquote(parsed.path)).name
    return Path(path).name


def _as_uri(path: Path) -> str:
    return path.resolve().as_uri()


def bind_existing_roi_jpegs(visual: VisualInput, frames_root: Path) -> VisualInput:
    """Attach already-materialized JPEGs. Does not recrop or rewrite files."""
    job = visual.job
    if job is None:
        raise ValueError("VisualInput.job is required")
    out_dir = (
        Path(frames_root)
        / job.sample.video_id
        / _safe_dirname(job.sample.sample_id)
        / job.condition
    )
    paths: list[str] = []
    missing: list[str] = []
    for stamp in job.global_stamps:
        dest = out_dir / _global_name(stamp)
        if not dest.is_file():
            missing.append(str(dest))
        paths.append(_as_uri(dest) if dest.is_file() else str(dest))
    for roi in job.rois:
        for stamp in job.roi_stamps.get(int(roi.roi_id), ()):
            dest = out_dir / _roi_name(int(roi.roi_id), stamp)
            if not dest.is_file():
                missing.append(str(dest))
            paths.append(_as_uri(dest) if dest.is_file() else str(dest))
    if missing:
        raise FileNotFoundError("missing existing ROI JPEGs:\n" + "\n".join(missing))
    job.frame_paths = paths
    job.input_image_resolutions = measured_image_resolutions(paths)
    fill_job_cost(job)
    visual.frame_paths = list(paths)
    visual.num_input_images = len(paths)
    visual.processed_pixels = int(job.total_input_pixels)
    visual.input_image_resolutions = [list(row) for row in job.input_image_resolutions]
    return visual


def restore_tube_packing(visual: VisualInput) -> TubeVideoPacking:
    """Restore tube membership from job metadata; verify filenames and source_types."""
    job = visual.job
    if job is None:
        raise ValueError("VisualInput.job is required")
    paths = list(job.frame_paths) or list(visual.frame_paths)
    n_global = len(job.global_stamps)
    n_roi = sum(len(job.roi_stamps.get(int(roi.roi_id), ())) for roi in job.rois)
    if len(paths) != n_global + n_roi:
        raise ValueError(
            f"frame_paths={len(paths)} != globals={n_global} + roi_crops={n_roi}"
        )
    if len(visual.source_types) != len(paths):
        raise ValueError("source_types length does not match frame_paths")

    idx = 0
    globals_: list[PackedFrame] = []
    for stamp in job.global_stamps:
        path = paths[idx]
        expect = _global_name(stamp)
        got = _basename(path)
        if got != expect:
            raise ValueError(f"global JPEG mismatch: {got} != {expect}")
        if visual.source_types[idx] != "global":
            raise ValueError(
                f"source_types[{idx}]={visual.source_types[idx]!r} expected 'global'"
            )
        globals_.append(
            PackedFrame(
                path=path,
                timestamp=float(stamp.timestamp),
                native_frame=int(stamp.native_frame),
                source="global",
            )
        )
        idx += 1

    tubes: list[tuple[int, list[PackedFrame]]] = []
    for roi in job.rois:
        roi_id = int(roi.roi_id)
        frames: list[PackedFrame] = []
        for stamp in job.roi_stamps.get(roi_id, ()):
            path = paths[idx]
            expect = _roi_name(roi_id, stamp)
            got = _basename(path)
            if got != expect:
                raise ValueError(f"tube {roi_id} JPEG mismatch: {got} != {expect}")
            expect_src = f"roi:{roi_id}"
            if visual.source_types[idx] != expect_src:
                raise ValueError(
                    f"source_types[{idx}]={visual.source_types[idx]!r} expected {expect_src!r}"
                )
            frames.append(
                PackedFrame(
                    path=path,
                    timestamp=float(stamp.timestamp),
                    native_frame=int(stamp.native_frame),
                    source=expect_src,
                )
            )
            idx += 1
        tubes.append((roi_id, frames))
    if idx != len(paths):
        raise ValueError(f"unconsumed frame_paths: {len(paths) - idx}")
    return TubeVideoPacking(global_frames=globals_, tubes=tubes)


def image_packed_sample(visual: VisualInput) -> Sample:
    """Existing image-packed ROI sample. Same constructor as the pilot runner."""
    from vlm_inference.sovabench_spatial import _motion_sample

    if visual.job is None:
        raise ValueError("VisualInput.job is required")
    return _motion_sample(visual.job)


def tube_video_sample(visual: VisualInput, packing: TubeVideoPacking) -> Sample:
    if visual.job is None:
        raise ValueError("VisualInput.job is required")
    job = visual.job
    bboxes = {int(roi.roi_id): tuple(int(v) for v in roi.bbox_px) for roi in job.rois}
    return Sample(
        job_name=f"{job.sample.sample_id}__{job.condition}__tube_video",
        sample_id=job.sample.sample_id,
        input_type=INPUT_TYPE_TUBE_VIDEOS,
        paths=list(job.frame_paths),
        frame_width=int(job.video_width),
        frame_height=int(job.video_height),
        roi_bboxes_px=bboxes or None,
        global_image_paths=[frame.path for frame in packing.global_frames],
        tube_video_paths=[
            (roi_id, [frame.path for frame in frames]) for roi_id, frames in packing.tubes
        ],
    )


def evidence_rows(packing: TubeVideoPacking) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in packing.global_frames:
        rows.append(
            {
                "source": frame.source,
                "tube_id": None,
                "timestamp": frame.timestamp,
                "native_frame": frame.native_frame,
                "path": frame.path,
                "basename": _basename(frame.path),
            }
        )
    for roi_id, frames in packing.tubes:
        for frame in frames:
            rows.append(
                {
                    "source": frame.source,
                    "tube_id": roi_id,
                    "timestamp": frame.timestamp,
                    "native_frame": frame.native_frame,
                    "path": frame.path,
                    "basename": _basename(frame.path),
                }
            )
    return rows


def same_jpeg_evidence(
    image_sample: Sample, tube_sample: Sample, packing: TubeVideoPacking
) -> dict[str, Any]:
    image_paths = [_basename(p) for p in image_sample.paths]
    tube_paths = [_basename(p) for p in tube_sample.global_image_paths]
    for _roi_id, paths in tube_sample.tube_video_paths:
        tube_paths.extend(_basename(p) for p in paths)
    packing_paths = [_basename(r["path"]) for r in evidence_rows(packing)]
    return {
        "image_packed_n": len(image_sample.paths),
        "tube_video_n": len(tube_paths),
        "same_basenames_multiset": sorted(image_paths) == sorted(tube_paths),
        "same_order_as_job": image_paths == packing_paths == tube_paths,
        "image_packed_basenames": image_paths,
        "tube_video_basenames": tube_paths,
    }


def summarize_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    n_image = 0
    n_video = 0
    for msg in messages:
        for item in msg.get("content") or []:
            typ = item.get("type")
            row: dict[str, Any] = {"type": typ}
            if typ == "image":
                n_image += 1
                row["path"] = item.get("image")
                row["min_pixels"] = item.get("min_pixels")
                row["max_pixels"] = item.get("max_pixels")
            elif typ == "video":
                n_video += 1
                video = item.get("video")
                row["n_frames"] = len(video) if isinstance(video, (list, tuple)) else 1
                row["is_frame_list"] = isinstance(video, (list, tuple))
                row["paths"] = list(video) if isinstance(video, (list, tuple)) else [video]
                row["min_pixels"] = item.get("min_pixels")
                row["max_pixels"] = item.get("max_pixels")
                row["total_pixels"] = item.get("total_pixels")
                row["sample_fps"] = item.get("sample_fps")
            elif typ == "text":
                text = str(item.get("text") or "")
                row["text_preview"] = text[:80].replace("\n", "\\n")
            items.append(row)
    return {"n_image": n_image, "n_video": n_video, "items": items}


def inspect_processor(captioner, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Same process_vision_info + processor call as Qwen3VLCaptioner.generate."""
    return inspect_processor_output(captioner.processor, messages)


def inspect_processor_output(processor, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Same process_vision_info + processor call as Qwen3VLCaptioner.generate."""
    from qwen_vl_utils import process_vision_info

    from vlm_inference.model import count_processor_tokens

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_shapes: list[list[int]] = []
    if videos is not None:
        videos_only, video_metadatas = zip(*videos)
        videos_only, video_metadatas = list(videos_only), list(video_metadatas)
        for video in videos_only:
            video_shapes.append(list(video.shape))
    else:
        videos_only, video_metadatas = None, None

    image_sizes = []
    if images:
        for image in images:
            image_sizes.append({"width": int(image.size[0]), "height": int(image.size[1])})

    inputs = processor(
        text=[text],
        images=images,
        videos=videos_only,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **(video_kwargs or {}),
    )
    token_stats = count_processor_tokens(
        processor, getattr(inputs, "input_ids", None)
    )
    keys = list(inputs.keys())
    ids = inputs["input_ids"][0] if "input_ids" in keys else None
    image_grid = inputs["image_grid_thw"].tolist() if "image_grid_thw" in keys else None
    video_grid = inputs["video_grid_thw"].tolist() if "video_grid_thw" in keys else None
    tps = int(processor.video_processor.temporal_patch_size)
    packed = []
    if video_grid is not None:
        for grid, shape in zip(video_grid, video_shapes):
            n_frames = int(shape[0])
            grid_t = int(grid[0])
            packed.append(
                {
                    "fetch_video_shape": shape,
                    "video_grid_thw": grid,
                    "temporal_patch_size": tps,
                    "grid_t_equals_nframes_div_tps": n_frames // tps == grid_t,
                    "n_frames": n_frames,
                }
            )
    n_image_pad = (
        int((ids == processor.image_token_id).sum().item()) if ids is not None else None
    )
    n_video_pad = (
        int((ids == processor.video_token_id).sum().item()) if ids is not None else None
    )
    return {
        "video_kwargs": video_kwargs,
        "pre_processor_image_sizes": image_sizes,
        "pre_processor_video_shapes": video_shapes,
        "image_grid_thw": image_grid,
        "video_grid_thw": video_grid,
        "pixel_values_shape": (
            list(inputs["pixel_values"].shape) if "pixel_values" in keys else None
        ),
        "pixel_values_videos_shape": (
            list(inputs["pixel_values_videos"].shape)
            if "pixel_values_videos" in keys
            else None
        ),
        "token_stats": token_stats,
        "n_image_pad": n_image_pad,
        "n_video_pad": n_video_pad,
        "temporal_pack": packed,
    }


def validate_tube_representation(
    *,
    packing: TubeVideoPacking,
    packed_sample: Sample,
    tube_sample: Sample,
    messages: list[dict[str, Any]],
    audit: dict[str, Any],
    duration_sec: float,
) -> list[str]:
    """Return structural/evidence failure codes. Empty list means pass."""
    failures: list[str] = []
    if not packing.global_frames and not packing.tubes:
        failures.append("empty_visual")
    if not packing.global_frames:
        failures.append("missing_global_anchor")

    roi_paths: list[str] = []
    for roi_id, frames in packing.tubes:
        if len(frames) < 2:
            failures.append(f"tube_{roi_id}_frames_lt_2")
        timestamps = [float(frame.timestamp) for frame in frames]
        if any(timestamps[i] > timestamps[i + 1] for i in range(len(timestamps) - 1)):
            failures.append(f"tube_{roi_id}_timestamps_not_monotonic")
        for frame in frames:
            if frame.timestamp < -1e-3 or frame.timestamp > float(duration_sec) + 0.05:
                failures.append(
                    f"tube_{roi_id}_timestamp_out_of_range:{frame.timestamp}"
                )
            roi_paths.append(frame.path)
    if len(roi_paths) != len(set(roi_paths)):
        failures.append("roi_frame_in_multiple_tubes")

    for frame in packing.global_frames:
        if frame.timestamp < -1e-3 or frame.timestamp > float(duration_sec) + 0.05:
            failures.append(f"global_timestamp_out_of_range:{frame.timestamp}")

    evidence = same_jpeg_evidence(packed_sample, tube_sample, packing)
    if not evidence["same_basenames_multiset"] or not evidence["same_order_as_job"]:
        failures.append("jpeg_evidence_mismatch")

    summary = summarize_messages(messages)
    if summary["n_image"] != len(packing.global_frames):
        failures.append(
            f"message_image_count:{summary['n_image']}!={len(packing.global_frames)}"
        )
    if summary["n_video"] != len(packing.tubes):
        failures.append(
            f"message_video_count:{summary['n_video']}!={len(packing.tubes)}"
        )
    video_items = [item for item in summary["items"] if item.get("type") == "video"]
    if any(not item.get("is_frame_list") for item in video_items):
        failures.append("video_not_frame_list")
    for (roi_id, frames), item in zip(packing.tubes, video_items):
        expected = [_basename(frame.path) for frame in frames]
        got = [_basename(path) for path in item.get("paths") or []]
        if expected != got:
            failures.append(f"tube_{roi_id}_video_paths_mismatch")
        other = {
            _basename(frame.path)
            for other_id, other_frames in packing.tubes
            if other_id != roi_id
            for frame in other_frames
        }
        if other.intersection(got):
            failures.append(f"tube_{roi_id}_mixed_with_other_tube")

    if packing.global_frames:
        grids = audit.get("image_grid_thw")
        if not grids:
            failures.append("missing_image_grid_thw")
        else:
            if len(grids) != len(packing.global_frames):
                failures.append("image_grid_count_mismatch")
            for grid in grids:
                if len(grid) != 3 or any(int(v) < 1 for v in grid):
                    failures.append(f"bad_image_grid_thw:{grid}")
    if packing.tubes:
        grids = audit.get("video_grid_thw")
        if not grids:
            failures.append("missing_video_grid_thw")
        else:
            if len(grids) != len(packing.tubes):
                failures.append("video_grid_count_mismatch")
            for grid in grids:
                if len(grid) != 3 or any(int(v) < 1 for v in grid):
                    failures.append(f"bad_video_grid_thw:{grid}")
        packs = audit.get("temporal_pack") or []
        if len(packs) != len(packing.tubes):
            failures.append("temporal_pack_count_mismatch")
        for pack in packs:
            if int(pack.get("temporal_patch_size") or 0) != 2:
                failures.append("temporal_patch_size_not_2")
            if not pack.get("grid_t_equals_nframes_div_tps"):
                failures.append("temporal_pack_mismatch")

    stats = audit.get("token_stats") or {}
    visual = stats.get("num_visual_tokens")
    if visual is None or int(visual) <= 0:
        failures.append("visual_token_count_invalid")
    pads = int(audit.get("n_image_pad") or 0) + int(audit.get("n_video_pad") or 0)
    if visual is not None and pads != int(visual):
        failures.append("visual_token_pad_mismatch")
    return failures
