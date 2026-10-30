"""Absolute timestamps for a Qwen3-VL JPEG frame list.

Installed qwen_vl_utils rebuilds video metadata as ``index / sample_fps``
starting at 0. ``VideoMetadata`` has no timestamp list; the processor computes
``frames_indices[i] / fps``. Encoding each real timestamp as
``round(timestamp * scale) / scale`` is the supported way to keep irregular
times and a non-zero origin.

``video.timestamp_mode=legacy`` leaves that fetcher clock unchanged.
``source_timestamp`` makes ``generate`` pass one group per video item.
The vlm-inference tube packing sends Global as an image, so ``global_is_video``
is false there. The SurveillanceVQA packing sends each Global JPEG as its own
one-frame video.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

TIMESTAMP_LEGACY = "legacy"
TIMESTAMP_SOURCE = "source_timestamp"
VALID_TIMESTAMP_MODES = frozenset({TIMESTAMP_LEGACY, TIMESTAMP_SOURCE})


def normalize_timestamp_mode(value: object) -> str:
    mode = str(TIMESTAMP_LEGACY if value is None else value).strip().lower()
    if mode not in VALID_TIMESTAMP_MODES:
        raise ValueError(
            f"video.timestamp_mode must be one of {sorted(VALID_TIMESTAMP_MODES)}, got {value!r}"
        )
    return mode


def frame_timestamps_for_generate(
    sample: Any,
    timestamp_mode: object,
    *,
    global_is_video: bool,
) -> list[list[float]] | None:
    """Return per-video source timestamps, or None for the legacy clock."""
    mode = normalize_timestamp_mode(timestamp_mode)
    if mode == TIMESTAMP_LEGACY:
        return None
    return source_video_timestamp_groups(sample, global_is_video=global_is_video)


def source_video_timestamp_groups(sample: Any, *, global_is_video: bool) -> list[list[float]]:
    """One timestamp list per video item, in message order."""
    explicit = getattr(sample, "video_timestamp_groups", None)
    if explicit is not None:
        groups = [[float(stamp) for stamp in group] for group in explicit]
        if not groups or any(len(group) == 0 for group in groups):
            raise ValueError("video_timestamp_groups must contain one non-empty group per video")
        return groups

    if getattr(sample, "input_type", None) == "combined_image_roi_full_video":
        # The start image is not a video, so it has no timestamp group.
        return combined_video_timestamp_groups(sample)

    if getattr(sample, "input_type", None) != "tube_global_roi_videos":
        raise ValueError(
            "timestamp_mode=source_timestamp needs sample.video_timestamp_groups "
            f"for input_type={getattr(sample, 'input_type', None)!r}"
        )

    groups: list[list[float]] = []
    for block in sample.tube_global_roi_blocks:
        sequence = getattr(block, "sequence_timestamps", None)
        if not sequence or len(sequence) != len(block.sequence_paths):
            raise ValueError(
                f"ROI {block.roi_id}: source_timestamp needs one timestamp per ROI frame"
            )
        if global_is_video:
            if getattr(block, "global_timestamp", None) is None:
                raise ValueError(f"ROI {block.roi_id}: missing Global frame timestamp")
            groups.append([float(block.global_timestamp)])
        groups.append([float(stamp) for stamp in sequence])
    if not groups:
        raise ValueError("source_timestamp sample has no video items")
    return groups


def combined_video_timestamp_groups(sample: Any) -> list[list[float]]:
    """ROI videos in sample order, then the full-frame video. The start image is omitted."""
    rois = list(getattr(sample, "combined_roi_videos", None) or [])
    if not rois:
        raise ValueError("combined input needs at least one ROI video")
    groups: list[list[float]] = []
    for roi in rois:
        sequence = list(getattr(roi, "sequence_timestamps", None) or [])
        paths = list(getattr(roi, "sequence_paths", None) or [])
        if not sequence or len(sequence) != len(paths):
            raise ValueError(
                f"ROI {getattr(roi, 'roi_id', '?')}: source_timestamp needs one timestamp per ROI frame"
            )
        groups.append([float(stamp) for stamp in sequence])
    full_paths = list(getattr(sample, "combined_full_paths", None) or [])
    full_times = list(getattr(sample, "combined_full_timestamps", None) or [])
    if not full_times or len(full_times) != len(full_paths):
        raise ValueError("combined full-frame video needs one timestamp per frame")
    groups.append([float(stamp) for stamp in full_times])
    return groups


def encode_timestamps(timestamps: Sequence[float], *, scale: int = 1000) -> tuple[list[int], float]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not timestamps:
        raise ValueError("timestamps must be non-empty")
    indices = [int(round(float(stamp) * int(scale))) for stamp in timestamps]
    return indices, float(scale)


def rewrite_fetched_metadata(
    metadata: Mapping,
    timestamps: Sequence[float],
    *,
    scale: int = 1000,
) -> dict:
    """Replace fake 0,1,2,... indices with encoded absolute timestamps.

    If the fetcher duplicated the last JPEG to satisfy temporal factor 2,
    that extra slot repeats the last real index. It does not advance time.
    """
    fetched = [int(value) for value in metadata["frames_indices"]]
    encoded, fps = encode_timestamps(timestamps, scale=scale)
    if len(fetched) < len(encoded):
        raise ValueError(
            f"fetcher returned {len(fetched)} frames for {len(encoded)} timestamps"
        )
    if len(fetched) > len(encoded):
        encoded = encoded + [encoded[-1]] * (len(fetched) - len(encoded))
    updated = dict(metadata)
    updated["frames_indices"] = encoded
    updated["fps"] = fps
    return updated


def rewrite_video_metadata_list(
    metadatas: Sequence[MutableMapping],
    groups: Sequence[Sequence[float]],
    *,
    scale: int = 1000,
) -> list[dict]:
    if len(metadatas) != len(groups):
        raise ValueError(
            f"{len(groups)} timestamp groups for {len(metadatas)} videos"
        )
    return [
        rewrite_fetched_metadata(metadata, stamps, scale=scale)
        for metadata, stamps in zip(metadatas, groups)
    ]
