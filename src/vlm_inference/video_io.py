"""PyAV video reader for qwen-vl-utils (torchvision.io.read_video is gone)."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote, urlparse

import av
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_local_path(video_path: str) -> str:
    if video_path.startswith("file://"):
        parsed = urlparse(video_path)
        return unquote(parsed.path)
    return video_path


def read_video_av(ele: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any], float]:
    """Read and sample a video with PyAV. Compatible with qwen_vl_utils backends."""
    from qwen_vl_utils.vision_process import calculate_video_frame_range, smart_nframes

    video_path = _to_local_path(ele["video"])
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    st = time.time()
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate) if stream.average_rate else 30.0
        if video_fps <= 0:
            video_fps = 30.0

        # Prefer stream metadata when available; otherwise count while decoding.
        total_frames = int(stream.frames) if stream.frames and stream.frames > 0 else 0

        frames_rgb: list[np.ndarray] = []
        for frame in container.decode(video=0):
            frames_rgb.append(frame.to_ndarray(format="rgb24"))
        if total_frames <= 0:
            total_frames = len(frames_rgb)
        else:
            total_frames = min(total_frames, len(frames_rgb))
            frames_rgb = frames_rgb[:total_frames]

        if total_frames == 0:
            raise RuntimeError(f"No frames decoded from {video_path}")

        start_frame, end_frame, frame_count = calculate_video_frame_range(
            ele, total_frames=total_frames, video_fps=video_fps
        )
        clipped = frames_rgb[start_frame : end_frame + 1]
        clipped_total = len(clipped)
        if clipped_total == 1 and ele.get("allow_single_frame_padding", False):
            # Explicit opt-in for a fixed-window pilot's one-frame final tail.
            # Duplicate that frame for Qwen's temporal patch, without adding
            # observations outside the segment or changing its recorded length.
            nframes = 2
        else:
            nframes = smart_nframes(ele, total_frames=clipped_total, video_fps=video_fps)
        idx = torch.linspace(0, clipped_total - 1, nframes).round().long()
        sample_fps = nframes / max(clipped_total, 1e-6) * video_fps

        selected = [clipped[int(i)] for i in idx.tolist()]
        video = torch.from_numpy(np.stack(selected, axis=0)).permute(0, 3, 1, 2).contiguous()

        abs_indices = (idx + start_frame).tolist()
        video_metadata = dict(
            fps=video_fps,
            frames_indices=abs_indices,
            total_num_frames=total_frames,
            video_backend="av",
        )
        logger.info(
            "av: path=%s total_frames=%s video_fps=%.3f nframes=%s time=%.3fs",
            video_path,
            total_frames,
            video_fps,
            nframes,
            time.time() - st,
        )
        return video, video_metadata, sample_fps
    finally:
        container.close()


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    native_fps: float
    frame_count: int
    width: int
    height: int

    def timestamp_of(self, frame_idx: int) -> float:
        fps = float(self.native_fps) if self.native_fps > 0 else 30.0
        return float(frame_idx) / fps

    def nearest_frame_index(self, timestamp_sec: float) -> int:
        if self.frame_count <= 0:
            return 0
        fps = float(self.native_fps) if self.native_fps > 0 else 30.0
        idx = int(round(float(timestamp_sec) * fps))
        return min(max(idx, 0), int(self.frame_count) - 1)


def inspect_video(path: str | Path) -> VideoInfo:
    video_path = str(Path(_to_local_path(str(path))).resolve())
    return _inspect_video_cached(video_path)


@lru_cache(maxsize=32)
def _inspect_video_cached(video_path: str) -> VideoInfo:
    video_path_obj = Path(video_path)
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate) if stream.average_rate else 30.0
        if native_fps <= 1e-6:
            native_fps = 30.0
        frame_count = int(stream.frames) if stream.frames and stream.frames > 0 else 0
        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)
        if frame_count <= 0 and stream.duration and stream.time_base:
            frame_count = int(round(float(stream.duration * stream.time_base) * native_fps))
    finally:
        container.close()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path_obj}")
    return VideoInfo(
        path=video_path_obj,
        native_fps=native_fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def resolve_video_file(video_id: str, roots: Sequence[str | Path]) -> Path:
    """Return the first existing `{video_id}.mp4` under the given roots."""
    searched: list[str] = []
    for root in roots:
        if root is None or str(root).strip() == "":
            continue
        path = Path(root) / f"{video_id}.mp4"
        searched.append(str(path.parent))
        if path.is_file():
            return path
    if not searched:
        raise FileNotFoundError(f"missing video {video_id}.mp4 (no search roots)")
    raise FileNotFoundError(
        f"missing video {video_id}.mp4 in: {', '.join(searched)}"
    )


def extract_video_frames(video_path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    """Sequential RGB decode of selected 0-based frame indices."""
    wanted = sorted(int(i) for i in indices if int(i) >= 0)
    if not wanted:
        return {}
    wanted_set = set(wanted)
    last = wanted[-1]
    path = Path(_to_local_path(str(video_path)))
    container = av.open(str(path))
    got: dict[int, np.ndarray] = {}
    try:
        frame_idx = 0
        for frame in container.decode(video=0):
            if frame_idx in wanted_set:
                got[frame_idx] = frame.to_ndarray(format="rgb24")
            if frame_idx >= last:
                break
            frame_idx += 1
    finally:
        container.close()
    missing = [i for i in wanted if i not in got]
    if missing:
        raise RuntimeError(
            f"{path.name}: failed to decode frames {missing[:8]}"
            f"{'…' if len(missing) > 8 else ''}"
        )
    return got


def install_av_video_reader() -> None:
    """Register PyAV backend and force qwen-vl-utils to use it."""
    import qwen_vl_utils.vision_process as vp

    vp.VIDEO_READER_BACKENDS["av"] = read_video_av
    vp.FORCE_QWENVL_VIDEO_READER = "av"
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "av"
    if hasattr(vp.get_video_reader_backend, "cache_clear"):
        vp.get_video_reader_backend.cache_clear()
