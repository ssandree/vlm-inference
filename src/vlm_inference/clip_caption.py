"""Frame-disjoint fixed windows and plain-text clip review exports."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ClipWindow:
    index: int
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int  # exclusive


def segment_video(frame_count: int, fps: float, duration_sec: float = 3.0, *, allow_empty_tail: bool = False) -> list[ClipWindow]:
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Positive frame count and finite fps required")
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("Positive finite clip duration required")
    duration = frame_count / fps
    windows = []
    for i in range(math.ceil(duration / duration_sec)):
        start, end = i * duration_sec, min((i + 1) * duration_sec, duration)
        # Frames belong to [start, end): no boundary frame shared by two inputs.
        first = min(frame_count, math.ceil(start * fps - 1e-9))
        stop = min(frame_count, math.ceil(end * fps - 1e-9))
        if stop <= first:
            if allow_empty_tail and i == math.ceil(duration / duration_sec) - 1:
                # The final source frame remains displayed across this boundary.
                # Retain the real sub-frame tail using that frame explicitly.
                first = frame_count - 1
            else:
                raise ValueError("Clip interval contains no source frame")
        windows.append(ClipWindow(i, start, end, first, stop))
    return windows


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def time_label(seconds: float) -> str:
    ms = round(seconds * 1000)
    minutes, remainder = divmod(ms, 60000)
    sec, fraction = divmod(remainder, 1000)
    return f'{minutes:02d}:{sec:02d}' + (f'.{fraction:03d}' if fraction else '')


def review_text(video: str, clips: list[dict]) -> str:
    return f'=== {video} ===\n\n' + ''.join(
        f"[{time_label(c['start_sec'])}–{time_label(c['end_sec'])}]\n{c['caption']}\n\n"
        for c in clips
    )
