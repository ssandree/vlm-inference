#!/usr/bin/env python3
"""Compare walking-segment duration cuts. Does not select a final threshold."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EVAL_ROOT = Path(__file__).resolve().parents[1]
PILOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_pilot_6"
SEG_ROOT = PILOT / "walking_analysis" / "motion_segment_pilot"
ANN_DIR = Path("/data/datasets/VIRAT/annotations")
OUT = SEG_ROOT / "minimum_duration"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
DIRECTION_KO = {
    "left_to_right": "왼쪽→오른쪽",
    "right_to_left": "오른쪽→왼쪽",
    "top_to_bottom": "위→아래",
    "bottom_to_top": "아래→위",
    "upper_left_to_lower_right": "왼쪽위→오른쪽아래",
    "upper_right_to_lower_left": "오른쪽위→왼쪽아래",
    "lower_left_to_upper_right": "왼쪽아래→오른쪽위",
    "lower_right_to_upper_left": "오른쪽아래→왼쪽위",
    None: "방향 없음",
}


def load_segments() -> list[dict]:
    rows = []
    for path in sorted(SEG_ROOT.glob("VIRAT_*/motion_segments.json")):
        rows.extend(json.loads(path.read_text())["segments"])
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def brief(segment: dict) -> dict:
    return {
        "segment_id": segment["segment_id"],
        "video_id": segment["video_id"],
        "source_track_id": segment["source_track_id"],
        "duration_sec": segment["duration_sec"],
        "displacement_px": segment["displacement_px"],
        "path_length_px": segment["path_length_px"],
        "direction_proposal": segment["direction_proposal"],
        "motion_state": segment["motion_state"],
        "start_reason": segment["start_reason"],
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
    }


def neighbors(segment: dict, grouped: dict[tuple[str, str], list[dict]]) -> tuple[dict | None, dict | None]:
    rows = grouped[(segment["video_id"], segment["source_track_id"])]
    index = next(i for i, row in enumerate(rows) if row["segment_id"] == segment["segment_id"])
    previous = rows[index - 1] if index else None
    following = rows[index + 1] if index + 1 < len(rows) else None
    return previous, following


def same_direction_continuation(segment: dict, following: dict | None) -> bool:
    if following is None or segment["direction_proposal"] is None:
        return False
    return following["direction_proposal"] == segment["direction_proposal"]


def histogram(values: list[float], edges: list[float]) -> list[dict]:
    rows = []
    for left, right in zip(edges, edges[1:]):
        rows.append({"from_sec": left, "to_sec": right, "n": sum(1 for value in values if left <= value < right)})
    return rows


def build_report(segments: list[dict]) -> dict:
    grouped = defaultdict(list)
    for segment in segments:
        grouped[(segment["video_id"], segment["source_track_id"])].append(segment)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["start_frame"])
    durations = [segment["duration_sec"] for segment in segments]
    displacements = [segment["displacement_px"] for segment in segments]
    paths = [segment["path_length_px"] for segment in segments]

    def dist(values: list[float]) -> dict:
        return {name: round(percentile(values, q), 2) for name, q in (("min", 0), ("p10", 0.1), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("max", 1))}

    comparisons = []
    for threshold in THRESHOLDS:
        removed = [segment for segment in segments if segment["duration_sec"] < threshold]
        retained = [segment for segment in segments if segment["duration_sec"] >= threshold]
        removed_sorted = sorted(removed, key=lambda segment: (segment["duration_sec"], -segment["displacement_px"]))
        retained_directional = [
            segment for segment in retained
            if segment["motion_state"] == "moving" and segment["direction_proposal"]
        ]
        comparisons.append(
            {
                "minimum_duration_sec": threshold,
                "n_retained": len(retained),
                "n_removed": len(removed),
                "n_retained_moving_with_direction": len(retained_directional),
                "removed_examples": [brief(segment) for segment in removed_sorted[:8]],
                "shortest_retained_moving_with_direction": brief(min(retained_directional, key=lambda segment: segment["duration_sec"])) if retained_directional else None,
                "removed_with_direction": [brief(segment) for segment in removed if segment["direction_proposal"]],
                "removed_stationary": [brief(segment) for segment in removed if segment["motion_state"] == "stationary"],
            }
        )

    short = [segment for segment in segments if segment["duration_sec"] < 2]
    classes = {"artifact_under_1s": [], "same_direction_head": [], "stationary_transition": [], "other_under_2s": []}
    for segment in short:
        _previous, following = neighbors(segment, grouped)
        if segment["duration_sec"] < 1 and segment["direction_proposal"] is None:
            classes["artifact_under_1s"].append(brief(segment))
        elif same_direction_continuation(segment, following):
            classes["same_direction_head"].append(brief(segment))
        elif segment["motion_state"] == "stationary" and segment["start_reason"] == "stationary_interval":
            classes["stationary_transition"].append(brief(segment))
        else:
            classes["other_under_2s"].append(brief(segment))

    directional_moving = [segment for segment in segments if segment["motion_state"] == "moving" and segment["direction_proposal"]]
    return {
        "final_threshold_selected": False,
        "recommended_minimum_duration_sec": 1.0,
        "n_segments": len(segments),
        "duration_sec": dist(durations),
        "duration_histogram_sec": histogram(durations, [0, 0.05, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 1000]),
        "displacement_px": dist(displacements),
        "path_length_px": dist(paths),
        "moving_with_direction_min_duration_sec": round(min(segment["duration_sec"] for segment in directional_moving), 2),
        "under_1s_all_direction_null": all(segment["direction_proposal"] is None for segment in segments if segment["duration_sec"] < 1),
        "under_1s_max_displacement_px": round(max(segment["displacement_px"] for segment in segments if segment["duration_sec"] < 1), 2),
        "classes_under_2s": {key: value for key, value in classes.items()},
        "comparisons": comparisons,
    }


def clip_jobs(segments: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for segment in segments:
        grouped[(segment["video_id"], segment["source_track_id"])].append(segment)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["start_frame"])

    def span(segment: dict, before: float, after: float) -> tuple[int, int]:
        previous, following = neighbors(segment, grouped)
        start = segment["start_frame"]
        end = segment["end_frame"]
        if previous is not None:
            start = max(previous["start_frame"], int(round(segment["start_frame"] - before * segment["fps"])))
        else:
            start = max(0, int(round(segment["start_frame"] - before * segment["fps"])))
        if following is not None:
            end = min(following["end_frame"], int(round(segment["end_frame"] + after * segment["fps"])))
        else:
            end = int(round(segment["end_frame"] + after * segment["fps"]))
        return start, end

    under_1 = [segment for segment in segments if segment["duration_sec"] < 1 and segment["direction_proposal"] is None]
    artifact = max(under_1, key=lambda segment: segment["displacement_px"])
    heads = []
    pauses = []
    for segment in segments:
        _previous, following = neighbors(segment, grouped)
        if 1 <= segment["duration_sec"] < 2 and same_direction_continuation(segment, following):
            heads.append(segment)
        if segment["start_reason"] == "stationary_interval" and 1 <= segment["duration_sec"] < 2:
            pauses.append(segment)
    shortest = min(
        (segment for segment in segments if segment["motion_state"] == "moving" and segment["direction_proposal"]),
        key=lambda segment: segment["duration_sec"],
    )
    pause = min(pauses, key=lambda segment: abs(segment["duration_sec"] - 1.63))
    head_15 = min((segment for segment in heads if segment["duration_sec"] < 1.5), key=lambda segment: -segment["displacement_px"])
    head_20 = min((segment for segment in heads if segment["duration_sec"] >= 1.5), key=lambda segment: -segment["displacement_px"])
    specs = [
        ("removed_at_1.0s", artifact, "1.0초 미만. 방향 없음. 이 기준에서 빠지는 조각", 2.0, 2.0),
        ("kept_at_1.0s_removed_at_1.5s", head_15, "1.0초에서는 남고 1.5초에서는 빠짐. 다음 구간과 같은 방향", 0.5, 3.0),
        ("kept_at_1.5s_removed_at_2.0s", head_20, "1.5초에서는 남고 2.0초에서는 빠짐. 다음 구간과 같은 방향", 0.5, 3.0),
        ("stationary_pause_around_1.6s", pause, "반대 방향 이동 사이의 정지. 1.5초에서는 남고 2.0초에서는 빠짐", 8.0, 8.0),
        ("shortest_directional_move_kept_at_2.0s", shortest, "8방위가 있는 이동 중 가장 짧음. 2.0초에서도 남음", 0.5, 3.0),
    ]
    jobs = []
    for role, segment, note, before, after in specs:
        start, end = span(segment, before, after)
        jobs.append({"role": role, "note": note, "segment": segment, "start_frame": start, "end_frame": max(end, segment["end_frame"])})
    return jobs


def load_boxes(video_id: str, track_id: str) -> dict[int, tuple[int, int, int, int]]:
    boxes = {}
    for line in (ANN_DIR / f"{video_id}.viratdata.objects.txt").read_text().splitlines():
        if not line.strip():
            continue
        obj_id, _dur, frame, x, y, width, height, obj_type = map(int, line.split()[:8])
        if obj_type == 1 and str(obj_id) == track_id:
            boxes[frame] = (x, y, width, height)
    return boxes


def draw(frame, job: dict, frame_index: int) -> None:
    segment = job["segment"]
    height, width = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pen = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 22 if width >= 1600 else 18)
    inside = segment["start_frame"] <= frame_index <= segment["end_frame"]
    mark = "이 구간" if inside else "앞뒤 맥락"
    direction = DIRECTION_KO.get(segment["direction_proposal"], str(segment["direction_proposal"]))
    lines = [
        f"{job['role']}  track {segment['source_track_id']}  {segment['duration_sec']:.2f}초  {direction}",
        job["note"],
        f"{mark}. 변위 {segment['displacement_px']:.0f}px, 경로 {segment['path_length_px']:.0f}px. 최종 기준으로 확정하지 않음.",
    ]
    line_h = 28 if width >= 1600 else 24
    pen.rectangle((0, 0, width, 12 + line_h * len(lines)), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        pen.text((12, 6 + index * line_h), line, font=font, fill=(245, 245, 245))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(jobs: list[dict]) -> list[dict]:
    by_video = defaultdict(list)
    for job in jobs:
        by_video[job["segment"]["video_id"]].append(job)
    written = []
    for video_id, video_jobs in by_video.items():
        sample = video_jobs[0]["segment"]
        width, height, fps = sample["frame_width"], sample["frame_height"], sample["fps"]
        boxes = {job["segment"]["source_track_id"]: load_boxes(video_id, job["segment"]["source_track_id"]) for job in video_jobs}
        encoders = []
        for job in video_jobs:
            segment = job["segment"]
            name = f"{job['role']}_{video_id}_t{segment['source_track_id']}_{segment['duration_sec']:.2f}s.mp4"
            out = OUT / "clips" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            encoder = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                    "-r", f"{fps:.6f}", "-i", "-", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", str(out),
                ],
                stdin=subprocess.PIPE,
            )
            encoders.append((job, encoder, out))
        cap = cv2.VideoCapture(sample["video_path"])
        frame_index = 0
        last = max(job["end_frame"] for job in video_jobs)
        try:
            while frame_index <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                for job, encoder, _out in encoders:
                    if job["start_frame"] <= frame_index <= job["end_frame"]:
                        painted = frame.copy()
                        box = boxes[job["segment"]["source_track_id"]].get(frame_index)
                        if box is not None:
                            x, y, w, h = box
                            color = (40, 220, 80) if job["segment"]["start_frame"] <= frame_index <= job["segment"]["end_frame"] else (180, 180, 180)
                            cv2.rectangle(painted, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), color, 2)
                        draw(painted, job, frame_index)
                        encoder.stdin.write(painted.tobytes())
                frame_index += 1
        finally:
            cap.release()
            for job, encoder, out in encoders:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed for {out}")
                written.append(
                    {
                        "file": str(out.relative_to(EVAL_ROOT)),
                        "role": job["role"],
                        "note": job["note"],
                        "segment_id": job["segment"]["segment_id"],
                        "duration_sec": job["segment"]["duration_sec"],
                        "displacement_px": job["segment"]["displacement_px"],
                        "direction_proposal": job["segment"]["direction_proposal"],
                    }
                )
        print(f"WROTE {video_id} clips={len(video_jobs)}", flush=True)
    return written


def main() -> None:
    segments = load_segments()
    report = build_report(segments)
    jobs = clip_jobs(segments)
    clips = render(jobs)
    report["boundary_clips"] = clips
    report["recommendation"] = (
        "1.0s. Every segment shorter than 1.0s has direction null and displacement below the meaningful-movement threshold. "
        "Every moving segment that already has an 8-way direction is at least 2.21s, so 1.0, 1.5, and 2.0 keep the same directional walks. "
        "1.5s and 2.0s start removing same-direction heads and a real pause between opposite walks."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "minimum_duration_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    # The policy points at the parent of this directory.
    alias = SEG_ROOT / "minimum_duration_comparison.json"
    alias.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"SEGMENTS {report['n_segments']} CLIPS {len(clips)}")
    for row in report["comparisons"]:
        print(row["minimum_duration_sec"], "retained", row["n_retained"], "removed", row["n_removed"], "directional", row["n_retained_moving_with_direction"], "removed_with_direction", len(row["removed_with_direction"]))


if __name__ == "__main__":
    main()
