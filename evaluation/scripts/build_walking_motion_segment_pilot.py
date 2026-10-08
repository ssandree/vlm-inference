#!/usr/bin/env python3
"""Motion-segment candidates for the 6-video walking temporal-unit pilot.

Does not confirm walking or wandering, and does not write caption-event GT.
Smoothing and the stationary cut are pilot probes. They are not policy.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EVAL_ROOT = Path(__file__).resolve().parents[1]
PILOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_pilot_6"
ANN_DIR = Path("/data/datasets/VIRAT/annotations")
OUT = PILOT / "walking_analysis" / "motion_segment_pilot"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# Pilot probes. Recorded in procedure.json. Not written into annotation policy.
SPEED_WINDOW_SEC = 1.0
STILL_1S_PX = 8.0
DIRECTIONS = [
    "left_to_right",
    "upper_left_to_lower_right",
    "top_to_bottom",
    "upper_right_to_lower_left",
    "right_to_left",
    "lower_right_to_upper_left",
    "bottom_to_top",
    "lower_left_to_upper_right",
]
FOCUS = {
    "VIRAT_S_010001_02_000195_000498": ["7", "9", "12", "37", "52"],
    "VIRAT_S_010106_02_000656_000692": ["5"],
    "VIRAT_S_010111_08_000920_000954": ["88"],
}
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
REASON_KO = {
    "track_start": "track 시작",
    "track_gap": "track이 끊긴 지점",
    "activity_boundary": "VIRAT activity 경계",
    "stationary_interval": "멈춰 있는 구간",
    "direction_reversal": "방향 전환",
    "coherent_movement": "이어서 이동",
}


def displacement_threshold(frame_width: int) -> float:
    return max(20.0, 0.05 * frame_width)


def direction_8(dx: float, dy: float, frame_width: int) -> str | None:
    if math.hypot(dx, dy) < displacement_threshold(frame_width):
        return None
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360.0
    return DIRECTIONS[int(math.floor((angle + 22.5) / 45.0)) % 8]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_person_tracks(video_id: str) -> dict[int, dict]:
    tracks: dict[int, dict] = {}
    path = ANN_DIR / f"{video_id}.viratdata.objects.txt"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj_id, _dur, frame, x, y, width, height, obj_type = map(int, line.split()[:8])
        if obj_type != 1:
            continue
        track = tracks.setdefault(obj_id, {"centroids": {}, "boxes": {}})
        track["centroids"][frame] = (x + width / 2.0, y + height / 2.0)
        track["boxes"][frame] = (x, y, width, height)
    return tracks


def subject_events(doc: dict, track_id: str) -> list[dict]:
    rows = []
    for event in doc["candidate_events"]:
        for track in event["tracks"]:
            if track["track_id"] == track_id and track["relation"] == "subject":
                rows.append(event)
                break
    return rows


def overlap_events(doc: dict, track_id: str, start: int, end: int, frames: set[int]) -> list[dict]:
    rows = []
    for event in doc["candidate_events"]:
        inter_start = max(start, event["start_frame"])
        inter_end = min(end, event["end_frame"])
        if inter_end < inter_start:
            continue
        role = "temporal_only"
        for track in event["tracks"]:
            if track["track_id"] == track_id:
                role = track["relation"]
                break
        rows.append(
            {
                "event_id": event["event_id"],
                "activity_class": event["activity_class"],
                "role": role,
                "overlap_frames": inter_end - inter_start + 1,
                "tracked_frames_inside_overlap": sum(1 for frame in range(inter_start, inter_end + 1) if frame in frames),
            }
        )
    return rows


def piece_bounds(frames: list[int], fps: float) -> list[tuple[int, int, str]]:
    bounds = [(frames[0], frames[0], "track_start")]
    window = max(1, int(round(SPEED_WINDOW_SEC * fps)))
    for previous, frame in zip(frames, frames[1:]):
        if frame - previous > window:
            bounds.append((frame, previous, "track_gap"))
    grouped = []
    start = frames[0]
    reason = "track_start"
    for frame, previous, boundary in bounds[1:]:
        grouped.append((start, previous, reason))
        start = frame
        reason = boundary
    grouped.append((start, frames[-1], reason))
    return grouped


def cut_activity(frames: list[int], pieces: list[tuple[int, int, str]], events: list[dict]) -> list[tuple[int, int, str]]:
    cuts = []
    for event in events:
        cuts.extend([event["start_frame"], event["end_frame"] + 1])
    output = []
    for start, end, reason in pieces:
        boundaries = [start]
        for cut in sorted(set(cuts)):
            if start < cut <= end and any(frame >= cut for frame in frames if start <= frame <= end):
                boundaries.append(cut)
        boundaries.append(end + 1)
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            inside = [frame for frame in frames if left <= frame < right and frame <= end]
            if not inside:
                continue
            output.append((inside[0], inside[-1], reason if index == 0 else "activity_boundary"))
    return output


def label_motion(frames: list[int], centroids: dict[int, tuple[float, float]], fps: float) -> dict[int, str]:
    window = max(1, int(round(SPEED_WINDOW_SEC * fps)))

    def displacement(origin: int, dest: int) -> float | None:
        if dest - origin > window * 1.5:
            return None
        x0, y0 = centroids[origin]
        x1, y1 = centroids[dest]
        return math.hypot(x1 - x0, y1 - y0)

    labels = {}
    for position, frame in enumerate(frames):
        past = displacement(frames[position - window], frame) if position >= window else None
        future = displacement(frame, frames[position + window]) if position + window < len(frames) else None
        moving = (past is not None and past >= STILL_1S_PX) or (future is not None and future >= STILL_1S_PX)
        labels[frame] = "moving" if moving else "stationary"
    return labels


def merge_short_runs(frames: list[int], labels: dict[int, str], fps: float) -> dict[int, str]:
    window = max(1, int(round(SPEED_WINDOW_SEC * fps)))
    runs = []
    for frame in frames:
        if not runs or runs[-1][0] != labels[frame] or frame - runs[-1][2] > 1:
            runs.append([labels[frame], frame, frame, [frame]])
        else:
            runs[-1][2] = frame
            runs[-1][3].append(frame)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for index, run in enumerate(runs):
            if run[2] - run[1] + 1 >= window:
                continue
            neighbor = index - 1 if index > 0 else index + 1
            if neighbor < 0 or neighbor >= len(runs):
                continue
            for frame in run[3]:
                labels[frame] = runs[neighbor][0]
            runs[neighbor][3].extend(run[3])
            runs[neighbor][1] = min(runs[neighbor][1], run[1])
            runs[neighbor][2] = max(runs[neighbor][2], run[2])
            del runs[index]
            changed = True
            break
    return labels


def reversal_cuts(frames: list[int], centroids: dict[int, tuple[float, float]], threshold: float) -> list[int]:
    if len(frames) < 2:
        return []

    def dist(left: int, right: int) -> float:
        x0, y0 = centroids[left]
        x1, y1 = centroids[right]
        return math.hypot(x1 - x0, y1 - y0)

    def vector(left: int, right: int) -> tuple[float, float]:
        x0, y0 = centroids[left]
        x1, y1 = centroids[right]
        return (x1 - x0, y1 - y0)

    origin = 0
    extreme = 0
    cuts = []
    for index in range(1, len(frames)):
        if dist(frames[origin], frames[index]) > dist(frames[origin], frames[extreme]):
            extreme = index
        if dist(frames[origin], frames[extreme]) >= threshold and dist(frames[extreme], frames[index]) >= threshold:
            outgoing = vector(frames[origin], frames[extreme])
            incoming = vector(frames[extreme], frames[index])
            denom = math.hypot(*outgoing) * math.hypot(*incoming)
            cosine = 0.0 if denom < 1 else (outgoing[0] * incoming[0] + outgoing[1] * incoming[1]) / denom
            if cosine < 0:
                cuts.append(frames[extreme])
            origin = extreme
            extreme = index
    return cuts


def motion_pieces(frames: list[int], centroids: dict[int, tuple[float, float]], fps: float, threshold: float, start_reason: str) -> list[tuple[int, int, str, str]]:
    labels = merge_short_runs(frames, label_motion(frames, centroids, fps), fps)
    runs = []
    for frame in frames:
        state = labels[frame]
        if not runs or runs[-1][0] != state or frame != runs[-1][2] + 1:
            reason = start_reason if not runs else ("stationary_interval" if state == "stationary" else "coherent_movement")
            runs.append([state, frame, frame, reason])
        else:
            runs[-1][2] = frame
    clean = []
    for state, start, end, reason in runs:
        run_frames = [frame for frame in frames if start <= frame <= end]
        if state != "moving":
            clean.append((run_frames[0], run_frames[-1], reason, state))
            continue
        edges = [run_frames[0]]
        for cut in reversal_cuts(run_frames, centroids, threshold):
            if run_frames[0] < cut < run_frames[-1]:
                edges.append(cut)
        edges.append(run_frames[-1])
        for index in range(len(edges) - 1):
            edge = edges[index]
            stop = edges[index + 1]
            if index < len(edges) - 2:
                inside = [frame for frame in run_frames if edge <= frame < stop]
            else:
                inside = [frame for frame in run_frames if edge <= frame <= stop]
            if not inside:
                continue
            clean.append((inside[0], inside[-1], reason if index == 0 else "direction_reversal", "moving"))
    return clean


def segment_geometry(frames: list[int], centroids: dict[int, tuple[float, float]], diagonal: float) -> dict:
    x0, y0 = centroids[frames[0]]
    x1, y1 = centroids[frames[-1]]
    net = math.hypot(x1 - x0, y1 - y0)
    path = 0.0
    previous = centroids[frames[0]]
    for frame in frames[1:]:
        point = centroids[frame]
        path += math.hypot(point[0] - previous[0], point[1] - previous[1])
        previous = point
    return {
        "displacement_px": round(net, 2),
        "path_length_px": round(path, 2),
        "normalized_displacement": round(net / diagonal, 4),
        "dx_px": round(x1 - x0, 2),
        "dy_px": round(y1 - y0, 2),
    }


def build() -> tuple[list[dict], dict]:
    candidates = json.loads((PILOT / "walking_analysis" / "candidates.json").read_text())["candidates"]
    by_video = defaultdict(list)
    for candidate in candidates:
        by_video[candidate["video_id"]].append(candidate)
    segments = []
    noise_steps = []
    noise_windows = []
    for video_id, rows in by_video.items():
        doc = json.loads((PILOT / video_id / "review.json").read_text())
        tracks = load_person_tracks(video_id)
        threshold = displacement_threshold(doc["frame_width"])
        diagonal = math.hypot(doc["frame_width"], doc["frame_height"])
        window = max(1, int(round(SPEED_WINDOW_SEC * doc["fps"])))
        for row in rows:
            track_id = int(row["track_id"])
            centroids = tracks[track_id]["centroids"]
            frames = sorted(centroids)
            for left, right in zip(frames, frames[1:]):
                if right == left + 1:
                    noise_steps.append(math.hypot(centroids[right][0] - centroids[left][0], centroids[right][1] - centroids[left][1]))
            for index, frame in enumerate(frames):
                if index + window >= len(frames):
                    break
                dest = frames[index + window]
                if dest - frame > window * 1.5:
                    continue
                noise_windows.append(
                    math.hypot(centroids[dest][0] - centroids[frame][0], centroids[dest][1] - centroids[frame][1])
                )
            pieces = piece_bounds(frames, doc["fps"])
            pieces = cut_activity(frames, pieces, subject_events(doc, row["track_id"]))
            ordinal = 1
            for start, end, reason in pieces:
                piece_frames = [frame for frame in frames if start <= frame <= end]
                if len(piece_frames) < 1:
                    continue
                for seg_start, seg_end, seg_reason, state in motion_pieces(piece_frames, centroids, doc["fps"], threshold, reason):
                    seg_frames = [frame for frame in piece_frames if seg_start <= frame <= seg_end]
                    if not seg_frames:
                        continue
                    geometry = segment_geometry(seg_frames, centroids, diagonal)
                    segments.append(
                        {
                            "segment_id": f"{video_id}_t{row['track_id']}_s{ordinal:02d}",
                            "video_id": video_id,
                            "video_path": doc["video_path"],
                            "fps": doc["fps"],
                            "frame_width": doc["frame_width"],
                            "frame_height": doc["frame_height"],
                            "source_track_id": row["track_id"],
                            "start_frame": seg_frames[0],
                            "end_frame": seg_frames[-1],
                            "start_time": round(seg_frames[0] / doc["fps"], 4),
                            "end_time": round(seg_frames[-1] / doc["fps"], 4),
                            "duration_sec": round((seg_frames[-1] - seg_frames[0]) / doc["fps"], 4),
                            **geometry,
                            "direction_proposal": direction_8(geometry["dx_px"], geometry["dy_px"], doc["frame_width"]),
                            "motion_state": state,
                            "start_reason": seg_reason,
                            "overlapping_virat_events": overlap_events(doc, row["track_id"], seg_frames[0], seg_frames[-1], set(seg_frames)),
                            "confirmed_walking": False,
                            "confirmed_wandering": False,
                            "human_review": {
                                "is_walking": None,
                                "is_wandering": None,
                                "keep_as_caption_event": None,
                                "direction_override": None,
                                "review_note": "",
                            },
                        }
                    )
                    ordinal += 1
    noise = {
        "smoothing_applied": False,
        "frame_step_px": {
            "n": len(noise_steps),
            "p50": round(percentile(noise_steps, 0.5), 2),
            "p90": round(percentile(noise_steps, 0.9), 2),
            "p99": round(percentile(noise_steps, 0.99), 2),
            "max": round(max(noise_steps), 2),
        },
        "forward_1s_displacement_px": {
            "n": len(noise_windows),
            "p50": round(percentile(noise_windows, 0.5), 2),
            "p90": round(percentile(noise_windows, 0.9), 2),
            "p99": round(percentile(noise_windows, 0.99), 2),
        },
        "meaningful_displacement_px_on_1280_width": 64,
        "note": "Per-frame centroid steps are a few pixels. The meaningful-displacement threshold is 64px, so bbox jitter does not by itself cross a reversal.",
    }
    return segments, noise


def summarize(segments: list[dict]) -> dict:
    by_track = defaultdict(int)
    for segment in segments:
        by_track[(segment["video_id"], segment["source_track_id"])] += 1
    short = [
        {
            "segment_id": segment["segment_id"],
            "duration_sec": segment["duration_sec"],
            "motion_state": segment["motion_state"],
            "start_reason": segment["start_reason"],
            "direction_proposal": segment["direction_proposal"],
            "displacement_px": segment["displacement_px"],
        }
        for segment in segments
        if segment["duration_sec"] < SPEED_WINDOW_SEC
    ]
    def rows(reason: str) -> list[dict]:
        return [
            {
                "segment_id": segment["segment_id"],
                "time": [segment["start_time"], segment["end_time"]],
                "motion_state": segment["motion_state"],
                "direction_proposal": segment["direction_proposal"],
                "displacement_px": segment["displacement_px"],
                "path_length_px": segment["path_length_px"],
            }
            for segment in segments
            if segment["start_reason"] == reason
        ]
    focus = []
    for video_id, track_ids in FOCUS.items():
        for track_id in track_ids:
            focus.append(
                {
                    "video_id": video_id,
                    "source_track_id": track_id,
                    "segments": [segment for segment in segments if segment["video_id"] == video_id and segment["source_track_id"] == track_id],
                }
            )
    return {
        "n_source_tracks": len(by_track),
        "n_segments": len(segments),
        "segments_per_track": {
            "1": sum(1 for count in by_track.values() if count == 1),
            "2": sum(1 for count in by_track.values() if count == 2),
            "3_or_more": sum(1 for count in by_track.values() if count >= 3),
            "max": max(by_track.values()),
        },
        "n_direction_reversal_segments": len(rows("direction_reversal")),
        "n_stationary_segments": len(rows("stationary_interval")),
        "n_activity_boundary_segments": len(rows("activity_boundary")),
        "n_track_gap_segments": len(rows("track_gap")),
        "n_segments_shorter_than_measurement_window": len(short),
        "short_segments": short,
        "direction_reversal_segments": rows("direction_reversal"),
        "stationary_segments": rows("stationary_interval"),
        "activity_boundary_segments": rows("activity_boundary"),
        "focus_tracks": focus,
    }


def draw_banner(frame, segment: dict) -> None:
    height, width = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pen = ImageDraw.Draw(image)
    body = ImageFont.truetype(FONT_PATH, 22 if width >= 1600 else 18)
    direction = DIRECTION_KO.get(segment["direction_proposal"], str(segment["direction_proposal"]))
    state = "이동" if segment["motion_state"] == "moving" else "정지"
    lines = [
        f"{segment['segment_id'].split('_t')[-1]}  {state}  {direction}  시작 이유: {REASON_KO.get(segment['start_reason'], segment['start_reason'])}",
        f"변위 {segment['displacement_px']:.0f}px  경로 {segment['path_length_px']:.0f}px  {segment['start_time']:.1f}-{segment['end_time']:.1f}s",
        "walking으로 확정하지 않은 motion segment 후보",
    ]
    line_h = 28 if width >= 1600 else 24
    pen.rectangle((0, 0, width, 12 + line_h * len(lines)), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        pen.text((12, 6 + index * line_h), line, font=body, fill=(245, 245, 245))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_focus(segments: list[dict], tracks_by_video: dict[str, dict]) -> list[str]:
    written = []
    for video_id, track_ids in FOCUS.items():
        video_segments = [
            segment for segment in segments
            if segment["video_id"] == video_id and segment["source_track_id"] in track_ids
        ]
        if not video_segments:
            continue
        sample = video_segments[0]
        width, height, fps = sample["frame_width"], sample["frame_height"], sample["fps"]
        encoders = []
        for track_id in track_ids:
            rows = [segment for segment in video_segments if segment["source_track_id"] == track_id]
            if not rows:
                continue
            out = OUT / "clips" / f"motion_t{track_id}_{video_id}.mp4"
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
            encoders.append((track_id, rows, encoder, out, rows[0]["start_frame"], rows[-1]["end_frame"]))
        cap = cv2.VideoCapture(sample["video_path"])
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                for track_id, rows, encoder, _out, start, end in encoders:
                    if start <= frame_index <= end:
                        painted = frame.copy()
                        active = next((row for row in rows if row["start_frame"] <= frame_index <= row["end_frame"]), rows[0])
                        box = tracks_by_video[video_id][int(track_id)]["boxes"].get(frame_index)
                        if box is not None:
                            x, y, w, h = box
                            cv2.rectangle(painted, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), (40, 220, 80), 2)
                        draw_banner(painted, active)
                        encoder.stdin.write(painted.tobytes())
                frame_index += 1
                if frame_index > max(item[5] for item in encoders):
                    break
        finally:
            cap.release()
            for _track_id, _rows, encoder, out, _start, _end in encoders:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed for {out}")
                written.append(str(out.relative_to(EVAL_ROOT)))
        print(f"WROTE {video_id} focus_clips={len(encoders)}", flush=True)
    return written


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    segments, noise = build()
    summary = summarize(segments)
    clips = render_focus(segments, {video_id: load_person_tracks(video_id) for video_id in FOCUS})
    by_video = defaultdict(list)
    for segment in segments:
        by_video[segment["video_id"]].append(segment)
    for video_id, rows in by_video.items():
        write_json(OUT / video_id / "motion_segments.json", {"video_id": video_id, "confirmed_walking": False, "n_segments": len(rows), "segments": rows})
    write_json(OUT / "trajectory_noise.json", noise)
    write_json(
        OUT / "procedure.json",
        {
            "finalized_in_annotation_policy": False,
            "smoothing": None,
            "why_no_smoothing": noise["note"],
            "speed_window_sec": SPEED_WINDOW_SEC,
            "stationary_if_past_or_future_window_displacement_px_below": STILL_1S_PX,
            "why_stationary_probe": "1초 변위는 대체로 5px 이하이거나 20px 이상이다. 8px은 그 사이에 둔 pilot probe이며 policy threshold가 아니다.",
            "runs_shorter_than_the_measurement_window_are_merged": True,
            "reversal_rule": "raw centroid leg closes when displacement and the reversal both reach max(20, 0.05 * frame_width) and the leg cosine is negative",
            "activity_boundary": "split where this track is the VIRAT subject and the event start or end falls inside the piece",
            "auto_confirm_walking": False,
            "auto_confirm_wandering": False,
        },
    )
    public_summary = {key: value for key, value in summary.items() if key != "focus_tracks"}
    write_json(OUT / "summary.json", public_summary)
    write_json(OUT / "focus_tracks.json", {"tracks": summary["focus_tracks"]})
    write_json(OUT / "clips.json", {"clips": clips})
    print(
        f"TRACKS {summary['n_source_tracks']} SEGMENTS {summary['n_segments']} "
        f"REVERSAL {summary['n_direction_reversal_segments']} "
        f"STILL {summary['n_stationary_segments']} "
        f"ACTIVITY {summary['n_activity_boundary_segments']} "
        f"SHORT {summary['n_segments_shorter_than_measurement_window']}"
    )


if __name__ == "__main__":
    main()
