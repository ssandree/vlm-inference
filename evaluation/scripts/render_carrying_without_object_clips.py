#!/usr/bin/env python3
"""Clip the 13 carrying candidates that have no mapped object track.

Draws only that event's subject track so the review JSON event id
can be matched to a person on screen. Does not modify VIRAT annotations,
the 59-video GT, or the full-video review.mp4 files.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

EVAL_ROOT = Path(__file__).resolve().parents[1]
PILOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_pilot_6"
ANN_DIR = Path("/data/datasets/VIRAT/annotations")
OUT_DIR = PILOT / "carrying_without_object"
PAD_SEC = 0.5


def carrying_events() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for review_path in sorted(PILOT.glob("VIRAT_*/review.json")):
        doc = json.loads(review_path.read_text())
        for event in doc["candidate_events"]:
            if event["activity_class"] != "person_carrying_object" or event["object"]:
                continue
            grouped[doc["video_id"]].append(
                {
                    "video_id": doc["video_id"],
                    "video_path": doc["video_path"],
                    "fps": doc["fps"],
                    "nb_frames": doc["nb_frames"],
                    "event_id": event["event_id"],
                    "track_id": int(event["subject"]["track_id"]),
                    "start_frame": event["start_frame"],
                    "end_frame": event["end_frame"],
                    "start_time": event["start_time"],
                    "end_time": event["end_time"],
                    "screen_location": event["screen_location"]["value"],
                    "track_motion": event["track_motion"]["value"],
                }
            )
    return grouped


def load_track_boxes(video_id: str, track_id: int) -> dict[int, tuple[int, int, int, int]]:
    boxes = {}
    path = ANN_DIR / f"{video_id}.viratdata.objects.txt"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj_id, _dur, frame, x, y, width, height, _obj_type = map(int, line.split()[:8])
        if obj_id == track_id:
            boxes[frame] = (x, y, width, height)
    return boxes


def draw(frame, box, event: dict, frame_index: int, in_event: bool) -> None:
    height, width = frame.shape[:2]
    label = (
        f"carry #{event['event_id']}  id {event['track_id']}  "
        f"{event['start_time']:.1f}-{event['end_time']:.1f}s  no object track"
    )
    if not in_event:
        label = "padding  " + label
    cv2.rectangle(frame, (0, 0), (width, 32), (0, 0, 0), -1)
    cv2.putText(frame, label[:180], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    if box is None:
        return
    x, y, w, h = box
    x2 = min(width - 1, x + w)
    y2 = min(height - 1, y + h)
    cv2.rectangle(frame, (x, y), (x2, y2), (80, 220, 60), 2)
    cv2.putText(
        frame,
        f"id {event['track_id']} carry #{event['event_id']}",
        (x, max(48, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 220, 60),
        1,
        cv2.LINE_AA,
    )


def open_encoder(width: int, height: int, fps: float, out_path: Path) -> subprocess.Popen:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )


def render_video(video_id: str, events: list[dict]) -> list[dict]:
    video_path = Path(events[0]["video_path"])
    fps = events[0]["fps"]
    nb_frames = events[0]["nb_frames"]
    pad = max(1, int(round(PAD_SEC * fps)))
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            text=True,
        )
    )["streams"][0]
    width, height = int(probe["width"]), int(probe["height"])
    jobs = []
    for event in events:
        clip_start = max(0, event["start_frame"] - pad)
        clip_end = min(nb_frames - 1, event["end_frame"] + pad)
        name = f"{video_id}__event{event['event_id']}_track{event['track_id']}.mp4"
        out_path = OUT_DIR / name
        jobs.append(
            {
                **event,
                "clip_start_frame": clip_start,
                "clip_end_frame": clip_end,
                "output": str(out_path),
                "boxes": load_track_boxes(video_id, event["track_id"]),
                "encoder": open_encoder(width, height, fps, out_path),
            }
        )
    cap = cv2.VideoCapture(str(video_path))
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            for job in jobs:
                if job["clip_start_frame"] <= frame_index <= job["clip_end_frame"]:
                    painted = frame.copy()
                    draw(
                        painted,
                        job["boxes"].get(frame_index),
                        job,
                        frame_index,
                        job["start_frame"] <= frame_index <= job["end_frame"],
                    )
                    job["encoder"].stdin.write(painted.tobytes())
            frame_index += 1
    finally:
        cap.release()
        for job in jobs:
            job["encoder"].stdin.close()
            code = job["encoder"].wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed for {job['output']} with code {code}")
    rows = []
    for job in jobs:
        rows.append(
            {
                "video_id": video_id,
                "event_id": job["event_id"],
                "subject_track_id": str(job["track_id"]),
                "activity_class": "person_carrying_object",
                "object_tracks": [],
                "start_time": job["start_time"],
                "end_time": job["end_time"],
                "start_frame": job["start_frame"],
                "end_frame": job["end_frame"],
                "screen_location": job["screen_location"],
                "track_motion": job["track_motion"],
                "clip": str(Path(job["output"]).relative_to(EVAL_ROOT)),
                "review_json": str((PILOT / video_id / "review.json").relative_to(EVAL_ROOT)),
            }
        )
    print(f"WROTE {video_id} clips={len(rows)} frames_read={frame_index}", flush=True)
    return rows


def main() -> None:
    grouped = carrying_events()
    rows = []
    for video_id, events in grouped.items():
        rows.extend(render_video(video_id, events))
    rows.sort(key=lambda item: (item["video_id"], int(item["event_id"])))
    index = {
        "note": "Carrying candidates whose VIRAT mapping has no object track. The green box is that event's subject person only.",
        "n_clips": len(rows),
        "clips": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(f"INDEX {OUT_DIR / 'index.json'} n={len(rows)}")


if __name__ == "__main__":
    main()
