#!/usr/bin/env python3
"""Overlay VIRAT object-track GT on the 59-video list.

Draws objects.txt boxes (detection + track id) on each frame.
A track that mapping.txt links to an activity is drawn thicker and
labeled with that activity while the event interval is active.
Type 0 boxes are the large static regions present in some files;
they are drawn thin and gray.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = EVAL_ROOT / "configs" / "virat_multi_roi_59.txt"
ANN_DIR = Path("/data/datasets/VIRAT/annotations")
OUT_DIR = EVAL_ROOT / "experiments" / "virat" / "gt_track_overlay"

ACTIVITY = {
    1: "load",
    2: "unload",
    3: "open_trunk",
    4: "close_trunk",
    5: "get_in",
    6: "get_out",
    7: "gesture",
    8: "dig",
    9: "carry",
    10: "run",
    11: "enter",
    12: "exit",
}
CATEGORY = {
    0: "static",
    1: "person",
    2: "car",
    3: "vehicle",
    4: "object",
    5: "bike",
}
COLOR = {
    0: (170, 170, 170),
    1: (80, 220, 60),
    2: (255, 170, 40),
    3: (220, 220, 40),
    4: (40, 140, 255),
    5: (220, 60, 220),
}


def read_tracks(path: Path) -> tuple[dict[int, list], dict[int, int]]:
    by_frame: dict[int, list] = defaultdict(list)
    types: dict[int, int] = {}
    if not path.is_file():
        return by_frame, types
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj_id, _dur, frame, x, y, w, h, obj_type = map(int, line.split()[:8])
        types[obj_id] = obj_type
        by_frame[frame].append((obj_id, x, y, w, h, obj_type))
    return by_frame, types


def read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        event_id, event_type, _dur, start, end, _n = map(int, tokens[:6])
        bits = [int(bit) for bit in tokens[6:]]
        object_ids = [index for index, bit in enumerate(bits) if bit == 1]
        events.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "name": ACTIVITY.get(event_type, f"type{event_type}"),
                "start": start,
                "end": end,
                "object_ids": set(object_ids),
            }
        )
    return events


def probe(video_path: Path) -> tuple[int, int, float]:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        text=True,
    )
    stream = json.loads(raw)["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    return int(stream["width"]), int(stream["height"]), fps


def draw_frame(frame, boxes, events, frame_index: int, video_id: str) -> None:
    height, width = frame.shape[:2]
    active = [event for event in events if event["start"] <= frame_index <= event["end"]]
    linked: dict[int, list[str]] = defaultdict(list)
    for event in active:
        for obj_id in event["object_ids"]:
            linked[obj_id].append(event["name"])

    for obj_id, x, y, w, h, obj_type in boxes:
        color = COLOR.get(obj_type, (255, 255, 255))
        names = linked.get(obj_id, [])
        thickness = 1 if obj_type == 0 else (3 if names else 2)
        x2 = min(width - 1, x + w)
        y2 = min(height - 1, y + h)
        cv2.rectangle(frame, (x, y), (x2, y2), color, thickness)
        label = f"{CATEGORY.get(obj_type, str(obj_type))}#{obj_id}"
        if names:
            label += " " + ",".join(names)
        cv2.putText(
            frame,
            label,
            (x, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    banner = f"{video_id}  f={frame_index}"
    if active:
        banner += "  EVENT " + " | ".join(
            f"{event['name']}#{event['event_id']}" for event in active
        )
    else:
        banner += "  no event"
    cv2.rectangle(frame, (0, 0), (width, 22), (0, 0, 0), -1)
    cv2.putText(
        frame,
        banner[:180],
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def render(video_path: Path, out_path: Path) -> dict:
    video_id = video_path.stem
    boxes_by_frame, types = read_tracks(ANN_DIR / f"{video_id}.viratdata.objects.txt")
    events = read_events(ANN_DIR / f"{video_id}.viratdata.mapping.txt")
    width, height, fps = probe(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
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
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
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
            draw_frame(frame, boxes_by_frame.get(frame_index, []), events, frame_index, video_id)
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
    finally:
        cap.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {video_id} with code {code}")
    return {
        "video_id": video_id,
        "frames": frame_index,
        "tracks": len(types),
        "events": len(events),
        "output": str(out_path),
    }


def render_job(job: tuple[str, str]) -> dict:
    video_path, out_path = Path(job[0]), Path(job[1])
    return render(video_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-list", type=Path, default=DEFAULT_LIST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    rows = []
    for line in args.video_list.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if args.video_id and path.stem not in args.video_id:
            continue
        rows.append(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(path), str(args.output_dir / f"{path.stem}.mp4")) for path in rows]
    summary = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_job, job): job[0] for job in jobs}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            summary.append(result)
            print(f"[{done}/{len(jobs)}] {result['video_id']} frames={result['frames']}", flush=True)
    summary.sort(key=lambda item: item["video_id"])
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
