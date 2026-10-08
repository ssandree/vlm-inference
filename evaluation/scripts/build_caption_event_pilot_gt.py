#!/usr/bin/env python3
"""6-video caption-event GT draft.

VIRAT activities with a policy-defined caption action become caption events.
Walking and wandering stay unconfirmed review items. No reference sentences.
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
SEG_ROOT = PILOT / "walking_analysis" / "motion_segment_pilot"
GT_ROOT = PILOT / "gt"
VIS_ROOT = PILOT / "review_visualization"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MIN_SEGMENT_SEC = 1.0
TEMPORAL_MIN = 0.3
SPATIAL_MAX = 0.15
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
ARROWS = {
    "left_to_right": "→",
    "right_to_left": "←",
    "top_to_bottom": "↓",
    "bottom_to_top": "↑",
    "upper_left_to_lower_right": "↘",
    "upper_right_to_lower_left": "↙",
    "lower_left_to_upper_right": "↗",
    "lower_right_to_upper_left": "↖",
}


def direction_8(dx: float, dy: float, threshold: float) -> str | None:
    if math.hypot(dx, dy) < threshold:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360.0
    return DIRECTIONS[int(math.floor((angle + 22.5) / 45.0)) % 8]


def caption_action(event: dict) -> str | None:
    if event["activity_class"] == "person_carrying_object":
        if event.get("object"):
            return "carrying_object"
        return None
    name = event["activity_class"]
    if name.startswith("person_"):
        return name[len("person_") :]
    return name


def load_boxes(video_id: str) -> dict[str, dict[int, tuple[int, int, int, int]]]:
    boxes: dict[str, dict[int, tuple[int, int, int, int]]] = defaultdict(dict)
    path = ANN_DIR / f"{video_id}.viratdata.objects.txt"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj_id, _dur, frame, x, y, width, height, _obj_type = map(int, line.split()[:8])
        boxes[str(obj_id)][frame] = (x, y, width, height)
    return boxes


def median_screen(boxes: dict[int, tuple[int, int, int, int]], width: int, height: int, start: int, end: int) -> dict | None:
    points = []
    for frame in range(start, end + 1):
        box = boxes.get(frame)
        if box is None:
            continue
        x, y, w, h = box
        points.append(((x + w / 2) / width, (y + h / 2) / height))
    if not points:
        return None
    points.sort()
    nx, ny = points[len(points) // 2]
    col = "left" if nx < 1 / 3 else "center" if nx < 2 / 3 else "right"
    row = "top" if ny < 1 / 3 else "middle" if ny < 2 / 3 else "bottom"
    return {"value": f"{col}_{row}", "source": "derived_from_track", "is_semantic_location": False}


def bron_kerbosch(nodes: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    neighbors = {node: set() for node in nodes}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    cliques: list[list[str]] = []

    def search(current: set[str], candidates: set[str], excluded: set[str]) -> None:
        if not candidates and not excluded:
            if len(current) >= 2:
                cliques.append(sorted(current))
            return
        for node in list(candidates):
            search(current | {node}, candidates & neighbors[node], excluded & neighbors[node])
            candidates.remove(node)
            excluded.add(node)

    search(set(), set(nodes), set())
    return cliques


def build_video(doc: dict, segments: list[dict], wandering_tracks: set[str], boxes: dict[str, dict]) -> dict:
    video_id = doc["video_id"]
    duplicate_ids = {
        event_id
        for group in doc.get("duplicate_candidate_groups", [])
        for event_id in group["member_event_ids"]
    }
    evidence = []
    queue = []

    def add_queue(reason: str, proposal: dict, ae_ids: list[str], track_ids: list[str], start_frame, end_frame, start_time, end_time) -> None:
        queue.append(
            {
                "review_id": f"RQ{len(queue) + 1:03d}",
                "video_id": video_id,
                "atomic_evidence_ids": ae_ids,
                "source_track_ids": track_ids,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "reason": reason,
                "current_proposal": proposal,
                "human_decision": {
                    "accept": True if reason in {"walking_candidate", "wandering_candidate"} else None,
                    "canonical_atomic_evidence_id": None,
                    "is_walking": True if reason == "walking_candidate" else None,
                    "is_wandering": True if reason == "wandering_candidate" else None,
                    "direction_override": None,
                    "semantic_location": None,
                    "note": "Confirmed by human review." if reason in {"walking_candidate", "wandering_candidate"} else "",
                },
            }
        )

    for event in doc["candidate_events"]:
        action = caption_action(event)
        motion = event.get("track_motion") or {}
        direction = None
        if motion.get("dx_px") is not None and motion.get("threshold_px") is not None:
            direction = direction_8(motion["dx_px"], motion["dy_px"], motion["threshold_px"])
        subject_track = (event.get("subject") or {}).get("track_id")
        track_ids = []
        if subject_track is not None:
            track_ids.append(str(subject_track))
        track_ids.extend(str(item["track_id"]) for item in event.get("object") or [])
        ae_id = f"AE{len(evidence) + 1:03d}"
        duplicate = event["event_id"] in duplicate_ids
        eligible = action is not None and not duplicate
        row = {
            "atomic_evidence_id": ae_id,
            "source": {
                "type": "virat_activity",
                "virat_event_ids": [event["event_id"]],
                "source_track_ids": track_ids,
            },
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "subject": {
                "category": (event.get("subject") or {}).get("category"),
                "track_id": None if subject_track is None else str(subject_track),
            },
            "action": action,
            "object": event.get("object") or [],
            "direction": direction,
            "screen_location": event.get("screen_location"),
            "semantic_location": None,
            "caption_evaluable": eligible,
            "activity_class": event["activity_class"],
            "duplicate_candidate": duplicate,
            "confirmed_caption_event": eligible,
            "review": {"human_verified": False, "note": ""},
        }
        evidence.append(row)
        if duplicate:
            continue
        if action is None and event["activity_class"] == "person_carrying_object":
            add_queue(
                "ambiguous_action",
                {"action": None, "reason": "carrying_without_mapped_object_track", "auto_convert_to_walking": False},
                [ae_id],
                track_ids,
                event["start_frame"],
                event["end_frame"],
                event["start_time"],
                event["end_time"],
            )

    walking_by_track = defaultdict(list)
    for segment in segments:
        if segment["video_id"] != video_id:
            continue
        if segment["duration_sec"] < MIN_SEGMENT_SEC:
            continue
        if segment["motion_state"] != "moving" or not segment["direction_proposal"]:
            continue
        ae_id = f"AE{len(evidence) + 1:03d}"
        track_id = str(segment["source_track_id"])
        conflicts = [
            {"event_id": item["event_id"], "activity_class": item["activity_class"]}
            for item in segment.get("overlapping_virat_events", [])
            if item.get("role") == "subject" and item.get("activity_class") == "person_running"
        ]
        row = {
            "atomic_evidence_id": ae_id,
            "source": {
                "type": "person_track_review",
                "virat_event_ids": [
                    item["event_id"]
                    for item in segment.get("overlapping_virat_events", [])
                    if item.get("role") == "subject"
                ],
                "source_track_ids": [track_id],
            },
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "subject": {"category": "person", "track_id": track_id},
            "action": "walking",
            "object": [],
            "direction": segment["direction_proposal"],
            "screen_location": median_screen(boxes.get(track_id, {}), doc["frame_width"], doc["frame_height"], segment["start_frame"], segment["end_frame"]),
            "semantic_location": None,
            "caption_evaluable": True,
            "confirmed_caption_event": True,
            "proposal": {
                "action": "walking",
                "direction": segment["direction_proposal"],
                "confirmed": True,
                "running_conflict": conflicts,
            },
            "review": {
                "human_verified": True,
                "is_walking": True,
                "is_wandering": False,
                "keep_as_caption_event": True,
                "direction_override": None,
                "note": "Confirmed by human review.",
            },
        }
        evidence.append(row)
        walking_by_track[track_id].append(ae_id)
        add_queue(
            "walking_candidate",
            {"action": "walking", "direction": segment["direction_proposal"], "confirmed": True, "running_conflict": conflicts},
            [ae_id],
            [track_id],
            segment["start_frame"],
            segment["end_frame"],
            segment["start_time"],
            segment["end_time"],
        )

    for track_id in sorted(wandering_tracks, key=int):
        related = walking_by_track.get(track_id, [])
        frames = [item for item in evidence if item["atomic_evidence_id"] in related]
        if not frames:
            continue
        ae_id = f"AE{len(evidence) + 1:03d}"
        evidence.append(
            {
                "atomic_evidence_id": ae_id,
                "source": {
                    "type": "person_track_review",
                    "virat_event_ids": [],
                    "source_track_ids": [track_id],
                },
                "start_frame": min(item["start_frame"] for item in frames),
                "end_frame": max(item["end_frame"] for item in frames),
                "start_time": min(item["start_time"] for item in frames),
                "end_time": max(item["end_time"] for item in frames),
                "subject": {"category": "person", "track_id": track_id},
                "action": "wandering",
                "object": [],
                "direction": None,
                "screen_location": None,
                "semantic_location": None,
                "caption_evaluable": True,
                "confirmed_caption_event": True,
                "constituent_walking_atomic_evidence_ids": related,
                "review": {
                    "human_verified": True,
                    "is_walking": False,
                    "is_wandering": True,
                    "keep_as_caption_event": True,
                    "note": "Confirmed by human review. Constituent walking segments are kept.",
                },
            }
        )
        add_queue(
            "wandering_candidate",
            {
                "action": "wandering",
                "direction": None,
                "confirmed": True,
                "constituent_walking_atomic_evidence_ids": related,
            },
            [ae_id],
            [track_id],
            min(item["start_frame"] for item in frames),
            max(item["end_frame"] for item in frames),
            min(item["start_time"] for item in frames),
            max(item["end_time"] for item in frames),
        )

    for group in doc.get("duplicate_candidate_groups", []):
        members = [item for item in evidence if item["source"]["type"] == "virat_activity" and item["source"]["virat_event_ids"][0] in group["member_event_ids"]]
        canonical = max(
            members,
            key=lambda item: (item["end_frame"] - item["start_frame"], -item["start_frame"]),
        )
        for item in members:
            chosen = item["atomic_evidence_id"] == canonical["atomic_evidence_id"]
            item["confirmed_caption_event"] = chosen
            item["caption_evaluable"] = chosen
            item["review"] = {
                "human_verified": True,
                "keep_as_caption_event": chosen,
                "note": (
                    "Human review: same person and same action kept once. Longer interval is the caption event."
                    if chosen
                    else "Not selected as the caption event. Source annotation is retained."
                ),
            }
        add_queue(
            "duplicate_canonical",
            {
                "virat_event_ids": group["member_event_ids"],
                "activity_class": group["activity_class"],
                "subject_track_id": group["subject_track_id"],
                "shorter_coverage": group["shorter_coverage"],
                "canonical_virat_event_id": canonical["source"]["virat_event_ids"][0],
                "canonical_atomic_evidence_id": canonical["atomic_evidence_id"],
                "auto_merge": False,
                "auto_delete": False,
            },
            [item["atomic_evidence_id"] for item in members],
            [str(group["subject_track_id"])],
            min(item["start_frame"] for item in members),
            max(item["end_frame"] for item in members),
            min(item["start_time"] for item in members),
            max(item["end_time"] for item in members),
        )
        queue[-1]["human_decision"] = {
            "accept": True,
            "canonical_atomic_evidence_id": canonical["atomic_evidence_id"],
            "is_walking": None,
            "is_wandering": None,
            "direction_override": None,
            "semantic_location": None,
            "note": "Human review: keep one caption event, the longer interval.",
        }

    eligible = [item for item in evidence if item["confirmed_caption_event"]]
    for item in eligible:
        add_queue(
            "semantic_location",
            {"semantic_location": None, "blocks_caption_event": False},
            [item["atomic_evidence_id"]],
            item["source"]["source_track_ids"],
            item["start_frame"],
            item["end_frame"],
            item["start_time"],
            item["end_time"],
        )

    events = []
    for item in sorted(eligible, key=lambda row: (row["start_frame"], row["atomic_evidence_id"])):
        events.append(
            {
                "caption_event_id": f"CE{len(events) + 1:03d}",
                "event_form": "individual",
                "source_atomic_evidence_ids": [item["atomic_evidence_id"]],
                "source_track_ids": [track for track in item["source"]["source_track_ids"] if track == item["subject"]["track_id"]],
                "constituent_track_ids": list(item["source"]["source_track_ids"]),
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "subject": {"category": item["subject"]["category"], "group": False, "track_id": item["subject"]["track_id"]},
                "action": item["action"],
                "object": item["object"],
                "direction": item["direction"],
                "semantic_location": None,
                "reference_sentence": None,
            }
        )

    diagonal = math.hypot(doc["frame_width"], doc["frame_height"])
    centroids = {
        track_id: {frame: (x + w / 2.0, y + h / 2.0) for frame, (x, y, w, h) in track_boxes.items()}
        for track_id, track_boxes in boxes.items()
    }
    by_id = {item["caption_event_id"]: item for item in events}
    edges = set()
    for left, right in ((a, b) for i, a in enumerate(events) for b in events[i + 1 :]):
        if left["action"] != right["action"]:
            continue
        if left["subject"]["track_id"] is None or left["subject"]["track_id"] == right["subject"]["track_id"]:
            continue
        if left["action"] == "walking" and (not left["direction"] or left["direction"] != right["direction"]):
            continue
        start = max(left["start_frame"], right["start_frame"])
        end = min(left["end_frame"], right["end_frame"])
        intersection = end - start + 1
        if intersection <= 0:
            continue
        shorter = min(left["end_frame"] - left["start_frame"] + 1, right["end_frame"] - right["start_frame"] + 1)
        if intersection / shorter < TEMPORAL_MIN:
            continue
        distances = []
        ca = centroids.get(left["subject"]["track_id"], {})
        cb = centroids.get(right["subject"]["track_id"], {})
        for frame in range(start, end + 1):
            if frame in ca and frame in cb:
                distances.append(math.hypot(ca[frame][0] - cb[frame][0], ca[frame][1] - cb[frame][1]) / diagonal)
        if not distances or sorted(distances)[len(distances) // 2] > SPATIAL_MAX:
            continue
        edges.add(tuple(sorted((left["caption_event_id"], right["caption_event_id"]))))
    groups = bron_kerbosch([item["caption_event_id"] for item in events], edges)
    for clique in groups:
        members = [by_id[event_id] for event_id in clique]
        directions = {item["direction"] for item in members}
        shared_direction = next(iter(directions)) if len(directions) == 1 else None
        start = max(item["start_frame"] for item in members)
        end = min(item["end_frame"] for item in members)
        events.append(
            {
                "caption_event_id": f"CE{len(events) + 1:03d}",
                "event_form": "group",
                "source_atomic_evidence_ids": [ae_id for item in members for ae_id in item["source_atomic_evidence_ids"]],
                "source_track_ids": [item["subject"]["track_id"] for item in members],
                "start_frame": start,
                "end_frame": end,
                "start_time": round(start / doc["fps"], 4),
                "end_time": round(end / doc["fps"], 4),
                "subject": {"category": "person", "group": True},
                "action": members[0]["action"],
                "object": [obj for item in members for obj in item["object"]],
                "direction": shared_direction,
                "semantic_location": None,
                "reference_sentence": None,
                "person_count_is_required_caption_slot": False,
            }
        )
    blocking = [
        item for item in queue
        if item["reason"] != "semantic_location" and item["human_decision"]["accept"] is not True
    ]
    summary = {
        "video_id": video_id,
        "n_virat_source_events": len(doc["candidate_events"]),
        "n_atomic_evidence": len(evidence),
        "n_caption_events": len(events),
        "n_individual_caption_events": sum(1 for item in events if item["event_form"] == "individual"),
        "n_group_caption_events": sum(1 for item in events if item["event_form"] == "group"),
        "n_walking_candidates": sum(1 for item in queue if item["reason"] == "walking_candidate"),
        "n_confirmed_walking": sum(1 for item in evidence if item.get("action") == "walking"),
        "n_wandering_candidates": sum(1 for item in queue if item["reason"] == "wandering_candidate"),
        "n_confirmed_wandering": sum(1 for item in evidence if item.get("action") == "wandering"),
        "n_human_review_pending": len(blocking),
        "n_semantic_location_unset": sum(1 for item in queue if item["reason"] == "semantic_location"),
        "n_duplicate_candidates": sum(1 for item in queue if item["reason"] == "duplicate_canonical"),
        "n_human_added_events": len(doc.get("human_added_events") or []),
        "n_carrying_without_object": sum(1 for item in queue if item["reason"] == "ambiguous_action"),
        "pending_human_review": [
            {
                "review_id": item["review_id"],
                "reason": item["reason"],
                "atomic_evidence_ids": item["atomic_evidence_ids"],
                "source_track_ids": item["source_track_ids"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "current_proposal": item["current_proposal"],
            }
            for item in blocking
        ],
    }
    return {"atomic_evidence": evidence, "caption_events": events, "human_review_queue": queue, "review_summary": summary}


def active(item: dict, frame: int) -> bool:
    return item["start_frame"] <= frame <= item["end_frame"]


def dashed_rectangle(image, x1: int, y1: int, x2: int, y2: int, color, dash: int = 8) -> None:
    edges = [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for xa, ya, xb, yb in edges:
        length = max(abs(xb - xa), abs(yb - ya))
        if length == 0:
            continue
        steps = max(1, length // dash)
        for step in range(0, steps, 2):
            t0 = step / steps
            t1 = min(1.0, (step + 1) / steps)
            cv2.line(
                image,
                (int(xa + (xb - xa) * t0), int(ya + (yb - ya) * t0)),
                (int(xa + (xb - xa) * t1), int(ya + (yb - ya) * t1)),
                color,
                2,
            )


def draw_overlay(frame, frame_index: int, bundle: dict, boxes: dict[str, dict], font, small) -> None:
    height, width = frame.shape[:2]
    events = bundle["caption_events"]
    evidence = {item["atomic_evidence_id"]: item for item in bundle["atomic_evidence"]}
    active_events = [item for item in events if active(item, frame_index)]
    active_candidates = [
        item for item in bundle["atomic_evidence"]
        if item["source"]["type"] == "person_track_review" and not item.get("confirmed_caption_event") and active(item, frame_index)
    ]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pen = ImageDraw.Draw(image)
    lines = ["ACTIVE CAPTION EVENTS", "실선 FINAL    점선 CANDIDATE    빨간 점선 REJECTED"]
    if not active_events:
        lines.append("(없음)")
    for item in active_events:
        arrow = ARROWS.get(item["direction"], "")
        who = "people" if item["subject"]["group"] else "person"
        form = " GROUP" if item["event_form"] == "group" else ""
        tracks = ",".join(item["source_track_ids"])
        source = ""
        if item["event_form"] == "individual":
            ae = evidence.get(item["source_atomic_evidence_ids"][0])
            if ae and ae["source"]["type"] == "virat_activity":
                source = f"  source=VIRAT {ae['source']['virat_event_ids'][0]}"
        lines.append(
            f"{item['caption_event_id']}{form}  {who} | {item['action']} {arrow}  "
            f"t={item['start_time']:.1f}-{item['end_time']:.1f}  tracks={tracks}{source}"
        )
    if active_candidates:
        lines.append("CANDIDATE  미확정")
        for item in active_candidates[:8]:
            arrow = ARROWS.get(item["direction"], "")
            lines.append(f"[CANDIDATE] {item['atomic_evidence_id']} track {item['subject']['track_id']}  walking? {arrow}")
        if len(active_candidates) > 8:
            lines.append(f"... 후보 {len(active_candidates) - 8}개 더")
    line_h = 24 if width >= 1600 else 20
    panel_h = 8 + line_h * len(lines)
    pen.rectangle((0, 0, width, panel_h), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        pen.text((10, 4 + index * line_h), line, font=small, fill=(245, 245, 245))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    labels = []

    def queue_label(track_id: str, text: str, color, dashed: bool) -> None:
        box = boxes.get(track_id, {}).get(frame_index)
        if box is None:
            return
        x, y, w, h = box
        x2, y2 = min(width - 1, x + w), min(height - 1, y + h)
        if dashed:
            dashed_rectangle(frame, x, y, x2, y2, color)
        else:
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        labels.append((track_id, x, y2, text, color))

    for item in active_events:
        if item["event_form"] != "individual":
            continue
        track_id = item["subject"]["track_id"]
        if track_id is None:
            continue
        arrow = ARROWS.get(item["direction"], "")
        action = item["action"] if not arrow else f"{item['action']} {arrow}"
        queue_label(track_id, f"T{track_id} {item['source_atomic_evidence_ids'][0]} {item['caption_event_id']} {action}", (80, 220, 80), False)
    for item in active_events:
        if item["event_form"] != "group":
            continue
        arrow = ARROWS.get(item["direction"], "")
        for track_id in item["source_track_ids"]:
            queue_label(track_id, f"T{track_id} {item['caption_event_id']} GROUP {item['action']} {arrow}".strip(), (40, 180, 255), False)
    for item in active_candidates:
        track_id = item["subject"]["track_id"]
        arrow = ARROWS.get(item["direction"], "")
        queue_label(track_id, f"CANDIDATE T{track_id} {item['atomic_evidence_id']} walking? {arrow}".strip(), (0, 165, 255), True)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    pen = ImageDraw.Draw(image)
    used = defaultdict(int)
    for track_id, x, y2, text, color in labels:
        yy = min(height - 22, max(panel_h + 2, y2 + 2 + used[track_id] * 18))
        used[track_id] += 1
        pen.text((x, yy), text, font=small, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_overlay(doc: dict, bundle: dict, boxes: dict[str, dict]) -> None:
    video_id = doc["video_id"]
    out = VIS_ROOT / video_id / "gt_overlay.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps = doc["frame_width"], doc["frame_height"], doc["fps"]
    font = ImageFont.truetype(FONT_PATH, 22 if width >= 1600 else 18)
    small = ImageFont.truetype(FONT_PATH, 20 if width >= 1600 else 16)
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
    cap = cv2.VideoCapture(doc["video_path"])
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            draw_overlay(frame, frame_index, bundle, boxes, font, small)
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
    finally:
        cap.release()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {out}")
    print(f"WROTE overlay {video_id} frames={frame_index}", flush=True)


def render_timeline(doc: dict, events: list[dict]) -> None:
    video_id = doc["video_id"]
    out = VIS_ROOT / video_id / "event_timeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(doc["duration_sec"], 1.0)
    label_w = 420
    plot_w = 1100
    row_h = 28
    height = 50 + row_h * max(1, len(events)) + 20
    image = Image.new("RGB", (label_w + plot_w + 20, height), (250, 250, 250))
    pen = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 16)
    pen.text((12, 12), video_id, font=font, fill=(20, 20, 20))
    if not events:
        pen.text((12, 40), "Caption Event 없음", font=font, fill=(80, 80, 80))
    for index, item in enumerate(events):
        y = 46 + index * row_h
        arrow = ARROWS.get(item["direction"], "")
        form = "GROUP " if item["event_form"] == "group" else ""
        tracks = ",".join(item["source_track_ids"])
        pen.text((8, y), f"{item['caption_event_id']} {form}{item['action']} {arrow}  t={tracks}"[:52], font=font, fill=(20, 20, 20))
        x1 = label_w + int(plot_w * item["start_time"] / duration)
        x2 = label_w + max(x1 - label_w + 2, int(plot_w * item["end_time"] / duration))
        color = (40, 110, 210) if item["event_form"] == "individual" else (210, 120, 30)
        pen.rectangle((x1, y + 4, x2, y + 20), fill=color)
    image.save(out)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    policy = json.loads((PILOT / "annotation_policy.json").read_text())
    rule = policy["walking"]["walking_temporal_unit"]["minimum_segment_rule"]
    if rule.get("minimum_segment_duration_sec") != 1.0 or rule.get("final_threshold_selected") is not True:
        raise RuntimeError("policy minimum segment duration is not the confirmed 1.0s rule")
    wandering = json.loads((PILOT / "walking_analysis" / "wandering_candidates.json").read_text())
    wandering_by_video = defaultdict(set)
    for item in wandering["candidates"]:
        wandering_by_video[item["video_id"]].add(item["track_id"])
    summaries = []
    for review_path in sorted(PILOT.glob("VIRAT_*/review.json")):
        doc = json.loads(review_path.read_text())
        video_id = doc["video_id"]
        segment_path = SEG_ROOT / video_id / "motion_segments.json"
        segments = json.loads(segment_path.read_text())["segments"] if segment_path.exists() else []
        boxes = load_boxes(video_id)
        bundle = build_video(doc, segments, wandering_by_video[video_id], boxes)
        out = GT_ROOT / video_id
        write_json(out / "atomic_evidence.json", {"video_id": video_id, "atomic_evidence": bundle["atomic_evidence"]})
        write_json(out / "caption_events.json", {"video_id": video_id, "reference_sentence": None, "caption_events": bundle["caption_events"]})
        write_json(out / "human_review_queue.json", {"video_id": video_id, "queue": bundle["human_review_queue"]})
        write_json(out / "review_summary.json", bundle["review_summary"])
        render_timeline(doc, bundle["caption_events"])
        render_overlay(doc, bundle, boxes)
        summaries.append(bundle["review_summary"])
        print(
            video_id,
            "AE", bundle["review_summary"]["n_atomic_evidence"],
            "CE", bundle["review_summary"]["n_caption_events"],
            "GROUP", bundle["review_summary"]["n_group_caption_events"],
            "PENDING", bundle["review_summary"]["n_human_review_pending"],
            flush=True,
        )
    write_json(GT_ROOT / "pilot_summary.json", {"videos": summaries})


if __name__ == "__main__":
    main()
