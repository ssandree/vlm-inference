#!/usr/bin/env python3
"""Build structured dense-event GT from existing VIRAT annotations.

Reads events / objects / mapping files and writes new JSON only.
Does not modify videos, VIRAT annotations, or existing experiment outputs.

By default this script refuses to process the full list. Pass explicit
--video-id values, or --all after the dry-run schema is confirmed.

Time:
  start_time / end_time are seconds of the annotated frames:
  frame_index / video_fps. The annotation itself stores frames, not seconds.
  The numbers embedded in a VIRAT filename are not used as the interval.

Subject role:
  mapping.txt has no subject/object role column. Every official activity
  class is person-centric. A participant is the subject only when the
  mapping contains exactly one type-1 (person) track. Zero or multiple
  type-1 tracks leave subject null and status absent or ambiguous.

Location:
  No semantic place name is read or guessed. Coarse location is the
  median centroid of the subject track inside [start_frame, end_frame],
  binned on a 3x3 grid of the frame.
  Horizontal edges are 1/3 and 2/3 of width: left, center, right.
  Vertical edges are 1/3 and 2/3 of height: top, middle, bottom.
  A coordinate exactly on a boundary falls into the higher bin.

Direction:
  Net displacement from the earliest to the latest subject-track box
  inside the event interval. Image +x is right and +y is down.
  threshold_px = max(MIN_DISP_PX, MIN_DISP_FRAC * frame_width)
  MIN_DISP_PX = 20
  MIN_DISP_FRAC = 0.05
  A cardinal label is emitted only when displacement >= threshold_px
  and |primary axis| >= AXIS_DOMINANCE * |secondary axis|.
  AXIS_DOMINANCE = 2.0
  Below threshold -> value null, reason below_displacement_threshold.
  Both axes large but neither dominates -> diagonal_<horizontal>_<vertical>.
  Fewer than two boxed frames -> value null, reason insufficient_track_points.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_LIST = EVAL_ROOT / "configs" / "virat_multi_roi_59.txt"
DEFAULT_ANNOTATION_DIR = Path("/data/datasets/VIRAT/annotations")
DEFAULT_OUTPUT_DIR = EVAL_ROOT / "experiments" / "virat" / "dense_event_gt"

# Official VIRAT Release 2.0 activity ids. Unknown ids are not renamed into
# a nearby class.
ACTIVITY_CLASS = {
    1: "person_loading_object_to_vehicle",
    2: "person_unloading_object_from_vehicle",
    3: "person_opening_vehicle_trunk",
    4: "person_closing_vehicle_trunk",
    5: "person_getting_into_vehicle",
    6: "person_getting_out_of_vehicle",
    7: "person_gesturing",
    8: "person_digging",
    9: "person_carrying_object",
    10: "person_running",
    11: "person_entering_facility",
    12: "person_exiting_facility",
}

# Official moving-object type ids. Type 0 appears in some object files as a
# static full-video box and is never given a place name here.
OBJECT_CATEGORY = {
    1: "person",
    2: "car",
    3: "vehicle",
    4: "object",
    5: "bike",
}
PERSON_TYPE_ID = 1

MIN_DISP_PX = 20.0
MIN_DISP_FRAC = 0.05
AXIS_DOMINANCE = 2.0
GRID_EDGES = (1.0 / 3.0, 2.0 / 3.0)

DERIVATION = {
    "time": "start_time = start_frame / fps; end_time = end_frame / fps",
    "subject_role_rule": "unique_type1_participant_of_person_centric_activity",
    "location_method": "median_centroid_3x3",
    "location_edges": {"x": list(GRID_EDGES), "y": list(GRID_EDGES)},
    "location_boundary": "value exactly on 1/3 or 2/3 falls in the higher bin",
    "direction_method": "endpoint_displacement",
    "direction_axes": "+x right, +y down",
    "MIN_DISP_PX": MIN_DISP_PX,
    "MIN_DISP_FRAC": MIN_DISP_FRAC,
    "AXIS_DOMINANCE": AXIS_DOMINANCE,
    "reference_sentence": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-list", type=Path, default=DEFAULT_VIDEO_LIST)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Video id to convert. Repeat for more than one. Required unless --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every video in --video-list. Omit this until the schema is confirmed.",
    )
    return parser.parse_args()


def load_video_list(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        video_path = Path(line)
        rows.append((video_path.stem, video_path))
    return rows


def probe_video(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    payload = json.loads(subprocess.check_output(cmd, text=True))
    stream = payload["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    nb_frames = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": int(nb_frames) if nb_frames not in (None, "N/A") else None,
        "duration_sec": float(stream["duration"]) if stream.get("duration") else None,
    }


def annotation_paths(annotation_dir: Path, video_id: str) -> dict[str, Path]:
    return {
        kind: annotation_dir / f"{video_id}.viratdata.{kind}.txt"
        for kind in ("events", "objects", "mapping")
    }


def read_objects(path: Path) -> dict[int, dict[str, Any]]:
    tracks: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj_id, duration, frame, x, y, width, height, obj_type = map(int, line.split()[:8])
        track = tracks.get(obj_id)
        if track is None:
            track = {
                "object_type_id": obj_type,
                "annotated_duration": duration,
                "boxes": {},
            }
            tracks[obj_id] = track
        track["boxes"][frame] = (x, y, width, height)
    return tracks


def read_mapping(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        event_id, event_type, duration, start_frame, end_frame, n_objects = map(int, tokens[:6])
        bits = [int(bit) for bit in tokens[6:]]
        object_ids = [index for index, bit in enumerate(bits) if bit == 1]
        events.append(
            {
                "event_id": event_id,
                "activity_type_id": event_type,
                "annotated_duration_frames": duration,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "mapping_num_objects": n_objects,
                "object_ids": object_ids,
                "bit_sum_matches_count": sum(bits) == n_objects,
            }
        )
    return events


def read_event_regions(path: Path) -> dict[int, list[tuple[int, int, int, int, int]]]:
    """Per-frame event boxes from events.txt. Not a semantic location."""
    regions: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        tokens = list(map(int, line.split()[:10]))
        event_id, _event_type, _duration, _start, _end, frame, x, y, width, height = tokens
        regions[event_id].append((frame, x, y, width, height))
    return regions


def category_for(object_type_id: int) -> str:
    return OBJECT_CATEGORY.get(object_type_id, f"unknown_type_{object_type_id}")


def bin3(norm: float) -> int:
    if norm < GRID_EDGES[0]:
        return 0
    if norm < GRID_EDGES[1]:
        return 1
    return 2


def coarse_location(
    boxes: list[tuple[int, int, int, int, int]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    if not boxes:
        return {
            "source": None,
            "value": None,
            "reason": "no_track_boxes_in_interval",
        }
    xs = [(x + width / 2.0) / frame_width for _frame, x, _y, width, _height in boxes]
    ys = [(y + height / 2.0) / frame_height for _frame, _x, y, _width, height in boxes]
    x_norm = statistics.median(xs)
    y_norm = statistics.median(ys)
    horiz = ("left", "center", "right")[bin3(x_norm)]
    vert = ("top", "middle", "bottom")[bin3(y_norm)]
    return {
        "source": "derived_from_track",
        "value": f"{horiz}_{vert}",
        "method": "median_centroid_3x3",
        "median_centroid_norm": [round(x_norm, 4), round(y_norm, 4)],
        "n_boxes": len(boxes),
    }


def direction_from_track(
    boxes: list[tuple[int, int, int, int, int]],
    frame_width: int,
) -> dict[str, Any]:
    ordered = sorted(boxes, key=lambda item: item[0])
    if len({frame for frame, *_rest in ordered}) < 2:
        return {
            "source": "derived_from_track" if ordered else None,
            "value": None,
            "method": "endpoint_displacement",
            "reason": "insufficient_track_points",
            "n_boxes": len(ordered),
        }
    _f0, x0, y0, w0, h0 = ordered[0]
    _f1, x1, y1, w1, h1 = ordered[-1]
    cx0, cy0 = x0 + w0 / 2.0, y0 + h0 / 2.0
    cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
    dx = cx1 - cx0
    dy = cy1 - cy0
    disp = math.hypot(dx, dy)
    threshold = max(MIN_DISP_PX, MIN_DISP_FRAC * frame_width)
    details = {
        "source": "derived_from_track",
        "method": "endpoint_displacement",
        "dx_px": round(dx, 2),
        "dy_px": round(dy, 2),
        "displacement_px": round(disp, 2),
        "threshold_px": round(threshold, 2),
        "first_frame": ordered[0][0],
        "last_frame": ordered[-1][0],
        "n_boxes": len(ordered),
    }
    if disp < threshold:
        details["value"] = None
        details["reason"] = "below_displacement_threshold"
        return details
    horiz = "left_to_right" if dx > 0 else "right_to_left"
    vert = "down" if dy > 0 else "up"
    if abs(dx) >= AXIS_DOMINANCE * abs(dy):
        details["value"] = horiz
    elif abs(dy) >= AXIS_DOMINANCE * abs(dx):
        details["value"] = vert
    else:
        details["value"] = f"diagonal_{horiz}_{vert}"
        details["reason"] = "no_dominant_axis"
    return details


def boxes_in_interval(
    track: dict[str, Any] | None,
    start_frame: int,
    end_frame: int,
) -> list[tuple[int, int, int, int, int]]:
    if track is None:
        return []
    return [
        (frame, x, y, width, height)
        for frame, (x, y, width, height) in track["boxes"].items()
        if start_frame <= frame <= end_frame
    ]


def assign_roles(
    object_ids: list[int],
    tracks: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    participants = []
    for object_id in object_ids:
        track = tracks.get(object_id)
        if track is None:
            participants.append(
                {
                    "track_id": str(object_id),
                    "category": None,
                    "object_type_id": None,
                    "source": "virat_mapping",
                    "missing_from_objects_file": True,
                }
            )
            continue
        participants.append(
            {
                "track_id": str(object_id),
                "category": category_for(track["object_type_id"]),
                "object_type_id": track["object_type_id"],
                "source": "virat_mapping",
                "missing_from_objects_file": False,
            }
        )
    persons = [
        item
        for item in participants
        if item["object_type_id"] == PERSON_TYPE_ID and not item["missing_from_objects_file"]
    ]
    if len(persons) == 1:
        subject = {
            "track_id": persons[0]["track_id"],
            "category": "person",
            "object_type_id": PERSON_TYPE_ID,
            "source": "virat_mapping",
            "status": "unique_person",
            "role_rule": DERIVATION["subject_role_rule"],
        }
        objects = [
            {
                "track_id": item["track_id"],
                "category": item["category"],
                "object_type_id": item["object_type_id"],
                "source": "virat_mapping",
            }
            for item in participants
            if item["track_id"] != subject["track_id"]
        ]
    elif len(persons) == 0:
        subject = {
            "track_id": None,
            "category": None,
            "object_type_id": None,
            "source": "virat_mapping",
            "status": "absent",
            "role_rule": "no_type1_participant",
            "candidate_track_ids": [],
        }
        objects = []
    else:
        subject = {
            "track_id": None,
            "category": None,
            "object_type_id": None,
            "source": "virat_mapping",
            "status": "ambiguous",
            "role_rule": "multiple_type1_participants",
            "candidate_track_ids": [item["track_id"] for item in persons],
        }
        objects = []
    return subject, objects, participants


def build_event(
    video_id: str,
    mapped: dict[str, Any],
    tracks: dict[int, dict[str, Any]],
    frame_width: int,
    frame_height: int,
    fps: float,
) -> dict[str, Any]:
    start_frame = mapped["start_frame"]
    end_frame = mapped["end_frame"]
    subject, objects, participants = assign_roles(mapped["object_ids"], tracks)
    interval_len = end_frame - start_frame + 1
    track_records = []
    for item in participants:
        object_id = int(item["track_id"])
        boxed = boxes_in_interval(tracks.get(object_id), start_frame, end_frame)
        if subject["status"] == "unique_person" and item["track_id"] == subject["track_id"]:
            relation = "subject"
        elif subject["status"] == "unique_person":
            relation = "object"
        else:
            relation = "unassigned"
        track_records.append(
            {
                **item,
                "relation": relation,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_boxes_in_interval": len(boxed),
                "interval_coverage": round(len(boxed) / interval_len, 4) if interval_len else None,
                "location": coarse_location(boxed, frame_width, frame_height),
                "direction": direction_from_track(boxed, frame_width),
            }
        )

    if subject["status"] == "unique_person":
        subject_track = next(record for record in track_records if record["relation"] == "subject")
        location = {
            "source": subject_track["location"]["source"],
            "value": subject_track["location"]["value"],
            "track_id": subject["track_id"],
            "method": subject_track["location"].get("method"),
            "median_centroid_norm": subject_track["location"].get("median_centroid_norm"),
            "n_boxes": subject_track["location"].get("n_boxes"),
            "reason": subject_track["location"].get("reason"),
        }
        direction = {
            "source": subject_track["direction"]["source"],
            "value": subject_track["direction"]["value"],
            "track_id": subject["track_id"],
            **{
                key: subject_track["direction"].get(key)
                for key in (
                    "method",
                    "dx_px",
                    "dy_px",
                    "displacement_px",
                    "threshold_px",
                    "first_frame",
                    "last_frame",
                    "n_boxes",
                    "reason",
                )
            },
        }
    else:
        location = {
            "source": None,
            "value": None,
            "reason": f"subject_status_{subject['status']}",
        }
        direction = {
            "source": None,
            "value": None,
            "reason": f"subject_status_{subject['status']}",
        }

    activity_id = mapped["activity_type_id"]
    return {
        "video_id": video_id,
        "event_id": str(mapped["event_id"]),
        "activity_class": ACTIVITY_CLASS.get(activity_id, f"unknown_event_type_{activity_id}"),
        "activity_type_id": activity_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time": round(start_frame / fps, 4),
        "end_time": round(end_frame / fps, 4),
        "time_unit": "seconds",
        "temporal_source": "virat_annotation",
        "annotated_duration_frames": mapped["annotated_duration_frames"],
        "duration_matches_inclusive_span": mapped["annotated_duration_frames"] == interval_len,
        "mapping_num_objects": mapped["mapping_num_objects"],
        "mapping_bit_sum_matches_count": mapped["bit_sum_matches_count"],
        "subject": subject,
        "object": objects,
        "tracks": track_records,
        "location": location,
        "direction": direction,
        "source": "virat_annotation",
        "reference_sentence": None,
    }


def convert_video(
    video_id: str,
    video_path: Path,
    annotation_dir: Path,
) -> dict[str, Any]:
    paths = annotation_paths(annotation_dir, video_id)
    missing = [kind for kind, path in paths.items() if not path.is_file()]
    if missing:
        details = {
            kind: ("directory" if path.is_dir() else "missing")
            for kind, path in paths.items()
        }
        raise FileNotFoundError(
            f"{video_id}: annotation files unavailable ({details})"
        )
    probe = probe_video(video_path)
    tracks = read_objects(paths["objects"])
    mapped = read_mapping(paths["mapping"])
    # events.txt is parsed only to confirm the file is readable. Its boxes are
    # event-region geometry, not a semantic location, so they are not copied
    # into location.value.
    event_regions = read_event_regions(paths["events"])
    events = [
        build_event(
            video_id,
            item,
            tracks,
            probe["width"],
            probe["height"],
            probe["fps"],
        )
        for item in mapped
    ]
    for event in events:
        region = event_regions.get(int(event["event_id"]), [])
        event["event_region_n_frames"] = len(region)
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "annotation_status": "available",
        "manual_annotation_target": False,
        "fps": probe["fps"],
        "frame_width": probe["width"],
        "frame_height": probe["height"],
        "nb_frames": probe["nb_frames"],
        "annotation_files": {kind: str(path) for kind, path in paths.items()},
        "derivation": DERIVATION,
        "n_events": len(events),
        "events": events,
    }


def annotation_file_state(annotation_dir: Path, video_id: str) -> dict[str, str]:
    state = {}
    for kind, path in annotation_paths(annotation_dir, video_id).items():
        if path.is_file():
            state[kind] = "file"
        elif path.is_dir():
            state[kind] = "empty_directory" if not any(path.iterdir()) else "directory"
        else:
            state[kind] = "missing"
    return state


def human_template(video_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "annotation_events": events,
        "human_added_events": [],
        "notes": "",
    }


def event_validation_issues(event: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    base = {"video_id": event["video_id"], "event_id": event["event_id"]}
    if not event["duration_matches_inclusive_span"]:
        issues.append({**base, "check": "duration_mismatch"})
    if not event["mapping_bit_sum_matches_count"]:
        issues.append({**base, "check": "mapping_bit_sum_mismatch"})
    if str(event["activity_class"]).startswith("unknown_event_type_"):
        issues.append({**base, "check": "unknown_activity_type", "activity_type_id": event["activity_type_id"]})
    if event["event_region_n_frames"] != event["annotated_duration_frames"]:
        issues.append(
            {
                **base,
                "check": "event_region_frame_count_mismatch",
                "event_region_n_frames": event["event_region_n_frames"],
                "annotated_duration_frames": event["annotated_duration_frames"],
            }
        )
    for track in event["tracks"]:
        if track.get("missing_from_objects_file"):
            issues.append({**base, "check": "track_missing_from_objects", "track_id": track["track_id"]})
        category = track.get("category") or ""
        if str(category).startswith("unknown_type_"):
            issues.append(
                {
                    **base,
                    "check": "unknown_object_type",
                    "track_id": track["track_id"],
                    "object_type_id": track.get("object_type_id"),
                }
            )
    return issues


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    listed = load_video_list(args.video_list)
    by_id = {video_id: video_path for video_id, video_path in listed}
    if args.all:
        selected = listed
    elif args.video_id:
        unknown = [video_id for video_id in args.video_id if video_id not in by_id]
        if unknown:
            raise SystemExit(f"video id not in {args.video_list}: {unknown}")
        selected = [(video_id, by_id[video_id]) for video_id in args.video_id]
    else:
        raise SystemExit("Pass --video-id (repeatable) or --all. Full-list conversion is not the default.")

    manifest_rows = []
    parsing_errors = []
    validation_issues = []
    derived_events: list[dict[str, Any]] = []
    for video_id, video_path in selected:
        template_path = args.output_dir / f"{video_id}.human_template.json"
        file_state = annotation_file_state(args.annotation_dir, video_id)
        if any(state != "file" for state in file_state.values()):
            write_json(template_path, human_template(video_id, []))
            manifest_rows.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "annotation_status": "missing",
                    "manual_annotation_target": True,
                    "annotation_files": file_state,
                    "n_events": 0,
                    "events_path": None,
                    "human_template_path": str(template_path),
                }
            )
            print(f"MISSING {video_id}")
            continue
        try:
            doc = convert_video(video_id, video_path, args.annotation_dir)
        except Exception as exc:
            write_json(template_path, human_template(video_id, []))
            parsing_errors.append(
                {
                    "video_id": video_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            manifest_rows.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "annotation_status": "parse_error",
                    "manual_annotation_target": True,
                    "annotation_files": file_state,
                    "n_events": 0,
                    "events_path": None,
                    "human_template_path": str(template_path),
                }
            )
            print(f"PARSE_ERROR {video_id}: {exc}")
            continue
        events_path = args.output_dir / f"{video_id}.events.json"
        write_json(events_path, doc)
        write_json(template_path, human_template(video_id, doc["events"]))
        for event in doc["events"]:
            validation_issues.extend(event_validation_issues(event))
        derived_events.extend(doc["events"])
        manifest_rows.append(
            {
                "video_id": video_id,
                "video_path": str(video_path),
                "annotation_status": "available",
                "manual_annotation_target": False,
                "n_events": doc["n_events"],
                "fps": doc["fps"],
                "frame_size": [doc["frame_width"], doc["frame_height"]],
                "events_path": str(events_path),
                "human_template_path": str(template_path),
            }
        )
        print(f"WROTE {video_id} events={doc['n_events']}")

    n_available = sum(row["annotation_status"] == "available" for row in manifest_rows)
    n_missing = sum(row["annotation_status"] == "missing" for row in manifest_rows)
    n_parse_error = sum(row["annotation_status"] == "parse_error" for row in manifest_rows)
    n_events = len(derived_events)
    activity_counts = Counter(event["activity_class"] for event in derived_events)
    subject_status = Counter(event["subject"]["status"] for event in derived_events)
    n_subject = sum(event["subject"]["track_id"] is not None for event in derived_events)
    n_object = sum(len(event["object"]) > 0 for event in derived_events)
    n_location = sum(event["location"]["value"] is not None for event in derived_events)
    n_direction = sum(event["direction"]["value"] is not None for event in derived_events)
    reference_non_null = sum(event["reference_sentence"] is not None for event in derived_events)
    validation_counts = Counter(issue["check"] for issue in validation_issues)

    def rate(count: int) -> float | None:
        if n_events == 0:
            return None
        return round(count / n_events, 4)

    manifest = {
        "video_list": str(args.video_list),
        "n_videos": len(manifest_rows),
        "n_annotation_available": n_available,
        "n_annotation_missing": n_missing,
        "n_parse_error": n_parse_error,
        "n_annotation_derived_events": n_events,
        "videos": manifest_rows,
    }
    summary = {
        "video_list": str(args.video_list),
        "n_videos": len(manifest_rows),
        "n_annotation_available": n_available,
        "n_annotation_missing": n_missing,
        "n_parse_error": n_parse_error,
        "n_annotation_derived_events": n_events,
        "activity_class_counts": dict(activity_counts),
        "subject_status_counts": dict(subject_status),
        "subject_present_rate": rate(n_subject),
        "object_nonempty_rate": rate(n_object),
        "location_non_null_rate": rate(n_location),
        "direction_non_null_rate": rate(n_direction),
        "n_subject_present": n_subject,
        "n_object_nonempty": n_object,
        "n_location_non_null": n_location,
        "n_direction_non_null": n_direction,
        "n_reference_sentence_non_null": reference_non_null,
        "parsing_errors": parsing_errors,
        "validation_issue_counts": dict(validation_counts),
        "validation_issues": validation_issues,
        "events_per_video": [
            {"video_id": row["video_id"], "annotation_status": row["annotation_status"], "n_events": row["n_events"]}
            for row in manifest_rows
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    summary_path = args.output_dir / "conversion_summary.json"
    write_json(manifest_path, manifest)
    write_json(summary_path, summary)
    print(
        f"MANIFEST {manifest_path} videos={len(manifest_rows)} "
        f"available={n_available} missing={n_missing} events={n_events}"
    )
    print(f"SUMMARY {summary_path}")


if __name__ == "__main__":
    main()
