#!/usr/bin/env python3
"""Caption-event GT for 191 VIRAT videos.

132 single-ROI videos with area < 0.90, plus 59 multi-ROI videos.
Writes atomic evidence and caption events. One review summary for the set.
Does not render overlays or timelines, and does not write a per-video queue.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import build_caption_event_pilot_gt as pilot
import build_virat_dense_event_gt as dense
import build_walking_motion_segment_pilot as segments
import analyze_virat_walking_pilot as walking

EVAL_ROOT = Path(__file__).resolve().parents[1]
OUT = EVAL_ROOT / "experiments" / "virat" / "caption_event_gt_191"
POLICY_SRC = EVAL_ROOT / "experiments" / "virat" / "caption_event_pilot_6" / "annotation_policy.json"
ANN_DIR = Path("/data/datasets/VIRAT/annotations")
LISTS = {
    "single_roi_area_lt_0.90": EVAL_ROOT / "configs" / "virat_single_roi_area_lt0.90_132.txt",
    "multi_roi": EVAL_ROOT / "configs" / "virat_multi_roi_59.txt",
}
# These six were confirmed by human review in the pilot. Other videos stay pending.
CONFIRMED_VIDEOS = {
    "VIRAT_S_010001_02_000195_000498",
    "VIRAT_S_010003_03_000219_000259",
    "VIRAT_S_010106_02_000656_000692",
    "VIRAT_S_010107_00_000019_000057",
    "VIRAT_S_010111_08_000920_000954",
    "VIRAT_S_010207_10_001549_001596",
}
MIN_SEGMENT_SEC = 1.0
TEMPORAL_MIN = 0.3
SPATIAL_MAX = 0.15


def load_lists() -> list[dict]:
    rows = []
    seen = set()
    for roi_set, path in LISTS.items():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            video_path = Path(line)
            video_id = video_path.stem
            if video_id in seen:
                continue
            seen.add(video_id)
            rows.append({"video_id": video_id, "video_path": video_path, "roi_set": roi_set})
    return rows


def duplicate_groups(events: list[dict]) -> list[list[dict]]:
    buckets = defaultdict(list)
    for event in events:
        track_id = (event.get("subject") or {}).get("track_id")
        if track_id:
            buckets[(track_id, event["activity_class"])].append(event)
    groups = []
    for rows in buckets.values():
        parent = {event["event_id"]: event["event_id"] for event in rows}

        def find(event_id: str) -> str:
            while parent[event_id] != event_id:
                parent[event_id] = parent[parent[event_id]]
                event_id = parent[event_id]
            return event_id

        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                start = max(left["start_frame"], right["start_frame"])
                end = min(left["end_frame"], right["end_frame"])
                intersection = end - start + 1
                if intersection <= 0:
                    continue
                shorter = min(
                    left["end_frame"] - left["start_frame"] + 1,
                    right["end_frame"] - right["start_frame"] + 1,
                )
                if shorter and intersection / shorter >= 0.8:
                    parent[find(right["event_id"])] = find(left["event_id"])
        clusters = defaultdict(list)
        for event in rows:
            clusters[find(event["event_id"])].append(event)
        groups.extend(cluster for cluster in clusters.values() if len(cluster) >= 2)
    return groups


def motion_segments(doc: dict, tracks: dict[int, dict]) -> tuple[list[dict], set[str]]:
    threshold = segments.displacement_threshold(doc["frame_width"])
    diagonal = math.hypot(doc["frame_width"], doc["frame_height"])
    doc["candidate_events"] = doc["events"]
    produced = []
    wandering = set()
    for track_id, track in tracks.items():
        centroids = track["centroids"]
        frames = sorted(centroids)
        if len(frames) < 2:
            continue
        x0, y0 = centroids[frames[0]]
        x1, y1 = centroids[frames[-1]]
        net = math.hypot(x1 - x0, y1 - y0)
        radius = max(math.hypot(centroids[frame][0] - x0, centroids[frame][1] - y0) for frame in frames)
        if max(net, radius) < threshold:
            continue
        legs = walking.significant_direction_legs(centroids, doc["frame_width"], doc["fps"])
        if walking.motion_pattern(legs) == "wandering":
            wandering.add(str(track_id))
        pieces = segments.piece_bounds(frames, doc["fps"])
        pieces = segments.cut_activity(frames, pieces, segments.subject_events(doc, str(track_id)))
        ordinal = 1
        for start, end, reason in pieces:
            piece_frames = [frame for frame in frames if start <= frame <= end]
            if not piece_frames:
                continue
            for seg_start, seg_end, seg_reason, state in segments.motion_pieces(
                piece_frames, centroids, doc["fps"], threshold, reason
            ):
                seg_frames = [frame for frame in piece_frames if seg_start <= frame <= seg_end]
                if not seg_frames:
                    continue
                geometry = segments.segment_geometry(seg_frames, centroids, diagonal)
                produced.append(
                    {
                        "video_id": doc["video_id"],
                        "source_track_id": str(track_id),
                        "start_frame": seg_frames[0],
                        "end_frame": seg_frames[-1],
                        "start_time": round(seg_frames[0] / doc["fps"], 4),
                        "end_time": round(seg_frames[-1] / doc["fps"], 4),
                        "duration_sec": round((seg_frames[-1] - seg_frames[0]) / doc["fps"], 4),
                        "direction_proposal": segments.direction_8(geometry["dx_px"], geometry["dy_px"], doc["frame_width"]),
                        "motion_state": state,
                        "overlapping_virat_events": segments.overlap_events(
                            doc, str(track_id), seg_frames[0], seg_frames[-1], set(seg_frames)
                        ),
                        "boxes": track["boxes"],
                    }
                )
                ordinal += 1
    return produced, wandering


def build_video(doc: dict, tracks: dict[int, dict]) -> dict:
    video_id = doc["video_id"]
    confirmed_video = video_id in CONFIRMED_VIDEOS
    groups = duplicate_groups(doc["events"])
    duplicate_ids = {event["event_id"] for group in groups for event in group}
    canonical_ids = set()
    resolved = []
    for group in groups:
        canonical = max(group, key=lambda event: (event["end_frame"] - event["start_frame"], -event["start_frame"]))
        canonical_ids.add(canonical["event_id"])
        resolved.append(
            {
                "video_id": video_id,
                "activity_class": canonical["activity_class"],
                "subject_track_id": canonical["subject"]["track_id"],
                "kept_virat_event_id": canonical["event_id"],
                "kept_time": [canonical["start_time"], canonical["end_time"]],
                "dropped_virat_event_ids": [event["event_id"] for event in group if event["event_id"] != canonical["event_id"]],
            }
        )

    evidence = []
    pending = []
    for event in doc["events"]:
        action = pilot.caption_action(event)
        motion = event.get("direction") or {}
        direction = None
        if motion.get("dx_px") is not None and motion.get("threshold_px") is not None:
            direction = pilot.direction_8(motion["dx_px"], motion["dy_px"], motion["threshold_px"])
        subject_track = (event.get("subject") or {}).get("track_id")
        track_ids = []
        if subject_track is not None:
            track_ids.append(str(subject_track))
        track_ids.extend(str(item["track_id"]) for item in event.get("object") or [])
        duplicate = event["event_id"] in duplicate_ids
        eligible = action is not None and (not duplicate or event["event_id"] in canonical_ids)
        ae_id = f"AE{len(evidence) + 1:03d}"
        evidence.append(
            {
                "atomic_evidence_id": ae_id,
                "source": {"type": "virat_activity", "virat_event_ids": [event["event_id"]], "source_track_ids": track_ids},
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "subject": {"category": (event.get("subject") or {}).get("category"), "track_id": None if subject_track is None else str(subject_track)},
                "action": action,
                "object": event.get("object") or [],
                "direction": direction,
                "screen_location": event.get("location"),
                "semantic_location": None,
                "caption_evaluable": eligible,
                "activity_class": event["activity_class"],
                "duplicate_candidate": duplicate,
                "confirmed_caption_event": eligible,
                "review": {
                    "human_verified": duplicate,
                    "keep_as_caption_event": eligible if duplicate else None,
                    "note": "Human review: keep the longer interval." if duplicate and eligible else ("Not selected. Source annotation retained." if duplicate else ""),
                },
            }
        )
        if action is None and event["activity_class"] == "person_carrying_object":
            pending.append(
                {
                    "video_id": video_id,
                    "reason": "ambiguous_action",
                    "atomic_evidence_id": ae_id,
                    "source_track_ids": track_ids,
                    "start_time": event["start_time"],
                    "end_time": event["end_time"],
                    "current_proposal": {"action": None, "reason": "carrying_without_mapped_object_track"},
                }
            )

    produced, wandering_tracks = motion_segments(doc, tracks)
    walking_by_track = defaultdict(list)
    for segment in produced:
        if segment["duration_sec"] < MIN_SEGMENT_SEC or segment["motion_state"] != "moving" or not segment["direction_proposal"]:
            continue
        track_id = segment["source_track_id"]
        conflicts = [
            {"event_id": item["event_id"], "activity_class": item["activity_class"]}
            for item in segment["overlapping_virat_events"]
            if item.get("role") == "subject" and item.get("activity_class") == "person_running"
        ]
        ae_id = f"AE{len(evidence) + 1:03d}"
        confirmed = confirmed_video
        evidence.append(
            {
                "atomic_evidence_id": ae_id,
                "source": {
                    "type": "person_track_review",
                    "virat_event_ids": [item["event_id"] for item in segment["overlapping_virat_events"] if item.get("role") == "subject"],
                    "source_track_ids": [track_id],
                },
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "subject": {"category": "person", "track_id": track_id},
                "action": "walking" if confirmed else None,
                "object": [],
                "direction": segment["direction_proposal"],
                "screen_location": pilot.median_screen(
                    segment["boxes"], doc["frame_width"], doc["frame_height"], segment["start_frame"], segment["end_frame"]
                ),
                "semantic_location": None,
                "caption_evaluable": confirmed,
                "confirmed_caption_event": confirmed,
                "proposal": {"action": "walking", "direction": segment["direction_proposal"], "confirmed": confirmed, "running_conflict": conflicts},
                "review": {
                    "human_verified": confirmed,
                    "is_walking": True if confirmed else None,
                    "keep_as_caption_event": True if confirmed else None,
                    "note": "Confirmed in the 6-video pilot." if confirmed else "",
                },
            }
        )
        walking_by_track[track_id].append(ae_id)
        if not confirmed:
            pending.append(
                {
                    "video_id": video_id,
                    "reason": "walking_candidate",
                    "atomic_evidence_id": ae_id,
                    "source_track_ids": [track_id],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "current_proposal": {"action": "walking", "direction": segment["direction_proposal"], "running_conflict": conflicts},
                }
            )

    for track_id in sorted(wandering_tracks, key=int):
        related = walking_by_track.get(track_id, [])
        frames = [item for item in evidence if item["atomic_evidence_id"] in related]
        if not frames:
            continue
        start_time = min(item["start_time"] for item in frames)
        end_time = max(item["end_time"] for item in frames)
        if confirmed_video:
            ae_id = f"AE{len(evidence) + 1:03d}"
            evidence.append(
                {
                    "atomic_evidence_id": ae_id,
                    "source": {"type": "person_track_review", "virat_event_ids": [], "source_track_ids": [track_id]},
                    "start_frame": min(item["start_frame"] for item in frames),
                    "end_frame": max(item["end_frame"] for item in frames),
                    "start_time": start_time,
                    "end_time": end_time,
                    "subject": {"category": "person", "track_id": track_id},
                    "action": "wandering",
                    "object": [],
                    "direction": None,
                    "screen_location": None,
                    "semantic_location": None,
                    "caption_evaluable": True,
                    "confirmed_caption_event": True,
                    "constituent_walking_atomic_evidence_ids": related,
                    "review": {"human_verified": True, "is_wandering": True, "note": "Confirmed in the 6-video pilot."},
                }
            )
        else:
            pending.append(
                {
                    "video_id": video_id,
                    "reason": "wandering_candidate",
                    "atomic_evidence_id": None,
                    "source_track_ids": [track_id],
                    "start_time": start_time,
                    "end_time": end_time,
                    "current_proposal": {"action": "wandering", "direction": None, "constituent_atomic_evidence_ids": related},
                }
            )

    events = []
    eligible = [item for item in evidence if item["confirmed_caption_event"]]
    for item in sorted(eligible, key=lambda row: (row["start_frame"], row["atomic_evidence_id"])):
        events.append(
            {
                "caption_event_id": f"CE{len(events) + 1:03d}",
                "event_form": "individual",
                "source_atomic_evidence_ids": [item["atomic_evidence_id"]],
                "source_track_ids": [track for track in item["source"]["source_track_ids"] if track == item["subject"]["track_id"]],
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

    centroids = {str(track_id): track["centroids"] for track_id, track in tracks.items()}
    diagonal = math.hypot(doc["frame_width"], doc["frame_height"])
    by_id = {item["caption_event_id"]: item for item in events}
    edges = set()
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            if left["action"] != right["action"] or not left["action"]:
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
            ca = centroids.get(left["subject"]["track_id"], {})
            cb = centroids.get(right["subject"]["track_id"], {})
            distances = []
            for frame in range(start, end + 1):
                if frame in ca and frame in cb:
                    distances.append(math.hypot(ca[frame][0] - cb[frame][0], ca[frame][1] - cb[frame][1]) / diagonal)
            if not distances or sorted(distances)[len(distances) // 2] > SPATIAL_MAX:
                continue
            edges.add(tuple(sorted((left["caption_event_id"], right["caption_event_id"]))))
    for clique in pilot.bron_kerbosch([item["caption_event_id"] for item in events], edges):
        members = [by_id[event_id] for event_id in clique]
        directions = {item["direction"] for item in members}
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
                "direction": next(iter(directions)) if len(directions) == 1 else None,
                "semantic_location": None,
                "reference_sentence": None,
                "person_count_is_required_caption_slot": False,
            }
        )
    for item in evidence:
        item.pop("boxes", None)
    summary = {
        "video_id": video_id,
        "roi_set": doc["roi_set"],
        "annotation_status": "available",
        "n_virat_source_events": len(doc["events"]),
        "n_atomic_evidence": len(evidence),
        "n_caption_events": len(events),
        "n_individual_caption_events": sum(1 for item in events if item["event_form"] == "individual"),
        "n_group_caption_events": sum(1 for item in events if item["event_form"] == "group"),
        "n_walking_candidates": sum(1 for item in pending if item["reason"] == "walking_candidate") + (sum(1 for item in evidence if item.get("action") == "walking") if confirmed_video else 0),
        "n_confirmed_walking": sum(1 for item in evidence if item.get("action") == "walking"),
        "n_wandering_candidates": (1 if confirmed_video else 0) * len(wandering_tracks) + sum(1 for item in pending if item["reason"] == "wandering_candidate"),
        "n_confirmed_wandering": sum(1 for item in evidence if item.get("action") == "wandering"),
        "n_human_review_pending": len(pending),
        "n_duplicate_resolved": len(resolved),
        "n_carrying_without_object": sum(1 for item in pending if item["reason"] == "ambiguous_action"),
    }
    return {"atomic_evidence": evidence, "caption_events": events, "pending": pending, "resolved_duplicates": resolved, "summary": summary}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    policy = json.loads(POLICY_SRC.read_text())
    rule = policy["walking"]["walking_temporal_unit"]["minimum_segment_rule"]
    if rule.get("minimum_segment_duration_sec") != 1.0 or rule.get("final_threshold_selected") is not True:
        raise RuntimeError("pilot policy is not the confirmed 1.0s rule")
    policy["scope"] = (
        "Same caption-event rules as the 6-video pilot, applied to 191 videos: "
        "132 single-ROI with area < 0.90 and 59 multi-ROI. "
        "Walking and wandering stay unconfirmed except the six videos already reviewed in the pilot."
    )
    policy["applied_to"] = "experiments/virat/caption_event_gt_191"
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "annotation_policy.json", policy)

    manifest_videos = []
    summaries = []
    pending = []
    resolved = []
    for row in load_lists():
        video_id = row["video_id"]
        state = dense.annotation_file_state(ANN_DIR, video_id)
        available = all(value == "file" for value in state.values())
        entry = {
            "video_id": video_id,
            "video_path": str(row["video_path"]),
            "roi_set": row["roi_set"],
            "annotation_status": "available" if available else "missing",
            "annotation_files": state,
        }
        if not available:
            entry["n_events"] = 0
            manifest_videos.append(entry)
            summaries.append({**entry, "n_atomic_evidence": 0, "n_caption_events": 0, "n_group_caption_events": 0, "n_human_review_pending": 0})
            print(f"MISSING {video_id}", flush=True)
            continue
        doc = dense.convert_video(video_id, row["video_path"], ANN_DIR)
        doc["roi_set"] = row["roi_set"]
        tracks = segments.load_person_tracks(video_id)
        bundle = build_video(doc, tracks)
        out = OUT / "gt" / video_id
        write_json(out / "atomic_evidence.json", {"video_id": video_id, "atomic_evidence": bundle["atomic_evidence"]})
        write_json(out / "caption_events.json", {"video_id": video_id, "reference_sentence": None, "caption_events": bundle["caption_events"]})
        entry.update({"n_events": doc["n_events"], "fps": doc["fps"], "frame_size": [doc["frame_width"], doc["frame_height"]]})
        manifest_videos.append(entry)
        summaries.append(bundle["summary"])
        pending.extend(bundle["pending"])
        resolved.extend(bundle["resolved_duplicates"])
        print(
            video_id,
            "AE", bundle["summary"]["n_atomic_evidence"],
            "CE", bundle["summary"]["n_caption_events"],
            "PENDING", bundle["summary"]["n_human_review_pending"],
            flush=True,
        )

    write_json(
        OUT / "manifest.json",
        {
            "video_lists": {key: str(path) for key, path in LISTS.items()},
            "n_videos": len(manifest_videos),
            "n_annotation_available": sum(1 for item in manifest_videos if item["annotation_status"] == "available"),
            "n_annotation_missing": sum(1 for item in manifest_videos if item["annotation_status"] == "missing"),
            "videos": manifest_videos,
        },
    )
    write_json(
        OUT / "review_summary.json",
        {
            "n_videos": len(summaries),
            "n_annotation_available": sum(1 for item in summaries if item.get("annotation_status") == "available"),
            "n_annotation_missing": sum(1 for item in summaries if item.get("annotation_status") == "missing"),
            "n_atomic_evidence": sum(item.get("n_atomic_evidence", 0) for item in summaries),
            "n_caption_events": sum(item.get("n_caption_events", 0) for item in summaries),
            "n_group_caption_events": sum(item.get("n_group_caption_events", 0) for item in summaries),
            "n_confirmed_walking": sum(item.get("n_confirmed_walking", 0) for item in summaries),
            "n_confirmed_wandering": sum(item.get("n_confirmed_wandering", 0) for item in summaries),
            "n_human_review_pending": len(pending),
            "n_pending_walking": sum(1 for item in pending if item["reason"] == "walking_candidate"),
            "n_pending_wandering": sum(1 for item in pending if item["reason"] == "wandering_candidate"),
            "n_pending_carrying_without_object": sum(1 for item in pending if item["reason"] == "ambiguous_action"),
            "n_duplicate_resolved": len(resolved),
            "videos": summaries,
            "resolved_duplicates": resolved,
            "pending_human_review": pending,
        },
    )
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
