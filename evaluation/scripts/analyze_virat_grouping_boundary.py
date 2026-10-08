#!/usr/bin/env python3
"""Boundary analysis for same-action multi-person grouping.

Reads annotation_policy.json and the 6-video pilot reviews.
Writes only under caption_event_pilot_6/grouping_analysis/.
Does not select a final threshold, create group events, or edit atomic GT.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2

EVAL_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = EVAL_ROOT / "experiments" / "virat" / "annotation_policy.json"
PILOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_pilot_6"
ANN_DIR = Path("/data/datasets/VIRAT/annotations")
OUT = PILOT / "grouping_analysis"

CAPTION_ACTION = {
    "person_entering_facility": "entering_facility",
    "person_exiting_facility": "exiting_facility",
    "person_running": "running",
    "person_getting_into_vehicle": "getting_into_vehicle",
    "person_getting_out_of_vehicle": "getting_out_of_vehicle",
    "person_opening_vehicle_trunk": "opening_vehicle_trunk",
    "person_closing_vehicle_trunk": "closing_vehicle_trunk",
    "person_gesturing": "gesturing",
    "person_digging": "digging",
    "person_loading_object_to_vehicle": "loading_object",
    "person_unloading_object_from_vehicle": "unloading_object",
}


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


def caption_semantics(event: dict) -> dict:
    if event["activity_class"] == "person_carrying_object":
        if event["object"]:
            return {"action": "carrying_object", "carrying_evaluable": True}
        return {"action": None, "carrying_evaluable": False}
    return {
        "action": CAPTION_ACTION.get(event["activity_class"], event["activity_class"]),
        "carrying_evaluable": None,
    }


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.9 * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_boxes(video_id: str, track_ids: set[int]) -> dict[int, dict[int, tuple[float, float]]]:
    centroids: dict[int, dict[int, tuple[float, float]]] = {track_id: {} for track_id in track_ids}
    path = ANN_DIR / f"{video_id}.viratdata.objects.txt"
    if not path.is_file():
        return centroids
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj_id, _dur, frame, x, y, width, height, _obj_type = map(int, line.split()[:8])
        if obj_id in centroids:
            centroids[obj_id][frame] = (x + width / 2.0, y + height / 2.0)
    return centroids


def maximal_cliques(nodes: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    neighbors = {node: set() for node in nodes}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    cliques: list[list[str]] = []

    def bron(current: set[str], candidates: set[str], excluded: set[str]) -> None:
        if not candidates and not excluded:
            if len(current) >= 2:
                cliques.append(sorted(current, key=int))
            return
        for node in list(candidates):
            bron(
                current | {node},
                candidates & neighbors[node],
                excluded & neighbors[node],
            )
            candidates.remove(node)
            excluded.add(node)

    bron(set(), set(nodes), set())
    return cliques


def components(nodes: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    neighbors = {node: set() for node in nodes}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    seen = set()
    groups = []
    for node in nodes:
        if node in seen or not neighbors[node]:
            continue
        stack = [node]
        seen.add(node)
        group = []
        while stack:
            current = stack.pop()
            group.append(current)
            for other in neighbors[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        if len(group) >= 2:
            groups.append(sorted(group, key=int))
    return groups


def build_pairs(policy: dict) -> tuple[list[dict], list[dict]]:
    temporal_floor = min(policy["grouping_analysis"]["pair"]["temporal_thresholds"])
    events = []
    for review_path in sorted(PILOT.glob("VIRAT_*/review.json")):
        doc = json.loads(review_path.read_text())
        diagonal = math.hypot(doc["frame_width"], doc["frame_height"])
        usable = []
        for event in doc["candidate_events"]:
            semantics = caption_semantics(event)
            if semantics["action"] is None or not event["subject"]["track_id"]:
                continue
            usable.append({**event, "caption_action": semantics["action"], "video_id": doc["video_id"]})
        boxes = load_boxes(doc["video_id"], {int(event["subject"]["track_id"]) for event in usable})
        for event in usable:
            event["_centroids"] = boxes[int(event["subject"]["track_id"])]
            event["_diagonal"] = diagonal
            event["_fps"] = doc["fps"]
            event["_video_path"] = doc["video_path"]
            event["_width"] = doc["frame_width"]
            event["_height"] = doc["frame_height"]
            events.append(event)
    pairs = []
    by_key = defaultdict(list)
    for event in events:
        by_key[(event["video_id"], event["caption_action"])].append(event)
    for (_video_id, _action), group in by_key.items():
        for left, right in combinations(group, 2):
            if left["subject"]["track_id"] == right["subject"]["track_id"]:
                continue
            start = max(left["start_frame"], right["start_frame"])
            end = min(left["end_frame"], right["end_frame"])
            intersection = end - start + 1
            if intersection <= 0:
                continue
            shorter = min(
                left["end_frame"] - left["start_frame"] + 1,
                right["end_frame"] - right["start_frame"] + 1,
            )
            ratio = intersection / shorter
            distances = []
            for frame in range(start, end + 1):
                c_left = left["_centroids"].get(frame)
                c_right = right["_centroids"].get(frame)
                if c_left is None or c_right is None:
                    continue
                distances.append(math.hypot(c_left[0] - c_right[0], c_left[1] - c_right[1]) / left["_diagonal"])
            pair = {
                "video_id": left["video_id"],
                "video_path": left["_video_path"],
                "fps": left["_fps"],
                "frame_width": left["_width"],
                "frame_height": left["_height"],
                "event_ids": [left["event_id"], right["event_id"]],
                "subject_track_ids": [left["subject"]["track_id"], right["subject"]["track_id"]],
                "activity_classes": [left["activity_class"], right["activity_class"]],
                "caption_action": left["caption_action"],
                "overlap_start_frame": start,
                "overlap_end_frame": end,
                "temporal_overlap_ratio": round(ratio, 4),
                "n_overlap_frames": intersection,
                "n_distance_frames": len(distances),
                "median_normalized_distance": round(sorted(distances)[len(distances) // 2], 4) if distances else None,
                "p90_normalized_distance": round(p90(distances), 4) if distances else None,
                "max_normalized_distance": round(max(distances), 4) if distances else None,
            }
            if ratio + 1e-9 >= temporal_floor and distances:
                pair["_median"] = pair["median_normalized_distance"]
                pairs.append(pair)
    public = [{key: value for key, value in pair.items() if not key.startswith("_")} for pair in pairs]
    return pairs, public


def analyze(pairs: list[dict], policy: dict) -> tuple[list[dict], list[dict], list[dict]]:
    temporal_thresholds = policy["grouping_analysis"]["pair"]["temporal_thresholds"]
    spatial_thresholds = policy["grouping_analysis"]["pair"]["spatial_thresholds"]
    stats = []
    boundaries = []
    chains = []
    for temporal in temporal_thresholds:
        for spatial in spatial_thresholds:
            eligible = [
                pair for pair in pairs
                if pair["temporal_overlap_ratio"] + 1e-9 >= temporal and pair["_median"] is not None
            ]
            accepted = [pair for pair in eligible if pair["_median"] <= spatial + 1e-12]
            rejected = [pair for pair in eligible if pair["_median"] > spatial + 1e-12]
            edges = set()
            nodes = set()
            for pair in accepted:
                left, right = pair["event_ids"]
                video = pair["video_id"]
                action = pair["caption_action"]
                nodes.add((video, action, left))
                nodes.add((video, action, right))
                a, b = sorted([left, right], key=int)
                edges.add((video, action, a, b))
            grouped_events = set()
            size_counts = defaultdict(int)
            n_groups = 0
            by_bucket = defaultdict(list)
            for video, action, event_id in nodes:
                by_bucket[(video, action)].append(event_id)
            edge_lookup = defaultdict(set)
            for video, action, left, right in edges:
                edge_lookup[(video, action)].add((left, right))
            population = {
                (pair["video_id"], pair["caption_action"], event_id)
                for pair in pairs
                for event_id in pair["event_ids"]
                if pair["temporal_overlap_ratio"] + 1e-9 >= temporal
            }
            for key, members in by_bucket.items():
                unique = sorted(set(members), key=int)
                cliques = maximal_cliques(unique, edge_lookup[key])
                for clique in cliques:
                    n_groups += 1
                    size_counts[str(len(clique))] += 1
                    grouped_events.update((key[0], key[1], event_id) for event_id in clique)
                for component in components(unique, edge_lookup[key]):
                    if len(component) < 3:
                        continue
                    component_edges = {
                        tuple(sorted((left, right), key=int))
                        for left, right in combinations(component, 2)
                        if tuple(sorted((left, right), key=int)) in edge_lookup[key]
                    }
                    missing = []
                    for left, right in combinations(component, 2):
                        ordered = tuple(sorted((left, right), key=int))
                        if ordered not in component_edges:
                            match = next(
                                pair for pair in eligible
                                if pair["video_id"] == key[0]
                                and pair["caption_action"] == key[1]
                                and set(pair["event_ids"]) == {left, right}
                            ) if any(
                                pair["video_id"] == key[0]
                                and pair["caption_action"] == key[1]
                                and set(pair["event_ids"]) == {left, right}
                                for pair in eligible
                            ) else None
                            missing.append(
                                {
                                    "event_ids": [left, right],
                                    "temporal_overlap_ratio": None if match is None else match["temporal_overlap_ratio"],
                                    "median_normalized_distance": None if match is None else match["median_normalized_distance"],
                                    "reason": "pair_absent_or_outside_spatial_threshold" if match is None or match["_median"] > spatial else "unknown",
                                }
                            )
                    if missing:
                        chains.append(
                            {
                                "temporal_threshold": temporal,
                                "spatial_threshold": spatial,
                                "video_id": key[0],
                                "caption_action": key[1],
                                "single_link_component": component,
                                "complete_link_cliques": [
                                    clique for clique in maximal_cliques(unique, edge_lookup[key])
                                    if set(clique) <= set(component)
                                ],
                                "missing_pairs": missing,
                            }
                        )
            def record(kind: str, pair: dict | None) -> dict | None:
                if pair is None:
                    return None
                return {
                    "role": kind,
                    "temporal_threshold": temporal,
                    "spatial_threshold": spatial,
                    "accepted": kind != "nearest_rejected",
                    **{key: value for key, value in pair.items() if not key.startswith("_")},
                }

            closest = min(accepted, key=lambda pair: pair["_median"]) if accepted else None
            farthest = max(accepted, key=lambda pair: pair["_median"]) if accepted else None
            nearest = min(rejected, key=lambda pair: pair["_median"]) if rejected else None
            boundary = {
                "temporal_threshold": temporal,
                "spatial_threshold": spatial,
                "closest_accepted": record("closest_accepted", closest),
                "farthest_accepted": record("farthest_accepted", farthest),
                "nearest_rejected": record("nearest_rejected", nearest),
            }
            boundaries.append(boundary)
            stats.append(
                {
                    "temporal_threshold": temporal,
                    "spatial_threshold": spatial,
                    "n_temporal_eligible_pairs": len(eligible),
                    "n_accepted_pairs": len(accepted),
                    "n_spatial_rejected_pairs": len(rejected),
                    "n_candidate_groups": n_groups,
                    "group_size_distribution": dict(sorted(size_counts.items(), key=lambda item: int(item[0]))),
                    "n_grouped_events": len(grouped_events),
                    "n_ungrouped_events": len(population - grouped_events),
                    "n_events_meeting_temporal_threshold": len(population),
                }
            )
    return stats, boundaries, chains


def draw_pair(frame, pair: dict, boxes: dict[int, tuple[int, int, int, int]], role: str) -> None:
    height, width = frame.shape[:2]
    colors = [(80, 220, 60), (40, 140, 255)]
    for track_id, color in zip(pair["subject_track_ids"], colors):
        box = boxes.get(int(track_id))
        if box is None:
            continue
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), color, 2)
        cv2.putText(
            frame,
            f"id {track_id} {pair['caption_action']}",
            (x, max(52, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    status = "ACCEPTED" if pair["accepted"] else "REJECTED"
    text = (
        f"{status} {role}  T={pair['temporal_threshold']:.2f} D={pair['spatial_threshold']:.2f}  "
        f"overlap={pair['temporal_overlap_ratio']:.2f}  "
        f"med={pair['median_normalized_distance']:.3f} p90={pair['p90_normalized_distance']:.3f} "
        f"max={pair['max_normalized_distance']:.3f}"
    )
    cv2.rectangle(frame, (0, 0), (width, 36), (0, 0, 0), -1)
    cv2.putText(frame, text[:170], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


def render_boundaries(boundaries: list[dict]) -> list[dict]:
    jobs = []
    for boundary in boundaries:
        for role in ("closest_accepted", "farthest_accepted", "nearest_rejected"):
            pair = boundary[role]
            if pair is None:
                continue
            tag_t = f"T{int(round(pair['temporal_threshold'] * 100)):03d}"
            tag_d = f"D{int(round(pair['spatial_threshold'] * 100)):03d}"
            name = (
                f"{tag_t}_{tag_d}_{role}_"
                f"{pair['video_id']}_e{pair['event_ids'][0]}_e{pair['event_ids'][1]}.mp4"
            )
            jobs.append({**pair, "role": role, "output_name": name})
    by_video = defaultdict(list)
    for job in jobs:
        by_video[job["video_id"]].append(job)
    written = []
    for video_id, video_jobs in by_video.items():
        boxes = {}
        needed = {int(track_id) for job in video_jobs for track_id in job["subject_track_ids"]}
        path = ANN_DIR / f"{video_id}.viratdata.objects.txt"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            obj_id, _dur, frame, x, y, width, height, _obj_type = map(int, line.split()[:8])
            if obj_id in needed:
                boxes.setdefault(obj_id, {})[frame] = (x, y, width, height)
        sample = video_jobs[0]
        width, height = sample["frame_width"], sample["frame_height"]
        fps = sample["fps"]
        pad = max(1, int(round(0.5 * fps)))
        encoders = []
        for job in video_jobs:
            job["clip_start"] = max(0, job["overlap_start_frame"] - pad)
            job["clip_end"] = job["overlap_end_frame"] + pad
            out = OUT / "clips" / job["output_name"]
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
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                for job, encoder, _out in encoders:
                    if job["clip_start"] <= frame_index <= job["clip_end"]:
                        painted = frame.copy()
                        frame_boxes = {
                            track_id: track.get(frame_index)
                            for track_id, track in boxes.items()
                        }
                        draw_pair(painted, job, frame_boxes, job["role"])
                        encoder.stdin.write(painted.tobytes())
                frame_index += 1
        finally:
            cap.release()
            for job, encoder, out in encoders:
                encoder.stdin.close()
                code = encoder.wait()
                if code != 0:
                    raise RuntimeError(f"ffmpeg failed for {out} ({code})")
                written.append(
                    {
                        "file": str(out.relative_to(EVAL_ROOT)),
                        "role": job["role"],
                        "temporal_threshold": job["temporal_threshold"],
                        "spatial_threshold": job["spatial_threshold"],
                        "video_id": job["video_id"],
                        "event_ids": job["event_ids"],
                        "subject_track_ids": job["subject_track_ids"],
                        "caption_action": job["caption_action"],
                        "temporal_overlap_ratio": job["temporal_overlap_ratio"],
                        "median_normalized_distance": job["median_normalized_distance"],
                        "p90_normalized_distance": job["p90_normalized_distance"],
                        "max_normalized_distance": job["max_normalized_distance"],
                        "accepted": job["accepted"],
                    }
                )
        print(f"WROTE {video_id} clips={len(video_jobs)}", flush=True)
    return written


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    policy = load_policy()
    pairs, public_pairs = build_pairs(policy)
    stats, boundaries, chains = analyze(pairs, policy)
    clips = render_boundaries(boundaries)
    write_json(OUT / "pairs.json", {"n_pairs": len(public_pairs), "pairs": public_pairs})
    write_json(OUT / "threshold_stats.json", {"combinations": stats})
    write_json(OUT / "boundary_cases.json", {"combinations": boundaries})
    write_json(OUT / "chain_cases.json", {"n_cases": len(chains), "cases": chains})
    write_json(OUT / "clips.json", {"n_clips": len(clips), "clips": clips})
    print(f"PAIRS {len(public_pairs)} CHAINS {len(chains)} CLIPS {len(clips)}")


if __name__ == "__main__":
    main()
