#!/usr/bin/env python3
"""Walking-candidate pilot for the 6 caption-event videos.

Does not confirm walking, create group events, or edit VIRAT events,
annotation_policy.json, or the existing grouping rule.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EVAL_ROOT = Path(__file__).resolve().parents[1]
PILOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_pilot_6"
ANN_DIR = Path("/data/datasets/VIRAT/annotations")
OUT = PILOT / "walking_analysis"

MIN_DISP_PX = 20.0
MIN_DISP_FRAC = 0.05
AXIS_DOMINANCE = 2.0
TEMPORAL_MIN = 0.3
SPATIAL_MAX = 0.15
DIRECTION_THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
CONFLICT_CLASSES = {"person_running"}

FUTURE_EVENT = {
    "event_type": "human_caption_event",
    "source": "person_track_review",
    "source_track_ids": [],
    "subject": {"category": "person", "group": False},
    "action": "walking",
    "direction": None,
    "start_time": None,
    "end_time": None,
    "semantic_location": None,
    "reference_sentence": None,
    "person_count_is_required_caption_slot": False,
}
FUTURE_GROUP = {
    **FUTURE_EVENT,
    "subject": {"category": "person", "group": True},
    "source_track_ids": [],
    "constituent_track_count": None,
}


def coarse_direction(dx: float, dy: float, frame_width: int) -> dict:
    disp = math.hypot(dx, dy)
    threshold = max(MIN_DISP_PX, MIN_DISP_FRAC * frame_width)
    result = {
        "value": None,
        "dx_px": round(dx, 2),
        "dy_px": round(dy, 2),
        "displacement_px": round(disp, 2),
        "threshold_px": round(threshold, 2),
        "method": "endpoint_displacement",
        "dominant_axis_ratio": AXIS_DOMINANCE,
        "source": "derived_from_track",
    }
    if disp < threshold:
        result["reason"] = "below_displacement_threshold"
        return result
    horizontal = "left_to_right" if dx > 0 else "right_to_left"
    vertical = "down" if dy > 0 else "up"
    if abs(dx) >= AXIS_DOMINANCE * abs(dy):
        result["value"] = horizontal
    elif abs(dy) >= AXIS_DOMINANCE * abs(dx):
        result["value"] = vertical
    else:
        result["value"] = f"diagonal_{horizontal}_{vertical}"
        result["reason"] = "no_dominant_axis"
    return result


def significant_direction_legs(
    centroids: dict[int, tuple[float, float]], frame_width: int, fps: float
) -> list[dict]:
    """Sustained legs. A reversal counts only after both the outgoing leg and the
    turn itself move at least the walking displacement threshold.
    """
    frames = sorted(centroids)
    if len(frames) < 2:
        return []
    points = [(frame, centroids[frame][0], centroids[frame][1]) for frame in frames]
    window = max(5, int(round(0.4 * fps)))
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = []
    for index, (frame, _x, _y) in enumerate(points):
        low = max(0, index - half)
        high = min(len(points), index + half + 1)
        smoothed.append(
            (
                frame,
                sum(points[j][1] for j in range(low, high)) / (high - low),
                sum(points[j][2] for j in range(low, high)) / (high - low),
            )
        )
    threshold = max(MIN_DISP_PX, MIN_DISP_FRAC * frame_width)

    def dist(left, right) -> float:
        return math.hypot(left[1] - right[1], left[2] - right[2])

    def make_leg(start, end) -> dict | None:
        direction = coarse_direction(end[1] - start[1], end[2] - start[2], frame_width)
        if direction["value"] is None:
            return None
        return {
            "start_frame": start[0],
            "end_frame": end[0],
            "direction": direction["value"],
            "displacement_px": direction["displacement_px"],
            "dx_px": direction["dx_px"],
            "dy_px": direction["dy_px"],
        }

    origin = smoothed[0]
    extreme = smoothed[0]
    legs = []
    for point in smoothed[1:]:
        if dist(point, origin) > dist(extreme, origin):
            extreme = point
        if dist(extreme, origin) >= threshold and dist(point, extreme) >= threshold:
            leg = make_leg(origin, extreme)
            if leg is not None:
                legs.append(leg)
            origin = extreme
            extreme = point
    final = make_leg(origin, smoothed[-1])
    if final is not None:
        legs.append(final)
    return legs


def legs_are_opposite(left: dict, right: dict) -> bool:
    denom = math.hypot(left["dx_px"], left["dy_px"]) * math.hypot(right["dx_px"], right["dy_px"])
    if denom < 1:
        return False
    return (left["dx_px"] * right["dx_px"] + left["dy_px"] * right["dy_px"]) / denom < 0


def motion_pattern(legs: list[dict]) -> str:
    if any(legs_are_opposite(left, right) for left, right in combinations(legs, 2)):
        return "wandering"
    if legs:
        return "directional"
    return "direction_not_committed"


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.9 * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_tracks(video_id: str) -> dict[int, dict]:
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


def trajectory(centroids: dict[int, tuple[float, float]]) -> dict:
    frames = sorted(centroids)
    x0, y0 = centroids[frames[0]]
    x1, y1 = centroids[frames[-1]]
    net = math.hypot(x1 - x0, y1 - y0)
    path = 0.0
    radius = 0.0
    prev = centroids[frames[0]]
    for frame in frames[1:]:
        point = centroids[frame]
        path += math.hypot(point[0] - prev[0], point[1] - prev[1])
        radius = max(radius, math.hypot(point[0] - x0, point[1] - y0))
        prev = point
    return {
        "frames": frames,
        "dx": x1 - x0,
        "dy": y1 - y0,
        "net": net,
        "path": path,
        "radius": radius,
    }


def overlap_events(track_frames: list[int], events: list[dict], track_id: str) -> tuple[list[dict], list[dict]]:
    start, end = track_frames[0], track_frames[-1]
    track_len = end - start + 1
    overlaps = []
    conflicts = []
    frame_set = set(track_frames)
    for event in events:
        inter_start = max(start, event["start_frame"])
        inter_end = min(end, event["end_frame"])
        if inter_end < inter_start:
            continue
        inter = inter_end - inter_start + 1
        role = "temporal_only"
        for track in event["tracks"]:
            if track["track_id"] == track_id:
                role = track["relation"]
                break
        covered = sum(1 for frame in range(inter_start, inter_end + 1) if frame in frame_set)
        row = {
            "event_id": event["event_id"],
            "activity_class": event["activity_class"],
            "role": role,
            "overlap_frames": inter,
            "overlap_ratio_of_track": round(inter / track_len, 4),
            "tracked_frames_inside_event": covered,
        }
        overlaps.append(row)
        if role == "subject" and event["activity_class"] in CONFLICT_CLASSES:
            conflicts.append(
                {
                    "event_id": event["event_id"],
                    "activity_class": event["activity_class"],
                    "reason": "subject_of_running",
                    "overlap_frames": inter,
                }
            )
    return overlaps, conflicts


def build_candidates() -> list[dict]:
    candidates = []
    for review_path in sorted(PILOT.glob("VIRAT_*/review.json")):
        doc = json.loads(review_path.read_text())
        video_id = doc["video_id"]
        diagonal = math.hypot(doc["frame_width"], doc["frame_height"])
        threshold = max(MIN_DISP_PX, MIN_DISP_FRAC * doc["frame_width"])
        tracks = load_tracks(video_id)
        for track_id, track in tracks.items():
            if len(track["centroids"]) < 2:
                continue
            stats = trajectory(track["centroids"])
            if max(stats["net"], stats["radius"]) < threshold:
                continue
            direction = coarse_direction(stats["dx"], stats["dy"], doc["frame_width"])
            legs = significant_direction_legs(track["centroids"], doc["frame_width"], doc["fps"])
            pattern = motion_pattern(legs)
            proposal = dict(direction)
            if pattern == "wandering":
                proposal["endpoint_direction"] = direction["value"]
                proposal["value"] = None
                proposal["reason"] = "clear_direction_reversal"
            overlaps, conflicts = overlap_events(stats["frames"], doc["candidate_events"], str(track_id))
            start_frame, end_frame = stats["frames"][0], stats["frames"][-1]
            candidates.append(
                {
                    "video_id": video_id,
                    "video_path": doc["video_path"],
                    "fps": doc["fps"],
                    "frame_width": doc["frame_width"],
                    "frame_height": doc["frame_height"],
                    "frame_diagonal": round(diagonal, 2),
                    "track_id": str(track_id),
                    "category": "person",
                    "object_type_id": 1,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": round(start_frame / doc["fps"], 4),
                    "end_time": round(end_frame / doc["fps"], 4),
                    "trajectory_displacement_px": round(stats["net"], 2),
                    "path_length_px": round(stats["path"], 2),
                    "max_distance_from_start_px": round(stats["radius"], 2),
                    "normalized_displacement": round(stats["net"] / diagonal, 4),
                    "displacement_threshold_px": round(threshold, 2),
                    "coarse_direction": direction["value"],
                    "motion_pattern_proposal": pattern,
                    "direction_legs": legs,
                    "overlapping_virat_events": overlaps,
                    "virat_conflicts": conflicts,
                    "confirmed_walking": False,
                    "human_review": {
                        "is_walking": None,
                        "keep_as_caption_event": None,
                        "motion_pattern": None,
                        "review_note": "",
                    },
                    "caption_semantics": {
                        "action": None,
                        "direction": None,
                        "direction_proposal": proposal,
                        "motion_pattern_proposal": pattern,
                    },
                    "_centroids": track["centroids"],
                    "_boxes": track["boxes"],
                }
            )
    return candidates


def cosine(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float | None:
    if left is None or right is None:
        return None
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if left_norm < 5 or right_norm < 5:
        return None
    return (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)


def overlap_vector(centroids: dict[int, tuple[float, float]], start: int, end: int) -> tuple[float, float] | None:
    frames = [frame for frame in range(start, end + 1) if frame in centroids]
    if len(frames) < 2:
        return None
    x0, y0 = centroids[frames[0]]
    x1, y1 = centroids[frames[-1]]
    return (x1 - x0, y1 - y0)


def build_pairs(candidates: list[dict]) -> list[dict]:
    by_video = defaultdict(list)
    for candidate in candidates:
        by_video[candidate["video_id"]].append(candidate)
    pairs = []
    for group in by_video.values():
        for left, right in combinations(group, 2):
            if "wandering" in (left["motion_pattern_proposal"], right["motion_pattern_proposal"]):
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
                distances.append(
                    math.hypot(c_left[0] - c_right[0], c_left[1] - c_right[1]) / left["frame_diagonal"]
                )
            similarity = cosine(
                overlap_vector(left["_centroids"], start, end),
                overlap_vector(right["_centroids"], start, end),
            )
            if not distances or similarity is None:
                continue
            pairs.append(
                {
                    "video_id": left["video_id"],
                    "video_path": left["video_path"],
                    "fps": left["fps"],
                    "frame_width": left["frame_width"],
                    "frame_height": left["frame_height"],
                    "track_ids": [left["track_id"], right["track_id"]],
                    "directions": [left["coarse_direction"], right["coarse_direction"]],
                    "motion_patterns": [left["motion_pattern_proposal"], right["motion_pattern_proposal"]],
                    "overlap_start_frame": start,
                    "overlap_end_frame": end,
                    "temporal_overlap_ratio": round(ratio, 4),
                    "median_normalized_distance": round(sorted(distances)[len(distances) // 2], 4),
                    "p90_normalized_distance": round(p90(distances), 4),
                    "max_normalized_distance": round(max(distances), 4),
                    "direction_cosine": round(similarity, 4),
                    "passes_temporal": ratio + 1e-9 >= TEMPORAL_MIN,
                    "passes_spatial": sorted(distances)[len(distances) // 2] <= SPATIAL_MAX + 1e-12,
                    "opposite_direction": similarity < 0,
                    "provisional": True,
                    "is_walking_confirmed": False,
                }
            )
    return pairs


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
            bron(current | {node}, candidates & neighbors[node], excluded & neighbors[node])
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


def analyze(pairs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    stats = []
    boundaries = []
    exclusions = []
    base = [pair for pair in pairs if pair["passes_temporal"] and pair["passes_spatial"]]
    for threshold in DIRECTION_THRESHOLDS:
        accepted = [pair for pair in base if pair["direction_cosine"] + 1e-12 >= threshold]
        rejected = [pair for pair in base if pair["direction_cosine"] < threshold]
        opposite = [pair for pair in rejected if pair["opposite_direction"]]
        spatial_same = [
            pair for pair in pairs
            if pair["passes_temporal"] and not pair["passes_spatial"] and pair["direction_cosine"] + 1e-12 >= threshold
        ]
        lowest = min(accepted, key=lambda pair: pair["direction_cosine"]) if accepted else None
        highest = max(rejected, key=lambda pair: pair["direction_cosine"]) if rejected else None
        opposite_case = min(opposite, key=lambda pair: pair["direction_cosine"]) if opposite else None
        spatial_case = min(spatial_same, key=lambda pair: pair["median_normalized_distance"]) if spatial_same else None

        edges_by_video = defaultdict(set)
        nodes_by_video = defaultdict(set)
        for pair in accepted:
            left, right = sorted(pair["track_ids"], key=int)
            edges_by_video[pair["video_id"]].add((left, right))
            nodes_by_video[pair["video_id"]].update([left, right])
        n_groups = 0
        sizes = defaultdict(int)
        grouped = set()
        for video_id, nodes in nodes_by_video.items():
            node_list = sorted(nodes, key=int)
            edges = edges_by_video[video_id]
            cliques = maximal_cliques(node_list, edges)
            for clique in cliques:
                n_groups += 1
                sizes[str(len(clique))] += 1
                grouped.update((video_id, track_id) for track_id in clique)
            for component in components(node_list, edges):
                if len(component) < 3:
                    continue
                missing = []
                for left, right in combinations(component, 2):
                    ordered = tuple(sorted((left, right), key=int))
                    if ordered in edges:
                        continue
                    match = next(
                        (
                            pair for pair in pairs
                            if pair["video_id"] == video_id and set(pair["track_ids"]) == {left, right}
                        ),
                        None,
                    )
                    missing.append(
                        {
                            "track_ids": [left, right],
                            "temporal_overlap_ratio": None if match is None else match["temporal_overlap_ratio"],
                            "median_normalized_distance": None if match is None else match["median_normalized_distance"],
                            "direction_cosine": None if match is None else match["direction_cosine"],
                        }
                    )
                if missing:
                    exclusions.append(
                        {
                            "direction_threshold": threshold,
                            "video_id": video_id,
                            "single_link_component": component,
                            "complete_link_cliques": [
                                clique for clique in cliques if set(clique) <= set(component)
                            ],
                            "excluded_pairs": missing,
                        }
                    )
        population = {
            (pair["video_id"], track_id)
            for pair in base
            for track_id in pair["track_ids"]
        }
        def pack(role: str, pair: dict | None) -> dict | None:
            if pair is None:
                return None
            return {"role": role, "direction_threshold": threshold, **pair}

        boundaries.append(
            {
                "direction_threshold": threshold,
                "lowest_coherence_accepted": pack("lowest_coherence_accepted", lowest),
                "highest_coherence_rejected": pack("highest_coherence_rejected", highest),
                "opposite_direction_rejected": pack("opposite_direction_rejected", opposite_case),
                "same_direction_spatial_rejected": pack("same_direction_spatial_rejected", spatial_case),
            }
        )
        stats.append(
            {
                "direction_threshold": threshold,
                "n_temporal_spatial_pairs": len(base),
                "n_accepted_pairs": len(accepted),
                "n_direction_rejected_pairs": len(rejected),
                "n_opposite_direction_pairs": len(opposite),
                "n_same_direction_spatial_rejected_pairs": len(spatial_same),
                "n_candidate_groups": n_groups,
                "group_size_distribution": dict(sorted(sizes.items(), key=lambda item: int(item[0]))),
                "n_grouped_tracks": len(grouped),
                "n_ungrouped_tracks": len(population - grouped),
            }
        )
    return stats, boundaries, exclusions


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONTS: dict[int, ImageFont.FreeTypeFont] = {}
BOX_COLORS = [
    (80, 220, 60),
    (40, 140, 255),
    (0, 220, 255),
    (255, 120, 40),
    (220, 80, 255),
    (80, 80, 255),
    (255, 200, 40),
    (180, 180, 180),
    (40, 180, 180),
]
DIRECTION_KO = {
    "left_to_right": "왼쪽→오른쪽",
    "right_to_left": "오른쪽→왼쪽",
    "up": "아래→위",
    "down": "위→아래",
    "diagonal_left_to_right_down": "왼쪽위→오른쪽아래",
    "diagonal_left_to_right_up": "왼쪽아래→오른쪽위",
    "diagonal_right_to_left_up": "오른쪽아래→왼쪽위",
    "diagonal_right_to_left_down": "오른쪽위→왼쪽아래",
    None: "방향 없음",
}


def get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in FONTS:
        FONTS[size] = ImageFont.truetype(FONT_PATH, size)
    return FONTS[size]


def direction_ko(value: str | None) -> str:
    return DIRECTION_KO.get(value, str(value))


def metric_sentence(job: dict) -> str:
    overlap = job.get("temporal_overlap_ratio")
    distance = job.get("median_normalized_distance")
    cosine = job.get("direction_cosine")
    overlap_text = "측정 안 됨" if overlap is None else f"{overlap:.2f}"
    distance_text = "측정 안 됨" if distance is None else f"{distance:.3f}"
    cosine_text = "측정 안 됨" if cosine is None else f"{cosine:.2f}"
    return (
        f"시간 겹침 {overlap_text} (0.30 이상이면 통과), "
        f"거리 {distance_text} (0.15 이하면 통과), "
        f"방향 유사도 {cosine_text} (1이면 같은 방향, 음수면 반대, 이번 기준 {job['direction_threshold']:.1f} 이상)."
    )


def person_direction_label(coarse: str | None, pattern: str | None) -> str:
    if pattern == "wandering":
        return "왕복"
    return direction_ko(coarse)


def pair_people(job: dict) -> str:
    directions = job.get("directions") or []
    patterns = job.get("motion_patterns") or []
    parts = []
    for index, track_id in enumerate(job["track_ids"]):
        coarse = directions[index] if index < len(directions) else None
        pattern = patterns[index] if index < len(patterns) else None
        parts.append(f"track {track_id} ({person_direction_label(coarse, pattern)})")
    return " / ".join(parts)


def attach_overlay(job: dict, exclusion: dict | None = None) -> None:
    role = job["role"]
    people = pair_people(job)
    metrics = metric_sentence(job)
    if role == "lowest_coherence_accepted":
        lines = [
            "이 영상: 그룹에 남은 쌍 중에서, 방향이 가장 덜 비슷한 쌍",
            "둘은 시간이 겹치고 거리도 가깝습니다. 방향 유사도도 기준을 넘어서 같은 그룹 후보로 남았습니다.",
            metrics,
            people + ". 사람이 walking인지 확인하기 전인 후보입니다.",
        ]
    elif role == "highest_coherence_rejected":
        lines = [
            "이 영상: 그룹에서 빠진 쌍 중에서, 방향이 가장 비슷한 쌍",
            "시간과 거리는 통과했습니다. 방향 유사도가 기준보다 낮아 같은 그룹에서 빠졌습니다.",
            "같은 기준의 '가장 덜 비슷한 통과 쌍'과 나란히 보면 방향 기준의 경계가 보입니다.",
            metrics,
            people + ".",
        ]
    elif role == "opposite_direction_rejected":
        lines = [
            "이 영상: 가까이 있고 시간도 겹치지만, 반대 방향으로 지나가는 쌍",
            "시간과 거리는 통과했습니다. 방향이 반대라 한 그룹으로 보지 않습니다.",
            metrics,
            people + ".",
        ]
    elif role == "same_direction_spatial_rejected":
        lines = [
            "이 영상: 방향은 비슷한데, 화면에서 멀리 떨어져 빠진 쌍",
            "시간과 방향은 통과했습니다. 두 사람 사이 거리가 0.15보다 커서 같은 그룹에서 빠졌습니다.",
            metrics,
            people + ".",
        ]
    else:
        cliques = exclusion["complete_link_cliques"] if exclusion else []
        largest = sorted(cliques, key=len, reverse=True)[:3]
        clique_text = ", ".join("[" + ", ".join(clique) + "]" for clique in largest)
        example = None
        if exclusion:
            scored = [
                pair for pair in exclusion["excluded_pairs"]
                if pair.get("direction_cosine") is not None and pair["direction_cosine"] < 0 and pair.get("median_normalized_distance") is not None
            ]
            if scored:
                example = min(scored, key=lambda pair: pair["median_normalized_distance"])
        example_line = "그룹 안의 모든 쌍이 시간, 거리, 방향을 만족해야 한 그룹입니다."
        if example is not None:
            example_line = (
                f"예를 들어 track {example['track_ids'][0]}과 {example['track_ids'][1]}은 "
                f"거리 {example['median_normalized_distance']:.3f}로 가깝지만 "
                f"방향 유사도 {example['direction_cosine']:.2f}라 서로 반대입니다."
            )
        lines = [
            "이 영상: 3명 이상일 때, 옆 사람과만 비슷하면 전원을 한 그룹으로 넣지 않는 경우",
            f"이어진 track {', '.join(job['track_ids'])}. 실제로 같이 둘 수 있는 묶음은 {clique_text}입니다.",
            example_line,
            "사람이 walking인지 확인하기 전인 후보입니다.",
        ]
    if "37" in job["track_ids"] and "010001_02" in job["video_id"]:
        lines.append("track 37은 기존 VIRAT 주석에서 running인 사람입니다. walking으로 확정하지 않았습니다.")
    job["overlay_lines"] = lines


def draw(frame, job: dict, boxes: dict[str, tuple[int, int, int, int] | None]) -> None:
    height, width = frame.shape[:2]
    directions = job.get("directions") or []
    for index, track_id in enumerate(job["track_ids"]):
        box = boxes.get(track_id)
        if box is None:
            continue
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), color, 2)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pen = ImageDraw.Draw(image)
    body = get_font(22 if width >= 1600 else 18)
    small = get_font(18 if width >= 1600 else 16)
    raw_lines = job.get("overlay_lines") or ["이 영상은 walking 후보의 그룹 경계입니다."]
    line_h = 28 if width >= 1600 else 24
    lines = []
    for raw in raw_lines:
        current = ""
        for char in raw:
            trial = current + char
            if body.getlength(trial) > width - 24 and current:
                lines.append(current)
                current = char
            else:
                current = trial
        if current:
            lines.append(current)
    panel_h = 16 + line_h * len(lines) + 8
    pen.rectangle((0, 0, width, panel_h), fill=(0, 0, 0))
    for line_index, line in enumerate(lines):
        pen.text((12, 8 + line_index * line_h), line, font=body, fill=(245, 245, 245))
    for index, track_id in enumerate(job["track_ids"]):
        box = boxes.get(track_id)
        if box is None:
            continue
        color = BOX_COLORS[index % len(BOX_COLORS)]
        rgb = (color[2], color[1], color[0])
        x, y, w, h = box
        patterns = job.get("motion_patterns") or []
        pattern = patterns[index] if index < len(patterns) else None
        coarse = directions[index] if index < len(directions) else None
        label = f"{track_id} {person_direction_label(coarse, pattern)}".strip()
        label_y = y - 22
        if label_y < panel_h + 2:
            label_y = min(height - 24, y + 4)
        pen.text((x, label_y), label, font=small, fill=rgb)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(jobs: list[dict], candidates: list[dict]) -> list[dict]:
    lookup = {(item["video_id"], item["track_id"]): item for item in candidates}
    by_video = defaultdict(list)
    for job in jobs:
        by_video[job["video_id"]].append(job)
    written = []
    for video_id, video_jobs in by_video.items():
        sample_track = lookup[(video_id, video_jobs[0]["track_ids"][0])]
        width, height = sample_track["frame_width"], sample_track["frame_height"]
        fps = sample_track["fps"]
        pad = max(1, int(round(0.5 * fps)))
        boxes = {track_id: lookup[(video_id, track_id)]["_boxes"] for track_id in {track for job in video_jobs for track in job["track_ids"]} if (video_id, track_id) in lookup}
        encoders = []
        for job in video_jobs:
            start = max(0, job["overlap_start_frame"] - pad)
            end = job["overlap_end_frame"] + pad
            tag = f"C{int(round(job['direction_threshold'] * 100)):03d}"
            tracks = "_".join(f"t{track_id}" for track_id in job["track_ids"])
            name = f"{tag}_{job['role']}_{video_id}_{tracks}.mp4"
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
            encoders.append((job, encoder, out, start, end))
        cap = cv2.VideoCapture(sample_track["video_path"])
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                for job, encoder, _out, start, end in encoders:
                    if start <= frame_index <= end:
                        painted = frame.copy()
                        frame_boxes = {
                            track_id: track_boxes.get(frame_index)
                            for track_id, track_boxes in boxes.items()
                        }
                        draw(painted, job, frame_boxes)
                        encoder.stdin.write(painted.tobytes())
                frame_index += 1
        finally:
            cap.release()
            for _job, encoder, out, _start, _end in encoders:
                encoder.stdin.close()
                code = encoder.wait()
                if code != 0:
                    raise RuntimeError(f"ffmpeg failed for {out} ({code})")
        for job, _encoder, out, _start, _end in encoders:
            written.append({"file": str(out.relative_to(EVAL_ROOT)), "role": job["role"], "direction_threshold": job["direction_threshold"], "video_id": video_id, "track_ids": job["track_ids"], "overlay_lines": job.get("overlay_lines", [])})
        print(f"WROTE {video_id} clips={len(video_jobs)}", flush=True)
    return written


def public_candidate(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    candidates = build_candidates()
    pairs = build_pairs(candidates)
    stats, boundaries, exclusions = analyze(pairs)
    render_jobs = []
    for boundary in boundaries:
        for role in (
            "lowest_coherence_accepted",
            "highest_coherence_rejected",
            "opposite_direction_rejected",
            "same_direction_spatial_rejected",
        ):
            pair = boundary[role]
            if pair is not None:
                attach_overlay(pair)
                render_jobs.append(pair)
    for case in exclusions:
        if not case["excluded_pairs"]:
            continue
        missing = case["excluded_pairs"][0]
        job = {
            "role": "complete_link_excluded",
            "direction_threshold": case["direction_threshold"],
            "video_id": case["video_id"],
            "video_path": next(item["video_path"] for item in candidates if item["video_id"] == case["video_id"]),
            "track_ids": case["single_link_component"],
            "directions": [
                next(item["coarse_direction"] for item in candidates if item["video_id"] == case["video_id"] and item["track_id"] == track_id)
                for track_id in case["single_link_component"]
            ],
            "motion_patterns": [
                next(item["motion_pattern_proposal"] for item in candidates if item["video_id"] == case["video_id"] and item["track_id"] == track_id)
                for track_id in case["single_link_component"]
            ],
            "temporal_overlap_ratio": missing["temporal_overlap_ratio"],
            "median_normalized_distance": missing["median_normalized_distance"],
            "direction_cosine": missing["direction_cosine"],
            "overlap_start_frame": min(item["start_frame"] for item in candidates if item["video_id"] == case["video_id"] and item["track_id"] in case["single_link_component"]),
            "overlap_end_frame": max(item["end_frame"] for item in candidates if item["video_id"] == case["video_id"] and item["track_id"] in case["single_link_component"]),
        }
        attach_overlay(job, case)
        render_jobs.append(job)
    clips = render(render_jobs, candidates) if render_jobs else []
    kept = {Path(item["file"]).name for item in clips}
    clips_dir = OUT / "clips"
    if clips_dir.exists():
        for old in clips_dir.glob("*.mp4"):
            if old.name not in kept:
                old.unlink()
    by_video = defaultdict(list)
    for candidate in candidates:
        by_video[candidate["video_id"]].append(public_candidate(candidate))
    for video_id, rows in by_video.items():
        write_json(
            OUT / video_id / "walking_review.json",
            {
                "video_id": video_id,
                "confirmed_walking": False,
                "final_events_created": False,
                "candidates": rows,
                "confirmed_walking_events": [],
                "confirmed_group_events": [],
                "future_event_template": FUTURE_EVENT,
                "future_group_event_template": FUTURE_GROUP,
            },
        )
    write_json(
        OUT / "walking_policy.json",
        {
            "does_not_modify": "experiments/virat/annotation_policy.json",
            "existing_grouping_rule_unchanged": {
                "temporal_overlap_ratio_min": TEMPORAL_MIN,
                "median_normalized_distance_max": SPATIAL_MAX,
                "groups_of_three_or_more": "complete_link",
                "different_actions_grouped": False,
                "screen_bin_3x3_used": False,
            },
            "walking_confirmed_automatically": False,
            "direction_cosine": "cosine of the two net displacement vectors over the overlap interval",
            "direction_thresholds_compared": DIRECTION_THRESHOLDS,
            "direction_threshold_selected": False,
            "wandering": {
                "auto_confirm_event": False,
                "excluded_from_directional_grouping": True,
                "rule": "At least two sustained legs have opposite directions, and each leg displacement is >= max(20px, 0.05 * frame_width).",
                "opposite": "cosine of the two leg displacement vectors < 0",
                "not_sufficient": "A small net displacement or an uncommitted endpoint direction, without two clear opposing legs.",
            },
            "reference_sentence": None,
        },
    )
    wandering = [
        {
            "video_id": item["video_id"],
            "track_id": item["track_id"],
            "direction_legs": item["direction_legs"],
            "endpoint_direction": item["coarse_direction"],
            "trajectory_displacement_px": item["trajectory_displacement_px"],
        }
        for item in candidates
        if item["motion_pattern_proposal"] == "wandering"
    ]
    write_json(OUT / "wandering_candidates.json", {"n_wandering": len(wandering), "confirmed": False, "candidates": wandering})
    write_json(OUT / "candidates.json", {"n_candidates": len(candidates), "candidates": [public_candidate(item) for item in candidates]})
    write_json(OUT / "pairs.json", {"n_pairs": len(pairs), "pairs": pairs})
    write_json(OUT / "direction_threshold_stats.json", {"combinations": stats})
    write_json(OUT / "boundary_cases.json", {"combinations": boundaries})
    write_json(OUT / "complete_link_exclusions.json", {"n_cases": len(exclusions), "cases": exclusions})
    write_json(OUT / "clips.json", {"n_clips": len(clips), "clips": clips})
    print(f"CANDIDATES {len(candidates)} PAIRS {len(pairs)} CLIPS {len(clips)} EXCLUSIONS {len(exclusions)}")


if __name__ == "__main__":
    main()
