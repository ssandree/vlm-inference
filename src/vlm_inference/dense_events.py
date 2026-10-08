"""Lossless event proposal parsing and temporal review pairing, not GT merging.

This module deliberately has no model, detection, or tracking dependencies.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ALIASES = {
    "walking": "walking", "walk": "walking", "locomotion": "walking",
    "running": "running", "run": "running",
    "entering_vehicle": "getting_into_vehicle", "exiting_vehicle": "getting_out_of_vehicle",
    "loading_object_into_vehicle": "loading_object_to_vehicle",
    "unloading_object_from_vehicle": "unloading_object_from_vehicle",
    "entering_building": "entering_facility", "exiting_building": "exiting_facility",
}
REVIEW_FIELDS = [
    "video_id", "candidate_id", "source", "start_sec", "end_sec", "action", "subject",
    "object_or_target", "description", "qwen_confidence", "matched_annotation_id",
    "annotation_action", "annotation_start_sec", "annotation_end_sec", "match_status",
    "human_verdict", "corrected_start_sec", "corrected_end_sec", "corrected_action", "human_note",
    "subject_id", "temporal_confidence", "start_range_sec", "end_range_sec", "uncertainty_note",
    "temporal_iou", "start_delta_sec", "end_delta_sec", "possible_annotation_ids", "review_flags",
]


def write_json(path: Path, value) -> None:
    """Never replace an existing artifact, including a human's review answers."""
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def canonical_action(action: str) -> str:
    name = re.sub(r"[\s-]+", "_", action.strip().lower())
    if name.startswith("person_"):
        name = name[7:]
    return ALIASES.get(name, name)


def numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_events(raw: str, duration: float) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        doc = json.loads(text)
    except (ValueError, TypeError) as exc:
        return {"events": [], "rejected_events": [], "errors": [str(exc)], "status": "invalid_json"}
    rows = doc.get("events") if isinstance(doc, dict) else doc
    if not isinstance(rows, list):
        return {"events": [], "rejected_events": [], "errors": ["Expected events array"], "status": "invalid_schema"}
    valid, rejected, seen = [], [], set()
    required = ("event_id", "start_sec", "end_sec", "subject", "action", "object_or_target", "location", "description", "confidence")
    for index, row in enumerate(rows):
        errors = []
        if not isinstance(row, dict):
            rejected.append({"index": index, "original": row, "errors": ["Not an object"]})
            continue
        errors += [f"Missing {k}" for k in required if k not in row]
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            errors.append("Invalid or duplicate event_id")
        for k in ("start_sec", "end_sec", "confidence"):
            if not numeric(row.get(k)):
                errors.append(f"{k} must be a finite number")
        if numeric(row.get("start_sec")) and numeric(row.get("end_sec")):
            if not 0 <= row["start_sec"] <= row["end_sec"] <= duration:
                errors.append("Boundary out of video range or reversed; never clamped")
        if numeric(row.get("confidence")) and not 0 <= row["confidence"] <= 1:
            errors.append("confidence outside [0,1]")
        for k in ("subject", "action", "description"):
            if not isinstance(row.get(k), str) or not row[k].strip():
                errors.append(f"{k} must be nonempty text")
        for k in ("object_or_target", "location", "subject_id", "uncertainty_note"):
            if row.get(k) is not None and not isinstance(row[k], str):
                errors.append(f"{k} must be text or null")
        for k, point in (("start_range_sec", "start_sec"), ("end_range_sec", "end_sec")):
            r = row.get(k)
            if r is not None and (not isinstance(r, list) or len(r) != 2 or not all(numeric(x) for x in r) or not numeric(row.get(point)) or not 0 <= r[0] <= row[point] <= r[1] <= duration):
                errors.append(f"Invalid {k}")
        tc = row.get("temporal_confidence")
        if tc is not None and (not numeric(tc) or not 0 <= tc <= 1):
            errors.append("Invalid temporal_confidence")
        if errors:
            rejected.append({"index": index, "original": row, "errors": errors})
            continue
        seen.add(event_id)
        event = dict(row)
        event["original_action"] = row["action"]
        event["action"] = canonical_action(row["action"])
        valid.append(event)
    return {"events": valid, "rejected_events": rejected, "errors": [], "status": "partial" if rejected else "ok"}


def interval_stats(left: dict, right: dict) -> dict:
    ls, le, rs, re_ = left["start_sec"], left["end_sec"], right["start_sec"], right["end_sec"]
    overlap = max(0.0, min(le, re_) - max(ls, rs))
    union = max(le, re_) - min(ls, rs)
    shorter = min(le - ls, re_ - rs)
    return {"overlap_sec": overlap, "temporal_iou": overlap / union if union > 0 else float(ls == rs),
            "overlap_over_shorter": overlap / shorter if shorter > 0 else 0.0,
            "gap_sec": max(0.0, max(ls, rs) - min(le, re_)),
            "start_delta_sec": ls - rs, "end_delta_sec": le - re_}


def compare_events(video_id: str, events: list[dict], annotations: list[dict], *, min_iou: float = .1, min_shorter_overlap: float = .5, nearby_sec: float = 5.0) -> dict:
    """Greedy one-to-one pairing; all other edges retained for manual inspection.

    Actor correspondence is unknown. Broad/ambiguous matches are NOT confirmations.
    Nearby different actions and non-overlap pairs remain visible without matching.
    """
    edges, nearby = [], {}
    for qi, q in enumerate(events):
        candidates = []
        for ai, a in enumerate(annotations):
            stats = interval_stats(q, a)
            same = canonical_action(q["action"]) == canonical_action(a["action"])
            eligible = same and stats["overlap_sec"] > 0 and stats["temporal_iou"] >= min_iou and stats["overlap_over_shorter"] >= min_shorter_overlap
            if stats["gap_sec"] <= nearby_sec:
                candidates.append({"annotation_id": a["event_id"], "action": a["action"], "start_sec": a["start_sec"], "end_sec": a["end_sec"], "subject": a.get("subject"), "same_action": same, "eligible_pair": eligible, **stats})
            if eligible:
                edges.append((stats["temporal_iou"], stats["overlap_over_shorter"], qi, ai, stats))
        nearby[q["event_id"]] = sorted(candidates, key=lambda x: (not x["same_action"], -x["temporal_iou"], x["gap_sec"]))
    assignments, used_q, used_a = {}, set(), set()
    for _, _, qi, ai, stats in sorted(edges, key=lambda x: (-x[0], -x[1], x[2], x[3])):
        if qi not in used_q and ai not in used_a:
            assignments[qi] = (ai, stats); used_q.add(qi); used_a.add(ai)
    rows, comparisons = [], []
    for qi, q in enumerate(events):
        matched = assignments.get(qi)
        a = annotations[matched[0]] if matched else None
        candidates = nearby[q["event_id"]]
        flags = ["actor_correspondence_unverified"] if a else []
        eligible_ids = [x["annotation_id"] for x in candidates if x["eligible_pair"]]
        if len(eligible_ids) > 1:
            flags.append("multiple_annotation_candidates_possible_event_conflation")
        if matched and matched[1]["temporal_iou"] < .5:
            flags.append("low_temporal_iou_check_boundaries")
        if q.get("original_action") == "locomotion":
            flags.append("broad_locomotion_alias_review")
        if q.get("temporal_confidence") is None or q.get("start_range_sec") is None or q.get("end_range_sec") is None:
            flags.append("missing_boundary_uncertainty")
        row = {k: "" for k in REVIEW_FIELDS}
        row.update(video_id=video_id, candidate_id=q["event_id"], source="qwen", start_sec=q["start_sec"], end_sec=q["end_sec"], action=q["action"], subject=q["subject"], object_or_target=q.get("object_or_target"), description=q["description"], qwen_confidence=q["confidence"], match_status="matched" if a else "qwen_only", subject_id=q.get("subject_id"), temporal_confidence=q.get("temporal_confidence"), start_range_sec=q.get("start_range_sec"), end_range_sec=q.get("end_range_sec"), uncertainty_note=q.get("uncertainty_note"), possible_annotation_ids=eligible_ids, review_flags=flags)
        if a:
            row.update(matched_annotation_id=a["event_id"], annotation_action=a["action"], annotation_start_sec=a["start_sec"], annotation_end_sec=a["end_sec"])
            row.update({k: matched[1][k] for k in ("temporal_iou", "start_delta_sec", "end_delta_sec")})
        rows.append(row)
        comparisons.append({"qwen": q, "paired_annotation": a, "match_status": row["match_status"], "nearby_annotations": candidates, "review_flags": flags})
    for ai, a in enumerate(annotations):
        if ai in used_a:
            continue
        row = {k: "" for k in REVIEW_FIELDS}
        row.update(video_id=video_id, candidate_id="ANN-" + a["event_id"], source="annotation", start_sec=a["start_sec"], end_sec=a["end_sec"], action=a["action"], subject=a.get("subject"), object_or_target=a.get("object_or_target"), description=a.get("description", ""), matched_annotation_id=a["event_id"], annotation_action=a["action"], annotation_start_sec=a["start_sec"], annotation_end_sec=a["end_sec"], match_status="annotation_only", review_flags=["annotation_incomplete_or_duplicate_possible", "not_proof_of_qwen_error"])
        rows.append(row)
    return {"purpose": "Human review candidates only; no GT merge or adjudication", "matching": {"method": "greedy_one_to_one_action_and_temporal_overlap", "min_iou": min_iou, "min_shorter_overlap": min_shorter_overlap, "nearby_sec": nearby_sec, "actor_matching": False}, "counts": {"qwen_events": len(events), "annotation_events": len(annotations), "matched": len(used_q), "qwen_only": len(events) - len(used_q), "annotation_only": len(annotations) - len(used_a)}, "comparisons": comparisons, "review_rows": rows}


def write_review_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
