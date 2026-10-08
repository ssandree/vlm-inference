#!/usr/bin/env python3
"""Compile one legacy VIRAT GT bundle into an additive, human-gated v5 bundle.

This command never reads source videos or rewrites source annotations, dense_event_gt,
ROI outputs, or the legacy GT. It requires an explicit video and writes to a separate
versioned output root. Existing per-video output is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_gt_191_v5"
POLICY_V4 = EVAL_ROOT / "experiments" / "virat" / "caption_event_gt_191" / "annotation_policy.json"
POLICY_V5 = EVAL_ROOT / "experiments" / "virat" / "caption_event_gt_191" / "annotation_policy.v5.json"
SOURCE_TYPES = {"virat_activity", "person_track_review", "human_added"}
IMPORTANCE = {"primary", "context"}
ACTION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def semantic_review(review_doc: dict[str, Any]) -> dict[str, Any]:
    state = review_doc.get("semantic_review") or {}
    reviewed = state.get("reviewed_for_visible_events", False)
    by_human = state.get("reviewed_by_human", False)
    if not isinstance(reviewed, bool) or not isinstance(by_human, bool):
        raise ValueError("semantic_review flags must be booleans")
    if reviewed and not by_human:
        raise ValueError("Visible-event completeness cannot be marked reviewed without human review")
    version = state.get("review_version", 1)
    if not isinstance(version, int) or version < 1:
        raise ValueError("semantic_review.review_version must be a positive integer")
    return {
        "reviewed_for_visible_events": reviewed,
        "reviewed_by_human": by_human,
        "review_version": version,
        "reviewed_at": state.get("reviewed_at"),
    }


def _action(value: Any, where: str) -> str:
    if not isinstance(value, str) or not ACTION_PATTERN.fullmatch(value):
        raise ValueError(f"{where} must be a canonical snake_case action string")
    return value


def normalize_human_added_events(
    video_id: str,
    rows: list[dict[str, Any]],
    first_index: int,
    fps: float | None,
) -> list[dict[str, Any]]:
    normalized = []
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Each human_added_events entry must be an object")
        review = row.get("review") or {}
        if review.get("human_verified") is not True:
            raise ValueError(
                "human_added_events must be human-confirmed; keep model/uncertain proposals in candidate_suggestions"
            )
        action = _action(row.get("action"), "human_added_events.action")
        subject = copy.deepcopy(row.get("subject") or {"category": None, "track_id": None})
        if not isinstance(subject, dict):
            raise ValueError("human_added_events.subject must be an object")
        subject_track = subject.get("track_id")
        track_ids = [str(item) for item in row.get("source_track_ids", []) if item is not None]
        if subject_track is not None:
            subject["track_id"] = str(subject_track)
            track_ids.append(str(subject_track))
        track_ids = list(dict.fromkeys(track_ids))
        start_frame = row.get("start_frame")
        end_frame = row.get("end_frame")
        start_time = row.get("start_time")
        end_time = row.get("end_time")
        if fps and start_time is None and start_frame is not None:
            start_time = round(start_frame / fps, 4)
        if fps and end_time is None and end_frame is not None:
            end_time = round(end_frame / fps, 4)
        source_event_id = row.get("human_added_event_id", f"HA{offset + 1:03d}")
        if row.get("virat_event_ids"):
            raise ValueError("human_added evidence cannot claim VIRAT event IDs")
        normalized.append(
            {
                "atomic_evidence_id": f"AE{first_index + offset:03d}",
                "source": {
                    "type": "human_added",
                    "human_added_event_id": str(source_event_id),
                    "virat_event_ids": [],
                    "source_track_ids": track_ids,
                },
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "subject": subject,
                "action": action,
                "object": copy.deepcopy(row.get("object") or []),
                "direction": row.get("direction"),
                "screen_location": copy.deepcopy(row.get("screen_location")),
                "semantic_location": row.get("semantic_location"),
                "caption_evaluable": True,
                "review": {
                    "human_verified": True,
                    "reviewer_id": review.get("reviewer_id"),
                    "reviewed_at": review.get("reviewed_at"),
                    "note": review.get("note", ""),
                },
            }
        )
    return normalized


def _ae_tracks(item: dict[str, Any]) -> list[str]:
    values = (item.get("source") or {}).get("source_track_ids") or []
    subject_track = (item.get("subject") or {}).get("track_id")
    if subject_track is not None:
        values = [*values, subject_track]
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def apply_atomic_evidence_reviews(
    evidence: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {item["atomic_evidence_id"]: item for item in evidence}
    allowed_fields = {
        "action", "subject", "object", "direction", "semantic_location",
        "start_frame", "end_frame", "start_time", "end_time",
    }
    for review in reviews:
        evidence_id = review.get("atomic_evidence_id")
        if evidence_id not in by_id:
            raise ValueError(f"Unknown atomic evidence ID in review: {evidence_id}")
        item = by_id[evidence_id]
        source = item.get("source") or {}
        original_source = copy.deepcopy(source)
        for field in allowed_fields:
            if field in review:
                item[field] = copy.deepcopy(review[field])
        if "source_track_ids" in review:
            source["source_track_ids"] = list(dict.fromkeys(str(value) for value in review["source_track_ids"]))
        if source != original_source:
            raise ValueError("AE review cannot modify source type or source VIRAT provenance")
        review_state = copy.deepcopy(item.get("review") or {})
        human_verified = review.get("human_verified") is True
        review_state.update(
            {
                "human_verified": human_verified,
                "reviewer_id": review.get("reviewer_id"),
                "reviewed_at": review.get("reviewed_at"),
                "note": review.get("note", review_state.get("note", "")),
            }
        )
        item["review"] = review_state
        item["caption_evaluable"] = human_verified
    return evidence


def _event_bounds(members: list[dict[str, Any]], decision: dict[str, Any], fps: float | None) -> dict[str, Any]:
    bounds = {}
    for field in ("start_frame", "end_frame", "start_time", "end_time"):
        values = [item.get(field) for item in members if item.get(field) is not None]
        if field.startswith("start"):
            bounds[field] = min(values) if values else None
        else:
            bounds[field] = max(values) if values else None
        if decision.get(field) is not None:
            bounds[field] = decision[field]
    if fps:
        if bounds["start_time"] is None and bounds["start_frame"] is not None:
            bounds["start_time"] = round(bounds["start_frame"] / fps, 4)
        if bounds["end_time"] is None and bounds["end_frame"] is not None:
            bounds["end_time"] = round(bounds["end_frame"] / fps, 4)
    return bounds


def _verified_participant_relation(decision: dict[str, Any]) -> bool:
    relation = decision.get("participant_relation") or {}
    return bool(relation.get("description")) and relation.get("human_verified") is True


def compile_caption_event(
    decision: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    event_id: str,
    fps: float | None,
) -> dict[str, Any]:
    form = decision.get("event_form")
    if form not in {"individual", "group", "composite"}:
        raise ValueError("event_form must be individual, group, or composite")
    evidence_ids = decision.get("source_atomic_evidence_ids") or []
    if len(evidence_ids) != len(set(evidence_ids)) or not evidence_ids:
        raise ValueError("source_atomic_evidence_ids must be a non-empty unique list")
    try:
        members = [evidence_by_id[item] for item in evidence_ids]
    except KeyError as exc:
        raise ValueError(f"Unknown atomic evidence ID: {exc.args[0]}") from exc
    actions = [_action(item.get("action"), f"{item['atomic_evidence_id']}.action") for item in members]
    tracks_by_member = [_ae_tracks(item) for item in members]
    shared_tracks = set(tracks_by_member[0]).intersection(*(set(items) for items in tracks_by_member[1:])) if len(members) > 1 else set()

    semantics = decision.get("event_semantics") or {}
    primary_action = _action(semantics.get("primary_action"), "event_semantics.primary_action")
    supporting_actions = list(dict.fromkeys(
        _action(item, "event_semantics.supporting_actions")
        for item in semantics.get("supporting_actions", [])
    ))
    human_review = decision.get("human_review") or {}
    human_verified = human_review.get("human_verified") is True
    relation_verified = _verified_participant_relation(decision)
    bounds = _event_bounds(members, decision, fps)

    if form == "individual":
        if len(members) != 1 or primary_action != actions[0] or supporting_actions:
            raise ValueError("individual CE requires exactly one AE and its action as primary_action")
    elif form == "group":
        unique_tracks = {track for tracks in tracks_by_member for track in tracks}
        membership_verified = decision.get("group_membership_verified") is True and human_verified
        if len(members) < 2 or len(set(actions)) != 1 or primary_action != actions[0] or supporting_actions:
            raise ValueError("group CE requires at least two AEs with the same action")
        if len(unique_tracks) < 2 and not membership_verified:
            raise ValueError("group CE requires distinct tracks or human-verified untracked group membership")
    else:
        if len(members) < 2:
            raise ValueError("composite CE requires at least two AEs")
        if primary_action not in actions:
            raise ValueError("composite primary_action must be one of the constituent AE actions")
        expected_support = list(dict.fromkeys(action for action in actions if action != primary_action))
        if set(supporting_actions) != set(expected_support):
            raise ValueError("composite supporting_actions must match the other constituent actions")
        if not shared_tracks and not relation_verified:
            raise ValueError("composite requires a shared track or an explicit human-verified participant_relation")

    importance = decision.get("importance") if human_verified else None
    sentence = decision.get("reference_sentence") if human_verified else None
    if human_verified:
        if importance not in IMPORTANCE:
            raise ValueError("human-verified CE requires importance primary or context")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("human-verified CE requires a human-verified reference_sentence")
        if human_review.get("reference_sentence_verified") is not True:
            raise ValueError("human-verified CE requires reference_sentence_verified")
        if human_review.get("temporal_bounds_verified") is not True:
            raise ValueError("human-verified CE requires temporal_bounds_verified")
        if not (
            bounds["start_time"] is not None and bounds["end_time"] is not None
        ) and not (bounds["start_frame"] is not None and bounds["end_frame"] is not None):
            raise ValueError("human-verified CE requires a complete time or frame interval")
        if form == "composite" and not (shared_tracks or relation_verified):
            raise ValueError("human-verified composite requires a verified participant relation")

    all_tracks = list(dict.fromkeys(track for tracks in tracks_by_member for track in tracks))
    categories = [((item.get("subject") or {}).get("category")) for item in members]
    subject_category = (decision.get("subject") or {}).get("category") or next(
        (value for value in categories if value is not None), None
    )
    if form == "group":
        subject = {"category": subject_category, "group": True, "track_id": None}
        if decision.get("group_membership_verified") is True:
            subject["group_membership_verified"] = True
    else:
        subject_track = next(iter(shared_tracks)) if len(shared_tracks) == 1 else (all_tracks[0] if len(all_tracks) == 1 else None)
        subject = {"category": subject_category, "group": False, "track_id": subject_track}

    action_alias = primary_action
    objects = decision.get("object")
    if objects is None:
        objects = [obj for item in members for obj in item.get("object", [])]
    direction = decision.get("direction")
    if "direction" not in decision:
        directions = {item.get("direction") for item in members}
        direction = next(iter(directions)) if len(directions) == 1 else None
    semantics_out = {"primary_action": primary_action, "supporting_actions": supporting_actions}
    event = {
        "caption_event_id": decision.get("caption_event_id", event_id),
        "event_form": form,
        "source_atomic_evidence_ids": list(evidence_ids),
        "source_track_ids": all_tracks,
        **bounds,
        "subject": subject,
        "action": action_alias,
        "event_semantics": semantics_out,
        "importance": importance,
        "object": copy.deepcopy(objects),
        "direction": direction,
        "semantic_location": decision.get("semantic_location"),
        "reference_sentence": sentence,
        "v5_review": {
            "human_verified": human_verified,
            "status": "verified" if human_verified else "pending_human_verification",
            "reference_sentence_verified": human_review.get("reference_sentence_verified") is True,
            "temporal_bounds_verified": human_review.get("temporal_bounds_verified") is True,
            "reviewer_id": human_review.get("reviewer_id"),
            "reviewed_at": human_review.get("reviewed_at"),
            "participant_relation": copy.deepcopy(decision.get("participant_relation")),
        },
        "evaluation_eligible": False,
    }
    if decision.get("importance") is not None and not human_verified:
        event["proposed_importance"] = decision["importance"]
    if decision.get("reference_sentence") and not human_verified:
        event["proposed_reference_sentence"] = decision["reference_sentence"]
    if form == "group":
        event["person_count_is_required_caption_slot"] = False
    return event


def build_bundle(
    video_id: str,
    source_ae_doc: dict[str, Any],
    source_ce_doc: dict[str, Any],
    review_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review_doc = review_doc or {"video_id": video_id}
    for doc in (source_ae_doc, source_ce_doc, review_doc):
        if doc.get("video_id") not in (None, video_id):
            raise ValueError("Input video_id does not match selected video")
    legacy_evidence = copy.deepcopy(source_ae_doc.get("atomic_evidence") or [])
    legacy_events = copy.deepcopy(source_ce_doc.get("caption_events") or [])
    evidence_ids = set()
    for item in legacy_evidence:
        evidence_id = item.get("atomic_evidence_id")
        if not evidence_id or evidence_id in evidence_ids:
            raise ValueError("Legacy atomic evidence IDs must be present and unique")
        source_type = (item.get("source") or {}).get("type")
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"Unsupported atomic evidence source type: {source_type}")
        evidence_ids.add(evidence_id)
    next_ae_index = max((int(match.group(1)) for item in evidence_ids if (match := re.fullmatch(r"AE(\d+)", item))), default=0) + 1
    fps = review_doc.get("fps")
    if fps is None:
        fps = source_ae_doc.get("fps")
    human_added = normalize_human_added_events(
        video_id,
        review_doc.get("human_added_events") or [],
        next_ae_index,
        fps,
    )
    for item in human_added:
        if item["atomic_evidence_id"] in evidence_ids:
            raise ValueError("Generated human_added AE ID collides with existing evidence")
        evidence_ids.add(item["atomic_evidence_id"])
    all_evidence = legacy_evidence + human_added
    apply_atomic_evidence_reviews(all_evidence, review_doc.get("atomic_evidence_reviews") or [])
    evidence_by_id = {item["atomic_evidence_id"]: item for item in all_evidence}

    used_ce_ids = set()
    caption_events = []
    for event in legacy_events:
        event_id = event.get("caption_event_id")
        if not event_id or event_id in used_ce_ids:
            raise ValueError("Legacy caption event IDs must be present and unique")
        used_ce_ids.add(event_id)
        action = event.get("action")
        event.setdefault("event_semantics", {"primary_action": action, "supporting_actions": []})
        event.setdefault("importance", None)
        member_ids = event.get("source_atomic_evidence_ids") or []
        member_rows = [item for item in legacy_evidence if item["atomic_evidence_id"] in member_ids]
        evidence_human_verified = bool(member_rows) and all(
            item.get("review", {}).get("human_verified") is True
            and item.get("action") == action
            for item in member_rows
        )
        event["v5_review"] = {
            "human_verified": False,
            "evidence_human_verified": evidence_human_verified,
            "temporal_bounds_verified": False,
            "reference_sentence_verified": False,
            "status": "awaiting_v5_event_review" if evidence_human_verified else "legacy_unverified",
        }
        event["evaluation_eligible"] = False
        caption_events.append(event)

    next_ce_index = max((int(match.group(1)) for item in used_ce_ids if (match := re.fullmatch(r"CE(\d+)", item))), default=0) + 1
    verified_decisions = []
    unverified_decisions = []
    for decision in review_doc.get("caption_event_decisions") or []:
        proposed_id = decision.get("caption_event_id")
        event_id = proposed_id or f"CE{next_ce_index:03d}"
        if not proposed_id:
            next_ce_index += 1
        is_update = event_id in used_ce_ids
        if is_update:
            caption_events = [item for item in caption_events if item["caption_event_id"] != event_id]
        else:
            used_ce_ids.add(event_id)
        event = compile_caption_event(decision, evidence_by_id, event_id, fps)
        if not all(evidence_by_id[item].get("review", {}).get("human_verified") is True for item in event["source_atomic_evidence_ids"]):
            raise ValueError("Caption events may only use human-verified atomic evidence")
        if event["v5_review"]["human_verified"]:
            verified_decisions.append(event)
        else:
            unverified_decisions.append(event)
    caption_events.extend(verified_decisions)

    state = semantic_review(review_doc)
    for event in caption_events:
        v5_review = event.get("v5_review") or {}
        eligible = (
            state["reviewed_for_visible_events"]
            and state["reviewed_by_human"]
            and v5_review.get("human_verified") is True
            and v5_review.get("temporal_bounds_verified") is True
            and v5_review.get("reference_sentence_verified") is True
            and event.get("importance") in IMPORTANCE
            and isinstance(event.get("reference_sentence"), str)
            and bool(event["reference_sentence"].strip())
        )
        event["evaluation_eligible"] = eligible

    human_added_candidates = [
        {
            "candidate_id": f"CC{index:03d}",
            "event_form": "individual",
            "source_atomic_evidence_ids": [item["atomic_evidence_id"]],
            "candidate_only": True,
            "human_review_required": True,
        }
        for index, item in enumerate(human_added, 1)
    ]
    return {
        "video_id": video_id,
        "version": "caption_event_gt_v5",
        "semantic_review": state,
        "atomic_evidence": all_evidence,
        "caption_events": caption_events,
        "caption_event_candidates": human_added_candidates + unverified_decisions,
        "candidate_suggestions": copy.deepcopy(review_doc.get("candidate_suggestions") or []),
        "evaluation_eligible_caption_event_ids": [
            item["caption_event_id"] for item in caption_events if item["evaluation_eligible"]
        ],
    }


def load_policy() -> dict[str, Any]:
    base = read_json(POLICY_V4)
    extension = read_json(POLICY_V5)
    if extension.get("version") != 5 or extension.get("inherits") != POLICY_V4.name:
        raise ValueError("v5 policy must explicitly inherit the existing 191-video policy")
    return deep_merge(base, extension)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="Compile exactly this one video")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing atomic_evidence.json and caption_events.json")
    parser.add_argument("--review-file", type=Path, help="Optional human-authored v5 review JSON; model suggestions remain candidates")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_id = args.video_id
    source_ae_doc = read_json(args.source_dir / "atomic_evidence.json")
    source_ce_doc = read_json(args.source_dir / "caption_events.json")
    review_doc = read_json(args.review_file) if args.review_file else {"video_id": video_id}
    bundle = build_bundle(video_id, source_ae_doc, source_ce_doc, review_doc)

    video_out = args.out_root / "gt" / video_id
    if video_out.exists() and any(video_out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing v5 video output: {video_out}")
    policy_out = args.out_root / "annotation_policy.json"
    if not policy_out.exists():
        write_json(policy_out, load_policy())
    write_json(
        video_out / "atomic_evidence.json",
        {
            "video_id": video_id,
            "version": "caption_event_gt_v5",
            "semantic_review": bundle["semantic_review"],
            "atomic_evidence": bundle["atomic_evidence"],
        },
    )
    write_json(
        video_out / "caption_events.json",
        {
            "video_id": video_id,
            "version": "caption_event_gt_v5",
            "semantic_review": bundle["semantic_review"],
            "caption_events": bundle["caption_events"],
            "caption_event_candidates": bundle["caption_event_candidates"],
            "candidate_suggestions": bundle["candidate_suggestions"],
            "evaluation_eligible_caption_event_ids": bundle["evaluation_eligible_caption_event_ids"],
        },
    )
    print(
        f"WROTE {video_id} AE={len(bundle['atomic_evidence'])} "
        f"CE={len(bundle['caption_events'])} "
        f"CE_CANDIDATES={len(bundle['caption_event_candidates'])} "
        f"EVAL_ELIGIBLE={len(bundle['evaluation_eligible_caption_event_ids'])}"
    )


if __name__ == "__main__":
    main()
