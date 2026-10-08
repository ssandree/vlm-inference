#!/usr/bin/env python3
"""Minimal one-video human annotation UI for the v5 VIRAT event GT."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

EVAL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EVAL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_caption_event_gt_v5 as v5

VIDEO_ID = "VIRAT_S_010001_02_000195_000498"
V5_ROOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_gt_191_v5"
DEFAULT_SOURCE = V5_ROOT / "gt" / VIDEO_ID
DEFAULT_PILOT_REVIEW = EVAL_ROOT / "experiments" / "virat" / "caption_event_pilot_6" / VIDEO_ID / "review.json"
HUMAN_REVIEW_ROOT = V5_ROOT / "human_review"
CANONICAL_ACTIONS = [
    "carrying_object", "closing_vehicle_trunk", "entering_facility", "exiting_facility",
    "getting_into_vehicle", "getting_out_of_vehicle", "gesturing", "loading_object_to_vehicle",
    "opening_vehicle_trunk", "running", "unloading_object_from_vehicle", "walking",
    "wandering", "riding_bicycle", "vehicle_moving", "crossing_road",
]
DIRECTIONS = [
    "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top",
    "upper_left_to_lower_right", "upper_right_to_lower_left",
    "lower_left_to_upper_right", "lower_right_to_upper_left",
]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return copy.deepcopy(default)
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seconds(value: Any) -> float:
    return 0.0 if value is None else float(value)


def frame_at(timestamp: float, fps: float) -> int:
    return max(0, round(timestamp * fps))


def timestamp_range(label: str, start: Any, end: Any, duration: float, key: str) -> tuple[float, float]:
    left, right = st.columns(2)
    start_value = left.number_input(
        f"{label} start (video seconds)", min_value=0.0, max_value=max(duration, 0.0),
        value=min(max(seconds(start), 0.0), max(duration, 0.0)), step=0.1, key=f"{key}_start",
    )
    end_value = right.number_input(
        f"{label} end (video seconds)", min_value=0.0, max_value=max(duration, 0.0),
        value=min(max(seconds(end), 0.0), max(duration, 0.0)), step=0.1, key=f"{key}_end",
    )
    return float(start_value), float(end_value)


def parse_json_list(text: str, field_name: str) -> list:
    value = json.loads(text or "[]")
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return value


def action_editor(label: str, current: str | None, actions: list[str], key: str) -> str:
    current = current or ""
    options = list(dict.fromkeys([*actions, current, "__custom__"]))
    default = current if current in options else "__custom__"
    selected = st.selectbox(label, options, index=options.index(default), key=f"{key}_choice", format_func=lambda value: "Custom action..." if value == "__custom__" else value)
    if selected == "__custom__":
        return st.text_input(f"{label} (canonical snake_case)", value=current if current != "__custom__" else "", key=f"{key}_custom").strip()
    return selected


def initial_review(video_id: str, fps: float) -> dict[str, Any]:
    path = HUMAN_REVIEW_ROOT / video_id / "review.json"
    loaded = read_json(path, {})
    if loaded:
        return loaded
    return {
        "video_id": video_id,
        "fps": fps,
        "human_added_events": [],
        "atomic_evidence_reviews": [],
        "caption_event_decisions": [],
        "candidate_suggestions": [],
        "semantic_review": {
            "reviewed_for_visible_events": False,
            "reviewed_by_human": False,
            "review_version": 1,
            "reviewed_at": None,
        },
    }


def default_review_paths() -> tuple[Path, Path, Path]:
    return DEFAULT_SOURCE, DEFAULT_PILOT_REVIEW, V5_ROOT / "gt" / VIDEO_ID / "atomic_evidence.json"


st.set_page_config(page_title="VIRAT Semantic Review", layout="wide")
st.title("VIRAT Semantic Event Review")
st.caption("Single-video pilot. Human review data and compiled v5 GT are stored separately.")

source_dir, default_review_path, _ = default_review_paths()
with st.sidebar:
    st.header("Video")
    video_id = st.text_input("Video ID", value=VIDEO_ID, disabled=True)
    source_dir_text = st.text_input("v5 source bundle", value=str(source_dir))
    review_source_text = st.text_input("Video metadata source", value=str(default_review_path))
    video_path_text = st.text_input("Video file", value="")
    video_id = video_id.strip()
    source_dir = Path(source_dir_text).expanduser()
    review_source_path = Path(review_source_text).expanduser()

if video_id != VIDEO_ID:
    st.error(f"This pilot interface is limited to {VIDEO_ID}.")
    st.stop()

try:
    source_ae_doc = v5.read_json(source_dir / "atomic_evidence.json")
    source_ce_doc = v5.read_json(source_dir / "caption_events.json")
except (OSError, ValueError) as error:
    st.error(f"Could not load the v5 source bundle: {error}")
    st.stop()

if source_ae_doc.get("video_id") != video_id or source_ce_doc.get("video_id") != video_id:
    st.error("The selected video ID must match both v5 source files.")
    st.stop()

metadata = read_json(review_source_path, {})
if metadata.get("video_id") not in (None, video_id):
    st.error("The metadata file's video ID does not match the selected video.")
    st.stop()
fps = float(metadata.get("fps") or 0.0)
duration = float(metadata.get("duration_sec") or 0.0)
video_path = Path(video_path_text).expanduser() if video_path_text.strip() else Path(metadata.get("video_path") or "")
if fps <= 0:
    st.error("A positive FPS value is required in the video metadata source.")
    st.stop()
if duration <= 0:
    duration = float(metadata.get("nb_frames", 0)) / fps
review_path = HUMAN_REVIEW_ROOT / video_id / "review.json"

if "review_doc" not in st.session_state or st.session_state.get("active_video_id") != video_id:
    st.session_state.review_doc = initial_review(video_id, fps)
    st.session_state.active_video_id = video_id
review_doc = st.session_state.review_doc
review_doc["video_id"] = video_id
review_doc["fps"] = fps

header_left, header_right = st.columns([3, 2])
with header_left:
    st.write(f"**FPS:** {fps:g}   **Duration:** {duration:.2f} s")
    if video_path.is_file():
        playback_start = st.number_input("Playback start (video seconds)", min_value=0.0, max_value=max(duration, 0.0), value=0.0, step=1.0)
        st.video(str(video_path), start_time=int(playback_start))
    else:
        st.warning(f"Video file not found: {video_path}. Set the correct path in the sidebar.")
with header_right:
    st.subheader("Current evidence")
    st.caption("Existing VIRAT/pilot records are editable semantic copies. Source provenance is read-only.")

atomic_evidence = source_ae_doc.get("atomic_evidence") or []
review_by_ae = {row["atomic_evidence_id"]: row for row in review_doc.get("atomic_evidence_reviews", [])}
human_added_rows = review_doc.get("human_added_events", [])
human_added_evidence = []
for index, row in enumerate(human_added_rows, 1):
    row["human_added_event_id"] = row.get("human_added_event_id", f"HA{index:03d}")
    ae_id = f"AE{len(atomic_evidence) + index:03d}"
    human_added_evidence.append(
        {
            "atomic_evidence_id": ae_id,
            "source": {"type": "human_added", "human_added_event_id": row["human_added_event_id"], "virat_event_ids": [], "source_track_ids": row.get("source_track_ids", [])},
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "subject": row.get("subject") or {},
            "action": row.get("action"),
            "object": row.get("object") or [],
            "direction": row.get("direction"),
            "semantic_location": row.get("semantic_location"),
            "review": row.get("review") or {},
        }
    )
    review_by_ae.setdefault(
        ae_id,
        {
            "atomic_evidence_id": ae_id,
            "action": row.get("action"),
            "subject": row.get("subject"),
            "object": row.get("object", []),
            "direction": row.get("direction"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "human_verified": (row.get("review") or {}).get("human_verified") is True,
            "reviewer_id": (row.get("review") or {}).get("reviewer_id"),
            "note": (row.get("review") or {}).get("note", ""),
        },
    )
action_options = sorted(set(CANONICAL_ACTIONS) | {str(row.get("action")) for row in atomic_evidence if row.get("action")})

st.subheader(f"Atomic Evidence ({len(atomic_evidence) + len(human_added_rows)})")
with st.container():
    displayed_evidence = []
    for row in [*atomic_evidence, *human_added_evidence]:
        existing = review_by_ae.get(row["atomic_evidence_id"], {})
        displayed_evidence.append({**row, **existing, "source": row.get("source")})
    st.dataframe(
        [
            {
                "AE": row.get("atomic_evidence_id"),
                "time": f"{row.get('start_time')}–{row.get('end_time')} s",
                "subject": f"{(row.get('subject') or {}).get('category')} / {(row.get('subject') or {}).get('track_id')}",
                "action": row.get("action"),
                "object": json.dumps(row.get("object") or [], ensure_ascii=False),
                "direction": row.get("direction"),
                "source": (row.get("source") or {}).get("type"),
                "human verified": (row.get("human_verified") if "human_verified" in row else (row.get("review") or {}).get("human_verified", False)),
            }
            for row in displayed_evidence
        ],
        width="stretch",
        hide_index=True,
    )
    for row in [*atomic_evidence, *human_added_evidence]:
        evidence_id = row["atomic_evidence_id"]
        existing = review_by_ae.get(evidence_id, {})
        source = row.get("source") or {}
        with st.expander(f"{evidence_id} · {row.get('action') or 'unlabeled'} · {row.get('start_time')}–{row.get('end_time')} s", expanded=False):
            st.caption(f"Source type: {source.get('type')} | VIRAT event IDs: {source.get('virat_event_ids', [])} | source tracks: {source.get('source_track_ids', [])}")
            with st.form(f"ae_form_{evidence_id}"):
                start, end = timestamp_range("AE time", existing.get("start_time", row.get("start_time")), existing.get("end_time", row.get("end_time")), duration, f"ae_{evidence_id}")
                subject_original = existing.get("subject", row.get("subject") or {})
                c1, c2 = st.columns(2)
                category = c1.text_input("Subject category", value=subject_original.get("category") or "", key=f"ae_{evidence_id}_category")
                track = c2.text_input("Subject track ID", value=str(subject_original.get("track_id") or ""), key=f"ae_{evidence_id}_track")
                action = action_editor("Action", existing.get("action", row.get("action")), action_options, f"ae_{evidence_id}_action")
                objects = st.text_area("Object (JSON array)", value=json.dumps(existing.get("object", row.get("object") or []), ensure_ascii=False), key=f"ae_{evidence_id}_object")
                direction = st.selectbox("Direction", [""] + DIRECTIONS, index=([""] + DIRECTIONS).index(existing.get("direction", row.get("direction")) or ""), key=f"ae_{evidence_id}_direction")
                semantic_location = st.text_input("Semantic location", value=existing.get("semantic_location", row.get("semantic_location")) or "", key=f"ae_{evidence_id}_location")
                verified_default = existing.get("human_verified", (row.get("review") or {}).get("human_verified", False))
                human_verified = st.checkbox("Human verified visible actor/action", value=bool(verified_default), key=f"ae_{evidence_id}_verified", disabled=source.get("type") == "human_added")
                note = st.text_input("Review note", value=existing.get("note", (row.get("review") or {}).get("note", "")), key=f"ae_{evidence_id}_note")
                submitted = st.form_submit_button("Save AE review")
            if submitted:
                try:
                    subject = {"category": category.strip() or None, "track_id": track.strip() or None}
                    review_by_ae[evidence_id] = {
                        "atomic_evidence_id": evidence_id,
                        "start_time": start,
                        "end_time": end,
                        "start_frame": frame_at(start, fps),
                        "end_frame": frame_at(end, fps),
                        "subject": subject,
                        "action": action,
                        "object": parse_json_list(objects, "object"),
                        "direction": direction or None,
                        "semantic_location": semantic_location.strip() or None,
                        "human_verified": human_verified,
                        "reviewer_id": "human_annotator",
                        "reviewed_at": now_iso() if human_verified else None,
                        "note": note,
                    }
                    st.session_state.review_doc["atomic_evidence_reviews"] = list(review_by_ae.values())
                    st.success(f"Saved review state for {evidence_id}. Use Save review JSON to persist it.")
                except (ValueError, json.JSONDecodeError) as error:
                    st.error(str(error))

    st.markdown("#### Add human-observed evidence")
    with st.form("human_added_form", clear_on_submit=True):
        ha_start, ha_end = timestamp_range("New AE time", 0.0, 0.0, duration, "human_added")
        h1, h2, h3 = st.columns(3)
        ha_category = h1.text_input("Actor category", value="person")
        ha_track = h2.text_input("Actor track ID", value="")
        ha_direction = h3.selectbox("Direction ", [""] + DIRECTIONS, key="human_added_direction")
        ha_action = action_editor("New action", None, action_options, "human_added_action")
        ha_object = st.text_area("Objects (JSON array)", value="[]", key="human_added_objects")
        ha_location = st.text_input("Semantic location (optional)")
        ha_verified = st.checkbox("I verified this visible event in the video", value=False, key="human_added_verified")
        ha_note = st.text_input("Annotation note")
        add_submitted = st.form_submit_button("Add human_added AE")
    if add_submitted:
        if not ha_verified:
            st.error("Human verification is required before adding human_added evidence.")
        elif not ha_action:
            st.error("Enter an action using canonical snake_case.")
        elif ha_end < ha_start:
            st.error("End time must be at or after start time.")
        else:
            try:
                rows = st.session_state.review_doc.setdefault("human_added_events", [])
                rows.append(
                    {
                        "human_added_event_id": f"HA{len(rows) + 1:03d}",
                        "action": ha_action,
                        "subject": {"category": ha_category.strip() or None, "track_id": ha_track.strip() or None},
                        "source_track_ids": [ha_track.strip()] if ha_track.strip() else [],
                        "object": parse_json_list(ha_object, "objects"),
                        "direction": ha_direction or None,
                        "semantic_location": ha_location.strip() or None,
                        "start_time": ha_start,
                        "end_time": ha_end,
                        "start_frame": frame_at(ha_start, fps),
                        "end_frame": frame_at(ha_end, fps),
                        "review": {"human_verified": True, "reviewer_id": "human_annotator", "reviewed_at": now_iso(), "note": ha_note},
                    }
                )
                st.success("Human-added AE captured. Save review JSON to persist it.")
            except (ValueError, json.JSONDecodeError) as error:
                st.error(str(error))

st.divider()
st.subheader("Caption Events")
current_evidence_reviews = {row["atomic_evidence_id"]: row for row in st.session_state.review_doc.get("atomic_evidence_reviews", [])}
evidence_by_id = {row["atomic_evidence_id"]: row for row in atomic_evidence}
all_ae_ids = list(evidence_by_id)
for index, row in enumerate(st.session_state.review_doc.get("human_added_events", []), 1):
    all_ae_ids.append(f"AE{len(atomic_evidence) + index:03d}")
existing_ce = source_ce_doc.get("caption_events") or []
ce_review_rows = st.session_state.review_doc.get("caption_event_decisions", [])
ce_reviews_by_id = {row.get("caption_event_id"): row for row in ce_review_rows if row.get("caption_event_id")}
ce_options = [row["caption_event_id"] for row in existing_ce]
ce_options.extend(row for row in ce_reviews_by_id if row not in ce_options)
ce_selector_options = ["New caption event", *ce_options]
selected_ce = st.selectbox("Edit CE", ce_selector_options)
base_ce = next((row for row in existing_ce if row["caption_event_id"] == selected_ce), {})
current_decision = ce_reviews_by_id.get(selected_ce, {}) if selected_ce != "New caption event" else {}
default_ae_ids = current_decision.get("source_atomic_evidence_ids", base_ce.get("source_atomic_evidence_ids", []))

with st.form("caption_event_form"):
    selected_ae_ids = st.multiselect("Source atomic evidence (provenance)", all_ae_ids, default=[item for item in default_ae_ids if item in all_ae_ids])
    event_forms = ["individual", "group", "composite"]
    current_form = current_decision.get("event_form", base_ce.get("event_form", "individual"))
    form = st.selectbox("Event form", event_forms, index=event_forms.index(current_form) if current_form in event_forms else 0)
    chosen_ae = []
    for ae_id in selected_ae_ids:
        if ae_id in evidence_by_id:
            chosen_ae.append(current_evidence_reviews.get(ae_id, evidence_by_id[ae_id]))
        else:
            offset = int(ae_id[2:]) - len(atomic_evidence) - 1
            if 0 <= offset < len(st.session_state.review_doc.get("human_added_events", [])):
                chosen_ae.append(st.session_state.review_doc["human_added_events"][offset])
    default_start = min([row.get("start_time") for row in chosen_ae if row.get("start_time") is not None] or [base_ce.get("start_time", 0.0)])
    default_end = max([row.get("end_time") for row in chosen_ae if row.get("end_time") is not None] or [base_ce.get("end_time", 0.0)])
    ce_start, ce_end = timestamp_range("CE time", current_decision.get("start_time", base_ce.get("start_time", default_start)), current_decision.get("end_time", base_ce.get("end_time", default_end)), duration, "ce")
    member_actions = list(dict.fromkeys(str(row.get("action")) for row in chosen_ae if row.get("action")))
    action_options_ce = sorted(set(CANONICAL_ACTIONS + member_actions))
    primary_action = action_editor("Primary action", (current_decision.get("event_semantics") or {}).get("primary_action", base_ce.get("action")), action_options_ce, "ce_primary")
    support_text = st.text_input("Supporting actions (comma-separated canonical actions)", value=", ".join((current_decision.get("event_semantics") or {}).get("supporting_actions", [])))
    importance_options = ["", "primary", "context"]
    current_importance = current_decision.get("importance", base_ce.get("importance")) or ""
    importance = st.selectbox("Importance", importance_options, index=importance_options.index(current_importance) if current_importance in importance_options else 0)
    sentence = st.text_area("Human-authored reference sentence", value=current_decision.get("reference_sentence", base_ce.get("reference_sentence") or ""))
    ce_direction_options = [""] + DIRECTIONS
    current_direction = current_decision.get("direction", base_ce.get("direction")) or ""
    ce_direction = st.selectbox("Direction", ce_direction_options, index=ce_direction_options.index(current_direction) if current_direction in ce_direction_options else 0)
    group_membership_verified = st.checkbox("Human verified group membership (needed if tracks are unavailable)", value=current_decision.get("group_membership_verified", False))
    relation_description = st.text_input("Composite participant relation (optional)", value=(current_decision.get("participant_relation") or {}).get("description", ""))
    relation_verified = st.checkbox("Human verified participant relation", value=(current_decision.get("participant_relation") or {}).get("human_verified", False))
    ce_verified = st.checkbox("Human verified this caption event, wording, and event interpretation", value=(current_decision.get("human_review") or {}).get("human_verified", False))
    ce_submit = st.form_submit_button("Save CE review")

if ce_submit:
    if not selected_ae_ids:
        st.error("Select at least one AE to preserve provenance.")
    elif ce_end < ce_start:
        st.error("CE end time must be at or after start time.")
    elif any(not (current_evidence_reviews.get(ae_id, {}).get("human_verified", evidence_by_id.get(ae_id, {}).get("review", {}).get("human_verified", False))) for ae_id in selected_ae_ids if ae_id in evidence_by_id):
        st.error("Approve every selected existing AE before using it in a CE.")
    elif not importance and ce_verified:
        st.error("Choose primary or context before verifying a CE.")
    elif not sentence.strip() and ce_verified:
        st.error("A human-verified CE requires a reference sentence.")
    else:
        ce_id = selected_ce if selected_ce != "New caption event" else None
        review_state = {
            "human_verified": ce_verified,
            "reference_sentence_verified": ce_verified,
            "temporal_bounds_verified": ce_verified,
            "reviewer_id": "human_annotator",
            "reviewed_at": now_iso() if ce_verified else None,
        }
        decision = {
            "caption_event_id": ce_id,
            "event_form": form,
            "source_atomic_evidence_ids": selected_ae_ids,
            "start_time": ce_start,
            "end_time": ce_end,
            "event_semantics": {"primary_action": primary_action, "supporting_actions": [item.strip() for item in support_text.split(",") if item.strip()]},
            "importance": importance or None,
            "reference_sentence": sentence.strip() or None,
            "direction": ce_direction or None,
            "group_membership_verified": group_membership_verified,
            "participant_relation": {"description": relation_description.strip(), "human_verified": relation_verified} if relation_description.strip() else None,
            "human_review": review_state,
        }
        existing_decisions = st.session_state.review_doc.setdefault("caption_event_decisions", [])
        if ce_id:
            existing_decisions[:] = [item for item in existing_decisions if item.get("caption_event_id") != ce_id]
        existing_decisions.append(decision)
        st.success("CE review captured. Use Save review JSON to persist it.")

st.divider()
st.subheader("Video Semantic Completeness")
semantic = st.session_state.review_doc.setdefault("semantic_review", {"reviewed_for_visible_events": False, "reviewed_by_human": False, "review_version": 1, "reviewed_at": None})
semantic["reviewed_for_visible_events"] = st.checkbox(
    "I reviewed the full video for visible semantic events, including actions outside VIRAT taxonomy and background vehicle activity.",
    value=bool(semantic.get("reviewed_for_visible_events", False)),
)
semantic["reviewed_by_human"] = st.checkbox(
    "I am the human reviewer completing this video review.",
    value=bool(semantic.get("reviewed_by_human", False)),
)
semantic["review_version"] = st.number_input("Review version", min_value=1, step=1, value=int(semantic.get("review_version", 1)))

save_left, compile_right = st.columns(2)
with save_left:
    if st.button("Save human review JSON", type="primary"):
        semantic["reviewed_at"] = now_iso() if semantic["reviewed_by_human"] and semantic["reviewed_for_visible_events"] else None
        review_doc["semantic_review"] = semantic
        review_doc["atomic_evidence_reviews"] = list({row["atomic_evidence_id"]: row for row in review_doc.get("atomic_evidence_reviews", [])}.values())
        write_json(review_path, review_doc)
        st.success(f"Human review saved: {review_path}")
with compile_right:
    st.caption("Compile uses the existing v5 validation and evaluation gate. Each compile is immutable and stored in a new review-version directory.")
    if st.button("Compile v5 GT and validate eligibility"):
        semantic["reviewed_at"] = now_iso() if semantic["reviewed_by_human"] and semantic["reviewed_for_visible_events"] else None
        review_doc["semantic_review"] = semantic
        review_doc["atomic_evidence_reviews"] = list({row["atomic_evidence_id"]: row for row in review_doc.get("atomic_evidence_reviews", [])}.values())
        write_json(review_path, review_doc)
        try:
            bundle = v5.build_bundle(video_id, source_ae_doc, source_ce_doc, review_doc)
            run_name = f"review_v{int(semantic['review_version'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = V5_ROOT / "compiled" / video_id / run_name
            if output_dir.exists():
                raise FileExistsError(output_dir)
            v5.write_json(output_dir / "atomic_evidence.json", {"video_id": video_id, "version": "caption_event_gt_v5", "semantic_review": bundle["semantic_review"], "atomic_evidence": bundle["atomic_evidence"]})
            v5.write_json(output_dir / "caption_events.json", {"video_id": video_id, "version": "caption_event_gt_v5", "semantic_review": bundle["semantic_review"], "caption_events": bundle["caption_events"], "caption_event_candidates": bundle["caption_event_candidates"], "candidate_suggestions": bundle["candidate_suggestions"], "evaluation_eligible_caption_event_ids": bundle["evaluation_eligible_caption_event_ids"]})
            eligible = bundle["evaluation_eligible_caption_event_ids"]
            st.success(f"Compiled to {output_dir}. Evaluation eligible CE: {len(eligible)}")
            if semantic["reviewed_for_visible_events"] and semantic["reviewed_by_human"] and not eligible:
                st.warning("Semantic completeness is marked done, but no CE passed the compiler's per-event verification gate.")
            if eligible:
                st.write("Evaluation eligible:", ", ".join(eligible))
        except (ValueError, FileExistsError) as error:
            st.error(f"v5 compiler validation failed: {error}")

st.caption(f"Review JSON: {review_path}")
