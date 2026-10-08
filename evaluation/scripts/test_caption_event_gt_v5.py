#!/usr/bin/env python3
"""Focused tests for the human-gated v5 caption-event compiler."""

from __future__ import annotations

import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import build_caption_event_gt_v5 as v5


VIDEO_ID = "VIRAT_TEST_VIDEO"


def evidence(evidence_id: str, track_id: str, action: str, start: int, end: int) -> dict:
    return {
        "atomic_evidence_id": evidence_id,
        "source": {
            "type": "virat_activity",
            "virat_event_ids": [evidence_id],
            "source_track_ids": [track_id],
        },
        "start_frame": start,
        "end_frame": end,
        "start_time": start / 10,
        "end_time": end / 10,
        "subject": {"category": "person", "track_id": track_id},
        "action": action,
        "object": [],
        "direction": None,
        "semantic_location": None,
    }


def base_docs() -> tuple[dict, dict]:
    atomic = evidence("AE001", "27", "walking", 10, 30)
    caption = {
        "caption_event_id": "CE001",
        "event_form": "individual",
        "source_atomic_evidence_ids": ["AE001"],
        "source_track_ids": ["27"],
        "start_frame": 10,
        "end_frame": 30,
        "start_time": 1.0,
        "end_time": 3.0,
        "subject": {"category": "person", "group": False, "track_id": "27"},
        "action": "walking",
        "object": [],
        "direction": None,
        "semantic_location": None,
        "reference_sentence": None,
    }
    return ({"video_id": VIDEO_ID, "atomic_evidence": [atomic]}, {"video_id": VIDEO_ID, "caption_events": [caption]})


class CaptionEventV5Tests(unittest.TestCase):
    def test_human_added_requires_human_confirmation_and_remains_candidate_until_ce_review(self) -> None:
        source_ae, source_ce = base_docs()
        review = {
            "video_id": VIDEO_ID,
            "fps": 10,
            "human_added_events": [
                {
                    "human_added_event_id": "HA004",
                    "action": "riding_bicycle",
                    "subject": {"category": "person", "track_id": "31"},
                    "source_track_ids": ["31"],
                    "start_frame": 40,
                    "end_frame": 80,
                    "review": {"human_verified": True},
                }
            ],
            "candidate_suggestions": [{"source": "vlm", "action": "vehicle_moving"}],
        }
        bundle = v5.build_bundle(VIDEO_ID, source_ae, source_ce, review)
        added = bundle["atomic_evidence"][1]
        self.assertEqual(added["source"]["type"], "human_added")
        self.assertEqual(added["start_time"], 4.0)
        self.assertEqual(bundle["caption_events"][0]["v5_review"]["status"], "legacy_unverified")
        self.assertFalse(bundle["caption_events"][0]["evaluation_eligible"])
        self.assertEqual(bundle["caption_event_candidates"][0]["source_atomic_evidence_ids"], ["AE002"])
        self.assertEqual(len(bundle["atomic_evidence"]), 2)
        self.assertEqual(bundle["candidate_suggestions"][0]["source"], "vlm")

        review["human_added_events"][0]["review"]["human_verified"] = False
        with self.assertRaisesRegex(ValueError, "human-confirmed"):
            v5.build_bundle(VIDEO_ID, source_ae, source_ce, review)

    def test_composite_requires_participant_link_and_video_completeness_for_eval(self) -> None:
        source_ae, source_ce = base_docs()
        source_ae["atomic_evidence"].append(evidence("AE002", "27", "entering_vehicle", 35, 50))
        review = {
            "video_id": VIDEO_ID,
            "fps": 10,
            "semantic_review": {
                "reviewed_for_visible_events": True,
                "reviewed_by_human": True,
                "review_version": 1,
            },
            "atomic_evidence_reviews": [
                {"atomic_evidence_id": "AE001", "human_verified": True, "reviewer_id": "reviewer"},
                {"atomic_evidence_id": "AE002", "human_verified": True, "reviewer_id": "reviewer"},
            ],
            "caption_event_decisions": [
                {
                    "event_form": "composite",
                    "source_atomic_evidence_ids": ["AE001", "AE002"],
                    "event_semantics": {
                        "primary_action": "entering_vehicle",
                        "supporting_actions": ["walking"],
                    },
                    "importance": "primary",
                    "reference_sentence": "A person walks toward a parked car and gets into it.",
                    "human_review": {
                        "human_verified": True,
                        "reference_sentence_verified": True,
                        "temporal_bounds_verified": True,
                        "reviewer_id": "reviewer",
                    },
                }
            ],
        }
        bundle = v5.build_bundle(VIDEO_ID, source_ae, source_ce, review)
        composite = bundle["caption_events"][1]
        self.assertEqual(composite["event_form"], "composite")
        self.assertEqual(composite["action"], "entering_vehicle")
        self.assertEqual(composite["source_atomic_evidence_ids"], ["AE001", "AE002"])
        self.assertTrue(composite["evaluation_eligible"])
        self.assertEqual(bundle["evaluation_eligible_caption_event_ids"], ["CE002"])

        review["semantic_review"]["reviewed_by_human"] = False
        with self.assertRaisesRegex(ValueError, "without human review"):
            v5.build_bundle(VIDEO_ID, source_ae, source_ce, review)

    def test_temporal_overlap_does_not_create_a_group(self) -> None:
        source_ae, source_ce = base_docs()
        source_ae["atomic_evidence"].append(evidence("AE002", "28", "walking", 15, 25))
        bundle = v5.build_bundle(VIDEO_ID, source_ae, source_ce)
        self.assertEqual(len(bundle["caption_events"]), 1)
        self.assertEqual(bundle["caption_events"][0]["event_form"], "individual")

    def test_legacy_human_verified_walking_evidence_is_preserved_but_not_finalized(self) -> None:
        source_ae, source_ce = base_docs()
        source_ae["atomic_evidence"][0]["review"] = {"human_verified": True, "is_walking": True}
        bundle = v5.build_bundle(VIDEO_ID, source_ae, source_ce)
        imported = bundle["caption_events"][0]
        self.assertTrue(imported["v5_review"]["evidence_human_verified"])
        self.assertFalse(imported["v5_review"]["human_verified"])
        self.assertFalse(imported["evaluation_eligible"])

    def test_ae_review_edits_semantics_without_changing_source_provenance(self) -> None:
        source_ae, source_ce = base_docs()
        original_source = source_ae["atomic_evidence"][0]["source"].copy()
        review = {
            "video_id": VIDEO_ID,
            "fps": 10,
            "atomic_evidence_reviews": [
                {
                    "atomic_evidence_id": "AE001",
                    "action": "crossing_road",
                    "start_time": 1.5,
                    "end_time": 2.5,
                    "start_frame": 15,
                    "end_frame": 25,
                    "human_verified": True,
                }
            ],
        }
        bundle = v5.build_bundle(VIDEO_ID, source_ae, source_ce, review)
        updated = bundle["atomic_evidence"][0]
        self.assertEqual(updated["action"], "crossing_road")
        self.assertEqual(updated["start_frame"], 15)
        self.assertEqual(updated["source"], original_source)
        self.assertTrue(updated["review"]["human_verified"])

    def test_existing_ce_update_can_pass_existing_evaluation_gate(self) -> None:
        source_ae, source_ce = base_docs()
        review = {
            "video_id": VIDEO_ID,
            "fps": 10,
            "semantic_review": {
                "reviewed_for_visible_events": True,
                "reviewed_by_human": True,
                "review_version": 1,
            },
            "atomic_evidence_reviews": [
                {"atomic_evidence_id": "AE001", "human_verified": True, "reviewer_id": "reviewer"},
            ],
            "caption_event_decisions": [
                {
                    "caption_event_id": "CE001",
                    "event_form": "individual",
                    "source_atomic_evidence_ids": ["AE001"],
                    "event_semantics": {"primary_action": "walking", "supporting_actions": []},
                    "importance": "primary",
                    "reference_sentence": "A person walks across the lot.",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "human_review": {
                        "human_verified": True,
                        "reference_sentence_verified": True,
                        "temporal_bounds_verified": True,
                    },
                }
            ],
        }
        bundle = v5.build_bundle(VIDEO_ID, source_ae, source_ce, review)
        self.assertEqual(len(bundle["caption_events"]), 1)
        self.assertEqual(bundle["caption_events"][0]["caption_event_id"], "CE001")
        self.assertTrue(bundle["caption_events"][0]["evaluation_eligible"])
        self.assertEqual(bundle["evaluation_eligible_caption_event_ids"], ["CE001"])

    def test_cli_review_save_compile_round_trip_uses_separate_output_root(self) -> None:
        source_ae, source_ce = base_docs()
        review = {
            "video_id": VIDEO_ID,
            "fps": 10,
            "semantic_review": {
                "reviewed_for_visible_events": True,
                "reviewed_by_human": True,
                "review_version": 1,
            },
            "atomic_evidence_reviews": [
                {"atomic_evidence_id": "AE001", "human_verified": True, "reviewer_id": "reviewer"},
            ],
            "caption_event_decisions": [
                {
                    "caption_event_id": "CE001",
                    "event_form": "individual",
                    "source_atomic_evidence_ids": ["AE001"],
                    "event_semantics": {"primary_action": "walking", "supporting_actions": []},
                    "importance": "primary",
                    "reference_sentence": "A person walks across the lot.",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "human_review": {
                        "human_verified": True,
                        "reference_sentence_verified": True,
                        "temporal_bounds_verified": True,
                    },
                }
            ],
        }
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "atomic_evidence.json").write_text(json.dumps(source_ae))
            (source / "caption_events.json").write_text(json.dumps(source_ce))
            review_path = root / "human_review.json"
            review_path.write_text(json.dumps(review))
            output_root = root / "compiled"
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "build_caption_event_gt_v5.py"),
                    "--video-id", VIDEO_ID,
                    "--source-dir", str(source),
                    "--review-file", str(review_path),
                    "--out-root", str(output_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            compiled = json.loads(
                (output_root / "gt" / VIDEO_ID / "caption_events.json").read_text()
            )
            self.assertEqual(compiled["evaluation_eligible_caption_event_ids"], ["CE001"])
            self.assertTrue(compiled["caption_events"][0]["evaluation_eligible"])

    def test_group_requires_same_action_and_multiple_actors(self) -> None:
        source_ae, _source_ce = base_docs()
        source_ae["atomic_evidence"].append(evidence("AE002", "28", "walking", 15, 25))
        event = v5.compile_caption_event(
            {
                "event_form": "group",
                "source_atomic_evidence_ids": ["AE001", "AE002"],
                "event_semantics": {"primary_action": "walking", "supporting_actions": []},
            },
            {item["atomic_evidence_id"]: item for item in source_ae["atomic_evidence"]},
            "CE002",
            10,
        )
        self.assertTrue(event["subject"]["group"])
        self.assertEqual(event["event_form"], "group")

        source_ae["atomic_evidence"][1]["action"] = "running"
        with self.assertRaisesRegex(ValueError, "same action"):
            v5.compile_caption_event(
                {
                    "event_form": "group",
                    "source_atomic_evidence_ids": ["AE001", "AE002"],
                    "event_semantics": {"primary_action": "walking", "supporting_actions": []},
                },
                {item["atomic_evidence_id"]: item for item in source_ae["atomic_evidence"]},
                "CE002",
                10,
            )


if __name__ == "__main__":
    unittest.main()
