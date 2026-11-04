#!/usr/bin/env python3
"""Blinded candidate-GT evaluation for the whole-video 8B experiment.

packets    : write judge packets. Times and spatial condition names are omitted.
aggregate  : read eval/judgments_<video_id>.json and score them.

Scores are provisional. The references are candidate captions, not human-verified GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "whole_video_spatial_8b"
EVAL = OUT / "eval"
LABELS = ("correct", "partial", "contradictory", "missing")
TIME_KEYS = {"start_sec", "end_sec", "start_time", "end_time", "window", "temporal_coverage_sec", "sampled_range"}


def walk_keys(obj, found: set[str]) -> None:
    if isinstance(obj, dict):
        found.update(obj)
        for v in obj.values():
            walk_keys(v, found)
    elif isinstance(obj, list):
        for v in obj:
            walk_keys(v, found)


def cmd_packets(_args) -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    rows = json.loads((OUT / "results.json").read_text())
    refs = json.loads((OUT / "gt_references.json").read_text())
    rng = random.Random(20261031)
    private = {}
    for video in refs["videos"]:
        vid = video["video_id"]
        got = [r for r in rows if r["short_id"] == vid and r["status"] == "ok"]
        rng.shuffle(got)
        captions = []
        for i, r in enumerate(got, 1):
            bid = f"{vid}-{i:02d}"
            private[bid] = {
                "condition_id": r["condition_id"],
                "spatial": r["spatial"],
                "video_name": r["video_id"],
            }
            captions.append(
                {
                    "blind_id": bid,
                    "text": r["caption"],
                    "generation_flags": [x for x in (r.get("caption_flags") or "").split("|") if x],
                }
            )
        packet = {
            "video_id": vid,
            "video_name": video["video_name"],
            "gt_status": "candidate",
            "human_verified": False,
            "instruction": "Judge the paragraph against each reference caption. Do not infer timestamps. Spatial condition names are withheld.",
            "reference_captions": [
                {
                    "ref_id": item["ref_id"],
                    "text": item["caption"],
                    "candidate_file_marked_review": bool(item["requires_human_review"]),
                }
                for item in video["eval_captions"]
            ],
            "captions": captions,
        }
        keys: set[str] = set()
        walk_keys(packet, keys)
        leaked = keys & TIME_KEYS
        if leaked:
            raise RuntimeError(f"{vid} packet contains time fields: {leaked}")
        meta = {k: v for k, v in packet.items() if k != "captions"}
        blob = json.dumps(meta)
        for banned in ("S0", "S1", "S2"):
            if re.search(rf"\b{banned}\b", blob):
                raise RuntimeError(f"{vid} packet leaks condition id {banned}")
        (EVAL / f"packet_{vid}.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False))
    (EVAL / "blind_map_PRIVATE.json").write_text(json.dumps(private, indent=2))
    print("packets", len(private))


def rate(n, d):
    return None if not d else round(n / d, 4)


def cmd_aggregate(_args) -> None:
    private = json.loads((EVAL / "blind_map_PRIVATE.json").read_text())
    refs = {v["video_id"]: v for v in json.loads((OUT / "gt_references.json").read_text())["videos"]}
    rows = {r["condition_id"]: r for r in json.loads((OUT / "results.json").read_text())}
    per_cond = []
    per_ref = []
    for vid, video in refs.items():
        jf = EVAL / f"judgments_{vid}.json"
        if not jf.is_file():
            continue
        packet = json.loads((EVAL / f"packet_{vid}.json").read_text())
        ref_ids = [c["ref_id"] for c in packet["reference_captions"]]
        judged = json.loads(jf.read_text())
        by_blind = {j["blind_id"]: j for j in judged["judgments"]}
        for cap in packet["captions"]:
            j = by_blind[cap["blind_id"]]
            meta = private[cap["blind_id"]]
            got = {e["ref_id"]: e for e in j["references"]}
            missing_ids = [rid for rid in ref_ids if rid not in got]
            if missing_ids:
                raise RuntimeError(f"{cap['blind_id']} missing refs {missing_ids}")
            labels = []
            for rid in ref_ids:
                lab = got[rid]["label"].lower()
                if lab not in LABELS:
                    raise RuntimeError(f"bad label {lab}")
                labels.append(lab)
                per_ref.append(
                    {
                        "condition_id": meta["condition_id"],
                        "video_id": meta["video_name"],
                        "short_id": vid,
                        "spatial": meta["spatial"],
                        "ref_id": rid,
                        "label": lab,
                        "requires_human_review": bool(got[rid].get("requires_human_review")),
                        "note": got[rid].get("note", ""),
                    }
                )
            n = len(labels)
            counts = Counter(labels)
            unsupported = j.get("unsupported_events") or []
            base = rows[meta["condition_id"]]
            per_cond.append(
                {
                    "condition_id": meta["condition_id"],
                    "video_id": meta["video_name"],
                    "short_id": vid,
                    "spatial": meta["spatial"],
                    "total_visual_tokens": base.get("total_visual_tokens"),
                    "n_gt": n,
                    "correct": counts["correct"],
                    "partial": counts["partial"],
                    "contradictory": counts["contradictory"],
                    "missing": counts["missing"],
                    "strict_match_rate": rate(counts["correct"], n),
                    "partial_inclusive_match_rate": rate(counts["correct"] + counts["partial"], n),
                    "gt_caption_coverage": rate(counts["correct"] + counts["partial"], n),
                    "n_unsupported_events": len(unsupported),
                    "unsupported_events": " || ".join(unsupported),
                    "caption_flags": base.get("caption_flags") or "",
                    "requires_human_review_count": sum(1 for rid in ref_ids if got[rid].get("requires_human_review")),
                    "gt_status": "candidate",
                    "human_verified": False,
                }
            )
    if not per_cond:
        raise RuntimeError("no judgments found")
    per_cond.sort(key=lambda r: (r["short_id"], r["spatial"]))
    for name, data in (("eval_per_condition", per_cond), ("eval_per_reference", per_ref)):
        (EVAL / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        with open(EVAL / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(len(per_cond), "conditions scored")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("packets")
    sub.add_parser("aggregate")
    args = ap.parse_args()
    {"packets": cmd_packets, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
