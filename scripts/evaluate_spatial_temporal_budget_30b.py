#!/usr/bin/env python3
"""Blinded 8B vs 30B evaluation for the spatial-temporal budget experiment.

Packets mix both models' captions and hide condition names. Metrics use the same
definitions as scripts/evaluate_spatial_temporal_budget.py. The 8B experiment
directory is only read.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EIGHT = REPO / "experiments" / "spatial_temporal_budget_40960"
OUT = REPO / "experiments" / "spatial_temporal_budget_40960_30b"
EVAL = OUT / "eval"

spec = importlib.util.spec_from_file_location("stb_eval8", REPO / "scripts" / "evaluate_spatial_temporal_budget.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def _load_rows(root: Path) -> list[dict]:
    path = root / "results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def cmd_packets(_args) -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    rubric = (EIGHT / "eval" / "JUDGE_RUBRIC.md").read_text()
    (EVAL / "JUDGE_RUBRIC.md").write_text(rubric)
    eight = _load_rows(EIGHT)
    thirty = _load_rows(OUT)
    rng = random.Random(20261031)
    private = {}
    for video in ev.CFG["videos"]:
        items = []
        for model, rows in (("8b", eight), ("30b", thirty)):
            for row in rows:
                if row["video_id"] == video["id"] and row["status"] == "ok":
                    items.append((model, row))
        rng.shuffle(items)
        gt = ev.gt_events(video)
        captions = []
        for i, (model, row) in enumerate(items, 1):
            bid = f"{video['id']}-{i:02d}"
            private[bid] = {"model": model, "condition_id": row["condition_id"], "video_id": video["id"]}
            sentences, seen = [], set()
            for k, sentence in enumerate(ev.split_sentences(row["caption"]), 1):
                key = ev.norm(sentence)
                sentences.append({"idx": k, "text": sentence, "exact_duplicate_of_earlier": key in seen})
                seen.add(key)
            captions.append({
                "blind_id": bid,
                "caption_truncated_by_token_limit": bool(row["hit_token_limit"]),
                "sentences": sentences,
            })
        packet = {
            "video_id": video["id"],
            "window_sec": [video["start"], video["end"]],
            "scene_context": gt["scene"],
            "gt_status": "candidate (not human verified)",
            "human_verified": False,
            "primary_gt_events": gt["primary"],
            "boundary_gt_events": gt["boundary"],
            "captions": captions,
            "judge_note": "Captions are a mixture of two systems. Do not infer which system produced a caption. Judge every caption with the same rubric.",
        }
        (EVAL / f"packet_{video['id']}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False))
    (EVAL / "blind_map_PRIVATE.json").write_text(json.dumps(private, indent=1))
    print(f"packets written: {len(private)} captions -> {EVAL}")


def _score_judgment(judgment: dict, packet: dict, row: dict, model: str) -> tuple[dict, list[dict]]:
    prim = {event["event_id"] for event in packet["primary_gt_events"]}
    boundary = {event["event_id"] for event in packet["boundary_gt_events"]}
    caps = {caption["blind_id"]: caption for caption in packet["captions"]}
    sentences = caps[judgment["blind_id"]]["sentences"]
    dup = {sentence["idx"] for sentence in sentences if sentence["exact_duplicate_of_earlier"]}
    events = {event["event_id"]: event for event in judgment["events"]}
    labels = {key: events[key]["label"] if key in events else "missing" for key in prim}
    n = len(prim)
    correct = sum(label == "correct" for label in labels.values())
    partial = sum(label == "partial" for label in labels.values())
    contra = sum(label == "contradictory" for label in labels.values())
    missing = sum(label == "missing" for label in labels.values())
    recall_strict = correct / n
    recall_len = (correct + partial) / n
    unique_sentences = [sentence for sentence in judgment["sentences"] if sentence["idx"] not in dup]
    supported = sum(sentence["status"] in ("supported", "boundary_supported") for sentence in unique_sentences)
    unsupported = sum(sentence["status"] == "unsupported" for sentence in unique_sentences)
    contradicted = sum(sentence["status"] == "contradictory" for sentence in unique_sentences)
    precision = supported / len(unique_sentences) if unique_sentences else None
    # Same-unit alternative: each primary detection and each extra unsupported sentence is one claim.
    tp_len = correct + partial
    fp_len = contra + unsupported
    precision_aligned = tp_len / (tp_len + fp_len) if (tp_len + fp_len) else None
    facet_ok = facet_bad = facet_abs = 0
    for key in prim:
        if key in events:
            for facet in ev.FACETS:
                state = events[key].get("facets", {}).get(facet, "na")
                facet_ok += state == "correct"
                facet_bad += state == "wrong"
                facet_abs += state == "absent"
    applicable = facet_ok + facet_bad + facet_abs
    direction_wrong = sum(1 for key in prim if key in events and events[key].get("facets", {}).get("direction") == "wrong")
    order_err = sum(1 for key in prim if key in events and events[key].get("order_error"))
    per_cond = {
        "model": model,
        "condition_id": row["condition_id"],
        "video": row["video_id"],
        "spatial": row["spatial"],
        "fps": row["fps"],
        "total_visual_tokens": row["total_visual_tokens"],
        "n_gt_primary": n,
        "correct": correct,
        "partial": partial,
        "contradictory": contra,
        "missing": missing,
        "event_recall_strict": round(recall_strict, 4),
        "event_recall_lenient": round(recall_len, 4),
        "n_unique_sentences": len(unique_sentences),
        "n_duplicate_sentences": len(dup),
        "supported_sentences": supported,
        "gt_unsupported_sentences": unsupported,
        "contradicted_sentences": contradicted,
        "event_precision": None if precision is None else round(precision, 4),
        "event_f1_lenient": None if ev.f1(precision, recall_len) is None else round(ev.f1(precision, recall_len), 4),
        "event_f1_strict": None if ev.f1(precision, recall_strict) is None else round(ev.f1(precision, recall_strict), 4),
        "event_precision_aligned": None if precision_aligned is None else round(precision_aligned, 4),
        "event_f1_aligned": None if ev.f1(precision_aligned, recall_len) is None else round(ev.f1(precision_aligned, recall_len), 4),
        "atomic_fact_correct": facet_ok,
        "atomic_fact_wrong": facet_bad,
        "atomic_fact_absent": facet_abs,
        "atomic_fact_accuracy_stated": None if facet_ok + facet_bad == 0 else round(facet_ok / (facet_ok + facet_bad), 4),
        "atomic_fact_coverage": None if applicable == 0 else round(facet_ok / applicable, 4),
        "direction_errors": direction_wrong,
        "order_errors": order_err,
        "gt_unsupported_rate": None if not unique_sentences else round(unsupported / len(unique_sentences), 4),
        "hit_token_limit": row["hit_token_limit"],
        "output_tokens": row.get("output_tokens", row.get("output_token_count")),
        "repeated_sentence_count": row.get("repeated_sentence_count", len(dup)),
    }
    event_rows = []
    for key in sorted(prim | boundary):
        event_rows.append({
            "model": model,
            "condition_id": row["condition_id"],
            "video": row["video_id"],
            "spatial": row["spatial"],
            "fps": row["fps"],
            "event_id": key,
            "scope": "primary" if key in prim else "boundary",
            "label": events[key]["label"] if key in events else "missing",
            **{f"facet_{facet}": events.get(key, {}).get("facets", {}).get(facet, "") for facet in ev.FACETS},
            "note": events.get(key, {}).get("note", ""),
        })
    return per_cond, event_rows


def cmd_aggregate(_args) -> None:
    private = json.loads((EVAL / "blind_map_PRIVATE.json").read_text())
    by_model = {"8b": {row["condition_id"]: row for row in _load_rows(EIGHT)}, "30b": {row["condition_id"]: row for row in _load_rows(OUT)}}
    per_cond, event_rows = [], []
    for video in ev.CFG["videos"]:
        judgment_path = EVAL / f"judgments_{video['id']}.json"
        if not judgment_path.is_file():
            continue
        packet = json.loads((EVAL / f"packet_{video['id']}.json").read_text())
        expected_ids = {caption["blind_id"] for caption in packet["captions"]}
        expected_events = {event["event_id"] for event in packet["primary_gt_events"] + packet["boundary_gt_events"]}
        payload = json.loads(judgment_path.read_text())
        seen = set()
        for judgment in payload["judgments"]:
            bid = judgment["blind_id"]
            if bid not in expected_ids:
                raise SystemExit(f"{video['id']}: unknown blind_id {bid}")
            seen.add(bid)
            got_events = {event["event_id"] for event in judgment["events"]}
            if got_events != expected_events:
                raise SystemExit(f"{bid}: events {sorted(got_events)} != {sorted(expected_events)}")
            got_sentences = {sentence["idx"] for sentence in judgment["sentences"]}
            need_sentences = {sentence["idx"] for sentence in next(c for c in packet["captions"] if c["blind_id"] == bid)["sentences"]}
            if got_sentences != need_sentences:
                raise SystemExit(f"{bid}: sentences {sorted(got_sentences)} != {sorted(need_sentences)}")
            meta = private[bid]
            row = by_model[meta["model"]][meta["condition_id"]]
            cond, events = _score_judgment(judgment, packet, row, meta["model"])
            per_cond.append(cond)
            event_rows.extend(events)
        missing = expected_ids - seen
        if missing:
            raise SystemExit(f"{video['id']}: missing judgments {sorted(missing)}")
    per_cond.sort(key=lambda row: (row["model"], row["video"], row["fps"], row["spatial"]))
    for name, rows in (("eval_per_condition", per_cond), ("eval_per_event", event_rows)):
        (EVAL / f"{name}.json").write_text(json.dumps(rows, indent=1))
        with open(EVAL / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(len(per_cond), "conditions scored ->", EVAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("packets")
    sub.add_parser("aggregate")
    args = parser.parse_args()
    {"packets": cmd_packets, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
