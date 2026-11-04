#!/usr/bin/env python3
"""Caption evaluation for the spatial-temporal budget experiment.

packets   : write blinded judge packets (candidate GT events inside the window + anonymised captions)
aggregate : read judge outputs (eval/judgments_<V>.json) and compute event/atomic-fact metrics

GT = collected_candidate_gt/*_candidate_gt.json (gt_status=candidate, human_verified=false).
Scores are therefore provisional; nothing here is a verified accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "spatial_temporal_budget_40960"
EVAL = OUT / "eval"
CFG = yaml.safe_load((OUT / "configs" / "experiment.yaml").read_text())
PRIMARY_OVERLAP = 0.5   # fraction of the GT event inside the window
BOUNDARY_MIN_SEC = 2.0  # shorter overlaps are out of scope; 2s..50% are "boundary"
FACETS = ("actor", "action", "object", "direction")


def gt_events(video: dict) -> dict:
    d = json.loads((REPO / "collected_candidate_gt" / f"{video['name']}_candidate_gt.json").read_text())
    s, e = float(video["start"]), float(video["end"])
    primary, boundary, outside = [], [], []
    for ev in d["level_2"]["events"]:
        a, b = float(ev["start_sec"]), float(ev["end_sec"])
        ov = max(0.0, min(b, e) - max(a, s))
        row = {"event_id": ev["event_id"], "start_sec": a, "end_sec": b, "caption": ev["caption"],
               "overlap_sec": round(ov, 2), "overlap_frac": round(ov / max(b - a, 1e-6), 3)}
        if ov / max(b - a, 1e-6) >= PRIMARY_OVERLAP:
            primary.append(row)
        elif ov >= BOUNDARY_MIN_SEC:
            boundary.append(row)
        else:
            outside.append(row)
    return {"scene": d["level_1"]["description"], "primary": primary, "boundary": boundary, "outside": outside,
            "gt_status": d["gt_status"], "human_verified": d["human_verified"]}


def split_sentences(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) <= 1 and lines:
        lines = [x.strip() for x in re.split(r"(?<=[.!?])\s+", lines[0]) if x.strip()]
    return lines


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def cmd_packets(_args) -> None:
    EVAL.mkdir(exist_ok=True)
    rows = json.loads((OUT / "results.json").read_text())
    rng = random.Random(20261030)
    private = {}
    for v in CFG["videos"]:
        vr = [r for r in rows if r["video_id"] == v["id"] and r["status"] == "ok"]
        rng.shuffle(vr)
        gt = gt_events(v)
        items = []
        for i, r in enumerate(vr, 1):
            bid = f"{v['id']}-{i:02d}"
            private[bid] = r["condition_id"]
            sents, seen = [], set()
            for k, s in enumerate(split_sentences(r["caption"]), 1):
                n = norm(s)
                sents.append({"idx": k, "text": s, "exact_duplicate_of_earlier": n in seen})
                seen.add(n)
            items.append({"blind_id": bid, "caption_truncated_by_token_limit": bool(r["hit_token_limit"]), "sentences": sents})
        packet = {
            "video_id": v["id"], "window_sec": [v["start"], v["end"]], "scene_context": gt["scene"],
            "gt_status": "candidate (not human verified)",
            "primary_gt_events": gt["primary"], "boundary_gt_events": gt["boundary"],
            "captions": items,
        }
        (EVAL / f"packet_{v['id']}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False))
    (EVAL / "blind_map_PRIVATE.json").write_text(json.dumps(private, indent=1))
    print("packets written:", len(private), "captions")


def f1(p, r):
    if p is None or r is None:
        return None
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def cmd_aggregate(_args) -> None:
    bmap = json.loads((EVAL / "blind_map_PRIVATE.json").read_text())
    rows = {r["condition_id"]: r for r in json.loads((OUT / "results.json").read_text())}
    per_cond = []
    event_rows = []
    for v in CFG["videos"]:
        jf = EVAL / f"judgments_{v['id']}.json"
        if not jf.is_file():
            continue
        packet = json.loads((EVAL / f"packet_{v['id']}.json").read_text())
        prim = {e["event_id"] for e in packet["primary_gt_events"]}
        bnd = {e["event_id"] for e in packet["boundary_gt_events"]}
        caps = {c["blind_id"]: c for c in packet["captions"]}
        for j in json.loads(jf.read_text())["judgments"]:
            cid = bmap[j["blind_id"]]
            sents = caps[j["blind_id"]]["sentences"]
            dup = {s["idx"] for s in sents if s["exact_duplicate_of_earlier"]}
            ev = {e["event_id"]: e for e in j["events"]}
            labels = {k: ev[k]["label"] if k in ev else "missing" for k in prim}
            n = len(prim)
            correct = sum(l == "correct" for l in labels.values())
            partial = sum(l == "partial" for l in labels.values())
            contra = sum(l == "contradictory" for l in labels.values())
            missing = sum(l == "missing" for l in labels.values())
            recall_strict = correct / n
            recall_len = (correct + partial) / n
            ss = [s for s in j["sentences"] if s["idx"] not in dup]
            sup = sum(s["status"] in ("supported", "boundary_supported") for s in ss)
            unsup = sum(s["status"] == "unsupported" for s in ss)
            contr_s = sum(s["status"] == "contradictory" for s in ss)
            prec = sup / len(ss) if ss else None
            facet_ok = facet_bad = facet_abs = 0
            for k in prim:
                if k in ev:
                    for f in FACETS:
                        st = ev[k].get("facets", {}).get(f, "na")
                        facet_ok += st == "correct"
                        facet_bad += st == "wrong"
                        facet_abs += st == "absent"
            applicable = facet_ok + facet_bad + facet_abs
            dir_wrong = sum(1 for k in prim if k in ev and ev[k].get("facets", {}).get("direction") == "wrong")
            order_err = sum(1 for k in prim if k in ev and ev[k].get("order_error"))
            r = rows[cid]
            per_cond.append({
                "condition_id": cid, "video": v["id"], "spatial": r["spatial"], "fps": r["fps"], "total_visual_tokens": r["total_visual_tokens"],
                "n_gt_primary": n, "correct": correct, "partial": partial, "contradictory": contra, "missing": missing,
                "event_recall_strict": round(recall_strict, 4), "event_recall_lenient": round(recall_len, 4),
                "n_unique_sentences": len(ss), "n_duplicate_sentences": len(dup),
                "supported_sentences": sup, "gt_unsupported_sentences": unsup, "contradicted_sentences": contr_s,
                "event_precision": None if prec is None else round(prec, 4),
                "event_f1_lenient": None if f1(prec, recall_len) is None else round(f1(prec, recall_len), 4),
                "event_f1_strict": None if f1(prec, recall_strict) is None else round(f1(prec, recall_strict), 4),
                "atomic_fact_correct": facet_ok, "atomic_fact_wrong": facet_bad, "atomic_fact_absent": facet_abs,
                "atomic_fact_accuracy_stated": None if facet_ok + facet_bad == 0 else round(facet_ok / (facet_ok + facet_bad), 4),
                "atomic_fact_coverage": None if applicable == 0 else round(facet_ok / applicable, 4),
                "direction_errors": dir_wrong, "order_errors": order_err,
                "gt_unsupported_rate": None if not ss else round(unsup / len(ss), 4),
                "hit_token_limit": r["hit_token_limit"],
            })
            for k in sorted(prim | bnd):
                event_rows.append({"condition_id": cid, "video": v["id"], "spatial": r["spatial"], "fps": r["fps"], "event_id": k,
                                   "scope": "primary" if k in prim else "boundary", "label": ev[k]["label"] if k in ev else "missing",
                                   **{f"facet_{f}": ev.get(k, {}).get("facets", {}).get(f, "") for f in FACETS}, "note": ev.get(k, {}).get("note", "")})
    per_cond.sort(key=lambda x: (x["video"], x["fps"], x["spatial"]))
    for name, rows_ in (("eval_per_condition", per_cond), ("eval_per_event", event_rows)):
        (EVAL / f"{name}.json").write_text(json.dumps(rows_, indent=1))
        with open(EVAL / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)
    print(len(per_cond), "conditions scored ->", EVAL)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("packets")
    sub.add_parser("aggregate")
    a = ap.parse_args()
    {"packets": cmd_packets, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
