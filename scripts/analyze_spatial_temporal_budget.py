#!/usr/bin/env python3
"""Markdown tables for the spatial-temporal budget report (resource usage, failure types, quality, equal-cost pairs)."""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "spatial_temporal_budget_40960"
FPS = [0.5, 1.0, 2.0, 3.0, 4.0]
SP = ["S0", "S1", "S2"]
rows = json.loads((OUT / "results.json").read_text())
R = {(r["video_id"], r["spatial"], r["fps"]): r for r in rows}
lines: list[str] = []


def md(header, body):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for b in body:
        lines.append("| " + " | ".join(str(x) for x in b) + " |")
    lines.append("")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def sentences(text):
    return [l.strip() for l in text.splitlines() if l.strip()] or [text.strip()]


# --- 1. token usage by spatial x fps (mean over videos)
lines.append("### T1. Mean visual tokens (image+video) by Spatial x FPS (mean of 4 videos)\n")
md(["Spatial"] + [f"{f:g} FPS" for f in FPS], [[s] + [f"{mean([R[(v, s, f)]['total_visual_tokens'] for v in 'ACDE']):.0f}" for f in FPS] for s in SP])
lines.append("### T2. Per-condition resources\n")
body = []
for v in "ACDE":
    for f in FPS:
        for s in SP:
            r = R[(v, s, f)]
            body.append([v, s, f"{f:g}", r["roi_count"], r["input_frames_per_video"], r["input_resolution"], r["image_tokens"], r["video_tokens"], r["total_visual_tokens"],
                         f"{r['total_visual_tokens'] / 40960:.0%}", r["binding_constraint"].replace("per_frame_max_pixels_or_native", "cap/native"), fmt(r["peak_gpu_memory_mib"], 0), fmt(r["inference_time_sec"]), r["output_tokens"],
                         "Y" if r["terminated_normally"] else "N"])
md(["Video", "Sp", "FPS", "ROI", "Frames/video", "Input res (WxH)", "Img tok", "Vid tok", "Total", "% of 40,960", "Binding", "Peak MiB", "Infer s", "Out tok", "EOS"], body)
lines.append("### T3. Mean inference time (s) / peak GPU MiB by Spatial x FPS\n")
md(["Spatial"] + [f"{f:g} FPS" for f in FPS],
   [[s] + [f"{mean([R[(v, s, f)]['inference_time_sec'] for v in 'ACDE']):.1f} s / {mean([R[(v, s, f)]['peak_gpu_memory_mib'] for v in 'ACDE']):.0f}" for f in FPS] for s in SP])

# --- 2. failure types
lines.append("### T4. Generation termination / repetition types\n")
body = []
for r in rows:
    ss = sentences(r["caption"])
    uniq = len({re.sub(r"[^a-z0-9 ]", "", s.lower()) for s in ss})
    kinds = []
    if r["hit_token_limit"]:
        kinds.append("hit max_new_tokens")
    if len(ss) > uniq:
        kinds.append(f"repeated sentences ({len(ss) - uniq} dup)")
    if len(ss) == 1 and r["output_tokens"] < 40:
        kinds.append("single-sentence summary")
    if kinds:
        body.append([r["condition_id"], r["output_tokens"], len(ss), uniq, "; ".join(kinds)])
md(["Condition", "Out tokens", "Sentences", "Unique", "Type"], body)

# --- 3. quality (if evaluated)
ev_path = OUT / "eval" / "eval_per_condition.json"
if ev_path.is_file():
    E = {(e["video"], e["spatial"], e["fps"]): e for e in json.loads(ev_path.read_text())}
    metrics = [("event_recall_strict", "Recall(strict)"), ("event_recall_lenient", "Recall(lenient)"), ("event_precision", "Precision"), ("event_f1_lenient", "F1(lenient)"), ("event_f1_strict", "F1(strict)")]
    for key, label in metrics:
        lines.append(f"### Q-{label}: mean over 4 videos, Spatial x FPS\n")
        md(["Spatial"] + [f"{f:g} FPS" for f in FPS] + ["mean"],
           [[s] + [fmt(mean([E[(v, s, f)][key] for v in "ACDE"]), 3) for f in FPS] + [fmt(mean([E[(v, s, f)][key] for v in "ACDE" for f in FPS]), 3)] for s in SP])
    lines.append("### Q-per-video: F1(lenient) / Recall(lenient) per condition\n")
    for v in "ACDE":
        lines.append(f"**Video {v}** (F1 / Recall-lenient / Recall-strict)\n")
        md(["Spatial"] + [f"{f:g} FPS" for f in FPS],
           [[s] + [f"{fmt(E[(v, s, f)]['event_f1_lenient'], 2)} / {E[(v, s, f)]['event_recall_lenient']:.2f} / {E[(v, s, f)]['event_recall_strict']:.2f}" for f in FPS] for s in SP])
    lines.append("### Q-best per video (by F1 lenient, ties -> lower tokens)\n")
    body = []
    for v in "ACDE":
        c = sorted([(E[(v, s, f)]["event_f1_lenient"] or 0, -E[(v, s, f)]["total_visual_tokens"], s, f) for s in SP for f in FPS], reverse=True)[:3]
        body.append([v] + [f"{s} {f:g}fps (F1 {a:.2f}, {-t} tok)" for a, t, s, f in c])
    md(["Video", "1st", "2nd", "3rd"], body)
    lines.append("### Q-mean over videos, per (Spatial, FPS): F1(lenient), tokens, tokens per F1\n")
    body = []
    for s in SP:
        for f in FPS:
            f1 = mean([E[(v, s, f)]["event_f1_lenient"] for v in "ACDE"])
            tk = mean([E[(v, s, f)]["total_visual_tokens"] for v in "ACDE"])
            body.append([s, f"{f:g}", fmt(f1, 3), fmt(mean([E[(v, s, f)]["event_recall_lenient"] for v in "ACDE"]), 3), f"{tk:.0f}", fmt(mean([E[(v, s, f)]["atomic_fact_accuracy_stated"] for v in "ACDE"]), 3), fmt(mean([E[(v, s, f)]["gt_unsupported_rate"] for v in "ACDE"]), 3)])
    md(["Spatial", "FPS", "F1 (lenient)", "Recall (lenient)", "Mean tokens", "Atomic acc. (stated)", "GT-unsupported rate"], body)

    # equal-cost pairs: same video, different spatial, tokens within +-5%
    lines.append("### Q-equal-cost pairs (same video, different Spatial input, total tokens within +-5%)\n")
    body = []
    for v in "ACDE":
        items = [(s, f, R[(v, s, f)]["total_visual_tokens"]) for s in SP for f in FPS]
        for (s1, f1_, t1), (s2, f2, t2) in combinations(items, 2):
            if s1 != s2 and abs(t1 - t2) / max(t1, t2) <= 0.05:
                a, b = E[(v, s1, f1_)], E[(v, s2, f2)]
                body.append([v, f"{s1} {f1_:g}fps", t1, fmt(a["event_f1_lenient"], 2), f"{s2} {f2:g}fps", t2, fmt(b["event_f1_lenient"], 2), f"{abs(t1 - t2) / max(t1, t2):.1%}"])
    md(["Video", "A", "tokens", "F1", "B", "tokens", "F1", "diff"], body) if body else lines.append("None.\n")
(OUT / "eval" / "tables.md").write_text("\n".join(lines))
print("tables written:", OUT / "eval" / "tables.md", len(lines), "lines")
