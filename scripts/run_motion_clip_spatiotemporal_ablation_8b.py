#!/usr/bin/env python3
"""Clip-level spatial x temporal event captioning on Qwen3-VL-8B-Instruct.

Uses the frozen M1 segments from motion_temporal_segmentation_feasibility.
Does not rewrite those files, candidate GT, motion tubes, or earlier experiments.

Commands: manifest | plan | infer | aggregate | check | packets | score
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vlm_inference.frame_clock import encode_timestamps  # noqa: E402

OUT = REPO / "experiments" / "motion_clip_spatiotemporal_ablation_8b"
FACTOR = 32
TEMPORAL = 2
LABELS = ("correct", "partial", "contradictory", "missing")
# Passed to generate() only when the experiment file sets them. seed is torch-only.
GENERATE_KEYS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "min_new_tokens",
    "eos_token_id",
    "pad_token_id",
    "use_cache",
)
log = logging.getLogger("clipablation")


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load((OUT / "config" / "experiment.yaml").read_text())


def load_common_prompt() -> str:
    path = OUT / "config" / "common_prompt.txt"
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if not text.strip():
        raise RuntimeError(f"{path} is empty")
    return text


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fps_tag(fps: float) -> str:
    return f"fps{fps:g}".replace(".", "p")


def cond_id(clip_id: str, spatial: str, fps: float) -> str:
    return f"{clip_id}_{spatial}_{fps_tag(fps)}"


def generation_record(cfg: dict[str, Any]) -> dict[str, Any]:
    g = cfg["generation"]
    unknown = sorted(set(g) - set(GENERATE_KEYS) - {"seed"})
    if unknown:
        raise RuntimeError(f"experiment.yaml generation has fields this runner does not pass: {unknown}")
    rec: dict[str, Any] = {"seed": int(g["seed"])}
    for key in GENERATE_KEYS:
        if key not in g or g[key] in (None, "null"):
            rec[key] = None
            continue
        value = g[key]
        if key in ("top_k", "max_new_tokens", "min_new_tokens", "pad_token_id"):
            value = int(value)
        elif key in ("temperature", "top_p", "repetition_penalty"):
            value = float(value)
        elif key == "do_sample" or key == "use_cache":
            value = bool(value)
        elif key == "eos_token_id":
            value = [int(v) for v in value]
        rec[key] = value
    rec["sha"] = sha16(json.dumps({k: rec[k] for k in rec if k != "sha"}, sort_keys=True))
    return rec


def generation_kwargs(gen: dict[str, Any]) -> dict[str, Any]:
    return {key: gen[key] for key in GENERATE_KEYS if gen.get(key) is not None}


def resolve_video(cfg: dict[str, Any], name: str) -> Path:
    hits = []
    for root in cfg["sources"]["video_roots"]:
        p = Path(root) / f"{name}.mp4"
        if p.is_file():
            hits.append(p)
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {hits}")
    return hits[0]


def probe(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        if c.duration is None:
            raise RuntimeError(f"{path} has no container duration")
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "width": int(st.codec_context.width),
            "height": int(st.codec_context.height),
            "native_fps": float(st.average_rate),
            "duration_sec": float(c.duration / 1e6),
            "frame_count": int(st.frames),
        }


def grid_times(start: float, end: float, fps: float) -> list[float]:
    n = int(math.ceil((end - start) * fps - 1e-9))
    times = [start + k / fps for k in range(n)]
    if not times or times[-1] >= end - 1e-9:
        raise RuntimeError(f"grid for {start}-{end} at {fps} is not inside the clip")
    return times


def smart_resize(h: int, w: int, min_px: int, max_px: int) -> tuple[int, int]:
    from qwen_vl_utils.vision_process import smart_resize as sr

    return sr(h, w, FACTOR, min_pixels=min_px, max_pixels=max_px)


def video_tokens(frames: int, h: int, w: int) -> int:
    return math.ceil(frames / TEMPORAL) * (h // FACTOR) * (w // FACTOR)


def image_tokens(h: int, w: int) -> int:
    return (h // FACTOR) * (w // FACTOR)


def input_description(spatial: str, start: float, end: float, n_roi: int, image_time: float | None) -> str:
    if spatial == "S0":
        return (
            f"The input is one full-frame video covering the original recording from {start:.2f}s to {end:.2f}s. "
            "Timestamps shown in the video are original recording times."
        )
    if image_time is None:
        raise RuntimeError("S1/S2 require the shared first-frame time")
    noun = "one ROI video" if n_roi == 1 else f"{n_roi} ROI videos"
    return (
        f"The input is one full-frame image at {image_time:.2f}s, followed by {noun}. "
        "Each ROI video is an independent spatial region of the same scene and covers only the time range stated in its label. "
        "Timestamps are original recording times. "
        "Use the full-frame image only for scene context. "
        "Do not merge events from different ROI videos unless the same actor clearly continues from one region into another."
    )


def full_prompt(spatial: str, start: float, end: float, n_roi: int, image_time: float | None) -> str:
    return input_description(spatial, start, end, n_roi, image_time) + "\n\n" + load_common_prompt()


def clip_xywh(x: int, y: int, w: int, h: int, width: int, height: int) -> dict[str, Any]:
    requested = [int(x), int(y), int(w), int(h)]
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(width, x2), min(height, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        raise RuntimeError(f"ROI {requested} is outside the {width}x{height} frame")
    applied = [cx1, cy1, cx2 - cx1, cy2 - cy1]
    reason = ""
    if applied != requested:
        reason = (
            f"requested xywh {requested} ends at ({x2}, {y2}); "
            f"frame is {width}x{height}; applied xywh {applied}"
        )
    return {
        "requested_xywh": requested,
        "applied_xywh": applied,
        "applied_xyxy": [cx1, cy1, cx2, cy2],
        "clipped_to_frame": applied != requested,
        "clip_reason": reason,
    }


def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def load_events(cfg: dict[str, Any], video_name: str) -> list[dict[str, Any]]:
    path = REPO / cfg["sources"]["gt"] / f"{video_name}_candidate_gt.json"
    payload = json.loads(path.read_text())
    events = payload["level_2"]["events"]
    if len(events) != int(payload["event_count"]):
        raise RuntimeError(f"{video_name} event_count {payload['event_count']} != {len(events)}")
    return [
        {
            "event_id": ev["event_id"],
            "start_sec": float(ev["start_sec"]),
            "end_sec": float(ev["end_sec"]),
            "caption": ev["caption"],
            "requires_human_review": ev["event_id"] in set(payload.get("review_required_event_ids") or []),
        }
        for ev in events
    ]


def motion_tubes(cfg: dict[str, Any], name: str, width: int, height: int) -> list[dict[str, Any]]:
    src = cfg["sources"]
    tracks_path = Path(src["motion_roi_tracks"]) / f"{name}_roi_tracks.json"
    tracks = json.loads(tracks_path.read_text())
    if tracks.get("variant") != "stage3_light_uf_compose":
        raise RuntimeError(f"{name} tube variant is {tracks.get('variant')}, expected stage3_light_uf_compose")
    z = np.load(Path(src["motion_fusion_npz"]) / name / f"{name}_gap_fusion_mean.npz")
    ts = np.asarray(z["timestamp_sec_curr"], dtype=np.float64)
    cell = int(tracks.get("cell_px") or 64)
    rois = []
    for tube in sorted(tracks["tracks"], key=lambda t: (int(t["t0"]), -int(t.get("num_cells") or 0), int(t["tube_id"]))):
        g = tube["spatial_bbox_grid"]
        x0, y0 = int(g[0]) * cell, int(g[1]) * cell
        x1, y1 = min(width, int(g[2]) * cell), min(height, int(g[3]) * cell)
        t0 = max(0, min(int(tube["t0"]), len(ts) - 1))
        t1 = max(0, min(int(tube["t1"]), len(ts) - 1))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"{name} tube{tube['tube_id']} has an empty box")
        rois.append(
            {
                "roi_id": f"tube{int(tube['tube_id'])}",
                "bbox_xyxy": [x0, y0, x1, y1],
                "roi_native_size": [x1 - x0, y1 - y0],
                "time_range": [float(ts[t0]), float(ts[t1])],
                "source": "motion_stage3_tube",
                "variant": tracks.get("variant"),
                "tube_index_range": [int(tube["t0"]), int(tube["t1"])],
                "num_cells": tube.get("num_cells"),
                "cell_px": cell,
                "tracks_file": str(tracks_path),
            }
        )
    return rois


def classify_event(ev: dict[str, Any], start: float, end: float, eps: float) -> str:
    es, ee = float(ev["start_sec"]), float(ev["end_sec"])
    if ee < es:
        raise RuntimeError(f"event {ev['event_id']} ends before it starts")
    ov = overlap_sec(es, ee, start, end)
    contained = es >= start - eps and ee <= end + eps
    if contained:
        return "contained"
    if ov > eps:
        return "boundary"
    return "none"


def manual_for_clip(cfg: dict[str, Any], video_name: str, start: float, end: float, width: int, height: int) -> dict[str, Any]:
    row = next((r for r in cfg["manual_rois"] if r["video"] == video_name), None)
    if row is None:
        return {
            "available": False,
            "reason": "no_human_box_for_video",
            "source": None,
            "annotated_window": None,
            "overlap_sec": None,
        }
    w0, w1 = float(row["annotated_window"][0]), float(row["annotated_window"][1])
    ov = overlap_sec(start, end, w0, w1)
    minimum = float(cfg["experiment"]["manual_roi_min_overlap_sec"])
    if ov < minimum:
        return {
            "available": False,
            "reason": "annotated_window_overlap_below_minimum" if ov > 0 else "clip_outside_annotated_window",
            "source": row["source"],
            "annotated_window": [w0, w1],
            "requested_xywh": [int(v) for v in row["xywh"]],
            "overlap_sec": round(ov, 4),
            "min_overlap_sec": minimum,
        }
    box = clip_xywh(*[int(v) for v in row["xywh"]], width, height)
    covers = w0 <= start + 1e-6 and w1 >= end - 1e-6
    box.update(
        {
            "available": True,
            "reason": "reused_human_box",
            "source": row["source"],
            "annotated_window": [w0, w1],
            "overlap_sec": round(ov, 4),
            "coverage": "full" if covers else "partial",
            "roi_mode": "single_manual_roi",
            "roi_count": 1,
        }
    )
    return box


def decode_nearest(path: str, targets: list[float]) -> list[dict[str, Any]]:
    import av

    if targets != sorted(targets):
        raise RuntimeError("decode targets must be sorted")
    lo, hi = min(targets) - 1.0, max(targets) + 1.0
    best: dict[int, dict[str, Any]] = {}
    with av.open(path) as c:
        st = c.streams.video[0]
        c.seek(int(max(0.0, lo) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base) if fr.pts is not None else float(fr.time)
            if t > hi:
                break
            if t < lo:
                continue
            j = bisect.bisect_left(targets, t)
            arr = None
            for k in (j - 1, j):
                if 0 <= k < len(targets):
                    d = abs(t - targets[k])
                    if k not in best or d < best[k]["abs_error_sec"]:
                        if arr is None:
                            arr = fr.to_ndarray(format="rgb24")
                        best[k] = {
                            "target_time_sec": targets[k],
                            "source_time_sec": t,
                            "abs_error_sec": d,
                            "array": arr,
                        }
    if len(best) != len(targets):
        raise RuntimeError(f"decoded {len(best)} of {len(targets)} target frames from {path}")
    return [best[k] for k in range(len(targets))]


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=90)


def draw_boxes(rgb: np.ndarray, boxes: list[dict[str, Any]]) -> Any:
    from PIL import Image, ImageDraw

    im = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(im)
    for box in boxes:
        x0, y0, x1, y1 = box["bbox_xyxy"]
        color = box["color"]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=4)
        draw.text((x0 + 6, max(0, y0 + 6)), box["label"], fill=color)
    return im


def build_manifest(cfg: dict[str, Any], export_frames: bool) -> dict[str, Any]:
    seg_dir = Path(cfg["sources"]["segmentation_dir"])
    files = sorted(seg_dir.glob("*_M1.json"))
    if len(files) != 8:
        raise RuntimeError(f"expected 8 M1 files, found {len(files)}")
    eps = float(cfg["experiment"]["containment_eps_sec"])
    short = cfg["short_ids"]
    clips: list[dict[str, Any]] = []
    gt_index: dict[str, Any] = {}
    owners: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    info_cache: dict[str, dict[str, Any]] = {}
    tube_cache: dict[str, list[dict[str, Any]]] = {}
    disagreements = []
    for path in files:
        payload = json.loads(path.read_text())
        name = payload["video_id"]
        if payload.get("gt_used_for_segmentation") is not False:
            raise RuntimeError(f"{name} segmentation used GT")
        if payload.get("method") != "M1":
            raise RuntimeError(f"{name} method is {payload.get('method')}")
        if name not in short:
            raise RuntimeError(f"no short id for {name}")
        events = load_events(cfg, name)
        gt_index[name] = {
            "video_name": name,
            "short_id": short[name],
            "gt_status": "candidate",
            "human_verified": False,
            "event_count": len(events),
            "events": events,
        }
        if name not in info_cache:
            info_cache[name] = probe(resolve_video(cfg, name))
            tube_cache[name] = motion_tubes(cfg, name, info_cache[name]["width"], info_cache[name]["height"])
        info = info_cache[name]
        boundaries = [float(b) for b in payload["boundaries_sec"]]
        segments = payload["segments"]
        expected_edges = [0.0, *boundaries, float(payload["duration_sec"])]
        if len(segments) != len(expected_edges) - 1:
            raise RuntimeError(f"{name} segment count does not match boundaries")
        for seg, left, right in zip(segments, expected_edges, expected_edges[1:]):
            if abs(float(seg["start_sec"]) - left) > 1e-6 or abs(float(seg["end_sec"]) - right) > 1e-6:
                raise RuntimeError(f"{name} segment {seg['segment_id']} does not follow the frozen boundaries")
            start, end = float(seg["start_sec"]), float(seg["end_sec"])
            contained, boundary = [], []
            for ev in events:
                kind = classify_event(ev, start, end, eps)
                ov = overlap_sec(ev["start_sec"], ev["end_sec"], start, end)
                row = {
                    "event_id": ev["event_id"],
                    "overlap_sec": round(max(0.0, ov), 4),
                    "event_start_sec": ev["start_sec"],
                    "event_end_sec": ev["end_sec"],
                }
                if kind == "contained":
                    contained.append(row)
                elif kind == "boundary":
                    boundary.append(row)
            file_full = set(seg["fully_contained_gt_events"])
            mine = {r["event_id"] for r in contained}
            if file_full != mine:
                disagreements.append(
                    {
                        "video_name": name,
                        "segment_id": seg["segment_id"],
                        "file_only": sorted(file_full - mine),
                        "recompute_only": sorted(mine - file_full),
                    }
                )
            manual = manual_for_clip(cfg, name, start, end, info["width"], info["height"])
            intersecting = []
            for tube in tube_cache[name]:
                t0, t1 = tube["time_range"]
                ov = overlap_sec(start, end, t0, t1)
                if ov > 1e-3:
                    intersecting.append(
                        {
                            "roi_id": tube["roi_id"],
                            "bbox_xyxy": tube["bbox_xyxy"],
                            "time_range": tube["time_range"],
                            "overlap_sec": round(ov, 4),
                            "num_cells": tube["num_cells"],
                        }
                    )
            clip_id = f"{short[name]}_c{int(seg['segment_id']):02d}"
            clip = {
                "clip_id": clip_id,
                "short_id": short[name],
                "video_name": name,
                "segment_id": int(seg["segment_id"]),
                "segment_file": str(path),
                "segment_sha256": sha256_file(path),
                "method": "M1",
                "gt_used_for_segmentation": False,
                "boundaries_sec": boundaries,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": float(seg["duration_sec"]),
                "video_duration_sec": float(payload["duration_sec"]),
                "width": info["width"],
                "height": info["height"],
                "native_fps": info["native_fps"],
                "video_path": info["path"],
                "n_gt_events_in_video": len(events),
                "fully_contained_event_ids": [r["event_id"] for r in contained],
                "boundary_event_ids": [r["event_id"] for r in boundary],
                "fully_contained_detail": contained,
                "boundary_detail": boundary,
                "n_fully_contained": len(contained),
                "n_boundary": len(boundary),
                "segmentation_file_fully_contained": list(seg["fully_contained_gt_events"]),
                "manual_roi": {k: v for k, v in manual.items() if k != "array"},
                "s1_status": "ready" if manual["available"] else "held_missing_manual_roi",
                "motion_tubes_intersecting": intersecting,
                "n_motion_tubes": len(intersecting),
            }
            clips.append(clip)
            for row in contained:
                owners[(name, row["event_id"])].append(
                    {"clip_id": clip_id, "overlap_sec": row["overlap_sec"], "start_sec": start}
                )
    if len(clips) != 34:
        raise RuntimeError(f"expected 34 M1 clips, built {len(clips)}")
    suppressed = []
    for key, group in owners.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda r: (-r["overlap_sec"], r["start_sec"], r["clip_id"]))
        keep = group[0]["clip_id"]
        for extra in group[1:]:
            clip = next(c for c in clips if c["clip_id"] == extra["clip_id"])
            eid = key[1]
            clip["fully_contained_event_ids"].remove(eid)
            clip["fully_contained_detail"] = [r for r in clip["fully_contained_detail"] if r["event_id"] != eid]
            clip["n_fully_contained"] = len(clip["fully_contained_event_ids"])
            clip.setdefault("duplicate_contained_suppressed", []).append(
                {"event_id": eid, "kept_in_clip": keep, "overlap_sec": extra["overlap_sec"]}
            )
            suppressed.append({"video_name": key[0], "event_id": eid, "dropped_clip": extra["clip_id"], "kept_clip": keep})
    owned = Counter()
    for clip in clips:
        for eid in clip["fully_contained_event_ids"]:
            owned[(clip["video_name"], eid)] += 1
    dupes = [k for k, v in owned.items() if v != 1]
    if dupes:
        raise RuntimeError(f"fully contained events still duplicated: {dupes[:8]}")
    if export_frames:
        export_annotation_frames(clips)
    manifest = {
        "created_at": utc_now(),
        "segmentation": "M1_official",
        "segmentation_modified": False,
        "gt_used_for_segmentation": False,
        "n_videos": 8,
        "n_clips": len(clips),
        "containment_eps_sec": eps,
        "manual_roi_rule": (
            "Reuse the gold-case human xywh when the clip overlaps that annotated window "
            f"by at least {cfg['experiment']['manual_roi_min_overlap_sec']}s. "
            "The same box is then applied to every frame of the clip. "
            "Otherwise S1 is held and a clean representative frame is exported. No box is invented."
        ),
        "motion_roi_source": "stage3_light_uf_compose tubes, times from the stage2 gap-fusion timeline. No separate stage4 tube file exists.",
        "containment_disagreements_vs_segmentation_file": disagreements,
        "duplicate_contained_suppressed": suppressed,
        "clips": clips,
    }
    return {"manifest": manifest, "gt_index": gt_index, "tubes": tube_cache}


def export_annotation_frames(clips: list[dict[str, Any]]) -> None:
    frame_dir = OUT / "manual_roi_annotation" / "frames"
    viz_dir = OUT / "roi_visualizations"
    frame_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for clip in clips:
        mid = (clip["start_sec"] + clip["end_sec"]) / 2.0
        first = clip["start_sec"]
        decoded = decode_nearest(clip["video_path"], sorted({first, mid}))
        by_target = {round(d["target_time_sec"], 6): d for d in decoded}
        mid_dec = by_target[round(mid, 6)]
        colors = [(0, 255, 255), (255, 64, 255), (255, 176, 0), (80, 255, 80), (80, 160, 255)]
        boxes = []
        manual = clip["manual_roi"]
        if manual.get("available"):
            boxes.append({"bbox_xyxy": manual["applied_xyxy"], "label": "manual", "color": (255, 220, 0)})
        for i, tube in enumerate(clip["motion_tubes_intersecting"]):
            t0, t1 = tube["time_range"]
            if t0 - 1e-6 <= mid <= t1 + 1e-6:
                boxes.append({"bbox_xyxy": tube["bbox_xyxy"], "label": tube["roi_id"], "color": colors[i % len(colors)]})
        viz_name = f"{clip['clip_id']}_mid.jpg"
        draw_boxes(mid_dec["array"], boxes).save(viz_dir / viz_name, quality=90)
        clip["roi_visualization"] = str(viz_dir / viz_name)
        if clip["s1_status"] != "held_missing_manual_roi":
            continue
        image_name = f"{clip['clip_id']}_representative.jpg"
        save_rgb(frame_dir / image_name, mid_dec["array"])
        first_dec = by_target[round(first, 6)]
        first_name = f"{clip['clip_id']}_first.jpg"
        save_rgb(frame_dir / first_name, first_dec["array"])
        row = {
            "clip_id": clip["clip_id"],
            "video_name": clip["video_name"],
            "start_sec": clip["start_sec"],
            "end_sec": clip["end_sec"],
            "duration_sec": clip["duration_sec"],
            "frame_width": clip["width"],
            "frame_height": clip["height"],
            "representative_time_sec": mid,
            "representative_source_time_sec": mid_dec["source_time_sec"],
            "representative_frame": str(frame_dir / image_name),
            "first_frame_time_sec": first,
            "first_frame_source_time_sec": first_dec["source_time_sec"],
            "first_frame": str(frame_dir / first_name),
            "manual_roi_xywh": None,
            "hold_reason": clip["manual_roi"]["reason"],
            "instruction": (
                "Draw one xywh in source pixels. It will be applied to every frame of this clip. "
                "Leave manual_roi_xywh null until a person fills it. Do not invent a box."
            ),
        }
        requests.append(row)
        clip["annotation_request"] = row
    write_json(
        OUT / "manual_roi_annotation" / "manifest.json",
        {
            "created_at": utc_now(),
            "n_clips_missing_manual_roi": len(requests),
            "note": "These clips have no reusable human box. S1 is not run. Frames contain no drawn box.",
            "clips": requests,
        },
    )


def assert_no_gt_leak(prompt: str, gt_index: dict[str, Any]) -> None:
    for video in gt_index.values():
        for ev in video["events"]:
            if ev["caption"] and ev["caption"] in prompt:
                raise RuntimeError("GT caption leaked into the prompt")


def allocate(cfg: dict[str, Any], items: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    vmin, vmax = int(cfg["video"]["min_pixels"]), int(cfg["video"]["max_pixels"])
    floor = int(vmin * 1.05)

    def total(cap: int) -> int:
        s = 0
        for it in items:
            h, w = smart_resize(it["native_h"], it["native_w"], vmin, max(cap, floor))
            s += video_tokens(it["frames"], h, w)
        return s

    cap = vmax
    uncapped = total(vmax) if items else 0
    if items and uncapped > budget:
        lo, hi = floor, vmax
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total(mid) <= budget:
                lo = mid
            else:
                hi = mid - 1
        cap = lo
        while total(cap) > budget and cap > floor:
            cap -= 256
    out = []
    for it in items:
        uh, uw = smart_resize(it["native_h"], it["native_w"], vmin, vmax)
        h, w = smart_resize(it["native_h"], it["native_w"], vmin, max(cap, floor))
        out.append(
            {
                "roi_id": it["roi_id"],
                "input_h": h,
                "input_w": w,
                "unconstrained_h": uh,
                "unconstrained_w": uw,
                "tokens": video_tokens(it["frames"], h, w),
                "tokens_at_max_pixels": video_tokens(it["frames"], uh, uw),
            }
        )
    binding = "budget" if items and uncapped > budget else "per_frame_max_pixels_or_native"
    used = sum(r["tokens"] for r in out)
    return {
        "cap_pixels": cap if items else None,
        "tokens_at_vmax": uncapped,
        "binding": binding,
        "fits": used <= budget,
        "items": out,
    }


def roi_records_for(
    spatial: str,
    grid: list[float],
    clip: dict[str, Any],
    tubes: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    start, end = clip["start_sec"], clip["end_sec"]
    width, height = clip["width"], clip["height"]
    if spatial == "S0":
        raw = [
            {
                "roi_id": "full",
                "bbox_xyxy": [0, 0, width, height],
                "time_range": [start, end],
                "source": "full_frame",
            }
        ]
    elif spatial == "S1":
        if not clip["manual_roi"].get("available"):
            return "held_missing_manual_roi", []
        manual = clip["manual_roi"]
        raw = [
            {
                "roi_id": "manual1",
                "bbox_xywh_requested": manual["requested_xywh"],
                "bbox_xywh": manual["applied_xywh"],
                "bbox_xyxy": manual["applied_xyxy"],
                "time_range": [start, end],
                "source": "manual_roi",
                "roi_mode": "single_manual_roi",
                "clipped_to_frame": manual["clipped_to_frame"],
                "clip_reason": manual["clip_reason"],
                "coverage": manual["coverage"],
                "annotated_window": manual["annotated_window"],
                "annotated_overlap_sec": manual["overlap_sec"],
            }
        ]
    else:
        raw = []
        for tube in tubes:
            t0, t1 = tube["time_range"]
            lo, hi = max(start, t0), min(end, t1)
            if hi <= lo + 1e-6:
                continue
            raw.append({**tube, "intersection_time_range": [lo, hi]})
    rois = []
    for r in raw:
        t0, t1 = r.get("intersection_time_range", r["time_range"])
        idx = [k for k, t in enumerate(grid) if t0 - 1e-6 <= t <= t1 + 1e-6]
        item = dict(r)
        item["grid_idx"] = idx
        if not idx:
            item["excluded"] = "no sampled frame inside the ROI and the clip"
        rois.append(item)
    used = [r for r in rois if r["grid_idx"]]
    if not used:
        return "error_no_roi_frames", rois
    return "planned", rois


def build_plans(cfg: dict[str, Any], manifest: dict[str, Any], gt_index: dict[str, Any], tubes: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    budget = int(cfg["experiment"]["token_budget"])
    lock_fps = float(cfg["experiment"]["resolution_lock_fps"])
    fps_list = [float(f) for f in cfg["experiment"]["fps"]]
    if lock_fps not in fps_list:
        raise RuntimeError("resolution_lock_fps must be one of the experiment FPS values")
    vmin = int(cfg["video"]["min_pixels"])
    imin, imax = int(cfg["image"]["min_pixels"]), int(cfg["image"]["max_pixels"])
    drafts: dict[tuple[str, str, float], dict[str, Any]] = {}
    for clip in manifest["clips"]:
        start, end = clip["start_sec"], clip["end_sec"]
        gh, gw = smart_resize(clip["height"], clip["width"], imin, imax)
        gtok = image_tokens(gh, gw)
        image_time = start
        for fps in fps_list:
            grid = grid_times(start, end, fps)
            for spatial in cfg["experiment"]["spatial"]:
                status, rois = roi_records_for(spatial, grid, clip, tubes[clip["video_name"]])
                drafts[(clip["clip_id"], spatial, fps)] = {
                    "clip": clip,
                    "fps": fps,
                    "spatial": spatial,
                    "status": status,
                    "rois": rois,
                    "grid": grid,
                    "image_time": image_time,
                    "global_hw": [gh, gw],
                    "image_tokens_one": gtok,
                }
    plans = []
    for clip in manifest["clips"]:
        for spatial in cfg["experiment"]["spatial"]:
            group = [drafts[(clip["clip_id"], spatial, fps)] for fps in fps_list]
            image_tokens_n = 0 if spatial == "S0" else group[0]["image_tokens_one"]
            video_budget = budget - image_tokens_n
            planned = [d for d in group if d["status"] == "planned"]
            ref = next((d for d in planned if d["fps"] == lock_fps), None)
            if ref is None and planned:
                ref = max(planned, key=lambda d: sum(len(r["grid_idx"]) for r in d["rois"]))
            if ref is None:
                alloc = None
                res_by_roi = {}
                used_lock = lock_fps
            else:
                items = []
                for r in ref["rois"]:
                    if not r["grid_idx"]:
                        continue
                    x0, y0, x1, y1 = r["bbox_xyxy"]
                    items.append(
                        {
                            "roi_id": r["roi_id"],
                            "native_h": y1 - y0,
                            "native_w": x1 - x0,
                            "frames": len(r["grid_idx"]),
                        }
                    )
                alloc = allocate(cfg, items, video_budget)
                res_by_roi = {a["roi_id"]: a for a in alloc["items"]}
                used_lock = float(ref["fps"])
            for draft in group:
                plan = assemble_condition(
                    cfg, draft, alloc, res_by_roi, image_tokens_n, video_budget, budget, vmin, gt_index, used_lock
                )
                plans.append(plan)
    if len(plans) != 34 * 4 * 3:
        raise RuntimeError(f"expected 408 conditions, built {len(plans)}")
    assert_resolution_lock(plans, lock_fps)
    return plans


def assemble_condition(
    cfg: dict[str, Any],
    draft: dict[str, Any],
    alloc: dict[str, Any] | None,
    res_by_roi: dict[str, dict[str, Any]],
    image_tokens_n: int,
    video_budget: int,
    budget: int,
    vmin: int,
    gt_index: dict[str, Any],
    lock_fps: float,
) -> dict[str, Any]:
    clip = draft["clip"]
    spatial = draft["spatial"]
    fps = draft["fps"]
    start, end = clip["start_sec"], clip["end_sec"]
    plan: dict[str, Any] = {
        "condition_id": cond_id(clip["clip_id"], spatial, fps),
        "clip_id": clip["clip_id"],
        "video_id": clip["short_id"],
        "video_name": clip["video_name"],
        "segment_id": clip["segment_id"],
        "spatial": spatial,
        "fps": fps,
        "window": [start, end],
        "duration_sec": clip["duration_sec"],
        "native_resolution": [clip["width"], clip["height"]],
        "native_fps": clip["native_fps"],
        "video_path": clip["video_path"],
        "grid_times": [round(t, 6) for t in draft["grid"]],
        "grid_frame_count": len(draft["grid"]),
        "token_budget": budget,
        "timestamp_mode": "source_timestamp",
        "resolution_lock_fps": lock_fps,
        "primary_event_ids": list(clip["fully_contained_event_ids"]),
        "boundary_event_ids": list(clip["boundary_event_ids"]),
        "status": draft["status"],
    }
    if spatial == "S0":
        plan["global_images"] = []
    else:
        gh, gw = draft["global_hw"]
        plan["global_images"] = [
            {
                "role": "clip_first_frame",
                "time_target": draft["image_time"],
                "native_resolution": [clip["width"], clip["height"]],
                "input_resolution": [gw, gh],
                "tokens": draft["image_tokens_one"],
                "shared_with": ["S1", "S2"],
            }
        ]
    if draft["status"] != "planned":
        plan["rois"] = draft["rois"]
        plan["roi_count_used"] = 0
        plan["planned_image_tokens"] = image_tokens_n if spatial != "S0" else 0
        plan["planned_video_tokens"] = 0
        plan["planned_total_visual_tokens"] = None
        plan["budget_ok"] = False
        plan["resolution_lowered_for_budget"] = False
        plan["hold_reason"] = clip["manual_roi"].get("reason", draft["status"])
        return plan
    used = [r for r in draft["rois"] if r["grid_idx"]]
    vmax = int(cfg["video"]["max_pixels"])
    own_items = []
    for r in used:
        x0, y0, x1, y1 = r["bbox_xyxy"]
        native_h, native_w = y1 - y0, x1 - x0
        locked = res_by_roi.get(r["roi_id"])
        if locked is None:
            cap = alloc["cap_pixels"] if alloc and alloc["cap_pixels"] else vmax
            h, w = smart_resize(native_h, native_w, vmin, max(int(cap), int(vmin * 1.05)))
            uh, uw = smart_resize(native_h, native_w, vmin, vmax)
            r["resolution_source"] = "pixel_cap_of_lock_fps_roi_absent_there"
        else:
            h, w = locked["input_h"], locked["input_w"]
            uh, uw = locked["unconstrained_h"], locked["unconstrained_w"]
            r["resolution_source"] = "locked_from_3fps"
        r["roi_native_size"] = [native_w, native_h]
        r["frames"] = len(r["grid_idx"])
        r["input_resolution"] = [w, h]
        r["unconstrained_resolution"] = [uw, uh]
        r["planned_video_tokens"] = video_tokens(r["frames"], h, w)
        r["tokens_at_max_pixels"] = video_tokens(r["frames"], uh, uw)
        r["resolution_lowered_for_budget"] = [w, h] != [uw, uh]
        r["model_frames_with_temporal_padding"] = math.ceil(r["frames"] / TEMPORAL) * TEMPORAL
        own_items.append(r["tokens_at_max_pixels"])
    plan["rois"] = draft["rois"]
    plan["roi_count_used"] = len(used)
    plan["roi_mode"] = {"S0": "full_frame", "S1": "single_manual_roi", "S2": "motion_multi_roi"}[spatial]
    plan["allocation"] = None if alloc is None else {k: v for k, v in alloc.items() if k != "items"}
    plan["planned_image_tokens"] = 0 if spatial == "S0" else image_tokens_n
    plan["planned_video_tokens"] = sum(r["planned_video_tokens"] for r in used)
    plan["planned_total_visual_tokens"] = plan["planned_image_tokens"] + plan["planned_video_tokens"]
    plan["tokens_at_max_pixels_this_fps"] = plan["planned_image_tokens"] + sum(own_items)
    plan["own_fps_fits_at_max_pixels"] = plan["tokens_at_max_pixels_this_fps"] <= budget
    plan["resolution_lowered_for_budget"] = any(r["resolution_lowered_for_budget"] for r in used)
    plan["lowered_because"] = (
        "locked_to_3fps_token_budget" if plan["resolution_lowered_for_budget"] else ""
    )
    plan["budget_ok"] = plan["planned_total_visual_tokens"] <= budget
    plan["lock_fps_fits"] = bool(alloc["fits"]) if alloc else False
    plan["video_token_budget_after_image"] = video_budget
    if not plan["budget_ok"]:
        plan["status"] = "budget_infeasible"
    image_time = None if spatial == "S0" else draft["image_time"]
    prompt = full_prompt(spatial, start, end, len(used), image_time)
    assert_no_gt_leak(prompt, gt_index)
    plan["prompt_text"] = prompt
    plan["prompt_sha"] = sha16(prompt)
    plan["common_prompt_sha"] = sha16(load_common_prompt())
    return plan


def assert_resolution_lock(plans: list[dict[str, Any]], lock_fps: float) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in plans:
        if p["status"] == "planned":
            groups[(p["clip_id"], p["spatial"])].append(p)
    for key, group in groups.items():
        ref = next((p for p in group if p["fps"] == lock_fps), None)
        if ref is None:
            ref = max(group, key=lambda p: p["grid_frame_count"])
        ref_res = {
            r["roi_id"]: tuple(r["input_resolution"])
            for r in ref["rois"]
            if r.get("grid_idx") and r.get("input_resolution")
        }
        for p in group:
            for r in p["rois"]:
                if not r.get("grid_idx") or not r.get("input_resolution"):
                    continue
                if r["roi_id"] in ref_res and tuple(r["input_resolution"]) != ref_res[r["roi_id"]]:
                    raise RuntimeError(
                        f"{p['condition_id']} resolution drifted from the {ref['fps']} FPS lock"
                    )
            if p["planned_total_visual_tokens"] > p["token_budget"]:
                raise RuntimeError(f"{p['condition_id']} exceeds the visual token budget")


def cmd_manifest(_args: argparse.Namespace) -> None:
    cfg = load_cfg()
    built = build_manifest(cfg, export_frames=True)
    write_json(OUT / "config" / "clip_manifest.json", built["manifest"])
    write_json(OUT / "config" / "gt_index.json", built["gt_index"])
    rows = []
    for c in built["manifest"]["clips"]:
        rows.append(
            {
                "clip_id": c["clip_id"],
                "video_name": c["video_name"],
                "segment_id": c["segment_id"],
                "start_sec": round(c["start_sec"], 4),
                "end_sec": round(c["end_sec"], 4),
                "duration_sec": round(c["duration_sec"], 4),
                "n_fully_contained_gt": c["n_fully_contained"],
                "n_boundary_gt": c["n_boundary"],
                "fully_contained_event_ids": "|".join(c["fully_contained_event_ids"]),
                "boundary_event_ids": "|".join(c["boundary_event_ids"]),
                "boundaries_sec": "|".join(f"{b:.4f}" for b in c["boundaries_sec"]),
                "s1_status": c["s1_status"],
                "manual_coverage": c["manual_roi"].get("coverage", ""),
                "manual_overlap_sec": c["manual_roi"].get("overlap_sec", ""),
                "n_motion_tubes": c["n_motion_tubes"],
                "width": c["width"],
                "height": c["height"],
            }
        )
    with (OUT / "config" / "clip_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ready = sum(1 for c in built["manifest"]["clips"] if c["s1_status"] == "ready")
    print(
        f"clips {len(rows)} s1_ready {ready} "
        f"disagreements {len(built['manifest']['containment_disagreements_vs_segmentation_file'])} "
        f"suppressed {len(built['manifest']['duplicate_contained_suppressed'])}"
    )


def cmd_plan(_args: argparse.Namespace) -> None:
    cfg = load_cfg()
    manifest_path = OUT / "config" / "clip_manifest.json"
    gt_path = OUT / "config" / "gt_index.json"
    if not manifest_path.is_file() or not gt_path.is_file():
        raise RuntimeError("run manifest first")
    manifest = json.loads(manifest_path.read_text())
    gt_index = json.loads(gt_path.read_text())
    tubes = {}
    for clip in manifest["clips"]:
        if clip["video_name"] not in tubes:
            tubes[clip["video_name"]] = motion_tubes(cfg, clip["video_name"], clip["width"], clip["height"])
    plans = build_plans(cfg, manifest, gt_index, tubes)
    payload = {
        "created_at": utc_now(),
        "generation": generation_record(cfg),
        "common_prompt": load_common_prompt(),
        "common_prompt_sha": sha16(load_common_prompt()),
        "n_conditions": len(plans),
        "conditions": plans,
    }
    write_json(OUT / "config" / "plan.json", payload)
    rows = []
    for p in plans:
        used = [r for r in p["rois"] if r.get("grid_idx")]
        rows.append(
            {
                "condition_id": p["condition_id"],
                "clip_id": p["clip_id"],
                "spatial": p["spatial"],
                "fps": p["fps"],
                "status": p["status"],
                "duration_sec": round(p["duration_sec"], 3),
                "grid_frames": p["grid_frame_count"],
                "roi_count": p.get("roi_count_used", 0),
                "roi_frames": "|".join(str(r.get("frames", "")) for r in used),
                "input_resolution": "|".join(
                    f"{r['input_resolution'][0]}x{r['input_resolution'][1]}" for r in used if r.get("input_resolution")
                ),
                "unconstrained_resolution": "|".join(
                    f"{r['unconstrained_resolution'][0]}x{r['unconstrained_resolution'][1]}"
                    for r in used
                    if r.get("unconstrained_resolution")
                ),
                "image_tokens": p.get("planned_image_tokens"),
                "video_tokens": p.get("planned_video_tokens"),
                "total_tokens": p.get("planned_total_visual_tokens"),
                "tokens_at_max_pixels_this_fps": p.get("tokens_at_max_pixels_this_fps"),
                "resolution_lowered_for_budget": p.get("resolution_lowered_for_budget"),
                "own_fps_fits_at_max_pixels": p.get("own_fps_fits_at_max_pixels"),
                "budget_ok": p.get("budget_ok"),
                "n_primary_events": len(p.get("primary_event_ids") or []),
            }
        )
    with (OUT / "config" / "plan.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pilot = set(cfg["experiment"]["pilot_clips"])
    bad = [r for r in rows if r["clip_id"] in pilot and r["status"] != "planned"]
    print(f"planned {len(rows)} status {Counter(r['status'] for r in rows)}")
    if bad:
        print("pilot not runnable:", [r["condition_id"] + ":" + r["status"] for r in bad])
        raise SystemExit(2)


def load_model(cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        import torch.backends.python_native as pn

        pn.triton.enabled = False
    except Exception:
        pass
    path = cfg["model"]["path"]
    if not (Path(path) / "config.json").is_file():
        raise FileNotFoundError(path)
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        dtype=getattr(torch, cfg["model"]["dtype"]),
        device_map=cfg["model"]["device_map"],
        attn_implementation=cfg["model"]["attn_implementation"],
    )
    model.eval()
    return processor, model


def resize_frames(frames: np.ndarray, h: int, w: int):
    import torch
    from torchvision import transforms

    t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if t.shape[2] == h and t.shape[3] == w:
        return t.float()
    return transforms.functional.resize(
        t, [h, w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    ).float()


def prepare_inputs(cfg: dict[str, Any], processor, plan: dict[str, Any], gt_index: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    t_start = time.perf_counter()
    start, end = plan["window"]
    grid = grid_times(start, end, plan["fps"])
    if [round(t, 6) for t in grid] != plan["grid_times"]:
        raise RuntimeError("grid drifted from the plan")
    global_targets = [g["time_target"] for g in plan["global_images"]]
    sel = decode_nearest(plan["video_path"], grid)
    gsel = decode_nearest(plan["video_path"], global_targets) if global_targets else []
    src_times = [s["source_time_sec"] for s in sel]
    if len({round(t, 6) for t in src_times}) != len(src_times):
        raise RuntimeError("duplicate source frames in the sampling grid")
    max_err = max(s["abs_error_sec"] for s in sel + gsel)
    native_fps = float(plan["native_fps"])
    if max_err > 1.5 / native_fps:
        raise RuntimeError(f"timestamp error {max_err:.4f}s")
    content: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []
    metadatas: list[dict[str, Any]] = []
    used = [r for r in plan["rois"] if r.get("grid_idx")]
    for g, s in zip(plan["global_images"], gsel):
        gw, gh = g["input_resolution"]
        img = Image.fromarray(s["array"])
        img = img.resize((gw, gh), Image.BICUBIC) if (gw, gh) != img.size else img
        images.append(img)
        g["source_time_sec"] = s["source_time_sec"]
        g["abs_error_sec"] = s["abs_error_sec"]
        content.append({"type": "text", "text": f"[Full-frame image, t={s['source_time_sec']:.2f}s]"})
        content.append({"type": "image", "image": "memory://global"})
    roi_ts: dict[str, list[float]] = {}
    width, height = plan["native_resolution"]
    for i, r in enumerate(used, 1):
        x0, y0, x1, y1 = r["bbox_xyxy"]
        frames = np.stack([sel[k]["array"][y0:y1, x0:x1] for k in r["grid_idx"]], axis=0)
        if frames.size == 0 or frames.shape[1] == 0 or frames.shape[2] == 0:
            raise RuntimeError(f"{r['roi_id']} crop is empty")
        if list(frames.shape[1:3]) != [r["roi_native_size"][1], r["roi_native_size"][0]]:
            raise RuntimeError(f"{r['roi_id']} crop {frames.shape} != native {r['roi_native_size']}")
        w, h = r["input_resolution"]
        videos.append(resize_frames(frames, h, w))
        stamps = [src_times[k] for k in r["grid_idx"]]
        roi_ts[r["roi_id"]] = stamps
        r["source_time_first_last"] = [stamps[0], stamps[-1]]
        indices, fps_enc = encode_timestamps(stamps, scale=1000)
        metadatas.append(
            {
                "fps": fps_enc,
                "frames_indices": indices,
                "total_num_frames": max(indices) + 1,
                "video_backend": "av",
            }
        )
        if plan["spatial"] == "S0":
            label = f"[Full-frame video, {stamps[0]:.2f}s-{stamps[-1]:.2f}s]"
        else:
            label = (
                f"[ROI {i} video: crop x={x0}-{x1}, y={y0}-{y1} of the {width}x{height} frame, "
                f"{stamps[0]:.2f}s-{stamps[-1]:.2f}s]"
            )
        content.append({"type": "text", "text": label})
        content.append({"type": "video", "video": f"memory://roi{i}"})
    image_time = gsel[0]["source_time_sec"] if gsel else None
    prompt = full_prompt(plan["spatial"], start, end, len(used), image_time)
    assert_no_gt_leak(prompt, gt_index)
    content.append({"type": "text", "text": prompt})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images or None,
        videos=videos or None,
        video_metadata=metadatas or None,
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )
    grids = [[int(v) for v in row] for row in inputs["video_grid_thw"].tolist()] if "video_grid_thw" in inputs else []
    igrids = [[int(v) for v in row] for row in inputs["image_grid_thw"].tolist()] if "image_grid_thw" in inputs else []
    vid_tok = sum(t * h * w // 4 for t, h, w in grids)
    img_tok = sum(t * h * w // 4 for t, h, w in igrids)
    ph_v = int((inputs["input_ids"] == processor.video_token_id).sum())
    ph_i = int((inputs["input_ids"] == processor.image_token_id).sum())
    if ph_v != vid_tok or ph_i != img_tok:
        raise RuntimeError(f"placeholder mismatch video {ph_v}/{vid_tok} image {ph_i}/{img_tok}")
    for r, grid_row in zip(used, grids):
        w, h = r["input_resolution"]
        expect = [math.ceil(r["frames"] / TEMPORAL), h // 16, w // 16]
        if grid_row != expect:
            raise RuntimeError(f"{r['roi_id']}: grid {grid_row} != expected {expect}")
        r["video_grid_thw"] = grid_row
        r["measured_video_tokens"] = grid_row[0] * grid_row[1] * grid_row[2] // 4
    if vid_tok != plan["planned_video_tokens"] or img_tok != plan["planned_image_tokens"]:
        raise RuntimeError(
            f"measured tokens {img_tok}+{vid_tok} differ from plan {plan['planned_image_tokens']}+{plan['planned_video_tokens']}"
        )
    if img_tok + vid_tok > int(cfg["experiment"]["token_budget"]):
        raise RuntimeError(f"visual tokens {img_tok + vid_tok} exceed 40960")
    decoded = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    expected = []
    for r in used:
        idx, scale = encode_timestamps(roi_ts[r["roi_id"]], scale=1000)
        st = [i / scale for i in idx]
        st = st + [st[-1]] * (len(st) % 2)
        expected += [f"{(st[i] + st[i + 1]) / 2:.1f}" for i in range(0, len(st), 2)]
    found = re.findall(r"<(\d+(?:\.\d+)?) seconds>", decoded)
    if found != expected:
        raise RuntimeError(f"timestamp markers differ: {found[:8]} vs {expected[:8]}")
    assert_no_gt_leak(decoded, gt_index)
    return {
        "inputs": inputs,
        "prompt_text": prompt,
        "image_tokens": img_tok,
        "video_tokens": vid_tok,
        "total_visual_tokens": img_tok + vid_tok,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "text_tokens": int(inputs["input_ids"].shape[1]) - img_tok - vid_tok,
        "sampling": [
            {
                "k": k,
                "target": s["target_time_sec"],
                "source": s["source_time_sec"],
                "abs_err": round(s["abs_error_sec"], 5),
            }
            for k, s in enumerate(sel)
        ],
        "global_sampling": [
            {
                "role": g["role"],
                "target": g["time_target"],
                "source": g["source_time_sec"],
                "abs_err": round(g["abs_error_sec"], 5),
                "input_resolution": g["input_resolution"],
                "tokens": g["tokens"],
            }
            for g in plan["global_images"]
        ],
        "max_abs_error_sec": max_err,
        "roi_timestamps": roi_ts,
        "timestamp_markers_verified": len(found),
        "prep_sec": time.perf_counter() - t_start,
        "plan": plan,
    }


def caption_flags(text: str, hit_limit: bool) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    flags = []
    if len(parts) <= 1:
        flags.append("single_sentence")
    norms = [re.sub(r"\s+", " ", p.lower()) for p in parts]
    if any(v >= 2 for v in Counter(norms).values()):
        flags.append("repeated_sentence")
    words = text.split()
    if len(words) >= 30:
        tail = " ".join(words[-12:])
        if tail and text.count(tail) >= 3:
            flags.append("repetition_loop")
    if re.search(r"(?m)^\s*(\d+[\.\)]|[-*])\s+", text):
        flags.append("list_or_numbering")
    if re.search(r"\b\d+(?:\.\d+)?\s*s\b|timestamp", text, re.I):
        flags.append("timestamp_like")
    if hit_limit:
        flags.append("hit_max_new_tokens")
    if hit_limit or "repetition_loop" in flags:
        flags.append("generation_failure")
    return flags


def generate_one(cfg: dict[str, Any], model, processor, prepared: dict[str, Any]) -> dict[str, Any]:
    import torch

    gen = generation_record(cfg)
    torch.manual_seed(int(gen["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(gen["seed"]))
    device = next(model.parameters()).device
    inputs = prepared["inputs"].to(device)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    oom = False
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, **generation_kwargs(gen))
    except BaseException as exc:  # noqa: BLE001
        oom = is_oom(exc)
        raise
    finally:
        if oom:
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peaks = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**20, 1) for i in range(torch.cuda.device_count())}
    n_in = int(inputs["input_ids"].shape[1])
    new = out[0, n_in:]
    text = processor.batch_decode(new.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    n_out = int(new.shape[0])
    last = int(new[-1]) if n_out else None
    eos_ids = list(gen["eos_token_id"] or [])
    hit = gen["max_new_tokens"] is not None and n_out >= int(gen["max_new_tokens"]) and last not in eos_ids
    flags = caption_flags(text, hit)
    eos = last in eos_ids if last is not None else False
    del inputs, out, new
    return {
        "caption": text,
        "output_token_count": n_out,
        "hit_token_limit": hit,
        "terminated_normally": (not hit) and eos,
        "eos": eos,
        "caption_flags": flags,
        "inference_time_sec": round(elapsed, 3),
        "peak_gpu_memory_mib": max(peaks.values()) if peaks else None,
        "peak_gpu_memory_mib_per_device": peaks,
        "oom": False,
    }


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def select_plans(plans: list[dict[str, Any]], cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pilot:
        clips = set(cfg["experiment"]["pilot_clips"])
    elif args.clips:
        clips = set(args.clips.split(","))
    else:
        clips = None
    spatials = set(args.spatial.split(",")) if args.spatial else None
    fps = {float(x) for x in args.fps.split(",")} if args.fps else None
    chosen = []
    for p in plans:
        if clips and p["clip_id"] not in clips:
            continue
        if spatials and p["spatial"] not in spatials:
            continue
        if fps and float(p["fps"]) not in fps:
            continue
        chosen.append(p)
    return chosen


def cmd_infer(args: argparse.Namespace) -> None:
    import torch

    cfg = load_cfg()
    gen = generation_record(cfg)
    prompt_sha = sha16(load_common_prompt())
    plan_payload = json.loads((OUT / "config" / "plan.json").read_text())
    gt_index = json.loads((OUT / "config" / "gt_index.json").read_text())
    ckpt_dir = OUT / "results" / "checkpoints"
    cap_dir = OUT / "captions"
    inp_dir = OUT / "inputs"
    for d in (ckpt_dir, cap_dir, inp_dir, OUT / "reports" / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    wanted = select_plans(plan_payload["conditions"], cfg, args)
    runnable = [p for p in wanted if p["status"] == "planned"]
    skipped = [p for p in wanted if p["status"] != "planned"]
    if skipped:
        log.info("skipping %d non-runnable conditions", len(skipped))

    def reusable(p: dict[str, Any]) -> bool:
        f = ckpt_dir / f"{p['condition_id']}.json"
        if not f.is_file():
            return False
        rec = json.loads(f.read_text())
        same = rec.get("generation_sha") == gen["sha"] and rec.get("prompt_sha_common") == prompt_sha
        if rec.get("status") == "ok" and same:
            return True
        if rec.get("status") not in (None, "ok") and same and not args.retry_failed:
            return True
        return False

    todo = [p for p in runnable if not reusable(p)]
    log.info("%d selected, %d runnable, %d to run", len(wanted), len(runnable), len(todo))
    if not todo:
        return
    processor, model = load_model(cfg)
    base = {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 2**20, 1) for i in range(torch.cuda.device_count())}
    log.info("model loaded; resident MiB %s", base)
    for p in todo:
        cid = p["condition_id"]
        rec: dict[str, Any] = {
            "condition_id": cid,
            "clip_id": p["clip_id"],
            "video_id": p["video_id"],
            "video_name": p["video_name"],
            "spatial": p["spatial"],
            "fps": p["fps"],
            "window": p["window"],
            "started_at": utc_now(),
            "model_path": cfg["model"]["path"],
            "model_id": cfg["model"]["id"],
            "model_revision": cfg["model"]["revision"],
            "model_resident_mib": base,
            "generation": gen,
            "generation_sha": gen["sha"],
            "prompt_sha_common": prompt_sha,
            "oom": False,
        }
        prepared = None
        try:
            prepared = prepare_inputs(cfg, processor, p, gt_index)
            produced = generate_one(cfg, model, processor, prepared)
            rec.update(status="ok", **produced)
            for key in (
                "image_tokens", "video_tokens", "total_visual_tokens", "prompt_tokens", "text_tokens",
                "prep_sec", "max_abs_error_sec", "timestamp_markers_verified", "roi_timestamps",
                "prompt_text", "global_sampling",
            ):
                rec[key] = prepared[key]
            rec["resolution_lowered_for_budget"] = p.get("resolution_lowered_for_budget")
            rec["input_resolution"] = [
                r.get("input_resolution") for r in prepared["plan"]["rois"] if r.get("grid_idx")
            ]
            rec["frame_counts"] = [r.get("frames") for r in prepared["plan"]["rois"] if r.get("grid_idx")]
            rec["primary_event_ids"] = p.get("primary_event_ids")
            rec["boundary_event_ids"] = p.get("boundary_event_ids")
            write_json(
                inp_dir / f"{cid}.json",
                {
                    "condition_id": cid,
                    "sampling": prepared["sampling"],
                    "global_sampling": prepared["global_sampling"],
                    "roi_timestamps": prepared["roi_timestamps"],
                    "prompt_text": prepared["prompt_text"],
                    "image_tokens": prepared["image_tokens"],
                    "video_tokens": prepared["video_tokens"],
                    "total_visual_tokens": prepared["total_visual_tokens"],
                    "resolution_lowered_for_budget": p.get("resolution_lowered_for_budget"),
                    "rois": [
                        {
                            "roi_id": r["roi_id"],
                            "bbox_xyxy": r["bbox_xyxy"],
                            "roi_native_size": r.get("roi_native_size"),
                            "input_resolution": r.get("input_resolution"),
                            "unconstrained_resolution": r.get("unconstrained_resolution"),
                            "frames": r.get("frames"),
                            "source": r.get("source"),
                            "time_range": r.get("time_range"),
                            "source_time_first_last": r.get("source_time_first_last"),
                            "measured_video_tokens": r.get("measured_video_tokens"),
                            "resolution_lowered_for_budget": r.get("resolution_lowered_for_budget"),
                        }
                        for r in prepared["plan"]["rois"]
                        if r.get("grid_idx")
                    ],
                },
            )
            (cap_dir / f"{cid}.txt").write_text(produced["caption"] + "\n", encoding="utf-8")
            log.info(
                "%s ok tok=%d out=%d flags=%s %.1fs peak=%s",
                cid, prepared["total_visual_tokens"], produced["output_token_count"],
                produced["caption_flags"], produced["inference_time_sec"], produced["peak_gpu_memory_mib"],
            )
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            rec.update(
                status="oom" if is_oom(exc) else "error",
                oom=is_oom(exc),
                error=f"{type(exc).__name__}: {exc}"[:2000],
                traceback=traceback.format_exc()[-3000:],
            )
            log.error("%s %s: %s", cid, rec["status"], str(exc)[:500])
        rec["finished_at"] = utc_now()
        write_json(ckpt_dir / f"{cid}.json", rec)
        del prepared
        gc.collect()
        torch.cuda.empty_cache()


def _load_record(condition_id: str) -> dict[str, Any] | None:
    path = OUT / "results" / "checkpoints" / f"{condition_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def result_row(p: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    pl = rec.get("plan", p) if rec.get("plan") else p
    used = [r for r in pl.get("rois", []) if r.get("grid_idx")]
    return {
        "condition_id": p["condition_id"],
        "clip_id": p["clip_id"],
        "video_id": p["video_id"],
        "video_name": p["video_name"],
        "spatial": p["spatial"],
        "fps": p["fps"],
        "status": rec.get("status", p["status"] if not rec else rec.get("status")),
        "plan_status": p["status"],
        "oom": rec.get("oom", False),
        "error": rec.get("error", ""),
        "window_start": p["window"][0],
        "window_end": p["window"][1],
        "duration_sec": p["duration_sec"],
        "native_resolution": "x".join(map(str, p["native_resolution"])),
        "roi_count": len(used),
        "input_resolution": "|".join(
            f"{r['input_resolution'][0]}x{r['input_resolution'][1]}" for r in used if r.get("input_resolution")
        ),
        "frame_count": "|".join(str(r.get("frames")) for r in used),
        "image_tokens": rec.get("image_tokens", p.get("planned_image_tokens")),
        "video_tokens": rec.get("video_tokens", p.get("planned_video_tokens")),
        "total_visual_tokens": rec.get("total_visual_tokens", p.get("planned_total_visual_tokens")),
        "resolution_lowered_for_budget": p.get("resolution_lowered_for_budget"),
        "lowered_because": p.get("lowered_because", ""),
        "own_fps_fits_at_max_pixels": p.get("own_fps_fits_at_max_pixels"),
        "tokens_at_max_pixels_this_fps": p.get("tokens_at_max_pixels_this_fps"),
        "text_tokens": rec.get("text_tokens"),
        "peak_gpu_memory_mib": rec.get("peak_gpu_memory_mib"),
        "inference_time_sec": rec.get("inference_time_sec"),
        "prep_sec": rec.get("prep_sec"),
        "output_tokens": rec.get("output_token_count"),
        "terminated_normally": rec.get("terminated_normally"),
        "eos": rec.get("eos"),
        "hit_token_limit": rec.get("hit_token_limit"),
        "caption_flags": "|".join(rec.get("caption_flags") or []),
        "max_abs_timestamp_error_sec": rec.get("max_abs_error_sec"),
        "n_primary_events": len(p.get("primary_event_ids") or []),
        "n_boundary_events": len(p.get("boundary_event_ids") or []),
        "primary_event_ids": "|".join(p.get("primary_event_ids") or []),
        "boundary_event_ids": "|".join(p.get("boundary_event_ids") or []),
        "caption": rec.get("caption", ""),
    }


def cmd_aggregate(_args: argparse.Namespace) -> None:
    plan = json.loads((OUT / "config" / "plan.json").read_text())
    rows = []
    blocks = [
        "# Motion-clip captions\n",
        "Each caption is one M1 clip. GT text was not in the prompt.\n",
        "S1/S2 receive one shared first-frame full image. Resolution is locked across FPS.\n",
    ]
    by_merge: dict[tuple[str, str, float], list[tuple[float, str]]] = defaultdict(list)
    for p in plan["conditions"]:
        rec = _load_record(p["condition_id"]) or {}
        row = result_row(p, rec)
        if not rec:
            row["status"] = p["status"] if p["status"] != "planned" else "not_run"
        rows.append(row)
        if rec.get("caption"):
            blocks.append(
                f"## {p['condition_id']}\n\n"
                f"duration={p['duration_sec']:.2f}s tokens={rec.get('total_visual_tokens')} "
                f"res={row['input_resolution']} frames={row['frame_count']} "
                f"lowered={p.get('resolution_lowered_for_budget')} flags={rec.get('caption_flags')} "
                f"eos={rec.get('eos')} gen_s={rec.get('inference_time_sec')}\n\n"
                f"{rec['caption']}\n"
            )
            by_merge[(p["video_name"], p["spatial"], float(p["fps"]))].append((p["window"][0], rec["caption"]))
    result_dir = OUT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "results.json", rows)
    with (result_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "captions" / "captions.md").write_text("\n".join(blocks), encoding="utf-8")
    merged_dir = OUT / "captions" / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "config" / "clip_manifest.json").read_text())
    n_clips = Counter(c["video_name"] for c in manifest["clips"])
    merge_index = []
    for (video, spatial, fps), parts in sorted(by_merge.items()):
        parts = [text for _, text in sorted(parts)]
        name = f"{video}_{spatial}_{fps_tag(fps)}.txt"
        text = "\n\n".join(parts)
        (merged_dir / name).write_text(text + "\n", encoding="utf-8")
        ready = sum(
            1
            for c in manifest["clips"]
            if c["video_name"] == video and (spatial != "S1" or c["s1_status"] == "ready")
        )
        merge_index.append(
            {
                "video_name": video,
                "spatial": spatial,
                "fps": fps,
                "n_clip_captions": len(parts),
                "n_clips_expected": n_clips[video] if spatial != "S1" else sum(
                    1 for c in manifest["clips"] if c["video_name"] == video and c["s1_status"] == "ready"
                ),
                "covers_every_clip": len(parts) == ready,
                "path": str(merged_dir / name),
            }
        )
    write_json(OUT / "captions" / "merged_index.json", merge_index)
    print(Counter(r["status"] for r in rows))


def cmd_check(args: argparse.Namespace) -> None:
    cfg = load_cfg()
    plan = json.loads((OUT / "config" / "plan.json").read_text())
    gt_index = json.loads((OUT / "config" / "gt_index.json").read_text())
    chosen = select_plans(plan["conditions"], cfg, args)
    hard = []
    notes = []
    rows = []
    groups: dict[tuple[str, str], dict[float, tuple]] = defaultdict(dict)
    for p in chosen:
        if p["status"] != "planned":
            notes.append(f"{p['condition_id']}: {p['status']}")
            continue
        rec = _load_record(p["condition_id"])
        if rec is None:
            hard.append(f"{p['condition_id']}: missing")
            continue
        flags = rec.get("caption_flags") or []
        used = [r for r in p["rois"] if r.get("grid_idx")]
        res = tuple(tuple(r["input_resolution"]) for r in used)
        groups[(p["clip_id"], p["spatial"])][float(p["fps"])] = res
        row = {
            "condition_id": p["condition_id"],
            "status": rec.get("status"),
            "tokens": rec.get("total_visual_tokens"),
            "budget_ok": (rec.get("total_visual_tokens") or 10**9) <= 40960,
            "resolution_lowered_for_budget": p.get("resolution_lowered_for_budget"),
            "eos": rec.get("eos"),
            "hit_limit": rec.get("hit_token_limit"),
            "flags": flags,
            "timestamp_error": rec.get("max_abs_error_sec"),
            "frame_counts": [r.get("frames") for r in used],
            "input_resolution": [r.get("input_resolution") for r in used],
        }
        rows.append(row)
        if rec.get("status") != "ok" or rec.get("oom"):
            hard.append(f"{p['condition_id']}: {rec.get('status')} {rec.get('error', '')[:180]}")
        if not row["budget_ok"]:
            hard.append(f"{p['condition_id']}: visual tokens over budget")
        if rec.get("total_visual_tokens") != p.get("planned_total_visual_tokens"):
            hard.append(f"{p['condition_id']}: measured tokens != plan")
        prompt = rec.get("prompt_text") or ""
        try:
            assert_no_gt_leak(prompt, gt_index)
        except RuntimeError as exc:
            hard.append(f"{p['condition_id']}: {exc}")
        if "generation_failure" in flags or "repetition_loop" in flags or rec.get("hit_token_limit"):
            notes.append(f"{p['condition_id']}: flags={flags}")
        if not rec.get("caption"):
            hard.append(f"{p['condition_id']}: empty caption")
    for key, by_fps in groups.items():
        vals = list(by_fps.values())
        if len(vals) >= 2 and any(v != vals[0] for v in vals[1:]):
            hard.append(f"{key}: resolution not locked across FPS {sorted(by_fps)}")
    payload = {"created_at": utc_now(), "n": len(rows), "hard": hard, "notes": notes, "rows": rows}
    write_json(OUT / "reports" / "pilot_check.json", payload)
    print(json.dumps({"n": len(rows), "hard": hard, "notes": notes}, indent=2, ensure_ascii=False))
    if hard:
        raise SystemExit(2)


def cmd_packets(_args: argparse.Namespace) -> None:
    plan = json.loads((OUT / "config" / "plan.json").read_text())
    gt_index = json.loads((OUT / "config" / "gt_index.json").read_text())
    by_video = gt_index
    packet_dir = OUT / "eval" / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for clip_id in sorted({p["clip_id"] for p in plan["conditions"]}):
        sample = next(p for p in plan["conditions"] if p["clip_id"] == clip_id)
        events = {ev["event_id"]: ev for ev in by_video[sample["video_name"]]["events"]}
        captions = []
        for p in plan["conditions"]:
            if p["clip_id"] != clip_id or p["status"] != "planned":
                continue
            rec = _load_record(p["condition_id"])
            if not rec or rec.get("status") != "ok":
                continue
            captions.append(
                {
                    "condition_id": p["condition_id"],
                    "text": rec["caption"],
                    "generation_flags": rec.get("caption_flags") or [],
                }
            )
            n += 1
        if not captions:
            continue
        packet = {
            "clip_id": clip_id,
            "video_name": sample["video_name"],
            "gt_status": "candidate",
            "human_verified": False,
            "instruction": (
                "Score only primary_references. Each primary event lies fully inside this clip. "
                "Judge the paragraph against each primary caption. Do not infer timestamps. "
                "unscored_context_captions are events that cross the clip boundary. "
                "Do not score them and do not call them unsupported if the paragraph mentions them. "
                "Do not mark a primary event missing merely because a boundary event is absent."
            ),
            "primary_references": [
                {
                    "event_id": eid,
                    "text": events[eid]["caption"],
                    "requires_human_review": events[eid]["requires_human_review"],
                }
                for eid in sample["primary_event_ids"]
            ],
            "unscored_context_captions": [
                {"event_id": eid, "text": events[eid]["caption"]}
                for eid in sample["boundary_event_ids"]
            ],
            "captions": captions,
        }
        blob = json.dumps(packet)
        for banned in ("start_sec", "end_sec", "window"):
            if banned in blob:
                raise RuntimeError(f"{clip_id} packet contains {banned}")
        (packet_dir / f"{clip_id}.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False))
    print("packets", n)


def rate(n: int, d: int):
    return None if not d else round(n / d, 4)


def cmd_score(_args: argparse.Namespace) -> None:
    judge_dir = OUT / "eval" / "judgments"
    files = sorted(judge_dir.glob("*.json"))
    if not files:
        raise RuntimeError("no judgments")
    plan = {p["condition_id"]: p for p in json.loads((OUT / "config" / "plan.json").read_text())["conditions"]}
    records = {}
    for p in plan.values():
        rec = _load_record(p["condition_id"])
        if rec:
            records[p["condition_id"]] = rec
    per_cond = []
    per_event = []
    for path in files:
        judged = json.loads(path.read_text())
        for cond in judged["conditions"]:
            cid = cond["condition_id"]
            p = plan[cid]
            base = records.get(cid, {})
            refs = cond["references"]
            labels = []
            for ref in refs:
                lab = ref["label"].lower()
                if lab not in LABELS:
                    raise RuntimeError(f"bad label {lab} in {cid}")
                labels.append(lab)
                per_event.append(
                    {
                        "condition_id": cid,
                        "clip_id": p["clip_id"],
                        "video_name": p["video_name"],
                        "spatial": p["spatial"],
                        "fps": p["fps"],
                        "duration_sec": round(p["duration_sec"], 4),
                        "event_id": ref["event_id"],
                        "scope": "fully_contained",
                        "label": lab,
                        "note": ref.get("note", ""),
                        "total_visual_tokens": base.get("total_visual_tokens"),
                        "resolution_lowered_for_budget": p.get("resolution_lowered_for_budget"),
                    }
                )
            counts = Counter(labels)
            n = len(labels)
            unsupported = cond.get("unsupported_events") or []
            flags = base.get("caption_flags") or []
            per_cond.append(
                {
                    "condition_id": cid,
                    "clip_id": p["clip_id"],
                    "video_name": p["video_name"],
                    "spatial": p["spatial"],
                    "fps": p["fps"],
                    "duration_sec": round(p["duration_sec"], 4),
                    "total_visual_tokens": base.get("total_visual_tokens"),
                    "inference_time_sec": base.get("inference_time_sec"),
                    "resolution_lowered_for_budget": p.get("resolution_lowered_for_budget"),
                    "input_resolution": "|".join(
                        f"{r['input_resolution'][0]}x{r['input_resolution'][1]}"
                        for r in p["rois"]
                        if r.get("grid_idx") and r.get("input_resolution")
                    ),
                    "frame_count": "|".join(str(r.get("frames")) for r in p["rois"] if r.get("grid_idx")),
                    "n_primary": n,
                    "correct": counts["correct"],
                    "partial": counts["partial"],
                    "contradictory": counts["contradictory"],
                    "missing": counts["missing"],
                    "strict_correct_rate": rate(counts["correct"], n),
                    "partial_inclusive_coverage": rate(counts["correct"] + counts["partial"], n),
                    "n_unsupported_events": len(unsupported),
                    "unsupported_events": " || ".join(unsupported),
                    "caption_flags": "|".join(flags),
                    "repetition": "repeated_sentence" in flags or "repetition_loop" in flags,
                    "generation_failure": "generation_failure" in flags or not base.get("eos", True),
                    "eos": base.get("eos"),
                    "hit_token_limit": base.get("hit_token_limit"),
                    "n_boundary_not_scored": len(p.get("boundary_event_ids") or []),
                    "gt_status": "candidate",
                    "human_verified": False,
                }
            )
    per_cond.sort(key=lambda r: (r["clip_id"], r["spatial"], r["fps"]))
    eval_dir = OUT / "eval"
    for name, data in (("eval_per_condition", per_cond), ("eval_per_event", per_event)):
        write_json(eval_dir / f"{name}.json", data)
        with (eval_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(len(per_cond), "conditions scored", len(per_event), "event labels")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    sub.add_parser("plan")
    inf = sub.add_parser("infer")
    sub.add_parser("aggregate")
    chk = sub.add_parser("check")
    sub.add_parser("packets")
    sub.add_parser("score")
    for s in (inf, chk):
        s.add_argument("--clips", default="")
        s.add_argument("--spatial", default="")
        s.add_argument("--fps", default="")
        s.add_argument("--pilot", action="store_true")
    inf.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    {
        "manifest": cmd_manifest,
        "plan": cmd_plan,
        "infer": cmd_infer,
        "aggregate": cmd_aggregate,
        "check": cmd_check,
        "packets": cmd_packets,
        "score": cmd_score,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
