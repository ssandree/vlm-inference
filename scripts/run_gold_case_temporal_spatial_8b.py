#!/usr/bin/env python3
"""Gold-case temporal x spatial captioning on Qwen3-VL-8B-Instruct.

5 clips x 4 FPS x 3 spatial inputs = 60 conditions.
S0 full-frame video, S1 one manual ROI video plus two global frames,
S2 motion-analyzer ROI videos plus the same two global frames.

Existing experiment directories are not written. --force is refused.

Commands: plan | visualize | infer | aggregate | check
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vlm_inference.frame_clock import encode_timestamps  # noqa: E402

OUT = REPO / "experiments" / "gold_case_temporal_spatial_8b"
FACTOR = 32
TEMPORAL = 2
EOS_IDS = [151645, 151643]
log = logging.getLogger("goldcase")

def load_common_prompt() -> str:
    """The caption instructions are config/common_prompt.txt. The script does not replace that file."""
    path = OUT / "config" / "common_prompt.txt"
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if not text.strip():
        raise RuntimeError(f"{path} is empty")
    return text


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load((OUT / "config" / "experiment.yaml").read_text())


def load_gt() -> dict[str, Any]:
    return json.loads((OUT / "config" / "gt_events.json").read_text())


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fps_tag(fps: float) -> str:
    return f"fps{fps:g}".replace(".", "p")


def cond_id(vid: str, spatial: str, fps: float) -> str:
    return f"{vid}_{spatial}_{fps_tag(fps)}"


def generation_record(cfg: dict[str, Any]) -> dict[str, Any]:
    g = cfg["generation"]
    if g.get("temperature") not in (None, "null"):
        raise RuntimeError("sampling is disabled; temperature must stay unset")
    rec = {
        "do_sample": False,
        "temperature": None,
        "repetition_penalty": float(g["repetition_penalty"]),
        "max_new_tokens": int(g["max_new_tokens"]),
        "seed": int(g["seed"]),
    }
    # 1.0 looped on the first pilot. This run uses the penalty set in experiment.yaml.
    if rec["do_sample"] or rec["repetition_penalty"] != 1.3 or rec["max_new_tokens"] != 1024:
        raise RuntimeError(f"generation settings drifted: {rec}")
    rec["sha"] = sha16(json.dumps({k: rec[k] for k in rec if k != "sha"}, sort_keys=True))
    return rec


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


def input_description(spatial: str, start: float, end: float, n_roi: int, image_times: list[float]) -> str:
    if spatial == "S0":
        return (
            f"The input is one full-frame video covering the original recording from {start:.2f}s to {end:.2f}s. "
            "Timestamps shown in the video are original recording times."
        )
    noun = "one ROI video" if n_roi == 1 else f"{n_roi} ROI videos"
    t0, t1 = image_times
    return (
        f"The input is two full-frame images, at {t0:.2f}s and {t1:.2f}s, followed by {noun}. "
        "Each ROI video is an independent spatial region of the same scene and covers only the time range stated in its label. "
        "Timestamps are original recording times. "
        "Use the full-frame images only for scene context. "
        "Do not merge events from different ROI videos unless the same actor clearly continues from one region into another."
    )


def full_prompt(spatial: str, start: float, end: float, n_roi: int, image_times: list[float]) -> str:
    return input_description(spatial, start, end, n_roi, image_times) + "\n\n" + load_common_prompt()


def clip_xywh(x: int, y: int, w: int, h: int, width: int, height: int) -> dict[str, Any]:
    requested = [int(x), int(y), int(w), int(h)]
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(width, x2), min(height, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        raise RuntimeError(f"ROI {requested} is outside the {width}x{height} frame")
    applied = [cx1, cy1, cx2 - cx1, cy2 - cy1]
    reasons = []
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        reasons.append(
            f"requested xywh {requested} ends at ({x2}, {y2}); "
            f"frame is {width}x{height}; applied xywh {applied}"
        )
    return {
        "requested_xywh": requested,
        "applied_xywh": applied,
        "applied_xyxy": [cx1, cy1, cx2, cy2],
        "clipped_to_frame": applied != requested,
        "clip_reason": reasons[0] if reasons else "",
    }


def motion_tubes(cfg: dict[str, Any], name: str, width: int, height: int) -> list[dict[str, Any]]:
    src = cfg["sources"]
    tracks = json.loads((Path(src["motion_roi_tracks"]) / f"{name}_roi_tracks.json").read_text())
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
                "tube_index_range": [int(tube["t0"]), int(tube["t1"])],
                "num_cells": tube.get("num_cells"),
                "cell_px": cell,
                "tracks_file": str(Path(src["motion_roi_tracks"]) / f"{name}_roi_tracks.json"),
            }
        )
    return rois


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
    uncapped = total(vmax)
    if uncapped > budget:
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
        h, w = smart_resize(it["native_h"], it["native_w"], vmin, max(cap, floor))
        out.append({"input_h": h, "input_w": w, "tokens": video_tokens(it["frames"], h, w)})
    binding = "budget" if uncapped > budget else "per_frame_max_pixels_or_native"
    return {"cap_pixels": cap, "tokens_at_vmax": uncapped, "binding": binding, "items": out}


def event_coverage(events: list[dict[str, Any]], start: float, end: float, tubes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ev in events:
        ev0, ev1 = float(ev["start"]), float(ev["end"])
        overlap0, overlap1 = max(ev0, start), min(ev1, end)
        inside = overlap1 > overlap0 + 1e-9
        tubes_hit = []
        for tube in tubes:
            t0, t1 = tube["time_range"]
            lo, hi = max(overlap0, t0), min(overlap1, t1)
            if hi > lo + 1e-9:
                tubes_hit.append(
                    {
                        "roi_id": tube["roi_id"],
                        "overlap_sec": [round(lo, 3), round(hi, 3)],
                        "bbox_xyxy": tube["bbox_xyxy"],
                    }
                )
        past = float(ev.get("end_past_clip_sec") or 0.0)
        rows.append(
            {
                "event_id": ev["event_id"],
                "event_time": [ev0, ev1],
                "clip_time_overlap": inside,
                "overlap_with_clip_sec": [round(overlap0, 3), round(overlap1, 3)] if inside else None,
                "end_extends_past_clip": ev1 > end + 1e-6,
                "end_past_clip_sec": past,
                "eval_tail_policy": (
                    "Do not score an ending action that would occur only after the clip end as an error."
                    if past > 0
                    else ""
                ),
                "motion_tubes_with_time_overlap": tubes_hit,
                "spatial_inclusion_in_roi": "not_labeled_no_event_box",
            }
        )
    return rows


def assert_no_gt_leak(prompt: str, gt: dict[str, Any]) -> None:
    for events in gt["videos"].values():
        for ev in events:
            if ev["text"] and ev["text"] in prompt:
                raise RuntimeError("GT event text leaked into the prompt")


def plan_all(cfg: dict[str, Any]) -> dict[str, Any]:
    gt = load_gt()
    listed = sum(len(v) for v in gt["videos"].values())
    if listed != 24 or gt["event_count"] != 24:
        raise RuntimeError(f"expected 24 GT events, found {listed}")
    budget = int(cfg["experiment"]["token_budget"])
    infos: dict[str, Any] = {}
    plans: list[dict[str, Any]] = []
    for video in cfg["videos"]:
        info = probe(resolve_video(cfg, video["name"]))
        if info["duration_sec"] <= float(video["end"]):
            raise RuntimeError(f"{video['id']} clip end {video['end']} exceeds duration {info['duration_sec']}")
        width, height = info["width"], info["height"]
        start, end = float(video["start"]), float(video["end"])
        manual = clip_xywh(*video["manual_roi_xywh"], width, height)
        manual["roi_mode"] = "single_manual_roi"
        manual["roi_count"] = 1
        tubes = motion_tubes(cfg, video["name"], width, height)
        gh, gw = smart_resize(height, width, int(cfg["image"]["min_pixels"]), int(cfg["image"]["max_pixels"]))
        image_times = [start, (start + end) / 2.0]
        events = event_coverage(gt["videos"][video["id"]], start, end, tubes)
        if any(not ev["clip_time_overlap"] for ev in events):
            raise RuntimeError(f"{video['id']} has a GT event with no overlap with the clip")
        infos[video["id"]] = {
            **info,
            "video_id": video["id"],
            "video_name": video["name"],
            "window": [start, end],
            "manual_roi": manual,
            "motion_tubes": tubes,
            "global_image_input_resolution": [gw, gh],
            "global_image_tokens_each": image_tokens(gh, gw),
            "global_image_times": image_times,
            "events": events,
        }
        for fps in cfg["experiment"]["fps"]:
            fps = float(fps)
            grid = grid_times(start, end, fps)
            for spatial in cfg["experiment"]["spatial"]:
                plan = build_condition(
                    cfg, video, info, manual, tubes, grid, spatial, fps, start, end, image_times, [gw, gh], budget, gt
                )
                plans.append(plan)
    if len(plans) != 60:
        raise RuntimeError(f"expected 60 conditions, built {len(plans)}")
    if any(not p["budget_ok"] for p in plans):
        bad = [p["condition_id"] for p in plans if not p["budget_ok"]]
        raise RuntimeError(f"visual token budget exceeded: {bad}")
    clipped = [v for v in infos.values() if v["manual_roi"]["clipped_to_frame"]]
    if [v["video_id"] for v in clipped] != ["A"]:
        raise RuntimeError(f"unexpected clipped manual ROIs: {[v['video_id'] for v in clipped]}")
    return {
        "created_at": utc_now(),
        "run_id": cfg["experiment"]["run_id"],
        "generation": generation_record(cfg),
        "common_prompt": load_common_prompt(),
        "common_prompt_sha": sha16(load_common_prompt()),
        "videos": infos,
        "conditions": plans,
    }


def build_condition(
    cfg: dict[str, Any],
    video: dict[str, Any],
    info: dict[str, Any],
    manual: dict[str, Any],
    tubes: list[dict[str, Any]],
    grid: list[float],
    spatial: str,
    fps: float,
    start: float,
    end: float,
    image_times: list[float],
    global_wh: list[int],
    budget: int,
    gt: dict[str, Any],
) -> dict[str, Any]:
    width, height = info["width"], info["height"]
    plan: dict[str, Any] = {
        "condition_id": cond_id(video["id"], spatial, fps),
        "run_id": cfg["experiment"]["run_id"],
        "video_id": video["id"],
        "video_name": video["name"],
        "spatial": spatial,
        "fps": fps,
        "window": [start, end],
        "native_resolution": [width, height],
        "native_fps": info["native_fps"],
        "grid_times": [round(t, 6) for t in grid],
        "grid_frame_count": len(grid),
        "token_budget": budget,
        "timestamp_mode": "source_timestamp",
        "status": "planned",
    }
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
            }
        ]
    else:
        raw = [dict(tube) for tube in tubes]
    rois = []
    for r in raw:
        t0, t1 = r["time_range"]
        idx = [k for k, t in enumerate(grid) if t0 - 1e-6 <= t <= t1 + 1e-6]
        item = dict(r, grid_idx=idx)
        if not idx:
            item["excluded"] = "no sampled frame inside the ROI lifetime and the clip"
        rois.append(item)
    globals_meta: list[dict[str, Any]] = []
    img_tok = 0
    if spatial == "S0":
        plan["global_images"] = []
    else:
        gw, gh = global_wh
        one = image_tokens(gh, gw)
        for role, target in (("clip_start", image_times[0]), ("clip_midpoint", image_times[1])):
            globals_meta.append(
                {
                    "role": role,
                    "time_target": target,
                    "native_resolution": [width, height],
                    "input_resolution": [gw, gh],
                    "tokens": one,
                }
            )
        img_tok = one * len(globals_meta)
        plan["global_images"] = globals_meta
    used = [r for r in rois if r["grid_idx"]]
    if not used:
        plan["status"] = "error_no_roi_frames"
        plan["rois"] = rois
        plan["budget_ok"] = False
        return plan
    items = [
        {
            "native_h": r["bbox_xyxy"][3] - r["bbox_xyxy"][1],
            "native_w": r["bbox_xyxy"][2] - r["bbox_xyxy"][0],
            "frames": len(r["grid_idx"]),
        }
        for r in used
    ]
    alloc = allocate(cfg, items, budget - img_tok)
    for r, it, a in zip(used, items, alloc["items"]):
        r["roi_native_size"] = [it["native_w"], it["native_h"]]
        r["frames"] = it["frames"]
        r["input_resolution"] = [a["input_w"], a["input_h"]]
        r["planned_video_tokens"] = a["tokens"]
        r["model_frames_with_temporal_padding"] = math.ceil(it["frames"] / TEMPORAL) * TEMPORAL
        r["temporal_padding_frames"] = r["model_frames_with_temporal_padding"] - it["frames"]
    plan["rois"] = rois
    plan["roi_count_used"] = len(used)
    plan["roi_mode"] = "single_manual_roi" if spatial == "S1" else ("full_frame" if spatial == "S0" else "motion_multi_roi")
    plan["allocation"] = {k: v for k, v in alloc.items() if k != "items"}
    plan["planned_image_tokens"] = img_tok
    plan["planned_video_tokens"] = sum(r["planned_video_tokens"] for r in used)
    plan["planned_total_visual_tokens"] = plan["planned_image_tokens"] + plan["planned_video_tokens"]
    plan["global_image_count"] = len(globals_meta)
    plan["budget_ok"] = plan["planned_total_visual_tokens"] <= budget
    times = image_times if globals_meta else [start, end]
    prompt = full_prompt(spatial, start, end, len(used), times)
    assert_no_gt_leak(prompt, gt)
    plan["prompt_text"] = prompt
    plan["prompt_sha"] = sha16(prompt)
    plan["common_prompt_sha"] = sha16(load_common_prompt())
    return plan


def cmd_plan(_args: argparse.Namespace) -> None:
    cfg = load_cfg()
    payload = plan_all(cfg)
    write_json(OUT / "config" / "plan.json", payload)
    log.info("common prompt from %s sha=%s", OUT / "config" / "common_prompt.txt", payload["common_prompt_sha"])
    rows = []
    for p in payload["conditions"]:
        used = [r for r in p["rois"] if r.get("grid_idx")]
        rows.append(
            {
                "condition_id": p["condition_id"],
                "video": p["video_id"],
                "spatial": p["spatial"],
                "fps": p["fps"],
                "status": p["status"],
                "clip": f"{p['window'][0]:g}-{p['window'][1]:g}",
                "grid_frames": p["grid_frame_count"],
                "roi_count": p["roi_count_used"],
                "roi_frames": "|".join(str(r["frames"]) for r in used),
                "roi_native": "|".join(f"{r['roi_native_size'][0]}x{r['roi_native_size'][1]}" for r in used),
                "input_resolution": "|".join(f"{r['input_resolution'][0]}x{r['input_resolution'][1]}" for r in used),
                "image_tokens": p["planned_image_tokens"],
                "video_tokens": p["planned_video_tokens"],
                "total_tokens": p["planned_total_visual_tokens"],
                "binding": p["allocation"]["binding"],
                "budget_ok": p["budget_ok"],
            }
        )
    path = OUT / "config" / "plan.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT / "config" / "video_roi_metadata.json", {k: _public_video(v) for k, v in payload["videos"].items()})
    print(f"planned {len(rows)} conditions -> {path}")
    for vid, info in payload["videos"].items():
        m = info["manual_roi"]
        print(
            f"{vid} frame {info['width']}x{info['height']} manual requested {m['requested_xywh']} "
            f"applied {m['applied_xywh']} clipped={m['clipped_to_frame']}"
        )


def _public_video(info: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "video_id", "video_name", "path", "width", "height", "native_fps", "duration_sec",
        "frame_count", "window", "manual_roi", "motion_tubes", "global_image_input_resolution",
        "global_image_tokens_each", "global_image_times", "events",
    )
    return {k: info[k] for k in keep}


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


def _draw(rgb: np.ndarray, boxes: list[dict[str, Any]]) -> Any:
    from PIL import Image, ImageDraw

    im = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(im)
    for box in boxes:
        if not box.get("active", True):
            continue
        x0, y0, x1, y1 = box["bbox_xyxy"]
        color = box["color"]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=4)
        draw.text((x0 + 6, max(0, y0 + 6)), box["label"], fill=color)
    return im


def _box_style(info: dict[str, Any], time_sec: float) -> list[dict[str, Any]]:
    """Draw the manual box and only motion tubes whose lifetime contains this time.

    Tubes outside their lifetime are listed in the sidecar, not drawn, so a frame
    does not look like those crops were fed to the model.
    """
    manual = info["manual_roi"]["applied_xyxy"]
    colors = [(0, 255, 255), (255, 64, 255), (255, 176, 0), (80, 255, 80)]
    boxes = [{"bbox_xyxy": manual, "label": "manual", "color": (255, 220, 0), "active": True}]
    for i, tube in enumerate(info["motion_tubes"]):
        t0, t1 = tube["time_range"]
        active = t0 - 1e-6 <= time_sec <= t1 + 1e-6
        boxes.append(
            {
                "bbox_xyxy": tube["bbox_xyxy"],
                "label": tube["roi_id"] if active else f"{tube['roi_id']} outside lifetime",
                "color": colors[i % len(colors)],
                "active": active,
                "time_range": tube["time_range"],
            }
        )
    return boxes


def cmd_visualize(_args: argparse.Namespace) -> None:
    plan = json.loads((OUT / "config" / "plan.json").read_text())
    viz = OUT / "roi_visualizations"
    viz.mkdir(parents=True, exist_ok=True)
    index = []
    for vid, info in plan["videos"].items():
        times = set(info["global_image_times"])
        for ev in info["events"]:
            if not ev["overlap_with_clip_sec"]:
                continue
            a, b = ev["overlap_with_clip_sec"]
            times.add(a)
            times.add(max(a, b - 0.05))
        # Representative sampled frames at the pilot rate.
        pilot = next(p for p in plan["conditions"] if p["video_id"] == vid and p["spatial"] == "S0" and p["fps"] == 2.0)
        grid = pilot["grid_times"]
        pick = np.linspace(0, len(grid) - 1, num=min(12, len(grid)), dtype=int)
        times.update(grid[int(i)] for i in pick)
        targets = sorted(times)
        decoded = decode_nearest(info["path"], targets)
        max_err = max(d["abs_error_sec"] for d in decoded)
        if max_err > 1.5 / info["native_fps"]:
            raise RuntimeError(f"{vid} visualization timestamp error {max_err:.4f}s")
        for dec in decoded:
            boxes = _box_style(info, dec["target_time_sec"])
            stamp = f"{dec['target_time_sec']:.2f}".replace(".", "p")
            name = f"{vid}_t{stamp}.jpg"
            _draw(dec["array"], boxes).save(viz / name, quality=90)
            index.append(
                {
                    "file": name,
                    "video_id": vid,
                    "target_time_sec": dec["target_time_sec"],
                    "source_time_sec": dec["source_time_sec"],
                    "abs_error_sec": dec["abs_error_sec"],
                    "boxes_active": [
                        {"label": b["label"], "bbox_xyxy": b["bbox_xyxy"]}
                        for b in boxes
                        if b.get("active", True)
                    ],
                    "boxes_outside_lifetime": [
                        {"label": b["label"], "bbox_xyxy": b["bbox_xyxy"], "time_range": b.get("time_range")}
                        for b in boxes
                        if not b.get("active", True)
                    ],
                }
            )
        log.info("%s saved %d overlays", vid, len(decoded))
    write_json(viz / "index.json", {"created_at": utc_now(), "frames": index})
    print(f"visualizations {len(index)} -> {viz}")


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
    gc_ = model.generation_config
    eos = gc_.eos_token_id
    eos_list = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    if eos_list != EOS_IDS or gc_.pad_token_id != EOS_IDS[1]:
        log.warning("Fixing generation stop tokens (was eos=%s pad=%s)", eos, gc_.pad_token_id)
        gc_.eos_token_id = list(EOS_IDS)
        gc_.pad_token_id = EOS_IDS[1]
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


def prepare_inputs(cfg: dict[str, Any], processor, plan: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    t_start = time.perf_counter()
    gt = load_gt()
    start, end = plan["window"]
    grid = grid_times(start, end, plan["fps"])
    if [round(t, 6) for t in grid] != plan["grid_times"]:
        raise RuntimeError("grid drifted from the plan")
    global_targets = [g["time_target"] for g in plan["global_images"]]
    sel = decode_nearest(info["path"], grid)
    gsel = decode_nearest(info["path"], global_targets) if global_targets else []
    src_times = [s["source_time_sec"] for s in sel]
    if len({round(t, 6) for t in src_times}) != len(src_times):
        raise RuntimeError("duplicate source frames in the sampling grid")
    max_err = max(s["abs_error_sec"] for s in sel + gsel)
    if max_err > 1.5 / info["native_fps"]:
        raise RuntimeError(f"timestamp error {max_err:.4f}s")
    if len(gsel) == 2 and abs(gsel[0]["source_time_sec"] - gsel[1]["source_time_sec"]) < 0.5:
        raise RuntimeError("the two global frames resolved to the same time")

    content: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []
    metadatas: list[dict[str, Any]] = []
    used = [r for r in plan["rois"] if r["grid_idx"]]
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
                f"[ROI {i} video: crop x={x0}-{x1}, y={y0}-{y1} of the {info['width']}x{info['height']} frame, "
                f"{stamps[0]:.2f}s-{stamps[-1]:.2f}s]"
            )
        content.append({"type": "text", "text": label})
        content.append({"type": "video", "video": f"memory://roi{i}"})

    image_times = [g["source_time_sec"] for g in plan["global_images"]] or [start, end]
    prompt = full_prompt(plan["spatial"], start, end, len(used), image_times)
    if sha16(load_common_prompt()) != plan["common_prompt_sha"]:
        raise RuntimeError("config/common_prompt.txt changed after planning")
    assert_no_gt_leak(prompt, gt)
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
    assert_no_gt_leak(decoded, gt)
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
    return flags


def generate_one(cfg: dict[str, Any], model, processor, prepared: dict[str, Any]) -> dict[str, Any]:
    import torch

    gen = generation_record(cfg)
    torch.manual_seed(int(gen["seed"]))
    device = next(model.parameters()).device
    inputs = prepared["inputs"].to(device)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    oom = False
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=False,
                repetition_penalty=float(gen["repetition_penalty"]),
                max_new_tokens=int(gen["max_new_tokens"]),
                use_cache=True,
                eos_token_id=list(EOS_IDS),
                pad_token_id=EOS_IDS[1],
            )
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
    hit = n_out >= int(gen["max_new_tokens"]) and last not in EOS_IDS
    flags = caption_flags(text, hit)
    del inputs, out, new
    return {
        "caption": text,
        "output_token_count": n_out,
        "hit_token_limit": hit,
        "terminated_normally": (not hit) and (last in EOS_IDS if last is not None else False),
        "eos": last in EOS_IDS if last is not None else False,
        "caption_flags": flags,
        "inference_time_sec": round(elapsed, 3),
        "peak_gpu_memory_mib": max(peaks.values()) if peaks else None,
        "peak_gpu_memory_mib_per_device": peaks,
        "oom": False,
    }


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def select_plans(plans: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    videos = set(args.videos.split(",")) if args.videos else None
    spatials = set(args.spatial.split(",")) if args.spatial else None
    fps = {float(x) for x in args.fps.split(",")} if args.fps else None
    chosen = []
    for p in plans:
        if p["status"] != "planned":
            continue
        if videos and p["video_id"] not in videos:
            continue
        if spatials and p["spatial"] not in spatials:
            continue
        if fps and float(p["fps"]) not in fps:
            continue
        if not p.get("budget_ok"):
            raise RuntimeError(f"{p['condition_id']} exceeds the visual-token budget")
        chosen.append(p)
    return chosen


def cmd_infer(args: argparse.Namespace) -> None:
    import torch

    if args.force:
        raise SystemExit(
            "Refusing --force. This experiment does not overwrite an existing run. "
            "A finished checkpoint is reused; an error checkpoint is replaced only with --retry-failed."
        )
    cfg = load_cfg()
    gen = generation_record(cfg)
    plan_all_payload = json.loads((OUT / "config" / "plan.json").read_text())
    if plan_all_payload["run_id"] != cfg["experiment"]["run_id"]:
        raise RuntimeError("plan.json run_id does not match experiment.yaml; refusing to mix runs")
    if plan_all_payload["common_prompt_sha"] != sha16(load_common_prompt()):
        raise RuntimeError("config/common_prompt.txt changed after planning")
    infos = plan_all_payload["videos"]
    ckpt_dir = OUT / "results" / "checkpoints"
    cap_dir = OUT / "captions"
    inp_dir = OUT / "inputs"
    for d in (ckpt_dir, cap_dir, inp_dir, OUT / "reports" / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    wanted = select_plans(plan_all_payload["conditions"], args)

    def reusable(p: dict[str, Any]) -> bool:
        f = ckpt_dir / f"{p['condition_id']}.json"
        if not f.is_file():
            return False
        rec = json.loads(f.read_text())
        same = rec.get("run_id") == cfg["experiment"]["run_id"] and rec.get("generation_sha") == gen["sha"]
        if rec.get("status") == "ok" and same:
            return True
        if rec.get("status") not in (None, "ok") and same and not args.retry_failed:
            return True
        return False

    todo = [p for p in wanted if not reusable(p)]
    log.info("%d selected, %d to run", len(wanted), len(todo))
    if not todo:
        return
    processor, model = load_model(cfg)
    base = {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 2**20, 1) for i in range(torch.cuda.device_count())}
    log.info("model loaded; resident MiB %s", base)
    for p in todo:
        cid = p["condition_id"]
        rec: dict[str, Any] = {
            "condition_id": cid,
            "run_id": cfg["experiment"]["run_id"],
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
            "oom": False,
        }
        prepared = None
        try:
            prepared = prepare_inputs(cfg, processor, p, infos[p["video_id"]])
            produced = generate_one(cfg, model, processor, prepared)
            rec.update(status="ok", **produced)
            for key in (
                "image_tokens", "video_tokens", "total_visual_tokens", "prompt_tokens", "text_tokens",
                "prep_sec", "max_abs_error_sec", "timestamp_markers_verified", "roi_timestamps",
                "prompt_text", "global_sampling",
            ):
                rec[key] = prepared[key]
            rec["plan"] = prepared["plan"]
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
                    "rois": [
                        {
                            "roi_id": r["roi_id"],
                            "bbox_xyxy": r["bbox_xyxy"],
                            "bbox_xywh_requested": r.get("bbox_xywh_requested"),
                            "bbox_xywh": r.get("bbox_xywh"),
                            "roi_native_size": r.get("roi_native_size"),
                            "input_resolution": r.get("input_resolution"),
                            "frames": r.get("frames"),
                            "source": r.get("source"),
                            "time_range": r.get("time_range"),
                            "source_time_first_last": r.get("source_time_first_last"),
                            "measured_video_tokens": r.get("measured_video_tokens"),
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


def cmd_aggregate(_args: argparse.Namespace) -> None:
    plan = json.loads((OUT / "config" / "plan.json").read_text())
    rows = []
    blocks = ["# Gold-case captions\n", "Captions are model output. GT text was not in the prompt.\n"]
    for p in plan["conditions"]:
        rec = _load_record(p["condition_id"]) or {"status": "not_run"}
        pl = rec.get("plan", p)
        used = [r for r in pl.get("rois", []) if r.get("grid_idx")]
        rows.append(
            {
                "condition_id": p["condition_id"],
                "video_id": p["video_id"],
                "video_name": p["video_name"],
                "spatial": p["spatial"],
                "fps": p["fps"],
                "status": rec.get("status"),
                "oom": rec.get("oom", False),
                "error": rec.get("error", ""),
                "window": f"{p['window'][0]:g}-{p['window'][1]:g}",
                "native_resolution": "x".join(map(str, p["native_resolution"])),
                "roi_count": len(used),
                "roi_native": "|".join(f"{r['roi_native_size'][0]}x{r['roi_native_size'][1]}" for r in used),
                "input_resolution": "|".join(
                    f"{r['input_resolution'][0]}x{r['input_resolution'][1]}" for r in used if r.get("input_resolution")
                ),
                "frame_count": "|".join(str(r.get("frames")) for r in used),
                "image_tokens": rec.get("image_tokens", pl.get("planned_image_tokens")),
                "video_tokens": rec.get("video_tokens", pl.get("planned_video_tokens")),
                "total_visual_tokens": rec.get("total_visual_tokens", pl.get("planned_total_visual_tokens")),
                "text_tokens": rec.get("text_tokens"),
                "binding_constraint": (pl.get("allocation") or {}).get("binding", ""),
                "peak_gpu_memory_mib": rec.get("peak_gpu_memory_mib"),
                "inference_time_sec": rec.get("inference_time_sec"),
                "output_tokens": rec.get("output_token_count"),
                "terminated_normally": rec.get("terminated_normally"),
                "eos": rec.get("eos"),
                "hit_token_limit": rec.get("hit_token_limit"),
                "caption_flags": "|".join(rec.get("caption_flags") or []),
                "max_abs_timestamp_error_sec": rec.get("max_abs_error_sec"),
                "caption": rec.get("caption", ""),
            }
        )
        if rec.get("caption"):
            blocks.append(
                f"## {p['condition_id']}\n\n"
                f"tokens={rec.get('total_visual_tokens')} flags={rec.get('caption_flags')} "
                f"eos={rec.get('eos')} peak_mib={rec.get('peak_gpu_memory_mib')}\n\n"
                f"{rec['caption']}\n"
            )
    result_dir = OUT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "results.json", rows)
    with (result_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "captions" / "captions.md").write_text("\n".join(blocks), encoding="utf-8")
    print(Counter(r["status"] for r in rows))


def cmd_check(args: argparse.Namespace) -> None:
    plan = json.loads((OUT / "config" / "plan.json").read_text())
    chosen = select_plans(plan["conditions"], args)
    problems = []
    rows = []
    for p in chosen:
        rec = _load_record(p["condition_id"])
        if rec is None:
            problems.append(f"{p['condition_id']}: missing")
            continue
        flags = rec.get("caption_flags") or []
        row = {
            "condition_id": p["condition_id"],
            "status": rec.get("status"),
            "oom": rec.get("oom"),
            "tokens": rec.get("total_visual_tokens"),
            "budget_ok": (rec.get("total_visual_tokens") or 10**9) <= 40960,
            "eos": rec.get("eos"),
            "hit_1024": rec.get("hit_token_limit"),
            "flags": flags,
            "timestamp_error": rec.get("max_abs_error_sec"),
        }
        rows.append(row)
        if rec.get("status") != "ok" or rec.get("oom"):
            problems.append(f"{p['condition_id']}: {rec.get('status')} {rec.get('error', '')[:180]}")
        if "repetition_loop" in flags:
            problems.append(f"{p['condition_id']}: repetition_loop")
        if not row["budget_ok"]:
            problems.append(f"{p['condition_id']}: visual tokens over budget")
    write_json(OUT / "reports" / "pilot_check.json", {"created_at": utc_now(), "rows": rows, "problems": problems})
    print(json.dumps({"n": len(rows), "problems": problems}, indent=2, ensure_ascii=False))
    if problems:
        raise SystemExit(2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("visualize")
    inf = sub.add_parser("infer")
    sub.add_parser("aggregate")
    chk = sub.add_parser("check")
    for s in (inf, chk):
        s.add_argument("--videos", default="")
        s.add_argument("--spatial", default="")
        s.add_argument("--fps", default="")
    inf.add_argument("--force", action="store_true")
    inf.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    {"plan": cmd_plan, "visualize": cmd_visualize, "infer": cmd_infer, "aggregate": cmd_aggregate, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
