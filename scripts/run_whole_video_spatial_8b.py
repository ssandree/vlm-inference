#!/usr/bin/env python3
"""Whole-video spatial captioning for eight VIRAT clips on Qwen3-VL-8B-Instruct.

S0 full-frame video, S1 motion tubes + two global frames, S2 manual lifetime
boxes + the same two global frames. The whole file is the input window.
Candidate GT captions are never placed in the prompt.

Commands: plan | references | infer | aggregate
"""

from __future__ import annotations

import argparse
import bisect
import csv
import difflib
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vlm_inference.frame_clock import encode_timestamps  # noqa: E402

OUT = REPO / "experiments" / "whole_video_spatial_8b"
CFG = yaml.safe_load((OUT / "configs" / "experiment.yaml").read_text())
FACTOR = 32
TEMPORAL = 2
EOS_IDS = [151645, 151643]
log = logging.getLogger("whole30b")

COMMON_PROMPT = """Describe all distinct visible events in this surveillance video, focusing on people, vehicles, and bicycles.

- Describe each actor's actions independently, including movements, interactions, and changes in behavior.
- Include small, distant, and brief actions whenever they are visually identifiable.
- For each event, specify who did what, to which object, and in which direction when clearly visible.
- Do not merge different actors or events into vague summaries.
- Do not describe static backgrounds, general scenery, or atmosphere.
- Do not invent actions or details that are not visually supported.

Write one chronological paragraph with one sentence per independent event. Include as many distinct observable events as possible without repetition. Do not use timestamps, numbering, or lists."""


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
    h.update(path.read_bytes())
    return h.hexdigest()


def generation_record() -> dict[str, Any]:
    g = CFG["generation"]
    rec = {
        "do_sample": bool(g["do_sample"]),
        "repetition_penalty": float(g["repetition_penalty"]),
        "max_new_tokens": int(g["max_new_tokens"]),
        "seed": int(g["seed"]),
    }
    rec["sha"] = sha16(json.dumps(rec, sort_keys=True))
    return rec


def resolve_video(name: str) -> Path:
    hits = []
    for root in CFG["sources"]["video_roots"]:
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
            "width": int(st.codec_context.width),
            "height": int(st.codec_context.height),
            "native_fps": float(st.average_rate),
            "duration_sec": float(c.duration / 1e6),
            "frame_count": int(st.frames),
        }


def grid_times(start: float, end: float, fps: float) -> list[float]:
    n = int(math.ceil((end - start) * fps - 1e-9))
    return [start + k / fps for k in range(n)]


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
    return input_description(spatial, start, end, n_roi, image_times) + "\n\n" + COMMON_PROMPT


def motion_rois(vid_name: str, width: int, height: int) -> list[dict[str, Any]]:
    """Stage3 tubes. Lifetime box from spatial_bbox_grid; times from the Stage2 timeline."""
    src = CFG["sources"]
    tracks = json.loads((Path(src["motion_roi_tracks"]) / f"{vid_name}_roi_tracks.json").read_text())
    z = np.load(Path(src["motion_fusion_npz"]) / vid_name / f"{vid_name}_gap_fusion_mean.npz")
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
            raise RuntimeError(f"{vid_name} tube{tube['tube_id']} has an empty box")
        rois.append(
            {
                "roi_id": f"tube{int(tube['tube_id'])}",
                "bbox_xyxy": [x0, y0, x1, y1],
                "time_range": [float(ts[t0]), float(ts[t1])],
                "source": "motion_stage3_tube",
                "tube_index_range": [int(tube["t0"]), int(tube["t1"])],
            }
        )
    return rois


def manual_table() -> dict[str, dict[str, Any]]:
    man = json.loads((REPO / CFG["sources"]["manual_lifetime_bbox_manifest"]).read_text())
    return {v["video_filename"]: v for v in man["videos"]}


def manual_rois(video_name: str, start: float, end: float) -> list[dict[str, Any]] | None:
    entry = manual_table().get(video_name)
    if entry is None:
        return None
    rois = []
    for i, (x, y, w, h) in enumerate(entry["bboxes"]):
        x, y, w, h = int(x), int(y), int(w), int(h)
        rois.append(
            {
                "roi_id": f"manual{i + 1}",
                "bbox_xywh": [x, y, w, h],
                "bbox_xyxy": [x, y, x + w, y + h],
                "time_range": [float(start), float(end)],
                "source": "gold_lifetime_bbox",
                "manifest_video_id": entry["video_id"],
            }
        )
    return rois


def allocate(items: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    vmin, vmax = int(CFG["video"]["min_pixels"]), int(CFG["video"]["max_pixels"])
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


def plan_condition(video: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    start, end = 0.0, float(info["duration_sec"])
    fps = float(CFG["experiment"]["fps"])
    budget = int(CFG["experiment"]["token_budget"])
    grid = grid_times(start, end, fps)
    W, H = info["width"], info["height"]
    image_times = [0.0, end / 2.0]
    plans = []
    for spatial in CFG["experiment"]["spatial"]:
        plan: dict[str, Any] = {
            "condition_id": f"{video['id']}_{spatial}",
            "video_id": video["id"],
            "video_name": video["name"],
            "spatial": spatial,
            "fps": fps,
            "window": [start, end],
            "full_duration_sec": end,
            "native_resolution": [W, H],
            "native_fps": info["native_fps"],
            "grid_frame_count": len(grid),
            "token_budget": budget,
            "status": "planned",
        }
        if spatial == "S2":
            raw = manual_rois(video["name"], start, end)
            if raw is None:
                plan["status"] = "held_missing_manual_roi"
                plan["hold_reason"] = (
                    "No lifetime box in experiments/gold_lifetime_crop/configs/manifest.json. "
                    "Coordinates were not invented."
                )
                plan["rois"] = []
                plan["global_images"] = []
                plan["roi_count_used"] = 0
                plan["planned_total_visual_tokens"] = None
                plan["budget_ok"] = None
                plan["prompt_text"] = None
                plans.append(plan)
                continue
        elif spatial == "S1":
            raw = motion_rois(video["name"], W, H)
        else:
            raw = [
                {
                    "roi_id": "full",
                    "bbox_xyxy": [0, 0, W, H],
                    "time_range": [start, end],
                    "source": "full_frame",
                }
            ]
        rois = []
        for r in raw:
            t0, t1 = r["time_range"]
            idx = [k for k, t in enumerate(grid) if t0 - 1e-6 <= t <= t1 + 1e-6]
            r = dict(r, grid_idx=idx)
            if not idx:
                r["excluded"] = "no sampled frame inside the ROI lifetime"
            rois.append(r)
        globals_meta = []
        img_tok = 0
        if spatial == "S0":
            plan["global_images"] = []
        else:
            gh, gw = smart_resize(H, W, int(CFG["image"]["min_pixels"]), int(CFG["image"]["max_pixels"]))
            one = image_tokens(gh, gw)
            for role, target in (("first_frame", image_times[0]), ("midpoint", image_times[1])):
                globals_meta.append(
                    {
                        "role": role,
                        "time_target": target,
                        "native_resolution": [W, H],
                        "input_resolution": [gw, gh],
                        "tokens": one,
                    }
                )
            img_tok = one * len(globals_meta)
            plan["global_images"] = globals_meta
        used = [r for r in rois if r["grid_idx"]]
        items = [
            {
                "native_h": r["bbox_xyxy"][3] - r["bbox_xyxy"][1],
                "native_w": r["bbox_xyxy"][2] - r["bbox_xyxy"][0],
                "frames": len(r["grid_idx"]),
            }
            for r in used
        ]
        if not items:
            plan["status"] = "error_no_roi_frames"
            plan["rois"] = rois
            plans.append(plan)
            continue
        alloc = allocate(items, budget - img_tok)
        for r, it, a in zip(used, items, alloc["items"]):
            r["roi_native_size"] = [it["native_w"], it["native_h"]]
            r["frames"] = it["frames"]
            r["input_resolution"] = [a["input_w"], a["input_h"]]
            r["planned_video_tokens"] = a["tokens"]
            r["model_frames_with_temporal_padding"] = math.ceil(it["frames"] / TEMPORAL) * TEMPORAL
            r["temporal_padding_frame"] = r["model_frames_with_temporal_padding"] - it["frames"]
            r["temporal_coverage_sec"] = [round(r["time_range"][0], 3), round(r["time_range"][1], 3)]
        plan["rois"] = rois
        plan["roi_count_used"] = len(used)
        plan["allocation"] = {k: v for k, v in alloc.items() if k != "items"}
        plan["planned_image_tokens"] = img_tok
        plan["planned_video_tokens"] = sum(r["planned_video_tokens"] for r in used)
        plan["planned_total_visual_tokens"] = plan["planned_image_tokens"] + plan["planned_video_tokens"]
        plan["global_image_count"] = len(globals_meta)
        plan["budget_ok"] = plan["planned_total_visual_tokens"] <= budget and img_tok <= budget
        prompt = full_prompt(spatial, start, end, len(used), image_times)
        plan["prompt_text"] = prompt
        plan["prompt_sha"] = sha16(prompt)
        plan["common_prompt_sha"] = sha16(COMMON_PROMPT)
        plans.append(plan)
    return plans


def cmd_plan(_args) -> None:
    infos = {}
    plans = []
    for v in CFG["videos"]:
        info = probe(resolve_video(v["name"]))
        infos[v["id"]] = info
        plans.extend(plan_condition(v, info))
    leak = []
    for p in plans:
        if not p.get("prompt_text"):
            continue
        gt = json.loads((REPO / "collected_candidate_gt" / f"{p['video_name']}_candidate_gt.json").read_text())
        for ev in gt["level_2"]["events"]:
            if ev["caption"] in p["prompt_text"]:
                leak.append(p["condition_id"])
    if leak:
        raise RuntimeError(f"GT caption leaked into prompts: {leak}")
    write_json(
        OUT / "configs" / "plan.json",
        {"created_at": utc_now(), "generation": generation_record(), "videos": infos, "conditions": plans},
    )
    rows = []
    for p in plans:
        used = [r for r in p["rois"] if r.get("grid_idx")]
        rows.append(
            {
                "condition_id": p["condition_id"],
                "video": p["video_name"],
                "spatial": p["spatial"],
                "status": p["status"],
                "duration_sec": round(p["full_duration_sec"], 3),
                "grid_frames": p["grid_frame_count"],
                "rois": p.get("roi_count_used", 0),
                "global_images": p.get("global_image_count", 0),
                "image_tokens": p.get("planned_image_tokens", ""),
                "video_tokens": p.get("planned_video_tokens", ""),
                "total_tokens": p.get("planned_total_visual_tokens", ""),
                "budget_ok": p.get("budget_ok", ""),
                "binding": (p.get("allocation") or {}).get("binding", ""),
                "roi_inputs": " | ".join(
                    f"{r['roi_id']}:{r['frames']}f@{r['input_resolution'][0]}x{r['input_resolution'][1]}"
                    f"[{r['time_range'][0]:.1f}-{r['time_range'][1]:.1f}s]"
                    for r in used
                ),
            }
        )
    with open(OUT / "configs" / "plan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    held = [r["video"] for r in rows if r["status"] == "held_missing_manual_roi"]
    bad = [r["condition_id"] for r in rows if r["budget_ok"] is False]
    print(f"{len(rows)} rows; held S2 videos: {sorted(set(held))}; budget violations: {bad}")


def norm_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def cmd_references(_args) -> None:
    """Eval captions are text only. Times stay in the audit block."""
    videos = []
    for v in CFG["videos"]:
        path = REPO / "collected_candidate_gt" / f"{v['name']}_candidate_gt.json"
        d = json.loads(path.read_text())
        if d.get("condition") != "D":
            raise RuntimeError(f"{v['name']} GT condition is {d.get('condition')}, expected D")
        events = d["level_2"]["events"]
        review = set(d.get("review_required_event_ids") or [])
        groups: dict[str, list[dict[str, Any]]] = {}
        audit = []
        for i, ev in enumerate(events):
            key = norm_caption(ev["caption"])
            rec = {
                "event_id": ev["event_id"],
                "json_pointer": f"level_2.events[{i}]",
                "start_sec": ev.get("start_sec"),
                "end_sec": ev.get("end_sec"),
                "caption": ev["caption"],
            }
            audit.append(rec)
            groups.setdefault(key, []).append(rec)
        eval_captions = []
        excluded = []
        for key, recs in groups.items():
            keeper = recs[0]
            item = {
                "ref_id": keeper["event_id"],
                "caption": keeper["caption"],
                "source_event_ids": [r["event_id"] for r in recs],
                "source_file": str(path.relative_to(REPO)),
                "requires_human_review": any(r["event_id"] in review for r in recs),
                "review_reason": "listed in review_required_event_ids" if any(r["event_id"] in review for r in recs) else "",
            }
            eval_captions.append(item)
            for extra in recs[1:]:
                excluded.append(
                    {
                        "event_id": extra["event_id"],
                        "kept_as": keeper["event_id"],
                        "reason": "exact caption duplicate after whitespace/case normalization",
                        "caption": extra["caption"],
                    }
                )
        near = []
        for a in range(len(eval_captions)):
            for b in range(a + 1, len(eval_captions)):
                ratio = difflib.SequenceMatcher(
                    None, norm_caption(eval_captions[a]["caption"]), norm_caption(eval_captions[b]["caption"])
                ).ratio()
                if ratio >= 0.85:
                    near.append({"a": eval_captions[a]["ref_id"], "b": eval_captions[b]["ref_id"], "ratio": round(ratio, 3)})
        videos.append(
            {
                "video_id": v["id"],
                "video_name": v["name"],
                "source_file": str(path.relative_to(REPO)),
                "source_sha256": sha256_file(path),
                "gt_condition": d["condition"],
                "gt_status": d["gt_status"],
                "human_verified": d["human_verified"],
                "original_event_count": len(events),
                "eval_caption_count": len(eval_captions),
                "eval_captions": eval_captions,
                "excluded_exact_duplicates": excluded,
                "near_duplicate_pairs_not_excluded": near,
                "audit_timeline_not_for_judge": audit,
            }
        )
    payload = {
        "created_at": utc_now(),
        "gt_status": "candidate",
        "human_verified": False,
        "score_meaning": "Provisional scores against candidate captions, not human-verified GT.",
        "judge_input_excludes": ["start_sec", "end_sec", "start_time", "end_time"],
        "construction": (
            "Caption text is copied from collected_candidate_gt level_2.events. "
            "condition D files only. Exact normalized duplicates are dropped from the eval list "
            "and retained in excluded_exact_duplicates. Original files are not modified."
        ),
        "videos": videos,
    }
    write_json(OUT / "gt_references.json", payload)
    print(
        "references",
        [(v["video_id"], v["original_event_count"], v["eval_caption_count"], len(v["excluded_exact_duplicates"]), len(v["near_duplicate_pairs_not_excluded"])) for v in videos],
    )


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
        raise RuntimeError(f"decoded {len(best)} of {len(targets)} target frames")
    return [best[k] for k in range(len(targets))]


def decode_mapped(path: str, targets: list[float]) -> list[dict[str, Any]]:
    order = sorted(range(len(targets)), key=lambda i: (targets[i], i))
    sorted_targets = [targets[i] for i in order]
    # Equal targets must stay distinct indices. decode_nearest keys by position.
    got = decode_nearest(path, sorted_targets)
    out: list[dict[str, Any] | None] = [None] * len(targets)
    for pos, src in enumerate(order):
        out[src] = got[pos]
    return out  # type: ignore[return-value]


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        import torch.backends.python_native as pn

        pn.triton.enabled = False
    except Exception:
        pass
    path = CFG["model"]["path"]
    if not (Path(path) / "config.json").is_file():
        raise FileNotFoundError(f"8B weights missing at {path}")
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        dtype=getattr(torch, CFG["model"]["dtype"]),
        device_map=CFG["model"]["device_map"],
        attn_implementation=CFG["model"]["attn_implementation"],
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


def prepare_inputs(processor, plan: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    t_start = time.perf_counter()
    start, end = plan["window"]
    grid = grid_times(start, end, plan["fps"])
    global_targets = [g["time_target"] for g in plan["global_images"]]
    # Separate decodes so a global target that equals a grid time is not a duplicate index.
    sel = decode_nearest(info["path"], grid)
    gsel = decode_mapped(info["path"], global_targets) if global_targets else []
    src_times = [s["source_time_sec"] for s in sel]
    if len({round(t, 6) for t in src_times}) != len(src_times):
        raise RuntimeError("duplicate source frames in the 2 fps grid")
    max_err = max(s["abs_error_sec"] for s in sel + gsel)
    if max_err > 1.5 / info["native_fps"]:
        raise RuntimeError(f"timestamp error {max_err:.4f}s too large")
    if len(gsel) == 2 and abs(gsel[0]["source_time_sec"] - gsel[1]["source_time_sec"]) < 0.5:
        raise RuntimeError("first and midpoint global frames resolved to the same time")

    content: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []
    metadatas: list[dict[str, Any]] = []
    spatial = plan["spatial"]
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
        if spatial == "S0":
            label = f"[Full-frame video, {stamps[0]:.2f}s-{stamps[-1]:.2f}s]"
        else:
            label = (
                f"[ROI {i} video: crop x={x0}-{x1}, y={y0}-{y1} of the {info['width']}x{info['height']} frame, "
                f"{stamps[0]:.2f}s-{stamps[-1]:.2f}s]"
            )
        content.append({"type": "text", "text": label})
        content.append({"type": "video", "video": f"memory://roi{i}"})

    image_times = [g["source_time_sec"] for g in plan["global_images"]] or [start, end]
    prompt = full_prompt(spatial, start, end, len(used), image_times if plan["global_images"] else [start, end])
    if spatial == "S0":
        prompt = full_prompt(spatial, start, end, len(used), [start, end])
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
    for r, grid in zip(used, grids):
        w, h = r["input_resolution"]
        expect = [math.ceil(r["frames"] / TEMPORAL), h // 16, w // 16]
        if grid != expect:
            raise RuntimeError(f"{r['roi_id']}: grid {grid} != expected {expect}")
        r["video_grid_thw"] = grid
        r["measured_video_tokens"] = grid[0] * grid[1] * grid[2] // 4
    if vid_tok != plan["planned_video_tokens"] or img_tok != plan["planned_image_tokens"]:
        raise RuntimeError(
            f"measured tokens {img_tok}+{vid_tok} differ from plan {plan['planned_image_tokens']}+{plan['planned_video_tokens']}"
        )
    if img_tok + vid_tok > int(CFG["experiment"]["token_budget"]):
        raise RuntimeError(f"visual tokens {img_tok + vid_tok} exceed the request budget")

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
    gt = json.loads((REPO / "collected_candidate_gt" / f"{plan['video_name']}_candidate_gt.json").read_text())
    for ev in gt["level_2"]["events"]:
        if ev["caption"] in prompt or ev["caption"] in decoded:
            raise RuntimeError("GT caption leaked into the model input")
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
    counts = Counter(norm_caption(p) for p in parts)
    if any(v >= 2 for v in counts.values()):
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


def generate_one(model, processor, prepared: dict[str, Any]) -> dict[str, Any]:
    import torch

    gen = generation_record()
    torch.manual_seed(int(gen["seed"]))
    device = next(model.parameters()).device
    inputs = prepared["inputs"].to(device)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
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
        "terminated_normally": not hit,
        "eos": last in EOS_IDS if last is not None else False,
        "caption_flags": flags,
        "inference_time_sec": round(elapsed, 3),
        "peak_gpu_memory_mib": max(peaks.values()) if peaks else None,
        "peak_gpu_memory_mib_per_device": peaks,
    }


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def runnable(plans: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    wanted = []
    for p in plans:
        if p["status"] != "planned":
            continue
        if args.videos and p["video_id"] not in args.videos.split(","):
            continue
        if args.spatial and p["spatial"] not in args.spatial.split(","):
            continue
        if not p.get("budget_ok"):
            raise RuntimeError(f"{p['condition_id']} is over the visual-token budget")
        wanted.append(p)
    return wanted


def cmd_infer(args) -> None:
    import torch

    out = OUT / (args.subdir or "")
    ckpt_dir, cap_dir, inp_dir = out / "checkpoints", out / "captions", out / "inputs"
    for d in (ckpt_dir, cap_dir, inp_dir, out / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    infos = plan_all["videos"]
    gen = generation_record()
    wanted = runnable(plan_all["conditions"], args)

    def done(p):
        f = ckpt_dir / f"{p['condition_id']}.json"
        if not f.is_file() or args.force:
            return False
        rec = json.loads(f.read_text())
        ok = rec.get("status") == "ok" and rec.get("generation_sha") == gen["sha"]
        return ok or (rec.get("status") not in (None, "ok") and not args.retry_failed and rec.get("generation_sha") == gen["sha"])

    todo = [p for p in wanted if not done(p)]
    log.info("%d selected, %d to run, generation %s", len(wanted), len(todo), gen)
    if not todo:
        return
    processor, model = load_model()
    base = {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 2**20, 1) for i in range(torch.cuda.device_count())}
    log.info("model loaded; resident MiB %s ; class %s", base, type(model).__name__)
    for p in todo:
        cid = p["condition_id"]
        rec: dict[str, Any] = {
            "condition_id": cid,
            "video_id": p["video_id"],
            "video_name": p["video_name"],
            "spatial": p["spatial"],
            "fps": p["fps"],
            "window": p["window"],
            "full_duration_sec": p["full_duration_sec"],
            "started_at": utc_now(),
            "model_path": CFG["model"]["path"],
            "model_id": CFG["model"]["id"],
            "model_class_expected": "Qwen3-VL-8B-Instruct",
            "model_resident_mib": base,
            "generation": gen,
            "generation_sha": gen["sha"],
            "plan": p,
        }
        prepared = None
        try:
            prepared = prepare_inputs(processor, p, infos[p["video_id"]])
            produced = generate_one(model, processor, prepared)
            rec.update(status="ok", **produced)
            rec.update(
                {
                    k: prepared[k]
                    for k in (
                        "image_tokens",
                        "video_tokens",
                        "total_visual_tokens",
                        "prompt_tokens",
                        "text_tokens",
                        "prep_sec",
                        "max_abs_error_sec",
                        "timestamp_markers_verified",
                        "roi_timestamps",
                        "prompt_text",
                        "global_sampling",
                    )
                }
            )
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
                },
            )
            (cap_dir / f"{cid}.txt").write_text(produced["caption"] + "\n", encoding="utf-8")
            log.info(
                "%s ok tok=%d out=%d flags=%s %.1fs peak=%s",
                cid,
                prepared["total_visual_tokens"],
                produced["output_token_count"],
                produced["caption_flags"],
                produced["inference_time_sec"],
                produced["peak_gpu_memory_mib"],
            )
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            rec.update(
                status="oom" if is_oom(exc) else "error",
                error=f"{type(exc).__name__}: {exc}"[:2000],
                traceback=traceback.format_exc()[-3000:],
            )
            log.error("%s %s: %s", cid, rec["status"], str(exc)[:400])
        rec["finished_at"] = utc_now()
        write_json(ckpt_dir / f"{cid}.json", rec)
        del prepared
        gc.collect()
        torch.cuda.empty_cache()


def cmd_aggregate(args) -> None:
    out = OUT / (args.subdir or "")
    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    rows = []
    blocks = ["# Whole-video captions\n", "Candidate-GT experiment. Captions are model output, not references.\n"]
    for p in plan_all["conditions"]:
        f = out / "checkpoints" / f"{p['condition_id']}.json"
        r = json.loads(f.read_text()) if f.is_file() else {"status": p["status"] if p["status"] != "planned" else "not_run", "plan": p}
        pl = r.get("plan", p)
        used = [x for x in pl.get("rois", []) if x.get("grid_idx")]
        rows.append(
            {
                "condition_id": p["condition_id"],
                "video_id": p["video_name"],
                "short_id": p["video_id"],
                "full_duration_sec": p["full_duration_sec"],
                "spatial": p["spatial"],
                "fps": p.get("fps", ""),
                "status": r.get("status"),
                "error": r.get("error", pl.get("hold_reason", "")),
                "native_resolution": "x".join(map(str, p["native_resolution"])),
                "roi_count": len(used),
                "rois": json.dumps(
                    [
                        {
                            "id": x["roi_id"],
                            "bbox_xyxy": x["bbox_xyxy"],
                            "bbox_xywh": x.get("bbox_xywh"),
                            "temporal_coverage_sec": x.get("temporal_coverage_sec", x.get("time_range")),
                            "sampled_range": x.get("source_time_first_last"),
                            "frames": x.get("frames"),
                            "roi_native_size": x.get("roi_native_size"),
                            "input_resolution": x.get("input_resolution"),
                            "source": x.get("source"),
                        }
                        for x in used
                    ]
                ),
                "frame_count": "|".join(str(x.get("frames")) for x in used),
                "input_resolution": "|".join(
                    f"{x['input_resolution'][0]}x{x['input_resolution'][1]}" for x in used if x.get("input_resolution")
                ),
                "global_image_count": pl.get("global_image_count", 0),
                "global_images": json.dumps(r.get("global_sampling", pl.get("global_images", []))),
                "image_tokens": r.get("image_tokens", pl.get("planned_image_tokens")),
                "video_tokens": r.get("video_tokens", pl.get("planned_video_tokens")),
                "total_visual_tokens": r.get("total_visual_tokens", pl.get("planned_total_visual_tokens")),
                "text_tokens": r.get("text_tokens"),
                "binding_constraint": (pl.get("allocation") or {}).get("binding", ""),
                "peak_gpu_memory_mib": r.get("peak_gpu_memory_mib"),
                "inference_time_sec": r.get("inference_time_sec"),
                "prep_sec": round(r["prep_sec"], 2) if r.get("prep_sec") else None,
                "output_tokens": r.get("output_token_count"),
                "terminated_normally": r.get("terminated_normally"),
                "eos": r.get("eos"),
                "hit_token_limit": r.get("hit_token_limit"),
                "caption_flags": "|".join(r.get("caption_flags") or []),
                "caption": r.get("caption", ""),
            }
        )
        if r.get("caption"):
            blocks.append(
                f"## {p['video_name']} · {p['spatial']}\n\n"
                f"tokens={r.get('total_visual_tokens')} flags={r.get('caption_flags')}\n\n"
                f"{r['caption']}\n"
            )
    write_json(out / "results.json", rows)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if not args.subdir:
        (OUT / "captions.md").write_text("\n".join(blocks), encoding="utf-8")
    print(Counter(r["status"] for r in rows), "->", out / "results.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("references")
    inf = sub.add_parser("infer")
    agg = sub.add_parser("aggregate")
    for s in (inf, agg):
        s.add_argument("--subdir", default="")
    inf.add_argument("--videos")
    inf.add_argument("--spatial")
    inf.add_argument("--force", action="store_true")
    inf.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    {"plan": cmd_plan, "references": cmd_references, "infer": cmd_infer, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
