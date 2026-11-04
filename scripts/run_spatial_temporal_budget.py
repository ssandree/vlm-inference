#!/usr/bin/env python3
"""Spatial-temporal input configuration under a fixed visual-token budget.

4 VIRAT clips x 5 FPS x 3 spatial inputs (S0 full-frame video, S1 automatic
Motion ROI + first-frame image, S2 manual lifetime ROI + first-frame image).

Existing results and code are not touched. Pieces reused from the repo:
  * frame_clock.encode_timestamps (source_timestamp mode)
  * qwen_vl_utils.smart_resize (same resize rule as the existing pipeline)
  * Stage3 roi_tracks / Stage2 fusion timeline (motion-detector outputs)
  * manual lifetime bboxes from experiments/gold_lifetime_crop/configs/manifest.json

Commands:  plan | infer | aggregate
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import gc
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
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

OUT = REPO / "experiments" / "spatial_temporal_budget_40960"
CFG = yaml.safe_load((OUT / "configs" / "experiment.yaml").read_text())
FACTOR = 32  # patch 16 * merge 2
TEMPORAL = 2
EOS_IDS = [151645, 151643]
log = logging.getLogger("stb")

# --------------------------------------------------------------------------- prompt
# Common caption instructions: identical for every condition (the 1-3/1-4 event-centric prompt).
COMMON_PROMPT = """Describe the distinct actions occurring in this surveillance video.

Write one concise sentence per event in chronological order.
Identify the actor, action, and involved objects.
Clearly distinguish entering, exiting, approaching, and moving.
Describe only visually supported actions.
Do not invent events or repeat the same continuous action.
Avoid irrelevant static-scene descriptions."""


def input_description(spatial: str, start: float, end: float, n_roi: int) -> str:
    if spatial == "S0":
        return (
            f"The input is one full-frame video covering the original recording from {start:.1f}s to {end:.1f}s. "
            "Timestamps shown in the video are original recording times."
        )
    noun = "one ROI video" if n_roi == 1 else f"{n_roi} ROI videos"
    text = (
        f"The input is a full-frame image captured at {start:.1f}s of the original recording, followed by {noun}. "
        "Each ROI video is an independent spatial crop of the same scene and covers only the time range stated in its label; "
        "its timestamps are original recording times. "
        "Use the full-frame image only for scene context."
    )
    if n_roi > 1:
        text += (
            " Events seen in different ROI videos are separate unless the same actor clearly continues; "
            "do not merge them into one event and do not describe the same event twice."
        )
    return text


def full_prompt(spatial: str, start: float, end: float, n_roi: int) -> str:
    return input_description(spatial, start, end, n_roi) + "\n\n" + COMMON_PROMPT


# --------------------------------------------------------------------------- helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fps_tag(fps: float) -> str:
    return f"fps{fps:g}".replace(".", "p")


def cond_id(vid: str, spatial: str, fps: float) -> str:
    return f"{vid}_{spatial}_{fps_tag(fps)}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def resolve_video(name: str) -> Path:
    for root in CFG["sources"]["video_roots"]:
        p = Path(root) / f"{name}.mp4"
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def probe(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
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


# --------------------------------------------------------------------------- ROI sources
def motion_rois(vid_name: str, width: int, height: int) -> list[dict[str, Any]]:
    """Stage3 tubes: lifetime AABB from spatial_bbox_grid, time range from Stage2 timeline (same rule as export_vlm_sampling)."""
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
        rois.append(
            {
                "roi_id": f"tube{int(tube['tube_id'])}",
                "bbox_xyxy": [x0, y0, x1, y1],
                "time_range": [float(ts[t0]), float(ts[t1])],
                "source": "motion_stage3_tube",
            }
        )
    return rois


def manual_rois(video_id: str, start: float, end: float) -> list[dict[str, Any]]:
    man = json.loads((REPO / CFG["sources"]["manual_lifetime_bbox_manifest"]).read_text())
    entry = next(v for v in man["videos"] if v["video_id"] == video_id)
    return [
        {
            "roi_id": f"manual{i + 1}",
            "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
            "time_range": [float(start), float(end)],  # lifetime bbox: fixed over the whole window
            "source": "gold_lifetime_bbox",
        }
        for i, (x, y, w, h) in enumerate(entry["bboxes"])
    ]


# --------------------------------------------------------------------------- planning
def allocate(items: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    """Largest common per-frame pixel cap (<= video.max_pixels) so that sum(video tokens) <= budget.

    Resize rule is qwen_vl_utils.smart_resize (min_pixels/max_pixels), as in the existing pipeline.
    """
    vmin, vmax = int(CFG["video"]["min_pixels"]), int(CFG["video"]["max_pixels"])

    def total(cap: int) -> int:
        s = 0
        for it in items:
            h, w = smart_resize(it["native_h"], it["native_w"], vmin, max(cap, int(vmin * 1.05)))
            s += video_tokens(it["frames"], h, w)
        return s

    cap = vmax
    uncapped = total(vmax)
    if uncapped > budget:
        lo, hi = int(vmin * 1.05), vmax
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total(mid) <= budget:
                lo = mid
            else:
                hi = mid - 1
        cap = lo
        while total(cap) > budget and cap > int(vmin * 1.05):
            cap -= 256
    out = []
    for it in items:
        h, w = smart_resize(it["native_h"], it["native_w"], vmin, max(cap, int(vmin * 1.05)))
        out.append({"input_h": h, "input_w": w, "tokens": video_tokens(it["frames"], h, w)})
    binding = "budget" if uncapped > budget else "per_frame_max_pixels_or_native"
    return {"cap_pixels": cap, "tokens_at_vmax": uncapped, "binding": binding, "items": out}


def plan_condition(video: dict[str, Any], info: dict[str, Any], spatial: str, fps: float) -> dict[str, Any]:
    start, end = float(video["start"]), float(video["end"])
    budget = int(CFG["experiment"]["token_budget"])
    grid = grid_times(start, end, fps)
    W, H = info["width"], info["height"]
    plan: dict[str, Any] = {
        "condition_id": cond_id(video["id"], spatial, fps),
        "video_id": video["id"],
        "video_name": video["name"],
        "spatial": spatial,
        "fps": fps,
        "window": [start, end],
        "native_resolution": [W, H],
        "native_fps": info["native_fps"],
        "grid_frame_count": len(grid),
        "token_budget": budget,
    }
    img_tok = 0
    if spatial == "S0":
        rois = [
            {"roi_id": "full", "bbox_xyxy": [0, 0, W, H], "time_range": [start, end], "source": "full_frame", "grid_idx": list(range(len(grid)))}
        ]
        plan["global_image"] = None
    else:
        raw = motion_rois(video["name"], W, H) if spatial == "S1" else manual_rois(video["id"], start, end)
        rois = []
        for r in raw:
            t0, t1 = r["time_range"]
            idx = [k for k, t in enumerate(grid) if t0 - 1e-6 <= t <= t1 + 1e-6]
            r = dict(r, grid_idx=idx)
            if not idx:
                r["excluded"] = "no sampled frame inside tube lifetime ∩ window"
            rois.append(r)
        gh, gw = smart_resize(H, W, int(CFG["image"]["min_pixels"]), int(CFG["image"]["max_pixels"]))
        img_tok = image_tokens(gh, gw)
        plan["global_image"] = {
            "time_target": grid[0],
            "native_resolution": [W, H],
            "input_resolution": [gw, gh],
            "tokens": img_tok,
        }
    used = [r for r in rois if r["grid_idx"]]
    items = [
        {
            "native_h": r["bbox_xyxy"][3] - r["bbox_xyxy"][1],
            "native_w": r["bbox_xyxy"][2] - r["bbox_xyxy"][0],
            "frames": len(r["grid_idx"]),
        }
        for r in used
    ]
    alloc = allocate(items, budget - img_tok)
    for r, it, a in zip(used, items, alloc["items"]):
        r["roi_native_size"] = [it["native_w"], it["native_h"]]
        r["frames"] = it["frames"]
        r["input_resolution"] = [a["input_w"], a["input_h"]]
        r["planned_video_tokens"] = a["tokens"]
        r["model_frames_with_temporal_padding"] = math.ceil(it["frames"] / TEMPORAL) * TEMPORAL
        r["temporal_padding_frame"] = r["model_frames_with_temporal_padding"] - it["frames"]
    plan["rois"] = rois
    plan["roi_count_used"] = len(used)
    plan["allocation"] = {k: v for k, v in alloc.items() if k != "items"}
    plan["planned_image_tokens"] = img_tok
    plan["planned_video_tokens"] = sum(r["planned_video_tokens"] for r in used)
    plan["planned_total_visual_tokens"] = plan["planned_image_tokens"] + plan["planned_video_tokens"]
    plan["budget_ok"] = plan["planned_total_visual_tokens"] <= budget
    return plan


def all_conditions() -> list[tuple[dict, str, float]]:
    return [(v, s, f) for v in CFG["videos"] for f in CFG["experiment"]["fps"] for s in CFG["experiment"]["spatial"]]


def cmd_plan(args) -> None:
    infos = {v["id"]: probe(resolve_video(v["name"])) for v in CFG["videos"]}
    plans = [plan_condition(v, infos[v["id"]], s, f) for v, s, f in all_conditions()]
    write_json(OUT / "configs" / "plan.json", {"created_at": utc_now(), "videos": infos, "conditions": plans})
    rows = []
    for p in plans:
        rows.append(
            {
                "condition_id": p["condition_id"], "video": p["video_id"], "spatial": p["spatial"], "fps": p["fps"],
                "grid_frames": p["grid_frame_count"], "rois": p["roi_count_used"],
                "image_tokens": p["planned_image_tokens"], "video_tokens": p["planned_video_tokens"],
                "total_tokens": p["planned_total_visual_tokens"], "budget_ok": p["budget_ok"],
                "binding": p["allocation"]["binding"],
                "roi_inputs": " | ".join(f"{r['roi_id']}:{r['frames']}f@{r['input_resolution'][0]}x{r['input_resolution'][1]}" for r in p["rois"] if r["grid_idx"]),
            }
        )
    with open(OUT / "configs" / "plan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r["condition_id"] for r in rows if not r["budget_ok"]]
    print(f"{len(rows)} conditions planned; budget violations: {bad}")


# --------------------------------------------------------------------------- decoding
def decode_nearest(path: str, targets: list[float]) -> list[dict[str, Any]]:
    """Nearest decoded frame (by PTS) for each target time."""
    import av

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
                        best[k] = {"target_time_sec": targets[k], "source_time_sec": t, "abs_error_sec": d, "array": arr}
    if len(best) != len(targets):
        raise RuntimeError(f"decoded {len(best)} of {len(targets)} target frames")
    return [best[k] for k in range(len(targets))]


# --------------------------------------------------------------------------- model
def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        import torch.backends.python_native as pn

        pn.triton.enabled = False
    except Exception:
        pass
    path = CFG["model"]["path"]
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation=CFG["model"]["attn_implementation"]
    )
    model.eval()
    return processor, model


def resize_frames(frames: np.ndarray, h: int, w: int):
    """frames: T,H,W,3 uint8 -> T,3,h,w float tensor, bicubic+antialias like qwen_vl_utils."""
    import torch
    from torchvision import transforms

    t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if t.shape[2] == h and t.shape[3] == w:
        return t.float()
    return transforms.functional.resize(t, [h, w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True).float()


def prepare_inputs(processor, plan: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    t_start = time.perf_counter()
    start, end = plan["window"]
    grid = grid_times(start, end, plan["fps"])
    sel = decode_nearest(info["path"], grid)
    src_times = [s["source_time_sec"] for s in sel]
    if len(set(round(t, 6) for t in src_times)) != len(src_times):
        raise RuntimeError("duplicate source frames in sampling grid")
    max_err = max(s["abs_error_sec"] for s in sel)
    if max_err > 1.5 / info["native_fps"]:
        raise RuntimeError(f"timestamp error {max_err:.4f}s too large")

    content: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []
    metadatas: list[dict[str, Any]] = []
    spatial = plan["spatial"]
    used = [r for r in plan["rois"] if r["grid_idx"]]

    if plan["global_image"]:
        g = plan["global_image"]
        gw, gh = g["input_resolution"]
        img = Image.fromarray(sel[0]["array"])
        img = img.resize((gw, gh), Image.BICUBIC) if (gw, gh) != img.size else img
        images.append(img)
        g["source_time_sec"] = src_times[0]
        content.append({"type": "text", "text": f"[Full-frame image, t={src_times[0]:.2f}s]"})
        content.append({"type": "image", "image": "memory://global"})

    roi_ts: dict[str, list[float]] = {}
    for r in used:
        x0, y0, x1, y1 = r["bbox_xyxy"]
        frames = np.stack([sel[k]["array"][y0:y1, x0:x1] for k in r["grid_idx"]], axis=0)
        w, h = r["input_resolution"]
        videos.append(resize_frames(frames, h, w))
        stamps = [src_times[k] for k in r["grid_idx"]]
        roi_ts[r["roi_id"]] = stamps
        r["source_time_first_last"] = [stamps[0], stamps[-1]]
        r["window_time_range"] = [stamps[0], stamps[-1]]
        indices, fps_enc = encode_timestamps(stamps, scale=1000)
        metadatas.append({"fps": fps_enc, "frames_indices": indices, "total_num_frames": max(indices) + 1, "video_backend": "av"})
        if spatial == "S0":
            label = f"[Full-frame video, {stamps[0]:.2f}s-{stamps[-1]:.2f}s]"
        else:
            label = (
                f"[{r['roi_id']} video: crop x={x0}-{x1}, y={y0}-{y1} of the {info['width']}x{info['height']} frame, "
                f"{stamps[0]:.2f}s-{stamps[-1]:.2f}s]"
            )
        content.append({"type": "text", "text": label})
        content.append({"type": "video", "video": f"memory://{r['roi_id']}"})

    prompt = full_prompt(spatial, start, end, len(used))
    content.append({"type": "text", "text": prompt})
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images or None,
        videos=videos or None,
        video_metadata=metadatas or None,
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )

    # ---- verify against the processor, not the plan
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
        if grid != [math.ceil(r["frames"] / TEMPORAL), h // 16, w // 16]:
            raise RuntimeError(f"{r['roi_id']}: grid {grid} != expected for {r['frames']}f {w}x{h}")
        r["video_grid_thw"] = grid
        r["measured_video_tokens"] = grid[0] * grid[1] * grid[2] // 4
    if vid_tok != plan["planned_video_tokens"] or img_tok != plan["planned_image_tokens"]:
        raise RuntimeError(f"measured tokens {img_tok}+{vid_tok} differ from plan {plan['planned_image_tokens']}+{plan['planned_video_tokens']}")

    decoded = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    # ---- the "<x.x seconds>" markers the processor wrote must equal the pair-mean of our source timestamps
    import re

    expected = []
    for r in used:
        idx, scale = encode_timestamps(roi_ts[r["roi_id"]], scale=1000)  # processor sees ms-quantised times
        st = [i / scale for i in idx]
        st = st + [st[-1]] * (len(st) % 2)
        expected += [f"{(st[i] + st[i + 1]) / 2:.1f}" for i in range(0, len(st), 2)]
    found = re.findall(r"<(\d+(?:\.\d+)?) seconds>", decoded)
    if found != expected:
        raise RuntimeError(f"timestamp markers differ from source timestamps: {found[:6]} vs {expected[:6]}")
    return {
        "inputs": inputs,
        "prompt_text": prompt,
        "decoded_prompt_head": decoded.replace("<|video_pad|>", "").replace("<|image_pad|>", "")[:6000],
        "image_tokens": img_tok,
        "video_tokens": vid_tok,
        "total_visual_tokens": img_tok + vid_tok,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "text_tokens": int(inputs["input_ids"].shape[1]) - img_tok - vid_tok,
        "sampling": [{"k": k, "target": s["target_time_sec"], "source": s["source_time_sec"], "abs_err": round(s["abs_error_sec"], 5)} for k, s in enumerate(sel)],
        "max_abs_error_sec": max_err,
        "roi_timestamps": roi_ts,
        "timestamp_markers_verified": len(found),
        "prep_sec": time.perf_counter() - t_start,
        "plan": plan,
    }


def generate_one(model, processor, prepared: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.manual_seed(int(CFG["generation"]["seed"]))
    inputs = prepared["inputs"].to("cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            repetition_penalty=float(CFG["generation"]["repetition_penalty"]),
            max_new_tokens=int(CFG["generation"]["max_new_tokens"]),
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    n_in = int(inputs["input_ids"].shape[1])
    gen = out[0, n_in:]
    text = processor.batch_decode(gen.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    n_out = int(gen.shape[0])
    last = int(gen[-1]) if n_out else None
    hit = n_out >= int(CFG["generation"]["max_new_tokens"]) and last not in EOS_IDS
    del inputs, out, gen
    return {
        "caption": text, "output_token_count": n_out, "hit_token_limit": hit, "terminated_normally": not hit,
        "inference_time_sec": round(elapsed, 3), "peak_gpu_memory_mib": round(peak / 2**20, 1),
        "peak_gpu_reserved_mib": round(reserved / 2**20, 1),
    }


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


# --------------------------------------------------------------------------- infer
def cmd_infer(args) -> None:
    import torch

    out = OUT / (args.subdir or "")
    ckpt_dir, cap_dir, samp_dir = out / "checkpoints", out / "captions", out / "sampling"
    for d in (ckpt_dir, cap_dir, samp_dir):
        d.mkdir(parents=True, exist_ok=True)
    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    infos = plan_all["videos"]
    wanted = [p for p in plan_all["conditions"]
              if (not args.videos or p["video_id"] in args.videos.split(","))
              and (not args.spatial or p["spatial"] in args.spatial.split(","))
              and (not args.fps or p["fps"] in [float(x) for x in args.fps.split(",")])]

    def done(p):
        f = ckpt_dir / f"{p['condition_id']}.json"
        if not f.is_file() or args.force:
            return False
        st = json.loads(f.read_text()).get("status")
        return st == "ok" or (st != "ok" and not args.retry_failed)

    todo = [p for p in wanted if not done(p)]
    log.info("%d selected, %d to run", len(wanted), len(todo))
    if not todo:
        return
    processor, model = load_model()
    log.info("model loaded; baseline allocated %.0f MiB", torch.cuda.memory_allocated() / 2**20)
    base_mib = round(torch.cuda.memory_allocated() / 2**20, 1)
    for p in todo:
        cid = p["condition_id"]
        rec: dict[str, Any] = {"condition_id": cid, "video_id": p["video_id"], "spatial": p["spatial"], "fps": p["fps"], "window": p["window"],
                               "started_at": utc_now(), "model_resident_mib": base_mib, "plan": p}
        prepared = None
        try:
            prepared = prepare_inputs(processor, p, infos[p["video_id"]])
            gen = generate_one(model, processor, prepared)
            rec.update(status="ok", **gen)
            rec.update({k: prepared[k] for k in ("image_tokens", "video_tokens", "total_visual_tokens", "prompt_tokens", "text_tokens",
                                                  "prep_sec", "max_abs_error_sec", "timestamp_markers_verified", "roi_timestamps", "prompt_text", "decoded_prompt_head")})
            rec["plan"] = prepared["plan"]
            write_json(samp_dir / f"{cid}.json", {"condition_id": cid, "sampling": prepared["sampling"], "roi_timestamps": prepared["roi_timestamps"]})
            (cap_dir / f"{cid}.txt").write_text(gen["caption"] + "\n", encoding="utf-8")
            log.info("%s ok tok=%d out=%d hit=%s %.1fs peak=%.0fMiB", cid, prepared["total_visual_tokens"], gen["output_token_count"],
                     gen["hit_token_limit"], gen["inference_time_sec"], gen["peak_gpu_memory_mib"])
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            rec.update(status="oom" if is_oom(exc) else "error", error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-3000:])
            log.error("%s %s: %s", cid, rec["status"], str(exc)[:200])
        rec["finished_at"] = utc_now()
        write_json(ckpt_dir / f"{cid}.json", rec)
        del prepared
        gc.collect()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- aggregate
def cmd_aggregate(args) -> None:
    out = OUT / (args.subdir or "")
    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    rows = []
    for p in plan_all["conditions"]:
        f = out / "checkpoints" / f"{p['condition_id']}.json"
        r = json.loads(f.read_text()) if f.is_file() else {"status": "not_run", "plan": p}
        pl = r.get("plan", p)
        used = [x for x in pl["rois"] if x["grid_idx"]]
        rows.append(
            {
                "condition_id": p["condition_id"], "video_id": p["video_id"], "window_start": p["window"][0], "window_end": p["window"][1],
                "spatial": p["spatial"], "fps": p["fps"], "status": r["status"], "error": r.get("error", ""),
                "native_resolution": "x".join(map(str, p["native_resolution"])),
                "roi_count": len(used),
                "rois": json.dumps([{"id": x["roi_id"], "bbox_xyxy": x["bbox_xyxy"], "tube_time_range": [round(v, 3) for v in x["time_range"]],
                                     "sampled_range": x.get("source_time_first_last"), "frames": x["frames"], "roi_native_size": x["roi_native_size"],
                                     "input_resolution": x["input_resolution"]} for x in used]),
                "input_frames_per_video": "|".join(str(x["frames"]) for x in used),
                "model_frames_with_temporal_padding": "|".join(str(x["model_frames_with_temporal_padding"]) for x in used),
                "input_resolution": "|".join(f"{x['input_resolution'][0]}x{x['input_resolution'][1]}" for x in used),
                "global_image": (json.dumps({"t": pl["global_image"].get("source_time_sec", pl["global_image"]["time_target"]), "input": pl["global_image"]["input_resolution"]}) if pl.get("global_image") else ""),
                "planned_total_tokens": pl["planned_total_visual_tokens"],
                "image_tokens": r.get("image_tokens"), "video_tokens": r.get("video_tokens"), "total_visual_tokens": r.get("total_visual_tokens"),
                "text_tokens": r.get("text_tokens"), "binding_constraint": pl["allocation"]["binding"],
                "peak_gpu_memory_mib": r.get("peak_gpu_memory_mib"), "inference_time_sec": r.get("inference_time_sec"), "prep_sec": round(r["prep_sec"], 2) if r.get("prep_sec") else None,
                "output_tokens": r.get("output_token_count"), "terminated_normally": r.get("terminated_normally"), "hit_token_limit": r.get("hit_token_limit"),
                "caption": r.get("caption", ""),
            }
        )
    write_json(out / "results.json", rows)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter

    print(Counter(r["status"] for r in rows), "->", out / "results.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    i = sub.add_parser("infer")
    a = sub.add_parser("aggregate")
    for s in (i, a):
        s.add_argument("--subdir", default="")
    i.add_argument("--videos")
    i.add_argument("--spatial")
    i.add_argument("--fps")
    i.add_argument("--force", action="store_true")
    i.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    {"plan": cmd_plan, "infer": cmd_infer, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
