#!/usr/bin/env python3
"""CPU-only input plan for the spatial-temporal budget trade-off.

Does not load the language model and does not write into the 1-3 or 1-4
result directories. Frame targets use the 1-4 uniform sampler. Visual tokens
use that experiment's formula and are checked against the Qwen3-VL processor
grid.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from temporal_ablation.inference import (  # noqa: E402
    align_up,
    formula_visual_tokens,
    grid_visual_tokens,
)
from temporal_ablation.sampling import probe_video, select_frames, uniform_times  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "temporal_ablation_main.yaml"
MEMORY_CSV = REPO / "results" / "temporal_ablation_main" / "temporal_results.csv"
CLIPS = ("A", "C", "D", "E")
BUDGETS = (5120, 10240, 20480)
FPS_LIST = (0.5, 1.0, 2.0, 4.0)
SPATIAL_FACTOR = 32
MIN_SIDE = 32
# Dedicated A6000, from the same host. Occupied GPUs are not used here.
A6000_TOTAL_MIB = 49140


def _load_clips() -> tuple[dict, str]:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    prompt = str(cfg["prompt"]).strip() + "\n"
    videos = []
    for item in cfg["videos"]:
        if item["video_id"] not in CLIPS:
            continue
        videos.append(item)
    order = {clip: index for index, clip in enumerate(CLIPS)}
    videos.sort(key=lambda item: order[item["video_id"]])
    if [item["video_id"] for item in videos] != list(CLIPS):
        raise RuntimeError("clip A/C/D/E missing from the 1-4 config")
    return {"videos": videos, "expected": cfg["expected_frame_counts"], "model": cfg["model"]}, prompt


def _nearest_sources(path: Path, native_fps: float, groups: dict[float, list[float]]) -> dict[float, list[dict]]:
    """One decode. Same nearest-PTS rule as temporal_ablation.sampling.select_frames."""
    all_targets = [stamp for stamps in groups.values() for stamp in stamps]
    margin = max(1.0 / native_fps, 0.05) if native_fps else 0.05
    window_start = min(all_targets) - margin
    window_end = max(all_targets) + margin
    decoded = []
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            timestamp = frame.time
            if timestamp is None or timestamp < window_start:
                continue
            decoded.append((int(frame_index), float(timestamp)))
            if timestamp > window_end:
                break
    finally:
        container.close()
    if not decoded:
        raise RuntimeError(f"no decoded frames in {path}")
    chosen = {}
    for fps, targets in groups.items():
        rows = []
        for target in targets:
            index, stamp = min(decoded, key=lambda item: abs(item[1] - target))
            rows.append(
                {
                    "target_time_sec": float(target),
                    "source_time_sec": stamp,
                    "source_frame_index": index,
                    "abs_error_sec": abs(stamp - target),
                }
            )
        chosen[fps] = rows
    return chosen


def _verify_against_select_frames(path: Path, native_fps: float, targets: list[float], rows: list[dict]) -> None:
    selected = select_frames(path, targets, native_fps=native_fps)
    got = [(row["source_frame_index"], round(row["source_time_sec"], 6)) for row in rows]
    ref = [
        (row["source_frame_index"], round(row["source_time_sec"], 6))
        for row in selected["frames"]
    ]
    if got != ref:
        raise RuntimeError("nearest-PTS pass does not match select_frames")
    del selected


def _search(num_frames: int, native_w: int, native_h: int, budget: int) -> dict | None:
    best = None
    for width in range(MIN_SIDE, native_w + 1):
        height = int(round(width * native_h / native_w))
        if height < MIN_SIDE or height > native_h:
            continue
        proc_w = align_up(width, SPATIAL_FACTOR)
        proc_h = align_up(height, SPATIAL_FACTOR)
        tokens = formula_visual_tokens(num_frames, proc_h, proc_w)
        if tokens > budget:
            continue
        # Larger processor canvas, then less padding, then tighter budget use.
        rank = (proc_w * proc_h, width * height, tokens)
        if best is None or rank > best[0]:
            best = (
                rank,
                {
                    "content_width": width,
                    "content_height": height,
                    "effective_width": proc_w,
                    "effective_height": proc_h,
                    "visual_tokens_formula": tokens,
                },
            )
    return None if best is None else best[1]


def _memory_model() -> tuple[float, float, int]:
    tokens = []
    peaks = []
    with MEMORY_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok" or not row.get("visual_tokens"):
                continue
            tokens.append(float(row["visual_tokens"]))
            peaks.append(float(row["peak_gpu_memory_mib"]))
    if len(tokens) < 4:
        raise RuntimeError(f"not enough 1-4 memory rows in {MEMORY_CSV}")
    slope, intercept = np.polyfit(np.array(tokens), np.array(peaks), 1)
    return float(slope), float(intercept), len(tokens)


def _risk(peak_mib: float) -> str:
    fraction = peak_mib / A6000_TOTAL_MIB
    if fraction >= 0.95:
        return "likely_oom"
    if fraction >= 0.85:
        return "high"
    if fraction >= 0.70:
        return "elevated"
    return "low"


def _processor_check(processor, prompt: str, num_frames: int, proc_h: int, proc_w: int) -> dict:
    tensor = torch.zeros((num_frames, 3, proc_h, proc_w), dtype=torch.uint8)
    messages = [{"role": "user", "content": [{"type": "video", "video": "dummy.mp4"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    metadata = {
        "fps": 30.0,
        "frames_indices": list(range(num_frames)),
        "total_num_frames": num_frames,
        "video_backend": "av",
    }
    inputs = processor(
        text=[text],
        videos=[tensor],
        video_metadata=[metadata],
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )
    grid = [int(value) for value in inputs["video_grid_thw"][0].tolist()]
    measured = grid_visual_tokens(grid)
    placeholder = int((inputs["input_ids"] == processor.video_token_id).sum().item())
    return {
        "video_grid_thw": grid,
        "visual_tokens": measured,
        "placeholder_visual_tokens": placeholder,
        "input_token_count": int(inputs["input_ids"].shape[1]),
    }


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    meta, prompt = _load_clips()
    slope, intercept, memory_n = _memory_model()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(meta["model"]["path"])
    rows = []
    for video in meta["videos"]:
        path = Path(video["source_path"])
        probe = probe_video(path)
        native_w, native_h = int(probe["width"]), int(probe["height"])
        groups = {
            float(fps): uniform_times(float(video["clip_start"]), float(video["clip_end"]), float(fps), 0.0)
            for fps in FPS_LIST
        }
        for fps, stamps in groups.items():
            table = meta["expected"][video["video_id"]]
            if fps in table:
                expected = int(table[fps])
            elif str(fps) in table:
                expected = int(table[str(fps)])
            elif fps.is_integer() and int(fps) in table:
                expected = int(table[int(fps)])
            else:
                raise KeyError(f"no expected frame count for {video['video_id']} fps {fps}")
            if len(stamps) != expected:
                raise RuntimeError(f"{video['video_id']} fps {fps}: {len(stamps)} != table {expected}")
        sources = _nearest_sources(path, float(probe["native_fps"]), groups)
        _verify_against_select_frames(path, float(probe["native_fps"]), groups[0.5], sources[0.5])
        for fps in FPS_LIST:
            sampled = sources[fps]
            num_frames = len(sampled)
            for budget in BUDGETS:
                found = _search(num_frames, native_w, native_h, budget)
                base = {
                    "clip_id": video["video_id"],
                    "filename": video["filename"],
                    "source_path": str(path),
                    "clip_start_sec": float(video["clip_start"]),
                    "clip_end_sec": float(video["clip_end"]),
                    "native_fps": probe["native_fps"],
                    "native_resolution": f"{native_w}x{native_h}",
                    "requested_fps": fps,
                    "budget_visual_tokens": budget,
                    "sampled_frame_count": num_frames,
                    "target_timestamps_sec": ";".join(f"{row['target_time_sec']:.6f}" for row in sampled),
                    "source_timestamps_sec": ";".join(f"{row['source_time_sec']:.6f}" for row in sampled),
                    "source_frame_indices": ";".join(str(row["source_frame_index"]) for row in sampled),
                    "duplicate_source_frames": len({row["source_frame_index"] for row in sampled}) != num_frames,
                    "max_abs_decoder_error_sec": round(max(row["abs_error_sec"] for row in sampled), 6),
                }
                if found is None:
                    minimum = formula_visual_tokens(num_frames, SPATIAL_FACTOR, SPATIAL_FACTOR)
                    rows.append(
                        {
                            **base,
                            "feasible": False,
                            "infeasible_reason": f"minimum 32x32 grid uses {minimum} tokens",
                            "content_resolution": "",
                            "effective_resolution": "",
                            "grid_pad_right": "",
                            "grid_pad_bottom": "",
                            "video_grid_thw": "",
                            "visual_tokens": "",
                            "visual_tokens_formula": minimum,
                            "placeholder_visual_tokens": "",
                            "processor_matches_formula": "",
                            "unused_visual_tokens": "",
                            "budget_utilization": "",
                            "full_native_fits": False,
                            "input_token_count": "",
                            "estimated_peak_gpu_mib": "",
                            "oom_risk": "",
                        }
                    )
                    continue
                checked = _processor_check(
                    processor,
                    prompt,
                    num_frames,
                    found["effective_height"],
                    found["effective_width"],
                )
                formula_tokens = found["visual_tokens_formula"]
                if checked["visual_tokens"] != formula_tokens or checked["placeholder_visual_tokens"] != formula_tokens:
                    raise RuntimeError(
                        f"processor mismatch {video['video_id']} fps {fps} budget {budget}: {checked} vs {formula_tokens}"
                    )
                native_tokens = formula_visual_tokens(
                    num_frames,
                    align_up(native_h, SPATIAL_FACTOR),
                    align_up(native_w, SPATIAL_FACTOR),
                )
                peak = intercept + slope * checked["visual_tokens"]
                rows.append(
                    {
                        **base,
                        "feasible": True,
                        "infeasible_reason": "",
                        "content_resolution": f"{found['content_width']}x{found['content_height']}",
                        "effective_resolution": f"{found['effective_width']}x{found['effective_height']}",
                        "grid_pad_right": found["effective_width"] - found["content_width"],
                        "grid_pad_bottom": found["effective_height"] - found["content_height"],
                        "video_grid_thw": str(checked["video_grid_thw"]),
                        "visual_tokens": checked["visual_tokens"],
                        "visual_tokens_formula": formula_tokens,
                        "placeholder_visual_tokens": checked["placeholder_visual_tokens"],
                        "processor_matches_formula": True,
                        "unused_visual_tokens": budget - checked["visual_tokens"],
                        "budget_utilization": round(checked["visual_tokens"] / budget, 6),
                        "full_native_fits": native_tokens <= budget,
                        "input_token_count": checked["input_token_count"],
                        "estimated_peak_gpu_mib": round(peak, 1),
                        "oom_risk": _risk(peak),
                    }
                )
                print(
                    video["video_id"],
                    fps,
                    budget,
                    rows[-1]["effective_resolution"],
                    rows[-1]["visual_tokens"],
                    "unused",
                    rows[-1]["unused_visual_tokens"],
                    rows[-1]["oom_risk"],
                )
    csv_path = OUT / "budget_feasible_configs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _heatmap(rows, OUT / "budget_fps_resolution_heatmap.png")
    _yaml(rows, meta, prompt, slope, intercept, memory_n, OUT / "budget_experiment_config.yaml")
    print("wrote", csv_path)
    return 0


def _heatmap(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, sharey=True)
    for axis, clip in zip(axes.ravel(), CLIPS):
        subset = [row for row in rows if row["clip_id"] == clip]
        grid = np.full((len(BUDGETS), len(FPS_LIST)), np.nan)
        labels = [["" for _ in FPS_LIST] for _ in BUDGETS]
        for row in subset:
            y = BUDGETS.index(int(row["budget_visual_tokens"]))
            x = FPS_LIST.index(float(row["requested_fps"]))
            if row["feasible"]:
                width, height = (int(part) for part in str(row["effective_resolution"]).split("x"))
                grid[y, x] = width * height / 1e6
                labels[y][x] = f"{row['effective_resolution']}\n{row['visual_tokens']} tok"
            else:
                labels[y][x] = "infeasible"
        image = axis.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=0, vmax=2.2)
        axis.set_xticks(range(len(FPS_LIST)), [str(fps) for fps in FPS_LIST])
        axis.set_yticks(range(len(BUDGETS)), [str(budget) for budget in BUDGETS])
        axis.set_title(f"clip {clip}")
        axis.set_xlabel("FPS")
        axis.set_ylabel("visual-token budget")
        for y in range(len(BUDGETS)):
            for x in range(len(FPS_LIST)):
                axis.text(x, y, labels[y][x], ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="effective megapixels")
    fig.suptitle("Maximum effective resolution under each visual-token budget")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _yaml(rows, meta, prompt, slope, intercept, memory_n, path: Path) -> None:
    payload = {
        "experiment": {
            "name": "spatial_temporal_budget_1_5",
            "run_inference": False,
            "reason": "Input plan only. Decide whether to run after 1-3 and 1-4 results.",
            "clips": list(CLIPS),
            "budgets": list(BUDGETS),
            "fps": list(FPS_LIST),
            "sampling": "temporal_ablation.sampling.uniform_times, phase 0, clip end exclusive; nearest decoded PTS",
            "resolution_search": "largest 32-padded canvas whose formula tokens fit, content aspect follows the native frame, no upscaling",
            "token_check": "Qwen3-VL processor video_grid_thw on a zero tensor of the chosen shape; do_resize=false; do_sample_frames=false",
        },
        "model": {
            "id": meta["model"]["id"],
            "path": meta["model"]["path"],
            "revision": meta["model"]["revision"],
            "dtype": meta["model"]["dtype"],
            "attn_implementation": meta["model"]["attn_implementation"],
            "do_sample": False,
            "max_new_tokens": meta["model"]["max_new_tokens"],
            "prompt": prompt,
        },
        "memory_estimate": {
            "source": str(MEMORY_CSV),
            "n_observed_runs": memory_n,
            "peak_mib_equals": f"{intercept:.4f} + {slope:.6f} * visual_tokens",
            "reference_gpu_mib": A6000_TOTAL_MIB,
            "note": "Linear fit to finished 1-4 runs at 1280x736, bf16, sdpa, max_new_tokens=256. Not a new measurement.",
        },
        "conditions": [
            {
                "clip_id": row["clip_id"],
                "fps": row["requested_fps"],
                "budget_visual_tokens": row["budget_visual_tokens"],
                "feasible": row["feasible"],
                "sampled_frame_count": row["sampled_frame_count"],
                "content_resolution": row["content_resolution"],
                "effective_resolution": row["effective_resolution"],
                "visual_tokens": row["visual_tokens"],
                "unused_visual_tokens": row["unused_visual_tokens"],
                "oom_risk_on_dedicated_a6000": row["oom_risk"],
                "estimated_peak_gpu_mib": row["estimated_peak_gpu_mib"],
            }
            for row in rows
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
