#!/usr/bin/env python3
"""Compute Qwen3-VL layer-4 spatiotemporal visual information volume.

The full sampled frame sequence is one multimodal input. A single full
forward caches the layer-4 decoder boundary; only the decoder suffix is
replayed for batched 2x2 visual-token interventions.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from urllib.parse import urlparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from vlm_inference.config import JobConfig, load_config
from vlm_inference.inputs import resolve_job
from vlm_inference.model import Qwen3VLCaptioner
from vlm_inference.prompts import load_job_prompt, load_sample_prompt
from vlm_inference.token_information import CachedQwenSuffix, make_blocks

LOG = logging.getLogger("video_information_volume")
TIMESTAMP_RE = re.compile(r"(?:^|[_-])t(?P<time>\d+(?:\.\d+)?)", re.IGNORECASE)


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _timestamps(paths: list[str], fps: float | None) -> list[float]:
    values = []
    for index, path in enumerate(paths):
        match = TIMESTAMP_RE.search(Path(path).stem)
        values.append(float(match.group("time")) if match else (index / fps if fps else float(index)))
    return values


def _local_path(path: str) -> Path:
    parsed = urlparse(path)
    return Path(parsed.path if parsed.scheme == "file" else path)


def _caption(raw: dict, video_id: str) -> str:
    info = raw["visual_information_volume"]
    if info.get("gt_caption"):
        return str(info["gt_caption"]).strip()
    refs = json.loads(Path(info["gt_reference_file"]).expanduser().read_text(encoding="utf-8"))
    value = refs.get("references", refs)[video_id]
    if isinstance(value, list):
        value = value[int(info.get("gt_reference_index", 0))]
    return str(value).strip()


def _prepare_inputs(captioner, sample, prompt, caption, fps):
    messages = captioner._build_messages(sample, prompt)
    chat_prefix = captioner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    else:
        video_metadata = None
    inputs = captioner.processor(
        text=[chat_prefix], images=images, videos=videos, video_metadata=video_metadata,
        return_tensors="pt", do_resize=False, **(video_kwargs or {})
    )
    target_ids = captioner.processor.tokenizer(caption, add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("GT caption tokenization produced no target tokens")
    prefix_length = inputs.input_ids.shape[1]
    target = torch.tensor([target_ids], dtype=inputs.input_ids.dtype)
    inputs["input_ids"] = torch.cat([inputs.input_ids, target], dim=1)
    inputs["attention_mask"] = torch.cat([inputs.attention_mask, torch.ones_like(target)], dim=1)
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat([inputs.mm_token_type_ids, torch.zeros_like(target)], dim=1)
    visual_ids = {int(captioner.model.config.image_token_id), int(captioner.model.config.video_token_id)}
    visual = [i for i, token in enumerate(inputs.input_ids[0].tolist()) if token in visual_ids]
    if not visual:
        raise ValueError("No Qwen visual tokens found in processor input")
    grid_key = "video_grid_thw" if "video_grid_thw" in inputs else "image_grid_thw"
    grid = inputs[grid_key].tolist()
    source_metadata = [
        {"path": path, "timestamp": timestamp, "original_width": Image.open(_local_path(path)).width,
         "original_height": Image.open(_local_path(path)).height}
        for path, timestamp in zip(sample.paths, _timestamps(sample.paths, fps))
    ]
    if len(grid) == 1 and int(grid[0][0]) != len(source_metadata):
        actual_frames = int(grid[0][0])
        if actual_frames < 1 or actual_frames > len(source_metadata):
            raise ValueError("Processor video frame count is outside the source sequence")
        indices = np.linspace(0, len(source_metadata) - 1, actual_frames).round().astype(int)
        frame_metadata = [source_metadata[int(index)] for index in indices]
    else:
        frame_metadata = source_metadata
    blocks, frames = make_blocks(visual, grid, frame_metadata)
    if not blocks:
        raise ValueError("No complete 64x64 visual blocks")
    predictors = list(range(prefix_length - 1, prefix_length + len(target_ids) - 1))
    return inputs, visual, predictors, target_ids, blocks, frames, chat_prefix


def run(raw: dict, config_path: str, batch_size: int) -> dict:
    cfg = load_config(config_path)
    info = raw["visual_information_volume"]
    jobs = cfg.jobs
    if len(jobs) != 1:
        raise ValueError("The volume config must contain exactly one video/frame-sequence job")
    sample = resolve_job(jobs[0])[0]
    video_id = str(info.get("video_id", sample.sample_id))
    caption = _caption(raw, video_id)
    prompt = load_sample_prompt(load_job_prompt(cfg, jobs[0]), sample.sample_id, sample.sidecar_dirs)
    captioner = Qwen3VLCaptioner(cfg)
    device = next(captioner.model.parameters()).device
    inputs, visual, predictors, target_ids, blocks, frames, chat_prefix = _prepare_inputs(
        captioner, sample, prompt, caption, cfg.video.fps
    )
    inputs = inputs.to(device)
    layer = 4
    engine = CachedQwenSuffix(captioner.model, [layer], visual, predictors, target_ids)
    started = time.perf_counter()
    engine.capture(inputs)
    baseline = engine.resume(layer, [[]])[0].numpy()
    values = []
    for start in range(0, len(blocks), batch_size):
        groups = [block["token_indices"] for block in blocks[start:start + batch_size]]
        scores = engine.resume(layer, groups).numpy()
        values.extend((scores.mean(axis=1) - baseline.mean()).tolist())
    elapsed = time.perf_counter() - started
    for block, value in zip(blocks, values):
        block["raw_information"] = float(value)
    shape = (len(frames), max(frame["block_rows"] for frame in frames), max(frame["block_cols"] for frame in frames))
    volume = np.full(shape, np.nan, dtype=np.float32)
    for block in blocks:
        volume[block["frame_idx"], block["block_row"], block["block_col"]] = block["raw_information"]
    if not np.isfinite(volume[~np.isnan(volume)]).all():
        raise RuntimeError("Information volume contains non-finite populated values")
    out_dir = Path(info.get("output_dir", cfg.output.dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "information_volume.npy", volume)
    np.save(out_dir / "positive_information_volume.npy", np.maximum(volume, 0.0))
    with (out_dir / "blocks.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["video_id", "timestamp", "frame_idx", "block_row", "block_col", "token_indices", "visual_token_indices", "raw_information"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block in blocks:
            writer.writerow({"video_id": video_id, **{field: json.dumps(block[field]) if isinstance(block[field], list) else block[field] for field in fields[1:]}})
    dump_json(out_dir / "blocks.json", [{"video_id": video_id, **block} for block in blocks])
    metadata = {
        "video_id": video_id, "timestamps": [frame["timestamp"] for frame in frames], "gt_caption": caption,
        "layer": layer, "block_size": 64, "visual_tokens_per_block": 4,
        "scoring_definition": "mean_j log P(y_j | y_<j, retained block at layer 4) - mean_j log P(y_j | y_<j, no visual tokens at layer 4)",
        "shape": list(volume.shape), "block_count": len(blocks), "batch_size": batch_size,
        "elapsed_seconds": elapsed, "counts": engine.counts, "text_only_baseline_reused": True,
        "raw_information_file": "information_volume.npy",
        "positive_visualization_file": "positive_information_volume.npy",
        "prompt": prompt, "chat_prefix": chat_prefix,
        "frames": frames,
    }
    dump_json(out_dir / "metadata.json", metadata)
    LOG.info("Wrote %s blocks, shape=%s, finite=%s, elapsed=%.2fs", len(blocks), volume.shape, bool(np.isfinite(volume[~np.isnan(volume)]).all()), elapsed)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config).resolve()
    import yaml
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run(raw, str(config_path), args.batch_size)


if __name__ == "__main__":
    main()