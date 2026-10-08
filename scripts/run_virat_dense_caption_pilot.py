#!/usr/bin/env python3
"""Five-video Qwen3-VL full-video dense-caption pilot; no GT dependencies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_inference.config import load_config
from vlm_inference.dense_caption import parse_dense_captions, write_new_json
from vlm_inference.inputs import resolve_job
from vlm_inference.video_io import inspect_video


def fingerprint(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/virat_dense_caption_pilot.yaml"))
    parser.add_argument("--output-dir", help="New destination for a rerun; edited results are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Actual processor inputs for all five videos; no model weights/generation")
    parser.add_argument("--resume", action="store_true", help="Skip completed, configuration-matching videos and preserve human edits")
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 6), help="Process at most five videos")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        cfg = load_config(args.config)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if cfg.mode != "default" or cfg.feasibility is not None:
        parser.error("The dense-caption pilot requires full-video default mode, not feasibility/ROI mode")
    if not cfg.jobs or len(cfg.jobs) > 5 or any(j.type != "video" for j in cfg.jobs):
        parser.error("Provide 1–5 explicit video jobs; directory/dataset-wide inference is disabled")
    if Path(cfg.model.path).name != "Qwen3-VL-30B-A3B-Instruct" or cfg.model.load_in_8bit:
        parser.error("This pilot requires the Qwen3-VL-30B-A3B-Instruct checkpoint without INT8 substitution")
    out = Path(args.output_dir or cfg.output.dir).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    if not cfg.prompt_file:
        parser.error("prompt_file is required")
    prompt_path = Path(cfg.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = cfg.config_dir / prompt_path
    template = prompt_path.read_text(encoding="utf-8")
    # Resolve every input before loading 30B weights.
    prepared_jobs = []
    for job in cfg.jobs[:args.limit]:
        sample = resolve_job(job)[0]
        info = inspect_video(job.path)
        duration = info.frame_count / info.native_fps
        prompt = template.replace("{duration_sec}", f"{duration:.6f}")
        prepared_jobs.append((sample, info, duration, prompt))
    if len({x[1].path.name for x in prepared_jobs}) != len(prepared_jobs):
        parser.error("Duplicate original filenames would collide in per-video outputs")
    settings = {"model": asdict(cfg.model), "generation": asdict(cfg.generation),
                "video": asdict(cfg.video), "prompt": template,
                "videos": [{"filename": info.path.name, "path": str(info.path),
                            "bytes": info.path.stat().st_size, "mtime_ns": info.path.stat().st_mtime_ns}
                           for _, info, _, _ in prepared_jobs]}
    run_id = fingerprint(settings)
    video_order = {v["filename"]: i for i, v in enumerate(settings["videos"])}
    name = "dry_run.json" if args.dry_run else cfg.output.file
    index_path = out / name
    if index_path.exists() and not args.resume:
        parser.error(f"Existing {index_path}; use --resume or a new --output-dir")
    # Separate dry-run records from actual generation metadata.
    area = out / ("dry_run" if args.dry_run else "inference")
    area.mkdir(parents=True, exist_ok=True)
    rows, pending = [], []
    for sample, info, duration, prompt in prepared_jobs:
        base = info.path.name
        meta_path = area / (base + ".metadata.json")
        candidate_path = out / "candidates" / (base + ".json")
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if not args.resume or old.get("run_fingerprint") != run_id:
                parser.error(f"Existing/mismatched {meta_path}; choose a new --output-dir")
            if old["status"] not in {"ok", "partial", "dry_run"}:
                parser.error(f"Previous failed video {base}; retained raw output. Retry in a new --output-dir")
            if not args.dry_run and not candidate_path.exists():
                parser.error(f"Candidate missing for completed {base}; choose a new --output-dir")
            rows.append(old)
        else:
            artifacts = [area / (base + ".raw.txt"), area / (base + ".prompt.txt"), area / (base + ".validation.json")]
            if not args.dry_run:
                artifacts.append(candidate_path)
            if any(p.exists() for p in artifacts):
                parser.error(f"Incomplete previous artifacts for {base}; choose a new --output-dir")
            pending.append((sample, info, duration, prompt))
    # Import the existing wrapper only after validating files/options.
    from vlm_inference.model import Qwen3VLCaptioner, _apply_video_opts, prepare_multimodal_inputs
    if pending:
        if args.dry_run:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(cfg.model.path)
            captioner = None
        else:
            captioner = Qwen3VLCaptioner(cfg)
            processor = captioner.processor
    for sample, info, duration, prompt in pending:
        base = info.path.name
        logging.info("Pilot %s: full duration %.2fs, requested fps=%s", base, duration, cfg.video.fps)
        start = time.perf_counter()
        item = {"type": "video", "video": info.path.as_uri()}
        _apply_video_opts(item, cfg.video, is_frame_list=False)
        messages = [{"role": "user", "content": [item, {"type": "text", "text": prompt}]}]
        # The exact processed tensors measured here are passed to generate, not reloaded.
        prepared = prepare_multimodal_inputs(processor, messages)
        if len(prepared.video_inputs) != 1:
            raise RuntimeError("Expected exactly one full video input")
        actual = prepared.video_inputs[0]
        meta = {"video_filename": base, "video_path": str(info.path), "pilot_role": sample.job_name,
                "duration_sec": duration, "source_width": info.width, "source_height": info.height,
                "model": asdict(cfg.model), "generation": asdict(cfg.generation),
                "sampling_requested": asdict(cfg.video), "sampling_actual": actual,
                "input_tokens": prepared.token_stats, "run_fingerprint": run_id,
                "prompt_file": str(prompt_path.resolve()), "status": "dry_run"}
        with (area / (base + ".prompt.txt")).open("x", encoding="utf-8") as f:
            f.write(prompt)
        logging.info("Actual input: %s frames, %.6f effective fps, %sx%s, %s tokens",
                     actual["sampled_frame_count"], actual["effective_sample_fps"],
                     actual["resized_width"], actual["resized_height"], prepared.token_stats)
        if not args.dry_run:
            result = captioner.generate(sample, prompt, prepared=prepared)
            with (area / (base + ".raw.txt")).open("x", encoding="utf-8") as f:
                f.write(result.text + "\n")
            parsed = parse_dense_captions(result.text, duration)
            if result.hit_token_limit:
                parsed["status"] = "truncated"
                parsed["errors"].append("Generation reached max_new_tokens without an EOS token")
            write_new_json(area / (base + ".validation.json"), parsed)
            meta.update(status=parsed["status"], num_events=len(parsed["events"]),
                        num_rejected_events=len(parsed["rejected_events"]),
                        num_output_tokens=result.num_output_tokens, hit_token_limit=result.hit_token_limit)
            if parsed["status"] in {"ok", "partial"}:
                candidate_path = out / "candidates" / (base + ".json")
                write_new_json(candidate_path, {"video_filename": base, "events": parsed["events"]})
                meta["candidate_file"] = str(candidate_path)
            logging.info("Result %s: %s, %s accepted events", base, meta["status"], len(parsed["events"]))
        meta["elapsed_sec"] = time.perf_counter() - start
        write_new_json(area / (base + ".metadata.json"), meta)
        rows.append(meta)
        # Keep an interrupted pilot's completed records usable without waiting for all five.
        index = {"purpose": "Human-review dense-caption GT candidates; not final GT", "run_fingerprint": run_id,
                 "model": cfg.model.path, "dry_run": args.dry_run,
                 "videos": sorted(rows, key=lambda x: video_order[x["video_filename"]])}
        temp = out / (name + ".tmp")
        temp.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(index_path)
        del prepared
    print(f"{'Processor-only dry run' if args.dry_run else 'Pilot inference'}: {index_path}")
    for row in rows:
        actual = row["sampling_actual"]
        print(f"{row['video_filename']}: {actual['sampled_frame_count']} frames / {actual['effective_sample_fps']:.6f} fps / {row['status']}")
    return int(any(r["status"] not in {"ok", "dry_run"} for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
