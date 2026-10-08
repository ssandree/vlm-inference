"""Multi-job inference runner with JSON output."""

from __future__ import annotations

import json
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .cache import (
    cache_reusable,
    ingest_results_json,
    load_cache,
    resolve_cache_path,
    save_cache,
)
from .config import AppConfig
from .feasibility import (
    alias_records_for,
    build_result_record,
    cache_key_payload,
    collect_image_stats,
    collect_method_inputs,
    dry_run_row,
    executable_methods,
    make_cache_key,
    print_dry_run_table,
)
from .inputs import Sample, resolve_job
from .per_tube_merge import DEFAULT_MERGE_PROMPT_FILE, PER_TUBE_MERGE_METHOD, run_per_tube_merge, tube_passes_as_dicts
from .prompts import load_job_prompt, load_sample_prompt
from .single_motion import prepare_single_motion_cache

logger = logging.getLogger(__name__)


def _timestamped_output_path(path: Path, when: datetime | None = None) -> Path:
    """Insert MMDDHHmmss before the suffix: results.json -> results_0803183345.json.

    Never returns an existing path: if the stamp collides, append _2, _3, ...
    So each run always gets a brand-new results file.
    """
    when = when or datetime.now()
    stamp = when.strftime("%m%d%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        alt = path.with_name(f"{path.stem}_{stamp}_{n}{path.suffix}")
        if not alt.exists():
            return alt
        n += 1


def _allocate_results_path(cfg: AppConfig) -> Path:
    """Create a fresh results path for this process (never reuse an old JSON)."""
    out_path = cfg.output.path
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path = _timestamped_output_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("New results file for this run: %s", out_path)
    print(f"New results file: {out_path}", flush=True)
    return out_path


def _result_record(
    *,
    sample: Sample,
    prompt: str,
    model_path: str,
    response: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "job": sample.job_name,
        "sample_id": sample.sample_id,
        "input_type": sample.input_type,
        "paths": sample.paths,
        "prompt": prompt,
        "response": response,
        "model": model_path,
        "error": error,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _caption_text(result: Any) -> str:
    return result.text if hasattr(result, "text") else result


def run_inference(
    cfg: AppConfig,
    *,
    dry_run: bool = False,
    resume: bool | None = None,
) -> Path:
    if cfg.feasibility is not None:
        return run_feasibility(
            cfg,
            dry_run=dry_run,
            resume=resume,
        )
    if dry_run:
        raise ValueError(
            "--dry-run requires a feasibility config "
            "(see configs/feasibility.yaml)"
        )
    return _run_legacy_jobs(cfg)


def _run_legacy_jobs(cfg: AppConfig) -> Path:
    out_path = _allocate_results_path(cfg)

    logger.info("Loading model from %s", cfg.model.path)
    logger.info("Inference mode: %s", cfg.mode)
    from .model import Qwen3VLCaptioner

    captioner = Qwen3VLCaptioner(cfg)

    results: list[dict[str, Any]] = []

    for job in cfg.jobs:
        logger.info("Resolving job '%s' (type=%s)", job.name, job.type)
        try:
            samples = resolve_job(job)
            job_prompt = load_job_prompt(cfg, job)
        except Exception as exc:
            logger.exception("Failed to prepare job '%s'", job.name)
            results.append(
                {
                    "job": job.name,
                    "sample_id": None,
                    "input_type": job.type,
                    "paths": [],
                    "prompt": None,
                    "response": None,
                    "model": cfg.model.path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_json(out_path, results)
            continue

        logger.info("Job '%s': %d sample(s)", job.name, len(samples))
        for i, sample in enumerate(samples, start=1):
            prompt = load_sample_prompt(
                job_prompt, sample.sample_id, sample.sidecar_dirs
            )
            logger.info(
                "[%s] (%d/%d) %s",
                job.name,
                i,
                len(samples),
                sample.sample_id,
            )
            try:
                generated = captioner.generate(sample, prompt)
                results.append(
                    _result_record(
                        sample=sample,
                        prompt=prompt,
                        model_path=cfg.model.path,
                        response=_caption_text(generated),
                    )
                )
            except Exception as exc:
                logger.error(
                    "Inference failed for %s/%s: %s\n%s",
                    job.name,
                    sample.sample_id,
                    exc,
                    traceback.format_exc(),
                )
                results.append(
                    _result_record(
                        sample=sample,
                        prompt=prompt,
                        model_path=cfg.model.path,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        _write_json(out_path, results)
        logger.info("Flushed results -> %s", out_path)

    logger.info("Done. Wrote %d record(s) to %s", len(results), out_path)
    return out_path


def _feasibility_prompt(cfg: AppConfig, method: str | None = None) -> str:
    from .prompts import _read_text, resolve_path

    feas = cfg.feasibility
    if method and feas is not None:
        prompt_file = feas.method_prompts.get(method)
        if prompt_file:
            return _read_text(resolve_path(cfg.config_dir, prompt_file))

    if cfg.prompt:
        return cfg.prompt.strip()
    if cfg.prompt_file:
        return _read_text(resolve_path(cfg.config_dir, cfg.prompt_file))
    raise ValueError("feasibility config requires a global prompt / prompt_file")


def _feasibility_prompts(cfg: AppConfig) -> dict[str, str]:
    feas = cfg.feasibility
    assert feas is not None
    return {method: _feasibility_prompt(cfg, method) for method in feas.methods}


def _output_payload(
    *,
    cfg: AppConfig,
    prompt: str | dict[str, str],
    results: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    feas = cfg.feasibility
    assert feas is not None
    return {
        "experiment": "virat9_feasibility",
        "dry_run": dry_run,
        "prompt": prompt,
        "model": cfg.model.path,
        "methods": list(feas.methods),
        "aliases": dict(feas.aliases),
        "videos": list(feas.videos),
        "input_order": cfg.experiment.input_order,
        "roi_frames": cfg.experiment.roi_frames,
        "global_insert": cfg.experiment.global_insert,
        "global_insert_source": cfg.experiment.global_insert_source,
        "global_scale": cfg.experiment.global_scale,
        "results": results,
    }


def _dump_method_input_order(
    *,
    cfg: AppConfig,
    video_id: str,
    method: str,
    inputs,
) -> Path | None:
    """Write input_order dump for one video/method before VLM call."""
    if inputs.errors:
        return None
    order = cfg.experiment.input_order
    if inputs.bundle is not None:
        from .metadata_format import format_input_order_dump_from_bundle

        text = format_input_order_dump_from_bundle(inputs.bundle, input_order=order)
    elif inputs.frames:
        from .experiment import format_input_order_dump

        text = format_input_order_dump(inputs.frames, input_order=order)
    else:
        return None

    out_dir = Path(cfg.output.dir) / "input_order"
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = float(cfg.experiment.global_scale)
    scale_tag = ""
    if abs(scale - 1.0) > 1e-12:
        from .global_resolution import format_scale

        scale_tag = f"__gs{format_scale(scale)}"
    path = out_dir / f"{video_id}__{method}__{order}{scale_tag}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _maybe_export_vlm_inputs(
    *,
    cfg: AppConfig,
    video_id: str,
    method: str,
    inputs,
) -> Path | None:
    if not cfg.experiment.save_vlm_inputs:
        return None
    from .export_vlm_inputs import export_vlm_inputs

    return export_vlm_inputs(
        cfg=cfg,
        video_id=video_id,
        method=method,
        inputs=inputs,
        input_order=cfg.experiment.input_order,
        plan=getattr(inputs, "global_insert_plan", None),
        mode=getattr(inputs, "roi_repr_mode", None)
        or getattr(inputs, "global_insert_mode", None),
    )


def run_feasibility(
    cfg: AppConfig,
    *,
    dry_run: bool = False,
    resume: bool | None = None,
) -> Path:
    feas = cfg.feasibility
    if feas is None:
        raise ValueError("feasibility config is missing")

    prompts = _feasibility_prompts(cfg)
    methods = executable_methods(feas)
    needs_single = any(
        m
        in (
            "global_0.25fps_single_motion_roi",
            "global_first_0.25fps_single_motion_roi",
            "single_motion_roi_original",
        )
        for m in methods
    )

    if needs_single and not dry_run:
        logger.info(
            "Preparing single-motion ROI under %s (roi_frames=%s)",
            feas.datasets.single_motion_roi,
            cfg.experiment.roi_frames,
        )
        prepare_single_motion_cache(
            feas,
            overwrite=(cfg.experiment.roi_frames == "fps_1"),
            roi_frames=cfg.experiment.roi_frames,
        )

    from .motion_adaptive import (
        ROI_SAMPLING_METHODS,
        prepare_motion_adaptive_sampling,
        sampling_outputs_ready,
    )

    needs_sampling = cfg.experiment.roi_sampling is not None or any(
        m in ROI_SAMPLING_METHODS for m in methods
    )
    if needs_sampling:
        overwrite = bool(cfg.experiment.roi_sampling_overwrite)
        if overwrite or not sampling_outputs_ready(cfg):
            logger.info(
                "Preparing motion-adaptive ROI sampling under %s", cfg.output.dir
            )
            prepare_motion_adaptive_sampling(cfg, overwrite=overwrite)
        else:
            logger.info(
                "Reusing existing motion-adaptive ROI sampling under %s",
                cfg.output.dir,
            )

    from .roi_union import ROI_REPR_METHODS, prepare_roi_repr

    needs_repr = cfg.experiment.roi_repr is not None or any(
        m in ROI_REPR_METHODS for m in methods
    )
    if needs_repr:
        logger.info("Preparing Separate/Union ROI representation under %s", cfg.output.dir)
        prepare_roi_repr(cfg, overwrite=bool(cfg.experiment.roi_repr_overwrite))

    # Allocate AFTER prepare so the filename matches when this run actually starts
    # writing results — and never reuse an existing JSON from a prior run.
    out_path = _allocate_results_path(cfg)

    if dry_run:
        rows: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for video_id in feas.videos:
            for method in methods:
                prompt = prompts[method]
                inputs = collect_method_inputs(
                    video_id, method, cfg, allow_missing_inputs=True
                )
                dump_path = _dump_method_input_order(
                    cfg=cfg, video_id=video_id, method=method, inputs=inputs
                )
                if dump_path is not None:
                    logger.info("Wrote input order dump: %s", dump_path)
                export_dir = _maybe_export_vlm_inputs(
                    cfg=cfg, video_id=video_id, method=method, inputs=inputs
                )
                if export_dir is not None:
                    logger.info("Wrote VLM input frames: %s", export_dir)
                row = dry_run_row(inputs)
                rows.append(row)
                image_stats = None
                if not inputs.errors and inputs.frames:
                    try:
                        image_stats = collect_image_stats(inputs)
                    except Exception as exc:  # noqa: BLE001
                        inputs.warnings.append(f"image stats failed: {exc}")
                payload = cache_key_payload(
                    video_id=video_id, inputs=inputs, prompt=prompt, cfg=cfg
                )
                key = make_cache_key(payload) if not inputs.errors else ""
                record = build_result_record(
                    inputs=inputs,
                    prompt=prompt,
                    cfg=cfg,
                    image_stats=image_stats,
                    cache_key=key,
                    error="; ".join(inputs.errors) or None,
                )
                results.append(record)
                for alias in alias_records_for(method, feas):
                    alias_inputs = collect_method_inputs(
                        video_id, method, cfg, allow_missing_inputs=True
                    )
                    alias_inputs.method = alias
                    results.append(
                        build_result_record(
                            inputs=alias_inputs,
                            prompt=prompt,
                            cfg=cfg,
                            image_stats=image_stats,
                            cache_key=key,
                            alias_of=method,
                            error="; ".join(inputs.errors) or None,
                        )
                    )
        print_dry_run_table(rows)
        n_err = sum(1 for r in rows if r["errors"])
        n_warn = sum(1 for r in rows if r["warnings"])
        print()
        print(
            f"dry-run videos={len(feas.videos)} methods={len(methods)} "
            f"jobs={len(rows)} errors={n_err} warnings={n_warn}"
        )
        _write_json(out_path, _output_payload(cfg=cfg, prompt=prompts, results=results, dry_run=True))
        logger.info("Dry-run plan written to %s", out_path)
        return out_path

    cache_path = resolve_cache_path(feas.cache_file)
    use_resume = feas.resume if resume is None else resume
    cache: dict[str, dict[str, Any]] = load_cache(cache_path) if use_resume else {}
    if use_resume:
        n_prev = ingest_results_json(out_path, cache)
        out_dir = out_path.parent
        for prev in sorted(out_dir.glob(f"{cfg.output.path.stem}_*.json")):
            ingest_results_json(prev, cache)
        if n_prev:
            logger.info("Ingested %d record(s) from current output path", n_prev)
        logger.info("Inference cache entries: %d (%s)", len(cache), cache_path)

    logger.info("Loading model from %s", cfg.model.path)
    logger.info("Inference mode: %s", cfg.mode)
    from .model import Qwen3VLCaptioner

    captioner = Qwen3VLCaptioner(cfg)

    results: list[dict[str, Any]] = []
    run_cache: dict[str, dict[str, Any]] = {}

    total = len(feas.videos) * len(methods)
    done = 0
    for video_id in feas.videos:
        for method in methods:
            done += 1
            prompt = prompts[method]
            logger.info("[%d/%d] %s / %s", done, total, video_id, method)
            try:
                inputs = collect_method_inputs(video_id, method, cfg)
                if inputs.errors:
                    raise RuntimeError("; ".join(inputs.errors))
                dump_path = _dump_method_input_order(
                    cfg=cfg, video_id=video_id, method=method, inputs=inputs
                )
                if dump_path is not None:
                    logger.info("Wrote input order dump: %s", dump_path)
                export_dir = _maybe_export_vlm_inputs(
                    cfg=cfg, video_id=video_id, method=method, inputs=inputs
                )
                if export_dir is not None:
                    logger.info("Wrote VLM input frames: %s", export_dir)
                image_stats = collect_image_stats(inputs)
                payload = cache_key_payload(
                    video_id=video_id, inputs=inputs, prompt=prompt, cfg=cfg
                )
                key = make_cache_key(payload)
                reused = run_cache.get(key) or (cache.get(key) if use_resume else None)
                token_stats = None
                caption = None
                roi_wise = None
                cache_hit = False
                if method != PER_TUBE_MERGE_METHOD and cache_reusable(reused):
                    caption = reused.get("caption") or reused.get("response")
                    token_stats = {
                        k: reused[k]
                        for k in (
                            "num_visual_tokens",
                            "num_text_tokens",
                            "num_total_input_tokens",
                        )
                        if k in reused
                    } or None
                    cache_hit = True
                    if reused.get("total_latency_s") is not None:
                        roi_wise = {
                            "vlm_call_count": 1,
                            "total_latency_s": float(reused["total_latency_s"]),
                            "cache_hit": True,
                        }
                    logger.info("  cache hit %s", key[:12])
                else:
                    sample = Sample(
                        job_name=f"{video_id}__{method}",
                        sample_id=video_id,
                        input_type="frames",
                        paths=inputs.input_paths,
                        sidecar_dirs=[],
                        frame_width=inputs.frame_width,
                        frame_height=inputs.frame_height,
                        roi_bboxes_px=inputs.roi_bboxes_px,
                        bundle=inputs.bundle,
                        bundle_dir=(
                            Path(inputs.bundle_dir) if inputs.bundle_dir else None
                        ),
                    )
                    if method == PER_TUBE_MERGE_METHOD:
                        merge_path = (cfg.config_dir / DEFAULT_MERGE_PROMPT_FILE).resolve()
                        merge_template = merge_path.read_text(encoding="utf-8")
                        merged = run_per_tube_merge(
                            captioner=captioner,
                            sample=sample,
                            tube_prompt=prompt,
                            merge_prompt_template=merge_template,
                            cfg=cfg,
                            method=method,
                        )
                        caption = merged.caption
                        token_stats = merged.token_stats
                        passes = tube_passes_as_dicts(merged.tube_passes)
                        visual = sum(
                            int((p.get("token_stats") or {}).get("num_visual_tokens") or 0)
                            for p in passes
                        )
                        text_tokens = sum(
                            int((p.get("token_stats") or {}).get("num_text_tokens") or 0)
                            for p in passes
                        ) + int((merged.merge_token_stats or {}).get("num_text_tokens") or 0)
                        roi_wise = {
                            "intermediate_event_lists": passes,
                            "merge_prompt": merged.merge_prompt,
                            "merge_latency_s": merged.merge_latency_s,
                            "total_latency_s": merged.total_latency_s,
                            "vlm_call_count": len(passes) + 1,
                            "total_input_images_across_calls": sum(
                                int(p["input_image_count"]) for p in passes
                            ),
                            "total_visual_tokens_across_roi_calls": visual,
                            "total_text_input_tokens_across_calls": text_tokens,
                        }
                    else:
                        started = time.perf_counter()
                        generated = captioner.generate(sample, prompt)
                        caption = generated.text
                        token_stats = generated.token_stats() or None
                        roi_wise = {
                            "vlm_call_count": 1,
                            "total_latency_s": time.perf_counter() - started,
                        }
                        cache[key] = {
                            "caption": caption,
                            "prompt": prompt,
                            "cache_key": key,
                            "total_latency_s": (roi_wise or {}).get("total_latency_s"),
                            **(token_stats or {}),
                        }
                        save_cache(cache_path, cache)
                run_cache[key] = {
                    "caption": caption,
                    "prompt": prompt,
                    "cache_key": key,
                    **(token_stats or {}),
                }
                record = build_result_record(
                    inputs=inputs,
                    prompt=prompt,
                    cfg=cfg,
                    image_stats=image_stats,
                    caption=caption,
                    cache_key=key,
                    cache_hit=cache_hit,
                    token_stats=token_stats,
                )
                results.append(record)
                if roi_wise is not None:
                    record["roi_wise"] = roi_wise
                for alias in alias_records_for(method, feas):
                    alias_inputs = inputs
                    alias_copy = build_result_record(
                        inputs=alias_inputs,
                        prompt=prompt,
                        cfg=cfg,
                        image_stats=image_stats,
                        caption=caption,
                        cache_key=key,
                        cache_hit=True,
                        alias_of=method,
                        token_stats=token_stats,
                    )
                    alias_copy["method"] = alias
                    alias_copy["job"] = f"{video_id}__{alias}"
                    results.append(alias_copy)
            except Exception as exc:
                logger.error(
                    "Feasibility job failed for %s/%s: %s\n%s",
                    video_id,
                    method,
                    exc,
                    traceback.format_exc(),
                )
                oom = "out of memory" in str(exc).lower() or type(exc).__name__ in {
                    "OutOfMemoryError",
                    "CUDAOutOfMemoryError",
                }
                if oom:
                    logger.error(
                        "OOM recorded for %s/%s; clearing CUDA cache and continuing",
                        video_id,
                        method,
                    )
                    try:
                        import gc

                        import torch

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                results.append(
                    {
                        "video_id": video_id,
                        "method": method,
                        "job": f"{video_id}__{method}",
                        "caption": None,
                        "prompt": prompt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "oom": oom,
                        "model": cfg.model.path,
                        "global_scale": cfg.experiment.global_scale,
                    }
                )

            _write_json(
                out_path,
                _output_payload(cfg=cfg, prompt=prompts, results=results, dry_run=False),
            )

    logger.info("Done. Wrote %d record(s) to %s", len(results), out_path)
    return out_path
