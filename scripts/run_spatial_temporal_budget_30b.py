#!/usr/bin/env python3
"""Re-run the 40,960-token spatial-temporal budget experiment on Qwen3-VL-30B-A3B-Instruct.

The 8B runner and experiments/spatial_temporal_budget_40960 are read-only.
Planning, prompts, resize, timestamp encoding, and generation kwargs come from that runner.
One process owns the model. Layers are split across the visible GPUs.

device_map="auto" is not used. On this host, a direct CUDA peer copy of bfloat16
tensors returns corrupted values (float32 copies and CPU staging do not). Accelerate's
hooks move activations with that copy, so greedy decoding collapses. Activations that
cross GPUs are cast to float32, copied, then cast back to bfloat16.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EIGHT = REPO / "experiments" / "spatial_temporal_budget_40960"
OUT = REPO / "experiments" / "spatial_temporal_budget_40960_30b"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

spec = importlib.util.spec_from_file_location("stb8", REPO / "scripts" / "run_spatial_temporal_budget.py")
stb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stb)

import yaml  # noqa: E402

stb.OUT = OUT
stb.CFG = yaml.safe_load((OUT / "configs" / "experiment.yaml").read_text())
log = logging.getLogger("stb30")


def assert_safe_out() -> None:
    out = OUT.resolve()
    eight = EIGHT.resolve()
    if out == eight or eight in out.parents or out.name != "spatial_temporal_budget_40960_30b":
        raise SystemExit(f"refusing to write outside the 30B experiment dir: {out}")


def split_sentences(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1 and lines:
        lines = [x.strip() for x in re.split(r"(?<=[.!?])\s+", lines[0]) if x.strip()]
    return lines


def norm_sentence(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def sentence_stats(text: str) -> dict[str, int]:
    sentences = split_sentences(text)
    seen: set[str] = set()
    dups = 0
    for sentence in sentences:
        key = norm_sentence(sentence)
        if key in seen:
            dups += 1
        seen.add(key)
    return {
        "n_sentences": len(sentences),
        "n_unique_sentences": len(seen),
        "n_duplicate_sentences": dups,
        "repeated_sentence_count": dups,
    }


def _close_seq(a: list[float], b: list[float], tol: float = 1e-3) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def match_against_8b(prepared: dict[str, Any]) -> dict[str, Any]:
    """Compare this request with the finished 8B checkpoint. Does not write the 8B tree."""
    cid = prepared["plan"]["condition_id"]
    ck = json.loads((EIGHT / "checkpoints" / f"{cid}.json").read_text())
    samp = json.loads((EIGHT / "sampling" / f"{cid}.json").read_text())
    issues: list[str] = []
    if prepared["prompt_text"] != ck.get("prompt_text"):
        issues.append("prompt_text")
    for key in ("image_tokens", "video_tokens", "total_visual_tokens"):
        if prepared[key] != ck.get(key):
            issues.append(f"{key}:{prepared[key]}!={ck.get(key)}")
    samp_new = prepared["sampling"]
    samp_old = samp["sampling"]
    if len(samp_new) != len(samp_old):
        issues.append(f"sampling_len:{len(samp_new)}!={len(samp_old)}")
    else:
        for left, right in zip(samp_new, samp_old):
            if left["k"] != right["k"] or abs(left["target"] - right["target"]) > 1e-6:
                issues.append(f"target_time_k{left['k']}")
                break
            if abs(left["source"] - right["source"]) > 1e-3:
                issues.append(f"source_time_k{left['k']}:{left['source']}!={right['source']}")
                break
    for roi_id, stamps in prepared["roi_timestamps"].items():
        old = samp.get("roi_timestamps", {}).get(roi_id)
        if old is None or not _close_seq(stamps, old):
            issues.append(f"roi_timestamps:{roi_id}")
    def geom(plan: dict[str, Any]) -> list[tuple]:
        rows = []
        for roi in plan["rois"]:
            if not roi.get("grid_idx"):
                continue
            rows.append((roi["roi_id"], tuple(roi["bbox_xyxy"]), roi["frames"], tuple(roi["input_resolution"]), tuple(roi.get("video_grid_thw") or [])))
        return rows
    if geom(prepared["plan"]) != geom(ck["plan"]):
        issues.append("roi_geometry_or_grid")
    return {
        "ok": not issues,
        "issues": issues,
        "tokens_8b": ck.get("total_visual_tokens"),
        "tokens_30b": prepared["total_visual_tokens"],
        "prompt_sha16_8b": stb.sha16(ck.get("prompt_text") or ""),
        "prompt_sha16_30b": stb.sha16(prepared["prompt_text"]),
    }


def safe_to(value, device):
    """Move tensors without a direct bfloat16 peer copy.

    A CUDA peer copy of bfloat16 on this host corrupts the values. Float32 peer
    copies and CPU staging do not. Same-device tensors are returned unchanged.
    """
    import torch

    if isinstance(value, tuple):
        return tuple(safe_to(item, device) for item in value)
    if isinstance(value, list):
        return [safe_to(item, device) for item in value]
    if not torch.is_tensor(value):
        return value
    dest = torch.device(device)
    if value.device == dest:
        return value
    if value.dtype == torch.bfloat16 and value.is_cuda and dest.type == "cuda":
        return value.float().to(dest).to(dtype=torch.bfloat16)
    return value.to(dest)


def _install_activation_bridge(layer, device: str) -> None:
    orig = layer.forward

    def wrapped(*args, **kwargs):
        args = tuple(safe_to(arg, device) for arg in args)
        kwargs = {
            key: (val if key == "past_key_values" else safe_to(val, device))
            for key, val in kwargs.items()
        }
        return orig(*args, **kwargs)

    layer.forward = wrapped


def _place_model(model) -> dict[str, int]:
    """Put whole decoder layers on successive GPUs. Vision stays with the first layers."""
    import torch

    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError("BF16 Qwen3-VL-30B-A3B needs at least 2 visible GPUs")
    text = model.model.language_model
    n_layers = len(text.layers)
    bounds = [round(i * n_layers / n) for i in range(n + 1)]
    if bounds[1] < 3:
        raise RuntimeError("the first GPU must hold the DeepStack decoder layers 0-2")
    text.embed_tokens.to("cuda:0")
    text.rotary_emb.to("cuda:0")
    model.model.visual.to("cuda:0")
    placement: dict[str, int] = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
    }
    for gpu in range(n):
        for index in range(bounds[gpu], bounds[gpu + 1]):
            layer = text.layers[index]
            layer.to(f"cuda:{gpu}")
            if gpu != 0:
                _install_activation_bridge(layer, f"cuda:{gpu}")
            placement[f"model.language_model.layers.{index}"] = gpu
    last = f"cuda:{n - 1}"
    text.norm.to(last)
    model.lm_head.to(last)
    placement["model.language_model.norm"] = n - 1
    placement["lm_head"] = n - 1
    off = [name for name, param in model.named_parameters() if param.device.type != "cuda"]
    if off:
        raise RuntimeError("parameters left off GPU: " + ", ".join(off[:8]))
    # DeepStack writes into the first decoder layers. Those layers share cuda:0 with vision.
    model.hf_device_map = placement
    return placement


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        import torch.backends.python_native as pn

        pn.triton.enabled = False
    except Exception:
        pass
    cfg = stb.CFG["model"]
    path = cfg["path"]
    if not (Path(path) / "config.json").is_file():
        raise FileNotFoundError(path)
    processor = AutoProcessor.from_pretrained(path)
    vp = processor.video_processor
    if int(vp.patch_size) != 16 or int(vp.merge_size) != 2 or int(vp.temporal_patch_size) != 2:
        raise RuntimeError(
            f"30B video processor patch/merge/temporal {vp.patch_size}/{vp.merge_size}/{vp.temporal_patch_size} "
            "does not match the 8B factor used to build the plan"
        )
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError("BF16 Qwen3-VL-30B-A3B needs at least 2 visible GPUs")
    experts = cfg.get("experts_implementation", "grouped_mm")
    log.info(
        "loading %s on %d GPUs attn=%s experts=%s (manual layer split, bf16 peer-copy bridge)",
        path, n, cfg["attn_implementation"], experts,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation=cfg["attn_implementation"],
        experts_implementation=experts,
        low_cpu_mem_usage=True,
    )
    model.eval()
    placement = _place_model(model)
    log.info("placement bounds %s", {dev: sum(v == dev for v in placement.values()) for dev in sorted(set(placement.values()))})
    # The checkpoint generation_config has do_sample=true. The 8B runner overrides that per call.
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.0
    return processor, model


def _first_device(model):
    return model.get_input_embeddings().weight.device


def generate_one(model, processor, prepared: dict[str, Any]) -> dict[str, Any]:
    """Greedy argmax with the 8B generation contract.

    model.generate is not used. On this two-GPU split it feeds position ids and a
    full-length multimodal mask into later steps, and those steps no longer match a
    direct forward. The loop below is still do_sample=false, repetition_penalty=1.0,
    max_new_tokens=256: each step takes the argmax and stops on EOS.
    """
    import torch

    torch.manual_seed(int(stb.CFG["generation"]["seed"]))
    device = _first_device(model)
    inputs = prepared["inputs"].to(device)
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.synchronize()
    t0 = __import__("time").perf_counter()
    gen_kwargs = {
        "do_sample": False,
        "repetition_penalty": float(stb.CFG["generation"]["repetition_penalty"]),
        "max_new_tokens": int(stb.CFG["generation"]["max_new_tokens"]),
        "use_cache": True,
    }
    if gen_kwargs["do_sample"] is not False or gen_kwargs["repetition_penalty"] != 1.0 or gen_kwargs["max_new_tokens"] != 256:
        raise RuntimeError(f"generation kwargs drifted from the 8B experiment: {gen_kwargs}")
    eos = set(stb.EOS_IDS)
    produced: list[int] = []
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        cache = out.past_key_values
        for _ in range(gen_kwargs["max_new_tokens"]):
            nxt = int(out.logits[0, -1].argmax())
            produced.append(nxt)
            if nxt in eos:
                break
            step_ids = torch.tensor([[nxt]], device=device, dtype=inputs["input_ids"].dtype)
            out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    torch.cuda.synchronize()
    elapsed = __import__("time").perf_counter() - t0
    per_gpu = [round(torch.cuda.max_memory_allocated(i) / 2**20, 1) for i in range(n_gpu)]
    gen = torch.tensor(produced, device=device)
    text = processor.batch_decode(gen.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    n_out = int(gen.shape[0])
    last = int(gen[-1]) if n_out else None
    hit = n_out >= int(stb.CFG["generation"]["max_new_tokens"]) and last not in eos
    ended_eos = last in eos
    del inputs, out, gen, cache
    return {
        "caption": text,
        "output_token_count": n_out,
        "hit_token_limit": hit,
        "terminated_normally": not hit,
        "ended_with_eos": ended_eos,
        "last_token_id": last,
        "inference_time_sec": round(elapsed, 3),
        "peak_gpu_memory_mib": round(sum(per_gpu), 1),
        "peak_gpu_memory_mib_max": max(per_gpu) if per_gpu else None,
        "peak_gpu_memory_per_gpu_mib": per_gpu,
        "generation_call": gen_kwargs,
        **sentence_stats(text),
    }


stb.load_model = load_model
stb.generate_one = generate_one


def _select(args) -> list[dict[str, Any]]:
    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    wanted = []
    ids = set(args.ids.split(",")) if getattr(args, "ids", "") else None
    for cond in plan_all["conditions"]:
        if ids and cond["condition_id"] not in ids:
            continue
        if args.videos and cond["video_id"] not in args.videos.split(","):
            continue
        if args.spatial and cond["spatial"] not in args.spatial.split(","):
            continue
        if args.fps and cond["fps"] not in [float(x) for x in args.fps.split(",")]:
            continue
        wanted.append(cond)
    return wanted


def cmd_plan(_args) -> None:
    assert_safe_out()
    stb.cmd_plan(_args)
    eight = json.loads((EIGHT / "configs" / "plan.json").read_text())
    new = json.loads((OUT / "configs" / "plan.json").read_text())
    def slim(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return payload["conditions"]
    same = slim(eight) == slim(new)
    report = {"plan_conditions_identical_to_8b": same, "n": len(new["conditions"])}
    if not same:
        report["note"] = "regenerated plan differs; infer will refuse until this is resolved"
    stb.write_json(OUT / "logs" / "plan_diff.json", report)
    print(json.dumps(report))
    if not same:
        raise SystemExit(2)


def cmd_verify(args) -> None:
    """Processor-only check of all selected conditions against the 8B inputs. No generation."""
    assert_safe_out()
    import torch
    from transformers import AutoProcessor

    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    infos = plan_all["videos"]
    path = stb.CFG["model"]["path"]
    processor = AutoProcessor.from_pretrained(path if (Path(path) / "tokenizer.json").is_file() else stb.CFG["model"]["id"])
    vp = processor.video_processor
    ip = processor.image_processor
    header = {
        "video_processor": {"patch_size": int(vp.patch_size), "merge_size": int(vp.merge_size), "temporal_patch_size": int(vp.temporal_patch_size)},
        "image_processor": {"patch_size": int(ip.patch_size), "merge_size": int(ip.merge_size)},
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
    }
    rows = []
    wanted = _select(args) if (args.videos or args.spatial or args.fps or args.ids) else plan_all["conditions"]
    for cond in wanted:
        cid = cond["condition_id"]
        try:
            prepared = stb.prepare_inputs(processor, json.loads(json.dumps(cond)), infos[cond["video_id"]])
            match = match_against_8b(prepared)
            rows.append({"condition_id": cid, "status": "match" if match["ok"] else "mismatch", **match,
                         "image_tokens": prepared["image_tokens"], "video_tokens": prepared["video_tokens"],
                         "max_abs_error_sec": prepared["max_abs_error_sec"]})
            log.info("%s %s tokens=%s", cid, rows[-1]["status"], prepared["total_visual_tokens"])
        except Exception as exc:  # noqa: BLE001
            rows.append({"condition_id": cid, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            log.error("%s verify error: %s", cid, exc)
        finally:
            prepared = None
    bad = [r for r in rows if r["status"] != "match"]
    payload = {"processor": header, "n": len(rows), "n_mismatch": len(bad), "rows": rows}
    stb.write_json(OUT / "logs" / "input_match_8b.json", payload)
    print(f"verify {len(rows)} mismatch {len(bad)} -> {OUT / 'logs' / 'input_match_8b.json'}")
    if bad:
        raise SystemExit(2)


def _resident_mib() -> dict[str, Any]:
    import torch

    per = [round(torch.cuda.memory_allocated(i) / 2**20, 1) for i in range(torch.cuda.device_count())]
    return {"model_resident_mib": round(sum(per), 1), "model_resident_per_gpu_mib": per}


def cmd_preflight(_args) -> None:
    assert_safe_out()
    import torch

    processor, model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_gpu = torch.cuda.device_count()
    devices = []
    for i in range(n_gpu):
        free, total = torch.cuda.mem_get_info(i)
        devices.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "total_mib": round(total / 2**20, 1),
            "free_mib_after_load": round(free / 2**20, 1),
            "allocated_mib": round(torch.cuda.memory_allocated(i) / 2**20, 1),
        })
    from collections import Counter

    payload = {
        "model_class": type(model).__name__,
        "path": stb.CFG["model"]["path"],
        "revision": stb.CFG["model"]["revision"],
        "dtype": str(next(model.parameters()).dtype),
        "n_parameters": n_params,
        "weight_bytes_index": stb.CFG["model"]["weight_bytes"],
        "n_gpus": n_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_map": "manual_layer_split",
        "bf16_peer_copy": "float32_bridge",
        "experts_implementation": stb.CFG["model"].get("experts_implementation", "grouped_mm"),
        "device_map_counts": dict(Counter(str(v) for v in model.hf_device_map.values())),
        "generation_config_after_override": {
            "do_sample": model.generation_config.do_sample,
            "repetition_penalty": model.generation_config.repetition_penalty,
            "eos_token_id": model.generation_config.eos_token_id,
        },
        "effective_generate_kwargs": {"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": 256, "seed": 0},
        "devices": devices,
        "processor_video": {
            "patch_size": int(processor.video_processor.patch_size),
            "merge_size": int(processor.video_processor.merge_size),
            "temporal_patch_size": int(processor.video_processor.temporal_patch_size),
        },
    }
    stb.write_json(OUT / "logs" / "preflight_load.json", payload)
    print(json.dumps({k: payload[k] for k in ("model_class", "dtype", "n_parameters", "n_gpus", "devices", "device_map_counts")}, indent=2))


def cmd_infer(args) -> None:
    assert_safe_out()
    import gc
    import torch

    ckpt_dir, cap_dir, samp_dir = OUT / "checkpoints", OUT / "captions", OUT / "sampling"
    for folder in (ckpt_dir, cap_dir, samp_dir, OUT / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    infos = plan_all["videos"]
    wanted = _select(args)

    def done(cond: dict[str, Any]) -> bool:
        path = ckpt_dir / f"{cond['condition_id']}.json"
        if not path.is_file() or args.force:
            return False
        status = json.loads(path.read_text()).get("status")
        return status == "ok" or (status != "ok" and not args.retry_failed)

    todo = [cond for cond in wanted if not done(cond)]
    log.info("%d selected, %d to run", len(wanted), len(todo))
    if not todo:
        return
    processor, model = load_model()
    resident = _resident_mib()
    log.info("model loaded resident_sum=%.0f MiB per_gpu=%s", resident["model_resident_mib"], resident["model_resident_per_gpu_mib"])
    for cond in todo:
        cid = cond["condition_id"]
        rec: dict[str, Any] = {
            "condition_id": cid,
            "video_id": cond["video_id"],
            "spatial": cond["spatial"],
            "fps": cond["fps"],
            "window": cond["window"],
            "started_at": stb.utc_now(),
            "model_id": stb.CFG["model"]["id"],
            "model_path": stb.CFG["model"]["path"],
            "model_revision": stb.CFG["model"]["revision"],
            "model_dtype": "bfloat16",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **resident,
            "plan": cond,
        }
        prepared = None
        try:
            prepared = stb.prepare_inputs(processor, cond, infos[cond["video_id"]])
            match = match_against_8b(prepared)
            rec["match_8b"] = match
            if not match["ok"]:
                rec.update(status="input_mismatch", error="; ".join(match["issues"]))
                log.error("%s input_mismatch %s", cid, rec["error"])
            else:
                gen = generate_one(model, processor, prepared)
                rec.update(status="ok", **gen)
                rec.update({k: prepared[k] for k in (
                    "image_tokens", "video_tokens", "total_visual_tokens", "prompt_tokens", "text_tokens",
                    "prep_sec", "max_abs_error_sec", "timestamp_markers_verified", "roi_timestamps",
                    "prompt_text", "decoded_prompt_head",
                )})
                rec["plan"] = prepared["plan"]
                stb.write_json(samp_dir / f"{cid}.json", {
                    "condition_id": cid,
                    "sampling": prepared["sampling"],
                    "roi_timestamps": prepared["roi_timestamps"],
                    "match_8b": match,
                })
                (cap_dir / f"{cid}.txt").write_text(gen["caption"] + "\n", encoding="utf-8")
                log.info(
                    "%s ok tok=%d out=%d eos=%s hit=%s repeats=%d %.1fs peak_sum=%.0fMiB",
                    cid, prepared["total_visual_tokens"], gen["output_token_count"], gen["ended_with_eos"],
                    gen["hit_token_limit"], gen["repeated_sentence_count"], gen["inference_time_sec"],
                    gen["peak_gpu_memory_mib"],
                )
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            rec.update(status="oom" if stb.is_oom(exc) else "error", error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-3000:])
            log.error("%s %s: %s", cid, rec["status"], str(exc)[:300])
        rec["finished_at"] = stb.utc_now()
        stb.write_json(ckpt_dir / f"{cid}.json", rec)
        del prepared
        gc.collect()
        torch.cuda.empty_cache()


def cmd_aggregate(_args) -> None:
    assert_safe_out()
    plan_all = json.loads((OUT / "configs" / "plan.json").read_text())
    rows = []
    for cond in plan_all["conditions"]:
        path = OUT / "checkpoints" / f"{cond['condition_id']}.json"
        rec = json.loads(path.read_text()) if path.is_file() else {"status": "not_run", "plan": cond}
        plan = rec.get("plan", cond)
        used = [roi for roi in plan["rois"] if roi["grid_idx"]]
        match = rec.get("match_8b") or {}
        rows.append({
            "condition_id": cond["condition_id"],
            "video_id": cond["video_id"],
            "window_start": cond["window"][0],
            "window_end": cond["window"][1],
            "spatial": cond["spatial"],
            "fps": cond["fps"],
            "status": rec.get("status", "not_run"),
            "error": rec.get("error", ""),
            "native_resolution": "x".join(map(str, cond["native_resolution"])),
            "roi_count": len(used),
            "rois": json.dumps([
                {
                    "id": roi["roi_id"],
                    "bbox_xyxy": roi["bbox_xyxy"],
                    "tube_time_range": [round(v, 3) for v in roi["time_range"]],
                    "sampled_range": roi.get("source_time_first_last"),
                    "frames": roi["frames"],
                    "roi_native_size": roi["roi_native_size"],
                    "input_resolution": roi["input_resolution"],
                }
                for roi in used
            ]),
            "input_frames_per_video": "|".join(str(roi["frames"]) for roi in used),
            "model_frames_with_temporal_padding": "|".join(str(roi["model_frames_with_temporal_padding"]) for roi in used),
            "input_resolution": "|".join(f"{roi['input_resolution'][0]}x{roi['input_resolution'][1]}" for roi in used),
            "global_image": (
                json.dumps({"t": plan["global_image"].get("source_time_sec", plan["global_image"]["time_target"]), "input": plan["global_image"]["input_resolution"]})
                if plan.get("global_image") else ""
            ),
            "planned_total_tokens": plan["planned_total_visual_tokens"],
            "image_tokens": rec.get("image_tokens"),
            "video_tokens": rec.get("video_tokens"),
            "total_visual_tokens": rec.get("total_visual_tokens"),
            "tokens_8b": match.get("tokens_8b"),
            "tokens_match_8b": match.get("ok"),
            "text_tokens": rec.get("text_tokens"),
            "binding_constraint": plan["allocation"]["binding"],
            "model_resident_mib": rec.get("model_resident_mib"),
            "peak_gpu_memory_mib": rec.get("peak_gpu_memory_mib"),
            "peak_gpu_memory_mib_max": rec.get("peak_gpu_memory_mib_max"),
            "peak_gpu_memory_per_gpu_mib": json.dumps(rec.get("peak_gpu_memory_per_gpu_mib")),
            "inference_time_sec": rec.get("inference_time_sec"),
            "prep_sec": round(rec["prep_sec"], 2) if rec.get("prep_sec") else None,
            "output_tokens": rec.get("output_token_count"),
            "terminated_normally": rec.get("terminated_normally"),
            "ended_with_eos": rec.get("ended_with_eos"),
            "hit_token_limit": rec.get("hit_token_limit"),
            "n_sentences": rec.get("n_sentences"),
            "n_unique_sentences": rec.get("n_unique_sentences"),
            "n_duplicate_sentences": rec.get("n_duplicate_sentences"),
            "repeated_sentence_count": rec.get("repeated_sentence_count"),
            "caption": rec.get("caption", ""),
        })
    stb.write_json(OUT / "results.json", rows)
    import csv

    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    parts = ["# Raw captions\n"]
    for row in rows:
        eos = row.get("ended_with_eos")
        parts.append(
            f"\n## {row['condition_id']}  (status {row['status']}, tokens {row['total_visual_tokens']}, "
            f"out {row['output_tokens']}, EOS={eos}, repeats {row['repeated_sentence_count']}, hit_limit={row['hit_token_limit']})\n\n```\n{row['caption']}\n```\n"
        )
    (OUT / "captions.md").write_text("".join(parts), encoding="utf-8")
    from collections import Counter

    print(Counter(row["status"] for row in rows), "->", OUT / "results.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("preflight")
    verify = sub.add_parser("verify-inputs")
    infer = sub.add_parser("infer")
    sub.add_parser("aggregate")
    for command in (verify, infer):
        command.add_argument("--videos")
        command.add_argument("--spatial")
        command.add_argument("--fps")
        command.add_argument("--ids", default="")
    infer.add_argument("--force", action="store_true")
    infer.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    {
        "plan": cmd_plan,
        "verify-inputs": cmd_verify,
        "preflight": cmd_preflight,
        "infer": cmd_infer,
        "aggregate": cmd_aggregate,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
