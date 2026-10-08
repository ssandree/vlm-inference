#!/usr/bin/env python3
"""Caption low-camera-motion clips: ROI-start Global + centroid-displacement ROI.

Reads the motion-detector export list, writes a feasibility YAML, then runs
Qwen3-VL captioning (or --dry-run).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_inference.config import load_config, load_model_catalog, resolve_model_spec
from vlm_inference.runner import run_inference

logger = logging.getLogger("run_low_camera_motion_captioning")

METHOD = "global_roi_start_only_roi_centroid_2fps"
PROMPT_FILE = "prompts/frames_roi_start_centroid.txt"
DEFAULT_VIDEOS = Path(
    "/home/ssandree/Motion-guided-ROI/motion-detector/outputs/harrypotter_low_camera_motion/vlm_ready_videos.txt"
)
VLM_DATA = Path("/data/datasets/low_camera_motion_vlm")
TRACKS_ROOT = Path(
    "/home/ssandree/Motion-guided-ROI/motion-detector/outputs/harrypotter_low_camera_motion"
)
MODEL_PATH = Path("/data/models/Qwen3-VL-8B-Instruct-BF16")
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "low_camera_motion_start_centroid"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument(
        "--config-out",
        type=Path,
        default=ROOT / "configs" / "low_camera_motion_start_centroid.yaml",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    p.add_argument(
        "--model",
        default="8b",
        help="VLM preset (8b, 30b, 8b-fp8, 30b-fp8) or a local weights path.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-vlm-inputs", action="store_true")
    p.add_argument("--max-videos", type=int, default=None)
    return p.parse_args()


def load_video_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"video list not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def build_config_payload(
    videos: list[str],
    *,
    output_dir: Path,
    save_vlm_inputs: bool,
    model_spec: str = "8b",
) -> dict:
    model_cfg = resolve_model_spec(model_spec)
    if model_cfg.name in load_model_catalog():
        model_block: str | dict[str, str] = model_cfg.name
    else:
        model_block = {
            "path": str(model_cfg.path),
            "dtype": model_cfg.dtype,
            "device_map": model_cfg.device_map,
            "attn_implementation": str(model_cfg.attn_implementation or "sdpa"),
        }
        if model_cfg.name:
            model_block["name"] = model_cfg.name
    return {
        "model": model_block,
        "generation": {
            "max_new_tokens": 384,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 8,
        },
        "video": {
            "fps": 2.0,
            "min_pixels": 4096,
            "max_pixels": 262144,
            "total_pixels": 20971520,
        },
        "image": {"min_pixels": 4096, "max_pixels": 262144},
        "mode": "experiment",
        "experiment": {
            "global_pattern": r"global_frame_(?P<frame>\d+)_t(?P<time>\d+(?:\.\d+)?)",
            "roi_pattern": (
                r"tube(?P<roi>\d+)_frame_(?P<frame>\d+)_t(?P<time>\d+(?:\.\d+)?)"
                r"(?:_fps_\d+)?"
            ),
            "global_label": "[Global] t={time}s",
            "roi_label": "[ROI{roi}] t={time}s",
            "input_order": "global_anchor",
            "roi_frames": "default",
            "global_insert": "start_only",
            "global_insert_source": "fps_1",
            "global_insert_dedup_eps": 0.5,
            "roi_sampling": None,
            "roi_sampling_tau": 8.0,
            "roi_sampling_overwrite": False,
            "save_vlm_inputs": bool(save_vlm_inputs),
            "centroid_motion": {
                "centroid_displacement_threshold": 0.02,
                "pad_sec": 0.5,
                "base_fps": 1.0,
                "high_fps": 2.0,
            },
        },
        "prompt_file": PROMPT_FILE,
        "output": {
            "dir": str(output_dir),
            "file": "feasibility_results_low_camera_motion_start_centroid.json",
        },
        "experiments": [METHOD],
        "feasibility": {
            "method_prompts": {METHOD: PROMPT_FILE},
            "videos": videos,
            "datasets": {
                "video_root": str(VLM_DATA / "videos"),
                "fps_0_25": str(VLM_DATA / "fps_sampling_frames" / "fps0_25"),
                "fps_1": str(VLM_DATA / "fps_sampling_frames" / "fps_1"),
                "fps_2": str(VLM_DATA / "fps_sampling_frames" / "fps_2"),
                "proposed_roi": str(VLM_DATA / "proposed_ROI"),
                "proposed_roi_w_fps": str(VLM_DATA / "proposed_ROI_w_fps"),
                "single_motion_roi": str(VLM_DATA / "single_motion_ROI"),
                "single_motion_roi_w_fps": str(VLM_DATA / "single_motion_ROI_w_fps"),
            },
            "tracks_root": str(TRACKS_ROOT),
            "jpeg_quality": 95,
            "cache_file": str(output_dir / "feasibility_cache.json"),
            "resume": True,
        },
    }


def write_captions_md(results_path: Path, output_dir: Path, prompt: str) -> Path:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = [r for r in (data.get("results") or []) if isinstance(r, dict)]
    ok = [r for r in rows if r.get("caption") and not r.get("error")]
    failed = [r for r in rows if r.get("error") or not r.get("caption")]
    model_field = data.get("model")
    if isinstance(model_field, dict):
        model_name = Path(str(model_field.get("path") or MODEL_PATH)).name
    else:
        model_name = Path(str(model_field or MODEL_PATH)).name
    lines = [
        "# Low-camera-motion captions",
        "",
        "ROI-start Global (`start_only`) + centroid-displacement ROI sampling "
        "(1 fps default, 2 fps around large centroid jumps).",
        f"Model: {model_name}. "
        f"Prompt: `{PROMPT_FILE}`.",
        "",
        f"Succeeded: {len(ok)}  |  Failed / empty: {len(failed)}  |  "
        f"Total jobs: {len(rows)}",
        "",
        f"[Results JSON]({results_path.name}) · [Run config](config.yaml)",
        "",
        "## Prompt",
        "",
        "```",
        prompt.strip(),
        "```",
        "",
    ]
    by_video = {}
    for row in rows:
        by_video.setdefault(str(row.get("video_id") or ""), []).append(row)
    for video_id, video_rows in by_video.items():
        row = video_rows[0]
        caption = (row.get("caption") or "").strip()
        err = row.get("error")
        n_global = row.get("num_global_images", row.get("num_global"))
        n_roi = row.get("num_roi_images", row.get("num_roi"))
        n_tubes = row.get("num_roi_tubes", row.get("num_tubes"))
        labels = row.get("roi_sampling_labels") or {}
        densify = []
        if isinstance(labels, dict):
            for roi_id, label in sorted(labels.items(), key=lambda kv: int(kv[0])):
                text = str(label)
                if "2fps" in text.lower() or "high" in text.lower():
                    densify.append(f"ROI {roi_id}: {text}")
        lines += [
            f"## {video_id}",
            "",
            f"Global frames: {n_global}  |  ROI frames: {n_roi}  |  ROI tubes: {n_tubes}",
            "",
        ]
        if densify:
            lines += ["**2fps / densify notes:** " + "; ".join(densify), ""]
        if err:
            lines += [f"**Error:** {err}", ""]
        elif caption:
            lines += ["**Generated caption**", "", caption, ""]
        else:
            lines += ["_(empty caption)_", ""]
    out = output_dir / "captions.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    model_cfg = resolve_model_spec(args.model)
    model_path = Path(model_cfg.path)
    if args.video_id:
        videos = list(args.video_id)
    else:
        videos = load_video_ids(args.videos)
    if args.max_videos is not None:
        videos = videos[: int(args.max_videos)]
    if not videos:
        raise SystemExit("no videos to caption")
    if not model_path.exists():
        raise SystemExit(f"VLM weights missing: {model_path}")

    output_dir = args.output_dir.resolve()
    if args.output_dir == DEFAULT_OUTPUT_DIR and model_cfg.name and model_cfg.name != "8b":
        output_dir = (ROOT / "outputs" / f"low_camera_motion_start_centroid_{model_cfg.name}").resolve()
        logger.info("output-dir defaulted to %s for model=%s", output_dir, model_cfg.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_config_payload(
        videos,
        output_dir=output_dir,
        save_vlm_inputs=bool(args.save_vlm_inputs),
        model_spec=args.model,
    )
    config_out = args.config_out.resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.copy2(config_out, output_dir / "config.yaml")
    logger.info("wrote config %s (%d videos)", config_out, len(videos))

    cfg = load_config(str(config_out))
    if args.no_resume:
        cfg.feasibility.resume = False  # type: ignore[union-attr]
    prompt = (ROOT / PROMPT_FILE).read_text(encoding="utf-8").strip()
    results_path = run_inference(cfg, dry_run=bool(args.dry_run), resume=cfg.feasibility.resume)
    captions_path = write_captions_md(Path(results_path), output_dir, prompt)
    logger.info("results: %s", results_path)
    logger.info("captions: %s", captions_path)
    print(f"results: {results_path}")
    print(f"captions: {captions_path}")
    print(f"videos: {len(videos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
