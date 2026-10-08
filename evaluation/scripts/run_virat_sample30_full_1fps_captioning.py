#!/usr/bin/env python3
"""sample30 30B Full-1fps captioning. Does not rewrite existing sample30 ROI results."""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

EVAL_ROOT = Path(__file__).resolve().parents[1]
VLM_INF_SRC = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/src")
for path in (str(EVAL_ROOT), str(VLM_INF_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vlm_inference.config import load_config
from vlm_inference.gt_action_cls import seed_everything
from vlm_inference.inputs import Sample, list_image_files
from vlm_inference.model import Qwen3VLCaptioner
from vlm_inference.video_io import install_av_video_reader

logger = logging.getLogger(__name__)

CONDITION = "full_1fps"
OUT_DIR = EVAL_ROOT / "experiments" / "virat" / "tube_video_captioning_30"
CAPTIONS_ONLY = OUT_DIR / "captions_only.json"
SAMPLE30_OUT = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/outputs/sample30_30b_start_centroid")
PROMPT_FILE = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/prompts/frames_full.txt")
DATASET_FPS1 = Path("/data/datasets/fps_sampling_frames/fps_1")
SAMPLE30_FPS1 = SAMPLE30_OUT / "full_frames" / "fps_1"
CONDITION_ORDER = (
    "full_1fps",
    "global_roi_start_only_roi_centroid_2fps_tube_video",
    "roi_fixed_1fps_tube_video",
    "roi_centroid_tube_video",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _local(path: str) -> Path:
    if path.startswith("file://"):
        return Path(unquote(urlparse(path).path))
    return Path(path)


def resolve_full_dir(video_id: str) -> Path:
    for root in (DATASET_FPS1, SAMPLE30_FPS1):
        directory = root / video_id
        if list_image_files(directory):
            return directory
    raise FileNotFoundError(f"no full_1fps JPEGs for {video_id}")


def jpeg_pixels(uris: list[str]) -> int:
    from PIL import Image

    total = 0
    for uri in uris:
        with Image.open(_local(uri)) as img:
            w, h = img.size
        total += int(w) * int(h)
    return total


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_captions_only(new_rows: list[dict[str, Any]]) -> None:
    existing = []
    if CAPTIONS_ONLY.is_file():
        existing = json.loads(CAPTIONS_ONLY.read_text(encoding="utf-8"))
    kept = [row for row in existing if row.get("condition") != CONDITION]
    slim = [
        {
            "video_id": row["video_id"],
            "condition": row["condition"],
            "caption": row.get("caption") or "",
        }
        for row in new_rows
        if not row.get("error")
    ]
    combined = kept + slim
    rank = {name: i for i, name in enumerate(CONDITION_ORDER)}
    combined.sort(
        key=lambda row: (row["video_id"], rank.get(row["condition"], 99), row["condition"])
    )
    write_json(CAPTIONS_ONLY, combined)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if SAMPLE30_OUT.resolve() == OUT_DIR.resolve():
        raise SystemExit("refusing to write into sample30 outputs")
    seed_everything(0)
    install_av_video_reader()
    cfg = load_config(EVAL_ROOT / "configs" / "virat_tube_video_audit.yaml")
    cfg.output.dir = str(OUT_DIR / "full_1fps")
    cfg.experiment.save_vlm_inputs = False
    videos = list(cfg.feasibility.videos)
    if len(videos) != 30:
        raise SystemExit(f"expected 30 videos, got {len(videos)}")
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    Path(cfg.output.dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading Qwen3-VL once from %s", cfg.model.path)
    captioner = Qwen3VLCaptioner(cfg)
    model_load_count = 1

    jsonl_path = OUT_DIR / "full_1fps_predictions.jsonl"
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("video_id") and rec.get("status") == "ok":
                rows.append(rec)
                done.add(rec["video_id"])
        logger.info("Resuming: %s already completed", len(done))

    for video_id in videos:
        if video_id in done:
            logger.info("skip completed %s", video_id)
            continue
        try:
            directory = resolve_full_dir(video_id)
            paths = [_as_uri(p) for p in list_image_files(directory)]
            if not paths:
                raise FileNotFoundError(directory)
            sample = Sample(
                job_name=f"{video_id}__{CONDITION}",
                sample_id=video_id,
                input_type="frames",
                paths=paths,
            )
            t0 = time.perf_counter()
            result = captioner.generate(sample, prompt)
            vlm_time = time.perf_counter() - t0
            rec = {
                "video_id": video_id,
                "condition": CONDITION,
                "caption": result.text,
                "status": "ok",
                "error": None,
                "num_full_frames": len(paths),
                "full_frame_dir": str(directory),
                "visual_tokens": result.num_visual_tokens,
                "processed_pixels": jpeg_pixels(paths),
                "vlm_time": vlm_time,
                "total_time": vlm_time,
                "model": cfg.model.path,
                "prompt_file": str(PROMPT_FILE),
            }
        except Exception as exc:
            logger.exception("full_1fps failed %s", video_id)
            rec = {
                "video_id": video_id,
                "condition": CONDITION,
                "caption": "",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "num_full_frames": None,
                "full_frame_dir": None,
                "visual_tokens": None,
                "processed_pixels": None,
                "vlm_time": 0.0,
                "total_time": 0.0,
                "model": cfg.model.path,
                "prompt_file": str(PROMPT_FILE),
            }
        rows.append(rec)
        append_jsonl(jsonl_path, rec)
        logger.info(
            "%s %s status=%s frames=%s tokens=%s vlm=%.2fs",
            video_id,
            CONDITION,
            rec["status"],
            rec.get("num_full_frames"),
            rec.get("visual_tokens"),
            rec.get("vlm_time") or 0.0,
        )

    ok = [row for row in rows if not row.get("error")]
    tokens = [float(row["visual_tokens"]) for row in ok if row.get("visual_tokens") is not None]
    summary = {
        "condition": CONDITION,
        "n_videos": len(videos),
        "completed": len(ok),
        "failed": len(rows) - len(ok),
        "model_load_count": model_load_count,
        "mean_visual_tokens": (sum(tokens) / len(tokens)) if tokens else None,
        "mean_vlm_time": (
            sum(float(row["vlm_time"]) for row in ok) / len(ok) if ok else None
        ),
        "existing_sample30_untouched": str(SAMPLE30_OUT),
        "outputs": {
            "predictions_json": str(OUT_DIR / "full_1fps_predictions.json"),
            "predictions_jsonl": str(jsonl_path),
            "summary": str(OUT_DIR / "full_1fps_summary.json"),
            "captions_only": str(CAPTIONS_ONLY),
        },
    }
    write_json(OUT_DIR / "full_1fps_predictions.json", {"model_load_count": model_load_count, "rows": rows})
    write_json(OUT_DIR / "full_1fps_summary.json", summary)
    merge_captions_only(rows)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
