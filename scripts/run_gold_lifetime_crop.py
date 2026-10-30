#!/usr/bin/env python3
"""Gold-case lifetime-bbox crop captioning at 1/2/5/8 fps.

Reuses the temporal-ablation sampling path: frames are chosen from original
timestamps, cropped in memory, and passed to Qwen3-VL with
do_sample_frames=False and do_resize=False. The processor therefore cannot
drop the run down to its default 2 fps or max_frames=768.

Completed checkpoints are kept. A new process starts after each generate
because a long-lived process in the previous ablation stalled after many
decodes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_ablation.sampling import probe_video, resize_rgb, select_frames, uniform_times  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "gold_lifetime_crop"
SEARCH_ROOTS = [
    "/data1/vailab02_dir/Classification_DB/VIRAT/videos-00",
    "/data1/vailab02_dir/Classification_DB/VIRAT/videos-01",
    "/data1/vailab02_dir/Classification_DB/VIRAT/videos-04",
    "/data/datasets/VIRAT/videos-00",
    "/data/datasets/VIRAT/videos-01",
    "/data/datasets/VIRAT/videos-04",
]

# 88_000 visual tokens is about 44.6 GiB on the fit from temporal ablation
# (MiB ≈ 16752 + 0.3164 * tokens; 73600 tokens measured 40038 MiB).
# A and G exceed that budget at native crop size and 8 fps. Their content
# size is reduced once and then reused for every FPS of that video.
TOKEN_BUDGET = 88000
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
SPATIAL_FACTOR = 32
EOS_TOKEN_IDS = [151645, 151643]

MODEL = {
    "id": "Qwen/Qwen3-VL-8B-Instruct",
    "path": "/data1/vailab02_dir/vlm_models/Qwen3-VL-8B-Instruct/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "device": "cuda:0",
    "do_sample": False,
    "temperature": 0,
    "max_new_tokens": 768,
    "seed": 0,
    "max_position_embeddings": 262144,
}

PROMPT = """Analyze the surveillance video carefully and describe the events that occur.

Focus on:
1. What actions are performed?
2. Who or what performs each action?
3. Which objects are involved?
4. What is the temporal order of the actions?
5. What direction do the people or vehicles move?

Only describe events that are visually supported by the provided frames.

Do not invent actions or infer unseen transitions.

If more than one video is attached, they are time-synchronized spatial crops of one clip. Write a single chronological description of that clip. Do not describe them as unrelated videos.

Return a concise, chronological description of the observed events.
"""

# bboxes are (x, y, width, height), half-open on the right and bottom.
VIDEOS: list[dict[str, Any]] = [
    {
        "video_id": "A",
        "filename": "VIRAT_S_000201_04_000682_000822",
        "bboxes": [(0, 200, 1280, 520)],
        "content_sizes": [(608, 247)],
        "scale_note": "Native 1280x520 at 8 fps is about 379k visual tokens, above both the 262144 context and the 48 GiB card. Content is resized to 608x247 (scale 0.475 on both axes) for every FPS of A.",
    },
    {
        "video_id": "B",
        "filename": "VIRAT_S_010001_05_000649_000684",
        "bboxes": [(0, 0, 1280, 497), (0, 552, 312, 168)],
        "content_sizes": [(1280, 497), (312, 168)],
        "scale_note": "Native crop size. Two videos are passed together; their boxes are not unioned.",
    },
    {
        "video_id": "C",
        "filename": "VIRAT_S_010106_05_000954_000996",
        "bboxes": [(0, 109, 1276, 380)],
        "content_sizes": [(1276, 380)],
        "scale_note": "Native crop size. Only factor-32 padding is added.",
    },
    {
        "video_id": "D",
        "filename": "VIRAT_S_040005_00_000003_000058",
        "bboxes": [(0, 256, 628, 424)],
        "content_sizes": [(628, 424)],
        "scale_note": "Native crop size. Only factor-32 padding is added.",
    },
    {
        "video_id": "E",
        "filename": "VIRAT_S_010116_00_000061_000121",
        "bboxes": [(0, 103, 1280, 284)],
        "content_sizes": [(1280, 284)],
        "scale_note": "Native crop size. 8 fps is near 84k visual tokens and is expected to fit.",
    },
    {
        "video_id": "F",
        "filename": "VIRAT_S_010000_07_000827_000860",
        "bboxes": [(0, 85, 1280, 420)],
        "content_sizes": [(1280, 420)],
        "scale_note": "Native crop size. Only factor-32 padding is added.",
    },
    {
        "video_id": "G",
        "filename": "VIRAT_S_010005_03_000215_000270",
        "bboxes": [(0, 90, 1280, 420)],
        "content_sizes": [(1120, 368)],
        "scale_note": "Native 1280x420 at 8 fps is about 114k visual tokens (~53 GiB). Content is resized to 1120x368 (0.875 of width; height rounds 367.5 to 368) for every FPS of G.",
    },
]

FPS_LIST = [1.0, 2.0, 5.0, 8.0]
TERMINAL = {"ok", "oom", "context_exceeded", "prepare_error", "inference_error", "sampling_error"}
CSV_FIELDS = [
    "video_id",
    "video_filename",
    "source_path",
    "original_width",
    "original_height",
    "original_fps",
    "duration_sec",
    "roi_count",
    "bbox_coordinates",
    "crop_resolution",
    "sampling_fps",
    "actual_frame_count",
    "actual_sampling_fps",
    "vlm_input_resolution",
    "visual_tokens",
    "inference_time_sec",
    "model_name",
    "caption",
    "status",
    "error_message",
]

log = logging.getLogger("gold_lifetime_crop")


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fps_tag(fps: float) -> str:
    return f"{fps:g}"


def job_name(video_id: str, fps: float) -> str:
    return f"{video_id}_{fps_tag(fps)}fps"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_text(bboxes: list[tuple[int, int, int, int]]) -> str:
    return " | ".join(f"{x},{y};{w},{h}" for x, y, w, h in bboxes)


def size_text(sizes: list[tuple[int, int]]) -> str:
    return " | ".join(f"{w}x{h}" for w, h in sizes)


def align_up(value: int, factor: int) -> int:
    return int(math.ceil(value / factor) * factor)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def resolve_video(filename: str) -> Path:
    name = filename if filename.endswith(".mp4") else f"{filename}.mp4"
    for root in SEARCH_ROOTS:
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find {name} under {SEARCH_ROOTS}")


def video_by_id(video_id: str) -> dict[str, Any]:
    for video in VIDEOS:
        if video["video_id"] == video_id:
            return video
    raise KeyError(video_id)


def prompt_sha() -> str:
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:16]


def fingerprint(video: dict[str, Any], fps: float, probe: dict[str, Any]) -> str:
    payload = {
        "video_id": video["video_id"],
        "filename": video["filename"],
        "bboxes": video["bboxes"],
        "content_sizes": video["content_sizes"],
        "fps": fps,
        "duration_sec": probe["duration_sec"],
        "model": MODEL["path"],
        "revision": MODEL["revision"],
        "prompt_sha": prompt_sha(),
        "max_new_tokens": MODEL["max_new_tokens"],
        "do_sample": False,
        "seed": MODEL["seed"],
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def ensure_dirs() -> None:
    for name in ("configs", "crops", "previews", "captions", "logs", "sampling", "checkpoints"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_dirs()
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(OUT / "logs" / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stream)


def validate_bbox(width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive bbox {bbox}")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"bbox {bbox} is outside {width}x{height}")


def collect_environment() -> dict[str, Any]:
    import torch

    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_vram_mib": int(props.total_memory / (1024**2)),
                    "free_vram_mib": int(free / (1024**2)),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    os_release = ""
    release_path = Path("/etc/os-release")
    if release_path.is_file():
        os_release = release_path.read_text(encoding="utf-8")
    return {
        "collected_at": utc_now(),
        "host": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "os_release": os_release,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus_visible": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "transformers": package_version("transformers"),
        "av": package_version("av"),
        "opencv": package_version("opencv-python"),
        "model_id": MODEL["id"],
        "model_path": MODEL["path"],
        "model_revision": MODEL["revision"],
        "do_sample": False,
        "temperature": MODEL["temperature"],
        "max_new_tokens": MODEL["max_new_tokens"],
        "seed": MODEL["seed"],
        "deterministic_algorithms": False,
        "seed_note": "torch.manual_seed is set before generate. do_sample is false. CUDA kernels are not forced deterministic.",
        "processor_do_resize": False,
        "processor_do_sample_frames": False,
        "processor_default_fps_not_used": 2,
        "processor_default_max_frames_not_used": 768,
        "visual_token_budget_per_condition": TOKEN_BUDGET,
        "token_budget_note": "Budget is a memory planning cap. Measured visual tokens are stored per run. Conditions are not silently retargeted to a lower FPS.",
    }


def _even_pix_fmt(width: int, height: int) -> str:
    if width % 2 == 0 and height % 2 == 0:
        return "yuv420p"
    return "yuv444p"


def write_crop_video(
    source: Path,
    dest: Path,
    bbox: tuple[int, int, int, int],
    preview_stem: str,
) -> dict[str, Any]:
    x, y, w, h = bbox
    probe = probe_video(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sidecar = dest.with_suffix(".json")
    if dest.is_file() and sidecar.is_file():
        previous = read_json(sidecar)
        if (
            previous.get("source_path") == str(source)
            and previous.get("bbox") == [x, y, w, h]
            and previous.get("status") == "ok"
            and (OUT / "previews" / f"{preview_stem}_first.jpg").is_file()
            and (OUT / "previews" / f"{preview_stem}_mid.jpg").is_file()
        ):
            log.info("reuse crop %s", dest)
            return previous

    input_container = av.open(str(source))
    output = av.open(str(dest), mode="w")
    try:
        in_stream = input_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        rate = in_stream.average_rate
        pix_fmt = _even_pix_fmt(w, h)
        out_stream = output.add_stream("libx264", rate=rate)
        out_stream.width = w
        out_stream.height = h
        out_stream.pix_fmt = pix_fmt
        out_stream.options = {"crf": "18", "preset": "veryfast"}
        expected = int(probe["frame_count_metadata"] or 0)
        mid_index = expected // 2 if expected else None
        saved_mid = False
        count = 0
        for frame in input_container.decode(in_stream):
            array = frame.to_ndarray(format="rgb24")
            if array.shape[1] != probe["width"] or array.shape[0] != probe["height"]:
                raise RuntimeError(f"decoded {array.shape} != probe {probe['width']}x{probe['height']}")
            crop = np.ascontiguousarray(array[y : y + h, x : x + w])
            if crop.shape[0] != h or crop.shape[1] != w:
                raise RuntimeError(f"crop shape {crop.shape} != {h}x{w}")
            if count == 0:
                _save_jpeg(OUT / "previews" / f"{preview_stem}_first.jpg", crop)
            if mid_index is not None and count == mid_index:
                _save_jpeg(OUT / "previews" / f"{preview_stem}_mid.jpg", crop)
                saved_mid = True
            out_frame = av.VideoFrame.from_ndarray(crop, format="rgb24")
            out_frame.pts = count
            for packet in out_stream.encode(out_frame):
                output.mux(packet)
            count += 1
            if count % 500 == 0:
                log.info("encoding %s frame %s", dest.name, count)
        for packet in out_stream.encode():
            output.mux(packet)
    finally:
        output.close()
        input_container.close()

    if mid_index is None:
        raise RuntimeError(f"source frame count metadata missing for {source}")
    if not saved_mid:
        raise RuntimeError(f"middle frame {mid_index} was not saved for {dest}")
    crop_probe = probe_video(dest)
    record = {
        "status": "ok",
        "source_path": str(source),
        "crop_path": str(dest),
        "bbox": [x, y, w, h],
        "crop_width": w,
        "crop_height": h,
        "decoded_frame_count": count,
        "source_frame_count_metadata": expected,
        "source_duration_sec": probe["duration_sec"],
        "source_fps": probe["native_fps"],
        "crop_width_probe": crop_probe["width"],
        "crop_height_probe": crop_probe["height"],
        "crop_fps": crop_probe["native_fps"],
        "crop_duration_sec": crop_probe["duration_sec"],
        "crop_frame_count_metadata": crop_probe["frame_count_metadata"],
        "pix_fmt": pix_fmt,
        "codec": "libx264",
        "crf": 18,
        "preview_first": str(OUT / "previews" / f"{preview_stem}_first.jpg"),
        "preview_mid": str(OUT / "previews" / f"{preview_stem}_mid.jpg"),
    }
    if crop_probe["width"] != w or crop_probe["height"] != h:
        record["status"] = "geometry_mismatch"
        record["error"] = f"encoded {crop_probe['width']}x{crop_probe['height']} != {w}x{h}"
    elif count != expected:
        record["status"] = "frame_count_mismatch"
        record["error"] = f"decoded {count} frames, metadata said {expected}"
    elif abs(float(crop_probe["native_fps"]) - float(probe["native_fps"])) > 0.05:
        record["status"] = "fps_mismatch"
        record["error"] = f"crop fps {crop_probe['native_fps']} != source {probe['native_fps']}"
    elif abs(float(crop_probe["duration_sec"]) - float(probe["duration_sec"])) > 0.2:
        record["status"] = "duration_mismatch"
        record["error"] = f"crop duration {crop_probe['duration_sec']} != source {probe['duration_sec']}"
    write_json(sidecar, record)
    if record["status"] != "ok":
        raise RuntimeError(record["error"])
    log.info("crop ok %s frames=%s %sx%s fps=%s", dest, count, w, h, crop_probe["native_fps"])
    return record


def _save_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed to write {path}")


def prepare(selected: list[str] | None) -> dict[str, Any]:
    ensure_dirs()
    (OUT / "configs" / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    manifest_videos = []
    for video in VIDEOS:
        if selected and video["video_id"] not in selected:
            continue
        source = resolve_video(video["filename"])
        probe = probe_video(source)
        for bbox in video["bboxes"]:
            validate_bbox(int(probe["width"]), int(probe["height"]), tuple(bbox))
        if len(video["bboxes"]) != len(video["content_sizes"]):
            raise RuntimeError(f"{video['video_id']} bbox/content size mismatch")
        crops = []
        for index, bbox in enumerate(video["bboxes"], start=1):
            dest = OUT / "crops" / video["video_id"] / f"roi{index}.mp4"
            crops.append(
                write_crop_video(
                    source,
                    dest,
                    tuple(bbox),
                    f"{video['video_id']}_roi{index}",
                )
            )
        entry = {
            "video_id": video["video_id"],
            "video_filename": video["filename"],
            "source_path": str(source),
            "original_width": probe["width"],
            "original_height": probe["height"],
            "original_fps": probe["native_fps"],
            "duration_sec": probe["duration_sec"],
            "frame_count_metadata": probe["frame_count_metadata"],
            "codec": probe["codec"],
            "roi_count": len(video["bboxes"]),
            "bboxes": video["bboxes"],
            "bbox_coordinates": bbox_text(video["bboxes"]),
            "crop_resolution": size_text([(b[2], b[3]) for b in video["bboxes"]]),
            "content_sizes": video["content_sizes"],
            "scale_note": video["scale_note"],
            "crops": crops,
            "frame_counts": {
                fps_tag(fps): len(uniform_times(0.0, float(probe["duration_sec"]), fps, 0.0)) for fps in FPS_LIST
            },
        }
        manifest_videos.append(entry)
        log.info(
            "video %s %sx%s fps=%s dur=%.3f frames=%s",
            video["video_id"],
            probe["width"],
            probe["height"],
            probe["native_fps"],
            probe["duration_sec"],
            entry["frame_counts"],
        )
    manifest = {
        "created_at": utc_now(),
        "spatial_condition": "gold_lifetime_bbox",
        "sampling_fps": FPS_LIST,
        "global_frame": False,
        "prompt_sha": prompt_sha(),
        "model": MODEL,
        "token_budget": TOKEN_BUDGET,
        "videos": manifest_videos,
    }
    write_json(OUT / "configs" / "manifest.json", manifest)
    write_json(OUT / "configs" / "model.json", MODEL)
    return manifest


def _disable_triton() -> None:
    try:
        import torch.backends.python_native as pn

        pn.triton.enabled = False
    except Exception:
        pass


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _disable_triton()
    processor = AutoProcessor.from_pretrained(MODEL["path"])
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL["path"],
        dtype=torch.bfloat16,
        device_map=MODEL["device"],
        attn_implementation=MODEL["attn_implementation"],
    )
    model.eval()
    return processor, model


def assert_processor_keeps_frames(processor) -> None:
    """Fail if the processor resamples or clips to its default max_frames."""
    import torch

    long = torch.zeros(800, 3, 32, 64, dtype=torch.uint8)
    short = torch.zeros(6, 3, 64, 96, dtype=torch.uint8)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "memory://roi-long"},
                {"type": "video", "video": "memory://roi-short"},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    metas = [
        {"fps": 30.0, "frames_indices": list(range(800)), "total_num_frames": 800, "video_backend": "torch"},
        {"fps": 30.0, "frames_indices": list(range(6)), "total_num_frames": 6, "video_backend": "torch"},
    ]
    inputs = processor(
        text=[text],
        videos=[long, short],
        video_metadata=metas,
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )
    grid = inputs["video_grid_thw"]
    if tuple(grid.shape) != (2, 3):
        raise RuntimeError(f"expected two video grids, got {tuple(grid.shape)}")
    if int(grid[0, 0]) != 400 or int(grid[1, 0]) != 3:
        raise RuntimeError(f"processor changed frame count, grid_t={grid[:, 0].tolist()} (800 frames must stay 400 temporal patches)")
    log.info("processor keeps supplied frames; default max_frames/fps were not applied")


def _stack_roi(
    arrays: list[np.ndarray],
    bbox: tuple[int, int, int, int],
    content_size: tuple[int, int],
) -> tuple[Any, bool, str, str]:
    import torch
    import torch.nn.functional as F

    x, y, w, h = bbox
    content_w, content_h = content_size
    frames = []
    resized_any = False
    for array in arrays:
        crop = np.ascontiguousarray(array[y : y + h, x : x + w])
        if crop.shape[0] != h or crop.shape[1] != w:
            raise RuntimeError(f"crop {crop.shape[:2]} != {(h, w)}")
        image, did = resize_rgb(crop, content_w, content_h)
        if image.shape[0] != content_h or image.shape[1] != content_w:
            raise RuntimeError(f"resize produced {image.shape[:2]} != {(content_h, content_w)}")
        resized_any = resized_any or did
        frames.append(image)
    stacked = np.stack(frames, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    proc_h, proc_w = align_up(content_h, SPATIAL_FACTOR), align_up(content_w, SPATIAL_FACTOR)
    pad_bottom = proc_h - content_h
    pad_right = proc_w - content_w
    if pad_bottom or pad_right:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), value=0)
    return tensor, resized_any, f"{content_w}x{content_h}", f"{proc_w}x{proc_h}"


def prepare_inputs(processor, video: dict[str, Any], probe: dict[str, Any], fps: float) -> dict[str, Any]:
    targets = uniform_times(0.0, float(probe["duration_sec"]), fps, 0.0)
    if not targets:
        raise RuntimeError("no sample timestamps")
    started = time.perf_counter()
    selected = select_frames(Path(probe["path"]), targets, native_fps=float(probe["native_fps"]))
    tensors = []
    content_res = []
    padded_res = []
    resized_flags = []
    for bbox, content_size in zip(video["bboxes"], video["content_sizes"]):
        tensor, resized, content, padded = _stack_roi(selected["arrays"], tuple(bbox), tuple(content_size))
        tensors.append(tensor)
        content_res.append(content)
        padded_res.append(padded)
        resized_flags.append(resized)
    if any(tensor.shape[0] != tensors[0].shape[0] for tensor in tensors):
        raise RuntimeError("ROI frame counts differ")
    num_frames = int(tensors[0].shape[0])
    rel_indices = [int(row["source_frame_index"]) for row in selected["frames"]]
    if len(rel_indices) != num_frames:
        raise RuntimeError("index count differs from stacked frames")
    if len(set(rel_indices)) != len(rel_indices):
        raise RuntimeError("duplicate source frames; the requested fps is not a distinct-frame sample")
    messages_content: list[dict[str, str]] = []
    for index in range(len(tensors)):
        messages_content.append({"type": "video", "video": f"memory://{video['video_id']}-roi{index + 1}"})
    messages_content.append({"type": "text", "text": PROMPT.strip()})
    text = processor.apply_chat_template(
        [{"role": "user", "content": messages_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    metadatas = [
        {
            "fps": float(probe["native_fps"]),
            "frames_indices": rel_indices,
            "total_num_frames": max(rel_indices) + 1,
            "video_backend": "av",
        }
        for _ in tensors
    ]
    inputs = processor(
        text=[text],
        videos=tensors,
        video_metadata=metadatas,
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )
    grids = [[int(v) for v in row] for row in inputs["video_grid_thw"].tolist()]
    measured = 0
    for grid, tensor in zip(grids, tensors):
        proc_h, proc_w = int(tensor.shape[2]), int(tensor.shape[3])
        tokens = 1
        for value in grid:
            tokens *= int(value)
        tokens //= MERGE_SIZE**2
        measured += tokens
        expected_t = int(math.ceil(num_frames / TEMPORAL_PATCH_SIZE))
        if grid[0] != expected_t:
            raise RuntimeError(f"temporal grid {grid[0]} != ceil(frames/2)={expected_t}; processor dropped frames")
        if grid[1] != proc_h // PATCH_SIZE or grid[2] != proc_w // PATCH_SIZE:
            raise RuntimeError(f"spatial grid {grid} does not match {proc_w}x{proc_h}")
    placeholder = int((inputs["input_ids"] == processor.video_token_id).sum().item())
    if placeholder != measured:
        raise RuntimeError(f"placeholder tokens {placeholder} != measured {measured}")
    times = [float(row["source_time_sec"]) for row in selected["frames"]]
    if len(times) >= 2:
        actual_fps = (len(times) - 1) / (times[-1] - times[0])
    else:
        actual_fps = None
    max_error = float(selected["max_abs_error_sec"])
    frame_period = 1.0 / float(probe["native_fps"])
    if max_error > 1.5 * frame_period:
        raise RuntimeError(f"timestamp error {max_error:.4f}s exceeds 1.5 source frames")
    return {
        "inputs": inputs,
        "elapsed_sec": time.perf_counter() - started,
        "num_frames": num_frames,
        "actual_sampling_fps": None if actual_fps is None else round(actual_fps, 6),
        "timestamps": [float(row["target_time_sec"]) for row in selected["frames"]],
        "source_times": times,
        "max_abs_error_sec": max_error,
        "sampling": selected["frames"],
        "content_resolution": " | ".join(content_res),
        "vlm_input_resolution": " | ".join(padded_res),
        "content_resized": any(resized_flags),
        "video_grid_thw": grids,
        "visual_tokens": measured,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "processor_frame_count": num_frames,
        "processor_temporal_patches": [grid[0] for grid in grids],
    }


def generate_one(model, processor, prepared: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.manual_seed(int(MODEL["seed"]))
    inputs = prepared["inputs"].to(MODEL["device"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(MODEL["max_new_tokens"]),
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    prompt_len = int(inputs["input_ids"].shape[1])
    generated = output_ids[0, prompt_len:]
    text = processor.batch_decode(
        generated.unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    output_tokens = int(generated.shape[0])
    last_id = int(generated[-1].item()) if output_tokens else None
    hit_limit = output_tokens >= int(MODEL["max_new_tokens"]) and last_id not in EOS_TOKEN_IDS
    del inputs, output_ids, generated
    return {
        "status": "ok",
        "caption": text,
        "output_token_count": output_tokens,
        "hit_token_limit": hit_limit,
        "inference_time_sec": round(elapsed, 4),
        "peak_gpu_memory_bytes": peak,
        "peak_gpu_memory_mib": round(peak / (1024**2), 1),
    }


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _base_record(video: dict[str, Any], probe: dict[str, Any], fps: float) -> dict[str, Any]:
    return {
        "video_id": video["video_id"],
        "video_filename": video["filename"],
        "source_path": probe["path"],
        "original_width": probe["width"],
        "original_height": probe["height"],
        "original_fps": probe["native_fps"],
        "duration_sec": probe["duration_sec"],
        "roi_count": len(video["bboxes"]),
        "bbox_coordinates": bbox_text(video["bboxes"]),
        "crop_resolution": size_text([(b[2], b[3]) for b in video["bboxes"]]),
        "sampling_fps": fps,
        "actual_frame_count": None,
        "actual_sampling_fps": None,
        "vlm_input_resolution": None,
        "vlm_content_resolution": size_text(video["content_sizes"]),
        "visual_tokens": None,
        "inference_time_sec": None,
        "model_name": MODEL["id"],
        "model_revision": MODEL["revision"],
        "caption": None,
        "status": None,
        "error_message": None,
        "fingerprint": fingerprint(video, fps, probe),
        "prompt_sha": prompt_sha(),
        "do_sample": False,
        "temperature": MODEL["temperature"],
        "max_new_tokens": MODEL["max_new_tokens"],
        "seed": MODEL["seed"],
        "processor_do_sample_frames": False,
        "processor_do_resize": False,
        "global_frame_included": False,
        "multi_roi_method": "simultaneous_videos" if len(video["bboxes"]) > 1 else "single_crop",
        "scale_note": video["scale_note"],
        "command": (
            "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "TOKENIZERS_PARALLELISM=false "
            f"{sys.executable} scripts/run_gold_lifetime_crop.py infer"
        ),
    }


def _write_caption(video_id: str, fps: float, text: str) -> None:
    path = OUT / "captions" / f"{job_name(video_id, fps)}.txt"
    path.write_text(text + "\n", encoding="utf-8")


def _pending_jobs(
    probes: dict[str, dict[str, Any]],
    selected_videos: list[str] | None,
    selected_fps: list[float],
    force: bool,
    retry_failed: bool,
) -> list[tuple[dict[str, Any], float]]:
    pending = []
    for video in VIDEOS:
        if selected_videos and video["video_id"] not in selected_videos:
            continue
        for fps in selected_fps:
            fps = float(fps)
            checkpoint = OUT / "checkpoints" / f"{job_name(video['video_id'], fps)}.json"
            if checkpoint.is_file() and not force:
                previous = read_json(checkpoint)
                same = previous.get("fingerprint") == fingerprint(video, fps, probes[video["video_id"]])
                if same and previous.get("status") in TERMINAL and (previous.get("status") == "ok" or not retry_failed):
                    continue
            pending.append((video, fps))
    return pending


def run_infer(selected_videos: list[str] | None, selected_fps: list[float], force: bool, retry_failed: bool) -> None:
    import torch

    manifest_path = OUT / "configs" / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("manifest missing; run the prepare stage first")
    probes = {}
    for video in VIDEOS:
        source = resolve_video(video["filename"])
        probes[video["video_id"]] = probe_video(source)
    jobs = _pending_jobs(probes, selected_videos, selected_fps, force, retry_failed)
    if not jobs:
        log.info("no pending jobs")
        aggregate()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    write_json(OUT / "configs" / "environment.json", collect_environment())
    processor, model = load_model()
    log.info("loaded %s", type(model).__name__)
    check_path = OUT / "logs" / "processor_frame_check.json"
    if not check_path.is_file():
        assert_processor_keeps_frames(processor)
        write_json(check_path, {"status": "ok", "checked_at": utc_now(), "long_frames": 800, "short_frames": 6})

    for video, fps in jobs:
        name = job_name(video["video_id"], fps)
        checkpoint = OUT / "checkpoints" / f"{name}.json"
        probe = probes[video["video_id"]]
        current_fp = fingerprint(video, fps, probe)
        if checkpoint.is_file() and not force:
            previous = read_json(checkpoint)
            same = previous.get("fingerprint") == current_fp and previous.get("status") in TERMINAL
            if same and (previous.get("status") == "ok" or not retry_failed):
                log.info("skip %s status=%s", name, previous.get("status"))
                continue
        record = _base_record(video, probe, fps)
        record["started_at"] = utc_now()
        log.info("start %s", name)
        try:
            prepared = prepare_inputs(processor, video, probe, fps)
        except Exception as exc:
            record.update(
                {
                    "status": "sampling_error" if "duplicate" in str(exc).lower() or "timestamp" in str(exc).lower() else "prepare_error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_now(),
                }
            )
            write_json(checkpoint, record)
            log.exception("prepare failed %s", name)
            if _reexec():
                return
            continue
        record.update(
            {
                "actual_frame_count": prepared["num_frames"],
                "actual_sampling_fps": prepared["actual_sampling_fps"],
                "vlm_input_resolution": prepared["vlm_input_resolution"],
                "vlm_content_resolution": prepared["content_resolution"],
                "visual_tokens": prepared["visual_tokens"],
                "prompt_token_count": prepared["prompt_tokens"],
                "video_grid_thw": prepared["video_grid_thw"],
                "processor_temporal_patches": prepared["processor_temporal_patches"],
                "content_resized": prepared["content_resized"],
                "max_abs_decoder_error_sec": prepared["max_abs_error_sec"],
                "preprocess_time_sec": round(prepared["elapsed_sec"], 4),
                "processor_do_sample_frames": False,
            }
        )
        sampling_payload = {
            "job": name,
            "fps": fps,
            "num_frames": prepared["num_frames"],
            "actual_sampling_fps": prepared["actual_sampling_fps"],
            "max_abs_error_sec": prepared["max_abs_error_sec"],
            "timestamps": prepared["timestamps"],
            "source_times": prepared["source_times"],
            "frames": prepared["sampling"],
            "processor_do_sample_frames": False,
            "processor_temporal_patches": prepared["processor_temporal_patches"],
        }
        write_json(OUT / "sampling" / f"{name}.json", sampling_payload)
        if prepared["prompt_tokens"] > int(MODEL["max_position_embeddings"]):
            record.update(
                {
                    "status": "context_exceeded",
                    "error_message": (
                        f"prompt tokens {prepared['prompt_tokens']} exceed "
                        f"max_position_embeddings {MODEL['max_position_embeddings']}"
                    ),
                    "finished_at": utc_now(),
                }
            )
            write_json(checkpoint, record)
            log.error("context exceeded %s", name)
            del prepared
            gc.collect()
            torch.cuda.empty_cache()
            if _reexec():
                return
            continue
        try:
            generated = generate_one(model, processor, prepared)
        except Exception as exc:
            torch.cuda.empty_cache()
            record.update(
                {
                    "status": "oom" if _is_oom(exc) else "inference_error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_now(),
                }
            )
            write_json(checkpoint, record)
            log.error("%s %s", record["status"], name)
            del prepared
            gc.collect()
            torch.cuda.empty_cache()
            if _reexec():
                return
            continue
        record.update(generated)
        record["finished_at"] = utc_now()
        record["error_message"] = None
        write_json(checkpoint, record)
        if generated.get("caption"):
            _write_caption(video["video_id"], fps, generated["caption"])
        log.info(
            "ok %s frames=%s tokens=%s time=%.1fs peak=%.0fMiB truncated=%s",
            name,
            record["actual_frame_count"],
            record["visual_tokens"],
            record["inference_time_sec"],
            record["peak_gpu_memory_mib"],
            record["hit_token_limit"],
        )
        del prepared
        gc.collect()
        torch.cuda.empty_cache()
        if _reexec():
            return
    aggregate()


def _reexec() -> bool:
    aggregate()
    if os.environ.get("GOLD_REEXEC", "1") == "0":
        return False
    log.info("reexec to release decoder and CUDA state")
    os.execv(sys.executable, [sys.executable, *sys.argv])
    return True


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def aggregate() -> None:
    ensure_dirs()
    records = []
    for path in sorted((OUT / "checkpoints").glob("*.json")):
        records.append(read_json(path))
    records.sort(key=lambda row: (row.get("video_id", ""), float(row.get("sampling_fps") or 0)))
    csv_path = OUT / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_value(record.get(field)) for field in CSV_FIELDS})
    manifest = read_json(OUT / "configs" / "manifest.json") if (OUT / "configs" / "manifest.json").is_file() else None
    environment = read_json(OUT / "configs" / "environment.json") if (OUT / "configs" / "environment.json").is_file() else None
    payload = {
        "experiment": "gold_lifetime_crop",
        "generated_at": utc_now(),
        "model": MODEL,
        "prompt": PROMPT,
        "prompt_sha": prompt_sha(),
        "token_budget": TOKEN_BUDGET,
        "spatial_condition": "gold_lifetime_bbox",
        "global_frame_included": False,
        "multi_roi_method_B": "Two crop videos are passed in one generate call. Boxes are not unioned and the full frame is not attached.",
        "vlm_pixel_source": "Frames are decoded from the original file and cropped in memory. The archived mp4 is not decoded for inference, so encoder loss is not part of the VLM input.",
        "manifest": manifest,
        "environment": environment,
        "results": records,
    }
    write_json(OUT / "results.json", payload)
    log.info("wrote %s rows=%s", csv_path, len(records))


def _parse_ids(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_fps(text: str | None) -> list[float]:
    if not text:
        return list(FPS_LIST)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold lifetime-bbox crop captioning")
    parser.add_argument("stage", choices=["prepare", "infer", "aggregate"])
    parser.add_argument("--videos", default="", help="Comma-separated ids, default all")
    parser.add_argument("--fps", default="", help="Comma-separated fps, default 1,2,5,8")
    parser.add_argument("--force", action="store_true", help="Rerun even successful checkpoints")
    parser.add_argument("--retry-failed", action="store_true", help="Rerun non-ok checkpoints")
    args = parser.parse_args()
    setup_logging()
    selected = _parse_ids(args.videos)
    if args.stage == "prepare":
        prepare(selected)
    elif args.stage == "infer":
        run_infer(selected, _parse_fps(args.fps), args.force, args.retry_failed)
    else:
        aggregate()


if __name__ == "__main__":
    main()
