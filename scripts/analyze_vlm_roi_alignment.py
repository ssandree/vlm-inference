#!/usr/bin/env python3
"""Compare Qwen visual information volumes with the existing VIRAT ROI tubes."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
import yaml
from PIL import Image

from analyze_video_information_volume import _prepare_inputs
from vlm_inference.config import load_config
from vlm_inference.inputs import Sample, list_image_files
from vlm_inference.model import Qwen3VLCaptioner
from vlm_inference.prompts import load_job_prompt
from vlm_inference.roi_margin import tracks_json_path
from vlm_inference.video_io import inspect_video

LOG = logging.getLogger("vlm_roi_alignment")
VIDEO_IDS = [
    "VIRAT_S_000201_02_000590_000623",
    "VIRAT_S_000200_01_000226_000268",
    "VIRAT_S_000201_06_001354_001397",
    "VIRAT_S_010003_08_000739_000778",
    "VIRAT_S_010004_00_000064_000100",
    "VIRAT_S_010106_05_000954_000996",
    "VIRAT_S_010200_05_000658_000700",
    "VIRAT_S_040000_02_000111_000145",
    "VIRAT_S_050000_05_000696_000732",
]
THRESHOLDS = {"positive": 0.0, "top30": 0.70, "top20": 0.80, "top10": 0.90}
ROI_TRACK_FPS = 5.0


def local_path(path: str) -> Path:
    parsed = urlparse(path)
    return Path(parsed.path if parsed.scheme == "file" else path)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def gt_caption(config_raw: dict, video_id: str) -> str:
    info = config_raw["visual_information_volume"]
    if info.get("gt_caption"):
        return str(info["gt_caption"]).strip()
    refs = json.loads(Path(info["gt_reference_file"]).read_text(encoding="utf-8"))
    value = refs.get("references", refs)[video_id]
    return str(value[int(info.get("gt_reference_index", 0))] if isinstance(value, list) else value).strip()


def source_sample(cfg, video_id: str) -> tuple[Sample, list[Path]]:
    directory = Path(cfg.feasibility.datasets.fps_1) / video_id
    paths = list_image_files(directory, "*.jpg")
    if not paths:
        raise FileNotFoundError(f"No full-frame sampling images under {directory}")
    sample = Sample(
        job_name="video_information_volume",
        sample_id=video_id,
        input_type="frames",
        paths=[path.resolve().as_uri() for path in paths],
        sidecar_dirs=[directory],
    )
    return sample, paths


def load_roi_cells(track_path: Path) -> dict[int, set[tuple[int, int]]]:
    data = json.loads(track_path.read_text(encoding="utf-8"))
    cells_by_frame: dict[int, set[tuple[int, int]]] = {}
    for track in data.get("tracks", data.get("tubes", [])):
        for frame in track.get("per_frame", []):
            frame_index = int(frame["frame_index"])
            cells = cells_by_frame.setdefault(frame_index, set())
            cells.update((int(cell[0]), int(cell[1])) for cell in frame.get("cells", []))
    if not cells_by_frame:
        raise ValueError(f"{track_path}: no per-frame ROI cells")
    return cells_by_frame


def project_roi_mask(blocks: list[dict], frames: list[dict], cells_by_frame: dict[int, set[tuple[int, int]]], video_path: Path) -> np.ndarray:
    info = inspect_video(video_path)
    mask = np.zeros((len(frames), max(f["block_rows"] for f in frames), max(f["block_cols"] for f in frames)), dtype=bool)
    for frame in frames:
        raw_index = int(round(float(frame["timestamp"]) * ROI_TRACK_FPS))
        cells = cells_by_frame.get(raw_index, set())
        sx = float(frame["mapped_width"]) / float(frame["original_width"])
        sy = float(frame["mapped_height"]) / float(frame["original_height"])
        for y, x in cells:
            mapped_x = (x + 0.5) * 64.0 * sx
            mapped_y = (y + 0.5) * 64.0 * sy
            col = min(max(int(mapped_x // 64), 0), mask.shape[2] - 1)
            row = min(max(int(mapped_y // 64), 0), mask.shape[1] - 1)
            mask[frame["frame_idx"], row, col] = True
    return mask


def attach_roi(blocks: list[dict], mask: np.ndarray) -> None:
    for block in blocks:
        block["in_roi"] = bool(mask[block["frame_idx"], block["block_row"], block["block_col"]])


def metrics(raw: np.ndarray, roi: np.ndarray) -> dict:
    positive = np.maximum(raw, 0.0)
    total = int(roi.size)
    roi_count = int(roi.sum())
    total_positive = float(positive.sum())
    roi_positive = float(positive[roi].sum())
    roi_density = float(positive[roi].mean()) if roi.any() else None
    non_roi = ~roi
    non_roi_density = float(positive[non_roi].mean()) if non_roi.any() else None
    return {
        "ROI_volume_ratio": float(roi_count / total) if total else None,
        "ROI_information_coverage": roi_positive / total_positive if total_positive else None,
        "information_enrichment": (roi_positive / total_positive) / (roi_count / total) if roi_count and total_positive else None,
        "ROI_density": roi_density,
        "nonROI_density": non_roi_density,
        "density_ratio": roi_density / non_roi_density if roi_density is not None and non_roi_density else None,
        "roi_blocks": roi_count,
        "total_blocks": total,
        "positive_information_sum": total_positive,
    }


def render_3d(raw: np.ndarray, roi: np.ndarray, out: Path, title: str, quantile: float) -> None:
    positive = np.maximum(raw, 0.0)
    vmax = float(positive.max()) if positive.size else 1.0
    threshold = float(np.quantile(positive[positive > 0], quantile)) if quantile > 0 and np.any(positive > 0) else 0.0
    selected = positive >= threshold if quantile > 0 else positive > 0
    # Matplotlib's voxel axes are array axes (x, y, z). Reorder the volume
    # from [t, y, x] to [x, t, y] so spatial y is vertical and time recedes
    # diagonally, matching the reference ROI-tube figure.
    positive_xyz = positive.transpose(2, 0, 1)
    selected_xyz = selected.transpose(2, 0, 1)
    roi_xyz = roi.transpose(2, 0, 1)
    fig = plt.figure(figsize=(13, 9), facecolor="#f7f9fc")
    ax = fig.add_subplot(111, projection="3d")
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1e-12))
    cmap = plt.get_cmap("turbo")
    colors = np.zeros((*positive_xyz.shape, 4), dtype=float)
    colors[selected_xyz] = cmap(norm(positive_xyz[selected_xyz]))
    colors[selected_xyz, 3] = 0.78
    if np.any(selected_xyz):
        ax.voxels(selected_xyz, facecolors=colors, edgecolor="none", shade=False)
    # Draw only the existing motion-tube block boundaries. Transparent faces
    # keep the information voxels visible wherever the two volumes overlap.
    if np.any(roi_xyz):
        ax.voxels(
            roi_xyz,
            facecolors=(0.0, 0.0, 0.0, 0.0),
            edgecolor=(0.0, 0.0, 0.0, 0.95),
            linewidth=0.7,
            shade=False,
        )
    ax.set(xlabel="x (64px block)", ylabel="t (frame)", zlabel="y (64px block)", title=title)
    ax.set_box_aspect((positive.shape[2], positive.shape[0], positive.shape[1]))
    ax.invert_zaxis()
    ax.view_init(elev=24, azim=-62)
    ax.grid(True, color="#9aa2ad", alpha=0.55)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(positive)
    fig.colorbar(mappable, ax=ax, shrink=0.58, pad=0.08, label="positive VLM information")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def render_heatmap(raw: np.ndarray, roi: np.ndarray, out: Path, title: str, quantile: float) -> None:
    positive = np.maximum(raw, 0.0)
    vmax = float(positive.max()) if positive.size else 1.0
    threshold = float(np.quantile(positive[positive > 0], quantile)) if quantile > 0 and np.any(positive > 0) else 0.0
    selected = positive > threshold if quantile > 0 else positive > 0
    indices = np.unique(np.linspace(0, raw.shape[0] - 1, min(6, raw.shape[0])).round().astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), squeeze=False, facecolor="#f7f9fc")
    for axis, index in zip(axes.flat, indices):
        image = np.where(selected[index], positive[index], 0.0)
        axis.imshow(image, origin="upper", cmap="turbo", vmin=0, vmax=max(vmax, 1e-12), interpolation="nearest")
        if roi[index].any():
            axis.contour(roi[index].astype(float), levels=[0.5], colors="black", linewidths=1.5)
        axis.set_title(f"t={index}", fontsize=10)
        axis.set_xlabel("x (64px block)", fontsize=8)
        axis.set_ylabel("y (64px block)", fontsize=8)
        axis.set_xticks(np.arange(0, positive.shape[2], max(1, positive.shape[2] // 6)))
        axis.set_yticks(np.arange(0, positive.shape[1], max(1, positive.shape[1] // 5)))
    for axis in axes.flat[len(indices):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.subplots_adjust(left=0.055, right=0.86, bottom=0.08, top=0.90, wspace=0.28, hspace=0.34)
    mappable = ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1e-12)), cmap="turbo")
    mappable.set_array(positive)
    colorbar_axis = fig.add_axes([0.89, 0.16, 0.022, 0.68])
    fig.colorbar(mappable, cax=colorbar_axis, label="positive VLM information")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_visualizations(raw: np.ndarray, roi: np.ndarray, out_dir: Path, video_id: str) -> None:
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for name, quantile in THRESHOLDS.items():
        render_3d(raw, roi, viz_dir / f"{name}_3d.png", f"{video_id} | {name} | VLM + ROI", quantile)
        render_heatmap(raw, roi, viz_dir / f"{name}_heatmap.png", f"{video_id} | {name} | time slices", quantile)


def load_existing_volume(existing_dir: Path) -> tuple[np.ndarray, list[dict], list[dict], str]:
    raw = np.load(existing_dir / "information_volume.npy")
    blocks = json.loads((existing_dir / "blocks.json").read_text(encoding="utf-8"))
    metadata = json.loads((existing_dir / "metadata.json").read_text(encoding="utf-8"))
    return raw, blocks, metadata["frames"], metadata["gt_caption"]


def rerender_existing(output_root: Path, cfg, video_ids: list[str]) -> None:
    """Regenerate figures from saved volumes without loading or running Qwen."""
    summaries = []
    for video_id in video_ids:
        out_dir = output_root / video_id
        raw = np.load(out_dir / "information_volume.npy")
        blocks = []
        rows = []
        with (out_dir / "blocks.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
                blocks.append({
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp": float(row["timestamp"]),
                    "block_row": int(row["block_row"]),
                    "block_col": int(row["block_col"]),
                    "x2": (int(row["block_col"]) + 1) * 64,
                    "y2": (int(row["block_row"]) + 1) * 64,
                })
        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        sample, paths = source_sample(cfg, video_id)
        with Image.open(paths[0]) as image:
            original_width, original_height = image.size
        mapped_width = max(int(block["x2"]) for block in blocks)
        mapped_height = max(int(block["y2"]) for block in blocks)
        frames = [
            {"frame_idx": index, "timestamp": timestamp, "original_width": original_width,
             "original_height": original_height, "mapped_width": mapped_width,
             "mapped_height": mapped_height, "block_rows": raw.shape[1],
             "block_cols": raw.shape[2]}
            for index, timestamp in enumerate(metadata["timestamps"])
        ]
        track_path = tracks_json_path(Path(cfg.feasibility.tracks_root), video_id)
        roi = project_roi_mask(blocks, frames, load_roi_cells(track_path), Path(cfg.feasibility.datasets.video_root) / f"{video_id}.mp4")
        attach_roi(blocks, roi)
        for row, block in zip(rows, blocks):
            row["in_roi"] = str(block["in_roi"])
        with (out_dir / "blocks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"video_id": video_id, "num_frames": int(raw.shape[0]), "num_blocks": int(raw.size), **metrics(raw, roi)}
        dump(out_dir / "summary.json", summary)
        save_visualizations(raw, roi, out_dir, video_id)
        summaries.append(summary)
    fields = ["video_id", "num_frames", "num_blocks", "ROI_volume_ratio", "ROI_information_coverage", "information_enrichment", "ROI_density", "nonROI_density", "density_ratio"]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: summary.get(field) for field in fields} for summary in summaries)
    numeric = {field: [float(summary[field]) for summary in summaries if summary.get(field) is not None] for field in fields[1:]}
    dump(output_root / "summary.json", {"video_count": len(summaries), "videos": summaries,
        "mean": {field: float(np.mean(values)) if values else None for field, values in numeric.items()},
        "std": {field: float(np.std(values)) if values else None for field, values in numeric.items()}})


def analyze_video(captioner, cfg, config_raw, video_id: str, output_root: Path, existing_dir: Path | None, batch_size: int) -> dict:
    sample, paths = source_sample(cfg, video_id)
    video_path = Path(cfg.feasibility.datasets.video_root) / f"{video_id}.mp4"
    out_dir = output_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = gt_caption(config_raw, video_id)
    if existing_dir is not None and (existing_dir / "information_volume.npy").is_file():
        raw, blocks, frames, caption = load_existing_volume(existing_dir)
        source = "reused_task1_volume"
    else:
        prompt = load_job_prompt(cfg, cfg.jobs[0]) if cfg.jobs else "Describe the entire video in one caption."
        inputs, visual, predictors, target_ids, blocks, frames, _ = _prepare_inputs(captioner, sample, prompt, caption, cfg.video.fps)
        engine = __import__("vlm_inference.token_information", fromlist=["CachedQwenSuffix"]).CachedQwenSuffix(captioner.model, [4], visual, predictors, target_ids)
        inputs = inputs.to(next(captioner.model.parameters()).device)
        engine.capture(inputs)
        baseline = engine.resume(4, [[]])[0].numpy()
        values = []
        for start in range(0, len(blocks), batch_size):
            scores = engine.resume(4, [block["token_indices"] for block in blocks[start:start + batch_size]]).numpy()
            values.extend((scores.mean(axis=1) - baseline.mean()).tolist())
        raw = np.full((len(frames), max(f["block_rows"] for f in frames), max(f["block_cols"] for f in frames)), np.nan, dtype=np.float32)
        for block, value in zip(blocks, values):
            block["raw_information"] = float(value)
            raw[block["frame_idx"], block["block_row"], block["block_col"]] = float(value)
        source = "computed_task1_layer4"
    track_path = tracks_json_path(Path(cfg.feasibility.tracks_root), video_id)
    cells = load_roi_cells(track_path)
    roi = project_roi_mask(blocks, frames, cells, video_path)
    attach_roi(blocks, roi)
    positive = np.maximum(raw, 0.0)
    np.save(out_dir / "information_volume.npy", raw)
    np.save(out_dir / "positive_information_volume.npy", positive)
    fields = ["video_id", "timestamp", "frame_idx", "block_row", "block_col", "token_indices", "visual_token_indices", "raw_information", "positive_information", "in_roi"]
    with (out_dir / "blocks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block in blocks:
            writer.writerow({"video_id": video_id, "timestamp": block["timestamp"], "frame_idx": block["frame_idx"], "block_row": block["block_row"], "block_col": block["block_col"], "token_indices": json.dumps(block["token_indices"]), "visual_token_indices": json.dumps(block["visual_token_indices"]), "raw_information": block["raw_information"], "positive_information": max(float(block["raw_information"]), 0.0), "in_roi": block["in_roi"]})
    summary = {"video_id": video_id, "num_frames": int(raw.shape[0]), "num_blocks": int(raw.size), **metrics(raw, roi)}
    dump(out_dir / "summary.json", summary)
    dump(out_dir / "metadata.json", {"video_id": video_id, "gt_caption": caption, "layer": 4, "block_size": 64, "visual_tokens_per_block": 4, "source": source, "full_frame_sampling_dir": str(Path(cfg.feasibility.datasets.fps_1) / video_id), "roi_tracks": str(track_path), "roi_coordinate_system": "original 64x64 cells projected by mapped/original pixel scale to Qwen blocks", "shape": list(raw.shape), "roi_mask_shape": list(roi.shape), "timestamps": [frame["timestamp"] for frame in frames], "raw_information_file": "information_volume.npy", "positive_information_file": "positive_information_volume.npy", "summary": summary})
    save_visualizations(raw, roi, out_dir, video_id)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/video_information_volume_9videos.yaml")
    parser.add_argument("--video-id", action="append")
    parser.add_argument("--output-root", default="outputs/vlm_roi_alignment")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-task1", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Regenerate figures from saved volumes without Qwen inference")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config).resolve()
    config_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = load_config(config_path)
    ids = args.video_id or VIDEO_IDS
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        rerender_existing(output_root, cfg, ids)
        LOG.info("Rerendered %d video visualization directories", len(ids))
        return
    existing = Path("outputs/video_information_volume") if args.reuse_task1 else None
    captioner = None if existing and len(ids) == 1 else Qwen3VLCaptioner(cfg)
    summaries = []
    for video_id in ids:
        reuse = existing if existing and video_id == "VIRAT_S_000201_06_001354_001397" else None
        summaries.append(analyze_video(captioner, cfg, config_raw, video_id, output_root, reuse, args.batch_size))
    fields = ["video_id", "num_frames", "num_blocks", "ROI_volume_ratio", "ROI_information_coverage", "information_enrichment", "ROI_density", "nonROI_density", "density_ratio"]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: summary.get(field) for field in fields} for summary in summaries)
    numeric = {field: [float(summary[field]) for summary in summaries if summary.get(field) is not None] for field in fields[1:]}
    aggregate = {"video_count": len(summaries), "videos": summaries, "mean": {field: float(np.mean(values)) if values else None for field, values in numeric.items()}, "std": {field: float(np.std(values)) if values else None for field, values in numeric.items()}}
    dump(output_root / "summary.json", aggregate)
    LOG.info("Wrote %d video summaries to %s", len(summaries), output_root)


if __name__ == "__main__":
    main()