"""Save the exact VLM input frames and reuse the existing board renderer."""

from __future__ import annotations

import json
import logging
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import AppConfig
from .global_insert import (
    GlobalInsertPlan,
    ROI_GLOBAL_INSERT_METHODS,
)
from .metadata_format import ordered_frame_items

logger = logging.getLogger(__name__)


def _local_path(path: str | Path) -> Path:
    text = str(path)
    if text.startswith("file://"):
        parsed = urlparse(text)
        return Path(unquote(parsed.path))
    return Path(text)


def _time_token(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    if "." not in text:
        text = f"{text}.0"
    whole, _, frac = text.partition(".")
    return f"{int(whole):02d}.{frac}"


def vlm_input_stem(*, index: int, kind: str, roi_id: int | None, timestamp: float) -> str:
    t = _time_token(timestamp)
    if kind == "global":
        return f"{index:03d}__Global__t{t}s"
    rid = int(roi_id) if roi_id is not None else 0
    return f"{index:03d}__ROI{rid:02d}__t{t}s"


def condition_dir_name(
    method: str, mode: str | None, *, global_scale: float = 1.0
) -> str:
    if mode:
        from .global_insert import VALID_GLOBAL_INSERT_MODES, condition_dirname

        if mode in VALID_GLOBAL_INSERT_MODES:
            name = condition_dirname(mode)
        else:
            name = str(mode)
    else:
        name = ROI_GLOBAL_INSERT_METHODS.get(method, method)
    if abs(float(global_scale) - 1.0) > 1e-12:
        from .global_resolution import format_scale

        name = f"{name}__global_scale_{format_scale(global_scale)}"
    return name


def vlm_inputs_root(cfg: AppConfig, method: str, mode: str | None) -> Path:
    return Path(cfg.output.dir) / "vlm_inputs" / condition_dir_name(
        method, mode, global_scale=float(cfg.experiment.global_scale)
    )


def _parse_item_label(text: str) -> tuple[str, int | None, float]:
    """Parse '[Global] t=0.0s' / '[ROI1] t=3.5s' dump labels."""
    label, _, rest = text.partition(" t=")
    label = label.strip()
    time_s = float(rest.strip().rstrip("s"))
    if label == "[Global]":
        return "global", None, time_s
    if label.startswith("[ROI") and label.endswith("]"):
        return "roi", int(label[4:-1]), time_s
    raise ValueError(f"unrecognized frame label: {text}")


def collect_ordered_frames(inputs, *, input_order: str) -> list[tuple[str, Path]]:
    if inputs.bundle is not None:
        rows: list[tuple[str, Path]] = []
        for item in ordered_frame_items(inputs.bundle, input_order=input_order):
            if item.image_path is None:
                continue
            rows.append((item.text, Path(item.image_path)))
        return rows
    return [(frame.label, _local_path(frame.path)) for frame in inputs.frames]


def _plan_payload(plan: GlobalInsertPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "mode": plan.mode,
        "source_dir": plan.source_dir,
        "dedup_eps": plan.dedup_eps,
        "kept_timestamps": [float(frame.timestamp) for frame in plan.frames],
        "hits": [
            {
                "roi_id": hit.request.roi_id,
                "role": hit.request.role,
                "target_t": hit.request.target_t,
                "selected_t": float(hit.selected.timestamp),
                "selected_file": str(hit.selected.path),
                "kept": hit.kept,
                "skip_reason": hit.skip_reason,
            }
            for hit in plan.hits
        ],
    }


def export_vlm_inputs(
    *,
    cfg: AppConfig,
    video_id: str,
    method: str,
    inputs,
    input_order: str,
    plan: GlobalInsertPlan | None = None,
    mode: str | None = None,
) -> Path | None:
    """Copy VLM frames (in model order) and write grouped + order boards."""
    if inputs.errors:
        return None
    if inputs.bundle is None and not getattr(inputs, "frames", None):
        return None

    ordered = collect_ordered_frames(inputs, input_order=input_order)
    if not ordered:
        return None

    video_dir = vlm_inputs_root(cfg, method, mode) / video_id
    frames_dir = video_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    dump_lines = [f"# input_order={input_order}", f"# video_id={video_id}", f"# method={method}"]
    manifest_frames: list[dict[str, Any]] = []
    copied_items: list[tuple[str, Path]] = []

    for i, (label, src) in enumerate(ordered):
        src_path = Path(src).resolve()
        kind, roi_id, time_s = _parse_item_label(label)
        stem = vlm_input_stem(index=i, kind=kind, roi_id=roi_id, timestamp=time_s)
        dest = frames_dir / f"{stem}{src_path.suffix.lower() or '.jpg'}"
        shutil.copy2(src_path, dest)
        dump_lines.append(f"{label}  ->  {dest}")
        copied_items.append((label, dest))
        manifest_frames.append(
            {
                "index": i,
                "kind": kind,
                "roi_id": roi_id,
                "timestamp": time_s,
                "label": label,
                "filename": dest.name,
                "source": str(src_path),
            }
        )

    dump_path = video_dir / "input_order.txt"
    dump_path.write_text("\n".join(dump_lines).rstrip() + "\n", encoding="utf-8")

    manifest = {
        "video_id": video_id,
        "method": method,
        "input_order": input_order,
        "global_insert": mode or ROI_GLOBAL_INSERT_METHODS.get(method),
        "n_frames": len(manifest_frames),
        "frames_dir": str(frames_dir),
        "frames": manifest_frames,
        "global_insert_plan": _plan_payload(plan),
    }
    (video_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        _write_boards(
            video_id=video_id,
            method=method,
            items=copied_items,
            out_dir=video_dir,
            tube_labels=getattr(inputs, "roi_repr_labels", None)
            or getattr(inputs, "roi_sampling_labels", None),
            extra_sections=getattr(inputs, "roi_repr_extra_sections", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("board render failed for %s / %s: %s", video_id, method, exc)

    logger.info("Wrote VLM inputs under %s (%d frames)", video_dir, len(manifest_frames))
    return video_dir


def _viz_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "viz_vlm_input_boards.py"
    spec = importlib.util.spec_from_file_location("viz_vlm_input_boards", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_boards(
    *,
    video_id: str,
    method: str,
    items: list[tuple[str, Path]],
    out_dir: Path,
    tube_labels: dict[int, str] | None = None,
    extra_sections: list[tuple[str, list[Path]]] | None = None,
) -> None:
    viz = _viz_module()
    args = Namespace(
        thumb_h=72,
        frame_gap=8,
        row_gap=10,
        section_gap=36,
        label_w=210,
        pad=24,
        canvas_w=2400,
        section_pad=12,
    )
    grouped = viz.render_grouped_canvas(
        items,
        video_id=video_id,
        method=method,
        args=args,
        tube_labels=tube_labels,
        extra_sections=extra_sections,
    )
    order = viz.render_order_canvas(items, video_id=video_id, method=method, args=args)
    grouped_path = out_dir / "board_grouped.png"
    order_path = out_dir / "board_order.png"
    if not viz.cv2.imwrite(str(grouped_path), grouped):
        raise RuntimeError(f"failed to write {grouped_path}")
    if not viz.cv2.imwrite(str(order_path), order):
        raise RuntimeError(f"failed to write {order_path}")
    boards_dir = out_dir.parent / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)
    viz.cv2.imwrite(str(boards_dir / f"{video_id}__grouped.png"), grouped)
    viz.cv2.imwrite(str(boards_dir / f"{video_id}__order.png"), order)
    logger.info("Wrote boards %s and %s", grouped_path, order_path)
