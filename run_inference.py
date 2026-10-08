#!/usr/bin/env python3
"""CLI entrypoint for VLM caption inference."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_inference.config import (
    apply_model_override,
    apply_output_dir_override,
    list_model_presets,
    load_config,
)
from vlm_inference.runner import run_inference


def parse_args() -> argparse.Namespace:
    try:
        preset_help = ", ".join(list_model_presets())
    except Exception:
        preset_help = "8b, 30b, 8b-fp8, 30b-fp8"
    parser = argparse.ArgumentParser(
        description="Qwen3-VL caption inference (Transformers + YAML config)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="SPEC",
        help=(
            "Swap VLM weights without editing the experiment YAML. "
            f"Presets: {preset_help}. "
            "Also accepts a local weights path such as "
            "/data/models/Qwen3-VL-30B-A3B-Instruct."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Override output.dir (feasibility cache is moved under this directory).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Feasibility: validate inputs and print video/method stats without loading the VLM",
    )
    parser.add_argument(
        "--prepare-single-motion",
        action="store_true",
        help="Feasibility: build single-motion union ROI cache from proposed tubes (no VLM)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Feasibility: ignore existing inference cache and re-run all jobs",
    )
    parser.add_argument(
        "--input-order",
        "--input_order",
        dest="input_order",
        choices=["timestamp", "grouped", "global_anchor", "tube_story"],
        default=None,
        help=(
            "Experiment VLM image order: "
            "'timestamp' = Global+ROI interleaved by time; "
            "'grouped' = all Global then ROI1, ROI2, ...; "
            "'global_anchor' = each Global then ROI1, ROI2, ... in [G_i, G_{i+1}); "
            "'tube_story' = each tube Global then that tube's full ROI sequence "
            "(overrides experiment.input_order in config)"
        ),
    )
    parser.add_argument(
        "--roi-frames",
        "--roi_frames",
        dest="roi_frames",
        choices=["default", "fps_1"],
        default=None,
        help=(
            "Which ROI JPEG set to use: "
            "'default' = metadata frames (~5/3 fps); "
            "'fps_1' = metadata frames_fps_1 (*_fps_1.jpg). "
            "Overrides experiment.roi_frames in config."
        ),
    )
    parser.add_argument(
        "--global-insert",
        dest="global_insert",
        choices=["video_start", "start_only", "start_end", "start_mid_end"],
        default=None,
        help=(
            "Full-frame insertion: "
            "video_start / start_only / start_end / start_mid_end. "
            "Overrides experiment.global_insert. If the config lists dedicated "
            "ablation methods, only the matching condition is run."
        ),
    )
    parser.add_argument(
        "--global-insert-source",
        dest="global_insert_source",
        choices=["fps_1", "fps_2", "fps_0_25"],
        default=None,
        help="Sampled Full-frame pool used to snap ROI timestamps (default: fps_1).",
    )
    parser.add_argument(
        "--save-vlm-inputs",
        dest="save_vlm_inputs",
        action="store_true",
        help="Copy actual VLM input frames and write board images under output.dir/vlm_inputs/",
    )
    parser.add_argument(
        "--roi-sampling",
        dest="roi_sampling",
        choices=[
            "spatial_occupancy",
            "piecewise_fps",
            "fixed_1fps",
            "fixed_2fps",
            "adaptive_0.5_1_2",
            "adaptive_0.5_1_4",
            "accumulated_motion",
            "accumulated_motion_cap2fps",
            "burst_2fps",
            "centroid_2fps",
        ],
        default=None,
        help="ROI motion-adaptive sampling policy (overrides experiment.roi_sampling).",
    )
    parser.add_argument(
        "--roi-sampling-tau",
        dest="roi_sampling_tau",
        type=float,
        default=None,
        help="Accumulated-motion threshold tau (overrides experiment.roi_sampling_tau).",
    )
    parser.add_argument(
        "--overwrite-roi-sampling",
        dest="overwrite_roi_sampling",
        action="store_true",
        help="Regenerate motion-adaptive ROI crops even if they already exist.",
    )
    parser.add_argument(
        "--roi-repr",
        dest="roi_repr",
        choices=["separate_roi", "union_roi"],
        default=None,
        help="ROI representation: separate_roi or union_roi (overrides experiment.roi_repr).",
    )
    parser.add_argument(
        "--overwrite-roi-repr",
        dest="overwrite_roi_repr",
        action="store_true",
        help="Regenerate Separate/Union ROI crops even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config(args.config)
    if args.model is not None:
        model = apply_model_override(cfg, args.model)
        logging.getLogger(__name__).info(
            "Using model=%s path=%s (CLI override)",
            model.name or model.path,
            model.path,
        )
    if args.output_dir is not None:
        apply_output_dir_override(cfg, args.output_dir)
        logging.getLogger(__name__).info(
            "Using output.dir=%s (CLI override)", cfg.output.dir
        )
    if args.input_order is not None:
        cfg.experiment.input_order = args.input_order
        logging.getLogger(__name__).info(
            "Using input_order=%s (CLI override)", args.input_order
        )
    if args.roi_frames is not None:
        cfg.experiment.roi_frames = args.roi_frames
        logging.getLogger(__name__).info(
            "Using roi_frames=%s (CLI override)", args.roi_frames
        )
    if args.global_insert is not None:
        cfg.experiment.global_insert = args.global_insert
        logging.getLogger(__name__).info(
            "Using global_insert=%s (CLI override)", args.global_insert
        )
        if cfg.feasibility is not None:
            from vlm_inference.feasibility import filter_methods_for_global_insert

            filtered = filter_methods_for_global_insert(
                cfg.feasibility.methods, args.global_insert
            )
            if filtered != list(cfg.feasibility.methods):
                cfg.feasibility.methods = filtered
                logging.getLogger(__name__).info(
                    "Filtered methods to %s", filtered
                )
    if args.global_insert_source is not None:
        cfg.experiment.global_insert_source = args.global_insert_source
        logging.getLogger(__name__).info(
            "Using global_insert_source=%s (CLI override)", args.global_insert_source
        )
    if args.save_vlm_inputs:
        cfg.experiment.save_vlm_inputs = True
        logging.getLogger(__name__).info("save_vlm_inputs=True (CLI override)")
    if args.roi_sampling is not None:
        cfg.experiment.roi_sampling = args.roi_sampling
        logging.getLogger(__name__).info(
            "Using roi_sampling=%s (CLI override)", args.roi_sampling
        )
        if cfg.feasibility is not None:
            from vlm_inference.motion_adaptive import filter_methods_for_roi_sampling

            filtered = filter_methods_for_roi_sampling(
                cfg.feasibility.methods, args.roi_sampling
            )
            if filtered != list(cfg.feasibility.methods):
                cfg.feasibility.methods = filtered
                logging.getLogger(__name__).info("Filtered methods to %s", filtered)
    if args.roi_sampling_tau is not None:
        cfg.experiment.roi_sampling_tau = float(args.roi_sampling_tau)
        logging.getLogger(__name__).info(
            "Using roi_sampling_tau=%s (CLI override)", args.roi_sampling_tau
        )
    if args.overwrite_roi_sampling:
        cfg.experiment.roi_sampling_overwrite = True
    if args.roi_repr is not None:
        cfg.experiment.roi_repr = args.roi_repr
        logging.getLogger(__name__).info(
            "Using roi_repr=%s (CLI override)", args.roi_repr
        )
        if cfg.feasibility is not None:
            from vlm_inference.roi_union import filter_methods_for_roi_repr

            filtered = filter_methods_for_roi_repr(
                cfg.feasibility.methods, args.roi_repr
            )
            if filtered != list(cfg.feasibility.methods):
                cfg.feasibility.methods = filtered
                logging.getLogger(__name__).info("Filtered methods to %s", filtered)
    if args.overwrite_roi_repr:
        cfg.experiment.roi_repr_overwrite = True

    if args.prepare_single_motion:
        if cfg.feasibility is None:
            raise SystemExit("--prepare-single-motion requires a feasibility config")
        from vlm_inference.single_motion import prepare_single_motion_cache

        rows = prepare_single_motion_cache(
            cfg.feasibility,
            overwrite=(cfg.experiment.roi_frames == "fps_1"),
            roi_frames=cfg.experiment.roi_frames,
        )
        print(
            f"Prepared single-motion ROI for {len(rows)} video(s) -> "
            f"{cfg.feasibility.datasets.single_motion_roi} "
            f"(roi_frames={cfg.experiment.roi_frames})"
        )
        if not args.dry_run:
            return

    out = run_inference(
        cfg,
        dry_run=args.dry_run,
        resume=False if args.no_resume else None,
    )
    print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
