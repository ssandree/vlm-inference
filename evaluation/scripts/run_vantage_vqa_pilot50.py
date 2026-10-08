#!/usr/bin/env python3
"""VANTAGE VQA pilot50 runner.

Default: dry-run validation only (no Qwen). Optional --smoke-test runs the
first QA through all 5 conditions with a single Qwen load. Full 50x5
inference requires both --full and --confirm-full-pilot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
VLM_INF_SRC = Path("/home/ssandree/Motion-guided-ROI/vlm-inference/src")
for path in (str(EVAL_ROOT), str(VLM_INF_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vlm_inference.config import apply_model_override, apply_output_dir_override, load_config
from vlm_inference.gt_action_cls import seed_everything

from vantage_vqa.runner import (
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_VIDEO_ROOT,
    dry_run_validate,
    load_pilot_items,
    prepare_unique_videos,
    run_full_pilot,
    smoke_test_first_qa,
    write_json,
)
from vantage_vqa.visual import USER_CONDITIONS, assert_visual_builder_isolated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument(
        "--confirm-full-pilot",
        action="store_true",
        default=False,
        help="Required together with --full to start the 50x5 VANTAGE VQA pilot.",
    )
    parser.add_argument("--skip-frames", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.full:
        if not args.confirm_full_pilot:
            raise SystemExit(
                "--full requires --confirm-full-pilot to start the 50x5 VANTAGE VQA pilot."
            )
    elif not args.dry_run and not args.smoke_test:
        args.dry_run = True

    cfg = load_config(args.config)
    if args.model:
        apply_model_override(cfg, args.model)
    apply_output_dir_override(cfg, args.output_dir)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_pilot_items(args.manifest)
    assert_visual_builder_isolated()

    if args.dry_run:
        prepared = prepare_unique_videos(
            items,
            output_dir=output_dir,
            video_root=args.video_root,
            cfg=cfg,
            jpeg_quality=95,
            skip_frames=args.skip_frames,
            skip_motion=args.skip_motion,
        )
        report = dry_run_validate(items, prepared)
        write_json(output_dir / "dry_run_validation.json", report)
        print(f"dry-run passed={report['passed']} issues={len(report['issues'])}")
        print(f"  n_qa={report['n_qa']} unique_videos={report['n_unique_videos']}")
        print(f"  conditions={list(USER_CONDITIONS)}")
        print(f"  token_accounting={report['token_accounting']}")
        print(f"  wrote {output_dir / 'dry_run_validation.json'}")
        if not report["passed"]:
            for issue in report["issues"]:
                print(f"  ISSUE: {issue}")
            return 1
        if not args.smoke_test and not args.full:
            return 0

    if args.smoke_test:
        # First QA only: reuse motion/frame cache, then one Qwen load.
        prepared = prepare_unique_videos(
            items[:1],
            output_dir=output_dir,
            video_root=args.video_root,
            cfg=cfg,
            jpeg_quality=95,
            skip_frames=args.skip_frames,
            skip_motion=args.skip_motion,
        )
        smoke = smoke_test_first_qa(items, prepared, cfg, output_dir=output_dir)
        print(f"smoke-test model_load_count={smoke['model_load_count']}")
        for row in smoke["rows"]:
            print(
                f"  {row['condition']}: letter={row['prediction_letter']} "
                f"images={row['num_input_images']} tokens={row['visual_tokens']} "
                f"pixels={row['processed_pixels']} build={row['input_build_time']:.2f}s "
                f"vlm={row['vlm_time']:.2f}s"
            )
        print(f"  wrote {output_dir / 'smoke_test_first_qa.json'}")
    if args.full:
        prepared = prepare_unique_videos(
            items,
            output_dir=output_dir,
            video_root=args.video_root,
            cfg=cfg,
            jpeg_quality=95,
            skip_frames=args.skip_frames,
            skip_motion=args.skip_motion,
        )
        summary = run_full_pilot(items, prepared, cfg, output_dir=output_dir)
        print(f"full-pilot completed={summary['completed']}/{summary['n_total']}")
        print(
            f"  errors={summary['errors']} skipped={summary['skipped']} "
            f"parse_failures={summary['parse_failures']} "
            f"model_load_count={summary['model_load_count']}"
        )
        for condition, stats in summary["conditions"].items():
            dist = stats["prediction_distribution"]
            print(
                f"  {condition}: {stats['completed']}/{stats['n']} "
                f"mean_images={stats['mean_images']} "
                f"mean_tokens={stats['mean_visual_tokens']} "
                f"mean_pixels={stats['mean_processed_pixels']} "
                f"mean_vlm={stats['mean_vlm_time']} "
                f"dist=A:{dist['A']} B:{dist['B']} C:{dist['C']} D:{dist['D']}"
            )
        print(f"  wrote {output_dir / 'predictions.json'}")
        print(f"  wrote {output_dir / 'full_pilot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
