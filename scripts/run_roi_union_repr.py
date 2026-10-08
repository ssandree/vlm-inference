#!/usr/bin/env python3
"""Prepare Separate vs Union ROI crops, logs, and boards (no VLM)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_inference.config import load_config
from vlm_inference.roi_union import VALID_ROI_REPR_MODES, prepare_roi_repr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/feasibility_roi_union.yaml",
    )
    p.add_argument(
        "--roi-repr",
        nargs="*",
        default=None,
        choices=sorted(VALID_ROI_REPR_MODES),
        help="Subset of conditions (default: all methods listed in the config).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config(args.config)
    summary = prepare_roi_repr(
        cfg,
        modes=args.roi_repr,
        overwrite=args.overwrite,
    )
    print(f"Wrote representation outputs under {cfg.output.dir}")
    for mode, payload in (summary.get("conditions") or {}).items():
        print(
            f"  {mode}: videos={payload.get('num_videos')} "
            f"roi_images={payload.get('num_roi_images')}"
        )


if __name__ == "__main__":
    main()
