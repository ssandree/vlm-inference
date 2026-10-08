#!/usr/bin/env python3
"""Build experiments/vantage/vantage_vqa_pilot50.json from official VANTAGE VQA data.

Does not run Qwen inference, motion analysis, or ROI generation.
Does not modify existing research algorithm code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vantage_vqa.adapter import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    write_pilot50_manifest,
)

DEFAULT_OUT = ROOT / "experiments" / "vantage" / "vantage_vqa_pilot50.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = write_pilot50_manifest(args.out, data_root=args.data_root)
    meta = payload["meta"]
    val = meta["validation"]
    print(json.dumps(
        {
            "out": str(args.out),
            "n_raw_qa": meta["full_dataset"]["n_raw_qa"],
            "n_tsv_qa": meta["full_dataset"]["n_tsv_qa"],
            "n_unique_videos_full": meta["full_dataset"]["n_unique_videos"],
            "n_pool200": meta["official_200_selection"]["n"],
            "n_pool200_videos": meta["official_200_selection"]["n_unique_videos"],
            "n_pilot": val["n_items"],
            "n_pilot_videos": val["n_unique_videos"],
            "validation_ok": val["ok"],
            "videos_with_multiple_qa": val["videos_with_multiple_qa"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
