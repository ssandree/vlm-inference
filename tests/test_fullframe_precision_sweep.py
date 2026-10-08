"""Precision-sweep config: 11 videos, 5 presets, full-frame vs proposed ROI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_inference.fullframe_motion_multiroi import (  # noqa: E402
    PRECISION_PRESETS,
    discover_model_sizes,
    load_sweep_request,
)

CONFIG = ROOT / "configs" / "fullframe_proposed_roi_5model.yaml"


class PrecisionSweepConfigTests(unittest.TestCase):
    def test_yaml_selects_eleven_videos_and_five_presets(self) -> None:
        request = load_sweep_request(CONFIG)
        self.assertEqual(list(request.models), list(PRECISION_PRESETS))
        self.assertEqual(len(request.videos or []), 11)
        self.assertEqual(
            request.conditions, ["full_frame", "full_frame_2fps", "motion_roi"]
        )
        self.assertEqual(request.prompts["full_frame"], "prompts/frames_full.txt")
        self.assertEqual(request.prompts["full_frame_2fps"], "prompts/frames_full.txt")
        self.assertEqual(
            request.prompts["motion_roi"], "prompts/frames_roi_per_tube.txt"
        )
        self.assertTrue(request.cfg.output.dir.endswith("fullframe_proposed_roi_5model"))

    def test_five_presets_resolve_to_local_checkpoints(self) -> None:
        found = discover_model_sizes(presets=PRECISION_PRESETS)
        self.assertEqual([item["name"] for item in found], list(PRECISION_PRESETS))
        by_name = {item["name"]: item for item in found}
        self.assertEqual(
            by_name["8b"]["path"], by_name["8b-int8"]["path"]
        )
        self.assertTrue(by_name["8b-int8"]["load_in_8bit"])
        self.assertFalse(by_name["8b"]["load_in_8bit"])
        self.assertFalse(by_name["8b-fp8"]["load_in_8bit"])
        self.assertIn("FP8", by_name["8b-fp8"]["path"])
        self.assertIn("FP8", by_name["30b-fp8"]["path"])
        self.assertNotIn("FP8", by_name["30b"]["path"])

    def test_59_video_fp16_config(self) -> None:
        request = load_sweep_request(ROOT / "configs" / "fullframe_proposed_roi_59_8b_fp16.yaml")
        self.assertEqual(request.models, ["8b-fp16"])
        self.assertEqual(len(request.videos or []), 59)
        self.assertEqual(request.videos[0], "VIRAT_S_000001")
        self.assertEqual(request.conditions, ["full_frame", "motion_roi"])
        found = discover_model_sizes(presets=request.models)
        self.assertEqual(found[0]["dtype"], "float16")
        self.assertTrue(found[0]["path"].endswith("Qwen3-VL-8B-Instruct-BF16"))

    def test_2fps_config_writes_into_the_five_model_runs(self) -> None:
        request = load_sweep_request(ROOT / "configs" / "fullframe_2fps_5model.yaml")
        self.assertEqual(list(request.models), list(PRECISION_PRESETS))
        self.assertEqual(len(request.videos or []), 11)
        self.assertEqual(request.conditions, ["full_frame_2fps"])
        self.assertEqual(request.prompts["full_frame_2fps"], "prompts/frames_full.txt")
        self.assertEqual(request.cfg.video.fps, 2.0)
        self.assertTrue(request.cfg.output.dir.endswith("fullframe_proposed_roi_5model"))


if __name__ == "__main__":
    unittest.main()
