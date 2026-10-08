"""Centroid displacement 1fps/2fps ROI sampling."""
import math
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_inference.centroid_motion import (
    CentroidMotionConfig,
    analyze_centroid_motion,
    centroid_displacement,
    union_intervals,
)
from vlm_inference.global_insert import (
    GLOBAL_INSERT_VIDEO_START,
    ROI_GLOBAL_INSERT_METHODS,
)
from vlm_inference.motion_adaptive import (
    ROI_CENTROID_2FPS,
    RoiMotionRecord,
    assign_policy,
    sampling_mode_for_method,
    select_piecewise,
)


SQUARE = (0, 0, 100, 100)


def record(t0=0.0, t1=10.0, analysis=None):
    rec = RoiMotionRecord(
        video_id="test",
        roi_id=1,
        tube_id=1,
        t0_stage3=0,
        t1_stage3=50,
        t0_sec=t0,
        t1_sec=t1,
        motion_score=0,
        motion_sum=0,
        n_motion_frames=0,
        native_fps=10,
        frame_count=101,
        bbox_px=SQUARE,
    )
    rec.centroid_analysis = analysis
    return rec


class CentroidDisplacementTests(unittest.TestCase):
    def test_displacement_has_no_delta_t(self):
        d_short = centroid_displacement((0.5, 0.5), (0.8, 0.5), SQUARE)
        expected = (0.3 * 100) / math.hypot(100, 100)
        self.assertAlmostEqual(d_short, expected)
        ts_fast = [0.0, 0.2]
        ts_slow = [0.0, 1.0]
        xy = [[0.5, 0.5], [0.8, 0.5]]
        cfg = CentroidMotionConfig(centroid_displacement_threshold=1.0)
        a = analyze_centroid_motion(ts_fast, xy, SQUARE, cfg)
        b = analyze_centroid_motion(ts_slow, xy, SQUARE, cfg)
        self.assertAlmostEqual(a["frames"][1]["displacement"], b["frames"][1]["displacement"])
        self.assertAlmostEqual(a["frames"][1]["displacement"], expected)

    def test_missing_centroid_is_not_origin_jump(self):
        ts = np.arange(5) * 0.2
        xy = np.full((5, 2), np.nan)
        xy[0] = (0.2, 0.2)
        xy[2] = (0.2, 0.2)
        xy[4] = (0.8, 0.8)
        out = analyze_centroid_motion(ts, xy, SQUARE)
        self.assertIsNone(out["frames"][0]["displacement"])
        self.assertIsNone(out["frames"][1]["displacement"])
        self.assertFalse(out["frames"][1]["is_trigger"])
        self.assertAlmostEqual(out["frames"][2]["displacement"], 0.0)
        self.assertEqual(out["frames"][2]["prev_valid_centroid_timestamp"], 0.0)
        origin_jump = math.hypot(0.8 * 100, 0.8 * 100) / math.hypot(100, 100)
        self.assertLess(out["frames"][4]["displacement"], origin_jump - 1e-6)
        self.assertEqual(out["frames"][4]["prev_valid_centroid_timestamp"], 0.4)

    def test_no_trailing_mean_or_speed_fields(self):
        ts = np.arange(6) * 0.2
        xy = np.repeat([[0.4, 0.6]], 6, axis=0)
        xy[4] = (0.9, 0.6)
        out = analyze_centroid_motion(ts, xy, SQUARE)
        for row in out["frames"]:
            self.assertNotIn("speed", row)
            self.assertNotIn("score", row)
        self.assertIsNone(getattr(CentroidMotionConfig, "lookback_sec", None))

    def test_trigger_only_when_displacement_reaches_tau(self):
        ts = [0.0, 0.2, 0.4]
        xy = [[0.0, 0.0], [0.01, 0.0], [0.5, 0.0]]
        cfg = CentroidMotionConfig(centroid_displacement_threshold=0.1)
        out = analyze_centroid_motion(ts, xy, SQUARE, cfg)
        d1 = out["frames"][1]["displacement"]
        d2 = out["frames"][2]["displacement"]
        self.assertLess(d1, 0.1)
        self.assertGreaterEqual(d2, 0.1)
        self.assertFalse(out["frames"][1]["is_trigger"])
        self.assertTrue(out["frames"][2]["is_trigger"])
        self.assertEqual(out["trigger_timestamps"], [0.4])

    def test_each_trigger_is_plus_minus_half_second_clipped(self):
        ts = np.arange(21) * 0.2
        xy = np.repeat([[0.2, 0.5]], 21, axis=0)
        xy[10] = (0.9, 0.5)
        xy[11:] = (0.9, 0.5)
        cfg = CentroidMotionConfig(centroid_displacement_threshold=0.02, pad_sec=0.5)
        out = analyze_centroid_motion(ts, xy, SQUARE, cfg, t0=0.0, t1=4.0)
        self.assertEqual(out["trigger_timestamps"], [2.0])
        window = out["high_fps_intervals"][0]
        self.assertAlmostEqual(window["start_sec"], 1.5)
        self.assertAlmostEqual(window["end_sec"], 2.5)
        near_start = analyze_centroid_motion(
            [0.0, 0.2], [[0.1, 0.1], [0.9, 0.1]], SQUARE, cfg, t0=0.0, t1=0.2
        )
        self.assertEqual(near_start["trigger_timestamps"], [0.2])
        self.assertAlmostEqual(near_start["high_fps_intervals"][0]["start_sec"], 0.0)
        self.assertAlmostEqual(near_start["high_fps_intervals"][0]["end_sec"], 0.2)

    def test_overlapping_and_touching_intervals_union(self):
        merged = union_intervals(
            [
                {"start_sec": 1.5, "end_sec": 2.5, "fps": 2.0},
                {"start_sec": 2.3, "end_sec": 3.3, "fps": 2.0},
            ],
            fps=2.0,
        )
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["start_sec"], 1.5)
        self.assertAlmostEqual(merged[0]["end_sec"], 3.3)
        touching = union_intervals(
            [
                {"start_sec": 1.5, "end_sec": 2.5, "fps": 2.0},
                {"start_sec": 2.5, "end_sec": 3.5, "fps": 2.0},
            ],
            fps=2.0,
        )
        self.assertEqual(len(touching), 1)
        self.assertAlmostEqual(touching[0]["end_sec"], 3.5)
        gapped = union_intervals(
            [
                {"start_sec": 1.0, "end_sec": 2.0, "fps": 2.0},
                {"start_sec": 3.1, "end_sec": 4.1, "fps": 2.0},
            ],
            fps=2.0,
        )
        self.assertEqual(len(gapped), 2)

    def test_sampling_is_1fps_outside_and_2fps_inside(self):
        analysis = {
            "config": {
                "base_fps": 1.0,
                "high_fps": 2.0,
                "centroid_displacement_threshold": 0.02,
            },
            "high_fps_intervals": [
                {"start_sec": 4.0, "end_sec": 5.0, "fps": 2.0},
            ],
        }
        rec = record(0.0, 8.0, analysis)
        assign_policy([rec], mode=ROI_CENTROID_2FPS, tau=8)
        stamps = [s.timestamp for s in rec.selected]
        inside = [t for t in stamps if 4.0 <= t <= 5.0]
        outside = [t for t in stamps if t < 4.0 or t > 5.0]
        inside_gaps = [round(b - a, 5) for a, b in zip(inside, inside[1:])]
        outside_gaps = [round(b - a, 5) for a, b in zip(outside, outside[1:]) if b <= 4.0 or a >= 5.0]
        self.assertTrue(any(abs(g - 0.5) < 1e-6 for g in inside_gaps))
        self.assertTrue(all(g >= 0.5 - 1e-6 for g in inside_gaps))
        left = [t for t in stamps if t <= 4.0]
        left_gaps = [round(b - a, 5) for a, b in zip(left, left[1:])]
        self.assertTrue(all(abs(g - 1.0) < 1e-6 or abs(g - 0.5) < 1e-6 for g in left_gaps))

    def test_selected_rate_never_exceeds_2fps_grid(self):
        rec = record(0.0, 5.0)
        stamps = [
            s.timestamp
            for s in select_piecewise(
                rec,
                default_fps=1.0,
                intervals=[{"start_sec": 1.0, "end_sec": 3.0, "fps": 2.0}],
            )
        ]
        gaps = [round(b - a, 5) for a, b in zip(stamps, stamps[1:])]
        self.assertTrue(all(g >= 0.5 - 1e-6 for g in gaps))
        self.assertEqual(stamps, [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

    def test_named_methods_use_video_start_and_centroid_mode(self):
        self.assertEqual(
            ROI_GLOBAL_INSERT_METHODS["global_roi_video_start_roi_centroid_2fps"],
            GLOBAL_INSERT_VIDEO_START,
        )
        self.assertEqual(
            sampling_mode_for_method("global_roi_video_start_roi_centroid_2fps", None),
            ROI_CENTROID_2FPS,
        )


if __name__ == "__main__":
    unittest.main()
