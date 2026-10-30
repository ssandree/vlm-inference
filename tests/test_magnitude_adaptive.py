"""Within-tube magnitude FPS, without video files or a VLM."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_inference.frame_clock import encode_timestamps, rewrite_fetched_metadata
from vlm_inference.magnitude_config import MagnitudeAdaptiveConfig
from vlm_inference.motion_adaptive import (
    ROI_CENTROID_2FPS,
    ROI_FIXED_1FPS,
    ROI_FIXED_2FPS,
    ROI_MAGNITUDE_ADAPTIVE,
    MotionSample,
    RoiMotionRecord,
    assign_policy,
    select_motion_magnitude_adaptive,
)


def _record(series):
    return RoiMotionRecord(
        video_id="synthetic",
        roi_id=1,
        tube_id=1,
        t0_stage3=0,
        t1_stage3=len(series) - 1,
        t0_sec=float(series[0].timestamp),
        t1_sec=float(series[-1].timestamp),
        motion_score=1.0,
        motion_sum=1.0,
        n_motion_frames=len(series),
        native_fps=10.0,
        frame_count=501,
        series=tuple(series),
    )


def _series():
    rows = []
    for index, stamp in enumerate([i * 0.2 for i in range(21)]):
        if stamp < 2.0:
            magnitude = 0.2
        elif stamp < 3.0:
            magnitude = 3.0
        else:
            magnitude = 8.0
        rows.append(
            MotionSample(
                stage3_index=index,
                timestamp=stamp,
                native_frame=int(round(stamp * 10)),
                magnitude=magnitude,
            )
        )
    return rows


class MagnitudeAdaptiveTests(unittest.TestCase):
    def test_rates_follow_absolute_magnitude(self):
        rec = _record(_series())
        selected = select_motion_magnitude_adaptive(rec, MagnitudeAdaptiveConfig())
        times = [round(stamp.timestamp, 1) for stamp in selected]
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 4.0)
        self.assertEqual(len(times), len(set(times)))
        # Quiet prefix is one 0.5 fps segment, so the 0.2 s grid is not kept.
        self.assertNotIn(0.2, times)
        self.assertNotIn(1.0, times)
        # High tail is 2 fps: 3.0, 3.5, 4.0.
        self.assertIn(3.5, times)
        self.assertIn(4.0, times)

    def test_existing_modes_still_assign(self):
        rec = _record(_series())
        fixed = assign_policy([rec], mode=ROI_FIXED_1FPS, tau=8.0)[0]
        self.assertEqual(fixed.assigned_mode, ROI_FIXED_1FPS)
        self.assertAlmostEqual(fixed.selected[0].timestamp, 0.0)
        self.assertAlmostEqual(fixed.selected[-1].timestamp, 4.0)
        dense = assign_policy([_record(_series())], mode=ROI_FIXED_2FPS, tau=8.0)[0]
        self.assertGreater(len(dense.selected), len(fixed.selected))
        adaptive = assign_policy(
            [_record(_series())], mode=ROI_MAGNITUDE_ADAPTIVE, tau=8.0
        )[0]
        self.assertEqual(adaptive.assigned_mode, ROI_MAGNITUDE_ADAPTIVE)
        self.assertNotEqual(
            [s.timestamp for s in adaptive.selected],
            [s.timestamp for s in fixed.selected],
        )

    def test_centroid_mode_name_unchanged(self):
        self.assertEqual(ROI_CENTROID_2FPS, "centroid_2fps")

    def test_timestamp_encoding_keeps_origin_and_irregular_gaps(self):
        stamps = [10.0, 11.0, 11.5, 12.0, 12.5, 13.5]
        indices, fps = encode_timestamps(stamps, scale=1000)
        restored = [index / fps for index in indices]
        self.assertEqual(restored, stamps)
        meta = rewrite_fetched_metadata(
            {"frames_indices": list(range(7)), "fps": 1.0},
            stamps,
        )
        self.assertEqual(len(meta["frames_indices"]), 7)
        self.assertEqual(meta["frames_indices"][-1], meta["frames_indices"][-2])
        decoded = [index / meta["fps"] for index in meta["frames_indices"]]
        self.assertEqual(decoded[:6], stamps)
        self.assertEqual(decoded[-1], 13.5)

    def test_two_tubes_do_not_share_selections(self):
        left = _record(_series())
        right = _record(_series())
        right.roi_id = 2
        right.series = tuple(
            MotionSample(
                stage3_index=sample.stage3_index,
                timestamp=sample.timestamp + 20.0,
                native_frame=sample.native_frame + 200,
                magnitude=0.1,
            )
            for sample in _series()
        )
        right.t0_sec = right.series[0].timestamp
        right.t1_sec = right.series[-1].timestamp
        assign_policy([left, right], mode=ROI_MAGNITUDE_ADAPTIVE, tau=8.0)
        self.assertLess(left.selected[-1].timestamp, 5.0)
        self.assertGreater(right.selected[0].timestamp, 19.0)
        self.assertLess(len(right.selected), len(left.selected))


if __name__ == "__main__":
    unittest.main()
