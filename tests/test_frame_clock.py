"""Timestamp mode stays legacy unless a sample carries source times."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_inference.frame_clock import (
    TIMESTAMP_LEGACY,
    frame_timestamps_for_generate,
    source_video_timestamp_groups,
)
from vlm_inference.inputs import Sample, TubeGlobalRoiBlock


def _block():
    return TubeGlobalRoiBlock(
        roi_id=2,
        global_image_path="global.jpg",
        sequence_paths=["a.jpg", "b.jpg", "c.jpg"],
        global_timestamp=3.3375,
        sequence_timestamps=[3.3375, 4.3375, 4.8375],
    )


class FrameClockTests(unittest.TestCase):
    def test_legacy_does_not_request_source_times(self):
        sample = Sample(
            job_name="job",
            sample_id="clip",
            input_type="tube_global_roi_videos",
            paths=[],
            tube_global_roi_blocks=[_block()],
        )
        self.assertIsNone(
            frame_timestamps_for_generate(sample, TIMESTAMP_LEGACY, global_is_video=False)
        )
        self.assertIsNone(
            frame_timestamps_for_generate(sample, None, global_is_video=True)
        )

    def test_caption_packing_lists_roi_videos_only(self):
        sample = Sample(
            job_name="job",
            sample_id="clip",
            input_type="tube_global_roi_videos",
            paths=[],
            tube_global_roi_blocks=[_block()],
        )
        groups = source_video_timestamp_groups(sample, global_is_video=False)
        self.assertEqual(groups, [[3.3375, 4.3375, 4.8375]])

    def test_vqa_packing_puts_global_before_each_roi(self):
        sample = Sample(
            job_name="job",
            sample_id="clip",
            input_type="tube_global_roi_videos",
            paths=[],
            tube_global_roi_blocks=[_block()],
        )
        groups = source_video_timestamp_groups(sample, global_is_video=True)
        self.assertEqual(groups, [[3.3375], [3.3375, 4.3375, 4.8375]])

    def test_unknown_mode_is_rejected(self):
        sample = Sample("job", "clip", "frames", [])
        with self.assertRaises(ValueError):
            frame_timestamps_for_generate(sample, "absolute", global_is_video=False)


if __name__ == "__main__":
    unittest.main()
