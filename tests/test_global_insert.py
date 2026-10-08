import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_inference.global_insert import (
    GLOBAL_INSERT_START_END,
    GLOBAL_INSERT_START_ONLY,
    GLOBAL_INSERT_VIDEO_START,
    parse_global_insert_mode,
    plan_global_insert,
    roi_anchor_requests,
)
from vlm_inference.metadata_format import (
    FrameRef,
    GlobalSection,
    RoiSection,
    VideoBundleMetadata,
)


def _frame(index: int, t: float, name: str) -> FrameRef:
    return FrameRef(
        frame_index=index,
        timestamp=t,
        file=name,
        path=Path("/tmp") / name,
    )


def _bundle(*ranges: tuple[int, tuple[float, float]]) -> VideoBundleMetadata:
    rois = []
    for roi_id, time_range in ranges:
        start, end = time_range
        frames = (
            _frame(roi_id * 10, start, f"tube{roi_id:02d}_frame_000001_t{start:.1f}.jpg"),
            _frame(roi_id * 10 + 1, end, f"tube{roi_id:02d}_frame_000002_t{end:.1f}.jpg"),
        )
        rois.append(
            RoiSection(
                roi_id=roi_id,
                time_range=time_range,
                source_region="motion",
                normalized_bbox=(0.0, 0.0, 1.0, 1.0),
                num_frames=len(frames),
                frames=frames,
            )
        )
    return VideoBundleMetadata(
        video_id="demo",
        root=Path("/tmp/demo"),
        global_section=GlobalSection(coverage="full scene", num_frames=0, frames=()),
        rois=tuple(rois),
    )


class GlobalInsertTests(unittest.TestCase):
    def test_parse_accepts_video_start(self):
        self.assertEqual(parse_global_insert_mode("video_start"), GLOBAL_INSERT_VIDEO_START)
        self.assertEqual(parse_global_insert_mode("start_only"), GLOBAL_INSERT_START_ONLY)
        self.assertEqual(parse_global_insert_mode("start_end"), GLOBAL_INSERT_START_END)
        self.assertIsNone(parse_global_insert_mode("none"))

    def test_video_start_is_one_request_at_t0(self):
        bundle = _bundle((1, (5.0, 10.0)), (2, (12.0, 18.0)))
        requests = roi_anchor_requests(bundle, GLOBAL_INSERT_VIDEO_START)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].role, "video_start")
        self.assertEqual(requests[0].target_t, 0.0)

    def test_start_only_one_per_roi(self):
        bundle = _bundle((1, (5.0, 10.0)), (2, (12.0, 18.0)))
        requests = roi_anchor_requests(bundle, GLOBAL_INSERT_START_ONLY)
        self.assertEqual([(r.roi_id, r.role, r.target_t) for r in requests], [
            (1, "start", 5.0),
            (2, "start", 12.0),
        ])

    def test_start_end_two_per_roi(self):
        bundle = _bundle((1, (5.0, 10.0)))
        requests = roi_anchor_requests(bundle, GLOBAL_INSERT_START_END)
        self.assertEqual([(r.role, r.target_t) for r in requests], [
            ("start", 5.0),
            ("end", 10.0),
        ])

    def test_plan_snaps_and_dedups(self):
        bundle = _bundle((1, (0.2, 4.0)), (2, (0.4, 8.0)))
        candidates = [
            _frame(1, 0.0, "global_frame_000001_t00.0.jpg"),
            _frame(5, 4.0, "global_frame_000005_t04.0.jpg"),
            _frame(9, 8.0, "global_frame_000009_t08.0.jpg"),
        ]
        video_plan = plan_global_insert(
            bundle,
            mode=GLOBAL_INSERT_VIDEO_START,
            candidates=candidates,
            source_dir="/src",
        )
        self.assertEqual([fr.timestamp for fr in video_plan.frames], [0.0])

        start_plan = plan_global_insert(
            bundle,
            mode=GLOBAL_INSERT_START_ONLY,
            candidates=candidates,
            source_dir="/src",
        )
        self.assertEqual([fr.timestamp for fr in start_plan.frames], [0.0])
        self.assertEqual(sum(1 for hit in start_plan.hits if not hit.kept), 1)

        start_end = plan_global_insert(
            bundle,
            mode=GLOBAL_INSERT_START_END,
            candidates=candidates,
            source_dir="/src",
        )
        self.assertEqual([fr.timestamp for fr in start_end.frames], [0.0, 4.0, 8.0])


if __name__ == "__main__":
    unittest.main()
