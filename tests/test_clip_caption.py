import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_inference.clip_caption import segment_video, review_text


class ClipCaptionTest(unittest.TestCase):
    def test_half_open_partition_noninteger_fps(self):
        windows = segment_video(620, 29.97)
        self.assertEqual(windows[0].start_sec, 0)
        self.assertEqual(windows[-1].end_sec, 620 / 29.97)
        self.assertLess(windows[-1].end_sec - windows[-1].start_sec, 3)
        self.assertEqual([i for w in windows for i in range(w.start_frame, w.end_frame)], list(range(620)))
        for a, b in zip(windows, windows[1:]):
            self.assertEqual(a.end_sec, b.start_sec)
            self.assertEqual(a.end_frame, b.start_frame)

    def test_exact_boundary_and_one_frame_tail(self):
        self.assertEqual(len(segment_video(180, 30)), 2)
        tail = segment_video(181, 30)[-1]
        self.assertEqual((tail.start_frame, tail.end_frame), (180, 181))
        self.assertAlmostEqual(tail.end_sec, 181 / 30)

    def test_review_keeps_all_sentences_and_repeated_clips(self):
        caption = 'Two people walk. One vehicle turns.'
        clips = [dict(start_sec=0, end_sec=3, caption=caption),
                 dict(start_sec=3, end_sec=3.125, caption=caption)]
        text = review_text('test.mp4', clips)
        self.assertEqual(text.count(caption), 2)
        self.assertIn('[00:03–00:03.125]', text)

class SingleFrameTailTest(unittest.TestCase):
    def test_single_frame_tail_padding_uses_no_neighbor_frame(self):
        import tempfile
        import av
        import numpy as np
        from vlm_inference.video_io import read_video_av
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tail.avi'
            with av.open(str(path), mode='w') as container:
                stream = container.add_stream('ffv1', rate=30)
                stream.width, stream.height, stream.pix_fmt = 32, 32, 'yuv420p'
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 127, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            tensor, metadata, fps = read_video_av(dict(video=str(path), fps=2, allow_single_frame_padding=True))
            self.assertEqual(metadata['frames_indices'], [0, 0])
            self.assertEqual(metadata['total_num_frames'], 1)
            self.assertTrue(tensor[0].equal(tensor[1]))
            self.assertEqual(fps, 60)


if __name__ == '__main__':
    unittest.main()
