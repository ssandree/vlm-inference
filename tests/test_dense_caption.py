"""Temporal and lossless validation for human-editable dense-caption candidates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_inference.dense_caption import parse_dense_captions, write_new_json


class DenseCaptionTests(unittest.TestCase):
    def test_chronological_sort_preserves_simultaneous_actors_and_continuous_walk(self):
        events = [
            {"timestamp": "8.0 - 9.0", "sentence": "A vehicle moves to the left."},
            {"timestamp": "1.0 - 12.0", "sentence": "A person walks across the foreground."},
            {"timestamp": "1.0 - 3.0", "sentence": "Another person walks near the entrance."},
        ]
        parsed = parse_dense_captions(json.dumps(events), 13)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["events"], [events[2], events[1], events[0]])

    def test_invalid_boundaries_preserved_without_clamping(self):
        rows = [{"timestamp": t, "sentence": "A person walks."}
                for t in ["2 - 4", "-1 - 3", "3 - 2", "1 - 1", "1 - 21", "nan - 4", "00:01 - 00:03"]]
        parsed = parse_dense_captions(json.dumps(rows), 20)
        self.assertEqual(parsed["status"], "partial")
        self.assertEqual(parsed["events"], rows[:1])
        self.assertEqual([r["original"] for r in parsed["rejected_events"]], rows[1:])

    def test_rejects_extra_fields_and_complex_schema(self):
        self.assertEqual(parse_dense_captions('{"events": []}', 20)["status"], "invalid_schema")
        rows = [{"timestamp": "1 - 2", "sentence": "A person walks.", "action": "walking"},
                {"timestamp": "1 - 2", "sentence": " "}, {"timestamp": 2, "sentence": "A person walks."}]
        self.assertEqual(len(parse_dense_captions(json.dumps(rows), 20)["rejected_events"]), 3)

    def test_empty_scene_valid_truncation_not_repaired(self):
        self.assertEqual(parse_dense_captions("[]", 20)["status"], "ok")
        self.assertEqual(parse_dense_captions('[{"timestamp":', 20)["status"], "invalid_json")
        self.assertEqual(parse_dense_captions('```json\n[]\n```', 20)["events"], [])

    def test_human_edits_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "clip.mp4.json"
            write_new_json(p, {"video_filename": "clip.mp4", "events": []})
            human_text = '{"video_filename":"clip.mp4","events":[],"human_note":"reviewed"}'
            p.write_text(human_text)
            with self.assertRaises(FileExistsError):
                write_new_json(p, {"events": []})
            self.assertEqual(p.read_text(), human_text)


if __name__ == "__main__":
    unittest.main()
