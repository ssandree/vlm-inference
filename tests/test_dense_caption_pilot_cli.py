"""The pilot must reject dataset-wide jobs before loading model weights."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dense_pilot_cli", ROOT / "scripts/run_virat_dense_caption_pilot.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class PilotScopeTests(unittest.TestCase):
    def run_rejected(self, jobs, mode="default"):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "pilot.yaml"
            config.write_text(yaml.safe_dump({"model": "30b", "mode": mode, "jobs": jobs}))
            with patch.object(sys, "argv", ["pilot", "--config", str(config)]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    cli.main()
            self.assertEqual(error.exception.code, 2)

    def test_six_videos_rejected_before_missing_video_or_model_load(self):
        self.run_rejected([{"name": str(i), "type": "video", "path": "/missing.mp4"} for i in range(6)])

    def test_directory_job_and_empty_jobs_rejected(self):
        self.run_rejected([{"name": "all", "type": "video_dir", "path": "/data/datasets/VIRAT"}])
        self.run_rejected([])

    def test_roi_mode_rejected(self):
        self.run_rejected([{"name": "roi", "type": "video", "path": "/missing.mp4"}], mode="experiment")


if __name__ == "__main__":
    unittest.main()
