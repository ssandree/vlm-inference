"""descriptions.txt layout for the full-frame vs proposed-ROI sweep."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract_fullframe_descriptions import render_descriptions  # noqa: E402


class ExtractFullframeDescriptionsTests(unittest.TestCase):
    def test_groups_models_and_keeps_video_condition_description(self) -> None:
        text = render_descriptions(
            {
                "meta": {
                    "models": ["8b", "8b-int8"],
                    "videos": ["VIRAT_S_010114_01_000235_000311"],
                    "conditions": ["full_frame", "motion_roi"],
                },
                "runs": [
                    {
                        "model": "8b-int8",
                        "video_id": "VIRAT_S_010114_01_000235_000311",
                        "condition": "motion_roi",
                        "generated_description": "int8 roi",
                    },
                    {
                        "model": "8b",
                        "video_id": "VIRAT_S_010114_01_000235_000311",
                        "condition": "full_frame",
                        "generated_description": "bf16 full",
                    },
                ],
            }
        )
        self.assertEqual(
            text,
            "\n".join(
                [
                    "# 8b",
                    "",
                    "video: VIRAT_S_010114_01_000235_000311.mp4",
                    "condition: full_frame",
                    "description: bf16 full",
                    "",
                    "# 8b-int8",
                    "",
                    "video: VIRAT_S_010114_01_000235_000311.mp4",
                    "condition: motion_roi",
                    "description: int8 roi",
                    "",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
