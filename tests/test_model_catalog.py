"""Model catalog aliases (8b / 30b) and path overlays."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_inference.config import (
    apply_model_override,
    apply_output_dir_override,
    load_config,
    resolve_model_spec,
)


class ModelCatalogTests(unittest.TestCase):
    def test_8b_and_30b_presets(self) -> None:
        eight = resolve_model_spec("8b")
        thirty = resolve_model_spec("30b")
        self.assertEqual(eight.name, "8b")
        self.assertEqual(eight.path, "/data/models/Qwen3-VL-8B-Instruct-BF16")
        self.assertEqual(thirty.name, "30b")
        self.assertEqual(thirty.path, "/data/models/Qwen3-VL-30B-A3B-Instruct")
        self.assertFalse(eight.load_in_8bit)
        self.assertNotEqual(eight.path, thirty.path)
        int8 = resolve_model_spec("8b-int8")
        self.assertEqual(int8.name, "8b-int8")
        self.assertEqual(int8.path, eight.path)
        self.assertTrue(int8.load_in_8bit)
        self.assertFalse(resolve_model_spec("8b-fp8").load_in_8bit)
        self.assertFalse(resolve_model_spec("30b-fp8").load_in_8bit)

    def test_alias_and_path_override(self) -> None:
        aliased = resolve_model_spec("qwen3-vl-30b-a3b")
        self.assertEqual(aliased.name, "30b")
        overlay = resolve_model_spec({"name": "8b", "attn_implementation": "eager"})
        self.assertEqual(overlay.path, "/data/models/Qwen3-VL-8B-Instruct-BF16")
        self.assertEqual(overlay.attn_implementation, "eager")
        pinned = resolve_model_spec({"name": "30b", "device_map": "cuda:0"})
        self.assertEqual(pinned.path, "/data/models/Qwen3-VL-30B-A3B-Instruct")
        self.assertEqual(pinned.device_map, "cuda:0")
        self.assertEqual(pinned.name, "30b")
        custom = resolve_model_spec("/data/models/Qwen3-VL-30B-A3B-Instruct")
        self.assertEqual(custom.path, "/data/models/Qwen3-VL-30B-A3B-Instruct")
        self.assertEqual(custom.name, "Qwen3-VL-30B-A3B-Instruct")

    def test_repo_configs_resolve_8b_or_30b(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs"
        for name in (
            "default.yaml",
            "feasibility.yaml",
            "feasibility_global_resolution_ablation.yaml",
            "token_information_feasibility.yaml",
        ):
            cfg = load_config(root / name)
            self.assertEqual(cfg.model.name, "8b")
            self.assertTrue(cfg.model.path.endswith("Qwen3-VL-8B-Instruct-BF16"))
            if name == "token_information_feasibility.yaml":
                self.assertEqual(cfg.model.device_map, "cuda:0")

    def test_yaml_alias_loads_and_cli_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "job.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "model: 8b",
                        "output:",
                        "  dir: outputs/from_yaml",
                        "  file: results.json",
                        "jobs:",
                        "  - name: demo",
                        "    type: images",
                        "    paths:",
                        "      - /tmp/unused.jpg",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = load_config(cfg_path)
            self.assertEqual(cfg.model.name, "8b")
            apply_model_override(cfg, "30b")
            self.assertEqual(cfg.model.name, "30b")
            self.assertEqual(
                cfg.model.path, "/data/models/Qwen3-VL-30B-A3B-Instruct"
            )
            apply_output_dir_override(cfg, Path(tmp) / "out30")
            self.assertEqual(cfg.output.dir, str(Path(tmp) / "out30"))


if __name__ == "__main__":
    unittest.main()
