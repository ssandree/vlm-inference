"""Global-resolution ablation: ROI stays fixed, only Global VLM tokens change."""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_inference.config import ImageConfig
from vlm_inference.experiment import LabeledFrame
from vlm_inference.feasibility import MethodInputs
from vlm_inference.global_resolution import (
    GLOBAL_SCALES,
    apply_global_resolution,
    global_pixel_budget,
    resolution_audit,
    scaled_wh,
    verify_invariants,
    vlm_smart_resize,
)
from vlm_inference.metadata_format import (
    FrameRef,
    GlobalSection,
    RoiSection,
    VideoBundleMetadata,
    formatted_items_to_qwen_content,
)


def _save(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=95)


def _frame(path: str, kind: str, time_s: float, roi_id: int | None = None) -> LabeledFrame:
    label = f"[Global] t={time_s}s" if kind == "global" else f"[ROI{roi_id}] t={time_s}s"
    return LabeledFrame(
        path=Path(path).resolve().as_uri(),
        label=label,
        time_s=time_s,
        kind=kind,
        roi_id=roi_id,
        frame_idx=1,
    )


def _inputs(tmp: Path, *, global_size=(1920, 1080), roi_size=(320, 240)) -> MethodInputs:
    g_path = tmp / "global_frame_000001_t0.0.jpg"
    r_path = tmp / "tube01_frame_000001_t0.5.jpg"
    _save(g_path, global_size, (10, 20, 30))
    _save(r_path, roi_size, (200, 10, 10))
    g_ref = FrameRef(1, 0.0, g_path.name, g_path)
    r_ref = FrameRef(1, 0.5, r_path.name, r_path)
    bundle = VideoBundleMetadata(
        video_id="demo",
        root=tmp,
        global_section=GlobalSection("full scene", 1, (g_ref,)),
        rois=(
            RoiSection(
                roi_id=1,
                time_range=(0.0, 1.0),
                source_region="center",
                normalized_bbox=(0.1, 0.2, 0.4, 0.5),
                num_frames=1,
                frames=(r_ref,),
            ),
        ),
    )
    return MethodInputs(
        video_id="demo",
        method="global_roi_start_only_roi_accumulated_motion",
        frames=[
            _frame(str(g_path), "global", 0.0),
            _frame(str(r_path), "roi", 0.5, 1),
        ],
        input_dirs=[str(tmp)],
        num_global_images=1,
        num_roi_images=1,
        num_roi_tubes=1,
        bundle=bundle,
        roi_bboxes_px={1: (10, 20, 40, 50)},
    )


class ScaleMathTests(unittest.TestCase):
    def test_scaled_wh_example(self):
        self.assertEqual(scaled_wh(1920, 1080, 1.0), (1920, 1080))
        self.assertEqual(scaled_wh(1920, 1080, 0.5), (960, 540))
        self.assertEqual(scaled_wh(1920, 1080, 0.25), (480, 270))
        self.assertEqual(scaled_wh(1920, 1080, 0.125), (240, 135))

    def test_identity_budget_matches_pipeline(self):
        image = ImageConfig(min_pixels=4096, max_pixels=262144)
        self.assertEqual(global_pixel_budget(image, 1.0), (4096, 262144))
        self.assertEqual(global_pixel_budget(image, 0.5), (4096, 65536))

    def test_tokens_drop_with_scale_under_max_pixels_cap(self):
        image = ImageConfig(min_pixels=4096, max_pixels=262144)
        tokens = []
        for scale in GLOBAL_SCALES:
            width, height = scaled_wh(1920, 1080, scale)
            min_p, max_p = global_pixel_budget(image, scale)
            _w, _h, n = vlm_smart_resize(
                width, height, min_pixels=min_p, max_pixels=max_p
            )
            tokens.append(n)
        jpeg_only = []
        for scale in GLOBAL_SCALES:
            width, height = scaled_wh(1920, 1080, scale)
            _w, _h, n = vlm_smart_resize(
                width, height, min_pixels=4096, max_pixels=262144
            )
            jpeg_only.append(n)
        self.assertEqual(jpeg_only[0], jpeg_only[1], "JPEG-only 1.0 vs 0.5 is flattened by max_pixels")
        self.assertGreater(tokens[0], tokens[1])
        self.assertGreater(tokens[1], tokens[2])
        self.assertGreater(tokens[2], tokens[3])


class ApplyScaleTests(unittest.TestCase):
    def test_scale_1_is_identity_and_roi_untouched(self):
        image = ImageConfig(min_pixels=4096, max_pixels=262144)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _inputs(root)
            roi_before = [f.path for f in inputs.frames if f.kind == "roi"]
            global_before = [f.path for f in inputs.frames if f.kind == "global"]
            out = apply_global_resolution(
                inputs,
                scale=1.0,
                image_cfg=image,
                cache_dir=root / "scaled",
                jpeg_quality=95,
            )
            self.assertEqual(
                [f.path for f in out.frames if f.kind == "roi"], roi_before
            )
            self.assertEqual(
                [f.path for f in out.frames if f.kind == "global"], global_before
            )
            self.assertEqual(out.original_global_resolution, (1920, 1080))
            self.assertFalse((root / "scaled").exists())

    def test_only_global_files_are_resized(self):
        image = ImageConfig(min_pixels=4096, max_pixels=262144)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _inputs(root)
            from urllib.parse import unquote, urlparse

            def local(uri: str) -> Path:
                parsed = urlparse(uri)
                return Path(unquote(parsed.path))

            roi_path = local(inputs.frames[1].path)
            roi_bytes = roi_path.read_bytes()
            roi_uri = inputs.frames[1].path
            out = apply_global_resolution(
                inputs,
                scale=0.5,
                image_cfg=image,
                cache_dir=root / "scaled",
                jpeg_quality=95,
            )
            self.assertEqual(out.original_global_resolution, (1920, 1080))
            g_path = local([f.path for f in out.frames if f.kind == "global"][0])
            with Image.open(g_path) as im:
                self.assertEqual(im.size, (960, 540))
            self.assertEqual([f.path for f in out.frames if f.kind == "roi"], [roi_uri])
            self.assertEqual(roi_path.read_bytes(), roi_bytes)
            self.assertEqual(out.bundle.rois[0].frames[0].path, inputs.bundle.rois[0].frames[0].path)


class ContentOptsTests(unittest.TestCase):
    def test_qwen_content_uses_scaled_max_pixels_only_for_global(self):
        from vlm_inference.metadata_format import FormattedItem

        image = ImageConfig(min_pixels=4096, max_pixels=262144)
        global_opts = ImageConfig(min_pixels=4096, max_pixels=65536)
        with tempfile.TemporaryDirectory() as tmp:
            g = Path(tmp) / "g.jpg"
            r = Path(tmp) / "r.jpg"
            _save(g, (64, 64), (0, 0, 0))
            _save(r, (64, 64), (1, 1, 1))
            items = [
                FormattedItem("[Global] t=0.0s", g),
                FormattedItem("[ROI1] t=0.5s", r),
            ]
            content = formatted_items_to_qwen_content(
                items, image_opts=image, global_image_opts=global_opts
            )
            images = [row for row in content if row.get("type") == "image"]
            self.assertEqual(images[0]["max_pixels"], 65536)
            self.assertEqual(images[1]["max_pixels"], 262144)


class InvariantTests(unittest.TestCase):
    def test_invariants_on_synthetic_four_scale_audits(self):
        image = ImageConfig(min_pixels=4096, max_pixels=262144)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _inputs(root)
            cfg_image = image
            rows_by_scale = {}
            baseline = None
            for scale in GLOBAL_SCALES:
                scaled = apply_global_resolution(
                    base,
                    scale=scale,
                    image_cfg=cfg_image,
                    cache_dir=root / "scaled" / str(scale),
                    jpeg_quality=95,
                )
                from vlm_inference.config import (
                    AppConfig,
                    ExperimentConfig,
                    GenerationConfig,
                    ModelConfig,
                    OutputConfig,
                    VideoConfig,
                )

                cfg = AppConfig(
                    model=ModelConfig(path="/tmp/model", name="8b"),
                    generation=GenerationConfig(),
                    video=VideoConfig(),
                    image=image,
                    output=OutputConfig(dir=str(root)),
                    experiment=replace(ExperimentConfig(), global_scale=scale),
                )
                row = resolution_audit(scaled, cfg=cfg, extra_io={
                    "global_timestamps": [0.0],
                    "roi_timestamps": {"1": [0.5]},
                })
                rows_by_scale[float(scale)] = [row]
                if scale == 1.0:
                    baseline = [row]
            report = verify_invariants(
                rows_by_scale,
                expected_videos=["demo"],
                baseline_rows=baseline,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["n_jobs"], 4)


if __name__ == "__main__":
    unittest.main()
