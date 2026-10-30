"""Qwen3-VL model wrapper using HuggingFace Transformers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import AppConfig, GenerationConfig, ImageConfig, VideoConfig
from .inputs import Sample, TubeInputBlock
from .video_io import install_av_video_reader

logger = logging.getLogger(__name__)

# Official Qwen3-VL Instruct stop tokens: <|im_end|> + <|endoftext|>.
QWEN3_EOS_TOKEN_IDS = [151645, 151643]
QWEN3_PAD_TOKEN_ID = 151643

# torchvision 0.28+ removed io.read_video; use PyAV via qwen-vl-utils backend hook.
install_av_video_reader()


def _disable_triton_python_native() -> None:
    """Avoid Triton JIT of cuda_utils (needs python3.12-dev / Python.h)."""
    try:
        import torch.backends.python_native as pn

        pn.triton.enabled = False
    except Exception:
        pass


_disable_triton_python_native()


def _dtype_from_name(name: str) -> torch.dtype | str:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(mapping)}")
    return mapping[key]


@dataclass
class GenerateResult:
    text: str
    num_visual_tokens: int | None = None
    num_text_tokens: int | None = None
    num_total_input_tokens: int | None = None
    video_inputs: list[dict[str, Any]] | None = None
    num_output_tokens: int | None = None
    hit_token_limit: bool = False
    # Per image and per video, from processor grids. None when grids are absent.
    # Existing num_visual_tokens stays the summed placeholder count.
    visual_token_detail: dict[str, Any] | None = None

    def token_stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.num_visual_tokens is not None:
            out["num_visual_tokens"] = self.num_visual_tokens
        if self.num_text_tokens is not None:
            out["num_text_tokens"] = self.num_text_tokens
        if self.num_total_input_tokens is not None:
            out["num_total_input_tokens"] = self.num_total_input_tokens
        return out


def count_processor_tokens(processor: Any, input_ids: torch.Tensor) -> dict[str, int] | None:
    """Count visual vs text tokens from actual processor input_ids.

    Visual tokens are the processor's image/video pad ids (the placeholders
    replaced by vision features). Returns None if those ids are unavailable.
    """
    if input_ids is None or input_ids.numel() == 0:
        return None
    ids = input_ids[0] if input_ids.ndim > 1 else input_ids
    total = int(ids.numel())
    visual_ids: list[int] = []
    for attr in ("image_token_id", "video_token_id"):
        token_id = getattr(processor, attr, None)
        if token_id is None:
            continue
        try:
            visual_ids.append(int(token_id))
        except (TypeError, ValueError):
            continue
    if not visual_ids:
        return None
    visual = 0
    for token_id in visual_ids:
        visual += int((ids == token_id).sum().item())
    return {
        "num_visual_tokens": visual,
        "num_text_tokens": total - visual,
        "num_total_input_tokens": total,
    }


@dataclass
class PreparedInputs:
    inputs: Any
    token_stats: dict[str, int] | None
    video_inputs: list[dict[str, Any]]
    visual_token_detail: dict[str, Any] | None = None


def _grid_visual_tokens(grid: Any, merge_size: int) -> list[int]:
    """Tokens per grid row. Qwen replaces one placeholder by prod(thw) // merge_size**2."""
    if grid is None:
        return []
    rows = grid.tolist() if hasattr(grid, "tolist") else list(grid)
    merge = int(merge_size) ** 2
    counts: list[int] = []
    for row in rows:
        product = 1
        for value in row:
            product *= int(value)
        counts.append(product // merge)
    return counts


def visual_token_detail_from_inputs(processor: Any, inputs: Any) -> dict[str, Any] | None:
    """Split visual tokens into one count per image and one count per video.

    The order matches processor image_grid_thw, then video_grid_thw, which follows
    message order. The summed grid count is what count_processor_tokens should match.
    """
    image_processor = getattr(processor, "image_processor", None)
    video_processor = getattr(processor, "video_processor", None)
    image_merge = getattr(image_processor, "merge_size", None)
    video_merge = getattr(video_processor, "merge_size", None)
    if image_merge is None or video_merge is None:
        return None
    image_tokens = _grid_visual_tokens(getattr(inputs, "image_grid_thw", None), int(image_merge))
    video_tokens = _grid_visual_tokens(getattr(inputs, "video_grid_thw", None), int(video_merge))
    return {
        "image_visual_tokens": image_tokens,
        "video_visual_tokens": video_tokens,
        "num_visual_tokens_from_grids": sum(image_tokens) + sum(video_tokens),
    }


def prepare_multimodal_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    frame_timestamps: list[list[float]] | None = None,
) -> PreparedInputs:
    """Shared Qwen preprocessing, with measured inputs rather than requested FPS.

    Used by normal generate and by a processor-only pilot dry-run. Metadata comes
    from the actual reader; processor frame sampling is explicitly disabled.

    ``frame_timestamps`` must align with the video items in order.
    ``generate`` passes them only when ``video.timestamp_mode`` is
    ``source_timestamp``. ``legacy`` passes None and keeps the fetcher clock.
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    tensors, metadatas, measured = None, None, []
    if videos is not None:
        tensors, metadatas = map(list, zip(*videos))
        metadatas = [dict(item) for item in metadatas]
        if frame_timestamps is not None:
            from .frame_clock import rewrite_video_metadata_list

            metadatas = rewrite_video_metadata_list(metadatas, frame_timestamps)
        for tensor, metadata in zip(tensors, metadatas):
            indices = [int(x) for x in metadata["frames_indices"]]
            fps = float(metadata["fps"])
            total = int(metadata["total_num_frames"])
            measured.append({
                "reader_backend": metadata.get("video_backend"),
                "sampled_frame_count": int(tensor.shape[0]),
                "resized_height": int(tensor.shape[2]),
                "resized_width": int(tensor.shape[3]),
                "native_fps": fps,
                "source_frame_count": total,
                "source_frame_indices": indices,
                "frame_timestamps_sec": [i / fps for i in indices],
                "effective_sample_fps": len(indices) * fps / total,
                "mean_frame_interval_sec": ((indices[-1] - indices[0]) / fps / (len(indices)-1)
                                            if len(indices) > 1 else None),
                "processor_resampling": False,
            })
    inputs = processor(
        text=[text], images=images, videos=tensors, video_metadata=metadatas,
        return_tensors="pt", do_resize=False, **(kwargs or {}),
    )
    grid = getattr(inputs, "video_grid_thw", None)
    if grid is not None:
        temporal = int(processor.video_processor.temporal_patch_size)
        for row, measurement in zip(grid.tolist(), measured):
            measurement["video_grid_thw"] = [int(x) for x in row]
            measurement["model_frame_count_with_temporal_padding"] = int(row[0]) * temporal
    return PreparedInputs(
        inputs,
        count_processor_tokens(processor, inputs.input_ids),
        measured,
        visual_token_detail_from_inputs(processor, inputs),
    )


class Qwen3VLCaptioner:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        model_cfg = cfg.model
        dtype = _dtype_from_name(model_cfg.dtype)

        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": model_cfg.device_map,
        }
        if model_cfg.attn_implementation:
            load_kwargs["attn_implementation"] = model_cfg.attn_implementation
        if model_cfg.load_in_8bit:
            from transformers import BitsAndBytesConfig

            # Quantize Linear weights on load. The BF16 checkpoint stays on disk.
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # Auto class selects dense Qwen3-VL (8B) or MoE Qwen3-VL (30B-A3B).
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_cfg.path, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_cfg.path)
        self.model.eval()
        logger.info(
            "Loaded %s from %s (preset=%s)",
            type(self.model).__name__,
            model_cfg.path,
            model_cfg.name or "-",
        )
        self._ensure_qwen_stop_tokens()

    def _ensure_qwen_stop_tokens(self) -> None:
        """Keep EOS/PAD aligned with official Qwen3-VL Instruct generation_config."""
        gc = self.model.generation_config
        eos = gc.eos_token_id
        pad = gc.pad_token_id
        eos_list = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
        if eos_list != QWEN3_EOS_TOKEN_IDS or pad != QWEN3_PAD_TOKEN_ID:
            logger.warning(
                "Fixing generation_config stop tokens (was eos=%s pad=%s)",
                eos,
                pad,
            )
            gc.eos_token_id = list(QWEN3_EOS_TOKEN_IDS)
            gc.pad_token_id = QWEN3_PAD_TOKEN_ID
        logger.info("generation_config: %s", gc)
        logger.info("eos: %s", gc.eos_token_id)
        logger.info("pad: %s", gc.pad_token_id)

    def _build_messages(self, sample: Sample, prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        video_opts = self.cfg.video
        image_opts = self.cfg.image

        if self.cfg.mode == "experiment" and sample.input_type in {"frames", "images"}:
            content.extend(self._build_experiment_image_content(sample))
        elif sample.input_type == "video":
            for path in sample.paths:
                item: dict[str, Any] = {"type": "video", "video": path}
                _apply_video_opts(item, video_opts, is_frame_list=False)
                content.append(item)
        elif sample.input_type == "frames":
            # Frame folder -> treated as a single video of frames
            item = {"type": "video", "video": list(sample.paths)}
            _apply_video_opts(item, video_opts, is_frame_list=True)
            content.append(item)
        elif sample.input_type == "leading_image_then_frames":
            # One Global full-frame image, then same-size ROI crops as a video.
            if not sample.paths:
                raise ValueError("leading_image_then_frames requires at least one path")
            image_item = {"type": "image", "image": sample.paths[0]}
            _apply_image_opts(image_item, image_opts)
            content.append(image_item)
            roi_paths = list(sample.paths[1:])
            if roi_paths:
                video_item = {"type": "video", "video": roi_paths}
                _apply_video_opts(video_item, video_opts, is_frame_list=True)
                content.append(video_item)
        elif sample.input_type == "global_then_tube_videos":
            content.extend(self._build_global_then_tube_video_content(sample))
        elif sample.input_type == "tube_global_anchor_videos":
            content.extend(self._build_tube_global_anchor_content(sample))
        elif sample.input_type == "tube_global_roi_videos":
            content.extend(self._build_tube_global_roi_content(sample))
        elif sample.input_type == "combined_image_roi_full_video":
            content.extend(self._build_combined_image_roi_full_content(sample))
        elif sample.input_type == "images":
            for path in sample.paths:
                item = {"type": "image", "image": path}
                _apply_image_opts(item, image_opts)
                content.append(item)
        else:
            raise ValueError(f"Unknown sample input_type: {sample.input_type}")

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _global_image_opts(self):
        from .global_resolution import global_image_config, is_identity_scale

        scale = float(self.cfg.experiment.global_scale)
        if is_identity_scale(scale):
            return self.cfg.image
        return global_image_config(self.cfg.image, scale)

    def _build_experiment_image_content(self, sample: Sample) -> list[dict[str, Any]]:
        """Spatial headers first, then Global/ROI images in configured input_order."""
        image_opts = self.cfg.image
        global_image_opts = self._global_image_opts()
        input_order = self.cfg.experiment.input_order

        # Prefer logical metadata bundle (composed or single-dir).
        if sample.bundle is not None or sample.bundle_dir is not None:
            from .metadata_format import (
                format_input_order_dump_from_bundle,
                format_video_bundle,
                formatted_items_to_qwen_content,
                load_video_bundle_metadata,
            )

            if sample.bundle is not None:
                metadata = sample.bundle
                formatted = format_video_bundle(metadata, input_order=input_order)
            else:
                assert sample.bundle_dir is not None
                metadata = load_video_bundle_metadata(
                    sample.bundle_dir,
                    roi_frames=self.cfg.experiment.roi_frames,
                )
                formatted = format_video_bundle(metadata, input_order=input_order)
            self._maybe_dump_input_order(
                sample,
                format_input_order_dump_from_bundle(metadata, input_order=input_order),
            )
            return formatted_items_to_qwen_content(
                formatted.items,
                image_opts=image_opts,
                as_file_uri=True,
                global_image_opts=global_image_opts,
            )

        from .experiment import (
            build_experiment_text_and_image_sequence,
            build_labeled_frames,
            build_roi_spatial_meta,
            format_input_order_dump,
        )

        frames = build_labeled_frames(sample.paths, self.cfg.experiment)
        spatial = None
        has_roi = any(f.kind == "roi" for f in frames)
        if has_roi:
            bboxes, width, height = self._resolve_roi_spatial(sample, frames)
            spatial = build_roi_spatial_meta(
                bboxes, frame_width=width, frame_height=height
            )

        self._maybe_dump_input_order(
            sample, format_input_order_dump(frames, input_order=input_order)
        )

        content: list[dict[str, Any]] = []
        for text, image_path in build_experiment_text_and_image_sequence(
            frames, spatial, input_order=input_order
        ):
            content.append({"type": "text", "text": text})
            if image_path is None:
                continue
            item: dict[str, Any] = {"type": "image", "image": image_path}
            opts = global_image_opts if text.lstrip().startswith("[Global]") else image_opts
            _apply_image_opts(item, opts)
            content.append(item)
        return content

    def _build_global_then_tube_video_content(self, sample: Sample) -> list[dict[str, Any]]:
        """Sparse Global images, then one Qwen video per ROI tube (JPEG frame lists)."""
        content: list[dict[str, Any]] = []
        global_image_opts = self._global_image_opts()
        video_opts = self.cfg.video
        for path in sample.global_image_paths:
            item: dict[str, Any] = {"type": "image", "image": path}
            _apply_image_opts(item, global_image_opts)
            content.append(item)
        for _roi_id, paths in sample.tube_video_paths:
            item = {"type": "video", "video": list(paths)}
            _apply_video_opts(item, video_opts, is_frame_list=True)
            content.append(item)
        return content

    def _build_tube_global_anchor_content(self, sample: Sample) -> list[dict[str, Any]]:
        """Per tube: Global image, enlarged start-anchor image, existing ROI video."""
        return build_tube_global_anchor_content(
            sample.tube_input_blocks,
            global_image=self._global_image_opts(),
            video=self.cfg.video,
        )

    def _build_tube_global_roi_content(self, sample: Sample) -> list[dict[str, Any]]:
        """Per tube: one [Global] image, then [ROI#] and the ROI video. No anchor."""
        return build_tube_global_roi_content(
            sample.tube_global_roi_blocks,
            global_image=self._global_image_opts(),
            video=self.cfg.video,
        )

    def _build_combined_image_roi_full_content(self, sample: Sample) -> list[dict[str, Any]]:
        """t=0 full-frame image, each ROI video, then one full-frame video."""
        return build_combined_image_roi_full_content(
            sample,
            image=self.cfg.image,
            video=self.cfg.video,
        )

    def _maybe_dump_input_order(self, sample: Sample, text: str) -> None:
        """Write pre-VLM input order dump under outputs/input_order/."""
        out_dir = Path(self.cfg.output.dir) / "input_order"
        out_dir.mkdir(parents=True, exist_ok=True)
        job = sample.job_name or sample.sample_id
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in job)
        order = self.cfg.experiment.input_order
        path = out_dir / f"{safe}__{order}.txt"
        path.write_text(text, encoding="utf-8")
        import logging

        logging.getLogger(__name__).info("Wrote input order dump: %s", path)

    def _resolve_roi_spatial(
        self, sample: Sample, frames: list
    ) -> tuple[dict[int, tuple[int, int, int, int]], int, int]:
        """Prefer Sample-attached bboxes; else load tracks via feasibility config."""
        if (
            sample.roi_bboxes_px is not None
            and sample.frame_width is not None
            and sample.frame_height is not None
        ):
            return sample.roi_bboxes_px, sample.frame_width, sample.frame_height

        feas = self.cfg.feasibility
        if feas is None:
            raise ValueError(
                "experiment mode with ROI frames requires Sample.roi_bboxes_px "
                "and frame_width/frame_height (or a feasibility config with tracks)"
            )

        from pathlib import Path

        from .roi_margin import load_tube_bboxes, tracks_json_path
        from .video_io import inspect_video

        tube_ids = sorted(
            {f.roi_id for f in frames if f.kind == "roi" and f.roi_id is not None}
        )
        video_path = Path(feas.datasets.video_root) / f"{sample.sample_id}.mp4"
        if not video_path.is_file():
            from .video_io import resolve_video_file

            video_path = resolve_video_file(
                sample.sample_id, feas.datasets.all_video_roots()
            )
        tracks_path = tracks_json_path(Path(feas.tracks_root), sample.sample_id)
        info = inspect_video(video_path)
        tubes = load_tube_bboxes(
            tracks_path, frame_width=info.width, frame_height=info.height
        )
        missing = [tid for tid in tube_ids if tid not in tubes]
        if missing:
            raise ValueError(
                f"missing track bboxes for tube(s) {missing} in {tracks_path}"
            )
        bboxes = {tid: tubes[tid].bbox_px for tid in tube_ids}
        return bboxes, int(info.width), int(info.height)

    @torch.inference_mode()
    def generate(
        self, sample: Sample, prompt: str, *, prepared: PreparedInputs | None = None
    ) -> GenerateResult:
        if prepared is None:
            from .frame_clock import frame_timestamps_for_generate

            # This packing emits each Global as an image, so only ROI videos
            # receive a timestamp group. SurveillanceVQA uses global_is_video=True.
            groups = frame_timestamps_for_generate(
                sample,
                self.cfg.video.timestamp_mode,
                global_is_video=False,
            )
            prepared = prepare_multimodal_inputs(
                self.processor,
                self._build_messages(sample, prompt),
                groups,
            )
        inputs = prepared.inputs
        token_stats = prepared.token_stats

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        gen = self.cfg.generation
        gc = self.model.generation_config
        gen_kwargs = _generation_kwargs(
            gen,
            eos_token_id=gc.eos_token_id,
            pad_token_id=gc.pad_token_id,
        )
        output_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return GenerateResult(
            text=texts[0].strip(),
            num_visual_tokens=(token_stats or {}).get("num_visual_tokens"),
            num_text_tokens=(token_stats or {}).get("num_text_tokens"),
            num_total_input_tokens=(token_stats or {}).get("num_total_input_tokens"),
            video_inputs=prepared.video_inputs,
            visual_token_detail=prepared.visual_token_detail,
            num_output_tokens=int(trimmed[0].numel()),
            hit_token_limit=(int(trimmed[0].numel()) >= gen.max_new_tokens
                             and int(trimmed[0][-1]) not in QWEN3_EOS_TOKEN_IDS),
        )

    @torch.inference_mode()
    def generate_text(self, prompt: str) -> GenerateResult:
        """Text-only generate (no images). Used to merge per-tube event lists."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            return_tensors="pt",
            do_resize=False,
        )
        token_stats = count_processor_tokens(
            self.processor, getattr(inputs, "input_ids", None)
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        gen = self.cfg.generation
        gc = self.model.generation_config
        gen_kwargs = _generation_kwargs(
            gen,
            eos_token_id=gc.eos_token_id,
            pad_token_id=gc.pad_token_id,
        )
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return GenerateResult(
            text=texts[0].strip(),
            num_visual_tokens=(token_stats or {}).get("num_visual_tokens"),
            num_text_tokens=(token_stats or {}).get("num_text_tokens"),
            num_total_input_tokens=(token_stats or {}).get("num_total_input_tokens"),
            num_output_tokens=int(trimmed[0].numel()),
            hit_token_limit=(int(trimmed[0].numel()) >= gen.max_new_tokens
                             and int(trimmed[0][-1]) not in QWEN3_EOS_TOKEN_IDS),
        )


def _apply_video_opts(
    item: dict[str, Any], opts: VideoConfig, *, is_frame_list: bool
) -> None:
    if opts.min_pixels is not None:
        item["min_pixels"] = opts.min_pixels
    if opts.max_pixels is not None:
        item["max_pixels"] = opts.max_pixels
    if opts.total_pixels is not None:
        item["total_pixels"] = opts.total_pixels

    if is_frame_list:
        if opts.sample_fps is not None:
            item["sample_fps"] = opts.sample_fps
        elif opts.fps is not None:
            item["sample_fps"] = opts.fps
    else:
        # Prefer nframes if set; otherwise fps
        if opts.nframes is not None:
            item["nframes"] = opts.nframes
        elif opts.fps is not None:
            item["fps"] = opts.fps


def build_combined_image_roi_full_content(
    sample: Sample,
    *,
    image: ImageConfig,
    video: VideoConfig,
) -> list[dict[str, Any]]:
    """One start image, then each ROI video, then the full-frame video.

    No extra text tags are inserted. Timestamp groups, when requested, are the
    ROI videos in this same order followed by the full-frame video.
    """
    from .frame_clock import combined_video_timestamp_groups

    if not sample.combined_start_image:
        raise ValueError("combined input needs the t=0 full-frame image")
    if not sample.combined_roi_videos:
        raise ValueError("combined input needs at least one ROI video")
    if not sample.combined_full_paths:
        raise ValueError("combined input needs the full-frame video")
    groups = combined_video_timestamp_groups(sample)
    explicit = sample.video_timestamp_groups
    if explicit is not None and len(explicit) != len(groups):
        raise ValueError(
            f"{len(explicit)} timestamp groups for {len(groups)} combined videos"
        )
    content: list[dict[str, Any]] = []
    image_item: dict[str, Any] = {"type": "image", "image": sample.combined_start_image}
    _apply_image_opts(image_item, image)
    content.append(image_item)
    for roi in sample.combined_roi_videos:
        item = {"type": "video", "video": list(roi.sequence_paths)}
        _apply_video_opts(item, video, is_frame_list=True)
        content.append(item)
    full_item = {"type": "video", "video": list(sample.combined_full_paths)}
    _apply_video_opts(full_item, video, is_frame_list=True)
    content.append(full_item)
    return content


def build_tube_global_roi_content(
    blocks: list,
    *,
    global_image: ImageConfig,
    video: VideoConfig,
) -> list[dict[str, Any]]:
    """One [Global] image at the start of each ROI sequence, then [ROI#] and that video.

    Tags stay ``[Global]`` and ``[ROI#]``. No anchor image is inserted.
    """
    content: list[dict[str, Any]] = []
    for block in blocks:
        content.append({"type": "text", "text": "[Global]"})
        global_item: dict[str, Any] = {"type": "image", "image": block.global_image_path}
        _apply_image_opts(global_item, global_image)
        content.append(global_item)

        content.append({"type": "text", "text": f"[ROI{int(block.roi_id)}]"})
        sequence = {"type": "video", "video": list(block.sequence_paths)}
        _apply_video_opts(sequence, video, is_frame_list=True)
        content.append(sequence)
    return content


def build_tube_global_anchor_content(
    blocks: list[TubeInputBlock],
    *,
    global_image: ImageConfig,
    video: VideoConfig,
) -> list[dict[str, Any]]:
    """Global, enlarged start anchor, one [ROI#] label, then that tube's ROI video.

    The ROI video uses the existing video pixel budget. The anchor image uses
    the per-tube budget stored on the block, which is higher than that video.
    """
    content: list[dict[str, Any]] = []
    for block in blocks:
        content.append({"type": "text", "text": "[Global]"})
        global_item: dict[str, Any] = {"type": "image", "image": block.global_image_path}
        _apply_image_opts(global_item, global_image)
        content.append(global_item)

        content.append({"type": "text", "text": "[ROI Anchor]"})
        anchor_item: dict[str, Any] = {"type": "image", "image": block.anchor_image_path}
        _apply_image_opts(
            anchor_item,
            ImageConfig(
                min_pixels=int(block.anchor_min_pixels),
                max_pixels=int(block.anchor_max_pixels),
            ),
        )
        content.append(anchor_item)

        content.append({"type": "text", "text": f"[ROI{int(block.roi_id)}]"})
        sequence = {"type": "video", "video": list(block.sequence_paths)}
        _apply_video_opts(sequence, video, is_frame_list=True)
        content.append(sequence)
    return content


def _apply_image_opts(item: dict[str, Any], opts: ImageConfig) -> None:
    if opts.min_pixels is not None:
        item["min_pixels"] = opts.min_pixels
    if opts.max_pixels is not None:
        item["max_pixels"] = opts.max_pixels


def _generation_kwargs(
    gen: GenerationConfig,
    *,
    eos_token_id: Any = None,
    pad_token_id: Any = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": gen.max_new_tokens,
        "do_sample": gen.do_sample,
        "repetition_penalty": gen.repetition_penalty,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if gen.do_sample:
        kwargs["temperature"] = gen.temperature
        kwargs["top_p"] = gen.top_p
        kwargs["top_k"] = gen.top_k
    return kwargs
