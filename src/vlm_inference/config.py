"""YAML config loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any

import yaml

from .spatial_occupancy import SpatialOccupancyConfig
from .centroid_motion import CentroidMotionConfig
from .magnitude_config import MagnitudeAdaptiveConfig
from .frame_clock import normalize_timestamp_mode

VALID_JOB_TYPES = frozenset(
    {"video", "video_dir", "images", "image_dir", "image_dirs"}
)
VALID_MODES = frozenset({"default", "experiment"})
VALID_ROI_FRAME_SETS = frozenset({"default", "fps_1"})

# Default ROI stem: optional `_fps_1` (or `_fps_N`) suffix after the time token.
DEFAULT_ROI_PATTERN = (
    r"tube(?P<roi>\d+)_frame_(?P<frame>\d+)_t(?P<time>\d+(?:\.\d+)?)"
    r"(?:_fps_\d+)?"
)
DEFAULT_GLOBAL_PATTERN = r"global_frame_(?P<frame>\d+)_t(?P<time>\d+(?:\.\d+)?)"


MODEL_PRESETS_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"

# Extra spellings that map onto keys in configs/models.yaml.
MODEL_ALIASES = {
    "8b": "8b",
    "8b-bf16": "8b",
    "qwen3-vl-8b": "8b",
    "qwen3-vl-8b-instruct": "8b",
    "8b-fp16": "8b-fp16",
    "8b-float16": "8b-fp16",
    "8b-int8": "8b-int8",
    "8b-fp8": "8b-fp8",
    "30b": "30b",
    "30b-a3b": "30b",
    "qwen3-vl-30b": "30b",
    "qwen3-vl-30b-a3b": "30b",
    "qwen3-vl-30b-a3b-instruct": "30b",
    "30b-fp8": "30b-fp8",
}


@dataclass
class ModelConfig:
    path: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = "sdpa"
    load_in_8bit: bool = False
    name: str | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 384
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    do_sample: bool = True


@dataclass
class VideoConfig:
    fps: float | None = 2.0
    nframes: int | None = None
    min_pixels: int | None = 4 * 32 * 32
    max_pixels: int | None = 256 * 32 * 32
    total_pixels: int | None = 20480 * 32 * 32
    sample_fps: float | None = None  # for frame lists
    # legacy keeps the fetcher clock. source_timestamp encodes selected times.
    timestamp_mode: str = "legacy"


@dataclass
class ImageConfig:
    min_pixels: int | None = 4 * 32 * 32
    max_pixels: int | None = 256 * 32 * 32


@dataclass
class OutputConfig:
    dir: str = "outputs"
    file: str = "results.json"

    @property
    def path(self) -> Path:
        return Path(self.dir) / self.file


@dataclass
class ExperimentConfig:
    """Filename patterns / labels for Global+ROI image prompts."""

    # Stem regexes with named groups: time (required), frame/roi (optional)
    global_pattern: str = DEFAULT_GLOBAL_PATTERN
    roi_pattern: str = DEFAULT_ROI_PATTERN
    global_label: str = "[Global] t={time}s"
    roi_label: str = "[ROI{roi}] t={time}s"
    # timestamp: Global+ROI interleaved by time
    # grouped: all Global, then ROI1, ROI2, ... (each ascending in time)
    # global_anchor: each Global, then ROI1 / ROI2 / ... in [G_i, G_{i+1})
    # tube_story: each tube's Global, then that tube's full ROI sequence
    input_order: str = "timestamp"
    # Which ROI JPEG set from metadata.json / disk to feed the VLM.
    # default: `frames` (~5/3 fps); fps_1: `frames_fps_1` (*_fps_1.jpg)
    roi_frames: str = "default"
    # Optional ROI-anchored Full-frame insertion (None = keep dataset globals).
    # video_start | start_only | start_end | start_mid_end
    global_insert: str | None = None
    # Sampled full-frame pool used to snap ROI timestamps: fps_1 | fps_2 | fps_0_25
    global_insert_source: str = "fps_1"
    # Merge Full-frames whose sampled timestamps differ by <= this many seconds.
    global_insert_dedup_eps: float = 0.5
    # Copy VLM input frames + board images under output.dir/vlm_inputs/
    save_vlm_inputs: bool = False
    # Global-only spatial scale for the resolution ablation. 1.0 = existing pipeline.
    # ROI crops / timestamps / bboxes are never scaled.
    global_scale: float = 1.0
    # ROI temporal sampling policy (None = keep dataset ROI frames).
    spatial_occupancy: SpatialOccupancyConfig = field(default_factory=SpatialOccupancyConfig)
    centroid_motion: CentroidMotionConfig = field(default_factory=CentroidMotionConfig)
    magnitude_adaptive: MagnitudeAdaptiveConfig = field(default_factory=MagnitudeAdaptiveConfig)
    roi_sampling: str | None = None
    roi_sampling_default_fps: float = 1.0
    roi_sampling_intervals: list[dict[str, float]] = field(default_factory=list)
    roi_sampling_tau: float = 8.0
    roi_sampling_overwrite: bool = False
    # ROI representation: None = keep dataset tubes; separate_roi | union_roi
    roi_repr: str | None = None
    roi_repr_overwrite: bool = False


@dataclass
class JobConfig:
    name: str
    type: str
    path: str | None = None
    paths: list[str] | None = None
    glob: str = "*"
    prompt: str | None = None
    prompt_file: str | None = None


@dataclass
class FeasibilityDatasets:
    video_root: str = "/data/datasets/virat_9vid"
    video_roots: list[str] = field(default_factory=list)
    fps_0_25: str = "/data/datasets/fps_sampling_frames/fps0_25"
    fps_1: str = "/data/datasets/fps_sampling_frames/fps_1"
    fps_2: str = "/data/datasets/fps_sampling_frames/fps_2"
    proposed_roi: str = "/data/datasets/proposed_ROI/gaptype5_fps2"
    proposed_roi_w_fps: str = "/data/datasets/proposed_ROI_w_fps/gaptype5_fps2"
    margin_root: str = "/data/datasets/proposed_ROI_margin"
    single_motion_roi: str = "/data/datasets/single_motion_ROI/fps2"
    single_motion_roi_w_fps: str = "/data/datasets/single_motion_ROI_w_fps/fps2"

    def all_video_roots(self) -> list[str]:
        roots: list[str] = []
        for root in [self.video_root, *list(self.video_roots or [])]:
            text = str(root).strip()
            if text and text not in roots:
                roots.append(text)
        return roots


@dataclass
class FeasibilityConfig:
    """9-video VIRAT feasibility sweep over already-sampled frame folders."""

    videos: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    datasets: FeasibilityDatasets = field(default_factory=FeasibilityDatasets)
    tracks_root: str = (
        "/home/ssandree/motion-detector/outputs/stage3/3_roi_tube/vlm_9vid_tracks"
    )
    jpeg_quality: int = 95
    cache_file: str = "outputs/feasibility_cache.json"
    resume: bool = True
    # Optional per-method prompt files (relative to config dir). Falls back to
    # global AppConfig.prompt / prompt_file when a method is omitted.
    method_prompts: dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    model: ModelConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    mode: str = "default"  # default | experiment
    jobs: list[JobConfig] = field(default_factory=list)
    prompt: str | None = None
    prompt_file: str | None = None
    config_dir: Path = field(default_factory=lambda: Path("."))
    feasibility: FeasibilityConfig | None = None


def _require(data: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required key '{key}' in {ctx}")
    return data[key]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "y"}:
            return True
        if key in {"0", "false", "no", "n", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Expected a boolean, got {value!r}")


def _as_dict(data: Any, ctx: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping for {ctx}, got {type(data).__name__}")
    return data


def load_model_catalog(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    catalog_path = Path(path) if path is not None else MODEL_PRESETS_PATH
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Model catalog not found: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Model catalog must be a non-empty mapping: {catalog_path}")
    catalog: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name or name.startswith("."):
            continue
        catalog[name] = _as_dict(value, f"models.{name}")
        if "path" not in catalog[name]:
            raise ValueError(f"models.{name} requires 'path'")
    if not catalog:
        raise ValueError(f"Model catalog has no presets: {catalog_path}")
    return catalog


def list_model_presets(catalog: dict[str, dict[str, Any]] | None = None) -> list[str]:
    return sorted((catalog or load_model_catalog()).keys())


def _model_config_from_mapping(
    data: dict[str, Any],
    *,
    name: str | None = None,
    ctx: str = "model",
) -> ModelConfig:
    return ModelConfig(
        path=str(_require(data, "path", ctx)),
        dtype=str(data.get("dtype", "bfloat16")),
        device_map=str(data.get("device_map", "auto")),
        attn_implementation=data.get("attn_implementation", "sdpa"),
        load_in_8bit=_as_bool(data.get("load_in_8bit", False)),
        name=name,
    )


def _looks_like_model_path(spec: str) -> bool:
    if spec.startswith("/") or spec.startswith(".") or "/" in spec:
        return True
    return Path(spec).exists()


def resolve_model_spec(
    spec: str | dict[str, Any],
    *,
    catalog: dict[str, dict[str, Any]] | None = None,
    catalog_path: str | Path | None = None,
) -> ModelConfig:
    """Resolve `8b` / `30b` / a weights path / a model mapping into ModelConfig."""
    presets = catalog if catalog is not None else load_model_catalog(catalog_path)
    if isinstance(spec, str):
        key = spec.strip()
        if not key:
            raise ValueError("model spec must not be empty")
        alias = MODEL_ALIASES.get(key.lower(), key)
        if alias in presets:
            return _model_config_from_mapping(
                presets[alias], name=alias, ctx=f"models.{alias}"
            )
        if _looks_like_model_path(key):
            path = Path(key)
            return ModelConfig(path=str(path), name=path.name)
        known = ", ".join(list_model_presets(presets))
        raise ValueError(
            f"Unknown model '{spec}'. Use a catalog name ({known}) or a local path."
        )

    data = _as_dict(spec, "model")
    name_raw = data.get("name")
    overlay = {k: v for k, v in data.items() if k != "name"}
    if name_raw:
        base = resolve_model_spec(
            str(name_raw), catalog=presets, catalog_path=catalog_path
        )
        merged = {
            "path": base.path,
            "dtype": base.dtype,
            "device_map": base.device_map,
            "attn_implementation": base.attn_implementation,
            "load_in_8bit": base.load_in_8bit,
        }
        merged.update(overlay)
        return _model_config_from_mapping(
            merged, name=base.name or str(name_raw), ctx="model"
        )
    return _model_config_from_mapping(overlay, ctx="model")


def apply_model_override(cfg: AppConfig, spec: str) -> ModelConfig:
    cfg.model = resolve_model_spec(spec)
    return cfg.model


def apply_output_dir_override(cfg: AppConfig, output_dir: str | Path) -> None:
    out = Path(output_dir)
    cfg.output.dir = str(out)
    if cfg.feasibility is not None:
        cache_name = Path(cfg.feasibility.cache_file).name or "feasibility_cache.json"
        cfg.feasibility.cache_file = str(out / cache_name)


def load_config(config_path: str | Path, *, allow_empty_jobs: bool = False) -> AppConfig:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    model = resolve_model_spec(_require(raw, "model", "config"))

    gen_raw = _as_dict(raw.get("generation"), "generation")
    generation = GenerationConfig(
        max_new_tokens=int(gen_raw.get("max_new_tokens", 384)),
        temperature=float(gen_raw.get("temperature", 0.7)),
        top_p=float(gen_raw.get("top_p", 0.8)),
        top_k=int(gen_raw.get("top_k", 20)),
        repetition_penalty=float(gen_raw.get("repetition_penalty", 1.05)),
        do_sample=bool(gen_raw.get("do_sample", True)),
    )

    video_raw = _as_dict(raw.get("video"), "video")
    video = VideoConfig(
        fps=video_raw.get("fps", 2.0),
        nframes=video_raw.get("nframes"),
        min_pixels=video_raw.get("min_pixels", 4 * 32 * 32),
        max_pixels=video_raw.get("max_pixels", 256 * 32 * 32),
        total_pixels=video_raw.get("total_pixels", 20480 * 32 * 32),
        sample_fps=video_raw.get("sample_fps"),
        timestamp_mode=normalize_timestamp_mode(video_raw.get("timestamp_mode", "legacy")),
    )

    image_raw = _as_dict(raw.get("image"), "image")
    image = ImageConfig(
        min_pixels=image_raw.get("min_pixels", 4 * 32 * 32),
        max_pixels=image_raw.get("max_pixels", 256 * 32 * 32),
    )

    out_raw = _as_dict(raw.get("output"), "output")
    output = OutputConfig(
        dir=str(out_raw.get("dir", "outputs")),
        file=str(out_raw.get("file", "results.json")),
    )

    mode = str(raw.get("mode", "default")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got '{mode}'")

    exp_raw = _as_dict(raw.get("experiment"), "experiment")
    input_order = str(exp_raw.get("input_order", "timestamp")).strip().lower()
    from .experiment import VALID_INPUT_ORDERS

    if input_order not in VALID_INPUT_ORDERS:
        raise ValueError(
            f"experiment.input_order must be one of {sorted(VALID_INPUT_ORDERS)}, "
            f"got '{input_order}'"
        )
    roi_frames = str(exp_raw.get("roi_frames", "default")).strip().lower()
    if roi_frames not in VALID_ROI_FRAME_SETS:
        raise ValueError(
            f"experiment.roi_frames must be one of {sorted(VALID_ROI_FRAME_SETS)}, "
            f"got '{roi_frames}'"
        )
    from .centroid_motion import CentroidMotionConfig
    from .global_insert import parse_global_insert_mode
    from .motion_adaptive import parse_roi_sampling_mode
    from .roi_union import parse_roi_repr_mode

    global_insert = parse_global_insert_mode(exp_raw.get("global_insert"))
    global_insert_source = str(exp_raw.get("global_insert_source", "fps_1")).strip()
    roi_sampling = parse_roi_sampling_mode(exp_raw.get("roi_sampling"))
    roi_repr = parse_roi_repr_mode(exp_raw.get("roi_repr"))
    experiment = ExperimentConfig(
        global_pattern=str(
            exp_raw.get("global_pattern", DEFAULT_GLOBAL_PATTERN)
        ),
        roi_pattern=str(exp_raw.get("roi_pattern", DEFAULT_ROI_PATTERN)),
        global_label=str(exp_raw.get("global_label", "[Global] t={time}s")),
        roi_label=str(exp_raw.get("roi_label", "[ROI{roi}] t={time}s")),
        input_order=input_order,
        roi_frames=roi_frames,
        global_insert=global_insert,
        global_insert_source=global_insert_source,
        global_insert_dedup_eps=float(exp_raw.get("global_insert_dedup_eps", 0.5)),
        save_vlm_inputs=bool(exp_raw.get("save_vlm_inputs", False)),
        roi_sampling=roi_sampling,
        spatial_occupancy=SpatialOccupancyConfig(**(exp_raw.get("spatial_occupancy") or {})),
        centroid_motion=CentroidMotionConfig(
            **{
                key: value
                for key, value in (exp_raw.get("centroid_motion") or {}).items()
                if key in {f.name for f in dataclass_fields(CentroidMotionConfig)}
            }
        ),
        magnitude_adaptive=MagnitudeAdaptiveConfig(
            **{
                key: value
                for key, value in (exp_raw.get("magnitude_adaptive") or {}).items()
                if key in {f.name for f in dataclass_fields(MagnitudeAdaptiveConfig)}
            }
        ),
        roi_sampling_default_fps=float(exp_raw.get("roi_sampling_default_fps", 1.0)),
        roi_sampling_intervals=list(exp_raw.get("roi_sampling_intervals") or []),
        roi_sampling_tau=float(exp_raw.get("roi_sampling_tau", 8.0)),
        roi_sampling_overwrite=bool(exp_raw.get("roi_sampling_overwrite", False)),
        roi_repr=roi_repr,
        roi_repr_overwrite=bool(exp_raw.get("roi_repr_overwrite", False)),
        global_scale=float(exp_raw.get("global_scale", 1.0)),
    )
    if experiment.global_scale <= 0:
        raise ValueError(
            f"experiment.global_scale must be positive, got {experiment.global_scale}"
        )

    jobs_raw = raw.get("jobs") or []
    if jobs_raw and not isinstance(jobs_raw, list):
        raise ValueError("'jobs' must be a list when provided")

    jobs: list[JobConfig] = []
    for i, job in enumerate(jobs_raw):
        ctx = f"jobs[{i}]"
        if not isinstance(job, dict):
            raise ValueError(f"{ctx} must be a mapping")
        name = str(_require(job, "name", ctx))
        jtype = str(_require(job, "type", ctx))
        if jtype not in VALID_JOB_TYPES:
            raise ValueError(
                f"{ctx}.type must be one of {sorted(VALID_JOB_TYPES)}, got '{jtype}'"
            )
        if jtype == "images":
            paths = job.get("paths")
            if not paths or not isinstance(paths, list):
                raise ValueError(f"{ctx}: type 'images' requires a non-empty 'paths' list")
        else:
            if not job.get("path"):
                raise ValueError(f"{ctx}: type '{jtype}' requires 'path'")

        jobs.append(
            JobConfig(
                name=name,
                type=jtype,
                path=job.get("path"),
                paths=[str(p) for p in job["paths"]] if job.get("paths") else None,
                glob=str(job.get("glob", "*")),
                prompt=job.get("prompt"),
                prompt_file=job.get("prompt_file"),
            )
        )

    feasibility = _parse_feasibility(raw)
    if not jobs and feasibility is None and not allow_empty_jobs:
        raise ValueError(
            "Config must include a non-empty 'jobs' list or a 'feasibility' / "
            "'experiments' section"
        )

    return AppConfig(
        model=model,
        generation=generation,
        video=video,
        image=image,
        output=output,
        experiment=experiment,
        mode=mode,
        jobs=jobs,
        prompt=raw.get("prompt"),
        prompt_file=raw.get("prompt_file"),
        config_dir=path.parent,
        feasibility=feasibility,
    )


def _parse_feasibility(raw: dict[str, Any]) -> FeasibilityConfig | None:
    feas_raw = raw.get("feasibility")
    experiments = raw.get("experiments")
    if feas_raw is None and experiments is None:
        return None
    data = _as_dict(feas_raw, "feasibility")

    methods = data.get("methods")
    if experiments is not None:
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("'experiments' must be a non-empty list of method names")
        methods = [str(m) for m in experiments]
    elif methods is None:
        methods = []
    else:
        if not isinstance(methods, list) or not methods:
            raise ValueError("feasibility.methods must be a non-empty list")
        methods = [str(m) for m in methods]

    if not methods:
        raise ValueError("feasibility methods / experiments list must not be empty")

    videos = data.get("videos") or []
    if not isinstance(videos, list) or not videos:
        raise ValueError("feasibility.videos must be a non-empty list")

    aliases_raw = data.get("aliases") or {}
    if not isinstance(aliases_raw, dict):
        raise ValueError("feasibility.aliases must be a mapping")
    aliases = {str(k): str(v) for k, v in aliases_raw.items()}

    ds_raw = _as_dict(data.get("datasets"), "feasibility.datasets")
    extra_roots = ds_raw.get("video_roots") or []
    if extra_roots and not isinstance(extra_roots, list):
        raise ValueError("feasibility.datasets.video_roots must be a list")
    datasets = FeasibilityDatasets(
        video_root=str(ds_raw.get("video_root", FeasibilityDatasets.video_root)),
        video_roots=[str(root) for root in extra_roots],
        fps_0_25=str(ds_raw.get("fps_0_25", FeasibilityDatasets.fps_0_25)),
        fps_1=str(ds_raw.get("fps_1", FeasibilityDatasets.fps_1)),
        fps_2=str(ds_raw.get("fps_2", FeasibilityDatasets.fps_2)),
        proposed_roi=str(ds_raw.get("proposed_roi", FeasibilityDatasets.proposed_roi)),
        proposed_roi_w_fps=str(
            ds_raw.get("proposed_roi_w_fps", FeasibilityDatasets.proposed_roi_w_fps)
        ),
        margin_root=str(ds_raw.get("margin_root", FeasibilityDatasets.margin_root)),
        single_motion_roi=str(
            ds_raw.get("single_motion_roi", FeasibilityDatasets.single_motion_roi)
        ),
        single_motion_roi_w_fps=str(
            ds_raw.get(
                "single_motion_roi_w_fps",
                FeasibilityDatasets.single_motion_roi_w_fps,
            )
        ),
    )

    method_prompts_raw = data.get("method_prompts") or {}
    if not isinstance(method_prompts_raw, dict):
        raise ValueError("feasibility.method_prompts must be a mapping")
    method_prompts = {str(k): str(v) for k, v in method_prompts_raw.items()}

    return FeasibilityConfig(
        videos=[str(v) for v in videos],
        methods=methods,
        aliases=aliases,
        datasets=datasets,
        tracks_root=str(data.get("tracks_root", FeasibilityConfig.tracks_root)),
        jpeg_quality=int(data.get("jpeg_quality", 95)),
        cache_file=str(data.get("cache_file", "outputs/feasibility_cache.json")),
        resume=bool(data.get("resume", True)),
        method_prompts=method_prompts,
    )
