"""Resolve job configs into inference samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import JobConfig
from .metadata_format import VideoBundleMetadata, has_metadata_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}


@dataclass(frozen=True)
class TubeInputBlock:
    """One ROI tube: Global image, enlarged start anchor, then the ROI video."""

    roi_id: int
    global_image_path: str
    anchor_image_path: str
    sequence_paths: list[str]
    anchor_min_pixels: int
    anchor_max_pixels: int


@dataclass(frozen=True)
class TubeGlobalRoiBlock:
    """One ROI tube: one Global image, then the ROI video. No anchor image."""

    roi_id: int
    global_image_path: str
    sequence_paths: list[str]
    global_timestamp: float | None = None
    sequence_timestamps: list[float] | None = None


@dataclass(frozen=True)
class CombinedRoiVideo:
    """One ROI-tube video inside a combined image + ROI videos + full-frame video sample."""

    roi_id: int
    sequence_paths: list[str]
    sequence_timestamps: list[float]


@dataclass
class Sample:
    job_name: str
    sample_id: str
    input_type: str  # "video" | "images" | "frames" | "global_then_tube_videos" | "tube_global_anchor_videos" | "tube_global_roi_videos" | "combined_image_roi_full_video"
    paths: list[str]
    sidecar_dirs: list[Path] = field(default_factory=list)
    # Prototype packing: sparse Global images + one video frame-list per ROI tube.
    # Unused by the existing image-packed experiment path.
    global_image_paths: list[str] = field(default_factory=list)
    tube_video_paths: list[tuple[int, list[str]]] = field(default_factory=list)
    # Start-anchor packing. Does not change global_then_tube_videos.
    tube_input_blocks: list[TubeInputBlock] = field(default_factory=list)
    # Per tube: [Global] image immediately before [ROI#] video. No anchor.
    tube_global_roi_blocks: list[TubeGlobalRoiBlock] = field(default_factory=list)
    # Original full-frame size + per-tube lifetime AABB (pixels). Used only for
    # experiment-mode ROI spatial metadata fallback (filename path).
    frame_width: int | None = None
    frame_height: int | None = None
    roi_bboxes_px: dict[int, tuple[int, int, int, int]] | None = None
    # Prefer logical metadata bundle (possibly composed). bundle_dir is only used
    # when bundle is None (single-directory metadata.json).
    bundle: VideoBundleMetadata | None = None
    bundle_dir: Path | None = None
    # One group per Qwen video item, in message order. Used only when
    # video.timestamp_mode is source_timestamp. None keeps the legacy clock.
    video_timestamp_groups: list[list[float]] | None = None
    # combined_image_roi_full_video only. Unused by the existing sample types.
    combined_start_image: str | None = None
    combined_roi_videos: list[CombinedRoiVideo] = field(default_factory=list)
    combined_full_paths: list[str] = field(default_factory=list)
    combined_full_timestamps: list[float] | None = None


def _sorted_glob(directory: Path, pattern: str) -> list[Path]:
    matches = sorted(directory.glob(pattern))
    return [p for p in matches if p.is_file()]


def _filter_images(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.suffix.lower() in IMAGE_EXTS]


def _filter_videos(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.suffix.lower() in VIDEO_EXTS]


def _as_file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def list_image_files(directory: Path, glob: str = "*.jpg") -> list[Path]:
    """Sorted image files in a directory. Empty if the directory is missing."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return _filter_images(_sorted_glob(root, glob))


def resolve_job(job: JobConfig) -> list[Sample]:
    if job.type == "video":
        path = Path(job.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"[{job.name}] video not found: {path}")
        return [
            Sample(
                job_name=job.name,
                sample_id=path.stem,
                input_type="video",
                paths=[_as_file_uri(path)],
                sidecar_dirs=[path.parent],
            )
        ]

    if job.type == "video_dir":
        root = Path(job.path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"[{job.name}] video_dir not found: {root}")
        videos = _filter_videos(_sorted_glob(root, job.glob))
        if not videos:
            # fallback: all video extensions under root if glob is "*"
            videos = _filter_videos(sorted(root.iterdir()) if job.glob == "*" else [])
        if not videos:
            raise FileNotFoundError(
                f"[{job.name}] no videos matched in {root} with glob '{job.glob}'"
            )
        return [
            Sample(
                job_name=job.name,
                sample_id=v.stem,
                input_type="video",
                paths=[_as_file_uri(v)],
                sidecar_dirs=[v.parent],
            )
            for v in videos
        ]

    if job.type == "images":
        paths = [Path(p).expanduser().resolve() for p in (job.paths or [])]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"[{job.name}] missing images: {missing}")
        sample_id = paths[0].stem if len(paths) == 1 else f"{paths[0].stem}_x{len(paths)}"
        return [
            Sample(
                job_name=job.name,
                sample_id=sample_id,
                input_type="images",
                paths=[_as_file_uri(p) for p in paths],
                sidecar_dirs=[paths[0].parent],
            )
        ]

    if job.type == "image_dir":
        root = Path(job.path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"[{job.name}] image_dir not found: {root}")
        if has_metadata_json(root):
            from .metadata_format import (
                iter_bundle_image_uris,
                load_video_bundle_metadata,
            )

            bundle = load_video_bundle_metadata(root, require_files=True)
            return [
                Sample(
                    job_name=job.name,
                    sample_id=bundle.video_id or root.name,
                    input_type="frames",
                    paths=iter_bundle_image_uris(bundle),
                    sidecar_dirs=[root, root.parent],
                    bundle=bundle,
                    bundle_dir=root,
                )
            ]
        images = _filter_images(_sorted_glob(root, job.glob))
        if not images:
            raise FileNotFoundError(
                f"[{job.name}] no images matched in {root} with glob '{job.glob}'"
            )
        return [
            Sample(
                job_name=job.name,
                sample_id=root.name,
                input_type="frames",
                paths=[_as_file_uri(p) for p in images],
                sidecar_dirs=[root, root.parent],
            )
        ]

    if job.type == "image_dirs":
        root = Path(job.path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"[{job.name}] image_dirs root not found: {root}")
        subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not subdirs:
            raise FileNotFoundError(f"[{job.name}] no subdirectories under {root}")
        samples: list[Sample] = []
        for sub in subdirs:
            if has_metadata_json(sub):
                from .metadata_format import (
                    iter_bundle_image_uris,
                    load_video_bundle_metadata,
                )

                bundle = load_video_bundle_metadata(sub, require_files=True)
                samples.append(
                    Sample(
                        job_name=job.name,
                        sample_id=bundle.video_id or sub.name,
                        input_type="frames",
                        paths=iter_bundle_image_uris(bundle),
                        sidecar_dirs=[sub, root],
                        bundle=bundle,
                        bundle_dir=sub,
                    )
                )
                continue
            images = _filter_images(_sorted_glob(sub, job.glob))
            if not images:
                continue
            samples.append(
                Sample(
                    job_name=job.name,
                    sample_id=sub.name,
                    input_type="frames",
                    paths=[_as_file_uri(p) for p in images],
                    sidecar_dirs=[sub, root],
                )
            )
        if not samples:
            raise FileNotFoundError(
                f"[{job.name}] no image subfolders with matches under {root}"
            )
        return samples

    raise ValueError(f"Unsupported job type: {job.type}")
