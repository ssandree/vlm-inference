"""Run existing motion-detector Stage1–3 on one VANTAGE video.

Does not change Farneback, fusion, or Stage3 parameters. Writes
``{video_id}_roi_tracks.json`` for the VLM pipeline to load.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _bootstrap() -> None:
    motion_src = Path("/home/ssandree/Motion-guided-ROI/motion-detector/src")
    if str(motion_src) not in sys.path:
        sys.path.insert(0, str(motion_src))
    from motion_analyzer.farneback import require_cuda_farneback

    require_cuda_farneback()


def compute_video_motion(
    *,
    video_id: str,
    video_root: Path,
    cache_root: Path,
) -> dict:
    _bootstrap()
    from motion_analyzer.aggregation import fuse_video
    from motion_analyzer.config import PipelineConfig
    from motion_analyzer.motion_map import compute_video_base_motion
    from stage3 import process_video

    video_root = Path(video_root)
    cache_root = Path(cache_root)
    tracks_dir = cache_root / "tracks"
    video_cache = cache_root / "videos" / video_id
    stage1 = video_cache / "stage1"
    stage2 = video_cache / "stage2"
    stage3 = video_cache / "stage3"
    tracks_path = tracks_dir / f"{video_id}_roi_tracks.json"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    stage1.mkdir(parents=True, exist_ok=True)
    stage2.mkdir(parents=True, exist_ok=True)
    stage3.mkdir(parents=True, exist_ok=True)

    if tracks_path.is_file():
        payload = json.loads(tracks_path.read_text(encoding="utf-8"))
        fusion = payload.get("fusion_npz")
        if fusion and Path(fusion).is_file():
            return {
                "video_id": video_id,
                "status": "cache_hit",
                "tracks_json": str(tracks_path),
                "n_tracks": len(payload.get("tracks") or []),
                "elapsed_sec": 0.0,
            }

    cfg = PipelineConfig(video_search_roots=[video_root])
    cfg.validate()
    t0 = time.perf_counter()
    compute_video_base_motion(video_id, cfg=cfg, data_root=stage1)
    fusion = fuse_video(
        video_id,
        cfg=cfg,
        output_root=stage2,
        data_root=stage1,
        fusion=cfg.fusion,
    )
    result = process_video(
        video_id,
        cfg=cfg,
        fusion_root=stage2,
        output_root=stage3,
        fusion=cfg.fusion,
        write_overlay=False,
        write_3d=False,
    )
    src_tracks = Path(result["tracks_json"])
    if src_tracks.resolve() != tracks_path.resolve():
        tracks_path.write_text(src_tracks.read_text(encoding="utf-8"), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    payload = json.loads(tracks_path.read_text(encoding="utf-8"))
    return {
        "video_id": video_id,
        "status": "computed",
        "tracks_json": str(tracks_path),
        "n_tracks": len(payload.get("tracks") or []),
        "stage1": str(stage1),
        "stage2": str(stage2),
        "stage3": str(stage3),
        "fusion_npz": fusion.get("fusion_npz") if isinstance(fusion, dict) else None,
        "elapsed_sec": float(elapsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    payload = compute_video_motion(
        video_id=args.video_id,
        video_root=args.video_root,
        cache_root=args.cache_root,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
