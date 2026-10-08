#!/usr/bin/env python3
"""Independent 3-second Qwen captions for the five previously selected VIRAT videos."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_inference.clip_caption import atomic_json, review_text, segment_video
from vlm_inference.config import load_config
from vlm_inference.inputs import Sample
from vlm_inference.video_io import inspect_video


def materialize_clip(source: Path, destination: Path, window, fps: float) -> None:
    """Lossless encode of every original frame in the half-open interval."""
    if destination.exists():
        info = inspect_video(destination)
        if info.frame_count != window.end_frame - window.start_frame or abs(info.native_fps - fps) > 1e-6:
            raise RuntimeError(f'Invalid cached segment: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.tmp.avi')
    subprocess.run([
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(source), '-map', '0:v:0', '-an',
        '-vf', f'trim=start_frame={window.start_frame}:end_frame={window.end_frame},setpts=PTS-STARTPTS',
        '-c:v', 'ffv1', '-level', '3', '-threads', '2', '-fps_mode', 'passthrough', str(temp),
    ], check=True)
    info = inspect_video(temp)
    if info.frame_count != window.end_frame - window.start_frame or abs(info.native_fps - fps) > 1e-6:
        raise RuntimeError(f'Segment frame count/fps mismatch: {temp}')
    temp.replace(destination)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/virat_clip_caption_pilot.yaml'))
    p.add_argument('--output-dir')
    p.add_argument('--dry-run', action='store_true', help='Materialize all segments and run actual processor, without model generation')
    p.add_argument('--resume', action='store_true', help='Reuse matching segments and completed clip records')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    cfg = load_config(args.config)
    baseline = load_config(ROOT / 'configs/virat_dense_caption_pilot.yaml')
    if (len(cfg.jobs) != 5 or any(j.type != 'video' for j in cfg.jobs)
            or [Path(j.path).resolve() for j in cfg.jobs] != [Path(j.path).resolve() for j in baseline.jobs]):
        p.error('Only the five previously selected explicit video jobs are allowed')
    if cfg.mode != 'default' or cfg.feasibility is not None:
        p.error('Requires default video mode without ROI inputs')
    if Path(cfg.model.path).name != 'Qwen3-VL-30B-A3B-Instruct' or cfg.model.load_in_8bit:
        p.error('Requires original Qwen3-VL-30B-A3B-Instruct checkpoint')
    if not cfg.prompt_file:
        p.error('prompt_file is required')
    prompt_path = Path(cfg.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = cfg.config_dir / prompt_path
    prompt = prompt_path.read_text(encoding='utf-8')
    out = Path(args.output_dir or cfg.output.dir).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    videos = [(job, inspect_video(job.path)) for job in cfg.jobs]
    settings = dict(model=asdict(cfg.model), generation=asdict(cfg.generation),
                    video=asdict(cfg.video), prompt=prompt, clip_duration_sec=3,
                    segmentation='half-open native frame intervals; lossless FFV1; local timestamps',
                    videos=[dict(path=str(info.path), bytes=info.path.stat().st_size,
                                 mtime_ns=info.path.stat().st_mtime_ns,
                                 windows=[asdict(w) for w in segment_video(info.frame_count, info.native_fps)])
                            for _, info in videos])
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error(f'Another pilot is using {out}')
    manifest = out / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['fingerprint'] != fingerprint:
            p.error('Settings/source mismatch: use a new --output-dir')
    else:
        if any(path.name != '.lock' for path in out.iterdir()):
            p.error('Destination must be empty')
        atomic_json(manifest, dict(fingerprint=fingerprint, settings=settings))
    area = out / ('dry_run' if args.dry_run else 'inference')
    if area.exists() and any(area.rglob('*.json')) and not args.resume:
        p.error('Existing clip records: use --resume or a new --output-dir')
    from vlm_inference.model import Qwen3VLCaptioner, _apply_video_opts, prepare_multimodal_inputs
    captioner, processor = None, None
    rows = []
    failed = False
    for job, info in videos:
        windows = segment_video(info.frame_count, info.native_fps)
        clips, records = [], []
        for window in windows:
            record_path = area / info.path.stem / f'{window.index:04d}.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record.get('fingerprint') != fingerprint:
                    p.error(f'Mismatched record: {record_path}')
                if record['status'] not in {'ok', 'dry_run', 'truncated', 'empty'}:
                    raise RuntimeError(f'Unrecognized clip status at {record_path}')
            else:
                segment = out / 'segments' / info.path.stem / f'{window.index:04d}.avi'
                materialize_clip(info.path, segment, window, info.native_fps)
                # One-frame tails cannot satisfy Qwen temporal factor 2. Keep the
                # segment intact; duplicate only its model input via reader opt-in.
                if processor is None:
                    if args.dry_run:
                        from transformers import AutoProcessor
                        processor = AutoProcessor.from_pretrained(cfg.model.path)
                    else:
                        captioner = Qwen3VLCaptioner(cfg)
                        processor = captioner.processor
                item = {'type': 'video', 'video': segment.resolve().as_uri()}
                _apply_video_opts(item, cfg.video, is_frame_list=False)
                if window.end_frame - window.start_frame == 1:
                    item['allow_single_frame_padding'] = True
                messages = [{'role': 'user', 'content': [item, {'type': 'text', 'text': prompt}]}]
                begin = time.perf_counter()
                prepared = prepare_multimodal_inputs(processor, messages)
                if len(prepared.video_inputs) != 1:
                    raise RuntimeError('Expected exactly one independent segment input')
                record = dict(fingerprint=fingerprint, video=info.path.name, **asdict(window),
                              segment=str(segment), sampling_actual=prepared.video_inputs[0],
                              input_tokens=prepared.token_stats, status='dry_run')
                if not args.dry_run:
                    sample = Sample(job.name, f'{info.path.stem}_{window.index:04d}', 'video', [item['video']])
                    result = captioner.generate(sample, prompt, prepared=prepared)
                    record.update(caption=result.text, num_output_tokens=result.num_output_tokens,
                                  hit_token_limit=result.hit_token_limit,
                                  status='truncated' if result.hit_token_limit else ('ok' if result.text else 'empty'))
                record['elapsed_sec'] = time.perf_counter() - begin
                atomic_json(record_path, record)
                del prepared
            records.append(record)
            if not args.dry_run:
                clips.append(dict(start_sec=window.start_sec, end_sec=window.end_sec, caption=record['caption']))
                atomic_json(out / 'captions' / f'{info.path.name}.json',
                            dict(video=info.path.name, clip_duration_sec=3, clips=clips))
                review = out / 'review' / f'{info.path.name}.txt'
                review.parent.mkdir(parents=True, exist_ok=True)
                review.write_text(review_text(info.path.name, clips), encoding='utf-8')
            logging.info('%s [%g–%g] %s: %s frames, %.3f fps%s', info.path.name,
                         window.start_sec, window.end_sec, record['status'],
                         record['sampling_actual']['sampled_frame_count'],
                         record['sampling_actual']['effective_sample_fps'],
                         ': ' + record.get('caption', '').replace('\n', ' ') if not args.dry_run else '')
            if record['status'] not in {'ok', 'dry_run'}:
                failed = True
        rows.append(dict(video=info.path.name, num_clips=len(records),
                         status='ok' if all(r['status'] in {'ok', 'dry_run'} for r in records) else 'failed',
                         clip_status_counts={status: sum(r['status'] == status for r in records)
                                             for status in sorted({r['status'] for r in records})},
                         records=[str(area / info.path.stem / f'{w.index:04d}.json') for w in windows]))
        atomic_json(out / ('dry_run.json' if args.dry_run else 'results.json'),
                    dict(fingerprint=fingerprint, dry_run=args.dry_run, videos=rows,
                         complete=len(rows) == 5, purpose='Independent clip captions only; no merge or GT'))
    print(f'Output: {out}')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
