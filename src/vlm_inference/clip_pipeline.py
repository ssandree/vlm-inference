"""Durable records and one-pass lossless segmentation for fixed-window captioning."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import logging
from pathlib import Path

import av

from .clip_caption import atomic_json
from .video_io import inspect_video


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def valid_record(path: Path, expected_fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    record = read_json(path)
    if record.get('fingerprint') != expected_fingerprint:
        raise ValueError(f'Settings/input mismatch: {path}; use a new output directory')
    return record if record.get('status') == 'ok' else None


def preserve_failed(path: Path) -> None:
    if path.exists():
        attempt = 1
        archive = path.with_name(f'{path.stem}.failed_{attempt}{path.suffix}')
        while archive.exists():
            attempt += 1
            archive = path.with_name(f'{path.stem}.failed_{attempt}{path.suffix}')
        path.rename(archive)


def verify_segment(path: Path, info, window) -> None:
    actual = inspect_video(path)
    if actual.frame_count != window.end_frame - window.start_frame or abs(actual.native_fps-info.native_fps) > 1e-6:
        raise ValueError(f'Segment metadata mismatch: {path}')


def materialize_segments(info, windows: list, directory: Path) -> None:
    """Decode a source at most once, encoding every native frame of each window.

    Very short tails without a new frame use the last frame still being displayed.
    Normal boundaries are half-open, disjoint native-frame ranges.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f'{w.index:04d}.avi' for w in windows]
    pending = set()
    for w, path in zip(windows, paths):
        if path.exists():
            verify_segment(path, info, w)
        else:
            pending.add(w.index)
    if not pending:
        return
    source = av.open(str(info.path))
    decoded = source.decode(video=0)
    frame = None
    frame_index = -1
    try:
        rate = source.streams.video[0].average_rate
        for w, path in zip(windows, paths):
            if w.index not in pending:
                continue
            temp = path.with_suffix('.tmp.avi')
            encoder = av.open(str(temp), mode='w')
            try:
                stream = encoder.add_stream('ffv1', rate=rate)
                stream.width, stream.height = info.width, info.height
                stream.pix_fmt = source.streams.video[0].codec_context.format.name
                stream.codec_context.thread_count = 2
                stream.options = {'level': '3'}
                for index in range(w.start_frame, w.end_frame):
                    while frame_index < index:
                        frame = next(decoded)
                        frame_index += 1
                    # CFR timestamps relative to the independent segment.
                    frame.pts = index - w.start_frame
                    frame.time_base = Fraction(1, 1) / rate
                    for packet in stream.encode(frame):
                        encoder.mux(packet)
                for packet in stream.encode():
                    encoder.mux(packet)
            finally:
                encoder.close()
            verify_segment(temp, info, w)
            temp.replace(path)
            if w.index % 25 == 0 or w.index == len(windows)-1:
                logging.getLogger(__name__).info('Segmented %s: %s/%s clips', info.path.name, w.index+1, len(windows))
    finally:
        source.close()


def caption_document(info, windows, records: list[dict]) -> dict:
    if len(records) != len(windows):
        raise ValueError('Video caption cannot be complete with missing clip records')
    clips = []
    for window, record in zip(windows, records):
        if record.get('status') != 'ok' or not record.get('caption', '').strip():
            raise ValueError('Only normal, nonempty clip responses form a complete caption file')
        if any(record.get(key) != value for key, value in asdict(window).items()):
            raise ValueError('Clip record interval does not match source segmentation')
        clips.append(dict(start_sec=window.start_sec, end_sec=window.end_sec, caption=record['caption']))
    return dict(video=info.path.name, clip_duration_sec=3, clips=clips)


def verify_published_caption(path: Path, info, windows, expected: dict) -> None:
    from .temporal_event_merge import validate_clip_document
    actual = read_json(path)
    validate_clip_document(actual)
    if actual != expected:
        raise ValueError(f'Caption export differs from normal records: {path}; preserve edits in a separate copy')
