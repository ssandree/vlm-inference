#!/usr/bin/env python3
"""Merge one existing clip-caption JSON using text-only Qwen3-VL inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_inference.config import load_config
from vlm_inference.dense_caption import write_new_json
from vlm_inference.temporal_event_merge import (
    build_prompt, comparison_markdown, validate_clip_document, validate_merged_events,
)

DEFAULT_INPUT = ROOT / 'outputs/virat_clip_caption_pilot/captions/VIRAT_S_010001_03_000537_000563.mp4.json'


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=str(DEFAULT_INPUT), help='One existing clip-caption JSON; no batch processing')
    p.add_argument('--config', default=str(ROOT / 'configs/virat_temporal_event_merge.yaml'))
    p.add_argument('--output-dir', help='New destination for a rerun; previous artifacts are never overwritten')
    p.add_argument('--dry-run', action='store_true', help='Validate captions and save exact text prompt without loading the model')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    cfg = load_config(args.config, allow_empty_jobs=True)
    if cfg.jobs or cfg.mode != 'default' or cfg.feasibility is not None:
        p.error('Text-only merge config must have no video/ROI jobs')
    if Path(cfg.model.path).name != 'Qwen3-VL-30B-A3B-Instruct' or cfg.model.load_in_8bit:
        p.error('Requires Qwen3-VL-30B-A3B-Instruct without quantization substitution')
    source = Path(args.input).expanduser().resolve()
    try:
        source_bytes = source.read_bytes()
        document = json.loads(source_bytes)
        clips = validate_clip_document(document)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if Path(document['video']).name != document['video']:
        p.error('video must be a filename, not a path')
    prompt_path = Path(cfg.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = cfg.config_dir / prompt_path
    prompt = build_prompt(prompt_path.read_text(encoding='utf-8'), clips)
    out = Path(args.output_dir or cfg.output.dir).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    if out.resolve() == source.parent:
        p.error('Use a separate output directory; original captions must be preserved')
    stem = document['video']
    prefix = stem + ('.dry_run' if args.dry_run else '')
    output = out / (stem + '.json')
    review = out / (stem + '.md')
    paths = {key: out / f'{prefix}.{suffix}' for key, suffix in
             [('prompt', 'prompt.txt'), ('raw', 'raw.txt'), ('validation', 'validation.json'), ('metadata', 'metadata.json')]}
    check_paths = [paths['prompt'], paths['metadata']] if args.dry_run else [*paths.values(), output, review]
    if any(path.exists() for path in check_paths):
        p.error('Existing artifacts: use a new --output-dir for a rerun')
    meta = dict(video=document['video'], input_file=str(source),
                input_sha256=hashlib.sha256(source_bytes).hexdigest(),
                model=asdict(cfg.model), generation=asdict(cfg.generation),
                num_input_clips=len(clips), input_type='text_only', video_inference=False,
                prompt_file=str(prompt_path.resolve()), purpose='Human-verification event candidates; not GT')
    write_text(paths['prompt'], prompt)
    if args.dry_run:
        write_new_json(paths['metadata'], dict(meta, status='dry_run'))
        print(paths['prompt'])
        return 0
    from vlm_inference.model import Qwen3VLCaptioner
    logging.info('Loading 30B model for text-only merge of %s (%s clips)', stem, len(clips))
    loading_started = time.perf_counter()
    captioner = Qwen3VLCaptioner(cfg)
    meta['model_load_sec'] = time.perf_counter() - loading_started
    logging.info('Starting text-only generate; no video decoding or vision input')
    started = time.perf_counter()
    result = captioner.generate_text(prompt)
    meta.update(elapsed_generation_sec=time.perf_counter()-started,
                input_tokens=result.token_stats(), num_output_tokens=result.num_output_tokens,
                hit_token_limit=result.hit_token_limit)
    write_text(paths['raw'], result.text + '\n')
    validated = validate_merged_events(result.text, clips)
    if result.hit_token_limit:
        validated['status'] = 'truncated'
        validated['errors'].append('Generation reached max_new_tokens without EOS')
    write_new_json(paths['validation'], validated)
    meta.update(status=validated['status'], num_events=len(validated['events']))
    write_new_json(paths['metadata'], meta)
    if validated['status'] != 'ok':
        logging.error('Merge validation failed; raw output retained: %s', paths['validation'])
        return 1
    if source.read_bytes() != source_bytes:
        raise RuntimeError('Input changed during inference; not publishing merged events')
    write_new_json(output, dict(video=document['video'], events=validated['events']))
    write_text(review, comparison_markdown(document['video'], clips, validated['events']))
    logging.info('Saved %s merged candidates: %s; review: %s', len(validated['events']), output, review)
    print(json.dumps(validated['events'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
