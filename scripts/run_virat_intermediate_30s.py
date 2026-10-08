#!/usr/bin/env python3
"""Text-only 30-second consolidation of existing selected-59 caption JSONs."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_inference.clip_caption import atomic_json
from vlm_inference.clip_pipeline import read_json, fingerprint
from vlm_inference.config import load_config
from vlm_inference.intermediate_caption import windows_for, window_prompt, review_markdown, audit_prompt

PILOT = 'VIRAT_S_010001_03_000537_000563.mp4'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=ROOT/'outputs/virat_clip_caption/captions')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/virat_intermediate_30s')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--pilot-only', action='store_true')
    args = p.parse_args()
    source_dir, out = args.input_dir.resolve(), args.output_dir.resolve()
    if out == source_dir or source_dir in out.parents or out in source_dir.parents:
        p.error('Input and output directories must be separate')
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        lock = resources.enter_context((out/'.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                            handlers=[logging.FileHandler(out/'pipeline.log'), logging.StreamHandler()])
        cfg = load_config(ROOT/'configs/virat_intermediate_30s.yaml', allow_empty_jobs=True)
        template = (cfg.config_dir / cfg.prompt_file).read_text()
        names = [Path(s.strip()).name for s in (ROOT/'configs/virat_multi_roi_59.txt').read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
        names = [PILOT] if args.pilot_only else [PILOT] + [s for s in names if s != PILOT]
        inputs = []
        # Validate every required JSON before loading any model. Never fall back to video or partial captions.
        for name in names:
            path = source_dir / (name+'.json')
            raw = path.read_bytes(); doc = json.loads(raw)
            if doc['video'] != name: raise ValueError(f'Video mismatch: {path}')
            inputs.append((path, hashlib.sha256(raw).hexdigest(), doc, windows_for(doc)))
        settings = dict(model=asdict(cfg.model), generation=asdict(cfg.generation), prompt=template, window_duration_sec=30)
        manifest = dict(settings=settings, inputs={str(path):sha for path,sha,_,_ in inputs})
        manifest_path = out/('pilot_manifest.json' if args.pilot_only else 'manifest.json')
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError('Input/settings changed; choose a new output directory')
        atomic_json(manifest_path, manifest)
        total = sum(len(ws) for _,_,_,ws in inputs)
        atomic_json(out/'preflight.json', dict(videos=len(inputs), windows=total, text_only=True, source_hashes=manifest['inputs']))
        if args.dry_run:
            pilot = inputs[0]
            (out/'pilot_prompt.txt').write_text(window_prompt(template, pilot[3][0]))
            logging.info('Dry run validated %s videos, %s windows; no model/video inference', len(inputs), total)
            return 0
        model = None; completed = 0; failed = []
        def progress(stage, **extra):
            atomic_json(out/'progress.json', dict(stage=stage,total_videos=len(inputs),total_windows=total,
                        completed_windows=completed,failed_windows=failed,updated_unix_sec=time.time(),**extra))
        def generate(prompt):
            nonlocal model
            if model is None:
                progress('model_loading')
                from vlm_inference.model import Qwen3VLCaptioner
                model = Qwen3VLCaptioner(cfg)
            return model.generate_text(prompt)
        for video_idx,(path,sha,doc,windows) in enumerate(inputs,1):
            result_windows = []
            for idx,window in enumerate(windows):
                prompt = window_prompt(template,window)
                fp = fingerprint(dict(settings=settings,source_sha256=sha,window=window))
                folder = out/'inference'/doc['video']; record_path = folder/f'{idx:04d}.json'
                old = read_json(record_path) if record_path.exists() else None
                if old and old['fingerprint'] != fp: raise ValueError(f'Record settings mismatch: {record_path}')
                if old and old['status'] == 'ok':
                    record = old
                else:
                    attempt=1
                    while (folder/f'{idx:04d}.attempt_{attempt}.json').exists(): attempt+=1
                    progress('text_generate', current_video=doc['video'],video_index=video_idx,current_window=idx)
                    logging.info('START %s window [%s,%s] (%s source clips)',doc['video'],window['start_sec'],window['end_sec'],len(window['source_segments']))
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/f'{idx:04d}.attempt_{attempt}.prompt.txt').write_text(prompt)
                    start=time.perf_counter(); result=generate(prompt)
                    record=dict(fingerprint=fp,status='truncated' if result.hit_token_limit else ('ok' if result.text.strip() else 'empty'),
                                **window,intermediate_caption=result.text,source_clips=[c['source_clip'] for c in window['source_segments']],
                                input_sha256=sha,input_type='text_only',num_output_tokens=result.num_output_tokens,
                                elapsed_sec=time.perf_counter()-start,hit_token_limit=result.hit_token_limit)
                    atomic_json(folder/f'{idx:04d}.attempt_{attempt}.json',record)
                    atomic_json(record_path,record)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=sha: raise ValueError('Input changed during inference')
                if record['status']!='ok':
                    failed.append(dict(video=doc['video'],window=idx,status=record['status']))
                    if video_idx==1:
                        progress('pilot_failed'); return 1
                    continue
                if video_idx==1:
                    audit_path=folder/f'{idx:04d}.audit.json'
                    if audit_path.exists():
                        audit=read_json(audit_path)
                        if audit.get('candidate_sha256')!=hashlib.sha256(record['intermediate_caption'].encode()).hexdigest(): raise ValueError('Stale pilot audit')
                    else:
                        progress('pilot_validation',current_video=doc['video'])
                        result=generate(audit_prompt(window,record['intermediate_caption']))
                        raw=result.text.strip();raw=re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)
                        try: verdict=json.loads(raw)
                        except ValueError: verdict={}
                        approved=(not result.hit_token_limit and verdict.get('approved') is True and
                                  all(verdict.get(k)==[] for k in ['missing_events','unsupported_claims','conflict_or_order_errors']))
                        audit=dict(approved=approved,verdict=verdict,raw=result.text,
                                   candidate_sha256=hashlib.sha256(record['intermediate_caption'].encode()).hexdigest(),
                                   validation_kind='model_based_evidence_check; not human or video verification')
                        atomic_json(audit_path,audit)
                    if not audit['approved']:
                        progress('pilot_failed',audit_file=str(audit_path));logging.error('Pilot evidence check failed: %s',audit_path);return 1
                result_windows.append({k:record[k] for k in ['start_sec','end_sec','intermediate_caption','source_clips','source_segments']})
                completed+=1
                progress('text_generate',current_video=doc['video'],video_index=video_idx)
            if len(result_windows)==len(windows):
                exported=dict(video=doc['video'],window_duration_sec=30,source_caption_json=str(path),source_caption_sha256=sha,windows=result_windows)
                atomic_json(out/'captions'/f"{doc['video']}.json",exported)
                (out/'captions'/f"{doc['video']}.md").write_text(review_markdown(exported))
            if video_idx==1:
                atomic_json(out/'pilot_validation.json',dict(video=doc['video'],passed=True,source_sha256=sha,validation='Structural validation + model-based evidence check',windows=len(windows)))
                logging.info('Pilot passed; continuing remaining selected videos')
        for path,sha,_,_ in inputs:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Input changed')
        progress('complete' if not failed else 'complete_with_failures')
        return int(bool(failed))

if __name__=='__main__':
    raise SystemExit(main())
