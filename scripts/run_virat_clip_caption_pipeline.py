#!/usr/bin/env python3
"""Resumable selected-59 VIRAT clip captions, followed by text-only event merging."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_inference.clip_caption import atomic_json, review_text, segment_video
from vlm_inference.clip_pipeline import (
    caption_document, fingerprint, materialize_segments, preserve_failed,
    read_json, valid_record, verify_published_caption,
)
from vlm_inference.config import load_config
from vlm_inference.inputs import Sample
from vlm_inference.temporal_event_merge import build_prompt, comparison_markdown, validate_clip_document, validate_merged_events
from vlm_inference.video_io import inspect_video


def prompt_text(cfg):
    path = Path(cfg.prompt_file)
    return (path if path.is_absolute() else cfg.config_dir / path).read_text(encoding='utf-8')


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def merge_budget(clips):
    # Long videos need enough tokens to spell out every supporting source ID.
    return max(4096, min(16384, 1024 + 48 * len(clips)))


def _main(resources):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/virat_clip_caption_59.yaml'))
    parser.add_argument('--merge-config', default=str(ROOT / 'configs/virat_temporal_event_merge.yaml'))
    parser.add_argument('--videos-file', default=str(ROOT / 'configs/virat_multi_roi_59.txt'))
    parser.add_argument('--output-dir')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Check all 59 sources and segmentation without loading model')
    args = parser.parse_args()
    out = Path(args.output_dir or ROOT / 'outputs/virat_clip_caption').expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = resources.enter_context((out / '.lock').open('a'))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another pipeline is using this output directory')
    log_handler = logging.FileHandler(out/'pipeline.log')
    resources.callback(log_handler.close)
    resources.callback(logging.getLogger().removeHandler, log_handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[log_handler, logging.StreamHandler()], force=True)
    cfg = load_config(args.config, allow_empty_jobs=True)
    merge_cfg = load_config(args.merge_config, allow_empty_jobs=True)
    for config in [cfg, merge_cfg]:
        if config.jobs or config.mode != 'default' or config.feasibility is not None:
            parser.error('Use standalone configs without video/ROI jobs')
        if config.model != cfg.model or Path(config.model.path).name != 'Qwen3-VL-30B-A3B-Instruct' or config.model.load_in_8bit:
            parser.error('Both stages require the same original 30B BF16 model')
    selected = [Path(line.strip()).resolve() for line in Path(args.videos_file).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    if len(selected) != 59 or len({p.name for p in selected}) != 59:
        parser.error('Provide exactly the selected 59 unique video files')
    baseline = [Path(line.strip()).resolve() for line in (ROOT/'configs/virat_multi_roi_59.txt').read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    if selected != baseline:
        parser.error('Video list must match the existing selected 59 list')
    caption_prompt, merge_template = prompt_text(cfg), prompt_text(merge_cfg)
    videos=[]
    for path in selected:
        info=inspect_video(path)
        windows=segment_video(info.frame_count,info.native_fps,allow_empty_tail=True)
        videos.append((info,windows))
    caption_settings=dict(model=asdict(cfg.model),generation=asdict(cfg.generation),video=asdict(cfg.video),prompt=caption_prompt,
                          clip_duration_sec=3,segmentation='one-pass lossless FFV1; final sub-frame tail retains displayed frame',
                          videos=[dict(path=str(i.path),bytes=i.path.stat().st_size,mtime_ns=i.path.stat().st_mtime_ns,
                                       frames=i.frame_count,fps=i.native_fps,windows=[asdict(w) for w in ws]) for i,ws in videos])
    merge_settings=dict(model=asdict(merge_cfg.model),generation=asdict(merge_cfg.generation),prompt=build_prompt(merge_template, []),
                        output_budget='max(4096,min(16384,1024+48*num_clips))')
    manifest=dict(caption_settings=caption_settings,merge_settings=merge_settings,
                  caption_fingerprint=fingerprint(caption_settings),merge_fingerprint=fingerprint(merge_settings))
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path)!=manifest:
            parser.error('Settings/source mismatch; use a new output directory')
        if not args.resume:
            parser.error('Existing run; pass --resume')
    else:
        atomic_json(manifest_path,manifest)
    total_clips=sum(len(ws) for _,ws in videos)
    logging.info('Preflight: 59 videos, %s clips; captions finish before any merging',total_clips)
    if args.dry_run:
        atomic_json(out/'preflight.json',dict(videos=59,total_clips=total_clips,
                    source_duration_sec=sum(i.frame_count/i.native_fps for i,_ in videos),
                    subframe_tails=[i.path.name for i,ws in videos if len(ws)>1 and ws[-1].start_frame<ws[-2].end_frame]))
        return 0
    gate_path=out/'pilot_gate.json'
    if gate_path.exists() and not read_json(gate_path).get('accepted_for_59_run',False):
        raise ValueError(f'Pilot merge quality gate has not passed: {gate_path}; full inference has not started')
    from vlm_inference.model import Qwen3VLCaptioner, _apply_video_opts, prepare_multimodal_inputs
    captioner=None
    completed_clips=0
    completed_videos=0
    failed_clips=[]
    def progress(stage, **extra):
        atomic_json(out/'progress.json',dict(stage=stage,total_videos=59,total_clips=total_clips,
                    normal_caption_clips=completed_clips,normal_caption_videos=completed_videos,
                    failed_clips=failed_clips,updated_unix_sec=time.time(),**extra))
    progress('caption_preflight')
    all_documents=[]
    for video_idx,(info,windows) in enumerate(videos,1):
        record_dir=out/'inference'/info.path.stem
        records=[]
        pending=[]
        for w in windows:
            path=record_dir/f'{w.index:04d}.json'
            record=valid_record(path,manifest['caption_fingerprint'])
            if record is not None:
                # Validate stored boundaries/contents before treating a record as complete.
                caption_document(info,[w],[record])
                completed_clips+=1
            else:
                pending.append(w.index)
            records.append(record)
        if pending:
            progress('caption_segmentation',current_video=info.path.name,video_index=video_idx)
            logging.info('Caption video %s/59 %s: %s pending clips',video_idx,info.path.name,len(pending))
            materialize_segments(info,windows,out/'segments'/info.path.stem)
            if captioner is None:
                progress('model_loading',current_video=info.path.name)
                logging.info('Loading Qwen30B once; reused for all videos and both stages')
                captioner=Qwen3VLCaptioner(cfg)
            captioner.cfg=cfg
            for w in windows:
                if records[w.index] is not None:
                    continue
                path=record_dir/f'{w.index:04d}.json'
                preserve_failed(path)
                segment=out/'segments'/info.path.stem/f'{w.index:04d}.avi'
                item=dict(type='video',video=segment.as_uri())
                _apply_video_opts(item,cfg.video,is_frame_list=False)
                if w.end_frame-w.start_frame==1:
                    item['allow_single_frame_padding']=True
                progress('caption_generate',current_video=info.path.name,video_index=video_idx,
                         current_clip=w.index,start_sec=w.start_sec,end_sec=w.end_sec)
                logging.info('START caption %s clip %s [%g,%g]',info.path.name,w.index,w.start_sec,w.end_sec)
                started=time.perf_counter()
                prepared=prepare_multimodal_inputs(captioner.processor,[dict(role='user',content=[item,dict(type='text',text=caption_prompt)])])
                sample=Sample(info.path.stem,f'{info.path.stem}_{w.index:04d}','video',[item['video']])
                result=captioner.generate(sample,caption_prompt,prepared=prepared)
                status='truncated' if result.hit_token_limit else ('ok' if result.text else 'empty')
                record=dict(fingerprint=manifest['caption_fingerprint'],video=info.path.name,**asdict(w),
                            caption=result.text,status=status,num_output_tokens=result.num_output_tokens,
                            hit_token_limit=result.hit_token_limit,input_tokens=prepared.token_stats,
                            sampling_actual=prepared.video_inputs[0],elapsed_sec=time.perf_counter()-started,
                            boundary_frame_reused=(w.index>0 and w.start_frame<windows[w.index-1].end_frame))
                atomic_json(path,record)
                records[w.index]=record
                del prepared
                if status=='ok':
                    completed_clips+=1
                else:
                    failed_clips.append(dict(video=info.path.name,clip=w.index,status=status))
                partial=[dict(start_sec=ws.start_sec,end_sec=ws.end_sec,caption=r['caption'],status=r['status'])
                         for ws,r in zip(windows,records) if r is not None]
                atomic_json(out/'partial_captions'/f'{info.path.name}.json',dict(video=info.path.name,clips=partial))
                logging.info('DONE %s clip %s: %s, %s tokens, %.2fs; %s/%s normal clips',
                             info.path.name,w.index,status,result.num_output_tokens,record['elapsed_sec'],completed_clips,total_clips)
        if all(r is not None and r['status']=='ok' for r in records):
            document=caption_document(info,windows,records)
            path=out/'captions'/f'{info.path.name}.json'
            if path.exists():
                verify_published_caption(path,info,windows,document)
            else:
                atomic_json(path,document)
            review=out/'captions'/f'{info.path.name}.txt'
            if not review.exists():atomic_text(review,review_text(info.path.name,document['clips']))
            all_documents.append(document)
            completed_videos+=1
        progress('caption',current_video=info.path.name,video_index=video_idx)
    if completed_videos!=59:
        progress('caption_failed_merge_not_started')
        logging.error('Caption barrier: %s/59 normal videos; failed clips retained. Resume retries only abnormal clips.',completed_videos)
        return 1
    completed_merges=0
    failed_merges=[]
    for document in all_documents:
        name=document['video'];source=out/'captions'/f'{name}.json'
        clips=validate_clip_document(document)
        input_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        fp=fingerprint(dict(settings=manifest['merge_fingerprint'],caption_sha256=input_hash))
        meta_path=out/'merge_inference'/f'{name}.json'
        output_path=out/'merged_events'/f'{name}.json'
        review_path=out/'merged_events'/f'{name}.md'
        old=valid_record(meta_path,fp)
        if old is not None:
            checked=validate_merged_events(json.dumps(old['events']),clips)
            if checked['status']!='ok':
                raise ValueError(f'Invalid normal merge record: {meta_path}')
            expected=dict(video=name,events=checked['events'])
            if output_path.exists():
                if read_json(output_path)!=expected:
                    raise ValueError(f'Merge export mismatch: {output_path}')
            else:
                atomic_json(output_path,expected)
            if not review_path.exists():atomic_text(review_path,comparison_markdown(name,clips,old['events']))
            completed_merges+=1
            progress('merge',normal_merged_videos=completed_merges,current_video=name)
            continue
        if output_path.exists():
            raise ValueError(f'Unverified existing merge export: {output_path}')
        if captioner is None:
            progress('merge_model_loading',normal_merged_videos=completed_merges)
            captioner=Qwen3VLCaptioner(merge_cfg)
        actual_cfg=replace(merge_cfg,generation=replace(merge_cfg.generation,max_new_tokens=merge_budget(clips)))
        captioner.cfg=actual_cfg
        prompt=build_prompt(merge_template,clips)
        attempt=1
        raw_dir=out/'merge_attempts'/name
        while (raw_dir/str(attempt)).exists():attempt+=1
        area=raw_dir/str(attempt)
        atomic_text(area/'prompt.txt',prompt)
        progress('merge_generate',normal_merged_videos=completed_merges,current_video=name)
        logging.info('START merge %s (%s clips; budget %s)',name,len(clips),actual_cfg.generation.max_new_tokens)
        started=time.perf_counter()
        result=captioner.generate_text(prompt)
        atomic_text(area/'raw.txt',result.text+'\n')
        checked=validate_merged_events(result.text,clips)
        if result.hit_token_limit:
            checked['status']='truncated';checked['errors'].append('Output token limit reached without EOS')
        meta=dict(fingerprint=fp,video=name,caption_sha256=input_hash,**checked,
                  generation=asdict(actual_cfg.generation),input_tokens=result.token_stats(),
                  num_output_tokens=result.num_output_tokens,hit_token_limit=result.hit_token_limit,
                  elapsed_sec=time.perf_counter()-started,attempt=str(area),video_inference=False)
        atomic_json(area/'metadata.json',meta)
        preserve_failed(meta_path)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=input_hash:
            raise ValueError(f'Caption source changed during merge: {source}')
        atomic_json(meta_path,meta)
        if checked['status']=='ok':
            atomic_json(output_path,dict(video=name,events=checked['events']))
            atomic_text(review_path,comparison_markdown(name,clips,checked['events']))
            completed_merges+=1
        else:
            failed_merges.append(dict(video=name,status=checked['status']))
        progress('merge',normal_merged_videos=completed_merges,failed_merges=failed_merges,current_video=name)
        logging.info('DONE merge %s: %s, %s events',name,checked['status'],len(checked['events']))
    progress('complete' if completed_merges==59 else 'merge_failed',normal_merged_videos=completed_merges,failed_merges=failed_merges)
    return int(completed_merges!=59)


def main():
    with ExitStack() as resources:
        return _main(resources)


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception('Pipeline stopped; intermediate results retained for --resume')
        raise
