#!/usr/bin/env python3
"""ONE-video A′ local-event/cross-window pipeline vs independent B′ direct pipeline."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict,replace
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_inference.clip_caption import atomic_json
from vlm_inference.clip_pipeline import fingerprint,read_json
from vlm_inference.config import load_config
from vlm_inference.temporal_event_merge import validate_clip_document
from vlm_inference.intermediate_caption import windows_for
from vlm_inference.event_comparison import insert,timestamped
from vlm_inference.event_prime import validate_events,mechanical_stats,comparison_markdown
VIDEO='VIRAT_S_010113_02_000434_000479.mp4'


def publish_json(path,value):
    if path.exists():
        if read_json(path)!=value:raise ValueError(f'Existing export differs: {path}; use a new output directory')
    else:atomic_json(path,value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'outputs/virat_clip_caption/captions'/(VIDEO+'.json'))
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/virat_event_ab_prime_pilot')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();source,out=args.input.resolve(),args.output_dir.resolve()
    if out==source.parent or out in source.parents or source.parent in out.parents:p.error('Use a separate output directory')
    if out.exists() and any(out.iterdir()) and not args.resume:p.error('Existing artifacts: use --resume or a new directory')
    raw=source.read_bytes();doc=json.loads(raw)
    if doc.get('video')!=VIDEO:p.error('This pilot is restricted to the requested one video')
    clips=validate_clip_document(doc);windows=windows_for(doc);duration=clips[-1]['end_sec']
    if [c for w in windows for c in w['source_segments']]!=clips:raise ValueError('Missing/duplicate source clips')
    out.mkdir(parents=True,exist_ok=True)
    with ExitStack() as resources:
        lock=resources.enter_context((out/'.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handler=logging.FileHandler(out/'pipeline.log');resources.callback(handler.close)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[handler,logging.StreamHandler()],force=True)
        cfg=load_config(ROOT/'configs/virat_event_ab_prime.yaml',allow_empty_jobs=True)
        prompt_paths={stage:ROOT/'prompts'/filename for stage,filename in [('A1','virat_abc_a_3s_to_30s_json.txt'),('A2','virat_abc_b_30s_to_full_json.txt'),('B','virat_abc_c_3s_to_full_json.txt')]}
        templates={stage:path.read_text() for stage,path in prompt_paths.items()}
        settings=dict(model=asdict(cfg.model),generation=asdict(cfg.generation),prompts=templates,
                      budget_policy='local 4096; full max(4096,min(16384,1024+64*input_count))',input_type='text_only')
        sha=hashlib.sha256(raw).hexdigest()
        manifest=dict(video=VIDEO,source=str(source),source_sha256=sha,settings=settings,prompt_paths={k:str(v) for k,v in prompt_paths.items()})
        publish_json(out/'manifest.json',manifest)
        model=None;records=[]
        def progress(stage,**extra):atomic_json(out/'progress.json',dict(stage=stage,video=VIDEO,source_3s_segments=len(clips),local_windows=len(windows),updated_unix_sec=time.time(),**extra))
        def text_call(stage,key,prompt,budget):
            nonlocal model
            branch=out/('A_hierarchical' if stage.startswith('A') else 'B_direct')
            folder=branch/'inference'/VIDEO/stage;record_path=folder/(key+'.json')
            actual=replace(cfg,generation=replace(cfg.generation,max_new_tokens=budget));fp=fingerprint(dict(settings=settings,prompt=prompt,budget=budget,source_sha256=sha))
            if record_path.exists():
                old=read_json(record_path)
                if old['fingerprint']!=fp:raise ValueError('Stage input/settings mismatch')
                if old['status']=='ok':records.append(dict(stage=stage,key=key,record=str(record_path),reused=True));return old['text']
            folder.mkdir(parents=True,exist_ok=True);attempt=1
            while (folder/f'{key}.attempt_{attempt}.json').exists():attempt+=1
            (folder/f'{key}.attempt_{attempt}.prompt.txt').write_text(prompt)
            if args.dry_run:return ''
            if model is None:
                progress('model_loading')
                from vlm_inference.model import Qwen3VLCaptioner
                start=time.perf_counter();model=Qwen3VLCaptioner(cfg);logging.info('Model loaded once in %.2fs',time.perf_counter()-start)
            model.cfg=actual;progress(stage,current_item=key);logging.info('START %s %s budget=%s',stage,key,budget)
            start=time.perf_counter();result=model.generate_text(prompt)
            record=dict(fingerprint=fp,text=result.text,status='truncated' if result.hit_token_limit else ('ok' if result.text.strip() else 'empty'),
                        num_output_tokens=result.num_output_tokens,hit_token_limit=result.hit_token_limit,elapsed_sec=time.perf_counter()-start,
                        input_type='text_only',generation=asdict(actual.generation))
            atomic_json(folder/f'{key}.attempt_{attempt}.json',record);atomic_json(record_path,record)
            records.append(dict(stage=stage,key=key,record=str(record_path),reused=False))
            logging.info('DONE %s %s: %s, %s tokens, %.2fs',stage,key,record['status'],result.num_output_tokens,record['elapsed_sec'])
            if record['status']!='ok':raise ValueError(f'{stage}: {record["status"]}; raw retained')
            return record['text']
        boundaries={b for c in clips for b in [c['start_sec'],c['end_sec']]}
        locals_=[];validation=dict(video=VIDEO,source_sha256=sha,local=[],final={},errors=[],duplicate_handling='report only; no automatic deletion')
        # Each local output is a standalone common-schema JSON; bounds/evidence are in its sidecar.
        for i,w in enumerate(windows):
            prompt=insert(templates['A1'],TIMESTAMPED_3S_CAPTIONS=timestamped(w['source_segments']))
            generated=text_call('A1',f'{i:04d}',prompt,4096)
            if args.dry_run:continue
            try:obj,report=validate_events(generated,boundaries,(w['start_sec'],w['end_sec']))
            except (ValueError,TypeError) as exc:
                progress('local_validation_failed',window_index=i);raise ValueError(f'A1 window {i}: {exc}')
            local_path=out/'A_hierarchical'/'local_30s'/VIDEO/f'{i:04d}.json'
            publish_json(local_path,obj)
            evidence=dict(window_start=w['start_sec'],window_end=w['end_sec'],source_clips=[c['source_clip'] for c in w['source_segments']],source_segments=w['source_segments'],validation=report)
            publish_json(local_path.with_suffix('.metadata.json'),evidence)
            locals_.append(dict(window_start=w['start_sec'],window_end=w['end_sec'],event_json=obj))
            validation['local'].append(dict(window_index=i,**report))
        if args.dry_run:
            # Placeholder only in dry-run A2; B still contains original observations exclusively.
            text_call('A2','final',insert(templates['A2'],WINDOW_EVENT_JSONS='[]'),4096)
            text_call('B','final',insert(templates['B'],ALL_TIMESTAMPED_3S_CAPTIONS=timestamped(clips)),4096)
            progress('dry_run');return 0
        if len(locals_)!=len(windows) or len({w['window_start'] for w in locals_})!=len(windows):raise ValueError('Missing/duplicate local windows')
        # A2 gets only local event JSON and its window timestamps, no original/prose output.
        prompt_a=insert(templates['A2'],WINDOW_EVENT_JSONS=json.dumps(locals_,ensure_ascii=False,indent=2))
        local_count=sum(len(w['event_json']['events']) for w in locals_)
        raw_a=text_call('A2','final',prompt_a,max(4096,min(16384,1024+64*local_count)))
        # Direct B receives no A output. Fresh user request, no conversation state.
        raw_b=text_call('B','final',insert(templates['B'],ALL_TIMESTAMPED_3S_CAPTIONS=timestamped(clips)),max(4096,min(16384,1024+64*len(clips))))
        finals={}
        for stage,generated,branch in [('A',raw_a,'A_hierarchical'),('B',raw_b,'B_direct')]:
            try:
                obj,report=validate_events(generated,boundaries,(0,duration));finals[stage]=obj;validation['final'][stage]=report
                publish_json(out/branch/'final'/(VIDEO+'.json'),obj)
            except (ValueError,TypeError) as exc:validation['errors'].append(stage+': '+str(exc))
        validation.update(source_unchanged=source.read_bytes()==raw,missing_windows=[],duplicated_windows=[],source_segment_count=len(clips),window_count=len(windows))
        if not validation['source_unchanged']:validation['errors'].append('Source changed')
        validation['passed']=not validation['errors'];atomic_json(out/'validation.json',validation)
        if not validation['passed']:progress('validation_failed');raise ValueError(str(validation['errors']))
        all_local=[e for w in locals_ for e in w['event_json']['events']]
        statistics=dict(source_3s_segment_count=len(clips),A_local=mechanical_stats(all_local),A_final=mechanical_stats(finals['A']['events']),B_final=mechanical_stats(finals['B']['events']))
        publish_json(out/'comparison'/(VIDEO+'.statistics.json'),statistics)
        comparison=out/'comparison'/(VIDEO+'.md');content=comparison_markdown(VIDEO,clips,locals_,finals['A'],finals['B'],statistics)
        if comparison.exists() and comparison.read_text()!=content:raise ValueError('Existing comparison differs')
        if not comparison.exists():comparison.write_text(content)
        if source.read_bytes()!=raw:raise ValueError('Source changed during inference')
        atomic_json(out/'execution.json',dict(calls=records,model_loaded=model is not None,semantic_judge_calls=0,full_dataset_started=False))
        progress('complete');print(content,flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
