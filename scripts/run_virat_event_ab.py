#!/usr/bin/env python3
"""Independent hierarchical/direct event generation from immutable 3s captions."""
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
from vlm_inference.event_comparison import insert,timestamped,parse_events,comparison_md,intermediate_boundaries
DEFAULT_VIDEO='VIRAT_S_010113_02_000434_000479.mp4'


def publish_json(path, value):
    if path.exists():
        if read_json(path) != value:raise ValueError(f'Existing output differs; preserve it and use a new directory: {path}')
    else:atomic_json(path,value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--video',default=None)
    selection.add_argument('--all',action='store_true',help='Explicitly run all selected 59; default is ONE pilot')
    p.add_argument('--input-dir',type=Path,default=ROOT/'outputs/virat_clip_caption/captions')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/virat_event_ab_pilot')
    p.add_argument('--pipeline',choices=['A','B','both'],default='both',help='Run A or B independently, or both for comparison')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    source_dir,out=args.input_dir.resolve(),args.output_dir.resolve()
    if out==source_dir or out in source_dir.parents or source_dir in out.parents:p.error('Use a separate output directory')
    if out.exists() and any(out.iterdir()) and not args.resume:p.error('Existing results: use --resume or a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    with ExitStack() as resources:
        lock=resources.enter_context((out/'.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handler=logging.FileHandler(out/'pipeline.log');resources.callback(handler.close)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[handler,logging.StreamHandler()],force=True)
        cfg=load_config(ROOT/'configs/virat_event_ab.yaml',allow_empty_jobs=True)
        templates={stage:(ROOT/'prompts'/filename).read_text() for stage,filename in
                   [('A1','virat_ab_a1_30s.txt'),('A2','virat_ab_a2_events.txt'),('B','virat_ab_b_direct.txt')]}
        selected=[Path(s.strip()).name for s in (ROOT/'configs/virat_multi_roi_59.txt').read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
        names=selected if args.all else [args.video or DEFAULT_VIDEO]
        if any(n not in selected for n in names):p.error('Video must belong to the selected 59')
        documents=[]
        for name in names:
            path=source_dir/(name+'.json');raw=path.read_bytes();doc=json.loads(raw)
            if doc['video']!=name:raise ValueError('Source filename mismatch')
            clips=validate_clip_document(doc);windows=windows_for(doc)
            documents.append((path,raw,doc,clips,windows))
        settings=dict(model=asdict(cfg.model),generation=asdict(cfg.generation),templates=templates,
                      event_budget='max(4096,min(16384,1024+64*input_count))',text_only=True)
        manifest=dict(settings=settings,source_hashes={str(path):hashlib.sha256(raw).hexdigest() for path,raw,_,_,_ in documents})
        if (out/'manifest.json').exists() and read_json(out/'manifest.json')!=manifest:raise ValueError('Source/settings mismatch')
        atomic_json(out/'manifest.json',manifest)
        model=None;completed=0
        def progress(stage,**extra):atomic_json(out/'progress.json',dict(stage=stage,videos=len(names),completed_videos=completed,updated_unix_sec=time.time(),**extra))
        def text_call(stage,name,key,prompt,budget):
            nonlocal model
            branch=out/('A_hierarchical' if stage.startswith('A') else 'B_direct')
            folder=branch/'inference'/name/stage;record_path=folder/(key+'.json')
            actual=replace(cfg,generation=replace(cfg.generation,max_new_tokens=budget))
            fp=fingerprint(dict(settings=settings,prompt=prompt,generation=asdict(actual.generation)))
            if record_path.exists():
                old=read_json(record_path)
                if old['fingerprint']!=fp:raise ValueError('Stage input/settings changed')
                if old['status']=='ok':return old['text']
            folder.mkdir(parents=True,exist_ok=True)
            attempt=1
            while (folder/f'{key}.attempt_{attempt}.json').exists():attempt+=1
            (folder/f'{key}.attempt_{attempt}.prompt.txt').write_text(prompt)
            if args.dry_run:return ''
            if model is None:
                progress('model_loading')
                from vlm_inference.model import Qwen3VLCaptioner
                start=time.perf_counter();model=Qwen3VLCaptioner(cfg)
                logging.info('Model loaded once in %.2fs; independent fresh text calls for A/B',time.perf_counter()-start)
            model.cfg=actual
            progress(stage,current_video=name,current_item=key)
            logging.info('START %s %s %s; max_new_tokens=%s',stage,name,key,budget)
            start=time.perf_counter();result=model.generate_text(prompt)
            record=dict(fingerprint=fp,status='truncated' if result.hit_token_limit else ('ok' if result.text.strip() else 'empty'),
                        text=result.text,input_type='text_only',generation=asdict(actual.generation),
                        elapsed_sec=time.perf_counter()-start,num_output_tokens=result.num_output_tokens,hit_token_limit=result.hit_token_limit)
            atomic_json(folder/f'{key}.attempt_{attempt}.json',record);atomic_json(record_path,record)
            logging.info('DONE %s %s: %s, %s tokens, %.2fs',stage,key,record['status'],result.num_output_tokens,record['elapsed_sec'])
            if record['status']!='ok':raise ValueError(f'Generation {record["status"]}; raw retained at {record_path}')
            return record['text']
        for path,raw,doc,clips,windows in documents:
            name=doc['video'];intermediate=[];outputs={};errors=[]
            if args.pipeline in ['A','both']:
                # A1 uses original captions only; no earlier intermediate output is read.
                for i,w in enumerate(windows):
                    prompt=insert(templates['A1'],window_start=w['start_sec'],window_end=w['end_sec'],timestamped_3s_captions=timestamped(w['source_segments']))
                    description=text_call('A1',name,f'{i:04d}',prompt,2048)
                    intermediate.append(dict(window_start=w['start_sec'],window_end=w['end_sec'],description=description,
                                             source_clips=[c['source_clip'] for c in w['source_segments']],source_segments=w['source_segments']))
                if not args.dry_run:
                    if [c for w in intermediate for c in w['source_segments']]!=clips:raise ValueError('Missing/duplicated source evidence')
                    if len({w['window_start'] for w in intermediate})!=len(windows):raise ValueError('Duplicated window')
                    publish_json(out/'A_hierarchical'/'intermediate'/(name+'.json'),dict(video=name,window_duration_sec=30,windows=intermediate))
                prompt_a=insert(templates['A2'],INPUT_DESCRIPTIONS=timestamped(intermediate,'window_start','window_end','description'))
                raw_a=text_call('A2',name,'events',prompt_a,max(4096,min(16384,1024+64*len(intermediate))))
                if not args.dry_run:
                    try:outputs['A']=parse_events(raw_a,intermediate_boundaries(intermediate,{b for c in clips for b in [c['start_sec'],c['end_sec']]}))
                    except (ValueError,TypeError) as exc:errors.append('A: '+str(exc))
            if args.pipeline in ['B','both']:
                # B uses only immutable original rows, never A output or conversational history.
                prompt_b=insert(templates['B'],INPUT_DESCRIPTIONS=timestamped(clips))
                raw_b=text_call('B',name,'events',prompt_b,max(4096,min(16384,1024+64*len(clips))))
                if not args.dry_run:
                    try:outputs['B']=parse_events(raw_b,{b for c in clips for b in [c['start_sec'],c['end_sec']]})
                    except (ValueError,TypeError) as exc:errors.append('B: '+str(exc))
            if args.dry_run:continue
            validation=dict(video=name,pipeline=args.pipeline,source_sha256=hashlib.sha256(raw).hexdigest(),source_unchanged=path.read_bytes()==raw,
                            input_clips=len(clips),intermediate_windows=len(intermediate),missing_windows=[],duplicated_windows=[],
                            fine_precision_A='Window endpoints plus explicitly timestamped evidence retained in A1',fine_precision_B='3s clip boundaries',errors=errors)
            if not validation['source_unchanged']:validation['errors'].append('Source changed')
            validation['passed']=not validation['errors']
            atomic_json(out/'validation'/(name+'.json'),validation)
            if not validation['source_unchanged']:raise ValueError('Source changed')
            # Independently valid branch outputs remain available even when the other branch fails.
            for stage,branch in [('A','A_hierarchical'),('B','B_direct')]:
                if stage in outputs:publish_json(out/branch/'events'/(name+'.json'),outputs[stage])
            if not validation['passed']:
                progress('validation_failed',current_video=name);raise ValueError(str(validation['errors']))
            files=[str(out/('A_hierarchical' if stage=='A' else 'B_direct')/'events'/(name+'.json')) for stage in outputs]
            if args.pipeline=='both':
                review=out/'comparison'/(name+'.md');review.parent.mkdir(parents=True,exist_ok=True)
                content=comparison_md(name,intermediate,outputs['A'],outputs['B'])
                if review.exists() and review.read_text()!=content:raise ValueError('Existing comparison differs; use a new directory')
                if not review.exists():review.write_text(content)
                files.append(str(review));print(review.read_text(),flush=True)
            validation['event_counts']={stage:len(v['events']) for stage,v in outputs.items()}
            validation['output_files']=files
            for f in files:
                if not Path(f).is_file():raise ValueError('Missing output file')
            atomic_json(out/'validation'/(name+'.json'),validation)
            completed+=1;progress('video_complete',current_video=name)
        for path,raw,_,_,_ in documents:
            if path.read_bytes()!=raw:raise ValueError('Source changed during run')
        progress('dry_run' if args.dry_run else 'complete')
        return 0

if __name__=='__main__':raise SystemExit(main())
