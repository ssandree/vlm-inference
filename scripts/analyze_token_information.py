#!/usr/bin/env python3
"""Feasibility: cached full-frame Qwen3-VL block information.

Each selected frame is scored independently against only the GT event
sentences marked visible at that timestamp. The decoder suffix engine is
unchanged: one full forward, then 2x2-token sufficiency interventions.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import yaml
from PIL import Image
from qwen_vl_utils import process_vision_info

from vlm_inference.config import load_config, JobConfig
from vlm_inference.inputs import Sample
from vlm_inference.experiment import parse_labeled_frame
from vlm_inference.model import Qwen3VLCaptioner
from vlm_inference.prompts import load_job_prompt
from vlm_inference.roi_margin import load_tube_bboxes
from vlm_inference.token_information import (
    CachedQwenSuffix, information_roi_metrics, make_blocks, roi_membership,
    sentence_means, sentence_token_spans,
)

LOG=logging.getLogger('token_information')
CSV_FIELDS=['video_id','frame_idx','timestamp','matched_event','layer',
            'ROI_block_ratio','ROI_information_coverage','information_enrichment',
            'ROI_information_density','nonROI_information_density','density_ratio']
MIN_FRAMES=5
MAX_FRAMES=10


def dump(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def csv_value(value):
    return '' if value is None else value


def load_reference_sentences(analysis, catalog):
    if catalog:
        return {int(k):str(v['sentence']).strip() for k,v in catalog.items()}
    if 'event_sentences' in analysis:
        return {i+1:s.strip() for i,s in enumerate(analysis['event_sentences'])}
    if 'gt_caption' in analysis:
        import re
        sentences=re.split(r'(?<=[.!?。！？])\s+', analysis['gt_caption'].strip())
        return {i+1:s.strip() for i,s in enumerate(sentences)}
    refs=json.loads(Path(analysis['gt_reference_file']).read_text())
    caption=refs.get('references',refs)[analysis['video_id']]
    if isinstance(caption,list):
        caption=caption[int(analysis.get('gt_reference_index',0))]
    import re
    sentences=re.split(r'(?<=[.!?。！？])\s+', caption.strip())
    return {i+1:s.strip() for i,s in enumerate(sentences)}


def load_frame_event_map(path):
    raw=yaml.safe_load(Path(path).read_text())
    events={int(k):dict(v) for k,v in (raw.get('events') or {}).items()}
    frames=[]
    for item in raw['frames']:
        frame=dict(item)
        frame['matched_event_ids']=[int(i) for i in frame.get('matched_event_ids') or []]
        frame['uncertain_event_ids']=[int(i) for i in frame.get('uncertain_event_ids') or []]
        if not frame['matched_event_ids']:
            raise ValueError(f"{frame.get('path')}: matched_event_ids is empty; edit the mapping file")
        if set(frame['matched_event_ids']) & set(frame['uncertain_event_ids']):
            raise ValueError(f"{frame.get('path')}: an event cannot be both matched and uncertain")
        frames.append(frame)
    return dict(video_id=raw['video_id'],events=events,frames=frames,raw=raw)


def resolve_analysis_frames(analysis):
    if 'frame_event_map' in analysis:
        mapping=load_frame_event_map(analysis['frame_event_map'])
        if mapping['video_id']!=analysis['video_id']:
            raise ValueError('frame_event_map video_id does not match token_information.video_id')
        sentences=load_reference_sentences(analysis, mapping['events'])
        for event_id,meta in mapping['events'].items():
            if event_id in sentences and meta.get('sentence') and meta['sentence'].strip()!=sentences[event_id]:
                raise ValueError(f'Event {event_id} sentence differs between map and GT reference')
            sentences[event_id]=str(meta.get('sentence') or sentences[event_id]).strip()
        return mapping['frames'], sentences, mapping
    frames=list(analysis.get('frames') or [])
    if any('matched_event_ids' not in f for f in frames):
        raise ValueError(
            'Provide token_information.frame_event_map or per-frame matched_event_ids. '
            'Scoring every video-level event sentence on a single frame is disabled.')
    for frame in frames:
        frame['matched_event_ids']=[int(i) for i in frame['matched_event_ids']]
        frame['uncertain_event_ids']=[int(i) for i in frame.get('uncertain_event_ids') or []]
    return frames, load_reference_sentences(analysis, {}), None


def resolve_roi(analysis, metadata):
    roi=analysis.get('roi',{})
    if 'json_file' in roi:
        roi=json.loads(Path(roi['json_file']).read_text())
    elif 'tracks_file' in roi:
        boxes_by_frame={}
        for f in metadata:
            boxes=load_tube_bboxes(Path(roi['tracks_file']),frame_width=f['original_width'],frame_height=f['original_height'])
            selected=roi.get('tube_ids',list(boxes))
            boxes_by_frame[str(f['frame_idx'])]={'bboxes':[list(boxes[int(i)].bbox_px) for i in selected]}
        roi={'frames':boxes_by_frame}
    return roi


def prepare_frame(captioner, cfg, analysis, frame, sentences, frame_idx):
    labeled=parse_labeled_frame(str(frame['path']),cfg.experiment)
    if labeled is None or labeled.kind!='global':
        raise ValueError('Each analysis image must match the configured Global filename pattern')
    if abs(labeled.time_s-float(frame['timestamp']))>1e-6:
        raise ValueError('Configured timestamp differs from Global image filename')
    if float(frame['timestamp'])<0 or not np.isfinite(float(frame['timestamp'])):
        raise ValueError('Frame timestamps must be finite and nonnegative')
    sample=Sample(job_name='token_information',sample_id=analysis['video_id'],input_type='images',
        paths=[Path(frame['path']).resolve().as_uri()])
    prompt=load_job_prompt(cfg,JobConfig(name='token_information',type='images'))
    messages=captioner._build_messages(sample,prompt)
    if any('[ROI' in item.get('text','') and item.get('text')!=prompt for item in messages[0]['content']):
        raise ValueError('Analysis accepts full-frame Global images only')
    prefix=captioner.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos,_video_kwargs=process_vision_info(messages,image_patch_size=16,
        return_video_kwargs=True,return_video_metadata=True)
    if videos is not None:
        raise ValueError('Use independent full-frame images, not temporally merged video tokens')
    inputs=captioner.processor(text=[prefix],images=images,return_tensors='pt',do_resize=False)
    event_ids=list(frame['matched_event_ids'])
    event_sentences=[sentences[i] for i in event_ids]
    if any(not s for s in event_sentences):
        raise ValueError(f"{frame['path']}: a matched event sentence is empty")
    caption=' '.join(event_sentences)
    ids,spans=sentence_token_spans(captioner.processor.tokenizer,caption,event_sentences)
    for span,event_id in zip(spans,event_ids):
        span['event_id']=int(event_id)
    prefix_length=inputs.input_ids.shape[1]
    target=torch.tensor([ids],dtype=inputs.input_ids.dtype)
    inputs['input_ids']=torch.cat([inputs.input_ids,target],dim=1)
    inputs['attention_mask']=torch.cat([inputs.attention_mask,torch.ones_like(target)],dim=1)
    if 'mm_token_type_ids' in inputs:
        inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(target)],dim=1)
    visual=(inputs.input_ids[0]==captioner.model.config.image_token_id).nonzero().flatten().tolist()
    if (inputs.input_ids==captioner.model.config.video_token_id).any():
        raise ValueError('Video tokens unsupported; use independent image sequence')
    with Image.open(frame['path']) as im:
        metadata=[dict(path=frame['path'],timestamp=float(frame['timestamp']),
                       original_width=im.width,original_height=im.height)]
    if captioner.model.config.vision_config.patch_size!=16 or captioner.model.config.vision_config.spatial_merge_size!=2:
        raise ValueError('Fixed 32px mapping requires 16px patches and merge size 2')
    blocks,metadata=make_blocks(visual,inputs.image_grid_thw.tolist(),metadata)
    if not blocks:
        raise ValueError('No complete 2x2 visual blocks')
    roi=resolve_roi(analysis, metadata)
    in_roi=roi_membership(blocks,metadata,roi)
    if 'frames' in roi and '0' in roi['frames']:
        roi={'frames':{str(frame_idx):roi['frames']['0']}}
    metadata[0]['frame_idx']=frame_idx
    for b,flag in zip(blocks,in_roi):
        b['frame_idx']=frame_idx
        b['in_roi']=bool(flag)
    return dict(inputs=inputs,visual=visual,predictors=list(range(prefix_length-1,prefix_length+len(ids)-1)),
        target_ids=ids,spans=spans,blocks=blocks,metadata=metadata[0],in_roi=in_roi,roi=roi,
        prompt=prompt,chat_prefix=prefix,prefix_tokens=prefix_length,gt_caption=caption,
        event_ids=event_ids,event_sentences=event_sentences)


def metric_row(video_id, frame_idx, timestamp, matched_event, layer, metrics):
    return dict(video_id=video_id,frame_idx=frame_idx,timestamp=timestamp,matched_event=matched_event,layer=layer,
        ROI_block_ratio=metrics['roi_block_ratio'],
        ROI_information_coverage=metrics['roi_information_coverage'],
        information_enrichment=metrics['information_enrichment'],
        ROI_information_density=metrics['roi_information_density'],
        nonROI_information_density=metrics['non_roi_information_density'],
        density_ratio=metrics['density_ratio'])


def mean_rows(rows, matched_event, layer):
    if not rows:
        return None
    out=dict(video_id=rows[0]['video_id'],frame_idx=None,timestamp=None,matched_event=matched_event,layer=layer)
    for key in CSV_FIELDS[5:]:
        values=[r[key] for r in rows if r[key] is not None]
        out[key]=float(np.mean(values)) if values else None
    return out


def block_grid(frame, blocks, values):
    grid=np.full((frame['block_rows'],frame['block_cols']),np.nan)
    for b,v in zip(blocks,values):
        grid[b['block_row'],b['block_col']]=v
    return grid


def roi_bboxes(frame, roi):
    frames=roi.get('frames',{})
    entry=frames.get(str(frame['frame_idx'])) or frames.get('0') or roi
    return [list(box) for box in entry.get('bboxes',[])]


def draw_roi(ax, frame, blocks, roi, facecolor=None, edgecolor='lime'):
    from matplotlib.patches import Rectangle
    for box in roi_bboxes(frame, roi):
        ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                               fill=facecolor is not None,facecolor=facecolor,edgecolor='red',linewidth=1.4,alpha=.25 if facecolor else 1))
    for b in blocks:
        if b['in_roi']:
            ax.add_patch(Rectangle((b['x1'],b['y1']),64,64,fill=False,edgecolor=edgecolor,linewidth=.35))


def save_heatmap(path, grid, vmax):
    import matplotlib.pyplot as plt
    plt.imsave(path, np.repeat(np.repeat(grid,64,axis=0),64,axis=1), cmap='coolwarm', vmin=-vmax, vmax=vmax)


def plot_overview(path, frame, blocks, values, roi, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vmax=max(float(np.nanmax(np.abs(values))), 1e-8)
    grid=block_grid(frame, blocks, values)
    save_heatmap(path.with_name(path.stem+'_heatmap.png'), grid, vmax)
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    with Image.open(frame['path']) as source:
        image=source.convert('RGB')
    axes[0,0].imshow(image); axes[0,0].set_title('original full frame'); axes[0,0].axis('off')
    axes[0,1].imshow(image)
    draw_roi(axes[0,1], frame, blocks, roi, facecolor=(0,1,0,.12))
    axes[0,1].set_xlim(0,frame['original_width']); axes[0,1].set_ylim(frame['original_height'],0)
    axes[0,1].set_title('ROI overlay'); axes[0,1].axis('off')
    im=axes[1,0].imshow(grid,cmap='coolwarm',vmin=-vmax,vmax=vmax,interpolation='nearest')
    axes[1,0].set_title('64×64 information heatmap'); axes[1,0].axis('off')
    axes[1,1].imshow(image)
    extent=(0,frame['block_cols']*64,frame['block_rows']*64,0)
    axes[1,1].imshow(grid,cmap='coolwarm',vmin=-vmax,vmax=vmax,alpha=.65,interpolation='nearest',extent=extent)
    draw_roi(axes[1,1], frame, blocks, roi)
    axes[1,1].set_xlim(0,frame['original_width']); axes[1,1].set_ylim(frame['original_height'],0)
    axes[1,1].set_title('information + ROI boundary'); axes[1,1].axis('off')
    fig.colorbar(im,ax=list(axes.ravel()),label='Information (nats / target token)',shrink=.6)
    fig.suptitle(title,fontsize=11)
    fig.savefig(path,dpi=130,bbox_inches='tight')
    plt.close(fig)


def plot_layer_comparison(out, frame, blocks, layer_values, layers, name, roi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vmax=max(float(np.max(np.abs(v))) for v in layer_values.values()) or 1e-8
    fig,axes=plt.subplots(1,len(layers),figsize=(5*len(layers),4),squeeze=False)
    im=None
    for ax,l in zip(axes[0],layers):
        grid=block_grid(frame, blocks, layer_values[l])
        directory=out/f'layer_{l:02d}'
        save_heatmap(directory/f"frame_{frame['frame_idx']:02d}_{name}_heatmap.png", grid, vmax)
        with Image.open(frame['path']) as source:
            ax.imshow(source)
        extent=(0,frame['block_cols']*64,frame['block_rows']*64,0)
        im=ax.imshow(grid,cmap='coolwarm',vmin=-vmax,vmax=vmax,alpha=.65,interpolation='nearest',extent=extent)
        draw_roi(ax, frame, blocks, roi)
        ax.set_xlim(0,frame['original_width']); ax.set_ylim(frame['original_height'],0)
        ax.set_title(f'layer {l}, {name}, t={frame["timestamp"]:g}s'); ax.axis('off')
    fig.colorbar(im,ax=list(axes[0]),label='Information (nats / target token)',shrink=.7)
    fig.savefig(out/f"frame_{frame['frame_idx']:02d}_{name}_layer_comparison.png",dpi=130,bbox_inches='tight')
    plt.close(fig)


def spatial_stats(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),std=float(values.std()),
        minimum=float(values.min()),maximum=float(values.max()),
        range=float(values.max()-values.min()),
        positive_block_ratio=float((values>0).mean()))


def analyze_frame(engine_factory, prepared, layers, batch_size, tolerance, video_id):
    inputs=prepared['inputs'].to(engine_factory['device'])
    engine=CachedQwenSuffix(engine_factory['model'],layers,prepared['visual'],prepared['predictors'],prepared['target_ids'])
    full=engine.capture(inputs)
    del inputs
    full_means=sentence_means(full,prepared['spans'])[0]
    layer_values={}
    summaries={}
    blocks=prepared['blocks']
    in_roi=prepared['in_roi']
    visual_tokens=len(prepared['visual'])
    event_ids=prepared['event_ids']
    for l in layers:
        layer_start=time.time()
        replay=engine.resume(l,None)
        replay_error=float((replay-full).abs().max())
        LOG.info('Layer %d retain-all max token log-prob error %.6f',l,replay_error)
        if replay_error>tolerance:
            raise RuntimeError(f'Layer {l} retain-all replay failed: {replay_error} > {tolerance}')
        baseline=engine.resume(l,[[]])
        baseline_means=sentence_means(baseline,prepared['spans'])[0]
        values=[]
        batch_error=None
        for start in range(0,len(blocks),batch_size):
            group=[b['token_indices'] for b in blocks[start:start+batch_size]]
            scores=engine.resume(l,group)
            if start==0:
                single=engine.resume(l,[group[0]])
                batch_error=float((single[0]-scores[0]).abs().max())
                if batch_error>tolerance:
                    raise RuntimeError(f'Batch/individual mismatch: {batch_error} > {tolerance}')
            values.append(sentence_means(scores,prepared['spans']).numpy()-baseline_means.numpy())
            if start % (batch_size*5)==0:
                LOG.info('Layer %d blocks %d/%d',l,min(start+batch_size,len(blocks)),len(blocks))
        information=np.concatenate(values)
        layer_values[l]=information
        rows=[]
        for b,v in zip(blocks,information):
            row=dict(video_id=video_id,layer=l,**b,importance_mean=float(v.mean()),importance_max=float(v.max()),
                **{f'importance_event_{event_id}':float(x) for event_id,x in zip(event_ids,v)})
            rows.append(row)
        per_event={event_id:information_roi_metrics(information[:,i],in_roi,visual_tokens)
                   for i,event_id in enumerate(event_ids)}
        max_map=information.max(axis=1)
        mean_map=information.mean(axis=1)
        summary=dict(layer=l,matched_event_ids=event_ids,
            baseline_token_log_probabilities=baseline[0].tolist(),
            baseline_sentence_log_probabilities=baseline_means.tolist(),
            baseline_caption_sentence_mean=float(baseline_means.mean()),
            event_roi_metrics={str(k):v for k,v in per_event.items()},
            max_event_roi_metrics=information_roi_metrics(max_map,in_roi,visual_tokens),
            mean_event_roi_metrics=information_roi_metrics(mean_map,in_roi,visual_tokens),
            validation=dict(retain_all_max_token_logprob_error=replay_error,
                            batch_single_max_token_logprob_error=batch_error,atol=tolerance),
            spatial_statistics=dict(max_event=spatial_stats(max_map),mean_event=spatial_stats(mean_map),
                **{f'event_{event_id}':spatial_stats(information[:,i]) for i,event_id in enumerate(event_ids)}),
            elapsed_seconds=time.time()-layer_start)
        summaries[l]=summary
        LOG.info('Layer %d complete in %.1fs; max-event coverage %s enrichment %s',
                 l,summary['elapsed_seconds'],
                 summary['max_event_roi_metrics']['roi_information_coverage'],
                 summary['max_event_roi_metrics']['information_enrichment'])
    if engine.counts['vision_forwards']!=1 or engine.counts['full_forwards']!=1:
        raise RuntimeError('Expected exactly one vision forward and one full decoder forward per frame')
    counts=dict(engine.counts)
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(full_token_log_probabilities=full[0].tolist(),
        full_sentence_log_probabilities=full_means.tolist(),
        full_caption_sentence_mean=float(full_means.mean()),
        layer_values=layer_values,summaries=summaries,counts=counts,
        )


def run(config, args):
    raw=yaml.safe_load(Path(config).read_text())
    analysis=raw['token_information']
    if args.layers is not None:
        analysis['layers']=args.layers
    if args.batch_size is not None:
        analysis['batch_size']=args.batch_size
    if args.output_dir:
        raw['output']['dir']=args.output_dir
    if args.map:
        analysis['frame_event_map']=args.map
    out=Path(raw['output']['dir']);out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(config);cfg.output.dir=str(out)
    if cfg.mode != 'experiment':
        raise ValueError('Use experiment mode to preserve Global frame labels and timestamps')
    frames,sentences,mapping=resolve_analysis_frames(analysis)
    if not MIN_FRAMES<=len(frames)<=MAX_FRAMES:
        raise ValueError(f'Feasibility uses {MIN_FRAMES}–{MAX_FRAMES} frames from one video; got {len(frames)}')
    if any(float(a['timestamp'])>=float(b['timestamp']) for a,b in zip(frames,frames[1:])):
        raise ValueError('Frames must be in strictly chronological order')
    layers=analysis.get('layers',[0,4,8])
    if any(not isinstance(l,int) or isinstance(l,bool) for l in layers):
        raise ValueError('Layer indices must be integers')
    if len(set(layers))!=len(layers):
        raise ValueError('Duplicate layers')
    batch_size=analysis.get('batch_size',8)
    if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:
        raise ValueError('batch_size must be a positive integer')
    tolerance=float(analysis.get('validation_atol',0.03))
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('validation_atol must be finite and positive')
    dump(out/'analysis_config.json',raw)
    if mapping is not None:
        dump(out/'frame_event_map.json',mapping['raw'])
    started=time.time()
    LOG.info('Loading model through existing Qwen3VLCaptioner')
    captioner=Qwen3VLCaptioner(cfg)
    prompt=load_job_prompt(cfg,JobConfig(name='token_information',type='images'))
    frame_results=[]
    all_rows=[]
    counts=dict(vision_forwards=0,full_forwards=0,suffix_batches=0,suffix_examples=0)
    for frame_idx,frame in enumerate(frames):
        LOG.info('Frame %d/%d t=%gs events=%s',frame_idx+1,len(frames),float(frame['timestamp']),frame['matched_event_ids'])
        prepared=prepare_frame(captioner,cfg,analysis,frame,sentences,frame_idx)
        result=analyze_frame(dict(model=captioner.model,device=next(captioner.model.parameters()).device),
            prepared,layers,batch_size,tolerance,analysis['video_id'])
        for key in counts:
            counts[key]+=result['counts'][key]
        frame_dir_meta=dict(prepared['metadata'])
        frame_dir_meta.update(
            frame_idx=frame_idx,timestamp=float(frame['timestamp']),path=frame['path'],
            matched_event_ids=prepared['event_ids'],matched_events=[
                dict(event_id=i,sentence=sentences[i]) for i in prepared['event_ids']],
            uncertain_event_ids=frame.get('uncertain_event_ids',[]),
            uncertain_events=[dict(event_id=i,sentence=sentences.get(i,''),
                                   note='Listed for review only; not scored.')
                              for i in frame.get('uncertain_event_ids',[])],
            note=frame.get('note',''),gt_caption=prepared['gt_caption'],sentence_spans=prepared['spans'])
        for l,information in result['layer_values'].items():
            directory=out/f'layer_{l:02d}';directory.mkdir(exist_ok=True)
            rows=[]
            for b,v in zip(prepared['blocks'],information):
                rows.append(dict(video_id=analysis['video_id'],layer=l,**b,
                    importance_mean=float(v.mean()),importance_max=float(v.max()),
                    **{f'importance_event_{event_id}':float(x) for event_id,x in zip(prepared['event_ids'],v)}))
            dump(directory/f"frame_{frame_idx:02d}_blocks.json",rows)
            with (directory/f"frame_{frame_idx:02d}_blocks.csv").open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
                for row in rows:
                    writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
            dump(directory/f"frame_{frame_idx:02d}_summary.json",result['summaries'][l])
            summary=result['summaries'][l]
            timestamp=float(frame['timestamp'])
            for event_id in prepared['event_ids']:
                all_rows.append(metric_row(analysis['video_id'],frame_idx,timestamp,f'event_{event_id}',l,
                    summary['event_roi_metrics'][str(event_id)]))
            all_rows.append(metric_row(analysis['video_id'],frame_idx,timestamp,'max_event',l,
                summary['max_event_roi_metrics']))
            all_rows.append(metric_row(analysis['video_id'],frame_idx,timestamp,'event_mean',l,
                summary['mean_event_roi_metrics']))
            plot_overview(directory/f"frame_{frame_idx:02d}_overview.png", prepared['metadata'],
                prepared['blocks'], information.max(axis=1), prepared['roi'],
                f"{analysis['video_id']}  t={timestamp:g}s  layer {l}  I_max  events={prepared['event_ids']}")
            for i,event_id in enumerate(prepared['event_ids']):
                save_heatmap(directory/f"frame_{frame_idx:02d}_event_{event_id}_heatmap.png",
                    block_grid(prepared['metadata'],prepared['blocks'],information[:,i]),
                    max(float(np.max(np.abs(information[:,i]))),1e-8))
            save_heatmap(directory/f"frame_{frame_idx:02d}_mean_heatmap.png",
                block_grid(prepared['metadata'],prepared['blocks'],information.mean(axis=1)),
                max(float(np.max(np.abs(information.mean(axis=1)))),1e-8))
        for i,event_id in enumerate(prepared['event_ids']):
            plot_layer_comparison(out, prepared['metadata'], prepared['blocks'],
                {l:result['layer_values'][l][:,i] for l in layers}, layers, f'event_{event_id}', prepared['roi'])
        plot_layer_comparison(out, prepared['metadata'], prepared['blocks'],
            {l:result['layer_values'][l].max(axis=1) for l in layers}, layers, 'max', prepared['roi'])
        plot_layer_comparison(out, prepared['metadata'], prepared['blocks'],
            {l:result['layer_values'][l].mean(axis=1) for l in layers}, layers, 'mean', prepared['roi'])
        dump(out/f'frame_{frame_idx:02d}_full_caption_scores.json',
             dict(token_log_probabilities=result['full_token_log_probabilities'],
                  sentence_log_probabilities=result['full_sentence_log_probabilities'],
                  caption_sentence_mean=result['full_caption_sentence_mean'],
                  matched_event_ids=prepared['event_ids']))
        frame_results.append(dict(frame=frame_dir_meta,layers=result['summaries'],
            visual_tokens=len(prepared['visual']),n_blocks=len(prepared['blocks']),
            roi=prepared['roi'],prompt=prepared['prompt'],chat_prefix=prepared['chat_prefix'],
            prefix_tokens=prepared['prefix_tokens'],target_token_ids=prepared['target_ids'],
            counts=result['counts']))
        dump(out/'progress.json',dict(completed_frames=frame_idx+1,total_frames=len(frames),
            completed_layers=layers,counts=counts))
    video_averages=[]
    for l in layers:
        for name in sorted({r['matched_event'] for r in all_rows}):
            subset=[r for r in all_rows if r['layer']==l and r['matched_event']==name]
            avg=mean_rows(subset,name,l)
            if avg is not None:
                video_averages.append(avg)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS);writer.writeheader()
        for row in all_rows+[{**r,'frame_idx':'video_mean','timestamp':'video_mean'} for r in video_averages]:
            writer.writerow({k:csv_value(row[k]) for k in CSV_FIELDS})
    manifest=dict(video_id=analysis['video_id'],prompt=prompt,
        frames=[fr['frame'] for fr in frame_results],
        events={str(k):v for k,v in sentences.items()},
        matching_rule='Only matched_event_ids are scored. uncertain_event_ids are recorded, never guessed into the target.',
        mapping='Fixed 32px/token and 64px/block, no inverse resize correction; bbox center determines ROI inclusion.',
        coverage_domain='Complete 2x2 groups only. Ungrouped edge tokens excluded from block coverage but counted in total visual token ratio.',
        layer_convention='After zero-based decoder layer l, including DeepStack addition.',
        baseline='All visual positions removed at the cached boundary; cached text still contains earlier visual context.',
        importance='Sentence mean log probability difference (nats/token). I_max = max over matched events at that frame. event_mean is auxiliary.',
        paper='https://arxiv.org/html/2512.07580v2')
    dump(out/'manifest.json',manifest)
    dump(out/'summary.json',dict(video_id=analysis['video_id'],layers=layers,
        frames=[{k:fr['frame'][k] for k in
                 ('frame_idx','timestamp','path','matched_event_ids','uncertain_event_ids','note')}
                for fr in frame_results],
        frame_metrics=all_rows,video_mean_metrics=video_averages,
        frame_layer_summaries=[{'frame_idx':fr['frame']['frame_idx'],'timestamp':fr['frame']['timestamp'],
                                'layers':fr['layers']} for fr in frame_results],
        counts=counts,elapsed_seconds=time.time()-started,
        torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,
        default_layer_note='No layer is auto-selected. Compare layers 0/4/8 on spatial discrimination and score stability.',
        validation='All retain-all and batch/individual checks passed. One vision/full decoder forward per frame.'))
    LOG.info('Saved analysis to %s (%.1fs)',out,time.time()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/token_information_feasibility.yaml')
    p.add_argument('--layers',type=int,nargs='+')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--output-dir')
    p.add_argument('--map',help='Override token_information.frame_event_map')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run(args.config,args)

if __name__=='__main__':
    main()
