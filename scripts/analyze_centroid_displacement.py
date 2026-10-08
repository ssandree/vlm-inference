#!/usr/bin/env python3
"""Read-only diagnostics of saved centroid pairs and fixed-padding cost.

No decoding, crop generation, frame selection, inference, or adaptive policy.
Paths in the analysis YAML are relative to the repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_inference.centroid_motion import centroid_displacement, trigger_windows


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values, prefix=''):
    a = np.asarray(values, dtype=float)
    keys = ['mean', 'median', 'std', 'min', 'max', 'p50', 'p75', 'p90', 'p95', 'p99']
    if not len(a):
        return {prefix + k: None for k in keys}
    vals = [a.mean(), np.median(a), a.std(ddof=0), a.min(), a.max(), *np.percentile(a, [50, 75, 90, 95, 99])]
    return {prefix + k: float(v) for k, v in zip(keys, vals)}


def rank_rows(rows, scope):
    """Ascending average rank, ties share rank; ECDF percentile includes ties."""
    a = np.sort([r['normalized_displacement'] for r in rows])
    for r in rows:
        lo = int(np.searchsorted(a, r['normalized_displacement'], side='left'))
        hi = int(np.searchsorted(a, r['normalized_displacement'], side='right'))
        r[f'trigger_rank_within_{scope}'] = (lo + 1 + hi) / 2
        r[f'trigger_percentile_within_{scope}'] = 100 * hi / len(a)


def summary(rows, nvalid, tau):
    triggers = [r for r in rows if r['is_trigger']]
    intervals = [r['previous_trigger_gap_sec'] for r in triggers if r['previous_trigger_gap_sec'] is not None]
    return dict(num_valid_centroids=nvalid, num_valid_displacements=len(rows),
                num_triggers=len(triggers), threshold=tau,
                trigger_fraction=len(triggers)/len(rows) if rows else None,
                **stats([r['normalized_displacement'] for r in rows], 'displacement_'),
                **stats([r['normalized_displacement'] for r in triggers], 'trigger_displacement_'),
                num_trigger_intervals=len(intervals), **stats(intervals, 'trigger_interval_sec_'),
                **stats([r['delta_t'] for r in rows], 'valid_pair_delta_t_sec_'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/centroid_displacement_analysis.yaml')
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    experiment_path = ROOT / cfg['experiment_config']
    exp = yaml.safe_load(experiment_path.read_text())
    videos = exp['feasibility']['videos']
    assert len(videos) == len(set(videos)) == 9
    detector = exp['experiment']['centroid_motion']
    tau = float(detector['centroid_displacement_threshold'])
    base_fps, high_fps = detector['base_fps'], detector['high_fps']
    source = ROOT / cfg['centroid_results_dir']
    out = ROOT / cfg['output_dir']
    assert out.resolve() != source.resolve() and source.resolve() not in out.resolve().parents
    out.mkdir(parents=True, exist_ok=True)
    assignment_path = source / 'assignments.json'
    assignments = read_json(assignment_path)['rois']
    assignments = [r for r in assignments if r['video_id'] in videos]
    assert {r['video_id'] for r in assignments} == set(videos)
    assert len({(r['video_id'],r['roi_id']) for r in assignments}) == len(assignments)
    inputs = [cfg_path, experiment_path, assignment_path]
    rows, roi_summaries, costs, interval_rows = [], [], [], []
    valid_by_video = {v: 0 for v in videos}
    for assignment in assignments:
        video, roi = assignment['video_id'], assignment['roi_id']
        path = source / video / 'debug' / f'roi_{roi:02d}_centroid.json'
        inputs.append(path)
        data = read_json(path)
        assert data['video_id'] == video and data['roi_id'] == roi
        for key in ('centroid_displacement_threshold', 'base_fps', 'high_fps'):
            assert data['config'][key] == detector[key], (path, key)
        previous, valid, local = None, 0, []
        timestamps = [f['timestamp'] for f in data['frames']]
        assert all(b > a for a,b in zip(timestamps,timestamps[1:]))
        bbox = data['bbox_px']
        for frame in data['frames']:
            xy = [frame['centroid_x'], frame['centroid_y']]
            if not all(v is not None and math.isfinite(v) for v in xy):
                assert frame['displacement'] is None and not frame['is_trigger']
                continue
            valid += 1
            if previous is not None:
                d = centroid_displacement([previous['centroid_x'],previous['centroid_y']],xy,bbox)
                assert d is not None and math.isclose(d,frame['displacement'],rel_tol=1e-12,abs_tol=1e-14)
                assert frame['prev_valid_centroid_timestamp'] == previous['timestamp']
                assert frame['is_trigger'] == (d >= tau)
                local.append(dict(video_id=video,roi_id=roi,timestamp=frame['timestamp'],
                    prev_valid_timestamp=previous['timestamp'],delta_t=frame['timestamp']-previous['timestamp'],
                    centroid_prev_x=previous['centroid_x'],centroid_prev_y=previous['centroid_y'],
                    centroid_x=xy[0],centroid_y=xy[1],normalized_displacement=d,is_trigger=d>=tau,
                    trigger_rank_within_roi=None,trigger_percentile_within_roi=None,
                    trigger_rank_within_video=None,trigger_percentile_within_video=None,
                    previous_trigger_gap_sec=None,next_trigger_gap_sec=None,trigger_tertile=None))
            else:
                assert frame['displacement'] is None and not frame['is_trigger']
            previous = frame
        triggers = [r for r in local if r['is_trigger']]
        ts = [r['timestamp'] for r in triggers]
        assert ts == data['trigger_timestamps'] == assignment['trigger_timestamps']
        assert len(local) == max(0,valid-1)
        rank_rows(triggers,'roi')
        for i,r in enumerate(triggers):
            r['previous_trigger_gap_sec'] = ts[i]-ts[i-1] if i else None
            r['next_trigger_gap_sec'] = ts[i+1]-ts[i] if i+1<len(ts) else None
        valid_by_video[video] += valid
        roi_summaries.append(dict(video_id=video,roi_id=roi,**summary(local,valid,tau)))
        rows.extend(local)
        t0,t1 = max(0.,assignment['t0_sec']),assignment['t1_sec']
        for pad in cfg['padding_seconds']:
            merged = trigger_windows(ts,pad_sec=pad,fps=high_fps,clip_t0=t0,clip_t1=t1)
            raw = sum(max(0.,min(t1,t+pad)-max(t0,t-pad)) for t in ts)
            union = sum(r['end_sec']-r['start_sec'] for r in merged)
            assert 0 <= union <= t1-t0+1e-8 and union <= raw+1e-8
            if pad == data['config']['pad_sec']:
                assert merged == data['high_fps_intervals']
            costs.append(dict(scope='roi',video_id=video,roi_id=roi,pad_sec=pad,num_triggers=len(ts),
                roi_duration_sec=t1-t0,nominal_window_duration_sum_sec=2*pad*len(ts),
                clipped_window_duration_sum_sec=raw,union_2fps_duration_sec=union,
                overlap_removed_sec=raw-union,num_union_intervals=len(merged),
                expected_additional_frames=union*(high_fps-base_fps),
                mean_clipped_window_sec_per_trigger=raw/len(ts) if ts else None,
                mean_unique_coverage_sec_per_trigger=union/len(ts) if ts else None,
                high_fps_coverage_fraction=union/(t1-t0) if t1>t0 else None))
            interval_rows.extend(dict(video_id=video,roi_id=roi,pad_sec=pad,**r) for r in merged)
    trigger_rows = [r for r in rows if r['is_trigger']]
    trigger_values = np.array([r['normalized_displacement'] for r in trigger_rows])
    cuts = np.percentile(trigger_values,[100/3,200/3]) if len(trigger_values) else [None,None]
    for r in trigger_rows:
        r['trigger_tertile'] = 'low' if r['normalized_displacement']<=cuts[0] else ('medium' if r['normalized_displacement']<=cuts[1] else 'high')
    video_summaries=[]
    for video in videos:
        local=[r for r in rows if r['video_id']==video]
        rank_rows([r for r in local if r['is_trigger']],'video')
        video_summaries.append(dict(video_id=video,num_rois=sum(r['video_id']==video for r in roi_summaries),**summary(local,valid_by_video[video],tau)))
    global_summary=dict(num_videos=len(videos),num_rois=len(assignments),**summary(rows,sum(valid_by_video.values()),tau))
    for multiplier in cfg['near_threshold_multipliers']:
        count=int(sum(tau<=v<=tau*multiplier for v in trigger_values))
        global_summary[f'triggers_tau_to_{multiplier:g}tau_count']=count
        global_summary[f'triggers_tau_to_{multiplier:g}tau_percent']=100*count/len(trigger_values) if len(trigger_values) else None
    global_summary['triggers_above_5tau_count']=int(sum(trigger_values>5*tau))
    global_summary['triggers_above_5tau_percent']=float(100*np.mean(trigger_values>5*tau)) if len(trigger_values) else None
    # Video/global cost sums ROI-seconds: simultaneous distinct ROI crops both cost frames.
    additive=['num_triggers','roi_duration_sec','nominal_window_duration_sum_sec','clipped_window_duration_sum_sec',
              'union_2fps_duration_sec','overlap_removed_sec','num_union_intervals','expected_additional_frames']
    for scope,groups in [('video',videos),('global',['ALL'])]:
        for video in groups:
            for pad in cfg['padding_seconds']:
                part=[r for r in costs if r['scope']=='roi' and r['pad_sec']==pad and (scope=='global' or r['video_id']==video)]
                rec=dict(scope=scope,video_id=video,roi_id='',pad_sec=pad,**{k:sum(r[k] for r in part) for k in additive})
                n=rec['num_triggers']
                rec.update(mean_clipped_window_sec_per_trigger=rec['clipped_window_duration_sum_sec']/n if n else None,
                           mean_unique_coverage_sec_per_trigger=rec['union_2fps_duration_sec']/n if n else None,
                           high_fps_coverage_fraction=rec['union_2fps_duration_sec']/rec['roi_duration_sec'])
                costs.append(rec)
    for scope,video,roi in {(r['scope'],r['video_id'],r['roi_id']) for r in costs}:
        group=sorted([r for r in costs if (r['scope'],r['video_id'],r['roi_id'])==(scope,video,roi)],key=lambda r:r['pad_sec'])
        baseline=group[0]['union_2fps_duration_sec']
        for r in group:
            r['extra_unique_coverage_vs_min_pad_sec_per_trigger']=(r['union_2fps_duration_sec']-baseline)/r['num_triggers'] if r['num_triggers'] else None
        assert all(b['union_2fps_duration_sec']>=a['union_2fps_duration_sec']-1e-8 for a,b in zip(group,group[1:]))
    bins=[]
    for name in ['low','medium','high']:
        group=[r for r in trigger_rows if r['trigger_tertile']==name]
        bins.append(dict(group=name,count=len(group),percent_of_all_triggers=100*len(group)/len(trigger_rows) if trigger_rows else 0,
                         **stats([r['normalized_displacement'] for r in group],'displacement_'),
                         **stats([r['delta_t'] for r in group],'delta_t_sec_'),
                         **stats([r['next_trigger_gap_sec'] for r in group if r['next_trigger_gap_sec'] is not None],'next_trigger_gap_sec_')))
    percentiles=[]
    for scope, summaries in [('roi',roi_summaries),('video',video_summaries),('global',[global_summary])]:
        for s in summaries:
            for population,prefix in [('all_valid_pairs','displacement_'),('triggers','trigger_displacement_')]:
                percentiles.append(dict(scope=scope,video_id=s.get('video_id','ALL'),roi_id=s.get('roi_id',''),population=population,
                                        **{p:s[prefix+p] for p in ['min','p50','p75','p90','p95','p99','max']}))
    for name,data in [('frame_displacements',rows),('roi_summary',roi_summaries),('video_summary',video_summaries),
                      ('global_summary',[global_summary]),('percentiles',percentiles),('trigger_tertiles',bins),
                      ('fixed_padding_cost',costs),('fixed_padding_union_intervals',interval_rows)]:
        write_csv(out/f'{name}.csv',data)
    (out/'frame_displacements.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    for name,values,title in [('displacement_histogram',[r['normalized_displacement'] for r in rows],'All valid centroid pairs'),
                              ('trigger_histogram',trigger_values,'Trigger-only displacement')]:
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax in axes:
            ax.hist(values,bins=cfg['histogram_bins'],color='#4477AA',edgecolor='white',linewidth=.3)
            ax.axvline(tau,color='#CC3311',linestyle='--',label=f'tau_c={tau:g}')
            ax.set(xlabel='Normalized centroid displacement D_t',ylabel='Count')
            ax.legend()
        axes[1].set_yscale('log');axes[1].set_title('Log count (same full displacement range)')
        fig.suptitle(f'{title} (n={len(values)})')
        fig.savefig(out/f'{name}.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    groups=[[r['normalized_displacement'] for r in trigger_rows if r['video_id']==v] for v in videos]
    ax.boxplot(groups,tick_labels=[f'{v.removeprefix("VIRAT_S_")}\n(n={len(g)})' for v,g in zip(videos,groups)],showfliers=True)
    ax.axhline(tau,color='#CC3311',linestyle='--',label=f'tau_c={tau:g}')
    ax.tick_params(axis='x',rotation=35);ax.set(ylabel='Trigger displacement D_t',title='Trigger displacement by video (whiskers: 1.5 IQR)');ax.legend()
    fig.savefig(out/'video_trigger_boxplot.png',dpi=180);plt.close(fig)
    provenance={'inputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
                'threshold':tau,'tertile_boundaries':list(cuts),'validation':'Saved pair displacement/trigger checks, ROI coverage, original intervals, and cost monotonicity passed.',
                'global_summary':global_summary}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    lines=['# Centroid displacement diagnostics','',
           f'9 videos, {len(assignments)} ROIs, {len(rows)} valid pairs, {len(trigger_rows)} triggers; tau_c={tau}.','',
           '## Definitions and limits','',
           '- Centroid x/y fields retain the existing ROI-normalized coordinates (not pixels). D_t uses x/y scaled by ROI width/height, divided by its diagonal; no delta_t normalization.',
           '- Invalid centroids are skipped; pairs connect consecutive valid centroids, with their actual delta_t.',
           '- Trigger ranks are ascending among triggers only: average ranks for ties, percentile = 100 * count(D <= value) / N. Trigger gaps are within the same ROI; video/global interval statistics pool those ROI gaps.',
           '- Standard deviation is population std (ddof=0); percentiles use NumPy linear interpolation. Global distributions pool pairs, rather than weighting each video equally.',
           '- Tertiles use pooled trigger 33.33/66.67 percentiles; equal values stay together, so group sizes may differ.',
           '- Actual trigger_windows implementation is symmetric [t-pad, t+pad], clipped to ROI bounds. The old module header mentions an asymmetric window but is stale.',
           '- Nominal window sum ignores clipping; clipped window sum counts overlaps repeatedly. Union is performed within each ROI, then summed across ROIs (ROI-seconds).',
           '- Expected additional frames = union duration * (high_fps-base_fps). This continuous-rate estimate excludes endpoint, native-frame rounding, and schedule-boundary effects; it is not an exact sampler frame count. No frames are selected or generated.',
           '- Mean unique coverage per trigger = union duration / trigger count. Extra coverage versus minimum pad uses pad=0.5 as baseline; this is amortized coverage, not isolated causal contribution.',
           '- Displacement distributions and trigger spacing alone cannot establish required caption context or optimal padding; that requires aligned caption-quality/event evidence.','',
           '## Global trigger displacement','',
           '| p50 | p75 | p90 | p95 | p99 | max |','|---:|---:|---:|---:|---:|---:|',
           '| '+' | '.join(f'{global_summary["trigger_displacement_"+k]:.6f}' for k in ['p50','p75','p90','p95','p99','max'])+' |','',
           '## Fixed-padding cost (summed ROI-seconds)','',
           '| pad sec | clipped window sum sec | union 2fps sec | expected extra frames | unique sec / trigger |',
           '|---:|---:|---:|---:|---:|']
    for r in costs:
        if r['scope']=='global':
            lines.append(f'| {r["pad_sec"]:.1f} | {r["clipped_window_duration_sum_sec"]:.3f} | {r["union_2fps_duration_sec"]:.3f} | {r["expected_additional_frames"]:.2f} | {r["mean_unique_coverage_sec_per_trigger"]:.3f} |')
    lines.extend(['','## Interpretation','',
        '- The trigger distribution is right-skewed, with a substantial magnitude range; see the percentiles and threshold concentration below.',
        '- Threshold-near values are common, but fewer than half of triggers fall within 1.5 times the threshold. Large triggers above 5 times the threshold are a minority.',
        '- Low/medium/high pooled trigger tertiles do not imply increasing required context. Their next-trigger gaps describe spacing, not caption performance.',
        '- Magnitude variation motivates investigating a hypothesis, but does not justify a magnitude-to-padding rule. There are no aligned context requirements or caption-quality targets in this analysis.',
        '', '## Largest trigger and valid-centroid gap', ''])
    if trigger_rows:
        largest = max(trigger_rows, key=lambda r:r['normalized_displacement'])
        lines.append(f'- Largest: {largest["video_id"]}, ROI {largest["roi_id"]}, t={largest["timestamp"]:.6f}s, D={largest["normalized_displacement"]:.6f}, delta_t={largest["delta_t"]:.6f}s. A long gap between valid centroids can produce a large D; D is not velocity.')
    lines.extend(['', '## Trigger tertiles', '', '| group | trigger count | percent | min D | max D | median next-trigger gap sec |', '|---|---:|---:|---:|---:|---:|'])
    for b in bins:
        lines.append(f'| {b["group"]} | {b["count"]} | {b["percent_of_all_triggers"]:.2f} | {b["displacement_min"]:.6f} | {b["displacement_max"]:.6f} | {b["next_trigger_gap_sec_median"]:.6f} |')
    lines.extend(['','## Threshold concentration',''])
    for m in cfg['near_threshold_multipliers']:
        lines.append(f'- tau <= D <= {m:g} tau: {global_summary[f"triggers_tau_to_{m:g}tau_count"]} triggers ({global_summary[f"triggers_tau_to_{m:g}tau_percent"]:.2f}%).')
    lines.extend(['',f'- D > 5 tau: {global_summary["triggers_above_5tau_count"]} ({global_summary["triggers_above_5tau_percent"]:.2f}%).','',
                  '## Run','',f'```bash\n.vlm-inf/bin/python scripts/analyze_centroid_displacement.py --config {args.config}\n```'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(global_summary,indent=2))
    print(f'Analysis written to {out}')


if __name__=='__main__':
    main()
