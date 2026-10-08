"""Prime pilot validation and mechanical statistics; never semantically repair events."""
import json
import math
import re
from .event_comparison import timestamped


def validate_events(raw, boundaries, interval):
    value=raw.strip()
    match=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',value)
    if match:value=match.group(1)
    obj=json.loads(value)
    if not isinstance(obj,dict) or set(obj)!={'events'} or not isinstance(obj['events'],list):
        raise ValueError('Expected exactly {"events": [...]}')
    lower,upper=interval;seen={};duplicates=[];out_of_order=[];previous=-1
    for i,event in enumerate(obj['events']):
        if not isinstance(event,dict) or set(event)!={'start','end','description'}:raise ValueError(f'Event {i}: invalid schema')
        for key in ['start','end']:
            v=event[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError(f'Event {i}: invalid {key}')
            if not lower-1e-9<=v<=upper+1e-9:raise ValueError(f'Event {i}: timestamp outside [{lower},{upper}]')
            if not any(math.isclose(v,b,rel_tol=0,abs_tol=1e-9) for b in boundaries):raise ValueError(f'Event {i}: {key}={v} is not an original observation boundary')
        if event['start']>event['end']:raise ValueError(f'Event {i}: start > end')
        if not isinstance(event['description'],str) or not event['description'].strip():raise ValueError(f'Event {i}: empty description')
        signature=(event['start'],event['end'],event['description'].strip())
        if signature in seen:duplicates.append(dict(first_event_index=seen[signature],duplicate_event_index=i))
        else:seen[signature]=i
        if event['start']<previous:out_of_order.append(i)
        previous=event['start']
    # Reports do not sort, remove duplicates, rewrite text or adjust times.
    return obj,dict(valid=True,exact_duplicate_pairs=duplicates,chronological_order_warning_indices=out_of_order,
                    overlapping_events_allowed=True,zero_duration_events_allowed=True)


STATIC_RE=re.compile(r'\bno\s+(?:(?:discernible|significant|noticeable|detectable|observable|identifiable)\s+)?(?:actions?|activity|motion|movement)\b|\b(?:scene|view|parking lot)\b.{0,40}\b(?:static|still|unchanged)\b|^\s*(?:parked (?:vehicles|cars)|vehicles remain stationary)',re.I)


def mechanical_stats(events):
    pairs=[]
    for i,a in enumerate(events):
        for j in range(i+1,len(events)):
            b=events[j]
            if min(a['end'],b['end'])-max(a['start'],b['start'])>1e-9:pairs.append([i,j])
    static=[i for i,e in enumerate(events) if STATIC_RE.search(e['description'])]
    return dict(event_count=len(events),events_spanning_gt_3s=sum(e['end']-e['start']>3+1e-9 for e in events),
                events_spanning_gt_6s=sum(e['end']-e['start']>6+1e-9 for e in events),
                overlapping_event_pairs=len(pairs),overlapping_event_pair_indices=pairs,
                static_no_action_events=len(static),static_no_action_event_indices=static,
                static_count_method='Conservative lexical heuristic; inspection required. No LLM judge.')


def comparison_markdown(video,clips,locals_,a,b,statistics):
    text=f'# {video}\n\nRaw A′/B′ pilot outputs for human inspection; no automatic semantic winner.\n'
    text+='\n## 1. Original 3s captions\n\n'+timestamped(clips)+'\n'
    text+='\n## 2. A′ 30s local Event JSON\n'
    for w in locals_:
        text+=f"\n### [{w['window_start']}–{w['window_end']}s]\n\n```json\n"+json.dumps(w['event_json'],ensure_ascii=False,indent=2)+'\n```\n'
    for title,obj in [('3. A′ final Event JSON',a),('4. B′ final Event JSON',b)]:
        text+='\n## '+title+'\n\n```json\n'+json.dumps(obj,ensure_ascii=False,indent=2)+'\n```\n'
    text+='\n## Mechanical statistics\n\n'
    text+=f"Source 3s segment count: {statistics['source_3s_segment_count']}\n\n"
    text+='| Metric | A′ local (all windows) | A′ final | B′ final |\n|---|---:|---:|---:|\n'
    for key,label in [('event_count','Event count'),('events_spanning_gt_3s','Events spanning >3s'),('events_spanning_gt_6s','Events spanning >6s'),('overlapping_event_pairs','Overlapping event pairs'),('static_no_action_events','Static/no-action candidates (lexical heuristic)')]:
        text+=f"| {label} | {statistics['A_local'][key]} | {statistics['A_final'][key]} | {statistics['B_final'][key]} |\n"
    text+='\nOverlap uses positive interval intersection; touching endpoints do not count. Static/no-action counts are conservative text-pattern flags, not semantic judgments. Exact duplicate candidates are reported in validation JSON and retained in outputs.\n'
    return text
