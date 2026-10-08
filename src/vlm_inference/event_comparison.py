"""Strict shared event schema, boundary validation and raw A/B comparison."""
import json
import math
import re
from .temporal_event_merge import _label


def insert(template, **values):
    # JSON examples use literal braces: substitute ONLY named input placeholders.
    for key, value in values.items():
        template = template.replace('{'+key+'}', str(value))
    return template


def timestamped(rows, start='start_sec', end='end_sec', text='caption'):
    return '\n\n'.join(f"[{_label(r[start])}s–{_label(r[end])}s]\n{r[text]}" for r in rows)


def parse_events(raw, boundaries):
    value=raw.strip()
    match=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', value)
    if match:value=match.group(1)
    obj=json.loads(value)
    if not isinstance(obj,dict) or set(obj)!={'events'} or not isinstance(obj['events'],list):
        raise ValueError('Expected exactly {"events": [...]}')
    previous=-1
    seen=set()
    for index,event in enumerate(obj['events']):
        if not isinstance(event,dict) or set(event)!={'start','end','description'}:
            raise ValueError(f'Event {index}: expected start/end/description only')
        for key in ['start','end']:
            v=event[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
                raise ValueError(f'Event {index}: invalid {key}')
            # Do not silently repair or quantize model-proposed timestamps.
            if not any(math.isclose(v,b,rel_tol=0,abs_tol=1e-9) for b in boundaries):
                raise ValueError(f'Event {index}: {key}={v} is not an available input boundary')
        if event['start']<0 or event['end']<=event['start']:
            raise ValueError(f'Event {index}: invalid time interval')
        if event['start']<previous:raise ValueError('Events must be chronological by start')
        previous=event['start']
        if not isinstance(event['description'],str) or not event['description'].strip():
            raise ValueError(f'Event {index}: empty description')
        key=(event['start'],event['end'],event['description'].strip())
        if key in seen:raise ValueError(f'Event {index}: exact duplicate')
        seen.add(key)
    return obj


def intermediate_boundaries(windows, source_boundaries):
    """Use window endpoints plus explicit times retained in the A1 text only."""
    found={b for w in windows for b in [w['window_start'],w['window_end']]}
    for w in windows:
        text=w['description']
        for m in re.finditer(r"\b(\d+):(\d{2})(?::(\d{2}))?(?:\.(\d+))?\b",text):
            first,second,third,fraction=m.groups()
            value=int(first)*60+int(second) if third is None else int(first)*3600+int(second)*60+int(third)
            if fraction:value+=float('0.'+fraction)
            if w['window_start']<=value<=w['window_end']:found.add(value)
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",text):
            value=float(m.group(1))
            if w['window_start']<=value<=w['window_end']:found.add(value)
    return {b for b in source_boundaries if any(math.isclose(b,v,rel_tol=0,abs_tol=1e-9) for v in found)}


def comparison_md(video, windows, a, b):
    lines=[f'# {video}\n','Raw output comparison for human inspection. Generated candidates, not verified GT.\n',
           '\n## 30-sec intermediate descriptions\n']
    for w in windows:
        lines.append(f"\n[{_label(w['window_start'])}–{_label(w['window_end'])}s]\n\n{w['description']}\n")
    for title,events in [('Hierarchical events: 3s → 30s → Events',a),('Direct events: 3s → Events',b)]:
        lines.append('\n## '+title+'\n')
        for i,event in enumerate(events['events'],1):
            lines.append(f"\nE{i} [{_label(event['start'])}–{_label(event['end'])}s] {event['description']}\n")
    return ''.join(lines)
