"""Read-only 3s evidence grouping for text-only 30s event consolidation."""
import json
from .temporal_event_merge import validate_clip_document, _label


def windows_for(document):
    clips = validate_clip_document(document)
    groups = []
    for clip in clips:
        index = int(clip['start_sec'] // 30)
        if index == len(groups):
            groups.append(dict(start_sec=index * 30, end_sec=clip['end_sec'], source_segments=[]))
        group = groups[index]
        if clip['end_sec'] > (index + 1) * 30 + 1e-9:
            raise ValueError('A source clip crosses a 30-second boundary')
        group['end_sec'] = clip['end_sec']
        group['source_segments'].append(clip)
    return groups


def window_prompt(template, window):
    lines = '\n\n'.join(f"[{_label(c['start_sec'])}s–{_label(c['end_sec'])}s]\n{c['caption']}" for c in window['source_segments'])
    return template.format(window_start=_label(window['start_sec']), window_end=_label(window['end_sec']), timestamped_3s_captions=lines)


def review_markdown(document):
    parts = [f"# {document['video']}\n", 'Intermediate event candidates for human verification.\n']
    for w in document['windows']:
        parts.append(f"\n## [{_label(w['start_sec'])}–{_label(w['end_sec'])}s]\n\n{w['intermediate_caption']}\n\nSources: {', '.join(w['source_clips'])}\n")
        for c in w['source_segments']:
            parts.append(f"\n- [{c['source_clip']}] {c['caption']}\n")
    return ''.join(parts)


def audit_prompt(window, caption):
    return ('Verify a consolidated event description against its source observations. '
            'Approve only if all distinct events, short actions, state changes, interactions and temporal order are retained, '
            'repeated observations are consolidated, uncertainty/conflicts are preserved, and no unsupported facts or hidden transitions are introduced. '
            'Do not require verbatim wording or every repeated detail. Return JSON only with exactly these keys: '
            '{"approved": true, "missing_events": [], "unsupported_claims": [], "conflict_or_order_errors": []}.\n'
            + json.dumps(dict(source_segments=window['source_segments'], consolidated_description=caption), ensure_ascii=False))
