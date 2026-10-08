"""Text-only event merging input, provenance checks and human review export."""
from __future__ import annotations

import json
import math
import re


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _label(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def validate_clip_document(document: dict) -> list[dict]:
    if not isinstance(document, dict) or not isinstance(document.get('video'), str) or not document['video']:
        raise ValueError('Input must contain a nonempty video name')
    if document.get('clip_duration_sec') != 3:
        raise ValueError('Expected a 3-second clip caption pilot')
    clips = document.get('clips')
    if not isinstance(clips, list) or not clips:
        raise ValueError('Input clips must be a nonempty list')
    result = []
    previous_end = 0
    for i, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise ValueError(f'Clip {i} must be an object')
        start, end, caption = (clip.get(k) for k in ('start_sec', 'end_sec', 'caption'))
        if not (_number(start) and _number(end) and start >= 0 and end > start):
            raise ValueError(f'Invalid interval in clip {i}')
        if not math.isclose(start, previous_end, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'Clip {i} is not consecutive or does not start at zero')
        duration = end - start
        if duration > 3 + 1e-9 or (i < len(clips)-1 and not math.isclose(duration, 3, rel_tol=0, abs_tol=1e-9)):
            raise ValueError('Only the final clip may be shorter than 3 seconds')
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f'Clip {i} has no usable caption')
        result.append(dict(source_clip=f'{_label(start)}-{_label(end)}',
                           start_sec=start, end_sec=end, caption=caption))
        previous_end = end
    return result


def build_prompt(template: str, clips: list[dict]) -> str:
    # Separate sentence observations without classifying actions or merging them.
    # Every sentence remains verbatim and retains its original clip boundaries.
    observations = []
    for clip in clips:
        sentences = re.split(r"(?<=[.!?])\s+", clip['caption'].strip())
        for index, sentence in enumerate(sentences, 1):
            observations.append(dict(source_clip=clip['source_clip'], start_sec=clip['start_sec'],
                                     end_sec=clip['end_sec'], observation_in_clip=index, caption=sentence))
    return (template.rstrip()
            + '\n\nEach input record is one sentence observation from an original clip. Several records can share the same source_clip. Follow each activity across adjacent clip intervals even when another activity is interleaved in the input list.\n'
            + 'Input clip caption sequence (the only event evidence):\n'
            + json.dumps(observations, ensure_ascii=False, indent=2, allow_nan=False)
            + '\n\nEach JSON object must describe ONE activity flow. Sitting and walking are different activities and require separate event objects, even when observed in the same clips. Split independent activities first, then aggressively merge each continuing activity across adjacent clips. A missing observation does not imply a person appeared or disappeared. Return a JSON array with as many event objects as needed, not one summary of the whole scene. For example, a continuously seated subject and a continuously walking group produce TWO overlapping event objects, each with its own supporting source_clips. Return JSON only.')


def validate_merged_events(raw: str, clips: list[dict]) -> dict:
    """Validate exact provenance/boundaries, not semantic truth or subject identity."""
    try:
        text = raw.strip()
        # Accept a single Markdown JSON fence as transport formatting only;
        # never extract from prose or repair event content/timestamps.
        fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence:
            text = fence.group(1)
        events = json.loads(text)
    except (ValueError, TypeError) as exc:
        return dict(status='invalid_json', events=[], rejected_events=[], errors=[str(exc)])
    if not isinstance(events, list):
        return dict(status='invalid_schema', events=[], rejected_events=[], errors=['Expected JSON array'])
    positions = {c['source_clip']: i for i, c in enumerate(clips)}
    accepted, rejected = [], []
    for index, event in enumerate(events):
        errors = []
        if not isinstance(event, dict):
            rejected.append(dict(index=index, original=event, errors=['Event must be an object']))
            continue
        if set(event) != {'start_sec', 'end_sec', 'sentence', 'source_clips'}:
            errors.append('Each event requires exactly start_sec, end_sec, sentence, source_clips')
        if not isinstance(event.get('sentence'), str) or not event['sentence'].strip():
            errors.append('sentence must be nonempty text')
        sources = event.get('source_clips')
        valid_sources = isinstance(sources, list) and bool(sources) and all(isinstance(s, str) and s in positions for s in sources)
        if not valid_sources:
            errors.append('source_clips must be a nonempty list of exact input identifiers')
        else:
            indices = [positions[s] for s in sources]
            if indices != list(range(indices[0], indices[-1] + 1)):
                errors.append('source_clips must be consecutive, chronological and unique')
            expected_start, expected_end = clips[indices[0]]['start_sec'], clips[indices[-1]]['end_sec']
            for field, expected in [('start_sec', expected_start), ('end_sec', expected_end)]:
                if not _number(event.get(field)) or not math.isclose(event[field], expected, rel_tol=0, abs_tol=1e-9):
                    errors.append(f'{field} must equal the supporting clip boundary {expected}')
        if errors:
            rejected.append(dict(index=index, original=event, errors=errors))
        else:
            # Preserve segmentation boundaries exactly; only float representation
            # differences within 1e-9 are accepted, never finer event timestamps.
            accepted.append(dict(event, start_sec=expected_start, end_sec=expected_end))
    accepted.sort(key=lambda e: (e['start_sec'], e['end_sec']))
    return dict(status='invalid_events' if rejected else 'ok', events=accepted,
                rejected_events=rejected, errors=[])


def comparison_markdown(video: str, clips: list[dict], events: list[dict]) -> str:
    lines = [f'# Temporal event candidates: {video}', '',
             'Text-only Qwen merge of existing clip captions. Human-review candidates, not GT.', '',
             '## Original clip captions', '']
    for clip in clips:
        lines += [f"### {clip['source_clip']} sec", '', clip['caption'], '']
    lines += ['## Qwen merged events and supporting clips', '']
    for i, event in enumerate(events, 1):
        lines += [f"### E{i}: {_label(event['start_sec'])}–{_label(event['end_sec'])} sec", '',
                  event['sentence'], '', 'Source clips: ' + ', '.join(f'`{s}`' for s in event['source_clips']), '']
    lines += ['## Clip-to-event mapping', '', '| Input clip (sec) | Event candidates |', '|---|---|']
    for clip in clips:
        ids = [f'E{i}' for i, event in enumerate(events, 1) if clip['source_clip'] in event['source_clips']]
        lines.append(f"| {clip['source_clip']} | {', '.join(ids) or '(no output event)'} |")
    return '\n'.join(lines) + '\n'
