#!/usr/bin/env python3
"""Verify the box screening against existing inputs without no-box inference."""
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_inference.config import load_config
from vlm_inference.feasibility import collect_method_inputs
from vlm_inference.metadata_format import format_video_bundle


def main():
    base = load_config(ROOT / 'configs/feasibility_global_roi_cross.yaml')
    box = load_config(ROOT / 'configs/feasibility_global_roi_start_box.yaml')
    assert base.feasibility.videos == box.feasibility.videos
    for key in ('model', 'generation', 'video', 'image'):
        assert getattr(base, key) == getattr(box, key)
    assert base.experiment.centroid_motion == box.experiment.centroid_motion
    records = []
    for video in box.feasibility.videos:
        for mode in ('centroid_2fps', 'fixed_2fps'):
            old = collect_method_inputs(video, f'global_roi_start_only_roi_{mode}', base)
            new = collect_method_inputs(video, f'global_roi_start_box_roi_{mode}', box)
            assert not old.errors and not new.errors, (old.errors, new.errors)
            a = format_video_bundle(old.bundle, input_order='global_anchor').items
            b = format_video_bundle(new.bundle, input_order='global_anchor').items
            assert [i.text for i in a] == [i.text for i in b]
            assert [(f.timestamp, f.frame_index) for f in old.bundle.global_section.frames] == [(f.timestamp, f.frame_index) for f in new.bundle.global_section.frames]
            for r0, r1 in zip(old.bundle.rois, new.bundle.rois, strict=True):
                assert r0.time_range == r1.time_range and r0.normalized_bbox == r1.normalized_bbox
                for f0, f1 in zip(r0.frames, r1.frames, strict=True):
                    assert (f0.timestamp, f0.frame_index) == (f1.timestamp, f1.frame_index)
                    assert hashlib.sha256(f0.path.read_bytes()).digest() == hashlib.sha256(f1.path.read_bytes()).digest()
            records.append({'video':video,'method':new.method,'roi_frames':new.num_roi_images,'global_frames':new.num_global_images})
    assert len(records) == 18
    out = ROOT / 'outputs/global_roi_start_box'
    previews = []
    for manifest in sorted((out / 'global_boxes').glob('*/boxes.json')):
        entries = json.loads(manifest.read_text())
        for entry in entries:
            original = np.array(Image.open(entry['source']).convert('RGB'))
            painted = np.array(Image.open(entry['image']).convert('RGB'))
            h, w = original.shape[:2]
            mask = Image.new('1', (w,h))
            draw = ImageDraw.Draw(mask)
            for bbox in entry['boxes']:
                draw.rectangle(bbox['bbox_px'], outline=1, width=max(4, round(min(w,h)/180))+4)
            changed = np.any(original != painted, axis=2)
            assert changed.any() and not changed[~np.array(mask, dtype=bool)].any()
            thumb = Image.open(entry['image']).convert('RGB')
            thumb.thumbnail((480,270))
            tile = Image.new('RGB',(480,300),'white')
            tile.paste(thumb,(0,30))
            ImageDraw.Draw(tile).text((4,4),f'{manifest.parent.name}\nt={entry["timestamp"]} ROI={[b["roi_id"] for b in entry["boxes"]]}',fill='black')
            previews.append(tile)
    sheet = Image.new('RGB',(480*3,300*((len(previews)+2)//3)),'white')
    for i, tile in enumerate(previews):
        sheet.paste(tile,((i%3)*480,(i//3)*300))
    sheet.save(out / 'box_contact_sheet.jpg')
    (out / 'input_verification.json').write_text(json.dumps({'jobs':records,'job_count':len(records),'global_frames_checked':len(previews),'roi_bytes_timestamps_labels_match':True,'changes_confined_to_box_strokes':True},indent=2)+'\n')
    print(f'PASS: {len(records)} jobs; ROI bytes, timestamps, labels match; {len(previews)} Global images checked')

if __name__ == '__main__':
    main()
