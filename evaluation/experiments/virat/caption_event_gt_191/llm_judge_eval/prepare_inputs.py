#!/usr/bin/env python3
"""Parse structural fields only. No lexical/regex event evaluation."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/ssandree/Motion-guided-ROI/vlm-inference/outputs/fullframe_proposed_roi_5model/descriptions.txt')
VIDEOS=['VIRAT_S_'+s for s in ['000200_00_000100_000171','010001_03_000537_000563','010003_03_000219_000259','010003_06_000526_000560','010003_10_000901_000934','010106_02_000656_000692','010114_01_000235_000311','010207_04_000929_000954','040005_08_001225_001276']]
MODELS=['8b','8b-int8','8b-fp8','30b','30b-fp8']
def save(name,data): (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def main():
    records=[]; current=None; model=None
    for line in SOURCE.read_text().splitlines():
        if line.startswith('# '): model=line[2:].strip()
        elif line.startswith('video: '):
            current={'model':model,'video_id':line[7:].strip().removesuffix('.mp4')}; records.append(current)
        elif line.startswith('condition: '): current['condition']=line[11:].strip()
        elif line.startswith('description: '): current['description']=line[13:].strip()
        elif line.strip() and current is not None and 'description' in current: current['description']+='\n'+line
    selected=[r for r in records if r['video_id'] in VIDEOS]
    assert len(selected)==90
    assert {(r['model'],r['video_id'],r['condition']) for r in selected}=={(m,v,c) for m in MODELS for v in VIDEOS for c in ['full_frame','motion_roi']}
    for i,r in enumerate(selected): r['record_id']=f'R{i:03d}'
    gt={v:json.loads((ROOT.parent/'gt'/v/'caption_events.json').read_text()) for v in VIDEOS}
    sources=[SOURCE, ROOT.parent/'annotation_policy.json']+[ROOT.parent/'gt'/v/'caption_events.json' for v in VIDEOS]
    save('inputs.json',{'records':selected,'ground_truth':gt})
    save('input_manifest.json',{'source_files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources], 'selected_records':len(selected),'excluded_records':len(records)-len(selected)})
    print('Selected',len(selected),'records; GT events:',sum(len(d['caption_events']) for d in gt.values()))
    for v,d in gt.items():
        actions=sorted({e['action'] for e in d['caption_events']}); dirs=sorted({(e['action'],e['direction']) for e in d['caption_events'] if e['direction']})
        print(v,len(d['caption_events']),actions,dirs)
if __name__=='__main__': main()
