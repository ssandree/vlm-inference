import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_inference.intermediate_caption import windows_for,window_prompt


def document(video='test.mp4',end=65):
    return dict(video=video,clip_duration_sec=3,clips=[dict(start_sec=s,end_sec=min(s+3,end),caption=f'Observation at {s}. A person opens a door.' if s==27 else f'Observation at {s}.') for s in range(0,end,3)])


class ConsolidationTests(unittest.TestCase):
    def test_boundaries_tail_and_verbatim_evidence(self):
        doc=document();windows=windows_for(doc)
        self.assertEqual([(w['start_sec'],w['end_sec']) for w in windows],[(0,30),(30,60),(60,65)])
        self.assertEqual([len(w['source_segments']) for w in windows],[10,10,2])
        template=(ROOT/'prompts/virat_intermediate_30s.txt').read_text()
        for w in windows:
            prompt=window_prompt(template,w)
            for c in w['source_segments']:self.assertIn(c['caption'],prompt)
        self.assertEqual(sum(len(w['source_segments']) for w in windows),len(doc['clips']))
        self.assertIn('A person opens a door.',window_prompt(template,windows[0]))

    def test_text_only_full_run_pilot_barrier_resume_and_immutable_sources(self):
        spec=importlib.util.spec_from_file_location('intermediate_cli',ROOT/'scripts/run_virat_intermediate_30s.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        names=[Path(s.strip()).name for s in (ROOT/'configs/virat_multi_roi_59.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
        calls=[];loads=[]
        class FakeModel:
            def __init__(self,cfg):loads.append(cfg)
            def generate_text(self,prompt):
                calls.append(prompt)
                text=json.dumps(dict(approved=True,missing_events=[],unsupported_claims=[],conflict_or_order_errors=[])) if prompt.startswith('Verify') else 'A person opens a door.'
                return SimpleNamespace(text=text,hit_token_limit=False,num_output_tokens=9)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source';source.mkdir();out=base/'out'
            for name in names:(source/(name+'.json')).write_text(json.dumps(document(name,end=9)))
            initial={p.name:p.read_bytes() for p in source.iterdir()}
            argv=['run','--input-dir',str(source),'--output-dir',str(out)]
            with patch.object(sys,'argv',argv),patch.dict(sys.modules,{'vlm_inference.model':SimpleNamespace(Qwen3VLCaptioner=FakeModel)}):
                self.assertEqual(cli.main(),0)
                self.assertEqual(len(loads),1);self.assertEqual(len(calls),60)
                self.assertTrue(calls[1].startswith('Verify'))
                self.assertEqual(len(list((out/'captions').glob('*.json'))),59)
                self.assertEqual(cli.main(),0);self.assertEqual(len(calls),60)
            self.assertEqual(initial,{p.name:p.read_bytes() for p in source.iterdir()})
            exported=json.loads((out/'captions'/(names[0]+'.json')).read_text())
            self.assertEqual(exported['windows'][0]['source_clips'],['0-3','3-6','6-9'])
            self.assertEqual(len(exported['windows'][0]['source_segments']),3)

    def test_failed_pilot_audit_stops_before_full_run(self):
        spec=importlib.util.spec_from_file_location('intermediate_fail_cli',ROOT/'scripts/run_virat_intermediate_30s.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        names=[Path(s.strip()).name for s in (ROOT/'configs/virat_multi_roi_59.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
        calls=[]
        class RejectingModel:
            def __init__(self,cfg):pass
            def generate_text(self,prompt):
                calls.append(prompt)
                text=json.dumps(dict(approved=False,missing_events=['door action'],unsupported_claims=[],conflict_or_order_errors=[])) if prompt.startswith('Verify') else 'A person walks.'
                return SimpleNamespace(text=text,hit_token_limit=False,num_output_tokens=9)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();out=Path(tmp)/'out'
            for name in names:(source/(name+'.json')).write_text(json.dumps(document(name,end=9)))
            with patch.object(sys,'argv',['run','--input-dir',str(source),'--output-dir',str(out)]),patch.dict(sys.modules,{'vlm_inference.model':SimpleNamespace(Qwen3VLCaptioner=RejectingModel)}):
                self.assertEqual(cli.main(),1)
            self.assertEqual(len(calls),2)
            self.assertEqual(json.loads((out/'progress.json').read_text())['stage'],'pilot_failed')
            self.assertFalse((out/'captions').exists())

if __name__=='__main__':unittest.main()
