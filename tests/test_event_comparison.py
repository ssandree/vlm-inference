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
from vlm_inference.event_comparison import insert,parse_events,intermediate_boundaries

class ComparisonTests(unittest.TestCase):
    def test_literal_schema_and_strict_timestamps(self):
        prompt=insert('{timestamped_3s_captions}\n{"events": []}',timestamped_3s_captions='evidence')
        self.assertEqual(prompt,'evidence\n{"events": []}')
        valid=json.dumps({'events':[dict(start=0,end=30,description='A person walks.'),dict(start=0,end=30,description='Another person sits.')]})
        self.assertEqual(len(parse_events(valid,{0,30})['events']),2)
        for obj in [{'events':[dict(start=1,end=30,description='walk')]},{'events':[dict(start=0,end=0,description='walk')]},
                    {'events':[dict(start=0,end=30,description='walk')]*2},{'events':[dict(start=True,end=30,description='walk')]}]:
            with self.assertRaises(ValueError):parse_events(json.dumps(obj),{0,30})

    def test_A_uses_only_explicitly_retained_temporal_evidence(self):
        rows=[dict(window_start=0,window_end=30,description='From 0:03 to 0:06 a person walks. At 12 seconds a door opens.'),dict(window_start=30,window_end=35,description='A person sits.')]
        self.assertEqual(intermediate_boundaries(rows,{0,3,6,9,12,15,18,21,24,27,30,33,35}),{0,3,6,12,30,35})

    def test_independent_branches_same_schema_source_unchanged_and_no_judge(self):
        spec=importlib.util.spec_from_file_location('ab_cli',ROOT/'scripts/run_virat_event_ab.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        calls=[];loads=[]
        class Fake:
            def __init__(self,cfg):loads.append(cfg)
            def generate_text(self,prompt):
                calls.append(prompt)
                text='UNIQUE_INTERMEDIATE: A person walks.' if 'Event-preserving description:' in prompt else json.dumps({'events':[dict(start=0,end=30,description='A person walks.')]})
                return SimpleNamespace(text=text,hit_token_limit=False,num_output_tokens=30)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();out=Path(tmp)/'both'
            path=source/(cli.DEFAULT_VIDEO+'.json')
            raw=json.dumps(dict(video=cli.DEFAULT_VIDEO,clip_duration_sec=3,clips=[dict(start_sec=s,end_sec=min(s+3,35),caption='A person walks.') for s in range(0,35,3)])).encode();path.write_bytes(raw)
            with patch.dict(sys.modules,{'vlm_inference.model':SimpleNamespace(Qwen3VLCaptioner=Fake)}),patch.object(sys,'argv',['run','--input-dir',str(source),'--output-dir',str(out)]):
                self.assertEqual(cli.main(),0)
            self.assertEqual(len(loads),1);self.assertEqual(len(calls),4)
            self.assertIn('UNIQUE_INTERMEDIATE',calls[2]);self.assertNotIn('UNIQUE_INTERMEDIATE',calls[3]);self.assertIn('[33s–35s]',calls[3])
            a=json.loads((out/'A_hierarchical/events'/(cli.DEFAULT_VIDEO+'.json')).read_text());b=json.loads((out/'B_direct/events'/(cli.DEFAULT_VIDEO+'.json')).read_text());self.assertEqual(a,b)
            self.assertEqual(path.read_bytes(),raw)
            with patch.dict(sys.modules,{'vlm_inference.model':SimpleNamespace(Qwen3VLCaptioner=Fake)}),patch.object(sys,'argv',['run','--pipeline','B','--input-dir',str(source),'--output-dir',str(Path(tmp)/'B')]):
                self.assertEqual(cli.main(),0)
            self.assertEqual(len(calls),5);self.assertFalse((Path(tmp)/'B/A_hierarchical').exists())

if __name__=='__main__':unittest.main()
