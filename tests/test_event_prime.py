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
from vlm_inference.event_prime import validate_events,mechanical_stats


def ev(start,end,description):return dict(start=start,end=end,description=description)

class PrimeTests(unittest.TestCase):
    def test_duplicate_report_overlap_zero_duration_retained(self):
        events=[ev(0,9,'A person walks.'),ev(3,6,'A van moves.'),ev(3,6,'A van moves.'),ev(9,9,'A door opens.')]
        obj,report=validate_events(json.dumps(dict(events=events)),{0,3,6,9},(0,9))
        self.assertEqual(obj['events'],events)
        self.assertEqual(report['exact_duplicate_pairs'],[dict(first_event_index=1,duplicate_event_index=2)])
        stats=mechanical_stats(events)
        self.assertEqual(stats['overlapping_event_pairs'],3)
        self.assertEqual(stats['events_spanning_gt_3s'],1)
        self.assertEqual(stats['events_spanning_gt_6s'],1)

    def test_invalid_boundaries_and_intervals(self):
        for e in [ev(1,3,'walk'),ev(6,3,'walk'),ev(0,12,'walk'),ev(False,3,'walk')]:
            with self.assertRaises(ValueError):validate_events(json.dumps(dict(events=[e])),{0,3,6,9},(0,9))
        with self.assertRaises(ValueError):validate_events('{"events": "wrong"}',{0,3},(0,3))
        # 3s boundaries are allowed within the local 30s window.
        self.assertTrue(validate_events(json.dumps(dict(events=[ev(3,6,'walk')])),{0,3,6,30},(0,30))[1]['valid'])

    def test_independent_input_local_schema_stats_and_resume(self):
        spec=importlib.util.spec_from_file_location('prime_cli',ROOT/'scripts/run_virat_event_ab_prime.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        calls=[];loads=[]
        class Fake:
            def __init__(self,cfg):loads.append(cfg)
            def generate_text(self,prompt):
                calls.append(prompt)
                if 'INPUT WINDOW EVENTS:' in prompt:events=[ev(0,35,'LOCAL_ONLY person walks.'),ev(3,6,'A van drives.')]
                elif '30-second interval' in prompt:
                    events=[ev(30,35,'LOCAL_ONLY person walks.')] if '[30s' in prompt else [ev(0,30,'LOCAL_ONLY person walks.'),ev(3,6,'A van drives.')]
                else:events=[ev(0,35,'A person walks.'),ev(3,6,'A van drives.')]
                return SimpleNamespace(text=json.dumps(dict(events=events)),hit_token_limit=False,num_output_tokens=40)
        with tempfile.TemporaryDirectory() as tmp:
            source_dir=Path(tmp)/'source';source_dir.mkdir();source=source_dir/'source.json';out=Path(tmp)/'out'
            raw=json.dumps(dict(video=cli.VIDEO,clip_duration_sec=3,clips=[dict(start_sec=s,end_sec=min(s+3,35),caption='ORIGINAL_ONLY A person walks.') for s in range(0,35,3)])).encode();source.write_bytes(raw)
            argv=['run','--input',str(source),'--output-dir',str(out)]
            with patch.dict(sys.modules,{'vlm_inference.model':SimpleNamespace(Qwen3VLCaptioner=Fake)}),patch.object(sys,'argv',argv):
                self.assertEqual(cli.main(),0)
            self.assertEqual(len(loads),1);self.assertEqual(len(calls),4)
            self.assertIn('LOCAL_ONLY',calls[2]);self.assertNotIn('ORIGINAL_ONLY',calls[2])
            self.assertIn('ORIGINAL_ONLY',calls[3]);self.assertNotIn('LOCAL_ONLY',calls[3])
            for f in (out/'A_hierarchical/local_30s').rglob('*.json'):
                if '.metadata.' not in f.name:self.assertEqual(set(json.loads(f.read_text())),{'events'})
            stat=json.loads((out/'comparison'/(cli.VIDEO+'.statistics.json')).read_text())
            self.assertEqual(stat['A_local']['event_count'],3);self.assertEqual(stat['A_final']['event_count'],2)
            self.assertEqual(stat['A_final']['overlapping_event_pairs'],1)
            with patch.dict(sys.modules,{'vlm_inference.model':SimpleNamespace(Qwen3VLCaptioner=Fake)}),patch.object(sys,'argv',argv+['--resume']):
                self.assertEqual(cli.main(),0)
            self.assertEqual(len(calls),4);self.assertEqual(source.read_bytes(),raw)

if __name__=='__main__':unittest.main()
