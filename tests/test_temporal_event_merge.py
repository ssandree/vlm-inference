from pathlib import Path
import sys
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_inference.temporal_event_merge import validate_clip_document, validate_merged_events, comparison_markdown, build_prompt


class TemporalEventMergeTests(unittest.TestCase):
    def setUp(self):
        self.clips = validate_clip_document(dict(video='test.mp4', clip_duration_sec=3, clips=[
            dict(start_sec=0, end_sec=3, caption='Two people walk. A person sits.'),
            dict(start_sec=3, end_sec=6, caption='Two people continue walking. A person sits.'),
            dict(start_sec=6, end_sec=7.125, caption='A person sits.'),
        ]))

    def event(self, start=0, end=6, sources=None):
        return dict(start_sec=start, end_sec=end, sentence='Two people walk.',
                    source_clips=sources if sources is not None else ['0-3', '3-6'])

    def test_prompt_preserves_each_original_sentence_and_source_boundary(self):
        prompt=build_prompt('Merge continuously.',self.clips)
        body=prompt.split('Input clip caption sequence (the only event evidence):\n',1)[1]
        observations,_=json.JSONDecoder().raw_decode(body)
        self.assertEqual(len(observations),5)
        self.assertEqual([r['caption'] for r in observations],['Two people walk.','A person sits.','Two people continue walking.','A person sits.','A person sits.'])
        self.assertEqual([r['source_clip'] for r in observations],['0-3','0-3','3-6','3-6','6-7.125'])
        self.assertEqual(observations[-1]['end_sec'],7.125)

    def test_concurrent_events_share_clip_provenance(self):
        result = validate_merged_events(json.dumps([
            self.event(), self.event(end=7.125, sources=['0-3','3-6','6-7.125'])]), self.clips)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(result['events']), 2)
        review = comparison_markdown('test.mp4', self.clips, result['events'])
        self.assertIn('| 0-3 | E1, E2 |', review)
        self.assertIn('| 6-7.125 | E2 |', review)

    def test_unknown_skipped_duplicate_and_reversed_sources_rejected(self):
        for sources in [['0-3','9-12'], ['0-3','6-7.125'], ['0-3','0-3'], ['3-6','0-3'], []]:
            with self.subTest(sources=sources):
                result = validate_merged_events(json.dumps([self.event(sources=sources)]), self.clips)
                self.assertEqual(result['status'], 'invalid_events')
                self.assertEqual(result['events'], [])
                self.assertEqual(result['rejected_events'][0]['original']['source_clips'], sources)

    def test_model_cannot_refine_interval_or_round_tail(self):
        for event in [self.event(start=0.1), self.event(end=5.9),
                      self.event(start=6, end=7.12, sources=['6-7.125']), self.event(start=True)]:
            self.assertEqual(validate_merged_events(json.dumps([event]), self.clips)['status'], 'invalid_events')

    def test_input_missing_clip_or_short_middle_rejected(self):
        for clips in [[dict(start_sec=3,end_sec=6,caption='Walk.')],
                      [dict(start_sec=0,end_sec=2,caption='Walk.'),dict(start_sec=2,end_sec=5,caption='Walk.')]]:
            with self.assertRaises(ValueError):
                validate_clip_document(dict(video='test.mp4',clip_duration_sec=3,clips=clips))

    def test_strict_json_schema_and_empty_events(self):
        self.assertEqual(validate_merged_events('[]',self.clips)['status'],'ok')
        self.assertEqual(validate_merged_events('```json\n[]\n```',self.clips)['status'],'ok')
        self.assertEqual(validate_merged_events('Explanation: []',self.clips)['status'],'invalid_json')
        self.assertEqual(validate_merged_events('{}',self.clips)['status'],'invalid_schema')
        event = self.event();event['subject']='person'
        self.assertEqual(validate_merged_events(json.dumps([event]),self.clips)['status'],'invalid_events')


class TextOnlyConfigTests(unittest.TestCase):
    def test_empty_jobs_requires_explicit_loader_opt_in(self):
        from vlm_inference.config import load_config
        config = Path(__file__).resolve().parents[1] / 'configs/virat_temporal_event_merge.yaml'
        with self.assertRaises(ValueError):
            load_config(config)
        cfg = load_config(config, allow_empty_jobs=True)
        self.assertEqual(cfg.jobs, [])
        self.assertEqual(cfg.model.name, '30b')


if __name__ == '__main__':
    unittest.main()
