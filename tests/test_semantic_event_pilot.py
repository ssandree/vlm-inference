import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('semantic_pilot', ROOT / 'scripts/rerun_virat_event_semantic_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class SemanticPilotStatsTests(unittest.TestCase):
    def test_overlap_pairs_exclude_touching_endpoints_and_static_context(self):
        events = [
            dict(start=0, end=6, description='A van drives while parked vehicles remain stationary.'),
            dict(start=3, end=12, description='A person walks.'),
            dict(start=6, end=9, description='No discernible actions occur.'),
        ]
        self.assertEqual(pilot.event_stats(events), {
            'number of events': 3,
            'number of events spanning >3 seconds': 2,
            'number of events spanning >6 seconds': 1,
            'number of overlapping event time ranges': 2,
            'number of static/no-action events': 1,
        })

    def test_explicit_absence_of_people_and_moving_vehicles_is_static(self):
        self.assertEqual(pilot.event_stats([dict(start=36, end=39, description='No people or moving vehicles are visible. The scene is entirely static.')])['number of static/no-action events'], 1)

    def test_report_required_order_and_common_prompt(self):
        groups = {title: {'events': []} for title in
                  ['Existing A events', 'New A events', 'Existing B events', 'New B events']}
        report, _ = pilot.comparison([dict(start_sec=0, end_sec=3, caption='A person walks.')], groups)
        titles = ['Original 3s captions', *groups, 'Mechanical statistics']
        positions = [report.index('## ' + title) for title in titles]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual((ROOT / 'prompts/virat_ab_a2_events.txt').read_bytes(),
                         (ROOT / 'prompts/virat_ab_b_direct.txt').read_bytes())


if __name__ == '__main__':
    unittest.main()
