"""Guard safety denominators and label-free paired hard-state probing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hard_case_evaluation import evaluate_hard_cases, native_summary


class HardCaseEvaluationTests(unittest.TestCase):
    def test_safe_alternatives_and_all_fatal_endpoints_stay_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / 'trace.jsonl'
            rows = []
            for risks in ({'up': {'life_lost': True}, 'left': {'life_lost': False}},
                          {'up': {'life_lost': True}, 'left': {'life_lost': True}}):
                rows.append({'request': {'state': {}}, 'diagnostics': {'choice': 'up', 'immediate_counterfactuals': risks,
                             'action_frames': 10, 'pellets_before': 2, 'pellets_after': 2}})
            trace.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            metrics = {name: 0 for name in ('decisions', 'simulation_frames', 'pellets_collected', 'life_losses',
                       'avoidable_immediate_deaths', 'avoidable_risk_decisions', 'all_immediate_actions_fatal',
                       'loop_decisions', 'first_life_pellets', 'post_respawn_pellets', 'power_pellets', 'ghosts_eaten')}
            metrics.update(avoidable_immediate_deaths=1, avoidable_risk_decisions=1,
                           longest_no_pellet_decisions=2, longest_loop_streak_decisions=0)
            report = {'episodes': [{'metrics': metrics, 'trace': str(trace)}], 'level_clears': 0, 'capped_episodes': 0}
            with patch('hard_case_evaluation.cohorts', return_value=['power_expiry']):
                result = native_summary(report)
            cohort = result['cohorts']['power_expiry']
            self.assertEqual(cohort['safe_alternative_opportunities'], 1)
            self.assertEqual(cohort['all_immediate_actions_fatal'], 1)
            self.assertEqual(cohort['fatal_choice_when_safe_available'], 1)

    def test_probes_do_not_leak_labels_and_reject_an_adapter_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'development.jsonl'
            record = {'_meta': {'id': 'hard-1'}, 'state': {'public': 'state'},
                      'questions': {'move': {'label': 'left', 'criteria': {'left': '', 'right': ''}}}}
            path.write_text(json.dumps(record) + '\n')
            def predict(requests):
                self.assertEqual(requests, [{'state': {'public': 'state'}}])
                return [{'answers': {'move': {'choice': 'left', 'probabilities': {'left': .8, 'right': .2}}}}]
            with patch('hard_case_evaluation.inference_request', side_effect=lambda r: {'state': r['state']}), \
                    patch('hard_case_evaluation.cohorts', return_value=['pellet_stall']):
                result = evaluate_hard_cases(path, predict, lambda: {'sha': 'v3'})
                self.assertEqual(result['teacher_agreement'], 1)
                self.assertAlmostEqual(result['rows'][0]['teacher_probability_margin'], .6)
                with self.assertRaisesRegex(RuntimeError, 'adapter changed'):
                    evaluate_hard_cases(path, predict, iter([{'sha': 'v3'}, {'sha': 'v4'}]).__next__)

    def test_behavior_roles_do_not_count_followthrough_as_additional_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'development.jsonl'
            rows = []
            for index, kind in enumerate(('targeted', 'informative')):
                rows.append({'_meta': {'id': str(index), 'class': kind,
                    'primary_role': 'actual_expiry_evasion', 'behavior': {'actual_expiry_evasion': True}},
                    'state': {'public': 'state'}, 'questions': {'move': {
                    'label': 'left', 'criteria': {'left': '', 'right': ''}}}})
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            response = {'answers': {'move': {'choice': 'left', 'probabilities': {'left': .8, 'right': .2}}}}
            with patch('hard_case_evaluation.inference_request', side_effect=lambda r: {'state': r['state']}), \
                    patch('hard_case_evaluation.cohorts', return_value=[]):
                result = evaluate_hard_cases(path, lambda requests: [response for _ in requests], lambda: {'sha': 'v3'})
            self.assertEqual(result['cohorts']['actual_expiry_evasion']['n'], 2)
            self.assertEqual(result['primary_roles']['actual_expiry_evasion']['n'], 1)
            self.assertIsNone(result['rows'][1]['primary_role'])


if __name__ == '__main__':
    unittest.main()
