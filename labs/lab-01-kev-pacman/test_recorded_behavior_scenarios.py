"""Saved-root provenance and follow-through prediction honesty checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import behavior_scenarios as behavior
from pacman_lab import body
import recorded_behavior_scenarios as recorded
import teacher_offline_data as offline
from teacher_validation import fingerprint


class RecordedProvenanceTests(unittest.TestCase):
    def state(self, turn=0, pellets=4):
        return {'turn': turn, 'pellets_remaining': pellets, 'maze': ['#####', '#   #', '#####'],
                'player': {'row': 1, 'column': 2, 'heading': 'left'}, 'legal_moves': ['left', 'right']}

    def test_recorded_response_is_bound_to_exact_trace_state_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); request = body(self.state()); identity = {'checkpoint_sha256': 'fixture'}
            response = {'answers': {'move': {'choice': 'right', 'probabilities': {'left': 0.1, 'right': 0.9}}}}
            row = {'request': request, 'response': response, 'active_checkpoint': identity, 'diagnostics': {'choice': 'right'}}
            behavior._write_rows(directory / 'trace.jsonl', [row]); behavior._save(directory / 'attempt.json', {'checkpoint': identity})
            evidence = {'attempt': 'attempt.json', 'attempt_sha256': behavior._hash(directory / 'attempt.json'),
                'trace': 'trace.jsonl', 'trace_sha256': behavior._hash(directory / 'trace.jsonl'), 'root_index': 0,
                'response': response, 'response_sha256': fingerprint(response), 'checkpoint': identity,
                'request_sha256': offline.request_fingerprint(request), 'state_sha256': fingerprint(request['state']),
                'choice': 'right', 'original_native_diagnostics': row['diagnostics']}
            candidate = {'request': request, 'alternative': 'right', 'state_sha256': fingerprint(request['state']),
                'prefix_actions': [], 'prefix_state_sha256': [fingerprint(request['state'])], 'recorded_learner_evidence': evidence}
            with mock.patch.object(recorded.v4, 'require_v3_identity'):
                self.assertEqual(recorded._validate_recorded(candidate, directory), evidence)
                edited = copy.deepcopy(candidate); edited['recorded_learner_evidence']['response']['answers']['move']['choice'] = 'left'
                with self.assertRaises(ValueError):
                    recorded._validate_recorded(edited, directory)

    def test_post_window_has_only_parent_prediction_and_prior_food_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); root_id = recorded.PREFIX + ':example'
            root = {'_meta': {'id': root_id, 'class': 'targeted', 'recorded_learner_evidence': {'choice': 'right'},
                             'learner_disagreement_measured': True}}
            rows = [{'request': offline._request(body(self.state(i, 4 if i < 4 else 3))), 'choice': 'left',
                     'state_sha256': fingerprint(self.state(i, 4 if i < 4 else 3)), 'diagnostics': {'life_index': 0}} for i in range(9)]
            behavior._write_rows(directory / 'cases/example/positive.jsonl', rows)
            with mock.patch.object(recorded, 'scenario_records', return_value=[root]):
                windows = recorded.scenario_windows(directory, root_id)
            self.assertEqual([r['_meta']['decision_offset'] for r in windows], [4, 8])
            for row in windows:
                self.assertFalse(row['_meta']['learner_disagreement_measured'])
                self.assertNotIn('recorded_learner_evidence', row['_meta'])
                self.assertEqual(row['_meta']['parent_recorded_learner_evidence'], {'choice': 'right'})
                self.assertEqual(row['_meta']['pellets_since_root'], 1)


if __name__ == '__main__':
    unittest.main()
