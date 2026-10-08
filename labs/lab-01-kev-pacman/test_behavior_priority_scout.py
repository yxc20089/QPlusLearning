"""CPU scout checks: cached prefixes, immutable resume and split quarantine."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import behavior_priority_scout as priority
import behavior_scenarios as behavior
from pacman_lab import body
import teacher_offline_data as offline
from teacher_validation import fingerprint


class PriorityScoutTests(unittest.TestCase):
    def make_episode(self, evidence, identifier, split):
        folder = evidence / 'episodes' / identifier; folder.mkdir(parents=True)
        receipt = {'accepted': True, 'metrics': {'life_losses': 0}, 'binding': {
            'job': {'split': split, 'seed': 101 if split == 'train' else 201, 'level': 1},
            'source_trace_sha256': 'fixture'}}
        behavior._save(folder / 'receipt.json', receipt)
        behavior._save(folder / 'source-config.json', {}); behavior._save(folder / 'source-attempt.json', {})
        rows = []; proofs = []
        for index in range(4):
            state = {'turn': index, 'maze': ['#####', '#   #', '#####'],
                     'player': {'row': 1, 'column': 2, 'heading': 'left'}, 'ghosts': [],
                     'legal_moves': ['left', 'right'], 'decisions_since_last_pellet': 0}
            request = body(state); risks = {'left': {'life_lost': False}, 'right': {'life_lost': False}}
            plan = {'candidates': [{'action': 'left', 'scenarios': [{'survived': True}]},
                                    {'action': 'right', 'scenarios': [{'survived': False}]}]}
            rows.append({'request': request, 'diagnostics': {'choice': 'left'}, 'teacher': plan})
            proofs.append({'request': request, 'source_index': index, 'choice': 'left',
                'request_sha256': offline.request_fingerprint(request),
                'diagnostics': {'immediate_counterfactuals': risks}})
        behavior._write_rows(folder / 'source-trace.jsonl', rows); behavior._write_rows(folder / 'verified.jsonl', proofs)

    def run_scout(self, evidence, output, train=3, dev=0):
        def bind(evidence_path, directory, identifier):
            source = evidence_path / 'episodes' / identifier / 'receipt.json'
            behavior._copy(source, directory / 'source-evidence/episodes' / identifier / 'receipt.json')
        config = {'generator_sha256': 'fixture', 'source_sha256': {}, 'qualification_sha256': 'fixture'}
        with mock.patch.object(behavior, '_assert_configuration', return_value=config), \
             mock.patch.object(offline, '_receipt_files'), mock.patch.object(offline, '_validate_source_binding'), \
             mock.patch.object(behavior, '_bind_source', side_effect=bind), \
             mock.patch.object(behavior, '_near', return_value=1), \
             mock.patch.object(priority, 'fingerprint', wraps=fingerprint) as hashing:
            result = priority.scout_priority_roots(evidence, output, 'unused', train, dev, 'wave')
        return result, sum('turn' in call.args[0] for call in hashing.call_args_list if isinstance(call.args[0], dict))

    def test_hash_each_state_once_and_resume_exact_same_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); evidence = root / 'evidence'; output = root / 'output'
            self.make_episode(evidence, 'teacher-a', 'train')
            report, state_hash_calls = self.run_scout(evidence, output)
            self.assertEqual(state_hash_calls, 4)
            self.assertEqual(report['selected'], {'train': 3})
            for entry in report['candidates']:
                candidate = behavior._load(output / 'cases' / entry['id'] / 'candidate.json')
                self.assertEqual(len(candidate['prefix_state_sha256']), len(candidate['prefix_actions']) + 1)
                self.assertEqual(candidate['state_sha256'], candidate['prefix_state_sha256'][-1])
            resumed, calls = self.run_scout(evidence, output)
            self.assertEqual(resumed, report); self.assertEqual(calls, 0)
            with self.assertRaises(ValueError):
                self.run_scout(evidence, output, train=4)

    def test_identical_train_development_inputs_are_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); evidence = root / 'evidence'; output = root / 'output'
            self.make_episode(evidence, 'teacher-a', 'train'); self.make_episode(evidence, 'teacher-b', 'development')
            report, _ = self.run_scout(evidence, output, train=2, dev=2)
            self.assertEqual(report['candidates'], [])
            self.assertEqual(report['deficits'], {'train': 2, 'development': 2})


if __name__ == '__main__':
    unittest.main()
