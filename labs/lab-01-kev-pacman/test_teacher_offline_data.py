"""CPU-only replay/admission/selection tests; no learner inference or training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import teacher_offline_data as offline
from gameplay_benchmark import BenchmarkEngine, diagnostic_record, summarize
from pacman_lab import ROOT, body
from teacher_validation import fingerprint, source_hashes


class OfflineTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reconstruct ONE existing qualified teacher game from its packaged
        # action-only evidence. It is a test fixture, never a new training seed.
        report = json.loads((ROOT / 'evaluation/teacher-qualification.json').read_text())
        episode = report['episodes'][0]
        cls.options = report['options']
        with zipfile.ZipFile(ROOT / 'evaluation' / report['replay_bundle']) as archive:
            actions = [json.loads(line)['choice'] for line in archive.read(episode['replay_entry']).splitlines()]
        cls.source_rows = []; diagnostics = []
        with BenchmarkEngine() as engine:
            state = engine.request('reset', options={'seed': episode['seed'], 'level': episode['level'], 'action_version': 2})
            cls.state = copy.deepcopy(state)
            for move in actions:
                risks = engine.request('risks'); response = offline._one_hot(state, move); step = engine.step(move)
                diag = diagnostic_record(state, response, step, risks, 0)
                diagnostics.append(diag)
                cls.source_rows.append({'request': body(state), 'response': response, 'diagnostics': diag,
                    'controller': 'qualified_teacher', 'teacher': {'choice': move,
                        'algorithm': 'native-rollout-mpc-food-routing-v2', 'scenario_seeds': cls.options['scenario_seeds'],
                        'buffers': cls.options['buffers'], 'horizon_frames': cls.options['horizon_frames']}})
                state = step['state']
        cls.metrics = summarize(diagnostics, 'level_cleared')
        cls.job = {'id': f'off-policy-fixture-train-level-{episode["level"]}-seed-{episode["seed"]}-normal',
            'split': 'train', 'level': episode['level'], 'seed': episode['seed'], 'mode': 'normal',
            'options': cls.options, 'limits': {'roots_per_episode': 64, 'perturb_roots_per_episode': 12, 'perturb_turns': 4},
            'perturb': True}

    def source_fixture(self, directory):
        folder = Path(directory) / 'source'; folder.mkdir()
        trace = folder / 'trace.jsonl'; offline._write_rows(trace, self.source_rows)
        attempt = {'id': self.job['id'], 'split': 'train', 'seed': self.job['seed'], 'level': self.job['level'],
            'mode': 'normal', 'job_sha256': fingerprint(self.job), 'accepted': True, 'failures': [],
            'metrics': self.metrics, 'full_metrics': self.metrics,
            'files': {f'episodes/{self.job["id"]}/trace.jsonl': offline._hash(trace)}}
        path = folder / 'attempt.json'; offline._save(path, attempt)
        config = {'version': offline.COLLECTION_PREFIX, 'qualification_sha256': 'c' * 64,
            'source_sha256': source_hashes(), 'teacher_options': self.options, 'protocol': offline.SPEC,
            'generator_sha256': offline._hash(ROOT / 'v4_data.py'),
            'episode_sets': {'off_policy/fixture': {'train': [self.job['seed']], 'development': [460003],
                'parameters': {**self.job['limits'], 'perturb': self.job['perturb']}}}}
        return {'job': self.job, 'source_attempt': str(path), 'source_trace': str(trace),
                'source_config': config, 'evidence_directory': str(Path(directory) / 'evidence')}

    def minimal_record(self, number, flags=(), family=None, digest=None, kind='targeted', parent=None):
        return {'_meta': {'id': f'root-{number}', 'class': kind, 'request_sha256': digest or f'digest-{number}',
            'group_id': family or f'family-{number}', 'episode': f'episode-{number}', 'source_index': number,
            'life_index': 0, 'cohorts': {k: True for k in flags}, 'parent_targeted_id': parent}}

    def test_native_full_game_replay_recomputes_every_transition_and_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.source_fixture(directory)
            replay, rows = offline._replay_trace(task['source_trace'], self.job, offline._load(task['source_attempt']))
            self.assertTrue(replay['accepted']); self.assertEqual(replay['metrics']['life_losses'], 0)
            self.assertEqual(replay['metrics']['pellets_remaining'], 0)
            self.assertEqual(len(rows), len(self.source_rows))
            self.assertEqual(rows[0]['request'], offline._request(body(self.state)))
            self.assertNotIn('teacher', rows[0]['request'])
            self.assertIn('power_transition', rows[0])
            changed = copy.deepcopy(self.source_rows); changed[2]['request']['state']['score'] += 10
            offline._write_rows(task['source_trace'], changed)
            with self.assertRaisesRegex(ValueError, 'observation/request differs'):
                offline._replay_trace(task['source_trace'], self.job, offline._load(task['source_attempt']))

    def test_native_counterfactual_tamper_and_premature_trace_end_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.source_fixture(directory)
            changed = copy.deepcopy(self.source_rows)
            move = changed[0]['diagnostics']['choice']
            changed[0]['diagnostics']['immediate_counterfactuals'][move]['action_frames'] += 1
            offline._write_rows(task['source_trace'], changed)
            with self.assertRaisesRegex(ValueError, 'risks/diagnostics differ'):
                offline._replay_trace(task['source_trace'], self.job, offline._load(task['source_attempt']))
            offline._write_rows(task['source_trace'], self.source_rows[:3])
            with self.assertRaisesRegex(ValueError, 'stopped before'):
                offline._replay_trace(task['source_trace'], self.job, offline._load(task['source_attempt']))

    def test_receipt_resume_reuses_exact_committed_native_evidence_only(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.source_fixture(directory)
            result = offline._verify_job(task)
            self.assertFalse(result['reused']); self.assertTrue(result['accepted'])
            with patch.object(offline, '_replay_trace', side_effect=AssertionError('Already verified')):
                self.assertTrue(offline._verify_job(task)['reused'])
            path = Path(task['evidence_directory']) / 'episodes' / self.job['id'] / 'verified.jsonl'
            path.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                offline._verify_job(task)

    def test_committed_receipt_is_last_and_source_changes_invalidate_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.source_fixture(directory)
            with patch.object(offline, '_replay_trace', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    offline._verify_job(task)
            receipt = Path(task['evidence_directory']) / 'episodes' / self.job['id'] / 'receipt.json'
            self.assertFalse(receipt.exists())
            offline._verify_job(task)
            source = Path(task['source_trace']); source.write_text(source.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'inputs/source changed'):
                offline._verify_job(task)

    def test_receipt_also_binds_declared_job_source_options_and_original_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.source_fixture(directory); result = offline._verify_job(task)
            receipt = offline._load(Path(task['evidence_directory']) / result['receipt'])
            attempt = offline._load(task['source_attempt'])
            offline._validate_source_binding(receipt, task['source_config'], attempt)
            changed = copy.deepcopy(task['source_config']); changed['teacher_options']['scenario_seeds'] = [1, 2]
            with self.assertRaisesRegex(ValueError, 'configuration receipt binding differs'):
                offline._validate_source_binding(receipt, changed, attempt)
            changed = copy.deepcopy(attempt); changed['accepted'] = False
            with self.assertRaisesRegex(ValueError, 'qualification admission differs'):
                offline._validate_source_binding(receipt, task['source_config'], changed)
            changed = copy.deepcopy(receipt); changed['binding']['job']['seed'] += 1
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                offline._validate_source_binding(changed, task['source_config'], attempt)

    def test_postrespawn_alone_is_not_a_hard_scenario_and_all_choices_remain(self):
        state = copy.deepcopy(self.state); state['life_epoch'] = 1
        risks = {move: {'life_lost': False, 'action_frames': 8} for move in state['legal_moves']}
        self.assertTrue(offline.cohort_flags(state, risks)['post_respawn'])
        self.assertFalse(offline.is_targeted(state, risks, state['legal_moves'][0]))
        risks[state['legal_moves'][1]]['life_lost'] = True
        self.assertTrue(offline.is_targeted(state, risks, state['legal_moves'][0]))
        self.assertFalse(offline.is_targeted(state, risks, state['legal_moves'][1]))
        request = body(state); proof = {'request': offline._request(request), 'choice': state['legal_moves'][0],
            'source_index': 0, 'life_index': 1, 'state_sha256': fingerprint(state),
            'request_sha256': offline.request_fingerprint(request), 'cohorts': offline.cohort_flags(state, risks),
            'diagnostics': {'immediate_counterfactuals': risks}}
        receipt = {'binding': {'job': self.job, 'source_trace_sha256': 'a' * 64,
                              'source_attempt_sha256': 'b' * 64, 'qualification_sha256': 'c' * 64}}
        record = offline._record(proof, receipt, self.job['id'], 'train')
        self.assertEqual(set(record['questions']['move']['criteria']), set(state['legal_moves']))
        self.assertEqual(record['_meta']['immediate_fatal_alternatives'], [state['legal_moves'][1]])
        self.assertFalse(record['_meta']['learner_disagreement_measured'])
        self.assertEqual(offline._request(record), offline._request(request))

    def test_power_pause_proxy_is_distinct_from_actual_expiry(self):
        state = copy.deepcopy(self.state); state['frightened'] = True
        state['timing']['power']['remaining_frames'] = 10
        state['ghosts'][0].update(mode='outside', frightened=True, row=state['player']['row'], column=state['player']['column'] - 1)
        risks = {move: {'life_lost': False, 'action_frames': 66} for move in state['legal_moves']}
        flags = offline.cohort_flags(state, risks, {'remaining_before': 10, 'remaining_after': 4, 'action_frames': 66})
        self.assertTrue(flags['potential_expiry_during_action_near_ghost'])
        self.assertFalse(flags['actual_power_expiry_during_teacher_action_near_ghost'])
        flags = offline.cohort_flags(state, risks, {'remaining_before': 10, 'remaining_after': 0, 'action_frames': 12})
        self.assertTrue(flags['actual_power_expiry_during_teacher_action_near_ghost'])

    def test_targeted_only_floors_and_exact_budgets_reject_easy_fill(self):
        records = [self.minimal_record(i, flags=('ghost_intercept',) if i == 0 else ()) for i in range(8)]
        selected = offline.select_partition(records, {}, {'targeted': 4, 'informative': 0}, {'ghost_intercept': 1})
        self.assertEqual(len(selected), 4)
        self.assertIn(records[0], selected)
        with self.assertRaisesRegex(ValueError, 'targeted-only behavior cohorts'):
            offline.select_partition(records, {}, {'targeted': 4, 'informative': 0}, {'power_expiry': 1})
        with self.assertRaisesRegex(ValueError, 'targeted scenarios/family capacity'):
            offline.select_partition(records[:2], {}, {'targeted': 4, 'informative': 0}, {})
        with self.assertRaisesRegex(ValueError, 'three quarters'):
            offline.select_partition(records, {}, {'targeted': 2, 'informative': 1}, {})

    def test_dedup_and_shared_family_cap_are_not_relaxed(self):
        repeated = [self.minimal_record(i, digest='same') for i in range(4)]
        with self.assertRaisesRegex(ValueError, 'targeted scenarios/family capacity'):
            offline.select_partition(repeated, {}, {'targeted': 4, 'informative': 0}, {})
        one_family = [self.minimal_record(i, family='one') for i in range(4)]
        with self.assertRaisesRegex(ValueError, 'targeted scenarios/family capacity'):
            offline.select_partition(one_family, {}, {'targeted': 4, 'informative': 0}, {}, family_cap=2)
        self.assertEqual(offline._select_replay(one_family, [], 0), [])

    def test_informative_selection_requires_selected_parent_and_nearby_available_window(self):
        roots = [self.minimal_record(i) for i in range(3)]
        windows = {root['_meta']['id']: [self.minimal_record(100 + i, kind='informative', parent=root['_meta']['id'])]
                   for i, root in enumerate(roots)}
        episodes = {r['_meta']['episode']: {} for r in roots}
        with patch.object(offline, '_window_candidates', side_effect=lambda root, episode: windows[root['_meta']['id']]):
            selected = offline.select_partition(roots, episodes, {'targeted': 3, 'informative': 1}, {})
        self.assertEqual(len(selected), 4)
        self.assertIn(selected[-1]['_meta']['parent_targeted_id'], {r['_meta']['id'] for r in selected[:3]})
        with patch.object(offline, '_window_candidates', return_value=[]), self.assertRaisesRegex(ValueError, 'same-life informative windows'):
            offline.select_partition(roots, episodes, {'targeted': 3, 'informative': 1}, {})
        with self.assertRaisesRegex(ValueError, 'targeted scenarios/family capacity'):
            offline.select_partition(roots, episodes, {'targeted': 3, 'informative': 0}, {},
                                     excluded_request_sha256={'digest-0'})

    def test_identical_requests_are_quarantined_across_disjoint_seed_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixed_digest = 'd' * 64
            config = {'dataset': offline.PREFIX, 'source_sha256': source_hashes(),
                'exporter_sha256': offline._hash(Path(offline.__file__)), 'recipe': offline.RECIPE,
                'targets': offline.TARGETS, 'minimums': offline.MINIMUMS, 'family_cap': offline.FAMILY_CAP,
                'collection_fixed_sha256': fixed_digest, 'qualification_sha256': 'c' * 64}
            offline._save(directory / 'offline-config.json', config)
            state = copy.deepcopy(self.state)
            risks = {move: {'life_lost': False, 'action_frames': 8} for move in state['legal_moves']}
            risks[state['legal_moves'][1]]['life_lost'] = True
            choice = state['legal_moves'][0]
            # Test pool policy in isolation. These tiny receipts are not used
            # by verify_sources or exported as independently replayed games.
            for split, seed in [('train', 400001), ('development', 460003)]:
                identifier = f'{split}-fixture'; folder = directory / 'episodes' / identifier
                proof = {'source_index': 0, 'life_index': 0, 'request': offline._request(body(state)),
                    'choice': choice, 'state_sha256': fingerprint(state), 'request_sha256': offline.request_fingerprint(body(state)),
                    'cohorts': offline.cohort_flags(state, risks, {'remaining_after': 0}),
                    'power_transition': {'remaining_before': 0, 'remaining_after': 0, 'action_frames': 8},
                    'diagnostics': {'action_frames': 8, 'immediate_counterfactuals': risks}}
                offline._write_rows(folder / 'verified.jsonl', [proof])
                receipt = {'binding': {'job': {'id': identifier, 'split': split, 'seed': seed, 'level': 1},
                    'source_sha256': config['source_sha256'], 'exporter_sha256': config['exporter_sha256'],
                    'collection_fixed_sha256': fixed_digest, 'qualification_sha256': 'c' * 64,
                    'source_trace_sha256': 'a' * 64, 'source_attempt_sha256': 'b' * 64},
                    'files': {}, 'verified_teacher_rows': 1, 'accepted': True, 'failures': [],
                    'metrics': self.metrics, 'native_replayed': True}
                offline._save(folder / 'receipt.json', receipt)
                offline._save(folder / 'source-config.json', {})
                offline._save(folder / 'source-attempt.json', {})
            with patch.object(offline, '_validate_source_binding'):
                pools, episodes, report = offline._load_pool(directory)
            self.assertEqual(pools, {'train': [], 'development': []})
            self.assertEqual(len(episodes), 2)
            self.assertEqual(report['cross_partition_requests_excluded'], 1)
            self.assertIn(proof['request_sha256'], episodes['train-fixture']['excluded_request_sha256'])

    def test_window_candidates_never_cross_respawn_or_teacher_prefix(self):
        state = self.state; request = offline._request(body(state)); risks = {move: {'life_lost': False} for move in state['legal_moves']}
        proofs = [{'source_index': i, 'life_index': 0 if i < 10 else 1, 'request': request,
            'choice': state['legal_moves'][0], 'state_sha256': fingerprint(state), 'request_sha256': f'digest-{i}',
            'cohorts': {}, 'diagnostics': {'pellets_before': 100, 'pellets_after': 99 if i == 9 else 100,
                                        'immediate_counterfactuals': risks}} for i in range(20)]
        receipt = {'binding': {'job': self.job, 'source_trace_sha256': 'a' * 64,
                              'source_attempt_sha256': 'b' * 64, 'qualification_sha256': 'c' * 64}}
        root = offline._record(proofs[8], receipt, self.job['id'], 'train')
        windows = offline._window_candidates(root, {'proofs': proofs, 'receipt': receipt})
        self.assertTrue(windows)
        self.assertTrue(all(r['_meta']['source_index'] < 10 for r in windows))
        self.assertTrue(all(r['_meta']['parent_targeted_id'] == root['_meta']['id'] for r in windows))
        proofs = [p for p in proofs if p['source_index'] != 7]
        windows = offline._window_candidates(root, {'proofs': proofs, 'receipt': receipt})
        self.assertFalse(any(r['_meta']['source_index'] < 8 for r in windows), 'A prefix gap breaks the same-life teacher sequence')

    def test_source_jobs_ignore_inflight_and_bind_hash_config_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'teacher_options': self.options, 'episode_sets': {'off_policy/wave-001': {
                'train': [400001], 'development': [460003], 'parameters': {
                    'roots_per_episode': 64, 'perturb_roots_per_episode': 12, 'perturb_turns': 4, 'perturb': True}}}}
            self.assertEqual(offline._source_jobs(directory, config), [])
            identifier = 'off-policy-wave-001-train-level-1-seed-400001-normal'
            folder = Path(directory) / 'episodes' / identifier; folder.mkdir(parents=True)
            (folder / 'trace.jsonl.tmp').write_text('inflight')
            self.assertEqual(offline._source_jobs(directory, config), [])
            job = {**self.job, 'id': identifier, 'seed': 400001, 'level': 1}
            trace = folder / 'trace.jsonl'; trace.write_text('complete')
            attempt = {**{k: job[k] for k in ('id', 'split', 'seed', 'level', 'mode')},
                'job_sha256': fingerprint(job), 'files': {f'episodes/{identifier}/trace.jsonl': offline._hash(trace)}}
            offline._save(folder / 'attempt.json', attempt)
            self.assertEqual(len(offline._source_jobs(directory, config)), 1)
            trace.write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'trace hash mismatch'):
                offline._source_jobs(directory, config)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            offline._safe_path('/private/tmp/evidence', '../secret')

    def test_replay_metrics_and_source_admission_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.source_fixture(directory); expected = offline._load(task['source_attempt'])
            expected['metrics']['longest_no_pellet_decisions'] += 1
            with self.assertRaisesRegex(ValueError, 'admission/outcome/metrics differ'):
                offline._replay_trace(task['source_trace'], self.job, expected)
            expected = offline._load(task['source_attempt']); expected['accepted'] = False
            with self.assertRaisesRegex(ValueError, 'admission/outcome/metrics differ'):
                offline._replay_trace(task['source_trace'], self.job, expected)


if __name__ == '__main__':
    unittest.main()
