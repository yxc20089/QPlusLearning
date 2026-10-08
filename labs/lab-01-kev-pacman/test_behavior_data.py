"""CPU checks for measured behaviors, exact matching and evidence boundaries.

Synthetic selector fixtures test combinatorial constraints. A separate native
fixture checks the pellet-advantage predicate against actual game transitions;
none of these fixtures are admitted as a new qualified training source.
"""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import behavior_data as balanced
import behavior_scenarios as scenarios
import teacher_offline_data as offline
from gameplay_benchmark import BenchmarkEngine, diagnostic_record
from pacman_lab import ROOT, body
from teacher_validation import fingerprint


class BalancedBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qualification = json.loads((ROOT / 'evaluation/teacher-qualification.json').read_text())
        episode = qualification['episodes'][0]
        with zipfile.ZipFile(ROOT / 'evaluation' / qualification['replay_bundle']) as archive:
            actions = [json.loads(line)['choice'] for line in archive.read(episode['replay_entry']).splitlines()]
        cls.native_proofs = {}
        with BenchmarkEngine() as engine:
            state = engine.request('reset', options={'seed': episode['seed'], 'level': episode['level'], 'action_version': 2})
            cls.state = copy.deepcopy(state)
            for index, choice in enumerate(actions[:200]):
                risks = engine.request('risks'); step = engine.step(choice)
                diag = diagnostic_record(state, offline._one_hot(state, choice), step, risks, 0)
                cls.native_proofs[index] = cls.proof(state, index, choice, diag=diag)
                state = step['state']

    @staticmethod
    def proof(state, index, choice=None, fatal=(), food=False, life=0, diag=None):
        state = copy.deepcopy(state); state['turn'] = index; state['life_epoch'] = life
        choice = choice or state['legal_moves'][0]
        risks = {move: {'life_lost': move in fatal, 'pellets': int(food and move == choice), 'action_frames': 8}
                 for move in state['legal_moves']}
        diag = copy.deepcopy(diag) if diag is not None else {
            'immediate_counterfactuals': risks, 'pellets_before': 244, 'pellets_after': 244 - int(food), 'action_frames': 8}
        request = offline._request(body(state)); risks = diag['immediate_counterfactuals']
        transition = {'remaining_before': state['timing']['power']['remaining_frames'],
                      'remaining_after': state['timing']['power']['remaining_frames'], 'action_frames': diag['action_frames']}
        return {'source_index': index, 'life_index': life, 'request': request, 'choice': choice,
            'state_sha256': fingerprint(state), 'request_sha256': offline.request_fingerprint(request),
            'diagnostics': diag, 'power_transition': transition,
            'cohorts': offline.cohort_flags(state, risks, transition)}

    @staticmethod
    def minimal(number, roles, family='level-1-seed-400001', digest=None, origin='verified_offline_trajectory'):
        return {'questions': {'move': {'label': 'left'}}, '_meta': {'id': f'root-{number}',
            'request_sha256': digest or f'digest-{number}', 'group_id': family, 'split': 'train',
            'class': 'targeted', 'primary_role': None, 'origin': origin,
            'behavior': {role: role in roles for role in balanced.ROLES}}}

    def context(self, food_index=3, break_index=None, blocked=()):
        proofs = {i: self.proof(self.state, i, food=i == food_index, life=int(break_index is not None and i >= break_index))
                  for i in range(25)}
        job = {'id': 'fixture', 'split': 'train', 'level': 1, 'seed': 400001}
        receipt = {'binding': {'job': job, 'source_trace_sha256': 'a' * 64,
            'source_attempt_sha256': 'b' * 64, 'qualification_sha256': 'c' * 64}}
        episode = {'receipt': receipt, 'receipt_sha256': 'd' * 64, 'proofs': list(proofs.values()),
                   'excluded_request_sha256': set(blocked)}
        original = offline._record(proofs[0], receipt, job['id'], 'train')
        root = balanced._wrap(original, {'type': 'verified_offline_trajectory', 'episode': job['id']}, {}, {})
        return root, {'episode': episode, 'proofs': proofs, 'proof': proofs[0], 'blocked_request_sha256': set(blocked)}

    def test_native_immediate_progress_is_measured_not_proximity(self):
        found = None
        for proof in self.native_proofs.values():
            flags, evidence = balanced.trajectory_behavior(proof, self.native_proofs)
            if flags['productive_routing']:
                found = proof, evidence; break
        self.assertIsNotNone(found, 'Existing qualified native trace must contain a measured productive junction')
        proof, evidence = found; risks = proof['diagnostics']['immediate_counterfactuals']
        self.assertGreater(risks[proof['choice']]['pellets'], 0)
        self.assertTrue(any(not risks[m]['life_lost'] and not risks[m]['pellets'] for m in evidence['surviving_zero_pellet_alternatives']))
        changed = copy.deepcopy(proof)
        for risk in changed['diagnostics']['immediate_counterfactuals'].values():
            risk['pellets'] = 0
        changed['request']['state']['pellets_remaining'] = 5
        flags, _ = balanced.trajectory_behavior(changed, {changed['source_index']: changed})
        self.assertFalse(flags['productive_routing']); self.assertFalse(flags['cleanup'])

    def test_fatal_alternative_and_post_escape_food_are_distinct(self):
        state = copy.deepcopy(self.state); state['player']['heading'] = 'right'; state['legal_moves'] = ['left', 'right']
        root = self.proof(state, 0, choice='left', fatal=('right',))
        sequence = {0: root, 1: self.proof(state, 1), 2: self.proof(state, 2, food=True)}
        flags, evidence = balanced.trajectory_behavior(root, sequence)
        self.assertTrue(flags['immediate_evasion']); self.assertTrue(flags['retreat_to_food'])
        self.assertEqual(evidence['first_food_decision_offset'], 2)
        sequence[2]['life_index'] = 1
        flags, _ = balanced.trajectory_behavior(root, sequence)
        self.assertFalse(flags['retreat_to_food'])
        food_on_root = self.proof(state, 0, choice='left', fatal=('right',), food=True)
        flags, _ = balanced.trajectory_behavior(food_on_root, {0: food_on_root})
        self.assertTrue(flags['immediate_evasion']); self.assertFalse(flags['retreat_to_food'])

    def test_geometric_escape_tracks_same_dangerous_ghost(self):
        state = copy.deepcopy(self.state); state['player']['heading'] = 'right'; state['legal_moves'] = ['left', 'right']
        root = self.proof(state, 0, choice='left'); sequence = {0: root, 1: self.proof(state, 1, food=True)}
        # A frightened/returning Blinky disappearing from the dangerous set
        # must not masquerade as an increased geometric gap to that ghost.
        with patch.object(balanced, '_danger_distances', side_effect=[{'Blinky': 2, 'Clyde': 6}, {'Clyde': 4}]):
            flags, evidence = balanced.trajectory_behavior(root, sequence)
        self.assertFalse(evidence['escape']); self.assertFalse(flags['retreat_to_food'])
        with patch.object(balanced, '_danger_distances', side_effect=[{'Blinky': 2, 'Clyde': 6}, {'Blinky': 4, 'Clyde': 6}]):
            flags, evidence = balanced.trajectory_behavior(root, sequence)
        self.assertTrue(flags['retreat_to_food']); self.assertEqual(evidence['escape_kind'], 'same_ghost_geometric')
        self.assertEqual(evidence['same_ghost_distances']['Blinky'], {'before': 2, 'after': 4})

    def test_expiry_requires_exact_branch_flag_not_elapsed_proxy(self):
        row = {'_meta': {'cohorts': {'potential_expiry_during_action_near_ghost': True}}}
        self.assertFalse(balanced._scenario_roles(row)['actual_expiry_evasion'])
        row['_meta']['cohorts']['actual_expiry_fatal_alternative'] = True
        flags = balanced._scenario_roles(row)
        self.assertTrue(flags['actual_expiry_evasion']); self.assertTrue(flags['immediate_evasion'])
        self.assertFalse(flags['anticipatory_escape'])

    def test_recorded_routing_alias_keeps_context_and_cleanup_proofs(self):
        row = {'state': {'legal_moves': ['up', 'left', 'right'], 'turn': 40,
            'decisions_since_last_pellet': 10, 'pellets_remaining': 150},
            '_meta': {'cohorts': {'productive_junction': True}}}
        self.assertTrue(balanced._scenario_roles(row)['productive_routing'])
        row['state']['turn'] = 0
        self.assertFalse(balanced._scenario_roles(row)['productive_routing'])
        row['state']['turn'] = 40
        row['state']['legal_moves'] = ['left', 'right']
        self.assertFalse(balanced._scenario_roles(row)['productive_routing'])
        row['_meta']['cohorts'] = {'sparse_cleanup': True}
        self.assertTrue(balanced._scenario_roles(row)['cleanup'])

    def test_quota_matching_reassigns_overlap_without_rare_role_theft(self):
        records = [self.minimal(0, ('immediate_evasion', 'anticipatory_escape')),
                   self.minimal(1, ('immediate_evasion',)), self.minimal(2, ('cleanup',))]
        selected = balanced.select_roots(records, {'immediate_evasion': 1, 'anticipatory_escape': 1, 'cleanup': 1}, family_cap=4)
        self.assertEqual(Counter(r['_meta']['primary_role'] for r in selected),
                         {'immediate_evasion': 1, 'anticipatory_escape': 1, 'cleanup': 1})
        self.assertEqual(next(r['_meta']['id'] for r in selected if r['_meta']['primary_role'] == 'anticipatory_escape'), 'root-0')
        self.assertEqual(records[0]['_meta']['primary_role'], None)

    def test_request_capacity_and_real_per_role_receipts_are_preserved(self):
        old = self.minimal(0, ('immediate_evasion',), digest='same')
        stronger = self.minimal(1, ('anticipatory_escape',), digest='same', origin='verified_behavior_recovery')
        with self.assertRaisesRegex(ValueError, 'distinct proved behavioral roots'):
            balanced.select_roots([old, stronger], {'immediate_evasion': 1, 'anticipatory_escape': 1}, family_cap=4)
        selected = balanced.select_roots([old, stronger], {'anticipatory_escape': 1}, family_cap=4)
        self.assertEqual(selected[0]['_meta']['id'], 'root-1')
        self.assertFalse(selected[0]['_meta']['behavior']['immediate_evasion'])
        conflicting = copy.deepcopy(stronger); conflicting['questions']['move']['label'] = 'right'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            balanced.select_roots([old, conflicting], {'anticipatory_escape': 1}, family_cap=4)

    def test_matching_prefers_actual_archived_error_within_same_role_budget(self):
        generic = self.minimal(0, ('immediate_evasion',), family='one')
        recorded = self.minimal(1, ('immediate_evasion',), family='two', origin='verified_recorded_learner_recovery')
        recorded['_meta']['learner_disagreement_measured'] = True
        selected = balanced.select_roots([generic, recorded], {'immediate_evasion': 1}, family_cap=4)
        self.assertEqual(selected[0]['_meta']['id'], 'root-1')
        self.assertEqual(selected[0]['_meta']['primary_role'], 'immediate_evasion')

    def test_family_reservation_and_excluded_inputs_cannot_be_relaxed(self):
        records = [self.minimal(i, ('cleanup',)) for i in range(4)]
        with self.assertRaisesRegex(ValueError, 'family capacity'):
            balanced.select_roots(records, {'cleanup': 4}, family_cap=4)
        with self.assertRaisesRegex(ValueError, 'family capacity'):
            balanced.select_roots(records, {'cleanup': 3}, family_cap=4, excluded=('digest-0', 'digest-1'))
        diverse = [self.minimal(i, ('cleanup',), family=f'family-{i}') for i in range(4)]
        self.assertEqual(len(balanced.select_roots(diverse, {'cleanup': 4}, family_cap=4)), 4)

    def test_explicit_full_plan_rejects_broad_filler_and_rare_zero(self):
        self.assertEqual(balanced.validate_plan(balanced.DEFAULT_PLAN), balanced.DEFAULT_PLAN)
        bad = copy.deepcopy(balanced.DEFAULT_PLAN); bad['train']['broad_exposure'] += 200; bad['train']['productive_routing'] -= 200
        with self.assertRaisesRegex(ValueError, 'full budgets'):
            balanced.validate_plan(bad)
        bad = copy.deepcopy(balanced.DEFAULT_PLAN); bad['train']['productive_routing'] += bad['train']['actual_expiry_evasion']; bad['train']['actual_expiry_evasion'] = 0
        with self.assertRaises(ValueError):
            balanced.validate_plan(bad)

    def test_windows_measure_progress_before_state_and_bound_same_life(self):
        root, context = self.context(food_index=3)
        rows = balanced._offline_windows(root, context); by_offset = {r['_meta']['window_decision_offset']: r for r in rows}
        self.assertEqual(by_offset[2]['_meta']['window_kind'], 'post_followthrough')
        self.assertEqual(by_offset[4]['_meta']['window_kind'], 'post_routing_progress')
        self.assertNotIn(1, by_offset)
        self.assertTrue(all(r['_meta']['parent_targeted_id'] == root['_meta']['id'] for r in rows))
        blocked = context['proofs'][4]['request_sha256']; context['blocked_request_sha256'].add(blocked)
        self.assertNotIn(blocked, {r['_meta']['request_sha256'] for r in balanced._offline_windows(root, context)})
        root, context = self.context(food_index=3, break_index=3)
        self.assertTrue(all(r['_meta']['window_decision_offset'] < 3 for r in balanced._offline_windows(root, context)))
        self.assertNotIn('behavior_evidence', offline._request(rows[0]))

    def test_opening_food_crossroad_does_not_count_as_routing_behavior(self):
        state = copy.deepcopy(self.state); state['legal_moves'] = ['left', 'right', 'up']; state['turn'] = 0
        root = self.proof(state, 0, food=True)
        flags, _ = balanced.trajectory_behavior(root, {0: root})
        self.assertFalse(flags['productive_routing'])
        state['turn'] = 12; state['visits_to_current_tile'] = 2
        root = self.proof(state, 12, food=True)
        flags, _ = balanced.trajectory_behavior(root, {12: root})
        self.assertTrue(flags['productive_routing'])

    def test_post_window_budget_rejects_lead_in_substitution(self):
        root = self.minimal(0, ('immediate_evasion',)); contexts = {'root-0': {}}
        lead = self.minimal(1, (), digest='lead'); lead['_meta'].update(class_='informative')
        lead['_meta']['class'] = 'informative'; lead['_meta']['window_kind'] = 'lead_in'
        with patch.object(balanced, 'windows_for', return_value=[lead]):
            with self.assertRaisesRegex(ValueError, 'post-progress'):
                balanced.select_windows([root], contexts, count=1, minimum_post=1)
        with self.assertRaisesRegex(ValueError, 'fit'):
            balanced.select_windows([root], contexts, count=1, minimum_post=2)

    def test_post_escape_floor_rejects_routing_and_dry_recovery_food(self):
        root, context = self.context(food_index=3)
        root['_meta']['behavior'] = {'productive_routing': True, 'retreat_to_food': True}
        contexts = {root['_meta']['id']: context}
        rows = balanced._offline_windows(root, context)
        self.assertTrue(any(r['_meta']['window_kind'] == 'post_recovery_progress' for r in rows))
        self.assertFalse(any(r['_meta']['window_kind'] == 'post_followthrough_with_progress' for r in rows))
        with self.assertRaisesRegex(ValueError, 'threat-escape'):
            balanced.select_windows([root], contexts, count=1, minimum_post=1)
        root['_meta']['behavior_evidence']['escape'] = True
        selected = balanced.select_windows([root], contexts, count=1, minimum_post=1)
        self.assertEqual(selected[0]['_meta']['window_kind'], 'post_followthrough_with_progress')
        self.assertGreaterEqual(selected[0]['_meta']['window_decision_offset'], 4)

    def test_scenario_adapter_uses_bound_receipt_and_actual_offset_fields(self):
        root, context = self.context(); original = copy.deepcopy(root)
        original['_meta'].update({'source_receipt': 'cases/real/receipt.json', 'source_receipt_sha256': 'f' * 64,
            'source_index': None, 'decision_offset': 4, 'window_kind': 'post_followthrough_with_progress',
            'cohorts': {'anticipatory_harm': True}})
        context = {'scenario_directory': '/fixture', 'scenario_root': root, 'blocked_request_sha256': set()}
        with patch.object(scenarios, 'scenario_windows', return_value=[original]):
            rows = balanced.windows_for(root, context)
        self.assertEqual(rows[0]['_meta']['window_decision_offset'], 4)
        self.assertEqual(rows[0]['_meta']['provenance']['receipt_sha256'], 'f' * 64)
        self.assertEqual(rows[0]['_meta']['provenance']['original_record_sha256'], fingerprint(original))
        self.assertTrue(rows[0]['_meta']['behavior']['anticipatory_escape'])

    def test_archived_learner_error_stays_at_root_not_future_window(self):
        root, _ = self.context(); archived = copy.deepcopy(root)
        archived['_meta'].update({'learner_disagreement_measured': True,
            'recorded_learner_evidence': {'trace_sha256': 'a' * 64, 'root_state_sha256': archived['_meta']['state_sha256'],
                                        'checkpoint_sha256': 'b' * 64}})
        result = balanced._wrap(archived, {'type': 'verified_behavior_recovery'}, {}, {})
        self.assertTrue(result['_meta']['learner_disagreement_measured'])
        self.assertEqual(result['_meta']['recorded_learner_evidence'], archived['_meta']['recorded_learner_evidence'])
        archived['_meta']['class'] = 'informative'
        window = balanced._wrap(archived, {'type': 'verified_behavior_recovery'}, {}, {})
        self.assertFalse(window['_meta']['learner_disagreement_measured'])
        archived['_meta']['class'] = 'targeted'; archived['_meta']['recorded_learner_evidence'] = None
        with self.assertRaisesRegex(ValueError, 'no recorded learner evidence'):
            balanced._wrap(archived, {'type': 'verified_behavior_recovery'}, {}, {})

    def test_pool_quarantines_shared_observation_without_reassigning_seeds(self):
        root, _ = self.context(); dev = copy.deepcopy(root); digest = root['_meta']['request_sha256']
        dev['_meta'].update({'id': 'dev-root', 'split': 'development', 'seed': 460003, 'group_id': 'level-1-seed-460003'})
        contexts = {r['_meta']['id']: {} for r in (root, dev)}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); qualification = directory / 'qualification.json'; qualification.write_text('{}')
            offline._save(directory / 'offline-config.json', {'qualification_sha256': offline._hash(qualification)})
            with patch.object(balanced, 'require_qualified_teacher'), patch.object(offline, '_load_pool', return_value=({}, {}, {})), \
                    patch.object(balanced, '_offline_records', return_value=([root, dev], contexts)), \
                    patch.object(scenarios, 'validate_scenarios'), patch.object(scenarios, 'scenario_records', return_value=[]):
                pools, _, _, report = balanced.load_pools(directory, '/fixture', qualification)
        self.assertEqual(pools, {'train': [], 'development': []})
        self.assertEqual(report['balanced_cross_partition_or_ambiguous_inputs_excluded'], 1)
        self.assertEqual(root['_meta']['split'], 'train'); self.assertEqual(dev['_meta']['split'], 'development')
        self.assertEqual(root['_meta']['request_sha256'], digest)

    def test_optional_recorded_catalog_preserves_prediction_provenance(self):
        root, _ = self.context(); root['_meta'].update({'cohorts': {'immediate_evasion': True},
            'source_receipt': 'cases/recorded/receipt.json', 'source_receipt_sha256': 'f' * 64,
            'learner_disagreement_measured': True,
            'recorded_learner_evidence': {'trace_sha256': 'a' * 64, 'checkpoint_sha256': 'b' * 64}})
        wrapper = SimpleNamespace(validate_scenarios=lambda *a, **kw: None, scenario_records=lambda directory: [root])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); qualification = directory / 'qualification.json'; qualification.write_text('{}')
            offline._save(directory / 'offline-config.json', {'qualification_sha256': offline._hash(qualification)})
            with patch.object(balanced, 'require_qualified_teacher'), patch.object(offline, '_load_pool', return_value=({}, {}, {})), \
                    patch.object(balanced, '_offline_records', return_value=([], {})), \
                    patch.object(scenarios, 'validate_scenarios'), patch.object(scenarios, 'scenario_records', return_value=[]), \
                    patch.dict('sys.modules', {'recorded_behavior_scenarios': wrapper}):
                pools, contexts, _, _ = balanced.load_pools(directory, '/behavior', qualification, recorded_directory='/recorded')
        selected = pools['train'][0]; meta = selected['_meta']
        self.assertEqual(meta['origin'], 'verified_recorded_learner_recovery')
        self.assertTrue(meta['learner_disagreement_measured'])
        self.assertEqual(meta['recorded_learner_evidence'], root['_meta']['recorded_learner_evidence'])
        self.assertEqual(meta['provenance']['original_record_sha256'], fingerprint(root))
        self.assertEqual(contexts[meta['id']]['scenario_module'], 'recorded_behavior_scenarios')
        self.assertNotIn('recorded_learner_evidence', offline._request(selected))

    def test_evidence_zip_checks_bytes_members_and_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); archive = directory / 'evidence.zip'; content = b'actual proof bytes'
            with zipfile.ZipFile(archive, 'w') as result:
                result.writestr('offline/episodes/one/receipt.json', content)
            binding = {'filename': archive.name, 'sha256': offline._hash(archive),
                'files': {'offline/episodes/one/receipt.json': hashlib.sha256(content).hexdigest()}}
            destination = directory / 'extracted'; balanced._extract_verified_bundle(directory, {'evidence': binding}, destination)
            self.assertEqual((destination / 'offline/episodes/one/receipt.json').read_bytes(), content)
            changed = copy.deepcopy(binding); changed['files']['offline/episodes/one/receipt.json'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'entry checksum'):
                balanced._extract_verified_bundle(directory, {'evidence': changed}, destination)
            with zipfile.ZipFile(archive, 'w') as result:
                result.writestr('../outside', content)
            binding = {'filename': archive.name, 'sha256': offline._hash(archive),
                'files': {'../outside': hashlib.sha256(content).hexdigest()}}
            with self.assertRaises(ValueError):
                balanced._extract_verified_bundle(directory, {'evidence': binding}, destination)


if __name__ == '__main__':
    unittest.main()
