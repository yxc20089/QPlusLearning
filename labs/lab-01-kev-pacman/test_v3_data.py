"""CPU checks for fresh-seed corrections, native replay and separate v3 recovery."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from checkpoint_backup import save_backup, restore_backup
from cloud_runtime import CloudRuntime, BASE_MODEL, BASE_REVISION
from gameplay_benchmark import BenchmarkEngine, diagnostic_record
from pacman_lab import ROOT, body, teacher
from planner_data import RECIPE
from v3_data import (OPPOSITE, balanced_select, behavior_summary, candidate_flags, check_seed_splits,
                     cpu_capacity, file_hash, recovery_progress, recovery_workers, replay_prefix,
                     require_v2_identity, run_recoveries, select_roots)


class V3CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with BenchmarkEngine() as engine:
            cls.state = engine.request('reset', options={'seed': 300017, 'level': 1, 'action_version': 2})

    def candidate(self, turn, origin='teacher_recovery', critical=False, reverse=False):
        state = copy.deepcopy(self.state)
        state['turn'] = turn
        request = body(state)
        request.pop('model')
        backwards = OPPOSITE[state['player']['heading']]
        move = backwards if reverse else state['player']['heading']
        request['questions']['move']['label'] = move
        request['_meta'] = {'origin': origin, 'immediate_survival_critical': critical,
                            'learner_choice': state['player']['heading'] if origin == 'learner_visited' else None}
        return request

    def test_worker_selection_respects_affinity_physical_cores_and_nested_quota(self):
        quota = {}

        def read(path, *args, **kwargs):
            name = str(path)
            if name == '/proc/self/cgroup':
                return '0::/lab/job\n'
            if name.endswith('/cpu.max'):
                return quota.get(name, 'max 100000')
            cpu = int(name.rsplit('/cpu', 1)[1].split('/')[0])
            return '0' if name.endswith('physical_package_id') else str(cpu // 2)

        with patch('v3_data.os.sched_getaffinity', return_value=set(range(48)), create=True), \
                patch.object(Path, 'read_text', read):
            capacity = cpu_capacity()
            self.assertEqual(capacity['logical_cpus'], 48)
            self.assertEqual(capacity['physical_cores'], 24)
            self.assertEqual(recovery_workers(None, 576, capacity), 24)
            self.assertEqual(recovery_workers(None, 3, capacity), 3)
            quota['/sys/fs/cgroup/lab/cpu.max'] = '650000 100000'
            self.assertEqual(cpu_capacity()['automatic_workers'], 7)
        for invalid in (0, -1, True, 2.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                recovery_workers(invalid, 10, capacity)
        self.assertEqual(recovery_workers(32, 576, capacity), 32)

    def test_eta_uses_only_current_completed_jobs_and_reports_warmup(self):
        self.assertIn('warming up', recovery_progress(0, 23, 500, 24, now=120))
        self.assertIn('remaining 40.0 min', recovery_progress(0, 24, 480, 24, now=120))
        self.assertIn('generation complete', recovery_progress(0, 24, 0, 24, now=120))

    def test_interrupt_cancels_bounded_pool_and_retains_completed_receipts(self):
        import v3_data
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            cancelled, pool = Mock(), Mock()
            pending = [{'id': str(i)} for i in range(576)]
            attempts = [{'id': 'already-committed'}]
            with patch.object(v3_data, 'ProcessPoolExecutor', return_value=pool), \
                    patch.object(v3_data.multiprocessing, 'get_context') as context, \
                    patch.object(v3_data, 'backup_teacher_collection') as backup, \
                    patch.object(v3_data, 'wait', side_effect=KeyboardInterrupt):
                context.return_value.Event.return_value = cancelled
                with self.assertRaises(KeyboardInterrupt):
                    run_recoveries(pending, attempts, folder, 24, backup_directory='drive-fixture')
                self.assertEqual(backup.call_count, 2, 'Protect cached input at start and completed work after orderly interrupt')
                backup.assert_called_with(folder, 'drive-fixture')
            self.assertEqual(pool.submit.call_count, 24, 'An interrupt must not leave 576 queued jobs')
            cancelled.set.assert_called_once()
            pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
            self.assertEqual(json.loads((Path(folder) / 'teacher-recoveries.json').read_text())['attempts'], attempts)
            with patch.object(v3_data, '_CANCEL_RECOVERIES', cancelled):
                cancelled.is_set.return_value = True
                with self.assertRaises(InterruptedError):
                    v3_data.continuation_job({'id': 'cancel-before-opening-any-files'})

    def test_native_recoveries_are_identical_with_one_or_two_workers(self):
        # CPU-only fixture: replay a qualified game to its last two decisions.
        # This test never publishes a dataset or uses these reserved seeds for training.
        with zipfile.ZipFile(ROOT / 'evaluation/teacher-qualification-replays.zip') as archive:
            proofs = [json.loads(line) for line in archive.read('level-1-seed-91009.jsonl').splitlines()]
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(folder)
            rows = []
            with BenchmarkEngine() as engine:
                state = engine.request('reset', options={'seed': 91009, 'level': 1, 'action_version': 2})
                for proof in proofs:
                    choice = proof['choice']
                    response = {'answers': {'move': {'choice': choice, 'probabilities': {
                        d: float(d == choice) for d in state['legal_moves']}}}}
                    risks = engine.request('risks')
                    step = engine.step(choice)
                    rows.append({'request': body(state), 'response': response,
                                 'diagnostics': diagnostic_record(state, response, step, risks, 0)})
                    state = step['state']
            trace = directory / 'learner.jsonl'
            trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            options = json.loads((ROOT / 'evaluation/teacher-qualification.json').read_text())['options']
            results = []
            for workers in (1, 2):
                attempts = []
                jobs = [{'id': str(root), 'split': 'train', 'seed': 91009, 'level': 1,
                         'root_index': root, 'learner_trace': str(trace), 'learner_trace_sha256': file_hash(trace),
                         'directory': str(directory / f'workers-{workers}'), 'options': options}
                        for root in (len(rows) - 2, len(rows) - 1)]
                self.assertEqual(run_recoveries(jobs, attempts, directory, workers), [])
                result = {}
                for attempt in attempts:
                    self.assertTrue(attempt['accepted'])
                    self.assertEqual(attempt['metrics']['outcome'], 'level_cleared')
                    candidate = Path(attempt['trace']).with_name('candidates.jsonl')
                    self.assertTrue(candidate.is_file())
                    result[attempt['id']] = (attempt['metrics'], Path(attempt['trace']).read_bytes(), candidate.read_bytes())
                results.append(result)
            self.assertEqual(results[0], results[1])

    def test_cancel_during_native_recovery_closes_node_without_committing_attempt(self):
        import v3_data
        engines = []

        def open_engine():
            engine = BenchmarkEngine()
            engines.append(engine)
            return engine

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with BenchmarkEngine() as engine:
                rows = self.native_rows(engine, turns=1)
            trace = directory / 'learner.jsonl'
            trace.write_text(json.dumps(rows[0]) + '\n')
            job = {'id': 'cancelled', 'split': 'train', 'seed': 300017, 'level': 1,
                   'root_index': 0, 'learner_trace': str(trace), 'learner_trace_sha256': file_hash(trace),
                   'directory': str(directory),
                   'options': json.loads((ROOT / 'evaluation/teacher-qualification.json').read_text())['options']}
            cancelled = Mock()
            cancelled.is_set.side_effect = [False, False, True]
            with patch.object(v3_data, '_CANCEL_RECOVERIES', cancelled), \
                    patch.object(v3_data, 'BenchmarkEngine', side_effect=open_engine), self.assertRaises(InterruptedError):
                v3_data.continuation_job(job)
            self.assertEqual(len(engines), 1)
            self.assertIsNotNone(engines[0].process.poll(), 'Cancellation must not orphan the native Node simulator')
            self.assertTrue((directory / 'cancelled/teacher.jsonl').is_file())
            self.assertFalse((directory / 'cancelled/attempt.json').exists(), 'An incomplete trajectory is not a cache hit')

    def test_fresh_seeds_never_overlap_benchmarks_or_old_partitions(self):
        check_seed_splits([300017], [310019])
        for train, development in [([50021], [310019]), ([300017], [300017]),
                                   ([300017, 300017], [310019]), ([], [310019]),
                                   ([201012], [310019])]:
            with self.subTest(train=train, development=development), self.assertRaises(ValueError):
                check_seed_splits(train, development)

    def test_collection_requires_the_completed_v2_adapter(self):
        identity = {'checkpoint_name': 'kev-4b-pacman-native-v2', 'base': BASE_MODEL,
                    'stage': 'pacman', 'pacman_fine_tuned': True, 'optimizer_steps': 762,
                    'adapter_sha256': 'a' * 64, 'checkpoint_sha256': 'b' * 64}
        require_v2_identity(identity)
        for key, value in [('checkpoint_name', 'kev-4b-skills'), ('stage', 'skills'),
                           ('pacman_fine_tuned', False), ('optimizer_steps', 0), ('adapter_sha256', '')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_v2_identity({**identity, key: value})

    def test_balancing_preserves_hard_safe_reverse_targets_and_deduplicates(self):
        records = [self.candidate(i) for i in range(20)]
        corrections = [self.candidate(100 + i, 'learner_visited', True, True) for i in range(5)]
        records += corrections + [copy.deepcopy(corrections[0])]
        selected, coverage = balanced_select(records, 10,
            {'learner_visited': 4, 'learner_disagreement': 4, 'critical_reverse': 4}, 7)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({r['state']['turn'] for r in selected}), 10)
        self.assertGreaterEqual(coverage['critical_reverse'], 4)
        for record in selected:
            if record['_meta']['origin'] == 'learner_visited':
                self.assertEqual(record['questions']['move']['label'], OPPOSITE[record['state']['player']['heading']])
        with self.assertRaisesRegex(ValueError, 'Coverage is insufficient'):
            balanced_select(records, 10, {'critical_reverse': 6}, 7)

    def test_root_selection_includes_earlier_trap_decisions_and_respects_life_boundary(self):
        records = []
        for i in range(40):
            state = copy.deepcopy(self.state)
            state.update(turn=i, life_epoch=0)
            records.append({'request': body(state), 'diagnostics': {
                'choice': state['legal_moves'][0], 'life_index': 0 if i < 20 else 1,
                'immediate_counterfactuals': {d: {'life_lost': False} for d in state['legal_moves']}}})
        last = records[30]['diagnostics']
        last['immediate_counterfactuals'][last['choice']]['life_lost'] = True
        self.assertTrue(candidate_flags(records[30])['avoidable_fatal'])
        roots = select_roots(records)
        self.assertIn(30, roots)
        self.assertTrue({22, 26, 29}.issubset(roots))
        self.assertNotIn(14, roots, 'A previous life is not an earlier decision in this trap')

    def native_rows(self, engine, turns=40):
        state = engine.request('reset', options={'seed': 300017, 'level': 1, 'action_version': 2})
        rows, life = [], 0
        for _ in range(turns):
            move = teacher(state)
            response = {'answers': {'move': {'choice': move, 'probabilities': {
                d: float(d == move) for d in state['legal_moves']}}}}
            risks = engine.request('risks')
            step = engine.step(move)
            rows.append({'request': body(state), 'response': response,
                         'diagnostics': diagnostic_record(state, response, step, risks, life)})
            state = step['state']
            if step['outcome'] == 'life_lost':
                continuation = engine.request('continue')
                if continuation['status'] != 'playing':
                    break
                state = continuation['state']
                life += 1
            elif step['outcome']:
                break
        return rows

    def test_native_learner_prefix_replays_exactly_and_rejects_a_changed_ghost(self):
        with BenchmarkEngine() as actual:
            rows = self.native_rows(actual)
        with BenchmarkEngine() as replay:
            replay.request('reset', options={'seed': 300017, 'level': 1, 'action_version': 2})
            replay_prefix(replay, rows, len(rows) - 1)
            self.assertEqual(replay.request('observe'), rows[-1]['request']['state'])
        corrupted = copy.deepcopy(rows)
        corrupted[4]['request']['state']['ghosts'][0]['heading'] = 'up'
        with BenchmarkEngine() as replay, self.assertRaisesRegex(ValueError, 'replay state differs'):
            replay.request('reset', options={'seed': 300017, 'level': 1, 'action_version': 2})
            replay_prefix(replay, corrupted, len(rows) - 1)

    def test_new_stage_initializes_v2_weights_but_does_not_resume_v2_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            v2, v3 = Path(folder) / 'checkpoints/kev-4b-pacman-native-v2', Path(folder) / 'checkpoints/kev-4b-pacman-native-v3'
            v2.mkdir(parents=True)
            for name in ['head.pt', 'adapter_config.json', 'adapter_model.safetensors']:
                (v2 / name).write_bytes(b'fixture')
            meta = {'base': BASE_MODEL, 'base_revision': BASE_REVISION, 'lora': 16,
                    'head_dim': 256, 'option_isolation': False, 'special_embeddings': False,
                    'weights_dtype': 'fp32', 'weights': 'lora'}
            with patch('cloud_runtime.subprocess.check_output', return_value=json.dumps(meta)), \
                    patch.object(runtime, '_train', return_value=v3) as train:
                runtime.finetune(Path(folder) / 'data/pacman-native-v3-train.jsonl', v3, init_from=v2)
            command = train.call_args.args[0]
            self.assertEqual(command[command.index('--init_from') + 1], str(v2))
            self.assertEqual(command[command.index('--out') + 1], str(v3.resolve()))
            self.assertNotIn('--resume', command)
            self.assertFalse(train.call_args.kwargs['resume'])
            for key, value in RECIPE.items():
                self.assertEqual(command[command.index('--' + key) + 1], str(value))

    def test_v3_drive_restore_includes_exact_data_development_and_native_proof(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace = Path(folder) / 'lab'
            output = workspace / 'checkpoints/kev-4b-pacman-native-v3'
            output.mkdir(parents=True)
            (output / 'run-evidence.json').write_text('{"stage":"pacman"}')
            (output / 'training_metrics.json').write_text('{"optimizer_steps":762}')
            (output / 'head.pt').write_bytes(b'head')
            data = workspace / 'data'
            data.mkdir()
            contents = {'pacman-native-v3-train.jsonl': b'train\n',
                        'pacman-native-v3-development.jsonl': b'dev\n',
                        'pacman-native-v3-replays.zip': b'native proof'}
            for name, content in contents.items():
                (data / name).write_bytes(content)
            manifest = data / 'pacman-native-v3-manifest.json'
            manifest.write_text(json.dumps({'files': {name: file_hash(data / name) for name in contents}}))
            expected_manifest = manifest.read_bytes()
            training = data / 'pacman-native-v3-train.jsonl'
            drive = Path(folder) / 'drive'
            receipt = save_backup(output, None, drive, workspace, training_data=training)
            self.assertEqual(len(receipt['training_artifacts']), 3)
            shutil.rmtree(output)
            shutil.rmtree(data)
            restore_backup(output, drive, workspace)
            for name, content in contents.items():
                self.assertEqual((data / name).read_bytes(), content)
            self.assertEqual(manifest.read_bytes(), expected_manifest)
            shutil.rmtree(output)
            (data / 'pacman-native-v3-development.jsonl').write_bytes(b'changed labels')
            with self.assertRaisesRegex(ValueError, 'changed local files'):
                restore_backup(output, drive, workspace)
            self.assertFalse(output.exists())

    def test_behavior_summary_weights_danger_opportunities_and_reached_lives(self):
        with tempfile.TemporaryDirectory() as folder:
            episodes = []
            for number, decisions in enumerate([1, 9]):
                fatal = number == 0
                choice = 'down' if fatal else 'up'
                row = {'diagnostics': {'player': {'heading': 'right'}, 'choice': choice,
                       'probabilities': {choice: .95}, 'immediate_counterfactuals': {
                           'down': {'life_lost': True}, 'left': {'life_lost': False}, 'up': {'life_lost': False}}}}
                path = Path(folder) / f'{number}.jsonl'
                path.write_text((json.dumps(row) + '\n') * decisions)
                metrics = {'decisions': decisions, 'life_losses': int(fatal),
                           'avoidable_immediate_deaths': int(fatal), 'avoidable_risk_decisions': decisions,
                           'loop_decisions': 0, 'power_pellets': 0, 'ghosts_eaten': 0,
                           'longest_no_pellet_decisions': 12,
                           'per_life': {'0': {'pellets_collected': 10 if fatal else 30}}}
                if fatal:
                    metrics['per_life']['1'] = {'pellets_collected': 6}
                episodes.append({'trace': str(path), 'metrics': metrics})
            result = behavior_summary({'episodes': episodes, 'level_clears': 0, 'capped_episodes': 0})
            self.assertEqual(result['fatal_choice_when_safe_available'], .1)
            self.assertEqual(result['totals']['avoidable_deaths_with_safe_reverse'], 1)
            self.assertEqual(result['totals']['confident_avoidable_deaths'], 1)
            self.assertEqual(result['mean_pellets_per_reached_life'], {'0': 20, '1': 6})
            self.assertEqual(result['games_reaching_each_life'], {'0': 2, '1': 1})

    def test_notebook_fresh_v3_resume_and_completed_restore_paths(self):
        import build_artifacts
        base = json.loads((ROOT / 'notebooks/pacman_kev_lab.ipynb').read_text())
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            temporary = Path(folder)
            (temporary / 'notebooks').mkdir()
            with patch.object(build_artifacts, 'ROOT', temporary):
                build_artifacts.v3_notebook(base)
            notebook = json.loads((temporary / 'notebooks/pacman_kev_v3.ipynb').read_text())
            source = next(''.join(c['source']) for c in notebook['cells'] if
                          ''.join(c['source']).startswith('# Warm-start a new v3 stage'))
            for scenario in ('fresh', 'resume', 'completed', 'wrong_parent'):
                with self.subTest(scenario=scenario):
                    v2, v3 = temporary / scenario / 'v2', temporary / scenario / 'v3'
                    runtime = Mock()
                    manifest = {'learner': {'checkpoint_sha256': 'other' if scenario == 'wrong_parent' else 'parent'}}
                    validator = Mock(return_value=manifest)
                    snapshot = Mock(return_value=None if scenario != 'resume' else 'v3-snapshot')

                    def finish(*args, **kwargs):
                        v3.mkdir(parents=True)
                        (v3 / 'training_metrics.json').write_text(json.dumps({
                            'optimizer_steps': 762, 'records_seen': 6096, 'requested_records': 6096}))
                        (v3 / 'run-evidence.json').write_text(json.dumps({'parent_checkpoint_sha256': 'parent'}))

                    runtime.finetune.side_effect = finish
                    if scenario == 'completed':
                        finish()
                    training = temporary / 'data/pacman-native-v3-train.jsonl'
                    namespace = dict(Path=Path, json=json, V2=v2, V3=v3,
                                     V3_DATA=temporary / 'data', V3_TRAINING=training,
                                     QUALIFICATION=temporary / 'qualification.json', runtime=runtime,
                                     validate_dataset=validator, latest_snapshot=snapshot,
                                     checkpoint_fingerprint=lambda path: 'parent')
                    if scenario == 'wrong_parent':
                        with self.assertRaisesRegex(AssertionError, 'different v2 weights'):
                            exec(compile(source, '<v3 training cell>', 'exec'), namespace)
                        runtime.finetune.assert_not_called()
                    else:
                        exec(compile(source, '<v3 training cell>', 'exec'), namespace)
                        if scenario == 'completed':
                            runtime.finetune.assert_not_called()
                            validator.assert_not_called()
                        else:
                            runtime.finetune.assert_called_once_with(training, v3, init_from=v2,
                                steps=0, resume=scenario == 'resume')
                            snapshot.assert_called_once_with(Path(str(v3) + '-recovery'))


if __name__ == '__main__':
    unittest.main()
