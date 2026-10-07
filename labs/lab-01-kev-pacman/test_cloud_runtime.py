"""Check hardware admission and the published-to-domain checkpoint transition."""
import hashlib
import io
import json
import math
from pathlib import Path
from itertools import product
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

from cloud_runtime import CloudRuntime, validate_gpu, BASE_MODEL, BASE_REVISION
from training_stages import backup_checkpoint, inspect_checkpoint, specifications

ROOT = Path(__file__).resolve().parent


def checkpoint_files(folder):
    """Placeholder files for command tests that mock the native metadata reader."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('head.pt', 'adapter_config.json', 'adapter_model.safetensors'):
        (folder / name).write_bytes(b'fixture')


class CloudRuntimeTests(unittest.TestCase):
    def test_explicit_correction_recipe_changes_replay_without_changing_architecture(self):
        from planner_data import RECIPE
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            inherited = ['python', '-m', 'kev.train', '--lora', '16', '--head_dim', '256',
                         '--batch', '1', '--accum', '8', '--replay', '2000']
            v4 = {**RECIPE, 'replay': 304}
            with patch.object(runtime, 'training_command', side_effect=lambda *args: list(inherited)):
                old = runtime.finetuning_command('data', 'v3', 'v2')
                new = runtime.finetuning_command('data', 'v4', 'v3', recipe=v4)
            self.assertEqual(old[old.index('--replay') + 1], '2000')
            self.assertEqual(new[new.index('--replay') + 1], '304')
            for flag, value in (('--lora', '16'), ('--head_dim', '256'), ('--batch', '4'), ('--accum', '2')):
                self.assertEqual(new[new.index(flag) + 1], value)
            with self.assertRaises(ValueError):
                runtime.finetuning_command('data', 'v4', 'v3', recipe={**v4, 'lora': 32})
            with self.assertRaises(ValueError):
                runtime.finetuning_command('data', 'v4', 'v3', recipe={**v4, 'accum': 8})

    def test_serving_identity_hashes_native_checkpoint_and_tracks_finetuned_stage(self):
        for stage in ('skills', 'pacman'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder)
                source = runtime.workspace / ('kev-4b-skills' if stage == 'skills' else 'kev-4b-pacman-arcade')
                checkpoint_files(source)
                (source / 'training_config.json').write_text(json.dumps({'args': {'base_revision': BASE_REVISION}}))
                (source / 'training_metrics.json').write_text(json.dumps({'optimizer_steps': 1915 if stage == 'skills' else 16}))
                (source / 'run-evidence.json').write_text(json.dumps({'stage': stage}))
                card = {'name': 'kev-latest', 'run': str(source), 'base': BASE_MODEL, 'lora': 16,
                        'dtype': 'torch.bfloat16', 'temperature': 1}
                response = lambda *args, **kwargs: io.StringIO(json.dumps({'models': [card]}))
                process = Mock(); process.poll.return_value = None
                with patch('cloud_runtime.subprocess.Popen', return_value=process), \
                        patch('cloud_runtime.urlopen', side_effect=response), patch('cloud_runtime.print'):
                    runtime.start(source)
                    info = runtime.active_model_info()
                    self.assertEqual(info['checkpoint'], str(source))
                    self.assertEqual(info['stage'], stage)
                    self.assertEqual(info['lora_rank'], 16)
                    self.assertEqual(info['pacman_fine_tuned'], stage == 'pacman')
                    self.assertEqual(info['adapter_sha256'], hashlib.sha256(b'fixture').hexdigest())
                    self.assertEqual(len(info['checkpoint_sha256']), 64)
                    self.assertEqual(json.loads((runtime.workspace / 'active-checkpoint.json').read_text()), info)
                    card['run'] = '/other/checkpoint'
                    with self.assertRaisesRegex(RuntimeError, 'changed outside'):
                        runtime.active_model_info()
                    runtime.stop()

    def test_missing_or_wrong_serving_checkpoint_never_gets_labelled_as_requested(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            source = runtime.workspace / 'kev-4b-skills'
            with patch('cloud_runtime.subprocess.Popen') as launch:
                with self.assertRaisesRegex(RuntimeError, 'no pointer head'):
                    runtime.start(source)
                launch.assert_not_called()
            checkpoint_files(source)
            card = {'name': 'kev-latest', 'run': 'runs/smoke'}
            process = Mock(); process.poll.return_value = None
            with patch('cloud_runtime.subprocess.Popen', return_value=process), \
                    patch('cloud_runtime.urlopen', return_value=io.StringIO(json.dumps({'models': [card]}))):
                with self.assertRaisesRegex(RuntimeError, 'instead of requested'):
                    runtime.start(source)
            self.assertIsNone(runtime.active_checkpoint)
            process.terminate.assert_called_once()

    def test_missing_intermediate_data_is_verified_before_training_launch(self):
        content = b'{"record":1}\n'
        spec = {'data': 'evals/documents-v1/train.jsonl', 'records': 1,
                'sha256': hashlib.sha256(content).hexdigest()}
        for mode in ('download', 'cached', 'corrupt', 'download_failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder)
                data = runtime.repo / spec['data']
                output = Path(folder) / 'documents'
                if mode in ('cached', 'corrupt'):
                    data.parent.mkdir(parents=True)
                    data.write_bytes(content if mode == 'cached' else b'changed')
                def prepare():
                    if mode == 'download_failure':
                        raise RuntimeError('dataset download failed')
                    data.parent.mkdir(parents=True)
                    data.write_bytes(content)
                def launch(*args, **kwargs):
                    self.assertEqual(data.read_bytes(), content)
                    return output
                with patch('cloud_runtime.specifications', return_value={'documents': spec}), \
                        patch.object(runtime, 'intermediate_command', return_value=['command']), \
                        patch.object(runtime, 'prepare_intermediate_data', side_effect=prepare) as download, \
                        patch.object(runtime, '_train', side_effect=launch) as train, patch('cloud_runtime.print'):
                    if mode in ('corrupt', 'download_failure'):
                        with self.assertRaises((ValueError, RuntimeError)):
                            runtime.intermediate('documents', output, 'dates')
                        train.assert_not_called()
                    else:
                        self.assertEqual(runtime.intermediate('documents', output, 'dates'), output)
                        train.assert_called_once()
                    self.assertEqual(download.call_count, int(mode in ('download', 'download_failure')))
                self.assertFalse(output.exists())

    def test_retry_preserves_empty_unstarted_output_and_requires_no_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            output = Path(folder) / 'documents'
            output.mkdir()
            runtime.training_preflight = {'result': 'bindings_verified'}
            (runtime.workspace / 'optimized-training-preflight.json').write_text(json.dumps(runtime.training_preflight))
            def train(*args, **kwargs):
                self.assertFalse(output.exists())
                self.assertNotIn('LAB_RESUME_FROM', kwargs['env'])
                output.mkdir()
                (output / 'training_metrics.json').write_text(json.dumps({'optimizer_steps': 1}))
            with patch('cloud_runtime.stream_training', side_effect=train), patch('cloud_runtime.print'):
                runtime._train(['python', '-m', 'kev.train', '--out', str(output)], output, 180, 'documents')
            preserved = list(output.parent.glob('documents-empty-attempt-*'))
            self.assertEqual(len(preserved), 1)
            self.assertEqual(list(preserved[0].iterdir()), [])
            self.assertIsNone(json.loads((output / 'run-evidence.json').read_text())['resumed_from'])

    def test_empty_output_with_recovery_directory_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            output = Path(folder) / 'documents'
            output.mkdir()
            recovery = Path(str(output) + '-recovery')
            recovery.mkdir()
            marker = recovery / 'incomplete-save'
            marker.write_bytes(b'keep')
            with patch('cloud_runtime.stream_training') as train:
                with self.assertRaisesRegex(RuntimeError, 'Choose a new checkpoint'):
                    runtime._train([], output, 180, 'documents')
                train.assert_not_called()
            self.assertTrue(output.is_dir())
            self.assertEqual(marker.read_bytes(), b'keep')
            self.assertFalse(list(output.parent.glob('documents-empty-attempt-*')))

    def test_notebook_continues_from_completed_dates_without_earlier_training_cells(self):
        notebook = json.loads((ROOT / 'notebooks/pacman_kev_lab.ipynb').read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        selected = [next(source for source in sources if source.startswith(prefix)) for prefix in
                    ('DOCUMENTS =', 'SKILLS =', '# Load completed Skills checkpoint')]
        meta = {'base': BASE_MODEL, 'base_revision': BASE_REVISION, 'lora': 16,
                'head_dim': 256, 'option_isolation': False, 'special_embeddings': False,
                'weights_dtype': 'fp32', 'weights': 'lora'}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(Path(folder) / 'lab')
            spec = specifications()
            for stage in spec.values():
                content = b'{"fixture":true}\n' * stage['records']
                data = runtime.repo / stage['data']
                data.parent.mkdir(parents=True, exist_ok=True)
                data.write_bytes(content)
                stage['sha256'] = hashlib.sha256(content).hexdigest()
            runtime.training_preflight = {'result': 'bindings_verified'}
            (runtime.workspace / 'optimized-training-preflight.json').write_text(json.dumps(runtime.training_preflight))
            root = runtime.workspace / 'checkpoints'
            dates = root / 'kev-4b-dates'
            def completed(stage, output, parent):
                checkpoint_files(output)
                args = {**runtime.selected_recipe(spec[stage]['args']), 'max_steps': 0}
                requests = spec[stage]['records'] + spec[stage]['replay']
                metrics = {'optimizer_steps': math.ceil(requests / (args['batch'] * args['accum'])),
                           'requested_records': requests, 'records_seen': requests,
                           'peak_device_bytes': 0, 'world_size': 1}
                (output / 'training_config.json').write_text(json.dumps({'args': args, 'init_source': str(parent)}))
                (output / 'training_metrics.json').write_text(json.dumps(metrics))
            completed('dates', dates, root / 'kev-4b-initial')
            (dates / 'run-evidence.json').write_text(json.dumps({'stage': 'dates'}))
            original_date_files = {file.name: file.read_bytes() for file in dates.iterdir()}
            namespace = {'runtime': runtime, 'LAB_DIR': runtime.workspace, 'CHECKPOINT_ROOT': root,
                         'STAGE_OWNERS': {}, 'Path': Path, 'json': json,
                         'specifications': lambda: spec, 'inspect_checkpoint': inspect_checkpoint,
                         'backup_checkpoint': backup_checkpoint, 'print': lambda *args, **kwargs: None}
            def train(command, **kwargs):
                flags = dict(zip(command[3::2], command[4::2]))
                completed(kwargs['stage'], Path(flags['--out']), Path(flags['--init_from']))
            with patch('cloud_runtime.specifications', return_value=spec), \
                    patch('cloud_runtime.subprocess.check_output', return_value=json.dumps(meta)), \
                    patch('cloud_runtime.stream_training', side_effect=train) as launches, \
                    patch.object(runtime, 'start', return_value={'data': []}) as serve:
                for source in selected:
                    exec(compile(source, '<notebook dates-only continuation>', 'exec'), namespace)
            self.assertEqual([call.kwargs['stage'] for call in launches.call_args_list], ['documents', 'skills'])
            self.assertEqual(namespace['missing_stage_archives'], ['initial'])
            self.assertIsNone(namespace['initial_metrics'])
            self.assertEqual(namespace['stage_metrics']['dates']['optimizer_steps'], 429)
            self.assertEqual(namespace['stage_metrics']['documents']['optimizer_steps'], 903)
            self.assertEqual(namespace['stage_metrics']['skills']['optimizer_steps'], 1915)
            self.assertEqual(namespace['STAGE_OWNERS'], dict.fromkeys(('dates', 'documents', 'skills'), 'learner'))
            serve.assert_called_once_with(root / 'kev-4b-skills')
            self.assertEqual(original_date_files, {file.name: file.read_bytes() for file in dates.iterdir()})
            self.assertFalse(Path(str(dates) + '-recovery').exists())

    def test_notebook_starts_cps_from_skills_only_without_training_or_recovery_imports(self):
        notebook = json.loads((ROOT / 'notebooks/pacman_kev_lab.ipynb').read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'
                      and ''.join(cell['source']).startswith('# Load completed Skills checkpoint'))
        for mode in ('learner', 'instructor', 'partial', 'missing_weights', 'wrong_stage'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder)
                skills = runtime.workspace / 'checkpoints/kev-4b-skills'
                checkpoint_files(skills)
                (skills / 'training_config.json').write_text(json.dumps({'args': {'max_steps': 0}}))
                (skills / 'training_metrics.json').write_text(json.dumps({
                    'optimizer_steps': 1000 if mode == 'partial' else 1915,
                    'requested_records': 15320}))
                (skills / 'run-evidence.json').write_text(json.dumps({'stage': 'dates' if mode == 'wrong_stage' else 'skills'}))
                if mode == 'missing_weights':
                    (skills / 'head.pt').unlink()
                original = {file.name: file.read_bytes() for file in skills.iterdir()}
                namespace = {'LAB_DIR': runtime.workspace, 'runtime': runtime, 'print': lambda *args, **kwargs: None}
                if mode == 'instructor':
                    namespace['STAGE_OWNERS'] = {'skills': 'instructor'}
                with patch.object(runtime, 'start', return_value={'data': []}) as serve, \
                        patch.object(runtime, 'intermediate') as train:
                    if mode in ('partial', 'missing_weights', 'wrong_stage'):
                        with self.assertRaises((ValueError, RuntimeError)):
                            exec(compile(source, '<notebook skills-only startup>', 'exec'), namespace)
                        serve.assert_not_called()
                    else:
                        exec(compile(source, '<notebook skills-only startup>', 'exec'), namespace)
                        serve.assert_called_once_with(skills)
                        self.assertEqual(namespace['STAGE_OWNERS']['skills'], mode)
                        self.assertEqual(namespace['missing_stage_archives'], ['initial', 'dates', 'documents'])
                        self.assertIsNone(namespace['initial_config'])
                        self.assertIsNone(namespace['dates_metrics'])
                        self.assertEqual(namespace['skills_metrics']['optimizer_steps'], 1915)
                    train.assert_not_called()
                self.assertEqual(original, {file.name: file.read_bytes() for file in skills.iterdir()})
                self.assertNotIn('latest_snapshot', namespace)

    def test_missing_or_partial_parent_explains_stage_one_prerequisite(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            initial = Path(folder) / 'checkpoints/kev-4b-initial'
            with patch('cloud_runtime.subprocess.check_output') as reader, patch('cloud_runtime.stream_training') as train:
                with self.assertRaisesRegex(RuntimeError, 'Run Stage 1 .* to completion'):
                    runtime.intermediate('dates', 'dates', initial)
                initial.mkdir(parents=True)
                recovery = Path(str(initial) + '-recovery')
                snapshot = recovery / 'step-0000900-fixture'
                snapshot.mkdir(parents=True)
                (snapshot / 'complete.json').write_text(json.dumps({'step': 900}))
                (recovery / 'latest.json').write_text(json.dumps({'directory': snapshot.name}))
                with self.assertRaisesRegex(RuntimeError, 'Recovery is available at optimizer step 900'):
                    runtime.intermediate('dates', 'dates', initial)
                reader.assert_not_called()
                train.assert_not_called()

    def test_checkpoint_reader_failure_exposes_actual_subprocess_stderr(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            runtime.python = Path(sys.executable)
            package = runtime.repo / 'kev'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('')
            (package / 'checkpoint.py').write_text("class Checkpoint:\n    def __init__(self, path):\n        raise ValueError('checkpoint metadata unreadable')\n")
            initial = Path(folder) / 'checkpoints/kev-4b-initial'
            checkpoint_files(initial)
            with self.assertRaisesRegex(RuntimeError, 'ValueError: checkpoint metadata unreadable') as raised:
                runtime.intermediate_command('dates', 'dates', initial)
            self.assertIn(str(initial), str(raised.exception))
            self.assertIn('Traceback', str(raised.exception))

    def test_explicit_resume_without_snapshot_gives_fresh_run_instruction(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            with patch('cloud_runtime.stream_training') as train:
                with self.assertRaisesRegex(RuntimeError, 'set RESUME_CHECKPOINT=None inside the Stage 1 cell'):
                    runtime._train([], Path(folder) / 'initial', 180, 'initial', resume_from='latest')
                train.assert_not_called()

    def test_notebook_initial_uses_4x2_for_fresh_resumed_or_completed_checkpoint(self):
        notebook = json.loads((ROOT / 'notebooks/pacman_kev_lab.ipynb').read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith('INITIAL ='))
        code = compile(source, '<notebook initial resume>', 'exec')
        previous_choices = (None, {'batch': 1, 'accum': 8, 'row_budget': 2048},
                            {'batch': 2, 'accum': 4, 'row_budget': 4096})
        for previous, mode in product(previous_choices, ('fresh', 'resume', 'complete')):
            with self.subTest(previous=previous, mode=mode), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder, initial_execution=previous)
                (runtime.repo / 'experiments').mkdir(parents=True)
                shutil.copy2(ROOT / 'vendor/kev/experiments/q35-4b-s23.json',
                             runtime.repo / 'experiments/q35-4b-s23.json')
                output = Path(folder) / 'checkpoints/kev-4b-initial'
                if mode != 'fresh':
                    output.mkdir(parents=True)
                    snapshot = Path(str(output) + '-recovery') / 'step-0000900-fixture'
                    snapshot.mkdir(parents=True)
                    (snapshot / 'complete.json').write_text(json.dumps({'step': 900}))
                    (snapshot.parent / 'latest.json').write_text(json.dumps({'directory': snapshot.name}))
                    if mode == 'complete':
                        (output / 'run-evidence.json').write_text('{}')
                namespace = {'runtime': runtime, 'CHECKPOINT_ROOT': output.parent, 'STAGE_OWNERS': {},
                             'json': json, 'Path': Path, 'print': lambda *args, **kwargs: None,
                             'inspect_checkpoint': lambda *args, **kwargs: ({}, {})}
                with patch.object(runtime, '_train', return_value=output) as train:
                    exec(code, namespace)
                if mode == 'complete':
                    train.assert_not_called()
                    command = runtime.pretraining_command(output)
                else:
                    train.assert_called_once()
                    command = train.call_args.args[0]
                    self.assertEqual(train.call_args.kwargs['resume_from'], 'latest' if mode == 'resume' else None)
                    self.assertEqual(train.call_args.kwargs['allow_execution_change'], mode == 'resume')
                flags = dict(zip(command[3::2], command[4::2]))
                self.assertEqual([flags['--' + key] for key in ('batch', 'accum', 'row_budget')], ['4', '2', '0'])
                self.assertEqual(flags['--out'], str(output.resolve()))
                self.assertEqual(flags['--checkpointing'], '1')
                self.assertEqual(flags['--lr'], '5e-05')
                self.assertEqual(runtime.selected_initial_recipe({})['batch'], 4)
                self.assertEqual(runtime.selected_recipe({})['batch'], 1)

    def test_initial_execution_changes_only_batching_and_keeps_published_training_recipe(self):
        recipe_file = ROOT / 'vendor/kev/experiments/q35-4b-s23.json'
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, initial_execution={'batch': 4, 'accum': 2, 'row_budget': 8192})
            (runtime.repo / 'experiments').mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / 'experiments/q35-4b-s23.json')
            flags = dict(zip(runtime.pretraining_command('initial')[3::2], runtime.pretraining_command('initial')[4::2]))
            self.assertEqual([flags['--' + key] for key in ('batch', 'accum', 'row_budget')], ['4', '2', '8192'])
            self.assertEqual(flags['--lr'], '5e-05')
            self.assertEqual(flags['--epochs'], '2')
            self.assertEqual(flags['--checkpointing'], '1')
            self.assertEqual(runtime.selected_initial_recipe({'batch': 4, 'accum': 2})['row_budget'], 8192)
            self.assertEqual(runtime.selected_recipe({'batch': 2, 'accum': 4})['batch'], 1)
            for settings in ({'batch': 4, 'accum': 4, 'row_budget': 0}, {'batch': 4, 'accum': 2, 'row_budget': -1},
                             {'batch': 4, 'accum': 2, 'row_budget': 0, 'lr': 0.01}):
                with self.assertRaises(ValueError):
                    CloudRuntime(folder, initial_execution=settings)
            with self.assertRaisesRegex(ValueError, 'restricted'):
                runtime.pretrain('initial', allow_execution_change=True)

    def test_resume_uses_latest_or_exact_selected_snapshot_and_preserves_existing_files(self):
        for selection in ('latest', 'chosen'):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder)
                output = Path(folder).resolve() / 'checkpoints/kev-4b-initial'
                output.mkdir(parents=True)
                recovery = Path(str(output) + '-recovery')
                for step in (900, 1000):
                    snapshot = recovery / f'step-{step:07d}-fixture'
                    snapshot.mkdir(parents=True)
                    (snapshot / 'complete.json').write_text(json.dumps({'step': step, 'steps': 3144}))
                    (snapshot / 'recovery.pt').write_bytes(b'fixture; no tensor restore in this selection test')
                (recovery / 'latest.json').write_text(json.dumps({'directory': 'step-0001000-fixture'}))
                chosen = recovery / 'step-0000900-fixture'
                requested = 'latest' if selection == 'latest' else chosen
                runtime.training_preflight = {'result': 'bindings_verified'}
                (Path(folder) / 'optimized-training-preflight.json').write_text(json.dumps(runtime.training_preflight))
                command = ['python', '-m', 'kev.train', '--out', str(output)]
                def training(*args, **kwargs):
                    (output / 'training_metrics.json').write_text(json.dumps({'optimizer_steps': 3144}))
                with patch('cloud_runtime.stream_training', side_effect=training) as stream, patch('cloud_runtime.print'):
                    runtime._train(command, output, 180, 'initial', resume_from=requested, allow_execution_change=True)
                expected = recovery / ('step-0001000-fixture' if selection == 'latest' else 'step-0000900-fixture')
                self.assertEqual(stream.call_args.kwargs['env']['LAB_RESUME_FROM'], str(expected))
                self.assertEqual(stream.call_args.kwargs['env']['LAB_ALLOW_EXECUTION_CHANGE'], '1')
                evidence = json.loads((output / 'run-evidence.json').read_text())
                self.assertEqual(evidence['resumed_from'], str(expected))
                self.assertTrue((chosen / 'recovery.pt').is_file())
                self.assertEqual(json.loads((recovery / 'latest.json').read_text())['directory'], 'step-0001000-fixture')

    def test_requested_missing_incomplete_or_unrelated_snapshot_never_starts_training(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            output = Path(folder).resolve() / 'initial'
            output.mkdir()
            recovery = Path(str(output) + '-recovery')
            incomplete = recovery / 'step-0000900-fixture'
            incomplete.mkdir(parents=True)
            unrelated = Path(folder) / 'timing-trial'
            unrelated.mkdir()
            (unrelated / 'complete.json').write_text(json.dumps({'step': 900}))
            with patch('cloud_runtime.stream_training') as stream:
                for selection in ('latest', incomplete, unrelated):
                    with self.subTest(selection=selection), self.assertRaises((RuntimeError, ValueError)):
                        runtime._train([], output, 180, 'initial', resume_from=selection)
                (incomplete / 'complete.json').write_text(json.dumps({'step': 900}))
                with self.assertRaisesRegex(ValueError, 'missing recovery.pt'):
                    runtime._train([], output, 180, 'initial', resume_from=incomplete)
                stream.assert_not_called()
            self.assertFalse((output / 'training_metrics.json').exists())

    def test_prepare_uses_fast_bindings_by_default_and_training_check_only_on_request(self):
        for full_check in (False, True):
            with self.subTest(full_check=full_check), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder)
                def audit(command, **kwargs):
                    Path(command[-1]).write_text(json.dumps({"suite_records": 12576}))
                def check(command, **kwargs):
                    report = Path(command[command.index("--report") + 1])
                    result = "bindings_verified" if "--bindings_only" in command else "passed"
                    report.write_text(json.dumps({"result": result}))
                with patch("cloud_runtime.subprocess.run", side_effect=audit), \
                        patch("cloud_runtime.stream_training", side_effect=check) as stream:
                    if full_check:
                        result = runtime.prepare_training(run_training_preflight=True)
                    else:
                        result = runtime.prepare_training()
                command = stream.call_args.args[0]
                self.assertEqual("--bindings_only" in command, not full_check)
                self.assertEqual(stream.call_args.kwargs["timeout_seconds"], 1200 if full_check else 120)
                self.assertFalse(stream.call_args.kwargs["require_metrics"])
                self.assertEqual(result["suite_records"], 12576)
                self.assertEqual(runtime.training_preflight["result"], "passed" if full_check else "bindings_verified")

    def test_real_training_accepts_verified_bindings_and_records_unrun_separate_check(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            output = Path(folder) / "initial"
            command = ["python", "-m", "kev.train", "--out", str(output)]
            with self.assertRaisesRegex(RuntimeError, "optimized bindings are required"):
                runtime._train(command, output, 90, "initial")
            runtime.training_preflight = {"result": "bindings_verified", "cuda_loss_backward": "not_run", "optimizer_steps": 0}
            (Path(folder) / "optimized-training-preflight.json").write_text(json.dumps(runtime.training_preflight))
            def train(*args, **kwargs):
                output.mkdir()
                (output / "training_metrics.json").write_text(json.dumps({"optimizer_steps": 1}))
            with patch("cloud_runtime.stream_training", side_effect=train):
                runtime._train(command, output, 90, "initial")
            evidence = json.loads((output / "run-evidence.json").read_text())
            self.assertEqual(evidence["optimized_training_preflight"]["cuda_loss_backward"], "not_run")
            self.assertEqual(evidence["optimized_training_preflight"]["optimizer_steps"], 0)

    def test_hardware_contract_and_explicit_fallback(self):
        info = {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "memory_gib": 96,
                "compute_capability": [12, 0], "torch": "2.8.0+cu128", "cuda": "12.8",
                "bf16_supported": True, "kernel_backward": "passed"}
        self.assertEqual(validate_gpu(info), "bf16")
        t4 = {**info, "name": "Tesla T4", "compute_capability": [7, 5], "bf16_supported": False}
        with self.assertRaisesRegex(RuntimeError, "received Tesla T4"):
            validate_gpu(t4)
        with self.assertRaisesRegex(RuntimeError, "must support BF16"):
            validate_gpu(t4, allow_other_gpu=True)
        p100 = {**t4, "name": "Tesla P100", "compute_capability": [6, 0]}
        with self.assertRaisesRegex(RuntimeError, "P100/Pascal"):
            validate_gpu(p100, allow_other_gpu=True)
        with self.assertRaisesRegex(RuntimeError, "pinned PyTorch"):
            validate_gpu({**info, "cuda": "12.6"})
        with self.assertRaisesRegex(RuntimeError, "must support BF16"):
            validate_gpu({**info, "bf16_supported": False})

    def test_initial_command_matches_selected_published_experiment(self):
        recipe_file = ROOT / "vendor/kev/experiments/q35-4b-s23.json"
        recipe = json.loads(recipe_file.read_text())[0]
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, training_profile="published_reference")
            (runtime.repo / "experiments").mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / "experiments/q35-4b-s23.json")
            command = runtime.pretraining_command(Path(folder) / "initial")
            flags = dict(zip(command[3::2], command[4::2]))
            for key, value in recipe.items():
                self.assertEqual(flags["--" + key], str(value))
            self.assertEqual(flags["--suite"], "evals/v7/decision-v7")
            self.assertEqual(flags["--lora"], "16")
            self.assertEqual(flags["--weights_dtype"], "fp32")
            self.assertNotIn("--init_from", flags)
            self.assertNotIn("--max_steps", flags)
            runtime.dtype = "fp32"
            with self.assertRaisesRegex(RuntimeError, "requires a BF16"):
                runtime.pretraining_command("unused")

    def test_domain_command_uses_learner_checkpoint_and_legal_action_set(self):
        meta = {"base": BASE_MODEL, "base_revision": BASE_REVISION, "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            initial = Path(folder) / "initial"
            checkpoint_files(initial)
            with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps(meta)) as read_meta:
                command = runtime.training_command("reviewed.jsonl", "candidate", init_from=initial)
            self.assertEqual(read_meta.call_args.args[0][-1], str(initial))
        flags = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(flags["--init_from"], str(initial))
        self.assertEqual(flags["--lr"], "2e-5")
        self.assertEqual(flags["--epochs"], "2")
        self.assertEqual(flags["--batch"], "1")
        self.assertEqual(flags["--accum"], "8")
        self.assertEqual(flags["--max_steps"], "0")
        self.assertEqual(flags["--dtype"], "bf16")
        for key in ["--p_none", "--p_none_distract", "--p_distract"]:
            self.assertEqual(flags[key], "0")

    def test_intermediate_stages_keep_replay_and_distinct_parent_checkpoints(self):
        meta = {"base": BASE_MODEL, "base_revision": BASE_REVISION, "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, training_profile="published_reference")
            for stage, parent in [("dates", "initial"), ("documents", "dates"), ("skills", "documents")]:
                checkpoint_files(Path(folder) / parent)
                with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps(meta)):
                    command = runtime.intermediate_command(stage, "new-checkpoint", Path(folder) / parent)
                flags = dict(zip(command[3::2], command[4::2]))
                for key, value in specifications()[stage]["args"].items():
                    self.assertEqual(flags["--" + key], str(value))
                self.assertEqual(flags["--init_from"], str(Path(folder) / parent))
                self.assertEqual(flags["--suite"], "evals/v7/decision-v7")
                self.assertEqual(flags["--p_none_pair"], "0.25")
                self.assertEqual(flags["--p_none"], "0.1")
                self.assertEqual(flags["--max_steps"], "0")
                if stage == "dates":
                    self.assertEqual(flags["--checkpointing"], "1")
                    self.assertEqual(flags["--max_state"], "384")

    def test_memory_profile_preserves_effective_batch_and_curriculum(self):
        recipe_file = ROOT / "vendor/kev/experiments/q35-4b-s23.json"
        meta = {"base": BASE_MODEL, "base_revision": BASE_REVISION, "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            (runtime.repo / "experiments").mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / "experiments/q35-4b-s23.json")
            commands = [runtime.pretraining_command("initial")]
            checkpoint_files(runtime.repo / 'parent')
            with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps(meta)):
                commands += [runtime.intermediate_command(s, s, "parent") for s in specifications()]
                commands += [runtime.training_command("reviewed.jsonl", "domain", "parent")]
            for command in commands:
                flags = dict(zip(command[3::2], command[4::2]))
                self.assertEqual(int(flags["--batch"]) * int(flags["--accum"]), 8)
                self.assertEqual(flags["--batch"], "1")
                self.assertEqual(flags["--checkpointing"], "1")
                self.assertEqual(flags["--row_budget"], "2048")
            initial = dict(zip(commands[0][3::2], commands[0][4::2]))
            self.assertEqual(initial["--epochs"], "2")
            self.assertEqual(initial["--p_none_pair"], "0.25")
            self.assertEqual(runtime.selected_recipe({"lr": 1e-4, "batch": 8})["lr"], 1e-4)
            self.assertEqual(runtime.environment()["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_domain_rejects_incompatible_backbone(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            checkpoint_files(runtime.repo / 'old-parent')
            with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps({
                    "base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "old"})):
                with self.assertRaisesRegex(RuntimeError, "0.8B adapters are incompatible"):
                    runtime.training_command("data.jsonl", "out", "old-parent")

    def test_old_failed_run_cannot_be_resumed_without_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            output = Path(folder) / "old"
            output.mkdir()
            (output / "training_config.json").write_text('{}')
            with self.assertRaisesRegex(RuntimeError, "No resumable LoRA snapshot"):
                runtime._train([], output, 90, "initial", resume=True)
            with self.assertRaisesRegex(RuntimeError, "Choose a new"):
                runtime._train([], output, 90, "initial")


if __name__ == "__main__":
    unittest.main()
