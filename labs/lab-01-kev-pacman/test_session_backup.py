"""CPU checks for preserving all runs and restoring end-of-session checkpoints."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from checkpoint_backup import backup_lab_to_drive, new_run_directory, restore_backup


class SessionBackupTests(unittest.TestCase):
    def cell(self, prefix):
        notebook = json.loads((Path(__file__).parent / 'notebooks/pacman_kev_lab.ipynb').read_text())
        return next(''.join(cell['source']) for cell in notebook['cells']
                    if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith(prefix))

    def checkpoint(self, workspace, name, stage='pacman'):
        output = workspace / 'checkpoints' / name
        output.mkdir(parents=True)
        (output / 'head.pt').write_bytes(b'head-' + name.encode())
        (output / 'adapter_model.safetensors').write_bytes(b'adapter-' + name.encode())
        (output / 'run-evidence.json').write_text(json.dumps({'stage': stage}))
        (output / 'training_metrics.json').write_text('{"optimizer_steps": 762}')
        (output / 'training_config.json').write_text('{"args": {"data": null}}')
        return output

    def test_all_checkpoint_versions_restore_and_all_recordings_survive(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, drive = Path(folder) / 'lab', Path(folder) / 'drive'
            outputs = [self.checkpoint(workspace, name) for name in
                       ('kev-4b-skills', 'kev-4b-pacman-planner-v1', 'kev-4b-pacman-native-v2')]
            (workspace / 'checkpoints/unstarted').mkdir()
            payloads = {'results/fine-tuned-player-trace.jsonl': b'old trace\n',
                        'results/comparison-one/gameplay-general/level-1-seed-7.jsonl': b'partial benchmark\n',
                        'results/interactive-v2/fine-tuned-player-trace.jsonl': b'new trace\n',
                        'logs/pacman/events.out.tfevents.test': b'tensorboard',
                        'data/reviewed.jsonl': b'labels\n', 'evaluation/replays.zip': b'replay',
                        'comparison.json': b'{"schema_version": 3}'}
            for name, content in payloads.items():
                path = workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            # Never ship model caches/environments in the recording archive.
            (workspace / 'kev').mkdir()
            (workspace / 'kev/foundation.safetensors').write_bytes(b'exclude')
            receipt = backup_lab_to_drive(workspace, drive)
            self.assertEqual(set(receipt['checkpoint_backups']), {o.name for o in outputs})
            self.assertFalse(receipt['checkpoint_errors'])
            self.assertEqual(len(receipt['skipped_checkpoints']), 1)
            archive = Path(receipt['archive'])
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), receipt['sha256'])
            with zipfile.ZipFile(archive) as saved:
                manifest = json.loads(saved.read('session-manifest.json'))
                self.assertEqual({entry['path'] for entry in manifest['files']}, set(payloads))
                for entry in manifest['files']:
                    content = saved.read(entry['path'])
                    self.assertEqual(content, payloads[entry['path']])
                    self.assertEqual(entry['sha256'], hashlib.sha256(content).hexdigest())
                    self.assertEqual(entry['bytes'], len(content))
                self.assertNotIn('kev/foundation.safetensors', saved.namelist())
            for output in outputs:
                expected = (output / 'adapter_model.safetensors').read_bytes()
                shutil.rmtree(output)
                restored = restore_backup(output, drive / 'backups', workspace)
                self.assertTrue(restored['completed_stage'])
                self.assertEqual((output / 'adapter_model.safetensors').read_bytes(), expected)

    def test_repeated_play_and_backups_do_not_overwrite_recordings(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, drive = Path(folder) / 'lab', Path(folder) / 'drive'
            first = new_run_directory(workspace / 'results', 'interactive-fine-tuned',
                                      active_checkpoint={'checkpoint': 'v1'}, seed=7)
            second = new_run_directory(workspace / 'results', 'interactive-fine-tuned',
                                       active_checkpoint={'checkpoint': 'v2'}, seed=7)
            self.assertNotEqual(first, second)
            (first / 'trace.jsonl').write_text('v1\n')
            (second / 'trace.jsonl').write_text('v2\n')
            one, two = backup_lab_to_drive(workspace, drive), backup_lab_to_drive(workspace, drive)
            self.assertNotEqual(one['archive'], two['archive'])
            self.assertTrue(Path(one['archive']).is_file())
            with zipfile.ZipFile(two['archive']) as saved:
                self.assertEqual(saved.read(str(first.relative_to(workspace) / 'trace.jsonl')), b'v1\n')
                self.assertEqual(saved.read(str(second.relative_to(workspace) / 'trace.jsonl')), b'v2\n')

    def test_unfinished_training_preserves_latest_recovery_and_custom_input_bytes(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, drive = Path(folder) / 'lab', Path(folder) / 'drive'
            output = self.checkpoint(workspace, 'kev-4b-pacman-native-v2')
            (output / 'run-evidence.json').unlink()
            data = workspace / 'data/reviewed.jsonl'
            data.parent.mkdir()
            data.write_bytes(b'{"label":"left"}\n')
            (output / 'training_config.json').write_text(json.dumps({'args': {'data': str(data)}}))
            root = Path(str(output) + '-recovery')
            for step in (100, 200):
                snapshot = root / f'step-{step:07d}-fixture'
                snapshot.mkdir(parents=True)
                content = f'optimizer/RNG/weights at {step}'.encode()
                (snapshot / 'recovery.pt').write_bytes(content)
                (snapshot / 'complete.json').write_text(json.dumps({'step': step, 'directory': snapshot.name,
                    'recovery_sha256': hashlib.sha256(content).hexdigest()}))
            (root / 'latest.json').write_text(json.dumps({'directory': snapshot.name}))
            receipt = backup_lab_to_drive(workspace, drive)
            checkpoint = receipt['checkpoint_backups'][output.name]
            self.assertEqual(checkpoint['step'], 200)
            self.assertFalse(checkpoint['completed_stage'])
            shutil.rmtree(output)
            shutil.rmtree(root)
            data.unlink()
            restore_backup(output, drive / 'backups', workspace)
            self.assertEqual((root / snapshot.name / 'recovery.pt').read_bytes(), content)
            self.assertEqual(data.read_bytes(), b'{"label":"left"}\n')

    def test_checkpoint_failure_still_preserves_partial_traces_and_reports_error(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, drive = Path(folder) / 'lab', Path(folder) / 'drive'
            output = self.checkpoint(workspace, 'kev-4b-pacman-native-v2')
            results = workspace / 'results'
            results.mkdir()
            (results / 'partial.jsonl').write_text('partial trace\n')
            copy = shutil.copyfile
            def fail_checkpoint(source, target):
                if Path(source).name.startswith('backup-'):
                    raise OSError('checkpoint copy interrupted')
                return copy(source, target)
            with patch('checkpoint_backup.shutil.copyfile', side_effect=fail_checkpoint):
                receipt = backup_lab_to_drive(workspace, drive)
            self.assertIn('checkpoint copy interrupted', receipt['checkpoint_errors'][output.name])
            with zipfile.ZipFile(receipt['archive']) as saved:
                self.assertEqual(saved.read('results/partial.jsonl'), b'partial trace\n')
            self.assertTrue((output / 'adapter_model.safetensors').is_file())

    def test_failed_session_upload_retains_previous_successful_pointer(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, drive = Path(folder) / 'lab', Path(folder) / 'drive'
            workspace.mkdir()
            previous = backup_lab_to_drive(workspace, drive)
            pointer = drive / 'session-backups/latest.json'
            expected = pointer.read_bytes()
            with patch('checkpoint_backup.shutil.copyfile', side_effect=OSError('Drive unavailable')):
                with self.assertRaisesRegex(OSError, 'Drive unavailable'):
                    backup_lab_to_drive(workspace, drive)
            self.assertEqual(pointer.read_bytes(), expected)
            self.assertTrue(Path(previous['archive']).is_file())

    def test_interrupted_notebook_comparison_preserves_baseline_and_partial_task_run(self):
        source = self.cell('# This comparison can also evaluate')
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace = Path(folder) / 'lab'
            runtime = types.SimpleNamespace(start=lambda _: None, active_model_info=lambda: {'adapter': 'fixture'})
            def interrupted_benchmark(*, model_info, trace_dir):
                trace_dir.mkdir(parents=True)
                (trace_dir / 'episode.jsonl').write_text('recorded decisions\n')
                if trace_dir.name == 'gameplay-fine-tuned':
                    raise RuntimeError('benchmark interrupted')
                return {'active_checkpoint': model_info(), 'completed': True}
            scope = {'LAB_DIR': workspace, 'CHECKPOINT_ROOT': workspace / 'checkpoints',
                     'runtime': runtime, 'new_run_directory': new_run_directory, 'json': json,
                     'benchmark_gameplay': interrupted_benchmark, 'print': lambda *args: None}
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'benchmark interrupted'):
                    exec(source, scope)
            runs = sorted((workspace / 'results').iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertTrue(json.loads((run / 'general.json').read_text())['completed'])
                for name in ('gameplay-general', 'gameplay-fine-tuned'):
                    self.assertEqual((run / name / 'episode.jsonl').read_text(), 'recorded decisions\n')
            receipt = backup_lab_to_drive(workspace, Path(folder) / 'drive')
            with zipfile.ZipFile(receipt['archive']) as saved:
                self.assertEqual(sum(name.endswith('episode.jsonl') for name in saved.namelist()), 4)

    def test_final_notebook_drive_cell_requires_no_benchmark_or_training_results(self):
        source = self.cell('# Back up all available checkpoints, trajectories and traces')
        colab = types.ModuleType('google.colab')
        colab.drive = types.SimpleNamespace(mount=lambda _: None)
        google = types.ModuleType('google')
        google.colab = colab
        receipt = {'checkpoint_errors': {}}
        scope = {'LAB_DIR': Path('/content/pacman-kev-lab'),
                 'CHECKPOINT_ROOT': Path('/content/pacman-kev-lab/checkpoints'),
                 'runtime': types.SimpleNamespace(stop=lambda: None), 'SAVE_TO_DRIVE': False,
                 'print': lambda *args: None}
        with patch.dict('sys.modules', {'google': google, 'google.colab': colab}), \
                patch('checkpoint_backup.backup_lab_to_drive', return_value=receipt) as backup:
            exec(source, scope)
        backup.assert_called_once_with(scope['LAB_DIR'],
            Path('/content/drive/MyDrive/QPlusLearning/lab-01-kev-pacman'), checkpoint_root=scope['CHECKPOINT_ROOT'])


if __name__ == '__main__':
    unittest.main()
