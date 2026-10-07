"""CPU checks of automatic archives and recovery after losing the local runtime."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from checkpoint_backup import save_backup, restore_backup
from cloud_runtime import CloudRuntime
from lora_recovery import COUNTERS, LoRARecovery, latest_snapshot
from test_lora_recovery import make_state, advance, export


class CheckpointBackupTests(unittest.TestCase):
    def test_notebook_preserves_restored_reviewed_labels_without_reformatting(self):
        notebook = json.loads((Path(__file__).parent / 'notebooks/pacman_kev_lab.ipynb').read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'
                      and 'EDITS = {}' in ''.join(cell['source']))
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            data = workspace / 'data'
            data.mkdir()
            row = {'_meta': {'id': 'board-000', 'teacher': {'candidates': []}, 'equally_ranked_actions':['left']}, 'state': {'player': [1, 1]},
                   'questions': {'move': {'criteria': {'left': 'Move left'}, 'label': 'left'}}}
            # Different serialization from the notebook; byte identity matters
            # for the recovery training-data fingerprint.
            original = json.dumps(row, separators=(',', ':')).encode() + b'\n'
            reviewed = data / 'pacman-native-v2-train-reviewed.jsonl'
            reviewed.write_bytes(original)
            (workspace/'evaluation').mkdir()
            (workspace/'evaluation/teacher-qualification.json').write_text('{"qualification":{"approved_for_training":true}}')
            scope = {'LAB_DIR': workspace, 'json': json, 'manifest': {'counts': {'train': 1}, 'coverage': {}},
                     'print': lambda *args, **kwargs: None,
                     'prepare_dataset': lambda *args: {'counts': {'train': 1}, 'coverage': {}}}
            exec(source, scope)
            self.assertEqual(reviewed.read_bytes(), original)
            self.assertEqual(scope['training'][0]['questions']['move']['label'], 'left')

    def prepare(self, folder):
        workspace = Path(folder).resolve() / 'lab'
        output = workspace / 'checkpoints/kev-4b-initial'
        output.mkdir(parents=True)
        data = workspace / 'data/reviewed.jsonl'
        data.parent.mkdir()
        data.write_text('{"label": "left"}\n')
        (output / 'training_config.json').write_text(json.dumps({'args': {'data': str(data)}}))
        state = make_state()
        state['a'].out, state['a'].data = str(output), str(data)
        return workspace, output, data, state

    def test_periodic_backup_restores_inputs_and_exact_dropout_optimizer_continuation(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, output, data, state = self.prepare(folder)
            drive = Path(folder) / 'drive'
            manager = LoRARecovery(str(output) + '-recovery', every_steps=1,
                                   backup_root=drive, backup_workspace=workspace)
            for step in (1, 2, 3):
                advance(state, step)
                manager.save_if_due(state, export)
            expected_losses = advance(state, 6)
            expected = {name: value.clone() for name, value in state['model'].state_dict().items()}
            manifests = list((drive / output.name).glob('backup-*.zip.json'))
            self.assertEqual(len(manifests), 2)
            self.assertEqual(sorted(json.loads(p.read_text())['step'] for p in manifests), [2, 3])
            shutil.rmtree(workspace)  # Simulate replacing the Colab VM.
            receipt = restore_backup(output, drive, workspace)
            self.assertEqual(receipt['step'], 3)
            self.assertEqual(data.read_text(), '{"label": "left"}\n')
            resumed = make_state()
            resumed['a'].out, resumed['a'].data = str(output), str(data)
            position = LoRARecovery(resume_from=latest_snapshot(str(output) + '-recovery')).restore(resumed)
            resumed.update(zip(COUNTERS, position))
            self.assertEqual(advance(resumed, 6), expected_losses)
            self.assertEqual(resumed['sched'].state_dict(), state['sched'].state_dict())
            for name, value in expected.items():
                self.assertTrue(torch.equal(resumed['model'].state_dict()[name], value), name)

    def test_interrupted_drive_copy_retains_previous_backup_and_new_local_snapshot(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as console:
            workspace, output, data, state = self.prepare(folder)
            drive = Path(folder) / 'drive'
            manager = LoRARecovery(str(output) + '-recovery', every_steps=1,
                                   backup_root=drive, backup_workspace=workspace)
            advance(state, 1)
            manager.save_if_due(state, export)
            pointer = drive / output.name / 'latest.json'
            previous = pointer.read_bytes()
            advance(state, 2)
            with patch('checkpoint_backup.shutil.copyfile', side_effect=OSError('Drive unavailable')):
                with self.assertRaisesRegex(OSError, 'Drive unavailable'):
                    manager.save_if_due(state, export)
            self.assertEqual(pointer.read_bytes(), previous)
            local = latest_snapshot(str(output) + '-recovery')
            self.assertEqual(json.loads((local / 'complete.json').read_text())['step'], 2)
            self.assertNotIn('Drive backup complete at optimizer step 2', console.getvalue())

    def test_configure_backs_up_existing_snapshot_and_restores_without_overwriting_local_files(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, output, data, state = self.prepare(folder)
            drive = Path(folder) / 'drive'
            advance(state, 1)
            LoRARecovery(str(output) + '-recovery').save_if_due(state, export)
            runtime = CloudRuntime(workspace)
            result = runtime.configure_backup(drive, output.parent)
            self.assertEqual(result['backed_up'][0]['step'], 1)
            pointer = drive / output.name / 'latest.json'
            previous = pointer.read_bytes()
            self.assertEqual(runtime.configure_backup(drive, output.parent)['backed_up'], [])
            self.assertEqual(pointer.read_bytes(), previous)
            shutil.rmtree(workspace)
            runtime = CloudRuntime(workspace)
            result = runtime.configure_backup(drive, output.parent)
            self.assertEqual(result['restored'][0]['step'], 1)
            (output / 'marker').write_text('keep local work')
            self.assertIsNone(restore_backup(output, drive, workspace))
            self.assertEqual((output / 'marker').read_text(), 'keep local work')
            runtime.configure_backup(None, output.parent)
            self.assertIsNone(runtime.backup_root)
            self.assertNotIn('LAB_BACKUP_ROOT', runtime.environment())

    def test_training_enables_child_backups_and_archives_completed_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace = Path(folder).resolve() / 'lab'
            workspace.mkdir()
            output = workspace / 'checkpoints/kev-4b-initial'
            drive = Path(folder).resolve() / 'drive'
            runtime = CloudRuntime(workspace)
            runtime.configure_backup(drive, output.parent)
            runtime.training_preflight = {'result': 'bindings_verified'}
            (workspace / 'optimized-training-preflight.json').write_text(json.dumps(runtime.training_preflight))
            def train(*args, **kwargs):
                output.mkdir(parents=True)
                (output / 'training_config.json').write_text(json.dumps({'args': {'data': None}}))
                (output / 'adapter_model.safetensors').write_bytes(b'learned adapter')
                (output / 'head.pt').write_bytes(b'learned head')
                (output / 'training_metrics.json').write_text(json.dumps({'optimizer_steps': 3144}))
            with patch('cloud_runtime.stream_training', side_effect=train) as stream:
                runtime._train(['python', '-m', 'kev.train', '--out', str(output)], output, 180, 'initial')
            self.assertEqual(stream.call_args.kwargs['env']['LAB_BACKUP_ROOT'], str(drive))
            self.assertEqual(stream.call_args.kwargs['env']['LAB_BACKUP_WORKSPACE'], str(workspace))
            manifest = json.loads((drive / output.name / 'latest.json').read_text())
            self.assertTrue(manifest['completed_stage'])
            shutil.rmtree(workspace)
            restore_backup(output, drive, workspace)
            self.assertEqual((output / 'head.pt').read_bytes(), b'learned head')
            self.assertEqual(json.loads((output / 'run-evidence.json').read_text())['stage'], 'initial')

    def test_corrupt_archive_or_changed_labels_never_restore_over_local_work(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            workspace, output, data, state = self.prepare(folder)
            drive = Path(folder) / 'drive'
            advance(state, 1)
            snapshot = LoRARecovery(str(output) + '-recovery').save_if_due(state, export)
            receipt = save_backup(output, snapshot, drive, workspace, training_data=data)
            shutil.rmtree(output)
            shutil.rmtree(str(output) + '-recovery')
            data.write_text('{"label": "right"}\n')
            with self.assertRaisesRegex(ValueError, 'changed labels'):
                restore_backup(output, drive, workspace)
            self.assertFalse(output.exists())
            self.assertEqual(data.read_text(), '{"label": "right"}\n')
            data.unlink()
            with (drive / output.name / receipt['archive']).open('ab') as corrupted:
                corrupted.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'archive checksum mismatch'):
                restore_backup(output, drive, workspace)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
