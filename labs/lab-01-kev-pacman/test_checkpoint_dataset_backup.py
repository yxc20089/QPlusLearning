"""Storage-only checks: keep the balanced teacher proofs with the adapter."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from checkpoint_backup import file_hash, restore_backup, save_backup


class BalancedDatasetBackupTests(unittest.TestCase):
    def prepare(self, directory):
        workspace = Path(directory) / 'lab'
        output = workspace / 'checkpoints/kev-4b-pacman-native-v4'
        output.mkdir(parents=True)
        (output / 'run-evidence.json').write_text('{"stage":"pacman"}')
        (output / 'training_metrics.json').write_text('{"optimizer_steps":762}')
        (output / 'adapter_model.safetensors').write_bytes(b'storage fixture')
        folder = workspace / 'data'
        folder.mkdir()
        prefix = 'pacman-native-v4-balanced'
        training = folder / (prefix + '-train.jsonl')
        development = folder / (prefix + '-development.jsonl')
        evidence = folder / (prefix + '-evidence.zip')
        training.write_bytes(b'{"label":"left"}\n')
        development.write_bytes(b'{"label":"right"}\n')
        evidence.write_bytes(b'complete positive/negative proof fixture')
        manifest = folder / (prefix + '-manifest.json')
        manifest.write_text(json.dumps({'files': {p.name: file_hash(p) for p in (training, development)},
            'evidence': {'filename': evidence.name, 'sha256': file_hash(evidence)}}))
        return workspace, output, training, development, evidence, manifest

    def test_runtime_replacement_restores_train_dev_manifest_and_branch_evidence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            workspace, output, training, development, evidence, manifest = self.prepare(directory)
            expected = {p: p.read_bytes() for p in (training, development, evidence, manifest)}
            drive = Path(directory) / 'drive'
            receipt = save_backup(output, None, drive, workspace, training_data=training)
            self.assertEqual({Path(p['path']).name for p in receipt['training_artifacts']},
                             {development.name, evidence.name, manifest.name})
            shutil.rmtree(workspace)
            restore_backup(output, drive, workspace)
            for path, content in expected.items():
                self.assertEqual(path.read_bytes(), content)

    def test_changed_evidence_cannot_replace_acknowledged_backup(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            workspace, output, training, _, evidence, _ = self.prepare(directory)
            drive = Path(directory) / 'drive'
            save_backup(output, None, drive, workspace, training_data=training)
            pointer = drive / output.name / 'latest.json'
            acknowledged = pointer.read_bytes()
            evidence.write_bytes(b'changed native proof')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                save_backup(output, None, drive, workspace, training_data=training)
            self.assertEqual(pointer.read_bytes(), acknowledged)


if __name__ == '__main__':
    unittest.main()
