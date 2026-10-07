"""Check interruption-safe data-generation backups and exact restoration."""
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from collection_backup import backup_collection, restore_collection
from checkpoint_backup import save_backup, restore_backup


class CollectionBackupTests(unittest.TestCase):
    def test_v4_checkpoint_restores_its_exact_inputs_and_development_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, drive = Path(temporary) / 'lab', Path(temporary) / 'drive'
            output, data = workspace / 'checkpoints/kev-4b-pacman-native-v4', workspace / 'data'
            output.mkdir(parents=True)
            data.mkdir()
            (output / 'run-evidence.json').write_text('{"stage":"pacman"}')
            (output / 'training_metrics.json').write_text('{"optimizer_steps":762}')
            contents = {'pacman-native-v4-train.jsonl': b'train\n',
                        'pacman-native-v4-development.jsonl': b'development\n',
                        'pacman-native-v4-replays.zip': b'verification'}
            for name, content in contents.items():
                (data / name).write_bytes(content)
            manifest = data / 'pacman-native-v4-manifest.json'
            manifest.write_text(json.dumps({'files': {name: hashlib.sha256(content).hexdigest()
                                                       for name, content in contents.items()}}))
            expected_manifest = manifest.read_bytes()
            save_backup(output, None, drive, workspace, data / 'pacman-native-v4-train.jsonl')
            shutil.rmtree(workspace)
            restore_backup(output, drive, workspace)
            for name, content in contents.items():
                self.assertEqual((data / name).read_bytes(), content)
            self.assertEqual(manifest.read_bytes(), expected_manifest)

    def test_delta_restores_latest_index_and_complete_cases_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            collection, drive = Path(temporary) / 'collection', Path(temporary) / 'drive'
            collection.mkdir()
            (collection / 'index.json').write_text('{"completed":1}')
            (collection / 'case-1.json').write_text('first')
            (collection / 'inflight.json').write_text('partial')
            first = backup_collection(collection, drive, ['index.json', 'case-1.json'])
            (collection / 'index.json').write_text('{"completed":2}')
            (collection / 'case-2.json').write_text('second')
            second = backup_collection(collection, drive, ['index.json', 'case-1.json', 'case-2.json'])
            self.assertEqual(len(second['archives']), 2)
            self.assertEqual(second['files']['case-1.json'], first['files']['case-1.json'])
            self.assertEqual(backup_collection(collection, drive, list(second['files'])), second)
            shutil.rmtree(collection)
            restore_collection(collection, drive)
            self.assertEqual(json.loads((collection / 'index.json').read_text()), {'completed': 2})
            self.assertEqual((collection / 'case-1.json').read_text(), 'first')
            self.assertEqual((collection / 'case-2.json').read_text(), 'second')
            self.assertFalse((collection / 'inflight.json').exists())
            (collection / 'index.json').write_text('local progress')
            restore_collection(collection, drive)
            self.assertEqual((collection / 'index.json').read_text(), 'local progress')

    def test_failed_upload_retains_pointer_and_corruption_never_publishes_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            collection, drive = Path(temporary) / 'collection', Path(temporary) / 'drive'
            collection.mkdir()
            (collection / 'case.json').write_text('first')
            receipt = backup_collection(collection, drive, ['case.json'])
            old_pointer = (drive / 'latest.json').read_bytes()
            (collection / 'case.json').write_text('new')
            with patch('collection_backup.shutil.copyfile', side_effect=OSError('Drive disconnected')):
                with self.assertRaises(OSError):
                    backup_collection(collection, drive, ['case.json'])
            self.assertEqual((drive / 'latest.json').read_bytes(), old_pointer)
            shutil.rmtree(collection)
            (drive / next(iter(receipt['archives']))).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restore_collection(collection, drive)
            self.assertFalse(collection.exists())

    def test_unsafe_inventory_and_mismatched_collection_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            collection, drive = Path(temporary) / 'collection', Path(temporary) / 'drive'
            collection.mkdir()
            (collection / 'case.json').write_text('first')
            for unsafe in ('../outside', '/outside', '.partial/file', 'a/../b', 'a\\b'):
                with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                    backup_collection(collection, drive, [unsafe])
            with self.assertRaises(ValueError):
                backup_collection(collection, drive)
            backup_collection(collection, drive, ['case.json'])
            with self.assertRaisesRegex(ValueError, 'different collection'):
                restore_collection(Path(temporary) / 'other', drive)


if __name__ == '__main__':
    unittest.main()
