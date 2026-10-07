"""CPU checks for incremental teacher prework backup across runtime loss."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from checkpoint_backup import backup_teacher_collection, restore_teacher_collection, file_hash


class TeacherBackupTests(unittest.TestCase):
    def inputs(self, collection):
        collection.mkdir()
        reports = {}
        for split in ('train', 'development'):
            trace = collection / f'learner-{split}/episode.jsonl'
            trace.parent.mkdir()
            trace.write_text(json.dumps({'request': {'state': {'split': split}}}) + '\n')
            name = f'learner-{split}.json'
            (collection / name).write_text(json.dumps({'episodes': [{'trace': str(trace)}]}))
            reports[split] = name
        (collection / 'collection.json').write_text(json.dumps({'reports': reports, 'learner': {'adapter_sha256': 'v2'}}))

    def attempt(self, collection, name, accepted=True, labels=True):
        folder = collection / 'teacher-recoveries' / name
        folder.mkdir(parents=True)
        state = {'turn': 17, 'ghosts': [{'name': 'Blinky', 'heading': 'left'}]}
        digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        proof = {'state_sha256': digest, 'choice': 'up', 'diagnostics': {
            'immediate_counterfactuals': {'up': {'life_lost': False}, 'down': {'life_lost': True}}}}
        trace = folder / 'teacher.jsonl'
        trace.write_text(json.dumps(proof) + '\n')
        (folder / 'attempt.json').write_text(json.dumps({'accepted': accepted, 'trace': str(trace),
            'trace_sha256': file_hash(trace), 'metrics': {'decisions': 1}}))
        candidate = {'state': state, 'questions': {'move': {'label': 'up'}}, '_meta': {'suffix_index': 0}}
        if accepted and labels:
            (folder / 'candidates.jsonl').write_text(json.dumps(candidate) + '\n')
        return folder, candidate

    def test_incremental_restore_reuses_both_admitted_and_rejected_attempts(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            collection, drive = root / 'v3-collection', root / 'drive'
            self.inputs(collection)
            self.attempt(collection, 'admitted')
            first = backup_teacher_collection(collection, drive)
            self.assertEqual(len(first['attempts']), 1)
            self.attempt(collection, 'rejected', accepted=False)
            unfinished = collection / 'teacher-recoveries/unfinished'
            unfinished.mkdir()
            (unfinished / 'teacher.jsonl').write_text('partial trajectory\n')
            second = backup_teacher_collection(collection, drive)
            self.assertEqual(len(second['archives']), 2)
            self.assertEqual(len(second['archives'][1]['files']), 2, 'Only the new rejected attempt is copied')
            self.assertEqual(set(second['attempts']), {'admitted', 'rejected'})
            self.assertEqual(backup_teacher_collection(collection, drive), second, 'No redundant archive on an unchanged collection')
            expected = {str(p.relative_to(collection)): p.read_bytes() for p in collection.rglob('*')
                        if p.is_file() and not p.is_relative_to(unfinished)}
            shutil.rmtree(collection)
            restored = restore_teacher_collection(collection, drive)
            self.assertEqual(restored, second)
            self.assertEqual({str(p.relative_to(collection)): p.read_bytes() for p in collection.rglob('*') if p.is_file()}, expected)
            report = json.loads((collection / 'learner-train.json').read_text())
            self.assertTrue(Path(report['episodes'][0]['trace']).is_file(), 'Original absolute learner paths must work after restore')
            (collection / 'local-work.txt').write_text('preserve')
            self.assertIsNone(restore_teacher_collection(collection, drive))
            self.assertEqual((collection / 'local-work.txt').read_text(), 'preserve')

    def test_old_receipt_is_skipped_until_all_labels_are_written(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            collection, drive = root / 'v3-collection', root / 'drive'
            self.inputs(collection)
            folder, candidate = self.attempt(collection, 'old-helper', labels=False)
            first = backup_teacher_collection(collection, drive)
            self.assertEqual(first['attempts'], {})
            (folder / 'candidates.jsonl').write_text('{"state":')
            self.assertEqual(backup_teacher_collection(collection, drive), first)
            (folder / 'candidates.jsonl').write_text(json.dumps(candidate) + '\n')
            self.assertEqual(set(backup_teacher_collection(collection, drive)['attempts']), {'old-helper'})

    def test_corrupt_archive_never_publishes_a_partial_restore(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            collection, drive = root / 'v3-collection', root / 'drive'
            self.inputs(collection)
            self.attempt(collection, 'admitted')
            receipt = backup_teacher_collection(collection, drive)
            (drive / receipt['archives'][0]['archive']).write_bytes(b'corrupted')
            shutil.rmtree(collection)
            with self.assertRaisesRegex(ValueError, 'archive checksum mismatch'):
                restore_teacher_collection(collection, drive)
            self.assertFalse(collection.exists())

    def test_changed_learner_or_teacher_evidence_preserves_previous_backup(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            collection, drive = root / 'v3-collection', root / 'drive'
            self.inputs(collection)
            folder, _ = self.attempt(collection, 'admitted')
            backup_teacher_collection(collection, drive)
            pointer = (drive / 'latest.json').read_bytes()
            (folder / 'attempt.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'receipt changed'):
                backup_teacher_collection(collection, drive)
            self.assertEqual((drive / 'latest.json').read_bytes(), pointer)
            (collection / 'learner-train/episode.jsonl').write_text('changed learner\n')
            with self.assertRaisesRegex(ValueError, 'different learner collection'):
                backup_teacher_collection(collection, drive)

    def test_failed_drive_copy_keeps_last_verified_pointer(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            collection, drive = root / 'v3-collection', root / 'drive'
            self.inputs(collection)
            backup_teacher_collection(collection, drive)
            pointer = (drive / 'latest.json').read_bytes()
            self.attempt(collection, 'admitted')
            copy = shutil.copyfile

            def bad_copy(source, destination):
                copy(source, destination)
                Path(destination).write_bytes(b'incomplete upload')

            with patch('checkpoint_backup.shutil.copyfile', side_effect=bad_copy), \
                    self.assertRaisesRegex(ValueError, 'archive checksum mismatch'):
                backup_teacher_collection(collection, drive)
            self.assertEqual((drive / 'latest.json').read_bytes(), pointer)


if __name__ == '__main__':
    unittest.main()
