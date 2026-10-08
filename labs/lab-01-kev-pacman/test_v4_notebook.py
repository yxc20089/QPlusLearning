"""Exercise the precomputed-data handoff and v4 training boundary without a GPU."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import build_artifacts
from behavior_data import PREFIX, RECIPE

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'c7c5c8326f74e86e161e05730db5004efdfd3b2351c6f2f9478581cfc13550b0'


def draft_notebook():
    """Render only in memory: do not pin helpers or overwrite any notebook."""
    base = json.loads((ROOT / 'notebooks/pacman_kev_lab.ipynb').read_text())
    with patch.object(Path, 'write_text', autospec=True) as write:
        build_artifacts.v4_notebook(base)
    write.assert_called_once()
    return json.loads(write.call_args.args[1])


def cell(prefix):
    return next(''.join(c['source']) for c in draft_notebook()['cells'] if c['cell_type'] == 'code'
                and ''.join(c['source']).startswith(prefix))


def complete_manifest():
    return {'dataset': PREFIX, 'files': {f'{PREFIX}-train.jsonl': 'trainhash',
            f'{PREFIX}-development.jsonl': 'devhash'},
        'expected_training_requests': 6096, 'expected_optimizer_steps': 762,
        'recipe': RECIPE, 'initialization': {'parent_checkpoint': 'kev-4b-pacman-native-v3', 'optimizer': 'fresh'},
        'counts': {'train': {'targeted': 3660, 'informative': 1220, 'replay': 912},
                   'development': {'targeted': 384, 'informative': 128}},
        'role_plan': {}, 'coverage': {}}


def write_completed_checkpoint(folder, parent_sha=PARENT_SHA, data_sha='trainhash'):
    folder.mkdir(exist_ok=True)
    (folder / 'training_metrics.json').write_text(json.dumps(
        {'optimizer_steps': 762, 'records_seen': 6096, 'requested_records': 6096}))
    (folder / 'run-evidence.json').write_text(json.dumps(
        {'stage': 'pacman', 'parent_checkpoint_sha256': parent_sha, 'training_data_sha256': data_sha}))
    args = {**RECIPE, 'lora': 16, 'lora_targets': 'all', 'head_dim': 256,
        'base': 'Qwen/Qwen3.5-4B-Base', 'base_revision': '1001bb4d826a52d1f399e183466143f4da7b741b'}
    (folder / 'training_config.json').write_text(json.dumps({'args': args}))


class V4NotebookTests(unittest.TestCase):
    def test_draft_has_no_generation_or_live_probe_and_preserves_native_comparison(self):
        notebook = draft_notebook()
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        for source in sources:
            if not source.startswith('%'):
                ast.parse(source)
        tree = ast.parse('\n'.join(s for s in sources if not s.startswith('%')))
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        for forbidden in ['harvest_scenarios', 'collect_learner', 'probe_candidates', 'verify_cases', 'build_dataset']:
            self.assertNotIn(forbidden, calls)
        games = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'benchmark_gameplay']
        self.assertEqual(len(games), 1)  # One unchanged call in the two-checkpoint loop.
        self.assertEqual({k.arg for k in games[0].keywords}, {'model_info', 'trace_dir'})
        self.assertIn("development[name]['primary_roles']", '\n'.join(sources))
        self.assertIn("for name, checkpoint in [('native-v3', V3), ('native-v4', V4)]", '\n'.join(sources))
        for helper in ['behavior_data.py', 'behavior_scenarios.py', 'recorded_behavior_scenarios.py']:
            self.assertIn(helper, build_artifacts.HELPERS)

    def _import_scope(self, root):
        source = root / 'drive'; source.mkdir()
        local = root / 'data'; local.mkdir()
        manifest = complete_manifest()
        payloads = {f'{PREFIX}-train.jsonl': b'train\n', f'{PREFIX}-development.jsonl': b'dev\n'}
        manifest['files'] = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
        for name, content in payloads.items():
            (source / name).write_bytes(content)
        evidence = b'proof-positive-and-negative-native-replay-evidence'
        evidence_name = f'{PREFIX}-evidence.zip'
        manifest['evidence'] = {'filename': evidence_name, 'sha256': hashlib.sha256(evidence).hexdigest()}
        parts = []
        for index, content in enumerate([evidence[:17], evidence[17:]], 1):
            name = f'{evidence_name}.part-{index:03d}'
            (source / name).write_bytes(content)
            parts.append({'file': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        receipt = {'schema': 'ordered-file-parts-v1', 'file': evidence_name, 'bytes': len(evidence),
                   'sha256': manifest['evidence']['sha256'], 'parts': parts}
        (source / f'{evidence_name}.parts.json').write_text(json.dumps(receipt))
        (source / f'{PREFIX}-manifest.json').write_text(json.dumps(manifest))
        return {'V4_DATA': local, 'V4_IMPORT_ROOT': source, 'V4_PREFIX': PREFIX,
                'V4_MANIFEST': local / f'{PREFIX}-manifest.json', 'Path': Path,
                'EXPECTED_V4_MANIFEST_SHA256': hashlib.sha256((source / f'{PREFIX}-manifest.json').read_bytes()).hexdigest(),
                'hashlib': hashlib, 'json': json, 'print': lambda *a, **kw: None}, evidence

    def test_import_restores_all_four_artifacts_from_parts_and_preserves_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope, evidence = self._import_scope(Path(temporary))
            source = cell('# Import the four completed balanced artifacts')
            exec(source, scope)
            local = scope['V4_DATA']; evidence_path = local / f'{PREFIX}-evidence.zip'
            self.assertEqual(evidence_path.read_bytes(), evidence)
            self.assertEqual(len(list(local.iterdir())), 4)
            exec(source, scope)  # A matching checkpoint restore is reusable.
            training = local / f'{PREFIX}-train.jsonl'
            training.write_bytes(b'previous-experiment')
            with self.assertRaisesRegex(ValueError, 'Conflicting local artifact'):
                exec(source, scope)
            self.assertEqual(training.read_bytes(), b'previous-experiment')

    def test_unapproved_manifest_cannot_import_or_reuse_semantic_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope, _ = self._import_scope(Path(temporary))
            source = scope['V4_IMPORT_ROOT'] / f'{PREFIX}-manifest.json'
            source.write_text(source.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'artifact checksum differs'):
                exec(cell('# Import the four completed balanced artifacts'), scope)
            self.assertEqual(list(scope['V4_DATA'].iterdir()), [])
        with tempfile.TemporaryDirectory() as temporary:
            scope = self._validation_scope(Path(temporary))
            exec(cell('# Strict validation must pass'), scope)
            scope['V4_MANIFEST'].write_text(scope['V4_MANIFEST'].read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'differs from the frozen'):
                scope['validated_v4_manifest']()
            self.assertIsNone(scope['_V4_VALIDATION_RECEIPT'])
            self.assertEqual(scope['validate_dataset'].call_count, 1)

    def test_corrupt_or_missing_part_never_installs_incomplete_evidence(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                scope, _ = self._import_scope(Path(temporary))
                part = scope['V4_IMPORT_ROOT'] / f'{PREFIX}-evidence.zip.part-002'
                if corrupt:
                    part.write_bytes(b'corrupt')
                else:
                    part.unlink()
                with self.assertRaises((FileNotFoundError, ValueError)):
                    exec(cell('# Import the four completed balanced artifacts'), scope)
                self.assertFalse((scope['V4_DATA'] / f'{PREFIX}-evidence.zip').exists())
                self.assertFalse(any(path.name.startswith('.') for path in scope['V4_DATA'].iterdir()))

    def test_unexpected_manifest_filename_stops_before_external_path_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope, _ = self._import_scope(Path(temporary))
            source = scope['V4_IMPORT_ROOT'] / f'{PREFIX}-manifest.json'
            manifest = json.loads(source.read_text())
            manifest['files'] = {'../outside.jsonl': 'unused'}
            source.write_text(json.dumps(manifest))
            scope['EXPECTED_V4_MANIFEST_SHA256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Unexpected balanced artifact names'):
                exec(cell('# Import the four completed balanced artifacts'), scope)
            self.assertEqual([path.name for path in scope['V4_DATA'].iterdir()], [f'{PREFIX}-manifest.json'])

    def _validation_scope(self, root):
        lab = root / 'lab'; lab.mkdir()
        for name in build_artifacts.HELPERS:
            path = lab / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        qualification = lab / 'evaluation/teacher-qualification.json'
        qualification.write_text(json.dumps({'replay_bundle': 'teacher-qualification-replays.zip'}))
        sources = {name: hashlib.sha256((lab / name).read_bytes()).hexdigest() for name in build_artifacts.HELPERS}
        data = lab / 'data'; manifest = complete_manifest()
        for suffix in ['train.jsonl', 'development.jsonl', 'evidence.zip']:
            (data / f'{PREFIX}-{suffix}').write_bytes(suffix.encode())
        manifest_path = data / f'{PREFIX}-manifest.json'; manifest_path.write_text(json.dumps(manifest))
        return {'validate_dataset': Mock(return_value=True), 'V4_DATA': data, 'V4_PREFIX': PREFIX,
                'QUALIFICATION': qualification, 'V4_MANIFEST': manifest_path, 'V4_RECIPE': RECIPE,
                'LAB_DIR': lab, 'FILES': sources, 'COURSE_REVISION': 'pinned-test-revision',
                'Path': Path, 'json': json,
                'EXPECTED_V4_MANIFEST_SHA256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'v4_file_sha256': lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'print': lambda *a, **kw: None}

    def test_boolean_validator_and_full_budgets_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = self._validation_scope(Path(temporary)); validate = scope['validate_dataset']
            manifest = complete_manifest()
            source = cell('# Strict validation must pass')
            exec(source, scope)
            validate.assert_called_once()
            self.assertEqual(scope['v4_manifest'], manifest)
            scope['_V4_VALIDATION_RECEIPT'] = None
            validate.return_value = False
            with self.assertRaisesRegex(ValueError, 'validation did not pass'):
                exec(source, scope)
            validate.return_value = True
            manifest['counts']['train']['targeted'] = 10
            scope['V4_MANIFEST'].write_text(json.dumps(manifest))
            scope['EXPECTED_V4_MANIFEST_SHA256'] = hashlib.sha256(scope['V4_MANIFEST'].read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'complete-data budget'):
                exec(source, scope)

    def test_validation_reuse_rehashes_data_sources_qualification_and_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = self._validation_scope(Path(temporary)); validate = scope['validate_dataset']
            source = cell('# Strict validation must pass')
            exec(source, scope)
            exec(source, scope)  # Redefining the cell keeps only the in-memory successful binding.
            self.assertEqual(validate.call_count, 1)
            for changed in [scope['V4_DATA'] / f'{PREFIX}-train.jsonl',
                            scope['LAB_DIR'] / 'behavior_data.py',
                            scope['QUALIFICATION'].parent / 'teacher-qualification-replays.zip',
                            scope['LAB_DIR'] / 'notebook-source.json']:
                changed.write_bytes(b'changed-input')
                before = validate.call_count
                scope['validated_v4_manifest']()
                self.assertEqual(validate.call_count, before + 1, changed.name)
                scope['validated_v4_manifest']()
                self.assertEqual(validate.call_count, before + 1)
            qualification = json.loads(scope['QUALIFICATION'].read_text())
            qualification['changed_receipt'] = True
            scope['QUALIFICATION'].write_text(json.dumps(qualification))
            scope['validated_v4_manifest']()
            self.assertEqual(validate.call_count, 6)
            self.assertEqual(set(scope['_V4_VALIDATION_RECEIPT']['local_helpers']), set(build_artifacts.HELPERS))

    def test_changed_invalid_data_cannot_reuse_old_success_or_launch_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); scope = self._validation_scope(root)
            exec(cell('# Strict validation must pass'), scope)
            (scope['V4_DATA'] / f'{PREFIX}-evidence.zip').write_bytes(b'corrupt-new-evidence')
            scope['validate_dataset'].side_effect = ValueError('native proof mismatch')
            runtime = Mock()
            scope.update(V3=root / 'native-v3', V4=root / 'native-v4', runtime=runtime,
                         verify_v3_parent=Mock(), latest_snapshot=Mock())
            with self.assertRaisesRegex(ValueError, 'native proof mismatch'):
                exec(cell('# Validate inputs and lineage'), scope)
            self.assertIsNone(scope['_V4_VALIDATION_RECEIPT'])
            runtime.finetune.assert_not_called()
            scope['latest_snapshot'].assert_not_called()

    def test_expected_canonical_extraction_is_cached_but_midcheck_source_change_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = self._validation_scope(Path(temporary)); validate = scope['validate_dataset']
            def extract(*args):
                for suffix in ['train', 'development']:
                    (scope['LAB_DIR'] / 'data' / f'pacman-native-v2-{suffix}.jsonl').write_bytes(b'canonical-extracted')
                return True
            validate.side_effect = extract
            exec(cell('# Strict validation must pass'), scope)
            scope['validated_v4_manifest']()
            self.assertEqual(validate.call_count, 1)
            (scope['V4_DATA'] / f'{PREFIX}-train.jsonl').write_bytes(b'new-data')
            def change_source(*args):
                (scope['LAB_DIR'] / 'teacher_data.py').write_bytes(b'changed-during-check')
                return True
            validate.side_effect = change_source
            with self.assertRaisesRegex(ValueError, 'changed during the semantic check'):
                scope['validated_v4_manifest']()
            self.assertIsNone(scope['_V4_VALIDATION_RECEIPT'])

    def _training_scope(self, root):
        v3, v4 = root / 'native-v3', root / 'native-v4'
        events = []
        scope = {'V3': v3, 'V4': v4, 'V4_RECIPE': RECIPE,
                 'V4_TRAINING': root / f'{PREFIX}-train.jsonl', 'EXPECTED_V3_SHA256': PARENT_SHA,
                 'validated_v4_manifest': Mock(side_effect=lambda: events.append('validated') or complete_manifest()),
                 'verify_v3_parent': Mock(side_effect=lambda: events.append('parent_checked')),
                 'checkpoint_fingerprint': lambda path: 'v4hash', 'latest_snapshot': Mock(return_value=None),
                 'runtime': Mock(), 'Path': Path, 'json': json, 'print': lambda *a, **kw: None}
        def train(data, output, init_from, steps, resume, recipe):
            events.append('trained')
            self.assertEqual(init_from, v3); self.assertEqual(output, v4)
            self.assertEqual(steps, 0); self.assertEqual(recipe, RECIPE)
            write_completed_checkpoint(v4)
        scope['runtime'].finetune.side_effect = train
        return scope, events

    def test_fresh_and_resumed_v4_validate_first_and_only_use_v4_recovery(self):
        for recovered in (False, True):
            with self.subTest(recovered=recovered), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); scope, events = self._training_scope(root)
                scope['latest_snapshot'].return_value = root / 'v4-recovery/step-100' if recovered else None
                exec(cell('# Validate inputs and lineage'), scope)
                self.assertEqual(events, ['validated', 'parent_checked', 'trained'])
                scope['latest_snapshot'].assert_called_once_with(Path(str(scope['V4']) + '-recovery'))
                self.assertEqual(scope['runtime'].finetune.call_args.kwargs['resume'], recovered)
                scope['runtime'].start.assert_not_called()

    def test_invalid_data_blocks_training_and_completed_v4_must_match_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope, _ = self._training_scope(Path(temporary))
            source = cell('# Validate inputs and lineage')
            scope['validated_v4_manifest'].side_effect = ValueError('bad proof bundle')
            with self.assertRaisesRegex(ValueError, 'bad proof bundle'):
                exec(source, scope)
            scope['runtime'].finetune.assert_not_called()
            scope['validated_v4_manifest'].side_effect = None
            scope['validated_v4_manifest'].return_value = complete_manifest()
            write_completed_checkpoint(scope['V4'], parent_sha='wrong-v3')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                exec(source, scope)
            scope['runtime'].finetune.assert_not_called()
            write_completed_checkpoint(scope['V4'], data_sha='old-unbalanced-data')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                exec(source, scope)
            write_completed_checkpoint(scope['V4'])
            exec(source, scope)
            scope['runtime'].finetune.assert_not_called()
            scope['latest_snapshot'].assert_not_called()

    def test_parent_check_is_exact_and_does_not_load_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); parent = root / 'kev-4b-pacman-native-v3'
            write_completed_checkpoint(parent)
            runtime = Mock()
            scope = {'CHECKPOINT_ROOT': root, 'LAB_DIR': root, 'Path': Path, 'json': json,
                     'runtime': runtime, 'print': lambda *a, **kw: None}
            with patch('training_stages.checkpoint_fingerprint', return_value=PARENT_SHA):
                exec(cell('from behavior_data import validate_dataset'), scope)
            runtime.start.assert_not_called()
            with patch('training_stages.checkpoint_fingerprint', return_value='different-adapter'):
                with self.assertRaisesRegex(ValueError, 'differs from the approved'):
                    exec(cell('from behavior_data import validate_dataset'), scope)
            runtime.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
