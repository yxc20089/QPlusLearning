"""Check fresh/resumed v4 lineage and refusal of undersized hard data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from v4_data import RECIPE

ROOT = Path(__file__).resolve().parent


def cell(prefix):
    notebook = json.loads((ROOT / 'notebooks/pacman_kev_v4.ipynb').read_text())
    return next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'
                and ''.join(c['source']).startswith(prefix))


class V4NotebookTests(unittest.TestCase):
    def test_fresh_and_recovered_training_only_use_v4_optimizer_and_v3_weights(self):
        for recovered in (False, True):
            with self.subTest(recovered=recovered), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                v3, v4 = root / 'native-v3', root / 'native-v4'
                runtime = Mock()
                def train(data, output, init_from, steps, resume, recipe):
                    self.assertEqual(init_from, v3)
                    self.assertEqual(output, v4)
                    self.assertEqual(recipe['replay'], 304)
                    self.assertEqual(resume, recovered)
                    v4.mkdir()
                    (v4 / 'training_metrics.json').write_text(json.dumps({'optimizer_steps': 762,
                        'records_seen': 6096, 'requested_records': 6096}))
                    (v4 / 'run-evidence.json').write_text('{"parent_checkpoint_sha256":"v3hash"}')
                runtime.finetune.side_effect = train
                latest = Mock(return_value=root / 'v4-recovery/step-100' if recovered else None)
                manifest = {'probe_checkpoint': {'checkpoint_sha256': 'v3hash'},
                            'expected_training_requests': 6096, 'expected_optimizer_steps': 762, 'recipe': RECIPE}
                scope = {'V3': v3, 'V4': v4, 'V4_DATA': root / 'data', 'V4_RECIPE': RECIPE,
                         'V4_TRAINING': root / 'data/train.jsonl', 'QUALIFICATION': root / 'qualification.json',
                         'validate_dataset': lambda *args: manifest, 'checkpoint_fingerprint': lambda p: 'v3hash',
                         'latest_snapshot': latest, 'runtime': runtime, 'Path': Path, 'json': json,
                         'print': lambda *args: None}
                exec(cell("if (V4 / 'run-evidence.json').is_file():"), scope)
                latest.assert_called_once_with(Path(str(v4) + '-recovery'))
                runtime.finetune.assert_called_once()

    def test_completed_v4_is_reused_and_partial_dataset_never_launches_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); v4 = root / 'v4'; v4.mkdir()
            (v4 / 'run-evidence.json').write_text('{}')
            (v4 / 'training_metrics.json').write_text('{}')
            runtime, validate = Mock(), Mock()
            scope = {'V4': v4, 'runtime': runtime, 'validate_dataset': validate,
                     'print': lambda *args: None}
            source = cell("if (V4 / 'run-evidence.json').is_file():")
            exec(source, scope)
            validate.assert_not_called(); runtime.finetune.assert_not_called()
            (v4 / 'run-evidence.json').unlink()
            validate.return_value = {'probe_checkpoint': {'checkpoint_sha256': 'v3hash'},
                                     'expected_training_requests': 80, 'expected_optimizer_steps': 10, 'recipe': RECIPE}
            scope.update(V3=root / 'v3', V4_DATA=root / 'data', QUALIFICATION=root / 'qualification',
                         checkpoint_fingerprint=lambda p: 'v3hash')
            with self.assertRaises(AssertionError):
                exec(source, scope)
            runtime.finetune.assert_not_called()


if __name__ == '__main__':
    unittest.main()
