"""Check the shared student route without downloading weights or starting training."""
import ast
import copy
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import build_artifacts


def source(cell):
    return ''.join(cell['source'])


class UnifiedNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(Path, 'write_text', autospec=True):
            cls.legacy = build_artifacts.notebook()
            cls.focused_v3 = build_artifacts.v3_notebook(cls.legacy)
            cls.focused_v4 = build_artifacts.v4_notebook(cls.legacy)
            cls.unified = build_artifacts.unified_notebook(cls.legacy, cls.focused_v4)

    def test_student_route_has_one_setup_storage_and_final_backup(self):
        cells = self.unified['cells']
        code = [source(c) for c in cells if c['cell_type'] == 'code']
        for prefix in ['from pathlib import Path\nfrom urllib.request',
                       'import subprocess\nsubprocess.check_call', 'runtime.setup()',
                       'RUN_TRAINING_PREFLIGHT = False', 'SAVE_TO_DRIVE =',
                       '# Back up all available checkpoints']:
            self.assertEqual(sum(s.startswith(prefix) for s in code), 1, prefix)
        self.assertTrue(next(s for s in code if s.startswith('SAVE_TO_DRIVE ='))
                        .startswith('SAVE_TO_DRIVE = True'))
        ids = [c['id'] for c in cells]
        self.assertEqual(len(ids), len(set(ids)))
        route = source(next(c for c in cells if c['id'] == 'lab-route'))
        self.assertTrue(set(re.findall(r'#scrollTo=([\w-]+)', route)).issubset(set(ids)))
        for required in ['skip training', 'read-only', 'supplied completed v3',
                         'private artifacts', 'optional prework']:
            self.assertIn(required, route)
        self.assertEqual(self.unified['metadata']['colab']['name'], 'pacman_kev_lab.ipynb')
        self.assertTrue(source(cells[-1]).startswith('# Back up all available checkpoints'))

    def test_v4_data_training_verification_and_gameplay_are_preserved_exactly(self):
        # The tested focused route must retain identical executable cells in the main notebook.
        by_id = {c['id']: c for c in self.unified['cells']}
        start = next(i for i, c in enumerate(self.focused_v4['cells'])
                     if source(c).startswith('## Verify the completed native-v3 parent'))
        stop = next(i for i, c in enumerate(self.focused_v4['cells'])
                    if source(c).startswith('## Final Drive backup'))
        for cell in self.focused_v4['cells'][start:stop]:
            self.assertEqual(by_id[cell['id']], cell)
        self.assertIn('## CP1 — Balanced teacher distillation into native-v4', source(by_id['v4-00']))
        headings = [source(c).splitlines()[0] for c in self.unified['cells'] if c['cell_type'] == 'markdown']
        self.assertEqual(sum(h.startswith('## CP1') for h in headings), 1)
        self.assertIn('## Earlier Pac-Man adaptation — native-v2 (optional prework)', headings)

    def test_earlier_curriculum_code_and_checkpoint_backup_are_preserved(self):
        by_id = {c['id']: c for c in self.unified['cells']}
        for cell in self.legacy['cells']:
            if cell['cell_type'] != 'code':
                continue
            expected = copy.deepcopy(cell)
            if source(cell).startswith('SAVE_TO_DRIVE = False'):
                expected['source'] = source(cell).replace('SAVE_TO_DRIVE = False', 'SAVE_TO_DRIVE = True', 1).splitlines(keepends=True)
            self.assertEqual(by_id[cell['id']], expected)
        for cell in self.unified['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(source(cell))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_focus_builders_can_reuse_main_without_changing_separate_notebooks(self):
        # Prevent a default-Drive change or duplicated main sections changing the retained routes.
        with patch.object(Path, 'write_text', autospec=True):
            self.assertEqual(build_artifacts.v3_notebook(self.unified), self.focused_v3)
            self.assertEqual(build_artifacts.v4_notebook(self.unified), self.focused_v4)
        legacy, focused = copy.deepcopy(self.legacy), copy.deepcopy(self.focused_v4)
        with patch.object(Path, 'write_text', autospec=True):
            build_artifacts.unified_notebook(legacy, focused)
        self.assertEqual(legacy, self.legacy)
        self.assertEqual(focused, self.focused_v4)

    def test_main_keeps_same_pinned_helpers_as_the_focused_route(self):
        def assignments(notebook):
            setup = next(source(c) for c in notebook['cells'] if c['cell_type'] == 'code'
                         and source(c).startswith('from pathlib import Path\nfrom urllib.request'))
            result = {}
            for statement in ast.parse(setup).body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name) and target.id in {'COURSE_REVISION', 'FILES'}:
                            result[target.id] = ast.literal_eval(statement.value)
            return result
        lock = json.loads((build_artifacts.ROOT / 'notebook-source.json').read_text())
        expected = {'COURSE_REVISION': lock['revision'], 'FILES': lock['files']}
        self.assertEqual(assignments(self.unified), expected)
        self.assertEqual(assignments(self.focused_v4), expected)


if __name__ == '__main__':
    unittest.main()
