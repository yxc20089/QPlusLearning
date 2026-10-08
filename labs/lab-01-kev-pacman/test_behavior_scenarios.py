"""CPU-only regression checks for measured native behavior admission."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import behavior_scenarios as behavior


class BehaviorAdmissionTests(unittest.TestCase):
    def fixtures(self):
        state = {'player': {'heading': 'right'}, 'frightened': True, 'decisions_since_last_pellet': 30}
        risks = {'left': {'life_lost': False}, 'right': {'life_lost': False}}
        positive = {'decisions': 60, 'simulation_frames': 480, 'life_losses': 0, 'level_cleared': True}
        negative = {'decisions': 70, 'simulation_frames': 560, 'life_losses': 0, 'level_cleared': True}
        transition = {'remaining_before': 4, 'remaining_after': 0,
                      'outside_frightened_nearest_before': 1, 'outside_dangerous_nearest_before': 999}
        positive_rows = [{'diagnostics': {'pellets_before': 5, 'pellets_after': 5, 'outcome': None}},
                         {'diagnostics': {'pellets_before': 5, 'pellets_after': 4, 'outcome': None}}]
        negative_rows = [{'power_transition': transition}]
        return state, risks, positive, negative, positive_rows, negative_rows

    def classify(self, fixture):
        state, risks, positive, negative, pos_rows, neg_rows = fixture
        return behavior.classify_behavior(state, risks, 'left', 'right', positive, negative, pos_rows, neg_rows)[0]

    def test_safe_looking_trap_needs_actual_later_failure(self):
        f = self.fixtures(); self.assertFalse(self.classify(f)['anticipatory_harm'])
        f[3]['life_losses'] = 1
        self.assertTrue(self.classify(f)['anticipatory_harm'])
        f[1]['right']['life_lost'] = True
        self.assertFalse(self.classify(f)['anticipatory_harm'])
        f[1]['right']['life_lost'] = False; f[1]['up'] = {'life_lost': True}
        self.assertFalse(self.classify(f)['anticipatory_harm'])

    def test_expiry_requires_observed_zero_timer_and_fatal_alt(self):
        f = self.fixtures(); self.assertFalse(self.classify(f)['actual_expiry_fatal_alternative'])
        f[1]['right']['life_lost'] = True
        self.assertTrue(self.classify(f)['actual_expiry_fatal_alternative'])
        # Long elapsed action/ghost-eating pause is not actual expiry.
        f[5][0]['power_transition']['remaining_after'] = 2
        self.assertFalse(self.classify(f)['actual_expiry_fatal_alternative'])
        f[5][0]['power_transition']['remaining_after'] = 0
        f[5][0]['power_transition']['outside_dangerous_nearest_before'] = 1
        self.assertFalse(self.classify(f)['actual_expiry_fatal_alternative'])

    def test_retreat_requires_real_later_food_and_full_route_cost(self):
        f = self.fixtures(); self.assertTrue(self.classify(f)['productive_retreat_to_food'])
        f[3]['life_losses'] = 1
        self.assertFalse(self.classify(f)['productive_retreat_to_food'])
        f[3]['life_losses'] = 0; f[3]['simulation_frames'] = 500
        self.assertFalse(self.classify(f)['productive_retreat_to_food'])
        f[3]['simulation_frames'] = 560; f[4][1]['diagnostics']['pellets_after'] = 5
        self.assertFalse(self.classify(f)['productive_retreat_to_food'])

    def test_atomic_committed_receipt_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            behavior._save(path, {'accepted': False}, immutable=True)
            behavior._save(path, {'accepted': False}, immutable=True)
            with self.assertRaises(ValueError):
                behavior._save(path, {'accepted': True}, immutable=True)
            self.assertEqual(json.loads(path.read_text()), {'accepted': False})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
