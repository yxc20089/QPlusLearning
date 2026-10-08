"""Small CPU native rewind check; no teacher training/generation."""
import unittest

from behavior_expiry_search import SnapshotEngine


class NativeSnapshotTests(unittest.TestCase):
    def test_branch_restore_preserves_native_clocks_state_and_risks(self):
        with SnapshotEngine() as engine:
            state = engine.request('reset', options={'seed': 123457, 'level': 1, 'action_version': 2})
            for _ in range(12):
                state = engine.step(state['legal_moves'][0])['state']
            handle = engine.request('snapshot')['handle']; risks = engine.request('risks')
            first = engine.step(state['legal_moves'][0])
            alternate = first['state']['legal_moves'][-1]
            engine.step(alternate)
            self.assertEqual(engine.request('restore', handle=handle), state)
            self.assertEqual(engine.request('risks'), risks)
            self.assertEqual(engine.step(state['legal_moves'][0]), first)
            self.assertEqual(engine.request('restore', handle=handle), state)


if __name__ == '__main__':
    unittest.main()
