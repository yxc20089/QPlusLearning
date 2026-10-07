"""Independent-state HTTP probes preserve order and surface server failures."""
import threading
import unittest
from unittest.mock import patch

from api_client import call_batch


class BatchProbeTests(unittest.TestCase):
    def test_independent_requests_overlap_and_keep_input_order(self):
        gate = threading.Barrier(3)

        def request(path, body):
            self.assertEqual(path, '/v1/systemone')
            gate.wait(timeout=3)
            return {'id': body['id']}, 1

        with patch('api_client.call', side_effect=request):
            results = call_batch('/v1/systemone', [{'id': i} for i in range(3)], concurrency=3)
        self.assertEqual([r[0]['id'] for r in results], [0, 1, 2])

    def test_failure_is_not_replaced_with_a_teacher_or_default_answer(self):
        with patch('api_client.call', side_effect=RuntimeError('bad schema')):
            with self.assertRaisesRegex(RuntimeError, 'bad schema'):
                call_batch('/v1/systemone', [{'id': 1}])
        for invalid in (0, 33, True):
            with self.assertRaises(ValueError):
                call_batch('/v1/systemone', [], concurrency=invalid)


if __name__ == '__main__':
    unittest.main()
