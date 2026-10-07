"""CPU tests of multi-life evaluation, counterfactual isolation and censoring."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from gameplay_benchmark import (BenchmarkEngine, SPEC, audit_trace, benchmark_gameplay,
                               diagnostic_record, evaluate_snapshots, paired_gameplay, summarize, verify_reserved_seeds)
from pacman_lab import body, teacher
from planner_data import prepare_dataset, PREFIX


def rule_predict(request):
    move = teacher(request['state'])
    return {'answers': {'move': {'choice': move, 'probabilities': {
        key: float(key == move) for key in request['questions']['move']['criteria']}}}}


class GameplayTests(unittest.TestCase):
    def test_snapshot_empty_strata_are_unavailable_and_danger_is_scored_separately(self):
        from pacman_lab import ROOT
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            for suffix in ('.zip', '-manifest.json'):
                shutil.copyfile(ROOT/'data'/f'{PREFIX}{suffix}', target/f'{PREFIX}{suffix}')
            prepare_dataset(target)
            report = evaluate_snapshots(target/f'{PREFIX}-evaluation.jsonl', rule_predict)
        self.assertEqual(report['strata']['post_respawn']['n'], 0)
        self.assertIsNone(report['strata']['post_respawn']['accuracy'])
        self.assertEqual(report['strata']['late_maze_30_or_fewer']['n'], 0)
        self.assertEqual(report['strata']['danger_within_1_tile']['n'], 8)
        self.assertEqual(report['strata']['search_survival_critical']['n'], 23)

    def test_counterfactuals_preserve_multi_life_release_rules_and_real_transitions(self):
        # This crosses the post-death global counter -> personal counter switch.
        # The upstream rewind alone diverges; diagnostics must not change play.
        losses = 0
        with BenchmarkEngine() as diagnostic, BenchmarkEngine() as control:
            state = diagnostic.reset(7); control.reset(7)
            for _ in range(1024):
                diagnostic.request('risks')
                self.assertEqual(diagnostic.request('observe'), state)
                move = teacher(state)
                actual, expected = diagnostic.step(move), control.step(move)
                self.assertEqual(actual, expected)
                state = actual['state']
                if actual['outcome'] == 'level_cleared':
                    break
                if actual['outcome'] == 'life_lost':
                    losses += 1
                    actual, expected = diagnostic.request('continue'), control.request('continue')
                    self.assertEqual(actual, expected)
                    if actual['status'] == 'game_over':
                        break
                    state = actual['state']
        self.assertEqual(losses, 3)
        self.assertEqual(actual['status'], 'game_over')
        self.assertEqual(state['pellets_remaining'], 22)

    def test_exact_trace_replay_checks_identity_and_state_through_respawns(self):
        identity = {'adapter_sha256': 'test-only-controller', 'stage': 'CPU fixture'}
        rows = []
        with BenchmarkEngine() as control:
            state = control.reset(7)
            for _ in range(600):
                request = body(state); response = rule_predict(request)
                response['active_checkpoint'] = identity
                rows.append({'request': request, 'response': response, 'active_checkpoint': identity})
                step = control.step(response['answers']['move']['choice']); state = step['state']
                if step['outcome']:
                    continuation = control.request('continue')
                    if continuation['status'] != 'playing':
                        break
                    state = continuation['state']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            def write(values):
                path.write_text(''.join(json.dumps(row)+'\n' for row in values))
            write(rows)
            report = audit_trace(path)
            self.assertEqual(report['exact_replay_matches'], len(rows))
            self.assertEqual(report['metrics']['life_losses'], 3)
            self.assertEqual(report['metrics']['outcome'], 'game_over')
            self.assertGreater(report['metrics']['post_respawn_pellets'], 0)
            changed = copy.deepcopy(rows); changed[5]['request']['state']['score'] += 10
            write(changed)
            with self.assertRaisesRegex(ValueError, 'request 5 differs'):
                audit_trace(path)
            changed = copy.deepcopy(rows); changed[2]['active_checkpoint'] = {'stage': 'different'}
            write(changed)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                audit_trace(path)

    def test_gameplay_cap_is_censored_and_paired_protocol_must_match(self):
        spec = {**SPEC, 'seeds': [50021], 'levels': [1], 'max_decisions': 12}
        with contextlib.redirect_stdout(io.StringIO()):
            before = benchmark_gameplay(rule_predict, spec=spec)
            after = benchmark_gameplay(rule_predict, spec=spec)
        self.assertEqual(before['capped_episodes'], 1)
        self.assertEqual(before['level_clears'], 0)
        metrics = before['episodes'][0]['metrics']
        self.assertEqual(metrics['outcome'], 'decision_cap')
        self.assertTrue(metrics['censored'])
        self.assertIsNone(metrics['unsafe_immediate_choice_rate'])  # No danger opportunity != perfect safety.
        self.assertTrue(all(v == 0 for v in paired_gameplay(before, after)['mean_after_minus_before'].values()))
        changed = copy.deepcopy(after); changed['protocol']['action_semantics'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'different observation/action'):
            paired_gameplay(before, changed)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            # First episode of the packaged generator: 41 + 1009.
            verify_reserved_seeds({**SPEC, 'seeds': [1050]})

    def test_loop_metrics_distinguish_revisits_from_progress_and_reset_per_life(self):
        with BenchmarkEngine() as engine:
            state = engine.reset()
            response = rule_predict(body(state)); risks = engine.request('risks')
            step = engine.step(response['answers']['move']['choice'])
            template = diagnostic_record(state, response, step, risks, 0)
        square = [(8, 1), (8, 2), (9, 2), (9, 1)]
        rows = []
        for i in range(12):
            row = copy.deepcopy(template)
            row.update(turn=i, score_before=0, score_after=0, pellets_before=10, pellets_after=10,
                       action_frames=8, outcome=None, events={'normal_pellets': 0, 'power_pellets': 0,
                                                            'ghosts_eaten': [], 'fruit_eaten': 0})
            row['player'].update(row=square[i%4][0], column=square[i%4][1])
            row['next_player'].update(row=square[(i+1)%4][0], column=square[(i+1)%4][1])
            rows.append(row)
        result = summarize(rows, 'decision_cap')
        self.assertEqual(result['loop_periods'], {4: 5})
        self.assertEqual(result['longest_loop_streak_decisions'], 5)
        self.assertEqual(result['longest_no_pellet_frames'], 96)
        sustained = rows + copy.deepcopy(rows[:3])
        self.assertEqual(summarize(sustained, 'decision_cap')['longest_loop_streak_decisions'], 8)
        progress = copy.deepcopy(rows)
        for row in progress:
            row['pellets_after'] -= 1
        self.assertEqual(summarize(progress, 'decision_cap')['loop_decisions'], 0)
        self.assertEqual(summarize(progress, 'decision_cap')['longest_loop_streak_decisions'], 0)
        respawn = copy.deepcopy(rows)
        for i, row in enumerate(respawn): row['life_index'] = i//4
        self.assertEqual(summarize(respawn, 'decision_cap')['loop_decisions'], 0)
        self.assertEqual(summarize(respawn, 'decision_cap')['longest_loop_streak_decisions'], 0)


if __name__ == '__main__':
    unittest.main()
