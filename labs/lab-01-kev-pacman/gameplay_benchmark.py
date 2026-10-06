"""Versioned whole-game evaluation and exact replay of private player traces.

Native counterfactuals are scoring diagnostics only. Kev receives body(state),
unchanged from v1 training; no planner or safety filter replaces its decisions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

from pacman_lab import (ArcadeEngine, ROOT, PACMAN_REVISION, body, call, destination,
                        distances, distribution, engine_script, ensure_node, evaluate, position)

SPEC = json.loads((ROOT / 'benchmark-spec.json').read_text())


def benchmark_script():
    source = engine_script()
    # Upstream rewind omits release mode and saves only the currently active
    # counter. Counterfactuals can cross GLOBAL -> PERSONAL after a respawn.
    # Augment save/load ONLY; forward physics and legacy player semantics agree
    # exactly with the unmodified source. Never run diagnostics on that source's
    # incomplete recovery hook during a live episode.
    start, end = source.index('var ghostReleaser = (function(){'), source.index('var elroyTimer = (function(){')
    section = source[start:end]
    save_start = section.index('    var save = function(t) {')
    save_end = section.index('\n    return {', save_start)
    section = section[:save_start] + '''    var savedMode = {};
    var save = function(t) {
        savedMode[t] = mode;
        savedFramesSinceLastDot[t] = framesSinceLastDot;
        savedGlobalCount[t] = globalCount;
        savedGhostCounts[t] = Object.assign({}, ghostCounts);
    };
    var load = function(t) {
        mode = savedMode[t];
        framesSinceLastDot = savedFramesSinceLastDot[t];
        globalCount = savedGlobalCount[t];
        ghostCounts = Object.assign({}, savedGhostCounts[t]);
    };
''' + section[save_end:]
    source = source[:start] + section + source[end:]
    end = source.rfind('})();')
    return source[:end] + (ROOT / 'games/benchmark-hooks.js').read_text() + '\n' + source[end:]


class BenchmarkEngine(ArcadeEngine):
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='kev-gameplay-')
        script = Path(self.directory.name) / 'pacman.js'
        script.write_text(benchmark_script())
        self.process = subprocess.Popen([ensure_node(), str(ROOT / 'games/benchmark-worker.cjs'), str(script)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)


def strata(state, candidates=()):
    maze, tile = state['maze'], position(state['player'])
    routes = distances(maze, tile)
    danger = min((routes.get(position(g), 999) for g in state['ghosts']
                  if g['mode'] == 'outside' and not g['frightened']), default=999)
    return {'junction': len(state['legal_moves']) >= 3, 'danger_within_3_tiles': danger <= 3,
            'danger_within_1_tile': danger <= 1, 'frightened': state['frightened'],
            'power_pellet_adjacent': any(maze[r][c] == 'o' for r, c in
                                       (destination(maze, tile, d) for d in state['legal_moves'])),
            'late_maze_30_or_fewer': state['pellets_remaining'] <= 30,
            'revisited_over_4_times': state['visits_to_current_tile'] > 4,
            'post_respawn': state['lives'] < 3,
            'search_survival_critical': len({c['survival'] for c in candidates}) > 1}


def evaluate_snapshots(path, predict=None):
    """Keep the old secondary metric, expose its sample coverage and hard strata."""
    result = evaluate(path, predict)
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    by_id = {r['_meta']['id']: r for r in records}
    buckets = {name: [] for name in strata(records[0]['state'])}
    for row in result['rows']:
        record = by_id[row['id']]
        groups = strata(record['state'], record['_meta'].get('teacher', {}).get('candidates', ()))
        row['strata'] = groups
        for name, included in groups.items():
            if included:
                buckets[name].append(row)
    result['strata'] = {}
    for name, rows in buckets.items():
        planned = [r for r in rows if 'lower_search_survival' in r]
        result['strata'][name] = {
            'n': len(rows), 'accuracy': sum(r['gold'] == r['prediction'] for r in rows)/len(rows) if rows else None,
            'caught_next_turn': sum(r['caught_next_turn'] for r in rows),
            'lower_search_survival_choices': sum(r['lower_search_survival'] for r in planned) if planned else None}
    return result


def diagnostic_record(state, response, step, risks, life_index, http_ms=0):
    chosen = response['answers']['move']['choice']
    events = step['events']
    groups = strata(state)
    groups['post_respawn'] = life_index > 0  # Native bonus lives can raise lives back to three.
    return {'turn': state['turn'], 'life_index': life_index, 'lives_before': state['lives'],
            'player': state['player'], 'next_player': step['state']['player'],
            'score_before': state['score'], 'score_after': step['state']['score'],
            'pellets_before': state['pellets_remaining'], 'pellets_after': step['state']['pellets_remaining'],
            'choice': chosen, 'probabilities': response['answers']['move']['probabilities'],
            'legal_moves': state['legal_moves'], 'strata': groups, 'events': events,
            'action_frames': step['action_frames'], 'outcome': step['outcome'], 'http_ms': http_ms,
            'immediate_counterfactuals': risks}


def summarize(rows, status):
    if not rows:
        raise ValueError('Cannot score an empty trajectory')
    metrics = {'decisions': len(rows), 'score': rows[-1]['score_after'],
               'pellets_collected': sum(r['pellets_before']-r['pellets_after'] for r in rows),
               'pellets_remaining': rows[-1]['pellets_after'],
               'life_losses': 0, 'avoidable_immediate_deaths': 0, 'immediate_risk_decisions': 0,
               'avoidable_risk_decisions': 0, 'all_immediate_actions_fatal': 0,
               'unsafe_immediate_choices': 0, 'loop_decisions': 0, 'longest_no_pellet_decisions': 0,
               'longest_no_pellet_frames': 0, 'simulation_frames': sum(r['action_frames'] for r in rows),
               'straight_available': 0, 'straight_chosen': 0, 'junction_straight_available': 0,
               'junction_straight_chosen': 0, 'safe_immediate_pellet_skips_at_junction': 0,
               'multi_tile_actions': 0, 'stationary_actions': 0, 'normal_pellets': 0,
               'power_pellets': 0, 'ghosts_eaten': 0, 'fruit_eaten': 0,
               'outcome': status, 'censored': status in ('decision_cap', 'simulation_frame_cap', 'trace_end'),
               'level_cleared': status == 'level_cleared', 'per_life': {}}
    periods, positions, previous_life = Counter(), [], None
    stalled, stalled_frames = 0, 0
    for row in rows:
        life = str(row['life_index'])
        if life != previous_life:
            positions, stalled, stalled_frames = [], 0, 0
            previous_life = life
        per_life = metrics['per_life'].setdefault(life, {'decisions': 0, 'pellets_collected': 0, 'life_lost': False})
        collected = row['pellets_before'] - row['pellets_after']
        per_life['decisions'] += 1
        per_life['pellets_collected'] += collected
        loss = row['outcome'] == 'life_lost'
        per_life['life_lost'] |= loss
        metrics['life_losses'] += loss
        risks = row['immediate_counterfactuals']
        any_risk, any_safe = any(r['life_lost'] for r in risks.values()), any(not r['life_lost'] for r in risks.values())
        metrics['immediate_risk_decisions'] += any_risk
        metrics['avoidable_risk_decisions'] += any_risk and any_safe
        metrics['all_immediate_actions_fatal'] += not any_safe
        unsafe = risks[row['choice']]['life_lost']
        if unsafe != loss:
            raise ValueError('Counterfactual diagnostic altered the real native transition')
        metrics['unsafe_immediate_choices'] += unsafe
        metrics['avoidable_immediate_deaths'] += loss and any(not r['life_lost'] for r in risks.values())
        heading = row['player']['heading']
        if heading in row['legal_moves']:
            metrics['straight_available'] += 1
            metrics['straight_chosen'] += row['choice'] == heading
            if row['strata']['junction']:
                metrics['junction_straight_available'] += 1
                metrics['junction_straight_chosen'] += row['choice'] == heading
        metrics['safe_immediate_pellet_skips_at_junction'] += (
            row['strata']['junction'] and collected == 0 and
            any(not r['life_lost'] and r['pellets'] > 0 for r in risks.values()))
        a, b = position(row['player']), position(row['next_player'])
        dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
        distance = dr + min(dc, 28-dc)  # Native classic maze has 28 columns.
        metrics['stationary_actions'] += distance == 0
        metrics['multi_tile_actions'] += distance > 1
        for name in ('normal_pellets', 'power_pellets', 'fruit_eaten'):
            metrics[name] += row['events'][name]
        metrics['ghosts_eaten'] += len(row['events']['ghosts_eaten'])
        if collected:
            positions, stalled, stalled_frames = [], 0, 0
        else:
            stalled += 1
            stalled_frames += row['action_frames']
            positions.append(a)
            positions = positions[-128:]
            metrics['longest_no_pellet_decisions'] = max(metrics['longest_no_pellet_decisions'], stalled)
            metrics['longest_no_pellet_frames'] = max(metrics['longest_no_pellet_frames'], stalled_frames)
            for period in range(2, min(64, len(positions)//2)+1):
                if positions[-period:] == positions[-2*period:-period] and len(set(positions[-period:])) >= 3:
                    metrics['loop_decisions'] += 1
                    periods[period] += 1
                    break
    metrics['loop_periods'] = dict(periods)
    metrics['first_life_pellets'] = metrics['per_life']['0']['pellets_collected']
    metrics['post_respawn_pellets'] = sum(v['pellets_collected'] for k, v in metrics['per_life'].items() if k != '0')
    metrics['unsafe_immediate_choice_rate'] = (metrics['unsafe_immediate_choices']/metrics['immediate_risk_decisions']
                                              if metrics['immediate_risk_decisions'] else None)
    metrics['avoidable_immediate_death_rate'] = (metrics['avoidable_immediate_deaths']/metrics['avoidable_risk_decisions']
                                               if metrics['avoidable_risk_decisions'] else None)
    metrics['straight_choice_rate'] = (metrics['straight_chosen']/metrics['straight_available']
                                     if metrics['straight_available'] else None)
    metrics['mean_http_ms'] = sum(r['http_ms'] for r in rows)/len(rows)
    return metrics


def verify_reserved_seeds(spec=SPEC):
    with zipfile.ZipFile(ROOT / 'data/pacman-planner-v1.zip') as archive:
        used = {json.loads(line)['_meta']['replay']['seed'] for name in archive.namelist()
                if name.endswith('.jsonl') for line in archive.read(name).splitlines()}
    if used.intersection(spec['seeds']):
        raise ValueError('Gameplay benchmark seeds overlap a packaged dataset split')


def audit_dataset():
    """Summarize saved labels and coverage without generating new examples."""
    archive_path = ROOT / 'data/pacman-planner-v1.zip'
    report = {'dataset_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(), 'splits': {}}
    with zipfile.ZipFile(archive_path) as archive:
        for split in ('train', 'development', 'evaluation'):
            records = [json.loads(line) for line in archive.read(f'pacman-planner-v1-{split}.jsonl').splitlines()]
            counts, lives = Counter(), Counter()
            heading_available, heading_selected = 0, 0
            junction_available, junction_selected = 0, 0
            for record in records:
                state = record['state']; lives[state['lives']] += 1
                candidates = record['_meta']['teacher']['candidates']
                for name, included in strata(state, candidates).items():
                    counts[name] += included
                best = candidates[0]
                counts['all_search_actions_die'] += best['survival'] == 0
                counts['multiple_exact_best_actions'] += sum((c['survival'], c['value']) ==
                    (best['survival'], best['value']) for c in candidates) > 1
                heading = state['player']['heading']
                if heading in state['legal_moves']:
                    heading_available += 1
                    heading_selected += record['questions']['move']['label'] == heading
                    if len(state['legal_moves']) >= 3:
                        junction_available += 1
                        junction_selected += record['questions']['move']['label'] == heading
            report['splits'][split] = {'n': len(records), 'lives': dict(lives), 'coverage': dict(counts),
                                      'straight_available': heading_available, 'straight_labelled': heading_selected,
                                      'junction_straight_available': junction_available,
                                      'junction_straight_labelled': junction_selected}
    return report


def benchmark_gameplay(predict=None, model_info=None, trace_dir=None, spec=SPEC):
    """Paired by fixed seed/level; three lives, no replacement policy on API errors."""
    verify_reserved_seeds(spec)
    if not spec['seeds'] or not spec['levels'] or spec['max_decisions'] <= 0 or spec['max_simulation_frames'] <= 0:
        raise ValueError('Benchmark episodes and caps must be nonempty and positive')
    predict = predict or (lambda request: call('/v1/systemone', request)[0])
    identity = model_info() if model_info else {'controller': 'unverified supplied predictor'}
    episodes = []
    if trace_dir is not None:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
    with BenchmarkEngine() as engine:
        for level in spec['levels']:
            for seed in spec['seeds']:
                state = engine.request('reset', options={'seed': seed, 'level': level})
                rows, life, active_frames, status = [], 0, 0, 'decision_cap'
                filename = trace_dir / f'level-{level}-seed-{seed}.jsonl' if trace_dir else None
                handle = filename.open('w') if filename else None
                print(f"[gameplay] level {level}, seed {seed}: up to {spec['max_decisions']} decisions, all lives", flush=True)
                try:
                    for turn in range(spec['max_decisions']):
                        if model_info and model_info() != identity:
                            raise RuntimeError('Active adapter changed during the gameplay benchmark')
                        risks = engine.request('risks')
                        request = body(state)
                        started = time.perf_counter()
                        response = predict(request)
                        elapsed = (time.perf_counter()-started)*1000
                        distribution(response['answers']['move'], request['questions']['move']['criteria'])
                        if model_info and model_info() != identity:
                            raise RuntimeError('Active adapter changed during the gameplay benchmark')
                        step = engine.step(response['answers']['move']['choice'])
                        row = diagnostic_record(state, response, step, risks, life, elapsed)
                        rows.append(row)
                        if handle:
                            handle.write(json.dumps({'request': request, 'response': response,
                                                     'active_checkpoint': identity, 'diagnostics': row})+'\n')
                            handle.flush()
                        active_frames += step['action_frames']
                        state = step['state']
                        if step['outcome'] == 'level_cleared':
                            status = 'level_cleared'; break
                        if step['outcome'] == 'life_lost':
                            continuation = engine.request('continue')
                            if continuation['status'] == 'game_over':
                                status = 'game_over'; break
                            state = continuation['state']; life += 1
                        if active_frames >= spec['max_simulation_frames']:
                            status = 'simulation_frame_cap'; break
                        if (turn+1) % 128 == 0:
                            print(f"[gameplay] {turn+1} decisions, score {state['score']}, pellets left {state['pellets_remaining']}, lives {state['lives']}", flush=True)
                finally:
                    if handle:
                        handle.close()
                episode = {'seed': seed, 'level': level, 'metrics': summarize(rows, status),
                           'trace': str(filename) if filename else None}
                episodes.append(episode)
                print('[gameplay] completed:', episode['metrics'], flush=True)
    means = {key: sum(e['metrics'][key] for e in episodes)/len(episodes) for key in
             ('score', 'pellets_collected', 'life_losses', 'avoidable_immediate_deaths', 'loop_decisions',
              'longest_no_pellet_decisions', 'first_life_pellets', 'post_respawn_pellets')}
    return {'protocol': spec, 'active_checkpoint': identity, 'episodes': episodes, 'means': means,
            'level_clears': sum(e['metrics']['level_cleared'] for e in episodes),
            'capped_episodes': sum(e['metrics']['censored'] for e in episodes),
            'interpretation': 'Five-seed same-maze comparison; caps are unresolved, not victories. No broad win-rate claim.'}


def paired_gameplay(before, after):
    if before['protocol'] != after['protocol']:
        raise ValueError('Cannot compare different observation/action/benchmark versions')
    b = {(e['level'], e['seed']): e for e in before['episodes']}
    a = {(e['level'], e['seed']): e for e in after['episodes']}
    if b.keys() != a.keys():
        raise ValueError('Paired episodes must use identical seeds and levels')
    keys = tuple(before['means'])
    rows = [{'level': level, 'seed': seed, 'before_outcome': b[level, seed]['metrics']['outcome'],
             'after_outcome': a[level, seed]['metrics']['outcome'],
             'after_minus_before': {k: a[level, seed]['metrics'][k]-b[level, seed]['metrics'][k] for k in keys}}
            for level, seed in b]
    return {'episodes': rows, 'mean_after_minus_before': {k: sum(r['after_minus_before'][k] for r in rows)/len(rows) for k in keys},
            'interpretation': 'Read safety, progress, clears and censoring together. Positive score alone is insufficient.'}


def audit_trace(path, seed=7, level=1):
    """Fail on any replay/identity mismatch; do not silently score a different game."""
    content = Path(path).read_bytes()
    trace = [json.loads(line) for line in content.splitlines()]
    if not trace:
        raise ValueError('Empty player trace')
    identity = trace[0]['active_checkpoint']
    records, life, status = [], 0, 'trace_end'
    with BenchmarkEngine() as engine:
        state = engine.request('reset', options={'seed': seed, 'level': level})
        for index, row in enumerate(trace):
            if row['active_checkpoint'] != identity or row['response'].get('active_checkpoint', identity) != identity:
                raise ValueError(f'Adapter identity changed at trace request {index}')
            if state != row['request']['state']:
                raise ValueError(f'Trace request {index} differs from exact native replay')
            distribution(row['response']['answers']['move'], row['request']['questions']['move']['criteria'])
            risks = engine.request('risks')
            step = engine.step(row['response']['answers']['move']['choice'])
            records.append(diagnostic_record(state, row['response'], step, risks, life, row.get('http_ms', 0)))
            state = step['state']
            if step['outcome']:
                continuation = engine.request('continue')
                if continuation['status'] != 'playing':
                    status = continuation['status']
                    if index != len(trace)-1:
                        raise ValueError('Player trace continues beyond the end of this game')
                    break
                life += 1
                state = continuation['state']
    return {'protocol': SPEC, 'trace_sha256': hashlib.sha256(content).hexdigest(),
            'seed': seed, 'level': level, 'exact_replay_matches': len(records),
            'active_checkpoint': identity, 'metrics': summarize(records, status), 'decisions': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a private recorded game; no GPU required.')
    parser.add_argument('trace', type=Path)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit_trace(args.trace, args.seed, args.level)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'exact_replay_matches': report['exact_replay_matches'], 'metrics': report['metrics']}, indent=2))
