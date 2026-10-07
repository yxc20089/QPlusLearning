"""Prework: full native games, published-method comparison, fail-closed teacher gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

from gameplay_benchmark import ROOT, SPEC, BenchmarkEngine, benchmark_gameplay, diagnostic_record, summarize

VALIDATION = json.loads((ROOT/'teacher-validation-spec.json').read_text())
SOURCE_FILES = ('games/arcade-teacher.js', 'games/arcade-engine.js', 'pacman_lab.py',
                'gameplay_benchmark.py', 'games/benchmark-hooks.js', 'games/benchmark-worker.cjs',
                'vendor/arcade-pacman/source.json', 'vendor/arcade-pacman/source.zip',
                'benchmark-spec.json', 'teacher-validation-spec.json', 'teacher_validation.py')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def create_replay_bundle(directory, episodes, filename):
    """Compact native replay evidence, not training inputs or model results."""
    with zipfile.ZipFile(Path(directory)/filename, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for episode in episodes:
            rows = []
            for line in Path(episode['trace']).read_text().splitlines():
                record = json.loads(line)
                diagnostics = {k:v for k,v in record['diagnostics'].items() if k != 'http_ms'}
                rows.append({'state_sha256': fingerprint(record['request']['state']),
                             'choice': record['response']['answers']['move']['choice'],
                             'diagnostics': diagnostics})
            name = f"level-{episode['level']}-seed-{episode['seed']}.jsonl"
            archive.writestr(name, ''.join(json.dumps(r,separators=(',', ':'))+'\n' for r in rows))
            episode['replay_entry'] = name
    return hashlib.sha256((Path(directory)/filename).read_bytes()).hexdigest()


def verify_replays(report, archive_path):
    """Recompute every native transition and all gate metrics from fixed actions."""
    if hashlib.sha256(Path(archive_path).read_bytes()).hexdigest() != report['replay_sha256']:
        raise ValueError('Teacher replay checksum mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != {e['replay_entry'] for e in report['episodes']}:
            raise ValueError('Teacher replay episodes differ from the report')
        for episode in report['episodes']:
            with BenchmarkEngine() as engine:
                state = engine.request('reset', options={'seed':episode['seed'], 'level':episode['level'], 'action_version':2})
                rows, life, status = [], 0, None
                records = [json.loads(line) for line in archive.read(episode['replay_entry']).splitlines()]
                for index, record in enumerate(records):
                    if fingerprint(state) != record['state_sha256']:
                        raise ValueError(f"Teacher replay state differs at {episode['replay_entry']} decision {index}")
                    move = record['choice']; risks = engine.request('risks')
                    response = {'answers':{'move':{'choice':move,'probabilities':{d:float(d==move) for d in state['legal_moves']}}}}
                    step = engine.step(move)
                    diagnostic = diagnostic_record(state,response,step,risks,life)
                    if {k:v for k,v in diagnostic.items() if k != 'http_ms'} != record['diagnostics']:
                        raise ValueError('Teacher replay transition/diagnostics differ')
                    rows.append(diagnostic); state = step['state']
                    if step['outcome'] == 'level_cleared': status = 'level_cleared'
                    elif step['outcome'] == 'life_lost':
                        continuation = engine.request('continue')
                        if continuation['status'] == 'game_over': status = 'game_over'
                        else: state = continuation['state']; life += 1
                    if status and index != len(records)-1:
                        raise ValueError('Teacher replay continued past its native terminal state')
                measured = summarize(rows, status or episode['metrics']['outcome'])
                expected = {k:v for k,v in episode['metrics'].items() if k != 'mean_http_ms'}
                if {k:v for k,v in measured.items() if k != 'mean_http_ms'} != expected:
                    raise ValueError('Teacher gate metrics disagree with native replay')
    return True


def require_qualified_teacher(path):
    """Fail closed; a boolean in an edited report is not an authorization."""
    path = Path(path); report = json.loads(path.read_text())
    result = qualification(report)
    if not result['approved_for_training']:
        raise ValueError('Teacher is not qualified: '+ '; '.join(result['failures']))
    if report['simulation_protocol'] != SPEC or report['qualification_spec'] != VALIDATION:
        raise ValueError('Teacher qualified against another benchmark/configuration')
    if report['source_sha256'] != source_hashes():
        raise ValueError('Teacher/native implementation changed after qualification; qualify again')
    verify_replays(report,path.parent/report['replay_bundle'])
    return report


def episode_job(algorithm, seed, level, directory, options):
    spec = {**SPEC, 'seeds': [seed], 'levels': [level]}
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = benchmark_gameplay(spec=spec, teacher_algorithm=algorithm, teacher_options=options,
                                    trace_dir=Path(directory)/algorithm)
    episode = result['episodes'][0]
    episode['wall_seconds'] = time.perf_counter()-started
    return episode


def qualification(report, specifications=VALIDATION):
    failures, episodes = [], report['episodes']
    gates = specifications['gates']
    if report.get('errors'):
        failures.append('Native simulation errors; inspect the failed episode details')
    if report['phase'] != 'qualification':
        failures.append('Development runs cannot authorize training')
    expected = {(level, seed) for level in specifications['levels'] for seed in specifications['qualification_seeds']}
    actual = {(e['level'], e['seed']) for e in episodes}
    if actual != expected or len(episodes) != len(expected):
        failures.append('Qualification suite is incomplete or has duplicate/unexpected episodes')
    if len(episodes) < gates['minimum_qualification_games']:
        failures.append('Too few qualification games')
    for level in specifications['levels']:
        cases = [e for e in episodes if e['level'] == level]
        rate = sum(e['metrics']['level_cleared'] for e in cases)/len(cases) if cases else 0
        if rate < gates['minimum_clear_rate_each_level']:
            failures.append(f'Level {level}: clear rate {rate:.3f}')
    if sum(e['metrics']['life_losses'] for e in episodes) > gates['maximum_life_losses_total']:
        failures.append('Too many lives lost')
    if sum(e['metrics']['avoidable_immediate_deaths'] for e in episodes) > gates['maximum_avoidable_immediate_deaths']:
        failures.append('Avoidable immediate deaths')
    for e in episodes:
        m = e['metrics']; prefix = f"level {e['level']} seed {e['seed']}"
        if m['loop_decisions'] > gates['maximum_loop_decisions_per_game']:
            failures.append(prefix+': no-pellet cycle')
        if m['longest_no_pellet_decisions'] > gates['maximum_no_pellet_decisions_per_game']:
            failures.append(prefix+': pellet stall')
        if m.get('nonterminal_multi_tile_actions', m['multi_tile_actions']) > gates['maximum_nonterminal_multi_tile_actions']:
            failures.append(prefix+': incorrect multi-tile action')
        if m.get('nonterminal_stationary_actions', m['stationary_actions']) > gates['maximum_nonterminal_stationary_actions']:
            failures.append(prefix+': incorrect stationary action')
        if m['outcome'] != 'level_cleared' or m['pellets_remaining'] != 0:
            failures.append(prefix+': maze incomplete')
    if report['algorithm'] != specifications['teacher'] or report['options'] != specifications['teacher_options']:
        failures.append('Algorithm/configuration differs from the fixed candidate')
    return {'approved_for_training': not failures, 'failures': failures,
            'interpretation': 'Empirical complete-game qualification on the declared finite suite; no universal optimality proof.'}


def validate(directory, algorithm='rollout_mpc', phase='development', workers=4):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if phase not in ('development','qualification') or workers<1 or workers>32:
        raise ValueError('Invalid validation phase/worker count')
    sources = source_hashes()
    options = VALIDATION['teacher_options'] if algorithm == 'rollout_mpc' else {}
    seeds = VALIDATION[phase+'_seeds']
    jobs = [(algorithm, seed, level, str(directory), options) for level in VALIDATION['levels'] for seed in seeds]
    episodes, errors = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(episode_job, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                e = future.result()
            except Exception as error:
                _, seed, level, _, _ = futures[future]
                errors.append({'level':level, 'seed':seed, 'error':f'{type(error).__name__}: {error}'})
                print(f'[{algorithm}/{phase}] ERROR level={level} seed={seed}: {error}', flush=True)
                continue
            episodes.append(e); m = e['metrics']
            print(f"[{algorithm}/{phase}] {len(episodes)}/{len(jobs)} level={e['level']} seed={e['seed']} "
                  f"{m['outcome']} pellets={m['pellets_collected']} losses={m['life_losses']} "
                  f"avoidable={m['avoidable_immediate_deaths']} loops={m['loop_decisions']} "
                  f"dry={m['longest_no_pellet_decisions']} time={e['wall_seconds']:.1f}s", flush=True)
    episodes.sort(key=lambda e:(e['level'],e['seed']))
    report = {'version': VALIDATION['version'], 'phase': phase, 'algorithm': algorithm, 'options': options,
              'simulation_protocol': SPEC, 'qualification_spec': VALIDATION, 'episodes': episodes, 'errors': errors,
              'wall_seconds': time.perf_counter()-started,
              'source_sha256': sources,
              'controller': 'CPU algorithm; not trained Kev', 'training_data_generated': False}
    if sources != source_hashes():
        raise RuntimeError('Validation source/configuration changed during the run; results cannot qualify')
    report['replay_bundle'] = f'{algorithm}-{phase}-replays.zip'
    report['replay_sha256'] = create_replay_bundle(directory,episodes,report['replay_bundle'])
    verify_replays(report,directory/report['replay_bundle'])
    report['native_replays_verified'] = True
    report['qualification'] = qualification(report)
    path = directory/f'{algorithm}-{phase}.json'; path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['qualification'],indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--algorithm', choices=('route_heuristic','legacy_beam','rollout_mpc'), default='rollout_mpc')
    parser.add_argument('--phase', choices=('development','qualification'), default='development')
    parser.add_argument('--workers', type=int, default=4)
    args=parser.parse_args();validate(args.out,args.algorithm,args.phase,args.workers)
