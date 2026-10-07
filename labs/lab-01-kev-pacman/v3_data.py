"""A v2 -> v3 correction round on fresh learner-visited native states.

Teacher lookahead is used for offline labels only. Full learner games and all
teacher continuation attempts, including rejected ones, remain replayable.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import signal
import time
import zipfile

from gameplay_benchmark import BenchmarkEngine, SPEC, benchmark_gameplay, diagnostic_record, strata, summarize
from pacman_lab import ROOT, body, distribution
from planner_data import RECIPE
from teacher_validation import VALIDATION, episode_quality_failures, fingerprint, require_qualified_teacher, source_hashes

PREFIX = 'pacman-native-v3'
TRAIN_SEEDS = [300017, 300029, 300043, 300059]
DEVELOPMENT_SEEDS = [310019, 310033]
COUNTS = {'train': 4096, 'development': 512}
OLD_REPLAY_COUNT = 2048
ROOTS_PER_GAME = 24
_CANCEL_RECOVERIES = None
OPPOSITE = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
MINIMUMS = {
    'train': {'learner_visited': 128, 'learner_disagreement': 64,
              'immediate_survival_critical': 512, 'critical_reverse': 256,
              'post_respawn': 512, 'stalled_current': 128, 'power_expiry': 128},
    'development': {'learner_visited': 32, 'learner_disagreement': 16,
                    'immediate_survival_critical': 64, 'critical_reverse': 32,
                    'post_respawn': 64, 'stalled_current': 16, 'power_expiry': 16},
}
for _split, _per_level in (('train', 256), ('development', 64)):
    MINIMUMS[_split].update({f'level_{level}': _per_level for level in SPEC['levels']})


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_capacity():
    """Count usable physical cores, respecting Linux affinity and CPU quotas."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    cores = set()
    for cpu in affinity:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            cores.add(((topology / 'physical_package_id').read_text().strip(),
                       (topology / 'core_id').read_text().strip()))
        except OSError:
            cores = set()
            break
    physical = len(cores) or len(affinity)
    quota = None
    # The visible cgroup root plus the process's group/ancestors may each limit it.
    root = Path('/sys/fs/cgroup')
    groups = [root]
    try:
        relative = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                        if line.startswith('0::'))
        group = root / relative.lstrip('/')
        if root in group.parents:
            groups += [group] + [p for p in group.parents if root in p.parents]
    except (OSError, StopIteration):
        pass
    for group in groups:
        try:
            maximum, period = (group / 'cpu.max').read_text().split()
            if maximum != 'max':
                limit = max(1, math.ceil(int(maximum) / int(period)))
                quota = limit if quota is None else min(quota, limit)
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return {'logical_cpus': len(affinity), 'physical_cores': physical,
            'quota_cpus': quota, 'automatic_workers': min(physical, quota or physical)}


def recovery_workers(requested, pending, capacity=None):
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested < 1):
        raise ValueError('Teacher workers must be a positive integer or None for automatic selection')
    capacity = cpu_capacity() if capacity is None else capacity
    return min(requested or capacity['automatic_workers'], pending)


def recovery_progress(started, completed, pending, workers, now=None):
    """Generation-only estimate; cached jobs never inflate the current run rate."""
    elapsed = max(0, (time.monotonic() if now is None else now) - started)
    if completed < workers or not elapsed:
        eta = 'ETA warming up'
    elif pending:
        eta = f'estimated generation remaining {elapsed * pending / completed / 60:.1f} min'
    else:
        eta = 'generation complete'
    return f'{workers} CPU workers; elapsed {elapsed / 60:.1f} min; {eta}'


def _initialize_recovery_worker(cancelled):
    global _CANCEL_RECOVERIES
    _CANCEL_RECOVERIES = cancelled
    # The parent handles notebook Interrupt, then workers close their Node engines.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _check_recovery_cancelled():
    if _CANCEL_RECOVERIES is not None and _CANCEL_RECOVERIES.is_set():
        raise InterruptedError('Teacher recovery interrupted; incomplete attempt will be retried')


def reserved_seeds():
    reserved = set(SPEC['seeds'] + VALIDATION['development_seeds'] + VALIDATION['qualification_seeds']
                   + VALIDATION.get('retired_qualification_seeds', []))
    old = load(ROOT / 'data/pacman-native-v2-manifest.json')
    reserved.update(e['seed'] for e in old['episodes'])
    with zipfile.ZipFile(ROOT / 'data/pacman-planner-v1.zip') as archive:
        reserved.update(json.loads(line)['_meta']['replay']['seed']
                        for name in archive.namelist() if name.endswith('.jsonl')
                        for line in archive.read(name).splitlines())
    return reserved


def behavior_summary(report):
    """Expose the retreat/safety/respawn gaps alongside full-game completion."""
    totals = Counter()
    life_food, life_games = Counter(), Counter()
    for episode in report['episodes']:
        metrics = episode['metrics']
        for name in ('decisions', 'life_losses', 'avoidable_immediate_deaths', 'avoidable_risk_decisions',
                     'loop_decisions', 'power_pellets', 'ghosts_eaten'):
            totals[name] += metrics[name]
        for life, values in metrics['per_life'].items():
            life_food[str(life)] += values['pellets_collected']
            life_games[str(life)] += 1
        for line in Path(episode['trace']).read_text().splitlines():
            record = json.loads(line)
            diag = record['diagnostics']
            reverse = OPPOSITE[diag['player']['heading']]
            totals['reversals'] += diag['choice'] == reverse
            risks = diag['immediate_counterfactuals']
            if risks[diag['choice']]['life_lost'] and any(not r['life_lost'] for r in risks.values()):
                safe_reverse = reverse in risks and not risks[reverse]['life_lost']
                totals['avoidable_deaths_with_safe_reverse'] += safe_reverse
                totals['confident_avoidable_deaths'] += diag['probabilities'][diag['choice']] >= .9
    opportunities = totals['avoidable_risk_decisions']
    return {'games': len(report['episodes']), 'level_clears': report['level_clears'],
            'capped_episodes': report['capped_episodes'], 'totals': dict(totals),
            'fatal_choice_when_safe_available': totals['avoidable_immediate_deaths'] / opportunities if opportunities else None,
            'reversal_rate': totals['reversals'] / totals['decisions'],
            'maximum_pellet_free_decisions': max(e['metrics']['longest_no_pellet_decisions'] for e in report['episodes']),
            'mean_pellets_per_reached_life': {life: life_food[life] / life_games[life] for life in life_games},
            'games_reaching_each_life': dict(life_games),
            'note': 'Risk exposure differs by controller trajectory. Native counterfactuals only score the next action; they do not prove eventual survival.'}


def check_seed_splits(train, development):
    a, b = set(train), set(development)
    if not a or not b or len(a) != len(train) or len(b) != len(development):
        raise ValueError('Use nonempty, unique fresh seeds for each partition')
    if a & b or (a | b) & reserved_seeds():
        raise ValueError('Correction seeds overlap another partition or reserved evaluation/previous data')


def require_v2_identity(identity):
    if (identity.get('checkpoint_name') != 'kev-4b-pacman-native-v2'
            or identity.get('base') != 'Qwen/Qwen3.5-4B-Base'
            or identity.get('stage') != 'pacman' or not identity.get('pacman_fine_tuned')
            or identity.get('optimizer_steps', 0) <= 0
            or not identity.get('adapter_sha256') or not identity.get('checkpoint_sha256')):
        raise ValueError('Collect corrections with the completed Pac-Man native-v2 adapter')


def collect(directory, model_info, predict=None, train_seeds=TRAIN_SEEDS, development_seeds=DEVELOPMENT_SEEDS):
    """Full native learner games. No teacher or safety filter alters decisions."""
    check_seed_splits(train_seeds, development_seeds)
    identity = model_info()
    require_v2_identity(identity)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split, seeds in (('train', train_seeds), ('development', development_seeds)):
        spec = {**SPEC, 'seeds': list(seeds)}
        target = directory / f'learner-{split}.json'
        if target.exists():
            report = load(target)
            if report['protocol'] != spec or report['active_checkpoint'] != identity:
                raise ValueError('Saved learner collection has different seeds or adapter; choose a new collection directory')
            if not all(Path(e['trace']).is_file() for e in report['episodes']):
                raise ValueError('Saved learner collection is missing trajectories')
            print(f'[v3] Reusing complete {split} learner collection', flush=True)
        else:
            print(f'[v3] Collecting {len(seeds) * len(SPEC["levels"])} fresh {split} learner games', flush=True)
            report = benchmark_gameplay(predict=predict, model_info=model_info,
                                        trace_dir=directory / f'learner-{split}', spec=spec)
            save(target, report)
        reports[split] = report
    save(directory / 'collection.json', {'dataset': PREFIX, 'learner': identity,
         'reports': {s: f'learner-{s}.json' for s in reports}, 'source_sha256': source_hashes(),
         'note': 'Fresh training/development episodes. Never label the 20 reserved benchmark starts.'})
    return reports


def mixed_risk(risks):
    fatal = [value['life_lost'] for value in risks.values()]
    return any(fatal) and not all(fatal)


def candidate_flags(record):
    state, diag = record['request']['state'], record['diagnostics']
    risks = diag['immediate_counterfactuals']
    return {
        'avoidable_fatal': mixed_risk(risks) and risks[diag['choice']]['life_lost'],
        'risk_opportunity': mixed_risk(risks),
        'stalled': state.get('decisions_since_last_pellet', 0) >= 32,
        'power_expiry': state['frightened'] and state['timing']['power']['remaining_frames'] <= 60,
        'post_respawn': state['life_epoch'] > 0,
        'late_maze': state['pellets_remaining'] <= 30,
    }


def select_roots(records, budget=ROOTS_PER_GAME):
    """Cover all six cohorts before filling the budget; select actual observations."""
    flags = [candidate_flags(record) for record in records]
    precursors = set()
    for index, value in enumerate(flags):
        if value['avoidable_fatal']:
            for offset in (1, 4, 8, 16):
                earlier = index - offset
                if earlier >= 0 and records[earlier]['diagnostics']['life_index'] == records[index]['diagnostics']['life_index']:
                    precursors.add(earlier)
    for i, value in enumerate(flags):
        value['pre_trap'] = i in precursors
    selected = []
    # Spread each cohort through its trajectory instead of picking adjacent rows.
    for name in ('avoidable_fatal', 'pre_trap', 'stalled', 'power_expiry', 'post_respawn', 'late_maze', 'risk_opportunity'):
        indices = [i for i, value in enumerate(flags) if value[name] and i not in selected]
        slots = min(3, len(indices), budget - len(selected))
        if slots:
            selected.extend(indices[(j * len(indices)) // slots] for j in range(slots))
    remaining = [i for i, value in enumerate(flags) if i not in selected and any(value.values())]
    random.Random(71).shuffle(remaining)
    selected.extend(remaining[:budget - len(selected)])
    return sorted(selected)


def replay_prefix(engine, rows, root_index):
    first = rows[0]['request']['state']
    # Episode seed is never exposed to Kev, so the caller resets separately.
    if engine.request('observe') != first:
        raise ValueError('Learner replay initial state differs')
    for record in rows[:root_index]:
        _check_recovery_cancelled()
        if engine.request('observe') != record['request']['state']:
            raise ValueError('Learner replay state differs before the correction root')
        diag = record['diagnostics']
        response = record['response']
        distribution(response['answers']['move'], body(record['request']['state'])['questions']['move']['criteria'])
        risks = engine.request('risks')
        step = engine.step(diag['choice'])
        measured = diagnostic_record(record['request']['state'], response, step, risks, diag['life_index'], diag['http_ms'])
        if measured != diag:
            raise ValueError('Learner replay diagnostics differ')
        if step['outcome'] == 'life_lost':
            if engine.request('continue')['status'] != 'playing':
                raise ValueError('Correction root occurs after native game-over')
        elif step['outcome']:
            raise ValueError('Correction root occurs after a maze clear')
    if engine.request('observe') != rows[root_index]['request']['state']:
        raise ValueError('Correction root is not the exact learner-visited native state')


def coverage_flags(record):
    state, meta = record['state'], record['_meta']
    flags = strata(state)
    flags.pop('search_survival_critical')
    flags.update(
        post_respawn=state['life_epoch'] > 0,
        learner_visited=meta['origin'] == 'learner_visited',
        learner_disagreement=meta.get('learner_choice') is not None and meta['learner_choice'] != record['questions']['move']['label'],
        immediate_survival_critical=meta['immediate_survival_critical'],
        critical_reverse=meta['immediate_survival_critical'] and record['questions']['move']['label'] == OPPOSITE[state['player']['heading']],
        stalled_current=state.get('decisions_since_last_pellet', 0) >= 32,
        power_expiry=state['frightened'] and state['timing']['power']['remaining_frames'] <= 60,
    )
    flags.update({f'level_{level}': state['level'] == level for level in SPEC['levels']})
    return flags


def continuation_job(job):
    """Reproduce a learner prefix, then require a complete teacher recovery."""
    _check_recovery_cancelled()
    rows = [json.loads(line) for line in Path(job['learner_trace']).read_text().splitlines()]
    if file_hash(job['learner_trace']) != job['learner_trace_sha256']:
        raise ValueError('Learner trajectory changed during correction collection')
    folder = Path(job['directory']) / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    trace = folder / 'teacher.jsonl'
    candidates, diagnostics, life = [], [], 0
    status, stale, frames = 'decision_cap', 0, 0
    with BenchmarkEngine() as engine, trace.open('w') as out:
        engine.request('reset', options={'seed': job['seed'], 'level': job['level'], 'action_version': 2})
        replay_prefix(engine, rows, job['root_index'])
        state = engine.request('observe')
        for index in range(SPEC['max_decisions']):
            _check_recovery_cancelled()
            risks = engine.request('risks')
            plan = engine.request('teacher', options=job['options'])
            if engine.request('observe') != state:
                raise ValueError('Teacher query altered the native correction root')
            move = plan['choice']
            response = {'answers': {'move': {'choice': move, 'probabilities': {
                d: float(d == move) for d in state['legal_moves']}}}}
            step = engine.step(move)
            diag = diagnostic_record(state, response, step, risks, life)
            diagnostics.append(diag)
            proof = {'state_sha256': fingerprint(state), 'choice': move, 'diagnostics': diag}
            out.write(json.dumps(proof, separators=(',', ':')) + '\n')
            out.flush()
            request = body(state)
            request.pop('model')
            request['questions']['move'].update(label=move, src='pacman_v3_verified_teacher')
            request['_meta'] = {'group_id': f'level-{job["level"]}-seed-{job["seed"]}',
                                'branch_id': job['id'], 'decision': state['turn'],
                                'suffix_index': index,
                                'origin': 'learner_visited' if index == 0 else 'teacher_recovery',
                                'learner_choice': rows[job['root_index']]['diagnostics']['choice'] if index == 0 else None,
                                'immediate_survival_critical': mixed_risk(risks),
                                'label_source': 'Frozen qualified teacher; complete native recovery verified'}
            if any(not value['life_lost'] for value in risks.values()):
                candidates.append(request)
            frames += step['action_frames']
            stale = stale + 1 if step['state']['pellets_remaining'] == state['pellets_remaining'] else 0
            state = step['state']
            if step['outcome'] == 'level_cleared':
                status = 'level_cleared'
                break
            if step['outcome'] == 'life_lost':
                continued = engine.request('continue')
                if continued['status'] == 'game_over':
                    status = 'game_over'
                    break
                state = continued['state']
                life += 1
                stale = 0
            if stale >= SPEC['max_no_pellet_decisions']:
                status = 'no_progress_watchdog'
                break
            if frames >= SPEC['max_simulation_frames']:
                status = 'simulation_frame_cap'
                break
    metrics = summarize(diagnostics, status)
    failures = episode_quality_failures(metrics)
    if all(value['life_lost'] for value in diagnostics[0]['immediate_counterfactuals'].values()):
        failures.append('root has no immediately surviving legal action')
    if metrics['life_losses'] > VALIDATION['gates']['maximum_life_losses_total']:
        failures.append('too many teacher life losses')
    receipt = {**job, 'metrics': metrics, 'accepted': not failures, 'rejection_reasons': failures,
               'trace': str(trace), 'trace_sha256': file_hash(trace)}
    # Rejected recoveries remain evidence, but never supply training labels.
    if not failures:
        temporary = folder / 'candidates.jsonl.tmp'
        with temporary.open('w') as out:
            for record in candidates:
                out.write(json.dumps(record, separators=(',', ':')) + '\n')
        temporary.replace(folder / 'candidates.jsonl')
    # Commit last: a completed receipt always has its full trace and labels.
    save(folder / 'attempt.json', receipt)
    return receipt


def run_recoveries(pending, attempts, directory, workers):
    """Bound submissions and stop cooperatively; reruns reuse committed attempts."""
    if not pending:
        return []
    started, finished, errors = time.monotonic(), 0, []
    total = len(attempts) + len(pending)
    jobs = iter(pending)
    context = multiprocessing.get_context()
    cancelled = context.Event()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                               initializer=_initialize_recovery_worker, initargs=(cancelled,))
    futures = {}

    def submit_next():
        job = next(jobs, None)
        if job is not None:
            futures[pool.submit(continuation_job, job)] = job

    def record_progress():
        save(Path(directory) / 'teacher-recoveries.json', {'attempts': attempts, 'errors': errors,
             'execution': {'workers': workers, 'completed_this_run': finished,
                           'elapsed_seconds': time.monotonic() - started}})

    try:
        for _ in range(workers):
            submit_next()
        while futures:
            ready, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
            if not ready:
                print(f'[v3] Teacher recovery running: {len(attempts)}/{total} complete; '
                      + recovery_progress(started, finished, len(pending) - finished, workers), flush=True)
            for future in ready:
                job = futures.pop(future)
                try:
                    attempt = future.result()
                except Exception as error:
                    errors.append({'id': job['id'], 'error': str(error)})
                else:
                    attempts.append(attempt)
                    m = attempt['metrics']
                    print(f'[v3] {len(attempts)}/{total}: {job["id"]}; admitted={attempt["accepted"]}, '
                          f'outcome={m["outcome"]}, deaths={m["life_losses"]}, avoidable={m["avoidable_immediate_deaths"]}, '
                          f'dry={m["longest_no_pellet_decisions"]}', flush=True)
                finished += 1
                print('[v3] ' + recovery_progress(started, finished, len(pending) - finished, workers), flush=True)
                record_progress()
                submit_next()
    except BaseException:
        cancelled.set()
        for future in futures:
            future.cancel()
        record_progress()
        print('[v3] Stopping workers after their current native query. Completed attempt receipts are retained; '
              'rerun this cell to retry only missing attempts.', flush=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return errors


def balanced_select(records, count, minimums, seed):
    """Unique native observations, with explicit overlapping coverage floors."""
    pool = {}
    for record in records:
        key = fingerprint(record['state'])
        # A learner-visited root takes precedence over the same expert suffix state.
        if key not in pool or record['_meta']['origin'] == 'learner_visited':
            pool[key] = record
    available = list(pool.values())
    random.Random(seed).shuffle(available)
    flags = [coverage_flags(record) for record in available]
    coverage = Counter()
    selected, unused = [], set(range(len(available)))
    while any(coverage[key] < needed for key, needed in minimums.items()):
        deficient = [key for key, needed in minimums.items() if coverage[key] < needed]
        chosen = max(sorted(unused), key=lambda i: sum(flags[i][key] / minimums[key] for key in deficient), default=None)
        if chosen is None or not any(flags[chosen][key] for key in deficient) or len(selected) >= count:
            missing = {key: needed - coverage[key] for key, needed in minimums.items() if needed > coverage[key]}
            raise ValueError(f'Coverage is insufficient; no dataset published. Missing: {missing}. Collect more fresh learner games.')
        selected.append(available[chosen])
        coverage.update({key: int(value) for key, value in flags[chosen].items()})
        unused.remove(chosen)
    for index in sorted(unused):
        if len(selected) >= count:
            break
        selected.append(available[index])
        coverage.update({key: int(value) for key, value in flags[index].items()})
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} distinct verified examples available; need {count}')
    return selected, dict(coverage)


def verify_native_evidence(manifest, bundle):
    """Independently replay learner prefixes and every teacher recovery attempt."""
    with zipfile.ZipFile(bundle) as archive:
        learner = {group: [json.loads(line) for line in archive.read(info['entry']).splitlines()]
                   for group, info in manifest['learner_replays'].items()}
        for number, attempt in enumerate(manifest['attempts'], 1):
            group = f'level-{attempt["level"]}-seed-{attempt["seed"]}'
            prefix = learner[group][:attempt['root_index']]
            suffix = [json.loads(line) for line in archive.read(f'{attempt["id"]}.jsonl').splitlines()]
            with BenchmarkEngine() as engine:
                state = engine.request('reset', options={'seed': attempt['seed'], 'level': attempt['level'], 'action_version': 2})
                for record in prefix:
                    if fingerprint(state) != record['state_sha256']:
                        raise ValueError('Recovery learner-prefix replay differs')
                    diag = record['diagnostics']
                    response = {'answers': {'move': {'choice': record['choice'], 'probabilities': diag['probabilities']}}}
                    risks = engine.request('risks')
                    step = engine.step(record['choice'])
                    if diagnostic_record(state, response, step, risks, diag['life_index'], diag['http_ms']) != diag:
                        raise ValueError('Recovery learner-prefix transition differs')
                    state = step['state']
                    if step['outcome']:
                        continued = engine.request('continue')
                        if continued['status'] != 'playing':
                            raise ValueError('Recovery begins after native terminal state')
                        state = continued['state']
                rows, life, terminal = [], 0, None
                for index, record in enumerate(suffix):
                    if fingerprint(state) != record['state_sha256']:
                        raise ValueError('Recovery teacher replay state differs')
                    move = record['choice']
                    response = {'answers': {'move': {'choice': move, 'probabilities': {
                        d: float(d == move) for d in state['legal_moves']}}}}
                    risks = engine.request('risks')
                    step = engine.step(move)
                    diag = diagnostic_record(state, response, step, risks, life)
                    if diag != record['diagnostics']:
                        raise ValueError('Recovery teacher transition differs')
                    rows.append(diag)
                    state = step['state']
                    if step['outcome'] == 'level_cleared':
                        terminal = 'level_cleared'
                    elif step['outcome'] == 'life_lost':
                        continued = engine.request('continue')
                        if continued['status'] == 'game_over':
                            terminal = 'game_over'
                        else:
                            state = continued['state']
                            life += 1
                    if terminal and index != len(suffix) - 1:
                        raise ValueError('Teacher replay continued beyond native terminal state')
                measured = summarize(rows, terminal or attempt['metrics']['outcome'])
                if fingerprint(measured) != fingerprint(attempt['metrics']):
                    raise ValueError('Recovery metrics disagree with native replay')
                failures = episode_quality_failures(measured)
                if all(value['life_lost'] for value in rows[0]['immediate_counterfactuals'].values()):
                    failures.append('root has no immediately surviving legal action')
                if measured['life_losses'] > VALIDATION['gates']['maximum_life_losses_total']:
                    failures.append('too many teacher life losses')
                if attempt['accepted'] != (not failures) or failures != attempt['rejection_reasons']:
                    raise ValueError('Teacher recovery admission differs from native replay')
            if number % 16 == 0 or number == len(manifest['attempts']):
                print(f'[v3] Native replay verification {number}/{len(manifest["attempts"])}', flush=True)
    return True


def old_expert_records():
    """Use the canonical v2 training partition, never development or benchmark states."""
    from teacher_data import prepare_dataset
    prepare_dataset(ROOT / 'data')
    manifest = load(ROOT / 'data/pacman-native-v2-manifest.json')
    records = [json.loads(line) for line in (ROOT / 'data/pacman-native-v2-train.jsonl').read_text().splitlines()]
    with zipfile.ZipFile(ROOT / 'data' / manifest['replay_bundle']) as archive:
        lookup = {}
        for episode in manifest['episodes']:
            if episode['split'] != 'train':
                continue
            group = f'level-{episode["level"]}-seed-{episode["seed"]}'
            for line in archive.read(episode['replay_entry']).splitlines():
                proof = json.loads(line)
                lookup[group, proof['state_sha256']] = proof
    result = []
    for record in records:
        proof = lookup[record['_meta']['group_id'], fingerprint(record['state'])]
        record['_meta'] = {'group_id': record['_meta']['group_id'], 'decision': record['state']['turn'],
                           'origin': 'v2_expert_replay', 'branch_id': None, 'learner_choice': None,
                           'immediate_survival_critical': mixed_risk(proof['diagnostics']['immediate_counterfactuals']),
                           'label_source': 'Unchanged canonical v2 qualified-teacher training label'}
        result.append(record)
    return result


def build(directory, output_directory, qualification_path, workers=None):
    """Generate the mixture only after teacher recovery and split checks pass."""
    directory, output_directory = Path(directory), Path(output_directory)
    target = output_directory / f'{PREFIX}-manifest.json'
    if target.exists():
        return validate_dataset(output_directory, qualification_path)
    teacher = require_qualified_teacher(qualification_path)
    collection = load(directory / 'collection.json')
    require_v2_identity(collection['learner'])
    if collection['source_sha256'] != source_hashes():
        raise ValueError('Native/teacher sources changed after learner collection')
    reports = {s: load(directory / name) for s, name in collection['reports'].items()}
    check_seed_splits(reports['train']['protocol']['seeds'], reports['development']['protocol']['seeds'])
    jobs = []
    for split, report in reports.items():
        if report['active_checkpoint'] != collection['learner'] or report['protocol'] != {**SPEC, 'seeds': report['protocol']['seeds']}:
            raise ValueError('Learner collection adapter/protocol mismatch')
        if {(e['level'], e['seed']) for e in report['episodes']} != {
                (level, seed) for level in SPEC['levels'] for seed in report['protocol']['seeds']}:
            raise ValueError('Learner collection has missing or unexpected starting cases')
        for episode in report['episodes']:
            rows = [json.loads(line) for line in Path(episode['trace']).read_text().splitlines()]
            if not rows or any(row['active_checkpoint'] != collection['learner'] for row in rows):
                raise ValueError('Learner episode is empty or changed adapters')
            measured = summarize([row['diagnostics'] for row in rows], episode['metrics']['outcome'])
            if fingerprint(measured) != fingerprint(episode['metrics']):
                raise ValueError('Learner episode metrics disagree with its trace')
            for index in select_roots(rows):
                branch = f'{split}-level-{episode["level"]}-seed-{episode["seed"]}-turn-{rows[index]["diagnostics"]["turn"]}'
                jobs.append({'id': branch, 'split': split, 'seed': episode['seed'], 'level': episode['level'],
                             'root_index': index, 'learner_trace': episode['trace'],
                             'learner_trace_sha256': file_hash(episode['trace']),
                             'directory': str(directory / 'teacher-recoveries'), 'options': teacher['options']})
    attempts = []
    pending = []
    for job in jobs:
        path = Path(job['directory']) / job['id'] / 'attempt.json'
        if path.exists():
            attempt = load(path)
            if any(attempt.get(k) != v for k, v in job.items()) or file_hash(attempt['trace']) != attempt['trace_sha256']:
                raise ValueError('Cached teacher recovery inputs/evidence changed')
            if attempt['accepted'] and not Path(attempt['trace']).with_name('candidates.jsonl').is_file():
                raise ValueError('Cached admitted teacher recovery is missing its labels')
            attempts.append(attempt)
        else:
            pending.append(job)
    print(f'[v3] {len(jobs)} teacher recovery attempts; {len(attempts)} cached, {len(pending)} pending', flush=True)
    capacity = cpu_capacity()
    workers = recovery_workers(workers, len(pending), capacity)
    print(f'[v3] Native JavaScript CPU teacher: {json.dumps(capacity)}; selected workers={workers}. '
          'GPU acceleration is not implemented for this native simulator.', flush=True)
    errors = run_recoveries(pending, attempts, directory, workers)
    if errors:
        raise ValueError('Native recovery jobs failed; evidence retained. Rerun to finish missing attempts.')
    attempts.sort(key=lambda x: x['id'])
    pools = {'train': [], 'development': []}
    for attempt in attempts:
        if attempt['accepted']:
            path = Path(attempt['trace']).with_name('candidates.jsonl')
            proofs = [json.loads(line) for line in Path(attempt['trace']).read_text().splitlines()]
            for line in path.read_text().splitlines():
                record = json.loads(line)
                proof = proofs[record['_meta']['suffix_index']]
                if fingerprint(record['state']) != proof['state_sha256'] or record['questions']['move']['label'] != proof['choice']:
                    raise ValueError('Candidate label no longer matches teacher recovery evidence')
                pools[attempt['split']].append(record)
    old = old_expert_records()
    prior_training_hashes = {fingerprint(record['state']) for record in old}
    random.Random(73).shuffle(old)
    old = old[:OLD_REPLAY_COUNT]
    replay_hashes = {fingerprint(record['state']) for record in old}
    pools['train'] = [r for r in pools['train'] if fingerprint(r['state']) not in replay_hashes]
    partitions, coverage = {}, {}
    fresh, fresh_coverage = balanced_select(pools['train'], COUNTS['train'] - OLD_REPLAY_COUNT, MINIMUMS['train'], 79)
    partitions['train'] = old + fresh
    random.Random(83).shuffle(partitions['train'])
    used_training_hashes = prior_training_hashes | {fingerprint(r['state']) for r in fresh}
    pools['development'] = [r for r in pools['development'] if fingerprint(r['state']) not in used_training_hashes]
    partitions['development'], _ = balanced_select(pools['development'], COUNTS['development'], MINIMUMS['development'], 89)
    seen = set()
    for split, records in partitions.items():
        counts = Counter()
        for i, record in enumerate(records):
            digest = fingerprint(record['state'])
            if digest in seen:
                raise ValueError('Exact state overlaps dataset partitions')
            seen.add(digest)
            record['_meta']['id'] = f'{split}-v3-{i:04d}'
            counts.update({key: int(value) for key, value in coverage_flags(record).items()})
        coverage[split] = dict(counts)
    if collection['source_sha256'] != source_hashes():
        raise ValueError('Native/teacher sources changed during recovery verification')
    output_directory.mkdir(parents=True, exist_ok=True)
    bundle = output_directory / f'{PREFIX}-replays.zip'
    learner_replays = {}
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for split, report in reports.items():
            for episode in report['episodes']:
                group = f'level-{episode["level"]}-seed-{episode["seed"]}'
                entry = f'learner-{split}-{group}.jsonl'
                compact = []
                for line in Path(episode['trace']).read_text().splitlines():
                    record = json.loads(line)
                    compact.append({'state_sha256': fingerprint(record['request']['state']),
                                    'choice': record['diagnostics']['choice'], 'diagnostics': record['diagnostics']})
                archive.writestr(entry, ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in compact))
                learner_replays[group] = {'entry': entry, 'split': split, 'seed': episode['seed'], 'level': episode['level']}
        for attempt in attempts:
            archive.write(attempt['trace'], f'{attempt["id"]}.jsonl')
    files = {bundle.name: file_hash(bundle)}
    for split, records in partitions.items():
        path = output_directory / f'{PREFIX}-{split}.jsonl'
        path.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in records))
        files[path.name] = file_hash(path)
    manifest = {'dataset': PREFIX, 'counts': COUNTS, 'recipe': RECIPE, 'coverage': coverage,
                'new_training_coverage': fresh_coverage,
                'minimums': MINIMUMS, 'old_expert_replay': OLD_REPLAY_COUNT,
                'new_teacher_corrections_and_recoveries': COUNTS['train'] - OLD_REPLAY_COUNT,
                'expected_training_requests': COUNTS['train'] + RECIPE['replay'],
                'expected_optimizer_steps': (COUNTS['train'] + RECIPE['replay'] + 7) // 8,
                'learner': collection['learner'], 'source_sha256': source_hashes(),
                'generator_sha256': file_hash(Path(__file__)),
                'qualification_sha256': file_hash(qualification_path), 'teacher_options': teacher['options'],
                'collection_directory': str(directory), 'train_seeds': reports['train']['protocol']['seeds'],
                'development_seeds': reports['development']['protocol']['seeds'],
                'attempts': attempts, 'files': files, 'replay_bundle': bundle.name,
                'learner_replays': learner_replays,
                'note': 'All rejected teacher recoveries retained; no labels admitted from them. Fresh development episodes never enter training. Reserved benchmark starts remain evaluation only.'}
    verify_native_evidence(manifest, bundle)
    manifest['native_recovery_replays_verified'] = True
    save(target, manifest)
    return validate_dataset(output_directory, qualification_path, replay=False)


def validate_dataset(directory, qualification_path, replay=True):
    """Verify hashes, source/qualification, coverage, splits and every label binding."""
    directory = Path(directory)
    manifest = load(directory / f'{PREFIX}-manifest.json')
    if (manifest['dataset'] != PREFIX or manifest['source_sha256'] != source_hashes()
            or manifest['generator_sha256'] != file_hash(Path(__file__))
            or manifest['qualification_sha256'] != file_hash(qualification_path)
            or manifest['recipe'] != RECIPE or manifest['counts'] != COUNTS
            or manifest['minimums'] != MINIMUMS):
        raise ValueError('v3 dataset configuration or provenance differs')
    teacher = require_qualified_teacher(qualification_path)
    require_v2_identity(manifest['learner'])
    if manifest['teacher_options'] != teacher['options']:
        raise ValueError('v3 teacher configuration differs from its qualification')
    check_seed_splits(manifest['train_seeds'], manifest['development_seeds'])
    for name, expected in manifest['files'].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError(f'v3 dataset checksum mismatch: {name}')
    if not manifest.get('native_recovery_replays_verified'):
        raise ValueError('v3 teacher recoveries have not been independently replayed')
    if replay:
        verify_native_evidence(manifest, directory / manifest['replay_bundle'])
    original = {fingerprint(r['state']): r['questions']['move']['label'] for r in old_expert_records()}
    attempts = {a['id']: a for a in manifest['attempts']}
    for attempt in attempts.values():
        split = attempt['split']
        seeds = manifest['train_seeds'] if split == 'train' else manifest['development_seeds'] if split == 'development' else []
        if attempt['seed'] not in seeds or attempt['level'] not in SPEC['levels']:
            raise ValueError('v3 teacher recovery seed/level belongs to another partition')
        if attempt['options'] != teacher['options']:
            raise ValueError('v3 recovery used another teacher configuration')
    seen, old_count = set(), 0
    with zipfile.ZipFile(directory / manifest['replay_bundle']) as archive:
        lookup = {a['id']: [json.loads(line) for line in archive.read(f'{a["id"]}.jsonl').splitlines()] for a in attempts.values()}
        learner_lookup = {group: [json.loads(line) for line in archive.read(info['entry']).splitlines()]
                          for group, info in manifest['learner_replays'].items()}
        for split, count in COUNTS.items():
            records = [json.loads(line) for line in (directory / f'{PREFIX}-{split}.jsonl').read_text().splitlines()]
            if len(records) != count:
                raise ValueError('v3 partition count differs')
            coverage, fresh_coverage = Counter(), Counter()
            for record in records:
                state, meta = record['state'], record['_meta']
                digest = fingerprint(state)
                if digest in seen:
                    raise ValueError('v3 state overlap between partitions')
                if split == 'development' and digest in original:
                    raise ValueError('v3 development state was already used to train v2')
                seen.add(digest)
                question = record['questions']['move']
                if {k: v for k, v in question.items() if k not in ('label', 'src')} != body(state)['questions']['move']:
                    raise ValueError('v3 training/inference request formatting differs')
                if question['label'] not in state['legal_moves']:
                    raise ValueError('v3 label is not legal')
                if meta['origin'] == 'v2_expert_replay':
                    if split != 'train' or original.get(digest) != question['label']:
                        raise ValueError('v2 expert replay label/split differs')
                    old_count += 1
                else:
                    attempt = attempts[meta['branch_id']]
                    if (not attempt['accepted'] or attempt['split'] != split
                            or episode_quality_failures(attempt['metrics'])):
                        raise ValueError('v3 label came from an unqualified teacher recovery')
                    proof = lookup[meta['branch_id']][meta['suffix_index']]
                    if not any(not value['life_lost'] for value in proof['diagnostics']['immediate_counterfactuals'].values()):
                        raise ValueError('v3 correction target has no immediately surviving action')
                    if proof['state_sha256'] != digest or proof['choice'] != question['label']:
                        raise ValueError('v3 label differs from verified teacher recovery')
                    if meta['immediate_survival_critical'] != mixed_risk(proof['diagnostics']['immediate_counterfactuals']):
                        raise ValueError('v3 safety cohort metadata differs')
                    if ((meta['origin'] == 'learner_visited') != (meta['suffix_index'] == 0)
                            or meta['group_id'] != f'level-{attempt["level"]}-seed-{attempt["seed"]}'):
                        raise ValueError('v3 source state/group differs')
                    learner_root = learner_lookup[meta['group_id']][attempt['root_index']]
                    if meta['origin'] == 'learner_visited' and (
                            learner_root['state_sha256'] != digest or meta['learner_choice'] != learner_root['choice']):
                        raise ValueError('v3 learner disagreement metadata differs from its actual rollout')
                    if meta['origin'] == 'teacher_recovery' and meta['learner_choice'] is not None:
                        raise ValueError('Teacher recovery states are not learner-visited correction roots')
                coverage.update({key: int(value) for key, value in coverage_flags(record).items()})
                if meta['origin'] != 'v2_expert_replay':
                    fresh_coverage.update({key: int(value) for key, value in coverage_flags(record).items()})
            if dict(coverage) != manifest['coverage'][split] or any(coverage[k] < n for k, n in MINIMUMS[split].items()):
                raise ValueError('v3 coverage floors differ or fail')
            if split == 'train' and (dict(fresh_coverage) != manifest['new_training_coverage'] or
                                     any(fresh_coverage[k] < n for k, n in MINIMUMS[split].items())):
                raise ValueError('New v3 corrections fail coverage floors; old replay cannot satisfy them')
    if old_count != OLD_REPLAY_COUNT:
        raise ValueError('v3 expert replay count differs')
    print('[v3] Dataset verified:', manifest['counts'], 'coverage:', manifest['coverage'], flush=True)
    return manifest
