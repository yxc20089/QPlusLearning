"""Hard-disagreement mining for native-v4; all game/teacher source stays pinned.

Generation and verification use the native CPU engine. Only ``probe_candidates``
and ``collect_learner`` call the frozen learner supplied by the notebook.
Every selected request is independent; full action/state/transition evidence is
retained separately. Insufficient hard examples fail rather than becoming easy
teacher-suffix examples. The large target mixture has not been generated here.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import signal
import re
import time
import zipfile
import uuid

from api_client import distribution
from gameplay_benchmark import BenchmarkEngine, SPEC, benchmark_gameplay, diagnostic_record, strata, summarize
from pacman_lab import ROOT, body, destination, distances, position
from planner_data import RECIPE as PREVIOUS_RECIPE
from teacher_data import prefix_move, COLLECTION
from teacher_validation import (VALIDATION, episode_quality_failures, fingerprint,
                                require_qualified_teacher, source_hashes)
from v3_data import cpu_capacity, recovery_workers, reserved_seeds, old_expert_records

PREFIX = 'pacman-native-v4'
RECIPE = {**PREVIOUS_RECIPE, 'replay': 304}
TARGETS = {'train': {'hard': 3660, 'informative': 1220, 'replay': 912},
           'development': {'hard': 384, 'informative': 128, 'replay': 0}}
MINIMUMS = {'train': {'ghost_intercept': 300, 'power_expiry': 120, 'food_dry': 180,
                     'sparse_food': 180, 'post_respawn': 120, 'anticipatory_harm': 300,
                     'expiry_during_action_with_near_ghost': 24},
            'development': {'ghost_intercept': 48, 'power_expiry': 16, 'food_dry': 24,
                            'sparse_food': 24, 'post_respawn': 16, 'anticipatory_harm': 48,
                            'expiry_during_action_with_near_ghost': 4}}
SOURCE_MIX = {'off_policy': .6, 'learner': .4}
DEFAULT_TRAIN_SEEDS = [400001 + 1009*i for i in range(8)]
DEFAULT_DEVELOPMENT_SEEDS = [460003 + 1013*i for i in range(4)]
HARVEST = {'roots_per_episode': 64, 'perturb_roots_per_episode': 12,
           'perturb_turns': 4, 'minimum_turn': 12, 'max_windows_per_case': 4,
           'max_selected_per_family': 128, 'modes': ['normal', 'post_respawn']}
HARM = {'minimum_extra_decisions': 8, 'minimum_extra_frames': 60,
        'minimum_extra_dry_decisions': 16}
_CANCELLED = None


def _load(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require_v3_identity(identity):
    if (identity.get('checkpoint_name') != 'kev-4b-pacman-native-v3'
            or identity.get('base') != 'Qwen/Qwen3.5-4B-Base'
            or identity.get('base_revision') != '1001bb4d826a52d1f399e183466143f4da7b741b'
            or identity.get('stage') != 'pacman' or not identity.get('pacman_fine_tuned')
            or identity.get('optimizer_steps', 0) <= 0
            or any(not isinstance(identity.get(k), str) or len(identity[k]) != 64
                   for k in ('checkpoint_sha256', 'adapter_sha256'))):
        raise ValueError('Use the completed native-v3 LoRA/head for v4 probes/rollouts')


def check_seed_splits(train, development, directory=None):
    train, development = list(train), list(development)
    if (not train or not development or len(set(train)) != len(train)
            or len(set(development)) != len(development)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in train+development)
            or set(train) & set(development)):
        raise ValueError('Use nonempty, unique, disjoint positive training/development seeds')
    reserved = reserved_seeds()
    # This band belongs to the frozen full-game evaluation family, including
    # unused neighboring seeds. Never mine a new training case from it.
    reserved.update(range(50000, 51000))
    reserved.update([300017, 300029, 300043, 300059, 310019, 310033])
    prior = ROOT/'data/pacman-native-v3-manifest.json'
    if prior.exists():
        manifest = _load(prior)
        reserved.update(manifest.get('train_seeds', [])); reserved.update(manifest.get('development_seeds', []))
    if (set(train) | set(development)) & reserved:
        raise ValueError('v4 generation seeds overlap benchmark, qualification or prior data')
    if directory and (Path(directory)/'collection.json').exists():
        for seeds in _load(Path(directory)/'collection.json').get('episode_sets', {}).values():
            if set(train) & set(seeds['development']) or set(development) & set(seeds['train']):
                raise ValueError('An episode family cannot change partition between collection sources')


def _configuration(directory, qualification_path):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    teacher = require_qualified_teacher(qualification_path)
    fixed = {'version': PREFIX, 'protocol': SPEC, 'teacher_options': teacher['options'],
             'source_sha256': source_hashes(), 'generator_sha256': _hash(Path(__file__)),
             'qualification_sha256': _hash(qualification_path), 'targets': TARGETS,
             'source_mix': SOURCE_MIX, 'harvest': HARVEST, 'harm_thresholds': HARM,
             'minimums': MINIMUMS}
    path = directory/'collection.json'
    if path.exists():
        config = _load(path)
        if any(config.get(k) != v for k, v in fixed.items()):
            raise ValueError('Frozen v4 collection inputs/source changed; use a new directory')
    else:
        config = {**fixed, 'episode_sets': {}}
        _save(path, config)
    return config


def _assert_configuration(directory):
    config = _load(Path(directory)/'collection.json')
    if config['version'] != PREFIX or config['source_sha256'] != source_hashes() or config['generator_sha256'] != _hash(Path(__file__)):
        raise ValueError('v4 source/configuration changed after collection')
    return config


def _set_seeds(directory, config, stage, train, development, wave_id='wave-001', parameters=None):
    if not isinstance(wave_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', wave_id):
        raise ValueError('wave_id must be a short identifier containing letters, numbers, hyphens or underscores')
    check_seed_splits(train, development, directory)
    key = f'{stage}/{wave_id}'
    selected = {'train': list(train), 'development': list(development), 'parameters': parameters or {}}
    existing = config['episode_sets'].get(key)
    if existing is not None and existing != selected:
        raise ValueError('Saved wave uses different seeds/limits; append a new wave_id')
    config['episode_sets'][key] = selected
    _save(Path(directory)/'collection.json', config)


def _harvest_limits(roots_per_episode, perturb_roots_per_episode, perturb_turns):
    values = dict(roots_per_episode=roots_per_episode, perturb_roots_per_episode=perturb_roots_per_episode,
                  perturb_turns=perturb_turns)
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'perturb_roots_per_episode' else 1):
            raise ValueError(f'{key} must be an integer within its positive range')
    if perturb_roots_per_episode > roots_per_episode or perturb_turns > 16:
        raise ValueError('Perturb at most the harvested root count, with at most 16 legal actions per branch')
    return values


def _initialize_worker(cancelled):
    global _CANCELLED
    _CANCELLED = cancelled
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _check_cancel():
    if _CANCELLED is not None and _CANCELLED.is_set():
        raise InterruptedError('v4 job cancelled; incomplete receipts will be retried')


def completed_files(directory):
    """Relative, committed files only. Mutable indices/configs are included."""
    directory = Path(directory); result = []
    for path in directory.glob('*.json'):
        if not path.name.endswith('.tmp'):
            _load(path); result.append(path.relative_to(directory).as_posix())
    for folder in (directory/'episodes').glob('*'):
        receipt = folder/'attempt.json'
        if receipt.exists():
            value = _load(receipt)
            for name, digest in value.get('files', {}).items():
                path = directory/name
                if not path.is_file() or _hash(path) != digest:
                    raise ValueError('Committed episode is missing or changed: '+name)
                result.append(name)
            result.append(receipt.relative_to(directory).as_posix())
            if (folder/'harvest.json').exists():
                _load(folder/'harvest.json'); result.append((folder/'harvest.json').relative_to(directory).as_posix())
    for folder in (directory/'cases').glob('*'):
        for name in ('candidate.json', 'probe.json', 'windows.json'):
            if (folder/name).is_file():
                _load(folder/name); result.append((folder/name).relative_to(directory).as_posix())
        receipt = folder/'verification.json'
        if receipt.exists():
            value = _load(receipt)
            for name, digest in value.get('files', {}).items():
                path = directory/name
                if not path.is_file() or _hash(path) != digest:
                    raise ValueError('Committed verification is missing or changed: '+name)
                result.append(name)
            result.append(receipt.relative_to(directory).as_posix())
    return sorted(set(result))


def _backup(directory, backup_directory):
    if backup_directory is not None:
        from collection_backup import backup_collection
        return backup_collection(directory, backup_directory, files=completed_files(directory))


def _run_jobs(jobs, worker, directory, stage, workers=None, backup_directory=None, backup_seconds=300):
    if backup_seconds <= 0: raise ValueError('backup_seconds must be positive')
    directory = Path(directory); _backup(directory, backup_directory)
    results, errors = [], []
    if not jobs:
        return {'stage': stage, 'status': 'complete', 'completed': 0, 'errors': []}
    workers = recovery_workers(workers, len(jobs)); context = multiprocessing.get_context()
    cancelled = context.Event(); started = time.monotonic(); last_backup = started
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                               initializer=_initialize_worker, initargs=(cancelled,))
    queued = iter(jobs); futures = {}
    def submit():
        job = next(queued, None)
        if job is not None: futures[pool.submit(worker, job)] = job
    def progress():
        _save(directory/f'{stage}-index.json', {'stage': stage, 'completed_this_run': len(results),
            'expected_this_run': len(jobs), 'workers': workers, 'errors': errors,
            'elapsed_seconds': time.monotonic()-started})
    try:
        for _ in range(workers): submit()
        while futures:
            ready, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
            if not ready:
                print(f'[v4/{stage}] {len(results)}/{len(jobs)} jobs; {workers} CPU workers; '
                      f'{(time.monotonic()-started)/60:.1f} min elapsed', flush=True)
            for future in ready:
                job = futures.pop(future)
                try: result = future.result()
                except Exception as error: errors.append({'id': job['id'], 'error': f'{type(error).__name__}: {error}'})
                else:
                    results.append(result)
                    print(f'[v4/{stage}] {len(results)}/{len(jobs)} {job["id"]}: '
                          f'{result.get("classification", result.get("accepted"))}', flush=True)
                progress(); submit()
            if backup_directory is not None and time.monotonic()-last_backup >= backup_seconds:
                _backup(directory, backup_directory); last_backup = time.monotonic()
    except BaseException:
        cancelled.set()
        for future in futures: future.cancel()
        progress()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        _backup(directory, backup_directory)
    return {'stage': stage, 'status': 'errors' if errors else 'complete',
            'completed': len(results), 'errors': errors, 'index': str(directory/f'{stage}-index.json')}


def cohort_flags(state, risks=None):
    routes = distances(state['maze'], position(state['player']))
    danger = min((routes.get(position(g), 999) for g in state['ghosts']
                  if g['mode'] == 'outside' and not g['frightened']), default=999)
    near_ghost = min((routes.get(position(g), 999) for g in state['ghosts'] if g['mode'] == 'outside'), default=999) <= 3
    power = state['timing']['power']['remaining_frames']
    return {'ghost_intercept': danger <= 6,
            'power_expiry': state['frightened'] and 0 < power <= 120,
            'expiry_during_action_with_near_ghost': bool(state['frightened'] and near_ghost and 0 < power <=
                max((r['action_frames'] for r in (risks or {}).values()), default=0)),
            'food_dry': state.get('decisions_since_last_pellet', 0) >= 24,
            'food_junction': len(state['legal_moves']) >= 3,
            'sparse_food': state['pellets_remaining'] <= 30,
            'post_respawn': state.get('life_epoch', 0) > 0,
            'immediate_critical': bool(risks and any(v['life_lost'] for v in risks.values())
                                      and any(not v['life_lost'] for v in risks.values()))}


def _one_hot(state, move):
    return {'answers': {'move': {'choice': move, 'probabilities': {d: float(d == move) for d in state['legal_moves']}}}}


def _replay_prefix(engine, candidate):
    state = engine.request('reset', options={'seed': candidate['seed'], 'level': candidate['level'], 'action_version': 2})
    hashes = candidate['prefix_state_sha256']
    if len(hashes) != len(candidate['prefix_actions'])+1: raise ValueError('Malformed native prefix evidence')
    for index, move in enumerate(candidate['prefix_actions']):
        _check_cancel()
        if fingerprint(state) != hashes[index]: raise ValueError('Native prefix observation differs')
        result = engine.step(move); state = result['state']
        if result['outcome'] == 'life_lost':
            continuation = engine.request('continue')
            if continuation['status'] != 'playing': raise ValueError('Candidate prefix passes native game-over')
            state = continuation['state']
        elif result['outcome']: raise ValueError('Candidate prefix passes a maze clear')
    if fingerprint(state) != hashes[-1] or state != candidate['request']['state']:
        raise ValueError('Candidate is not the exact native state at its prefix boundary')
    return state


def _candidate(directory, *, split, seed, level, origin, source, state, actions, hashes,
               case_role='root', parent_case=None, window_kind=None, teacher_hint=None):
    family = f'level-{level}-seed-{seed}'
    semantic = {'family': family, 'state': fingerprint(state), 'origin': origin, 'case_role': case_role,
                'parent_case': parent_case, 'window_kind': window_kind, 'source': source}
    identifier = fingerprint(semantic)[:24]
    value = {'id': identifier, 'split': split, 'seed': seed, 'level': level, 'family': family,
             'origin': origin, 'source': source, 'case_role': case_role, 'parent_case': parent_case,
             'window_kind': window_kind, 'request': body(state), 'state_sha256': fingerprint(state),
             'prefix_actions': list(actions), 'prefix_state_sha256': list(hashes),
             'cohorts': cohort_flags(state), 'teacher_choice_hint': teacher_hint}
    path = Path(directory)/'cases'/identifier/'candidate.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    try:
        try: os.link(temporary, path)
        except FileExistsError: pass
    finally: temporary.unlink()
    if path.exists():
        saved = _load(path)
        if saved != value: raise ValueError('Candidate ID collision or changed immutable source')
    return identifier


def _episode_job(job):
    """Coherent teacher episode, optionally preceded by legal native respawn."""
    directory = Path(job['directory']); folder = directory/'episodes'/job['id']; folder.mkdir(parents=True, exist_ok=True)
    temporary = folder/'trace.jsonl.tmp'; rows = []; teacher_rows = []; life = stale = frames = prefix = 0
    teacher_started = job['mode'] == 'normal'; status = 'decision_cap'
    with BenchmarkEngine() as engine, temporary.open('w') as handle:
        state = engine.request('reset', options={'seed': job['seed'], 'level': job['level'], 'action_version': 2})
        for _ in range(SPEC['max_decisions']):
            _check_cancel(); risks = engine.request('risks')
            if teacher_started:
                plan = engine.request('teacher', options=job['options']); move = plan['choice']; controller = 'qualified_teacher'
            else:
                if prefix >= COLLECTION['prefix_max_decisions']: raise RuntimeError('Native post-respawn prefix did not finish')
                move = prefix_move(state, risks); plan = None; controller = 'legal_perturbation'; prefix += 1
            response = _one_hot(state, move); step = engine.step(move)
            diag = diagnostic_record(state, response, step, risks, life); rows.append(diag)
            if teacher_started: teacher_rows.append({**diag, 'life_index': life-(1 if job['mode'] == 'post_respawn' else 0)})
            handle.write(json.dumps({'request': body(state), 'response': response, 'diagnostics': diag,
                                     'controller': controller, 'teacher': plan}, separators=(',', ':'))+'\n'); handle.flush()
            frames += step['action_frames']; stale = stale+1 if step['state']['pellets_remaining'] == state['pellets_remaining'] else 0
            state = step['state']
            if step['outcome'] == 'level_cleared': status = 'level_cleared'; break
            if step['outcome'] == 'life_lost':
                continuation = engine.request('continue')
                if continuation['status'] != 'playing': status = 'game_over'; break
                state = continuation['state']; life += 1; stale = 0; teacher_started = True
            if frames >= SPEC['max_simulation_frames']: status = 'simulation_frame_cap'; break
            if teacher_started and stale >= SPEC['max_no_pellet_decisions']: status = 'no_progress_watchdog'; break
    trace = folder/'trace.jsonl'; temporary.replace(trace)
    measured = summarize(teacher_rows or rows, status)
    failures = episode_quality_failures(measured)
    if not teacher_rows: failures.append('no teacher-controlled suffix')
    if measured['life_losses'] > VALIDATION['gates']['maximum_life_losses_total']: failures.append('too many teacher life losses')
    receipt = {'id': job['id'], 'job_sha256': fingerprint({k:v for k,v in job.items() if k != 'directory'}),
               'split': job['split'], 'seed': job['seed'], 'level': job['level'], 'mode': job['mode'],
               'accepted': not failures, 'failures': failures, 'metrics': measured,
               'full_metrics': summarize(rows, status), 'files': {trace.relative_to(directory).as_posix(): _hash(trace)}}
    _save(folder/'attempt.json', receipt)
    return receipt


def _root_indices(rows, budget, minimum_turn=0):
    """Round-robin rare risk/progress cohorts before taking distributed residual roots."""
    flags = {i: cohort_flags(r['request']['state'], r['diagnostics']['immediate_counterfactuals'])
             for i, r in enumerate(rows) if r['request']['state']['turn'] >= minimum_turn
             and len(r['request']['state']['legal_moves']) >= 2 and r.get('controller', 'qualified_teacher') == 'qualified_teacher'}
    queues = []
    for cohort in ('expiry_during_action_with_near_ghost', 'immediate_critical', 'power_expiry',
                   'ghost_intercept', 'food_dry', 'sparse_food', 'post_respawn', 'food_junction'):
        indices = [i for i in flags if flags[i][cohort]]
        if indices:
            # Spread across a cohort's full episode rather than taking its first contiguous prefix.
            queues.append([indices[j*len(indices)//min(budget, len(indices))] for j in range(min(budget, len(indices)))])
    result, seen = [], set()
    while queues and len(result) < budget:
        for queue in queues:
            while queue and queue[0] in seen: queue.pop(0)
            if queue and len(result) < budget:
                index = queue.pop(0); result.append(index); seen.add(index)
        queues = [q for q in queues if q]
    return result


def _harvest_episode(directory, receipt, perturb=True, limits=None):
    limits = limits or HARVEST
    directory = Path(directory); trace_name = next(iter(receipt['files'])); rows = _read_rows(directory/trace_name)
    eligible = _root_indices(rows, limits['roots_per_episode'], HARVEST['minimum_turn'])
    actions = [r['diagnostics']['choice'] for r in rows]
    hashes = [fingerprint(r['request']['state']) for r in rows]
    source = {'kind': 'teacher_episode', 'episode': receipt['id'], 'trace': trace_name,
              'trace_sha256': receipt['files'][trace_name]}
    ids = []
    for i in eligible:
        state = rows[i]['request']['state']
        ids.append(_candidate(directory, split=receipt['split'], seed=receipt['seed'], level=receipt['level'],
            origin='off_policy', source={**source, 'root_index': i}, state=state, actions=actions[:i],
            hashes=hashes[:i+1], teacher_hint=rows[i]['diagnostics']['choice']))
    if perturb:
        for i in eligible[:limits['perturb_roots_per_episode']]:
            root = rows[i]['request']['state']; original = rows[i]['diagnostics']['choice']
            risks = rows[i]['diagnostics']['immediate_counterfactuals']
            alternatives = [a for a in root['legal_moves'] if a != original and not risks[a]['life_lost']]
            if not alternatives: continue
            for kind in ('wrong_turn', 'delayed_retreat'):
                candidate = {'seed': receipt['seed'], 'level': receipt['level'], 'prefix_actions': actions[:i],
                             'prefix_state_sha256': hashes[:i+1], 'request': body(root)}
                with BenchmarkEngine() as engine:
                    state = _replay_prefix(engine, candidate); prefix_actions = list(actions[:i]); prefix_hashes = list(hashes[:i+1])
                    for offset in range(limits['perturb_turns']):
                        _check_cancel()
                        heading = state['player']['heading']
                        move = heading if kind == 'delayed_retreat' and heading in state['legal_moves'] else alternatives[0] if offset == 0 else heading
                        if move not in state['legal_moves']: break
                        step = engine.step(move); prefix_actions.append(move); state = step['state']
                        if step['outcome'] == 'level_cleared': break
                        if step['outcome'] == 'life_lost':
                            continuation = engine.request('continue')
                            if continuation['status'] != 'playing': break
                            state = continuation['state']
                        prefix_hashes.append(fingerprint(state))
                        ids.append(_candidate(directory, split=receipt['split'], seed=receipt['seed'], level=receipt['level'],
                            origin='off_policy', source={**source, 'root_index': i, 'perturbation': kind, 'perturbation_actions': prefix_actions[i:]},
                            state=state, actions=prefix_actions, hashes=prefix_hashes))
    return ids


def _episode_harvest_job(job):
    """Per-episode receipt commits survive interruption between simulation and mining."""
    directory = Path(job['directory']); folder = directory/'episodes'/job['id']
    if (folder/'attempt.json').exists():
        receipt = _load(folder/'attempt.json')
        if receipt['job_sha256'] != fingerprint({k:v for k,v in job.items() if k != 'directory'}):
            raise ValueError('Cached teacher episode inputs changed')
    else: receipt = _episode_job(job)
    ids = _harvest_episode(directory, receipt, job['perturb'], job['limits']) if receipt['accepted'] else []
    completed = {'id': job['id'], 'attempt_sha256': _hash(folder/'attempt.json'),
                 'candidates': sorted(set(ids)), 'accepted': receipt['accepted']}
    _save(folder/'harvest.json', completed)
    return completed


def _candidate_index(directory):
    directory = Path(directory); entries = []
    for path in sorted((directory/'cases').glob('*/candidate.json')):
        c = _load(path); entries.append({'id': c['id'], 'split': c['split'], 'family': c['family'],
            'origin': c['origin'], 'state_sha256': c['state_sha256'], 'case_role': c['case_role'],
            'candidate_sha256': _hash(path), 'probed': path.with_name('probe.json').exists(),
            'verified': path.with_name('verification.json').exists()})
    _save(directory/'candidate-index.json', {'entries': entries})
    return entries


def harvest_scenarios(directory, qualification_path, train_seeds=DEFAULT_TRAIN_SEEDS,
                      development_seeds=DEFAULT_DEVELOPMENT_SEEDS, workers=None,
                      backup_directory=None, backup_seconds=300, perturb=True, wave_id='wave-001',
                      roots_per_episode=64, perturb_roots_per_episode=12, perturb_turns=4):
    limits = _harvest_limits(roots_per_episode, perturb_roots_per_episode, perturb_turns)
    config = _configuration(directory, qualification_path)
    _set_seeds(directory, config, 'off_policy', train_seeds, development_seeds, wave_id, {**limits, 'perturb': perturb})
    jobs = []
    for split, seeds in [('train', train_seeds), ('development', development_seeds)]:
        for level in SPEC['levels']:
            for seed in seeds:
                for mode in HARVEST['modes']:
                    identifier = f'off-policy-{wave_id}-{split}-level-{level}-seed-{seed}-{mode}'
                    job = {'id': identifier, 'split': split, 'level': level, 'seed': seed, 'mode': mode,
                           'options': config['teacher_options'], 'directory': str(directory),
                           'limits': limits, 'perturb': perturb}
                    path = Path(directory)/'episodes'/identifier/'attempt.json'
                    if path.exists():
                        receipt = _load(path)
                        if receipt['job_sha256'] != fingerprint({k:v for k,v in job.items() if k != 'directory'}):
                            raise ValueError('Cached episode job changed')
                        harvested = path.with_name('harvest.json')
                        if not harvested.exists(): jobs.append(job)
                        elif _load(harvested)['attempt_sha256'] != _hash(path): raise ValueError('Cached harvest source changed')
                    else: jobs.append(job)
    result = _run_jobs(jobs, _episode_harvest_job, directory, 'harvest', workers, backup_directory, backup_seconds)
    try:
        entries = _candidate_index(directory)
        result.update(candidates=len(entries), pending_probe_count=sum(not e['probed'] for e in entries))
    finally: _backup(directory, backup_directory)
    return result


def collect_learner(directory, model_info, predict, qualification_path,
                    train_seeds=DEFAULT_TRAIN_SEEDS, development_seeds=DEFAULT_DEVELOPMENT_SEEDS,
                    backup_directory=None, backup_seconds=300, wave_id='wave-001', roots_per_episode=64):
    _harvest_limits(roots_per_episode, 0, 1)
    config = _configuration(directory, qualification_path)
    _set_seeds(directory, config, 'learner', train_seeds, development_seeds, wave_id, {'roots_per_episode': roots_per_episode})
    identity = copy.deepcopy(model_info()); require_v3_identity(identity)
    if config.get('probe_checkpoint') not in (None, identity): raise ValueError('Different v3 adapter in the collection')
    config['probe_checkpoint'] = identity; _save(Path(directory)/'collection.json', config)
    _backup(directory, backup_directory); completed = 0; last_backup = time.monotonic()
    try:
        for split, seeds in [('train', train_seeds), ('development', development_seeds)]:
            for level in SPEC['levels']:
                for seed in seeds:
                    identifier = f'learner-{wave_id}-{split}-level-{level}-seed-{seed}'; folder = Path(directory)/'episodes'/identifier
                    path = folder/'attempt.json'
                    if path.exists():
                        receipt = _load(path)
                        if receipt.get('checkpoint') != identity: raise ValueError('Saved learner game uses another adapter')
                    else:
                        report = benchmark_gameplay(predict=predict, model_info=model_info,
                            trace_dir=folder, spec={**SPEC, 'levels': [level], 'seeds': [seed]})
                        episode = report['episodes'][0]; trace = Path(episode['trace'])
                        receipt = {**episode, 'id': identifier, 'split': split, 'checkpoint': identity,
                            'accepted': True, 'kind': 'learner', 'files': {trace.relative_to(Path(directory)).as_posix(): _hash(trace)}}
                        _save(path, receipt); completed += 1
                    trace_name = next(iter(receipt['files'])); rows = _read_rows(Path(directory)/trace_name)
                    actions = [r['diagnostics']['choice'] for r in rows]; hashes = [fingerprint(r['request']['state']) for r in rows]
                    from v3_data import select_roots
                    selected = set(_root_indices(rows, roots_per_episode))
                    selected.update(select_roots(rows, max(1, roots_per_episode-len(selected))))
                    selected = sorted(selected)[:roots_per_episode]
                    for i in selected:
                        _candidate(directory, split=split, seed=seed, level=level, origin='learner',
                            source={'kind': 'v3_episode', 'episode': identifier, 'trace': trace_name,
                                    'trace_sha256': receipt['files'][trace_name], 'root_index': i, 'checkpoint': identity},
                            state=rows[i]['request']['state'], actions=actions[:i], hashes=hashes[:i+1])
                    _candidate_index(directory)
                    if backup_directory is not None and time.monotonic()-last_backup >= backup_seconds:
                        _backup(directory, backup_directory); last_backup = time.monotonic()
    finally: _backup(directory, backup_directory)
    return {'stage': 'learner', 'status': 'complete', 'completed': completed,
            'pending_probe_count': sum(not e['probed'] for e in _candidate_index(directory)), 'errors': []}


def probe_candidates(directory, model_info, predict_batch, batch_size=16,
                     backup_directory=None, backup_seconds=300):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1: raise ValueError('batch_size must be positive')
    config = _assert_configuration(directory); identity = copy.deepcopy(model_info()); require_v3_identity(identity)
    if config.get('probe_checkpoint') not in (None, identity): raise ValueError('Probes must keep the same completed v3 adapter')
    config['probe_checkpoint'] = identity; _save(Path(directory)/'collection.json', config)
    _backup(directory, backup_directory); last_backup = time.monotonic(); done = 0
    paths = sorted((Path(directory)/'cases').glob('*/candidate.json'))
    pending = []
    for path in paths:
        if path.with_name('probe.json').exists():
            p = _load(path.with_name('probe.json'))
            if p['candidate_sha256'] != _hash(path) or p['checkpoint'] != identity: raise ValueError('Cached v3 probe source changed')
        else: pending.append(path)
    try:
        for begin in range(0, len(pending), batch_size):
            batch = pending[begin:begin+batch_size]
            if model_info() != identity: raise RuntimeError('Active adapter changed during v4 probing')
            responses = predict_batch([_load(p)['request'] for p in batch])
            if not isinstance(responses, (list, tuple)) or len(responses) != len(batch): raise ValueError('Probe callback must return one ordered response per request')
            if model_info() != identity: raise RuntimeError('Active adapter changed during v4 probing')
            validated = []
            for path, response in zip(batch, responses):
                candidate = _load(path); probabilities = distribution(response['answers']['move'], candidate['request']['questions']['move']['criteria'])
                if response.get('active_checkpoint', identity) != identity: raise ValueError('Probe response uses another adapter')
                validated.append((path, {'candidate_sha256': _hash(path), 'checkpoint': identity,
                    'response': response, 'probabilities': probabilities, 'choice': response['answers']['move']['choice']}))
            for path, probe in validated: _save(path.with_name('probe.json'), probe); done += 1
            _save(Path(directory)/'probe-index.json', {'checkpoint': identity, 'completed_total': len(paths)-len(pending)+done,
                    'pending': len(pending)-done})
            if backup_directory is not None and time.monotonic()-last_backup >= backup_seconds:
                _backup(directory, backup_directory); last_backup = time.monotonic()
    finally: _backup(directory, backup_directory)
    entries = _candidate_index(directory)
    return {'stage': 'probe', 'status': 'complete', 'completed': done,
            'pending_probe_count': sum(not e['probed'] for e in entries),
            'pending_verification_count': sum(e['probed'] and not e['verified'] for e in entries), 'errors': []}


def _continuation(candidate, options, trace, forced_first=None, caps=SPEC):
    temporary = Path(str(trace)+'.tmp'); temporary.parent.mkdir(parents=True, exist_ok=True)
    rows = []; life = frames = stale = 0; status = 'decision_cap'
    with BenchmarkEngine() as engine, temporary.open('w') as handle:
        state = _replay_prefix(engine, candidate)
        for index in range(caps['max_decisions']):
            _check_cancel(); risks = engine.request('risks')
            plan = engine.request('teacher', options=options) if index or forced_first is None else None
            move = forced_first if index == 0 and forced_first is not None else plan['choice']
            response = _one_hot(state, move); step = engine.step(move)
            diag = diagnostic_record(state, response, step, risks, life); rows.append(diag)
            handle.write(json.dumps({'state_sha256': fingerprint(state), 'choice': move, 'diagnostics': diag}, separators=(',', ':'))+'\n'); handle.flush()
            frames += step['action_frames']; stale = stale+1 if step['state']['pellets_remaining'] == state['pellets_remaining'] else 0
            state = step['state']
            if step['outcome'] == 'level_cleared': status = 'level_cleared'; break
            if step['outcome'] == 'life_lost':
                continuation = engine.request('continue')
                if continuation['status'] != 'playing': status = 'game_over'; break
                state = continuation['state']; life += 1; stale = 0
            if stale >= caps['max_no_pellet_decisions']: status = 'no_progress_watchdog'; break
            if frames >= caps['max_simulation_frames']: status = 'simulation_frame_cap'; break
    temporary.replace(trace)
    return summarize(rows, status)


def _verify_branch(candidate, trace, metrics):
    proof = _read_rows(trace); measured = []; life = 0; status = None
    with BenchmarkEngine() as engine:
        state = _replay_prefix(engine, candidate)
        for index, row in enumerate(proof):
            _check_cancel()
            if fingerprint(state) != row['state_sha256']: raise ValueError('Continuation replay state changed')
            risks = engine.request('risks'); step = engine.step(row['choice'])
            diag = diagnostic_record(state, _one_hot(state, row['choice']), step, risks, life)
            if diag != row['diagnostics']: raise ValueError('Continuation replay transition changed')
            measured.append(diag); state = step['state']
            if step['outcome'] == 'level_cleared': status = 'level_cleared'
            if step['outcome'] == 'life_lost':
                continuation = engine.request('continue')
                if continuation['status'] != 'playing': status = 'game_over'
                else: state = continuation['state']; life += 1
            if status and index != len(proof)-1: raise ValueError('Continuation extends beyond native terminal state')
    if fingerprint(summarize(measured, status or metrics['outcome'])) != fingerprint(metrics):
        raise ValueError('Continuation metrics changed')
    return True


def negative_evidence(teacher_metrics, learner_metrics, risks, teacher_choice, learner_choice, thresholds=HARM):
    """Same subsequent controller isolates the first move; preference alone is not harm."""
    delta = {name: learner_metrics[name]-teacher_metrics[name] for name in
             ('life_losses', 'decisions', 'simulation_frames', 'longest_no_pellet_decisions', 'loop_decisions')}
    if risks[learner_choice]['life_lost'] and not risks[teacher_choice]['life_lost']:
        reason = 'immediate_fatal_with_safe_teacher'
    elif learner_metrics['life_losses'] > teacher_metrics['life_losses']:
        reason = 'extra_life_loss_with_same_continuation_controller'
    elif teacher_metrics['level_cleared'] and not learner_metrics['level_cleared']:
        reason = 'incomplete_forced_first_branch_with_same_controller'
    elif (teacher_metrics['level_cleared'] and learner_metrics['level_cleared']
          and delta['life_losses'] == 0
          and delta['decisions'] >= thresholds['minimum_extra_decisions']
          and delta['simulation_frames'] >= thresholds['minimum_extra_frames']):
        reason = 'verified_route_cost'
    elif (delta['longest_no_pellet_decisions'] >= thresholds['minimum_extra_dry_decisions']
          and delta['simulation_frames'] >= thresholds['minimum_extra_frames']
          and delta['life_losses'] == 0 and teacher_metrics['level_cleared']):
        reason = 'verified_extended_pellet_stall'
    else: reason = None
    return {'harmful': reason is not None, 'kind': reason, 'after_minus_teacher': delta,
            'comparison': 'Forced teacher-first versus frozen-v3-first; the same fixed teacher controls both afterward. Finite deterministic episode continuation, not global action optimality.'}


def _verification_job(job):
    directory = Path(job['directory']); folder = directory/'cases'/job['id']
    candidate = _load(folder/'candidate.json'); probe = _load(folder/'probe.json')
    if _hash(folder/'candidate.json') != probe['candidate_sha256']: raise ValueError('Probe input changed')
    with BenchmarkEngine() as engine:
        state = _replay_prefix(engine, candidate); risks = engine.request('risks')
        plan = engine.request('teacher', options=job['options']); teacher_choice = plan['choice']
        if engine.request('observe') != state: raise ValueError('Teacher query altered candidate root')
    probability = probe['probabilities']; learner_choice = probe['choice']
    margin = math.log(max(probability[learner_choice], 1e-8)/max(probability[teacher_choice], 1e-8))
    receipt = {'id': candidate['id'], 'candidate_sha256': _hash(folder/'candidate.json'),
        'probe_sha256': _hash(folder/'probe.json'), 'checkpoint': probe['checkpoint'],
        'teacher_choice': teacher_choice, 'learner_choice': learner_choice,
        'teacher_probability': probability[teacher_choice], 'learner_probability': probability[learner_choice],
        'learner_minus_teacher_log_probability': margin, 'immediate_counterfactuals': risks,
        'teacher_plan': plan, 'source_sha256': job['source_sha256'], 'options': job['options'],
        'protocol': job.get('caps', SPEC),
        'accepted': False, 'classification': 'rejected', 'failures': [], 'files': {}, 'native_verified': False}
    if teacher_choice == learner_choice and candidate['case_role'] == 'root':
        receipt['failures'] = ['v3 already selects the teacher action; not a hard disagreement']
    else:
        teacher_trace = folder/'teacher.jsonl'
        teacher_metrics = _continuation(candidate, job['options'], teacher_trace, teacher_choice, job.get('caps', SPEC))
        _verify_branch(candidate, teacher_trace, teacher_metrics)
        receipt['teacher_metrics'] = teacher_metrics
        receipt['files'][teacher_trace.relative_to(directory).as_posix()] = _hash(teacher_trace)
        receipt['failures'] = episode_quality_failures(teacher_metrics)
        if teacher_metrics['life_losses'] > VALIDATION['gates']['maximum_life_losses_total']:
            receipt['failures'].append('too many teacher life losses')
        if teacher_choice != learner_choice:
            negative_trace = folder/'v3-first-then-teacher.jsonl'
            negative_metrics = _continuation(candidate, job['options'], negative_trace, learner_choice, job.get('caps', SPEC))
            _verify_branch(candidate, negative_trace, negative_metrics)
            receipt['negative_metrics'] = negative_metrics
            receipt['files'][negative_trace.relative_to(directory).as_posix()] = _hash(negative_trace)
            receipt['negative_evidence'] = negative_evidence(teacher_metrics, negative_metrics, risks, teacher_choice, learner_choice)
        receipt['cohorts'] = cohort_flags(state, risks)
        receipt['cohorts']['anticipatory_harm'] = bool(not risks[learner_choice]['life_lost'] and
            receipt.get('negative_evidence', {}).get('kind') in
            ('extra_life_loss_with_same_continuation_controller', 'incomplete_forced_first_branch_with_same_controller'))
        receipt['native_verified'] = True
        if not receipt['failures']:
            if receipt.get('negative_evidence', {}).get('harmful'):
                receipt['accepted'] = True; receipt['classification'] = 'hard'
            elif candidate['case_role'] == 'informative':
                parent = directory/'cases'/candidate['parent_case']/'verification.json'
                if parent.exists() and _load(parent).get('classification') == 'hard' and _load(parent).get('accepted'):
                    receipt['accepted'] = True; receipt['classification'] = 'informative'
                else: receipt['failures'].append('informative window lacks an admitted hard parent')
            else: receipt['failures'].append('different useful action or insignificant counterfactual cost; not a negative')
    _save(folder/'verification.json', receipt)
    return receipt


def _derive_windows(directory, candidate, receipt):
    """At most four windows; level-one windows never spawn descendants."""
    if candidate['case_role'] != 'root' or not receipt['accepted'] or receipt['classification'] != 'hard': return []
    directory = Path(directory); ids = []; prefix = candidate['prefix_actions']; hashes = candidate['prefix_state_sha256']
    for offset in (1, 4, 8):
        target = len(prefix)-offset
        if target < 0: continue
        with BenchmarkEngine() as engine:
            engine.request('reset', options={'seed': candidate['seed'], 'level': candidate['level'], 'action_version': 2})
            state = engine.request('observe')
            for i, move in enumerate(prefix[:target]):
                _check_cancel()
                if fingerprint(state) != hashes[i]: raise ValueError('Lead-in prefix changed')
                step = engine.step(move); state = step['state']
                if step['outcome'] == 'life_lost': state = engine.request('continue')['state']
            if state['life_epoch'] != candidate['request']['state']['life_epoch']: continue
            if fingerprint(state) != hashes[target]: raise ValueError('Lead-in boundary changed')
        ids.append(_candidate(directory, split=candidate['split'], seed=candidate['seed'], level=candidate['level'],
            origin=candidate['origin'], source={'kind': 'verified_lead_in', 'parent': candidate['id'],
                'parent_verification_sha256': _hash(directory/'cases'/candidate['id']/'verification.json'), 'offset': offset},
            state=state, actions=prefix[:target], hashes=hashes[:target+1], case_role='informative',
            parent_case=candidate['id'], window_kind='earlier_same_life_trap_or_progress'))
    trace = directory/'cases'/candidate['id']/'teacher.jsonl'; proof = _read_rows(trace)
    next_food = next((i for i,row in enumerate(proof) if row['diagnostics']['pellets_after'] < row['diagnostics']['pellets_before']), None)
    if next_food is not None and next_food > 0 and len(ids) < HARVEST['max_windows_per_case']:
        with BenchmarkEngine() as engine:
            state = _replay_prefix(engine, candidate)
            actions = list(prefix); augmented_hashes = list(hashes)
            for row in proof[:next_food]:
                _check_cancel()
                if fingerprint(state) != row['state_sha256']: raise ValueError('Recovery window changed')
                step = engine.step(row['choice']); actions.append(row['choice']); state = step['state']
                if step['outcome'] == 'life_lost': state = engine.request('continue')['state']
                augmented_hashes.append(fingerprint(state))
        ids.append(_candidate(directory, split=candidate['split'], seed=candidate['seed'], level=candidate['level'],
            origin=candidate['origin'], source={'kind': 'verified_recovery_progress', 'parent': candidate['id'],
                'parent_verification_sha256': _hash(directory/'cases'/candidate['id']/'verification.json'),
                'teacher_trace_sha256': _hash(trace), 'suffix_index': next_food}, state=state,
            actions=actions, hashes=augmented_hashes, case_role='informative', parent_case=candidate['id'],
            window_kind='retreat_to_first_pellet', teacher_hint=proof[next_food]['choice']))
    return ids


def _window_job(job):
    directory = Path(job['directory']); folder = directory/'cases'/job['id']
    candidate, receipt = _load(folder/'candidate.json'), _load(folder/'verification.json')
    ids = _derive_windows(directory, candidate, receipt)
    completed = {'id': job['id'], 'parent_verification_sha256': _hash(folder/'verification.json'),
        'candidates': {identifier: _hash(directory/'cases'/identifier/'candidate.json') for identifier in ids},
        'classification': 'bounded_windows'}
    _save(folder/'windows.json', completed)
    return completed


def verify_cases(directory, qualification_path, workers=None, backup_directory=None,
                 backup_seconds=300, create_windows=True):
    config = _configuration(directory, qualification_path); entries = _candidate_index(directory); jobs = []
    for entry in entries:
        folder = Path(directory)/'cases'/entry['id']
        if entry['verified']:
            receipt = _load(folder/'verification.json')
            if receipt['candidate_sha256'] != _hash(folder/'candidate.json') or receipt['probe_sha256'] != _hash(folder/'probe.json'):
                raise ValueError('Cached case verification inputs changed')
        elif entry['probed']:
            jobs.append({'id': entry['id'], 'directory': str(directory), 'options': config['teacher_options'],
                         'source_sha256': config['source_sha256']})
    result = _run_jobs(jobs, _verification_job, directory, 'verification', workers, backup_directory, backup_seconds)
    before = {e['id'] for e in entries}; window_jobs = []
    try:
        if create_windows:
            for path in sorted((Path(directory)/'cases').glob('*/verification.json')):
                receipt = _load(path); candidate = _load(path.with_name('candidate.json'))
                if candidate['case_role'] != 'root' or not receipt['accepted'] or receipt['classification'] != 'hard': continue
                completed = path.with_name('windows.json')
                if completed.exists():
                    saved = _load(completed)
                    if saved['parent_verification_sha256'] != _hash(path): raise ValueError('Window parent changed')
                    for identifier, digest in saved['candidates'].items():
                        if _hash(Path(directory)/'cases'/identifier/'candidate.json') != digest:
                            raise ValueError('Committed window candidate changed')
                else: window_jobs.append({'id': candidate['id'], 'directory': str(directory)})
            windows = _run_jobs(window_jobs, _window_job, directory, 'windows', workers, backup_directory, backup_seconds)
            result['errors'].extend(windows['errors'])
            if result['errors']: result['status'] = 'errors'
        entries = _candidate_index(directory)
        result.update(new_informative_candidates=len({e['id'] for e in entries}-before),
                      pending_probe_count=sum(not e['probed'] for e in entries),
                      pending_verification_count=sum(e['probed'] and not e['verified'] for e in entries))
    finally: _backup(directory, backup_directory)
    return result


def _record(candidate, receipt, config):
    request = copy.deepcopy(candidate['request']); request.pop('model', None)
    request['questions']['move'].update(label=receipt['teacher_choice'], src='pacman_v4_verified_same_state_teacher')
    request['_meta'] = {'id': candidate['id'], 'group_id': candidate['family'], 'split': candidate['split'],
        'origin': candidate['origin'], 'class': receipt['classification'], 'case_role': candidate['case_role'],
        'parent_case': candidate['parent_case'], 'window_kind': candidate['window_kind'],
        'cohorts': receipt['cohorts'], 'state_sha256': candidate['state_sha256'],
        'source': candidate['source'], 'probe_checkpoint_sha256': receipt['checkpoint']['checkpoint_sha256'],
        'probe_adapter_sha256': receipt['checkpoint']['adapter_sha256'],
        'learner_choice': receipt['learner_choice'], 'teacher_probability': receipt['teacher_probability'],
        'learner_probability': receipt['learner_probability'], 'margin': receipt['learner_minus_teacher_log_probability'],
        'negative_evidence': receipt.get('negative_evidence'), 'immediate_counterfactuals': receipt['immediate_counterfactuals'],
        'qualification_sha256': config['qualification_sha256'], 'native_verified': receipt['native_verified']}
    return request


def _validate_binding(candidate, probe, receipt, config, candidate_sha256, probe_sha256):
    """Recompute observed labels, margins/cohorts and harm instead of trusting counts."""
    probabilities = distribution(probe['response']['answers']['move'], candidate['request']['questions']['move']['criteria'])
    legal = candidate['request']['state']['legal_moves']; teacher_choice = receipt['teacher_choice']; learner_choice = probe['choice']
    if (candidate['request'] != body(candidate['request']['state'])
            or candidate['state_sha256'] != fingerprint(candidate['request']['state'])
            or candidate['family'] != f'level-{candidate["level"]}-seed-{candidate["seed"]}'
            or len(candidate['prefix_state_sha256']) != len(candidate['prefix_actions'])+1
            or candidate['prefix_state_sha256'][-1] != candidate['state_sha256']
            or receipt['candidate_sha256'] != candidate_sha256 or probe['candidate_sha256'] != candidate_sha256
            or receipt['probe_sha256'] != probe_sha256 or receipt['checkpoint'] != config['probe_checkpoint']
            or probe['checkpoint'] != config['probe_checkpoint'] or receipt['source_sha256'] != config['source_sha256']
            or receipt['options'] != config['teacher_options'] or receipt['protocol'] != SPEC
            or receipt['learner_choice'] != learner_choice or learner_choice != probe['response']['answers']['move']['choice']
            or teacher_choice not in legal or set(receipt['immediate_counterfactuals']) != set(legal)
            or probe['probabilities'] != probabilities or not receipt['native_verified']
            or episode_quality_failures(receipt['teacher_metrics'])
            or receipt['teacher_metrics']['life_losses'] > VALIDATION['gates']['maximum_life_losses_total']):
        raise ValueError('v4 candidate, probe, teacher or fixed native evidence is unbound')
    margin = math.log(max(probabilities[learner_choice], 1e-8)/max(probabilities[teacher_choice], 1e-8))
    if (receipt['teacher_probability'] != probabilities[teacher_choice] or receipt['learner_probability'] != probabilities[learner_choice]
            or not math.isclose(receipt['learner_minus_teacher_log_probability'], margin, abs_tol=1e-12)):
        raise ValueError('v4 probability margin changed')
    negative = None
    if learner_choice != teacher_choice:
        negative = negative_evidence(receipt['teacher_metrics'], receipt['negative_metrics'],
            receipt['immediate_counterfactuals'], teacher_choice, learner_choice)
        if negative != receipt.get('negative_evidence'): raise ValueError('v4 counterfactual harm classification changed')
    flags = cohort_flags(candidate['request']['state'], receipt['immediate_counterfactuals'])
    flags['anticipatory_harm'] = bool(not receipt['immediate_counterfactuals'][learner_choice]['life_lost'] and
        (negative or {}).get('kind') in ('extra_life_loss_with_same_continuation_controller',
            'incomplete_forced_first_branch_with_same_controller'))
    if receipt['cohorts'] != flags: raise ValueError('v4 native behavior cohort changed')
    if receipt['classification'] == 'hard':
        if learner_choice == teacher_choice or not (negative or {}).get('harmful'):
            raise ValueError('Hard count contains an agreement/harmless preference')
    elif receipt['classification'] == 'informative':
        if candidate['case_role'] != 'informative' or not candidate['parent_case']:
            raise ValueError('Ordinary easy suffix entered informative quota')
    else: raise ValueError('Rejected case entered the training pool')


def _pool(directory, config):
    pools = {'train': [], 'development': []}; seen = {}; duplicates = Counter(); ambiguous = {}; splits = {}
    for path in sorted((Path(directory)/'cases').glob('*/verification.json')):
        receipt = _load(path)
        if not receipt['accepted']: continue
        candidate = _load(path.with_name('candidate.json')); probe = _load(path.with_name('probe.json'))
        _validate_binding(candidate, probe, receipt, config, _hash(path.with_name('candidate.json')), _hash(path.with_name('probe.json')))
        for name, digest in receipt['files'].items():
            if _hash(Path(directory)/name) != digest: raise ValueError('Accepted native evidence changed')
        if candidate['parent_case']:
            parent = _load(Path(directory)/'cases'/candidate['parent_case']/'verification.json')
            if not parent['accepted'] or parent['classification'] != 'hard': raise ValueError('Informative hard parent is missing')
        record = _record(candidate, receipt, config); digest = candidate['state_sha256']
        if digest in splits and splits[digest] != candidate['split']:
            raise ValueError('Exact native observation overlaps train/development; collect different families')
        splits[digest] = candidate['split']
        if digest in ambiguous:
            ambiguous[digest]['case_ids'].append(candidate['id'])
            ambiguous[digest]['teacher_choices'] = sorted(set(ambiguous[digest]['teacher_choices']+[receipt['teacher_choice']]))
            duplicates['exact_state_duplicates'] += 1
            continue
        priority = (receipt['classification'] == 'hard', receipt['learner_minus_teacher_log_probability'])
        previous = seen.get(digest)
        if previous:
            duplicates['exact_state_duplicates'] += 1
            if previous[0]['questions']['move']['label'] != receipt['teacher_choice']:
                ambiguous[digest] = {'case_ids':[previous[0]['_meta']['id'], candidate['id']],
                    'teacher_choices': sorted({previous[0]['questions']['move']['label'], receipt['teacher_choice']})}
                del seen[digest]; continue
            if priority <= previous[1]: continue
        seen[digest] = (record, priority)
    for record, _ in seen.values(): pools[record['_meta']['split']].append(record)
    if ambiguous: duplicates['ambiguous_teacher_label_states'] = len(ambiguous)
    _save(Path(directory)/'ambiguous-states.json', {'states':ambiguous,
        'policy':'Every identical observation with conflicting teacher actions is excluded; no confidence-based label replacement.'})
    return pools, dict(duplicates)


def select_partition(records, budgets, *, source_mix=SOURCE_MIX, family_cap=HARVEST['max_selected_per_family'],
                     seed=107, minimums=None):
    """Exact hard/source quotas; no ordinary suffix may replace a missing root."""
    selected = []; missing = {}; family_counts = Counter(); seen = set(); rng = random.Random(seed)
    minimums = minimums or {}
    if budgets['hard'] < 3*budgets['informative']: raise ValueError('Disagreement roots must be at least three quarters of the hard pool')
    for kind in ('hard', 'informative'):
        target = budgets[kind]; allocations = {'off_policy': int(round(target*source_mix['off_policy']))}
        allocations['learner'] = target-allocations['off_policy']
        admitted_by_origin = Counter(); cohort_counts = Counter()
        if kind == 'hard' and minimums:
            candidates = [r for r in records if r['_meta']['class'] == kind]
            rng.shuffle(candidates); candidates.sort(key=lambda r: -r['_meta'].get('margin', 0))
            while any(cohort_counts[name] < floor for name,floor in minimums.items()):
                eligible = [r for r in candidates if r['_meta']['state_sha256'] not in seen and
                    admitted_by_origin[r['_meta']['origin']] < allocations.get(r['_meta']['origin'], 0) and
                    family_counts[r['_meta']['group_id']] < family_cap]
                deficit = {name: floor-cohort_counts[name] for name,floor in minimums.items() if floor > cohort_counts[name]}
                scarcity = {name: sum(bool(r['_meta']['cohorts'].get(name)) for r in eligible) for name in deficit}
                def score(row):
                    return sum(1/max(scarcity[name], 1) for name in deficit if row['_meta']['cohorts'].get(name))
                chosen = max(eligible, key=score, default=None)
                if chosen is None or score(chosen) == 0:
                    raise ValueError('Insufficient hard-only native behavior cohorts: '+json.dumps(deficit, sort_keys=True))
                selected.append(chosen); seen.add(chosen['_meta']['state_sha256']); family_counts[chosen['_meta']['group_id']] += 1
                admitted_by_origin[chosen['_meta']['origin']] += 1
                cohort_counts.update(k for k,v in chosen['_meta']['cohorts'].items() if v)
        for origin, count in allocations.items():
            count -= admitted_by_origin[origin]
            if count == 0: continue
            available = [r for r in records if r['_meta']['class'] == kind and r['_meta']['origin'] == origin]
            rng.shuffle(available)
            available.sort(key=lambda r: -r['_meta'].get('margin', 0))
            admitted = 0
            for record in available:
                digest = record['_meta']['state_sha256']; family = record['_meta']['group_id']
                if digest in seen or family_counts[family] >= family_cap: continue
                selected.append(record); seen.add(digest); family_counts[family] += 1; admitted += 1
                if admitted == count: break
            if admitted < count: missing[f'{kind}/{origin}'] = count-admitted
    if missing: raise ValueError('Insufficient unique qualified hard/source quotas: '+json.dumps(missing, sort_keys=True))
    return selected


def build_dataset(directory, output_directory, qualification_path, allow_partial=False, backup_directory=None):
    config = _configuration(directory, qualification_path)
    if 'probe_checkpoint' not in config: raise ValueError('No frozen v3 probes were collected')
    pools, duplicates = _pool(directory, config)
    budgets = copy.deepcopy(TARGETS); recipe = copy.deepcopy(RECIPE); scale = 1
    replay = old_expert_records(); rng = random.Random(113); rng.shuffle(replay)
    if allow_partial:
        ratios = []
        for split, targets in budgets.items():
            for kind in ('hard', 'informative'):
                for origin in SOURCE_MIX:
                    needed = int(round(targets[kind]*SOURCE_MIX[origin]))
                    available = sum(r['_meta']['class'] == kind and r['_meta']['origin'] == origin for r in pools[split])
                    if needed: ratios.append(available/needed)
        scale = min([1, len(replay)/TARGETS['train']['replay']]+ratios)
        if scale <= 0: raise ValueError('No feasible hard-focused partial mixture; missing a required class/source/partition')
        for split in budgets:
            for kind in budgets[split]: budgets[split][kind] = int(math.floor(budgets[split][kind]*scale))
            budgets[split]['hard'] = max(budgets[split]['hard'], 3*budgets[split]['informative'])
        recipe['replay'] = int(math.floor(RECIPE['replay']*scale))
    minimums = {split:{k:int(math.floor(v*scale)) for k,v in floors.items()} for split,floors in MINIMUMS.items()}
    partitions = {}; used = set(); families = {'train': set(), 'development': set()}
    for split in ('train', 'development'):
        partitions[split] = select_partition(pools[split], budgets[split], seed=107 if split == 'train' else 109,
                                              minimums=minimums[split])
        for row in partitions[split]: used.add(row['_meta']['state_sha256']); families[split].add(row['_meta']['group_id'])
    if families['train'] & families['development']: raise ValueError('Related episode families cross partitions')
    needed = budgets['train']['replay']; added = 0
    for row in replay if needed else []:
        digest = fingerprint(row['state'])
        if digest in used: continue
        row = copy.deepcopy(row); row['_meta'].update(classification='replay', state_sha256=digest,
            origin='previous_pacman_replay', split='train')
        row['_meta']['class'] = 'replay'; partitions['train'].append(row); used.add(digest); added += 1
        if added == needed: break
    if added != needed: raise ValueError(f'Only {added} unique Pac-Man replay rows available; need {needed}')
    output_directory = Path(output_directory); output_directory.mkdir(parents=True, exist_ok=True)
    if (output_directory/f'{PREFIX}-manifest.json').exists():
        return validate_dataset(output_directory, qualification_path)
    files = {}; coverage = {}; selection = {}
    for split, rows in partitions.items():
        rng.shuffle(rows); counts = Counter(); by_source = Counter(); cohorts = Counter()
        for row in rows:
            counts[row['_meta']['class']] += 1; by_source[row['_meta']['origin']] += 1
            cohorts.update(k for k,v in row['_meta'].get('cohorts', cohort_flags(row['state'])).items() if v)
        coverage[split] = {'classes': dict(counts), 'sources': dict(by_source), 'cohorts': dict(cohorts),
                           'distinct_states': len({fingerprint(r['state']) for r in rows}),
                           'distinct_families': len({r['_meta']['group_id'] for r in rows})}
        path = output_directory/f'{PREFIX}-{split}.jsonl'; temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows)); temporary.replace(path)
        files[path.name] = _hash(path)
        selection[split] = [r['_meta']['id'] for r in rows if r['_meta']['class'] != 'replay']
    bundle = output_directory/f'{PREFIX}-replays.zip'; temporary = bundle.with_name(bundle.name+'.tmp')
    included = set(selection['train']+selection['development']); dependencies = list(included); episodes = set()
    while dependencies:
        identifier = dependencies.pop(); candidate = _load(Path(directory)/'cases'/identifier/'candidate.json')
        if candidate.get('parent_case') and candidate['parent_case'] not in included:
            included.add(candidate['parent_case']); dependencies.append(candidate['parent_case'])
        if candidate['source'].get('episode'): episodes.add(candidate['source']['episode'])
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(Path(directory)/'collection.json', 'collection.json')
        for episode in sorted(episodes):
            folder = Path(directory)/'episodes'/episode
            archive.write(folder/'attempt.json', f'episodes/{episode}/attempt.json')
            for name in _load(folder/'attempt.json')['files']: archive.write(Path(directory)/name, name)
        for identifier in sorted(included):
            folder = Path(directory)/'cases'/identifier
            for name in ('candidate.json', 'probe.json', 'verification.json'):
                archive.write(folder/name, f'cases/{identifier}/{name}')
            receipt = _load(folder/'verification.json')
            for name in receipt['files']: archive.write(Path(directory)/name, name)
    temporary.replace(bundle); files[bundle.name] = _hash(bundle)
    requested = len(partitions['train'])+recipe['replay']
    manifest = {'dataset': PREFIX, 'counts': {s:len(r) for s,r in partitions.items()},
        'target_budgets': TARGETS, 'selected_budgets': budgets, 'partial': allow_partial and budgets != TARGETS,
        'scale': scale, 'source_mix': SOURCE_MIX, 'minimums': MINIMUMS, 'selected_minimums': minimums,
        'recipe': recipe, 'expected_training_requests': requested, 'expected_optimizer_steps': (requested+7)//8,
        'probe_checkpoint': config['probe_checkpoint'], 'source_sha256': config['source_sha256'],
        'previous_replay_sha256': {name: _hash(ROOT/'data'/name) for name in
            ('pacman-native-v2-manifest.json', 'pacman-native-v2-train.jsonl')},
        'generator_sha256': config['generator_sha256'], 'qualification_sha256': config['qualification_sha256'],
        'teacher_options': config['teacher_options'], 'coverage': coverage, 'duplicates': duplicates,
        'selection': selection, 'files': files, 'replay_bundle': bundle.name, 'native_verified': True,
        'collection_directory': str(directory), 'collection_files': completed_files(directory),
        'scope': 'Same-state v3 hard negatives with complete fixed-teacher recovery and same-controller counterfactuals. Reserved starts excluded; failed cases retained only as evidence.'}
    _save(output_directory/f'{PREFIX}-manifest.json', manifest)
    validate_dataset(output_directory, qualification_path)
    _backup(directory, backup_directory)
    return manifest


def validate_dataset(output_directory, qualification_path, verify_native=False):
    output_directory = Path(output_directory); manifest = _load(output_directory/f'{PREFIX}-manifest.json')
    teacher = require_qualified_teacher(qualification_path)
    scale = manifest['scale']
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1: raise ValueError('Invalid explicit partial scale')
    expected_budgets = {split:{k:int(math.floor(v*scale)) for k,v in budget.items()} for split,budget in TARGETS.items()}
    for budget in expected_budgets.values(): budget['hard'] = max(budget['hard'], 3*budget['informative'])
    expected_minimums = {split:{k:int(math.floor(v*scale)) for k,v in floors.items()} for split,floors in MINIMUMS.items()}
    if (manifest['dataset'] != PREFIX or manifest['source_sha256'] != source_hashes()
            or manifest['generator_sha256'] != _hash(Path(__file__))
            or manifest['qualification_sha256'] != _hash(qualification_path)
            or manifest['teacher_options'] != teacher['options'] or manifest['target_budgets'] != TARGETS
            or manifest['source_mix'] != SOURCE_MIX or manifest['minimums'] != MINIMUMS
            or manifest['selected_budgets'] != expected_budgets or manifest['selected_minimums'] != expected_minimums
            or manifest['partial'] != (expected_budgets != TARGETS)
            or manifest['recipe'] != {**RECIPE, 'replay': int(math.floor(RECIPE['replay']*scale))}):
        raise ValueError('v4 dataset fixed provenance or budget changed')
    require_v3_identity(manifest['probe_checkpoint'])
    for name, digest in manifest['files'].items():
        if _hash(output_directory/name) != digest: raise ValueError('v4 dataset artifact changed: '+name)
    for name, digest in manifest['previous_replay_sha256'].items():
        if _hash(ROOT/'data'/name) != digest: raise ValueError('Canonical previous Pac-Man replay changed')
    replay_lookup = {fingerprint(r['state']):(r['questions']['move']['label'], r['_meta']['group_id'])
        for r in _read_rows(ROOT/'data/pacman-native-v2-train.jsonl')}
    seen = set(); families = {'train': set(), 'development': set()}
    with zipfile.ZipFile(output_directory/manifest['replay_bundle']) as archive:
        config = json.loads(archive.read('collection.json'))
        if (config['probe_checkpoint'] != manifest['probe_checkpoint'] or config['source_sha256'] != manifest['source_sha256']
                or config['generator_sha256'] != manifest['generator_sha256'] or config['protocol'] != SPEC
                or config['teacher_options'] != manifest['teacher_options'] or config['qualification_sha256'] != manifest['qualification_sha256']):
            raise ValueError('v4 replay bundle collection changed')
        train_seeds = set(); development_seeds = set()
        for seeds in config['episode_sets'].values(): train_seeds.update(seeds['train']); development_seeds.update(seeds['development'])
        check_seed_splits(sorted(train_seeds), sorted(development_seeds))
        for split, count in manifest['counts'].items():
            rows = _read_rows(output_directory/f'{PREFIX}-{split}.jsonl')
            if len(rows) != count: raise ValueError('v4 dataset row count changed')
            classes, sources, hard_cohorts, family_counts, all_cohorts = Counter(), Counter(), Counter(), Counter(), Counter()
            selected = []
            for row in rows:
                digest = fingerprint(row['state']); meta = row['_meta']; classes[meta['class']] += 1
                if digest in seen or digest != meta['state_sha256']: raise ValueError('v4 duplicate or changed state')
                seen.add(digest); families[split].add(meta['group_id'])
                if {k:v for k,v in row['questions']['move'].items() if k not in ('label','src')} != body(row['state'])['questions']['move']:
                    raise ValueError('v4 training/inference options changed')
                sources[meta['origin']] += 1
                if meta['class'] == 'replay':
                    if split != 'train' or replay_lookup.get(digest) != (row['questions']['move']['label'], meta['group_id']):
                        raise ValueError('Replay label is not unchanged canonical previous training data')
                    all_cohorts.update(k for k,v in cohort_flags(row['state']).items() if v)
                    continue
                family_counts[meta['group_id']] += 1; selected.append(meta['id'])
                prefix = f'cases/{meta["id"]}/'; candidate = json.loads(archive.read(prefix+'candidate.json'))
                probe = json.loads(archive.read(prefix+'probe.json')); receipt = json.loads(archive.read(prefix+'verification.json'))
                _validate_binding(candidate, probe, receipt, config,
                    hashlib.sha256(archive.read(prefix+'candidate.json')).hexdigest(),
                    hashlib.sha256(archive.read(prefix+'probe.json')).hexdigest())
                allowed = train_seeds if split == 'train' else development_seeds
                if candidate['split'] != split or candidate['seed'] not in allowed or not receipt['accepted'] or _record(candidate, receipt, config) != row:
                    raise ValueError('v4 selected request is unbound to archived native evidence')
                all_cohorts.update(k for k,v in receipt['cohorts'].items() if v)
                if meta['class'] == 'hard': hard_cohorts.update(k for k,v in receipt['cohorts'].items() if v)
                if candidate['parent_case']:
                    parent_path = f'cases/{candidate["parent_case"]}/verification.json'
                    parent = json.loads(archive.read(parent_path))
                    if (not parent['accepted'] or parent['classification'] != 'hard' or
                        candidate['source']['parent_verification_sha256'] != hashlib.sha256(archive.read(parent_path)).hexdigest()):
                        raise ValueError('v4 informative parent evidence changed')
                if candidate['source'].get('episode'):
                    episode_path = f'episodes/{candidate["source"]["episode"]}/attempt.json'
                    episode = json.loads(archive.read(episode_path)); source = candidate['source']
                    if (episode['seed'] != candidate['seed'] or episode['split'] != candidate['split']
                            or episode['files'].get(source['trace']) != source['trace_sha256']
                            or hashlib.sha256(archive.read(source['trace'])).hexdigest() != source['trace_sha256']):
                        raise ValueError('v4 source episode provenance changed')
                for name, digest in receipt['files'].items():
                    content = archive.read(name)
                    if hashlib.sha256(content).hexdigest() != digest: raise ValueError('Native case proof changed')
                    proof = [json.loads(line) for line in content.splitlines()]
                    metric = receipt['teacher_metrics'] if Path(name).name == 'teacher.jsonl' else receipt['negative_metrics']
                    if fingerprint(summarize([r['diagnostics'] for r in proof], metric['outcome'])) != fingerprint(metric):
                        raise ValueError('Native case trajectory and counters differ')
                    expected_first = receipt['teacher_choice'] if Path(name).name == 'teacher.jsonl' else receipt['learner_choice']
                    if proof[0]['choice'] != expected_first or proof[0]['state_sha256'] != candidate['state_sha256']:
                        raise ValueError('Counterfactual does not start at the exact candidate/action')
                    if verify_native:
                        import tempfile
                        with tempfile.TemporaryDirectory() as folder:
                            path = Path(folder)/Path(name).name; path.write_bytes(content)
                            _verify_branch(candidate, path, metric)
            if dict(classes) != {k:v for k,v in manifest['selected_budgets'][split].items() if v}:
                raise ValueError('v4 class budgets changed')
            if selected != manifest['selection'][split]: raise ValueError('v4 selected case index changed')
            for kind in ('hard', 'informative'):
                allocation = int(round(expected_budgets[split][kind]*SOURCE_MIX['off_policy']))
                counted = Counter(r['_meta']['origin'] for r in rows if r['_meta']['class'] == kind)
                if counted != Counter({k:v for k,v in {'off_policy':allocation, 'learner':expected_budgets[split][kind]-allocation}.items() if v}):
                    raise ValueError('v4 hard/source quota changed')
            if any(hard_cohorts[name] < floor for name,floor in expected_minimums[split].items()):
                raise ValueError('v4 hard-only behavior cohort floor missing')
            if any(count > HARVEST['max_selected_per_family'] for count in family_counts.values()):
                raise ValueError('v4 per-family cap exceeded')
            actual_coverage = {'classes':dict(classes), 'sources':dict(sources), 'cohorts':dict(all_cohorts),
                'distinct_states':len(rows), 'distinct_families':len(families[split])}
            if actual_coverage != manifest['coverage'][split]: raise ValueError('v4 observed coverage report changed')
    if families['train'] & families['development']: raise ValueError('v4 related family overlap')
    if manifest['expected_training_requests'] != manifest['counts']['train']+manifest['recipe']['replay']:
        raise ValueError('v4 total request budget changed')
    if manifest['expected_optimizer_steps'] != (manifest['expected_training_requests']+7)//8:
        raise ValueError('v4 optimizer update budget changed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['harvest', 'verify', 'build', 'validate', 'status'])
    parser.add_argument('directory', type=Path); parser.add_argument('--qualification', type=Path, default=ROOT/'evaluation/teacher-qualification.json')
    parser.add_argument('--output', type=Path, default=ROOT/'data'); parser.add_argument('--workers', type=int)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument('--development-seeds', type=int, nargs='+', default=DEFAULT_DEVELOPMENT_SEEDS)
    parser.add_argument('--allow-partial', action='store_true'); parser.add_argument('--verify-native', action='store_true')
    args = parser.parse_args()
    if args.stage == 'harvest': result = harvest_scenarios(args.directory, args.qualification, args.train_seeds, args.development_seeds, args.workers)
    elif args.stage == 'verify': result = verify_cases(args.directory, args.qualification, args.workers)
    elif args.stage == 'build': result = build_dataset(args.directory, args.output, args.qualification, args.allow_partial)
    elif args.stage == 'validate': result = validate_dataset(args.output, args.qualification, args.verify_native)
    else:
        entries = _candidate_index(args.directory)
        result = {'candidates': len(entries), 'pending_probe': sum(not e['probed'] for e in entries),
                  'pending_verification': sum(e['probed'] and not e['verified'] for e in entries)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
