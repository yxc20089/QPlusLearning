"""Independently verified, teacher-only native-v4 scenario distillation.

This is deliberately a different dataset from v4 learner disagreements. It
never calls Kev, invents a probe, or calls a safe alternative a harmful one.
The source teacher's full suffix must win without losing a life. Every legal
action remains in the request; exact immediate fatal alternatives are evidence,
not a runtime safety filter. CPU replay is sufficient for this preparation.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import zipfile

from gameplay_benchmark import BenchmarkEngine, SPEC, diagnostic_record, summarize
from pacman_lab import ROOT, body, distances, position
from planner_data import RECIPE as PREVIOUS_RECIPE
from teacher_data import COLLECTION
from teacher_validation import (VALIDATION, episode_quality_failures, fingerprint,
                                require_qualified_teacher, source_hashes)
from v3_data import reserved_seeds

PREFIX = 'pacman-native-v4-offline'
COLLECTION_PREFIX = 'pacman-native-v4'
RECIPE = {**PREVIOUS_RECIPE, 'replay': 304}
TARGETS = {'train': {'targeted': 3660, 'informative': 1220, 'replay': 912},
           'development': {'targeted': 384, 'informative': 128, 'replay': 0}}
MINIMUMS = {'train': {'ghost_intercept': 300, 'power_expiry': 120, 'food_dry': 180,
                     'sparse_food': 180, 'post_respawn': 120,
                     'potential_expiry_during_action_near_ghost': 24},
            'development': {'ghost_intercept': 48, 'power_expiry': 16, 'food_dry': 24,
                            'sparse_food': 24, 'post_respawn': 16,
                            'potential_expiry_during_action_near_ghost': 4}}
FAMILY_CAP = 128
WINDOW_OFFSETS = (-8, -4, -1, 1, 4, 8)
MAX_WINDOWS_PER_ROOT = 4
MAX_RECOVERY_WINDOW = 24
HARD_FLAGS = ('immediate_critical', 'ghost_intercept', 'power_expiry', 'food_dry', 'sparse_food')
INTERPRETATION = {
    'label_source': 'Qualified CPU teacher action on its exact independently replayed native trajectory',
    'positive_evidence': 'The source teacher-controlled suffix cleared the maze with zero life losses and passed loop/stall gates.',
    'negative_evidence': 'Only native immediate life-loss counterfactuals are measured; safe alternatives are not proved inferior.',
    'learner_disagreement': 'Not measured: no native-v3 predictions or learner disagreement quotas are present.',
    'anticipatory_harm': 'Not measured: no forced-alternative continuation comparison is present.',
    'expiry': 'Potential expiry compares the current timer to elapsed legal-action durations. Ghost-eating pauses can freeze that timer. Actual expiry is reported separately from the chosen transition before/after power clocks.',
    'optimality': 'Teacher-labelled hard scenarios are exposures, not a proof of globally optimal actions.',
    'input': 'Original observation/action v2 and all original legal options; future teacher plans and results remain metadata/evidence.',
    'initialization': 'Recommended parent native-v3 with a fresh optimizer; this exporter does not train or resume optimizer state.',
}


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    temporary.replace(path)


def _safe_path(directory, name):
    directory = Path(directory).resolve(); path = (directory / name).resolve()
    if not path.is_relative_to(directory) or path == directory:
        raise ValueError('Evidence filename escapes its directory')
    return path


def _request(record):
    """Exactly the model input, excluding labels and evidence metadata."""
    result = {'state': copy.deepcopy(record['state']), 'questions': copy.deepcopy(record['questions'])}
    for question in result['questions'].values():
        question.pop('label', None); question.pop('src', None)
    return result


def request_fingerprint(record):
    return fingerprint(_request(record))


def cohort_flags(state, risks, power_transition=None):
    routes = distances(state['maze'], position(state['player']))
    outside = [g for g in state['ghosts'] if g['mode'] == 'outside']
    danger = min((routes.get(position(g), 999) for g in outside if not g['frightened']), default=999)
    near = min((routes.get(position(g), 999) for g in outside), default=999) <= 3
    remaining = state['timing']['power']['remaining_frames']
    return {'ghost_intercept': danger <= 6,
            'power_expiry': bool(state['frightened'] and 0 < remaining <= 120),
            'potential_expiry_during_action_near_ghost': bool(state['frightened'] and near and 0 < remaining <=
                max((r['action_frames'] for r in risks.values()), default=0)),
            'actual_power_expiry_during_teacher_action_near_ghost': bool(state['frightened'] and near and
                remaining > 0 and power_transition is not None and power_transition['remaining_after'] == 0),
            'food_dry': state.get('decisions_since_last_pellet', 0) >= 24,
            'sparse_food': state['pellets_remaining'] <= 30,
            'post_respawn': state.get('life_epoch', 0) > 0,
            'immediate_critical': bool(any(r['life_lost'] for r in risks.values()) and
                                       any(not r['life_lost'] for r in risks.values()))}


def is_targeted(state, risks, choice):
    flags = cohort_flags(state, risks)
    return len(state['legal_moves']) >= 2 and not risks[choice]['life_lost'] and any(flags[k] for k in HARD_FLAGS)


def _fixed_collection(config):
    return {k: v for k, v in config.items() if k != 'episode_sets'}


def _collection_config(directory, qualification_path):
    config = _load(Path(directory) / 'collection.json')
    teacher = require_qualified_teacher(qualification_path)
    if (config.get('version') != COLLECTION_PREFIX or config.get('protocol') != SPEC
            or config.get('source_sha256') != source_hashes()
            or config.get('generator_sha256') != _hash(ROOT / 'v4_data.py')
            or config.get('qualification_sha256') != _hash(qualification_path)
            or config.get('teacher_options') != teacher['options']):
        raise ValueError('Source collection/qualification/configuration is not the pinned native teacher')
    seeds = {'train': set(), 'development': set()}
    for key, wave in config['episode_sets'].items():
        if not key.startswith('off_policy/'):
            continue
        for split in seeds:
            values = wave[split]
            if any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in values) or len(set(values)) != len(values):
                raise ValueError('Invalid source seed wave')
            seeds[split].update(values)
    forbidden = reserved_seeds() | set(range(50000, 51000)) | {300017, 300029, 300043, 300059, 310019, 310033}
    previous = ROOT / 'data/pacman-native-v3-manifest.json'
    if previous.exists():
        prior = _load(previous)
        forbidden.update(prior.get('train_seeds', [])); forbidden.update(prior.get('development_seeds', []))
    if seeds['train'] & seeds['development'] or (seeds['train'] | seeds['development']) & forbidden:
        raise ValueError('Source seed partitions overlap each other or frozen evaluation/prior families')
    return config


def _source_jobs(directory, config):
    directory = Path(directory); jobs = []; known = set()
    for key, wave in config['episode_sets'].items():
        if not key.startswith('off_policy/'):
            continue
        wave_id = key.split('/', 1)[1]; parameters = wave['parameters']
        limits = {name: parameters[name] for name in ('roots_per_episode', 'perturb_roots_per_episode', 'perturb_turns')}
        for split in ('train', 'development'):
            for level in SPEC['levels']:
                for seed in wave[split]:
                    for mode in ('normal', 'post_respawn'):
                        identifier = f'off-policy-{wave_id}-{split}-level-{level}-seed-{seed}-{mode}'
                        known.add(identifier)
                        path = directory / 'episodes' / identifier / 'attempt.json'
                        if not path.is_file():
                            continue  # In-flight traces/partial files never enter the pool.
                        receipt = _load(path)
                        job = {'id': identifier, 'split': split, 'level': level, 'seed': seed, 'mode': mode,
                               'options': config['teacher_options'], 'limits': limits, 'perturb': parameters['perturb']}
                        if any(receipt.get(k) != job[k] for k in ('id', 'split', 'level', 'seed', 'mode')) or receipt.get('job_sha256') != fingerprint(job):
                            raise ValueError('Completed source episode is not bound to its exact job/configuration')
                        expected_name = f'episodes/{identifier}/trace.jsonl'
                        if set(receipt.get('files', {})) != {expected_name}:
                            raise ValueError('Source receipt does not bind its full native trace')
                        if _hash(_safe_path(directory, expected_name)) != receipt['files'][expected_name]:
                            raise ValueError('Source trace hash mismatch')
                        jobs.append({'job': job, 'source_attempt': str(path),
                                     'source_trace': str(directory / expected_name),
                                     'source_config': copy.deepcopy(config)})
    unexpected = {p.parent.name for p in (directory / 'episodes').glob('off-policy-*/attempt.json')} - known
    if unexpected:
        raise ValueError('Completed teacher episodes absent from declared collection waves')
    return sorted(jobs, key=lambda job: job['job']['id'])


def _one_hot(state, move):
    return {'answers': {'move': {'choice': move, 'probabilities': {d: float(d == move) for d in state['legal_moves']}}}}


def _metrics_equal(left, right):
    return fingerprint({k: v for k, v in left.items() if k != 'mean_http_ms'}) == fingerprint(
        {k: v for k, v in right.items() if k != 'mean_http_ms'})


def _replay_trace(trace_path, job, expected):
    """Fresh native closure per full episode, including any intentional prefix."""
    source = _rows(trace_path)
    if not source or len(source) > SPEC['max_decisions']:
        raise ValueError('Empty/over-budget completed source trajectory')
    teacher_started = job['mode'] == 'normal'
    life = prefix_count = frames = stale = 0
    status = None; teacher_rows = []; all_rows = []; verified = []
    with BenchmarkEngine() as engine:
        state = engine.request('reset', options={'seed': job['seed'], 'level': job['level'], 'action_version': 2})
        for index, row in enumerate(source):
            if status:
                raise ValueError('Source trace continued past a terminal/watchdog boundary')
            if row['request'] != body(state):
                raise ValueError(f'Native source observation/request differs at decision {index}')
            controller = 'qualified_teacher' if teacher_started else 'legal_perturbation'
            if row.get('controller') != controller:
                raise ValueError('Source prefix/teacher controller boundary differs')
            move = row['diagnostics']['choice']
            if move not in state['legal_moves'] or row['response'] != _one_hot(state, move):
                raise ValueError('Source action/probabilities do not match the original legal options')
            if teacher_started:
                plan = row.get('teacher') or {}
                if (plan.get('choice') != move or plan.get('algorithm') != 'native-rollout-mpc-food-routing-v2'
                        or plan.get('scenario_seeds') != job['options']['scenario_seeds']
                        or plan.get('buffers') != job['options']['buffers']
                        or plan.get('horizon_frames') not in {job['options']['horizon_frames'],
                            job['options']['danger_horizon_frames'], job['options']['endgame_horizon_frames']}):
                    raise ValueError('Teacher-labelled source action lacks its pinned planner provenance')
            else:
                prefix_count += 1
                if prefix_count > COLLECTION['prefix_max_decisions'] or row.get('teacher') is not None:
                    raise ValueError('Invalid intentional respawn prefix')
            risks = engine.request('risks')
            if set(risks) != set(state['legal_moves']):
                raise ValueError('Native risks do not cover every original legal action')
            step = engine.step(move)
            diagnostic = diagnostic_record(state, _one_hot(state, move), step, risks, life)
            if fingerprint(diagnostic) != fingerprint(row['diagnostics']):
                raise ValueError(f'Native action/risks/diagnostics differ at decision {index}')
            all_rows.append(diagnostic)
            if teacher_started:
                teacher_rows.append({**diagnostic, 'life_index': life - int(job['mode'] == 'post_respawn')})
                power_transition = {'remaining_before': state['timing']['power']['remaining_frames'],
                                    'remaining_after': step['state']['timing']['power']['remaining_frames'],
                                    'action_frames': step['action_frames']}
                verified.append({'source_index': index, 'life_index': life, 'request': _request(body(state)),
                    'choice': move, 'state_sha256': fingerprint(state), 'request_sha256': request_fingerprint(body(state)),
                    'cohorts': cohort_flags(state, risks, power_transition), 'power_transition': power_transition,
                    'diagnostics': diagnostic})
            frames += step['action_frames']
            stale = stale + 1 if step['state']['pellets_remaining'] == state['pellets_remaining'] else 0
            state = step['state']
            if step['outcome'] == 'level_cleared':
                status = 'level_cleared'
            elif step['outcome'] == 'life_lost':
                continuation = engine.request('continue')
                if continuation['status'] != 'playing':
                    status = 'game_over'
                else:
                    state = continuation['state']; life += 1; stale = 0; teacher_started = True
            if status is None and frames >= SPEC['max_simulation_frames']:
                status = 'simulation_frame_cap'
            if status is None and teacher_started and stale >= SPEC['max_no_pellet_decisions']:
                status = 'no_progress_watchdog'
    if status is None:
        if len(source) != SPEC['max_decisions']:
            raise ValueError('Completed source trace stopped before a native terminal/watchdog boundary')
        status = 'decision_cap'
    full_metrics = summarize(all_rows, status)
    metrics = summarize(teacher_rows or all_rows, status)
    failures = episode_quality_failures(metrics)
    if not teacher_rows:
        failures.append('no teacher-controlled suffix')
    source_failures = list(failures)
    if metrics['life_losses'] > VALIDATION['gates']['maximum_life_losses_total']:
        source_failures.append('too many teacher life losses')
    if (not _metrics_equal(metrics, expected['metrics']) or not _metrics_equal(full_metrics, expected['full_metrics'])
            or expected.get('accepted') != (not source_failures) or expected.get('failures') != source_failures):
        raise ValueError('Source admission/outcome/metrics differ from independent native replay')
    # Stronger local gate: even if an older qualification allowed a loss, no
    # teacher-controlled death is admitted to this offline positive dataset.
    if metrics['life_losses'] != 0:
        failures.append('teacher-controlled life loss')
    return {'accepted': not failures, 'failures': failures, 'metrics': metrics,
            'full_metrics': full_metrics, 'intentional_prefix_decisions': prefix_count}, verified


def _receipt_files(directory, receipt):
    for name, digest in receipt['files'].items():
        if _hash(_safe_path(directory, name)) != digest:
            raise ValueError('Committed independent replay evidence is missing or changed')


def _validate_source_binding(receipt, config, attempt):
    """A replay receipt also binds the original declared job and admission."""
    binding = receipt['binding']; job = binding['job']; identifier = job['id']
    expected_names = {f'episodes/{identifier}/{name}' for name in
                      ('source-trace.jsonl', 'source-attempt.json', 'source-config.json', 'verified.jsonl')}
    if (binding['dataset'] != PREFIX or set(receipt['files']) != expected_names
            or fingerprint(_fixed_collection(config)) != binding['collection_fixed_sha256']
            or config['source_sha256'] != binding['source_sha256']
            or config['qualification_sha256'] != binding['qualification_sha256']
            or config['teacher_options'] != job['options']
            or config['generator_sha256'] != _hash(ROOT / 'v4_data.py')
            or config['version'] != COLLECTION_PREFIX or config['protocol'] != SPEC):
        raise ValueError('Source episode/configuration receipt binding differs')
    declared = False
    for key, wave in config['episode_sets'].items():
        if not key.startswith('off_policy/'):
            continue
        parameters = wave['parameters']; wave_id = key.split('/', 1)[1]
        expected_job = {'id': f'off-policy-{wave_id}-{job["split"]}-level-{job["level"]}-seed-{job["seed"]}-{job["mode"]}',
            'split': job['split'], 'level': job['level'], 'seed': job['seed'], 'mode': job['mode'],
            'options': config['teacher_options'],
            'limits': {name: parameters[name] for name in ('roots_per_episode', 'perturb_roots_per_episode', 'perturb_turns')},
            'perturb': parameters['perturb']}
        if (job['split'] in ('train', 'development') and job['seed'] in wave[job['split']]
                and job['level'] in SPEC['levels'] and job['mode'] in ('normal', 'post_respawn') and job == expected_job):
            declared = True
    if (not declared or attempt['job_sha256'] != fingerprint(job)
            or any(attempt[k] != job[k] for k in ('id', 'split', 'level', 'seed', 'mode'))
            or attempt['files'] != {f'episodes/{identifier}/trace.jsonl': binding['source_trace_sha256']}
            or not _metrics_equal(attempt['metrics'], receipt['metrics'])
            or not _metrics_equal(attempt['full_metrics'], receipt['full_metrics'])):
        raise ValueError('Original source job/trace/metrics provenance differs')
    failures = episode_quality_failures(receipt['metrics'])
    if receipt['verified_teacher_rows'] == 0:
        failures.append('no teacher-controlled suffix')
    source_failures = list(failures)
    if receipt['metrics']['life_losses'] > VALIDATION['gates']['maximum_life_losses_total']:
        source_failures.append('too many teacher life losses')
    if receipt['metrics']['life_losses']:
        failures.append('teacher-controlled life loss')
    if (receipt['accepted'] != (not failures) or receipt['failures'] != failures
            or attempt['accepted'] != (not source_failures) or attempt['failures'] != source_failures):
        raise ValueError('Source/independent qualification admission differs')


def _verify_job(task):
    """Only an atomic receipt makes an episode reusable after interruption."""
    evidence = Path(task['evidence_directory']); job = task['job']; identifier = job['id']
    folder = evidence / 'episodes' / identifier; folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'receipt.json'
    binding = {'dataset': PREFIX, 'job': job, 'source_attempt_sha256': _hash(task['source_attempt']),
               'source_trace_sha256': _hash(task['source_trace']),
               'collection_fixed_sha256': fingerprint(_fixed_collection(task['source_config'])),
               'source_sha256': source_hashes(), 'exporter_sha256': _hash(Path(__file__)),
               'qualification_sha256': task['source_config']['qualification_sha256']}
    if path.is_file():
        receipt = _load(path)
        if receipt.get('binding') != binding or not receipt.get('native_replayed'):
            raise ValueError('Cached independent receipt inputs/source changed')
        _receipt_files(evidence, receipt)
        return {'id': identifier, 'accepted': receipt['accepted'], 'reused': True,
                'receipt': path.relative_to(evidence).as_posix(), 'receipt_sha256': _hash(path)}
    copied_trace = folder / 'source-trace.jsonl'; copied_attempt = folder / 'source-attempt.json'
    shutil.copyfile(task['source_trace'], copied_trace); shutil.copyfile(task['source_attempt'], copied_attempt)
    if _hash(copied_trace) != binding['source_trace_sha256'] or _hash(copied_attempt) != binding['source_attempt_sha256']:
        raise ValueError('Source changed while copying independent evidence')
    _save(folder / 'source-config.json', task['source_config'])
    replay, verified = _replay_trace(copied_trace, job, _load(copied_attempt))
    _write_rows(folder / 'verified.jsonl', verified)
    names = [f'episodes/{identifier}/{name}' for name in
             ('source-trace.jsonl', 'source-attempt.json', 'source-config.json', 'verified.jsonl')]
    receipt = {**replay, 'binding': binding, 'native_replayed': True,
               'verified_teacher_rows': len(verified), 'files': {name: _hash(evidence / name) for name in names}}
    if binding['source_sha256'] != source_hashes() or binding['exporter_sha256'] != _hash(Path(__file__)):
        raise ValueError('Native/exporter source changed during replay')
    _save(path, receipt)
    return {'id': identifier, 'accepted': receipt['accepted'], 'reused': False,
            'receipt': path.relative_to(evidence).as_posix(), 'receipt_sha256': _hash(path)}


def verify_sources(collection_directory, evidence_directory, qualification_path, workers=10):
    """Read completed source episodes; replay concurrently in a separate directory."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError('Use 1..32 independent CPU replay workers')
    collection_directory = Path(collection_directory).resolve(); evidence_directory = Path(evidence_directory).resolve()
    if (collection_directory == evidence_directory or evidence_directory.is_relative_to(collection_directory)
            or collection_directory.is_relative_to(evidence_directory)):
        raise ValueError('Keep independent evidence outside the running source collection')
    config = _collection_config(collection_directory, qualification_path)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    fixed = {'dataset': PREFIX, 'collection_directory': str(collection_directory),
             'collection_fixed_sha256': fingerprint(_fixed_collection(config)),
             'source_sha256': source_hashes(), 'exporter_sha256': _hash(Path(__file__)),
             'qualification_sha256': _hash(qualification_path), 'recipe': RECIPE,
             'targets': TARGETS, 'minimums': MINIMUMS, 'family_cap': FAMILY_CAP,
             'interpretation': INTERPRETATION}
    config_path = evidence_directory / 'offline-config.json'
    if config_path.exists() and _load(config_path) != fixed:
        raise ValueError('Independent replay setup changed; use another evidence directory')
    _save(config_path, fixed)
    qualification = _load(qualification_path)
    shutil.copyfile(qualification_path, evidence_directory / 'qualification.json')
    shutil.copyfile(Path(qualification_path).parent / qualification['replay_bundle'], evidence_directory / qualification['replay_bundle'])
    jobs = _source_jobs(collection_directory, config)
    for task in jobs:
        task['evidence_directory'] = str(evidence_directory)
    results = []; started = time.monotonic()
    def save_index():
        _save(evidence_directory / 'index.json', {'dataset': PREFIX, 'entries': sorted(results, key=lambda r: r['id']),
              'completed_sources_in_snapshot': len(jobs), 'workers': workers,
              'elapsed_seconds': time.monotonic() - started, 'offline_config_sha256': _hash(config_path)})
    pool = ProcessPoolExecutor(max_workers=min(workers, max(len(jobs), 1)))
    queued = iter(jobs); pending = {}
    def submit():
        job = next(queued, None)
        if job is not None:
            pending[pool.submit(_verify_job, job)] = job
    try:
        for _ in range(min(workers, len(jobs))):
            submit()
        while pending:
            ready, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not ready:
                print(f'[offline/replay] {len(results)}/{len(jobs)} complete; {workers} CPU workers', flush=True)
            for future in ready:
                pending.pop(future); result = future.result(); results.append(result); save_index(); submit()
                print(f'[offline/replay] {len(results)}/{len(jobs)} {result["id"]}: accepted={result["accepted"]}, reused={result["reused"]}', flush=True)
    except BaseException:
        for future in pending:
            future.cancel()
        save_index()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    save_index()
    return report_pool(evidence_directory)


def _record(proof, receipt, identifier, split, kind='targeted', parent=None, window_kind=None):
    record = copy.deepcopy(proof['request']); choice = proof['choice']
    record['questions']['move'].update(label=choice, src=PREFIX)
    binding = receipt['binding']; job = binding['job']
    record['_meta'] = {'id': f'{PREFIX}:{identifier}:{proof["source_index"]}', 'split': split,
        'class': kind, 'origin': 'verified_teacher_trajectory', 'group_id': f'level-{job["level"]}-seed-{job["seed"]}',
        'seed': job['seed'], 'level': job['level'], 'episode': identifier, 'source_index': proof['source_index'],
        'life_index': proof['life_index'], 'state_sha256': proof['state_sha256'], 'request_sha256': proof['request_sha256'],
        'cohorts': proof['cohorts'], 'parent_targeted_id': parent, 'window_kind': window_kind,
        'label_source': INTERPRETATION['label_source'], 'source_trace_sha256': binding['source_trace_sha256'],
        'source_attempt_sha256': binding['source_attempt_sha256'], 'qualification_sha256': binding['qualification_sha256'],
        'source_receipt': f'episodes/{identifier}/receipt.json',
        'immediate_fatal_alternatives': [move for move, risk in proof['diagnostics']['immediate_counterfactuals'].items() if risk['life_lost']],
        'learner_disagreement_measured': False, 'anticipatory_harm_measured': False}
    return record


def _load_pool(evidence_directory):
    evidence_directory = Path(evidence_directory); config = _load(evidence_directory / 'offline-config.json')
    if (config['dataset'] != PREFIX or config['source_sha256'] != source_hashes()
            or config['exporter_sha256'] != _hash(Path(__file__)) or config['recipe'] != RECIPE
            or config['targets'] != TARGETS or config['minimums'] != MINIMUMS or config['family_cap'] != FAMILY_CAP):
        raise ValueError('Offline replay/configuration source changed')
    pools = {'train': [], 'development': []}; episodes = {}; labels = {}; duplicates = Counter()
    split_by_seed = {}; request_splits = {}; request_choices = {}; requests_seen = set()
    for path in sorted((evidence_directory / 'episodes').glob('*/receipt.json')):
        receipt = _load(path); _receipt_files(evidence_directory, receipt); job = receipt['binding']['job']
        _validate_source_binding(receipt, _load(path.with_name('source-config.json')), _load(path.with_name('source-attempt.json')))
        if (receipt['binding']['source_sha256'] != config['source_sha256']
                or receipt['binding']['exporter_sha256'] != config['exporter_sha256']
                or receipt['binding']['collection_fixed_sha256'] != config['collection_fixed_sha256']
                or receipt['binding']['qualification_sha256'] != config['qualification_sha256']
                or not receipt['native_replayed']):
            raise ValueError('Independent episode receipt is unbound')
        split = job['split']; identifier = job['id']
        if split not in pools or split_by_seed.setdefault(job['seed'], split) != split:
            raise ValueError('Source seed group overlaps train/development')
        proofs = _rows(path.with_name('verified.jsonl'))
        if len(proofs) != receipt['verified_teacher_rows']:
            raise ValueError('Independent receipt teacher row count differs')
        if not receipt['accepted']:
            continue
        if receipt['failures'] or receipt['metrics']['life_losses'] != 0 or episode_quality_failures(receipt['metrics']):
            raise ValueError('Nonqualifying teacher suffix entered the offline pool')
        episode = {'receipt': receipt, 'proofs': proofs, 'receipt_sha256': _hash(path)}
        episodes[identifier] = episode
        for proof in proofs:
            state = proof['request']['state']; risks = proof['diagnostics']['immediate_counterfactuals']; choice = proof['choice']
            if (proof['state_sha256'] != fingerprint(state) or proof['request_sha256'] != request_fingerprint(proof['request'])
                    or proof['request'] != _request(body(state)) or proof['cohorts'] != cohort_flags(state, risks, proof['power_transition'])
                    or proof['power_transition']['remaining_before'] != state['timing']['power']['remaining_frames']
                    or proof['power_transition']['action_frames'] != proof['diagnostics']['action_frames']
                    or choice not in state['legal_moves'] or set(risks) != set(state['legal_moves'])
                    or risks[choice]['life_lost']):
                raise ValueError('Verified row input/label/risk evidence differs')
            digest = proof['request_sha256']
            request_splits.setdefault(digest, set()).add(split)
            request_choices.setdefault(digest, set()).add(choice)
            if digest in requests_seen:
                duplicates['all_teacher_duplicate_requests'] += 1
            requests_seen.add(digest)
            if is_targeted(state, risks, choice):
                if digest in labels:
                    duplicates['targeted_duplicate_requests'] += 1
                else:
                    labels[digest] = _record(proof, receipt, identifier, split)
    # Deterministic openings can coincide across disjoint seed families.
    # Quarantine those inputs, including possible sequence windows, rather than
    # reject/reassign the entire source game or leak identical observations.
    overlap = {digest for digest, splits in request_splits.items() if len(splits) > 1}
    ambiguous = {digest for digest, choices in request_choices.items() if len(choices) > 1}
    blocked = overlap | ambiguous
    for digest, record in labels.items():
        if digest not in blocked:
            pools[record['_meta']['split']].append(record)
    for episode in episodes.values():
        episode['excluded_request_sha256'] = blocked
    duplicates['cross_partition_requests_excluded'] = len(overlap)
    duplicates['ambiguous_teacher_labels_excluded'] = len(ambiguous)
    return pools, episodes, dict(duplicates)


def _coverage(rows):
    return dict(Counter(name for row in rows for name, flag in row['_meta']['cohorts'].items() if flag))


def report_pool(evidence_directory):
    pools, episodes, duplicates = _load_pool(evidence_directory)
    receipts = [_load(p) for p in (Path(evidence_directory) / 'episodes').glob('*/receipt.json')]
    result = {'dataset': PREFIX, 'verified_source_episodes': len(receipts),
              'qualified_source_episodes': len(episodes),
              'rejected_source_episodes': len(receipts) - len(episodes),
              'rejection_reasons': dict(Counter(reason for r in receipts if not r['accepted'] for reason in r['failures'])),
              'duplicates': duplicates,
              'interpretation': INTERPRETATION, 'splits': {}}
    for split, rows in pools.items():
        families = Counter(r['_meta']['group_id'] for r in rows)
        result['splits'][split] = {'qualified_source_episodes': sum(e['receipt']['binding']['job']['split'] == split for e in episodes.values()),
            'unique_targeted_available': len(rows), 'targeted_coverage': _coverage(rows),
            'seed_groups': sorted({r['_meta']['seed'] for r in rows}), 'families': dict(families),
            'fresh_family_capacity': len(families) * FAMILY_CAP,
            'required_fresh_rows': TARGETS[split]['targeted'] + TARGETS[split]['informative'],
            'missing_targeted_cohort_floors': {name: floor - _coverage(rows).get(name, 0)
                for name, floor in MINIMUMS[split].items() if floor > _coverage(rows).get(name, 0)}}
    _save(Path(evidence_directory) / 'pool-report.json', result)
    return result


def _window_candidates(root, episode):
    """Nearby same-life teacher frames only; never arbitrary easy suffix fill."""
    proofs = {p['source_index']: p for p in episode['proofs']}; meta = root['_meta']; index = meta['source_index']
    offsets = list(WINDOW_OFFSETS)
    # An actual first pellet after the root provides a short recovery-followthrough
    # endpoint. This is not a measured route-cost comparison with another action.
    for offset in range(1, MAX_RECOVERY_WINDOW + 1):
        proof = proofs.get(index + offset)
        if proof is None or proof['life_index'] != meta['life_index']:
            break
        if proof['diagnostics']['pellets_after'] < proof['diagnostics']['pellets_before']:
            offsets.insert(3, offset); break
    seen = set(); result = []
    for offset in offsets:
        proof = proofs.get(index + offset)
        if (proof is None or proof['life_index'] != meta['life_index'] or offset in seen
                or len(proof['request']['state']['legal_moves']) < 2
                or proof['request_sha256'] in episode.get('excluded_request_sha256', ())):
            continue
        # Require every intervening decision to be in the same teacher-controlled
        # life. A nearby coordinate in another suffix is not a sequence window.
        if any(i not in proofs or proofs[i]['life_index'] != meta['life_index']
               for i in range(min(index, index + offset), max(index, index + offset) + 1)):
            continue
        risks = proof['diagnostics']['immediate_counterfactuals']
        if risks[proof['choice']]['life_lost']:
            continue
        seen.add(offset)
        kind = 'lead_in' if offset < 0 else 'recovery_followthrough'
        result.append(_record(proof, episode['receipt'], meta['episode'], meta['split'],
                              'informative', meta['id'], kind))
    return result


def select_partition(records, episodes, budgets, minimums, family_cap=FAMILY_CAP, seed=107,
                     excluded_request_sha256=()):
    """Exact scenario/window budgets, rare floors and a shared family cap."""
    rng = random.Random(seed); candidates = list(records); rng.shuffle(candidates)
    target, window_target = budgets['targeted'], budgets['informative']
    if target < 3 * window_target:
        raise ValueError('Targeted scenarios must be at least three quarters of fresh data')
    # Reserve the intended 3:1 allocation inside each family, rather than fill
    # every family with roots and discover that no room remains for windows.
    targeted_cap = family_cap if not window_target else math.floor(family_cap * target / (target + window_target))
    selected = []; seen = set(excluded_request_sha256); families = Counter(); coverage = Counter()
    def eligible(row):
        return row['_meta']['request_sha256'] not in seen and families[row['_meta']['group_id']] < targeted_cap
    def add(row):
        selected.append(row); seen.add(row['_meta']['request_sha256']); families[row['_meta']['group_id']] += 1
        coverage.update(k for k, flag in row['_meta']['cohorts'].items() if flag)
    while any(coverage[name] < floor for name, floor in minimums.items()):
        deficits = {name: floor - coverage[name] for name, floor in minimums.items() if floor > coverage[name]}
        available = [r for r in candidates if eligible(r)]
        scarcity = {name: sum(bool(r['_meta']['cohorts'].get(name)) for r in available) for name in deficits}
        def score(row):
            return sum(1 / max(scarcity[name], 1) for name in deficits if row['_meta']['cohorts'].get(name))
        chosen = max(available, key=score, default=None)
        if chosen is None or score(chosen) == 0 or len(selected) >= target:
            raise ValueError('Insufficient targeted-only behavior cohorts: ' + json.dumps(deficits, sort_keys=True))
        add(chosen)
    # Spread residual roots across source families. Consecutive local windows
    # remain evidence, not thousands of unlimited nearly identical easy frames.
    queues = {}
    for row in candidates:
        queues.setdefault(row['_meta']['group_id'], []).append(row)
    while len(selected) < target:
        changed = False
        for queue in queues.values():
            while queue and not eligible(queue[0]):
                queue.pop(0)
            if queue:
                add(queue.pop(0)); changed = True
                if len(selected) == target:
                    break
        if not changed:
            raise ValueError(f'Insufficient unique targeted scenarios/family capacity: {len(selected)}/{target}')
    if window_target == 0:
        return selected
    roots = list(selected); rng.shuffle(roots); windows = []; per_root = Counter()
    queues = [(root, _window_candidates(root, episodes[root['_meta']['episode']])) for root in roots]
    while len(windows) < window_target:
        changed = False
        for root, queue in queues:
            parent = root['_meta']['id']
            while queue and (queue[0]['_meta']['request_sha256'] in seen
                    or families[queue[0]['_meta']['group_id']] >= family_cap):
                queue.pop(0)
            if queue and per_root[parent] < MAX_WINDOWS_PER_ROOT:
                row = queue.pop(0); windows.append(row); seen.add(row['_meta']['request_sha256'])
                families[row['_meta']['group_id']] += 1; per_root[parent] += 1; changed = True
                if len(windows) == window_target:
                    break
        if not changed:
            raise ValueError(f'Insufficient nearby same-life informative windows: {len(windows)}/{window_target}')
    return selected + windows


def load_old_expert_records(data_directory=ROOT / 'data'):
    """Canonical v2 TRAIN labels with their original verified replay provenance."""
    from teacher_data import prepare_dataset
    directory = Path(data_directory); manifest = prepare_dataset(directory)
    filename = 'pacman-native-v2-train.jsonl'
    records = _rows(directory / filename)
    binding = {'dataset': 'pacman-native-v2', 'split': 'train', 'train_file': filename,
        'train_sha256': _hash(directory / filename), 'manifest_sha256': _hash(directory / 'pacman-native-v2-manifest.json'),
        'replay_bundle': manifest['replay_bundle'], 'replay_sha256': manifest['replay_sha256'],
        'qualification_sha256': manifest['qualification_sha256'], 'source_sha256': manifest['source_sha256']}
    result = []
    for original in records:
        record = copy.deepcopy(original); meta = record['_meta']; source_id = meta['id']
        record['_meta'] = {'id': f'{PREFIX}:old:{source_id}', 'split': 'train', 'class': 'replay',
            'origin': 'canonical_v2_expert_train', 'group_id': meta['group_id'],
            'state_sha256': fingerprint(record['state']), 'request_sha256': request_fingerprint(record),
            'cohorts': {}, 'source_record_sha256': fingerprint(original), 'source_id': source_id,
            'source_binding': binding, 'label_source': 'Unchanged canonical v2 qualified-teacher TRAIN label',
            'learner_disagreement_measured': False, 'anticipatory_harm_measured': False}
        result.append(record)
    return result, binding


def _select_replay(records, existing, count, family_cap=FAMILY_CAP):
    if count == 0:
        return []
    seen = {r['_meta']['request_sha256'] for r in existing}; families = Counter(r['_meta']['group_id'] for r in existing)
    available = list(records); random.Random(113).shuffle(available); selected = []
    for row in available:
        meta = row['_meta']; digest = meta['request_sha256']
        if digest in seen or families[meta['group_id']] >= family_cap:
            continue
        selected.append(row); seen.add(digest); families[meta['group_id']] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Insufficient canonical old expert replay: {len(selected)}/{count}')
    return selected


def export_evidence(evidence_directory, filename):
    """Portable receipt/full-native-trace bundle; include rejected evidence too."""
    directory = Path(evidence_directory); _load_pool(directory)
    config = _load(directory / 'offline-config.json'); qualification = _load(directory / 'qualification.json')
    names = {'offline-config.json', 'qualification.json', qualification['replay_bundle']}
    for name in ('index.json', 'pool-report.json'):
        if (directory / name).is_file():
            names.add(name)
    for path in (directory / 'episodes').glob('*/receipt.json'):
        receipt = _load(path); _receipt_files(directory, receipt)
        names.add(path.relative_to(directory).as_posix()); names.update(receipt['files'])
    if _hash(directory / 'qualification.json') != config['qualification_sha256']:
        raise ValueError('Copied qualification receipt changed')
    filename = Path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + '.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(names):
            archive.write(_safe_path(directory, name), name)
    temporary.replace(filename)
    return {'filename': filename.name, 'sha256': _hash(filename), 'files': {name: _hash(directory / name) for name in sorted(names)}}


def build_dataset(evidence_directory, output_directory, qualification_path, old_data_directory=ROOT / 'data'):
    """Full fixed recipe only. Shortage diagnostics never become a partial dataset."""
    require_qualified_teacher(qualification_path)
    pools, episodes, duplicates = _load_pool(evidence_directory)
    config = _load(Path(evidence_directory) / 'offline-config.json')
    if _hash(qualification_path) != config['qualification_sha256']:
        raise ValueError('Offline data uses another qualification')
    report_pool(evidence_directory)
    chosen = {'development': select_partition(pools['development'], episodes, TARGETS['development'],
                                               MINIMUMS['development'], seed=108)}
    chosen['train'] = select_partition(pools['train'], episodes, TARGETS['train'], MINIMUMS['train'], seed=107,
        excluded_request_sha256={r['_meta']['request_sha256'] for r in chosen['development']})
    old, old_binding = load_old_expert_records(old_data_directory)
    # Canonical replay can never duplicate a new development request.
    all_new = chosen['train'] + chosen['development']
    chosen['train'] += _select_replay(old, all_new, TARGETS['train']['replay'])
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    files = {}; counts = {}; coverage = {}; groups = {}; labels = {}
    for split, rows in chosen.items():
        filename = f'{PREFIX}-{split}.jsonl'; _write_rows(output / filename, rows)
        files[filename] = _hash(output / filename); counts[split] = dict(Counter(r['_meta']['class'] for r in rows))
        coverage[split] = _coverage([r for r in rows if r['_meta']['class'] == 'targeted'])
        groups[split] = dict(Counter(r['_meta']['group_id'] for r in rows))
        labels[split] = dict(Counter(r['questions']['move']['label'] for r in rows))
    bundle = export_evidence(evidence_directory, output / f'{PREFIX}-evidence.zip')
    manifest = {'dataset': PREFIX, 'schema': 'teacher-only-exact-native-trajectories-v1',
        'protocol': SPEC, 'source_sha256': config['source_sha256'], 'exporter_sha256': config['exporter_sha256'],
        'qualification_sha256': config['qualification_sha256'], 'recipe': RECIPE, 'targets': TARGETS,
        'minimums': MINIMUMS, 'family_cap': FAMILY_CAP, 'counts': counts, 'coverage': coverage,
        'families': groups, 'labels': labels, 'files': files, 'evidence': bundle, 'duplicates': duplicates,
        'source_receipts': {identifier: e['receipt_sha256'] for identifier, e in episodes.items()},
        'old_expert_source': old_binding, 'interpretation': INTERPRETATION,
        'expected_training_requests': sum(TARGETS['train'].values()) + RECIPE['replay'],
        'expected_optimizer_steps': math.ceil((sum(TARGETS['train'].values()) + RECIPE['replay']) / 8),
        'train_seeds': sorted({r['_meta']['seed'] for r in chosen['train'] if r['_meta']['class'] != 'replay'}),
        'development_seeds': sorted({r['_meta']['seed'] for r in chosen['development']}),
        'initialization': {'parent_checkpoint': 'kev-4b-pacman-native-v3', 'optimizer': 'fresh'},
        'teacher_cpu_replay_verified': True, 'gpu_training_performed': False}
    _save(output / f'{PREFIX}-manifest.json', manifest)
    validate_dataset(output, qualification_path, old_data_directory=old_data_directory)
    return manifest


def validate_dataset(output_directory, qualification_path, verify_native=False, old_data_directory=ROOT / 'data'):
    """Bind counts, labels, full source receipts and exact recipe; optionally replay again."""
    output = Path(output_directory); manifest = _load(output / f'{PREFIX}-manifest.json')
    require_qualified_teacher(qualification_path)
    if (manifest['dataset'] != PREFIX or manifest['schema'] != 'teacher-only-exact-native-trajectories-v1'
            or manifest['protocol'] != SPEC or manifest['source_sha256'] != source_hashes()
            or manifest['exporter_sha256'] != _hash(Path(__file__)) or manifest['qualification_sha256'] != _hash(qualification_path)
            or manifest['recipe'] != RECIPE or manifest['targets'] != TARGETS or manifest['minimums'] != MINIMUMS
            or manifest['family_cap'] != FAMILY_CAP or manifest['interpretation'] != INTERPRETATION
            or manifest['expected_training_requests'] != 6096 or manifest['expected_optimizer_steps'] != 762
            or manifest['initialization'] != {'parent_checkpoint': 'kev-4b-pacman-native-v3', 'optimizer': 'fresh'}
            or not manifest['teacher_cpu_replay_verified'] or manifest['gpu_training_performed']):
        raise ValueError('Offline dataset/configuration/recipe differs')
    expected_files = {f'{PREFIX}-{split}.jsonl' for split in TARGETS}
    if set(manifest['files']) != expected_files:
        raise ValueError('Unexpected offline data files')
    for name, digest in manifest['files'].items():
        if _hash(output / name) != digest:
            raise ValueError('Offline dataset file checksum mismatch')
    bundle = manifest['evidence']; bundle_path = _safe_path(output, bundle['filename'])
    if _hash(bundle_path) != bundle['sha256']:
        raise ValueError('Offline source evidence bundle changed')
    old, old_binding = load_old_expert_records(old_data_directory)
    if old_binding != manifest['old_expert_source']:
        raise ValueError('Canonical old expert provenance changed')
    old_lookup = {r['_meta']['source_id']: r for r in old}
    with zipfile.ZipFile(bundle_path) as archive:
        if set(archive.namelist()) != set(bundle['files']) or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Unexpected/duplicate offline evidence members')
        contents = {name: archive.read(name) for name in bundle['files']}
        for name, digest in bundle['files'].items():
            if hashlib.sha256(contents[name]).hexdigest() != digest:
                raise ValueError('Offline evidence entry checksum mismatch')
        receipts = {}; proofs = {}
        for identifier, digest in manifest['source_receipts'].items():
            name = f'episodes/{identifier}/receipt.json'
            if hashlib.sha256(contents[name]).hexdigest() != digest:
                raise ValueError('Source receipt hash mismatch')
            receipt = json.loads(contents[name]); receipts[identifier] = receipt
            _validate_source_binding(receipt, json.loads(contents[f'episodes/{identifier}/source-config.json']),
                                     json.loads(contents[f'episodes/{identifier}/source-attempt.json']))
            if (not receipt['accepted'] or receipt['failures'] or receipt['metrics']['life_losses']
                    or episode_quality_failures(receipt['metrics']) or not receipt['native_replayed']
                    or receipt['binding']['qualification_sha256'] != manifest['qualification_sha256']
                    or receipt['binding']['source_sha256'] != manifest['source_sha256']
                    or receipt['binding']['exporter_sha256'] != manifest['exporter_sha256']):
                raise ValueError('Source teacher episode is not qualified/bound')
            for filename, digest in receipt['files'].items():
                if hashlib.sha256(contents[filename]).hexdigest() != digest:
                    raise ValueError('Receipt references changed source evidence')
            attempt_name = f'episodes/{identifier}/source-attempt.json'
            trace_name = f'episodes/{identifier}/source-trace.jsonl'
            if (hashlib.sha256(contents[attempt_name]).hexdigest() != receipt['binding']['source_attempt_sha256']
                    or hashlib.sha256(contents[trace_name]).hexdigest() != receipt['binding']['source_trace_sha256']):
                raise ValueError('Full native prefix/attempt hashes differ')
            rows = [json.loads(line) for line in contents[f'episodes/{identifier}/verified.jsonl'].splitlines()]
            if len(rows) != receipt['verified_teacher_rows']:
                raise ValueError('Source receipt row count differs')
            proofs[identifier] = {p['source_index']: p for p in rows}
            if len(proofs[identifier]) != len(rows):
                raise ValueError('Duplicate source decision indices')
            if verify_native:
                import tempfile
                with tempfile.TemporaryDirectory(prefix='offline-native-replay-') as directory:
                    path = Path(directory) / 'trace.jsonl'; path.write_bytes(contents[trace_name])
                    replay, measured = _replay_trace(path, receipt['binding']['job'], json.loads(contents[attempt_name]))
                expected = {k: receipt[k] for k in replay}
                if fingerprint(replay) != fingerprint(expected) or fingerprint(measured) != fingerprint(rows):
                    raise ValueError('Fresh validation replay differs from committed receipt')
        seen = set(); seeds = {}; all_ids = set(); split_rows = {}
        for split, budget in TARGETS.items():
            rows = _rows(output / f'{PREFIX}-{split}.jsonl'); split_rows[split] = rows
            measured_counts = dict(Counter(r['_meta']['class'] for r in rows))
            expected_counts = {k: n for k, n in budget.items() if n}
            if measured_counts != expected_counts or measured_counts != manifest['counts'][split]:
                raise ValueError('Offline class counts differ')
            selected_roots = {r['_meta']['id']: r for r in rows if r['_meta']['class'] == 'targeted'}
            per_root = Counter(); families = Counter()
            for row in rows:
                meta = row['_meta']; digest = request_fingerprint(row)
                if (set(row) != {'state', 'questions', '_meta'} or meta['split'] != split
                        or meta['request_sha256'] != digest or meta['state_sha256'] != fingerprint(row['state'])
                        or digest in seen or meta['id'] in all_ids
                        or meta['learner_disagreement_measured'] or meta['anticipatory_harm_measured']):
                    raise ValueError('Offline request/identity/dedup/input boundary differs')
                seen.add(digest); all_ids.add(meta['id']); families[meta['group_id']] += 1
                if meta['class'] == 'replay':
                    if split != 'train' or row != old_lookup.get(meta['source_id']):
                        raise ValueError('Old replay label/input/provenance differs from canonical training source')
                    continue
                identifier = meta['episode']; receipt = receipts[identifier]; job = receipt['binding']['job']
                proof = proofs[identifier][meta['source_index']]
                if job['split'] != split or seeds.setdefault(job['seed'], split) != split:
                    raise ValueError('Source seed groups overlap partitions')
                expected = _record(proof, receipt, identifier, split, meta['class'], meta['parent_targeted_id'], meta['window_kind'])
                if row != expected or proof['request'] != _request(body(row['state'])):
                    raise ValueError('Offline gold/state/options/provenance differs from verified source')
                risks = proof['diagnostics']['immediate_counterfactuals']
                if set(risks) != set(row['state']['legal_moves']) or risks[proof['choice']]['life_lost']:
                    raise ValueError('Gold action is unsafe or a legal alternative was removed')
                if meta['cohorts'] != cohort_flags(row['state'], risks, proof['power_transition']):
                    raise ValueError('Offline source cohort differs')
                if meta['class'] == 'targeted':
                    if meta['parent_targeted_id'] or meta['window_kind'] or not is_targeted(row['state'], risks, proof['choice']):
                        raise ValueError('Easy/postrespawn-only state counted as targeted')
                elif meta['class'] == 'informative':
                    parent = selected_roots.get(meta['parent_targeted_id'])
                    if parent is None or parent['_meta']['episode'] != identifier or parent['_meta']['life_index'] != meta['life_index']:
                        raise ValueError('Informative frame is not tied to a selected same-life targeted root')
                    episode = {'receipt': receipt, 'proofs': list(proofs[identifier].values())}
                    if row not in _window_candidates(parent, episode):
                        raise ValueError('Informative frame is not an actual bounded lead-in/recovery window')
                    per_root[parent['_meta']['id']] += 1
            if any(n > FAMILY_CAP for n in families.values()) or any(n > MAX_WINDOWS_PER_ROOT for n in per_root.values()):
                raise ValueError('Offline family/window cap exceeded')
            coverage = _coverage(list(selected_roots.values()))
            if (coverage != manifest['coverage'][split] or dict(families) != manifest['families'][split]
                    or dict(Counter(r['questions']['move']['label'] for r in rows)) != manifest['labels'][split]
                    or any(coverage.get(name, 0) < floor for name, floor in MINIMUMS[split].items())):
                raise ValueError('Offline targeted-only floors/coverage/labels differ')
        if (manifest['train_seeds'] != sorted(s for s, split in seeds.items() if split == 'train')
                or manifest['development_seeds'] != sorted(s for s, split in seeds.items() if split == 'development')):
            raise ValueError('Offline source seed manifest differs')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify'); verify.add_argument('--collection', type=Path, required=True)
    verify.add_argument('--evidence', type=Path, required=True); verify.add_argument('--qualification', type=Path, required=True)
    verify.add_argument('--workers', type=int, default=10)
    report = sub.add_parser('report'); report.add_argument('--evidence', type=Path, required=True)
    build = sub.add_parser('build'); build.add_argument('--evidence', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True); build.add_argument('--qualification', type=Path, required=True)
    validate = sub.add_parser('validate'); validate.add_argument('--out', type=Path, required=True)
    validate.add_argument('--qualification', type=Path, required=True); validate.add_argument('--replay', action='store_true')
    export = sub.add_parser('export'); export.add_argument('--evidence', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify_sources(args.collection, args.evidence, args.qualification, args.workers)
    elif args.command == 'report':
        result = report_pool(args.evidence)
    elif args.command == 'build':
        result = build_dataset(args.evidence, args.out, args.qualification)
    elif args.command == 'validate':
        result = validate_dataset(args.out, args.qualification, args.replay)
    else:
        result = export_evidence(args.evidence, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
