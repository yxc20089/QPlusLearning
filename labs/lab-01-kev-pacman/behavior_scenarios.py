"""Measured native recovery scenarios, without querying a learner model.

An exposure is not a harmful action. This separate generator measures complete
forced-first branches under the same frozen teacher and admits only an exact,
zero-loss winning positive. Source suffix reuse requires the original full
action prefix and an independently replayed source receipt. Every rejected
proof is retained. No current-v3 disagreement or global optimality is claimed.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import uuid

from gameplay_benchmark import BenchmarkEngine, SPEC, diagnostic_record, summarize
from pacman_lab import ROOT, body, destination, distances, position
import teacher_offline_data as offline
from teacher_validation import episode_quality_failures, fingerprint, require_qualified_teacher, source_hashes
import v4_data as v4

PREFIX = 'pacman-native-v4-behavior'
RULES = {'near_ghost_tiles': 3, 'scout_ghost_tiles': 6, 'expiry_scout_frames': 32,
         'dry_decisions': 24, 'food_followthrough_decisions': 32,
         'minimum_route_extra_decisions': 8, 'minimum_route_extra_frames': 60,
         'window_offsets': [1, 4, 8, 16, 24, 32], 'max_windows': 4}
_CANCEL = None


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _save(path, value, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    if immutable:
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if _load(path) != value:
                    raise ValueError('Committed behavior artifact changed')
        finally:
            temporary.unlink()
    else:
        temporary.replace(path)


def _write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    temporary.replace(path)


def _copy(source, target):
    source, target = Path(source), Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _hash(source) != _hash(target):
            raise ValueError('Immutable copied source changed')
        return
    temporary = target.with_name(target.name + f'.{uuid.uuid4().hex}.tmp')
    shutil.copyfile(source, temporary)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _hash(temporary) != _hash(target):
                raise ValueError('Concurrent source copy differs')
    finally:
        temporary.unlink()


def _check():
    if _CANCEL is not None and _CANCEL.is_set():
        raise InterruptedError('Behavior generation interrupted; committed cases remain resumable')


def _worker_setup(cancel):
    global _CANCEL
    _CANCEL = cancel
    # The existing exact-prefix helper also checks this cooperative flag.
    v4._initialize_worker(cancel)


def _near(state, frightened=None):
    routes = distances(state['maze'], position(state['player']))
    return min((routes.get(position(ghost), 999) for ghost in state['ghosts']
                if ghost['mode'] == 'outside' and (frightened is None or ghost['frightened'] == frightened)), default=999)


def _configuration(collection, evidence, directory, qualification_path):
    collection, evidence, directory = map(lambda p: Path(p).resolve(), (collection, evidence, directory))
    if directory == collection or directory == evidence or directory.is_relative_to(collection) or directory.is_relative_to(evidence):
        raise ValueError('Keep behavior evidence separate from frozen source collections')
    teacher = require_qualified_teacher(qualification_path)
    offline._collection_config(collection, qualification_path)
    # Check every full-source file/job/qualification binding. Recomputing broad
    # exposure tags for every row is unnecessary here: selected roots and both
    # branches are independently replayed again before behavior admission.
    source_config = _load(evidence / 'offline-config.json')
    if (source_config['source_sha256'] != source_hashes()
            or source_config['exporter_sha256'] != _hash(ROOT / 'teacher_offline_data.py')):
        raise ValueError('Independent source replay configuration changed')
    for path in (evidence / 'episodes').glob('*/receipt.json'):
        receipt = _load(path); offline._receipt_files(evidence, receipt)
        offline._validate_source_binding(receipt, _load(path.with_name('source-config.json')),
                                        _load(path.with_name('source-attempt.json')))
        if (not receipt['native_replayed'] or receipt['binding']['source_sha256'] != source_config['source_sha256']
                or receipt['binding']['exporter_sha256'] != source_config['exporter_sha256']
                or receipt['binding']['qualification_sha256'] != source_config['qualification_sha256']
                or receipt['binding']['collection_fixed_sha256'] != source_config['collection_fixed_sha256']):
            raise ValueError('Unbound independent full-source replay receipt')
    fixed = {'dataset': PREFIX, 'rules': RULES, 'protocol': SPEC, 'teacher_options': teacher['options'],
             'source_sha256': source_hashes(), 'generator_sha256': _hash(Path(__file__)),
             'v4_generator_sha256': _hash(ROOT / 'v4_data.py'),
             'offline_exporter_sha256': _hash(ROOT / 'teacher_offline_data.py'),
             'qualification_sha256': _hash(qualification_path),
             'collection_fixed_sha256': fingerprint(offline._fixed_collection(_load(collection / 'collection.json'))),
             'offline_config_sha256': _hash(evidence / 'offline-config.json'),
             'interpretation': {'learner_disagreement': False, 'global_optimality': False,
               'negative_controller': 'One forced legal first action, then the same frozen qualified teacher at every subsequent state.',
               'positive_gate': 'Complete native level clear with zero teacher-controlled life losses; original movement, cycle and stall gates unchanged.'}}
    directory.mkdir(parents=True, exist_ok=True)
    _save(directory / 'behavior-config.json', fixed, immutable=True)
    _copy(qualification_path, directory / 'qualification.json')
    _copy(Path(qualification_path).parent / teacher['replay_bundle'], directory / teacher['replay_bundle'])
    _copy(evidence / 'offline-config.json', directory / 'source-evidence/offline-config.json')
    return fixed


def _assert_configuration(directory, qualification_path=None):
    directory = Path(directory); config = _load(directory / 'behavior-config.json')
    if (config['dataset'] != PREFIX or config['rules'] != RULES or config['protocol'] != SPEC
            or config['source_sha256'] != source_hashes() or config['generator_sha256'] != _hash(Path(__file__))
            or config['v4_generator_sha256'] != _hash(ROOT / 'v4_data.py')
            or config['offline_exporter_sha256'] != _hash(ROOT / 'teacher_offline_data.py')
            or config['qualification_sha256'] != _hash(directory / 'qualification.json')
            or config['offline_config_sha256'] != _hash(directory / 'source-evidence/offline-config.json')):
        raise ValueError('Behavior source/protocol/configuration changed; use a new output directory')
    if qualification_path is not None:
        teacher = require_qualified_teacher(qualification_path)
        if _hash(qualification_path) != config['qualification_sha256'] or teacher['options'] != config['teacher_options']:
            raise ValueError('Behavior qualification/teacher options changed')
    return config


def _bind_source(evidence, directory, identifier):
    source = Path(evidence) / 'episodes' / identifier
    receipt = _load(source / 'receipt.json')
    if not receipt['accepted'] or receipt['metrics']['life_losses'] or receipt['failures']:
        raise ValueError('Behavior root lacks an accepted zero-loss independent source game')
    for name in ('receipt.json', 'source-trace.jsonl', 'source-attempt.json', 'source-config.json', 'verified.jsonl'):
        _copy(source / name, Path(directory) / 'source-evidence/episodes' / identifier / name)
    return receipt


def _source_candidate(identifier, receipt, source, proof):
    index = proof['source_index']; job = receipt['binding']['job']; state = proof['request']['state']
    prefix = [r['diagnostics']['choice'] for r in source[:index]]
    hashes = [fingerprint(r['request']['state']) for r in source[:index]] + [fingerprint(state)]
    candidate = {'id': fingerprint({'episode': identifier, 'source_index': index, 'state': fingerprint(state)})[:24],
        'split': job['split'], 'seed': job['seed'], 'level': job['level'],
        'family': f'level-{job["level"]}-seed-{job["seed"]}', 'origin': 'verified_teacher_trajectory',
        'source': {'kind': 'verified_teacher_episode', 'episode': identifier,
                   'trace_sha256': receipt['binding']['source_trace_sha256'], 'root_index': index},
        'request': body(state), 'state_sha256': fingerprint(state), 'prefix_actions': prefix,
        'prefix_state_sha256': hashes, 'teacher_choice_hint': proof['choice']}
    return candidate


def _alternatives(state, risks, plan, choice):
    # Teacher portfolio predictions are scout priorities, never admission proof.
    portfolio = {r['action']: r for r in (plan or {}).get('candidates', [])}
    ghosts = [g for g in state['ghosts'] if g['mode'] == 'outside']
    result = []
    for move in state['legal_moves']:
        if move == choice:
            continue
        predicted = portfolio.get(move, {}).get('scenarios', [])
        predicted_losses = sum(not r['survived'] for r in predicted)
        tile = destination(state['maze'], position(state['player']), move)
        routes = distances(state['maze'], tile)
        proximity = min((routes.get(position(g), 999) for g in ghosts), default=999)
        result.append((predicted_losses, int(risks.get(move, {}).get('life_lost', False)), -proximity, move))
    return [r[-1] for r in sorted(result, reverse=True)]


def scout_scenarios(collection_directory, evidence_directory, output_directory, qualification_path,
                    max_roots=256, wave_id='pilot-001'):
    """Cheap full-source prioritization; no weak state predicate is called harm."""
    if isinstance(max_roots, bool) or not isinstance(max_roots, int) or not 1 <= max_roots <= 10000:
        raise ValueError('Use a bounded 1..10000 scout root budget')
    config = _configuration(collection_directory, evidence_directory, output_directory, qualification_path)
    directory = Path(output_directory); evidence = Path(evidence_directory); collection = Path(collection_directory)
    candidates = []; counts = Counter(); seen = set(); accepted_sources = set()
    for path in sorted((evidence / 'episodes').glob('*/receipt.json')):
        _check(); receipt = _load(path)
        if not receipt['accepted']:
            continue
        identifier = path.parent.name; accepted_sources.add(identifier)
        source = _rows(path.with_name('source-trace.jsonl'))
        for proof in _rows(path.with_name('verified.jsonl')):
            state = proof['request']['state']; risks = proof['diagnostics']['immediate_counterfactuals']; choice = proof['choice']
            if len(state['legal_moves']) < 2 or risks[choice]['life_lost']:
                continue
            near, danger = _near(state), _near(state, False)
            remaining = state['timing']['power']['remaining_frames']; index = proof['source_index']
            all_safe = all(not r['life_lost'] for r in risks.values())
            plan = source[index].get('teacher', {})
            predicted_losses = max((sum(not s['survived'] for s in c.get('scenarios', []))
                                    for c in plan.get('candidates', []) if c['action'] != choice), default=0)
            flags = {'expiry': bool(state['frightened'] and 0 < remaining <= RULES['expiry_scout_frames'] and near <= 6),
                     'anticipatory': bool(all_safe and danger <= 6),
                     'dry_recovery': bool(state.get('decisions_since_last_pellet', 0) >= RULES['dry_decisions'])}
            for name, flag in flags.items():
                counts[f'{receipt["binding"]["job"]["split"]}/{name}_raw'] += flag
            if not any(flags.values()):
                continue
            # Protect rarity first, then multi-scenario warnings and close traps.
            score = (int(flags['expiry'] and any(r['life_lost'] for r in risks.values())),
                     int(flags['expiry']), predicted_losses, int(flags['anticipatory']), -min(danger, 20),
                     int(flags['dry_recovery']), -remaining if flags['expiry'] else 0)
            candidate = _source_candidate(identifier, receipt, source, proof)
            key = offline.request_fingerprint(candidate['request'])
            if key in seen:
                continue
            seen.add(key)
            candidates.append((score, candidate, risks, plan, flags, path))
    # Also consider coherent legal perturbations; these must earn a new winning
    # positive rather than borrow a different native state's source suffix.
    for path in sorted((collection / 'cases').glob('*/candidate.json')):
        candidate = _load(path); source_id = candidate['source']['episode']
        if source_id not in accepted_sources or 'perturbation' not in candidate['source']:
            continue
        state = candidate['request']['state']; remaining = state['timing']['power']['remaining_frames']
        flags = {'expiry': bool(state['frightened'] and 0 < remaining <= 32 and _near(state) <= 6),
                 'anticipatory': _near(state, False) <= 3,
                 'dry_recovery': state.get('decisions_since_last_pellet', 0) >= 24}
        for name, flag in flags.items():
            counts[f'{candidate["split"]}/{name}_fork_raw'] += flag
        if not any(flags.values()):
            continue
        key = offline.request_fingerprint(candidate['request'])
        if key in seen:
            continue
        seen.add(key)
        score = (0, int(flags['expiry']), 0, int(flags['anticipatory']), -min(_near(state, False), 20),
                 int(flags['dry_recovery']), -remaining if flags['expiry'] else 0)
        candidates.append((score, candidate, None, None, flags, path))
    # Round-robin behavior/split/family prevents a single convenient source or
    # many low-clock exposures supplying the entire pilot. Warnings rank roots
    # inside each bucket; they never become admitted labels by themselves.
    buckets = {}
    for item in sorted(candidates, key=lambda r: (r[0], r[1]['id']), reverse=True):
        candidate = item[1]; flags = item[4]
        kind = 'expiry' if flags['expiry'] else 'anticipatory' if flags['anticipatory'] else 'dry_recovery'
        buckets.setdefault((kind, candidate['split'], candidate['family']), []).append(item)
    selected = []; keys = sorted(buckets, key=lambda k: (k[2], k[1] != 'development', k[0]))
    while len(selected) < max_roots and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < max_roots:
                selected.append(buckets[key].pop(0))
    jobs = []
    for score, candidate, risks, plan, flags, path in selected:
        identifier = candidate['source']['episode']
        _bind_source(evidence, directory, identifier)
        original = copy.deepcopy(candidate)
        source_binding = {'episode': identifier,
             'receipt': f'source-evidence/episodes/{identifier}/receipt.json',
             'receipt_sha256': _hash(directory / f'source-evidence/episodes/{identifier}/receipt.json'),
             'original_candidate_sha256': _hash(path) if path.name == 'candidate.json' else None,
             'source_index': candidate['source']['root_index']}
        moves = _alternatives(candidate['request']['state'], risks or {}, plan,
                              candidate.get('teacher_choice_hint'))
        if not moves:
            continue
        case_id = fingerprint({'candidate': original, 'alternative': moves[0], 'source_binding': source_binding})[:24]
        candidate.update(id=case_id, source_binding=source_binding, alternative= moves[0],
                         scout_flags=flags, scout_score=list(score), reuse_source_suffix=risks is not None)
        folder = directory / 'cases' / case_id
        _save(folder / 'candidate.json', candidate, immutable=True)
        jobs.append({'id': case_id, 'split': candidate['split'], 'family': candidate['family'],
                     'candidate_sha256': _hash(folder / 'candidate.json')})
    wave = {'wave_id': wave_id, 'source_receipts_sha256': fingerprint(sorted(_hash(p) for p in
             (evidence / 'episodes').glob('*/receipt.json'))), 'max_roots': max_roots,
             'candidates': jobs, 'raw_scout_counts': dict(counts), 'unique_scout_roots': len(candidates),
             'selected_roots': len(jobs), 'generator_sha256': config['generator_sha256']}
    _save(directory / 'scout-waves' / f'{wave_id}.json', wave, immutable=True)
    _save(directory / 'scout-report.json', wave)
    print(f'[behavior/scout] {len(jobs)} selected / {len(candidates)} distinct plausible roots; no harmful-alternative labels yet', flush=True)
    return wave


def _one_hot(state, move):
    return {'answers': {'move': {'choice': move, 'probabilities': {d: float(d == move) for d in state['legal_moves']}}}}


def _transition(state, step):
    return {'remaining_before': state['timing']['power']['remaining_frames'],
            'remaining_after': step['state']['timing']['power']['remaining_frames'],
            'frightened_before': state['frightened'], 'frightened_after': step['state']['frightened'],
            'action_frames': step['action_frames'], 'outcome': step['outcome'],
            'outside_frightened_nearest_before': _near(state, True),
            'outside_dangerous_nearest_before': _near(state, False)}


def _run_branch(candidate, options, trace, forced_first):
    rows = []; diagnostics = []; life = frames = stale = 0; status = 'decision_cap'
    with BenchmarkEngine() as engine:
        state = v4._replay_prefix(engine, candidate)
        for index in range(SPEC['max_decisions']):
            _check(); risks = engine.request('risks')
            plan = engine.request('teacher', options=options) if index else None
            move = forced_first if index == 0 else plan['choice']
            step = engine.step(move); diag = diagnostic_record(state, _one_hot(state, move), step, risks, life)
            diagnostics.append(diag)
            rows.append({'state_sha256': fingerprint(state), 'request': offline._request(body(state)),
                         'choice': move, 'diagnostics': diag, 'power_transition': _transition(state, step),
                         'teacher_plan': plan})
            frames += step['action_frames']; stale = stale + 1 if step['state']['pellets_remaining'] == state['pellets_remaining'] else 0
            state = step['state']
            if step['outcome'] == 'level_cleared':
                status = 'level_cleared'; break
            if step['outcome'] == 'life_lost':
                continuation = engine.request('continue')
                if continuation['status'] != 'playing':
                    status = 'game_over'; break
                state = continuation['state']; life += 1; stale = 0
            if stale >= SPEC['max_no_pellet_decisions']:
                status = 'no_progress_watchdog'; break
            if frames >= SPEC['max_simulation_frames']:
                status = 'simulation_frame_cap'; break
    _write_rows(trace, rows)
    return summarize(diagnostics, status)


def _reused_positive(directory, candidate, trace):
    identifier = candidate['source']['episode']; folder = Path(directory) / 'source-evidence/episodes' / identifier
    receipt = _load(folder / 'receipt.json'); source = _rows(folder / 'source-trace.jsonl')
    proofs = {row['source_index']: row for row in _rows(folder / 'verified.jsonl')}
    index = candidate['source']['root_index']; root = source[index]
    if (not receipt['accepted'] or receipt['metrics']['life_losses'] or root['request'] != candidate['request']
            or candidate['prefix_actions'] != [row['diagnostics']['choice'] for row in source[:index]]
            or candidate['prefix_state_sha256'] != [fingerprint(row['request']['state']) for row in source[:index]] + [candidate['state_sha256']]
            or receipt['binding']['source_trace_sha256'] != _hash(folder / 'source-trace.jsonl')):
        raise ValueError('Source suffix is not bound to this exact native action prefix')
    rows = []; diagnostics = []
    for offset, original in enumerate(source[index:]):
        if original['controller'] != 'qualified_teacher':
            raise ValueError('Positive suffix contains an unqualified source action')
        proof = proofs[index + offset]; diag = copy.deepcopy(original['diagnostics'])
        diag['life_index'] = 0; diag['strata']['post_respawn'] = False
        if diag['outcome'] == 'life_lost':
            raise ValueError('Positive reused suffix loses a life')
        transition = {**proof['power_transition'], 'frightened_before': original['request']['state']['frightened'],
             'frightened_after': None if offset == len(source[index:]) - 1 else source[index + offset + 1]['request']['state']['frightened'],
             'outcome': diag['outcome'], 'outside_frightened_nearest_before': _near(original['request']['state'], True),
             'outside_dangerous_nearest_before': _near(original['request']['state'], False)}
        rows.append({'state_sha256': proof['state_sha256'], 'request': offline._request(original['request']),
                     'choice': proof['choice'], 'diagnostics': diag, 'power_transition': transition,
                     'teacher_plan': original['teacher']})
        diagnostics.append(diag)
    _write_rows(trace, rows)
    return summarize(diagnostics, 'level_cleared')


def _replay_branch(candidate, path, metrics):
    rows = _rows(path); diagnostics = []; life = frames = stale = 0; status = None
    with BenchmarkEngine() as engine:
        state = v4._replay_prefix(engine, candidate)
        for index, row in enumerate(rows):
            _check()
            if status or row['request'] != offline._request(body(state)) or fingerprint(state) != row['state_sha256']:
                raise ValueError('Independent continuation replay observation differs')
            risks = engine.request('risks'); step = engine.step(row['choice'])
            diag = diagnostic_record(state, _one_hot(state, row['choice']), step, risks, life)
            if fingerprint(diag) != fingerprint(row['diagnostics']):
                raise ValueError('Independent continuation replay transition differs')
            measured = _transition(state, step)
            # Source's terminal observation is not in its original JSONL; check
            # only its recorded clock/action/outcome there, never invent one.
            for name in ('remaining_before', 'remaining_after', 'action_frames', 'outcome'):
                if measured[name] != row['power_transition'][name]:
                    raise ValueError('Native power transition differs from its proof')
            diagnostics.append(diag); frames += step['action_frames']
            stale = stale + 1 if step['state']['pellets_remaining'] == state['pellets_remaining'] else 0
            state = step['state']
            if step['outcome'] == 'level_cleared':
                status = 'level_cleared'
            elif step['outcome'] == 'life_lost':
                continuation = engine.request('continue')
                if continuation['status'] != 'playing':
                    status = 'game_over'
                else:
                    state = continuation['state']; life += 1; stale = 0
            if status is None and stale >= SPEC['max_no_pellet_decisions']:
                status = 'no_progress_watchdog'
            if status is None and frames >= SPEC['max_simulation_frames']:
                status = 'simulation_frame_cap'
        if status is None:
            if len(rows) != SPEC['max_decisions']:
                raise ValueError('Proof stopped before a terminal native/watchdog boundary')
            status = 'decision_cap'
    if fingerprint(summarize(diagnostics, status)) != fingerprint(metrics):
        raise ValueError('Independently replayed continuation metrics differ')
    return True


def classify_behavior(state, risks, teacher_choice, alternative, positive, negative, positive_rows, negative_rows):
    """Recomputable exact predicates; a portfolio warning never admits a root."""
    all_safe = all(not risk['life_lost'] for risk in risks.values())
    later_failure = negative['life_losses'] > 0 or not negative['level_cleared']
    expiry = negative_rows[0]['power_transition']
    actual_expiry = bool(risks[alternative]['life_lost'] and state['frightened']
        and expiry['remaining_before'] > 0 and expiry['remaining_after'] == 0
        and expiry['outside_frightened_nearest_before'] <= RULES['near_ghost_tiles']
        and expiry['outside_dangerous_nearest_before'] > RULES['near_ghost_tiles']
        and not risks[teacher_choice]['life_lost'])
    first_food = next((index for index, row in enumerate(positive_rows)
                       if row['diagnostics']['pellets_after'] < row['diagnostics']['pellets_before']), None)
    reverse = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}[state['player']['heading']]
    route_extra = {'decisions': negative['decisions'] - positive['decisions'],
                   'frames': negative['simulation_frames'] - positive['simulation_frames']}
    productive = bool(state.get('decisions_since_last_pellet', 0) >= RULES['dry_decisions']
        and teacher_choice == reverse and first_food is not None and 1 <= first_food <= RULES['food_followthrough_decisions']
        and all(row['diagnostics']['outcome'] != 'life_lost' for row in positive_rows[:first_food + 1])
        and negative['life_losses'] == 0 and negative['level_cleared']
        and route_extra['decisions'] >= RULES['minimum_route_extra_decisions']
        and route_extra['frames'] >= RULES['minimum_route_extra_frames'])
    flags = {'anticipatory_harm': bool(all_safe and later_failure),
             'actual_expiry_fatal_alternative': actual_expiry,
             'productive_retreat_to_food': productive}
    return flags, {'all_legal_first_actions_survive': all_safe, 'alternative_first_survives': not risks[alternative]['life_lost'],
        'alternative_later_losses': negative['life_losses'], 'alternative_level_cleared': negative['level_cleared'],
        'native_alternative_power_transition': expiry, 'positive_first_food_offset': first_food,
        'route_extra': route_extra, 'same_continuation_controller': True,
        'not_measured': ['Current-v3 disagreement', 'global action optimality']}


def _verification_job(job):
    started = time.monotonic(); directory = Path(job['directory']); config = job['config']
    folder = directory / 'cases' / job['id']; candidate_path = folder / 'candidate.json'; candidate = _load(candidate_path)
    binding = {'candidate_sha256': _hash(candidate_path), 'generator_sha256': config['generator_sha256'],
               'source_sha256': config['source_sha256'], 'qualification_sha256': config['qualification_sha256'],
               'options': config['teacher_options'], 'protocol': SPEC}
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        receipt = _load(receipt_path)
        if receipt['binding'] != binding:
            raise ValueError('Completed case inputs changed')
        for name, digest in receipt['files'].items():
            if _hash(directory / name) != digest:
                raise ValueError('Completed case proof changed')
        return {**receipt, 'reused': True}
    with BenchmarkEngine() as engine:
        state = v4._replay_prefix(engine, candidate); risks = engine.request('risks')
        plan = engine.request('teacher', options=config['teacher_options']); teacher_choice = plan['choice']
        if engine.request('observe') != state:
            raise ValueError('Teacher query changed native root')
    alternative = candidate['alternative']
    receipt = {'id': candidate['id'], 'binding': binding, 'accepted': False, 'failures': [],
        'native_replayed': False, 'source_binding': candidate['source_binding'], 'teacher_choice': teacher_choice,
        'alternative': alternative, 'teacher_plan': plan, 'immediate_counterfactuals': risks,
        'cohorts': {name: False for name in ('anticipatory_harm', 'actual_expiry_fatal_alternative', 'productive_retreat_to_food')},
        'files': {candidate_path.relative_to(directory).as_posix(): _hash(candidate_path)}}
    if alternative == teacher_choice:
        receipt['failures'].append('Scouted alternative is also the exact teacher choice')
    else:
        positive_path = folder / 'positive.jsonl'
        if candidate['reuse_source_suffix']:
            if teacher_choice != candidate['teacher_choice_hint']:
                raise ValueError('Teacher root choice differs from the bound source action')
            positive = _reused_positive(directory, candidate, positive_path)
            receipt['positive_mode'] = 'exact_independently_verified_source_suffix'
        else:
            positive = _run_branch(candidate, config['teacher_options'], positive_path, teacher_choice)
            receipt['positive_mode'] = 'fresh_teacher_recovery'
        _replay_branch(candidate, positive_path, positive)
        receipt['positive_metrics'] = positive
        receipt['files'][positive_path.relative_to(directory).as_posix()] = _hash(positive_path)
        receipt['failures'] += episode_quality_failures(positive)
        if positive['life_losses'] != 0:
            receipt['failures'].append('Teacher-controlled positive continuation loses a life')
        # Retain positive rejections without spending another complete teacher
        # continuation on an unusable label; no negative proof is invented.
        if not receipt['failures']:
            negative_path = folder / 'negative.jsonl'
            negative = _run_branch(candidate, config['teacher_options'], negative_path, alternative)
            _replay_branch(candidate, negative_path, negative)
            receipt['negative_metrics'] = negative
            receipt['files'][negative_path.relative_to(directory).as_posix()] = _hash(negative_path)
            flags, evidence = classify_behavior(state, risks, teacher_choice, alternative, positive, negative,
                                                _rows(positive_path), _rows(negative_path))
            receipt['cohorts'] = flags; receipt['behavior_evidence'] = evidence
            receipt['accepted'] = any(flags.values())
            if not receipt['accepted']:
                receipt['failures'].append('No measured anticipatory failure, native expiry fatality or productive retreat cost')
        receipt['native_replayed'] = True
    receipt['elapsed_seconds'] = time.monotonic() - started
    if source_hashes() != config['source_sha256'] or _hash(Path(__file__)) != config['generator_sha256']:
        raise ValueError('Frozen native/generator source changed while proving a case')
    _save(receipt_path, receipt, immutable=True)
    return receipt


def _report(directory):
    receipts = [_load(path) for path in sorted((Path(directory) / 'cases').glob('*/receipt.json'))]
    accepted = [r for r in receipts if r['accepted']]
    counts = Counter(name for r in accepted for name, value in r['cohorts'].items() if value)
    split_counts = Counter((_load(Path(directory) / 'cases' / r['id'] / 'candidate.json')['split'], name)
                          for r in accepted for name, value in r['cohorts'].items() if value)
    return {'dataset': PREFIX, 'completed_cases': len(receipts), 'accepted_cases': len(accepted),
            'rejected_cases': len(receipts) - len(accepted), 'behavior_counts': dict(counts),
            'by_split': {split: {name: n for (s, name), n in split_counts.items() if s == split}
                         for split in ('train', 'development')},
            'worker_seconds': sum(r.get('elapsed_seconds', 0) for r in receipts),
            'failure_counts': dict(Counter(reason for r in receipts for reason in r['failures'])),
            'learner_disagreement_measured': False}


def generate_scenarios(output_directory, qualification_path, workers=8, max_cases=24, wave_id=None):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('Use 1..8 independent CPU native workers')
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1:
        raise ValueError('Use a positive bounded case budget')
    directory = Path(output_directory); config = _assert_configuration(directory, qualification_path)
    waves = sorted((directory / 'scout-waves').glob('*.json'))
    if wave_id is not None:
        waves = [directory / 'scout-waves' / f'{wave_id}.json']
    entries = []; seen = set()
    for path in waves:
        for row in _load(path)['candidates']:
            if row['id'] not in seen:
                seen.add(row['id']); entries.append(row)
    # Completed work never consumes the next invocation's new proof budget.
    jobs = [{'id': row['id'], 'directory': str(directory), 'config': config} for row in entries
            if not (directory / 'cases' / row['id'] / 'receipt.json').exists()][:max_cases]
    context = multiprocessing.get_context('spawn'); cancel = context.Event(); started = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_setup, initargs=(cancel,))
    pending = {}; queued = iter(jobs); completed = 0
    def submit():
        job = next(queued, None)
        if job is not None:
            pending[pool.submit(_verification_job, job)] = job
    try:
        for _ in range(min(workers, len(jobs))):
            submit()
        while pending:
            ready, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not ready:
                print(f'[behavior] {completed}/{len(jobs)} new proofs; {time.monotonic()-started:.0f}s elapsed; {len(pending)} CPU workers', flush=True)
            for future in ready:
                job = pending.pop(future); receipt = future.result(); completed += 1
                _save(directory / 'generation-report.json', _report(directory)); submit()
                print(f'[behavior] {completed}/{len(jobs)} {job["id"]}: accepted={receipt["accepted"]}, flags={receipt["cohorts"]}, {receipt["elapsed_seconds"]:.1f}s', flush=True)
    except BaseException:
        cancel.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        report = {**_report(directory), 'invocation_elapsed_seconds': time.monotonic() - started,
                  'invocation_completed': completed, 'workers': workers}
        _save(directory / 'generation-report.json', report)
    return report


def _validated_receipts(directory):
    directory = Path(directory); config = _assert_configuration(directory); result = []
    for path in sorted((directory / 'cases').glob('*/receipt.json')):
        receipt = _load(path); candidate_path = path.with_name('candidate.json'); candidate = _load(candidate_path)
        expected_binding = {'candidate_sha256': _hash(candidate_path), 'generator_sha256': config['generator_sha256'],
             'source_sha256': config['source_sha256'], 'qualification_sha256': config['qualification_sha256'],
             'options': config['teacher_options'], 'protocol': SPEC}
        if (receipt['id'] != candidate['id'] or receipt['binding'] != expected_binding
                or receipt['source_binding'] != candidate['source_binding']
                or fingerprint(candidate['request']['state']) != candidate['state_sha256']
                or candidate['request'] != body(candidate['request']['state'])
                or receipt['alternative'] != candidate['alternative']):
            raise ValueError('Behavior candidate/receipt binding differs')
        for name, digest in receipt['files'].items():
            target = (directory / name).resolve()
            if not target.is_relative_to(directory.resolve()) or _hash(target) != digest:
                raise ValueError('Behavior proof file escapes or differs from receipt')
        source = directory / candidate['source_binding']['receipt']
        if _hash(source) != candidate['source_binding']['receipt_sha256']:
            raise ValueError('Copied full source replay receipt changed')
        source_receipt = _load(source)
        offline._receipt_files(directory / 'source-evidence', source_receipt)
        offline._validate_source_binding(source_receipt, _load(source.with_name('source-config.json')),
                                        _load(source.with_name('source-attempt.json')))
        if not source_receipt['accepted'] or source_receipt['metrics']['life_losses']:
            raise ValueError('Case source lacks a qualified zero-loss suffix')
        source_job = source_receipt['binding']['job']
        if any(candidate[key] != source_job[key] for key in ('split', 'seed', 'level')):
            raise ValueError('Case seed/family does not match its declared source partition')
        if receipt['accepted']:
            positive = _rows(path.with_name('positive.jsonl')); negative = _rows(path.with_name('negative.jsonl'))
            pm, nm = receipt['positive_metrics'], receipt['negative_metrics']
            if (receipt['failures'] or not receipt['native_replayed'] or pm['life_losses']
                    or episode_quality_failures(pm) or positive[0]['choice'] != receipt['teacher_choice']
                    or negative[0]['choice'] != receipt['alternative']):
                raise ValueError('Accepted positive/negative behavior proof is not qualified')
            if (positive[0]['request'] != offline._request(candidate['request'])
                    or negative[0]['request'] != offline._request(candidate['request'])
                    or set(receipt['immediate_counterfactuals']) != set(candidate['request']['state']['legal_moves'])
                    or fingerprint(summarize([r['diagnostics'] for r in positive], pm['outcome'])) != fingerprint(pm)
                    or fingerprint(summarize([r['diagnostics'] for r in negative], nm['outcome'])) != fingerprint(nm)):
                raise ValueError('Stored behavior state/transition counts differ from complete proof')
            flags, evidence = classify_behavior(candidate['request']['state'], receipt['immediate_counterfactuals'],
                receipt['teacher_choice'], receipt['alternative'], pm, nm, positive, negative)
            if flags != receipt['cohorts'] or evidence != receipt['behavior_evidence'] or not any(flags.values()):
                raise ValueError('Measured behavior flags differ from exact proof')
        result.append((candidate, receipt, path))
    return result


def _record(candidate, receipt, path, directory):
    record = offline._request(candidate['request']); record['questions']['move'].update(label=receipt['teacher_choice'], src=PREFIX)
    state = record['state']; primary = ('actual_expiry_evasion' if receipt['cohorts']['actual_expiry_fatal_alternative'] else
            'anticipatory_escape' if receipt['cohorts']['anticipatory_harm'] else 'retreat_to_food')
    record['_meta'] = {'id': f'{PREFIX}:{candidate["id"]}', 'split': candidate['split'], 'seed': candidate['seed'],
        'level': candidate['level'], 'group_id': candidate['family'], 'class': 'targeted',
        'origin': 'verified_behavior_recovery', 'state_sha256': fingerprint(state),
        'request_sha256': offline.request_fingerprint(record), 'cohorts': receipt['cohorts'],
        'primary_behavior': primary, 'source_receipt': path.relative_to(directory).as_posix(),
        'source_receipt_sha256': _hash(path), 'candidate_sha256': receipt['binding']['candidate_sha256'],
        'source_binding': candidate['source_binding'], 'source_index': candidate['source']['root_index'],
        'life_index': state.get('life_epoch', 0), 'gold': receipt['teacher_choice'],
        'alternative': receipt['alternative'], 'behavior_evidence': receipt['behavior_evidence'],
        'learner_disagreement_measured': False, 'anticipatory_harm_measured': receipt['cohorts']['anticipatory_harm']}
    return record


def scenario_records(directory):
    directory = Path(directory); rows = [_record(c, r, p, directory) for c, r, p in _validated_receipts(directory) if r['accepted']]
    # One exact request counts once. Conflicting labels are quarantined instead
    # of being resolved by arbitrary source priority or counted twice.
    grouped = {}
    for row in rows:
        grouped.setdefault(row['_meta']['request_sha256'], []).append(row)
    result = []
    for candidates in grouped.values():
        if len({r['_meta']['split'] for r in candidates}) != 1 or len({r['questions']['move']['label'] for r in candidates}) != 1:
            continue
        result.append(sorted(candidates, key=lambda r: r['_meta']['id'])[0])
    return sorted(result, key=lambda r: r['_meta']['id'])


def scenario_windows(directory, root_id):
    directory = Path(directory); identifier = root_id.removeprefix(PREFIX + ':'); folder = directory / 'cases' / identifier
    candidate = _load(folder / 'candidate.json'); receipt = _load(folder / 'receipt.json')
    if not receipt['accepted']:
        return []
    # Root's receipt remains the proof for all bounded same-life POST frames.
    root = _record(candidate, receipt, folder / 'receipt.json', directory); positive = _rows(folder / 'positive.jsonl')
    chosen = []; root_life = positive[0]['diagnostics']['life_index']
    for offset in RULES['window_offsets']:
        if offset >= len(positive):
            continue
        row = positive[offset]
        if row['diagnostics']['life_index'] != root_life:
            continue
        progress = positive[0]['request']['state']['pellets_remaining'] - row['request']['state']['pellets_remaining']
        if progress <= 0 or offset < 2:
            continue
        record = copy.deepcopy(row['request']); record['questions']['move'].update(label=row['choice'], src=PREFIX)
        record['_meta'] = {**root['_meta'], 'id': root['_meta']['id'] + f':post-{offset}', 'class': 'informative',
            'parent_targeted_id': root['_meta']['id'], 'window_kind': 'post_followthrough_with_progress',
            'decision_offset': offset, 'pellets_since_root': progress,
            'state_sha256': row['state_sha256'], 'request_sha256': offline.request_fingerprint(record),
            'gold': row['choice'], 'source_index': None, 'cohorts': receipt['cohorts']}
        chosen.append(record)
        if len(chosen) == RULES['max_windows']:
            break
    return chosen


def validate_scenarios(directory, qualification_path, replay=False):
    _assert_configuration(directory, qualification_path)
    rows = _validated_receipts(directory)
    if replay:
        for candidate, receipt, path in rows:
            if receipt.get('positive_metrics'):
                _replay_branch(candidate, path.with_name('positive.jsonl'), receipt['positive_metrics'])
            if receipt.get('negative_metrics'):
                _replay_branch(candidate, path.with_name('negative.jsonl'), receipt['negative_metrics'])
    return {**_report(directory), 'native_replay_repeated': replay,
            'unique_admitted_requests': len(scenario_records(directory))}


def completed_files(directory):
    directory = Path(directory); names = []
    for name in ('behavior-config.json', 'qualification.json', 'scout-report.json', 'generation-report.json', 'source-evidence/offline-config.json'):
        if (directory / name).is_file():
            names.append(name)
    config = _load(directory / 'behavior-config.json') if (directory / 'behavior-config.json').exists() else None
    if config:
        qualification = _load(directory / 'qualification.json')
        if (directory / qualification['replay_bundle']).exists():
            names.append(qualification['replay_bundle'])
    names += [p.relative_to(directory).as_posix() for p in (directory / 'scout-waves').glob('*.json')]
    for path in (directory / 'cases').glob('*/candidate.json'):
        names.append(path.relative_to(directory).as_posix())
    for path in (directory / 'source-evidence/episodes').glob('*/receipt.json'):
        receipt = _load(path)
        names += ['source-evidence/' + name for name in receipt['files']]
        names.append(path.relative_to(directory).as_posix())
    for path in (directory / 'cases').glob('*/receipt.json'):
        receipt = _load(path); names.extend(receipt['files']); names.append(path.relative_to(directory).as_posix())
    return sorted(set(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='stage', required=True)
    scout = sub.add_parser('scout'); scout.add_argument('--collection', type=Path, required=True)
    scout.add_argument('--evidence', type=Path, required=True); scout.add_argument('--output', type=Path, required=True)
    scout.add_argument('--qualification', type=Path, required=True); scout.add_argument('--max-roots', type=int, default=256)
    scout.add_argument('--wave-id', default='pilot-001')
    generate = sub.add_parser('generate'); generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--qualification', type=Path, required=True); generate.add_argument('--workers', type=int, default=8)
    generate.add_argument('--max-cases', type=int, default=24); generate.add_argument('--wave-id')
    validate = sub.add_parser('validate'); validate.add_argument('--output', type=Path, required=True)
    validate.add_argument('--qualification', type=Path, required=True); validate.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.stage == 'scout':
        result = scout_scenarios(args.collection, args.evidence, args.output, args.qualification, args.max_roots, args.wave_id)
    elif args.stage == 'generate':
        result = generate_scenarios(args.output, args.qualification, args.workers, args.max_cases, args.wave_id)
    else:
        result = validate_scenarios(args.output, args.qualification, args.replay)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
