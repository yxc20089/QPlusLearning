"""Native recovery proofs for saved, non-evaluation current-v3 decisions.

No live Kev query is made. A measured disagreement is bound to an original
response, checkpoint, full trace and exact native root. Teacher follow-through
windows inherit ancestry, never an invented prediction at their new state.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import json
import multiprocessing
from pathlib import Path
import time

import behavior_scenarios as behavior
from pacman_lab import ROOT, body
import teacher_offline_data as offline
from teacher_validation import episode_quality_failures, fingerprint
from gameplay_benchmark import summarize
import v4_data as v4

PREFIX = 'pacman-native-v4-recorded-recovery'


def prepare_recorded(recorded_source, evidence_directory, output_directory, template_directory,
                     qualification_path, max_routing=8):
    source, evidence, directory, template = map(Path, (recorded_source, evidence_directory, output_directory, template_directory))
    config = behavior._assert_configuration(template, qualification_path); collection = behavior._load(source / 'collection.json')
    if (collection['source_sha256'] != config['source_sha256']
            or collection['generator_sha256'] != config['v4_generator_sha256']
            or collection['protocol'] != config['protocol']
            or collection['parallel_learner_helper_sha256'] != behavior._hash(ROOT / 'v4_parallel_collection.py')):
        raise ValueError('Saved learner collection is not the pinned native current-v3 collector')
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('behavior-config.json', 'qualification.json', 'source-evidence/offline-config.json'):
        behavior._copy(template / name, directory / name)
    qualification = behavior._load(template / 'qualification.json')
    behavior._copy(template / qualification['replay_bundle'], directory / qualification['replay_bundle'])
    behavior._copy(source / 'collection.json', directory / 'recorded-source/collection.json')
    fixed = {'dataset': PREFIX, 'wrapper_sha256': behavior._hash(Path(__file__)),
             'behavior_generator_sha256': config['generator_sha256'], 'recorded_collection_sha256': behavior._hash(source / 'collection.json'),
             'source_sha256': config['source_sha256'], 'qualification_sha256': config['qualification_sha256'],
             'maximum_routing_roots': max_routing, 'live_model_queries': 0,
             'interpretation': 'Saved learner root choices only; full teacher recovery and same-controller negative native continuations.'}
    behavior._save(directory / 'recorded-config.json', fixed, immutable=True)
    proposed = []; identities = set(); source_snapshots = {}
    for attempt_path in sorted((source / 'episodes').glob('*/attempt.json')):
        attempt = behavior._load(attempt_path); v4.require_v3_identity(attempt['checkpoint'])
        identifier = attempt['id']; split = attempt['split']; seed, level = attempt['seed'], attempt['level']
        declared = any(key.startswith('learner/') and seed in wave.get(split, [])
                       for key, wave in collection['episode_sets'].items())
        if not declared or split != 'train' or len(attempt['files']) != 1:
            raise ValueError('Recorded source episode is outside its declared training seed partition')
        relative, digest = next(iter(attempt['files'].items())); trace = source / relative
        if behavior._hash(trace) != digest or relative != f'episodes/{identifier}/level-{level}-seed-{seed}.jsonl':
            raise ValueError('Recorded full learner trace differs from original immutable attempt')
        rows = behavior._rows(trace); identities.add(attempt['checkpoint']['checkpoint_sha256'])
        teacher_sources = []
        for path in (evidence / 'episodes').glob(f'*-{split}-level-{level}-seed-{seed}-*/receipt.json'):
            receipt = behavior._load(path)
            if receipt['accepted'] and receipt['metrics']['life_losses'] == 0:
                teacher_sources.append(path)
        if not teacher_sources:
            continue
        teacher_path = sorted(teacher_sources)[0]; teacher_id = teacher_path.parent.name
        behavior._bind_source(evidence, directory, teacher_id)
        behavior._copy(attempt_path, directory / f'recorded-source/episodes/{identifier}/attempt.json')
        behavior._copy(trace, directory / 'recorded-source' / relative)
        source_snapshots[identifier] = {'attempt': f'recorded-source/episodes/{identifier}/attempt.json',
            'attempt_sha256': behavior._hash(attempt_path), 'trace': 'recorded-source/' + relative, 'trace_sha256': digest}
        # Cache the source's exact prefix once, not anew per selected state.
        hashes = [fingerprint(row['request']['state']) for row in rows]
        actions = [row['diagnostics']['choice'] for row in rows]
        for index, row in enumerate(rows):
            if row['request'] != body(row['request']['state']) or row.get('active_checkpoint') != attempt['checkpoint']:
                raise ValueError('Recorded request/checkpoint identity changed inside saved trajectory')
            state = row['request']['state']; answer = row['response']['answers']['move']; choice = answer['choice']
            risks = row['diagnostics']['immediate_counterfactuals']
            if choice not in state['legal_moves'] or choice != actions[index] or set(answer['probabilities']) != set(state['legal_moves']):
                raise ValueError('Recorded response/action/options differ from the native trajectory')
            immediate = risks[choice]['life_lost'] and any(not risk['life_lost'] for risk in risks.values())
            routing = (not risks[choice]['life_lost'] and risks[choice]['pellets'] == 0
                       and any(not risk['life_lost'] and risk['pellets'] > 0 for risk in risks.values())
                       and (len(state['legal_moves']) >= 3 or state.get('decisions_since_last_pellet', 0) >= 24))
            if not immediate and not routing:
                continue
            evidence_record = {**source_snapshots[identifier], 'episode': identifier, 'root_index': index,
                'checkpoint': attempt['checkpoint'], 'response': row['response'], 'response_sha256': fingerprint(row['response']),
                'request_sha256': offline.request_fingerprint(row['request']), 'state_sha256': hashes[index],
                'choice': choice, 'protocol_sha256': fingerprint(collection['protocol']),
                'collection_sha256': fixed['recorded_collection_sha256'], 'original_native_diagnostics': row['diagnostics']}
            candidate = {'split': split, 'seed': seed, 'level': level, 'family': f'level-{level}-seed-{seed}',
                'origin': 'recorded_v3_recovery', 'source': {'kind': 'recorded_v3_episode', 'episode': teacher_id,
                    'root_index': index, 'recorded_episode': identifier, 'recorded_trace_sha256': digest},
                'request': body(state), 'state_sha256': hashes[index], 'prefix_actions': actions[:index],
                'prefix_state_sha256': hashes[:index + 1], 'teacher_choice_hint': None, 'alternative': choice,
                'reuse_source_suffix': False,
                'source_binding': {'episode': teacher_id, 'receipt': f'source-evidence/episodes/{teacher_id}/receipt.json',
                    'receipt_sha256': behavior._hash(directory / f'source-evidence/episodes/{teacher_id}/receipt.json'),
                    'original_candidate_sha256': None, 'source_index': index},
                'scout_flags': {'anticipatory': not immediate, 'expiry': state['frightened'], 'dry_recovery': routing},
                'scout_score': [], 'recorded_learner_evidence': evidence_record}
            candidate['id'] = fingerprint(candidate)[:24]
            proposed.append((not immediate, -state.get('decisions_since_last_pellet', 0), candidate))
    if len(identities) > 1:
        raise ValueError('Recorded source combines different learner checkpoints')
    selected = []; routing_used = 0; seen = set()
    for noncritical, _, candidate in sorted(proposed, key=lambda p: (p[0], p[1], p[2]['id'])):
        key = offline.request_fingerprint(candidate['request'])
        if key in seen or noncritical and routing_used >= max_routing:
            continue
        seen.add(key); routing_used += noncritical; selected.append(candidate)
        behavior._save(directory / 'cases' / candidate['id'] / 'candidate.json', candidate, immutable=True)
    index = {'dataset': PREFIX, 'wrapper_sha256': fixed['wrapper_sha256'], 'source_snapshots': source_snapshots,
             'candidates': [{'id': c['id'], 'candidate_sha256': behavior._hash(directory / 'cases' / c['id'] / 'candidate.json')}
                            for c in selected], 'critical_proposals': sum(not p[0] for p in proposed),
             'selected_critical': len(selected) - routing_used, 'selected_routing': routing_used,
             'checkpoint_sha256': next(iter(identities), None)}
    behavior._save(directory / 'recorded-index.json', index, immutable=True)
    return index


def _assert(directory, qualification_path=None):
    directory = Path(directory); config = behavior._assert_configuration(directory, qualification_path)
    fixed = behavior._load(directory / 'recorded-config.json')
    if (fixed['dataset'] != PREFIX or fixed['wrapper_sha256'] != behavior._hash(Path(__file__))
            or fixed['behavior_generator_sha256'] != config['generator_sha256']
            or fixed['recorded_collection_sha256'] != behavior._hash(directory / 'recorded-source/collection.json')):
        raise ValueError('Recorded recovery helper/source/configuration changed')
    return config, fixed


def _validate_recorded(candidate, directory):
    evidence = candidate['recorded_learner_evidence']; attempt = behavior._load(Path(directory) / evidence['attempt'])
    if (behavior._hash(Path(directory) / evidence['attempt']) != evidence['attempt_sha256']
            or behavior._hash(Path(directory) / evidence['trace']) != evidence['trace_sha256']):
        raise ValueError('Original saved learner episode changed')
    rows = behavior._rows(Path(directory) / evidence['trace']); row = rows[evidence['root_index']]
    if (row['request'] != candidate['request'] or row['response'] != evidence['response']
            or row['active_checkpoint'] != evidence['checkpoint'] or attempt['checkpoint'] != evidence['checkpoint']
            or fingerprint(row['response']) != evidence['response_sha256']
            or offline.request_fingerprint(row['request']) != evidence['request_sha256']
            or evidence['state_sha256'] != candidate['state_sha256']
            or evidence['choice'] != candidate['alternative']
            or evidence['original_native_diagnostics'] != row['diagnostics']
            or candidate['prefix_actions'] != [r['diagnostics']['choice'] for r in rows[:evidence['root_index']]]
            or candidate['prefix_state_sha256'] != [fingerprint(r['request']['state']) for r in rows[:evidence['root_index']+1]]):
        raise ValueError('Measured learner root/response/prefix is not its saved native trajectory')
    v4.require_v3_identity(evidence['checkpoint'])
    return evidence


def _wrapped_job(job):
    directory = Path(job['directory']); folder = directory / 'cases' / job['id']; candidate = behavior._load(folder / 'candidate.json')
    recorded = _validate_recorded(candidate, directory)
    inner = behavior._verification_job(job)
    wrapper_path = folder / 'recorded-receipt.json'
    binding = {'wrapper_sha256': job['wrapper_sha256'], 'behavior_receipt_sha256': behavior._hash(folder / 'receipt.json'),
               'candidate_sha256': behavior._hash(folder / 'candidate.json'), 'recorded_evidence_sha256': fingerprint(recorded)}
    if wrapper_path.exists():
        saved = behavior._load(wrapper_path)
        if saved['binding'] != binding:
            raise ValueError('Completed recorded recovery proof changed')
        return saved
    accepted = False; cohorts = {**inner['cohorts'], 'immediate_evasion': False, 'productive_junction': False, 'sparse_cleanup': False}
    failures = []
    if not inner.get('positive_metrics') or not inner.get('negative_metrics'):
        failures.append('No complete paired recovery proof (same teacher choice or unqualified positive)')
    else:
        positive = inner['positive_metrics']; risks = inner['immediate_counterfactuals']; gold = inner['teacher_choice']; alt = candidate['alternative']
        original_risks = recorded['original_native_diagnostics']['immediate_counterfactuals']
        if risks != original_risks:
            raise ValueError('Saved native root risks differ from independently replayed recovery root')
        qualified = inner['native_replayed'] and positive['life_losses'] == 0 and not episode_quality_failures(positive)
        cohorts['immediate_evasion'] = bool(qualified and risks[alt]['life_lost'] and not risks[gold]['life_lost'])
        productive = bool(qualified and not risks[alt]['life_lost'] and not risks[gold]['life_lost']
                          and risks[gold]['pellets'] > risks[alt]['pellets'])
        cohorts['sparse_cleanup'] = productive and candidate['request']['state']['pellets_remaining'] <= 30
        cohorts['productive_junction'] = productive and not cohorts['sparse_cleanup']
        accepted = bool(qualified and gold != alt and any(cohorts.values()))
        if not accepted:
            failures.append('Saved choice is not a proven admitted recovery/local-progress root')
    receipt = {'id': job['id'], 'binding': binding, 'accepted': accepted, 'failures': failures,
               'cohorts': cohorts, 'teacher_choice': inner['teacher_choice'], 'recorded_learner_evidence': recorded,
               'learner_disagreement_measured': accepted,
               'files': {**inner['files'], (folder / 'receipt.json').relative_to(directory).as_posix(): binding['behavior_receipt_sha256']},
               'interpretation': 'Current-v3 saved root disagrees with qualified teacher. Immediate fatality or native local pellet advantage is measured; safe local-progress alternatives are not claimed globally harmful.'}
    behavior._save(wrapper_path, receipt, immutable=True)
    return receipt


def generate_recorded(directory, qualification_path, workers=8):
    if not 1 <= workers <= 8:
        raise ValueError('Use1..8independentCPUworkers')
    directory = Path(directory); config, fixed = _assert(directory, qualification_path)
    entries = behavior._load(directory / 'recorded-index.json')['candidates']
    jobs = [{'id': row['id'], 'directory': str(directory), 'config': config, 'wrapper_sha256': fixed['wrapper_sha256']} for row in entries]
    context = multiprocessing.get_context('spawn'); cancel = context.Event()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=behavior._worker_setup, initargs=(cancel,))
    started = time.monotonic(); pending = {}; queued = iter(jobs); results = []
    def submit():
        task = next(queued, None)
        if task is not None:
            pending[pool.submit(_wrapped_job, task)] = task
    try:
        for _ in range(min(workers, len(jobs))):
            submit()
        while pending:
            ready, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not ready:
                print(f'[recorded/recovery] {len(results)}/{len(jobs)} proofs; {time.monotonic()-started:.0f}s', flush=True)
            for future in ready:
                pending.pop(future); receipt = future.result(); results.append(receipt); submit()
                print(f'[recorded/recovery] {len(results)}/{len(jobs)} admitted={receipt["accepted"]}, flags={receipt["cohorts"]}', flush=True)
    except BaseException:
        cancel.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    report = {'dataset': PREFIX, 'completed': len(results), 'accepted': sum(r['accepted'] for r in results),
              'elapsed_seconds': time.monotonic()-started, 'live_model_queries': 0}
    behavior._save(directory / 'recorded-report.json', report)
    return report


def _receipts(directory):
    directory = Path(directory); _, fixed = _assert(directory)
    originals = {c['id']: (c, r, p) for c, r, p in behavior._validated_receipts(directory)}
    result = []
    for path in sorted((directory / 'cases').glob('*/recorded-receipt.json')):
        receipt = behavior._load(path); candidate, inner, _ = originals[receipt['id']]; recorded = _validate_recorded(candidate, directory)
        binding = {'wrapper_sha256': fixed['wrapper_sha256'], 'behavior_receipt_sha256': behavior._hash(path.with_name('receipt.json')),
                   'candidate_sha256': behavior._hash(path.with_name('candidate.json')), 'recorded_evidence_sha256': fingerprint(recorded)}
        if receipt['binding'] != binding or receipt['recorded_learner_evidence'] != recorded:
            raise ValueError('Recorded wrapper receipt binding changed')
        for name, digest in receipt['files'].items():
            if behavior._hash(directory / name) != digest:
                raise ValueError('Complete recorded recovery evidence changed')
        if receipt['accepted']:
            pm = inner['positive_metrics']; risks = inner['immediate_counterfactuals']; gold = inner['teacher_choice']; alt = candidate['alternative']
            if (pm['life_losses'] or episode_quality_failures(pm) or not inner['native_replayed'] or gold == alt
                    or not receipt['learner_disagreement_measured'] or receipt['failures']
                    or risks != recorded['original_native_diagnostics']['immediate_counterfactuals']):
                raise ValueError('Recorded recovery label lacks complete qualifying proof')
            positive = behavior._rows(path.with_name('positive.jsonl')); negative = behavior._rows(path.with_name('negative.jsonl'))
            nm = inner['negative_metrics']
            if (positive[0]['request'] != offline._request(candidate['request'])
                    or negative[0]['request'] != offline._request(candidate['request'])
                    or positive[0]['choice'] != gold or negative[0]['choice'] != alt
                    or fingerprint(summarize([r['diagnostics'] for r in positive], pm['outcome'])) != fingerprint(pm)
                    or fingerprint(summarize([r['diagnostics'] for r in negative], nm['outcome'])) != fingerprint(nm)):
                raise ValueError('Stored recorded full-game counts differ from their transition proofs')
            if receipt['cohorts']['immediate_evasion'] != bool(risks[alt]['life_lost'] and not risks[gold]['life_lost']):
                raise ValueError('Recorded immediate evasion predicate differs')
            productive = bool(not risks[alt]['life_lost'] and not risks[gold]['life_lost'] and risks[gold]['pellets'] > risks[alt]['pellets'])
            sparse = productive and candidate['request']['state']['pellets_remaining'] <= 30
            if receipt['cohorts']['sparse_cleanup'] != sparse or receipt['cohorts']['productive_junction'] != (productive and not sparse):
                raise ValueError('Recorded native local-progress predicate differs')
        result.append((candidate, inner, receipt, path))
    return result


def scenario_records(directory):
    directory = Path(directory); records = []
    for candidate, inner, receipt, path in _receipts(directory):
        if not receipt['accepted']:
            continue
        record = behavior._record(candidate, inner, path, directory)
        record['questions']['move']['src'] = PREFIX
        primary = ('immediate_evasion' if receipt['cohorts']['immediate_evasion'] else
                   'actual_expiry_evasion' if receipt['cohorts']['actual_expiry_fatal_alternative'] else
                   'anticipatory_escape' if receipt['cohorts']['anticipatory_harm'] else
                   'retreat_to_food' if receipt['cohorts']['productive_retreat_to_food'] else
                   'sparse_cleanup' if receipt['cohorts']['sparse_cleanup'] else 'productive_junction')
        record['_meta'].update(id=f'{PREFIX}:{candidate["id"]}', primary_behavior=primary, cohorts=receipt['cohorts'],
            recorded_learner_evidence=receipt['recorded_learner_evidence'], learner_disagreement_measured=True)
        records.append(record)
    return records


def scenario_windows(directory, root_id):
    directory = Path(directory); identifier = root_id.removeprefix(PREFIX + ':'); root = next(
        (row for row in scenario_records(directory) if row['_meta']['id'] == root_id), None)
    if root is None:
        return []
    positive = behavior._rows(directory / 'cases' / identifier / 'positive.jsonl'); result = []
    for offset in behavior.RULES['window_offsets']:
        if offset < 2 or offset >= len(positive):
            continue
        row = positive[offset]; progress = positive[0]['request']['state']['pellets_remaining'] - row['request']['state']['pellets_remaining']
        if progress <= 0 or row['diagnostics']['life_index'] != positive[0]['diagnostics']['life_index']:
            continue
        record = copy.deepcopy(row['request']); record['questions']['move'].update(label=row['choice'], src=PREFIX)
        meta = {**root['_meta'], 'id': root_id + f':post-{offset}', 'class': 'informative', 'parent_targeted_id': root_id,
            'window_kind': 'post_followthrough_with_progress', 'decision_offset': offset, 'pellets_since_root': progress,
            'state_sha256': row['state_sha256'], 'request_sha256': offline.request_fingerprint(record), 'gold': row['choice'],
            'learner_disagreement_measured': False, 'parent_recorded_learner_evidence': root['_meta']['recorded_learner_evidence']}
        meta.pop('recorded_learner_evidence'); record['_meta'] = meta; result.append(record)
        if len(result) == behavior.RULES['max_windows']:
            break
    return result


def validate_scenarios(directory, qualification_path, replay=False):
    _assert(directory, qualification_path); rows = _receipts(directory)
    if replay:
        for candidate, inner, _, path in rows:
            if inner.get('positive_metrics'):
                behavior._replay_branch(candidate, path.with_name('positive.jsonl'), inner['positive_metrics'])
            if inner.get('negative_metrics'):
                behavior._replay_branch(candidate, path.with_name('negative.jsonl'), inner['negative_metrics'])
    return {'dataset': PREFIX, 'completed': len(rows), 'accepted': sum(r['accepted'] for _, _, r, _ in rows),
            'live_model_queries': 0, 'native_replay_repeated': replay}


def completed_files(directory):
    directory = Path(directory); names = behavior.completed_files(directory)
    names += [p.relative_to(directory).as_posix() for p in directory.glob('recorded-*.json')]
    names += [p.relative_to(directory).as_posix() for p in (directory / 'recorded-source').rglob('*.json')]
    names += [p.relative_to(directory).as_posix() for p in (directory / 'recorded-source').rglob('*.jsonl')]
    names += [p.relative_to(directory).as_posix() for p in (directory / 'cases').glob('*/recorded-receipt.json')]
    return sorted(set(names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='stage', required=True)
    prepare = sub.add_parser('prepare'); prepare.add_argument('--source', type=Path, required=True)
    prepare.add_argument('--evidence', type=Path, required=True); prepare.add_argument('--template', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True); prepare.add_argument('--qualification', type=Path, required=True)
    prepare.add_argument('--max-routing', type=int, default=8)
    generate = sub.add_parser('generate'); generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--qualification', type=Path, required=True); generate.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare_recorded(args.source, args.evidence, args.output, args.template, args.qualification, args.max_routing)
        print(json.dumps({k: v for k, v in result.items() if k not in ('candidates', 'source_snapshots')}, indent=2))
    else:
        print(json.dumps(generate_recorded(args.output, args.qualification, args.workers), indent=2))
