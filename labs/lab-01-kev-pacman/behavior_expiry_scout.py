"""Find real power-expiry collision roots using only legal native perturbations.

This is scouting, not label admission. Each proposed root must subsequently
pass behavior_scenarios' full zero-loss teacher recovery and exact negative
continuation proofs. Original seed partitions and frozen game/teacher sources
are preserved. No learner or GPU is queried.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import json
import multiprocessing
from pathlib import Path
import time

import behavior_scenarios as behavior
from gameplay_benchmark import BenchmarkEngine
from pacman_lab import body, destination, distances, position
import teacher_offline_data as offline
from teacher_validation import fingerprint
import v4_data as v4


def _job(task):
    started = time.monotonic(); base = task['candidate']; roots = []; explored = []
    # Two coherent legal approach paths target the nearest frightened ghosts.
    state = base['request']['state']; routes = distances(state['maze'], position(state['player']))
    targets = sorted((g for g in state['ghosts'] if g['mode'] == 'outside' and g['frightened']),
                     key=lambda g: (routes.get(position(g), 999), g['name']))[:2]
    for target in targets:
        candidate = copy.deepcopy(base)
        with BenchmarkEngine() as engine:
            state = v4._replay_prefix(engine, candidate)
            for depth in range(task['steps'] + 1):
                behavior._check(); risks = engine.request('risks')
                record = {'depth': depth, 'target': target['name'], 'state_sha256': fingerprint(state),
                          'remaining_frames': state['timing']['power']['remaining_frames'],
                          'prefix_actions': list(candidate['prefix_actions'][len(base['prefix_actions']):]),
                          'immediate_counterfactuals': risks, 'fatal_transitions': {}}
                if state['frightened'] and behavior._near(state, True) <= 3 and behavior._near(state, False) > 3:
                    for alternative, risk in risks.items():
                        if not risk['life_lost']:
                            continue
                        with BenchmarkEngine() as branch:
                            exact = v4._replay_prefix(branch, candidate); step = branch.step(alternative)
                            transition = behavior._transition(exact, step)
                        record['fatal_transitions'][alternative] = transition
                        if transition['remaining_before'] > 0 and transition['remaining_after'] == 0:
                            root = copy.deepcopy(candidate)
                            root.update(alternative=alternative, reuse_source_suffix=False, teacher_choice_hint=None,
                                scout_flags={'expiry': True, 'anticipatory': False, 'dry_recovery': False},
                                scout_score=[], derivation={'kind': 'legal_native_expiry_approach', 'target': target['name'],
                                    'steps_from_source': depth, 'parent_state_sha256': base['state_sha256'],
                                    'helper_sha256': task['helper_sha256'], 'job_sha256': fingerprint(task)},
                                native_expiry_scout_transition=transition)
                            root['source'] = {**root['source'], 'kind': 'legal_native_expiry_fork',
                                              'perturbation': 'approach_frightened_ghost_before_expiry',
                                              'perturbation_actions': record['prefix_actions']}
                            root['id'] = fingerprint({'request': root['request'], 'prefix': root['prefix_actions'],
                                'alternative': alternative, 'source_binding': root['source_binding'],
                                'derivation': root['derivation']})[:24]
                            roots.append(root)
                explored.append(record)
                if depth == task['steps'] or not state['frightened'] or state['timing']['power']['remaining_frames'] == 0:
                    break
                ghost = next((g for g in state['ghosts'] if g['name'] == target['name'] and g['mode'] == 'outside'), None)
                if ghost is None:
                    break
                ghost_routes = distances(state['maze'], position(ghost))
                safe = [move for move in state['legal_moves'] if not risks[move]['life_lost']]
                if not safe:
                    break
                # This is a controlled wrong-direction exposure, not a teacher.
                move = min(safe, key=lambda d: (ghost_routes.get(destination(state['maze'], position(state['player']), d), 999),
                                               d != state['player']['heading'], d))
                step = engine.step(move)
                if step['outcome']:
                    break
                candidate['prefix_actions'].append(move); candidate['prefix_state_sha256'].append(fingerprint(step['state']))
                state = step['state']; candidate['request'] = body(state); candidate['state_sha256'] = fingerprint(state)
    return {'job_sha256': fingerprint(task), 'helper_sha256': task['helper_sha256'],
            'found': roots, 'explored': explored, 'elapsed_seconds': time.monotonic() - started}


def scout_expiry_forks(evidence_directory, output_directory, qualification_path,
                       max_parents=96, workers=8, steps=4, wave_id='expiry-001'):
    if not 1 <= workers <= 8 or not 1 <= max_parents <= 1000 or not 1 <= steps <= 4:
        raise ValueError('Use1..8workers,1..1000parents,1..4legal approach moves')
    directory = Path(output_directory); evidence = Path(evidence_directory)
    config = behavior._assert_configuration(directory, qualification_path); helper_hash = behavior._hash(Path(__file__))
    parents = []; seen = set()
    for path in sorted((evidence / 'episodes').glob('*/receipt.json')):
        receipt = behavior._load(path)
        if not receipt['accepted']:
            continue
        source = None
        for proof in behavior._rows(path.with_name('verified.jsonl')):
            state = proof['request']['state']; remaining = state['timing']['power']['remaining_frames']
            if not state['frightened'] or not 0 < remaining <= 40 or len(state['legal_moves']) < 2:
                continue
            if behavior._near(state, True) > 6:
                continue
            digest = offline.request_fingerprint(proof['request'])
            if digest in seen:
                continue
            seen.add(digest)
            if source is None:
                source = behavior._rows(path.with_name('source-trace.jsonl'))
            candidate = behavior._source_candidate(path.parent.name, receipt, source, proof)
            parents.append((behavior._near(state, True), remaining, candidate, path.parent.name))
    buckets = {}
    for near, remaining, candidate, identifier in sorted(parents, key=lambda p: (p[0], p[1], p[2]['id'])):
        buckets.setdefault((candidate['split'], candidate['family']), []).append((candidate, identifier))
    selected = []; keys = sorted(buckets, key=lambda k: (k[1], k[0] != 'development'))
    while len(selected) < max_parents and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < max_parents:
                selected.append(buckets[key].pop(0))
    jobs = []
    for candidate, identifier in selected:
        behavior._bind_source(evidence, directory, identifier)
        source_receipt = directory / f'source-evidence/episodes/{identifier}/receipt.json'
        candidate['source_binding'] = {'episode': identifier, 'receipt': source_receipt.relative_to(directory).as_posix(),
            'receipt_sha256': behavior._hash(source_receipt), 'original_candidate_sha256': None,
            'source_index': candidate['source']['root_index']}
        job = {'candidate': candidate, 'steps': steps, 'helper_sha256': helper_hash, 'source_sha256': config['source_sha256']}
        job['id'] = fingerprint(job)[:24]
        jobs.append(job)
    context = multiprocessing.get_context('spawn'); cancel = context.Event()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=behavior._worker_setup, initargs=(cancel,))
    started = time.monotonic(); pending = {}; completed = []; queued = iter(jobs)
    def submit():
        job = next(queued, None)
        if job is not None:
            path = directory / 'scout-waves' / f'{wave_id}-parent-{job["id"]}.json'
            if path.exists():
                receipt = behavior._load(path)
                if receipt['job_sha256'] != fingerprint(job) or receipt['helper_sha256'] != helper_hash:
                    raise ValueError('Completed expiry scout inputs/helper changed')
                for entry in receipt['candidates']:
                    if behavior._hash(directory / 'cases' / entry['id'] / 'candidate.json') != entry['candidate_sha256']:
                        raise ValueError('Committed expiry candidate changed')
                completed.append(receipt); return submit()
            pending[pool.submit(_job, job)] = job
    try:
        for _ in range(min(workers, len(jobs))):
            submit()
        while pending:
            ready, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not ready:
                print(f'[expiry/scout] {len(completed)}/{len(jobs)} parents; {time.monotonic()-started:.0f}s', flush=True)
            for future in ready:
                job = pending.pop(future); proof = future.result(); entries = []
                for candidate in proof.pop('found'):
                    path = directory / 'cases' / candidate['id'] / 'candidate.json'
                    behavior._save(path, candidate, immutable=True)
                    entries.append({'id': candidate['id'], 'split': candidate['split'], 'family': candidate['family'],
                                    'candidate_sha256': behavior._hash(path)})
                receipt = {**proof, 'wave_id': wave_id, 'job': job, 'candidates': entries,
                           'label_admission': 'Pending full zero-loss teacher positive and native negative continuation proofs'}
                behavior._save(directory / 'scout-waves' / f'{wave_id}-parent-{job["id"]}.json', receipt, immutable=True)
                completed.append(receipt); submit()
                print(f'[expiry/scout] {len(completed)}/{len(jobs)} parents; {len(entries)} real expiry-fatal roots found', flush=True)
    except BaseException:
        cancel.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    report = {'wave_id': wave_id, 'helper_sha256': helper_hash, 'plausible_distinct_parents': len(parents),
              'completed_parents': len(completed), 'found_candidates': sum(len(p['candidates']) for p in completed),
              'worker_seconds': sum(p['elapsed_seconds'] for p in completed),
              'elapsed_seconds': time.monotonic() - started, 'learner_disagreement_measured': False}
    behavior._save(directory / 'scout-waves' / f'{wave_id}-summary.json', {**report, 'candidates': []}, immutable=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True); parser.add_argument('--max-parents', type=int, default=96)
    parser.add_argument('--workers', type=int, default=8); parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--wave-id', default='expiry-001'); args = parser.parse_args()
    print(json.dumps(scout_expiry_forks(args.evidence, args.output, args.qualification,
                     args.max_parents, args.workers, args.steps, args.wave_id), indent=2))
