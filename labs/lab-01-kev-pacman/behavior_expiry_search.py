"""Bounded legal native beam search for actual expiry-fatal alternatives.

Opaque rewind handles avoid repeatedly reconstructing the source prefix during
CPU scouting. The game, physics, qualified teacher and its future-RNG policy
remain unchanged. A discovered root is not admitted until the original native
engine cold-replays its full prefix and both complete continuation proofs.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import json
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import time

import behavior_scenarios as behavior
from gameplay_benchmark import BenchmarkEngine, benchmark_script
from pacman_lab import ROOT, body, ensure_node
import teacher_offline_data as offline
from teacher_validation import fingerprint
import v4_data as v4

WORKER = ROOT / 'games/behavior-scout-worker.cjs'


class SnapshotEngine(BenchmarkEngine):
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='kev-expiry-scout-')
        script = Path(self.directory.name) / 'pacman.js'; script.write_text(benchmark_script())
        self.process = subprocess.Popen([ensure_node(), str(WORKER), str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _rank(state):
    remaining = state['timing']['power']['remaining_frames']; near = behavior._near(state, True)
    pause = state['timing']['power']['eating_pause_frames']
    # Collision is native tile equality after the clock update. At distance2
    # expiry often occurs one action before the collision, leaving only an
    # already-unpowered next root. Favor the actual0..1tile boundary instead.
    return (int(1 <= remaining <= 7), int(near <= 1), -abs(near - 1), int(pause == 0), -remaining)


def _job(job):
    started = time.monotonic(); base = job['candidate']; discovered = []; explored = []; skipped = []
    with SnapshotEngine() as engine:
        state = v4._replay_prefix(engine, base); root = engine.request('snapshot')['handle']
        beam = [(root, copy.deepcopy(base))]; seen_prefixes = {fingerprint(base['prefix_actions'])}
        for depth in range(job['depth'] + 1):
            children = []
            for handle, candidate in beam:
                behavior._check(); state = engine.request('restore', handle=handle)
                if state != candidate['request']['state']:
                    raise ValueError('Opaque native snapshot changed the exact candidate observation')
                risks = engine.request('risks')
                record = {'depth': depth, 'state_sha256': fingerprint(state),
                    'remaining_frames': state['timing']['power']['remaining_frames'],
                    'legal_prefix': candidate['prefix_actions'][len(base['prefix_actions']):],
                    'immediate_counterfactuals': risks, 'fatal_transitions': {}}
                if state['frightened'] and behavior._near(state, True) <= 3 and behavior._near(state, False) > 3:
                    for alternative, risk in risks.items():
                        if not risk['life_lost']:
                            continue
                        engine.request('restore', handle=handle); step = engine.step(alternative)
                        transition = behavior._transition(state, step); record['fatal_transitions'][alternative] = transition
                        if transition['remaining_before'] > 0 and transition['remaining_after'] == 0:
                            found = copy.deepcopy(candidate)
                            found.update(alternative=alternative, reuse_source_suffix=False, teacher_choice_hint=None,
                                scout_flags={'expiry': True, 'anticipatory': False, 'dry_recovery': False}, scout_score=[],
                                derivation={'kind': 'legal_native_expiry_beam', 'helper_sha256': job['helper_sha256'],
                                    'worker_sha256': job['worker_sha256'], 'job_sha256': fingerprint(job),
                                    'steps_from_source': depth, 'parent_state_sha256': base['state_sha256']},
                                native_expiry_scout_transition=transition)
                            found['source'] = {**found['source'], 'kind': 'legal_native_expiry_fork',
                                'perturbation': 'beam_legal_delay_or_intercept', 'perturbation_actions': record['legal_prefix']}
                            found['id'] = fingerprint(found)[:24]; discovered.append(found)
                explored.append(record)
                if depth == job['depth'] or not state['frightened']:
                    continue
                for move, risk in risks.items():
                    if risk['life_lost']:
                        continue
                    engine.request('restore', handle=handle); step = engine.step(move)
                    if step['outcome'] or step['events']['ghosts_eaten'] or step['events']['power_pellets']:
                        skipped.append({'state_sha256': fingerprint(state), 'move': move,
                                        'outcome': step['outcome'], 'events': step['events']}); continue
                    next_state = step['state']
                    # Native remaining_frames0 may still be the last active
                    # powered frame. Only the native active flag ends search.
                    if not next_state['frightened']:
                        continue
                    child = copy.deepcopy(candidate); child['prefix_actions'].append(move)
                    child['prefix_state_sha256'].append(fingerprint(next_state))
                    child['request'] = body(next_state); child['state_sha256'] = fingerprint(next_state)
                    digest = fingerprint(child['prefix_actions'])
                    if digest in seen_prefixes:
                        continue
                    seen_prefixes.add(digest)
                    snapshot = engine.request('snapshot')['handle']
                    children.append((_rank(next_state), snapshot, child))
            if not children:
                break
            ordered = sorted(children, key=lambda row: (row[0], row[2]['state_sha256']), reverse=True)
            beam = [(handle, candidate) for _, handle, candidate in ordered[:job['beam']]]
            for _, handle, _ in ordered[job['beam']:]:
                engine.request('forget', handle=handle)
    return {'job_sha256': fingerprint(job), 'helper_sha256': job['helper_sha256'], 'worker_sha256': job['worker_sha256'],
            'found': discovered, 'explored': explored, 'skipped_ghost_eating_or_power_reset': skipped,
            'elapsed_seconds': time.monotonic()-started}


def search_expiry(evidence_directory, output_directory, qualification_path,
                  max_parents=96, workers=8, depth=12, beam=8, wave_id='expiry-beam-001'):
    if not 1 <= workers <= 8 or not 1 <= max_parents <= 1000 or not 1 <= depth <= 12 or not 1 <= beam <= 8:
        raise ValueError('Use1..8workers,1..1000parents,depth1..12andbeam1..8')
    directory = Path(output_directory); evidence = Path(evidence_directory); config = behavior._assert_configuration(directory, qualification_path)
    helper_hash, worker_hash = behavior._hash(Path(__file__)), behavior._hash(WORKER)
    buckets = {}; seen = set()
    for path in sorted((evidence / 'episodes').glob('*/receipt.json')):
        receipt = behavior._load(path)
        if not receipt['accepted']:
            continue
        source = None
        for proof in behavior._rows(path.with_name('verified.jsonl')):
            state = proof['request']['state']; remaining = state['timing']['power']['remaining_frames']
            if not state['frightened'] or not 0 < remaining <= 96 or len(state['legal_moves']) < 2:
                continue
            near = behavior._near(state, True)
            if near > 6:
                continue
            digest = proof['request_sha256']
            if digest in seen:
                continue
            seen.add(digest)
            if source is None:
                source = behavior._rows(path.with_name('source-trace.jsonl'))
            candidate = behavior._source_candidate(path.parent.name, receipt, source, proof)
            key = (candidate['split'], candidate['family'])
            # Favor clocks long enough to intercept/delay without immediate eat.
            buckets.setdefault(key, []).append((int(near >= 2), -abs(remaining-40), near, candidate, path.parent.name))
    selected = []; keys = sorted(buckets, key=lambda k: (k[1], k[0]))
    for key in keys:
        buckets[key].sort(key=lambda r: (r[:3], r[3]['state_sha256']), reverse=True)
    while len(selected) < max_parents and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < max_parents:
                selected.append(buckets[key].pop(0))
    jobs = []
    for _, _, _, candidate, identifier in selected:
        behavior._bind_source(evidence, directory, identifier)
        source_receipt = directory / f'source-evidence/episodes/{identifier}/receipt.json'
        candidate['source_binding'] = {'episode': identifier, 'receipt': source_receipt.relative_to(directory).as_posix(),
            'receipt_sha256': behavior._hash(source_receipt), 'original_candidate_sha256': None,
            'source_index': candidate['source']['root_index']}
        job = {'candidate': candidate, 'depth': depth, 'beam': beam, 'source_maximum_power_frames': 96, 'helper_sha256': helper_hash,
               'worker_sha256': worker_hash, 'source_sha256': config['source_sha256']}
        job['id'] = fingerprint(job)[:24]; jobs.append(job)
    context = multiprocessing.get_context('spawn'); cancel = context.Event()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=behavior._worker_setup, initargs=(cancel,))
    started = time.monotonic(); pending = {}; completed = []; queued = iter(jobs)
    def submit():
        job = next(queued, None)
        if job is not None:
            path = directory / 'scout-waves' / f'{wave_id}-parent-{job["id"]}.json'
            if path.exists():
                receipt = behavior._load(path)
                if receipt['job_sha256'] != fingerprint(job) or receipt['helper_sha256'] != helper_hash or receipt['worker_sha256'] != worker_hash:
                    raise ValueError('Committed beam scout source/job changed')
                completed.append(receipt); return submit()
            pending[pool.submit(_job, job)] = job
    try:
        for _ in range(min(workers, len(jobs))):
            submit()
        while pending:
            ready, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not ready:
                print(f'[expiry/beam] {len(completed)}/{len(jobs)} parents; {time.monotonic()-started:.0f}s', flush=True)
            for future in ready:
                job = pending.pop(future); proof = future.result(); entries = []
                for candidate in proof.pop('found'):
                    path = directory / 'cases' / candidate['id'] / 'candidate.json'
                    behavior._save(path, candidate, immutable=True)
                    entries.append({'id': candidate['id'], 'split': candidate['split'], 'family': candidate['family'],
                                    'candidate_sha256': behavior._hash(path)})
                receipt = {**proof, 'wave_id': wave_id, 'job': job, 'candidates': entries,
                           'label_admission': 'Pending full zero-loss qualified teacher recovery and cold original-engine replay'}
                behavior._save(directory / 'scout-waves' / f'{wave_id}-parent-{job["id"]}.json', receipt, immutable=True)
                completed.append(receipt); submit()
                print(f'[expiry/beam] {len(completed)}/{len(jobs)} parents; {len(entries)} real expiry-fatal roots', flush=True)
    except BaseException:
        cancel.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    report = {'wave_id': wave_id, 'helper_sha256': helper_hash, 'worker_sha256': worker_hash,
              'plausible_distinct_parents': len(seen), 'completed_parents': len(completed),
              'found_candidates': sum(len(p['candidates']) for p in completed),
              'worker_seconds': sum(p['elapsed_seconds'] for p in completed),
              'elapsed_seconds': time.monotonic()-started, 'learner_disagreement_measured': False}
    behavior._save(directory / 'scout-waves' / f'{wave_id}-summary.json', {**report, 'candidates': []}, immutable=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True); parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-parents', type=int, default=96); parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--beam', type=int, default=8); parser.add_argument('--wave-id', default='expiry-beam-001')
    args = parser.parse_args()
    print(json.dumps(search_expiry(args.evidence, args.output, args.qualification, args.max_parents,
                                  args.workers, args.depth, args.beam, args.wave_id), indent=2))
