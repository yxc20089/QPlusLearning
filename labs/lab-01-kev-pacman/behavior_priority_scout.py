"""Fast, self-hashed additional scout waves for causal ghost-escape proofs.

State hashes are computed once per full source episode. Exact prefixes are
constructed only after selection; every proposed case still needs the frozen
behavior_scenarios verifier's full native proof. Portfolio predictions rank
scouting and never count as admission evidence.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import behavior_scenarios as behavior
from pacman_lab import body
import teacher_offline_data as offline
from teacher_validation import fingerprint


def scout_priority_roots(evidence_directory, output_directory, qualification_path,
                         train_roots=800, development_roots=200, wave_id='anticipatory-002'):
    if any(isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 10000
           for n in (train_roots, development_roots)) or train_roots + development_roots == 0:
        raise ValueError('Use a bounded nonempty train/development root budget')
    directory = Path(output_directory); evidence = Path(evidence_directory)
    config = behavior._assert_configuration(directory, qualification_path); helper_hash = behavior._hash(Path(__file__))
    path = directory / 'scout-waves' / f'{wave_id}.json'
    budgets = {'train': train_roots, 'development': development_roots}
    binding = {'helper_sha256': helper_hash, 'generator_sha256': config['generator_sha256'],
               'source_sha256': config['source_sha256'], 'qualification_sha256': config['qualification_sha256'],
               'budgets': budgets, 'wave_id': wave_id,
               'source_receipts_sha256': fingerprint(sorted(behavior._hash(p) for p in (evidence / 'episodes').glob('*/receipt.json')))}
    if path.exists():
        saved = behavior._load(path)
        if saved['binding'] != binding:
            raise ValueError('Committed scout wave inputs/budget/helper changed')
        for entry in saved['candidates']:
            if behavior._hash(directory / 'cases' / entry['id'] / 'candidate.json') != entry['candidate_sha256']:
                raise ValueError('Committed scout candidate changed')
        return saved
    existing = {offline.request_fingerprint(behavior._load(p)['request'])
                for p in (directory / 'cases').glob('*/candidate.json')}
    episodes = {}; candidates = []; states = {}; counts = Counter()
    for receipt_path in sorted((evidence / 'episodes').glob('*/receipt.json')):
        receipt = behavior._load(receipt_path)
        if not receipt['accepted']:
            continue
        offline._receipt_files(evidence, receipt)
        offline._validate_source_binding(receipt, behavior._load(receipt_path.with_name('source-config.json')),
                                        behavior._load(receipt_path.with_name('source-attempt.json')))
        if receipt['metrics']['life_losses']:
            raise ValueError('Priority scout source contains a teacher loss')
        identifier = receipt_path.parent.name; source = behavior._rows(receipt_path.with_name('source-trace.jsonl'))
        # One state fingerprint per source row, independent of candidate count.
        hashes = [fingerprint(row['request']['state']) for row in source]
        actions = [row['diagnostics']['choice'] for row in source]
        episodes[identifier] = (receipt, source, hashes, actions)
        for proof in behavior._rows(receipt_path.with_name('verified.jsonl')):
            state = proof['request']['state']; risks = proof['diagnostics']['immediate_counterfactuals']
            if len(state['legal_moves']) < 2 or any(r['life_lost'] for r in risks.values()):
                continue
            index = proof['source_index']; choice = proof['choice']; plan = source[index]['teacher']
            alternatives = [row for row in plan['candidates'] if row['action'] != choice]
            predicted_losses = max((sum(not s['survived'] for s in row['scenarios']) for row in alternatives), default=0)
            if predicted_losses == 0:
                continue
            digest = proof['request_sha256']; split = receipt['binding']['job']['split']
            # Quarantine cross-partition or differently labelled observations
            # after scanning the complete source snapshot.
            states.setdefault(digest, set()).add((split, choice))
            if digest in existing:
                continue
            near = behavior._near(state, False)
            if near > behavior.RULES['scout_ghost_tiles']:
                continue
            counts[f'{split}/plausible_warning_roots'] += 1
            score = (predicted_losses, -near, int(len(state['legal_moves']) >= 3),
                     state.get('decisions_since_last_pellet', 0), -proof['source_index'])
            candidates.append((score, identifier, proof, digest))
    buckets = {}; seen = set()
    for entry in sorted(candidates, key=lambda r: (r[0], r[1], r[2]['source_index']), reverse=True):
        digest = entry[3]
        if digest in seen or len(states[digest]) != 1:
            continue
        seen.add(digest); receipt = episodes[entry[1]][0]; job = receipt['binding']['job']
        buckets.setdefault((job['split'], job['level'], job['seed']), []).append(entry)
    selected = []; used = Counter(); keys = sorted(buckets, key=lambda k: (k[1], k[2], k[0]))
    while any(buckets.values()) and any(used[s] < budgets[s] for s in budgets):
        progress = False
        for key in keys:
            if buckets[key] and used[key[0]] < budgets[key[0]]:
                selected.append(buckets[key].pop(0)); used[key[0]] += 1; progress = True
        if not progress:
            break
    jobs = []; source_bound = set()
    for score, identifier, proof, digest in selected:
        receipt, source, hashes, actions = episodes[identifier]; job = receipt['binding']['job']; index = proof['source_index']
        if identifier not in source_bound:
            behavior._bind_source(evidence, directory, identifier); source_bound.add(identifier)
        state = proof['request']['state']; candidate = {'split': job['split'], 'seed': job['seed'], 'level': job['level'],
            'family': f'level-{job["level"]}-seed-{job["seed"]}', 'origin': 'verified_teacher_trajectory',
            'source': {'kind': 'verified_teacher_episode', 'episode': identifier,
                       'trace_sha256': receipt['binding']['source_trace_sha256'], 'root_index': index},
            'request': body(state), 'state_sha256': hashes[index], 'prefix_actions': actions[:index],
            'prefix_state_sha256': hashes[:index + 1], 'teacher_choice_hint': proof['choice'],
            'source_binding': {'episode': identifier, 'receipt': f'source-evidence/episodes/{identifier}/receipt.json',
                'receipt_sha256': behavior._hash(directory / f'source-evidence/episodes/{identifier}/receipt.json'),
                'original_candidate_sha256': None, 'source_index': index},
            'reuse_source_suffix': True, 'scout_flags': {'anticipatory': True, 'expiry': False, 'dry_recovery': False},
            'scout_score': list(score), 'derivation': {'kind': 'cached_prefix_priority_scout',
                 'helper_sha256': helper_hash, 'wave_binding_sha256': fingerprint(binding),
                 'request_sha256': digest, 'portfolio_predictions_are_not_admission_proof': True}}
        alternative = behavior._alternatives(state, proof['diagnostics']['immediate_counterfactuals'],
                                             source[index]['teacher'], proof['choice'])[0]
        candidate['alternative'] = alternative
        candidate['id'] = fingerprint(candidate)[:24]
        candidate_path = directory / 'cases' / candidate['id'] / 'candidate.json'
        behavior._save(candidate_path, candidate, immutable=True)
        jobs.append({'id': candidate['id'], 'split': candidate['split'], 'family': candidate['family'],
                     'candidate_sha256': behavior._hash(candidate_path)})
        if len(jobs) % 100 == 0:
            print(f'[priority/scout] {len(jobs)}/{len(selected)} exact native root prefixes committed', flush=True)
    report = {'wave_id': wave_id, 'binding': binding, 'candidates': jobs, 'source_warning_counts': dict(counts),
              'selected': dict(used), 'deficits': {s: budgets[s] - used[s] for s in budgets},
              'interpretation': 'All first actions survive; portfolio danger only prioritizes. Full paired native proof is still required.'}
    behavior._save(path, report, immutable=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True); parser.add_argument('--train-roots', type=int, default=800)
    parser.add_argument('--development-roots', type=int, default=200); parser.add_argument('--wave-id', default='anticipatory-002')
    args = parser.parse_args()
    result = scout_priority_roots(args.evidence, args.output, args.qualification,
                                 args.train_roots, args.development_roots, args.wave_id)
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}, indent=2))
