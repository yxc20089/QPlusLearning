"""Read native full-game evidence and probe disjoint hard development states.

These diagnostics do not modify the frozen benchmark protocol or teacher source.
Snapshot agreement is secondary to native full-game survival and food progress.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from api_client import distribution
from gameplay_benchmark import strata
from pacman_lab import inference_request


def cohorts(state, risks=None):
    from v4_data import cohort_flags
    result = strata(state)
    result.update(cohort_flags(state, risks))
    return [name for name, included in result.items() if included]


def native_summary(report):
    """Count safety opportunities and progress with their actual denominators."""
    totals = Counter()
    buckets = defaultdict(Counter)
    for episode in report['episodes']:
        metrics = episode['metrics']
        for name in ('decisions', 'simulation_frames', 'pellets_collected', 'life_losses',
                     'avoidable_immediate_deaths', 'avoidable_risk_decisions',
                     'all_immediate_actions_fatal', 'loop_decisions', 'first_life_pellets',
                     'post_respawn_pellets', 'power_pellets', 'ghosts_eaten'):
            totals[name] += metrics[name]
        if not episode.get('trace'):
            raise ValueError('Native cohort evaluation requires saved full trajectories')
        for line in Path(episode['trace']).read_text().splitlines():
            row = json.loads(line)
            diag, state = row['diagnostics'], row['request']['state']
            risk = diag['immediate_counterfactuals']
            opportunity = any(r['life_lost'] for r in risk.values()) and any(not r['life_lost'] for r in risk.values())
            fatal = risk[diag['choice']]['life_lost']
            for cohort in cohorts(state, risk):
                values = buckets[cohort]
                values['decisions'] += 1
                values['frames'] += diag['action_frames']
                values['pellets'] += diag['pellets_before'] - diag['pellets_after']
                values['safe_alternative_opportunities'] += opportunity
                values['avoidable_immediate_deaths'] += fatal and opportunity
                values['all_immediate_actions_fatal'] += all(r['life_lost'] for r in risk.values())
    def rate(numerator, denominator):
        return numerator / denominator if denominator else None
    return {'games': len(report['episodes']), 'clears': report['level_clears'],
            'capped_episodes': report['capped_episodes'], 'totals': dict(totals),
            'pellets_per_1000_native_frames': rate(1000 * totals['pellets_collected'], totals['simulation_frames']),
            'fatal_choice_when_safe_available': rate(totals['avoidable_immediate_deaths'], totals['avoidable_risk_decisions']),
            'mean_first_life_pellets': rate(totals['first_life_pellets'], len(report['episodes'])),
            'maximum_pellet_free_decisions': max(e['metrics']['longest_no_pellet_decisions'] for e in report['episodes']),
            'maximum_loop_streak': max(e['metrics']['longest_loop_streak_decisions'] for e in report['episodes']),
            'cohorts': {name: {**dict(values), 'fatal_choice_when_safe_available':
                              rate(values['avoidable_immediate_deaths'], values['safe_alternative_opportunities'])}
                        for name, values in buckets.items()},
            'interpretation': 'Cohort exposures differ across controller trajectories. '
                              'Immediate counterfactuals do not prove eventual survival. '
                              'Efficiency uses native action frames, excluding HTTP latency and life animations.'}


def evaluate_hard_cases(path, predict_batch, model_info, batch_size=16):
    """Frozen, disjoint development labels; strip labels/meta before inference."""
    path = Path(path)
    content = path.read_bytes()
    records = [json.loads(line) for line in content.splitlines()]
    if not records or batch_size <= 0:
        raise ValueError('Use nonempty hard development data and a positive probe batch')
    rows, identity = [], model_info()
    for start in range(0, len(records), batch_size):
        if model_info() != identity:
            raise RuntimeError('Active adapter changed during hard-state evaluation')
        part = records[start:start + batch_size]
        answers = predict_batch([inference_request(record) for record in part])
        if len(answers) != len(part):
            raise ValueError('Probe response count differs from its input batch')
        for record, response in zip(part, answers):
            question = record['questions']['move']
            answer = response['answers']['move']
            p = distribution(answer, question['criteria'])
            gold = question['label']
            if gold not in p:
                raise ValueError('Teacher label is not a legal move')
            others = [v for key, v in p.items() if key != gold]
            groups = set(cohorts(record['state']))
            groups.update(name for name, included in record['_meta'].get('cohorts', {}).items() if included)
            rows.append({'id': record['_meta']['id'], 'gold': gold, 'prediction': answer['choice'],
                         'probabilities': answer['probabilities'],
                         'teacher_probability': p[gold], 'cross_entropy': -math.log(max(p[gold], 1e-12)),
                         'teacher_probability_margin': p[gold] - max(others, default=0),
                         'cohorts': sorted(groups)})
        print(f'[hard development] {min(start + len(part), len(records))}/{len(records)}', flush=True)
    if model_info() != identity:
        raise RuntimeError('Active adapter changed during hard-state evaluation')
    def summary(group):
        return {'n': len(group), 'teacher_agreement': sum(r['gold'] == r['prediction'] for r in group) / len(group),
                'mean_cross_entropy': sum(r['cross_entropy'] for r in group) / len(group),
                'mean_teacher_probability_margin': sum(r['teacher_probability_margin'] for r in group) / len(group)}
    groups = defaultdict(list)
    for row in rows:
        for name in row['cohorts']:
            groups[name].append(row)
    return {'active_checkpoint': identity, 'dataset_sha256': hashlib.sha256(content).hexdigest(),
            **summary(rows), 'cohorts': {name: summary(group) for name, group in groups.items()},
            'rows': rows, 'interpretation': 'Single-state teacher agreement diagnoses imitation. '
                                          'CE uses rounded API probabilities with a 1e-12 floor; it is not the training loss. '
                                          'Native full-game outcomes remain the primary benchmark.'}
