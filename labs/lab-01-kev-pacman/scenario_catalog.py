"""Categorize selected teacher-only examples without changing training inputs.

One primary scenario per targeted root makes counts additive. Informative
windows inherit their root's primary scenario; their own overlapping tags are
also retained. These are game-state categories, not measured learner errors.
Run after teacher_offline_data's admission/validation has succeeded.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

PREFIX = 'pacman-native-v4-offline'
SCENARIOS = [
    ('immediate_collision_avoidance', 'Immediate collision avoidance', 'immediate_critical'),
    ('actual_power_expiry_near_ghost', 'Power expires during the chosen move near a ghost', 'actual_power_expiry_during_teacher_action_near_ghost'),
    ('potential_power_expiry_near_ghost', 'Power may expire during a move near a ghost', 'potential_expiry_during_action_near_ghost'),
    ('low_power_clock', 'Low remaining power clock, other positions', 'power_expiry'),
    ('nearby_dangerous_ghost', 'Nearby dangerous ghost', 'ghost_intercept'),
    ('pellet_stall_routing', 'Routing after a pellet-free interval', 'food_dry'),
    ('sparse_pellet_cleanup', 'Remaining-pellet cleanup', 'sparse_food'),
]
REPLAY = 'prior_pacman_replay'


def primary_scenario(cohorts):
    for identifier, _, flag in SCENARIOS:
        if cohorts.get(flag):
            return identifier
    raise ValueError('A targeted root has no qualifying scenario')


def categorize(rows):
    roots = {r['_meta']['id']: r for r in rows if r['_meta']['class'] == 'targeted'}
    catalog = []
    for row in rows:
        meta = row['_meta']; kind = meta['class']
        if kind == 'replay':
            scenario = REPLAY
        elif kind == 'targeted':
            scenario = primary_scenario(meta['cohorts'])
        elif kind == 'informative':
            parent = roots.get(meta['parent_targeted_id'])
            if parent is None or parent['_meta']['episode'] != meta['episode']:
                raise ValueError('Informative row has no same-episode targeted parent')
            scenario = primary_scenario(parent['_meta']['cohorts'])
        else:
            raise ValueError('Unknown selected example class')
        catalog.append({'id': meta['id'], 'split': meta['split'], 'class': kind,
            'primary_scenario': scenario, 'tags': sorted(k for k, value in meta['cohorts'].items() if value),
            'parent_targeted_id': meta.get('parent_targeted_id'), 'window_kind': meta.get('window_kind'),
            'episode': meta.get('episode'), 'group_id': meta['group_id'], 'source_index': meta.get('source_index'),
            'source_record_id': meta.get('source_id'), 'teacher_move': row['questions']['move']['label'],
            'state_sha256': meta['state_sha256'], 'request_sha256': meta['request_sha256']})
    return catalog


def write_catalog(directory):
    directory = Path(directory)
    manifest_path = directory / f'{PREFIX}-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['dataset'] != PREFIX:
        raise ValueError('Use the separately admitted teacher-only dataset')
    catalog = []; partitions = {}
    for split in ('train', 'development'):
        path = directory / f'{PREFIX}-{split}.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][path.name]:
            raise ValueError('Selected data differ from the admitted manifest')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        measured = Counter(r['_meta']['class'] for r in rows)
        if dict(measured) != manifest['counts'][split] or any(r['_meta']['split'] != split for r in rows):
            raise ValueError('Selected split/class counts differ')
        selected = categorize(rows); catalog.extend(selected)
        groups = {}
        for identifier in [s[0] for s in SCENARIOS] + [REPLAY]:
            subset = [r for r in selected if r['primary_scenario'] == identifier]
            counts = Counter(r['class'] for r in subset)
            groups[identifier] = {'targeted': counts['targeted'], 'informative': counts['informative'],
                'replay': counts['replay'], 'total_examples': len(subset),
                'contributing_source_games': len({r['episode'] for r in subset}) if identifier != REPLAY else None,
                'source_families': len({r['group_id'] for r in subset}),
                'window_types': dict(Counter(r['window_kind'] for r in subset if r['class'] == 'informative'))}
        if sum(g['total_examples'] for g in groups.values()) != len(rows):
            raise ValueError('Primary categories do not account for every selected request')
        partitions[split] = {'total_examples': len(rows), 'primary_scenarios': groups,
                            'overlapping_tags': dict(Counter(tag for r in selected if r['class'] == 'targeted' for tag in r['tags']))}
    csv_path = directory / f'{PREFIX}-example-categories.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog[0])); writer.writeheader()
        writer.writerows({**r, 'tags': json.dumps(r['tags'])} for r in catalog)
    summary = {'dataset': PREFIX, 'schema': 'selected-teacher-scenario-catalog-v1',
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'data_sha256': manifest['files'], 'catalog_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'classifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'primary_priority': [{'id': identifier, 'name': name, 'predicate': flag} for identifier, name, flag in SCENARIOS],
        'rules': ['First matching scenario in the stated priority; each request has exactly one primary category.',
                  'Informative windows inherit the selected parent root category; their own tags are recorded separately.',
                  'Respawn is an overlapping context tag; it alone is not a hard scenario.',
                  'Game counts overlap across categories; example totals do not.',
                  'These categories do not claim measured v3 mistakes, optimality, or learned v4 performance.'],
        'partitions': partitions}
    path = directory / f'{PREFIX}-scenario-catalog.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_catalog(args.dataset), indent=2))
