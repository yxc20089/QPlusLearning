"""Balanced, evidence-grounded CPU teacher distillation for native-v4.

The original offline artifact and source hashes stay unchanged. New behaviors
are proved on native trajectories, never inferred from a nearby ghost or a low
timer alone. No live model probe is run. An archived learner disagreement is
reported only for a checkpoint/trace/state-bound recorded root.
"""
import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import tempfile
import zipfile

import teacher_offline_data as offline
from pacman_lab import ROOT, body, distances, position
from teacher_validation import fingerprint, require_qualified_teacher, source_hashes

PREFIX = 'pacman-native-v4-balanced'
RECIPE = copy.deepcopy(offline.RECIPE)
TARGETS = copy.deepcopy(offline.TARGETS)
FAMILY_CAP = 128
POST_MINIMUMS = {'train': 640, 'development': 64}
CRITICAL_MINIMUMS = {'train': 512, 'development': 64}
BROAD_MAX_FRACTION = .15
MAX_WINDOWS_PER_ROOT = 4
WINDOW_LIMIT = 24
ROLES = ('immediate_evasion', 'anticipatory_escape', 'actual_expiry_evasion',
         'retreat_to_food', 'productive_routing', 'cleanup', 'broad_exposure')
# Provisional allocation. A caller may supply another explicit full-budget plan
# after the CPU yield scout; it is recorded and validated, never silently filled.
DEFAULT_PLAN = {
    'train': dict(zip(ROLES, (512, 256, 128, 512, 1152, 700, 400))),
    'development': dict(zip(ROLES, (64, 24, 12, 48, 128, 72, 36))),
}
OPPOSITE = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
CLAIMS = {
    'controller': 'Qualified CPU teacher; no live learner/model probes. Archived learner predictions are evidence only when explicitly checkpoint/trace/state bound.',
    'immediate_evasion': 'Gold is native-immediately safe and another original legal action loses a life.',
    'anticipatory_escape': 'Every first action is immediately safe; a forced alternative followed by the same teacher later loses a life or fails while gold clears without a loss.',
    'actual_expiry_evasion': 'A measured alternative expires the actual native power clock and loses a life; gold is safe and its teacher continuation wins without loss.',
    'retreat_to_food': 'A measured safe escape/retreat is followed by actual same-life pellet progress within 24 decisions and a zero-loss maze clear.',
    'productive_routing': 'At a non-opening junction in a dry, revisited or depleted-food routing context, gold consumes a pellet immediately while another surviving legal action consumes none. This proves immediate progress, not global harm of the safe alternative.',
    'cleanup': 'With <=30 pellets, gold has the same measured immediate pellet advantage over a surviving alternative; sparse-food exposure alone does not qualify.',
    'broad_exposure': 'A bounded residual proximity/timer exposure; not a proven negative and at most 15% of targeted roots.',
    'windows': 'Actual bounded same-life teacher frames linked to selected roots. The640/64 escape-progress floor requires food consumed before the window state AND an immediate, actual-expiry, paired anticipatory or measured geometric threat escape parent. Dry routing recovery and ordinary later food do not count toward that floor.',
    'optimality': 'No universal action optimality or blanket native-v3 disagreement claim. Only explicitly bound archived roots measure learner disagreement. Full-game student evaluation is still required.',
    'input': 'Original observation/action v2, instructions and every original legal option. Teacher futures/results are metadata only.',
}


def validate_plan(plan):
    if set(plan) != set(TARGETS):
        raise ValueError('Use explicit training/development role plans')
    for split, quotas in plan.items():
        if (set(quotas) != set(ROLES) or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in quotas.values())
                or sum(quotas.values()) != TARGETS[split]['targeted']
                or quotas['broad_exposure'] > math.floor(BROAD_MAX_FRACTION * TARGETS[split]['targeted'])
                or quotas['immediate_evasion'] < CRITICAL_MINIMUMS[split]
                or any(quotas[role] == 0 for role in ('anticipatory_escape', 'actual_expiry_evasion', 'retreat_to_food', 'productive_routing', 'cleanup'))):
            raise ValueError('Role plan violates full budgets, grounded behavior diversity, critical floor or broad cap')
    return copy.deepcopy(plan)


def _danger_distances(state):
    routes = distances(state['maze'], position(state['player']))
    return {g['name']: routes.get(position(g), 999) for g in state['ghosts']
            if g['mode'] == 'outside' and not g['frightened']}


def _sequence(proofs, root_index, life_index, limit=WINDOW_LIMIT):
    result = []
    for index in range(root_index, root_index + limit + 1):
        proof = proofs.get(index)
        if proof is None or proof['life_index'] != life_index:
            break
        result.append(proof)
    return result


def trajectory_behavior(proof, proofs):
    """Only measured immediate outcomes and a real qualified teacher sequence."""
    state = proof['request']['state']; risks = proof['diagnostics']['immediate_counterfactuals']; gold = proof['choice']
    safe = not risks[gold]['life_lost']
    alternatives = {move: risk for move, risk in risks.items() if move != gold and not risk['life_lost']}
    pellet_advantage = bool(safe and risks[gold]['pellets'] > 0 and any(r['pellets'] == 0 for r in alternatives.values()))
    critical = bool(safe and any(r['life_lost'] for r in risks.values()))
    sequence = _sequence(proofs, proof['source_index'], proof['life_index'])
    first_food = next((p['source_index'] for p in sequence[1:]
        if p['diagnostics']['pellets_after'] < p['diagnostics']['pellets_before']), None)
    following = proofs.get(proof['source_index'] + 1)
    before_distances = _danger_distances(state); before = min(before_distances.values(), default=999)
    after_distances = _danger_distances(following['request']['state']) if following and following['life_index'] == proof['life_index'] else {}
    after = min(after_distances.values(), default=999) if following and following['life_index'] == proof['life_index'] else None
    # Either a genuinely critical reverse or a turn that measurably increases
    # the gap to currently dangerous ghosts. Becoming powered is not silently
    # described as a geometric retreat.
    nearest_names = {name for name, distance in before_distances.items() if distance == before}
    geometric = bool(before <= 6 and after is not None and after > before and nearest_names
        and all(name in after_distances and after_distances[name] > before_distances[name] for name in nearest_names))
    safe_reverse = critical and gold == OPPOSITE[state['player']['heading']]
    escape = bool(safe and gold != state['player']['heading'] and (safe_reverse or geometric))
    # The two-stage behavior has food AFTER escape. Eating food on the root
    # action alone is useful routing evidence but is not follow-through.
    retreat = escape and first_food is not None
    current_visits = state.get('visits_to_current_tile', 0)
    routing_context = (state['turn'] >= 12 and (state.get('decisions_since_last_pellet', 0) >= 8
                        or current_visits >= 2 or state['pellets_remaining'] <= 122))
    flags = {
        'immediate_evasion': critical, 'anticipatory_escape': False, 'actual_expiry_evasion': False,
        'retreat_to_food': bool(retreat),
        'productive_routing': pellet_advantage and len(state['legal_moves']) >= 3 and routing_context,
        'cleanup': pellet_advantage and state['pellets_remaining'] <= 30,
        'broad_exposure': bool(proof['cohorts']['ghost_intercept'] or proof['cohorts']['power_expiry']),
    }
    evidence = {'gold': gold, 'immediate_fatal_alternatives': [m for m, r in risks.items() if r['life_lost']],
        'surviving_zero_pellet_alternatives': [m for m, r in alternatives.items() if r['pellets'] == 0],
        'gold_immediate_pellets': risks[gold]['pellets'], 'first_food_source_index': first_food,
        'non_opening_routing_context': routing_context,
        'escape': escape, 'danger_distance_before': before, 'danger_distance_after': after,
        'escape_kind': 'critical_safe_reverse' if escape and safe_reverse else 'same_ghost_geometric' if escape else None,
        'same_ghost_distances': {name: {'before': distance, 'after': after_distances.get(name)}
                               for name, distance in before_distances.items()},
        'first_food_decision_offset': None if first_food is None else first_food - proof['source_index'],
        'native_actual_power_transition': proof['power_transition']}
    return flags, evidence


def _wrap(record, provenance, behavior, evidence):
    result = copy.deepcopy(record); result['questions']['move']['src'] = PREFIX
    old = result['_meta']; identifier = fingerprint({'request': offline.request_fingerprint(record), 'provenance': provenance})[:32]
    measured_disagreement = bool(old.get('class') == 'targeted' and old.get('learner_disagreement_measured', False))
    if measured_disagreement and not isinstance(old.get('recorded_learner_evidence'), dict):
        raise ValueError('Measured archived disagreement has no recorded learner evidence')
    result['_meta'] = {'id': f'{PREFIX}:{identifier}', 'split': old['split'], 'class': 'targeted',
        'primary_role': None, 'origin': provenance['type'], 'group_id': old['group_id'],
        'seed': old['seed'], 'level': old['level'], 'state_sha256': fingerprint(record['state']),
        'request_sha256': offline.request_fingerprint(record), 'life_index': old.get('life_index', 0),
        'source_index': old.get('source_index', record['state']['turn']), 'behavior': behavior,
        'behavior_evidence': evidence, 'cohorts': copy.deepcopy(old.get('cohorts', {})),
        'provenance': provenance, 'parent_targeted_id': None, 'window_kind': None,
        'window_decision_offset': None, 'learner_disagreement_measured': measured_disagreement,
        'recorded_learner_evidence': copy.deepcopy(old.get('recorded_learner_evidence')),
        'parent_recorded_learner_evidence': copy.deepcopy(old.get('parent_recorded_learner_evidence'))}
    return result


def _offline_records(episodes):
    records = []; contexts = {}
    for identifier, episode in episodes.items():
        receipt = episode['receipt']; job = receipt['binding']['job']; proofs = {p['source_index']: p for p in episode['proofs']}
        for proof in episode['proofs']:
            if len(proof['request']['state']['legal_moves']) < 2 or proof['request_sha256'] in episode['excluded_request_sha256']:
                continue
            flags, evidence = trajectory_behavior(proof, proofs)
            if not any(flags.values()):
                continue
            original = offline._record(proof, receipt, identifier, job['split'])
            provenance = {'type': 'verified_offline_trajectory', 'episode': identifier,
                'source_index': proof['source_index'], 'source_receipt': f'episodes/{identifier}/receipt.json',
                'receipt_sha256': episode['receipt_sha256'], 'original_record_sha256': fingerprint(original)}
            row = _wrap(original, provenance, flags, evidence); records.append(row)
            contexts[row['_meta']['id']] = {'episode': episode, 'proofs': proofs, 'proof': proof}
    return records, contexts


def _scenario_roles(record):
    meta = record['_meta']; cohorts = meta.get('cohorts', {})
    # These flags are supplied by, and checked against native branch proofs in,
    # behavior_scenarios.validate_scenarios. No proximity-derived substitution.
    # The expiry predicate requires an immediately fatal original alternative
    # and a safe gold action. It therefore also proves immediate criticality.
    expiry = bool(cohorts.get('actual_expiry_fatal_alternative') or cohorts.get('actual_expiry_evasion'))
    critical = bool(cohorts.get('immediate_critical') or cohorts.get('immediate_evasion') or expiry)
    state = record.get('state', {})
    contextual_junction = bool(len(state.get('legal_moves', [])) >= 3 and state.get('turn', 0) >= 12
        and (state.get('decisions_since_last_pellet', 0) >= 8
             or state.get('visits_to_current_tile', 0) >= 2 or state.get('pellets_remaining', 244) <= 122))
    return {'immediate_evasion': critical,
        'anticipatory_escape': bool(cohorts.get('anticipatory_harm') or cohorts.get('anticipatory_escape')),
        'actual_expiry_evasion': expiry,
        'retreat_to_food': bool(cohorts.get('productive_retreat_to_food') or cohorts.get('retreat_to_food')),
        'productive_routing': bool(cohorts.get('productive_routing')
                                   or cohorts.get('productive_junction') and contextual_junction),
        'cleanup': bool(cohorts.get('cleanup') or cohorts.get('sparse_cleanup')), 'broad_exposure': False}


def load_pools(offline_directory, scenario_directory, qualification_path, verify_native=False, recorded_directory=None):
    require_qualified_teacher(qualification_path)
    config = offline._load(Path(offline_directory) / 'offline-config.json')
    if (config['qualification_sha256'] != offline._hash(qualification_path)
            or offline._hash(Path(offline_directory) / 'qualification.json') != config['qualification_sha256']):
        raise ValueError('Offline evidence and supplied teacher qualification differ')
    _, episodes, old_duplicates = offline._load_pool(offline_directory)
    if verify_native:
        for identifier, episode in episodes.items():
            folder = Path(offline_directory) / 'episodes' / identifier
            receipt = episode['receipt']
            trace = folder / 'source-trace.jsonl'
            # The existing exporter has already independently replayed every
            # source; an explicitly requested repeat compares the same full
            # trace and all recomputed proofs, not just selected root labels.
            measured, proofs = offline._replay_trace(trace, receipt['binding']['job'], offline._load(folder / 'source-attempt.json'))
            if not measured['accepted'] or proofs != episode['proofs']:
                raise ValueError('Repeated offline native replay differs')
    rows, contexts = _offline_records(episodes)
    catalogs = [('behavior', 'behavior_scenarios', scenario_directory)]
    if recorded_directory is not None:
        catalogs.append(('recorded', 'recorded_behavior_scenarios', recorded_directory))
    for catalog, module_name, directory in catalogs:
        scenarios = importlib.import_module(module_name)
        scenarios.validate_scenarios(directory, qualification_path, replay=verify_native)
        for original in scenarios.scenario_records(directory):
            meta = original['_meta']; flags = _scenario_roles(original)
            if not any(flags.values()):
                continue
            provenance = {'type': 'verified_recorded_learner_recovery' if catalog == 'recorded' else 'verified_behavior_recovery',
                'source_catalog': catalog, 'root_id': meta['id'], 'source_receipt': meta['source_receipt'],
                'receipt_sha256': meta['source_receipt_sha256'], 'original_record_sha256': fingerprint(original)}
            row = _wrap(original, provenance, flags, copy.deepcopy(meta.get('behavior_evidence', {})))
            rows.append(row); contexts[row['_meta']['id']] = {'scenario_root': original,
                'scenario_directory': str(directory), 'scenario_module': module_name, 'source_catalog': catalog}
    pools = {'train': [], 'development': []}; by_request = {}; request_splits = defaultdict(set); request_labels = defaultdict(set)
    split_by_seed = {}; duplicates = Counter(old_duplicates)
    for row in rows:
        meta = row['_meta']; split = meta['split']; digest = meta['request_sha256']
        if split not in pools or split_by_seed.setdefault(meta['seed'], split) != split:
            raise ValueError('Behavior source seed family overlaps train/development')
        if meta['group_id'] != f'level-{meta["level"]}-seed-{meta["seed"]}' or offline._request(row) != offline._request(body(row['state'])):
            raise ValueError('Behavior source family/original input/options differ')
        request_splits[digest].add(split); request_labels[digest].add(row['questions']['move']['label'])
        by_request.setdefault(digest, []).append(row)
    blocked = {digest for digest in by_request if len(request_splits[digest]) > 1 or len(request_labels[digest]) > 1}
    duplicates['balanced_cross_partition_or_ambiguous_inputs_excluded'] = len(blocked)
    for digest, candidates in by_request.items():
        if digest in blocked:
            continue
        # Keep every proof variant in the pool: a role solver may select the
        # stronger branch proof, but the request can appear only once in data.
        for row in candidates:
            pools[row['_meta']['split']].append(row)
            contexts[row['_meta']['id']]['blocked_request_sha256'] = blocked
        duplicates['balanced_duplicate_provenances'] += len(candidates) - 1
    return pools, contexts, episodes, dict(duplicates)


def report_pool(offline_directory, scenario_directory, qualification_path, recorded_directory=None):
    pools, _, _, duplicates = load_pools(offline_directory, scenario_directory, qualification_path, recorded_directory=recorded_directory)
    return {'dataset': PREFIX, 'duplicates': duplicates, 'roles': {split: {
        role: len({r['_meta']['request_sha256'] for r in rows if r['_meta']['behavior'][role]}) for role in ROLES}
        for split, rows in pools.items()}, 'families': {split: len({r['_meta']['group_id'] for r in rows}) for split, rows in pools.items()},
        'origins': {split: dict(Counter(r['_meta']['origin'] for r in rows)) for split, rows in pools.items()},
        'recorded_disagreements': {split: len({r['_meta']['request_sha256'] for r in rows
                                  if r['_meta']['learner_disagreement_measured']}) for split, rows in pools.items()},
        'claims': CLAIMS}


class _Flow:
    """Integral capacitated matching; no optional optimization dependency."""
    def __init__(self, count):
        self.edges = [[] for _ in range(count)]

    def add(self, start, end, capacity):
        forward = [end, len(self.edges[end]), capacity]
        backward = [start, len(self.edges[start]), 0]
        self.edges[start].append(forward); self.edges[end].append(backward)
        return forward

    def solve(self, start, end):
        total = 0
        while True:
            levels = [-1] * len(self.edges); levels[start] = 0; queue = deque([start])
            while queue:
                node = queue.popleft()
                for target, _, capacity in self.edges[node]:
                    if capacity and levels[target] < 0:
                        levels[target] = levels[node] + 1; queue.append(target)
            if levels[end] < 0:
                return total
            visited = [0] * len(self.edges)
            def send(node, maximum):
                if node == end:
                    return maximum
                while visited[node] < len(self.edges[node]):
                    edge = self.edges[node][visited[node]]; target, reverse, capacity = edge
                    if capacity and levels[target] == levels[node] + 1:
                        amount = send(target, min(maximum, capacity))
                        if amount:
                            edge[2] -= amount; self.edges[target][reverse][2] += amount
                            return amount
                    visited[node] += 1
                return 0
            while (amount := send(start, 10 ** 9)):
                total += amount


def select_roots(records, quotas, family_cap=FAMILY_CAP, seed=107, excluded=()):
    """Exact role/request/family matching, preserving each role's real proof.

    A request has capacity one. Different receipts can prove different roles;
    labels, inputs and a single source family must agree. For a deterministic
    opening shared by families we first choose the family with the strongest
    grounded evidence, and keep its per-role original receipt variants. The
    solver may reassign an overlapping root, never manufacture a proof.
    """
    if any(role not in ROLES or isinstance(n, bool) or not isinstance(n, int) or n < 0 for role, n in quotas.items()):
        raise ValueError('Use nonnegative integer quotas for known roles')
    rng = random.Random(seed); used = set(excluded); grouped = defaultdict(list)
    for row in records:
        if row['_meta']['request_sha256'] not in used:
            grouped[row['_meta']['request_sha256']].append(row)
    roles = [role for role, n in quotas.items() if n]
    if not roles:
        return []
    available_counts = {role: sum(any(r['_meta']['behavior'].get(role) for r in variants)
                                  for variants in grouped.values()) for role in roles}
    proof_by_role = {}; family_by_request = {}
    for digest in sorted(grouped):
        variants = grouped[digest]
        if len({r['_meta']['split'] for r in variants}) > 1 or len({r['questions']['move']['label'] for r in variants}) > 1:
            raise ValueError('Matching pool contains ambiguous or cross-partition inputs')
        by_family = defaultdict(list)
        for row in variants:
            by_family[row['_meta']['group_id']].append(row)
        family_names = sorted(by_family); rng.shuffle(family_names)
        def strength(name):
            flags = {role for r in by_family[name] for role in roles if r['_meta']['behavior'].get(role)}
            return (sum(quotas[role] / max(available_counts[role], 1) for role in flags if role != 'broad_exposure'),
                    any(r['_meta'].get('learner_disagreement_measured', False) for r in by_family[name]), len(flags))
        family = max(family_names, key=strength)
        family_by_request[digest] = family
        for role in roles:
            options = [r for r in by_family[family] if r['_meta']['behavior'].get(role)]
            if options:
                proof_by_role[role, digest] = min(options, key=lambda r: (
                    not r['_meta'].get('learner_disagreement_measured', False),
                    r['_meta'].get('origin') not in ('verified_behavior_recovery', 'verified_recorded_learner_recovery'), r['_meta']['id']))
    requests = sorted({digest for _, digest in proof_by_role}); rng.shuffle(requests)
    # Actual archived disagreements are scarce grounded evidence. Prefer them
    # within the unchanged behavior quotas; never count a prediction at a
    # nearby teacher state or reserve a fabricated disagreement bucket.
    requests.sort(key=lambda digest: not any(proof_by_role[role, digest]['_meta'].get('learner_disagreement_measured', False)
                                            for role in roles if (role, digest) in proof_by_role))
    families = sorted({family_by_request[digest] for digest in requests}); rng.shuffle(families)
    # Reserve a quarter of each family budget for actual sequence windows.
    root_cap = math.floor(.75 * family_cap)
    source = 0; role_nodes = {role: i + 1 for i, role in enumerate(roles)}
    offset = 1 + len(roles); request_nodes = {digest: offset + i for i, digest in enumerate(requests)}
    offset += len(requests); family_nodes = {family: offset + i for i, family in enumerate(families)}
    sink = offset + len(families); flow = _Flow(sink + 1); assignments = []
    for role in roles:
        flow.add(source, role_nodes[role], quotas[role])
        for digest in requests:
            if (role, digest) in proof_by_role:
                edge = flow.add(role_nodes[role], request_nodes[digest], 1)
                assignments.append((role, digest, edge))
    for digest in requests:
        flow.add(request_nodes[digest], family_nodes[family_by_request[digest]], 1)
    for family in families:
        flow.add(family_nodes[family], sink, root_cap)
    matched = flow.solve(source, sink)
    if matched != sum(quotas.values()):
        assigned = Counter(role for role, _, edge in assignments if edge[2] == 0)
        shortage = {role: quotas[role] - assigned[role] for role in roles if quotas[role] > assigned[role]}
        raise ValueError('Insufficient distinct proved behavioral roots/family capacity: ' + json.dumps(shortage, sort_keys=True))
    chosen = []
    for role, digest, edge in assignments:
        if edge[2] == 0:
            selected = copy.deepcopy(proof_by_role[role, digest]); selected['_meta']['primary_role'] = role
            chosen.append(selected)
    return chosen


def _escape_parent(root):
    flags = root['_meta']['behavior']; evidence = root['_meta'].get('behavior_evidence', {})
    return bool(any(flags.get(role, False) for role in ('immediate_evasion', 'actual_expiry_evasion', 'anticipatory_escape'))
                or evidence.get('escape', False))


def _post_kind(root):
    if _escape_parent(root):
        return 'post_followthrough_with_progress'
    if root['_meta']['behavior'].get('retreat_to_food', False):
        return 'post_recovery_progress'
    return 'post_routing_progress'


def _offline_windows(root, context):
    episode = context['episode']; proof = context['proof']; proofs = context['proofs']; job = episode['receipt']['binding']['job']
    index = proof['source_index']; sequence = _sequence(proofs, index, proof['life_index'])
    first_food = next((p['source_index'] for p in sequence if p['diagnostics']['pellets_after'] < p['diagnostics']['pellets_before']), None)
    # Spread AFTER escape through real progress, then optional lead-ins. The
    # first post frame is at least2decisions later, not just a token next-step.
    post = [2, 4, 8, 16, 24]
    if first_food is not None:
        post = [max(2, first_food - index + 1), max(4, first_food - index + 4)] + post
    result = []; seen = set()
    for offset in post + [-1, -4, -8]:
        target = proofs.get(index + offset)
        if (offset in seen or target is None or target['life_index'] != proof['life_index']
                or len(target['request']['state']['legal_moves']) < 2 or abs(offset) > WINDOW_LIMIT
                or target['request_sha256'] in episode['excluded_request_sha256']
                or target['request_sha256'] in context.get('blocked_request_sha256', ())):
            continue
        if any(i not in proofs or proofs[i]['life_index'] != proof['life_index']
               for i in range(min(index, index + offset), max(index, index + offset) + 1)):
            continue
        progress = offset > 0 and any(p['diagnostics']['pellets_after'] < p['diagnostics']['pellets_before']
                                     for p in _sequence(proofs, index, proof['life_index'], offset - 1))
        kind = _post_kind(root) if progress else 'post_followthrough' if offset > 0 else 'lead_in'
        original = offline._record(target, episode['receipt'], job['id'], job['split'])
        behavior, evidence = trajectory_behavior(target, proofs)
        provenance = {'type': 'verified_offline_trajectory', 'episode': job['id'], 'source_index': target['source_index'],
            'source_receipt': f'episodes/{job["id"]}/receipt.json', 'receipt_sha256': episode['receipt_sha256'],
            'original_record_sha256': fingerprint(original)}
        row = _wrap(original, provenance, behavior, evidence); row['_meta'].update(
            {'class': 'informative', 'parent_targeted_id': root['_meta']['id'], 'window_kind': kind, 'window_decision_offset': offset})
        result.append(row); seen.add(offset)
    return result


def windows_for(root, context):
    if 'episode' in context:
        return _offline_windows(root, context)
    scenarios = importlib.import_module(context.get('scenario_module', 'behavior_scenarios'))
    result = []
    for original in scenarios.scenario_windows(context['scenario_directory'], context['scenario_root']['_meta']['id']):
        meta = original['_meta']; kind = meta['window_kind']; offset = meta['decision_offset']
        if (abs(offset) > WINDOW_LIMIT or offset < 2 or meta['life_index'] != root['_meta']['life_index']
                or meta['request_sha256'] in context.get('blocked_request_sha256', ())):
            continue
        if kind == 'post_followthrough_with_progress':
            kind = _post_kind(root)
        provenance = {'type': root['_meta']['origin'], 'source_catalog': context.get('source_catalog', 'behavior'),
            'root_id': context['scenario_root']['_meta']['id'],
            'source_receipt': meta['source_receipt'], 'receipt_sha256': meta['source_receipt_sha256'],
            'original_record_sha256': fingerprint(original), 'window_source_index': meta['source_index'],
            'decision_offset': offset}
        row = _wrap(original, provenance, _scenario_roles(original), copy.deepcopy(meta.get('behavior_evidence', {})))
        row['_meta'].update({'class': 'informative', 'parent_targeted_id': root['_meta']['id'],
                            'window_kind': kind, 'window_decision_offset': offset})
        result.append(row)
    return result


def select_windows(roots, contexts, count, minimum_post, family_cap=FAMILY_CAP, excluded=(), seed=113):
    if minimum_post < 0 or count < minimum_post:
        raise ValueError('Post-progress minimum must fit the full window budget')
    rng = random.Random(seed); ordered = list(roots); rng.shuffle(ordered)
    used = set(excluded) | {r['_meta']['request_sha256'] for r in roots}
    families = Counter(r['_meta']['group_id'] for r in roots); per_parent = Counter(); selected = []
    queues = {r['_meta']['id']: windows_for(r, contexts[r['_meta']['id']]) for r in ordered}
    for require_post, target in ((True, minimum_post), (False, count)):
        while len(selected) < target:
            changed = False
            for root in ordered:
                parent = root['_meta']['id']; queue = queues[parent]
                eligible = [r for r in queue if r['_meta']['request_sha256'] not in used
                    and families[r['_meta']['group_id']] < family_cap and per_parent[parent] < MAX_WINDOWS_PER_ROOT
                    and (not require_post or (_escape_parent(root) and r['_meta']['window_kind'] == 'post_followthrough_with_progress'))]
                if not eligible:
                    continue
                # Reserve POST evidence first; residual also favors actual
                # post-progress over lead-ins without inventing transitions.
                row = min(eligible, key=lambda r: (r['_meta']['window_kind'] != 'post_followthrough_with_progress',
                    r['_meta']['window_kind'] == 'lead_in', per_parent[parent]))
                selected.append(row); used.add(row['_meta']['request_sha256']); families[row['_meta']['group_id']] += 1
                per_parent[parent] += 1; queue.remove(row); changed = True
                if len(selected) == target:
                    break
            if not changed:
                raise ValueError(f'Insufficient actual same-life {"post-progress threat-escape" if require_post else "informative"} windows: {len(selected)}/{target}')
    return selected


def _replay_rows(offline_directory, chosen):
    old, binding = offline.load_old_expert_records()
    selected = offline._select_replay(old, chosen, TARGETS['train']['replay'], FAMILY_CAP)
    result = []
    for row in selected:
        value = copy.deepcopy(row); value['_meta']['id'] = value['_meta']['id'].replace(offline.PREFIX, PREFIX, 1)
        value['questions']['move']['src'] = PREFIX
        value['_meta'].update({'primary_role': 'old_expert_replay', 'behavior': {}, 'behavior_evidence': {},
                              'provenance': {'type': 'canonical_v2_expert_train', 'source_id': row['_meta']['source_id']}})
        result.append(value)
    return result, binding


def export_evidence(offline_directory, scenario_directory, filename, recorded_directory=None):
    filename = Path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='balanced-evidence-') as temporary:
        old_zip = Path(temporary) / 'offline.zip'; offline.export_evidence(offline_directory, old_zip)
        catalogs = [('behavior', 'behavior_scenarios', scenario_directory)]
        if recorded_directory is not None:
            catalogs.append(('recorded', 'recorded_behavior_scenarios', recorded_directory))
        temporary_zip = filename.with_name(filename.name + '.tmp')
        with zipfile.ZipFile(old_zip) as source, zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in source.namelist():
                archive.writestr('offline/' + name, source.read(name))
            for catalog, module_name, directory in catalogs:
                scenarios = importlib.import_module(module_name)
                for name in scenarios.completed_files(directory):
                    archive.write(offline._safe_path(directory, name), catalog + '/' + name)
        temporary_zip.replace(filename)
    with zipfile.ZipFile(filename) as archive:
        hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    return {'filename': filename.name, 'sha256': offline._hash(filename), 'files': hashes}


def _coverage(rows):
    roots = {r['_meta']['id']: r for r in rows if r['_meta']['class'] == 'targeted'}
    return {'primary_roles': dict(Counter(r['_meta']['primary_role'] for r in rows if r['_meta']['class'] == 'targeted')),
        'behavior_overlaps': dict(Counter(k for r in rows if r['_meta']['class'] == 'targeted'
                                         for k, yes in r['_meta']['behavior'].items() if yes)),
        'window_kinds': dict(Counter(r['_meta']['window_kind'] for r in rows if r['_meta']['class'] == 'informative')),
        'critical_actual_expiry_joint': sum(r['_meta']['class'] == 'targeted' and r['_meta']['behavior'].get('immediate_evasion', False)
                                          and r['_meta']['behavior'].get('actual_expiry_evasion', False) for r in rows),
        'targeted_origins': dict(Counter(r['_meta']['origin'] for r in rows if r['_meta']['class'] == 'targeted')),
        'recorded_learner_disagreements': sum(r['_meta']['class'] == 'targeted' and r['_meta']['learner_disagreement_measured'] for r in rows),
        'post_escape_progress_windows': sum(r['_meta']['class'] == 'informative'
            and r['_meta']['window_kind'] == 'post_followthrough_with_progress'
            and r['_meta']['parent_targeted_id'] in roots and _escape_parent(roots[r['_meta']['parent_targeted_id']]) for r in rows),
        'families': dict(Counter(r['_meta']['group_id'] for r in rows))}


def build_dataset(offline_directory, scenario_directory, output_directory, qualification_path, plan=DEFAULT_PLAN, recorded_directory=None):
    plan = validate_plan(plan)
    pools, contexts, episodes, duplicates = load_pools(offline_directory, scenario_directory, qualification_path, recorded_directory=recorded_directory)
    selected = {}; seen = set()
    for i, split in enumerate(('development', 'train')):
        roots = select_roots(pools[split], plan[split], seed=107 + i, excluded=seen)
        windows = select_windows(roots, contexts, TARGETS[split]['informative'], POST_MINIMUMS[split], excluded=seen, seed=113 + i)
        selected[split] = roots + windows; seen.update(r['_meta']['request_sha256'] for r in selected[split])
    replay, old_binding = _replay_rows(offline_directory, selected['train'] + selected['development'])
    selected['train'] += replay
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    files = {}; coverage = {}; counts = {}
    for split, rows in selected.items():
        name = f'{PREFIX}-{split}.jsonl'; offline._write_rows(output / name, rows); files[name] = offline._hash(output / name)
        coverage[split] = _coverage(rows); counts[split] = dict(Counter(r['_meta']['class'] for r in rows))
    bundle = export_evidence(offline_directory, scenario_directory, output / f'{PREFIX}-evidence.zip', recorded_directory)
    manifest = {'dataset': PREFIX, 'schema': 'native-measured-behaviors-and-winning-teacher-v1',
        'source_sha256': source_hashes(), 'exporter_sha256': offline._hash(Path(__file__)), 'protocol': offline.SPEC,
        'qualification_sha256': offline._hash(qualification_path), 'recipe': RECIPE, 'targets': TARGETS,
        'role_plan': plan, 'critical_minimums': CRITICAL_MINIMUMS, 'post_minimums': POST_MINIMUMS,
        'broad_max_fraction': BROAD_MAX_FRACTION, 'family_cap': FAMILY_CAP, 'counts': counts,
        'coverage': coverage, 'files': files, 'evidence': bundle, 'old_expert_source': old_binding,
        'duplicates': duplicates, 'claims': CLAIMS, 'expected_training_requests': 6096, 'expected_optimizer_steps': 762,
        'recorded_catalog_included': recorded_directory is not None,
        'initialization': {'parent_checkpoint': 'kev-4b-pacman-native-v3', 'optimizer': 'fresh'},
        'train_seeds': sorted({r['_meta']['seed'] for r in selected['train'] if r['_meta']['class'] != 'replay'}),
        'development_seeds': sorted({r['_meta']['seed'] for r in selected['development']}),
        'native_teacher_evidence_verified': True, 'gpu_training_performed': False}
    offline._save(output / f'{PREFIX}-manifest.json', manifest)
    validate_dataset(output, qualification_path)
    return manifest


def _extract_verified_bundle(output, manifest, destination):
    bundle = manifest['evidence']; path = offline._safe_path(output, bundle['filename'])
    if offline._hash(path) != bundle['sha256']:
        raise ValueError('Balanced evidence bundle checksum differs')
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != set(bundle['files']) or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Balanced evidence members differ/duplicate')
        for name, digest in bundle['files'].items():
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Balanced source evidence entry checksum differs')
            target = offline._safe_path(destination, name); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)


def validate_dataset(output_directory, qualification_path, verify_native=False):
    output = Path(output_directory); manifest = offline._load(output / f'{PREFIX}-manifest.json')
    plan = validate_plan(manifest['role_plan'])
    if (manifest['dataset'] != PREFIX or manifest['schema'] != 'native-measured-behaviors-and-winning-teacher-v1'
            or manifest['source_sha256'] != source_hashes() or manifest['exporter_sha256'] != offline._hash(Path(__file__))
            or manifest['qualification_sha256'] != offline._hash(qualification_path) or manifest['protocol'] != offline.SPEC
            or manifest['recipe'] != RECIPE or manifest['targets'] != TARGETS or manifest['family_cap'] != FAMILY_CAP
            or manifest['critical_minimums'] != CRITICAL_MINIMUMS or manifest['post_minimums'] != POST_MINIMUMS
            or manifest['broad_max_fraction'] != BROAD_MAX_FRACTION or manifest['claims'] != CLAIMS
            or manifest['expected_training_requests'] != 6096 or manifest['expected_optimizer_steps'] != 762
            or manifest['initialization'] != {'parent_checkpoint': 'kev-4b-pacman-native-v3', 'optimizer': 'fresh'}
            or not manifest['native_teacher_evidence_verified'] or manifest['gpu_training_performed']):
        raise ValueError('Balanced dataset/configuration/recipe differs')
    if set(manifest['files']) != {f'{PREFIX}-{s}.jsonl' for s in TARGETS}:
        raise ValueError('Unexpected balanced training/development files')
    for name, digest in manifest['files'].items():
        if offline._hash(output / name) != digest:
            raise ValueError('Balanced data checksum differs')
    with tempfile.TemporaryDirectory(prefix='balanced-validation-') as temporary:
        evidence = Path(temporary); _extract_verified_bundle(output, manifest, evidence)
        if bool((evidence / 'recorded').exists()) != manifest['recorded_catalog_included']:
            raise ValueError('Recorded learner catalog manifest/evidence differs')
        pools, contexts, _, _ = load_pools(evidence / 'offline', evidence / 'behavior', qualification_path, verify_native,
            recorded_directory=evidence / 'recorded' if manifest['recorded_catalog_included'] else None)
        source_rows = {r['_meta']['id']: r for rows in pools.values() for r in rows}
        old, binding = offline.load_old_expert_records()
        if binding != manifest['old_expert_source']:
            raise ValueError('Old replay provenance differs')
        old_lookup = {r['_meta']['source_id']: r for r in old}
        seen = set(); ids = set(); seed_partition = {}; all_rows = {}
        for split, budget in TARGETS.items():
            rows = offline._rows(output / f'{PREFIX}-{split}.jsonl'); all_rows[split] = rows
            counts = dict(Counter(r['_meta']['class'] for r in rows))
            if counts != {k: v for k, v in budget.items() if v} or counts != manifest['counts'][split]:
                raise ValueError('Balanced full class budgets differ')
            roots = {r['_meta']['id']: r for r in rows if r['_meta']['class'] == 'targeted'}; parent_counts = Counter()
            for row in rows:
                meta = row['_meta']; digest = offline.request_fingerprint(row)
                if (set(row) != {'state', 'questions', '_meta'} or meta['split'] != split or digest in seen or meta['id'] in ids
                        or meta['request_sha256'] != digest or meta['state_sha256'] != fingerprint(row['state'])):
                    raise ValueError('Balanced input/identity/dedup boundary differs')
                seen.add(digest); ids.add(meta['id'])
                if meta['class'] == 'replay':
                    original = copy.deepcopy(old_lookup[meta['source_id']]); original['_meta']['id'] = original['_meta']['id'].replace(offline.PREFIX, PREFIX, 1)
                    original['questions']['move']['src'] = PREFIX
                    original['_meta'].update({'primary_role': 'old_expert_replay', 'behavior': {}, 'behavior_evidence': {},
                        'provenance': {'type': 'canonical_v2_expert_train', 'source_id': meta['source_id']}})
                    if split != 'train' or row != original:
                        raise ValueError('Canonical old training input/label/provenance differs')
                    continue
                if seed_partition.setdefault(meta['seed'], split) != split:
                    raise ValueError('Balanced source seed group overlaps partitions')
                if meta['class'] == 'targeted':
                    original = copy.deepcopy(source_rows[meta['id']]); original['_meta']['primary_role'] = meta['primary_role']
                    if row != original or not meta['behavior'].get(meta['primary_role']):
                        raise ValueError('Primary behavior/input/label differs from proved source')
                elif meta['class'] == 'informative':
                    parent = roots.get(meta['parent_targeted_id'])
                    if parent is None or row not in windows_for(parent, contexts[parent['_meta']['id']]):
                        raise ValueError('Window is not a selected-root bounded actual same-life teacher frame')
                    parent_counts[parent['_meta']['id']] += 1
            measured = _coverage(rows)
            if (measured != manifest['coverage'][split] or measured['primary_roles'] != {role: n for role, n in plan[split].items() if n}
                    or measured['behavior_overlaps'].get('immediate_evasion', 0) < CRITICAL_MINIMUMS[split]
                    or measured['post_escape_progress_windows'] < POST_MINIMUMS[split]
                    or any(n > FAMILY_CAP for n in measured['families'].values())
                    or any(n > MAX_WINDOWS_PER_ROOT for n in parent_counts.values())):
                raise ValueError('Balanced behavior floors, role quotas, post windows or family caps differ')
        for split in TARGETS:
            seeds = sorted({r['_meta']['seed'] for r in all_rows[split] if r['_meta']['class'] != 'replay'})
            if seeds != manifest[split + '_seeds']:
                raise ValueError('Balanced seed partition manifest differs')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('report', 'build', 'validate'))
    parser.add_argument('--offline', type=Path); parser.add_argument('--scenarios', type=Path)
    parser.add_argument('--recorded', type=Path, help='Optional checkpoint-bound archived learner error/recovery catalog')
    parser.add_argument('--qualification', type=Path, required=True); parser.add_argument('--out', type=Path)
    parser.add_argument('--plan', type=Path); parser.add_argument('--replay', action='store_true'); args = parser.parse_args()
    if args.command == 'report':
        result = report_pool(args.offline, args.scenarios, args.qualification, args.recorded)
    elif args.command == 'build':
        result = build_dataset(args.offline, args.scenarios, args.out, args.qualification,
                               offline._load(args.plan) if args.plan else DEFAULT_PLAN, args.recorded)
    else:
        result = validate_dataset(args.out, args.qualification, args.replay)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
