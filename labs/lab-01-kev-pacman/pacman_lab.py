"""Structured player decisions over the pinned upstream classic Pac-Man engine.

The browser and CPU evaluation execute the same JavaScript, not Python physics.
"""
import base64
from collections import deque
import copy
import hashlib
import html
import json
from pathlib import Path
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
from urllib.request import urlopen
import zipfile

from api_client import call, distribution

ROOT = Path(__file__).resolve().parent
PACMAN_REVISION = '7407174c1d6a38be8cd230577489e39e0873145b'
POLICY = ('Control Pac-Man in classic arcade Pac-Man. Collect pellets and fruit, use power '
          'pellets to eat frightened ghosts, and avoid dangerous ghosts. All four ghosts '
          'follow the arcade engine. Prefer uncollected pellets reachable by safe maze '
          'paths; avoid repeatedly revisiting the same tiles or reversing without progress. '
          'Use the recent positions, ghost modes and legal moves. Select one supplied direction.')
PLANNER_POLICY = (POLICY + ' In chase mode, Blinky targets your current tile; Pinky aims four tiles '
                  'ahead; Inky reflects a two-tile-ahead point around Blinky; Clyde chases from '
                  'far away and retreats to his corner when close. Upward targeting includes '
                  'the classic leftward offset. In scatter mode ghosts target their own corners. '
                  'Normal ghost targeting is deterministic; frightened turns can vary. '
                  'Ghosts leave home gradually using pellet counters and simulation time. '
                  'Anticipate future collisions and power-pellet opportunities, not just current proximity.')
DIRECTIONS = {'up': (-1, 0), 'left': (0, -1), 'down': (1, 0), 'right': (0, 1)}
DATA_PREFIX = 'pacman-arcade'


def upstream_files():
    """Verify the archive and every unmodified upstream file before execution."""
    folder = ROOT / 'vendor/arcade-pacman'
    meta = json.loads((folder / 'source.json').read_text())
    content = (folder / 'source.zip').read_bytes()
    expected = meta.get('archive_sha256') or meta['sha256']
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Pac-Man source archive checksum mismatch')
    with zipfile.ZipFile(folder / 'source.zip') as archive:
        files = {name: archive.read(name) for name in archive.namelist() if not name.endswith('/')}
    for name, expected in meta['files'].items():
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError(f'Pac-Man source checksum mismatch: {name}')
    return files


def engine_script():
    source = upstream_files()['pacman.js'].decode()
    # Complete native rewind without changing forward physics. Expose current
    # clocks for v2 observations; seeds and private RNG never enter the request.
    start, stop = source.index('var ghostReleaser = (function(){'), source.index('var elroyTimer = (function(){')
    section = source[start:stop]
    begin = section.index('    var save = function(t) {')
    end = section.index('\n    return {', begin)
    section = section[:begin] + '''    var savedMode = {};
    var save = function(t) {
        savedMode[t] = mode;
        savedFramesSinceLastDot[t] = framesSinceLastDot;
        savedGlobalCount[t] = globalCount;
        savedGhostCounts[t] = Object.assign({}, ghostCounts);
    };
    var load = function(t) {
        mode = savedMode[t];
        framesSinceLastDot = savedFramesSinceLastDot[t];
        globalCount = savedGlobalCount[t];
        ghostCounts = Object.assign({}, savedGhostCounts[t]);
    };
''' + section[end:]
    section = section.replace('    return {', '''    return {
        currentState: function() { return {mode: mode === MODE_GLOBAL ? 'global' : 'personal',
            frames_since_last_dot: framesSinceLastDot, global_count: globalCount,
            personal_counts: Object.assign({}, ghostCounts)}; },''', 1)
    source = source[:start] + section + source[stop:]
    for anchor, getter in (
        ('var ghostCommander = (function()', "currentState: function() { return {phase_clock_frames: frame}; },"),
        ('var energizer = (function()', "currentState: function() { return {remaining_frames: active ? Math.max(0,getDuration()-count) : 0, ghost_points: points, eating_pause_frames: pointsFramesLeft}; },"),
        ('var elroyTimer = (function()', "currentState: function() { return {wait_for_clyde: waitForClyde}; },"),
    ):
        begin = source.index(anchor)
        at = source.index('    return {', begin) + len('    return {')
        source = source[:at] + '\n        ' + getter + source[at:]
    end = source.rfind('})();')
    if end < 0:
        raise ValueError('Pinned arcade closure was not found')
    return source[:end] + (ROOT / 'games/arcade-engine.js').read_text() + '\n' + source[end:]


def ensure_node(workspace=ROOT):
    """Use Node already installed, or one checksum-pinned official Linux binary."""
    node = shutil.which('node')
    if node and int(subprocess.check_output([node, '--version'], text=True).strip()[1:].split('.')[0]) >= 18:
        return node
    target = Path(workspace) / '.tools/node-v22.17.0/bin/node'
    if target.is_file():
        return str(target)
    import platform
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise RuntimeError('Install Node.js 18 or newer to execute the shared arcade engine.')
    url = 'https://nodejs.org/dist/v22.17.0/node-v22.17.0-linux-x64.tar.xz'
    print('Installing pinned Node.js 22.17.0 for arcade replay (separate from Torch).', flush=True)
    with urlopen(url, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != '325c0f1261e0c61bcae369a1274028e9cfb7ab7949c05512c5b1e630f7e80e12':
        raise ValueError('Node download checksum mismatch')
    import io
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:xz') as archive:
        for name in ('bin/node', 'LICENSE'):
            member = archive.getmember('node-v22.17.0-linux-x64/' + name)
            destination = target.parent.parent / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
    target.chmod(0o755)
    return str(target)


class ArcadeEngine:
    """A small JSON-lines bridge to the actual browser game's simulation."""
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='kev-arcade-')
        script = Path(self.directory.name) / 'pacman.js'
        script.write_text(engine_script())
        self.process = subprocess.Popen([ensure_node(), str(ROOT / 'games/arcade-worker.js'), str(script)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)

    def request(self, command, **args):
        self.process.stdin.write(json.dumps({'command': command, **args}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Arcade worker stopped: ' + self.process.stderr.read())
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result['result']

    def reset(self, seed=7):
        return self.request('reset', options={'seed': seed})

    def step(self, direction):
        return self.request('step', direction=direction)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.communicate(timeout=10)
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def position(actor):
    return actor['row'], actor['column']


def destination(maze, pos, direction):
    dr, dc = DIRECTIONS[direction]
    return pos[0] + dr, (pos[1] + dc) % len(maze[0])


def legal(maze, pos):
    return [d for d in DIRECTIONS if 0 <= destination(maze, pos, d)[0] < len(maze)
            and maze[destination(maze, pos, d)[0]][destination(maze, pos, d)[1]] != '#']


def distances(maze, target, blocked=()):
    found, queue = {target: 0}, deque([target])
    while queue:
        tile = queue.popleft()
        for direction in legal(maze, tile):
            nxt = destination(maze, tile, direction)
            if nxt not in found and nxt not in blocked:
                found[nxt] = found[tile] + 1
                queue.append(nxt)
    return found


def initial_state():
    with ArcadeEngine() as engine:
        return engine.reset()


def body(state, instructions=PLANNER_POLICY):
    criteria = {d: f'Move {d} toward row {destination(state["maze"], position(state["player"]), d)[0]}, '
                   f'column {destination(state["maze"], position(state["player"]), d)[1]}.'
                for d in state['legal_moves']}
    return {'state': state, 'model': 'kev-latest', 'questions': {'move': {
        'type': 'choice', 'instructions': instructions, 'criteria': criteria}}}


def teacher(state):
    """A transparent maze heuristic, not human expertise or an optimal controller."""
    maze, pac = state['maze'], position(state['player'])
    danger = [position(g) for g in state['ghosts'] if g['mode'] in ('outside', 'leaving_home') and not g['frightened']]
    pellets = [(r, c) for r, row in enumerate(maze) for c, ch in enumerate(row) if ch in '.o']
    recent = [position(p) for p in state['recent_positions']]
    def rank(direction):
        nxt = destination(maze, pac, direction)
        routes = distances(maze, nxt, danger)
        nearest_dot = min((routes.get(dot, 999) for dot in pellets), default=0)
        separation = min((distances(maze, nxt).get(g, 999) for g in danger), default=999)
        return (separation <= 1, separation <= 3, nearest_dot + 3 * recent.count(nxt), -min(separation, 8))
    return min(state['legal_moves'], key=rank)


def make_data(directory, counts=(64, 16, 16), seed=7):
    """Snapshots from valid engine trajectories; whole episodes stay in one split."""
    rng, fingerprints, manifest = random.Random(seed), set(), {'counts': {}, 'groups': {}, 'labels': {}}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    episode = 0
    with ArcadeEngine() as engine:
        for split, count in zip(('train', 'development', 'evaluation'), counts):
            rows, groups = [], []
            while len(rows) < count:
                episode += 1
                episode_seed = seed + episode * 1009
                state, replay = engine.reset(episode_seed), {'seed': episode_seed, 'actions': []}
                group = f'arcade-episode-{episode:04d}'
                for turn in range(384):
                    chosen = teacher(state)
                    fingerprint = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
                    if turn >= 12 and len(state['legal_moves']) >= 2 and fingerprint not in fingerprints and turn % 12 == 0:
                        request = body(state, instructions=POLICY)
                        request.pop('model')
                        request['questions']['move'].update(label=chosen, src='pacman_teacher')
                        request['_meta'] = {'id': f'{split}-board-{len(rows):03d}', 'group_id': group,
                                            'source': 'classic_pacman_engine', 'variant': 'clean',
                                            'label_source': 'maze safety/pellet/loop heuristic', 'replay': copy.deepcopy(replay)}
                        rows.append(request); fingerprints.add(fingerprint)
                        if group not in groups:
                            groups.append(group)
                        if len(rows) == count:
                            break
                    # Exploration creates broader positions; the label always uses the stated heuristic.
                    move = rng.choice(state['legal_moves']) if rng.random() < .2 else chosen
                    result = engine.step(move)
                    state, replay = result['state'], result['replay']
                    if result['outcome']:
                        break
            (directory / f'{DATA_PREFIX}-{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            manifest['counts'][split], manifest['groups'][split] = count, groups
            manifest['labels'][split] = {d: sum(row['questions']['move']['label'] == d for row in rows) for d in DIRECTIONS}
    manifest.update(seed=seed, game_commit=PACMAN_REVISION, label_policy=POLICY,
                    interpretation='Valid seeded arcade trajectories on one maze; episode-disjoint imitation labels, not human demonstrations or a win-rate benchmark.')
    (directory / f'{DATA_PREFIX}-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def inference_request(record):
    return body(record['state'])


def evaluate(path, predict=None):
    predict = predict or (lambda request: call('/v1/systemone', request)[0])
    rows = []
    with ArcadeEngine() as engine:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            replayed = engine.request('replay', replay=record['_meta']['replay'])
            if replayed != record['state']:
                raise ValueError('Evaluation snapshot differs from the pinned engine replay')
            started = time.perf_counter()
            response = predict(inference_request(record))
            answer = response['answers']['move']
            distribution(answer, record['questions']['move']['criteria'])
            result = engine.step(answer['choice'])
            row = {'id': record['_meta']['id'], 'gold': record['questions']['move']['label'],
                         'prediction': answer['choice'], 'probabilities': answer['probabilities'],
                         'caught_next_turn': result['outcome'] == 'life_lost',
                         'http_ms': (time.perf_counter() - started) * 1000}
            if 'teacher' in record['_meta']:
                candidates = record['_meta']['teacher']['candidates']
                best, selected = candidates[0], next(c for c in candidates if c['action'] == answer['choice'])
                row['teacher_tie_correct'] = (selected['survival'], selected['value']) == (best['survival'], best['value'])
                row['lower_search_survival'] = selected['survival'] < best['survival']
                row['search_value_regret'] = max(0, best['value']-selected['value']) if selected['survival'] == best['survival'] else None
            rows.append(row)
    if not rows:
        raise ValueError('No labelled decisions to evaluate')
    result = {'n': len(rows), 'accuracy': sum(r['gold'] == r['prediction'] for r in rows) / len(rows),
              'caught_next_turn': sum(r['caught_next_turn'] for r in rows), 'rows': rows, 'engine_revision': PACMAN_REVISION}
    planned = [r for r in rows if 'teacher_tie_correct' in r]
    if planned:
        regrets = [r['search_value_regret'] for r in planned if r['search_value_regret'] is not None]
        result.update(tie_aware_teacher_accuracy=sum(r['teacher_tie_correct'] for r in planned)/len(planned),
                      lower_search_survival_choices=sum(r['lower_search_survival'] for r in planned),
                      mean_same_survival_search_regret=sum(regrets)/len(regrets) if regrets else None)
    return result


def rollout(predict=None, max_turns=128, seed=7):
    predict = predict or (lambda request: call('/v1/systemone', request)[0])
    with ArcadeEngine() as engine:
        state = engine.reset(seed)
        pellets, frames, outcome = state['pellets_remaining'], [copy.deepcopy(state)], None
        for _ in range(max_turns):
            request = body(state)
            answer = predict(request)['answers']['move']
            distribution(answer, request['questions']['move']['criteria'])
            result = engine.step(answer['choice'])
            state, outcome = result['state'], result['outcome']
            frames.append(copy.deepcopy(state))
            if outcome:
                break
    return {'turns': state['turn'], 'dots_collected': pellets - state['pellets_remaining'],
            'score': state['score'], 'simulation_frames': state['simulation_frames'],
            'repeated_tiles': sum(position(s['player']) in [position(p['player']) for p in frames[max(0, i-12):i]] for i, s in enumerate(frames)),
            'outcome': outcome or 'turn_limit', 'frames': frames, 'engine_revision': PACMAN_REVISION, 'seed': seed}


class NotebookBridge:
    def __init__(self, trace_path, model_info):
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info

    def model(self):
        return self.model_info()

    def decide(self, state):
        if state.get('engine_revision') != PACMAN_REVISION or state.get('game') != 'classic-pacman':
            raise ValueError('Load the current classic Pac-Man board before requesting a decision.')
        identity = self.model()
        request = body(state)
        response, elapsed = call('/v1/systemone', request)
        distribution(response['answers']['move'], request['questions']['move']['criteria'])
        if self.model() != identity:
            raise RuntimeError('The checkpoint changed during this decision. Restart with the active adapter.')
        response['active_checkpoint'] = identity
        with self.trace_path.open('a') as handle:
            handle.write(json.dumps({'request': request, 'response': response, 'http_ms': elapsed, 'active_checkpoint': identity}) + '\n')
        return response


def notebook_game(action_version=1):
    """Embed upstream renderer/font/audio in a Colab callback-controlled board."""
    files = upstream_files()
    source = engine_script()
    # Resolve original assets inside srcdoc; no external game server or ROM needed.
    for name, content in files.items():
        if name.startswith('sounds/') and name.endswith('.mp3'):
            source = source.replace(name, 'data:audio/mpeg;base64,' + base64.b64encode(content).decode())
    end = source.rfind('})();')
    source = source[:end] + f'\nglobalThis.LAB_ACTION_VERSION = {int(action_version)};\n' + (ROOT / 'games/arcade-browser.js').read_text() + '\n' + source[end:]
    inner = files['index.html'].decode()
    inner = inner.replace('font/ARCADE_R.TTF', 'data:font/ttf;base64,' + base64.b64encode(files['font/ARCADE_R.TTF']).decode())
    inner = inner.replace('<script src="pacman.js"></script>', '<script>' + source.replace('</script', '<\\/script') + '</script>')
    # Icons are irrelevant to an embedded board, while renderer/font/audio are upstream.
    import re
    inner = re.sub(r'<link[^>]+href="icon/[^>]+>', '', inner)
    template = (ROOT / 'games/arcade-shell.html').read_text()
    return template.replace('ARCADE_SRCDOC', html.escape(inner, quote=True))
