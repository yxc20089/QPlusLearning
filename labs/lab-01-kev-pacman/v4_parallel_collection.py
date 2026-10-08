"""Collect independent frozen-v3 games through Kev's existing batching queue.

Each thread owns one native benchmark engine and makes its decisions in order.
Only the coordinator writes resumable receipts, candidates, indices and backups.
Do not run another mutating collection stage against the same directory at the
same time. This helper does not change the pinned game/teacher or v4 generator.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from pathlib import Path
import threading
import time

from gameplay_benchmark import benchmark_gameplay
from teacher_validation import fingerprint
from v3_data import select_roots
import v4_data as v4

MAX_WORKERS = 8


def _check_cancel(cancelled):
    if cancelled.is_set():
        if getattr(cancelled, 'identity_error', None):
            raise RuntimeError(cancelled.identity_error)
        raise InterruptedError('Parallel v3 collection cancelled; unfinished games will be retried')


def _identity(model_info, expected, cancelled):
    _check_cancel(cancelled)
    actual = copy.deepcopy(model_info())
    _check_cancel(cancelled)
    if actual != expected:
        cancelled.identity_error = 'Active v3 adapter changed during parallel collection'
        cancelled.set()
        raise RuntimeError(cancelled.identity_error)
    return actual


def _play_game(job, model_info, predict, expected, cancelled):
    """Worker writes only its isolated raw benchmark trace, never a receipt."""
    _check_cancel(cancelled)
    def checked_info():
        return _identity(model_info, expected, cancelled)
    def checked_predict(request):
        _check_cancel(cancelled)
        response = predict(request)
        _check_cancel(cancelled)
        if response.get('active_checkpoint', expected) != expected:
            cancelled.identity_error = 'Inference response uses a different v3 adapter'
            cancelled.set()
            raise RuntimeError(cancelled.identity_error)
        return response
    report = benchmark_gameplay(predict=checked_predict, model_info=checked_info,
        trace_dir=job['folder'], spec={**v4.SPEC, 'levels':[job['level']], 'seeds':[job['seed']]})
    checked_info()
    return report


def _check_receipt(directory, job, receipt, expected):
    if (receipt.get('id') != job['id'] or receipt.get('split') != job['split']
            or receipt.get('seed') != job['seed'] or receipt.get('level') != job['level']
            or receipt.get('checkpoint') != expected or receipt.get('kind') != 'learner'
            or not receipt.get('accepted') or len(receipt.get('files', {})) != 1):
        raise ValueError('Saved learner game uses different seeds, partition or adapter')
    trace_name, digest = next(iter(receipt['files'].items()))
    trace = Path(directory)/trace_name
    if trace.resolve().parent != Path(job['folder']).resolve() or not trace.is_file() or v4._hash(trace) != digest:
        raise ValueError('Saved learner trajectory is missing, moved or changed')
    return trace_name


def _mine_completed(directory, job, receipt, expected, roots_per_episode):
    """Exact sequential v4 sampling/provenance; coordinator-only mutation."""
    trace_name = _check_receipt(directory, job, receipt, expected)
    rows = v4._read_rows(Path(directory)/trace_name)
    if not rows or any(row.get('active_checkpoint') != expected or
                      row['response'].get('active_checkpoint', expected) != expected for row in rows):
        raise ValueError('Saved learner trajectory contains mixed or missing adapter identity')
    actions = [r['diagnostics']['choice'] for r in rows]
    hashes = [fingerprint(r['request']['state']) for r in rows]
    selected = set(v4._root_indices(rows, roots_per_episode))
    selected.update(select_roots(rows, max(1, roots_per_episode-len(selected))))
    selected = sorted(selected)[:roots_per_episode]
    for index in selected:
        v4._candidate(directory, split=job['split'], seed=job['seed'], level=job['level'], origin='learner',
            source={'kind':'v3_episode', 'episode':job['id'], 'trace':trace_name,
                    'trace_sha256':receipt['files'][trace_name], 'root_index':index, 'checkpoint':expected},
            state=rows[index]['request']['state'], actions=actions[:index], hashes=hashes[:index+1])
    return len(selected)


def _commit_game(directory, job, report, expected, roots_per_episode):
    spec = {**v4.SPEC, 'levels':[job['level']], 'seeds':[job['seed']]}
    if report.get('active_checkpoint') != expected or report.get('protocol') != spec or len(report.get('episodes', [])) != 1:
        raise ValueError('Completed learner game report does not match the fixed adapter/protocol')
    episode = report['episodes'][0]
    if episode['seed'] != job['seed'] or episode['level'] != job['level']:
        raise ValueError('Completed learner report belongs to another episode')
    trace = Path(episode['trace'])
    receipt = {**episode, 'id':job['id'], 'split':job['split'], 'checkpoint':expected,
        'accepted':True, 'kind':'learner', 'files':{trace.relative_to(Path(directory)).as_posix():v4._hash(trace)}}
    # Validate the isolated completed file before exposing its receipt to backup.
    _check_receipt(directory, job, receipt, expected)
    rows = v4._read_rows(trace)
    if not rows or any(r.get('active_checkpoint') != expected or
                      r['response'].get('active_checkpoint', expected) != expected for r in rows):
        raise ValueError('Completed trace is not bound to the frozen v3 adapter')
    v4._save(Path(job['folder'])/'attempt.json', receipt)
    _mine_completed(directory, job, receipt, expected, roots_per_episode)
    return receipt


def collect_learner_parallel(directory, model_info, predict, qualification_path,
                    train_seeds=v4.DEFAULT_TRAIN_SEEDS, development_seeds=v4.DEFAULT_DEVELOPMENT_SEEDS,
                    backup_directory=None, backup_seconds=300, wave_id='wave-001', roots_per_episode=64, workers=8):
    """Sequential-v4-compatible full games, bounded independent concurrency.

    Completed games are reused on restart, including games made by the original
    sequential collector. An interrupted in-flight game has no committed receipt
    and restarts from its native starting board. HTTP calls already in flight
    finish or reach the caller's timeout before cooperative shutdown completes.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError('Use between 1 and 8 independent native-game workers')
    if not isinstance(backup_seconds, (int,float)) or isinstance(backup_seconds,bool) or backup_seconds <= 0:
        raise ValueError('backup_seconds must be positive')
    directory = Path(directory)
    v4._harvest_limits(roots_per_episode, 0, 1)
    config = v4._configuration(directory, qualification_path)
    helper_sha256 = v4._hash(Path(__file__))
    if config.get('parallel_learner_helper_sha256') not in (None, helper_sha256):
        raise ValueError('Parallel collector source changed; resume with the pinned helper')
    expected = copy.deepcopy(model_info()); v4.require_v3_identity(expected)
    if config.get('probe_checkpoint') not in (None, expected):
        raise ValueError('Different v3 adapter in the collection')
    v4._set_seeds(directory, config, 'learner', train_seeds, development_seeds, wave_id,
                  {'roots_per_episode':roots_per_episode})
    config['probe_checkpoint'] = expected; config['parallel_learner_helper_sha256'] = helper_sha256
    v4._save(directory/'collection.json',config)
    cancelled = threading.Event(); jobs = []
    for split,seeds in [('train',train_seeds),('development',development_seeds)]:
        for level in v4.SPEC['levels']:
            for seed in seeds:
                identifier = f'learner-{wave_id}-{split}-level-{level}-seed-{seed}'
                jobs.append({'id':identifier,'split':split,'level':level,'seed':seed,
                             'folder':str(directory/'episodes'/identifier)})
    reused = completed = 0; errors = []; futures = {}; pending = []
    started = time.monotonic(); last_backup = started
    def progress():
        v4._save(directory/'learner-parallel-index.json',{
            'stage':'learner','wave_id':wave_id,'completed_this_run':completed,'reused_games':reused,
            'expected_games':len(jobs),'workers':workers,'inflight_games':len(futures),
            'errors':errors,'elapsed_seconds':time.monotonic()-started,
            'helper_sha256':helper_sha256,'checkpoint':expected})
    # Even a cached receipt can be interrupted midway through candidate mining;
    # rerunning the deterministic coordinator restores any missing candidates.
    pool = None
    try:
        v4._backup(directory,backup_directory)
        for job in jobs:
            _identity(model_info,expected,cancelled)
            path = Path(job['folder'])/'attempt.json'
            if path.exists():
                _mine_completed(directory,job,v4._load(path),expected,roots_per_episode); reused += 1
            else: pending.append(job)
            if backup_directory is not None and time.monotonic()-last_backup >= backup_seconds:
                v4._candidate_index(directory); progress(); v4._backup(directory,backup_directory)
                last_backup = time.monotonic()
        v4._candidate_index(directory); progress()
        if pending:
            pool = ThreadPoolExecutor(max_workers=min(workers,len(pending)),thread_name_prefix='kev-native-game')
            queued = iter(pending)
            def submit():
                job = next(queued,None)
                if job is not None: futures[pool.submit(_play_game,job,model_info,predict,expected,cancelled)] = job
            for _ in range(min(workers,len(pending))): submit()
            while futures:
                ready,_ = wait(futures,timeout=15,return_when=FIRST_COMPLETED)
                if not ready:
                    print(f'[v4/learner] {completed+reused}/{len(jobs)} games committed; '
                          f'{len(futures)} independent games; {(time.monotonic()-started)/60:.1f} min elapsed',flush=True)
                for future in sorted(ready,key=lambda f:futures[f]['id']):
                    job = futures.pop(future)
                    try:
                        report = future.result(); _identity(model_info,expected,cancelled)
                        if v4._hash(Path(__file__)) != helper_sha256:
                            raise ValueError('Parallel collector source changed during generation')
                        receipt = _commit_game(directory,job,report,expected,roots_per_episode)
                    except BaseException as error:
                        errors.append({'id':job['id'],'error':f'{type(error).__name__}: {error}'})
                        raise
                    completed += 1; v4._candidate_index(directory); progress()
                    print(f'[v4/learner] {completed+reused}/{len(jobs)} {job["id"]}: '
                          f'{receipt["metrics"]["outcome"]}',flush=True)
                    submit()
                if backup_directory is not None and time.monotonic()-last_backup >= backup_seconds:
                    v4._backup(directory,backup_directory); last_backup = time.monotonic()
    except BaseException:
        cancelled.set()
        for future in futures: future.cancel()
        progress()
        raise
    finally:
        if pool is not None: pool.shutdown(wait=True,cancel_futures=True)
        v4._candidate_index(directory); progress(); v4._backup(directory,backup_directory)
    entries = v4._candidate_index(directory)
    return {'stage':'learner','status':'complete','completed':completed,'reused':reused,'errors':[],
        'pending_probe_count':sum(not e['probed'] for e in entries),
        'index':str(directory/'learner-parallel-index.json'),'helper_sha256':helper_sha256}
