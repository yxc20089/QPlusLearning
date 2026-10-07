"""Verified checkpoint archives for a mounted Google Drive directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import zipfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.writing-' + uuid.uuid4().hex + '.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def new_run_directory(results_root, label, **metadata):
    """Preserve each invocation, including runs that are later interrupted."""
    if not label or Path(label).name != label or label in ('.', '..'):
        raise ValueError('Run label must be a single directory name')
    run_id = time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    directory = Path(results_root) / f'{label}-{run_id}'
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / 'session.json', {'run_id': run_id, 'label': label,
                'created_ns': time.time_ns(), **metadata})
    return directory


def _teacher_collection_inputs(collection, original=None):
    metadata = collection / 'collection.json'
    info = json.loads(metadata.read_text())
    files = {metadata}
    for name in info['reports'].values():
        report = collection / name
        files.add(report)
        for episode in json.loads(report.read_text())['episodes']:
            # Restored report bytes retain the original Colab absolute paths.
            relative = Path(episode['trace']).resolve().relative_to(original or collection)
            files.add(collection / relative)
    for file in files:
        if file.is_symlink() or not file.resolve().is_relative_to(collection):
            raise ValueError('Teacher collection input is outside its collection directory')
    return {str(file.relative_to(collection)): file_hash(file) for file in sorted(files)}


def _completed_teacher_files(folder):
    """Old helpers wrote the receipt before labels; verify that labels are complete."""
    marker, trace = folder / 'attempt.json', folder / 'teacher.jsonl'
    receipt = json.loads(marker.read_text())
    if file_hash(trace) != receipt['trace_sha256']:
        raise ValueError('Completed teacher trace checksum mismatch')
    proofs = [json.loads(line) for line in trace.read_text().splitlines()]
    if len(proofs) != receipt['metrics']['decisions']:
        raise ValueError('Completed teacher trace length differs from its receipt')
    files = [marker, trace]
    if receipt['accepted']:
        candidates = folder / 'candidates.jsonl'
        if not candidates.is_file():
            return None  # A currently-running older helper may still be writing labels.
        expected = {i for i, proof in enumerate(proofs) if any(
            not risk['life_lost'] for risk in proof['diagnostics']['immediate_counterfactuals'].values())}
        seen = set()
        for line in candidates.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return None
            index = record['_meta']['suffix_index']
            digest = hashlib.sha256(json.dumps(record['state'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if (index not in expected or index in seen or digest != proofs[index]['state_sha256']
                    or record['questions']['move']['label'] != proofs[index]['choice']):
                raise ValueError('Completed teacher labels disagree with their native trace')
            seen.add(index)
        if seen != expected:
            return None
        files.append(candidates)
    return files


def backup_teacher_collection(collection, backup_directory):
    """Incremental, verified Drive archives of immutable inputs/completed attempts."""
    collection, backup_directory = Path(collection).resolve(), Path(backup_directory).resolve()
    if not collection.is_dir() or backup_directory.is_relative_to(collection):
        raise ValueError('Use an existing teacher collection and a separate backup directory')
    base = _teacher_collection_inputs(collection)
    pointer = backup_directory / 'latest.json'
    previous = json.loads(pointer.read_text()) if pointer.is_file() else None
    if previous and (previous['version'] != 1 or previous['collection_path'] != str(collection)
                     or previous['base_files'] != base):
        raise ValueError('Drive teacher backup belongs to a different learner collection')
    backed = dict(previous['attempts']) if previous else {}
    files = [] if previous else [collection / name for name in base]
    newly_backed = {}
    for marker in sorted((collection / 'teacher-recoveries').glob('*/attempt.json')):
        name = marker.parent.name
        if name in backed:
            if file_hash(marker) != backed[name]:
                raise ValueError('An already-backed-up teacher receipt changed')
            continue
        complete = _completed_teacher_files(marker.parent)
        if complete is not None:
            files.extend(complete)
            newly_backed[name] = file_hash(marker)
    if not files:
        return previous
    backup_directory.mkdir(parents=True, exist_ok=True)
    name = f'collection-{time.time_ns()}-{uuid.uuid4().hex[:8]}.zip'
    entries = []
    print(f'Teacher Drive backup starting: {len(newly_backed)} new completed attempts', flush=True)
    with tempfile.TemporaryDirectory(prefix='kev-teacher-backup-') as temporary:
        archive = Path(temporary) / name
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as saved:
            for file in sorted(files):
                if file.is_symlink() or not file.resolve().is_relative_to(collection):
                    raise ValueError('Teacher backup contains a symbolic link or external file')
                relative = str(file.relative_to(collection))
                digest, size = hashlib.sha256(), 0
                with file.open('rb') as source, saved.open(relative, 'w', force_zip64=True) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                entries.append({'path': relative, 'bytes': size, 'sha256': digest.hexdigest()})
        copied = {entry['path']: entry['sha256'] for entry in entries}
        if any(copied[path] != checksum for path, checksum in base.items() if path in copied) or any(
                copied[f'teacher-recoveries/{attempt}/attempt.json'] != checksum
                for attempt, checksum in newly_backed.items()):
            raise ValueError('Teacher collection changed during backup; previous backup retained')
        checksum = file_hash(archive)
        uploading = backup_directory / ('.uploading-' + name)
        shutil.copyfile(archive, uploading)
        if file_hash(uploading) != checksum:
            raise ValueError('Teacher Drive archive checksum mismatch; previous backup retained')
        os.replace(uploading, backup_directory / name)
    receipt = {'version': 1, 'collection_path': str(collection), 'created_ns': time.time_ns(),
               'base_files': base, 'attempts': {**backed, **newly_backed},
               'archives': [*(previous['archives'] if previous else []),
                            {'archive': name, 'sha256': checksum, 'files': entries}]}
    atomic_json(backup_directory / (name + '.json'), receipt)
    atomic_json(pointer, receipt)
    print(f'Teacher Drive backup complete: {len(receipt["attempts"])} completed attempts; {backup_directory}', flush=True)
    return receipt


def restore_teacher_collection(collection, backup_directory):
    """Restore a verified collection at its original path, never over local work."""
    collection, backup_directory = Path(collection).resolve(), Path(backup_directory).resolve()
    pointer = backup_directory / 'latest.json'
    if collection.exists() or not pointer.is_file():
        return None
    receipt = json.loads(pointer.read_text())
    if receipt['version'] != 1 or receipt['collection_path'] != str(collection):
        raise ValueError('Restore the teacher collection at its original absolute path')
    print(f'Teacher Drive restore starting: {len(receipt["attempts"])} completed attempts', flush=True)
    collection.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restoring-teacher-', dir=collection.parent) as temporary:
        staging = Path(temporary) / 'collection'
        staging.mkdir()
        seen = set()
        for chunk in receipt['archives']:
            archive = backup_directory / chunk['archive']
            if archive.parent != backup_directory or file_hash(archive) != chunk['sha256']:
                raise ValueError('Teacher Drive archive checksum mismatch')
            with zipfile.ZipFile(archive) as saved:
                names = [info['path'] for info in chunk['files']]
                if len(set(saved.namelist())) != len(saved.namelist()) or set(saved.namelist()) != set(names):
                    raise ValueError('Teacher archive entries differ from its receipt')
                for info in chunk['files']:
                    relative = Path(info['path'])
                    if relative.is_absolute() or '..' in relative.parts or info['path'] in seen:
                        raise ValueError('Invalid or duplicate teacher archive path')
                    seen.add(info['path'])
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with saved.open(info['path']) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    if target.stat().st_size != info['bytes'] or file_hash(target) != info['sha256']:
                        raise ValueError('Restored teacher file checksum mismatch')
        if _teacher_collection_inputs(staging, original=collection) != receipt['base_files']:
            raise ValueError('Restored teacher inputs differ from the backup')
        for name, checksum in receipt['attempts'].items():
            folder = staging / 'teacher-recoveries' / name
            if file_hash(folder / 'attempt.json') != checksum or _completed_teacher_files(folder) is None:
                raise ValueError('Restored teacher attempt is incomplete')
        os.replace(staging, collection)
    print(f'Teacher Drive restore complete: {len(receipt["attempts"])} attempts reused at {collection}', flush=True)
    return receipt


def backup_lab_to_drive(workspace, drive_root, checkpoint_root=None):
    """Publish restore-compatible checkpoints and a dated archive of all recordings.

    Missing stages/comparisons are allowed. A failed checkpoint copy does not
    prevent preserving the traces; the returned errors must still be reported.
    Foundation downloads and environments are outside the selected directories.
    """
    from lora_recovery import latest_snapshot
    workspace, drive_root = Path(workspace).resolve(), Path(drive_root).resolve()
    checkpoint_root = (Path(checkpoint_root).resolve() if checkpoint_root is not None
                       else workspace / 'checkpoints')
    if not workspace.is_dir() or drive_root.is_relative_to(workspace):
        raise ValueError('Use an existing lab workspace and a separate Drive directory')
    checkpoints, skipped, errors = {}, [], {}
    if checkpoint_root.is_dir():
        for output in sorted(checkpoint_root.iterdir()):
            if not output.is_dir() or output.name.endswith('-recovery'):
                continue
            try:
                snapshot = latest_snapshot(Path(str(output) + '-recovery'))
                if snapshot is None and not (output / 'run-evidence.json').is_file():
                    skipped.append({'checkpoint': str(output), 'reason': 'No complete learned snapshot or final checkpoint'})
                    continue
                config_path = output / 'training_config.json'
                data = json.loads(config_path.read_text()).get('args', {}).get('data') if config_path.is_file() else None
                if data and not Path(data).is_absolute():
                    data = workspace / 'kev' / data
                # Preserve available input bytes, as in automatic training
                # backups; a restored final checkpoint may lack older inputs.
                data = Path(data).resolve() if data else None
                data = data if data and data.is_relative_to(workspace) and data.is_file() else None
                checkpoints[output.name] = save_backup(output, snapshot, drive_root / 'backups',
                                                       workspace, training_data=data)
            except Exception as error:
                errors[output.name] = f'{type(error).__name__}: {error}'
                print(f'Checkpoint backup failed for {output.name}: {errors[output.name]}', flush=True)

    directories = ('results', 'logs', 'data', 'evaluation')
    files = {file for name in directories for file in (workspace / name).rglob('*') if file.is_file()}
    files.update(file for pattern in ('*.json', '*.log') for file in workspace.glob(pattern) if file.is_file())
    manifest = {'version': 1, 'workspace': str(workspace), 'created_ns': time.time_ns(),
                'checkpoint_backups': checkpoints, 'skipped_checkpoints': skipped, 'checkpoint_errors': errors,
                'missing_directories': [name for name in directories if not (workspace / name).is_dir()],
                'files': []}
    folder = drive_root / 'session-backups'
    folder.mkdir(parents=True, exist_ok=True)
    name = 'session-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:8] + '.zip'
    print(f'Session backup starting: {len(files)} recording/log/data files', flush=True)
    with tempfile.TemporaryDirectory(prefix='kev-session-backup-') as temporary:
        archive = Path(temporary) / name
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as saved:
            for file in sorted(files):
                if file.is_symlink() or not file.resolve().is_relative_to(workspace):
                    raise ValueError('Session backup contains a symbolic link or external file')
                relative = str(file.relative_to(workspace))
                digest, size = hashlib.sha256(), 0
                with file.open('rb') as source, saved.open(relative, 'w', force_zip64=True) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                manifest['files'].append({'path': relative, 'bytes': size, 'sha256': digest.hexdigest()})
            saved.writestr('session-manifest.json', json.dumps(manifest, indent=2) + '\n')
        checksum = file_hash(archive)
        uploading = folder / ('.uploading-' + name)
        shutil.copyfile(archive, uploading)
        if file_hash(uploading) != checksum:
            raise ValueError('Drive session archive checksum mismatch; previous backup retained')
        os.replace(uploading, folder / name)
    receipt = {'version': 1, 'archive': str(folder / name), 'sha256': checksum,
               'created_ns': manifest['created_ns'], 'files': len(files),
               'checkpoint_backups': checkpoints, 'skipped_checkpoints': skipped, 'checkpoint_errors': errors}
    atomic_json(folder / (name + '.json'), receipt)
    atomic_json(folder / 'latest.json', receipt)
    print(f'Session backup complete: {folder / name}', flush=True)
    return receipt


def save_backup(output, snapshot, backup_root, workspace, training_data=None):
    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    snapshot = Path(snapshot).resolve() if snapshot else None
    recovery_root = Path(str(output) + '-recovery')
    if not output.is_dir():
        raise ValueError('Checkpoint output is missing; cannot back it up')
    if snapshot is not None:
        if snapshot.parent != recovery_root or not (snapshot / 'complete.json').is_file():
            raise ValueError('Back up only a complete snapshot for this output')
        recovery = json.loads((snapshot / 'complete.json').read_text())
        if file_hash(snapshot / 'recovery.pt') != recovery['recovery_sha256']:
            raise ValueError('Recovery checksum mismatch; Drive backup not published')
        step = recovery['step']
    elif (output / 'run-evidence.json').is_file():
        step = json.loads((output / 'training_metrics.json').read_text())['optimizer_steps']
    else:
        raise ValueError('No learned recovery snapshot or completed checkpoint to back up')
    data = Path(training_data).resolve() if training_data else None
    if data is not None and (not data.is_relative_to(workspace) or not data.is_file()):
        raise ValueError('Training data backup must be a file in the lab workspace')
    artifacts = []
    # The correction round generates private evidence at runtime. Restore its
    # dataset receipt/development/replays together with the exact training file.
    if data is not None and data.name in ('pacman-native-v3-train.jsonl', 'pacman-native-v4-train.jsonl'):
        manifest_path = data.with_name(data.name.removesuffix('-train.jsonl') + '-manifest.json')
        manifest = json.loads(manifest_path.read_text())
        for artifact_name, expected in manifest['files'].items():
            artifact = data.parent / artifact_name
            if (Path(artifact_name).name != artifact_name or artifact.is_symlink()
                    or not artifact.resolve().is_relative_to(workspace) or file_hash(artifact) != expected):
                raise ValueError('Correction dataset evidence changed; Drive backup not published')
            if artifact != data:
                artifacts.append(artifact)
        artifacts.append(manifest_path)
    artifact_info = [{'path': str(p.resolve()), 'sha256': file_hash(p),
                      'entry': 'inputs/artifacts/' + p.name} for p in artifacts]
    folder = Path(backup_root).resolve() / output.name
    folder.mkdir(parents=True, exist_ok=True)
    name = f'backup-{time.time_ns()}-{uuid.uuid4().hex[:8]}.zip'
    print(f'Drive backup starting at optimizer step {step}: {folder}', flush=True)
    with tempfile.TemporaryDirectory(prefix='kev-checkpoint-backup-') as temporary:
        archive = Path(temporary) / name
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as saved:
            for directory in [output, *([snapshot] if snapshot else [])]:
                for file in sorted(directory.rglob('*')):
                    if file.is_symlink():
                        raise ValueError('Checkpoint backup contains a symbolic link')
                    if file.is_file():
                        saved.write(file, file.relative_to(output.parent))
            if snapshot is not None:
                saved.writestr(recovery_root.name + '/latest.json', json.dumps(recovery) + '\n')
            if data is not None:
                saved.write(data, 'inputs/training.jsonl')
            for artifact, info in zip(artifacts, artifact_info):
                saved.write(artifact, info['entry'])
        checksum = file_hash(archive)
        uploading = folder / ('.uploading-' + name)
        shutil.copyfile(archive, uploading)
        if file_hash(uploading) != checksum:
            raise ValueError('Drive archive checksum mismatch; previous backup retained')
        os.replace(uploading, folder / name)
    receipt = {'version': 1, 'archive': name, 'sha256': checksum, 'created_ns': time.time_ns(),
               'output_path': str(output), 'workspace_path': str(workspace), 'step': step,
               'snapshot': snapshot.name if snapshot else None,
               'completed_stage': (output / 'run-evidence.json').is_file(),
               'training_data': {'path': str(data), 'sha256': file_hash(data)} if data else None,
               'training_artifacts': artifact_info}
    atomic_json(folder / (name + '.json'), receipt)
    atomic_json(folder / 'latest.json', receipt)
    # Keep the two most recently published backups, including an older selected
    # checkpoint continued as a new attempt. Ignore interrupted uploads.
    previous = sorted(folder.glob('backup-*.zip.json'),
                      key=lambda p: json.loads(p.read_text())['created_ns'])
    for metadata in previous[:-2]:
        old = json.loads(metadata.read_text())
        archive = folder / old['archive']
        if archive.parent != folder:
            raise ValueError('Invalid backup archive name')
        archive.unlink(missing_ok=True)
        metadata.unlink()
    print(f'Drive backup complete at optimizer step {step}: {folder / name}', flush=True)
    return receipt


def restore_backup(output, backup_root, workspace):
    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    recovery_root = Path(str(output) + '-recovery')
    folder = Path(backup_root).resolve() / output.name
    pointer = folder / 'latest.json'
    if not pointer.is_file() or output.exists() or recovery_root.exists():
        return None  # Existing local training is never replaced.
    receipt = json.loads(pointer.read_text())
    if receipt['version'] != 1 or receipt['output_path'] != str(output) or receipt['workspace_path'] != str(workspace):
        raise ValueError('Restore this backup at its original lab workspace and output paths')
    archive = folder / receipt['archive']
    print(f'Drive restore starting at optimizer step {receipt["step"]}: {archive}', flush=True)
    if archive.parent != folder or file_hash(archive) != receipt['sha256']:
        raise ValueError('Drive backup archive checksum mismatch')
    data_info = receipt.get('training_data')
    data = Path(data_info['path']).resolve() if data_info else None
    if data is not None and (not data.is_relative_to(workspace) or
                            (data.exists() and file_hash(data) != data_info['sha256'])):
        raise ValueError('Restore training data at its original path without replacing changed labels')
    artifacts = receipt.get('training_artifacts', [])
    for info in artifacts:
        path = Path(info['path']).resolve()
        if (not path.is_relative_to(workspace) or info['entry'] != 'inputs/artifacts/' + path.name
                or (path.exists() and file_hash(path) != info['sha256'])):
            raise ValueError('Restore correction evidence without replacing changed local files')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restoring-', dir=output.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as saved:
            allowed = {output.name, recovery_root.name, 'inputs'}
            for member in saved.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging.resolve()) or Path(member.filename).parts[0] not in allowed:
                    raise ValueError('Unsafe checkpoint archive path')
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Checkpoint archive contains a symbolic link')
            saved.extractall(staging)
        if not (staging / output.name).is_dir():
            raise ValueError('Backup is missing the checkpoint output')
        if receipt['snapshot'] is not None:
            from lora_recovery import latest_snapshot
            snapshot = latest_snapshot(staging / recovery_root.name)
            if snapshot is None or snapshot.name != receipt['snapshot']:
                raise ValueError('Backup is missing its selected recovery snapshot')
            saved_receipt = json.loads((snapshot / 'complete.json').read_text())
            if file_hash(snapshot / 'recovery.pt') != saved_receipt['recovery_sha256']:
                raise ValueError('Restored recovery checksum mismatch')
        if data is not None:
            source = staging / 'inputs/training.jsonl'
            if file_hash(source) != data_info['sha256']:
                raise ValueError('Restored training data checksum mismatch')
        for info in artifacts:
            if file_hash(staging / info['entry']) != info['sha256']:
                raise ValueError('Restored correction evidence checksum mismatch')
        for destination, source in ([ (data, staging / 'inputs/training.jsonl') ] if data is not None else []) + [
                (Path(info['path']), staging / info['entry']) for info in artifacts]:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        (staging / output.name).rename(output)
        if (staging / recovery_root.name).is_dir():
            (staging / recovery_root.name).rename(recovery_root)
    print(f'Restored Drive backup at optimizer step {receipt["step"]}: {output}', flush=True)
    return receipt
