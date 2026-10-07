"""Incremental, verified Drive backup for committed data-generation artifacts.

Callers list only atomically completed files. Mutable indexes get a new version
in the next chunk; restoration uses the latest version of each indexed file.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import uuid
import zipfile


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _name(value):
    if not isinstance(value, str):
        raise ValueError('Collection entries must be relative POSIX paths')
    name = PurePosixPath(value)
    if (not value or name.is_absolute() or '..' in name.parts or '\\' in value
            or str(name) != value or any(p.startswith('.') for p in name.parts)):
        raise ValueError('Unsafe collection entry')
    return value


def _receipt(folder, collection):
    pointer = folder / 'latest.json'
    if not pointer.is_file():
        return None
    receipt = json.loads(pointer.read_text())
    if receipt.get('version') != 1 or receipt.get('collection_path') != str(collection):
        raise ValueError('Collection backup belongs to a different collection path')
    for name, info in receipt['files'].items():
        _name(name)
        _name(info['archive'])
        if '/' in info['archive'] or info['archive'] not in receipt['archives']:
            raise ValueError('Invalid collection archive reference')
    return receipt


def backup_collection(collection, backup_directory, files=None):
    """Publish a SHA-verified delta and then its pointer; retain prior backups.

    ``files`` is the caller's complete committed artifact inventory. Passing it
    explicitly is required so an active worker's partial output cannot be saved.
    """
    collection, folder = Path(collection).resolve(), Path(backup_directory).resolve()
    if files is None:
        raise ValueError('Supply the completed artifact inventory for backup')
    names = sorted({_name(str(name)) for name in files})
    folder.mkdir(parents=True, exist_ok=True)
    previous = _receipt(folder, collection)
    index = dict(previous['files']) if previous else {}
    archives = dict(previous['archives']) if previous else {}
    changed = []
    for name in names:
        source = collection / name
        if (not source.is_file() or source.is_symlink() or
                not source.resolve().is_relative_to(collection)):
            raise ValueError('Committed collection artifact is missing or unsafe: ' + name)
        sha = _hash(source)
        if index.get(name, {}).get('sha256') != sha:
            changed.append((name, source, sha))
    if not changed and previous:
        return previous
    archive_name = f'collection-{time.time_ns()}-{uuid.uuid4().hex[:8]}.zip'
    with tempfile.TemporaryDirectory(prefix='kev-collection-backup-') as temporary:
        archive = Path(temporary) / archive_name
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as saved:
            for name, source, expected in changed:
                content = source.read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError('Collection artifact changed during backup; previous pointer retained')
                saved.writestr(name, content)
                index[name] = {'archive': archive_name, 'sha256': expected, 'bytes': len(content)}
        sha = _hash(archive)
        uploading = folder / ('.uploading-' + archive_name)
        shutil.copyfile(archive, uploading)
        if _hash(uploading) != sha:
            raise ValueError('Collection Drive upload checksum mismatch; previous pointer retained')
        os.replace(uploading, folder / archive_name)
        archives[archive_name] = sha
    receipt = {'version': 1, 'collection_path': str(collection), 'created_ns': time.time_ns(),
               'files': index, 'archives': archives}
    pending = folder / ('.latest-' + uuid.uuid4().hex + '.json')
    pending.write_text(json.dumps(receipt, indent=2) + '\n')
    os.replace(pending, folder / 'latest.json')
    print(f'Collection Drive backup complete: {len(index)} committed files; '
          f'{len(changed)} new/changed files; {folder}', flush=True)
    return receipt


def restore_collection(collection, backup_directory):
    """Restore absent local collections atomically after checking every file."""
    collection, folder = Path(collection).resolve(), Path(backup_directory).resolve()
    if collection.exists():
        print('Existing local collection preserved:', collection, flush=True)
        return None
    receipt = _receipt(folder, collection)
    if receipt is None:
        return None
    collection.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.collection-restore-', dir=collection.parent) as temporary:
        restored = Path(temporary) / 'restored'
        restored.mkdir()
        needed = {info['archive'] for info in receipt['files'].values()}
        for name in needed:
            archive = folder / name
            if archive.is_symlink() or _hash(archive) != receipt['archives'][name]:
                raise ValueError('Collection archive checksum mismatch; restore not published')
            with zipfile.ZipFile(archive) as saved:
                entries = saved.namelist()
                if len(entries) != len(set(entries)):
                    raise ValueError('Duplicate collection archive entries')
                for entry in entries:
                    _name(entry)
                    if (saved.getinfo(entry).external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Collection archive contains a symbolic link')
                for entry, info in receipt['files'].items():
                    if info['archive'] != name:
                        continue
                    content = saved.read(entry)
                    if (len(content) != info['bytes'] or
                            hashlib.sha256(content).hexdigest() != info['sha256']):
                        raise ValueError('Collection file checksum mismatch; restore not published')
                    target = restored / entry
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
        if collection.exists():
            raise ValueError('Local collection appeared during restore; local files preserved')
        os.replace(restored, collection)
    print('Collection restored from Drive:', collection, flush=True)
    return receipt
