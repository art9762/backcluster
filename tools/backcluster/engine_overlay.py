"""Materialize reviewed engine patches without changing the published gitlink."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .artifacts import load_json
from .devices import ROOT, engine_status, probe_environment


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def patch_spec(root=ROOT):
    root = Path(root)
    manifest = load_json(root / 'engine-patches/manifest.json')
    if manifest.get('schema_version') != 1 or type(manifest.get('schema_version')) is not int:
        raise ValueError('unsupported engine patch manifest')
    pin = engine_status(root)['expected_commit']
    if not pin or manifest.get('base_commit') != pin:
        raise ValueError('engine patch base does not match the published pin')
    patches = manifest.get('patches')
    if not isinstance(patches, list) or not patches:
        raise ValueError('engine patches required')
    names = set()
    for patch in patches:
        if not isinstance(patch, dict):
            raise ValueError('invalid engine patch')
        name = patch.get('file')
        expected = patch.get('sha256')
        if not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.patch', name) is None or name in names:
            raise ValueError('invalid engine patch filename')
        names.add(name)
        if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
            raise ValueError('invalid engine patch hash')
        path = root / 'engine-patches' / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError('engine patch content mismatch')
    identity = hashlib.sha256(json.dumps({'base_commit': pin, 'patches': patches},
                                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return manifest, identity


def expected_identity(root=ROOT):
    return patch_spec(root)[1]


def _source_hashes(source):
    result = {}
    for path in sorted(Path(source).rglob('*')):
        relative = path.relative_to(source)
        # All build output must stay in these ignored directories.
        if any(part == 'build' or part.startswith('build-') or part in ('.git', '__pycache__') for part in relative.parts[:-1]):
            continue
        if relative.as_posix() == '.backcluster-source.json':
            continue
        if path.is_symlink():
            raise ValueError('unexpected symlink in generated engine source')
        if path.is_file():
            result[relative.as_posix()] = digest(path)
    return result


def prepare_engine(root=ROOT):
    """Return source_dir/base_commit/build_id; reject base or cache tampering."""
    root = Path(root)
    manifest, identity = patch_spec(root)
    status = engine_status(root)
    if not status['matches_pin']:
        raise ValueError('engine overlay requires the clean pinned submodule')
    parent = root / '.backcluster/engine-source'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / identity
    # POSIX locks work on both supported development hosts and are released on exit.
    import fcntl
    with (parent / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        temporary = Path(tempfile.mkdtemp(prefix='.prepare-', dir=parent))
        try:
            archive = temporary / 'source.tar'
            with archive.open('wb') as output:
                subprocess.run(['git', '-C', str(root / 'llama.cpp'), 'archive', manifest['base_commit']],
                               stdout=output, check=True, env=probe_environment())
            source = temporary / 'source'
            source.mkdir()
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                for member in members:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                        raise ValueError('unsupported path in pinned engine archive')
                # Explicit path validation also supports Python 3.9.
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(source, members=members, filter='data')
                else:
                    tar.extractall(source, members=members)
            patch_env = dict(probe_environment(), GIT_CEILING_DIRECTORIES=str(parent))
            for patch in manifest['patches']:
                subprocess.run(['git', 'apply', '--check', str(root / 'engine-patches' / patch['file'])],
                               cwd=source, env=patch_env, check=True)
                subprocess.run(['git', 'apply', str(root / 'engine-patches' / patch['file'])],
                               cwd=source, env=patch_env, check=True)
            expected = {'schema_version': 1, 'base_commit': manifest['base_commit'],
                        'build_id': identity, 'source_hashes': _source_hashes(source)}
            if target.exists():
                # Reconstruct from the pin and patches. An editable local cache
                # receipt cannot authorize changed source, even if rehashed.
                receipt = load_json(target / '.backcluster-source.json')
                if (receipt != expected or type(receipt.get('schema_version')) is not int or
                        expected['source_hashes'] != _source_hashes(target)):
                    raise ValueError('generated engine source changed; remove its cache and rebuild')
            else:
                with (source / '.backcluster-source.json').open('w', encoding='utf-8') as output:
                    json.dump(expected, output, sort_keys=True)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(source, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return {'source_dir': target, 'base_commit': manifest['base_commit'], 'build_id': identity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-path', action='store_true')
    parser.add_argument('--build-id', action='store_true')
    parser.add_argument('--header', type=Path, help='write the native worker build identity header')
    args = parser.parse_args(argv)
    prepared = prepare_engine()
    if args.header:
        args.header.parent.mkdir(parents=True, exist_ok=True)
        content = '#pragma once\n#define BACKCLUSTER_ENGINE_COMMIT "%s"\n#define BACKCLUSTER_ENGINE_BUILD_ID "%s"\n' % (prepared['base_commit'], prepared['build_id'])
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.header.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
        try:
            os.replace(temporary, args.header)
        finally:
            temporary.unlink(missing_ok=True)
    if args.source_path:
        print(prepared['source_dir'])
    elif args.build_id:
        print(prepared['build_id'])
    else:
        print(json.dumps({k: str(v) for k, v in prepared.items()}, indent=2))


if __name__ == '__main__':
    main()
