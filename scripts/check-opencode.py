#!/usr/bin/env python3
"""Explicit opt-in OpenCode 1.19.0 software gate; portable checks stay offline.

Download the official Linux x64 baseline npm artifact, verify its pinned SHA512
SRI before extracting or executing it, then run isolated installed-client tests.
Nothing is installed globally. A supplied --binary permits offline invocation.
"""
import argparse
import base64
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.19.0'
PACKAGE = 'opencode-linux-x64-baseline'
URL = 'https://registry.npmjs.org/opencode-linux-x64-baseline/-/opencode-linux-x64-baseline-1.19.0.tgz'
# npm package metadata for opencode-linux-x64-baseline@1.19.0. Updating this pin
# requires checking the official package metadata and rerunning both real-client
# integrations; no runtime resolution of "latest" or trust-on-first-use.
INTEGRITY = 'sha512-9EopGD/LvTfUBF24NyJFnkLd12CG0Tu7JO/BqHmjwASnaJbCmFQjx8KCJ5UBMPG/wwsBiXZiP8T0VMNTPWfnnw=='
MAX_ARCHIVE = 200 * 1024 * 1024
MAX_BINARY = 200 * 1024 * 1024


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def download_binary(directory, opener=urllib.request.urlopen):
    archive = directory / 'opencode.tgz'
    digest = hashlib.sha512()
    size = 0
    deadline = time.monotonic() + 180
    with opener(URL, timeout=30) as source, archive.open('wb') as destination:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_ARCHIVE or time.monotonic() > deadline:
                raise ValueError('OpenCode download exceeds byte or time limit')
            digest.update(chunk)
            destination.write(chunk)
    actual = 'sha512-' + base64.b64encode(digest.digest()).decode('ascii')
    if actual != INTEGRITY:
        raise ValueError('OpenCode npm artifact integrity mismatch; binary was not executed')
    executable = directory / 'opencode'
    with tarfile.open(archive, 'r:gz') as package:
        metadata = package.getmember('package/package.json')
        binary = package.getmember('package/bin/opencode')
        if not metadata.isfile() or metadata.size > 16384 or not binary.isfile() or not 0 < binary.size <= MAX_BINARY:
            raise ValueError('OpenCode artifact contains an invalid package member')
        with package.extractfile(metadata) as stream:
            document = json.load(stream)
        if document.get('name') != PACKAGE or document.get('version') != VERSION:
            raise ValueError('OpenCode package identity mismatch')
        with package.extractfile(binary) as source, executable.open('wb') as destination:
            count = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_BINARY:
                    raise ValueError('OpenCode binary exceeds size limit')
                destination.write(chunk)
        if count != binary.size:
            raise ValueError('truncated OpenCode binary')
    executable.chmod(0o500)
    return executable, {'source': 'official_npm_pinned_artifact', 'package': PACKAGE,
                        'version': VERSION, 'tarball_url': URL, 'tarball_integrity': INTEGRITY,
                        'tarball_bytes': size, 'binary_sha256': sha256(executable)}


def private_write(path, document):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.opencode-smoke-', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, help='use an already installed OpenCode 1.19.0 binary offline')
    parser.add_argument('--output', type=Path, default=ROOT / '.backcluster/opencode-client-smoke.json')
    args = parser.parse_args(argv)
    provenance = None
    try:
        with tempfile.TemporaryDirectory(prefix='backcluster-opencode-gate-') as tmp:
            directory = Path(tmp)
            if args.binary:
                executable = args.binary.resolve(strict=True)
                if not executable.is_file() or not os.access(executable, os.X_OK):
                    raise ValueError('supplied OpenCode binary must be an executable regular file')
                provenance = {'source': 'supplied_local_binary', 'version': VERSION, 'binary_sha256': sha256(executable)}
            else:
                if sys.platform != 'linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
                    raise ValueError('automatic pinned download supports Linux x64; use --binary on other systems')
                executable, provenance = download_binary(directory)
            clean = {'PATH': os.environ.get('PATH', ''), 'HOME': str(directory),
                'XDG_CONFIG_HOME': str(directory / 'config'), 'XDG_DATA_HOME': str(directory / 'data'),
                'XDG_CACHE_HOME': str(directory / 'cache'), 'OPENCODE_DISABLE_MODELS_FETCH': '1',
                'OPENCODE_DISABLE_AUTOUPDATE': '1', 'OPENCODE_DISABLE_DEFAULT_PLUGINS': '1'}
            version = subprocess.run([str(executable), '--version'], env=clean, cwd=directory,
                capture_output=True, text=True, timeout=20, check=True).stdout.strip()
            if version != VERSION:
                raise ValueError('OpenCode client version differs from the pinned software gate')
            env = dict(os.environ, BACKCLUSTER_OPENCODE_BIN=str(executable))
            # The real client subprocesses in test_quality receive a separate
            # empty HOME and a clean environment; credentials are never passed.
            command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/backcluster',
                       '-p', 'test_quality.py', '-v']
            tests = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                capture_output=True, timeout=180, check=False)
            sys.stdout.buffer.write(tests.stdout); sys.stdout.flush()
            sys.stderr.buffer.write(tests.stderr); sys.stderr.flush()
            required = ('test_installed_client_config_sse_reasoning_and_local_alias',
                        'test_installed_client_read_edit_test_round_trip')
            test_log = tests.stderr.decode('utf-8', errors='replace')
            client_cases_passed = all(re.search(re.escape(name) + r' \([^\n]+\) \.\.\. ok(?:\n|$)', test_log)
                                      for name in required)
            source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                capture_output=True, text=True, timeout=5, check=False).stdout.strip()
            report = {'document_kind': 'backcluster_opencode_client_smoke', 'schema_version': 1,
                'captured_at': datetime.now(timezone.utc).isoformat(), 'ok': tests.returncode == 0 and client_cases_passed,
                'hardware_validated': False, 'model_inference_validated': False,
                'backend': 'deterministic_loopback_fake_openai', 'client': provenance,
                'source_commit': source if re.fullmatch('[0-9a-f]{40}', source) else None,
                'test_files_sha256': {name: sha256(ROOT / name) for name in
                    ('tools/backcluster/quality.py', 'tools/backcluster/benchmark.py',
                     'tests/backcluster/test_quality.py', 'scripts/check-opencode.py')},
                'test_suite': 'test_quality.py', 'test_suite_exit_code': tests.returncode,
                'installed_client_cases_passed': client_cases_passed,
                'test_stdout_sha256': hashlib.sha256(tests.stdout).hexdigest(),
                'test_stderr_sha256': hashlib.sha256(tests.stderr).hexdigest(),
                'checks': ['native config schema', 'one local locked model alias', 'SSE and separate reasoning',
                          'actual OpenCode read/edit/bash regression-test round-trip'],
                'limitations': ['Fake model decisions are fixed fixtures, not real-model coding quality',
                                'Target Mac, real Qwen weights, offline installation and long sessions remain hardware acceptance']}
            private_write(args.output, report)
            print('OpenCode %s software gate: %s' % (VERSION, 'passed' if report['ok'] else 'failed'))
            return 0 if report['ok'] else 1
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
        print('check-opencode: ' + str(error), file=sys.stderr)
        try:
            private_write(args.output, {'document_kind': 'backcluster_opencode_client_smoke', 'schema_version': 1,
                'ok': False, 'hardware_validated': False, 'model_inference_validated': False,
                'client': provenance, 'error': 'software_gate_failed'})
        except OSError:
            print('check-opencode: could not write the software receipt', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
