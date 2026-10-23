"""Durable opaque distributed state and coordinator-side tool receipts.

The store cannot interpret KV/recurrent state. A complete snapshot proves that
every agreed stage's bytes were saved at one boundary; engine restore and
numerical correctness still need native validation. Tool callbacks have
at-most-once execution: a crash after the durable ``running`` record requires
explicit reconciliation, because replay could repeat an external side effect.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_BLOB_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * MAX_BLOB_BYTES
MAX_STAGES = 64
_STAGE = re.compile(r'[a-z][a-z0-9_-]{0,31}\Z')
_SNAPSHOT = re.compile(r'checkpoint-[0-9a-f]{32}\Z')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_ENGINE = re.compile(r'[0-9a-f]{40}\Z')
_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)
_DIRECTORY = getattr(os, 'O_DIRECTORY', 0)


class CheckpointError(ValueError):
    """A checkpoint is incomplete, corrupt, incompatible, or unsafe."""


class ToolExecutionIndeterminate(RuntimeError):
    """The callback may already have had side effects; automatic replay is unsafe."""


def _positive_limit(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise CheckpointError('invalid ' + name)


def _text(value, name, maximum=256):
    if not isinstance(value, str) or not value or '\0' in value or len(value.encode('utf-8')) > maximum:
        raise ValueError('invalid ' + name)


def _json_bytes(value, limit):
    # Bound both recursion and item counts before encoding, and reject exotic
    # Python objects / non-string dictionary keys rather than coercing them.
    count = [0]

    def visit(item, depth):
        count[0] += 1
        if depth > 32 or count[0] > 100000:
            raise ValueError('JSON structure exceeds limits')
        if item is None or type(item) in (bool, int, float, str):
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child, depth + 1)
            return
        raise ValueError('value is not plain JSON')

    visit(value, 0)
    try:
        data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, OverflowError, UnicodeError) as error:
        raise ValueError('invalid JSON value') from error
    if len(data) > limit:
        raise ValueError('JSON document exceeds limit')
    return data


def _parse_json(data, limit):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value

    if len(data) > limit:
        raise ValueError('JSON document exceeds limit')
    try:
        value = json.loads(data, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON')))
        _json_bytes(value, limit)
        return value
    except (UnicodeError, RecursionError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('invalid JSON document') from error


def _identity(identity, limit):
    if type(identity) is not dict:
        raise CheckpointError('identity must be a dictionary')
    if not isinstance(identity.get('engine_commit'), str) or not _ENGINE.fullmatch(identity['engine_commit']):
        raise CheckpointError('invalid engine identity')
    for key in ('model_sha256', 'template_sha256', 'tokenizer_sha256', 'topology_sha256'):
        if not isinstance(identity.get(key), str) or not _SHA256.fullmatch(identity[key]):
            raise CheckpointError('invalid ' + key)
    for key in ('state_format', 'context'):
        _integer(identity.get(key), key, minimum=1)
    try:
        return _json_bytes(identity, limit)
    except ValueError as error:
        raise CheckpointError(str(error)) from error


def _stages(stages):
    stages = list(stages)
    if any(not isinstance(stage, str) or not _STAGE.fullmatch(stage) for stage in stages):
        raise CheckpointError('invalid stage name')
    if not 1 <= len(stages) <= MAX_STAGES or len(set(stages)) != len(stages):
        raise CheckpointError('invalid or duplicate stage set')
    return set(stages)


def _open_directory(path, create=False):
    """Walk from / using directory descriptors; no component follows a symlink."""
    path = Path(os.path.abspath(os.fspath(path)))
    # macOS owns these fixed filesystem aliases (including tempfile's /var
    # prefix). Expand only the documented target, then use the same no-follow
    # walk for every application-controlled component.
    if sys.platform == 'darwin' and len(path.parts) > 1 and path.parts[1] in ('var', 'tmp', 'etc'):
        alias = '/' + path.parts[1]
        if os.path.islink(alias):
            target = '/private/' + path.parts[1]
            if os.path.normpath(os.path.join('/', os.readlink(alias))) != target:
                raise CheckpointError('unsafe system directory alias')
            path = Path(target).joinpath(*path.parts[2:])
    descriptor = os.open('/', os.O_RDONLY | _DIRECTORY)
    try:
        for component in path.parts[1:]:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return path, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular(directory, name, flags=os.O_RDONLY, mode=0o600):
    # A corrupt snapshot can contain a FIFO/device. Avoid blocking on open
    # before fstat has a chance to reject it; regular file I/O is unaffected.
    descriptor = os.open(name, flags | _NOFOLLOW | os.O_NONBLOCK, mode, dir_fd=directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise CheckpointError('checkpoint entry must be a regular file')
    return descriptor


def _write_file(directory, name, data):
    descriptor = _open_regular(directory, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, 'wb', closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_file(directory, name, limit, expected_size=None):
    descriptor = _open_regular(directory, name)
    try:
        size = os.fstat(descriptor).st_size
        if size > limit or expected_size is not None and size != expected_size:
            raise CheckpointError('checkpoint file size mismatch or limit exceeded')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            data = stream.read(size + 1)
        if len(data) != size:
            raise CheckpointError('checkpoint file changed while reading')
        return data
    finally:
        os.close(descriptor)


def _receipt(name, data):
    return {'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    manifest: dict
    identity: dict
    session: str
    epoch: int
    sequence: int
    position: int
    parts: dict
    sampler_state: bytes
    journal_state: bytes


class CheckpointStore:
    """Publish ``latest.json`` only after every state file is durable.

    Limits can be raised explicitly for a measured native state size. Concurrent
    saves are serialized, and old complete generations are retained so a
    failed save does not invalidate the previous boundary.
    """

    def __init__(self, root, *, max_document_bytes=MAX_DOCUMENT_BYTES,
                 max_blob_bytes=MAX_BLOB_BYTES, max_total_bytes=MAX_TOTAL_BYTES):
        self.root = Path(os.path.abspath(os.fspath(root)))
        self.max_document_bytes = _positive_limit(max_document_bytes, 'max_document_bytes')
        self.max_blob_bytes = _positive_limit(max_blob_bytes, 'max_blob_bytes')
        self.max_total_bytes = _positive_limit(max_total_bytes, 'max_total_bytes')
        _, descriptor = _open_directory(self.root, create=True)
        os.close(descriptor)

    @contextlib.contextmanager
    def _locked_root(self, exclusive):
        _, directory = _open_directory(self.root)
        lock = None
        try:
            lock = _open_regular(directory, '.lock', os.O_RDWR | os.O_CREAT)
            fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield directory
        finally:
            if lock is not None:
                os.close(lock)
            os.close(directory)

    def save(self, parts, identity, session, epoch, sequence, position,
             sampler_state, journal_state=b''):
        if type(parts) is not dict:
            raise CheckpointError('parts must be a stage dictionary')
        parts = dict(parts)
        _stages(parts)
        identity_bytes = _identity(identity, self.max_document_bytes)
        identity = _parse_json(identity_bytes, self.max_document_bytes)
        _text(session, 'session')
        for key, value in (('epoch', epoch), ('sequence', sequence), ('position', position)):
            _integer(value, key)
        if position > identity['context']:
            raise CheckpointError('position exceeds context')
        blobs = list(parts.values()) + [sampler_state, journal_state]
        if any(type(blob) is not bytes or len(blob) > self.max_blob_bytes for blob in blobs):
            raise CheckpointError('state must be bytes within the blob limit')
        if sum(map(len, blobs)) > self.max_total_bytes:
            raise CheckpointError('total checkpoint state exceeds limit')
        generation = uuid.uuid4().hex
        pending, complete = '.pending-' + generation, 'checkpoint-' + generation
        manifest = {'document_kind': 'backcluster_checkpoint', 'schema_version': 1,
                    'identity': identity,
                    'identity_sha256': hashlib.sha256(identity_bytes).hexdigest(),
                    'session': session, 'epoch': epoch, 'sequence': sequence, 'position': position,
                    'stages': {stage: _receipt('stage-' + stage + '.bin', data)
                               for stage, data in sorted(parts.items())},
                    'sampler': _receipt('sampler.bin', sampler_state),
                    'journal': _receipt('journal.bin', journal_state)}
        manifest_bytes = _json_bytes(manifest, self.max_document_bytes)
        latest_bytes = _json_bytes({'document_kind': 'backcluster_checkpoint_pointer', 'schema_version': 1,
                                    'directory': complete, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()},
                                   self.max_document_bytes)
        with self._locked_root(True) as root:
            os.mkdir(pending, 0o700, dir_fd=root)
            directory = os.open(pending, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=root)
            published = False
            temporary_pointer = '.latest-' + generation
            try:
                for stage, data in sorted(parts.items()):
                    _write_file(directory, 'stage-' + stage + '.bin', data)
                _write_file(directory, 'sampler.bin', sampler_state)
                _write_file(directory, 'journal.bin', journal_state)
                _write_file(directory, 'manifest.json', manifest_bytes)
                os.fsync(directory)
                os.rename(pending, complete, src_dir_fd=root, dst_dir_fd=root)
                published = True
                os.fsync(root)
                _write_file(root, temporary_pointer, latest_bytes)
                os.replace(temporary_pointer, 'latest.json', src_dir_fd=root, dst_dir_fd=root)
                os.fsync(root)
            finally:
                if not published:
                    for entry in os.listdir(directory):
                        os.unlink(entry, dir_fd=directory)
                    os.rmdir(pending, dir_fd=root)
                try:
                    os.unlink(temporary_pointer, dir_fd=root)
                except FileNotFoundError:
                    pass
                os.close(directory)
        return self.root / complete

    def load(self, expected_identity, expected_stages):
        identity_bytes = _identity(expected_identity, self.max_document_bytes)
        stages = _stages(expected_stages)
        try:
            with self._locked_root(False) as root:
                latest = _parse_json(_read_file(root, 'latest.json', self.max_document_bytes), self.max_document_bytes)
                if (type(latest) is not dict or set(latest) != {'document_kind', 'schema_version', 'directory', 'manifest_sha256'}
                        or latest['document_kind'] != 'backcluster_checkpoint_pointer' or type(latest['schema_version']) is not int
                        or latest['schema_version'] != 1 or not isinstance(latest['directory'], str)
                        or not _SNAPSHOT.fullmatch(latest['directory']) or not isinstance(latest['manifest_sha256'], str)
                        or not _SHA256.fullmatch(latest['manifest_sha256'])):
                    raise CheckpointError('invalid checkpoint pointer')
                directory = os.open(latest['directory'], os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=root)
                try:
                    data = _read_file(directory, 'manifest.json', self.max_document_bytes)
                    if hashlib.sha256(data).hexdigest() != latest['manifest_sha256']:
                        raise CheckpointError('manifest checksum mismatch')
                    manifest = _parse_json(data, self.max_document_bytes)
                    required = {'document_kind', 'schema_version', 'identity', 'identity_sha256', 'session',
                                'epoch', 'sequence', 'position', 'stages', 'sampler', 'journal'}
                    if (type(manifest) is not dict or set(manifest) != required or manifest['document_kind'] != 'backcluster_checkpoint'
                            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1):
                        raise CheckpointError('invalid checkpoint manifest')
                    actual_identity = _identity(manifest['identity'], self.max_document_bytes)
                    if actual_identity != identity_bytes or manifest['identity_sha256'] != hashlib.sha256(actual_identity).hexdigest():
                        raise CheckpointError('checkpoint identity mismatch')
                    _text(manifest['session'], 'session')
                    for key in ('epoch', 'sequence', 'position'):
                        _integer(manifest[key], key)
                    if manifest['position'] > manifest['identity']['context']:
                        raise CheckpointError('position exceeds context')
                    if type(manifest['stages']) is not dict or _stages(manifest['stages']) != stages:
                        raise CheckpointError('checkpoint stage set mismatch')
                    entries = [('stage-' + stage + '.bin', manifest['stages'][stage]) for stage in sorted(stages)]
                    entries += [('sampler.bin', manifest['sampler']), ('journal.bin', manifest['journal'])]
                    if set(os.listdir(directory)) != {'manifest.json'} | {name for name, _ in entries}:
                        raise CheckpointError('unexpected or missing checkpoint files')
                    total = 0
                    for name, receipt in entries:
                        if (type(receipt) is not dict or set(receipt) != {'file', 'bytes', 'sha256'}
                                or receipt['file'] != name or type(receipt['bytes']) is not int
                                or not 0 <= receipt['bytes'] <= self.max_blob_bytes
                                or not isinstance(receipt['sha256'], str) or not _SHA256.fullmatch(receipt['sha256'])):
                            raise CheckpointError('invalid state receipt')
                        total += receipt['bytes']
                    if total > self.max_total_bytes:
                        raise CheckpointError('total checkpoint state exceeds limit')
                    states = {}
                    for name, receipt in entries:
                        state = _read_file(directory, name, self.max_blob_bytes, receipt['bytes'])
                        if hashlib.sha256(state).hexdigest() != receipt['sha256']:
                            raise CheckpointError('state checksum mismatch')
                        states[name] = state
                    return LoadedCheckpoint(self.root / latest['directory'], manifest, manifest['identity'],
                                            manifest['session'], manifest['epoch'], manifest['sequence'], manifest['position'],
                                            {stage: states['stage-' + stage + '.bin'] for stage in stages},
                                            states['sampler.bin'], states['journal.bin'])
                finally:
                    os.close(directory)
        except FileNotFoundError:
            # Missing latest is a normal empty store; a missing referenced
            # generation/file is corruption, not a reason to restore a subset.
            with self._locked_root(False) as root:
                try:
                    os.stat('latest.json', dir_fd=root, follow_symlinks=False)
                except FileNotFoundError:
                    raise FileNotFoundError('no complete checkpoint exists') from None
            raise CheckpointError('incomplete checkpoint') from None
        except (OSError, ValueError, TypeError) as error:
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointError(str(error)) from error


class TranscriptJournal:
    """SQLite transcript and durable, at-most-once JSON tool result journal."""

    def __init__(self, path, *, max_document_bytes=MAX_DOCUMENT_BYTES, max_records=100000, create=True):
        self.path = Path(os.path.abspath(os.fspath(path)))
        self.max_document_bytes = _positive_limit(max_document_bytes, 'max_document_bytes')
        self.max_records = _positive_limit(max_records, 'max_records')
        self._lock = threading.RLock()
        _, directory = _open_directory(self.path.parent, create=create)
        try:
            descriptor = _open_regular(directory, self.path.name, os.O_RDWR | (os.O_CREAT if create else 0))
            os.close(descriptor)
            # SQLite creates these itself. Refuse pre-existing links before it
            # opens any auxiliary file in the private journal directory.
            for suffix in ('-wal', '-shm', '-journal'):
                try:
                    info = os.stat(self.path.name + suffix, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('unsafe journal auxiliary file')
            os.fsync(directory)
        finally:
            os.close(directory)
        self._db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
        try:
            version = self._db.execute('PRAGMA user_version').fetchone()[0]
            if version not in ((0, 1, 2) if create else (1, 2)):
                raise ValueError('unsupported or uninitialized transcript journal version')
            self._db.execute('PRAGMA journal_mode=WAL')
            self._db.execute('PRAGMA synchronous=FULL')
            self._db.execute('PRAGMA trusted_schema=OFF')
            self._db.executescript('''
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
                    sha256 TEXT NOT NULL, created_ns INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY, tool TEXT NOT NULL, arguments TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('running','completed')),
                    result TEXT, result_sha256 TEXT, started_ns INTEGER NOT NULL, completed_ns INTEGER,
                    CHECK((status = 'running' AND result IS NULL AND result_sha256 IS NULL AND completed_ns IS NULL)
                       OR (status = 'completed' AND result IS NOT NULL AND result_sha256 IS NOT NULL AND completed_ns IS NOT NULL)));
                CREATE TABLE IF NOT EXISTS reconciliations (
                    call_id TEXT PRIMARY KEY REFERENCES calls(call_id), evidence TEXT NOT NULL,
                    sha256 TEXT NOT NULL);
                PRAGMA user_version=2;
                COMMIT;
            ''')
            _, parent = _open_directory(self.path.parent)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except BaseException:
            self._db.close()
            raise

    def close(self):
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextlib.contextmanager
    def _transaction(self):
        with self._lock:
            self._db.execute('BEGIN IMMEDIATE')
            try:
                yield
                self._db.execute('COMMIT')
            except BaseException:
                self._db.execute('ROLLBACK')
                raise

    def _capacity(self):
        count = self._db.execute('SELECT (SELECT COUNT(*) FROM messages) + (SELECT COUNT(*) FROM calls)').fetchone()[0]
        if count >= self.max_records:
            raise ValueError('journal record limit exceeded')

    @contextlib.contextmanager
    def _executors(self, exclusive=False):
        # A running callback must finish (or its process must exit) before an
        # operator can assert its outcome. Shared locks allow unrelated tools
        # to execute concurrently; the exclusive reconciliation lock never waits.
        _, directory = _open_directory(self.path.parent)
        descriptor = None
        try:
            descriptor = _open_regular(directory, self.path.name + '.executors.lock', os.O_RDWR | os.O_CREAT)
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ToolExecutionIndeterminate('journal has active executors or reconciliation; stop them before retrying') from None
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory)

    def record_message(self, role, content):
        _text(role, 'message role', 64)
        data = _json_bytes(content, self.max_document_bytes)
        with self._transaction():
            self._capacity()
            cursor = self._db.execute('INSERT INTO messages(role,content,sha256,created_ns) VALUES(?,?,?,?)',
                                      (role, data.decode(), hashlib.sha256(data).hexdigest(), time.time_ns()))
            return cursor.lastrowid

    def messages(self):
        with self._lock:
            rows = self._db.execute('SELECT id,role,content,sha256,created_ns FROM messages ORDER BY id').fetchall()
            result = []
            for identifier, role, content, checksum, created in rows:
                data = content.encode('utf-8')
                if hashlib.sha256(data).hexdigest() != checksum:
                    raise ValueError('transcript checksum mismatch')
                result.append({'id': identifier, 'role': role, 'content': _parse_json(data, self.max_document_bytes),
                               'created_ns': created})
            return result

    def _call(self, call_id):
        row = self._db.execute('SELECT tool,arguments,fingerprint,status,result,result_sha256,started_ns,completed_ns '
                               'FROM calls WHERE call_id=?', (call_id,)).fetchone()
        if row is None:
            return None
        tool, arguments, fingerprint, status, result, result_hash, started, completed = row
        args = _parse_json(arguments.encode(), self.max_document_bytes)
        expected = hashlib.sha256(_json_bytes({'tool': tool, 'arguments': args}, self.max_document_bytes)).hexdigest()
        if fingerprint != expected or status not in ('running', 'completed'):
            raise ValueError('tool receipt checksum or status mismatch')
        value = None
        if status == 'completed':
            if result is None or hashlib.sha256(result.encode()).hexdigest() != result_hash:
                raise ValueError('tool result checksum mismatch')
            value = _parse_json(result.encode(), self.max_document_bytes)
        reconciled = self._db.execute('SELECT evidence,sha256 FROM reconciliations WHERE call_id=?', (call_id,)).fetchone()
        evidence = None
        if reconciled is not None:
            data = reconciled[0].encode()
            if status != 'completed' or hashlib.sha256(data).hexdigest() != reconciled[1]:
                raise ValueError('tool reconciliation checksum or status mismatch')
            evidence = _parse_json(data, self.max_document_bytes)
            _text(evidence, 'reconciliation evidence', self.max_document_bytes)
        return {'call_id': call_id, 'tool': tool, 'arguments': args, 'status': status, 'result': value,
                'started_ns': started, 'completed_ns': completed, 'reconciliation_evidence': evidence}

    def get_call(self, call_id):
        _text(call_id, 'tool call ID')
        with self._lock:
            return self._call(call_id)

    def execute(self, call_id, tool, arguments, executor):
        """Run a zero-argument callback once, or return its committed JSON result.

        Callback exceptions, process death, and a failure to commit its result
        leave the record running. Subsequent attempts always raise indeterminate
        instead of calling the executor again. IDs must persist across re-prefill.
        """
        with self._executors():
            return self._execute(call_id, tool, arguments, executor)

    def _execute(self, call_id, tool, arguments, executor):
        _text(call_id, 'tool call ID')
        _text(tool, 'tool name', 128)
        if not callable(executor):
            raise ValueError('executor must be callable')
        args = _json_bytes(arguments, self.max_document_bytes)
        fingerprint = hashlib.sha256(_json_bytes({'tool': tool, 'arguments': _parse_json(args, self.max_document_bytes)},
                                                 self.max_document_bytes)).hexdigest()
        with self._transaction():
            previous = self._call(call_id)
            if previous is not None:
                if previous['tool'] != tool or _json_bytes(previous['arguments'], self.max_document_bytes) != args:
                    raise ValueError('tool call ID reused with different request')
                if previous['status'] == 'running':
                    raise ToolExecutionIndeterminate('tool call ' + call_id + ' may already have executed; reconcile its result')
                return previous['result']
            self._capacity()
            self._db.execute('INSERT INTO calls(call_id,tool,arguments,fingerprint,status,started_ns) VALUES(?,?,?,?,?,?)',
                             (call_id, tool, args.decode(), fingerprint, 'running', time.time_ns()))
        # The FULL-synchronous transaction is committed before any side effect.
        result = executor()
        data = _json_bytes(result, self.max_document_bytes)
        with self._transaction():
            self._db.execute("UPDATE calls SET status='completed',result=?,result_sha256=?,completed_ns=? WHERE call_id=? AND status='running'",
                             (data.decode(), hashlib.sha256(data).hexdigest(), time.time_ns(), call_id))
            if self._db.execute('SELECT changes()').fetchone()[0] != 1:
                raise ToolExecutionIndeterminate('tool result cannot be committed')
        return _parse_json(data, self.max_document_bytes)

    def reconcile(self, call_id, tool, arguments, result, *, evidence):
        """Commit an independently verified outcome, without invoking any tool.

        Stop all journal executors first and inspect the actual side effect.
        An unknown outcome must stay running. Request identity, result and
        evidence are immutable; repeating the exact reconciliation is safe.
        """
        _text(call_id, 'tool call ID')
        _text(tool, 'tool name', 128)
        _text(evidence, 'reconciliation evidence', self.max_document_bytes)
        if not evidence.strip():
            raise ValueError('reconciliation evidence must describe the verified outcome')
        args = _json_bytes(arguments, self.max_document_bytes)
        data = _json_bytes(result, self.max_document_bytes)
        proof = _json_bytes(evidence, self.max_document_bytes)
        with self._executors(exclusive=True), self._transaction():
            previous = self._call(call_id)
            if previous is None:
                raise ValueError('tool call does not exist')
            if previous['tool'] != tool or _json_bytes(previous['arguments'], self.max_document_bytes) != args:
                raise ValueError('tool call ID reused with different request')
            if previous['status'] == 'completed':
                if (previous['reconciliation_evidence'] == evidence and
                        _json_bytes(previous['result'], self.max_document_bytes) == data):
                    return previous['result']
                raise ValueError('completed tool call cannot be reconciled or changed')
            self._db.execute("UPDATE calls SET status='completed',result=?,result_sha256=?,completed_ns=? WHERE call_id=?",
                             (data.decode(), hashlib.sha256(data).hexdigest(), time.time_ns(), call_id))
            self._db.execute('INSERT INTO reconciliations(call_id,evidence,sha256) VALUES(?,?,?)',
                             (call_id, proof.decode(), hashlib.sha256(proof).hexdigest()))
        return _parse_json(data, self.max_document_bytes)

    def snapshot(self):
        """Export a consistent audit snapshot, never a license to replay tools.

        The live SQLite journal must survive inference rollback. Replacing it
        with an older checkpoint would discard newer tool execution receipts.
        """
        with self._transaction():
            calls = [self._call(row[0]) for row in self._db.execute('SELECT call_id FROM calls ORDER BY started_ns,call_id').fetchall()]
            return _json_bytes({'document_kind': 'backcluster_transcript', 'schema_version': 2,
                                'messages': self.messages(), 'calls': calls}, self.max_document_bytes)
