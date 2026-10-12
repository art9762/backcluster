"""Portable transaction harness for stage adapters, with a bounded socket transport.

This is a software correctness harness, not a model inference implementation.
Production adapters must implement snapshot/restore for ALL KV and recurrent
state. The native SPT inference protocol is deliberately a separate protocol.
"""
import hashlib
import json
import math
import re
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass

PROTOCOL_VERSION = 1
MAX_METADATA = 64 * 1024
MAX_PAYLOAD = 16 * 1024 * 1024
FRAME = struct.Struct('!4sIII')
MAGIC = b'BCTX'
ALIAS = re.compile(r'[a-z][a-z0-9_-]{0,31}')
SESSION = re.compile(r'[A-Za-z0-9_-]{1,64}')


class RuntimeErrorCode(ValueError):
    """Stable error code; no remote exception, prompt, path or tensor is exposed."""


def _integer(value, low=0, high=2**63 - 1):
    return type(value) is int and low <= value <= high


def _sha(value, width=64):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{%d}' % width, value) is not None


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeErrorCode('duplicate_field')
            result[key] = value
        return result
    def constant(_):
        raise RuntimeErrorCode('nonfinite_json')
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):
        raise RuntimeErrorCode('invalid_json') from None
    if not isinstance(value, dict):
        raise RuntimeErrorCode('invalid_metadata')
    return value


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, RecursionError):
        raise RuntimeErrorCode('invalid_metadata') from None


@dataclass(frozen=True)
class Identity:
    engine_commit: str
    model_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    topology_sha256: str
    state_format: int
    context: int
    hidden_size: int
    dtype: str = 'f32'
    kv_dtype: str = 'f16'
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if not _sha(self.engine_commit, 40) or any(not _sha(getattr(self, name)) for name in
                ('model_sha256', 'tokenizer_sha256', 'template_sha256', 'topology_sha256')):
            raise RuntimeErrorCode('invalid_identity')
        if (not _integer(self.state_format, 1) or not _integer(self.context, 1, 131072)
                or not _integer(self.hidden_size, 1, 131072) or self.dtype not in ('f32', 'f16')
                or self.kv_dtype not in ('f32', 'f16', 'q8_0') or self.protocol_version != PROTOCOL_VERSION
                or type(self.protocol_version) is not int):
            raise RuntimeErrorCode('invalid_identity')


@dataclass(frozen=True)
class Stage:
    alias: str
    layer_start: int
    layer_end: int
    shard_sha256: str

    def __post_init__(self):
        if (not isinstance(self.alias, str) or ALIAS.fullmatch(self.alias) is None
                or not _integer(self.layer_start) or not _integer(self.layer_end, 1)
                or self.layer_start >= self.layer_end or not _sha(self.shard_sha256)):
            raise RuntimeErrorCode('invalid_stage')


def topology_identity(stages, layer_count):
    """Require one exact, contiguous cover, preserving stage order."""
    if not isinstance(stages, (list, tuple)) or not stages or not _integer(layer_count, 1):
        raise RuntimeErrorCode('invalid_topology')
    cursor, aliases = 0, set()
    for stage in stages:
        if not isinstance(stage, Stage) or stage.alias in aliases or stage.layer_start != cursor:
            raise RuntimeErrorCode('invalid_topology')
        aliases.add(stage.alias)
        cursor = stage.layer_end
    if cursor != layer_count:
        raise RuntimeErrorCode('invalid_topology')
    return hashlib.sha256(canonical([asdict(stage) for stage in stages])).hexdigest()


def pack_f32(values):
    values = tuple(values)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise RuntimeErrorCode('invalid_activation')
    try:
        return struct.pack('<%df' % len(values), *values)
    except (OverflowError, struct.error):
        raise RuntimeErrorCode('invalid_activation') from None


def unpack_f32(payload):
    if len(payload) % 4:
        raise RuntimeErrorCode('invalid_activation')
    values = struct.unpack('<%df' % (len(payload) // 4), payload)
    if any(not math.isfinite(v) for v in values):
        raise RuntimeErrorCode('invalid_activation')
    return values


def _activation(identity, payload, tokens):
    size = 4 if identity.dtype == 'f32' else 2
    if (not _integer(tokens, 1, 4096) or not isinstance(payload, bytes)
            or len(payload) != tokens * identity.hidden_size * size or len(payload) > MAX_PAYLOAD):
        raise RuntimeErrorCode('invalid_activation')
    fmt = 'f' if size == 4 else 'e'
    if any(not math.isfinite(v) for v in struct.unpack('<%d%s' % (len(payload) // size, fmt), payload)):
        raise RuntimeErrorCode('invalid_activation')


class ReferenceStage:
    """Deterministic stateful arithmetic adapter for fault tests; no neural model."""
    def __init__(self, offset):
        self.offset = offset
        self.total_tokens = 0
        self.calls = 0

    def reset(self):
        self.total_tokens = 0

    def snapshot(self):
        return struct.pack('<Q', self.total_tokens)

    def restore(self, state):
        if len(state) != 8:
            raise RuntimeErrorCode('invalid_adapter_state')
        self.total_tokens, = struct.unpack('<Q', state)

    def apply(self, payload, tokens, position):
        if position != self.total_tokens:
            raise RuntimeErrorCode('adapter_position_mismatch')
        self.calls += 1
        self.total_tokens += tokens
        return pack_f32(v + self.offset for v in unpack_f32(payload))


class StageWorker:
    """One active session, transactional activation/state boundary, reconnect safe.

    An adapter is an object with reset(), snapshot()->bytes, restore(bytes), and
    apply(payload:bytes, token_count:int, position:int)->bytes. Snapshot includes
    the adapter's complete state. An incomplete adapter snapshot is not safe.
    """
    def __init__(self, identity, stage, adapter):
        self.identity, self.stage, self.adapter = identity, stage, adapter
        self.session, self.epoch, self.sequence, self.position = None, 0, 0, 0
        self.pending = None
        self.receipt = None
        self.cancelled = False
        self.lock = threading.RLock()

    def status(self):
        return {'session': self.session, 'epoch': self.epoch, 'sequence': self.sequence,
                'position': self.position, 'cancelled': self.cancelled,
                'pending': self.pending is not None, 'stage': self.stage.alias}

    def _session(self, meta):
        if meta.get('session') != self.session or type(meta.get('epoch')) is not int or meta['epoch'] != self.epoch:
            raise RuntimeErrorCode('session_epoch_mismatch')
        if self.cancelled:
            raise RuntimeErrorCode('session_cancelled')

    def _snapshot(self):
        if self.pending is not None:
            raise RuntimeErrorCode('uncommitted_state')
        state = self.adapter.snapshot()
        if not isinstance(state, bytes) or len(state) > MAX_PAYLOAD:
            raise RuntimeErrorCode('invalid_adapter_state')
        return state

    def call(self, meta, payload=b''):
        with self.lock:
            if not isinstance(meta, dict) or not isinstance(payload, bytes) or len(payload) > MAX_PAYLOAD:
                raise RuntimeErrorCode('invalid_request')
            op = meta.get('op')
            if op == 'hello':
                try:
                    remote_identity = Identity(**meta.get('identity', {}))
                    remote_stage = Stage(**meta.get('stage', {}))
                except (TypeError, RuntimeErrorCode):
                    raise RuntimeErrorCode('compatibility_mismatch') from None
                if payload or remote_identity != self.identity or remote_stage != self.stage:
                    raise RuntimeErrorCode('compatibility_mismatch')
                return dict(self.status(), identity=asdict(self.identity), stage_spec=asdict(self.stage)), b''
            if op == 'begin':
                session, epoch = meta.get('session'), meta.get('epoch')
                if (payload or not isinstance(session, str) or SESSION.fullmatch(session) is None
                        or not _integer(epoch, 1) or epoch <= self.epoch):
                    raise RuntimeErrorCode('invalid_epoch')
                if self.session and not self.cancelled:
                    raise RuntimeErrorCode('active_session')
                self.adapter.reset()
                self.session, self.epoch = session, epoch
                self.sequence, self.position, self.pending, self.receipt, self.cancelled = 0, 0, None, None, False
                return self.status(), b''
            self._session(meta)
            if op == 'status':
                if payload:
                    raise RuntimeErrorCode('unexpected_payload')
                return self.status(), b''
            if op == 'forward':
                sequence, position, tokens = (meta.get(k) for k in ('sequence', 'position', 'tokens'))
                _activation(self.identity, payload, tokens)
                if not _integer(sequence) or not _integer(position) or position + tokens > self.identity.context:
                    raise RuntimeErrorCode('invalid_position')
                fingerprint = hashlib.sha256(canonical([sequence, position, tokens]) + payload).hexdigest()
                # Retries, including a lost commit ACK, never apply state twice.
                cached = self.pending if self.pending is not None else self.receipt
                if cached and sequence == cached['sequence']:
                    if fingerprint != cached['fingerprint']:
                        raise RuntimeErrorCode('sequence_collision')
                    return cached['reply'].copy(), cached['output']
                if self.pending is not None or sequence != self.sequence or position != self.position:
                    raise RuntimeErrorCode('position_sequence_mismatch')
                before = self._snapshot()
                try:
                    output = self.adapter.apply(payload, tokens, position)
                    _activation(self.identity, output, tokens)
                except Exception:
                    self.adapter.restore(before)
                    raise RuntimeErrorCode('stage_compute_failed') from None
                reply = dict(self.status(), sequence=sequence, position=position + tokens, tokens=tokens)
                self.pending = {'sequence': sequence, 'fingerprint': fingerprint, 'before': before,
                                'output': output, 'reply': reply, 'tokens': tokens}
                return reply.copy(), output
            if payload and op not in ('restore',):
                raise RuntimeErrorCode('unexpected_payload')
            if op == 'commit':
                sequence = meta.get('sequence')
                if not _integer(sequence):
                    raise RuntimeErrorCode('invalid_sequence')
                if self.receipt and sequence == self.receipt['sequence']:
                    return self.status(), b''
                if self.pending is None or sequence != self.pending['sequence']:
                    raise RuntimeErrorCode('missing_transaction')
                self.receipt, self.pending = self.pending, None
                self.receipt['sealed'] = False
                self.sequence += 1
                self.position += self.receipt['tokens']
                return self.status(), b''
            if op == 'seal':
                if not _integer(meta.get('sequence')) or self.receipt is None or meta['sequence'] != self.receipt['sequence']:
                    raise RuntimeErrorCode('missing_transaction')
                self.receipt['sealed'] = True
                self.receipt.pop('before', None)
                return self.status(), b''
            if op == 'rollback':
                if not _integer(meta.get('sequence')):
                    raise RuntimeErrorCode('invalid_sequence')
                sequence = meta['sequence']
                if self.pending is not None:
                    if sequence != self.pending['sequence']:
                        raise RuntimeErrorCode('missing_transaction')
                    self.adapter.restore(self.pending['before'])
                    self.pending = None
                elif self.receipt is not None and sequence == self.receipt['sequence']:
                    if self.receipt.get('sealed'):
                        raise RuntimeErrorCode('transaction_already_sealed')
                    self.adapter.restore(self.receipt['before'])
                    self.sequence -= 1
                    self.position -= self.receipt['tokens']
                    self.receipt = None
                elif sequence != self.sequence:
                    raise RuntimeErrorCode('missing_transaction')
                return self.status(), b''
            if op == 'cancel':
                if self.pending is not None:
                    self.adapter.restore(self.pending['before'])
                    self.pending = None
                self.cancelled = True
                return self.status(), b''
            if op == 'snapshot':
                return self.status(), self._snapshot()
            if op == 'restore':
                epoch, sequence, position = (meta.get(k) for k in ('new_epoch', 'sequence', 'position'))
                if (not _integer(epoch, self.epoch + 1) or not _integer(sequence)
                        or not _integer(position, 0, self.identity.context) or self.pending is not None):
                    raise RuntimeErrorCode('invalid_restore')
                before = self._snapshot()
                try:
                    self.adapter.restore(payload)
                except Exception:
                    self.adapter.restore(before)
                    raise RuntimeErrorCode('invalid_adapter_state') from None
                self.epoch, self.sequence, self.position, self.receipt = epoch, sequence, position, None
                return self.status(), b''
            raise RuntimeErrorCode('unsupported_operation')


def send_frame(connection, metadata, payload=b''):
    data = canonical(metadata)
    if (len(data) > MAX_METADATA or not isinstance(payload, bytes) or len(payload) > MAX_PAYLOAD
            or not isinstance(metadata, dict)):
        raise RuntimeErrorCode('frame_too_large')
    connection.sendall(FRAME.pack(MAGIC, PROTOCOL_VERSION, len(data), len(payload)) + data + payload)


def receive_frame(connection, deadline):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeErrorCode('timeout')
            connection.settimeout(remaining)
            try:
                block = connection.recv(min(65536, size - len(data)))
            except socket.timeout:
                raise RuntimeErrorCode('timeout') from None
            if not block:
                raise RuntimeErrorCode('unexpected_eof')
            data.extend(block)
        return bytes(data)
    magic, version, metadata_size, payload_size = FRAME.unpack(exact(FRAME.size))
    if magic != MAGIC or version != PROTOCOL_VERSION:
        raise RuntimeErrorCode('incompatible_framing')
    if metadata_size > MAX_METADATA or payload_size > MAX_PAYLOAD:
        raise RuntimeErrorCode('frame_too_large')
    return _json(exact(metadata_size)), exact(payload_size)


class SocketStage:
    """An explicit host harness endpoint. The harness listens on loopback only."""
    def __init__(self, host, port, timeout=3.0):
        if host not in ('127.0.0.1', '::1') or not _integer(port, 1, 65535):
            raise RuntimeErrorCode('loopback_endpoint_required')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise RuntimeErrorCode('invalid_timeout')
        self.host, self.port, self.timeout = host, port, timeout

    def call(self, metadata, payload=b''):
        deadline = time.monotonic() + self.timeout
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as connection:
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                send_frame(connection, metadata, payload)
                reply, data = receive_frame(connection, deadline)
        except OSError:
            raise RuntimeErrorCode('transport_failed') from None
        if reply.get('ok') is not True:
            code = reply.get('error')
            raise RuntimeErrorCode(code if isinstance(code, str) and re.fullmatch(r'[a-z_]{1,64}', code) else 'invalid_reply')
        if not isinstance(reply.get('result'), dict):
            raise RuntimeErrorCode('invalid_reply')
        return reply['result'], data


class WorkerServer:
    """Bounded, stoppable loopback server for integration and adapter fault tests."""
    def __init__(self, worker, timeout=3.0):
        self.worker, self.timeout = worker, timeout
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('127.0.0.1', 0))
        self.socket.listen(4)
        self.socket.settimeout(0.1)
        self.port = self.socket.getsockname()[1]
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def _serve(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with connection:
                try:
                    meta, payload = receive_frame(connection, time.monotonic() + self.timeout)
                    result, output = self.worker.call(meta, payload)
                    send_frame(connection, {'ok': True, 'result': result}, output)
                except RuntimeErrorCode as error:
                    try:
                        send_frame(connection, {'ok': False, 'error': str(error)})
                    except OSError:
                        pass
                except OSError:
                    pass
                except Exception:
                    try:
                        send_frame(connection, {'ok': False, 'error': 'worker_failed'})
                    except OSError:
                        pass

    def __exit__(self, *_):
        self.stopped.set()
        self.socket.close()
        self.thread.join(self.timeout + 1)
        if self.thread.is_alive():
            raise RuntimeErrorCode('worker_shutdown_failed')


class Coordinator:
    """Sequential stages with an all-stage commit boundary and failure poisoning.

    If a worker is lost and rollback cannot be confirmed everywhere, no more
    tokens are accepted; recover from a COMPLETE checkpoint with a new epoch.
    """
    def __init__(self, identity, stages, clients, layer_count):
        if topology_identity(stages, layer_count) != identity.topology_sha256 or len(clients) != len(stages):
            raise RuntimeErrorCode('invalid_topology')
        self.identity, self.stages, self.clients = identity, list(stages), list(clients)
        self.session, self.epoch, self.sequence, self.position = None, 0, 0, 0
        self.poisoned = False
        self.cancelled = False
        self.trace = []
        self.lock = threading.RLock()

    def _meta(self, op, **fields):
        return dict(op=op, session=self.session, epoch=self.epoch, **fields)

    def _watermark(self, reply, sequence, position):
        return (isinstance(reply, dict) and _integer(reply.get('sequence')) and _integer(reply.get('position'))
                and _integer(reply.get('epoch'), 1) and reply['sequence'] == sequence
                and reply['position'] == position and reply['epoch'] == self.epoch
                and reply.get('session') == self.session)

    def begin(self, session, epoch=1):
        with self.lock:
            # Check every identity BEFORE mutating any worker.
            for stage, client in zip(self.stages, self.clients):
                client.call({'op': 'hello', 'identity': asdict(self.identity), 'stage': asdict(stage)})
            self.session, self.epoch = session, epoch
            started = []
            try:
                for client in self.clients:
                    started.append(client)
                    reply, _ = client.call(self._meta('begin'))
                    if not self._watermark(reply, 0, 0):
                        raise RuntimeErrorCode('invalid_begin_reply')
            except Exception:
                self.poisoned = True
                for client in started:
                    try:
                        client.call(self._meta('cancel'))
                    except Exception:
                        pass
                raise
            self.sequence, self.position, self.poisoned, self.cancelled = 0, 0, False, False

    def forward(self, payload, tokens):
        with self.lock:
            if self.session is None or self.poisoned or self.cancelled:
                raise RuntimeErrorCode('coordinator_unavailable')
            _activation(self.identity, payload, tokens)
            sequence, position = self.sequence, self.position
            touched = []
            try:
                for stage, client in zip(self.stages, self.clients):
                    touched.append(client)  # include a request whose reply may be lost
                    start = time.perf_counter_ns()
                    reply, payload = client.call(self._meta('forward', sequence=sequence, position=position, tokens=tokens), payload)
                    if (not self._watermark(reply, sequence, position + tokens)
                            or not _integer(reply.get('tokens'), 1) or reply['tokens'] != tokens):
                        raise RuntimeErrorCode('invalid_stage_reply')
                    _activation(self.identity, payload, tokens)
                    self.trace.append({'stage': stage.alias, 'sequence': sequence, 'tokens': tokens,
                                       'elapsed_ns': time.perf_counter_ns() - start})
                for client in self.clients:
                    reply, _ = client.call(self._meta('commit', sequence=sequence))
                    if not self._watermark(reply, sequence + 1, position + tokens):
                        raise RuntimeErrorCode('invalid_commit_reply')
            except Exception:
                rolled_back = True
                for client in touched:
                    try:
                        reply, _ = client.call(self._meta('rollback', sequence=sequence))
                        if not self._watermark(reply, sequence, position):
                            rolled_back = False
                    except Exception:
                        rolled_back = False
                self.poisoned = not rolled_back
                raise RuntimeErrorCode('distributed_step_failed' if rolled_back else 'distributed_state_uncertain') from None
            self.sequence += 1
            self.position += tokens
            try:
                for client in self.clients:
                    reply, _ = client.call(self._meta('seal', sequence=sequence))
                    if not self._watermark(reply, self.sequence, self.position):
                        raise RuntimeErrorCode('invalid_seal_reply')
            except Exception:
                # Some workers may already have sealed the commit. Rolling
                # back now would itself create inconsistent stage watermarks.
                self.poisoned = True
                raise RuntimeErrorCode('distributed_commit_unconfirmed') from None
            return payload

    def cancel(self):
        with self.lock:
            errors = False
            for client in self.clients:
                try:
                    reply, _ = client.call(self._meta('cancel'))
                    if not self._watermark(reply, self.sequence, self.position) or reply.get('cancelled') is not True:
                        errors = True
                except Exception:
                    errors = True
            self.cancelled = True
            self.poisoned |= errors
            if errors:
                raise RuntimeErrorCode('cancellation_unconfirmed')

    def checkpoint(self, store, sampler_state, journal_state=b''):
        with self.lock:
            if self.poisoned or self.cancelled or self.session is None:
                raise RuntimeErrorCode('coordinator_unavailable')
            parts = {}
            for stage, client in zip(self.stages, self.clients):
                reply, state = client.call(self._meta('snapshot'))
                if not self._watermark(reply, self.sequence, self.position):
                    raise RuntimeErrorCode('checkpoint_position_mismatch')
                parts[stage.alias] = state
            return store.save(parts, asdict(self.identity), self.session, self.epoch, self.sequence,
                              self.position, sampler_state, journal_state)

    def restore(self, store):
        with self.lock:
            checkpoint = store.load(asdict(self.identity), [stage.alias for stage in self.stages])
            if self.session != checkpoint.session or self.cancelled:
                raise RuntimeErrorCode('restore_session_mismatch')
            try:
                # Read identity-bound status without using the old epoch. A prior
                # interrupted restore may have moved only some participants.
                statuses = []
                for stage, client in zip(self.stages, self.clients):
                    reply, _ = client.call({'op': 'hello', 'identity': asdict(self.identity), 'stage': asdict(stage)})
                    if reply.get('session') != self.session or reply.get('cancelled') or not _integer(reply.get('epoch'), 1):
                        raise RuntimeErrorCode('restore_session_mismatch')
                    statuses.append(reply)
                new_epoch = max(self.epoch, checkpoint.epoch, *(reply['epoch'] for reply in statuses)) + 1
                for stage, client, reply in zip(self.stages, self.clients, statuses):
                    metadata = dict(session=self.session, epoch=reply['epoch'])
                    if reply.get('pending'):
                        client.call(dict(metadata, op='rollback', sequence=reply['sequence']))
                    restored, _ = client.call(dict(metadata, op='restore', new_epoch=new_epoch, sequence=checkpoint.sequence,
                                                   position=checkpoint.position), checkpoint.parts[stage.alias])
                    if (restored.get('session') != self.session or type(restored.get('epoch')) is not int
                            or restored['epoch'] != new_epoch or type(restored.get('sequence')) is not int
                            or restored['sequence'] != checkpoint.sequence or type(restored.get('position')) is not int
                            or restored['position'] != checkpoint.position):
                        raise RuntimeErrorCode('invalid_restore_reply')
            except Exception:
                self.poisoned = True
                raise RuntimeErrorCode('checkpoint_restore_incomplete') from None
            self.epoch, self.sequence, self.position = new_epoch, checkpoint.sequence, checkpoint.position
            self.poisoned = False
            return checkpoint


def runtime_selftest():
    """Exercise actual sockets and lifecycle without weights or Apple hardware.

    Returns an explicit software-only receipt. It neither launches inference
    nor claims that the native SPT worker, Apple GPU or numerical model passed.
    """
    from contextlib import ExitStack
    from tempfile import TemporaryDirectory
    from .checkpoints import CheckpointStore

    stages = [Stage(alias, i, i + 1, str(i + 1) * 64) for i, alias in enumerate(('mac', 'ipad', 'iphone'))]
    identity = Identity('a' * 40, 'b' * 64, 'c' * 64, 'd' * 64, topology_identity(stages, 3), 1, 32, 2)
    adapters = [ReferenceStage(i + 1) for i in range(3)]
    workers = [StageWorker(identity, stage, adapter) for stage, adapter in zip(stages, adapters)]

    class LostCommitAck:
        def __init__(self, client):
            self.client, self.fail = client, False
        def call(self, meta, payload=b''):
            result = self.client.call(meta, payload)
            if self.fail and meta.get('op') == 'commit':
                self.fail = False
                raise RuntimeErrorCode('transport_failed')
            return result

    checks = []
    with ExitStack() as stack, TemporaryDirectory(prefix='backcluster-runtime-') as directory:
        servers = [stack.enter_context(WorkerServer(worker)) for worker in workers]
        clients = [SocketStage('127.0.0.1', server.port) for server in servers]
        failure = LostCommitAck(clients[1])
        clients[1] = failure
        coordinator = Coordinator(identity, stages, clients, 3)
        coordinator.begin('software-selftest', 1)
        output = coordinator.forward(pack_f32([1, 2, 3, 4]), 2)
        checks.append({'name': 'three_stage_socket_forward', 'ok': unpack_f32(output) == (7., 8., 9., 10.)})
        # Same sequence after a lost ACK is answered from the receipt.
        calls = adapters[0].calls
        result, _ = clients[0].call(coordinator._meta('forward', sequence=0, position=0, tokens=2), pack_f32([1, 2, 3, 4]))
        checks.append({'name': 'idempotent_sequence_retry', 'ok': adapters[0].calls == calls and result['position'] == 2})
        store = CheckpointStore(directory)
        coordinator.checkpoint(store, b'sampler-state')
        failure.fail = True
        try:
            coordinator.forward(pack_f32([0, 0]), 1)
            rollback_ok = False
        except RuntimeErrorCode as error:
            rollback_ok = str(error) == 'distributed_step_failed' and not coordinator.poisoned
        checks.append({'name': 'lost_commit_ack_rolls_back_all_stages',
                       'ok': rollback_ok and all(adapter.total_tokens == 2 for adapter in adapters)})
        replay_input = pack_f32([2, 5])
        expected = coordinator.forward(replay_input, 1)
        saved = coordinator.restore(store)
        actual = coordinator.forward(replay_input, 1)
        checks.append({'name': 'complete_checkpoint_replay', 'ok': expected == actual and saved.sampler_state == b'sampler-state'
                       and coordinator.epoch == 2 and all(adapter.total_tokens == 3 for adapter in adapters)})
        coordinator.cancel()
        try:
            coordinator.forward(replay_input, 1)
            cancelled = False
        except RuntimeErrorCode:
            cancelled = True
        checks.append({'name': 'all_stage_cancellation', 'ok': cancelled and all(worker.cancelled for worker in workers)})
        trace = coordinator.trace
    return {'document_kind': 'backcluster_runtime_selftest', 'schema_version': 1,
            'software_only': True, 'hardware_validated': False, 'inference_validated': False,
            'adapter': 'deterministic_reference_arithmetic', 'protocol': 'BCTX',
            'ok': all(check['ok'] for check in checks), 'checks': checks, 'trace': trace}
