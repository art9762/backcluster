"""Artifact corruption, memory/identity gates and real network failure tests."""
import contextlib
import copy
import http.server
import io
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.artifacts import TYPE_SIZES, inspect_gguf, make_lock, verify_lock
from backcluster.benchmark import completion, run_bench
from backcluster.cli import main
from backcluster.devices import collect_profile
from backcluster.linkbench import sample, run_link_bench
from backcluster.planning import admission, make_plan, registry_checks, validate_registry
from backcluster.profiles import normalize_worker
from backcluster.rpc import Endpoint, ProbeError
from backcluster.serving import clean_environment, launch_spec
from test_diagnostics import native_reply

PIN = '9f17fc31cf60753eb6bd6ec075894edab773cb91'


def gguf_fixture(path, *, overlap=False, missing=False, template=True, duplicate=False):
    def string(s):
        b = s.encode(); return struct.pack('<Q', len(b)) + b
    def value(v):
        return struct.pack('<I', 8) + string(v) if isinstance(v, str) else struct.pack('<II', 4, v)
    metadata = {'general.architecture': 'qwen35', 'qwen35.block_count': 2, 'qwen35.context_length': 32768}
    if template: metadata['tokenizer.chat_template'] = '{{ messages }}'
    tensors = ['token_embd.weight', 'blk.0.attn.weight', 'blk.1.attn.weight']
    if missing: tensors.pop()
    if duplicate: tensors[-1] = tensors[1]
    data = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for k, v in metadata.items(): data += string(k) + value(v)
    for i, name in enumerate(tensors):
        data += string(name) + struct.pack('<IQIQ', 1, 8, 0, 0 if overlap else i * 32)
    data += b'\0' * ((-len(data)) % 32) + b'\0' * (len(tensors) * 32)
    path.write_bytes(data)


def worker_profile():
    d = native_reply()
    d['runtime'].update(engine_commit=PIN, control_protocol_version=1)
    d['capabilities'].append('link_bench')
    return normalize_worker(d)


def registry():
    return {'document_kind': 'backcluster_registry', 'schema_version': 1, 'workers': [{
        'alias': 'ipad', 'host': '169.254.10.2', 'control_port': 50061, 'interface': 'en7', # sensitive-ok: synthetic test endpoint
        'expected_model': 'iPad17,1', 'role': 'remote_kv', 'engine_commit': PIN,
        'phone_attn_version': 4, 'tail_protocol_version': 2, 'tail_state_format': 1,
        'control_protocol_version': 1, 'allocation_budget_bytes': 4 * 2**30, 'reserve_bytes': 2**30}]}


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / 'tiny.gguf'; gguf_fixture(self.file)

    def test_content_lock_and_layer_bytes(self):
        lock = make_lock(self.file, 'fixture/model', 'a' * 40, PIN)
        self.assertEqual(lock['gguf']['layer_tensor_bytes'], [32, 32])
        self.assertEqual(lock['gguf']['shared_tensor_bytes'], 32)
        self.assertEqual(lock['gguf']['tensor_bytes'], 96)
        self.assertFalse(lock['hardware_validated'])
        self.assertEqual(verify_lock(lock, self.file, PIN)['sha256'], lock['sha256'])

    def test_same_size_weight_corruption_rejected(self):
        lock = make_lock(self.file, 'fixture/model', 'a' * 40, PIN)
        with self.file.open('r+b') as f: f.seek(-1, 2); f.write(b'X')
        with self.assertRaises(ValueError): verify_lock(lock, self.file, PIN)

    def test_engine_and_metadata_tampering_rejected(self):
        lock = make_lock(self.file, 'fixture/model', 'a' * 40, PIN)
        with self.assertRaises(ValueError): verify_lock(lock, self.file, 'b' * 40)
        lock['gguf']['layer_tensor_bytes'][0] += 1
        with self.assertRaises(ValueError): verify_lock(lock, self.file, PIN)

    def test_truncated_tensor_and_metadata_rejected(self):
        for length in (10, self.file.stat().st_size - 1):
            gguf_fixture(self.file); self.file.write_bytes(self.file.read_bytes()[:length])
            with self.assertRaises(ValueError): inspect_gguf(self.file)

    def test_invalid_tensor_table_rejected(self):
        for option in ('overlap', 'missing', 'duplicate'):
            gguf_fixture(self.file, **{option: True})
            with self.subTest(option=option), self.assertRaises(ValueError): inspect_gguf(self.file)

    def test_huge_metadata_and_unknown_type_rejected(self):
        self.file.write_bytes(b'GGUF' + struct.pack('<IQQQ', 3, 1, 1, 2**63))
        with self.assertRaises(ValueError): inspect_gguf(self.file)

    def test_cli_model_round_trip_and_dry_launch(self):
        lock = Path(self.tmp.name) / 'lock.json'
        self.assertEqual(main(['model', '--file', str(self.file), '--repository', 'fixture/model', '--revision', 'a'*40, '--output', str(lock)]), 0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['model', '--file', str(self.file), '--check', str(lock)]), 0)
            self.assertEqual(main(['serve', '--file', str(self.file), '--lock', str(lock), '--dry-run']), 0)
        spec = launch_spec(json.loads(lock.read_text()), self.file)
        self.assertEqual(spec['bind'], '127.0.0.1')
        self.assertEqual(spec['remote_workers'], [])
        self.assertIn('none', spec['argv'])
        self.assertIn('mmap', spec['argv'])
        self.assertNotIn('--rpc', spec['argv'])

    def test_template_and_context_required(self):
        gguf_fixture(self.file, template=False)
        lock = make_lock(self.file, 'fixture/model', 'a'*40, PIN)
        with self.assertRaises(ValueError): launch_spec(lock, self.file)
        gguf_fixture(self.file)
        lock = make_lock(self.file, 'fixture/model', 'a'*40, PIN)
        with self.assertRaises(ValueError): launch_spec(lock, self.file, 65536)

    def test_no_model_paths_persisted_in_lock(self):
        lock = make_lock(self.file, 'fixture/model', 'a'*40, PIN)
        self.assertNotIn(self.tmp.name, json.dumps(lock))

    def test_inherited_inference_and_proxy_settings_discarded(self):
        with patch.dict(os.environ, {'GGML_METAL_REMOTE': '169.254.10.2', 'HTTP_PROXY': 'secret', 'DYLD_INSERT_LIBRARIES': '/private'}): # sensitive-ok: synthetic test endpoint
            env = clean_environment()
        self.assertNotIn('GGML_METAL_REMOTE', env)
        self.assertNotIn('HTTP_PROXY', env)
        self.assertNotIn('DYLD_INSERT_LIBRARIES', env)
        self.assertEqual(env['GGML_METAL_FA_SME'], '0')


class AdmissionTests(unittest.TestCase):
    def test_registry_rejects_ambiguous_and_non_cable_endpoints(self):
        for key, value in [('host', '192.168.1.2'), ('host', '::1'), ('allocation_budget_bytes', None), ('engine_commit', 'main'), ('control_port', True)]: # sensitive-ok: synthetic rejected LAN endpoint
            r = registry(); r['workers'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_registry(r)
        r = registry(); r['workers'].append(copy.deepcopy(r['workers'][0]))
        with self.assertRaises(ValueError): validate_registry(r)

    def test_per_device_budget_takes_smallest_limit_without_summing(self):
        result = admission(worker_profile(), 4*2**30, 2**30, 3*2**30)
        self.assertTrue(result['ok']); self.assertEqual(result['usable_bytes'], 3*2**30)
        self.assertFalse(admission(worker_profile(), 4*2**30, 2**30, 3*2**30+1)['ok'])

    def test_unknown_memory_and_thermal_refuse_admission(self):
        device = worker_profile()
        for key in device['memory']:
            modified = copy.deepcopy(device); modified['memory'][key] = None
            self.assertFalse(admission(modified, 4*2**30, 2**30, 1)['ok'])
        for thermal in (2, 3, None):
            modified = copy.deepcopy(device); modified['thermal_state'] = thermal
            self.assertFalse(admission(modified, 4*2**30, 2**30, 1)['ok'])

    def test_wrong_worker_and_stale_protocol_refuse_compatibility(self):
        snapshot = collect_profile([])
        snapshot['workers'] = [{'alias': 'ipad', 'status': 'ok', 'device': worker_profile(),
                                'link': {'transport': 'unverified', 'control_round_trip_ns': 1}}]
        self.assertTrue(registry_checks(registry(), snapshot, PIN)[0]['ok'])
        for key in ('engine_commit', 'tail_protocol_version', 'phone_attn_version', 'control_protocol_version'):
            bad = copy.deepcopy(snapshot); bad['workers'][0]['device']['runtime'][key] = None
            self.assertFalse(registry_checks(registry(), bad, PIN)[0]['ok'])
        bad = copy.deepcopy(snapshot); bad['workers'][0]['device']['hardware']['model'] = 'iPhone18,2'
        self.assertFalse(registry_checks(registry(), bad, PIN)[0]['ok'])

    def test_plan_is_not_hardware_authorization(self):
        snapshot = collect_profile([]); snapshot['host']['platform'] = 'macOS'
        snapshot['workers'] = [{'alias': 'ipad', 'status': 'ok', 'device': worker_profile(),
                                'link': {'transport': 'unverified', 'control_round_trip_ns': 1}}]
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'tiny.gguf'; gguf_fixture(file)
            lock = make_lock(file, 'fixture/model', 'a'*40, PIN)
        p = make_plan(lock, snapshot, registry(), PIN, 'remote-kv', 16*2**30, 4*2**30, 16384, [('ipad', 2**30)])
        self.assertTrue(p['worker_admission'][0]['ok'])
        self.assertFalse(p['launch_ready']); self.assertFalse(p['hardware_validated'])
        self.assertNotIn('169.254', json.dumps(p))
        snapshot['captured_at'] = '2020-01-01T00:00:00+00:00'
        self.assertIn('snapshot_stale', make_plan(lock, snapshot, None, PIN, 'mac', 16*2**30, 4*2**30, 16384, [])['blockers'])


@contextlib.contextmanager
def payload_server(bad=False, old=False, stall=False):
    listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen(20); listener.settimeout(.1)
    stop = threading.Event()
    def worker():
        while not stop.is_set():
            try: conn = listener.accept()[0]
            except socket.timeout: continue
            except OSError: break
            with conn:
                conn.settimeout(1)
                try:
                    command = bytearray()
                    while not command.endswith(b'\n'): command.extend(conn.recv(1))
                    size = int(command.split()[-1])
                    if old: conn.sendall(b'{"error":"unknown"}\n'); continue
                    conn.sendall(json.dumps({'document_kind': 'backcluster_link_ready', 'version': 1, 'bytes': size}).encode()+b'\n')
                    if stall: stop.wait(.2); continue
                    payload = bytearray()
                    while len(payload) < size:
                        b = conn.recv(size-len(payload))
                        if not b: break
                        payload.extend(b)
                    conn.sendall(json.dumps({'received_bytes': size}).encode()+b'\n')
                    if bad: payload[0] ^= 1
                    conn.sendall(payload)
                except OSError: pass
    thread = threading.Thread(target=worker, daemon=True); thread.start()
    try: yield Endpoint('ipad', '127.0.0.1', listener.getsockname()[1])
    finally: stop.set(); listener.close(); thread.join(2)


class LinkTests(unittest.TestCase):
    def test_verified_payload_and_raw_summary(self):
        with payload_server() as endpoint:
            result = run_link_bench([endpoint], 2, sizes=(16384,))
        self.assertTrue(result['ok']); self.assertEqual(len(result['samples']), 3)
        self.assertEqual(result['summary'][0]['samples'], 2)
        self.assertNotIn('127.0.0.1', json.dumps(result))

    def test_old_worker_corruption_and_timeout(self):
        for options, error in [({'old': True}, 'unsupported_link_bench'), ({'bad': True}, 'payload_mismatch'), ({'stall': True}, 'timeout')]:
            with self.subTest(options=options), payload_server(**options) as endpoint:
                with self.assertRaisesRegex(ProbeError, error): sample(endpoint, 16384, timeout=.1)

    def test_simultaneous_workers_and_partial_failure(self):
        with payload_server() as a, payload_server(old=True) as b:
            b = Endpoint('iphone', b.host, b.port)
            result = run_link_bench([a, b], 1, sizes=(16384,), concurrent=True)
        self.assertFalse(result['ok'])
        self.assertTrue(any(r['alias'] == 'ipad' and r['status'] == 'ok' for r in result['samples']))
        self.assertTrue(any(r['alias'] == 'iphone' and r['status'] == 'error' for r in result['samples']))

    def test_duplicate_endpoints_and_oversized_transfer_rejected(self):
        endpoint = Endpoint('ipad', '127.0.0.1', 1)
        with self.assertRaises(ValueError): run_link_bench([endpoint, endpoint])
        with self.assertRaises(ValueError): sample(endpoint, 4194305)


@contextlib.contextmanager
def sse_server(truncated=False):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"test"}}]}\n\n')
            self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":4}}\n\n')
            if not truncated: self.wfile.write(b'data: [DONE]\n\n')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try: yield server.server_port
    finally: server.shutdown(); server.server_close(); thread.join(2)


class BenchTests(unittest.TestCase):
    def test_sse_usage_and_no_prompt_storage(self):
        with sse_server() as port:
            result = run_bench(port, 'private prompt', 'fixture', {'sha256': 'a'*64, 'engine_commit': PIN}, 2)
        self.assertTrue(result['ok']); self.assertEqual(result['summary']['samples'], 2)
        self.assertEqual(result['samples'][1]['completion_tokens'], 4)
        self.assertNotIn('private prompt', json.dumps(result))
        self.assertFalse(result['server_identity_verified'])

    def test_incomplete_sse_is_failure_not_a_successful_benchmark(self):
        with sse_server(truncated=True) as port:
            with self.assertRaises(ValueError): completion(port, 'test', 'fixture')

    def test_invalid_lock_is_safe_cli_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'; path.write_text('[]')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['bench', '--lock', str(path), '--prompt', str(path)]), 1)


if __name__ == '__main__': unittest.main()
