"""Local HTTP fixtures verify transport, evidence and comparison rejection."""
import contextlib
import copy
import hashlib
import http.server
import json
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.benchmark import (HASH_FORMAT, compare_benchmarks, completion,
                                  run_bench, stream_chat)

LOCK = {'sha256': 'a' * 64, 'engine_commit': 'b' * 40}
MODEL = 'backcluster-' + LOCK['sha256'][:16]


def launch(port):
    return {'document_kind': 'backcluster_launch', 'schema_version': 1, 'mode': 'mac',
        'model_sha256': LOCK['sha256'], 'engine_commit': LOCK['engine_commit'],
        'bind': '127.0.0.1', 'port': port, 'context': 16384, 'draft_enabled': False,
        'remote_workers': [], 'argv': ['/private/server', '--model', '/private/weights.gguf',
        '--host', '127.0.0.1', '--port', str(port), '--alias', MODEL, '--ctx-size', '16384',
        '--parallel', '1', '--spec-type', 'none']}


def event(packet):
    return b'data: ' + json.dumps(packet).encode() + b'\n\n'


def delta(content='ok', **fields):
    return {'model': MODEL, 'choices': [{'index': 0, 'delta': dict(content=content, **fields)}]}


def finished(reason='stop'):
    return {'model': MODEL, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': reason}]}


@contextlib.contextmanager
def endpoint(raw=None, header_drip=False, packets=None, alias=MODEL):
    state = {'requests': [], 'client_closed': threading.Event()}
    raw = raw if raw is not None else b''.join(event(p) for p in packets or [delta(), finished(),
        {'model': MODEL, 'choices': [], 'usage': {'completion_tokens': 2, 'prompt_tokens': 8}}]) + b'data: [DONE]\n\n'
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            body = json.dumps({'object': 'list', 'data': [{'id': alias}]}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            state['requests'].append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            try:
                if header_drip:
                    self.connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: ')
                    for i in range(25):
                        self.connection.sendall(b' '); time.sleep(.03)
                    return
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                # Fragment inside UTF-8 and JSON/SSE boundaries, not only events.
                for offset in range(0, len(raw), 11):
                    self.wfile.write(raw[offset:offset + 11]); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state['client_closed'].set()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try: yield server.server_port, state
    finally: server.shutdown(); server.server_close(); thread.join(2)


class StreamTests(unittest.TestCase):
    def test_real_fragmented_sse_multiline_crlf_reasoning_and_usage(self):
        raw = (b': heartbeat\r\n\r\n' +
               b'data: {"model":"' + MODEL.encode() + b'",\r\ndata: "choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}\r\n\r\n' +
               event(delta('привет')) + event(finished()) +
               event({'choices': [], 'usage': {'completion_tokens': 3, 'prompt_tokens': 5}}) + b'data: [DONE]\n\n')
        with endpoint(raw=raw) as (port, state):
            result = stream_chat(port, [{'role': 'user', 'content': 'private'}], MODEL)
        self.assertEqual(result['choices'][0]['content'], 'привет')
        self.assertEqual(result['choices'][0]['reasoning_content'], 'think')
        self.assertEqual(result['completion_tokens'], 3)
        self.assertEqual(result['response_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['response_bytes'], len(raw))
        self.assertEqual(state['requests'][0]['seed'], 0)
        self.assertLessEqual(result['ttft_ns'], result['wall_ns'])

    def test_tools_normalize_fragmentation_and_volatile_ids(self):
        a = [delta('', tool_calls=[{'index': 0, 'id': 'one', 'function': {'name': 'read_file', 'arguments': '{"path":"x"}'}}]), finished('tool_calls')]
        b = [delta('', tool_calls=[{'index': 0, 'id': 'other', 'function': {'name': 'read_', 'arguments': '{"path":'}}]),
             delta('', tool_calls=[{'index': 0, 'function': {'name': 'file', 'arguments': '"x"}'}}]), finished('tool_calls')]
        with endpoint(packets=a) as (port, _): left = completion(port, 'private', MODEL)
        with endpoint(packets=b) as (port, _): right = completion(port, 'private', MODEL)
        self.assertEqual(left['output_sha256'], right['output_sha256'])
        self.assertNotEqual(left['response_sha256'], right['response_sha256'])
        self.assertNotIn('choices', left)

    def test_strict_schema_rejects_mismatch_error_delta_and_duplicate_members(self):
        bad = [event(dict(delta(), model='other')) + b'data: [DONE]\n\n',
               event({'error': {'message': 'failed'}}) + b'data: [DONE]\n\n',
               event({'choices': [{'index': True, 'delta': {'content': 'x'}}]}) + b'data: [DONE]\n\n',
               event({'choices': [{'delta': {'content': ['x']}}]}) + b'data: [DONE]\n\n',
               b'data: {"choices":[],"choices":[]}\n\ndata: [DONE]\n\n',
               event(delta()) + event(finished()) + event(delta('later')) + b'data: [DONE]\n\n',
               event(delta()) + event({'usage': {'completion_tokens': True}}) + b'data: [DONE]\n\n']
        for raw in bad:
            with self.subTest(raw=raw[:80]), endpoint(raw=raw) as (port, _):
                with self.assertRaises(ValueError): completion(port, 'private', MODEL)

    def test_header_drip_is_cut_off_by_absolute_deadline(self):
        with endpoint(header_drip=True) as (port, _):
            start = time.monotonic()
            with self.assertRaises((OSError, ValueError, TimeoutError, http.client.HTTPException)):
                completion(port, 'private', MODEL, timeout=.12)
            elapsed = time.monotonic() - start
        self.assertLess(elapsed, .6)

    def test_incomplete_and_oversized_events_fail(self):
        for raw in [event(delta()), b'data: ' + b'x' * 65537 + b'\n\n']:
            with self.subTest(size=len(raw)), endpoint(raw=raw) as (port, _):
                with self.assertRaises(ValueError): completion(port, 'private', MODEL)

    def test_failed_repeat_preserves_partial_wire_evidence(self):
        raw = event(delta())
        with endpoint(raw=raw) as (port, _):
            report = run_bench(port, 'private', MODEL, LOCK, repeats=2)
        self.assertFalse(report['ok']); self.assertEqual(len(report['samples']), 1)
        failed = report['samples'][0]
        self.assertEqual(failed['response_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(failed['response_bytes'], len(raw))
        self.assertEqual(failed['content_chunks'], 1)
        self.assertGreater(failed['wall_ns'], 0)

    def test_benchmark_without_usage_does_not_invent_tokens(self):
        with endpoint(packets=[delta(), finished()]) as (port, _):
            report = run_bench(port, 'private prompt', MODEL, LOCK, repeats=2)
        self.assertTrue(report['ok'])
        self.assertIsNone(report['samples'][1]['accepted_tokens_s'])
        self.assertIsNone(report['samples'][1]['completion_tokens'])
        self.assertNotIn('private prompt', json.dumps(report))
        self.assertFalse(report['server_identity_verified'])

    def test_receipt_preflight_and_private_runtime_identity(self):
        with endpoint() as (port, _):
            report = run_bench(port, 'private prompt', MODEL, LOCK, repeats=1, launch=launch(port))
        self.assertTrue(report['api_model_identity_verified'])
        self.assertFalse(report['server_identity_verified'])
        self.assertEqual(report['experiment']['context'], 16384)
        self.assertNotIn('/private/', json.dumps(report))
        with endpoint(alias='unrelated') as (port, _):
            with self.assertRaises(ValueError): run_bench(port, 'p', MODEL, LOCK, launch=launch(port))
        with endpoint() as (port, _):
            receipt = launch(port); receipt['argv'].extend(['--parallel', '2'])
            with self.assertRaises(ValueError): run_bench(port, 'p', MODEL, LOCK, launch=receipt)

    def test_cluster_receipt_tracks_build_environment_and_repeated_flags(self):
        with endpoint() as (port, _):
            receipt = launch(port)
            receipt.update(mode='split-decode', engine_build_id='f' * 64,
                environment={'LLAMA_SPLIT_TAIL': '192.0.2.1:50060', 'LLAMA_SPLIT_L': '20'}, remote_workers=[{'alias': 'fixture'}])
            receipt['argv'].extend(['--override-tensor', '^blk.20=CPU', '--override-tensor', '^output=CPU', '--no-repack', '--jinja'])
            report = run_bench(port, 'p', MODEL, LOCK, repeats=1, launch=receipt)
        experiment = report['experiment']
        self.assertEqual(experiment['engine_build_id'], 'f' * 64)
        self.assertEqual(experiment['runtime_environment']['LLAMA_SPLIT_L'], '20')
        self.assertEqual(experiment['runtime_environment']['GGML_METAL_FA_SME'], '0')
        self.assertEqual(experiment['runtime_settings']['--override-tensor'], ['^blk.20=CPU', '^output=CPU'])
        self.assertTrue(experiment['runtime_settings']['--jinja'])
        self.assertNotIn('192.0.2.1', json.dumps(report))


class ComparisonTests(unittest.TestCase):
    def reports(self):
        row = {'status': 'ok', 'ttft_ns': 10, 'wall_ns': 100, 'completion_tokens': 2,
               'output_sha256': 'e' * 64, 'output_hash_format': HASH_FORMAT,
               'response_bytes': 100, 'response_sha256': 'e' * 64, 'finish_reasons': ['stop']}
        report = {'document_kind': 'backcluster_baseline_bench', 'schema_version': 2, 'ok': True,
            'api_model_identity_verified': True,
            'model_sha256': LOCK['sha256'], 'engine_commit': LOCK['engine_commit'], 'prompt_sha256': 'c' * 64,
            'output_token_limit': 256, 'repeats': 5, 'generation_settings': {'temperature': 0, 'seed': 0, 'stream': True},
            'experiment': {'context': 16384, 'local_context': 16384, 'mode': 'mac', 'engine_build_id': 'f' * 64,
                'runtime_settings': {'--parallel': '1'}, 'runtime_environment': {'GGML_METAL_FA_SME': '0'},
                'launch_receipt_sha256': 'a' * 64, 'draft_enabled': False, 'remote_worker_count': 0},
            'samples': [dict(row, sample=i, warmup=i == 0) for i in range(6)],
            'summary': {'median_wall_ns': -10}}
        right = copy.deepcopy(report)
        for sample in right['samples']: sample['wall_ns'] = 50
        return report, right

    def test_recomputes_from_raw_rows_and_keeps_warmup_out(self):
        left, right = self.reports(); left['samples'][0]['wall_ns'] = 100000
        result = compare_benchmarks(left, right)
        self.assertTrue(result['ok']); self.assertEqual(result['observed_wall_ratio'], 2)
        self.assertEqual(result['baseline_wall_ns'], [100] * 5)
        self.assertFalse(result['hardware_validated'])

    def test_identity_settings_and_failed_samples_block_comparison(self):
        for field in ['model_sha256', 'engine_commit', 'prompt_sha256', 'output_token_limit', 'generation_settings']:
            left, right = self.reports(); right[field] = 'different'
            with self.subTest(field=field), self.assertRaises(ValueError): compare_benchmarks(left, right)
        for field in ['context', 'runtime_settings', 'draft_enabled']:
            left, right = self.reports(); right['experiment'][field] = 'different'
            with self.subTest(field=field), self.assertRaises(ValueError): compare_benchmarks(left, right)
        left, right = self.reports(); right['samples'][3]['status'] = 'error'
        with self.assertRaises(ValueError): compare_benchmarks(left, right)
        left, right = self.reports(); right['samples'][2]['wall_ns'] = float('nan')
        with self.assertRaises(ValueError): compare_benchmarks(left, right)

    def test_output_or_token_changes_block_speed_ratio(self):
        for key, value in [('output_sha256', 'd' * 64), ('completion_tokens', 3)]:
            left, right = self.reports(); right['samples'][1][key] = value
            result = compare_benchmarks(left, right)
            self.assertFalse(result['ok']); self.assertIsNone(result['observed_wall_ratio'])

    def test_pair_changes_are_recorded_and_tuning_needs_explicit_policy(self):
        left, right = self.reports()
        right['experiment'].update(mode='split-decode', remote_worker_count=1)
        right['experiment']['runtime_environment']['LLAMA_SPLIT_L'] = '20'
        right['experiment']['runtime_settings']['--no-repack'] = True
        result = compare_benchmarks(left, right)
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['runtime_differences']), 4)
        self.assertFalse(result['single_runtime_setting_changed'])
        left, right = self.reports()
        right['experiment']['runtime_environment']['GGML_METAL_FA_SME'] = '1'
        with self.assertRaises(ValueError): compare_benchmarks(left, right)
        result = compare_benchmarks(left, right, allow_runtime_changes=True)
        self.assertEqual(result['runtime_differences'][0]['setting'], 'runtime_environment.GGML_METAL_FA_SME')
        self.assertTrue(result['single_runtime_setting_changed'])
        right['experiment']['engine_build_id'] = 'e' * 64
        with self.assertRaises(ValueError): compare_benchmarks(left, right, allow_runtime_changes=True)


if __name__ == '__main__': unittest.main()
