"""Regression coverage for merged tooling, coding streams and cache identity."""
import contextlib
import copy
import http.client
import http.server
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.artifacts import load_json, make_lock, validate_lock
from backcluster.benchmark import completion
from backcluster.cli import main
from backcluster.devices import collect_profile
from backcluster.planning import make_plan
from test_preparation import PIN, gguf_fixture

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def stream_server(packets):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for packet in packets:
                self.wfile.write(b'data: ' + json.dumps(packet).encode() + b'\r\n\r\n')
            self.wfile.write(b'data: [DONE]\r\n\r\n')

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.file = Path(self.temp.name) / 'model.gguf'
        gguf_fixture(self.file)
        self.lock = make_lock(self.file, 'fixture/model', 'a' * 40, PIN)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        path = Path(self.temp.name) / 'invalid.json'
        for text in ('{"schema_version": 1, "schema_version": 2}', '{"value": NaN}'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_json(path)

    def test_lock_identity_rejected_before_benchmark(self):
        for key, value in (('sha256', 'x' * 64), ('engine_commit', 'b' * 40), ('schema_version', True), ('bytes', False)):
            bad = copy.deepcopy(self.lock)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_lock(bad, PIN)

    def test_host_budget_is_clamped_and_unknown_memory_blocks(self):
        profile = collect_profile([])
        profile['host']['platform'] = 'macOS'
        profile['host']['memory']['physical_bytes'] = 24 * 2**30
        plan = make_plan(self.lock, profile, None, PIN, 'mac', 48 * 2**30, 8 * 2**30, 16384, [])
        self.assertEqual(plan['assignments'][0]['usable_bytes'], 16 * 2**30)
        self.assertIn('host_budget_exceeds_physical_memory', plan['blockers'])
        profile['host']['memory']['physical_bytes'] = None
        plan = make_plan(self.lock, profile, None, PIN, 'mac', 16 * 2**30, 8 * 2**30, 16384, [])
        self.assertFalse(plan['host_weight_budget_ok'])
        self.assertIn('host_physical_memory_unknown', plan['blockers'])
        with self.assertRaises(ValueError):
            make_plan(self.lock, profile, None, PIN, 'mac', 16 * 2**30, 8 * 2**30, 65536, [])

    def test_readable_doctor_retains_json_receipt(self):
        receipt = Path(self.temp.name) / 'doctor.json'
        with patch('backcluster.devices.platform.system', return_value='Linux'), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['doctor', '--human', '--output', str(receipt)]), 1)
        self.assertIn('[FIX] apple_silicon_host', output.getvalue())
        self.assertIn('Metal toolchain', output.getvalue())
        self.assertFalse(json.loads(receipt.read_text())['ok'])

    def test_build_failure_has_no_traceback_or_command_paths(self):
        with patch('backcluster.cli.build_engine', side_effect=subprocess.CalledProcessError(7, ['/private/compiler'])), contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main(['build']), 1)
        self.assertIn('exit 7', output.getvalue())
        self.assertNotIn('/private', output.getvalue())
        self.assertNotIn('Traceback', output.getvalue())

    def test_installer_is_repeatable_and_protects_existing_commands(self):
        prefix = Path(self.temp.name) / 'bin'
        env = dict(os.environ, BACKCLUSTER_BIN_DIR=str(prefix))
        for _ in range(2):
            result = subprocess.run(['bash', str(ROOT / 'install.sh')], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((prefix / 'backcluster').resolve(), ROOT / 'backcluster')
        (prefix / 'backcluster').unlink()
        (prefix / 'backcluster').write_text('existing command')
        result = subprocess.run(['bash', str(ROOT / 'install.sh')], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((prefix / 'backcluster').read_text(), 'existing command')

    def test_legacy_launcher_preserves_server_failure(self):
        binary = Path(self.temp.name) / 'llama-server'
        binary.write_text('#!/bin/bash\nexit 7\n')
        binary.chmod(0o755)
        env = dict(os.environ, BIN=self.temp.name, MODEL=str(self.file), DRAFT='', PHONE='0',
                   PROXY='0', LOAD_MODE='mmap', CACHE_DIR=str(Path(self.temp.name) / 'cache'), SME='0', MM_SME='0')
        result = subprocess.run(['bash', str(ROOT / 'scripts/serve.sh')], env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_tool_only_stream_and_chunk_independent_output_hash(self):
        def delta(value):
            return {'choices': [{'index': 0, 'delta': value}]}
        whole = [delta({'tool_calls': [{'index': 0, 'id': 'random-a', 'function': {'name': 'read_file', 'arguments': '{"path":"x"}'}}]})]
        split = [delta({'tool_calls': [{'index': 0, 'id': 'random-b', 'function': {'name': 'read_', 'arguments': '{"path":'}}]}),
                 delta({'tool_calls': [{'index': 0, 'function': {'name': 'file', 'arguments': '"x"}'}}]})]
        hashes = []
        for packets in (whole, split):
            with stream_server(packets) as port:
                result = completion(port, 'test', 'fixture')
            self.assertGreater(result['ttft_ns'], 0)
            hashes.append(result['output_sha256'])
        self.assertEqual(hashes[0], hashes[1])

    def test_reasoning_and_content_in_same_packet_both_contribute(self):
        hashes = []
        for reasoning in ('first', 'second'):
            with stream_server([{'choices': [{'delta': {'content': 'answer', 'reasoning_content': reasoning}}]}]) as port:
                hashes.append(completion(port, 'test', 'fixture')['output_sha256'])
        self.assertNotEqual(hashes[0], hashes[1])


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        spec = importlib.util.spec_from_file_location('backcluster_proxy_fixture', ROOT / 'scripts/proxy.py')
        self.proxy = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'argv', ['proxy.py', '--cache', self.temp.name]):
            spec.loader.exec_module(self.proxy)

    def test_tool_calls_are_reconstructed_and_rewritten_calls_rejected(self):
        packet = {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function',
                                                        'function': {'name': 'read_file', 'arguments': '{}'}}]}}]}
        raw = ('data: ' + json.dumps(packet) + '\n\ndata: [DONE]\n\n').encode()
        reply = self.proxy.parse_reply(raw)
        self.assertEqual(reply['tool_calls'][0]['id'], 'call_1')
        message = dict(reply, role='assistant')
        self.assertTrue(self.proxy.same_reply(message, reply))
        changed = copy.deepcopy(message)
        changed['tool_calls'][0]['function']['name'] = 'delete_file'
        self.assertFalse(self.proxy.same_reply(changed, reply))
        self.assertIsNone(self.proxy.parse_reply(raw.split(b'data: [DONE]')[0]))

    def test_settings_change_prevents_continuation(self):
        body = {'messages': [{'role': 'user', 'content': 'test'}], 'tools': []}
        self.proxy.State.msgs = body['messages']
        self.proxy.State.reply = {'content': 'answer', 'tool_calls': []}
        self.proxy.State.context_key = self.proxy.context_key(body)
        body['messages'] = body['messages'] + [{'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'next'}]
        self.assertEqual(self.proxy.prepare(body), 'continue')
        body['reasoning_effort'] = 'low'
        with patch.object(self.proxy, 'preamble_prefix', return_value=None):
            self.assertIn('cold', self.proxy.prepare(body))

    def test_unavailable_upstream_returns_502_and_invalid_chat_400(self):
        self.proxy.args.upstream = 0
        server = self.proxy.Server(('127.0.0.1', 0), self.proxy.H)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for body, expected in (([], 400), ({'messages': []}, 502)):
                connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, expected)
                response.read()
                connection.close()
            self.assertIsNone(self.proxy.State.msgs)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == '__main__':
    unittest.main()
