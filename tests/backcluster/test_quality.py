"""Coding harness exercises real HTTP/SSE, tools and isolated fixture tests."""
import contextlib
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.quality import CALCULATOR, TESTS, opencode_config, run_quality

LOCK = {'sha256': 'a' * 64, 'engine_commit': 'b' * 40, 'gguf': {'context_length': 32768}}
MODEL = 'backcluster-' + LOCK['sha256'][:16]


def event(packet):
    return b'data: ' + json.dumps(packet).encode() + b'\n\n'


@contextlib.contextmanager
def quality_server(alias=MODEL, bad_tool=False, reused_id=False, bad_json=False, truncated=False):
    state = {'requests': [], 'tool_results': [], 'disconnected': threading.Event()}
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            payload = json.dumps({'object': 'list', 'data': [{'id': alias}]}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append(request)
            messages = request['messages']
            state['tool_results'].extend(m for m in messages if m['role'] == 'tool')
            if not request['stream']:
                message = '{"answer":"backcluster","n":7}' if not bad_json else '{"answer":"backcluster","n":true}'
                payload = json.dumps({'model': MODEL, 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': message}}]}).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload); return
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            packets = []
            tool = request.get('tool_choice')
            if isinstance(tool, dict):
                name = tool['function']['name']
                args = {'read_file': {'path': '../../escape.py' if bad_tool else 'calculator.py'},
                        'edit_file': {'path': 'calculator.py', 'old_text': 'return a - b', 'new_text': 'return a + b'},
                        'run_tests': {}}[name]
                argument = json.dumps(args)
                identifier = 'call_same' if reused_id else 'call_' + name
                packets += [{'choices': [{'index': 0, 'delta': {'reasoning_content': 'tool selection'}}]},
                    {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': identifier,
                        'type': 'function', 'function': {'name': name[:3], 'arguments': argument[:4]}}]}}]},
                    {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0,
                        'function': {'name': name[3:], 'arguments': argument[4:]}}]}}]}]
                finish = 'tool_calls'
            else:
                if tool == 'none':
                    previous = json.loads(messages[-1]['content'])
                    content = 'BACKCLUSTER_TOOLS_OK' if previous.get('passed') is True else 'test failed'
                elif messages[-1]['content'].startswith('Count from'):
                    # Keep the stream active so cancellation must actually
                    # close the peer's socket rather than consume a full reply.
                    try:
                        for i in range(100):
                            self.wfile.write(event({'model': MODEL, 'choices': [{'index': 0,
                                'delta': {'content': str(i + 1) + '\n'}}]})); self.wfile.flush()
                            time.sleep(.01)
                    except (BrokenPipeError, ConnectionResetError):
                        state['disconnected'].set()
                    return
                else:
                    content = 'BACKCLUSTER_OK'
                    packets.append({'choices': [{'index': 0, 'delta': {'reasoning_content': 'private thinking'}}]})
                packets.append({'choices': [{'index': 0, 'delta': {'content': content}}]})
                finish = 'stop'
            packets.append({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})
            packets.append({'choices': [], 'usage': {'completion_tokens': 12, 'prompt_tokens': 42}})
            try:
                for packet in packets:
                    packet['model'] = MODEL
                    raw = event(packet)
                    for offset in range(0, len(raw), 7):
                        self.wfile.write(raw[offset:offset + 7]); self.wfile.flush()
                if not truncated: self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state['disconnected'].set()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler); server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try: yield server.server_port, state
    finally: server.shutdown(); server.server_close(); thread.join(2)


class QualityTests(unittest.TestCase):
    def test_complete_coding_round_trip_and_honest_remaining_gates(self):
        with quality_server() as (port, state):
            report = run_quality(port, MODEL, LOCK, timeout=5)
            self.assertTrue(state['disconnected'].wait(1), 'cancelling client did not close the live socket')
        self.assertTrue(report['ok'], report)
        self.assertEqual(len(report['cases']), 4)
        tool_case = next(c for c in report['cases'] if c['name'] == 'coding_read_edit_test_round_trip')
        self.assertNotEqual(tool_case['before_tests']['returncode'], 0)
        self.assertEqual(tool_case['after_tests']['returncode'], 0)
        self.assertEqual([row['name'] for row in tool_case['tools']], ['read_file', 'edit_file', 'run_tests'])
        self.assertTrue(report['api_model_identity_verified'])
        self.assertFalse(report['server_identity_verified']); self.assertFalse(report['hardware_validated'])
        self.assertFalse(report['cases'][-1]['server_stop_verified'])
        self.assertTrue(report['cases'][1]['reasoning_observed'])
        self.assertTrue(any(m['tool_call_id'] == 'call_run_tests' and json.loads(m['content'])['passed']
                            for m in state['tool_results']))
        self.assertNotIn('private thinking', json.dumps(report))
        self.assertNotIn('BACKCLUSTER_TOOLS_OK', json.dumps(report))

    def test_other_model_preflight_prevents_all_completions(self):
        with quality_server(alias='another-model') as (port, state):
            report = run_quality(port, MODEL, LOCK, timeout=5)
        self.assertFalse(report['ok']); self.assertEqual(state['requests'], [])
        self.assertEqual(report['cases'][0]['name'], 'model_alias_preflight')

    def test_invalid_tool_arguments_never_run_or_edit(self):
        with quality_server(bad_tool=True) as (port, state):
            report = run_quality(port, MODEL, LOCK, timeout=5)
        self.assertFalse(report['ok'])
        tools = [r.get('tool_choice', {}).get('function', {}).get('name') for r in state['requests']
                 if isinstance(r.get('tool_choice'), dict)]
        self.assertEqual(tools, ['read_file'])
        self.assertEqual(state['tool_results'], [])
        failed = next(c for c in report['cases'] if c['name'] == 'coding_read_edit_test_round_trip')
        self.assertEqual(failed['evidence']['stage'], 'read_file')
        self.assertEqual(len(failed['evidence']['samples']), 1)
        self.assertGreater(failed['evidence']['samples'][0]['response_bytes'], 0)

    def test_reused_correlation_id_fails_before_third_tool(self):
        with quality_server(reused_id=True) as (port, state):
            report = run_quality(port, MODEL, LOCK, timeout=5)
        self.assertFalse(report['ok'])
        self.assertFalse(any(r.get('tool_choice') == {'type': 'function', 'function': {'name': 'run_tests'}}
                             for r in state['requests']))

    def test_json_semantics_and_sse_completion_are_required(self):
        with quality_server(bad_json=True) as (port, _): report = run_quality(port, MODEL, LOCK, timeout=5)
        self.assertFalse(report['ok']); self.assertEqual(report['cases'][0]['status'], 'error')
        with quality_server(truncated=True) as (port, _): report = run_quality(port, MODEL, LOCK, timeout=5)
        self.assertFalse(report['ok']); self.assertEqual(report['cases'][1]['status'], 'error')
        self.assertGreater(report['cases'][1]['evidence']['samples'][0]['response_bytes'], 0)

    def test_lock_alias_and_minimum_output_budget_are_gates(self):
        with self.assertRaises(ValueError): run_quality(8080, 'wrong', LOCK)
        with self.assertRaises(ValueError): run_quality(8080, MODEL, LOCK, tokens=32)


class OpenCodeTests(unittest.TestCase):
    def test_pinned_local_model_and_limits(self):
        config = opencode_config(LOCK, 16384, version='opencode 1.2.3')
        self.assertEqual(config['model'], config['small_model'])
        self.assertEqual(config['enabled_providers'], ['backcluster'])
        provider = config['provider']['backcluster']
        self.assertEqual(provider['options']['baseURL'], 'http://127.0.0.1:8080/v1')
        self.assertEqual(list(provider['models']), [MODEL])
        self.assertEqual(provider['models'][MODEL]['limit'], {'context': 16384, 'output': 4096})
        self.assertFalse(provider['models'][MODEL]['reasoning'])
        self.assertTrue(opencode_config(LOCK, 16384, version='1.2.3', reasoning=True)['provider']['backcluster']['models'][MODEL]['reasoning'])

    def test_unknown_versions_and_overbudget_context_are_rejected(self):
        for version in [None, 'latest', '2.0.0', '1.1.1-beta', '0.15.0']:
            with self.subTest(version=version), self.assertRaises(ValueError): opencode_config(LOCK, 16384, version=version)
        for context, output in [(65536, 4096), (4096, 4096), (100, 64), (True, 64)]:
            with self.subTest(context=context), self.assertRaises(ValueError):
                opencode_config(LOCK, context, output_tokens=output, version='1.2.3')


@contextlib.contextmanager
def opencode_endpoint(directory, coding=False):
    """A deterministic model fixture drives actual installed OpenCode tools."""
    state = {'requests': [], 'stage': 0, 'tool_messages': [], 'lock': threading.Lock()}
    calculator = str(directory / 'calculator.py')
    calls = [
        ('read', {'filePath': calculator}),
        ('edit', {'filePath': calculator, 'oldString': 'return a - b', 'newString': 'return a + b'}),
        ('bash', {'command': 'python3 -I -B -m unittest discover -s . -p test_calculator.py',
                  'description': 'Run the fixed addition regression fixture', 'timeout': 10000}),
    ]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with state['lock']:
                state['requests'].append(request)
                state['tool_messages'].extend(m for m in request.get('messages', []) if m.get('role') == 'tool')
                main = bool(request.get('tools'))
                stage = state['stage']
                if main: state['stage'] += 1
            metadata = {'id': 'chatcmpl-fixture-' + str(stage), 'object': 'chat.completion.chunk',
                        'created': int(time.time()), 'model': MODEL}
            if coding and main and stage < len(calls):
                name, arguments = calls[stage]
                args = json.dumps(arguments)
                packets = [{'choices': [{'index': 0, 'delta': {'tool_calls': [{
                    'index': 0, 'id': 'call_fixture_' + str(stage), 'type': 'function',
                    'function': {'name': name, 'arguments': args[:5]}}]}, 'finish_reason': None}]},
                    {'choices': [{'index': 0, 'delta': {'tool_calls': [{
                        'index': 0, 'function': {'arguments': args[5:]}}]}, 'finish_reason': None}]},
                    {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]}]
            else:
                packets = [{'choices': [{'index': 0, 'delta': {'reasoning_content': 'OPENCODE_REASONING_FIXTURE'}, 'finish_reason': None}]},
                    {'choices': [{'index': 0, 'delta': {'content': 'BACKCLUSTER_OPENCODE_OK'}, 'finish_reason': None}]},
                    {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}]
            packets.append({'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 5,
                'total_tokens': 105, 'completion_tokens_details': {'reasoning_tokens': 1}}})
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            try:
                for packet in packets:
                    raw = event(dict(metadata, **packet))
                    for offset in range(0, len(raw), 13):
                        self.wfile.write(raw[offset:offset + 13]); self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler); server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try: yield server.server_port, state
    finally: server.shutdown(); server.server_close(); thread.join(2)


@unittest.skipUnless(os.environ.get('BACKCLUSTER_OPENCODE_BIN'), 'set BACKCLUSTER_OPENCODE_BIN for installed-client integration')
class OpenCodeCliTests(unittest.TestCase):
    def run_client(self, coding=False):
        executable = Path(os.environ['BACKCLUSTER_OPENCODE_BIN']).resolve()
        with tempfile.TemporaryDirectory(prefix='backcluster-opencode-smoke-') as tmp:
            directory = Path(tmp)
            project = directory / 'project'; project.mkdir()
            home = directory / 'home'; home.mkdir()
            env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(home),
                'XDG_CONFIG_HOME': str(home / 'config'), 'XDG_DATA_HOME': str(home / 'data'),
                'XDG_CACHE_HOME': str(home / 'cache'), 'OPENCODE_DISABLE_MODELS_FETCH': '1',
                'OPENCODE_DISABLE_AUTOUPDATE': '1', 'OPENCODE_DISABLE_DEFAULT_PLUGINS': '1'}
            version = subprocess.run([str(executable), '--version'], env=env, cwd=project,
                capture_output=True, text=True, timeout=20, check=True).stdout.strip()
            (project / 'calculator.py').write_text(CALCULATOR, encoding='utf-8')
            (project / 'test_calculator.py').write_text(TESTS, encoding='utf-8')
            with opencode_endpoint(project, coding) as (port, state):
                config = opencode_config(LOCK, 16384, port=port, version=version, reasoning=True)
                # All model decisions in this test are fixed above and can
                # touch only this temporary fixture. No approval interaction.
                config['permission'] = {'read': 'allow', 'edit': 'allow', 'bash': 'allow'}
                path = project / 'opencode.json'; path.write_text(json.dumps(config))
                env['OPENCODE_CONFIG'] = str(path)
                parsed = subprocess.run([str(executable), 'debug', 'config'], cwd=project, env=env,
                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=True)
                self.assertEqual(json.loads(parsed.stdout)['model'], config['model'])
                result = subprocess.run([str(executable), 'run', '--pure', '--format', 'json', '--thinking',
                    '--title', 'BackCluster fixture', 'Fix the addition fixture and run its tests' if coding else
                    'Reply exactly BACKCLUSTER_OPENCODE_OK'], cwd=project, env=env,
                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
                self.assertFalse(any(row['type'] == 'error' for row in rows), rows)
                texts = [row['part']['text'] for row in rows if row['type'] == 'text']
                self.assertIn('BACKCLUSTER_OPENCODE_OK', texts)
                self.assertFalse(any('OPENCODE_REASONING_FIXTURE' in text for text in texts))
                self.assertTrue(any(row['type'] == 'reasoning' and row['part']['text'] == 'OPENCODE_REASONING_FIXTURE'
                                    for row in rows), rows)
                self.assertTrue(state['requests'])
                self.assertTrue(all(r['model'] == MODEL for r in state['requests']))
                if coding:
                    self.assertEqual((project / 'calculator.py').read_text(), CALCULATOR.replace('return a - b', 'return a + b'))
                    tools = [row['part'] for row in rows if row['type'] == 'tool_use']
                    self.assertEqual([row['tool'] for row in tools], ['read', 'edit', 'bash'])
                    self.assertTrue(all(row['state']['status'] == 'completed' for row in tools), tools)
                    self.assertIn('OK', tools[-1]['state']['output'])
                    self.assertGreaterEqual(len(state['tool_messages']), 3)

    def test_installed_client_config_sse_reasoning_and_local_alias(self):
        self.run_client()

    def test_installed_client_read_edit_test_round_trip(self):
        self.run_client(coding=True)


if __name__ == '__main__': unittest.main()
