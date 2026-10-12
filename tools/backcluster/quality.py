"""Offline-friendly OpenAI coding acceptance fixtures and OpenCode config.

The only executable tool is a fixed unittest fixture. Model-supplied paths,
patches, shell commands, and program text are never executed. A successful fake
server test validates this harness; it does not accept target-device inference.
"""
import hashlib
import http.client
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import (canonical_hash, experiment_identity, lock_identity, parse_json,
                        probe_server, request_json, stream_chat, _bounds)

FIXTURE_VERSION = 1
TEXT_PROMPT = 'Return exactly BACKCLUSTER_OK as your final answer. Do not call tools. Keep reasoning separate from the final answer.'
JSON_PROMPT = 'Return only the JSON object {"answer":"backcluster","n":7}. No additional members or text.'
TOOL_SYSTEM = ('Fix calculator.py using the supplied read_file, edit_file and run_tests tools, in that order. '
               'After the tests pass, reply exactly BACKCLUSTER_TOOLS_OK. Do not claim success before checking the tool results.')
TOOL_USER = 'Addition returns subtraction. Read the file, replace return a - b with return a + b, then run tests.'
CANCEL_PROMPT = 'Count from 1 to 1000, one number per line.'
TOOLS = [
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read the fixture calculator.',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'enum': ['calculator.py']}},
                       'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'edit_file', 'description': 'Fix addition in the fixture.',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'enum': ['calculator.py']},
            'old_text': {'type': 'string', 'enum': ['return a - b']},
            'new_text': {'type': 'string', 'enum': ['return a + b']}},
            'required': ['path', 'old_text', 'new_text'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'run_tests', 'description': 'Run fixed addition regression tests.',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
]
CALCULATOR = 'def add(a, b):\n    return a - b\n'
TESTS = '''import unittest
from calculator import add
class Addition(unittest.TestCase):
    def test_positive(self): self.assertEqual(add(7, 5), 12)
    def test_negative(self): self.assertEqual(add(-4, -6), -10)
    def test_zero(self): self.assertEqual(add(0, 3), 3)
'''


def _private_stream(result):
    row = {key: value for key, value in result.items() if key != 'choices'}
    row['finish_reasons'] = [choice['finish_reason'] for choice in result['choices']]
    row['reasoning_characters'] = sum(len(choice['reasoning_content']) + len(choice['reasoning'])
                                      for choice in result['choices'])
    row['tool_call_count'] = sum(len(choice['tool_calls']) for choice in result['choices'])
    return row


def _choice(result, finish):
    choices = result['choices']
    if len(choices) != 1 or choices[0]['finish_reason'] != finish:
        raise ValueError('unexpected completion finish reason')
    return choices[0]


def _text_case(port, model, tokens, timeout, evidence):
    result = stream_chat(port, [{'role': 'user', 'content': TEXT_PROMPT}], model, tokens, timeout)
    evidence['samples'].append(_private_stream(result))
    choice = _choice(result, 'stop')
    if choice['content'].strip() != 'BACKCLUSTER_OK' or choice['tool_calls']:
        raise ValueError('final content or reasoning separation failed')
    return {'name': 'sse_final_and_reasoning', 'status': 'ok', 'sample': _private_stream(result),
            'reasoning_observed': bool(choice['reasoning_content'] or choice['reasoning'])}


def _json_case(port, model, tokens, timeout, evidence):
    payload = {'model': model, 'messages': [{'role': 'user', 'content':
        JSON_PROMPT}],
        'temperature': 0, 'seed': 0, 'stream': False, 'max_tokens': tokens,
        'response_format': {'type': 'json_object'}}
    result = request_json(port, '/v1/chat/completions', timeout, payload, evidence=evidence)
    choices = result.get('choices')
    if (result.get('model') != model or not isinstance(choices, list) or len(choices) != 1 or
            not isinstance(choices[0], dict) or choices[0].get('index') != 0 or
            choices[0].get('finish_reason') != 'stop'):
        raise ValueError('invalid nonstream completion identity or finish')
    message = choices[0].get('message')
    if (not isinstance(message, dict) or message.get('role') != 'assistant' or
            not isinstance(message.get('content'), str) or message.get('tool_calls')):
        raise ValueError('invalid nonstream message')
    parsed = parse_json(message['content'])
    if (not isinstance(parsed, dict) or set(parsed) != {'answer', 'n'} or
            parsed['answer'] != 'backcluster' or type(parsed['n']) is not int or parsed['n'] != 7):
        raise ValueError('JSON fixture mismatch')
    return {'name': 'openai_nonstream_json', 'status': 'ok',
            'output_sha256': hashlib.sha256(message['content'].encode()).hexdigest(),
            'response_sha256': canonical_hash(result), 'response_hash_format': 'canonical_json_v1'}


def _test_fixture(directory, timeout):
    # The allowed edit preserves source length. Disable bytecode caching so a
    # same-second change cannot reload the deliberately broken pre-edit code.
    process = subprocess.run([sys.executable, '-I', '-B', '-m', 'unittest', 'discover', '-s', str(directory),
                              '-p', 'test_calculator.py'], cwd=directory, stdin=subprocess.DEVNULL,
                             capture_output=True, timeout=min(timeout, 30), check=False)
    return {'returncode': process.returncode, 'stdout_sha256': hashlib.sha256(process.stdout).hexdigest(),
            'stderr_sha256': hashlib.sha256(process.stderr).hexdigest(),
            'stdout_bytes': len(process.stdout), 'stderr_bytes': len(process.stderr)}


def _execute_tool(name, arguments, directory, timeout):
    if not isinstance(arguments, dict):
        raise ValueError('tool arguments must be an object')
    path = directory / 'calculator.py'
    if name == 'read_file':
        if arguments != {'path': 'calculator.py'}:
            raise ValueError('read fixture arguments mismatch')
        return {'path': 'calculator.py', 'content': path.read_text(encoding='utf-8')}, None
    if name == 'edit_file':
        if arguments != {'path': 'calculator.py', 'old_text': 'return a - b', 'new_text': 'return a + b'}:
            raise ValueError('edit fixture arguments mismatch')
        old = path.read_text(encoding='utf-8')
        if old != CALCULATOR:
            raise ValueError('fixture changed before edit')
        path.write_text(CALCULATOR.replace('return a - b', 'return a + b'), encoding='utf-8')
        return {'path': 'calculator.py', 'changed': True}, None
    if name == 'run_tests':
        if arguments:
            raise ValueError('test fixture accepts no arguments')
        evidence = _test_fixture(directory, timeout)
        return {'passed': evidence['returncode'] == 0, 'tests': 3}, evidence
    raise ValueError('unregistered tool')


def _tool_case(port, model, tokens, timeout, evidence):
    messages = [{'role': 'system', 'content': TOOL_SYSTEM}, {'role': 'user', 'content': TOOL_USER}]
    samples, tool_rows, ids = [], [], set()
    with tempfile.TemporaryDirectory(prefix='backcluster-quality-') as tmp:
        directory = Path(tmp)
        (directory / 'calculator.py').write_text(CALCULATOR, encoding='utf-8')
        (directory / 'test_calculator.py').write_text(TESTS, encoding='utf-8')
        before = _test_fixture(directory, timeout)
        evidence['before_tests'] = before
        if before['returncode'] == 0:
            raise ValueError('fixture did not fail before edit')
        for name in ('read_file', 'edit_file', 'run_tests'):
            evidence['stage'] = name
            result = stream_chat(port, messages, model, tokens, timeout, tools=TOOLS,
                tool_choice={'type': 'function', 'function': {'name': name}})
            evidence['samples'].append(_private_stream(result))
            choice = _choice(result, 'tool_calls')
            if len(choice['tool_calls']) != 1:
                raise ValueError('expected exactly one fixture tool call')
            call = choice['tool_calls'][0]
            identifier = call['id']
            if (not isinstance(identifier, str) or re.fullmatch('[A-Za-z0-9_-]{1,128}', identifier) is None or
                    identifier in ids or call['function']['name'] != name):
                raise ValueError('invalid, reused or unexpected tool-call identity')
            ids.add(identifier)
            arguments = parse_json(call['function']['arguments'])
            tool_result, test_evidence = _execute_tool(name, arguments, directory, timeout)
            if name == 'run_tests' and not tool_result['passed']:
                raise ValueError('fixed fixture tests failed')
            samples.append(_private_stream(result))
            tool_rows.append({'name': name, 'arguments_sha256': canonical_hash(arguments),
                              'result_sha256': canonical_hash(tool_result), 'test_evidence': test_evidence})
            messages.append({'role': 'assistant', 'content': choice['content'] or None, 'tool_calls': [call]})
            messages.append({'role': 'tool', 'tool_call_id': identifier, 'content': json.dumps(tool_result)})
        evidence['stage'] = 'final_answer'
        result = stream_chat(port, messages, model, tokens, timeout, tools=TOOLS, tool_choice='none')
        evidence['samples'].append(_private_stream(result))
        choice = _choice(result, 'stop')
        if choice['content'].strip() != 'BACKCLUSTER_TOOLS_OK' or choice['tool_calls']:
            raise ValueError('final tool round-trip fixture mismatch')
        samples.append(_private_stream(result))
        after = _test_fixture(directory, timeout)
        evidence['after_tests'] = after
        if after['returncode'] != 0:
            raise ValueError('fixture failed independent post-edit verification')
    return {'name': 'coding_read_edit_test_round_trip', 'status': 'ok', 'samples': samples,
            'tools': tool_rows, 'tool_choices_forced': True, 'before_tests': before, 'after_tests': after}


def _cancel_case(port, model, tokens, timeout, evidence):
    result = stream_chat(port, [{'role': 'user', 'content': CANCEL_PROMPT}],
                         model, tokens, timeout, cancel_after_chunks=1)
    evidence['samples'].append(_private_stream(result))
    if result['status'] != 'cancelled':
        raise ValueError('local stream cancellation failed')
    return {'name': 'local_stream_close', 'status': 'ok', 'sample': _private_stream(result),
            'local_socket_closed': True, 'server_stop_verified': False}


def run_quality(port, model, lock, timeout=120, tokens=256, launch=None):
    _bounds(port, tokens, timeout)
    if tokens < 64:
        raise ValueError('quality fixtures require at least 64 output tokens')
    if model != lock_identity(lock):
        raise ValueError('quality model differs from locked alias')
    experiment = experiment_identity(launch, lock, port)
    report = {'document_kind': 'backcluster_quality', 'schema_version': 1,
        'fixture_version': FIXTURE_VERSION, 'fixture_sha256': canonical_hash({'tools': TOOLS,
            'calculator': CALCULATOR, 'tests': TESTS, 'version': FIXTURE_VERSION,
            'prompts': [TEXT_PROMPT, JSON_PROMPT, TOOL_SYSTEM, TOOL_USER, CANCEL_PROMPT]}),
        'captured_at': datetime.now(timezone.utc).isoformat(), 'model_sha256': lock['sha256'],
        'engine_commit': lock['engine_commit'], 'experiment': experiment, 'output_token_limit': tokens,
        'hardware_validated': False, 'server_identity_verified': False,
        'api_model_identity_verified': False, 'ok': False, 'cases': [],
        'remaining_hardware_gates': ['Confirm actual loaded weights and engine against lock/receipt',
            'Confirm server stops inference and frees its slot after cancellation',
            'Run OpenCode with the installed version on the target Mac',
            'Run sustained coding tasks, memory and thermal acceptance on devices']}
    report['model_alias_evidence'] = {}
    try:
        report.update(probe_server(port, model, min(timeout, 10), report['model_alias_evidence']))
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
        report['cases'].append({'name': 'model_alias_preflight', 'status': 'error', 'error': 'local_identity_failed'})
        return report
    for check in (_json_case, _text_case, _tool_case, _cancel_case):
        names = {_json_case: 'openai_nonstream_json', _text_case: 'sse_final_and_reasoning',
                 _tool_case: 'coding_read_edit_test_round_trip', _cancel_case: 'local_stream_close'}
        evidence = {'samples': []}
        try:
            report['cases'].append(check(port, model, tokens, timeout, evidence))
        except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException, subprocess.SubprocessError) as error:
            if hasattr(error, 'evidence'):
                evidence['samples'].append(error.evidence)
            report['cases'].append({'name': names[check], 'status': 'error', 'error': 'quality_fixture_failed',
                                    'evidence': evidence})
    report['ok'] = all(row['status'] == 'ok' for row in report['cases'])
    return report


def opencode_config(lock, context, output_tokens=4096, port=8080, version=None, reasoning=False):
    """Generate the OpenCode 1.x OpenAI-compatible provider config offline.

    The caller must supply the installed version. Config generation does not
    claim OpenCode smoke acceptance, tool capability, or reasoning support.
    """
    model = lock_identity(lock)
    _bounds(port, output_tokens, 10)
    if (type(context) is not int or not 128 <= context <= 131072 or output_tokens >= context or
            type(reasoning) is not bool):
        raise ValueError('invalid client context/output limits')
    model_context = lock.get('gguf', {}).get('context_length')
    if type(model_context) is int and context > model_context:
        raise ValueError('client context exceeds locked model support')
    if not isinstance(version, str) or re.fullmatch(r'(?:opencode\s+)?1\.[0-9]+\.[0-9]+', version.strip()) is None:
        raise ValueError('provide the installed stable OpenCode 1.x version; other versions need schema review')
    return {'$schema': 'https://opencode.ai/config.json', 'model': 'backcluster/' + model,
        'small_model': 'backcluster/' + model, 'enabled_providers': ['backcluster'], 'share': 'disabled',
        'provider': {'backcluster': {'npm': '@ai-sdk/openai-compatible', 'name': 'BackCluster',
            'options': {'baseURL': 'http://127.0.0.1:' + str(port) + '/v1', 'apiKey': 'local-only'},
            'models': {model: {'name': 'BackCluster locked local model',
                'limit': {'context': context, 'output': output_tokens},
                'tool_call': True, 'reasoning': reasoning}}}}}
