"""Bounded loopback OpenAI transport and comparable, prompt-private baselines.

API aliases and launch receipts establish declared experiment identity. Neither
can prove the weights or executable loaded by another process.
"""
import hashlib
import http.client
import json
import math
import re
import socket
import statistics
import threading
import time
from datetime import datetime, timezone

MAX_RESPONSE = 8 * 1024 * 1024
MAX_EVENT = 65536
HASH_FORMAT = 'normalized_choices_v2'


class StreamFailure(ValueError):
    """Stream failure with bounded private evidence, never server text."""
    def __init__(self, evidence):
        super().__init__(evidence['error'])
        self.evidence = evidence


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON member')
        result[key] = value
    return result


def parse_json(data):
    def invalid_constant(value):
        raise ValueError('non-finite JSON number')
    try:
        return json.loads(data, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    except RecursionError as error:
        raise ValueError('JSON nesting exceeds parser limit') from error


def _bounds(port, tokens, timeout):
    if (type(port) is not int or not 1024 <= port <= 65535 or
            type(tokens) is not int or not 1 <= tokens <= 4096 or
            type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600):
        raise ValueError('invalid request bounds')


def lock_identity(lock):
    if (not isinstance(lock, dict) or not isinstance(lock.get('sha256'), str) or
            re.fullmatch('[0-9a-f]{64}', lock['sha256']) is None or
            not isinstance(lock.get('engine_commit'), str) or
            re.fullmatch('[0-9a-f]{40}', lock['engine_commit']) is None):
        raise ValueError('invalid model/engine identity')
    return 'backcluster-' + lock['sha256'][:16]


def experiment_identity(launch, lock, port):
    """Retain comparable runtime settings, without local paths or addresses."""
    if launch is None:
        return None
    alias = lock_identity(lock)
    if (not isinstance(launch, dict) or launch.get('document_kind') != 'backcluster_launch' or
            launch.get('schema_version') != 1 or launch.get('model_sha256') != lock['sha256'] or
            launch.get('engine_commit') != lock['engine_commit'] or launch.get('bind') != '127.0.0.1' or
            launch.get('port') != port or type(launch.get('context')) is not int or
            not 128 <= launch['context'] <= 131072):
        raise ValueError('launch receipt identity mismatch')
    argv = launch.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv):
        raise ValueError('invalid launch argv')
    options = {}
    # Preserve inference options, including repeated tensor placement patterns.
    # The receipt hash retains paths and routing without disclosing either.
    path_flags = {'--model', '--model-draft'}
    route_flags = {'--host', '--port', '--alias', '--rpc'}
    boolean_flags = {'--no-repack', '--jinja'}
    repeated_flags = {'--override-tensor'}
    index = 1
    while index < len(argv):
        flag = argv[index]
        index += 1
        if not flag.startswith('--') or (flag in options and flag not in repeated_flags):
            raise ValueError('invalid or duplicate launch option')
        if flag in boolean_flags:
            value = True
        else:
            if index == len(argv) or argv[index].startswith('--'):
                raise ValueError('launch option requires an explicit value')
            value = argv[index]
            index += 1
        if flag in repeated_flags:
            options.setdefault(flag, []).append(value)
        else:
            options[flag] = value
    if (options.get('--host') != '127.0.0.1' or options.get('--port') != str(port) or
            options.get('--alias') != alias or options.get('--ctx-size') != str(launch['context']) or
            not options.get('--model')):
        raise ValueError('launch argv disagrees with receipt')
    settings = {key: value for key, value in options.items() if key not in path_flags | route_flags}
    environment = launch.get('environment', {})
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               for k, v in environment.items()):
        raise ValueError('invalid launch environment')
    # Mirror serving.clean_environment's explicit defaults. Endpoint strings
    # stay private; their counts and the receipt hash retain routing evidence.
    routing_env = {'LLAMA_SPLIT_TAIL', 'LLAMA_KV_REMOTE', 'LLAMA_SPLIT_MIDDLE'}
    environment = dict({'GGML_METAL_FA_SME': '0', 'GGML_METAL_MM_SME': '0'}, **environment)
    env_settings = {k: v for k, v in environment.items() if k not in routing_env}
    build_id = launch.get('engine_build_id')
    if build_id is not None and (not isinstance(build_id, str) or re.fullmatch('[0-9a-f]{64}', build_id) is None):
        raise ValueError('invalid engine build identity')
    workers = launch.get('remote_workers', [])
    effective_context = launch.get('total_context', launch['context'])
    if type(effective_context) is not int or not launch['context'] <= effective_context <= 262144 or not isinstance(workers, list):
        raise ValueError('invalid launch context or workers')
    return {'context': effective_context, 'local_context': launch['context'], 'runtime_settings': settings,
            'runtime_environment': env_settings, 'engine_build_id': build_id,
            'mode': launch.get('mode'), 'draft_enabled': launch.get('draft_enabled'),
            'remote_worker_count': len(workers),
            'launch_receipt_sha256': canonical_hash(launch)}


class LocalRequest:
    """Absolute request deadline, including slow HTTP headers and SSE drips."""
    def __init__(self, port, timeout):
        _bounds(port, 1, timeout)
        self.start = time.perf_counter_ns()
        self.deadline = time.monotonic() + timeout
        self.connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
        self.sock = None
        self.expired = threading.Event()
        self.timer = threading.Timer(timeout, self._expire)
        self.timer.daemon = True

    def _expire(self):
        self.expired.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.expired.is_set():
            raise TimeoutError('local request deadline exceeded')
        return remaining

    def open(self, method, path, body=None):
        self.timer.start()
        self.connection.connect()
        self.sock = self.connection.sock
        self.sock.settimeout(self.remaining())
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        self.connection.request(method, path, body, headers)
        self.sock.settimeout(self.remaining())
        response = self.connection.getresponse()
        try:
            self.remaining()
            if response.status != 200:
                raise ValueError('local server rejected request')
        except (OSError, ValueError):
            response.close()
            raise
        return response

    def read(self, response):
        self.remaining()
        if response.isclosed():
            return b''
        self.sock.settimeout(self.remaining())
        chunk = response.read1(4096)
        self.remaining()
        return chunk

    def close(self):
        self.timer.cancel()
        self.connection.close()


def request_json(port, path='/v1/models', timeout=10, payload=None, evidence=None):
    request = LocalRequest(port, timeout)
    raw = bytearray()
    response = None
    try:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        if body is not None and len(body) > 1024 * 1024:
            raise ValueError('request exceeds limit')
        response = request.open('GET' if payload is None else 'POST', path, body)
        if response.getheader('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            raise ValueError('server did not return JSON')
        while True:
            chunk = request.read(response)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_RESPONSE:
                raise ValueError('oversized JSON response')
        result = parse_json(raw)
        if not isinstance(result, dict) or 'error' in result:
            raise ValueError('invalid OpenAI response')
        return result
    finally:
        if response is not None:
            response.close()
        request.close()
        if evidence is not None:
            evidence.update(response_bytes=len(raw), response_sha256=hashlib.sha256(raw).hexdigest(),
                            response_hash_format='raw_response_v1', wall_ns=time.perf_counter_ns() - request.start)


def probe_server(port, model, timeout=10, evidence=None):
    document = request_json(port, timeout=timeout, evidence=evidence)
    models = document.get('data')
    if (document.get('object') != 'list' or not isinstance(models, list) or len(models) != 1 or
            not isinstance(models[0], dict) or models[0].get('id') != model):
        raise ValueError('expected one local model with the locked alias')
    return {'api_model_identity_verified': True, 'server_identity_verified': False}


def _new_choice(index):
    return {'index': index, 'content': [], 'reasoning_content': [], 'reasoning': [],
            'tool_calls': {}, 'finish_reason': None}


def _append_text(target, key, value):
    if value is None:
        return False
    if not isinstance(value, str):
        raise ValueError('invalid content delta')
    if value:
        target[key].append(value)
        return True
    return False


def _packet(packet, output, model, usage):
    if not isinstance(packet, dict) or 'error' in packet:
        raise ValueError('invalid OpenAI event')
    if 'model' in packet and packet['model'] != model:
        raise ValueError('response model alias mismatch')
    choices = packet.get('choices', [])
    if not isinstance(choices, list) or len(choices) > 1:
        raise ValueError('invalid choices')
    generated = 0
    for choice in choices:
        if not isinstance(choice, dict) or choice.get('index', 0) != 0 or type(choice.get('index', 0)) is not int:
            raise ValueError('expected one choice at index zero')
        message = output.setdefault(0, _new_choice(0))
        delta = choice.get('delta', {})
        if not isinstance(delta, dict) or delta.get('role', 'assistant') != 'assistant':
            raise ValueError('invalid assistant delta')
        changed = False
        for field in ('content', 'reasoning_content', 'reasoning'):
            changed |= _append_text(message, field, delta.get(field))
        calls = delta.get('tool_calls')
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            raise ValueError('invalid tool-call delta')
        for call in calls:
            if not isinstance(call, dict) or call.get('type', 'function') != 'function':
                raise ValueError('invalid tool-call delta')
            index = call.get('index', 0)
            if type(index) is not int or index < 0 or index > 63:
                raise ValueError('invalid tool-call index')
            function = call.get('function', {})
            if not isinstance(function, dict):
                raise ValueError('invalid tool-call function')
            target = message['tool_calls'].setdefault(index, {'id': [], 'name': [], 'arguments': []})
            _append_text(target, 'id', call.get('id'))
            for field in ('name', 'arguments'):
                changed |= _append_text(target, field, function.get(field))
        if message['finish_reason'] is not None and (changed or calls):
            raise ValueError('output after finish_reason')
        finish = choice.get('finish_reason')
        if finish is not None:
            if finish not in ('stop', 'length', 'tool_calls', 'content_filter', 'function_call') or message['finish_reason'] is not None:
                raise ValueError('invalid finish_reason')
            message['finish_reason'] = finish
        generated += int(changed)
    if packet.get('usage') is not None:
        if not isinstance(packet['usage'], dict):
            raise ValueError('invalid token usage')
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = packet['usage'].get(field)
            if value is not None:
                if type(value) is not int or value < 0 or (field in usage and usage[field] != value):
                    raise ValueError('invalid or conflicting token usage')
                usage[field] = value
    return generated


def _normalize(output):
    normalized = []
    for index in sorted(output):
        message = output[index]
        calls = message['tool_calls']
        if sorted(calls) != list(range(len(calls))):
            raise ValueError('noncontiguous tool-call indices')
        normalized.append(dict(index=index,
            **{field: ''.join(message[field]) for field in ('content', 'reasoning_content', 'reasoning')},
            finish_reason=message['finish_reason'],
            tool_calls=[{'id': ''.join(calls[i]['id']), 'type': 'function',
                         'function': {'name': ''.join(calls[i]['name']), 'arguments': ''.join(calls[i]['arguments'])}}
                        for i in sorted(calls)]))
    return normalized


def stream_chat(port, messages, model, tokens=256, timeout=120, tools=None, tool_choice=None,
                response_format=None, cancel_after_chunks=None):
    """Return private output to callers; benchmark reports keep only hashes."""
    _bounds(port, tokens, timeout)
    if not isinstance(model, str) or not model or len(model) > 256 or not isinstance(messages, list) or not messages:
        raise ValueError('invalid request model or messages')
    if cancel_after_chunks is not None and (type(cancel_after_chunks) is not int or cancel_after_chunks < 1):
        raise ValueError('invalid cancellation threshold')
    payload = {'model': model, 'messages': messages, 'max_tokens': tokens, 'temperature': 0,
               'seed': 0, 'stream': True, 'stream_options': {'include_usage': True}}
    for key, value in [('tools', tools), ('tool_choice', tool_choice), ('response_format', response_format)]:
        if value is not None:
            payload[key] = value
    body = json.dumps(payload, allow_nan=False).encode()
    if len(body) > 1024 * 1024:
        raise ValueError('request exceeds limit')
    request = LocalRequest(port, timeout)
    response = None
    output, usage = {}, {}
    chunks, first, total, done, cancelled = 0, None, 0, False, False
    raw_hash = hashlib.sha256()
    try:
        response = request.open('POST', '/v1/chat/completions', body)
        if response.getheader('Content-Type', '').split(';')[0].strip().lower() != 'text/event-stream':
            raise ValueError('server did not return SSE')
        pending, lines, event_size = bytearray(), [], 0
        while not done and not cancelled:
            chunk = request.read(response)
            total += len(chunk)
            raw_hash.update(chunk)
            if not chunk or total > MAX_RESPONSE:
                raise ValueError('incomplete or oversized SSE response')
            pending.extend(chunk)
            while b'\n' in pending:
                line, extra = pending.split(b'\n', 1)
                pending[:] = extra
                line = line.rstrip(b'\r')
                event_size += len(line) + 1
                if event_size > MAX_EVENT:
                    raise ValueError('oversized SSE event')
                if line:
                    if line.startswith(b'data:'):
                        data = line[5:]
                        lines.append(data[1:] if data.startswith(b' ') else data)
                    continue
                event_size = 0
                if not lines:
                    continue
                data = b'\n'.join(lines)
                lines.clear()
                if data == b'[DONE]':
                    done = True
                    break
                generated = _packet(parse_json(data), output, model, usage)
                if generated:
                    first = first or time.perf_counter_ns()
                    chunks += generated
                    if cancel_after_chunks is not None and chunks >= cancel_after_chunks:
                        cancelled = True
                        break
            if len(pending) > MAX_EVENT:
                raise ValueError('oversized SSE event')
        elapsed = time.perf_counter_ns() - request.start
        if first is None:
            raise ValueError('no generated content')
        normalized = _normalize(output)
        # Tool IDs are transport correlation IDs, not model output equivalence.
        stable = [dict(choice, tool_calls=[{'type': c['type'], 'function': c['function']}
                                          for c in choice['tool_calls']]) for choice in normalized]
        count = usage.get('completion_tokens')
        return {'status': 'cancelled' if cancelled else 'ok', 'ttft_ns': first - request.start,
                'wall_ns': elapsed, 'completion_tokens': count, 'prompt_tokens': usage.get('prompt_tokens'),
                'content_chunks': chunks, 'output_sha256': canonical_hash(stable),
                'output_hash_format': HASH_FORMAT, 'response_bytes': total,
                'response_sha256': raw_hash.hexdigest(), 'accepted_tokens_s': count * 1e9 / elapsed if count is not None else None,
                'choices': normalized}
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException) as error:
        evidence = {'status': 'error',
            'error': 'local_stream_deadline' if isinstance(error, TimeoutError) or request.expired.is_set() else 'local_stream_failed',
            'ttft_ns': first - request.start if first is not None else None,
            'wall_ns': time.perf_counter_ns() - request.start, 'content_chunks': chunks,
            'response_bytes': total, 'response_sha256': raw_hash.hexdigest()}
        raise StreamFailure(evidence) from error
    finally:
        if response is not None:
            response.close()
        request.close()


def completion(port, prompt, model, tokens=256, timeout=120):
    result = stream_chat(port, [{'role': 'user', 'content': prompt}], model, tokens, timeout)
    result['finish_reasons'] = [choice['finish_reason'] for choice in result.pop('choices')]
    return result


def _summary(rows):
    wall = sorted(r['wall_ns'] for r in rows if r['status'] == 'ok' and not r['warmup'])
    if not wall:
        return None
    return {'samples': len(wall), 'median_wall_ns': statistics.median(wall),
            'p95_wall_ns': wall[math.ceil(.95 * len(wall)) - 1],
            'median_ttft_ns': statistics.median(r['ttft_ns'] for r in rows if r['status'] == 'ok' and not r['warmup'])}


def run_bench(port, prompt, model, lock, repeats=5, tokens=256, timeout=120, launch=None):
    _bounds(port, tokens, timeout)
    lock_identity(lock)
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError('invalid benchmark repeats')
    if not isinstance(prompt, str) or not prompt or len(prompt.encode()) > 1024 * 1024:
        raise ValueError('prompt is empty or exceeds limit')
    experiment = experiment_identity(launch, lock, port)
    identity = {'api_model_identity_verified': False, 'server_identity_verified': False}
    identity_evidence = None
    if launch is not None:
        if model != lock_identity(lock):
            raise ValueError('benchmark model differs from lock alias')
        identity_evidence = {}
        identity = probe_server(port, model, min(timeout, 10), identity_evidence)
    rows = []
    for index in range(repeats + 1):
        try:
            row = completion(port, prompt, model, tokens, timeout)
        except StreamFailure as error:
            row = error.evidence
        except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
            row = {'status': 'error', 'error': 'local_stream_failed'}
        rows.append(dict(row, sample=index, warmup=index == 0))
        if row['status'] != 'ok':
            break
    return {'document_kind': 'backcluster_baseline_bench', 'schema_version': 2,
            'captured_at': datetime.now(timezone.utc).isoformat(), 'hardware_validated': False,
            'model_sha256': lock['sha256'], 'engine_commit': lock['engine_commit'], **identity,
            'model_alias_evidence': identity_evidence,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'output_token_limit': tokens, 'repeats': repeats, 'experiment': experiment,
            'generation_settings': {'temperature': 0, 'seed': 0, 'stream': True},
            'ok': all(r['status'] == 'ok' for r in rows) and len(rows) == repeats + 1,
            'samples': rows, 'summary': _summary(rows)}


def _comparison_samples(report):
    if (not isinstance(report, dict) or report.get('document_kind') != 'backcluster_baseline_bench' or
            report.get('schema_version') != 2 or report.get('ok') is not True or
            report.get('api_model_identity_verified') is not True or not isinstance(report.get('experiment'), dict)):
        raise ValueError('comparison requires successful v2 reports with launch receipts')
    experiment = report['experiment']
    if (type(experiment.get('context')) is not int or not 128 <= experiment['context'] <= 262144 or
            not isinstance(experiment.get('runtime_settings'), dict) or
            not isinstance(experiment.get('runtime_environment'), dict) or
            not isinstance(experiment.get('launch_receipt_sha256'), str) or
            re.fullmatch('[0-9a-f]{64}', experiment['launch_receipt_sha256']) is None):
        raise ValueError('invalid experiment identity')
    lock_identity({'sha256': report.get('model_sha256'), 'engine_commit': report.get('engine_commit')})
    if (not isinstance(report.get('prompt_sha256'), str) or re.fullmatch('[0-9a-f]{64}', report['prompt_sha256']) is None or
            type(report.get('output_token_limit')) is not int or not 1 <= report['output_token_limit'] <= 4096 or
            report.get('generation_settings') != {'temperature': 0, 'seed': 0, 'stream': True}):
        raise ValueError('invalid benchmark generation identity')
    repeats, rows = report.get('repeats'), report.get('samples')
    if type(repeats) is not int or not 1 <= repeats <= 20 or not isinstance(rows, list) or len(rows) != repeats + 1:
        raise ValueError('incomplete benchmark repeats')
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or row.get('status') != 'ok' or type(row.get('sample')) is not int or row['sample'] != index or
                row.get('warmup') is not (index == 0) or row.get('output_hash_format') != HASH_FORMAT or
                not isinstance(row.get('output_sha256'), str) or re.fullmatch('[0-9a-f]{64}', row['output_sha256']) is None):
            raise ValueError('invalid benchmark sample')
        for key in ('wall_ns', 'ttft_ns'):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise ValueError('invalid benchmark timing')
        if (type(row.get('response_bytes')) is not int or row['response_bytes'] <= 0 or
                not isinstance(row.get('response_sha256'), str) or re.fullmatch('[0-9a-f]{64}', row['response_sha256']) is None):
            raise ValueError('missing raw response evidence')
        if row.get('finish_reasons') not in (['stop'], ['length']):
            raise ValueError('baseline requires a finished text completion')
        if row['ttft_ns'] > row['wall_ns']:
            raise ValueError('invalid benchmark timing order')
        count = row.get('completion_tokens')
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError('invalid benchmark token count')
    return rows[1:]


def compare_benchmarks(baseline, candidate, allow_runtime_changes=False):
    """Recompute from raw repeats; reject confounded identity/settings changes."""
    left, right = _comparison_samples(baseline), _comparison_samples(candidate)
    fields = ('model_sha256', 'engine_commit', 'prompt_sha256', 'output_token_limit', 'generation_settings', 'repeats')
    for key in fields:
        if baseline.get(key) is None or baseline[key] != candidate.get(key):
            raise ValueError('benchmark mismatch: ' + key)
    a_exp, b_exp = baseline['experiment'], candidate['experiment']
    for key in ('context', 'engine_build_id'):
        if a_exp.get(key) != b_exp.get(key):
            raise ValueError('benchmark mismatch: ' + key)
    if type(allow_runtime_changes) is not bool:
        raise ValueError('invalid runtime change policy')
    mode_changed = a_exp.get('mode') != b_exp.get('mode')
    differences = []
    placement_flags = {'--ctx-size', '--override-tensor', '--no-repack', '--batch-size', '--ubatch-size'}
    for section in ('runtime_settings', 'runtime_environment'):
        a_settings, b_settings = a_exp[section], b_exp[section]
        for key in sorted(set(a_settings) | set(b_settings)):
            if a_settings.get(key) == b_settings.get(key):
                continue
            placement = (key in placement_flags if section == 'runtime_settings' else
                         key.startswith(('LLAMA_SPLIT_', 'LLAMA_KV_REMOTE', 'GGML_METAL_REMOTE', 'LLAMA_REMOTE_PIPE')))
            if not allow_runtime_changes and not (mode_changed and placement):
                raise ValueError('benchmark mismatch: ' + section + '.' + key)
            differences.append({'setting': section + '.' + key,
                                'baseline': a_settings.get(key), 'candidate': b_settings.get(key)})
    for key in ('mode', 'remote_worker_count', 'local_context', 'draft_enabled'):
        if a_exp.get(key) != b_exp.get(key):
            if key == 'draft_enabled' and not allow_runtime_changes:
                raise ValueError('benchmark mismatch: ' + key)
            differences.append({'setting': key, 'baseline': a_exp.get(key), 'candidate': b_exp.get(key)})
    hashes = {row['output_sha256'] for row in left + right}
    tokens = {row.get('completion_tokens') for row in left + right}
    quality_match = len(hashes) == 1 and len(tokens) == 1
    a, b = statistics.median(r['wall_ns'] for r in left), statistics.median(r['wall_ns'] for r in right)
    return {'document_kind': 'backcluster_benchmark_comparison', 'schema_version': 1,
            'hardware_validated': False, 'ok': quality_match, 'quality_match': quality_match,
            'minimum_five_repeats_met': len(left) >= 5 and len(right) >= 5,
            'model_sha256': baseline['model_sha256'], 'engine_commit': baseline['engine_commit'],
            'prompt_sha256': baseline['prompt_sha256'], 'context': baseline['experiment']['context'],
            'baseline_report_sha256': canonical_hash(baseline), 'candidate_report_sha256': canonical_hash(candidate),
            'baseline_wall_ns': [r['wall_ns'] for r in left], 'candidate_wall_ns': [r['wall_ns'] for r in right],
            'baseline_ttft_ns': [r['ttft_ns'] for r in left], 'candidate_ttft_ns': [r['ttft_ns'] for r in right],
            'baseline_median_wall_ns': a, 'candidate_median_wall_ns': b,
            'observed_wall_ratio': a / b if quality_match else None,
            'engine_build_id': a_exp.get('engine_build_id'), 'runtime_differences': differences,
            'runtime_changes_explicitly_allowed': allow_runtime_changes,
            'single_runtime_setting_changed': len(differences) == 1,
            'server_identity_verified': False,
            'limitations': ['API and receipts do not attest loaded weights or engine',
                            'Output hashes establish this fixture equivalence, not general model quality',
                            'Measured timings require target-device review before performance acceptance',
                            'Multiple runtime differences compare configurations; they do not isolate a single cause']}
