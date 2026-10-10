"""Bounded local OpenAI SSE baseline, raw repeats without stored prompts."""
import hashlib
import http.client
import json
import math
import time
from datetime import datetime, timezone

MAX_RESPONSE = 8 * 1024 * 1024


def completion(port, prompt, model, tokens=256, timeout=120):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': tokens, 'temperature': 0, 'stream': True,
                       'stream_options': {'include_usage': True}}).encode()
    start = time.perf_counter_ns()
    deadline = time.monotonic() + timeout
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    content = hashlib.sha256()
    chunks, first, usage, total, done = 0, None, None, 0, False
    try:
        connection.connect()
        sock = connection.sock
        connection.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError('local server rejected request')
        if 'text/event-stream' not in response.getheader('Content-Type', ''):
            raise ValueError('server did not return SSE')
        pending = bytearray()
        while not done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            sock.settimeout(remaining)
            # read1 performs at most one raw socket read. A peer dripping one
            # byte cannot reset an entire readline's timeout indefinitely.
            chunk = response.read1(4096)
            total += len(chunk)
            if not chunk or total > MAX_RESPONSE:
                raise ValueError('incomplete or oversized SSE response')
            pending.extend(chunk)
            while b'\n' in pending:
                line, extra = pending.split(b'\n', 1)
                pending[:] = extra
                if len(line) > 65536:
                    raise ValueError('oversized SSE event')
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                packet = json.loads(data)
                for choice in packet.get('choices', []):
                    delta = choice.get('delta', {})
                    text = delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if text:
                        if not isinstance(text, str):
                            raise ValueError('invalid content delta')
                        first = first or time.perf_counter_ns()
                        content.update(text.encode())
                        chunks += 1
                if isinstance(packet.get('usage'), dict):
                    count = packet['usage'].get('completion_tokens')
                    if type(count) is int and count >= 0:
                        usage = count
            if len(pending) > 65536:
                raise ValueError('oversized SSE event')
        elapsed = time.perf_counter_ns() - start
        if not done or first is None:
            raise ValueError('no generated content')
        return {'status': 'ok', 'ttft_ns': first - start, 'wall_ns': elapsed,
                'completion_tokens': usage, 'content_chunks': chunks, 'output_sha256': content.hexdigest(),
                'accepted_tokens_s': usage * 1e9 / elapsed if usage is not None else None}
    finally:
        connection.close()


def run_bench(port, prompt, model, lock, repeats=5, tokens=256, timeout=120):
    if not 1024 <= port <= 65535 or not 1 <= repeats <= 20 or not 1 <= tokens <= 4096 or not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise ValueError('invalid benchmark bounds')
    if not prompt or len(prompt.encode()) > 1024 * 1024:
        raise ValueError('prompt is empty or exceeds limit')
    rows = []
    for index in range(repeats + 1):
        try:
            row = completion(port, prompt, model, tokens, timeout)
        except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
            row = {'status': 'error', 'error': 'local_stream_failed'}
        rows.append(dict(row, sample=index, warmup=index == 0))
        if row['status'] != 'ok':
            break
    wall = sorted(r['wall_ns'] for r in rows if r['status'] == 'ok' and not r['warmup'])
    summary = None
    if wall:
        mid = len(wall) // 2
        summary = {'samples': len(wall), 'median_wall_ns': wall[mid] if len(wall) % 2 else (wall[mid-1] + wall[mid]) / 2,
                   'p95_wall_ns': wall[math.ceil(.95 * len(wall)) - 1]}
    return {'document_kind': 'backcluster_baseline_bench', 'schema_version': 1,
            'captured_at': datetime.now(timezone.utc).isoformat(), 'hardware_validated': False,
            'model_sha256': lock['sha256'], 'engine_commit': lock['engine_commit'],
            'server_identity_verified': False, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'output_token_limit': tokens, 'repeats': repeats,
            'ok': all(r['status'] == 'ok' for r in rows), 'samples': rows, 'summary': summary}
