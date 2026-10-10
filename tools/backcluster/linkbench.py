"""Finite bidirectional payload samples, optionally concurrent across workers."""
import json
import math
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .rpc import ProbeError
from .devices import engine_status

MAX_BYTES = 4 * 1024 * 1024
SIZES = (16 * 1024, 1024 * 1024, 2600 * 1024)


def sample(endpoint, size, timeout=10):
    if type(size) is not int or not 1 <= size <= MAX_BYTES or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('invalid payload size or timeout')
    pattern = bytes(range(251))
    payload = (pattern * ((size + 250) // 251))[:size]
    deadline = time.monotonic() + timeout

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise ProbeError('timeout')
        return left

    try:
        with socket.create_connection((endpoint.host, endpoint.port), remaining()) as conn:
            pending = bytearray()
            def send(data):
                conn.settimeout(remaining())
                conn.sendall(data)
            def read_line():
                while b'\n' not in pending:
                    if len(pending) >= 1024:
                        raise ProbeError('invalid_reply')
                    conn.settimeout(remaining())
                    chunk = conn.recv(1)  # do not consume binary bytes while timing the acknowledgment
                    if not chunk:
                        raise ProbeError('unexpected_eof')
                    pending.extend(chunk)
                line, extra = pending.split(b'\n', 1)
                pending[:] = extra
                return json.loads(line)
            send(('linkbench 1 %d\n' % size).encode())
            ready = read_line()
            if ready != {'document_kind': 'backcluster_link_ready', 'version': 1, 'bytes': size} or type(ready.get('version')) is not int or type(ready.get('bytes')) is not int:
                raise ProbeError('unsupported_link_bench')
            start = time.perf_counter_ns()
            send(payload)
            if read_line() != {'received_bytes': size}:
                raise ProbeError('invalid_reply')
            acknowledged = time.perf_counter_ns()
            received = bytearray()
            while len(received) < size:
                conn.settimeout(remaining())
                chunk = conn.recv(min(65536, size - len(received)))
                if not chunk:
                    raise ProbeError('unexpected_eof')
                received.extend(chunk)
            complete = time.perf_counter_ns()
            if received != payload:
                raise ProbeError('payload_mismatch')
            upload = max(1, acknowledged - start)
            download = max(1, complete - acknowledged)
            return {'payload_bytes': size, 'upload_ack_ns': upload, 'download_ns': download,
                    'round_trip_ns': complete - start,
                    'upload_effective_bytes_s': size * 1e9 / upload,
                    'download_effective_bytes_s': size * 1e9 / download, 'verified': True}
    except (TimeoutError, socket.timeout):
        raise ProbeError('timeout') from None
    except (ValueError, UnicodeError, TypeError, AttributeError):
        raise ProbeError('invalid_reply') from None
    except OSError:
        raise ProbeError('connection_failed') from None


def run_link_bench(endpoints, repeats=5, timeout=10, sizes=SIZES, concurrent=False):
    if not endpoints or len({(e.host, e.port) for e in endpoints}) != len(endpoints):
        raise ValueError('distinct explicit worker endpoints required')
    def worker(endpoint):
        rows = []
        for size in sizes:
            for index in range(repeats + 1):
                try:
                    row = sample(endpoint, size, timeout)
                    row.update(alias=endpoint.alias, sample=index, warmup=index == 0, status='ok')
                except ProbeError as error:
                    row = {'alias': endpoint.alias, 'payload_bytes': size, 'sample': index,
                           'warmup': index == 0, 'status': 'error', 'error': str(error)}
                    rows.append(row)
                    break  # no repeated stalled probes after a failed transfer of this size
                rows.append(row)
        return rows
    if concurrent:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            rows = [row for group in pool.map(worker, endpoints) for row in group]
    else:
        rows = [row for e in endpoints for row in worker(e)]
    summary = []
    for endpoint in endpoints:
        for size in sizes:
            values = sorted(r['round_trip_ns'] for r in rows if r['alias'] == endpoint.alias and
                            r['payload_bytes'] == size and r['status'] == 'ok' and not r['warmup'])
            if values:
                mid = len(values) // 2
                median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
                summary.append({'alias': endpoint.alias, 'payload_bytes': size, 'samples': len(values),
                                'median_round_trip_ns': median,
                                'p95_round_trip_ns': values[math.ceil(.95 * len(values)) - 1]})
    return {'document_kind': 'backcluster_link_bench', 'schema_version': 1,
            'captured_at': datetime.now(timezone.utc).isoformat(), 'engine': engine_status(),
            'concurrent': concurrent, 'transport': 'unverified', 'hardware_validated': False,
            'ok': all(r['status'] == 'ok' for r in rows), 'samples': rows, 'summary': summary}
