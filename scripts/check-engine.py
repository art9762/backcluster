#!/usr/bin/env python3
"""Build the reviewed CPU engine and test real three-stage inference without hardware."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests/backcluster'))
from generate_tiny_model import generate  # noqa: E402
from tools.backcluster.sharding import split_gguf  # noqa: E402

MAGIC = 0x4C545053


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def exact(sock, n):
    result = bytearray()
    while len(result) < n:
        data = sock.recv(n - len(result))
        if not data:
            raise RuntimeError('worker closed during reply')
        result.extend(data)
    return bytes(result)


def hello(port, source_hash, start, end, *, request=None):
    base = struct.pack('<8I', 2, 1, start, 8, 64, 32, 128, 4)
    settings = struct.pack('<iiIII4iQI', 1, 1, 0, 0, 0, 0, 0, 0, 0, 1729, 0)
    payload = base + settings + struct.pack('<IQ65s', end, 1729, source_hash.encode())
    with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
        sock.sendall(struct.pack('<IIQ', MAGIC, 19, len(payload)) + payload)
        magic, kind, size = struct.unpack('<IIQ', exact(sock, 16))
        if magic != MAGIC or size > 4096:
            raise RuntimeError('invalid HELLO response')
        reply = exact(sock, size)
        if request is None:
            if kind != 10:
                raise RuntimeError('incompatible stage HELLO was accepted')
        else:
            if kind != 20:
                raise RuntimeError('valid stage HELLO rejected: ' + reply.decode(errors='replace'))
            sock.sendall(struct.pack('<IIQ', MAGIC, 21, len(request)) + request)
            _, kind, size = struct.unpack('<IIQ', exact(sock, 16))
            exact(sock, size)
            if kind != 10:
                raise RuntimeError('invalid stage CHUNK was accepted')


def check_protocol(port, source_hash):
    hello(port, '0' * 64, 2, 6)
    hello(port, source_hash, 2, 4)
    # A request with a valid residual but an unrelated epoch cannot mutate memory.
    residual = struct.pack('<64f', *([0.0] * 64))
    request = struct.pack('<iIIIQQ', 0, 1, 0, 0, 999, 1) + residual
    hello(port, source_hash, 2, 6, request=request)
    request = struct.pack('<iIIIQQ', 0, 1, 0, 0, 1729, 2) + residual
    hello(port, source_hash, 2, 6, request=request)
    with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
        sock.sendall(struct.pack('<IIQ', MAGIC, 19, (512 << 20) + 1))
        if sock.recv(1):
            raise RuntimeError('oversize frame was accepted')



def check_state_restore(port, source_hash):
    base = struct.pack('<8I', 2, 1, 2, 8, 64, 32, 128, 4)
    settings = struct.pack('<iiIII4iQI', 1, 1, 0, 0, 0, 0, 0, 0, 0, 1729, 0)
    handshake = base + settings + struct.pack('<IQ65s', 6, 1729, source_hash.encode())
    with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
        def exchange(kind, payload, expected):
            sock.sendall(struct.pack('<IIQ', MAGIC, kind, len(payload)) + payload)
            magic, reply, size = struct.unpack('<IIQ', exact(sock, 16))
            if magic != MAGIC or reply != expected or size > 512 << 20:
                raise RuntimeError('invalid state restore response')
            return exact(sock, size)
        exchange(19, handshake, 20)
        residual = struct.pack('<64f', *[0.01 * (i % 7 - 3) for i in range(64)])
        def chunk(pos, request_id):
            request = struct.pack('<iIIIQQ', pos, 1, 0, 0, 1729, request_id) + residual
            # Packed chunk_rep3 is 44 bytes. The residual follows it.
            return exchange(21, request, 22)[44:]
        chunk(0, 1)
        snapshot = exchange(7, b'', 8)
        before = chunk(1, 2)
        restored = exchange(23, struct.pack('<QI', 1729, 1) + snapshot, 15)
        if restored != struct.pack('<I', 1):
            raise RuntimeError('restored watermark mismatch')
        after = chunk(1, 3)
        if len(before) != 64 * 4 or before != after:
            raise RuntimeError('GDN/KV snapshot restore changed residual output')
    print('PASS: native stage GDN/KV state snapshot, restore and exact residual replay.')


def check_invalid_state(port, source_hash):
    # These headers must be rejected without waiting for their advertised body.
    for kind, size in ((19, 512 << 20), (16, 4096), (23, 512 << 20)):
        with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
            sock.sendall(struct.pack('<IIQ', MAGIC, kind, size))
            magic, reply, length = struct.unpack('<IIQ', exact(sock, 16))
            if magic != MAGIC or reply != 10 or length > 4096:
                raise RuntimeError('invalid frame was not rejected before payload reception')
            exact(sock, length)
    base = struct.pack('<8I', 2, 1, 2, 8, 64, 32, 128, 4)
    settings = struct.pack('<iiIII4iQI', 1, 1, 0, 0, 0, 0, 0, 0, 0, 1729, 0)
    handshake = base + settings + struct.pack('<IQ65s', 6, 1729, source_hash.encode())
    # Seq state starts with magic/seq_id, then hybrid KV n_stream/cell_count.
    # The first case must fail before reserving a batch of 2^32-1 cells.
    huge_kv = struct.pack('<IiII', 0xaf143cd8, 0, 1, 0xffffffff)
    # An empty KV stream has no tensor layout; its next word is recurrent count.
    huge_recurrent = struct.pack('<IiIII', 0xaf143cd8, 0, 1, 0, 0xffffffff)
    for state in (huge_kv, huge_recurrent):
        with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
            def exchange(kind, payload, expected):
                sock.sendall(struct.pack('<IIQ', MAGIC, kind, len(payload)) + payload)
                magic, reply, size = struct.unpack('<IIQ', exact(sock, 16))
                if magic != MAGIC or reply != expected or size > 4096:
                    raise RuntimeError('invalid malformed-state response')
                return exact(sock, size)
            exchange(19, handshake, 20)
            exchange(23, struct.pack('<QI', 1729, 0) + state, 10)
    # Invalid restores must not kill the worker or leave a partial KV/GDN state.
    check_state_restore(port, source_hash)
    print('PASS: oversized inner KV/recurrent counts fail before state allocation; worker recovers.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / '.backcluster/native-test-build')
    parser.add_argument('--engine-source', type=Path, help='use an explicit source tree for local development')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 2, 4))
    args = parser.parse_args()
    if args.engine_source:
        source = args.engine_source.resolve()
    else:
        from tools.backcluster.engine_overlay import prepare_engine
        source = prepare_engine()['source_dir']
    build = args.build_dir.resolve()
    if not args.skip_build:
        cmake = shutil.which('cmake')
        if cmake is None:
            raise RuntimeError('cmake is required for the native inference gate')
        run([cmake, '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF',
             '-DGGML_AVX=OFF', '-DGGML_AVX2=OFF', '-DGGML_FMA=OFF', '-DGGML_F16C=OFF',
             '-DGGML_OPENMP=OFF', '-DGGML_METAL=OFF', '-DGGML_BLAS=OFF', '-DLLAMA_BUILD_TESTS=OFF',
             '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_OPENSSL=OFF', '-DBUILD_SHARED_LIBS=ON'])
        run([cmake, '--build', build, '--target', 'llama-split-prefill', '-j', args.jobs])
    binary = build / 'bin'
    test = binary / 'backcluster-pipeline-test'
    run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-pthread',
         ROOT / 'tests/backcluster/pipeline-native-test.cpp', '-I' + str(source / 'include'),
         '-I' + str(source / 'ggml/include'), '-isystem', str(source / 'tools/split-prefill'), '-L' + str(binary), '-Wl,-rpath,' + str(binary),
         '-lllama', '-lggml', '-lggml-base', '-o', test])
    processes = []
    with tempfile.TemporaryDirectory(prefix='backcluster-native-') as temporary:
        directory = Path(temporary)
        full = directory / 'full.gguf'; generate(full)
        middle = directory / 'middle.gguf'; tail = directory / 'tail.gguf'
        from tools.backcluster.devices import engine_status
        pin = engine_status()['expected_commit']
        split_gguf(full, middle, start=2, end=6, role='middle', engine_commit=pin)
        split_gguf(full, tail, start=6, end=8, role='tail', engine_commit=pin)
        source_hash = hashlib.sha256(full.read_bytes()).hexdigest()
        try:
            for shard in (middle, tail):
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
                log = (directory / (shard.stem + '.log')).open('w')
                env = {key: value for key, value in os.environ.items() if not key.startswith('LLAMA_SPLIT_')}
                process = subprocess.Popen([str(test), '--worker', str(port), str(shard)],
                                           stdout=log, stderr=log, env=env)
                processes.append((process, port, log))
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError('worker failed to load: ' + (directory / (shard.stem + '.log')).read_text())
                    try:
                        with socket.create_connection(('127.0.0.1', port), timeout=.1):
                            break
                    except OSError:
                        time.sleep(.05)
                else:
                    raise RuntimeError('worker startup timed out')
            check_protocol(processes[0][1], source_hash)
            check_state_restore(processes[0][1], source_hash)
            check_invalid_state(processes[0][1], source_hash)
            for resid_f16 in (False, True):
                env = dict(os.environ)
                if resid_f16: env['LLAMA_SPLIT_RESID_F16'] = '1'
                else: env.pop('LLAMA_SPLIT_RESID_F16', None)
                run([test, full, '127.0.0.1:' + str(processes[0][1]), '127.0.0.1:' + str(processes[1][1]),
                     2, 6, source_hash, processes[0][0].pid, processes[1][0].pid], env=env, timeout=120)
            env = dict(os.environ); env.pop('LLAMA_SPLIT_RESID_F16', None)
            run([test, full, 'unused', '127.0.0.1:' + str(processes[1][1]),
                 6, 0, source_hash, processes[0][0].pid, processes[1][0].pid], env=env, timeout=120)
        except Exception:
            for path in directory.glob('*.log'):
                print(path.name + ':\n' + path.read_text()[-12000:], file=sys.stderr)
            raise
        finally:
            for process, _, log in processes:
                process.kill(); process.wait(); log.close()
    print('Native engine gate passed: real CPU inference, framing, source/range/epoch checks, cancellation and recovery.')


if __name__ == '__main__':
    main()
