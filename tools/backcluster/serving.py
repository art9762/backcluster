"""Fork-safe, loopback-only Mac baseline launcher with no discovery or tuning."""
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .artifacts import verify_lock
from .devices import ROOT, engine_status


def launch_spec(lock, model, context=16384, port=8080):
    engine = engine_status()
    verify_lock(lock, model, engine['expected_commit'])
    if lock['gguf']['split']:
        raise ValueError('Mac baseline requires an unsplit model')
    if lock['gguf']['chat_template_sha256'] is None:
        raise ValueError('chat template required for coding baseline')
    limit = lock['gguf'].get('context_length')
    if type(context) is not int or not 128 <= context <= 131072 or (type(limit) is int and context > limit):
        raise ValueError('context exceeds supported model range')
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('invalid local server port')
    executable = ROOT / 'llama.cpp/build-metal/bin/llama-server'
    # No upstream shell launcher: it auto-selects workers and may tune sysctl.
    argv = [str(executable), '--model', str(Path(model).resolve()), '--host', '127.0.0.1', '--port', str(port),
            '--alias', 'backcluster-' + lock['sha256'][:16], '--ctx-size', str(context), '--n-gpu-layers', '99', '--parallel', '1', '--batch-size', '512',
            '--ubatch-size', '128', '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--load-mode', 'mmap', '--spec-type', 'none']
    return {'document_kind': 'backcluster_launch', 'schema_version': 1, 'mode': 'mac',
            'model_sha256': lock['sha256'], 'engine_commit': engine['expected_commit'],
            'bind': '127.0.0.1', 'port': port, 'context': context, 'hardware_validated': False,
            'draft_enabled': False, 'remote_workers': [], 'argv': argv}


def clean_environment():
    # Keep only normal host/build variables. Remove undocumented inference
    # overrides and proxy/loader injection from an inherited shell session.
    keys = ('PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'DEVELOPER_DIR')
    env = {k: os.environ[k] for k in keys if k in os.environ}
    env.update(GGML_METAL_FA_SME='0', GGML_METAL_MM_SME='0')
    return env


def build_engine():
    status = engine_status()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not status['matches_pin']:
        raise ValueError('build requires Apple Silicon and a clean pinned engine')
    if not shutil.which('cmake'):
        raise ValueError('cmake required')
    target = ROOT / 'llama.cpp/build-metal'
    subprocess.run(['cmake', '-S', str(ROOT / 'llama.cpp'), '-B', str(target), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_METAL=ON', '-DGGML_RPC=ON', '-DLLAMA_OPENSSL=OFF'],
                   env=clean_environment(), check=True)
    subprocess.run(['cmake', '--build', str(target), '--target', 'llama-server', '-j', str(min(os.cpu_count() or 4, 8))],
                   env=clean_environment(), check=True)


def serve(spec):
    status = engine_status()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not status['matches_pin']:
        raise ValueError('serve requires Apple Silicon and a clean pinned engine')
    if not Path(spec['argv'][0]).is_file():
        raise ValueError('build the pinned engine first')
    # exec preserves Ctrl-C/cancellation and exit status; no orphan process.
    os.execve(spec['argv'][0], spec['argv'], clean_environment())
