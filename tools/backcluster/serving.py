"""Explicit Mac, pair and remote-KV launch receipts with fresh cable admission.

No discovery, route changes, sysctl tuning, drafter or implicit fallback policy.
"""
import copy
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import digest, verify_lock, load_json
from .devices import ROOT, engine_status, collect_profile, run
from .planning import (active_workers, admission, host_admission, make_plan,
                       registry_checks, state_estimate, tensor_map, kv_bytes_per_token, KV_TYPES)
from .rpc import Endpoint
from .profiles import validate_profile

def _build_identity():
    # A checkout without overlay patches remains usable for the Mac baseline.
    if not (ROOT / 'engine-patches/manifest.json').is_file():
        return None
    from .engine_overlay import expected_identity
    return expected_identity()


def launch_spec(lock, model, context=16384, port=8080):
    engine = engine_status()
    verify_lock(lock, model, engine['expected_commit'])
    split = lock['gguf']['split']
    if any(key in split for key in ('split.layer_start', 'split.layer_end', 'split.role', 'split.n_layer_full')) or split.get('split.count', 1) != 1:
        raise ValueError('Mac baseline requires an unsplit model')
    if lock['gguf']['chat_template_sha256'] is None:
        raise ValueError('chat template required for coding baseline')
    limit = lock['gguf'].get('context_length')
    if type(context) is not int or not 128 <= context <= 131072 or (type(limit) is int and context > limit):
        raise ValueError('context exceeds supported model range')
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('invalid local server port')
    executable = ROOT / 'llama.cpp/build-metal/bin/llama-server'
    argv = [str(executable), '--model', str(Path(model).resolve()), '--host', '127.0.0.1', '--port', str(port),
            '--alias', 'backcluster-' + lock['sha256'][:16], '--ctx-size', str(context), '--n-gpu-layers', '99',
            '--parallel', '1', '--batch-size', '512', '--ubatch-size', '128', '--flash-attn', 'on',
            '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--load-mode', 'mmap', '--spec-type', 'none',
            '--cache-ram', '0', '--ctx-checkpoints', '0', '--jinja']
    return {'document_kind': 'backcluster_launch', 'schema_version': 1, 'mode': 'mac',
            'model_sha256': lock['sha256'], 'engine_commit': engine['expected_commit'],
            'engine_build_id': _build_identity(), 'bind': '127.0.0.1', 'port': port, 'context': context,
            'hardware_validated': False, 'draft_enabled': False, 'remote_workers': [], 'argv': argv,
            'environment': {}, 'pending_live_checks': [],
            'artifacts': [{'kind': 'model', 'path': str(Path(model).resolve()), 'receipt': copy.deepcopy(lock)}]}


def _set_arg(argv, name, value):
    argv[argv.index(name) + 1] = str(value)


def remote_position_counts(total, workers, block=4096):
    """Pinned engine's block-cyclic mapping, including the unequal final block."""
    cycle = workers * block
    rounds, remainder = divmod(total, cycle)
    return [rounds * block + max(0, min(block, remainder - i * block)) for i in range(workers)]


def _tail_identity(device, manifest, spec):
    tail = device.get('loaded_tail')
    if not isinstance(tail, dict) or tail.get('state') not in ('ready', 'connected'):
        raise ValueError('tail worker has no idle, content-verified model loaded')
    for key, expected in (('artifact_sha256', manifest['sha256']), ('source_sha256', spec['model_sha256']),
                          ('engine_commit', spec['engine_commit']), ('layer_start', manifest['layer_start']),
                          ('layer_end', manifest['layer_end']), ('output_head', manifest['output_head']), ('bytes', manifest['bytes'])):
        if tail.get(key) != expected:
            raise ValueError('loaded tail artifact identity or layer range differs from launch receipt')
    if spec.get('engine_build_id') is not None and tail.get('engine_build_id') != spec['engine_build_id']:
        raise ValueError('loaded tail engine build identity differs from host')
    # Native receipt context is the loaded model's supported training context;
    # the actual cache is created by the engine HELLO after this admission.
    if type(tail.get('context')) is not int or tail['context'] < spec['context']:
        raise ValueError('loaded tail model context is smaller than the launch context')
    return tail


def cluster_launch_spec(lock, model, context=16384, port=8080, *, mode='mac', registry=None, profile=None,
                        host_budget=None, host_reserve=None, host_state_bytes=None, state_estimates=(),
                        split_layer=None, shard_manifest=None, shard_model=None, total_context=None,
                        cache_type='q8_0', timeout=3.0, middle_end=None, middle_manifest=None,
                        middle_model=None, host_manifest=None, host_model=None):
    """Construct an offline receipt; serve() repeats identity and memory live.

    Worker state estimates describe peak KV/recurrent/compute/transfer allocations
    separately from weights. Integer values are conservative operator aggregates.
    """
    spec = launch_spec(lock, model, context, port)
    if type(timeout) not in (int, float) or not 0.1 <= timeout <= 30:
        raise ValueError('worker probe timeout must be between 0.1 and 30 seconds')
    if cache_type not in KV_TYPES:
        raise ValueError('supported cache types: q8_0, q4_0, f16')
    _set_arg(spec['argv'], '--cache-type-k', cache_type)
    _set_arg(spec['argv'], '--cache-type-v', cache_type)
    workers = active_workers(registry, mode)
    if mode == 'mac':
        if any(v is not None for v in (split_layer, shard_manifest, shard_model, total_context,
                                      middle_end, middle_manifest, middle_model, host_manifest, host_model)) or state_estimates:
            raise ValueError('distributed placement arguments require a distributed launch mode')
        if profile is None:
            if any(v is not None for v in (host_budget, host_reserve, host_state_bytes)):
                raise ValueError('host admission estimates require an explicit macOS snapshot')
            return spec
    if mode != 'mac' and lock['gguf'].get('architecture') != 'qwen35':
        raise ValueError('pinned distributed engine supports qwen35 only')
    if profile is None or host_state_bytes is None or host_budget is None or host_reserve is None:
        raise ValueError('cluster launch requires explicit snapshot, host budget/reserve and peak state estimates')
    states = dict(state_estimates)
    if set(states) != {w['alias'] for w in workers}:
        raise ValueError('each active worker requires exactly one explicit state estimate')
    if len(states) != len(state_estimates):
        raise ValueError('worker state estimate aliases must be unique')
    host_state = state_estimate(host_state_bytes)
    states = {alias: state_estimate(value) for alias, value in states.items()}
    gguf, layers = tensor_map(lock)
    trunk = gguf['trunk_layers']
    stage_artifacts = []
    if mode in ('split-prefill', 'split-decode', 'capacity', 'pipeline'):
        if type(split_layer) is not int or not 0 < split_layer < trunk:
            raise ValueError('pair launch requires an explicit split inside trunk layers')
        if shard_manifest is None or shard_model is None:
            raise ValueError('pair launch requires the local verified tail shard and its manifest')
        from .sharding import verify_shard, validate_partition
        verify_shard(shard_manifest, shard_model, spec['engine_commit'], source_path=model)
        tail_start = middle_end if mode == 'pipeline' else split_layer
        if (shard_manifest.get('source', {}).get('sha256') != lock['sha256'] or
                shard_manifest.get('role') != 'tail' or shard_manifest.get('layer_start') != tail_start or
                shard_manifest.get('layer_end') != trunk):
            raise ValueError('tail shard must match this source and the explicit split range')
        if mode in ('split-decode', 'capacity', 'pipeline') and shard_manifest.get('output_head') is not True:
            raise ValueError('split decode requires a tail with its output head')
        transform = shard_manifest.get('transform', {})
        if transform.get('no_ffn') or transform.get('no_ffn_first', 0):
            raise ValueError('ANE-only tail weights are not supported by this launcher')
        if mode in ('split-decode', 'capacity', 'pipeline') and context > 65536:
            raise ValueError('split decode context is limited to 65536 in this launcher')
        if total_context not in (None, context):
            raise ValueError('tail compute and remote KV cannot be combined')
        worker = workers[-1]
        stage_artifacts.append({'kind': 'shard', 'alias': worker['alias'], 'path': str(Path(shard_model).resolve()), 'receipt': copy.deepcopy(shard_manifest)})
        env = {'LLAMA_SPLIT_TAIL': '%s:%d' % (worker['host'], worker.get('tail_port', 50060)),
               'LLAMA_SPLIT_L': str(split_layer), 'LLAMA_SPLIT_TIMEOUT_S': '20' if mode == 'split-prefill' else '120',
               'LLAMA_SPLIT_PUSH_ASYNC': '0', 'LLAMA_SPLIT_MODEL_SHA256': lock['sha256']}
        if mode == 'pipeline':
            if type(middle_end) is not int or not split_layer < middle_end < trunk or middle_manifest is None or middle_model is None:
                raise ValueError('pipeline needs a middle shard and explicit ordered layer boundaries')
            verify_shard(middle_manifest, middle_model, spec['engine_commit'], source_path=model)
            if (middle_manifest.get('source') != shard_manifest.get('source') or
                    middle_manifest.get('role') != 'middle' or middle_manifest.get('layer_start') != split_layer or
                    middle_manifest.get('layer_end') != middle_end or middle_manifest.get('output_head') is not False):
                raise ValueError('middle shard identity, range or output policy differs from pipeline')
            middle_transform = middle_manifest['transform']
            if middle_transform.get('no_ffn') or middle_transform.get('no_ffn_first'):
                raise ValueError('pipeline requires GPU-owned FFN weights on each worker')
            middle = workers[0]
            stage_artifacts.append({'kind': 'shard', 'alias': middle['alias'], 'path': str(Path(middle_model).resolve()), 'receipt': copy.deepcopy(middle_manifest)})
            env.update(LLAMA_SPLIT_MIDDLE='%s:%d' % (middle['host'], middle.get('tail_port', 50060)),
                       LLAMA_SPLIT_MIDDLE_END=str(middle_end))
            spec['middle_end'] = middle_end
        elif middle_end is not None or middle_manifest is not None or middle_model is not None:
            raise ValueError('middle stage requires pipeline mode')
        if (host_manifest is None) != (host_model is None):
            raise ValueError('host partition verification requires both its manifest and shard')
        if host_manifest is not None:
            verify_shard(host_manifest, host_model, spec['engine_commit'], source_path=model)
            if host_manifest.get('layer_end') != split_layer:
                raise ValueError('host shard range differs from launch boundary')
            spec['partition'] = validate_partition([host_manifest] + [a['receipt'] for a in stage_artifacts], spec['engine_commit'])
            spec['artifacts'].append({'kind': 'partition_host', 'path': str(Path(host_model).resolve()), 'receipt': copy.deepcopy(host_manifest)})
        spec['artifacts'].extend(stage_artifacts)
        if mode in ('split-decode', 'capacity', 'pipeline'):
            env['LLAMA_SPLIT_DECODE'] = '1'
            env['LLAMA_SPLIT_GPU_WARM_US'] = '0'
            # Enumerate actual block IDs. No comma in the override's regex: the
            # engine uses commas to split overrides. Include MTP on the CPU.
            suffix = '|'.join(str(i) for i in range(split_layer, gguf['block_count']))
            spec['argv'].extend(['--override-tensor', '^blk\\.(' + suffix + ')\\.=CPU',
                                 '--override-tensor', '^output=CPU', '--no-repack', '--no-context-shift'])
        else:
            # Engine dispatches all but the final ubatch to the tail. These
            # settings guarantee that a full batch reaches the 512-token gate.
            _set_arg(spec['argv'], '--batch-size', 1024)
            _set_arg(spec['argv'], '--ubatch-size', 256)
            env.update(LLAMA_SPLIT_MIN='512', LLAMA_SPLIT_ONE_BATCH='1')
        spec.update(environment=env, split_layer=split_layer, total_context=context)
    elif mode == 'remote-kv':
        if any(v is not None for v in (middle_end, middle_manifest, middle_model, host_manifest, host_model)):
            raise ValueError('remote KV cannot be combined with a distributed compute partition')
        if split_layer is not None or shard_manifest is not None or shard_model is not None:
            raise ValueError('remote KV cannot be combined with tail compute')
        query, heads, dk, dv = (gguf.get(k) for k in ('head_count', 'head_count_kv', 'key_length', 'value_length'))
        if any(type(n) is not int or n <= 0 for n in (query, heads, dk, dv)) or dk != 256 or dv != 256 or query != heads * 6:
            raise ValueError('remote KV requires inspected 256-dimensional heads and six query heads per KV head')
        limit = gguf.get('context_length')
        if type(total_context) is not int or not context < total_context <= 262144 or type(limit) is not int or total_context > limit:
            raise ValueError('remote KV needs explicit total context above local context and within model metadata')
        if context % 256 or total_context % 256 or context < 4096:
            raise ValueError('remote KV contexts must be multiples of 256, with at least 4096 local cells')
        bytes_per_token = kv_bytes_per_token(gguf, cache_type=cache_type)
        if bytes_per_token is None or not bytes_per_token:
            raise ValueError('remote KV requires inspected hybrid attention layout')
        remote_positions = ((total_context - context + 4095) // 4096) * 4096
        counts = remote_position_counts(remote_positions, len(workers))
        for worker, count in zip(workers, counts):
            if states[worker['alias']]['total_bytes'] < count * bytes_per_token:
                raise ValueError('worker state estimate is smaller than its block-cyclic KV allocation')
        if host_state['total_bytes'] < context * bytes_per_token:
            raise ValueError('host state estimate is smaller than its local KV allocation')
        spec.update(total_context=total_context, remote_positions=[{'alias': w['alias'], 'positions': n,
                    'minimum_kv_bytes': n * bytes_per_token} for w, n in zip(workers, counts)],
                    environment={'LLAMA_KV_REMOTE': ','.join('%s:%d' % (w['host'], w.get('kv_port', 50062)) for w in workers),
                                 'LLAMA_KV_REMOTE_CTX': str(total_context), 'GGML_METAL_REMOTE_BLOCK': '4096',
                                 'LLAMA_KV_REMOTE_PAGE': '4096', 'LLAMA_REMOTE_PIPE': '0'})
    elif mode != 'mac':
        raise ValueError('unsupported launch mode')
    if mode != 'remote-kv':
        head_end = split_layer if mode in ('capacity', 'split-decode', 'pipeline') else trunk
        host_kv = kv_bytes_per_token(gguf, 0, head_end, cache_type)
        if host_kv is not None and host_state['total_bytes'] < context * host_kv:
            raise ValueError('host state estimate is smaller than its full-attention KV allocation')
        for artifact in stage_artifacts:
            manifest = artifact['receipt']
            worker_kv = kv_bytes_per_token(gguf, manifest['layer_start'], manifest['layer_end'], cache_type)
            if worker_kv is not None and states[artifact['alias']]['total_bytes'] < context * worker_kv:
                raise ValueError('worker state estimate is smaller than its full-attention KV allocation')
    plan = make_plan(lock, profile, registry, spec['engine_commit'], mode, host_budget, host_reserve, context,
                     [(alias, state['components']) for alias, state in states.items()], host_state_bytes=host_state['components'],
                     split_layer=split_layer, middle_end=middle_end, engine_build_id=spec['engine_build_id'], cache_type=cache_type)
    # Admission uses the exact transformed shard weights, including output and
    # any replicated embedding, instead of the planner's conservative shared map.
    if workers and mode != 'remote-kv':
        plan['worker_admission'] = []
        for worker in workers:
            artifact = next(a for a in stage_artifacts if a['alias'] == worker['alias'])
            manifest = artifact['receipt']
            assignment = next(a for a in plan['assignments'] if a['alias'] == worker['alias'])
            assignment['weight_bytes'] = manifest['gguf']['tensor_bytes']
            assignment['requested_bytes'] = assignment['weight_bytes'] + states[worker['alias']]['total_bytes']
            device = next((row['device'] for row in profile['workers'] if row.get('alias') == worker['alias'] and row.get('status') == 'ok'), {})
            loaded = device.get('loaded_tail') or {}
            existing = assignment['weight_bytes'] if loaded.get('artifact_sha256') == manifest['sha256'] else 0
            a = admission(device, worker['allocation_budget_bytes'], worker['reserve_bytes'], assignment['requested_bytes'], existing_bytes=existing)
            assignment.update(a)
            plan['worker_admission'].append(dict(a, alias=worker['alias']))
        if all(a['ok'] for a in plan['worker_admission']):
            plan['blockers'] = [reason for reason in plan['blockers'] if reason != 'worker_admission_failed']
        # Replace the conservative search result consistently with this exact,
        # source-verified static layout in every part of the dry-run receipt.
        for candidate in plan['split_candidates']:
            if candidate['split_layer'] == split_layer:
                candidate['worker_admission'] = copy.deepcopy(plan['worker_admission'])
                candidate['feasible_estimate'] = all(a['ok'] for a in candidate['assignments'])
    active_aliases = {w['alias'] for w in workers}
    if any(not c['ok'] for c in plan['compatibility'] if c['alias'] in active_aliases):
        raise ValueError('snapshot worker model, protocol or engine identity mismatch')
    if any(not a['ok'] for a in plan['assignments']):
        raise ValueError('explicit host or worker peak memory estimate is not admitted')
    plan['feasible_estimate'] = not plan['blockers']
    spec.update(mode=mode, remote_workers=[w['alias'] for w in workers], snapshot_plan=plan,
                host_budget_bytes=host_budget, host_reserve_bytes=host_reserve, host_state=host_state,
                worker_states=states, registry=copy.deepcopy(registry), timeout=float(timeout),
                pending_live_checks=['route', 'worker_model_protocol_engine_identity', 'memory', 'thermal'] +
                (['loaded_tail_artifact_identity'] if mode in ('split-prefill', 'split-decode', 'capacity', 'pipeline') else []))
    spec['failure_policy'] = ('rerun_full_batch_on_mac_with_mirrored_state' if mode == 'split-prefill'
                              else 'stop_on_remote_state_or_transport_failure' if workers else 'local_engine_error')
    return spec


def verify_routes(registry, mode, route_reader=None):
    reader = route_reader or (lambda host: run('/sbin/route', '-n', 'get', host))
    for worker in active_workers(registry, mode):
        route = reader(worker['host'])
        if not isinstance(route, str) or not any(line.strip() == 'interface: ' + worker['interface'] for line in route.splitlines()):
            raise ValueError('live cable route differs from registry')


def check_live_launch(spec, *, route_reader=None, profile_reader=None):
    """Fresh identity, thermal and allocation check immediately before exec."""
    registry = spec.get('registry')
    if registry is None and spec['mode'] == 'mac' and 'host_budget_bytes' not in spec:
        return {'ok': True, 'checks': [], 'hardware_validated': False}
    workers = active_workers(registry, spec['mode'])
    verify_routes(registry, spec['mode'], route_reader)
    endpoints = [Endpoint(w['alias'], w['host'], w['control_port']) for w in workers]
    profile = (profile_reader or collect_profile)(endpoints, spec.get('timeout', 3.0))
    if validate_profile(profile):
        raise ValueError('invalid live device profile')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(profile['captured_at'].replace('Z', '+00:00'))).total_seconds()
    if not 0 <= age <= 60:
        raise ValueError('live device profile is stale')
    checks = registry_checks(registry, profile, spec['engine_commit'], spec.get('engine_build_id')) if registry else []
    aliases = {w['alias'] for w in workers}
    if any(not row['ok'] for row in checks if row['alias'] in aliases):
        raise ValueError('live worker model, protocol or engine identity mismatch')
    host = spec['snapshot_plan']['assignments'][0]
    host_a = host_admission(profile, spec['host_budget_bytes'], spec['host_reserve_bytes'], host['requested_bytes'])
    if profile['host'].get('platform') != 'macOS' or not host_a['ok']:
        raise ValueError('live host memory budget is not admitted')
    rows = {row['alias']: row.get('device', {}) for row in profile['workers']}
    admissions = []
    for w in workers:
        assignment = next(a for a in spec['snapshot_plan']['assignments'] if a['alias'] == w['alias'])
        existing = 0
        if spec['mode'] in ('split-prefill', 'split-decode', 'capacity', 'pipeline'):
            manifest = next(a['receipt'] for a in spec['artifacts'] if a['kind'] == 'shard' and a['alias'] == w['alias'])
            _tail_identity(rows[w['alias']], manifest, spec)
            existing = assignment['weight_bytes']
        a = admission(rows[w['alias']], w['allocation_budget_bytes'], w['reserve_bytes'], assignment['requested_bytes'], existing_bytes=existing)
        if not a['ok']:
            raise ValueError('live worker peak memory or thermal state is not admitted')
        admissions.append(dict(a, alias=w['alias']))
    return {'ok': True, 'checks': checks, 'host_admission': host_a, 'worker_admission': admissions, 'hardware_validated': False}


def clean_environment(settings=None):
    keys = ('PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'DEVELOPER_DIR')
    env = {k: os.environ[k] for k in keys if k in os.environ}
    env.update(GGML_METAL_FA_SME='0', GGML_METAL_MM_SME='0')
    if settings:
        env.update(settings)
    return env


def build_environment():
    """Keep configured build networking and certificate trust out of inference."""
    env = clean_environment()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
                'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'GIT_SSL_CAINFO'):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _runtime_files(directory):
    """The server executable alone does not identify its dynamic engine code."""
    return {path.name: digest(path) for path in sorted(Path(directory).iterdir())
            if path.is_file() and (path.name == 'llama-server' or path.suffix in ('.dylib', '.metallib'))}


def build_engine():
    status = engine_status()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not status['matches_pin']:
        raise ValueError('build requires Apple Silicon macOS and a clean pinned engine; run backcluster doctor --human')
    if not shutil.which('cmake'):
        raise ValueError('cmake is missing; install CMake, then run backcluster build')
    source = ROOT / 'llama.cpp'
    build_id = _build_identity()
    if build_id is not None:
        from .engine_overlay import prepare_engine
        prepared = prepare_engine()
        source = prepared['source_dir']
        if prepared['build_id'] != build_id:
            raise ValueError('engine overlay identity changed during preparation')
    target = ROOT / 'llama.cpp/build-metal'
    # Reusing a build from the base checkout would silently omit the overlay.
    cache = target / 'CMakeCache.txt'
    if cache.is_file():
        home = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', cache.read_text(), re.MULTILINE)
        if home and Path(home.group(1)).resolve() != source.resolve():
            raise ValueError('existing build directory uses another source; remove llama.cpp/build-metal and rebuild')
    printed_commit = status['expected_commit'][:12] + ('-bc-' + build_id[:12] if build_id else '')
    subprocess.run(['cmake', '-S', str(source), '-B', str(target), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_METAL=ON', '-DGGML_RPC=ON', '-DLLAMA_OPENSSL=OFF',
                    '-DLLAMA_BUILD_NUMBER=0', '-DLLAMA_BUILD_COMMIT=' + printed_commit], env=build_environment(), check=True)
    subprocess.run(['cmake', '--build', str(target), '--target', 'llama-server', '-j', str(min(os.cpu_count() or 4, 8))],
                   env=build_environment(), check=True)
    if _build_identity() != build_id:
        raise ValueError('engine overlay changed during build; rebuild before serving')
    receipt = {'engine_commit': status['expected_commit'], 'engine_build_id': build_id,
               'executable_sha256': digest(target / 'bin/llama-server'),
               'runtime_files': _runtime_files(target / 'bin')}
    (target / 'backcluster-build.json').write_text(json.dumps(receipt, sort_keys=True) + '\n', encoding='utf-8')


def serve(spec):
    status = engine_status()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not status['matches_pin']:
        raise ValueError('serve requires Apple Silicon macOS and a clean pinned engine; run backcluster doctor --human')
    executable = ROOT / 'llama.cpp/build-metal/bin/llama-server'
    if Path(spec['argv'][0]) != executable or not executable.is_file():
        raise ValueError('llama-server is missing; run backcluster build first')
    if spec['engine_commit'] != status['expected_commit'] or spec.get('engine_build_id') != _build_identity():
        raise ValueError('launch receipt engine identity changed; regenerate the launch')
    if spec.get('engine_build_id') is not None:
        build = load_json(executable.parent.parent / 'backcluster-build.json')
        if build.get('engine_commit') != spec['engine_commit'] or build.get('engine_build_id') != spec['engine_build_id'] or build.get('executable_sha256') != digest(executable) or build.get('runtime_files') != _runtime_files(executable.parent):
            raise ValueError('server binary does not match this engine build; run backcluster build')
    source = next(a for a in spec['artifacts'] if a['kind'] == 'model')
    verify_lock(source['receipt'], source['path'], spec['engine_commit'])
    for artifact in spec['artifacts']:
        if artifact['kind'] in ('shard', 'partition_host'):
            from .sharding import verify_shard
            verify_shard(artifact['receipt'], artifact['path'], spec['engine_commit'], source_path=source['path'])
    check_live_launch(spec)
    os.execve(spec['argv'][0], spec['argv'], clean_environment(spec.get('environment')))
