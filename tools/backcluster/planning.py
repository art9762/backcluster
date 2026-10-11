"""Explicit device placement, independent memory limits and inspectable estimates.

A plan never proves numerical correctness or grants permission to skip live checks.
"""
import ipaddress
import math
import re
from datetime import datetime, timezone

from .profiles import integer, valid_alias, validate_profile
from .rpc import Endpoint
from .artifacts import validate_lock

SHA = re.compile(r'[0-9a-f]{40}')
ROLES = {'remote_kv', 'tail_stage', 'middle_stage', 'idle'}
MODES = {'mac', 'remote-kv', 'split-prefill', 'split-decode', 'capacity', 'pipeline'}
STATE_COMPONENTS = {'kv_bytes', 'recurrent_bytes', 'compute_bytes', 'transfer_bytes',
                    'checkpoint_bytes', 'other_bytes'}
KV_TYPES = {'q8_0': (32, 34), 'q4_0': (32, 18), 'f16': (1, 2)}


def kv_bytes_per_token(gguf, start=0, end=None, cache_type='q8_0'):
    """Minimum cache rows, using the source's actual hybrid attention map."""
    if cache_type not in KV_TYPES:
        raise ValueError('supported cache types: q8_0, q4_0, f16')
    end = gguf['trunk_layers'] if end is None else end
    interval = gguf.get('full_attention_interval')
    heads, dk, dv = (gguf.get(k) for k in ('head_count_kv', 'key_length', 'value_length'))
    if any(type(n) is not int or n <= 0 for n in (interval, heads, dk, dv)):
        return None
    block, size = KV_TYPES[cache_type]
    if (dk * heads) % block or (dv * heads) % block:
        raise ValueError('KV row dimensions are not block aligned')
    recurrent = gguf.get('recurrent_layers')
    if isinstance(recurrent, list) and len(recurrent) >= end and all(type(flag) is bool for flag in recurrent):
        attention_layers = sum(not recurrent[layer] for layer in range(start, end))
    else:
        attention_layers = sum((layer + 1) % interval == 0 for layer in range(start, end))
    return attention_layers * (((dk * heads) // block) + ((dv * heads) // block)) * size


def validate_registry(registry):
    if not isinstance(registry, dict) or registry.get('document_kind') != 'backcluster_registry' or type(registry.get('schema_version')) is not int or registry['schema_version'] != 1:
        raise ValueError('unsupported registry')
    workers = registry.get('workers')
    if not isinstance(workers, list) or not 1 <= len(workers) <= 2:
        raise ValueError('registry requires one or two workers')
    aliases, hosts, endpoints, interfaces = set(), set(), set(), set()
    for w in workers:
        if not isinstance(w, dict):
            raise ValueError('invalid worker registration')
        alias, host, port, interface = (w.get(k) for k in ('alias', 'host', 'control_port', 'interface'))
        if not valid_alias(alias) or alias == 'mac' or alias in aliases:
            raise ValueError('invalid or duplicate alias')
        aliases.add(alias)
        try:
            address = ipaddress.ip_address(host)
        except (ValueError, TypeError):
            raise ValueError('registry requires literal cable IPv4 addresses') from None
        if address.version != 4 or address not in ipaddress.ip_network('169.254.0.0/16') or host in ('169.254.0.0', '169.254.255.255') or host in hosts:
            raise ValueError('each worker requires a distinct cable link-local IPv4')
        hosts.add(host)
        for key, default in (('control_port', None), ('tail_port', 50060), ('kv_port', 50062)):
            p = w.get(key, default)
            if not integer(p, 1) or p > 65535 or (host, p) in endpoints:
                raise ValueError('invalid or duplicate worker endpoint')
            endpoints.add((host, p))
        if not isinstance(interface, str) or not re.fullmatch(r'en[0-9]{1,3}', interface) or interface in interfaces:
            raise ValueError('each worker needs a distinct Mac cable interface')
        interfaces.add(interface)
        if w.get('role') not in ROLES:
            raise ValueError('unsupported worker role')
        model = w.get('expected_model')
        if not isinstance(model, str) or re.fullmatch(r'(iPhone|iPad)[0-9]+,[0-9]+', model) is None:
            raise ValueError('expected model identifier required')
        commit = w.get('engine_commit')
        if not isinstance(commit, str) or SHA.fullmatch(commit) is None:
            raise ValueError('worker engine pin required')
        if w.get('engine_build_id') is not None and (not isinstance(w['engine_build_id'], str) or re.fullmatch(r'[0-9a-f]{64}', w['engine_build_id']) is None):
            raise ValueError('invalid worker engine build identity')
        for k in ('phone_attn_version', 'tail_protocol_version', 'tail_state_format', 'control_protocol_version'):
            if not integer(w.get(k), 1):
                raise ValueError('worker protocol versions required')
        for k in ('allocation_budget_bytes', 'reserve_bytes'):
            if not integer(w.get(k), 1):
                raise ValueError('positive measured budget and reserve required')
        if w['reserve_bytes'] >= w['allocation_budget_bytes']:
            raise ValueError('worker reserve must be smaller than its budget')
    return workers


def active_workers(registry, mode):
    """Registry order is explicit position assignment; never discover or reorder."""
    if mode not in MODES:
        raise ValueError('unsupported planning mode')
    workers = validate_registry(registry) if registry else []
    if mode == 'mac':
        return []
    selected = [w for w in workers if w['role'] != 'idle']
    if mode == 'remote-kv':
        if not selected or any(w['role'] != 'remote_kv' for w in selected):
            raise ValueError('remote-kv requires one or two registered remote_kv workers; tail and remote KV cannot be combined')
    elif mode == 'pipeline':
        if len(selected) != 2 or {w['role'] for w in selected} != {'middle_stage', 'tail_stage'}:
            raise ValueError('pipeline requires exactly one middle_stage and one tail_stage worker')
        selected.sort(key=lambda w: w['role'] != 'middle_stage')
    elif len(selected) != 1 or selected[0]['role'] != 'tail_stage':
        raise ValueError('pair split mode requires exactly one tail_stage worker; other registered workers must be idle')
    return selected


def registry_endpoints(registry):
    return [Endpoint(w['alias'], w['host'], w['control_port']) for w in validate_registry(registry)]


def registry_checks(registry, profile, engine_commit, engine_build_id=None):
    workers = validate_registry(registry)
    if validate_profile(profile):
        raise ValueError('invalid diagnostic snapshot')
    rows = {w['alias']: w for w in profile['workers']}
    checks = []
    for expected in workers:
        row = rows.get(expected['alias'], {})
        device = row.get('device', {})
        runtime = device.get('runtime', {})
        errors = []
        if row.get('status') != 'ok':
            errors.append('worker_unavailable')
        if device.get('hardware', {}).get('model') != expected['expected_model']:
            errors.append('model_identity_mismatch')
        for k in ('engine_commit', 'phone_attn_version', 'tail_protocol_version', 'tail_state_format', 'control_protocol_version'):
            if runtime.get(k) != expected[k]:
                errors.append('compatibility_mismatch_' + k)
        if expected['engine_commit'] != engine_commit:
            errors.append('host_engine_mismatch')
        if engine_build_id is not None and (expected.get('engine_build_id') != engine_build_id or runtime.get('engine_build_id') != engine_build_id):
            errors.append('engine_build_identity_mismatch')
        capability = 'tail_stage' if expected['role'] == 'middle_stage' else expected['role']
        if expected['role'] != 'idle' and capability not in device.get('capabilities', []):
            errors.append('role_not_supported')
        checks.append({'alias': expected['alias'], 'role': expected['role'], 'ok': not errors, 'errors': errors})
    return checks


def state_estimate(value):
    """Keep KV, recurrent, compute and transfer estimates independently inspectable.

    An integer is an explicit conservative aggregate supplied by the operator.
    No context scaling is inferred from that aggregate.
    """
    if integer(value):
        return {'components': {'other_bytes': value}, 'total_bytes': value, 'source': 'operator_aggregate'}
    if not isinstance(value, dict) or not value or set(value) - STATE_COMPONENTS or any(not integer(n) for n in value.values()):
        raise ValueError('state estimate requires nonnegative bytes or named KV/recurrent/compute/transfer components')
    total = sum(value.values())
    if not integer(total):
        raise ValueError('state estimate exceeds supported byte range')
    return {'components': dict(sorted(value.items())), 'total_bytes': total, 'source': 'operator_components'}


def admission(device, budget, reserve, requested, *, existing_bytes=0):
    if not all(integer(v) for v in (budget, reserve, requested, existing_bytes)) or not budget or not reserve or reserve >= budget:
        raise ValueError('invalid memory admission estimate')
    memory = device.get('memory', {})
    available, physical = memory.get('process_available_bytes'), memory.get('physical_bytes')
    footprint, metal = memory.get('process_footprint_bytes'), memory.get('metal_recommended_bytes')
    allocated = memory.get('metal_allocated_bytes')
    if not all(integer(v) for v in (available, physical, footprint, metal, allocated)):
        return {'ok': False, 'reason': 'unknown_memory_budget', 'usable_bytes': None, 'requested_bytes': requested}
    # A content-verified resident artifact may already be included in footprint
    # and Metal allocation. Credit at most the smaller observed allocation.
    # os_proc_available_memory is additional headroom, not a second RAM pool.
    credit = min(existing_bytes, footprint, allocated)
    usable = max(0, min(budget, available + credit, max(0, physical - footprint) + credit,
                        max(0, metal - allocated) + credit) - reserve)
    thermal = device.get('thermal_state')
    ok = requested <= usable and thermal in (0, 1)
    return {'ok': ok, 'reason': 'admitted_estimate' if ok else ('thermal_or_unknown_state' if thermal not in (0, 1) else 'insufficient_memory'),
            'usable_bytes': usable, 'requested_bytes': requested, 'resident_credit_bytes': credit}


def tensor_map(lock):
    gguf = lock.get('gguf', {})
    layers, trunk = gguf.get('layer_tensor_bytes'), gguf.get('trunk_layers')
    if not integer(trunk, 1) or not isinstance(layers, list) or len(layers) < trunk or any(not integer(n, 1) for n in layers) or not integer(gguf.get('shared_tensor_bytes')):
        raise ValueError('invalid model tensor map')
    if gguf.get('tensor_bytes') != sum(layers) + gguf['shared_tensor_bytes']:
        raise ValueError('inconsistent model tensor byte map')
    return gguf, layers[:trunk]


def host_admission(profile, budget, reserve, requested):
    if not all(integer(v, 1) for v in (budget, reserve)) or reserve >= budget or not integer(requested):
        raise ValueError('positive host budget and smaller reserve required')
    physical = profile['host']['memory'].get('physical_bytes')
    usable = max(0, min(budget, physical) - reserve) if integer(physical, 1) else 0
    return {'ok': requested <= usable and integer(physical, 1) and budget <= physical,
            'usable_bytes': usable, 'requested_bytes': requested,
            'reason': 'host_physical_memory_unknown' if not integer(physical, 1) else ('host_budget_exceeds_physical_memory' if budget > physical else ('admitted_estimate' if requested <= usable else 'insufficient_memory'))}


def wire_latency(gguf, link, *, batch_tokens=1, residual_bytes=4):
    """Sequential request+reply lower bound from measured link inputs, never TOPS."""
    if link is None:
        return {'estimated_ms': None, 'reason': 'link_measurements_missing'}
    if not isinstance(link, dict) or set(link) - {'round_trip_ns', 'upload_bytes_per_second', 'download_bytes_per_second'}:
        raise ValueError('invalid measured link costs')
    rtt, up, down = (link.get(k) for k in ('round_trip_ns', 'upload_bytes_per_second', 'download_bytes_per_second'))
    if not integer(rtt) or not all(type(n) in (int, float) and math.isfinite(n) and n > 0 for n in (up, down)):
        raise ValueError('positive measured link rates and nonnegative RTT required')
    hidden, vocab = gguf.get('embedding_length'), gguf.get('vocab_size')
    if not integer(hidden, 1) or not integer(vocab, 1):
        return {'estimated_ms': None, 'reason': 'wire_dimensions_missing'}
    sent = hidden * batch_tokens * residual_bytes
    received = vocab * 4  # the last token's f32 logits; headers/copies are extra
    return {'estimated_ms': rtt / 1e6 + 1000 * (sent / up + received / down),
            'residual_bytes': sent, 'logits_bytes': received, 'includes': 'RTT_and_payload_only',
            'excludes': ['compute', 'state_sync', 'serialization', 'copies', 'scheduling']}


def pipeline_wire_latency(gguf, links, workers):
    aliases = {w['alias'] for w in workers}
    if links is not None and (not isinstance(links, dict) or set(links) != aliases):
        raise ValueError('pipeline link costs must describe each explicit worker hop')
    stages = [dict(wire_latency(gguf, links.get(w['alias']) if links else None), alias=w['alias']) for w in workers]
    estimates = [stage['estimated_ms'] for stage in stages]
    return {'estimated_ms': sum(estimates) if all(n is not None for n in estimates) else None,
            'stages': stages, 'includes': 'sequential_RTT_and_payload_for_each_hop_only'}


def make_plan(lock, profile, registry, engine_commit, mode, host_budget, host_reserve, context, kv_estimates,
              *, host_state_bytes=None, split_layer=None, middle_end=None, link_profile=None, engine_build_id=None,
              cache_type='q8_0'):
    validate_lock(lock, engine_commit)
    gguf, layers = tensor_map(lock)
    if validate_profile(profile) or profile['host']['platform'] != 'macOS':
        raise ValueError('planning requires a valid macOS diagnostic snapshot')
    if mode not in MODES or type(context) is not int or not 128 <= context <= 131072:
        raise ValueError('unsupported mode or context')
    limit = gguf.get('context_length')
    if type(limit) is int and context > limit:
        raise ValueError('context exceeds model metadata')
    selected = active_workers(registry, mode)
    checks = registry_checks(registry, profile, engine_commit, engine_build_id) if registry else []
    selected_aliases = {w['alias'] for w in selected}
    states = dict(kv_estimates)
    known = {w['alias'] for w in validate_registry(registry)} if registry else set()
    if set(states) - known:
        raise ValueError('state estimates must name registered workers')
    states = {alias: state_estimate(n) for alias, n in states.items()}
    host_state = state_estimate(host_state_bytes) if host_state_bytes is not None else None
    host_memory = host_admission(profile, host_budget, host_reserve, 0)
    rows = {w['alias']: w.get('device', {}) for w in profile['workers']}
    trunk = gguf['trunk_layers']
    if split_layer is not None and (type(split_layer) is not int or not 0 < split_layer < trunk):
        raise ValueError('split boundary must be inside trunk layers')
    interval = gguf.get('full_attention_interval') or (4 if gguf.get('architecture') == 'qwen35' else 1)
    if not integer(interval, 1) or (split_layer is not None and split_layer % interval):
        raise ValueError('split boundary must preserve the attention interval')
    if mode == 'split-prefill' and split_layer is None:
        raise ValueError('split-prefill needs an explicit layer boundary')
    if mode == 'pipeline' and (type(middle_end) is not int or split_layer is None or not split_layer < middle_end < trunk or middle_end % interval):
        raise ValueError('pipeline requires ordered explicit boundaries preserving the attention interval')
    # Exact per-layer bytes; shared tensors are conservatively charged to both
    # nodes unless inspection identifies the output tensors moved off the Mac.
    shared = gguf['shared_tensor_bytes']
    output_bytes = gguf.get('output_weight_tensor_bytes', 0) + gguf.get('output_norm_tensor_bytes', 0)
    if not integer(output_bytes) or output_bytes > shared:
        raise ValueError('invalid output tensor byte map')
    candidates = []
    boundaries = list(range(interval, trunk, interval)) if mode in ('capacity', 'split-decode') and split_layer is None else [split_layer]
    if not boundaries:
        raise ValueError('model has no supported split boundary')
    for boundary in boundaries:
        split = mode in ('capacity', 'split-decode', 'pipeline')
        host_weights = sum(layers[:boundary]) + shared - output_bytes if split else gguf['tensor_bytes']
        host_needed = host_weights + (host_state['total_bytes'] if host_state else 0)
        host_a = host_admission(profile, host_budget, host_reserve, host_needed)
        host_kv = kv_bytes_per_token(gguf, 0, boundary if split else trunk, cache_type)
        host_minimum = context * host_kv if host_kv is not None else None
        if host_state is not None and host_minimum is not None and host_state['total_bytes'] < host_minimum:
            host_a.update(ok=False, reason='state_estimate_below_kv_minimum')
        assignments = [{'alias': 'mac', 'layer_start': 0, 'layer_end': boundary if split else trunk,
                        'weight_bytes': host_weights, 'state': host_state, 'minimum_kv_bytes': host_minimum, **host_a}]
        admissions = []
        for w in selected:
            if mode == 'remote-kv':
                weight_bytes = 0
                start, end = 0, trunk
            else:
                start, end = boundary if boundary is not None else 0, trunk
                if mode == 'pipeline':
                    start, end = (boundary, middle_end) if w['role'] == 'middle_stage' else (middle_end, trunk)
                weight_bytes = sum(layers[start:end]) + shared
            state = states.get(w['alias'])
            worker_kv = kv_bytes_per_token(gguf, start, end, cache_type) if mode != 'remote-kv' else None
            worker_minimum = context * worker_kv if worker_kv is not None else None
            if state is None:
                a = {'ok': False, 'reason': 'missing_context_state_estimate', 'requested_bytes': None, 'usable_bytes': None}
            else:
                a = admission(rows.get(w['alias'], {}), w['allocation_budget_bytes'], w['reserve_bytes'], weight_bytes + state['total_bytes'])
                if worker_minimum is not None and state['total_bytes'] < worker_minimum:
                    a.update(ok=False, reason='state_estimate_below_kv_minimum')
            admissions.append(dict(a, alias=w['alias']))
            assignments.append({'alias': w['alias'], 'layer_start': start, 'layer_end': end,
                                'weight_bytes': weight_bytes, 'state': state, 'minimum_kv_bytes': worker_minimum, **a})
        valid_checks = all(c['ok'] for c in checks if c['alias'] in selected_aliases)
        candidates.append({'split_layer': boundary, 'assignments': assignments, 'worker_admission': admissions,
                           'feasible_estimate': bool(host_state is not None and host_a['ok'] and valid_checks and all(a['ok'] for a in admissions))})
    # Prefer maximum host headroom, then the smaller tail (deterministic). This
    # is a memory choice, never an unmeasured throughput recommendation.
    feasible = [c for c in candidates if c['feasible_estimate']]
    preferred = max(feasible, key=lambda c: (c['assignments'][0]['usable_bytes'] - c['assignments'][0]['requested_bytes'], c['split_layer'] or 0)) if feasible else candidates[0]
    blockers = []
    if host_state is None:
        blockers.append('missing_host_runtime_state_estimate')
    if not host_memory['ok']:
        blockers.append(host_memory['reason'])
    if not preferred['assignments'][0]['ok']:
        blockers.append('host_weights_and_state_exceed_budget')
    if any(not c['ok'] for c in checks if c['alias'] in selected_aliases) or any(not a['ok'] for a in preferred['worker_admission']):
        blockers.append('worker_admission_failed')
    captured = datetime.fromisoformat(profile['captured_at'].replace('Z', '+00:00'))
    age = (datetime.now(timezone.utc) - captured).total_seconds()
    if not 0 <= age <= 300:
        blockers.append('snapshot_stale')
    return {'document_kind': 'backcluster_plan', 'schema_version': 1, 'mode': mode, 'context': context,
            'model_sha256': lock['sha256'], 'hardware_validated': False, 'launch_ready': False,
            'feasible_estimate': preferred['feasible_estimate'] and not blockers,
            'host_weight_budget_ok': preferred['assignments'][0]['ok'], 'assignments': preferred['assignments'],
            'split_layer': preferred['split_layer'], 'compatibility': checks,
            'worker_admission': preferred['worker_admission'], 'blockers': blockers,
            'split_candidates': candidates if mode in ('capacity', 'split-decode') else [],
            'link_decode_cost': (wire_latency(gguf, link_profile) if mode in ('capacity', 'split-decode') else
                                 pipeline_wire_latency(gguf, link_profile, selected) if mode == 'pipeline' else None),
            'notes': ['operator_state_estimates_are_not_measurements', 'shared_tensors_charged_per_device',
                      'live_route_identity_and_memory_rechecked_at_launch', 'hardware_quality_and_thermal_tests_pending']}
