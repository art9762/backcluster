"""Explicit local registry and conservative per-device memory admission.

Plans are reviewable estimates. They cannot enable unvalidated inference modes.
"""
import ipaddress
import re
from datetime import datetime, timezone

from .profiles import integer, valid_alias, validate_profile
from .rpc import Endpoint

SHA = re.compile(r'[0-9a-f]{40}')
ROLES = {'remote_kv', 'tail_stage', 'idle'}


def validate_registry(registry):
    if not isinstance(registry, dict) or registry.get('document_kind') != 'backcluster_registry' or type(registry.get('schema_version')) is not int or registry['schema_version'] != 1:
        raise ValueError('unsupported registry')
    workers = registry.get('workers')
    if not isinstance(workers, list) or not 1 <= len(workers) <= 2:
        raise ValueError('registry requires one or two workers')
    aliases, endpoints, interfaces = set(), set(), set()
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
        if address.version != 4 or address not in ipaddress.ip_network('169.254.0.0/16') or host in ('169.254.0.0', '169.254.255.255'):
            raise ValueError('registry requires cable link-local IPv4')
        if not integer(port, 1) or port > 65535 or (host, port) in endpoints:
            raise ValueError('invalid or duplicate control endpoint')
        endpoints.add((host, port))
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
        for k in ('phone_attn_version', 'tail_protocol_version', 'tail_state_format', 'control_protocol_version'):
            if not integer(w.get(k), 1):
                raise ValueError('worker protocol versions required')
        for k in ('allocation_budget_bytes', 'reserve_bytes'):
            if not integer(w.get(k), 1):
                raise ValueError('positive measured budget and reserve required')
    return workers


def registry_endpoints(registry):
    return [Endpoint(w['alias'], w['host'], w['control_port']) for w in validate_registry(registry)]


def registry_checks(registry, profile, engine_commit):
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
        if expected['role'] != 'idle' and expected['role'] not in device.get('capabilities', []):
            errors.append('role_not_supported')
        # A snapshot cannot verify the live route; serve must not consume this as authorization.
        checks.append({'alias': expected['alias'], 'role': expected['role'], 'ok': not errors, 'errors': errors})
    return checks


def admission(device, budget, reserve, requested):
    memory = device.get('memory', {})
    available, physical = memory.get('process_available_bytes'), memory.get('physical_bytes')
    footprint, metal = memory.get('process_footprint_bytes'), memory.get('metal_recommended_bytes')
    allocated = memory.get('metal_allocated_bytes')
    if not all(integer(v) for v in (available, physical, footprint, metal, allocated)):
        return {'ok': False, 'reason': 'unknown_memory_budget', 'usable_bytes': None, 'requested_bytes': requested}
    # os_proc_available_memory is *additional* memory; Metal's recommended budget
    # is a total. Never sum the limits. Physical headroom is only an upper bound.
    usable = max(0, min(budget, available, max(0, physical - footprint), max(0, metal - allocated)) - reserve)
    thermal = device.get('thermal_state')
    ok = requested <= usable and thermal in (0, 1)
    return {'ok': ok, 'reason': 'admitted_estimate' if ok else ('thermal_or_unknown_state' if thermal not in (0, 1) else 'insufficient_memory'),
            'usable_bytes': usable, 'requested_bytes': requested}


def make_plan(lock, profile, registry, engine_commit, mode, host_budget, host_reserve, context, kv_estimates):
    if lock.get('document_kind') != 'backcluster_model_lock' or lock.get('engine_commit') != engine_commit:
        raise ValueError('model lock and engine pin must agree')
    gguf = lock.get('gguf', {})
    layers = gguf.get('layer_tensor_bytes')
    if not isinstance(layers, list) or not layers or any(not integer(n, 1) for n in layers) or not integer(gguf.get('shared_tensor_bytes')):
        raise ValueError('invalid model tensor map')
    if validate_profile(profile):
        raise ValueError('invalid diagnostic snapshot')
    if profile['host']['platform'] != 'macOS':
        raise ValueError('planning requires a macOS snapshot')
    if type(context) is not int or context < 1 or not all(integer(n, 1) for n in (host_budget, host_reserve)):
        raise ValueError('positive context, host budget and reserve required')
    checks = registry_checks(registry, profile, engine_commit) if registry else []
    workers = validate_registry(registry) if registry else []
    # Future sharding is not admitted by summing mobile RAM. Host owns all weights
    # in the only implemented launcher; tail assignments need measured split search.
    assignments = [{'alias': 'mac', 'layer_start': 0, 'layer_end': gguf['trunk_layers'],
                    'weight_bytes': gguf['tensor_bytes'], 'usable_bytes': max(0, host_budget - host_reserve)}]
    admissions = []
    rows = {w['alias']: w.get('device', {}) for w in profile['workers']}
    kv_estimates = dict(kv_estimates)
    if set(kv_estimates) - {w['alias'] for w in workers}:
        raise ValueError('KV estimates must name registered workers')
    for w in workers:
        needed = kv_estimates.get(w['alias'])
        if needed is None:
            a = {'ok': w['role'] == 'idle', 'reason': 'missing_context_state_estimate', 'requested_bytes': None, 'usable_bytes': None}
        else:
            a = admission(rows.get(w['alias'], {}), w['allocation_budget_bytes'], w['reserve_bytes'], needed)
        admissions.append(dict(a, alias=w['alias']))
    host_ok = gguf['tensor_bytes'] < max(0, host_budget - host_reserve)
    reasons = ['runtime_state_and_kv_require_measurement', 'target_device_forward_not_validated']
    if mode != 'mac':
        reasons.append('distributed_launch_requires_pair_baseline')
    if not host_ok:
        reasons.append('host_weights_exceed_budget')
    if any(not c['ok'] for c in checks) or any(not a['ok'] for a in admissions):
        reasons.append('worker_admission_failed')
    captured = datetime.fromisoformat(profile['captured_at'].replace('Z', '+00:00'))
    age = (datetime.now(timezone.utc) - captured).total_seconds()
    if not 0 <= age <= 300:
        reasons.append('snapshot_stale')
    return {'document_kind': 'backcluster_plan', 'schema_version': 1, 'mode': mode, 'context': context,
            'model_sha256': lock['sha256'], 'hardware_validated': False, 'launch_ready': False,
            'host_weight_budget_ok': host_ok, 'assignments': assignments,
            'compatibility': checks, 'worker_admission': admissions, 'blockers': reasons}
