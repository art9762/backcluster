"""Offline placement receipts and fail-closed live launch admission."""
import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.artifacts import make_lock
from backcluster.devices import collect_profile
from backcluster.planning import (admission, make_plan, registry_checks, state_estimate,
                                 validate_registry, wire_latency)
from backcluster.profiles import normalize_worker, validate_profile
from backcluster.serving import (check_live_launch, clean_environment, cluster_launch_spec, build_engine, build_environment,
                                kv_bytes_per_token, remote_position_counts)
from backcluster.sharding import Field, split_gguf
from generate_tiny_model import write_fixture
from test_diagnostics import native_reply
from test_preparation import registry, PIN

BUILD = 'b' * 64
GIB = 2**30


def fixture(path, *, tied=False):
    fields = {
        'general.architecture': Field(8, 'qwen35'), 'general.name': Field(8, 'offline-launch-fixture'),
        'qwen35.block_count': Field(4, 8), 'qwen35.context_length': Field(4, 65536),
        'qwen35.full_attention_interval': Field(4, 2), 'qwen35.embedding_length': Field(4, 1536),
        'qwen35.attention.head_count': Field(4, 6), 'qwen35.attention.head_count_kv': Field(4, 1),
        'qwen35.attention.key_length': Field(4, 256), 'qwen35.attention.value_length': Field(4, 256),
        'qwen35.vocab_size': Field(4, 8), 'tokenizer.chat_template': Field(8, '{{ messages }}'),
    }
    tensors = [('token_embd.weight', (1536, 8), 0, b'E' * (1536 * 8 * 4)),
               ('output_norm.weight', (1536,), 0, b'N' * (1536 * 4))]
    if not tied:
        tensors.append(('output.weight', (1536, 8), 0, b'O' * (1536 * 8 * 4)))
    for i in range(8):
        tensors.append((f'blk.{i}.attn.weight', (32,), 8, bytes([i + 1]) * 34))
    write_fixture(path, fields, tensors)


def cluster_registry(mode='pair'):
    value = registry()
    first = value['workers'][0]
    first.update(engine_build_id=BUILD, role='tail_stage' if mode == 'pair' else 'remote_kv')
    if mode in ('dual', 'pipeline'):
        second = copy.deepcopy(first)
        second.update(alias='iphone', host='169.254.11.2', interface='en8', expected_model='iPhone18,2')  # sensitive-ok: synthetic cable endpoint
        value['workers'].append(second)
    if mode == 'pipeline':
        first['role'] = 'middle_stage'
        second['role'] = 'tail_stage'
    return value


def snapshot(registration):
    result = collect_profile([])
    result['host']['platform'] = 'macOS'
    result['host']['memory']['physical_bytes'] = 24 * GIB
    for worker in registration['workers']:
        reply = native_reply()
        reply['hardware']['model'] = worker['expected_model']
        reply['platform'] = 'iPadOS' if worker['expected_model'].startswith('iPad') else 'iOS'
        reply['runtime'].update(engine_commit=PIN, control_protocol_version=1, engine_build_id=BUILD)
        result['workers'].append({'alias': worker['alias'], 'status': 'ok', 'device': normalize_worker(reply),
                                  'link': {'transport': 'unverified', 'control_round_trip_ns': 1}})
    return result


def attach_tail(profile, alias, manifest):
    device = next(row['device'] for row in profile['workers'] if row['alias'] == alias)
    device['loaded_tail'] = {
        'artifact_sha256': manifest['sha256'], 'source_sha256': manifest['source']['sha256'],
        'engine_commit': PIN, 'engine_build_id': BUILD, 'layer_start': manifest['layer_start'],
        'layer_end': manifest['layer_end'], 'output_head': manifest['output_head'],
        'bytes': manifest['bytes'], 'context': 65536, 'state': 'ready',
    }


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'model.gguf'; fixture(self.source)
        self.lock = make_lock(self.source, 'fixture/model', 'a' * 40, PIN)
        self.identity = patch('backcluster.serving._build_identity', return_value=BUILD)
        self.identity.start(); self.addCleanup(self.identity.stop)
        self.registration = cluster_registry()
        self.profile = snapshot(self.registration)
        self.tail = self.directory / 'tail.gguf'
        self.manifest = split_gguf(self.source, self.tail, 4, 8, 'tail', source_lock=self.lock)

    def spec(self, **overrides):
        options = dict(mode='capacity', registry=self.registration, profile=self.profile,
                       host_budget=16 * GIB, host_reserve=4 * GIB, host_state_bytes=GIB,
                       state_estimates=[('ipad', {'kv_bytes': GIB, 'compute_bytes': GIB // 8})],
                       split_layer=4, shard_manifest=self.manifest, shard_model=self.tail)
        options.update(overrides)
        return cluster_launch_spec(self.lock, self.source, 4096, **options)

    def live(self, spec, profile=None, route=None):
        profile = copy.deepcopy(self.profile if profile is None else profile)
        if spec['mode'] != 'remote-kv':
            attach_tail(profile, 'ipad', self.manifest)
        return check_live_launch(spec, route_reader=lambda host: 'interface: en7' if route is None else route,
                                 profile_reader=lambda endpoints, timeout: profile)

    def test_capacity_receipt_and_exact_cpu_override(self):
        spec = self.spec()
        self.assertFalse(spec['hardware_validated'])
        self.assertEqual(spec['environment']['LLAMA_SPLIT_DECODE'], '1')
        self.assertEqual(spec['environment']['LLAMA_SPLIT_TIMEOUT_S'], '120')
        self.assertEqual(spec['timeout'], 3.0)
        self.assertEqual(spec['environment']['LLAMA_SPLIT_MODEL_SHA256'], self.lock['sha256'])
        override = spec['argv'][spec['argv'].index('--override-tensor') + 1].split('=')[0]
        for layer in range(8):
            self.assertEqual(bool(re.match(override, f'blk.{layer}.weight')), layer >= 4)
        self.assertIn('--no-repack', spec['argv'])
        self.assertEqual(spec['snapshot_plan']['assignments'][1]['weight_bytes'], self.manifest['gguf']['tensor_bytes'])
        self.assertEqual(spec['snapshot_plan']['assignments'][0]['weight_bytes'],
                         sum(self.lock['gguf']['layer_tensor_bytes'][:4]) + self.lock['gguf']['embedding_tensor_bytes'])
        self.assertTrue(self.live(spec)['ok'])

    def test_tail_identity_not_replaced_by_snapshot(self):
        spec = self.spec()
        with self.assertRaisesRegex(ValueError, 'content-verified'):
            check_live_launch(spec, route_reader=lambda host: 'interface: en7',
                              profile_reader=lambda endpoints, timeout: self.profile)
        for field, value in [('artifact_sha256', 'a' * 64), ('source_sha256', 'a' * 64),
                             ('layer_start', 2), ('output_head', False), ('engine_build_id', 'a' * 64), ('bytes', 1),
                             ('context', 128)]:
            modified = copy.deepcopy(self.profile); attach_tail(modified, 'ipad', self.manifest)
            modified['workers'][0]['device']['loaded_tail'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_live_launch(spec, route_reader=lambda host: 'interface: en7',
                                  profile_reader=lambda endpoints, timeout: modified)

    def test_mac_options_are_applied_or_rejected_explicitly(self):
        spec = cluster_launch_spec(self.lock, self.source, 4096, cache_type='q4_0')
        self.assertEqual(spec['argv'][spec['argv'].index('--cache-type-k') + 1], 'q4_0')
        for option in ({'split_layer': 4}, {'total_context': 8192}, {'middle_end': 6},
                       {'state_estimates': [('ipad', GIB)]}, {'host_budget': GIB}, {'cache_type': 'unknown'}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                cluster_launch_spec(self.lock, self.source, 4096, **option)

    def test_route_fails_before_network_probe(self):
        probe = Mock()
        with self.assertRaisesRegex(ValueError, 'route'):
            check_live_launch(self.spec(), route_reader=lambda host: 'interface: en8', profile_reader=probe)
        probe.assert_not_called()

    def test_live_thermal_memory_build_and_stale_profile_fail(self):
        spec = self.spec()
        for field, value in [('thermal_state', 2), ('memory', dict(self.profile['workers'][0]['device']['memory'], process_available_bytes=0)),
                             ('runtime', dict(self.profile['workers'][0]['device']['runtime'], engine_build_id='c' * 64))]:
            modified = copy.deepcopy(self.profile); modified['workers'][0]['device'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.live(spec, modified)
        modified = copy.deepcopy(self.profile); modified['captured_at'] = '2020-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'stale'): self.live(spec, modified)

    def test_headless_tail_only_prefill(self):
        path = self.directory / 'headless.gguf'
        manifest = split_gguf(self.source, path, 4, 8, 'tail', no_head=True, source_lock=self.lock)
        with self.assertRaisesRegex(ValueError, 'output head'):
            self.spec(shard_model=path, shard_manifest=manifest)
        spec = self.spec(mode='split-prefill', shard_model=path, shard_manifest=manifest)
        self.assertNotIn('LLAMA_SPLIT_DECODE', spec['environment'])
        batch = int(spec['argv'][spec['argv'].index('--batch-size') + 1])
        ubatch = int(spec['argv'][spec['argv'].index('--ubatch-size') + 1])
        self.assertGreaterEqual(batch - ubatch, int(spec['environment']['LLAMA_SPLIT_MIN']))

    def test_shard_corruption_wrong_range_and_source_fail(self):
        with self.assertRaises(ValueError): self.spec(split_layer=2)
        with self.tail.open('r+b') as file: file.seek(-1, 2); file.write(b'X')
        with self.assertRaises(ValueError): self.spec()

    def test_tied_embedding_remains_charged_to_host(self):
        source = self.directory / 'tied.gguf'; fixture(source, tied=True)
        lock = make_lock(source, 'fixture/model', 'a' * 40, PIN)
        plan = make_plan(lock, self.profile, self.registration, PIN, 'capacity', 16*GIB, 4*GIB,
                         4096, [('ipad', GIB)], host_state_bytes=GIB, split_layer=4)
        self.assertEqual(plan['assignments'][0]['weight_bytes'],
                         sum(lock['gguf']['layer_tensor_bytes'][:4]) + lock['gguf']['embedding_tensor_bytes'])

    def test_dual_remote_kv_preserves_explicit_order_and_peak_position_mapping(self):
        registration = cluster_registry('dual'); profile = snapshot(registration)
        spec = self.spec(mode='remote-kv', registry=registration, profile=profile, split_layer=None,
                         shard_manifest=None, shard_model=None, total_context=12288,
                         state_estimates=[('ipad', GIB), ('iphone', GIB)])
        self.assertEqual(spec['remote_workers'], ['ipad', 'iphone'])
        self.assertEqual([row['positions'] for row in spec['remote_positions']], [4096, 4096])
        self.assertNotIn('LLAMA_SPLIT_TAIL', spec['environment'])
        self.assertEqual(spec['environment']['LLAMA_KV_REMOTE'].split(',')[0], '169.254.10.2:50062')  # sensitive-ok: synthetic cable endpoint
        before = copy.deepcopy(profile)
        before['workers'][1]['device']['thermal_state'] = 3
        with self.assertRaises(ValueError):
            check_live_launch(spec, route_reader=lambda host: 'interface: ' + ('en7' if host.endswith('10.2') else 'en8'),
                              profile_reader=lambda endpoints, timeout: before)

    def test_remote_peak_state_underestimate_and_mixed_roles_fail(self):
        registration = cluster_registry('dual'); profile = snapshot(registration)
        with self.assertRaisesRegex(ValueError, 'block-cyclic'):
            self.spec(mode='remote-kv', registry=registration, profile=profile, split_layer=None,
                      shard_manifest=None, shard_model=None, total_context=16384,
                      state_estimates=[('ipad', 1), ('iphone', 1)])
        registration['workers'][1]['role'] = 'tail_stage'
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            self.spec(mode='remote-kv', registry=registration, profile=profile)

    def test_compute_state_estimates_cannot_understate_known_kv(self):
        with self.assertRaisesRegex(ValueError, 'worker state estimate'):
            self.spec(state_estimates=[('ipad', 0)])
        with self.assertRaisesRegex(ValueError, 'host state estimate'):
            self.spec(host_state_bytes=0)

    def test_pipeline_verifies_both_stages_and_optional_partition(self):
        registration = cluster_registry('pipeline'); profile = snapshot(registration)
        middle_path = self.directory / 'middle.gguf'
        middle = split_gguf(self.source, middle_path, 2, 6, 'middle', source_lock=self.lock)
        tail_path = self.directory / 'pipeline-tail.gguf'
        tail = split_gguf(self.source, tail_path, 6, 8, 'tail', source_lock=self.lock)
        host_path = self.directory / 'head.gguf'
        head = split_gguf(self.source, host_path, 0, 2, 'host', source_lock=self.lock)
        spec = self.spec(mode='pipeline', registry=registration, profile=profile, split_layer=2, middle_end=6,
                         shard_manifest=tail, shard_model=tail_path, middle_manifest=middle, middle_model=middle_path,
                         host_manifest=head, host_model=host_path, state_estimates=[('ipad', GIB), ('iphone', GIB)])
        self.assertEqual(spec['environment']['LLAMA_SPLIT_MIDDLE_END'], '6')
        self.assertEqual(spec['partition']['source_sha256'], self.lock['sha256'])
        attach_tail(profile, 'ipad', middle); attach_tail(profile, 'iphone', tail)
        self.assertFalse(validate_profile(profile))
        check_live_launch(spec, route_reader=lambda host: 'interface: ' + ('en7' if host.endswith('10.2') else 'en8'),
                          profile_reader=lambda endpoints, timeout: profile)
        profile['workers'][0]['device']['loaded_tail']['output_head'] = True
        with self.assertRaises(ValueError):
            check_live_launch(spec, route_reader=lambda host: 'interface: ' + ('en7' if host.endswith('10.2') else 'en8'),
                              profile_reader=lambda endpoints, timeout: profile)

    def test_full_partition_is_optional_but_never_forged(self):
        spec = self.spec()
        self.assertNotIn('partition', spec)
        self.assertEqual(spec['artifacts'][0]['kind'], 'model')
        self.assertEqual(spec['argv'][spec['argv'].index('--model') + 1], str(self.source))

    def test_inherited_endpoint_flags_never_enter_launch(self):
        with patch.dict('os.environ', {'LLAMA_KV_REMOTE': 'bad', 'LLAMA_SPLIT_MIDDLE': 'bad', 'DYLD_INSERT_LIBRARIES': 'bad'}):
            env = clean_environment(self.spec()['environment'])
        self.assertNotIn('LLAMA_KV_REMOTE', env)
        self.assertNotIn('LLAMA_SPLIT_MIDDLE', env)
        self.assertNotIn('DYLD_INSERT_LIBRARIES', env)
        self.assertEqual(env['LLAMA_SPLIT_L'], '4')

    def test_build_uses_overlay_source_and_identifies_dynamic_runtime(self):
        source = self.directory / 'overlay'; source.mkdir()
        target = self.directory / 'llama.cpp/build-metal/bin'; target.mkdir(parents=True)
        (target / 'llama-server').write_bytes(b'executable')
        (target / 'libllama.dylib').write_bytes(b'engine')
        status = {'expected_commit': PIN, 'matches_pin': True}
        with patch('backcluster.serving.ROOT', self.directory), patch('backcluster.serving.engine_status', return_value=status), \
                patch('backcluster.serving.platform.system', return_value='Darwin'), \
                patch('backcluster.serving.platform.machine', return_value='arm64'), \
                patch('backcluster.serving.shutil.which', return_value='/fixture/cmake'), \
                patch('backcluster.engine_overlay.prepare_engine', return_value={'source_dir': source, 'base_commit': PIN, 'build_id': BUILD}), \
                patch('backcluster.serving.subprocess.run') as execute:
            build_engine()
        configure = execute.call_args_list[0].args[0]
        self.assertEqual(configure[configure.index('-S') + 1], str(source))
        self.assertIn('-DLLAMA_BUILD_COMMIT=' + PIN[:12] + '-bc-' + BUILD[:12], configure)
        self.assertIn('-DLLAMA_BUILD_NUMBER=0', configure)
        receipt = json.loads((target.parent / 'backcluster-build.json').read_text())
        self.assertEqual(receipt['engine_build_id'], BUILD)
        self.assertEqual(set(receipt['runtime_files']), {'llama-server', 'libllama.dylib'})


class PlanningTests(unittest.TestCase):
    def test_state_components_separate_and_nonfinite_boolean_negative_rejected(self):
        estimate = state_estimate({'kv_bytes': 12, 'recurrent_bytes': 4, 'transfer_bytes': 8})
        self.assertEqual(estimate['total_bytes'], 24)
        for value in (True, -1, {'kv_bytes': -1}, {'kv_bytes': float('nan')}, {'unknown': 1}):
            with self.subTest(value=value), self.assertRaises(ValueError): state_estimate(value)

    def test_equal_position_share_includes_unequal_final_block(self):
        self.assertEqual(remote_position_counts(4096, 2), [4096, 0])
        self.assertEqual(remote_position_counts(12288, 2), [8192, 4096])
        self.assertEqual(remote_position_counts(8193, 2), [4097, 4096])

    def test_verified_resident_weights_not_double_counted_or_added_to_ram_pool(self):
        d = normalize_worker(native_reply())
        d['memory'].update(process_available_bytes=2*GIB, process_footprint_bytes=5*GIB,
                           metal_allocated_bytes=5*GIB, metal_recommended_bytes=8*GIB)
        result = admission(d, 8*GIB, GIB, 6*GIB, existing_bytes=5*GIB)
        self.assertTrue(result['ok']); self.assertEqual(result['usable_bytes'], 6*GIB)
        self.assertFalse(admission(d, 8*GIB, GIB, 6*GIB+1, existing_bytes=5*GIB)['ok'])

    def test_registry_rejects_duplicate_host_even_with_distinct_ports(self):
        registration = cluster_registry('dual')
        registration['workers'][1].update(host=registration['workers'][0]['host'], control_port=50063)
        with self.assertRaises(ValueError): validate_registry(registration)

    def test_build_identity_required_for_overlay_workers(self):
        registration = cluster_registry(); profile = snapshot(registration)
        self.assertTrue(registry_checks(registration, profile, PIN, BUILD)[0]['ok'])
        del registration['workers'][0]['engine_build_id']
        self.assertFalse(registry_checks(registration, profile, PIN, BUILD)[0]['ok'])

    def test_loaded_tail_allowlist_and_unverified_null(self):
        reply = native_reply(); reply['loaded_tail'] = None
        self.assertNotIn('loaded_tail', normalize_worker(reply))
        reply['loaded_tail'] = {'path': '/private/worker/model.gguf'}
        with self.assertRaises(ValueError): normalize_worker(reply)

    def test_link_cost_uses_sequential_rtt_request_and_reply(self):
        result = wire_latency({'embedding_length': 1536, 'vocab_size': 8},
                              {'round_trip_ns': 2_000_000, 'upload_bytes_per_second': 6144000,
                               'download_bytes_per_second': 32000})
        self.assertAlmostEqual(result['estimated_ms'], 4)
        self.assertIn('compute', result['excludes'])
        self.assertIsNone(wire_latency({}, None)['estimated_ms'])

    def test_build_network_trust_kept_for_cmake_but_not_inference(self):
        with patch.dict('os.environ', {'HTTPS_PROXY': 'fixture-proxy', 'SSL_CERT_FILE': '/fixture/ca.pem',
                                       'DYLD_INSERT_LIBRARIES': '/fixture/inject'}):
            build = build_environment(); inference = clean_environment()
        self.assertEqual(build['HTTPS_PROXY'], 'fixture-proxy')
        self.assertEqual(build['SSL_CERT_FILE'], '/fixture/ca.pem')
        self.assertNotIn('DYLD_INSERT_LIBRARIES', build)
        self.assertNotIn('HTTPS_PROXY', inference)
        self.assertNotIn('SSL_CERT_FILE', inference)

    def test_kv_rows_include_both_storage_shapes_and_explicit_recurrent_map(self):
        model = {'trunk_layers': 4, 'full_attention_interval': 2, 'head_count_kv': 2,
                 'key_length': 16, 'value_length': 32}
        self.assertEqual(kv_bytes_per_token(model), 2 * (34 + 68))
        model['recurrent_layers'] = [False, False, False, True]
        self.assertEqual(kv_bytes_per_token(model), 3 * (34 + 68))


if __name__ == '__main__':
    unittest.main()
