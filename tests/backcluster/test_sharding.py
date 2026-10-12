"""Verbatim quantized ranges, source identity and complete stage partitions."""
import copy
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.artifacts import digest, inspect_gguf, make_lock
from backcluster.sharding import (Field, _document, _fields_hash, split_gguf,
                                 validate_partition, validate_shard_manifest, verify_shard)
from generate_tiny_model import generate, write_fixture

PIN = '9f17fc31cf60753eb6bd6ec075894edab773cb91'
ROOT = Path(__file__).resolve().parents[2]


def fixture(path, *, mtp=2, tied=False, bad_array=False, recurrent=True):
    count, trunk = 8 + mtp, 8
    fields = {
        'general.architecture': Field(8, 'qwen35'), 'general.name': Field(8, 'fixture'),
        'general.alignment': Field(4, 64), 'qwen35.block_count': Field(4, count),
        'qwen35.nextn_predict_layers': Field(4, mtp), 'qwen35.full_attention_interval': Field(4, 2),
        'qwen35.context_length': Field(4, 512), 'qwen35.embedding_length': Field(4, 32),
        'qwen35.attention.head_count': Field(9, list(range(count)), 4),
        'qwen35.experimental.by_layer': Field(9, list(range(count)), 4),
        'qwen35.rope.dimension_sections': Field(9, [2, 2, 0, 0], 5),
        'tokenizer.ggml.model': Field(8, 'gpt2'),
        # Its length deliberately equals trunk: it must never be layer-sliced.
        'tokenizer.ggml.tokens': Field(9, [f'token{i}' for i in range(trunk)], 8),
        'tokenizer.ggml.scores': Field(9, [float(i) for i in range(trunk)], 6),
        'tokenizer.chat_template': Field(8, '{{ messages }}'),
    }
    if recurrent:
        fields['qwen35.attention.recurrent_layers'] = Field(9, [i < trunk and i % 2 == 0 for i in range(count - int(bad_array))], 7)
    tensors = [('token_embd.weight', (32, 8), 0, bytes(range(256)) * 4),
               ('output_norm.weight', (32,), 0, b'N' * 128)]
    if not tied:
        tensors.append(('output.weight', (32, 8), 0, b'O' * 1024))
    for layer in range(count):
        # Q8_0 and F32 deliberately have different storage sizes and alignment.
        tensors.append((f'blk.{layer}.attn.weight', (32,), 8, bytes([layer + 1]) * 34))
        for rest in ('ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight'):
            tensors.append((f'blk.{layer}.{rest}', (32,), 0, bytes([layer + 11]) * 128))
    write_fixture(path, fields, tensors, 64)


class ShardingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.source = self.directory / 'source.gguf'; fixture(self.source)
        self.lock = make_lock(self.source, 'fixture/model', 'a' * 40, PIN)

    def split(self, start, end, role, name=None, **options):
        path = self.directory / (name or role + '.gguf')
        receipt = split_gguf(self.source, path, start, end, role, source_lock=self.lock, **options)
        return path, receipt

    def test_three_ranges_cover_once_and_copy_quantized_bytes(self):
        results = [self.split(0, 2, 'host'), self.split(2, 6, 'middle'), self.split(6, 8, 'tail')]
        partition = validate_partition([r for _, r in reversed(results)], PIN)
        self.assertEqual([s['layer_map'] for s in partition['shards']], [[0, 1], [2, 3, 4, 5], [6, 7]])
        self.assertFalse(partition['hardware_validated'])
        for path, receipt in results:
            verify_shard(receipt, path, PIN, self.source)
            _, fields, tensors, _, _ = _document(path)
            self.assertEqual(fields['split.source_sha256'].value, self.lock['sha256'])
            self.assertEqual(fields['backcluster.source.sha256'].value, self.lock['sha256'])
            self.assertEqual(fields['qwen35.attention.head_count'].value, receipt['layer_map'])
            self.assertEqual(fields['qwen35.rope.dimension_sections'].value, [2, 2, 0, 0])
            self.assertEqual(fields['tokenizer.ggml.tokens'].value, [f'token{i}' for i in range(8)])
            self.assertTrue(all(t.kind == 8 for t in tensors if t.name.endswith('attn.weight')))
        host, middle, tail = [receipt for _, receipt in results]
        self.assertTrue(host['has_token_embd']); self.assertFalse(host['has_output_weight'])
        self.assertFalse(middle['has_token_embd']); self.assertFalse(middle['output_head'])
        self.assertTrue(tail['has_token_embd']); self.assertTrue(tail['has_output_weight']); self.assertTrue(tail['output_head'])

    def test_mtp_retained_after_local_trunk_and_layer_arrays(self):
        path, receipt = self.split(4, 8, 'tail', keep_mtp=True)
        self.assertEqual(receipt['layer_map'], [4, 5, 6, 7, 8, 9])
        self.assertEqual(receipt['gguf']['nextn_layers'], 2)
        _, fields, _, _, _ = _document(path)
        self.assertEqual(fields['qwen35.attention.recurrent_layers'].value, [True, False, True, False, False, False])
        verify_shard(receipt, path, PIN, self.source)

    def test_headless_and_tied_tail(self):
        path, receipt = self.split(4, 8, 'tail', no_head=True)
        self.assertFalse(receipt['output_head']); self.assertFalse(receipt['has_token_embd']); self.assertFalse(receipt['has_output_weight'])
        verify_shard(receipt, path, PIN, self.source)
        fixture(self.source, tied=True)
        self.lock = make_lock(self.source, 'fixture/model', 'a' * 40, PIN)
        path, receipt = self.split(4, 8, 'tail', name='tied.gguf')
        self.assertTrue(receipt['output_tied']); self.assertTrue(receipt['has_token_embd']); self.assertFalse(receipt['has_output_weight'])
        verify_shard(receipt, path, PIN, self.source)

    def test_partial_ffn_removal_preserves_rest_and_mtp(self):
        path, receipt = self.split(4, 8, 'tail', no_ffn_first=2, keep_mtp=True)
        _, _, tensors, _, _ = _document(path)
        names = {t.name for t in tensors}
        for layer in range(6):
            self.assertEqual(f'blk.{layer}.ffn_up.weight' in names, layer >= 2)
        verify_shard(receipt, path, PIN, self.source)
        host = self.split(0, 4, 'host')[1]
        with self.assertRaisesRegex(ValueError, 'external FFN'):
            validate_partition([host, receipt], PIN)

    def test_explicit_extra_layer_arrays_and_recurrent_fallback(self):
        fixture(self.source, recurrent=False)
        self.lock = make_lock(self.source, 'fixture/model', 'a' * 40, PIN)
        path, receipt = self.split(2, 6, 'middle', layer_array_keys=['qwen35.experimental.by_layer'])
        _, fields, _, _, _ = _document(path)
        self.assertEqual(fields['qwen35.experimental.by_layer'].value, [2, 3, 4, 5])
        self.assertEqual(fields['qwen35.attention.recurrent_layers'].value, [True, False, True, False])
        verify_shard(receipt, path, PIN, self.source)

    def test_invalid_ranges_flags_and_keys_do_not_write(self):
        cases = [(-1, 4, 'host', {}), (0, 8, 'host', {}), (0, 4, 'tail', {}),
                 (2, 8, 'middle', {}), (3, 6, 'middle', {}), (2, 5, 'middle', {}),
                 (True, 4, 'host', {}), (4, 4, 'middle', {}),
                 (2, 6, 'middle', {'keep_mtp': True}), (4, 8, 'tail', {'keep_mtp': True, 'no_head': True}),
                 (0, 4, 'host', {'no_head': True}), (4, 8, 'tail', {'no_ffn_first': -1}),
                 (4, 8, 'tail', {'no_ffn_first': 5}), (4, 8, 'tail', {'no_ffn': True, 'no_ffn_first': 1}),
                 (4, 8, 'tail', {'no_head': 1}), (4, 8, 'tail', {'layer_array_keys': ['tokenizer.ggml.tokens']}),
                 (4, 8, 'tail', {'layer_array_keys': ['qwen35.missing']})]
        for start, end, role, options in cases:
            path = self.directory / 'invalid.gguf'
            with self.subTest(start=start, end=end, role=role, options=options), self.assertRaises(ValueError):
                split_gguf(self.source, path, start, end, role, engine_commit=PIN, **options)
            self.assertFalse(path.exists())

    def test_invalid_layer_array_and_source_without_head_rejected(self):
        fixture(self.source, bad_array=True)
        with self.assertRaisesRegex(ValueError, 'array length'):
            split_gguf(self.source, self.directory / 'bad.gguf', 4, 8, 'tail', engine_commit=PIN)
        fixture(self.source)
        _, fields, tensors, data_start, _ = _document(self.source)
        data = self.source.read_bytes()
        rows = [(t.name, t.shape, t.kind, data[data_start + t.offset:data_start + t.offset + t.nbytes]) for t in tensors if t.name != 'output_norm.weight']
        write_fixture(self.source, fields, rows, 64)
        with self.assertRaisesRegex(ValueError, 'output norm'):
            split_gguf(self.source, self.directory / 'bad.gguf', 4, 8, 'tail', engine_commit=PIN)

    def test_manifest_tampering_and_weight_corruption_rejected(self):
        path, receipt = self.split(4, 8, 'tail')
        for key, value in [('engine_commit', 'b' * 40), ('sha256', '0' * 64), ('layer_map', [4, 5, 6, 8]),
                           ('output_head', False), ('layer_start', True), ('bytes', True), ('tokenizer_sha256', '1' * 64)]:
            bad = copy.deepcopy(receipt); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): verify_shard(bad, path, PIN)
        with path.open('r+b') as file:
            file.seek(-1, 2); file.write(b'X')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            verify_shard(receipt, path, PIN)

    def test_rehashed_wrong_weights_still_fail_exact_source_verification(self):
        path, receipt = self.split(4, 8, 'tail')
        _, _, tensors, data_start, _ = _document(path)
        with path.open('r+b') as file:
            file.seek(data_start + tensors[-1].offset); file.write(b'X')
        receipt['sha256'] = digest(path)
        with self.assertRaisesRegex(ValueError, 'weight bytes'):
            verify_shard(receipt, path, PIN, self.source)

    def test_rehashed_forged_embedded_source_identity_rejected(self):
        path, receipt = self.split(4, 8, 'tail')
        receipt['source']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'embedded'):
            verify_shard(receipt, path, PIN)

    def test_source_lock_mismatch_or_derived_parent_rejected(self):
        with self.assertRaises(ValueError):
            split_gguf(self.source, self.directory / 'pin.gguf', 4, 8, 'tail', source_lock=self.lock, engine_commit='b' * 40)
        path, _ = self.split(4, 8, 'tail')
        with self.assertRaisesRegex(ValueError, 'already'):
            split_gguf(path, self.directory / 'again.gguf', 2, 4, 'tail', engine_commit=PIN)
        with self.source.open('r+b') as file:
            file.seek(-1, 2); file.write(b'X')
        with self.assertRaisesRegex(ValueError, 'model lock'):
            self.split(0, 4, 'host')

    def test_partition_gaps_overlaps_headless_and_source_mix_rejected(self):
        host = self.split(0, 2, 'host')[1]
        tail = self.split(4, 8, 'tail')[1]
        with self.assertRaisesRegex(ValueError, 'gap'):
            validate_partition([host, tail], PIN)
        middle = self.split(2, 6, 'middle')[1]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_partition([host, middle, tail], PIN)
        tail = self.split(2, 8, 'tail', name='headless.gguf', no_head=True)[1]
        with self.assertRaisesRegex(ValueError, 'output head'):
            validate_partition([host, tail], PIN)
        fixture(self.source, tied=True)
        self.lock = make_lock(self.source, 'fixture/model', 'a' * 40, PIN)
        tail = self.split(2, 8, 'tail', name='other-source.gguf')[1]
        with self.assertRaisesRegex(ValueError, 'mixes'):
            validate_partition([host, tail], PIN)

    def test_atomic_failure_cleanup_and_existing_destination(self):
        path = self.directory / 'fail.gguf'
        with patch('backcluster.sharding._write', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): split_gguf(self.source, path, 4, 8, 'tail', engine_commit=PIN)
        self.assertFalse(path.exists()); self.assertEqual(list(self.directory.glob('.*.tmp')), [])
        path.write_bytes(b'keep me')
        with self.assertRaises(ValueError): split_gguf(self.source, path, 4, 8, 'tail', engine_commit=PIN)
        self.assertEqual(path.read_bytes(), b'keep me')

    def test_source_mutation_during_copy_never_publishes(self):
        from backcluster.sharding import _write
        def mutate(*args):
            _write(*args)
            with self.source.open('ab') as file: file.write(b'changed')
        path = self.directory / 'fail.gguf'
        with patch('backcluster.sharding._write', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed while'):
                split_gguf(self.source, path, 4, 8, 'tail', engine_commit=PIN)
        self.assertFalse(path.exists())

    def test_inspector_reports_tokenizer_and_specific_weight_sizes(self):
        info = inspect_gguf(self.source)
        self.assertEqual(info['vocab_size'], 8); self.assertEqual(info['embedding_length'], 32)
        self.assertEqual(info['embedding_tensor_bytes'], 1024)
        self.assertEqual(info['output_head_tensor_bytes'], 1152)
        self.assertEqual(info['mtp_tensor_bytes'], 2 * (34 + 3 * 128))
        self.assertEqual(info['tokenizer_sha256'], _fields_hash(_document(self.source)[1], 'tokenizer.'))

    def test_legacy_script_persists_receipt_without_numpy(self):
        path = self.directory / 'legacy.gguf'
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts/split-gguf.py'), str(self.source), str(path),
                                    '-L', '4', '--no-head', '--source-lock', str(self.directory / 'source.json')], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        (self.directory / 'source.json').write_text(json.dumps(self.lock))
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts/split-gguf.py'), str(self.source), str(path),
                                    '-L', '4', '--no-head', '--source-lock', str(self.directory / 'source.json')], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(Path(str(path) + '.lock.json').read_text())
        verify_shard(receipt, path, PIN, self.source)
        self.assertNotIn(self.tmp.name, json.dumps(receipt))

    def test_tiny_inference_fixture_is_deterministic_and_shardable(self):
        a, b = self.directory / 'tiny-a.gguf', self.directory / 'tiny-b.gguf'
        generate(a); generate(b)
        self.assertEqual(digest(a), digest(b))
        info = inspect_gguf(a)
        self.assertEqual(info['trunk_layers'], 8); self.assertEqual(info['embedding_length'], 64)
        receipt = split_gguf(a, self.directory / 'tiny-middle.gguf', 2, 6, 'middle', engine_commit=PIN)
        verify_shard(receipt, self.directory / 'tiny-middle.gguf', PIN, a)


if __name__ == '__main__':
    unittest.main()
