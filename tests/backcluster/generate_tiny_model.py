#!/usr/bin/env python3
"""Generate deterministic, untrained Qwen35 weights for CPU integration tests.

The model has both GDN and full attention, 8 layers, 64-dimensional residuals,
and a 32-token dummy vocabulary. It tests inference plumbing, not model quality.
No third-party Python packages or downloaded weights are required.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.sharding import Field, _string, _value  # noqa: E402


def write_fixture(path, fields, tensors, alignment=32):
    """Write tensors as (name, GGML dimensions, storage type, raw bytes)."""
    with Path(path).open('wb') as file:
        file.write(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(fields)))
        for key, field in fields.items():
            file.write(_string(key)); file.write(struct.pack('<I', field.kind)); file.write(_value(field))
        offset = 0
        for name, shape, kind, data in tensors:
            file.write(_string(name)); file.write(struct.pack('<I', len(shape)))
            file.write(struct.pack('<' + 'Q' * len(shape), *shape)); file.write(struct.pack('<IQ', kind, offset))
            offset += len(data) + (-len(data) % alignment)
        file.write(b'\0' * (-file.tell() % alignment))
        for _, _, _, data in tensors:
            file.write(data); file.write(b'\0' * (-len(data) % alignment))


def generate(path, *, layers=8, interval=2, tied=False):
    if type(layers) is not int or layers < 4 or type(interval) is not int or not 1 <= interval <= layers:
        raise ValueError('at least four layers and a valid attention interval required')
    fields = {
        'general.architecture': Field(8, 'qwen35'),
        'general.name': Field(8, 'BackCluster deterministic CPU fixture'),
        'general.file_type': Field(4, 0),
        'qwen35.context_length': Field(4, 512),
        'qwen35.block_count': Field(4, layers),
        'qwen35.embedding_length': Field(4, 64),
        'qwen35.feed_forward_length': Field(4, 128),
        'qwen35.attention.head_count': Field(4, 4),
        'qwen35.attention.head_count_kv': Field(4, 2),
        'qwen35.attention.key_length': Field(4, 16),
        'qwen35.attention.value_length': Field(4, 16),
        'qwen35.attention.layer_norm_rms_epsilon': Field(6, 1e-6),
        'qwen35.full_attention_interval': Field(4, interval),
        'qwen35.attention.recurrent_layers': Field(9, [(i + 1) % interval != 0 for i in range(layers)], 7),
        'qwen35.rope.dimension_count': Field(4, 16),
        'qwen35.rope.dimension_sections': Field(9, [4, 4, 0, 0], 5),
        'qwen35.rope.freq_base': Field(6, 1000000.0),
        'qwen35.ssm.conv_kernel': Field(4, 4),
        'qwen35.ssm.inner_size': Field(4, 64),
        'qwen35.ssm.state_size': Field(4, 16),
        'qwen35.ssm.time_step_rank': Field(4, 4),
        'qwen35.ssm.group_count': Field(4, 2),
        'qwen35.vocab_size': Field(4, 32),
        'tokenizer.ggml.model': Field(8, 'none'),
        'tokenizer.chat_template': Field(8, '{{ messages }}'),
    }
    tensors = []

    def add(name, shape, *, norm=False, constant=None):
        rng = random.Random(int.from_bytes(hashlib.sha256(name.encode()).digest(), 'little'))
        count = math.prod(shape)
        data = struct.pack('<' + 'f' * count, *[
            (constant if constant is not None else (1.0 if norm else rng.uniform(-0.03, 0.03)))
            for _ in range(count)])
        tensors.append((name, shape, 0, data))

    add('token_embd.weight', (64, 32))
    add('output_norm.weight', (64,), norm=True)
    if not tied:
        add('output.weight', (64, 32))
    for layer in range(layers):
        prefix = f'blk.{layer}.'
        add(prefix + 'attn_norm.weight', (64,), norm=True)
        add(prefix + 'post_attention_norm.weight', (64,), norm=True)
        if (layer + 1) % interval == 0:
            add(prefix + 'attn_q.weight', (64, 128))
            add(prefix + 'attn_k.weight', (64, 32))
            add(prefix + 'attn_v.weight', (64, 32))
            add(prefix + 'attn_output.weight', (64, 64))
            add(prefix + 'attn_q_norm.weight', (16,), norm=True)
            add(prefix + 'attn_k_norm.weight', (16,), norm=True)
        else:
            add(prefix + 'attn_qkv.weight', (64, 128))
            add(prefix + 'attn_gate.weight', (64, 64))
            add(prefix + 'ssm_conv1d.weight', (4, 128))
            add(prefix + 'ssm_dt.bias', (4,), constant=0.0)
            add(prefix + 'ssm_a', (4,), constant=-1.0)
            add(prefix + 'ssm_beta.weight', (64, 4))
            add(prefix + 'ssm_alpha.weight', (64, 4))
            add(prefix + 'ssm_norm.weight', (16,), norm=True)
            add(prefix + 'ssm_out.weight', (64, 64))
        add(prefix + 'ffn_gate.weight', (64, 128))
        add(prefix + 'ffn_up.weight', (64, 128))
        add(prefix + 'ffn_down.weight', (128, 64))
    write_fixture(path, fields, tensors)
    return Path(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--tied', action='store_true')
    args = parser.parse_args()
    generate(args.output, layers=args.layers, interval=args.interval, tied=args.tied)
    print(f'{args.output}: deterministic untrained Qwen35 CPU fixture')
