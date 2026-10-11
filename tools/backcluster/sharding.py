"""Streaming GGUF range transforms and content-addressed derived receipts.

Weights are copied verbatim, including quantization blocks. A successful receipt
proves the transform and identity; it does not prove a device's forward pass.
Only little-endian GGUF v3 is accepted, as in the model-lock inspector.
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifacts import MAX_ITEMS, Reader, SCALARS, digest, inspect_gguf, verify_lock

BLOCK = re.compile(r'blk\.([0-9]+)\.(.+)')
SHA = re.compile(r'[0-9a-f]{64}')
PIN = re.compile(r'[0-9a-f]{40}')
COPY_BYTES = 8 * 1024 * 1024
FFN = {'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight'}
# These are loaded by the engine as scalars or block_count-sized arrays.
LAYER_ARRAYS = {'attention.recurrent_layers', 'attention.head_count',
                'attention.head_count_kv', 'feed_forward_length',
                'expert_used_count', 'rope.dimension_count',
                'rope.dimension_count_swa', 'attention.sliding_window'}


@dataclass(frozen=True)
class Field:
    kind: int
    value: object
    subtype: int | None = None


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    kind: int
    offset: int
    nbytes: int


def _string(value):
    data = value.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def _value(field):
    if field.kind in SCALARS:
        return struct.pack('<' + SCALARS[field.kind], field.value)
    if field.kind == 8:
        return _string(field.value)
    if field.kind == 9:
        return struct.pack('<IQ', field.subtype, len(field.value)) + b''.join(
            _value(Field(field.subtype, value)) for value in field.value)
    raise ValueError('unsupported GGUF field type')


def _fields_hash(fields, prefix=None):
    h = hashlib.sha256()
    for key, field in sorted(fields.items()):
        if prefix is None or key.startswith(prefix):
            h.update(_string(key)); h.update(struct.pack('<I', field.kind)); h.update(_value(field))
    return h.hexdigest()


def _document(path):
    """Inspect first, then retain the bounded typed metadata and tensor table."""
    info = inspect_gguf(path)
    from .artifacts import TYPE_SIZES
    with Path(path).open('rb') as file:
        r = Reader(file)
        r.read(8)
        tensor_count, kv_count = r.number('Q'), r.number('Q')
        fields = {}
        for _ in range(kv_count):
            key, kind = r.string(), r.number('I')
            if kind == 9:
                subtype, count = r.number('I'), r.number('Q')
                if count > MAX_ITEMS or subtype not in set(SCALARS) | {8}:
                    raise ValueError('invalid GGUF array')
                field = Field(kind, [r.value(subtype) for _ in range(count)], subtype)
            else:
                field = Field(kind, r.value(kind))
            fields[key] = field
        tensors = []
        for _ in range(tensor_count):
            name, dims = r.string(), r.number('I')
            shape = tuple(r.number('Q') for _ in range(dims))
            kind, offset = r.number('I'), r.number('Q')
            block, size = TYPE_SIZES[kind]
            elements = 1
            for dimension in shape:
                elements *= dimension
            tensors.append(Tensor(name, shape, kind, offset, elements // block * size))
        alignment = fields.get('general.alignment', Field(4, 32)).value
        data_start = (file.tell() + alignment - 1) // alignment * alignment
    return info, fields, tensors, data_start, alignment


def _pin(engine_commit):
    if not isinstance(engine_commit, str) or PIN.fullmatch(engine_commit) is None:
        raise ValueError('immutable 40-character engine pin required')


def _range(start, end, role, trunk, interval):
    if any(type(n) is not int for n in (start, end, trunk, interval)) or interval < 1:
        raise ValueError('integer layer range and positive attention interval required')
    if not 0 <= start < end <= trunk:
        raise ValueError('layer range must be nonempty and inside the trunk')
    if start % interval or (end != trunk and end % interval):
        raise ValueError('layer boundaries must preserve the full-attention interval')
    if role == 'host':
        valid = start == 0 and end < trunk
    elif role == 'middle':
        valid = 0 < start < end < trunk
    elif role == 'tail':
        valid = 0 < start and end == trunk
    else:
        valid = False
    if not valid:
        raise ValueError('role does not match host, middle or tail layer range')


def _transform(info, fields, tensors, start, end, role, transform, source_sha, engine_commit):
    arch, trunk, nextn = info['architecture'], info['trunk_layers'], info['nextn_layers']
    interval = info['full_attention_interval']
    interval = (4 if arch == 'qwen35' else 1) if interval is None else interval
    _range(start, end, role, trunk, interval)
    for key in ('keep_mtp', 'no_head', 'no_ffn'):
        if type(transform[key]) is not bool:
            raise ValueError('transform flags must be booleans')
    first = transform['no_ffn_first']
    if type(first) is not int or not 0 <= first <= end - start:
        raise ValueError('no_ffn_first exceeds the selected trunk range')
    if transform['no_ffn'] and first:
        raise ValueError('use either no_ffn or no_ffn_first')
    if transform['keep_mtp'] and (role != 'tail' or transform['no_head']):
        raise ValueError('MTP is supported only on a tail with its output head')
    if role != 'tail' and transform['no_head']:
        raise ValueError('no_head applies only to a tail; middle has no output head')
    extra = transform['layer_array_keys']
    if not isinstance(extra, list) or any(not isinstance(k, str) or not k.startswith(arch + '.') for k in extra) or len(set(extra)) != len(extra):
        raise ValueError('layer-array keys must be unique architecture metadata keys')
    for key in extra:
        if key not in fields or fields[key].kind != 9:
            raise ValueError('explicit layer-array key is missing or is not an array')
    names = {t.name for t in tensors}
    if 'token_embd.weight' not in names or 'output_norm.weight' not in names:
        raise ValueError('source must contain token embedding and output norm tensors')
    mtp_count = nextn if transform['keep_mtp'] else 0
    layer_map = list(range(start, end)) + list(range(trunk, trunk + mtp_count))
    count = len(layer_map)
    output_fields = {}
    for key, field in fields.items():
        # Source must be an unsplit model. Standard GGUF multipart annotations
        # describe the old file and must not leak into a newly written shard.
        if key.startswith('split.') or key.startswith('GGUF.'):
            continue
        if key == arch + '.block_count':
            field = Field(field.kind, count)
        elif key == arch + '.nextn_predict_layers':
            field = Field(field.kind, mtp_count)
        elif key == 'general.name':
            field = Field(8, f'{field.value} [{role} {start}:{end}]')
        elif field.kind == 9 and (key in extra or (key.startswith(arch + '.') and key[len(arch) + 1:] in LAYER_ARRAYS)):
            if len(field.value) not in (info['block_count'], trunk):
                raise ValueError('invalid per-layer array length: ' + key)
            if mtp_count and len(field.value) != info['block_count']:
                raise ValueError('per-layer array is missing MTP entries: ' + key)
            field = Field(9, [field.value[i] for i in layer_map], field.subtype)
        output_fields[key] = field
    # Write the global recurrent pattern explicitly. This avoids local-index
    # inference after renumbering and ensures MTP entries remain non-recurrent.
    recurrent_key = arch + '.attention.recurrent_layers'
    if arch == 'qwen35' and recurrent_key not in output_fields:
        output_fields[recurrent_key] = Field(9, [i < trunk and (i + 1) % interval != 0 for i in layer_map], 7)
    output_fields.update({
        'split.format_version': Field(4, 1), 'split.layer_start': Field(4, start),
        'split.layer_end': Field(4, end), 'split.n_layer_full': Field(4, trunk),
        'split.layer_map': Field(9, layer_map, 4), 'split.role': Field(8, role),
        'split.source_sha256': Field(8, source_sha), 'split.engine_commit': Field(8, engine_commit),
        'split.no_head': Field(4, int(role == 'middle' or transform['no_head'])),
        'split.no_ffn': Field(4, int(transform['no_ffn'] or bool(first))),
    })
    output_fields['backcluster.source.sha256'] = Field(8, source_sha)
    local_layers = {global_layer: local_layer for local_layer, global_layer in enumerate(layer_map)}
    selected = []
    for tensor in tensors:
        match = BLOCK.fullmatch(tensor.name)
        if match:
            layer, rest = int(match[1]), match[2]
            if layer not in local_layers:
                continue
            if layer < trunk and rest in FFN and (transform['no_ffn'] or layer < start + first):
                continue
            name = f'blk.{local_layers[layer]}.{rest}'
        else:
            if tensor.name == 'output.weight' and (role != 'tail' or transform['no_head']):
                continue
            if tensor.name == 'token_embd.weight' and (role == 'middle' or transform['no_head']):
                continue
            name = tensor.name
        selected.append((name, tensor))
    present = {int(BLOCK.fullmatch(name)[1]) for name, _ in selected if BLOCK.fullmatch(name)}
    if present != set(range(count)):
        raise ValueError('transform removes every tensor from a selected layer')
    return output_fields, selected, layer_map, interval


def _write(src, dst, fields, selected, data_start, alignment):
    with Path(src).open('rb') as source, Path(dst).open('wb') as output:
        output.write(b'GGUF' + struct.pack('<IQQ', 3, len(selected), len(fields)))
        for key, field in fields.items():
            output.write(_string(key)); output.write(struct.pack('<I', field.kind)); output.write(_value(field))
        offset = 0
        for name, tensor in selected:
            output.write(_string(name)); output.write(struct.pack('<I', len(tensor.shape)))
            output.write(struct.pack('<' + 'Q' * len(tensor.shape), *tensor.shape))
            output.write(struct.pack('<IQ', tensor.kind, offset))
            offset = (offset + tensor.nbytes + alignment - 1) // alignment * alignment
        output.write(b'\0' * (-output.tell() % alignment))
        for _, tensor in selected:
            source.seek(data_start + tensor.offset)
            remaining = tensor.nbytes
            while remaining:
                data = source.read(min(remaining, COPY_BYTES))
                if not data:
                    raise ValueError('source tensor truncated during transform')
                output.write(data); remaining -= len(data)
            output.write(b'\0' * (-tensor.nbytes % alignment))
        output.flush(); os.fsync(output.fileno())


def _source_identity(path, info, fields):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size,
            'architecture': info['architecture'], 'trunk_layers': info['trunk_layers'],
            'nextn_layers': info['nextn_layers'], 'tensor_index_sha256': info['tensor_index_sha256'],
            'tokenizer_sha256': _fields_hash(fields, 'tokenizer.'),
            'chat_template_sha256': info['chat_template_sha256']}


def split_gguf(src, dst, start, end, role, *, keep_mtp=False, no_head=False,
               no_ffn=False, no_ffn_first=0, layer_array_keys=(), source_lock=None,
               engine_commit=None):
    """Create a new shard atomically; existing files are never overwritten.

    The caller may persist the returned receipt as JSON. With a source_lock,
    the entire source is checked against its immutable model receipt first.
    """
    if source_lock is not None and not isinstance(source_lock, dict):
        raise ValueError('source model lock must be an object')
    if not isinstance(layer_array_keys, (tuple, list)) or any(not isinstance(key, str) for key in layer_array_keys):
        raise ValueError('layer-array keys must be a sequence of strings')
    if engine_commit is None and source_lock is not None:
        engine_commit = source_lock.get('engine_commit')
    _pin(engine_commit)
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve() or dst.exists():
        raise ValueError('shard destination must be a new file distinct from the source')
    before = src.stat()
    if source_lock is not None:
        verify_lock(source_lock, src, engine_commit)
    info, fields, tensors, data_start, alignment = _document(src)
    if any(k.startswith('split.') and k not in {'split.count', 'split.no', 'split.tensors.count'} for k in fields):
        raise ValueError('source is already a derived shard')
    source = _source_identity(src, info, fields)
    transform = {'keep_mtp': keep_mtp, 'no_head': no_head, 'no_ffn': no_ffn,
                 'no_ffn_first': no_ffn_first, 'layer_array_keys': sorted(layer_array_keys)}
    output_fields, selected, layer_map, interval = _transform(
        info, fields, tensors, start, end, role, transform, source['sha256'], engine_commit)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix='.' + dst.name + '.', suffix='.tmp', dir=dst.parent)
        os.close(descriptor)
        _write(src, temporary, output_fields, selected, data_start, alignment)
        summary, result_fields, result_tensors, _, _ = _document(temporary)
        after = src.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('source changed while splitting')
        names = {t.name for t in result_tensors}
        result = {'document_kind': 'backcluster_shard_lock', 'schema_version': 1,
                  'engine_commit': engine_commit, 'filename': dst.name,
                  'sha256': digest(temporary), 'bytes': Path(temporary).stat().st_size,
                  'source': source, 'gguf': summary, 'layer_start': start, 'layer_end': end,
                  'role': role, 'layer_map': layer_map, 'full_attention_interval': interval,
                  'transform': transform, 'metadata_sha256': _fields_hash(result_fields),
                  'tokenizer_sha256': _fields_hash(result_fields, 'tokenizer.'),
                  'output_head': role == 'tail' and not no_head,
                  'has_token_embd': 'token_embd.weight' in names,
                  'has_output_weight': 'output.weight' in names,
                  'output_tied': not any(t.name == 'output.weight' for t in tensors),
                  'hardware_validated': False}
        if source_lock is not None:
            result['source_model'] = {k: source_lock[k] for k in ('repository', 'revision')}
        validate_shard_manifest(result, engine_commit)
        # link is an atomic no-clobber publish, including a concurrent writer.
        os.link(temporary, dst)
        return result
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def validate_shard_manifest(manifest, engine_commit=None):
    if not isinstance(manifest, dict) or manifest.get('document_kind') != 'backcluster_shard_lock' or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('unsupported derived shard receipt')
    _pin(manifest.get('engine_commit'))
    if engine_commit is not None and manifest['engine_commit'] != engine_commit:
        raise ValueError('shard engine pin differs from this checkout')
    source, info = manifest.get('source'), manifest.get('gguf')
    if not isinstance(source, dict) or not isinstance(info, dict):
        raise ValueError('shard source and GGUF metadata required')
    for container, keys in ((manifest, ('sha256', 'metadata_sha256', 'tokenizer_sha256')),
                            (source, ('sha256', 'tensor_index_sha256', 'tokenizer_sha256'))):
        for key in keys:
            if not isinstance(container.get(key), str) or SHA.fullmatch(container[key]) is None:
                raise ValueError('invalid shard content identity')
    template = source.get('chat_template_sha256')
    if template is not None and (not isinstance(template, str) or SHA.fullmatch(template) is None):
        raise ValueError('invalid source chat-template identity')
    for container in (manifest, source):
        if type(container.get('bytes')) is not int or container['bytes'] <= 0:
            raise ValueError('invalid shard/source size')
    trunk, nextn = source.get('trunk_layers'), source.get('nextn_layers')
    if type(trunk) is not int or not 1 <= trunk <= 10000 or type(nextn) is not int or not 0 <= nextn <= 10000:
        raise ValueError('invalid source layer map')
    start, end, role = (manifest.get(k) for k in ('layer_start', 'layer_end', 'role'))
    _range(start, end, role, trunk, manifest.get('full_attention_interval'))
    transform = manifest.get('transform')
    if not isinstance(transform, dict) or set(transform) != {'keep_mtp', 'no_head', 'no_ffn', 'no_ffn_first', 'layer_array_keys'}:
        raise ValueError('invalid shard transform')
    if any(type(transform[k]) is not bool for k in ('keep_mtp', 'no_head', 'no_ffn')):
        raise ValueError('invalid shard transform flags')
    if type(transform['no_ffn_first']) is not int or not 0 <= transform['no_ffn_first'] <= end - start or (transform['no_ffn'] and transform['no_ffn_first']):
        raise ValueError('invalid shard FFN transform')
    extra = transform['layer_array_keys']
    arch = source.get('architecture')
    if not isinstance(arch, str) or re.fullmatch('[a-z0-9_]{1,64}', arch) is None or not isinstance(extra, list) or any(not isinstance(k, str) or not k.startswith(arch + '.') for k in extra) or len(set(extra)) != len(extra):
        raise ValueError('invalid source architecture or layer arrays')
    mtp = nextn if transform['keep_mtp'] else 0
    if transform['keep_mtp'] and (role != 'tail' or transform['no_head']):
        raise ValueError('MTP requires the tail output head')
    if role != 'tail' and transform['no_head']:
        raise ValueError('no_head applies only to tail')
    expected = list(range(start, end)) + list(range(trunk, trunk + mtp))
    layer_map = manifest.get('layer_map')
    if not isinstance(layer_map, list) or any(type(n) is not int for n in layer_map) or layer_map != expected:
        raise ValueError('shard global layer map mismatch')
    if info.get('architecture') != arch or type(info.get('block_count')) is not int or info['block_count'] != len(expected) or type(info.get('trunk_layers')) is not int or info['trunk_layers'] != end - start or type(info.get('nextn_layers')) is not int or info['nextn_layers'] != mtp:
        raise ValueError('shard GGUF layer metadata mismatch')
    for key in ('output_head', 'has_token_embd', 'has_output_weight', 'output_tied'):
        if type(manifest.get(key)) is not bool:
            raise ValueError('invalid shard head metadata')
    if manifest['output_head'] != (role == 'tail' and not transform['no_head']):
        raise ValueError('shard head policy mismatch')
    if manifest['has_token_embd'] != (role != 'middle' and not transform['no_head']) or (manifest['has_output_weight'] and not manifest['output_head']):
        raise ValueError('shard token embedding/output policy mismatch')
    if manifest['output_head'] and not (manifest['has_output_weight'] or (manifest['output_tied'] and manifest['has_token_embd'])):
        raise ValueError('shard output head is missing')
    if manifest['tokenizer_sha256'] != source['tokenizer_sha256'] or info.get('chat_template_sha256') != source.get('chat_template_sha256'):
        raise ValueError('shard tokenizer/template identity mismatch')
    return manifest


def verify_shard(manifest, path, engine_commit=None, source_path=None):
    """Verify shard bytes, embedded identity and optionally the exact transform."""
    validate_shard_manifest(manifest, engine_commit)
    before = Path(path).stat()
    info, fields, tensors, data_start, _ = _document(path)
    if before.st_size != manifest['bytes'] or digest(path) != manifest['sha256'] or info != manifest['gguf'] or _fields_hash(fields) != manifest['metadata_sha256']:
        raise ValueError('shard artifact does not match its receipt')
    expected_split = {'split.format_version': 1, 'split.layer_start': manifest['layer_start'],
                      'split.layer_end': manifest['layer_end'], 'split.n_layer_full': manifest['source']['trunk_layers'],
                      'split.role': manifest['role'], 'split.layer_map': manifest['layer_map'],
                      'split.source_sha256': manifest['source']['sha256'], 'split.engine_commit': manifest['engine_commit'],
                      'split.no_head': int(manifest['role'] == 'middle' or manifest['transform']['no_head']),
                      'split.no_ffn': int(manifest['transform']['no_ffn'] or bool(manifest['transform']['no_ffn_first']))}
    if {k: f.value for k, f in fields.items() if k.startswith('split.')} != expected_split:
        raise ValueError('embedded shard identity or layer map mismatch')
    if fields.get('backcluster.source.sha256') != Field(8, manifest['source']['sha256']):
        raise ValueError('embedded worker source identity mismatch')
    names = {t.name for t in tensors}
    if ('token_embd.weight' in names) != manifest['has_token_embd'] or ('output.weight' in names) != manifest['has_output_weight'] or 'output_norm.weight' not in names:
        raise ValueError('shard head tensors mismatch')
    if _fields_hash(fields, 'tokenizer.') != manifest['tokenizer_sha256']:
        raise ValueError('shard tokenizer metadata mismatch')
    if source_path is not None:
        source_before = Path(source_path).stat()
        source_info, source_fields, source_tensors, source_start, _ = _document(source_path)
        if _source_identity(source_path, source_info, source_fields) != manifest['source']:
            raise ValueError('source artifact differs from shard parent')
        expected_fields, selected, _, _ = _transform(
            source_info, source_fields, source_tensors, manifest['layer_start'], manifest['layer_end'],
            manifest['role'], manifest['transform'], manifest['source']['sha256'], manifest['engine_commit'])
        if fields != expected_fields or [(t.name, t.shape, t.kind, t.nbytes) for t in tensors] != [(name, t.shape, t.kind, t.nbytes) for name, t in selected]:
            raise ValueError('shard tensor/metadata transform mismatch')
        with Path(source_path).open('rb') as src, Path(path).open('rb') as dst:
            for actual, (_, original) in zip(tensors, selected):
                src.seek(source_start + original.offset); dst.seek(data_start + actual.offset)
                remaining = actual.nbytes
                while remaining:
                    count = min(remaining, COPY_BYTES)
                    if src.read(count) != dst.read(count):
                        raise ValueError('shard tensor differs from source weight bytes')
                    remaining -= count
        source_after = Path(source_path).stat()
        if (source_before.st_size, source_before.st_mtime_ns, source_before.st_ino) != (source_after.st_size, source_after.st_mtime_ns, source_after.st_ino):
            raise ValueError('source changed during shard verification')
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('shard changed during verification')
    return manifest


def validate_partition(manifests, engine_commit=None):
    """Require host, optional middle, tail and one exact trunk coverage."""
    if not isinstance(manifests, (list, tuple)) or not 2 <= len(manifests) <= 3:
        raise ValueError('partition requires host, optional middle and tail')
    shards = sorted((validate_shard_manifest(m, engine_commit) for m in manifests), key=lambda m: m['layer_start'])
    source = shards[0]['source']
    expected_roles = ['host', 'tail'] if len(shards) == 2 else ['host', 'middle', 'tail']
    cursor = 0
    for shard, role in zip(shards, expected_roles):
        if shard['source'] != source or shard['engine_commit'] != shards[0]['engine_commit']:
            raise ValueError('partition mixes source models or engine revisions')
        if shard['role'] != role or shard['layer_start'] != cursor:
            raise ValueError('partition contains a gap, overlap or misplaced role')
        if shard['transform']['no_ffn'] or shard['transform']['no_ffn_first']:
            raise ValueError('partition needs an explicit external FFN owner')
        cursor = shard['layer_end']
    if cursor != source['trunk_layers'] or not shards[-1]['output_head']:
        raise ValueError('partition must cover the trunk and retain its tail output head')
    return {'document_kind': 'backcluster_shard_partition', 'schema_version': 1,
            'source_sha256': source['sha256'], 'engine_commit': shards[0]['engine_commit'],
            'trunk_layers': source['trunk_layers'],
            'shards': [{'role': s['role'], 'layer_start': s['layer_start'], 'layer_end': s['layer_end'],
                        'sha256': s['sha256'], 'bytes': s['bytes'], 'layer_map': s['layer_map']} for s in shards],
            'hardware_validated': False}
