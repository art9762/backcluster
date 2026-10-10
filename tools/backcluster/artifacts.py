"""Bounded GGUF inspection and offline, content-addressed artifact locks.

No weights are loaded into memory. Architecture support remains the engine's
responsibility; a lock proves identity, not inference correctness.
"""
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path

MAX_METADATA = 128 * 1024 * 1024
MAX_ITEMS = 2_000_000
MAX_STRING = 32 * 1024 * 1024
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
# ggml type -> (block elements, storage bytes). Unknown future types fail closed.
TYPE_SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
              6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
              10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
              14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
              18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
              22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2),
              26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
              34: (256, 54), 35: (256, 66), 39: (32, 17), 40: (64, 36),
              41: (128, 18), 42: (64, 18)}
BLOCK = re.compile(r'blk\.([0-9]+)\..+')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Reader:
    def __init__(self, file):
        self.file = file

    def read(self, n):
        if n < 0 or self.file.tell() + n > MAX_METADATA:
            raise ValueError('GGUF metadata exceeds inspection limit')
        data = self.file.read(n)
        if len(data) != n:
            raise ValueError('truncated GGUF metadata')
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, keep=True):
        n = self.number('Q')
        if n > MAX_STRING:
            raise ValueError('GGUF string exceeds inspection limit')
        data = self.read(n)
        return data.decode('utf-8') if keep else None

    def value(self, kind, keep=True):
        if kind in SCALARS:
            v = self.number(SCALARS[kind])
            return v if keep else None
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            subtype, n = self.number('I'), self.number('Q')
            if n > MAX_ITEMS or subtype == 9 or subtype not in set(SCALARS) | {8}:
                raise ValueError('invalid GGUF array')
            values = [] if keep else None
            if not keep and subtype in SCALARS:
                self.read(n * struct.calcsize(SCALARS[subtype]))
            else:
                for _ in range(n):
                    v = self.value(subtype, keep)
                    if keep:
                        values.append(v)
            return values
        raise ValueError('unknown GGUF metadata type')


def inspect_gguf(path):
    path = Path(path)
    with path.open('rb') as f:
        r = Reader(f)
        if r.read(4) != b'GGUF' or r.number('I') != 3:
            raise ValueError('only little-endian GGUF v3 is supported')
        tensor_count, kv_count = r.number('Q'), r.number('Q')
        if not 0 < tensor_count <= MAX_ITEMS or kv_count > MAX_ITEMS:
            raise ValueError('invalid GGUF counts')
        metadata, keys = {}, set()
        for _ in range(kv_count):
            key, kind = r.string(), r.number('I')
            if key in keys:
                raise ValueError('duplicate GGUF metadata')
            keys.add(key)
            keep = (key in {'general.architecture', 'general.alignment', 'general.file_type', 'tokenizer.chat_template',
                            'split.layer_start', 'split.n_layer_full', 'split.count', 'split.no_head', 'split.no_ffn'}
                    or key.endswith(('.block_count', '.nextn_predict_layers', '.full_attention_interval', '.context_length')))
            value = r.value(kind, keep)
            if keep:
                metadata[key] = value
        arch = metadata.get('general.architecture')
        if not isinstance(arch, str) or not re.fullmatch(r'[a-z0-9_]{1,64}', arch):
            raise ValueError('missing GGUF architecture')
        count = metadata.get(arch + '.block_count')
        nextn = metadata.get(arch + '.nextn_predict_layers', 0)
        if type(count) is not int or not 1 <= count <= 10000 or type(nextn) is not int or not 0 <= nextn < count:
            raise ValueError('invalid GGUF layer metadata')
        if metadata.get('split.count', 1) != 1:
            raise ValueError('multipart GGUF must be merged before locking')
        alignment = metadata.get('general.alignment', 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError('invalid GGUF alignment')
        layers, names, spans, types = {}, set(), [], Counter()
        shared = 0
        tensor_index = hashlib.sha256()
        for _ in range(tensor_count):
            name, dims = r.string(), r.number('I')
            if name in names or not 1 <= dims <= 4:
                raise ValueError('duplicate or invalid tensor')
            names.add(name)
            shape = [r.number('Q') for _ in range(dims)]
            kind, offset = r.number('I'), r.number('Q')
            if kind not in TYPE_SIZES or any(d == 0 or d > 2**31 for d in shape):
                raise ValueError('unsupported tensor type/shape')
            block, size = TYPE_SIZES[kind]
            if shape[0] % block:
                raise ValueError('quantization row is not block aligned')
            elements = 1
            for d in shape:
                elements *= d
            nbytes = elements // block * size
            if offset % alignment:
                raise ValueError('unaligned tensor offset')
            spans.append((offset, offset + nbytes))
            types[str(kind)] += 1
            tensor_index.update(json.dumps([name, shape, kind, offset, nbytes], separators=(',', ':')).encode())
            match = BLOCK.fullmatch(name)
            if match:
                layer = int(match[1])
                if layer >= count:
                    raise ValueError('tensor layer outside block_count')
                layers[layer] = layers.get(layer, 0) + nbytes
            else:
                shared += nbytes
        if set(layers) != set(range(count)):
            raise ValueError('missing GGUF layers')
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        previous = 0
        for start, end in sorted(spans):
            if start < previous or data_start + end > path.stat().st_size:
                raise ValueError('overlapping or truncated GGUF tensors')
            previous = end
        template = metadata.get('tokenizer.chat_template')
        if template is not None and not isinstance(template, str):
            raise ValueError('unsupported chat-template metadata')
        return {'architecture': arch, 'block_count': count, 'trunk_layers': count - nextn,
                'nextn_layers': nextn, 'tensor_count': tensor_count,
                'tensor_bytes': shared + sum(layers.values()), 'shared_tensor_bytes': shared,
                'layer_tensor_bytes': [layers[i] for i in range(count)],
                'tensor_types': dict(sorted(types.items())), 'tensor_index_sha256': tensor_index.hexdigest(),
                'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest() if template is not None else None,
                'context_length': metadata.get(arch + '.context_length'),
                'full_attention_interval': metadata.get(arch + '.full_attention_interval'),
                'split': {k: v for k, v in metadata.items() if k.startswith('split.')},
                'file_type': metadata.get('general.file_type')}


def make_lock(path, repository, revision, engine_commit):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('use a repository and immutable 40-character revision')
    if not isinstance(engine_commit, str) or not re.fullmatch(r'[0-9a-f]{40}', engine_commit):
        raise ValueError('engine pin is required')
    path = Path(path)
    before = path.stat()
    result = {'document_kind': 'backcluster_model_lock', 'schema_version': 1,
              'repository': repository, 'revision': revision, 'engine_commit': engine_commit,
              'filename': path.name, 'bytes': before.st_size, 'sha256': digest(path), 'gguf': inspect_gguf(path),
              'hardware_validated': False}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('artifact changed while locking')
    return result


def verify_lock(lock, path, engine_commit):
    validate_lock(lock, engine_commit)
    actual = make_lock(path, lock.get('repository', ''), lock.get('revision', ''), engine_commit)
    # Filename is portable: the same artifact can be moved or renamed locally.
    for key in ('engine_commit', 'bytes', 'sha256', 'gguf'):
        if actual[key] != lock.get(key):
            raise ValueError('artifact or engine does not match model lock')
    return actual


def validate_lock(lock, engine_commit):
    """Validate an offline receipt without pretending to verify weight bytes."""
    if not isinstance(lock, dict) or lock.get('document_kind') != 'backcluster_model_lock' or type(lock.get('schema_version')) is not int or lock['schema_version'] != 1:
        raise ValueError('unsupported model lock; create it with backcluster model')
    for key, width in (('sha256', 64), ('engine_commit', 40), ('revision', 40)):
        if not isinstance(lock.get(key), str) or re.fullmatch('[0-9a-f]{%d}' % width, lock[key]) is None:
            raise ValueError('invalid model lock identity')
    if lock['engine_commit'] != engine_commit:
        raise ValueError('model lock engine pin differs from this checkout')
    if type(lock.get('bytes')) is not int or lock['bytes'] <= 0 or not isinstance(lock.get('gguf'), dict):
        raise ValueError('invalid model lock size or metadata')
    return lock


def load_json(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('JSON document exceeds limit')
    def constant(_):
        raise ValueError('nonfinite JSON')
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=constant, object_pairs_hook=unique_pairs)
