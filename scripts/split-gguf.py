#!/usr/bin/env python3
"""Stream a GGUF range and persist its source/engine/content receipt.

Legacy tail usage: split-gguf.py IN.gguf OUT.gguf -L 48 [--no-head]
General ranges:    split-gguf.py IN.gguf OUT.gguf --start 16 --end 48 --role middle
No numpy or initialized engine checkout is needed for the byte transform.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from backcluster.artifacts import inspect_gguf, load_json  # noqa: E402
from backcluster.devices import engine_status  # noqa: E402
from backcluster.sharding import split_gguf  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('-L', type=int, help='legacy: first tail layer')
    ap.add_argument('--start', type=int, help='first global trunk layer, inclusive')
    ap.add_argument('--end', type=int, help='last global trunk layer, exclusive')
    ap.add_argument('--role', choices=('host', 'middle', 'tail'))
    ap.add_argument('--keep-mtp', action='store_true', help='retain MTP only on a tail with its head')
    ap.add_argument('--no-head', action='store_true', help='tail returns normalized residual; drop embedding/output')
    ap.add_argument('--no-ffn', action='store_true', help='external FFN owns all selected trunk layers')
    ap.add_argument('--no-ffn-first', type=int, default=0, metavar='N')
    ap.add_argument('--layer-array', action='append', default=[], metavar='KEY', help='additional per-layer metadata array')
    ap.add_argument('--source-lock', help='verify the full source against this model lock')
    ap.add_argument('--engine-commit', help='40-character engine pin; default: this checkout gitlink')
    ap.add_argument('--manifest', help='receipt JSON path; default: OUT.gguf.lock.json')
    args = ap.parse_args(argv)
    if args.L is not None:
        if any(v is not None for v in (args.start, args.end, args.role)):
            ap.error('-L cannot be combined with --start, --end or --role')
    elif any(v is None for v in (args.start, args.end, args.role)):
        ap.error('provide -L or all of --start, --end and --role')
    receipt_path = Path(args.manifest or args.dst + '.lock.json')
    temporary = None
    created_shard = None
    try:
        if receipt_path.exists() or receipt_path.resolve() in (Path(args.src).resolve(), Path(args.dst).resolve()):
            raise ValueError('receipt destination must be a new file distinct from source and shard')
        source_lock = load_json(args.source_lock) if args.source_lock else None
        pin = args.engine_commit or engine_status()['expected_commit']
        start, end, role = args.start, args.end, args.role
        if args.L is not None:
            start, end, role = args.L, inspect_gguf(args.src)['trunk_layers'], 'tail'
        descriptor, temporary = tempfile.mkstemp(prefix='.' + receipt_path.name + '.', dir=receipt_path.parent)
        os.close(descriptor)
        result = split_gguf(args.src, args.dst, start, end, role,
                            keep_mtp=args.keep_mtp, no_head=args.no_head,
                            no_ffn=args.no_ffn, no_ffn_first=args.no_ffn_first,
                            layer_array_keys=args.layer_array, source_lock=source_lock, engine_commit=pin)
        created_shard = Path(args.dst).stat()
        with Path(temporary).open('w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2); file.write('\n')
            file.flush(); os.fsync(file.fileno())
        os.link(temporary, receipt_path)
        print(f"{args.dst}: {role} layers [{start}, {end}), {result['gguf']['tensor_count']} tensors, "
              f"{result['bytes'] / 1e9:.2f} GB; receipt: {receipt_path}")
        return 0
    except (ValueError, OSError) as error:
        # A failed receipt publication must not leave an apparently usable,
        # untracked shard. Only remove the new inode created by this operation.
        if created_shard is not None:
            try:
                current = Path(args.dst).stat()
                if (current.st_dev, current.st_ino) == (created_shard.st_dev, created_shard.st_ino):
                    Path(args.dst).unlink()
            except OSError:
                pass
        print(f'split-gguf: {error}', file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
