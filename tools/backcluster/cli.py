"""Offline preparation, explicit diagnostics and a Mac baseline launcher."""
import argparse
import ipaddress
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import VERSION
from .devices import collect_profile, doctor, engine_status, run
from .profiles import valid_alias, validate_profile
from .rpc import Endpoint
from .artifacts import load_json, make_lock, verify_lock, validate_lock
from .planning import make_plan, registry_endpoints, registry_checks, active_workers
from .linkbench import run_link_bench
from .serving import launch_spec, cluster_launch_spec, verify_routes, serve, build_engine
from .benchmark import run_bench, compare_benchmarks
from .quality import run_quality, opencode_config
from .sharding import split_gguf, verify_shard, validate_partition
from .runtime import runtime_selftest
from .engine_overlay import expected_identity


def endpoint(value):
    try:
        alias, address = value.split("=", 1)
        if not valid_alias(alias) or alias == "mac":
            raise ValueError
        # Brackets for IPv6, optional control port; literal addresses only.
        if address.startswith("["):
            host, suffix = address[1:].split("]", 1)
            if suffix and not suffix.startswith(":"):
                raise ValueError
            port = int(suffix[1:]) if suffix else 50061
        elif address.count(":") <= 1:
            host, separator, port_text = address.partition(":")
            port = int(port_text) if separator else 50061
        else:
            host, port = address, 50061
        if host != "localhost":
            ipaddress.ip_address(host)
        if not 1 <= port <= 65535:
            raise ValueError
        return Endpoint(alias, host, port)
    except ValueError:
        # Do not echo an invalid endpoint (it could contain an ID or name).
        raise argparse.ArgumentTypeError("use alias=literal-IP[:port], alias=[IPv6]:port, or alias=localhost:port") from None


def timeout_seconds(value):
    try:
        result = float(value)
    except ValueError:
        result = 0
    if not math.isfinite(result) or not 0.1 <= result <= 30:
        raise argparse.ArgumentTypeError("timeout must be between 0.1 and 30 seconds")
    return result


def inference_timeout(value):
    try:
        result = float(value)
    except ValueError:
        result = 0
    if not math.isfinite(result) or not 1 <= result <= 600:
        raise argparse.ArgumentTypeError('inference timeout must be between 1 and 600 seconds')
    return result


def emit(document, output=None):
    text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    # Private local snapshots; replace atomically so interruption leaves the
    # previous file intact. No repository paths or endpoints are persisted.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bounded_int(low, high):
    def parse(value):
        try:
            result = int(value)
        except ValueError:
            result = low - 1
        if not low <= result <= high:
            raise argparse.ArgumentTypeError("integer must be between %d and %d" % (low, high))
        return result
    return parse


def byte_assignment(value):
    try:
        alias, amount = value.split("=", 1)
        if not valid_alias(alias):
            raise ValueError
        amount = int(amount)
        if not 1 <= amount <= 2**63 - 1:
            raise ValueError
        return alias, amount
    except ValueError:
        raise argparse.ArgumentTypeError("use alias=positive-integer-bytes") from None


def protect_inputs(args):
    inputs = []
    for name in ('file', 'lock', 'profile', 'registry', 'check', 'prompt', 'launch', 'baseline',
                 'candidate', 'source', 'tail_file', 'tail_manifest', 'middle_file', 'middle_manifest', 'link_profile'):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            inputs.append(value)
    manifest = getattr(args, 'manifest', None)
    if args.command != 'shard' and manifest:
        inputs.extend(manifest if isinstance(manifest, list) else [manifest])
    outputs = [getattr(args, 'output', None)]
    if args.command == 'shard':
        outputs.append(manifest)
    outputs = [path for path in outputs if isinstance(path, Path)]
    for index, output in enumerate(outputs):
        for source in inputs + outputs[:index]:
            same = output.resolve() == source.resolve()
            if not same and output.exists() and source.exists():
                same = os.path.samefile(output, source)
            if same:
                raise ValueError('output paths must be distinct from inputs and from each other')


def main(argv=None):
    parser = argparse.ArgumentParser(prog="backcluster", description="BackCluster offline preparation and bounded hardware tests")
    parser.add_argument("--version", action="version", version="backcluster " + VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "profile"):
        command = commands.add_parser(name, help='check prerequisites' if name == 'doctor' else 'collect or validate device snapshots')
        command.add_argument("--worker", action="append", type=endpoint, default=[], metavar="ALIAS=IP[:PORT]")
        command.add_argument("--registry", type=Path, help="explicit private worker registry")
        command.add_argument("--timeout", type=timeout_seconds, default=3.0)
        command.add_argument("--output", type=Path, help="write JSON atomically instead of stdout")
        if name == "profile":
            command.add_argument("--check", type=Path, help="validate a saved diagnostic snapshot, without contacting devices")
        else:
            command.add_argument('--human', action='store_true', help='show readable checks and suggested fixes; JSON remains the default')
    model = commands.add_parser("model", help="lock or verify a local GGUF, without downloading")
    model.add_argument("--file", type=Path, required=True)
    model.add_argument("--repository")
    model.add_argument("--revision")
    model.add_argument("--check", type=Path)
    model.add_argument("--output", type=Path)
    shard = commands.add_parser('shard', help='stream a reproducible contiguous GGUF shard and manifest')
    shard.add_argument('--file', type=Path, required=True, help='full source GGUF')
    shard.add_argument('--lock', type=Path, required=True, help='content lock for the full source')
    shard.add_argument('--start', type=bounded_int(0, 10000), required=True)
    shard.add_argument('--end', type=bounded_int(1, 10000), required=True)
    shard.add_argument('--role', choices=('host', 'middle', 'tail'), required=True)
    shard.add_argument('--keep-mtp', action='store_true')
    shard.add_argument('--no-head', action='store_true')
    shard.add_argument('--output', type=Path, required=True, help='derived GGUF path')
    shard.add_argument('--manifest', type=Path, required=True, help='private derived identity receipt')
    shard_check = commands.add_parser('shard-check', help='verify a derived GGUF against its manifest')
    shard_check.add_argument('--file', type=Path, required=True)
    shard_check.add_argument('--manifest', type=Path, required=True)
    shard_check.add_argument('--source', type=Path, help='also verify the original parent GGUF')
    shard_check.add_argument('--output', type=Path)
    partition = commands.add_parser('partition-check', help='verify exact host/middle/tail layer coverage')
    partition.add_argument('--manifest', type=Path, action='append', required=True)
    partition.add_argument('--output', type=Path)
    plan = commands.add_parser("plan", help="review conservative estimates; never launch a cluster")
    plan.add_argument("--lock", type=Path, required=True)
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--registry", type=Path)
    plan.add_argument("--mode", choices=('mac', 'remote-kv', 'split-prefill', 'split-decode', 'capacity', 'pipeline'), default="mac")
    plan.add_argument("--host-budget-bytes", type=bounded_int(1, 2**63-1), required=True)
    plan.add_argument("--host-reserve-bytes", type=bounded_int(1, 2**63-1), required=True)
    plan.add_argument("--context", type=bounded_int(128, 131072), default=16384)
    plan.add_argument("--state-bytes", type=byte_assignment, action="append", default=[])
    plan.add_argument('--host-state-bytes', type=bounded_int(1, 2**63-1))
    plan.add_argument('--split-layer', type=bounded_int(1, 10000))
    plan.add_argument('--middle-end', type=bounded_int(2, 10000))
    plan.add_argument('--link-profile', type=Path, help='measured payload link report for latency estimates')
    plan.add_argument("--output", type=Path)
    link = commands.add_parser("link-bench", help="measure explicit workers, including simultaneous links")
    link.add_argument("--worker", type=endpoint, action="append", default=[])
    link.add_argument("--registry", type=Path)
    link.add_argument("--timeout", type=timeout_seconds, default=10)
    link.add_argument("--repeats", type=bounded_int(1, 20), default=5)
    link.add_argument("--concurrent", action="store_true")
    link.add_argument("--output", type=Path)
    commands.add_parser("build", help="build pinned Mac llama-server; no install or sysctl changes")
    launch = commands.add_parser("serve", help="launch a locked baseline or explicit cluster on loopback")
    launch.add_argument("--lock", type=Path, required=True)
    launch.add_argument("--file", type=Path, required=True)
    launch.add_argument("--context", type=bounded_int(128, 131072), default=16384)
    launch.add_argument("--port", type=bounded_int(1024, 65535), default=8080)
    launch.add_argument('--mode', choices=('mac', 'remote-kv', 'split-prefill', 'split-decode', 'capacity', 'pipeline'), default='mac')
    launch.add_argument('--registry', type=Path)
    launch.add_argument('--profile', type=Path, help='snapshot for dry-run; live launch probes devices again')
    launch.add_argument('--host-budget-bytes', type=bounded_int(1, 2**63-1))
    launch.add_argument('--host-reserve-bytes', type=bounded_int(1, 2**63-1))
    launch.add_argument('--host-state-bytes', type=bounded_int(1, 2**63-1))
    launch.add_argument('--state-bytes', type=byte_assignment, action='append', default=[])
    launch.add_argument('--split-layer', type=bounded_int(1, 10000))
    launch.add_argument('--tail-file', type=Path)
    launch.add_argument('--tail-manifest', type=Path)
    launch.add_argument('--middle-file', type=Path)
    launch.add_argument('--middle-manifest', type=Path)
    launch.add_argument('--middle-end', type=bounded_int(2, 10000))
    launch.add_argument('--total-context', type=bounded_int(128, 131072), help='remote-KV total context; --context is the local window')
    launch.add_argument('--cache-type', choices=('q8_0', 'q4_0', 'f16'), default='q8_0')
    launch.add_argument('--timeout', type=timeout_seconds, default=3)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--output", type=Path, help="private launch receipt (includes local model path)")
    bench = commands.add_parser("bench", help="bounded loopback SSE baseline with private raw samples")
    bench.add_argument("--lock", type=Path, required=True)
    bench.add_argument("--prompt", type=Path, required=True)
    bench.add_argument("--port", type=bounded_int(1024, 65535), default=8080)
    bench.add_argument("--repeats", type=bounded_int(1, 20), default=5)
    bench.add_argument("--tokens", type=bounded_int(1, 4096), default=256)
    bench.add_argument('--launch', type=Path, help='launch receipt required for comparable reports')
    bench.add_argument('--timeout', type=inference_timeout, default=120)
    bench.add_argument("--output", type=Path)
    comparison = commands.add_parser('bench-compare', help='compare locked raw benchmark evidence and output hashes')
    comparison.add_argument('--baseline', type=Path, required=True)
    comparison.add_argument('--candidate', type=Path, required=True)
    comparison.add_argument('--allow-runtime-changes', action='store_true', help='compare an explicitly changed tuning experiment')
    comparison.add_argument('--output', type=Path)
    quality = commands.add_parser('quality', help='check loopback SSE, JSON, reasoning and read/edit/test tool round-trip')
    quality.add_argument('--lock', type=Path, required=True)
    quality.add_argument('--launch', type=Path)
    quality.add_argument('--port', type=bounded_int(1024, 65535), default=8080)
    quality.add_argument('--tokens', type=bounded_int(1, 4096), default=256)
    quality.add_argument('--timeout', type=inference_timeout, default=120)
    quality.add_argument('--output', type=Path)
    opencode = commands.add_parser('opencode-config', help='write an explicit local OpenCode provider configuration')
    opencode.add_argument('--lock', type=Path, required=True)
    opencode.add_argument('--context', type=bounded_int(128, 131072), required=True)
    opencode.add_argument('--tokens', type=bounded_int(1, 4096), default=4096)
    opencode.add_argument('--port', type=bounded_int(1024, 65535), default=8080)
    opencode.add_argument('--client-version', required=True, help='installed OpenCode semantic version')
    opencode.add_argument('--reasoning', action='store_true', help='enable only after verifying separate reasoning output')
    opencode.add_argument('--output', type=Path, required=True)
    runtime_check = commands.add_parser('runtime-selftest', help='check transactional stages and recovery using real loopback sockets')
    runtime_check.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not hasattr(args, "worker"):
        args.worker = []
    if len({e.alias for e in args.worker}) != len(args.worker):
        parser.error("worker aliases must be unique")
    if getattr(args, "check", None) is not None and (args.worker or getattr(args, "registry", None)):
        parser.error("--check cannot be combined with --worker or --registry")
    try:
        protect_inputs(args)
        registry = load_json(args.registry) if getattr(args, "registry", None) else None
        build_id = expected_identity() if (registry or args.command == 'plan') and (Path(__file__).resolve().parents[2] / 'engine-patches/manifest.json').is_file() else None
        if registry and args.command in ("doctor", "profile", "link-bench"):
            if args.worker:
                parser.error("--registry cannot be combined with --worker")
            args.worker = registry_endpoints(registry)
            # A registry is an assertion about the physical route. Verify the
            # actual route on this Mac before any live probes. No route mutation.
            for expected in registry["workers"]:
                route = run("/sbin/route", "-n", "get", expected["host"])
                if not any(line.strip() == "interface: " + expected["interface"] for line in route.splitlines()):
                    raise ValueError("worker route mismatch")
        if args.command == "build":
            build_engine()
            return 0
        if args.command == 'shard':
            if args.output.resolve() == args.manifest.resolve() or args.file.resolve() == args.manifest.resolve():
                raise ValueError('shard, source and manifest paths must be distinct')
            result = split_gguf(args.file, args.output, args.start, args.end, args.role,
                                keep_mtp=args.keep_mtp, no_head=args.no_head,
                                source_lock=load_json(args.lock), engine_commit=engine_status()['expected_commit'])
            emit(result, args.manifest)
            return 0
        if args.command == 'runtime-selftest':
            result = runtime_selftest()
            code = 0 if result['ok'] else 1
        elif args.command == 'shard-check':
            verify_shard(load_json(args.manifest), args.file, engine_status()['expected_commit'], args.source)
            result = {'ok': True, 'hardware_validated': False}
            code = 0
        elif args.command == 'partition-check':
            result = validate_partition([load_json(path) for path in args.manifest], engine_status()['expected_commit'])
            code = 0
        elif args.command == "model":
            pin = engine_status()["expected_commit"]
            if args.check:
                verify_lock(load_json(args.check), args.file, pin)
                result = {"ok": True, "hardware_validated": False}
            else:
                if not args.repository or not args.revision:
                    parser.error("model lock requires --repository and --revision")
                result = make_lock(args.file, args.repository, args.revision, pin)
            code = 0
        elif args.command == "plan":
            if len(dict(args.state_bytes)) != len(args.state_bytes):
                parser.error("state estimate aliases must be unique")
            result = make_plan(load_json(args.lock), load_json(args.profile), registry,
                               engine_status()["expected_commit"], args.mode, args.host_budget_bytes,
                               args.host_reserve_bytes, args.context, args.state_bytes,
                               host_state_bytes=args.host_state_bytes, split_layer=args.split_layer,
                               middle_end=args.middle_end,
                               engine_build_id=build_id,
                               link_profile=load_json(args.link_profile) if args.link_profile else None)
            code = 0 # a blocked plan is a valid reviewable result, never launch authorization
        elif args.command == "link-bench":
            if registry:
                checks = registry_checks(registry, collect_profile(args.worker, args.timeout), engine_status()["expected_commit"], build_id)
                if any(not c["ok"] for c in checks):
                    raise ValueError("worker compatibility mismatch")
            result = run_link_bench(args.worker, args.repeats, args.timeout, concurrent=args.concurrent)
            code = 0 if result["ok"] else 1
        elif args.command == "serve":
            snapshot = load_json(args.profile) if args.profile else None
            if args.mode != 'mac' and snapshot is None:
                if not registry:
                    parser.error('cluster serve requires --registry')
                verify_routes(registry, args.mode)
                workers = active_workers(registry, args.mode)
                snapshot = collect_profile([Endpoint(w['alias'], w['host'], w['control_port']) for w in workers], args.timeout)
            options = dict(mode=args.mode, registry=registry, profile=snapshot,
                           host_budget=args.host_budget_bytes, host_reserve=args.host_reserve_bytes,
                           host_state_bytes=args.host_state_bytes, state_estimates=args.state_bytes,
                           split_layer=args.split_layer, shard_model=args.tail_file,
                           shard_manifest=load_json(args.tail_manifest) if args.tail_manifest else None,
                           middle_model=args.middle_file,
                           middle_manifest=load_json(args.middle_manifest) if args.middle_manifest else None,
                           middle_end=args.middle_end, total_context=args.total_context,
                           cache_type=args.cache_type, timeout=args.timeout)
            result = cluster_launch_spec(load_json(args.lock), args.file, args.context, args.port, **options)
            if not args.dry_run:
                if args.output:
                    emit(result, args.output)
                serve(result)
            code = 0
        elif args.command == "bench":
            lock = validate_lock(load_json(args.lock), engine_status()['expected_commit'])
            if args.prompt.stat().st_size > 1024 * 1024:
                raise ValueError("prompt exceeds limit")
            result = run_bench(args.port, args.prompt.read_text(encoding="utf-8"),
                               "backcluster-" + lock["sha256"][:16], lock, args.repeats, args.tokens,
                               timeout=args.timeout, launch=load_json(args.launch) if args.launch else None)
            code = 0 if result["ok"] else 1
        elif args.command == 'bench-compare':
            result = compare_benchmarks(load_json(args.baseline), load_json(args.candidate),
                                        allow_runtime_changes=args.allow_runtime_changes)
            code = 0 if result['ok'] else 1
        elif args.command == 'quality':
            lock = validate_lock(load_json(args.lock), engine_status()['expected_commit'])
            result = run_quality(args.port, 'backcluster-' + lock['sha256'][:16], lock,
                                 timeout=args.timeout, tokens=args.tokens,
                                 launch=load_json(args.launch) if args.launch else None)
            code = 0 if result['ok'] else 1
        elif args.command == 'opencode-config':
            lock = validate_lock(load_json(args.lock), engine_status()['expected_commit'])
            result = opencode_config(lock, args.context, args.tokens, args.port,
                                     version=args.client_version, reasoning=args.reasoning)
            code = 0
        elif args.command == "doctor":
            result = doctor(args.worker, args.timeout)
            code = 0 if result["ok"] else 1
        elif args.check is not None:
            document = load_json(args.check)
            errors = validate_profile(document)
            result = {"ok": not errors, "errors": errors, "cluster_ready": False}
            code = 0 if not errors else 1
        else:
            result = collect_profile(args.worker, args.timeout)
            errors = validate_profile(result)
            if errors:
                print("backcluster: invalid collected profile", file=sys.stderr)
                return 1
            code = 0 if result["host"]["platform"] == "macOS" and all(w["status"] == "ok" for w in result["workers"]) else 1
        if registry and args.command in ("profile", "doctor"):
            snapshot = result["profile"] if args.command == "doctor" else result
            checks = registry_checks(registry, snapshot, engine_status()["expected_commit"], build_id)
            result["registry_checks"] = checks
            if any(not c["ok"] for c in checks):
                code = 1
                if args.command == "doctor":
                    result["ok"] = False
        if args.command == 'doctor' and args.human:
            if args.output:
                emit(result, args.output)
            print('BackCluster ' + VERSION + ' — environment checks')
            for check in result['checks'] + result.get('registry_checks', []):
                print('  [%s] %s' % ('OK' if check['ok'] else 'FIX', check.get('name', check.get('alias'))))
                if check.get('hint'):
                    print('        ' + check['hint'])
                if check.get('errors'):
                    print('        ' + ', '.join(check['errors']))
            print('Environment ready.' if result['ok'] else 'Resolve the checks marked FIX before building.')
            print('Next: ' + result['next_step'])
        else:
            emit(result, args.output)
        return code
    except subprocess.CalledProcessError as error:
        print('backcluster: build command failed (exit %d); see compiler output above' % error.returncode, file=sys.stderr)
        return 1
    except OSError as error:
        detail = {2: 'required file or executable is missing', 13: 'permission denied', 28: 'not enough disk space'}.get(error.errno, 'file or system operation failed')
        print('backcluster: ' + detail + '; see command help', file=sys.stderr)
        return 1
    except (UnicodeError, ValueError) as error:
        detail = str(error) if not isinstance(error, UnicodeError) else 'input must be UTF-8'
        print('backcluster: ' + (detail or 'invalid input') + '; see command help', file=sys.stderr)
        return 1
    except (TypeError, KeyError, AttributeError, RecursionError):
        print('backcluster: invalid document structure; see command help and PRE_HARDWARE.md', file=sys.stderr)
        return 1
