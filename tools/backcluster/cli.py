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
from .planning import make_plan, registry_endpoints, registry_checks
from .linkbench import run_link_bench
from .serving import launch_spec, serve, build_engine
from .benchmark import run_bench


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
    plan = commands.add_parser("plan", help="review conservative estimates; never launch a cluster")
    plan.add_argument("--lock", type=Path, required=True)
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--registry", type=Path)
    plan.add_argument("--mode", choices=("mac", "remote-kv", "capacity"), default="mac")
    plan.add_argument("--host-budget-bytes", type=bounded_int(1, 2**63-1), required=True)
    plan.add_argument("--host-reserve-bytes", type=bounded_int(1, 2**63-1), required=True)
    plan.add_argument("--context", type=bounded_int(128, 131072), default=16384)
    plan.add_argument("--state-bytes", type=byte_assignment, action="append", default=[])
    plan.add_argument("--output", type=Path)
    link = commands.add_parser("link-bench", help="measure explicit workers, including simultaneous links")
    link.add_argument("--worker", type=endpoint, action="append", default=[])
    link.add_argument("--registry", type=Path)
    link.add_argument("--timeout", type=timeout_seconds, default=10)
    link.add_argument("--repeats", type=bounded_int(1, 20), default=5)
    link.add_argument("--concurrent", action="store_true")
    link.add_argument("--output", type=Path)
    commands.add_parser("build", help="build pinned Mac llama-server; no install or sysctl changes")
    launch = commands.add_parser("serve", help="launch the locked Mac-only baseline on loopback")
    launch.add_argument("--lock", type=Path, required=True)
    launch.add_argument("--file", type=Path, required=True)
    launch.add_argument("--context", type=bounded_int(128, 131072), default=16384)
    launch.add_argument("--port", type=bounded_int(1024, 65535), default=8080)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--output", type=Path, help="private launch receipt (includes local model path)")
    bench = commands.add_parser("bench", help="bounded loopback SSE baseline with private raw samples")
    bench.add_argument("--lock", type=Path, required=True)
    bench.add_argument("--prompt", type=Path, required=True)
    bench.add_argument("--port", type=bounded_int(1024, 65535), default=8080)
    bench.add_argument("--repeats", type=bounded_int(1, 20), default=5)
    bench.add_argument("--tokens", type=bounded_int(1, 4096), default=256)
    bench.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not hasattr(args, "worker"):
        args.worker = []
    if len({e.alias for e in args.worker}) != len(args.worker):
        parser.error("worker aliases must be unique")
    if getattr(args, "check", None) is not None and (args.worker or getattr(args, "registry", None)):
        parser.error("--check cannot be combined with --worker or --registry")
    try:
        registry = load_json(args.registry) if getattr(args, "registry", None) else None
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
        if args.command == "model":
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
                               args.host_reserve_bytes, args.context, args.state_bytes)
            code = 0 # a blocked plan is a valid reviewable result, never launch authorization
        elif args.command == "link-bench":
            if registry:
                checks = registry_checks(registry, collect_profile(args.worker, args.timeout), engine_status()["expected_commit"])
                if any(not c["ok"] for c in checks):
                    raise ValueError("worker compatibility mismatch")
            result = run_link_bench(args.worker, args.repeats, args.timeout, concurrent=args.concurrent)
            code = 0 if result["ok"] else 1
        elif args.command == "serve":
            result = launch_spec(load_json(args.lock), args.file, args.context, args.port)
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
                               "backcluster-" + lock["sha256"][:16], lock, args.repeats, args.tokens)
            code = 0 if result["ok"] else 1
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
            checks = registry_checks(registry, snapshot, engine_status()["expected_commit"])
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
