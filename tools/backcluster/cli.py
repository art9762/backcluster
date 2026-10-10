"""Offline diagnostics CLI. No inference launcher or automatic device selection."""
import argparse
import ipaddress
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from . import VERSION
from .devices import collect_profile, doctor
from .profiles import valid_alias, validate_profile
from .rpc import Endpoint


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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="backcluster", description="Read-only BackCluster hardware diagnostics")
    parser.add_argument("--version", action="version", version="backcluster " + VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "profile"):
        command = commands.add_parser(name)
        command.add_argument("--worker", action="append", type=endpoint, default=[], metavar="ALIAS=IP[:PORT]")
        command.add_argument("--timeout", type=timeout_seconds, default=3.0)
        command.add_argument("--output", type=Path, help="write JSON atomically instead of stdout")
        if name == "profile":
            command.add_argument("--check", type=Path, help="validate a saved diagnostic snapshot, without contacting devices")
    args = parser.parse_args(argv)
    if len({e.alias for e in args.worker}) != len(args.worker):
        parser.error("worker aliases must be unique")
    if getattr(args, "check", None) is not None and args.worker:
        parser.error("--check cannot be combined with --worker")
    try:
        if args.command == "doctor":
            result = doctor(args.worker, args.timeout)
            code = 0 if result["ok"] else 1
        elif args.check is not None:
            if args.check.stat().st_size > 1024 * 1024:
                raise ValueError
            document = json.loads(args.check.read_text(encoding="utf-8"))
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
        emit(result, args.output)
        return code
    except (OSError, UnicodeError, ValueError, RecursionError):
        print("backcluster: could not read or write a valid JSON document", file=sys.stderr)
        return 1
