"""Bounded, read-only access to the existing worker control port."""
import json
import socket
import time
from dataclasses import dataclass

MAX_REPLY_BYTES = 16 * 1024


class ProbeError(Exception):
    """A safe error code: never include a remote reply or endpoint in diagnostics."""


@dataclass(frozen=True)
class Endpoint:
    alias: str
    host: str
    port: int = 50061


def request(endpoint, command="profile", timeout=3.0):
    if command not in ("profile", "mem"):
        raise ValueError("only read-only profile/mem requests are supported")
    deadline = time.monotonic() + timeout

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise ProbeError("timeout")
        return left

    def invalid_constant(_):
        raise ValueError("non-finite JSON number")

    try:
        # Resolve only literal IPs/localhost at the CLI boundary. No discovery,
        # relaunch, DNS, pairing or changes to the app's network policy.
        with socket.create_connection((endpoint.host, endpoint.port), timeout=remaining()) as conn:
            conn.settimeout(remaining())
            conn.sendall((command + "\n").encode("ascii"))
            data = bytearray()
            while True:
                conn.settimeout(remaining())
                chunk = conn.recv(min(4096, MAX_REPLY_BYTES + 1 - len(data)))
                if not chunk:
                    raise ProbeError("unexpected_eof")
                data.extend(chunk)
                if len(data) > MAX_REPLY_BYTES:
                    raise ProbeError("reply_too_large")
                if b"\n" in data:
                    line, extra = data.split(b"\n", 1)
                    if extra.strip():
                        raise ProbeError("invalid_reply")
                    result = json.loads(line.decode("utf-8"), parse_constant=invalid_constant)
                    if not isinstance(result, dict):
                        raise ProbeError("invalid_reply")
                    return result
    except (TimeoutError, socket.timeout):
        raise ProbeError("timeout") from None
    except (UnicodeError, ValueError, RecursionError):
        raise ProbeError("invalid_reply") from None
    except OSError:
        raise ProbeError("connection_failed") from None
