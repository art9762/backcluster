"""Diagnostic contracts, socket failure behavior and privacy regressions."""
import contextlib
import argparse
import io
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from backcluster.cli import emit, endpoint, main, timeout_seconds
from backcluster.devices import collect_profile, engine_status, host_profile
from backcluster.profiles import normalize_worker, validate_profile
from backcluster.rpc import Endpoint, MAX_REPLY_BYTES, ProbeError, request


def native_reply():
    return {
        "document_kind": "backcluster_worker_profile", "schema_version": 1,
        "platform": "iPadOS", "os_version": "27.0.0",
        "hardware": {"model": "iPad17,1", "cpu_cores": 10},
        "memory": {"physical_bytes": 12 * 2**30, "process_footprint_bytes": 2**29,
                   "process_available_bytes": 5 * 2**30, "metal_recommended_bytes": 8 * 2**30,
                   "metal_allocated_bytes": 2**28},
        "thermal_state": 0,
        "runtime": {"app_version": "0.0.5", "app_build": "5", "phone_attn_version": 4,
                    "tail_protocol_version": 2, "tail_state_format": 1},
        "capabilities": ["memory_profile", "remote_kv", "tail_stage"],
    }


@contextlib.contextmanager
def server(chunks, delay=0):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    port = listener.getsockname()[1]

    def serve():
        try:
            with listener.accept()[0] as conn:
                conn.settimeout(2)
                conn.recv(64)
                for chunk in chunks:
                    if delay:
                        time.sleep(delay)
                    conn.sendall(chunk)
        except OSError:
            pass  # expected when the client rejects a reply or times out

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield Endpoint("ipad", "127.0.0.1", port)
    finally:
        listener.close()
        thread.join(timeout=2)


class RpcTests(unittest.TestCase):
    def test_fragmented_json(self):
        reply = json.dumps(native_reply()).encode() + b"\n"
        with server([reply[:17], reply[17:36], reply[36:]]) as worker:
            self.assertEqual(request(worker), native_reply())

    def test_eof_without_newline(self):
        with server([b'{}']) as worker:
            with self.assertRaisesRegex(ProbeError, "unexpected_eof"):
                request(worker)

    def test_oversized_reply(self):
        with server([b"x" * (MAX_REPLY_BYTES + 1)]) as worker:
            with self.assertRaisesRegex(ProbeError, "reply_too_large"):
                request(worker)

    def test_invalid_json(self):
        for data in (b"[]\n", b"not-json\n", b'{"x":NaN}\n', b'{}\nprivate-data'):
            with self.subTest(data=data), server([data]) as worker:
                with self.assertRaisesRegex(ProbeError, "invalid_reply"):
                    request(worker)

    def test_slow_drip_obeys_total_deadline(self):
        with server([b"{"] * 10, delay=0.03) as worker:
            start = time.monotonic()
            with self.assertRaisesRegex(ProbeError, "timeout"):
                request(worker, timeout=0.1)
            self.assertLess(time.monotonic() - start, 0.5)

    def test_mutating_commands_refused(self):
        with self.assertRaises(ValueError):
            request(Endpoint("ipad", "localhost"), "pair")


class ProfileTests(unittest.TestCase):
    def test_real_worker_collection_and_privacy(self):
        native = native_reply()
        native.update(device_name="private-device", endpoint="private-address", serial="private-serial")
        native["hardware"]["owner"] = "private-owner"
        with server([json.dumps(native).encode() + b"\n"]) as worker:
            profile = collect_profile([worker])
        text = json.dumps(profile)
        self.assertNotIn("private-", text)
        self.assertNotIn("127.0.0.1", text)
        self.assertEqual(validate_profile(profile), [])
        self.assertFalse(profile["hardware_validated"])
        row = profile["workers"][0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["device"]["memory"]["physical_bytes"], 12 * 2**30)
        self.assertEqual(row["link"]["transport"], "unverified")

    def test_old_app_not_silently_accepted(self):
        with server([b'{"phys_mb":12288,"avail_mb":6144}\n']) as worker:
            profile = collect_profile([worker])
        self.assertEqual(profile["workers"][0]["error"], "unsupported_profile")

    def test_bad_worker_profiles(self):
        mutations = (
            ("schema_version", 2), ("schema_version", True),
            ("memory", {"physical_bytes": -1}), ("thermal_state", 4),
            ("runtime", {"phone_attn_version": True}),
            ("hardware", {"model": "private-device"}),
            ("os_version", "private-label"),
            ("platform", "macOS"),
        )
        for key, value in mutations:
            data = native_reply()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                normalize_worker(data)

    def test_unknown_fields_and_nonstring_capabilities_discarded(self):
        data = native_reply()
        data["capabilities"].extend([{}, "secret-capability"])
        device = normalize_worker(data)
        self.assertEqual(device["capabilities"], ["memory_profile", "remote_kv", "tail_stage"])

    def test_unknown_host_data_stays_unknown(self):
        with patch("backcluster.devices.platform.system", return_value="Darwin"), patch("backcluster.devices.run", return_value=""):
            profile = host_profile()
        self.assertIsNone(profile["memory"]["physical_bytes"])
        self.assertIsNone(profile["memory"]["process_available_bytes"])

    def test_profile_schema_and_duplicate_aliases(self):
        profile = collect_profile([])
        self.assertEqual(validate_profile(profile), [])
        profile["workers"] = [{"alias": "ipad", "status": "error", "error": "timeout"}] * 2
        self.assertIn("invalid or duplicate worker alias", validate_profile(profile))
        profile["schema_version"] = True
        profile["hardware_validated"] = True
        self.assertGreaterEqual(len(validate_profile(profile)), 3)

    def test_malformed_nested_profile_is_validation_error(self):
        profile = collect_profile([])
        profile["workers"] = [{"alias": [], "status": "error", "error": []}]
        self.assertTrue(validate_profile(profile))


class CliTests(unittest.TestCase):
    def test_endpoint_parsing(self):
        self.assertEqual(endpoint("ipad=127.0.0.1:50061"), Endpoint("ipad", "127.0.0.1"))
        self.assertEqual(endpoint("iphone=[::1]:50161"), Endpoint("iphone", "::1", 50161))
        for value in ("mac=localhost", "ipad=private-name", "ipad=localhost:0", "ipad=localhost:65536", "invalid", "UPPER=localhost"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError) as error:
                endpoint(value)
            self.assertNotIn("private-name", str(error.exception))

    def test_nonfinite_timeout_refused(self):
        for value in ("nan", "inf", "-1", "31"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                timeout_seconds(value)

    def test_atomic_output_and_offline_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            emit(collect_profile([]), output)
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(main(["profile", "--check", str(output)]), 0)
            self.assertFalse(json.loads(captured.getvalue())["cluster_ready"])
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_unsupported_platform_diagnostic_failure(self):
        with patch("backcluster.devices.platform.system", return_value="Linux"), contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(["profile"]), 1)
        self.assertEqual(json.loads(captured.getvalue())["host"]["platform"], "unsupported")

    def test_missing_submodule_not_parent_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "llama.cpp").mkdir()
            status = engine_status(root)
        self.assertIsNone(status["checkout_commit"])
        self.assertFalse(status["matches_pin"])


if __name__ == "__main__":
    unittest.main()
