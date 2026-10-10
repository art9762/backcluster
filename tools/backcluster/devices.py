"""Hardware facts only; never infer an inference budget from physical RAM."""
import json
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .profiles import PROFILE_KIND, SCHEMA_VERSION, empty_memory, normalize_worker
from .rpc import ProbeError, request

ROOT = Path(__file__).resolve().parents[2]


def run(*args, timeout=10, cwd=None):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""


def sysctl(key, numeric=False):
    value = run("/usr/sbin/sysctl", "-n", key)
    if numeric:
        try:
            return int(value) if int(value) >= 0 else None
        except ValueError:
            return None
    return value or None


def host_profile():
    mac = platform.system() == "Darwin"
    memory = empty_memory()
    hardware = {"architecture": platform.machine(), "model": None, "chip": None,
                "cpu_cores": None, "p_cores": None, "e_cores": None, "gpu_cores": None}
    if mac:
        memory["physical_bytes"] = sysctl("hw.memsize", numeric=True)
        hardware.update(model=sysctl("hw.model"), chip=sysctl("machdep.cpu.brand_string"),
                        cpu_cores=sysctl("hw.physicalcpu", numeric=True),
                        p_cores=sysctl("hw.perflevel0.physicalcpu", numeric=True),
                        e_cores=sysctl("hw.perflevel1.physicalcpu", numeric=True))
        # SPDisplaysDataType may contain private display names/serials: extract
        # only the core count, never persist the raw system_profiler response.
        try:
            displays = json.loads(run("/usr/sbin/system_profiler", "SPDisplaysDataType", "-json", timeout=30) or "{}")
            for gpu in displays.get("SPDisplaysDataType", []):
                if gpu.get("sppci_cores"):
                    hardware["gpu_cores"] = int(gpu["sppci_cores"])
                    break
        except (ValueError, TypeError, AttributeError):
            pass
    return {
        "platform": "macOS" if mac else "unsupported",
        "os_version": run("/usr/bin/sw_vers", "-productVersion") if mac else None,
        "hardware": hardware, "memory": memory, "thermal_state": None,
        "runtime": {}, "capabilities": [],
    }


def collect_profile(endpoints, timeout=3.0):
    profile = {"document_kind": PROFILE_KIND, "schema_version": SCHEMA_VERSION,
               "captured_at": datetime.now(timezone.utc).isoformat(),
               "hardware_validated": False, "host": host_profile(), "workers": []}
    for endpoint in endpoints:
        row = {"alias": endpoint.alias}
        try:
            start = time.perf_counter_ns()
            reply = request(endpoint, timeout=timeout)
            elapsed = time.perf_counter_ns() - start
            device = normalize_worker(reply)
            row.update(status="ok", device=device,
                       link={"transport": "unverified", "control_round_trip_ns": elapsed})
        except ProbeError as error:
            row.update(status="error", error=str(error))
        except (ValueError, TypeError, KeyError):
            row.update(status="error", error="invalid_worker_profile")
            if isinstance(reply, dict) and reply.get("document_kind") != "backcluster_worker_profile":
                row["error"] = "unsupported_profile"
        profile["workers"].append(row)
    return profile


def engine_status(root=ROOT):
    entry = run("git", "ls-tree", "HEAD", "llama.cpp", cwd=root).split()
    expected = entry[2] if len(entry) >= 3 and entry[0] == "160000" else None
    # git -C in an empty submodule finds the PARENT repo; never report that as
    # the engine revision. An initialized submodule has its own .git entry.
    engine = root / "llama.cpp"
    checkout = run("git", "rev-parse", "HEAD", cwd=engine) if (engine / ".git").exists() else None
    checkout = checkout if checkout and re.fullmatch(r"[0-9a-f]{40}", checkout) else None
    dirty = bool(run("git", "status", "--porcelain", cwd=engine)) if checkout else None
    return {"expected_commit": expected, "checkout_commit": checkout, "dirty": dirty,
            "matches_pin": bool(expected and checkout == expected and not dirty)}


def doctor(endpoints, timeout=3.0):
    profile = collect_profile(endpoints, timeout)
    engine = engine_status()
    xcode = re.search(r"^Xcode ([0-9]+)", run("xcodebuild", "-version"))
    checks = [
        {"name": "apple_silicon_host", "ok": profile["host"]["platform"] == "macOS"
         and profile["host"]["hardware"]["architecture"] == "arm64"},
        {"name": "physical_memory_measured", "ok": bool(profile["host"]["memory"]["physical_bytes"])},
        {"name": "engine_pin", "ok": engine["matches_pin"]},
        {"name": "cmake", "ok": shutil.which("cmake") is not None},
        {"name": "xcode_26_or_newer", "ok": bool(xcode and int(xcode[1]) >= 26)},
        {"name": "metal_toolchain", "ok": bool(run("xcrun", "-sdk", "iphoneos", "-find", "metal"))},
    ]
    checks.extend({"name": "worker_" + w["alias"], "ok": w["status"] == "ok"} for w in profile["workers"])
    return {"document_kind": "backcluster_doctor", "schema_version": 1,
            "ok": all(c["ok"] for c in checks), "checks": checks, "engine": engine,
            "profile": profile, "cluster_ready": False,
            "next_step": "Hardware forward/link benchmarks and model lock are still required."}
