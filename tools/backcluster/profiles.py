"""Versioned diagnostic snapshots, deliberately distinct from runtime admission."""
import re
from datetime import datetime

PROFILE_KIND = "backcluster_device_profile"
SCHEMA_VERSION = 1
MEMORY_KEYS = (
    "physical_bytes", "process_footprint_bytes", "process_available_bytes",
    "metal_recommended_bytes", "metal_allocated_bytes",
)
CAPABILITIES = {"memory_profile", "remote_kv", "tail_stage"}
ERROR_CODES = {
    "timeout", "unexpected_eof", "reply_too_large", "invalid_reply",
    "connection_failed", "unsupported_profile", "invalid_worker_profile",
}


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def label(value):
    # Collected fields are product/OS strings, never user-visible device names.
    return value is None or (isinstance(value, str) and len(value) <= 120
                             and not any(ord(c) < 32 for c in value))


def valid_alias(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value) is not None


def empty_memory():
    return dict.fromkeys(MEMORY_KEYS)


def normalize_worker(reply):
    """Allowlist the native reply before persisting anything from the device."""
    if reply.get("document_kind") != "backcluster_worker_profile" or type(reply.get("schema_version")) is not int or reply["schema_version"] != 1:
        raise ValueError("unsupported_profile")
    hardware, memory, runtime = (reply.get(key) for key in ("hardware", "memory", "runtime"))
    if not all(isinstance(value, dict) for value in (hardware, memory, runtime)):
        raise ValueError("invalid_worker_profile")
    platform = reply.get("platform")
    if platform not in ("iOS", "iPadOS"):
        raise ValueError("invalid_worker_profile")
    product = hardware.get("model")
    prefix = "iPad" if platform == "iPadOS" else "iPhone"
    if not isinstance(product, str) or re.fullmatch(prefix + r"[0-9]+,[0-9]+", product) is None:
        raise ValueError("invalid_worker_profile")
    # These native fields are numeric OS/bundle versions, not arbitrary labels.
    for value in (reply.get("os_version"), runtime.get("app_version"), runtime.get("app_build")):
        if value is not None and (not isinstance(value, str) or re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,3}", value) is None):
            raise ValueError("invalid_worker_profile")
    device = {
        "platform": platform,
        "os_version": reply.get("os_version"),
        "hardware": {key: hardware.get(key) for key in ("model", "cpu_cores")},
        "memory": {key: memory.get(key) for key in MEMORY_KEYS},
        "thermal_state": reply.get("thermal_state"),
        "runtime": {key: runtime.get(key) for key in (
            "app_version", "app_build", "phone_attn_version", "tail_protocol_version", "tail_state_format",
        )},
        "capabilities": [c for c in reply.get("capabilities", []) if isinstance(c, str) and c in CAPABILITIES]
            if isinstance(reply.get("capabilities"), list) else [],
    }
    errors = device_errors(device)
    if errors or not integer(device["memory"]["physical_bytes"], 1) or not integer(device["memory"]["process_available_bytes"]):
        raise ValueError("invalid_worker_profile")
    if not all(integer(device["runtime"][k], 1) for k in (
        "phone_attn_version", "tail_protocol_version", "tail_state_format",
    )) or "memory_profile" not in device["capabilities"]:
        raise ValueError("invalid_worker_profile")
    return device


def device_errors(device):
    errors = []
    if not isinstance(device, dict):
        return ["device must be an object"]
    if device.get("platform") not in ("macOS", "iOS", "iPadOS", "unsupported"):
        errors.append("unsupported platform value")
    if not label(device.get("os_version")):
        errors.append("invalid OS version")
    hardware = device.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("hardware must be an object")
    else:
        for key in ("model", "chip", "architecture"):
            if not label(hardware.get(key)):
                errors.append("invalid hardware label")
        for key in ("cpu_cores", "p_cores", "e_cores", "gpu_cores"):
            if hardware.get(key) is not None and not integer(hardware[key]):
                errors.append("invalid core count")
    memory = device.get("memory")
    if not isinstance(memory, dict) or set(memory) != set(MEMORY_KEYS):
        errors.append("missing or unknown memory fields")
    else:
        for value in memory.values():
            if value is not None and not integer(value):
                errors.append("memory must contain nonnegative integer bytes or null")
    thermal = device.get("thermal_state")
    if thermal is not None and (not integer(thermal) or thermal > 3):
        errors.append("invalid thermal state")
    runtime = device.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime must be an object")
    else:
        for key in ("app_version", "app_build"):
            if not label(runtime.get(key)):
                errors.append("invalid app version")
        for key in ("phone_attn_version", "tail_protocol_version", "tail_state_format"):
            if runtime.get(key) is not None and not integer(runtime[key], 1):
                errors.append("invalid runtime version")
    caps = device.get("capabilities")
    if not isinstance(caps, list) or any(not isinstance(c, str) or c not in CAPABILITIES for c in caps):
        errors.append("unknown capabilities")
    return errors


def validate_profile(profile):
    if not isinstance(profile, dict):
        return ["profile must be an object"]
    errors = []
    if profile.get("document_kind") != PROFILE_KIND or type(profile.get("schema_version")) is not int or profile["schema_version"] != SCHEMA_VERSION:
        errors.append("unsupported profile kind/schema version")
    if profile.get("hardware_validated") is not False:
        errors.append("diagnostic snapshots must set hardware_validated=false")
    try:
        when = datetime.fromisoformat(profile.get("captured_at", "").replace("Z", "+00:00"))
        if when.tzinfo is None:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        errors.append("captured_at must be an ISO timestamp with timezone")
    host = profile.get("host")
    errors.extend("host: " + e for e in device_errors(host))
    workers = profile.get("workers")
    if not isinstance(workers, list):
        return errors + ["workers must be a list"]
    aliases = {"mac"}
    for worker in workers:
        if not isinstance(worker, dict):
            errors.append("worker must be an object")
            continue
        alias = worker.get("alias")
        if not valid_alias(alias) or alias in aliases:
            errors.append("invalid or duplicate worker alias")
        if isinstance(alias, str):
            aliases.add(alias)
        if worker.get("status") == "ok":
            errors.extend("worker: " + e for e in device_errors(worker.get("device")))
            link = worker.get("link")
            if not isinstance(link, dict) or link.get("transport") != "unverified" or not integer(link.get("control_round_trip_ns")):
                errors.append("invalid link sample")
        elif worker.get("status") != "error" or not isinstance(worker.get("error"), str) or worker["error"] not in ERROR_CODES:
            errors.append("invalid worker status")
    return errors
