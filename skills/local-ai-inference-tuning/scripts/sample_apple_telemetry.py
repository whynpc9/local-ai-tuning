#!/usr/bin/env python3
"""Sample privacy-bounded Apple memory, thermal, and optional process telemetry as JSONL."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime as dt
import json
import math
import platform
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, TextIO

from collect_hardware import (
    apple_memory,
    run_command,
    sanitized_apple_power_settings,
    sanitized_apple_power_source,
)


SCHEMA_VERSION = "1.0"
THERMAL_STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def probe_succeeded(result: Any, *, require_stdout: bool = True) -> bool:
    return bool(
        isinstance(result, dict)
        and result.get("available") is True
        and result.get("returncode") == 0
        and (not require_stdout or str(result.get("stdout") or "").strip())
    )


def process_info_thermal_state() -> dict[str, Any]:
    """Read NSProcessInfo.thermalState without PyObjC or a process listing."""

    try:
        foundation_path = ctypes.util.find_library("Foundation")
        objc_path = ctypes.util.find_library("objc")
        if not foundation_path or not objc_path:
            return {"available": False, "error": "Foundation or Objective-C runtime was not found"}
        ctypes.CDLL(foundation_path)
        objc = ctypes.CDLL(objc_path)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        send_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        process_info_class = objc.objc_getClass(b"NSProcessInfo")
        process_info = send_object(process_info_class, objc.sel_registerName(b"processInfo"))
        raw_state = int(send_integer(process_info, objc.sel_registerName(b"thermalState")))
        state = THERMAL_STATES.get(raw_state)
        if state is None:
            return {"available": False, "raw_state": raw_state, "error": "unknown NSProcessInfo thermal state"}
        return {"available": True, "raw_state": raw_state, "state": state}
    except (OSError, TypeError, ValueError) as exc:
        return {"available": False, "error": str(exc)}


def process_probe(pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {"requested": False}
    # The explicit stats list excludes process names and command lines.
    result = run_command(
        [
            "top",
            "-l",
            "1",
            "-pid",
            str(pid),
            "-stats",
            "pid,cpu,mem,rprvt,purg,vsize,threads,state,time,pageins",
        ],
        timeout=10,
    )
    result["pid_observed"] = bool(
        re.search(rf"(?m)^\s*{re.escape(str(pid))}\s+", str(result.get("stdout") or ""))
    )
    result["privacy_note"] = "top stats intentionally exclude command/name/path columns"
    return result


def capture_sample(run_id: str, sequence: int, pid: int | None) -> dict[str, Any]:
    vm_stat_result = run_command(["vm_stat"])
    pressure_result = run_command(["memory_pressure", "-Q"])
    swap_result = run_command(["sysctl", "vm.swapusage"])
    thermal_limits = run_command(["pmset", "-g", "therm"])
    thermal_state = process_info_thermal_state()
    power_state = sanitized_apple_power_source()
    power_profile = sanitized_apple_power_settings()
    process = process_probe(pid)
    memory = apple_memory(vm_stat_result, pressure_result, swap_result)
    # A sandbox may deny `sysctl vm.swapusage` while vm_stat still exposes the
    # cumulative swap-in/out counters required for a window delta.
    swap_counter_source_ok = probe_succeeded(swap_result) or all(
        field in memory for field in ("swapins", "swapouts")
    )
    required_health = {
        "vm_stat": probe_succeeded(vm_stat_result),
        "memory_pressure": probe_succeeded(pressure_result),
        "swap_counters": swap_counter_source_ok,
        "thermal_state": thermal_state.get("available") is True,
        "power_state": probe_succeeded(power_state),
        "power_profile": probe_succeeded(power_profile)
        and isinstance(power_profile.get("power_modes"), dict)
        and any(bool(mode) for mode in power_profile["power_modes"].values()),
        "process": pid is None or (probe_succeeded(process) and process.get("pid_observed") is True),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sample",
        "run_id": run_id,
        "sequence": sequence,
        "wall_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "memory": memory,
        "thermal_limits": thermal_limits,
        "thermal_state": thermal_state,
        "power_state": power_state,
        "power_profile": power_profile,
        "process": process,
        "probe_health": {
            "ok": all(required_health.values()),
            "required": required_health,
            "failed": sorted(name for name, ok in required_health.items() if not ok),
            "diagnostic_thermal_limits_ok": probe_succeeded(thermal_limits),
            "diagnostic_swap_usage_ok": probe_succeeded(swap_result),
        },
    }


def numeric_delta(first: dict[str, Any], last: dict[str, Any]) -> dict[str, int | float]:
    deltas: dict[str, int | float] = {}
    for key, value in first.items():
        other = last.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(other, (int, float)) and not isinstance(other, bool):
            deltas[key] = other - value
    return deltas


def memory_admission(memory_delta: dict[str, Any]) -> dict[str, Any]:
    required_zero_counters = ("swapins", "swapouts", "pageouts")
    missing = [field for field in required_zero_counters if not isinstance(memory_delta.get(field), (int, float))]
    nonzero = {
        field: memory_delta.get(field)
        for field in required_zero_counters
        if isinstance(memory_delta.get(field), (int, float)) and memory_delta.get(field) != 0
    }
    swap_used_growth = memory_delta.get("swap_used_bytes")
    swap_used_grew = isinstance(swap_used_growth, (int, float)) and swap_used_growth > 0
    return {
        "passed": not missing and not nonzero and not swap_used_grew,
        "missing_required_counters": missing,
        "nonzero_required_counters": nonzero,
        "swap_used_bytes_grew": swap_used_grew,
        "pageins_delta_diagnostic": memory_delta.get("pageins"),
        "compression_churn_diagnostic": {
            "compressions": memory_delta.get("compressions"),
            "decompressions": memory_delta.get("decompressions"),
            "compressor_bytes": memory_delta.get("compressor_bytes"),
        },
    }


def emit(document: dict[str, Any], stream: TextIO) -> None:
    stream.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def validate_timing(duration_s: float, interval_s: float) -> None:
    if not math.isfinite(duration_s) or not math.isfinite(interval_s):
        raise ValueError("duration and interval must be finite")
    if not 0.2 <= interval_s <= 60:
        raise ValueError("interval must be between 0.2 and 60 seconds")
    if duration_s < interval_s:
        raise ValueError("duration must be at least one interval so the window has two samples")


def run_sampling(run_id: str, duration_s: float, interval_s: float, pid: int | None, stream: TextIO) -> bool:
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    sequence = 0
    while True:
        sample = capture_sample(run_id, sequence, pid)
        samples.append(sample)
        emit(sample, stream)
        elapsed = time.monotonic() - started
        if elapsed >= duration_s:
            break
        sequence += 1
        time.sleep(min(interval_s, max(0.0, duration_s - elapsed)))
    first_memory = samples[0].get("memory") or {}
    last_memory = samples[-1].get("memory") or {}
    memory_delta = numeric_delta(first_memory, last_memory)
    memory_gate = memory_admission(memory_delta)
    probe_failures = sorted(
        {
            failure
            for sample in samples
            for failure in (sample.get("probe_health") or {}).get("failed", [])
        }
    )
    thermal_states = sorted(
        {
            str((sample.get("thermal_state") or {}).get("state"))
            for sample in samples
            if (sample.get("thermal_state") or {}).get("state")
        }
    )
    power_sources = sorted(
        {
            str((sample.get("power_state") or {}).get("source"))
            for sample in samples
            if (sample.get("power_state") or {}).get("source")
        }
    )
    power_profile_hashes = sorted(
        {
            str((sample.get("power_profile") or {}).get("profile_sha256"))
            for sample in samples
            if (sample.get("power_profile") or {}).get("profile_sha256")
        }
    )
    power_modes = [
        (sample.get("power_profile") or {}).get("power_modes") or {}
        for sample in samples
    ]
    serious_or_critical = bool({"serious", "critical"}.intersection(thermal_states))
    stable_power_source = len(power_sources) == 1
    stable_power_profile = len(power_profile_hashes) == 1
    valid_window = (
        not probe_failures
        and len(samples) >= 2
        and not serious_or_critical
        and stable_power_source
        and stable_power_profile
        and memory_gate["passed"]
    )
    summary = {
            "schema_version": SCHEMA_VERSION,
            "kind": "summary",
            "run_id": run_id,
            "sample_count": len(samples),
            "duration_s": (samples[-1]["monotonic_ns"] - samples[0]["monotonic_ns"]) / 1_000_000_000,
            "memory_delta": memory_delta,
            "performance_admission_fields": {
                "swapouts_delta": memory_delta.get("swapouts"),
                "swapins_delta": memory_delta.get("swapins"),
                "pageouts_delta": memory_delta.get("pageouts"),
                "pageins_delta": memory_delta.get("pageins"),
                "compressor_bytes_delta": memory_delta.get("compressor_bytes"),
                "compressions_delta": memory_delta.get("compressions"),
                "decompressions_delta": memory_delta.get("decompressions"),
                "swap_used_bytes_delta": memory_delta.get("swap_used_bytes"),
                "thermal_states": thermal_states,
                "serious_or_critical_thermal_observed": serious_or_critical,
                "power_sources": power_sources,
                "stable_power_source": stable_power_source,
                "power_profile_sha256s": power_profile_hashes,
                "power_modes": power_modes[0] if power_modes and all(mode == power_modes[0] for mode in power_modes) else power_modes,
                "stable_power_profile": stable_power_profile,
            },
            "window_health": {
                "valid": valid_window,
                "probe_failures": probe_failures,
                "minimum_sample_count_met": len(samples) >= 2,
                "thermal_gate_passed": not serious_or_critical and bool(thermal_states),
                "power_source_gate_passed": stable_power_source,
                "power_profile_gate_passed": stable_power_profile,
                "memory_gate": memory_gate,
            },
            "notes": [
                "A positive sustained swap/pageout/compressor signal requires inspection and normally disqualifies a maximum-tok/s trial.",
                "NSProcessInfo thermalState is the serious/critical gate; pmset thermal limits remain best-effort diagnostics.",
                "The active source and SHA-256 of the full pmset custom profile must remain stable; allowlisted lowpowermode/powermode values are reported explicitly.",
                "A formal performance window requires zero swapins, swapouts, and pageouts and no growth in swap-used bytes when that diagnostic is available; pageins/compression remain explicit diagnostics for workload-calibrated churn gates.",
                "This sampler does not invoke powermetrics, system_profiler, ioreg, network tools, or process command listings.",
            ],
        }
    emit(summary, stream)
    return valid_window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=str(uuid.uuid4()), help="shared benchmark run identifier")
    parser.add_argument("--duration", type=float, default=60.0, help="sampling duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples (0.2 to 60)")
    parser.add_argument("--pid", type=int, help="optional inference-server PID; command/name/path are not collected")
    parser.add_argument("--output", default="-", help="JSONL path, or - for stdout")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        parser.error("Apple telemetry requires native arm64 macOS; Rosetta/x86_64 is not accepted")
    try:
        validate_timing(args.duration, args.interval)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pid is not None and args.pid <= 0:
        parser.error("--pid must be a positive process id")
    if args.output == "-":
        valid_window = run_sampling(args.run_id, args.duration, args.interval, args.pid, sys.stdout)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            valid_window = run_sampling(args.run_id, args.duration, args.interval, args.pid, stream)
    if not valid_window:
        print("Apple telemetry window failed probe, sample-count, thermal, or power-state admission; inspect the JSONL summary", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
