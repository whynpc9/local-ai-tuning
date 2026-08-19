#!/usr/bin/env python3
"""Collect a read-only, credential-avoiding hardware/runtime inventory as JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


APPLE_PACKAGE_NAMES = (
    "mlx",
    "mlx-lm",
    "mlx-vlm",
    "omlx",
    "llama-cpp-python",
)

APPLE_CHIP_PATTERN = re.compile(r"^Apple M[1-9][0-9]*(?: (?:Pro|Max|Ultra))?$")


def run_command(argv: list[str], timeout: int = 10) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    if executable is None:
        return {"argv": argv, "available": False}
    try:
        completed = subprocess.run(
            [executable, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        result = {"argv": argv, "available": True, "timed_out": True, "timeout_s": timeout}
        if stdout.strip():
            result["stdout"] = stdout.strip()
        if stderr.strip():
            result["stderr"] = stderr.strip()
        return result
    except OSError as exc:
        return {"argv": argv, "available": True, "error": str(exc)}
    result: dict[str, Any] = {
        "argv": argv,
        "available": True,
        "returncode": completed.returncode,
    }
    if completed.stdout.strip():
        result["stdout"] = completed.stdout.strip()
    if completed.stderr.strip():
        result["stderr"] = completed.stderr.strip()
    return result


def linux_memory() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    wanted = {
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SwapTotal",
        "SwapFree",
    }
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        if key in wanted:
            number = rest.strip().split()[0]
            values[f"{key}_kib"] = int(number)
    return values


def sanitize_apple_system_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Allowlist non-identifying Apple hardware/software fields.

    `system_profiler -json` also returns serial numbers, platform UUIDs,
    provisioning UDIDs, computer/user names, and other host identifiers.  Never
    persist the raw document in an inventory.
    """

    hardware_allowed = {
        "_name",
        "machine_name",
        "machine_model",
        "chip_type",
        "physical_memory",
        "number_processors",
        "number_cores",
        "platform_CPU_htt",
    }
    display_allowed = {
        "_name",
        "sppci_model",
        "sppci_cores",
        "spdisplays_vendor",
        "spdisplays_metal",
        "spdisplays_device-id",
        "spdisplays_revision-id",
    }
    software_allowed = {
        "os_version",
        "kernel_version",
        "boot_mode",
        "secure_vm",
        "system_integrity",
    }

    def first_section(name: str) -> dict[str, Any]:
        section = raw.get(name)
        if not isinstance(section, list) or not section or not isinstance(section[0], dict):
            return {}
        return section[0]

    hardware = first_section("SPHardwareDataType")
    software = first_section("SPSoftwareDataType")
    displays = raw.get("SPDisplaysDataType")
    if not isinstance(displays, list):
        displays = []
    return {
        "hardware": {key: hardware[key] for key in hardware_allowed if key in hardware},
        "displays": [
            {key: display[key] for key in display_allowed if key in display}
            for display in displays
            if isinstance(display, dict)
        ],
        "software": {key: software[key] for key in software_allowed if key in software},
    }


def apple_system_profile() -> dict[str, Any]:
    argv = [
        "system_profiler",
        "SPHardwareDataType",
        "SPDisplaysDataType",
        "SPSoftwareDataType",
        "-json",
    ]
    result = run_command(argv, timeout=30)
    safe: dict[str, Any] = {
        "argv": argv,
        "available": result.get("available", False),
    }
    if "returncode" in result:
        safe["returncode"] = result["returncode"]
    if result.get("timed_out"):
        safe["timed_out"] = True
    if result.get("error"):
        safe["error"] = result["error"]
    if result.get("stderr"):
        safe["stderr"] = result["stderr"]
    stdout = result.get("stdout")
    if result.get("returncode") == 0 and isinstance(stdout, str):
        try:
            safe["profile"] = sanitize_apple_system_profile(json.loads(stdout))
        except json.JSONDecodeError as exc:
            safe["parse_error"] = str(exc)
    return safe


def apple_memory(vm_stat_result: dict[str, Any], pressure_result: dict[str, Any], swap_result: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    vm_text = str(vm_stat_result.get("stdout") or "")
    page_match = re.search(r"page size of (\d+) bytes", vm_text)
    if page_match:
        page_size = int(page_match.group(1))
        values["page_size_bytes"] = page_size
        keys = {
            "Pages free": "free_bytes",
            "Pages active": "active_bytes",
            "Pages inactive": "inactive_bytes",
            "Pages speculative": "speculative_bytes",
            "Pages wired down": "wired_bytes",
            "Pages stored in compressor": "stored_in_compressor_bytes",
            "Pages occupied by compressor": "compressor_bytes",
            "Compressions": "compressions",
            "Decompressions": "decompressions",
            "Pageins": "pageins",
            "Pageouts": "pageouts",
            "Swapins": "swapins",
            "Swapouts": "swapouts",
        }
        for line in vm_text.splitlines():
            label, separator, raw_value = line.partition(":")
            if not separator or label not in keys:
                continue
            match = re.search(r"(\d+)", raw_value.replace(".", ""))
            if not match:
                continue
            number = int(match.group(1))
            target = keys[label]
            values[target] = number * page_size if target.endswith("_bytes") else number
    pressure_text = str(pressure_result.get("stdout") or "")
    free_match = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure_text)
    if free_match:
        values["system_free_percent"] = int(free_match.group(1))
    swap_text = str(swap_result.get("stdout") or "")
    swap_match = re.search(
        r"total\s*=\s*([\d.]+)([KMG])\s+used\s*=\s*([\d.]+)([KMG])\s+free\s*=\s*([\d.]+)([KMG])",
        swap_text,
        re.IGNORECASE,
    )
    if swap_match:
        multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
        for field, value_index, unit_index in (
            ("swap_total_bytes", 1, 2),
            ("swap_used_bytes", 3, 4),
            ("swap_free_bytes", 5, 6),
        ):
            values[field] = int(float(swap_match.group(value_index)) * multipliers[swap_match.group(unit_index).upper()])
    return values


def capacity_bytes(value: Any) -> int | None:
    match = re.fullmatch(r"\s*([\d.]+)\s*(GB|TB)\s*", str(value or ""), re.IGNORECASE)
    if not match:
        return None
    multiplier = 1024**3 if match.group(2).upper() == "GB" else 1024**4
    return int(float(match.group(1)) * multiplier)


def metal_is_supported(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"spdisplays_supported", "supported"}


def successful_nonempty_probe(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("available") is True
        and value.get("returncode") == 0
        and str(value.get("stdout") or "").strip()
    )


def apple_power_mode_probe_ok(value: Any) -> bool:
    modes = value.get("power_modes") if isinstance(value, dict) else None
    return successful_nonempty_probe(value) and isinstance(modes, dict) and any(
        isinstance(profile, dict) and bool(profile) for profile in modes.values()
    )


def sanitized_apple_power_source() -> dict[str, Any]:
    """Capture active power state without persisting a battery identifier."""

    raw = run_command(["pmset", "-g", "batt"])
    safe = {key: raw[key] for key in ("argv", "available", "returncode", "timed_out", "timeout_s", "error", "stderr") if key in raw}
    stdout = str(raw.get("stdout") or "")
    source_match = re.search(r"Now drawing from ['\"]([^'\"]+)['\"]", stdout)
    battery_match = re.search(r"\t?(\d+)%\s*;\s*([^;\n]+)", stdout)
    parts = []
    if source_match:
        safe["source"] = source_match.group(1).strip()
        parts.append(f"source={safe['source']}")
    if battery_match:
        safe["battery_percent"] = int(battery_match.group(1))
        safe["battery_state"] = battery_match.group(2).strip()
        parts.extend((f"battery_percent={safe['battery_percent']}", f"battery_state={safe['battery_state']}"))
    if parts:
        safe["stdout"] = "; ".join(parts)
    elif raw.get("returncode") == 0:
        safe["parse_error"] = "pmset output did not match the allowlisted power-state grammar"
    return safe


def sanitized_apple_power_settings() -> dict[str, Any]:
    """Fingerprint the full pmset profile while exposing only power-mode keys."""

    raw = run_command(["pmset", "-g", "custom"])
    safe = {key: raw[key] for key in ("argv", "available", "returncode", "timed_out", "timeout_s", "error", "stderr") if key in raw}
    stdout = str(raw.get("stdout") or "")
    if raw.get("returncode") != 0 or not stdout.strip():
        return safe
    profiles: dict[str, dict[str, int | str]] = {}
    current_profile: str | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in {"AC Power", "Battery Power", "UPS Power"}:
            current_profile = stripped[:-1]
            profiles.setdefault(current_profile, {})
            continue
        if current_profile is None:
            continue
        match = re.fullmatch(r"(lowpowermode|powermode)\s+(.+)", stripped, re.IGNORECASE)
        if not match:
            continue
        raw_value = match.group(2).strip()
        profiles[current_profile][match.group(1).lower()] = int(raw_value) if raw_value.isdigit() else raw_value
    safe["profile_sha256"] = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
    safe["power_modes"] = profiles
    safe["stdout"] = json.dumps(profiles, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not any(values for values in profiles.values()):
        safe["parse_warning"] = "pmset profile was fingerprinted but no lowpowermode/powermode key was exposed"
    return safe


def apple_identity_checks(
    profile: dict[str, Any],
    probes: dict[str, Any],
    *,
    is_darwin: bool,
    machine: str,
) -> dict[str, bool]:
    """Fail closed on the identity tuple used for Apple performance comparisons.

    `sysctl.proc_translated` can be unavailable in a sandbox.  Native arm64 is
    therefore authoritative, but an explicit translated=1 result is always a
    rejection.
    """

    hardware = profile.get("hardware") if isinstance(profile, dict) else {}
    displays = profile.get("displays") if isinstance(profile, dict) else []
    software = profile.get("software") if isinstance(profile, dict) else {}
    if not isinstance(hardware, dict):
        hardware = {}
    if not isinstance(displays, list):
        displays = []
    if not isinstance(software, dict):
        software = {}
    displays = [display for display in displays if isinstance(display, dict)]

    chip = str(hardware.get("chip_type") or "").strip()
    rosetta_probe = probes.get("apple_rosetta_translation") or {}
    rosetta_value = str(rosetta_probe.get("stdout") or "").strip()
    memory_bytes = capacity_bytes(hardware.get("physical_memory"))
    matching_displays = [
        display
        for display in displays
        if str(display.get("sppci_model") or "").strip() == chip
    ]

    def positive_gpu_cores(display: dict[str, Any]) -> bool:
        try:
            return int(str(display.get("sppci_cores") or "").strip()) > 0
        except ValueError:
            return False

    return {
        "darwin_host": is_darwin,
        "native_arm64_abi": machine.lower() == "arm64",
        "rosetta_not_explicitly_active": rosetta_value != "1",
        "exact_apple_chip": bool(APPLE_CHIP_PATTERN.fullmatch(chip)),
        "chassis_identity": bool(hardware.get("machine_model") and hardware.get("machine_name")),
        "physical_memory_sane": memory_bytes is not None and memory_bytes >= 8 * 1024**3,
        "gpu_chip_matches": bool(matching_displays),
        "gpu_cores_reported": any(positive_gpu_cores(display) for display in matching_displays),
        "metal_supported": any(metal_is_supported(display.get("spdisplays_metal")) for display in matching_displays),
        "os_identity": bool(software.get("os_version")),
        "power_settings_captured": apple_power_mode_probe_ok(probes.get("apple_power_settings")),
        "power_source_captured": successful_nonempty_probe(probes.get("apple_power_source")),
    }


def filesystem_probe_argv(path: Path, system: str) -> list[str]:
    """Return a filesystem-capacity probe supported by the host OS."""

    if system == "Darwin":
        return ["df", "-kP", str(path)]
    return ["df", "-B1", "-T", str(path)]


def installed_python_packages() -> dict[str, str]:
    versions = {}
    for name in APPLE_PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def inventory(container_images: list[str] | None = None, storage_paths: list[str] | None = None) -> dict[str, Any]:
    system = platform.system()
    is_darwin = system == "Darwin"
    commands = {
        "os_release": ["cat", "/etc/os-release"],
        "lscpu": ["lscpu", "--json"],
        "nvidia_gpu_query": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total,driver_version,pstate,power.limit",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_topology": ["nvidia-smi", "topo", "-m"],
        "nvidia_driver": ["nvidia-smi"],
        "nvidia_power_details": ["nvidia-smi", "-q", "-d", "POWER,CLOCK,PERFORMANCE"],
        "nvpmodel": ["nvpmodel", "-q", "--verbose"],
        "cuda_compiler": ["nvcc", "--version"],
        "container_toolkit": ["nvidia-ctk", "--version"],
        "docker": ["docker", "version", "--format", "{{json .}}"],
        "rdma_links": ["rdma", "link", "show"],
        "ib_devices": ["ibdev2netdev"],
        "network_links": ["ip", "-details", "link", "show"],
        "tegrastats_probe": ["tegrastats", "--interval", "1000"],
        "vmstat": ["vmstat", "1", "2"],
        "block_devices": ["lsblk", "--json", "--bytes", "--output", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL"],
        "workspace_filesystem": ["df", "-kP", "."] if is_darwin else ["df", "-B1", "-T", "."],
        "systemd_default_target": ["systemctl", "get-default"],
        "login_sessions": ["loginctl", "list-sessions", "--no-legend"],
    }
    results = {
        name: run_command(argv, timeout=2 if name == "tegrastats_probe" else 10)
        for name, argv in commands.items()
    }
    apple_profile = apple_system_profile() if is_darwin else {"available": False}
    if is_darwin:
        results.update(
            {
                "apple_os_version": run_command(["sw_vers"]),
                "apple_rosetta_translation": run_command(["sysctl", "-in", "sysctl.proc_translated"]),
                "apple_hw_memsize": run_command(["sysctl", "-n", "hw.memsize"]),
                "apple_hw_pagesize": run_command(["sysctl", "-n", "hw.pagesize"]),
                "apple_hw_ncpu": run_command(["sysctl", "-n", "hw.ncpu"]),
                "apple_hw_physicalcpu": run_command(["sysctl", "-n", "hw.physicalcpu"]),
                "apple_hw_logicalcpu": run_command(["sysctl", "-n", "hw.logicalcpu"]),
                "apple_hw_nperflevels": run_command(["sysctl", "-n", "hw.nperflevels"]),
                "apple_vm_stat": run_command(["vm_stat"]),
                "apple_memory_pressure": run_command(["memory_pressure", "-Q"]),
                "apple_swap_usage": run_command(["sysctl", "vm.swapusage"]),
                "apple_power_settings": sanitized_apple_power_settings(),
                "apple_power_source": sanitized_apple_power_source(),
                "omlx_version": run_command(["omlx", "--version"]),
                "brew_omlx_version": run_command(["brew", "list", "--versions", "omlx"]),
            }
        )
    image_results = {
        image: run_command(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                '{"Id":{{json .Id}},"RepoDigests":{{json .RepoDigests}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"Created":{{json .Created}}}',
            ]
        )
        for image in (container_images or [])
    }
    storage_results = {}
    for raw_path in storage_paths or []:
        path = Path(raw_path).expanduser()
        if not path.exists():
            storage_results[raw_path] = {"path": raw_path, "exists": False}
        else:
            filesystem_argv = filesystem_probe_argv(path, system)
            storage_results[raw_path] = {
                "path": str(path.resolve()),
                "exists": True,
                "filesystem_unit_bytes": 1024 if is_darwin else 1,
                "filesystem": run_command(filesystem_argv),
            }
    gpu_probe = results["nvidia_gpu_query"]
    gpu_probe_ok = bool(
        gpu_probe.get("available")
        and gpu_probe.get("returncode") == 0
        and gpu_probe.get("stdout")
    )
    identity_checks = apple_identity_checks(
        apple_profile.get("profile") or {},
        results,
        is_darwin=is_darwin,
        machine=platform.machine(),
    )
    apple_probe_ok = all(identity_checks.values())
    memory = (
        apple_memory(
            results.get("apple_vm_stat") or {},
            results.get("apple_memory_pressure") or {},
            results.get("apple_swap_usage") or {},
        )
        if is_darwin
        else linux_memory()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "kernel": platform.release(),
            "cpu_count": os.cpu_count(),
        },
        "memory": memory,
        "apple_silicon": {
            "system_profile": apple_profile,
            "python_packages": installed_python_packages() if is_darwin else {},
            "powermetrics_available": bool(shutil.which("powermetrics")) if is_darwin else False,
        },
        "selected_environment": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "DISPLAY",
                "WAYLAND_DISPLAY",
                "XDG_SESSION_TYPE",
                "MLX_METAL_FAST_SYNCH",
                "MLX_METAL_DEBUG",
                "OMLX_MTP_ROWWISE_BATCH",
                "OMLX_WITH_CUSTOM_KERNEL",
            )
            if key in os.environ
        },
        "probes": results,
        "container_images": image_results,
        "storage_paths": storage_results,
        "validation": {
            "nvidia_gpu_probe_ok": gpu_probe_ok,
            "apple_silicon_probe_ok": apple_probe_ok,
            "apple_identity_checks": identity_checks,
            "all_requested_images_inspected": all(
                result.get("returncode") == 0 for result in image_results.values()
            ),
            "all_requested_storage_inspected": all(
                result.get("exists") is True
                and (result.get("filesystem") or {}).get("returncode") == 0
                for result in storage_results.values()
            ),
        },
        "notes": [
            "No credential-like environment variables are collected.",
            "Container inspection intentionally records only image ID, RepoDigests, architecture, OS, and creation time; Config.Env and labels are excluded because they may contain secrets.",
            "On macOS, system_profiler output is parsed through a strict allowlist; serial numbers, UUIDs, provisioning IDs, computer/user names, and the raw profile are never stored.",
            "The active macOS power source is reduced to source, charge percentage, and charge state; the raw pmset battery identifier is not persisted.",
            "The full pmset custom profile is retained only as a SHA-256 fingerprint; lowpowermode/powermode values are allowlisted for comparison, and unrelated settings are not persisted.",
            "Apple sysctl probes query a small explicit allowlist. The collector never runs sysctl -a, ioreg, network profiling, process command listings, or power-profile system_profiler payloads.",
            "This inventory is not anonymized: hostname, GPU UUID/PCI data, topology, and network identifiers may be present. Review or sanitize it before sharing outside the target environment.",
            "On UMA systems, nvidia-smi memory fields may be unsupported; use memory, vmstat, tegrastats, and framework logs together.",
            "Run this on every node. Pass every candidate image with --container-image to capture its local architecture, ID, and RepoDigests.",
            "The two-second tegrastats timeout is intentional: partial stdout is the sample, not a probe failure.",
            "powermetrics is never started by this collector because it commonly requires elevated privileges; run an explicitly authorized sampler beside the benchmark when energy or frequency data is needed.",
            "memory_pressure -Q is retained as a best-effort diagnostic because its availability/semantics vary; hard Apple admission uses vm_stat/swap deltas, process and Metal/MLX footprint, and thermal evidence.",
            "Requested storage capacity uses POSIX df -kP (1 KiB blocks) on macOS and df -B1 -T (bytes plus filesystem type) on Linux; a failed requested probe fails validation.",
        ],
    }


def write_json(value: dict[str, Any], output: str, pretty: bool) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True)
    if output == "-":
        print(text)
        return
    Path(output).write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--container-image", action="append", default=[], help="locally present image to inspect; repeatable")
    parser.add_argument("--storage-path", action="append", default=[], help="model/cache path whose filesystem capacity should be recorded")
    parser.add_argument("--require-nvidia", action="store_true", help="exit nonzero if the NVIDIA GPU identity probe fails")
    parser.add_argument("--require-apple-silicon", action="store_true", help="exit nonzero unless a native ARM64 Apple Silicon + Metal profile is validated")
    args = parser.parse_args()
    result = inventory(args.container_image, args.storage_path)
    write_json(result, args.output, args.pretty)
    if args.require_nvidia and not result["validation"]["nvidia_gpu_probe_ok"]:
        print("NVIDIA GPU identity probe failed; inventory was saved for diagnosis", file=sys.stderr)
        return 2
    if args.require_apple_silicon and not result["validation"]["apple_silicon_probe_ok"]:
        print("native Apple Silicon / Metal identity probe failed; inventory was saved for diagnosis", file=sys.stderr)
        return 4
    if args.container_image and not result["validation"]["all_requested_images_inspected"]:
        print("one or more requested local images could not be inspected", file=sys.stderr)
        return 3
    if args.storage_path and not result["validation"]["all_requested_storage_inspected"]:
        print("one or more requested storage paths could not be inspected", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
