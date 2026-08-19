from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import types
import unittest
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


benchmark = load_script("benchmark_openai")
planner = load_script("plan_experiments")
collector = load_script("collect_hardware")
comparator = load_script("compare_runs")
apple_sampler = load_script("sample_apple_telemetry")


class ScriptUnitTests(unittest.TestCase):
    def test_percentile_and_distribution(self):
        self.assertEqual(benchmark.percentile([1, 2, 3], 0.5), 2)
        values = benchmark.distribution([0.01, 0.02, 0.03], scale=1000)
        self.assertEqual(values["p50"], 20)
        self.assertEqual(values["count"], 3)

    def test_summary_keeps_correctness_out_of_goodput(self):
        base = {
            "success": True,
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "ttft_s": 0.1,
            "e2e_s": 1.0,
            "tpot_s": 0.1,
            "ttfo_s": 0.2,
            "completion_decode_tok_per_s": 10.0,
            "visible_answer_tok_per_s": 8.0,
            "visible_answer_tokens": 8,
            "finish_reason_valid": True,
        }
        result = benchmark.summarize(
            [{**base, "output_valid": True}, {**base, "output_valid": False}],
            wall_s=2.0,
            slos={"ttft_ms": 200, "tpot_ms": 200, "e2e_ms": 2000},
        )
        self.assertEqual(result["correctness_failed"], 1)
        self.assertEqual(result["goodput_completed"], 1)
        self.assertEqual(result["aggregate_completion_tok_per_s_including_reasoning"], 10)
        self.assertEqual(result["aggregate_visible_answer_tok_per_s"], 8)

    def test_secret_redaction_is_recursive(self):
        value = benchmark.redact(
            {
                "api_key": "x",
                "hf_token": "hf-secret",
                "github_token": "gh-secret",
                "tokenizer_revision": "abc123",
                "nested": {"password": "y", "num_speculative_tokens": 3, "safe": 1},
            }
        )
        self.assertEqual(value["api_key"], "<redacted>")
        self.assertEqual(value["hf_token"], "<redacted>")
        self.assertEqual(value["github_token"], "<redacted>")
        self.assertEqual(value["tokenizer_revision"], "abc123")
        self.assertEqual(value["nested"]["password"], "<redacted>")
        self.assertEqual(value["nested"]["num_speculative_tokens"], 3)
        self.assertEqual(value["nested"]["safe"], 1)

    def test_structured_comparison_contract(self):
        contract = {
            "base_model_id": "Qwen/Qwen3.8-27B",
            "artifact_id": "Qwen/Qwen3.8-27B",
            "artifact_revision": "abc",
            "tokenizer_revision": "def",
            "transformation": "none",
            "quality_contract_id": "sealed-v1",
            "advertised_context_tokens": 262144,
            "hardware_inventory_sha256s": ["a" * 64],
            "power_profile": "nvpmodel-maxn",
            "client_id": "same-host-stdlib-client-v1",
            "cache_state": "warm-engine-cold-prefix",
            "comparison_mode": "common-denominator",
            "allowed_candidate_differences": [],
        }
        parsed, digest = benchmark.load_comparison_contract(json.dumps(contract))
        self.assertEqual(parsed, contract)
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(ValueError, "missing non-empty fields"):
            benchmark.load_comparison_contract('{"artifact_id":"x"}')

    def test_structured_quality_gate_is_bound_to_exact_configuration(self):
        contract = {
            "artifact_id": "Qwen/Qwen3.8-27B",
            "artifact_revision": "artifact-rev",
            "quality_contract_id": "sealed-v1",
        }
        metadata = {
            "image_digest": "sha256:image",
            "framework_revision": "framework-rev",
            "server_config_sha256": "b" * 64,
        }
        gate = {
            "schema_version": "1.0",
            "run_id": "quality-run-1",
            "status": "passed",
            "comparison_contract_sha256": "a" * 64,
            "quality_contract_id": "sealed-v1",
            "artifact_id": "Qwen/Qwen3.8-27B",
            "artifact_revision": "artifact-rev",
            "image_digest": "sha256:image",
            "framework_revision": "framework-rev",
            "server_config_sha256": "b" * 64,
            "suite_manifest_sha256": "c" * 64,
            "results_sha256": "d" * 64,
            "gates": {"strict-json": "passed", "long-context": "passed"},
        }
        parsed, digest = benchmark.load_quality_gate(json.dumps(gate), contract, "a" * 64, metadata)
        self.assertEqual(parsed, gate)
        self.assertEqual(len(digest), 64)
        gate["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "status passed"):
            benchmark.load_quality_gate(json.dumps(gate), contract, "a" * 64, metadata)

    def test_workload_fingerprint_excludes_candidate_quality_identity(self):
        args = types.SimpleNamespace(
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            reasoning_effort=None,
            seed=42,
            concurrency=1,
            requests=7,
            warmups=1,
            unique_prefix=True,
            objective="single-stream",
            workload_case_id="isl-2048-osl-512",
            ttft_slo_ms=None,
            tpot_slo_ms=None,
            e2e_slo_ms=None,
        )
        payload = benchmark.build_workload_fingerprint_payload(
            args,
            [{"id": "p", "messages": [{"role": "user", "content": "x"}]}],
            {},
            "contract",
            "plan",
        )
        self.assertNotIn("quality_gate_sha256", payload)
        self.assertNotIn("candidate_id", payload)
        self.assertNotIn("server_config_sha256", payload)

    def test_result_writer_creates_requested_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "result.json"
            written = benchmark.write_result({"ok": True}, str(target))
            self.assertEqual(written, target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"ok": True})

    def test_comparator_requires_repeats_and_variance_separation(self):
        paths = []
        with tempfile.TemporaryDirectory() as directory:
            order_index = 0
            for label, values in (("fast", [10.0, 10.1, 9.9]), ("slow", [8.0, 8.1, 7.9])):
                for index, value in enumerate(values):
                    document = {
                        "schema_version": "1.1",
                        "run_id": f"{label}-{index}",
                        "objective": "single-stream",
                        "workload_fingerprint": "same-workload",
                        "comparison_contract_sha256": "same-contract",
                        "metadata": {
                            "label": label,
                            "order_design": "randomized",
                            "experiment_id": "experiment-1",
                            "trial_id": f"trial-{order_index}",
                            "order_index": order_index,
                            "image_digest": f"sha256:{label}",
                            "framework_revision": label,
                            "server_config_sha256": label,
                            "quality_gate_run_id": "quality-1",
                            "quality_gate_sha256": f"quality-{label}",
                            "served_model_name": "served-model",
                            "candidate_id": label,
                        },
                        "summary": {
                            "single_request_completion_decode_tok_per_s_including_reasoning": {"p50": value},
                            "failed": 0,
                            "correctness_failed": 0,
                            "token_accounted": 1,
                            "completed": 1,
                        },
                    }
                    path = Path(directory) / f"{label}-{index}.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    paths.append(str(path))
                    order_index += 1
            output = io.StringIO()
            with mock.patch("sys.argv", ["compare_runs.py", *paths]), redirect_stdout(output):
                self.assertEqual(comparator.main(), 0)
            self.assertIn("measured winner for this contract: fast", output.getvalue())

    def test_comparator_rejects_reused_run_file(self):
        document = {
            "schema_version": "1.1",
            "run_id": "same-run",
            "objective": "single-stream",
            "summary": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            error = io.StringIO()
            with mock.patch("sys.argv", ["compare_runs.py", str(path), str(path)]), redirect_stderr(error):
                self.assertEqual(comparator.main(), 2)
            self.assertIn("repeated files are not independent trials", error.getvalue())

    def test_planner_includes_specialized_and_magnitude_lanes(self):
        plan = planner.build_plan("qwen3.8-27b", "dgx-spark-1", "single-stream")
        ids = {candidate["id"] for candidate in plan["candidates"]}
        self.assertIn("qwen-sglang-sm121", ids)
        self.assertIn("qwen-sglang-dflash2-sm121", ids)
        self.assertIn("qwen-vllm-gb10-challenger", ids)
        self.assertIn("qwen-magnitude-gguf-control", ids)
        self.assertTrue(all("artifact_class" in candidate for candidate in plan["candidates"]))

    def test_apple_planner_has_same_artifact_control_and_portable_lane(self):
        plan = planner.build_plan("qwen3.8-27b", "apple-silicon-1", "single-stream")
        by_id = {candidate["id"]: candidate for candidate in plan["candidates"]}
        self.assertIn("qwen-omlx-oq4e-target-only-control", by_id)
        self.assertIn("qwen-omlx-oq4e-native-mtp", by_id)
        self.assertIn("qwen-mlx-dflash2", by_id)
        self.assertIn("qwen-magnitude-llamacpp-metal-control", by_id)
        self.assertEqual(by_id["qwen-omlx-oq4e-target-only-control"]["baseline"]["speculative"], "off")
        self.assertEqual(
            by_id["qwen-omlx-oq4e-target-only-control"]["artifact"],
            "Jundot/Qwen3.8-27B-oQ4e-mtp pinned by immutable revision",
        )
        self.assertIn("same exact oMLX artifact", by_id["qwen-omlx-oq4e-native-mtp"]["comparison_group"])
        self.assertIn("dedf8df68adfb1afeaf7b7480c0a0243108177b4", by_id["qwen-mlx-dflash2"]["artifact"])
        self.assertEqual(by_id["qwen-mlx-dflash2"]["sweeps"]["block_size"], [2, 3, 4, 5])
        self.assertIn("thermal-stability", plan["platform_policy"]["execution_lanes"])
        self.assertEqual(plan["platform_policy"]["warmup_gate"]["minimum_attempts"], 5)
        self.assertIn("observed_depth_histogram", plan["platform_policy"]["speculation_fields"])

    def test_rtx_planner_includes_dflash2_challenger(self):
        plan = planner.build_plan("qwen3.8-27b", "rtx-pro-6000-1", "single-stream")
        by_id = {candidate["id"]: candidate for candidate in plan["candidates"]}
        self.assertIn("qwen-sglang-dflash2-sm120", by_id)
        self.assertIn("c14312a66420b75ca9a11bf1817c4db1fa26b097", by_id["qwen-sglang-dflash2-sm120"]["stack"])

    def test_planner_labels_default_objective_and_accepts_canonical_id(self):
        plan = planner.build_plan("deepseek-ai/DeepSeek-V4-Flash-0731", "dgx-spark-1")
        self.assertEqual(plan["objective"], "single-stream")
        self.assertEqual(plan["objective_source"], "assumed-default")
        self.assertTrue(plan["assumptions"][0]["active"])

    def test_planner_rejects_invalid_objective(self):
        with self.assertRaisesRegex(ValueError, "unknown objective"):
            planner.build_plan("qwen3.8-27b", "dgx-spark-1", "singlestream")

    def test_all_planner_profiles_pass_candidate_schema(self):
        for model in ("qwen3.8-27b", "deepseek-v4-flash-0731"):
            for hardware in planner.HARDWARE_PROFILES:
                for objective in planner.OBJECTIVES:
                    plan = planner.build_plan(model, hardware, objective)
                    self.assertTrue(plan["candidates"])
                    self.assertTrue(all(candidate["evidence_tier"] in {"A", "B", "C", "D"} for candidate in plan["candidates"]))

    def test_hardware_profile_rejects_unvalidated_or_wrong_host(self):
        invalid = {"validation": {"nvidia_gpu_probe_ok": False}, "host": {"machine": "arm64"}, "probes": {}}
        with self.assertRaisesRegex(ValueError, "validated NVIDIA"):
            planner.validate_hardware_inventories("dgx-spark-1", [invalid])
        wrong_host = {
            "validation": {"nvidia_gpu_probe_ok": True},
            "host": {"machine": "x86_64"},
            "probes": {"nvidia_gpu_query": {"stdout": "0, NVIDIA GB10, 12.1"}},
        }
        with self.assertRaisesRegex(ValueError, "ARM64"):
            planner.validate_hardware_inventories("dgx-spark-1", [wrong_host])

    def test_apple_profile_requires_native_arm64_chip_memory_and_metal(self):
        valid = {
            "validation": {"apple_silicon_probe_ok": True, "nvidia_gpu_probe_ok": False},
            "host": {"machine": "arm64"},
            "probes": {
                "apple_rosetta_translation": {"available": True, "returncode": 0, "stdout": "0"},
                "apple_power_settings": {
                    "available": True,
                    "returncode": 0,
                    "stdout": "{\"AC Power\":{\"lowpowermode\":0}}",
                    "power_modes": {"AC Power": {"lowpowermode": 0}},
                },
                "apple_power_source": {"available": True, "returncode": 0, "stdout": "Now drawing from AC Power"},
            },
            "apple_silicon": {
                "system_profile": {
                    "profile": {
                        "hardware": {
                            "chip_type": "Apple M4 Max",
                            "physical_memory": "128 GB",
                            "machine_model": "Mac16,5",
                            "machine_name": "MacBook Pro",
                        },
                        "displays": [{
                            "sppci_model": "Apple M4 Max",
                            "sppci_cores": "40",
                            "spdisplays_metal": "spdisplays_supported",
                        }],
                        "software": {"os_version": "macOS 26.5.2 (25F84)"},
                    }
                }
            },
        }
        planner.validate_hardware_inventories("apple-silicon-1", [valid])
        deepseek_plan = planner.build_plan("deepseek-v4-flash-0731", "apple-silicon-1", "single-stream")
        planner.apply_inventory_admission(deepseek_plan, "apple-silicon-1", [valid], False)
        self.assertEqual(deepseek_plan["artifact_boundary"]["official_checkpoint"]["status"], "rejected")
        self.assertEqual(deepseek_plan["phase_scope"], "stopped at official capacity gate; alternatives listed but blocked")
        self.assertTrue(
            next(candidate for candidate in deepseek_plan["candidates"] if candidate["id"] == "deepseek-apple-official-admission")["rejected"]
        )
        with self.assertRaisesRegex(ValueError, "native ARM64"):
            planner.validate_hardware_inventories(
                "apple-silicon-1",
                [{**valid, "validation": {"apple_silicon_probe_ok": False}}],
            )
        with self.assertRaisesRegex(ValueError, "native arm64 ABI"):
            planner.validate_hardware_inventories(
                "apple-silicon-1",
                [{**valid, "host": {"machine": "x86_64"}}],
            )
        translated = {**valid, "probes": {**valid["probes"], "apple_rosetta_translation": {"stdout": "1"}}}
        with self.assertRaisesRegex(ValueError, "Rosetta"):
            planner.validate_hardware_inventories("apple-silicon-1", [translated])
        mystery = json.loads(json.dumps(valid))
        mystery["apple_silicon"]["system_profile"]["profile"]["hardware"]["chip_type"] = "Apple Mystery"
        with self.assertRaisesRegex(ValueError, "exact Apple chip"):
            planner.validate_hardware_inventories("apple-silicon-1", [mystery])
        tiny = json.loads(json.dumps(valid))
        tiny["apple_silicon"]["system_profile"]["profile"]["hardware"]["physical_memory"] = "1 GB"
        with self.assertRaisesRegex(ValueError, "sane physical"):
            planner.validate_hardware_inventories("apple-silicon-1", [tiny])
        missing_cores = json.loads(json.dumps(valid))
        missing_cores["apple_silicon"]["system_profile"]["profile"]["displays"][0].pop("sppci_cores")
        with self.assertRaisesRegex(ValueError, "GPU core"):
            planner.validate_hardware_inventories("apple-silicon-1", [missing_cores])
        false_metal = json.loads(json.dumps(valid))
        false_metal["apple_silicon"]["system_profile"]["profile"]["displays"][0]["spdisplays_metal"] = "not supported"
        with self.assertRaisesRegex(ValueError, "Metal support"):
            planner.validate_hardware_inventories("apple-silicon-1", [false_metal])

    def test_apple_collector_identity_and_filesystem_gates(self):
        profile = {
            "hardware": {
                "chip_type": "Apple M5 Max",
                "physical_memory": "128 GB",
                "machine_model": "Mac17,1",
                "machine_name": "MacBook Pro",
            },
            "displays": [{
                "sppci_model": "Apple M5 Max",
                "sppci_cores": "40",
                "spdisplays_metal": "spdisplays_supported",
            }],
            "software": {"os_version": "macOS 27"},
        }
        probes = {
            "apple_rosetta_translation": {"available": True, "returncode": 0, "stdout": "0"},
            "apple_power_settings": {
                "available": True,
                "returncode": 0,
                "stdout": "{\"AC Power\":{\"lowpowermode\":0}}",
                "power_modes": {"AC Power": {"lowpowermode": 0}},
            },
            "apple_power_source": {"available": True, "returncode": 0, "stdout": "AC Power"},
        }
        self.assertTrue(all(collector.apple_identity_checks(profile, probes, is_darwin=True, machine="arm64").values()))
        translated = {**probes, "apple_rosetta_translation": {"stdout": "1"}}
        self.assertFalse(collector.apple_identity_checks(profile, translated, is_darwin=True, machine="arm64")["rosetta_not_explicitly_active"])
        false_metal = json.loads(json.dumps(profile))
        false_metal["displays"][0]["spdisplays_metal"] = "not supported"
        self.assertFalse(collector.apple_identity_checks(false_metal, probes, is_darwin=True, machine="arm64")["metal_supported"])
        self.assertEqual(collector.filesystem_probe_argv(Path("/tmp"), "Darwin"), ["df", "-kP", "/tmp"])
        self.assertEqual(collector.filesystem_probe_argv(Path("/tmp"), "Linux"), ["df", "-B1", "-T", "/tmp"])

    def test_apple_system_profile_is_strictly_allowlisted(self):
        raw = {
            "SPHardwareDataType": [
                {
                    "chip_type": "Apple M4 Max",
                    "physical_memory": "128 GB",
                    "machine_model": "Mac16,5",
                    "serial_number": "SECRET-SERIAL",
                    "platform_UUID": "SECRET-UUID",
                    "provisioning_UDID": "SECRET-UDID",
                }
            ],
            "SPDisplaysDataType": [
                {
                    "sppci_model": "Apple M4 Max",
                    "sppci_cores": "40",
                    "spdisplays_metal": "spdisplays_supported",
                    "spdisplays_display-serial-number": "SECRET-DISPLAY",
                }
            ],
            "SPSoftwareDataType": [
                {
                    "os_version": "macOS 26",
                    "kernel_version": "Darwin 25",
                    "user_name": "private-user",
                    "computer_name": "private-host",
                }
            ],
        }
        safe = collector.sanitize_apple_system_profile(raw)
        rendered = json.dumps(safe)
        self.assertIn("Apple M4 Max", rendered)
        for secret in ("SECRET-SERIAL", "SECRET-UUID", "SECRET-UDID", "SECRET-DISPLAY", "private-user", "private-host"):
            self.assertNotIn(secret, rendered)

    def test_apple_memory_parses_pressure_and_swap(self):
        memory = collector.apple_memory(
            {"stdout": "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages wired down: 20.\nSwapins: 3.\n"},
            {"stdout": "System-wide memory free percentage: 42%"},
            {"stdout": "vm.swapusage: total = 2.00G  used = 512.00M  free = 1.50G"},
        )
        self.assertEqual(memory["free_bytes"], 100 * 16384)
        self.assertEqual(memory["wired_bytes"], 20 * 16384)
        self.assertEqual(memory["swapins"], 3)
        self.assertEqual(memory["system_free_percent"], 42)
        self.assertEqual(memory["swap_total_bytes"], 2 * 1024**3)

    def test_apple_telemetry_deltas_only_numeric_fields(self):
        deltas = apple_sampler.numeric_delta(
            {"swapouts": 3, "compressor_bytes": 100, "label": "first", "flag": False},
            {"swapouts": 8, "compressor_bytes": 80, "label": "last", "flag": True},
        )
        self.assertEqual(deltas, {"swapouts": 5, "compressor_bytes": -20})

    def test_apple_telemetry_memory_gate_rejects_swap_or_pageout(self):
        self.assertTrue(
            apple_sampler.memory_admission({"swapins": 0, "swapouts": 0, "pageouts": 0})["passed"]
        )
        for field in ("swapins", "swapouts", "pageouts"):
            delta = {"swapins": 0, "swapouts": 0, "pageouts": 0, field: 1}
            self.assertFalse(apple_sampler.memory_admission(delta)["passed"], field)
        self.assertFalse(
            apple_sampler.memory_admission(
                {"swapins": 0, "swapouts": 0, "pageouts": 0, "swap_used_bytes": 1}
            )["passed"]
        )

    def test_apple_process_probe_excludes_command_columns(self):
        with mock.patch.object(apple_sampler, "run_command", return_value={"available": True, "returncode": 0}) as command:
            result = apple_sampler.process_probe(123)
        argv = command.call_args.args[0]
        self.assertIn("pid,cpu,mem,rprvt,purg,vsize,threads,state,time,pageins", argv)
        self.assertFalse(any(value in {"command", "name", "args"} for value in argv))
        self.assertIn("exclude command", result["privacy_note"])
        self.assertFalse(result["pid_observed"])

        with mock.patch.object(
            apple_sampler,
            "run_command",
            return_value={"available": True, "returncode": 0, "stdout": "PID %CPU MEM\n 123 95.0 18G"},
        ):
            observed = apple_sampler.process_probe(123)
        self.assertTrue(observed["pid_observed"])

    def test_apple_telemetry_rejects_nonfinite_or_single_sample_windows(self):
        for invalid_duration in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                apple_sampler.validate_timing(invalid_duration, 1.0)
        with self.assertRaisesRegex(ValueError, "at least one interval"):
            apple_sampler.validate_timing(0.0, 1.0)
        apple_sampler.validate_timing(0.2, 0.2)

    def test_apple_telemetry_probe_failure_is_machine_readable(self):
        unavailable = {"available": False}
        with (
            mock.patch.object(apple_sampler, "run_command", return_value=unavailable),
            mock.patch.object(apple_sampler, "sanitized_apple_power_source", return_value=unavailable),
            mock.patch.object(apple_sampler, "process_info_thermal_state", return_value=unavailable),
        ):
            sample = apple_sampler.capture_sample("probe-failure", 0, None)
        self.assertFalse(sample["probe_health"]["ok"])
        self.assertIn("vm_stat", sample["probe_health"]["failed"])
        self.assertIn("thermal_state", sample["probe_health"]["failed"])

    def test_power_source_sanitizer_drops_battery_identifier(self):
        raw = {
            "available": True,
            "returncode": 0,
            "stdout": "Now drawing from 'AC Power'\n -InternalBattery-0 (id=35127395)\t87%; charging; 1:00 remaining",
        }
        with mock.patch.object(collector, "run_command", return_value=raw):
            safe = collector.sanitized_apple_power_source()
        self.assertEqual(safe["source"], "AC Power")
        self.assertEqual(safe["battery_percent"], 87)
        self.assertNotIn("35127395", json.dumps(safe))

    def test_power_settings_sanitizer_fingerprints_but_drops_unrelated_values(self):
        raw = {
            "available": True,
            "returncode": 0,
            "stdout": "AC Power:\n lowpowermode 2\n hibernatefile /private/secret/path\nBattery Power:\n lowpowermode 1",
        }
        with mock.patch.object(collector, "run_command", return_value=raw):
            safe = collector.sanitized_apple_power_settings()
        self.assertEqual(safe["power_modes"]["AC Power"]["lowpowermode"], 2)
        self.assertEqual(len(safe["profile_sha256"]), 64)
        self.assertNotIn("/private/secret/path", json.dumps(safe))

    def test_qwen_apple_capacity_rejects_24gb(self):
        plan = planner.build_plan("qwen3.8-27b", "apple-silicon-1", "single-stream")
        inventory = {
            "apple_silicon": {"system_profile": {"profile": {"hardware": {"physical_memory": "24 GB"}}}}
        }
        planner.apply_inventory_admission(plan, "apple-silicon-1", [inventory], False)
        self.assertEqual(plan["hardware_admission"]["status"], "rejected")
        self.assertTrue(all(candidate.get("rejected") for candidate in plan["candidates"]))

        plan_32 = planner.build_plan("qwen3.8-27b", "apple-silicon-1", "single-stream")
        inventory["apple_silicon"]["system_profile"]["profile"]["hardware"]["physical_memory"] = "32 GB"
        planner.apply_inventory_admission(plan_32, "apple-silicon-1", [inventory], False)
        self.assertEqual(plan_32["hardware_admission"]["status"], "conditional-short-context-single-stream")
        self.assertTrue(all(candidate.get("blocked_pending_capacity_admission") for candidate in plan_32["candidates"]))

        aggregate_32 = planner.build_plan("qwen3.8-27b", "apple-silicon-1", "aggregate")
        planner.apply_inventory_admission(aggregate_32, "apple-silicon-1", [inventory], False)
        self.assertTrue(all(candidate.get("rejected") for candidate in aggregate_32["candidates"]))
        self.assertIn("serving objective infeasible", aggregate_32["status"])

    def test_every_apple_converted_artifact_requires_transformation_manifest(self):
        plan = planner.build_plan("qwen3.8-27b", "apple-silicon-1", "single-stream")
        for candidate in plan["candidates"]:
            self.assertTrue(
                benchmark.candidate_requires_transformation(candidate),
                f"{candidate['id']} must not be allowed to claim transformation=none",
            )
        self.assertFalse(benchmark.candidate_requires_transformation({"artifact_class": "official-checkpoint"}))

    def test_every_nonofficial_planner_artifact_requires_transformation_manifest(self):
        for model in ("qwen3.8-27b", "deepseek-v4-flash-0731"):
            for hardware in planner.HARDWARE_PROFILES:
                for candidate in planner.build_plan(model, hardware, "single-stream")["candidates"]:
                    expected = candidate["artifact_class"] != "official-checkpoint"
                    self.assertEqual(benchmark.candidate_requires_transformation(candidate), expected, candidate["id"])

    def test_deepseek_single_spark_preserves_artifact_boundary(self):
        plan = planner.build_plan("deepseek-v4-flash-0731", "dgx-spark-1", "single-stream")
        by_id = {candidate["id"]: candidate for candidate in plan["candidates"]}
        self.assertTrue(by_id["deepseek-official-single-spark"]["rejected"])
        self.assertEqual(plan["artifact_boundary"]["official_checkpoint"]["status"], "rejected")
        self.assertEqual(by_id["deepseek-sparkinfer-transformed-single"]["artifact_class"], "transformed-target")
        self.assertTrue(by_id["deepseek-sparkinfer-transformed-single"]["blocked_pending_scope_acceptance"])
        self.assertEqual(plan["phase_scope"], "stopped at official capacity gate; alternatives listed but blocked")
        self.assertTrue(by_id["deepseek-magnitude-gguf-admission-control"]["rejected"])

        accepted = planner.build_plan(
            "deepseek-v4-flash-0731",
            "dgx-spark-1",
            "single-stream",
            allow_alternative_artifacts=True,
        )
        accepted_by_id = {candidate["id"]: candidate for candidate in accepted["candidates"]}
        self.assertFalse(accepted_by_id["deepseek-sparkinfer-transformed-single"]["blocked_pending_scope_acceptance"])
        self.assertTrue(accepted["scope_decision"]["alternative_artifacts_accepted"])


class OpenAIStreamIntegrationTest(unittest.TestCase):
    def test_usage_tokens_and_exact_output(self):
        expected = "amber cedar"
        events = [
            {"choices": [{"delta": {"reasoning_content": "brief thought"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": expected}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "completion_tokens": 4,
                    "total_tokens": 16,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
        ]
        lines = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
        lines.append(b"data: [DONE]\n\n")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def __iter__(self):
                return iter(lines)

        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return FakeResponse()

        with mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = benchmark.perform_request(
                request_id="test",
                prompt={"id": "p", "messages": [{"role": "user", "content": "copy"}], "expected_output": expected},
                endpoint="http://127.0.0.1:8000/v1/chat/completions",
                api_key=None,
                model="test-model",
                max_tokens=16,
                temperature=0,
                top_p=1,
                reasoning_effort=None,
                seed=42,
                timeout=5,
                unique_prefix=True,
                extra_body={},
                save_text=True,
            )
        self.assertEqual(result["completion_tokens"], 4)
        self.assertEqual(result["cached_prompt_tokens"], 2)
        self.assertEqual(result["processed_prompt_tokens"], 10)
        self.assertGreater(result["ttft_derived_processed_prompt_tok_per_s"], 0)
        self.assertEqual(result["reasoning_tokens"], 2)
        self.assertEqual(result["visible_answer_tokens"], 2)
        self.assertTrue(result["finish_reason_valid"])
        self.assertTrue(result["output_valid"])
        self.assertEqual(result["answer_text"], expected)
        self.assertGreater(result["completion_decode_tok_per_s"], 0)
        self.assertTrue(captured["payload"]["stream_options"]["include_usage"])


if __name__ == "__main__":
    unittest.main()
