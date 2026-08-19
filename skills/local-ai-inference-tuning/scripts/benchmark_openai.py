#!/usr/bin/env python3
"""Run a dependency-free, usage-token-based microbenchmark against an OpenAI chat API.

This is a same-client smoke/C1/closed-loop comparator, not a replacement for
framework-native offline benchmarks or a Poisson-load serving benchmark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.1"
SENSITIVE_KEYS = {
    "token",
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "auth_token",
    "bearer_token",
    "credential",
    "credentials",
}
SENSITIVE_SUFFIXES = (
    "_api_key",
    "_password",
    "_secret",
    "_access_token",
    "_auth_token",
    "_bearer_token",
    "_token",
)
RESERVED_EXTRA_KEYS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "reasoning_effort",
    "seed",
}
REQUIRED_COMPARISON_FIELDS = {
    "base_model_id",
    "artifact_id",
    "artifact_revision",
    "tokenizer_revision",
    "transformation",
    "quality_contract_id",
    "advertised_context_tokens",
    "hardware_inventory_sha256s",
    "power_profile",
    "client_id",
    "cache_state",
    "comparison_mode",
    "allowed_candidate_differences",
}
REQUIRED_PERFORMANCE_METADATA = {
    "label",
    "order_design",
    "experiment_id",
    "trial_id",
    "order_index",
    "image_digest",
    "framework_revision",
    "server_config_sha256",
    "quality_gate_run_id",
    "quality_gate_sha256",
    "served_model_name",
}


def candidate_requires_transformation(candidate: dict[str, Any]) -> bool:
    """Only exact official controls may truthfully use transformation=none."""

    return candidate.get("artifact_class") != "official-checkpoint"


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float], scale: float = 1.0) -> dict[str, float] | None:
    data = [value * scale for value in values if value is not None and math.isfinite(value)]
    if not data:
        return None
    return {
        "count": len(data),
        "min": min(data),
        "p50": percentile(data, 0.50),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "max": max(data),
        "mean": statistics.fmean(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0.0,
    }


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            sensitive = lowered in SENSITIVE_KEYS or any(lowered.endswith(suffix) for suffix in SENSITIVE_SUFFIXES)
            result[key] = "<redacted>" if sensitive else redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper().startswith("REPLACE_")
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    return False


def load_json_object(spec: str | None) -> dict[str, Any]:
    if not spec:
        return {}
    path = Path(spec)
    raw = path.read_text(encoding="utf-8") if path.exists() else spec
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def load_comparison_contract(spec: str) -> tuple[dict[str, Any], str]:
    value = load_json_object(spec)
    missing = sorted(
        field
        for field in REQUIRED_COMPARISON_FIELDS
        if field not in value or (field != "allowed_candidate_differences" and value.get(field) in (None, "", []))
    )
    if missing:
        raise ValueError(f"comparison contract missing non-empty fields: {missing}")
    if contains_placeholder(value):
        raise ValueError("comparison contract still contains REPLACE_ template placeholders")
    context = value["advertised_context_tokens"]
    if not isinstance(context, int) or isinstance(context, bool) or context < 1:
        raise ValueError("comparison contract advertised_context_tokens must be a positive integer")
    inventory_hashes = value["hardware_inventory_sha256s"]
    if not isinstance(inventory_hashes, list) or not inventory_hashes or not all(
        isinstance(item, str) and len(item) == 64 for item in inventory_hashes
    ):
        raise ValueError("comparison contract hardware_inventory_sha256s must be a non-empty list of SHA-256 strings")
    if value["comparison_mode"] not in {"common-denominator", "best-achievable"}:
        raise ValueError("comparison contract comparison_mode must be common-denominator or best-achievable")
    differences = value["allowed_candidate_differences"]
    if not isinstance(differences, list) or not all(isinstance(item, str) and item for item in differences):
        raise ValueError("comparison contract allowed_candidate_differences must be a string list")
    if value["comparison_mode"] == "best-achievable" and not differences:
        raise ValueError("best-achievable contract must declare allowed_candidate_differences")
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def load_quality_gate(
    spec: str,
    comparison_contract: dict[str, Any],
    comparison_contract_sha256: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    value = load_json_object(spec)
    required = {
        "schema_version",
        "run_id",
        "status",
        "comparison_contract_sha256",
        "quality_contract_id",
        "artifact_id",
        "artifact_revision",
        "image_digest",
        "framework_revision",
        "server_config_sha256",
        "suite_manifest_sha256",
        "results_sha256",
        "gates",
    }
    missing = sorted(field for field in required if value.get(field) in (None, "", {}))
    if missing:
        raise ValueError(f"quality gate missing non-empty fields: {missing}")
    if contains_placeholder(value):
        raise ValueError("quality gate still contains REPLACE_ template placeholders")
    if value["schema_version"] != "1.0" or value["status"] != "passed":
        raise ValueError("quality gate must use schema_version 1.0 and status passed")
    expected = {
        "comparison_contract_sha256": comparison_contract_sha256,
        "quality_contract_id": comparison_contract["quality_contract_id"],
        "artifact_id": comparison_contract["artifact_id"],
        "artifact_revision": comparison_contract["artifact_revision"],
        "image_digest": metadata.get("image_digest"),
        "framework_revision": metadata.get("framework_revision"),
        "server_config_sha256": metadata.get("server_config_sha256"),
    }
    mismatched = sorted(field for field, expected_value in expected.items() if value.get(field) != expected_value)
    if mismatched:
        raise ValueError(f"quality gate does not match this artifact/configuration: {mismatched}")
    if not is_sha256(value["suite_manifest_sha256"]) or not is_sha256(value["results_sha256"]):
        raise ValueError("quality gate suite_manifest_sha256 and results_sha256 must be SHA-256 strings")
    gates = value["gates"]
    if not isinstance(gates, dict) or not gates or not all(
        isinstance(name, str) and name and status == "passed" for name, status in gates.items()
    ):
        raise ValueError("quality gate gates must be a non-empty object whose every value is passed")
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_prompts(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        answer = "\n".join(
            ["amber cedar cobalt delta ember fable granite harbor iris juniper"] * 12
        )
        return [
            {
                "id": "default-1",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Copy exactly the bytes between ANSWER_BLOCK_BEGIN and "
                            "ANSWER_BLOCK_END. Do not add a code fence or explanation.\n"
                            f"ANSWER_BLOCK_BEGIN\n{answer}\nANSWER_BLOCK_END"
                        ),
                    }
                ],
                "expected_output": answer,
            }
        ]
    source = Path(path)
    prompts: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{source}:{line_number}: expected JSON object")
        if "messages" not in item:
            if "prompt" not in item:
                raise ValueError(f"{source}:{line_number}: expected messages or prompt")
            item["messages"] = [{"role": "user", "content": str(item.pop("prompt"))}]
        if not isinstance(item["messages"], list):
            raise ValueError(f"{source}:{line_number}: messages must be a list")
        item.setdefault("id", f"line-{line_number}")
        prompts.append(item)
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def prompt_fingerprint(prompts: list[dict[str, Any]]) -> str:
    canonical = json.dumps(prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_workload_fingerprint_payload(
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    extra_body: dict[str, Any],
    comparison_contract_sha256: str,
    plan_sha256: str | None,
) -> dict[str, Any]:
    """Describe only the common workload, never candidate/config/quality identity."""
    return {
        "prompt_sha256": prompt_fingerprint(prompts),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "warmups": args.warmups,
        "unique_prefix": args.unique_prefix,
        "objective": args.objective,
        "comparison_contract_sha256": comparison_contract_sha256,
        "plan_sha256": plan_sha256,
        "workload_case_id": args.workload_case_id,
        "slos": {
            "ttft_ms": args.ttft_slo_ms,
            "tpot_ms": args.tpot_slo_ms,
            "e2e_ms": args.e2e_slo_ms,
        },
        "extra_body": redact(extra_body),
    }


def delta_text(delta: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("reasoning_content", "content"):
        value = delta.get(key)
        if isinstance(value, str):
            pieces.append(value)
        elif value:
            pieces.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        pieces.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))
    return "".join(pieces)


def token_count_within_prompt_bounds(prompt: dict[str, Any], kind: str, actual: Any) -> bool | None:
    lower = prompt.get(f"expected_{kind}_tokens_min")
    upper = prompt.get(f"expected_{kind}_tokens_max")
    if lower is None and upper is None:
        return None
    if not isinstance(lower, int) or isinstance(lower, bool) or not isinstance(upper, int) or isinstance(upper, bool):
        raise ValueError(f"expected_{kind}_tokens_min/max must be integers")
    if lower < 0 or upper < lower:
        raise ValueError(f"invalid expected_{kind}_tokens_min/max range")
    return isinstance(actual, int) and lower <= actual <= upper


def perform_request(
    *,
    request_id: str,
    prompt: dict[str, Any],
    endpoint: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_effort: str | None,
    seed: int | None,
    timeout: float,
    unique_prefix: bool,
    extra_body: dict[str, Any],
    save_text: bool,
) -> dict[str, Any]:
    messages = json.loads(json.dumps(prompt["messages"], ensure_ascii=False))
    if unique_prefix:
        nonce = f"benchmark nonce {request_id}; do not repeat it in the answer"
        messages = [{"role": "system", "content": nonce}, *messages]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if seed is not None:
        payload["seed"] = seed
    forbidden = RESERVED_EXTRA_KEYS.intersection(extra_body)
    if forbidden:
        raise ValueError(f"extra body cannot override reserved keys: {sorted(forbidden)}")
    payload.update(extra_body)

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    first_output: float | None = None
    last_output: float | None = None
    first_answer: float | None = None
    usage: dict[str, Any] | None = None
    output_parts: list[str] = []
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")
                    delta = choice.get("delta") or {}
                    if isinstance(delta, dict):
                        piece = delta_text(delta)
                        if piece:
                            observed_at = time.perf_counter()
                            if first_output is None:
                                first_output = observed_at
                            last_output = observed_at
                            output_parts.append(piece)
                        content = delta.get("content")
                        if isinstance(content, str):
                            if content and first_answer is None:
                                first_answer = time.perf_counter()
                            answer_parts.append(content)
                        if delta.get("tool_calls") and first_answer is None:
                            first_answer = time.perf_counter()
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str):
                            reasoning_parts.append(reasoning)
        finished = time.perf_counter()
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc

    if first_output is None:
        raise RuntimeError("stream completed without a reasoning, content, or tool-call output event")
    completion_tokens = usage.get("completion_tokens") if usage else None
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    cached_prompt_tokens = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    reasoning_tokens = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    visible_answer_tokens = None
    if isinstance(completion_tokens, int) and isinstance(reasoning_tokens, int):
        visible_answer_tokens = max(0, completion_tokens - reasoning_tokens)
    e2e = finished - started
    ttft = first_output - started
    tpot = None
    completion_decode_tps = None
    if isinstance(completion_tokens, int) and completion_tokens > 1 and e2e > ttft:
        tpot = (e2e - ttft) / (completion_tokens - 1)
        completion_decode_tps = 1.0 / tpot
    client_output_window_s = last_output - first_output if last_output is not None and last_output > first_output else None
    client_output_window_tps = None
    if isinstance(completion_tokens, int) and completion_tokens > 1 and client_output_window_s:
        client_output_window_tps = (completion_tokens - 1) / client_output_window_s
    processed_prompt_tokens = None
    if isinstance(prompt_tokens, int) and isinstance(cached_prompt_tokens, int):
        processed_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)
    ttft_derived_processed_prompt_tps = (
        processed_prompt_tokens / ttft
        if isinstance(processed_prompt_tokens, int) and ttft > 0
        else None
    )
    ttfo = first_answer - started if first_answer is not None else None
    visible_tpot = None
    visible_answer_tps = None
    if (
        isinstance(visible_answer_tokens, int)
        and visible_answer_tokens > 1
        and first_answer is not None
        and finished > first_answer
    ):
        visible_tpot = (finished - first_answer) / (visible_answer_tokens - 1)
        visible_answer_tps = 1.0 / visible_tpot
    output = "".join(output_parts)
    answer = "".join(answer_parts)
    reasoning = "".join(reasoning_parts)
    expected_output = prompt.get("expected_output")
    answer_valid = None if expected_output is None else answer == expected_output
    allowed_finish_reasons = prompt.get("allowed_finish_reasons", ["stop", "tool_calls"])
    if not isinstance(allowed_finish_reasons, list) or not all(isinstance(item, str) for item in allowed_finish_reasons):
        raise ValueError("allowed_finish_reasons must be a list of strings")
    finish_reason_valid = finish_reason in allowed_finish_reasons
    prompt_tokens_valid = token_count_within_prompt_bounds(prompt, "prompt", prompt_tokens)
    completion_tokens_valid = token_count_within_prompt_bounds(prompt, "completion", completion_tokens)
    shape_invalid = prompt_tokens_valid is False or completion_tokens_valid is False
    output_valid = False if not finish_reason_valid or answer_valid is False or shape_invalid else answer_valid
    result: dict[str, Any] = {
        "request_id": request_id,
        "prompt_id": prompt.get("id"),
        "success": True,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "processed_prompt_tokens": processed_prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_answer_tokens": visible_answer_tokens,
        "ttft_s": ttft,
        "ttfo_s": ttfo,
        "e2e_s": e2e,
        "tpot_s": tpot,
        "completion_decode_tok_per_s": completion_decode_tps,
        "client_output_window_s": client_output_window_s,
        "client_output_window_tok_per_s": client_output_window_tps,
        "ttft_derived_processed_prompt_tok_per_s": ttft_derived_processed_prompt_tps,
        "visible_answer_tpot_s": visible_tpot,
        "visible_answer_tok_per_s": visible_answer_tps,
        "finish_reason": finish_reason,
        "allowed_finish_reasons": allowed_finish_reasons,
        "finish_reason_valid": finish_reason_valid,
        "answer_valid": answer_valid,
        "prompt_tokens_valid": prompt_tokens_valid,
        "completion_tokens_valid": completion_tokens_valid,
        "output_valid": output_valid,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "reasoning_sha256": hashlib.sha256(reasoning.encode("utf-8")).hexdigest(),
    }
    if save_text:
        result["answer_text"] = answer
        result["reasoning_text"] = reasoning
    return result


def failed_result(request_id: str, prompt_id: Any, exc: BaseException) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "prompt_id": prompt_id,
        "success": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def write_result(document: dict[str, Any], output_spec: str) -> Path:
    output = Path(output_spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def summarize(results: list[dict[str, Any]], wall_s: float, slos: dict[str, float | None]) -> dict[str, Any]:
    successful = [result for result in results if result.get("success")]
    token_accounted = [result for result in successful if isinstance(result.get("completion_tokens"), int)]
    correctness_failed = [result for result in successful if result.get("output_valid") is False]
    correctness_unchecked = [result for result in successful if result.get("output_valid") is None]
    finish_reason_failed = [result for result in successful if result.get("finish_reason_valid") is False]
    token_shape_failed = [
        result
        for result in successful
        if result.get("prompt_tokens_valid") is False or result.get("completion_tokens_valid") is False
    ]
    completion_tokens = sum(result["completion_tokens"] for result in token_accounted)
    visible_token_accounted = [
        result for result in token_accounted if isinstance(result.get("visible_answer_tokens"), int)
    ]
    visible_answer_tokens = sum(result["visible_answer_tokens"] for result in visible_token_accounted)
    prompt_tokens = sum(result["prompt_tokens"] for result in token_accounted if isinstance(result.get("prompt_tokens"), int))
    cached_prompt_token_results = [
        result for result in token_accounted if isinstance(result.get("cached_prompt_tokens"), int)
    ]
    good = []
    for result in token_accounted:
        checks = [
            slos["ttft_ms"] is None or result["ttft_s"] * 1000 <= slos["ttft_ms"],
            slos["tpot_ms"] is None or (result.get("tpot_s") is not None and result["tpot_s"] * 1000 <= slos["tpot_ms"]),
            slos["e2e_ms"] is None or result["e2e_s"] * 1000 <= slos["e2e_ms"],
        ]
        if all(checks) and result.get("output_valid") is True:
            good.append(result)
    return {
        "wall_s": wall_s,
        "attempted": len(results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "token_accounted": len(token_accounted),
        "visible_token_accounted": len(visible_token_accounted),
        "correctness_failed": len(correctness_failed),
        "correctness_unchecked": len(correctness_unchecked),
        "finish_reason_failed": len(finish_reason_failed),
        "token_shape_failed": len(token_shape_failed),
        "request_per_s": len(successful) / wall_s if wall_s else None,
        "aggregate_completion_tok_per_s_including_reasoning": completion_tokens / wall_s if wall_s else None,
        "aggregate_visible_answer_tok_per_s": (
            visible_answer_tokens / wall_s
            if wall_s and len(visible_token_accounted) == len(token_accounted)
            else None
        ),
        "aggregate_total_tok_per_s": (prompt_tokens + completion_tokens) / wall_s if wall_s else None,
        "actual_concurrency_mean": sum(result["e2e_s"] for result in successful) / wall_s if wall_s else None,
        "goodput_per_s": len(good) / wall_s if wall_s else None,
        "goodput_completion_tok_per_s_including_reasoning": (
            sum(result["completion_tokens"] for result in good) / wall_s if wall_s else None
        ),
        "goodput_visible_answer_tok_per_s": (
            sum(result["visible_answer_tokens"] for result in good) / wall_s
            if wall_s and good and all(isinstance(result.get("visible_answer_tokens"), int) for result in good)
            else None
        ),
        "goodput_completed": len(good),
        "cached_prompt_tokens": distribution(
            result["cached_prompt_tokens"] for result in cached_prompt_token_results
        ),
        "ttft_ms": distribution((result["ttft_s"] for result in successful), scale=1000),
        "ttfo_ms": distribution((result["ttfo_s"] for result in successful if result.get("ttfo_s") is not None), scale=1000),
        "tpot_ms": distribution((result["tpot_s"] for result in successful if result.get("tpot_s") is not None), scale=1000),
        "e2e_ms": distribution((result["e2e_s"] for result in successful), scale=1000),
        "single_request_completion_decode_tok_per_s_including_reasoning": distribution(
            (
                result["completion_decode_tok_per_s"]
                for result in successful
                if result.get("completion_decode_tok_per_s") is not None
            )
        ),
        "single_request_visible_answer_tok_per_s": distribution(
            (
                result["visible_answer_tok_per_s"]
                for result in successful
                if result.get("visible_answer_tok_per_s") is not None
            )
        ),
        "client_output_window_tok_per_s_including_reasoning": distribution(
            (
                result["client_output_window_tok_per_s"]
                for result in successful
                if result.get("client_output_window_tok_per_s") is not None
            )
        ),
        "ttft_derived_processed_prompt_tok_per_s": distribution(
            (
                result["ttft_derived_processed_prompt_tok_per_s"]
                for result in successful
                if result.get("ttft_derived_processed_prompt_tok_per_s") is not None
            )
        ),
        "warnings": [
            "Completion-token rates include reasoning tokens; visible-answer rates are reported separately only when usage exposes reasoning_tokens.",
            "Aggregate completion tok/s includes request wall time; single-request completion decode tok/s excludes TTFT.",
            "SSE events were not counted as tokens; metrics require API usage token counts.",
            "client_output_window_tok_per_s ends at the last output event; the primary TPOT metric retains the complete response tail. TTFT-derived prompt rate is not pure server prefill throughput.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, normally ending in /v1; required for a run")
    parser.add_argument("--model", help="Served model name; required for a run")
    parser.add_argument("--output", help="Result JSON path; required for a run")
    parser.add_argument("--prompts", help="tokenizer-verified JSONL with id and messages or prompt")
    parser.add_argument(
        "--allow-default-smoke-prompt",
        action="store_true",
        help="use the bundled short exact-copy prompt for smoke testing only, never as the planned 2K/8K/32K workload",
    )
    parser.add_argument("--requests", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key-env", default="LOCAL_AI_API_KEY")
    prefix_group = parser.add_mutually_exclusive_group()
    prefix_group.add_argument(
        "--unique-prefix",
        dest="unique_prefix",
        action="store_true",
        default=True,
        help="add an early per-request nonce to avoid prefix reuse (default)",
    )
    prefix_group.add_argument(
        "--shared-prefix",
        dest="unique_prefix",
        action="store_false",
        help="explicit warm-prefix lane; allows prompt-prefix reuse",
    )
    parser.add_argument("--extra-body-json", help="JSON object or path; cannot override request-controlled model/length/sampling/stream fields")
    parser.add_argument("--metadata-json", help="Stack/provenance metadata JSON object or path; required fields are enforced for performance runs")
    parser.add_argument("--quality-gate-json", help="prior passed quality-gate record bound to this exact artifact/image/framework/server configuration")
    parser.add_argument("--plan-json", help="plan_experiments.py output to bind this run to")
    parser.add_argument("--candidate-id", help="candidate ID from --plan-json")
    parser.add_argument("--workload-case-id", help="one shape case_id from --plan-json; each shape is a separate result file")
    parser.add_argument(
        "--comparison-contract",
        required=True,
        help="JSON object or path pinning artifact/revisions/transformation/context/quality contract",
    )
    parser.add_argument(
        "--print-comparison-contract-sha256",
        action="store_true",
        help="validate the comparison contract, print its canonical SHA-256, and exit",
    )
    parser.add_argument("--objective", choices=("single-stream", "aggregate", "goodput"), default="single-stream")
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--allow-missing-usage", action="store_true")
    parser.add_argument("--save-text", action="store_true", help="Save generated text; may contain sensitive data")
    args = parser.parse_args()

    if args.print_comparison_contract_sha256:
        try:
            _, digest = load_comparison_contract(args.comparison_contract)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))
        print(digest)
        return 0
    missing_run_args = [name for name in ("base_url", "model", "output") if getattr(args, name) in (None, "")]
    if missing_run_args:
        parser.error(f"benchmark run missing required arguments: {missing_run_args}")

    if args.requests < 1 or args.concurrency < 1 or args.warmups < 0 or args.max_tokens < 1:
        parser.error("requests/concurrency/max-tokens must be positive and warmups non-negative")
    if args.objective == "single-stream" and args.concurrency != 1:
        parser.error("single-stream objective requires --concurrency 1")
    if args.objective == "goodput" and all(
        value is None for value in (args.ttft_slo_ms, args.tpot_slo_ms, args.e2e_slo_ms)
    ):
        parser.error("goodput objective requires at least one TTFT, TPOT, or E2E SLO")
    if args.prompts is None and not args.allow_default_smoke_prompt:
        parser.error("performance runs require --prompts; use --allow-default-smoke-prompt only for a labeled smoke test")
    if args.prompts is not None and not (args.plan_json and args.candidate_id and args.workload_case_id):
        parser.error("performance runs require --plan-json, --candidate-id, and --workload-case-id")
    if args.prompts is not None and not args.quality_gate_json:
        parser.error("performance runs require --quality-gate-json from a prior broad correctness evaluation")

    prompts = load_prompts(args.prompts)
    extra_body = load_json_object(args.extra_body_json)
    metadata = redact(load_json_object(args.metadata_json))
    try:
        comparison_contract, comparison_contract_sha256 = load_comparison_contract(args.comparison_contract)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    quality_gate = None
    quality_gate_sha256 = None
    if args.prompts is not None:
        try:
            quality_gate, quality_gate_sha256 = load_quality_gate(
                args.quality_gate_json,
                comparison_contract,
                comparison_contract_sha256,
                metadata,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))
        if metadata.get("quality_gate_run_id") not in (None, quality_gate["run_id"]):
            parser.error("performance metadata quality_gate_run_id conflicts with --quality-gate-json")
        if metadata.get("quality_gate_sha256") not in (None, quality_gate_sha256):
            parser.error("performance metadata quality_gate_sha256 conflicts with --quality-gate-json")
        metadata["quality_gate_run_id"] = quality_gate["run_id"]
        metadata["quality_gate_sha256"] = quality_gate_sha256
    expected_cache_state = "warm-engine-cold-prefix" if args.unique_prefix else "warm-prefix-separate-lane"
    if comparison_contract["cache_state"] != expected_cache_state:
        parser.error(
            f"comparison contract cache_state must be {expected_cache_state!r} for the selected prefix mode"
        )
    plan_sha256 = None
    if len([value for value in (args.plan_json, args.candidate_id, args.workload_case_id) if value]) not in {0, 3}:
        parser.error("--plan-json, --candidate-id, and --workload-case-id must be supplied together")
    if args.plan_json:
        plan_path = Path(args.plan_json)
        raw_plan = plan_path.read_bytes()
        plan = json.loads(raw_plan)
        if plan.get("schema_version") != "1.0" or plan.get("objective") != args.objective:
            parser.error("plan schema/objective does not match this benchmark")
        if plan.get("model") != comparison_contract["base_model_id"]:
            parser.error("comparison contract base_model_id does not match the supplied plan")
        candidates = {candidate.get("id"): candidate for candidate in plan.get("candidates", [])}
        if args.candidate_id not in candidates:
            parser.error("candidate ID is absent from the supplied plan")
        if candidates[args.candidate_id].get("rejected"):
            parser.error("candidate is rejected by the supplied plan")
        candidate = candidates[args.candidate_id]
        if candidate.get("blocked_pending_scope_acceptance"):
            parser.error("candidate is blocked until the plan is regenerated with explicit alternative-artifact scope acceptance")
        if candidate.get("blocked_pending_capacity_admission"):
            parser.error("candidate is blocked until framework-native load/context capacity admission passes and a narrowed executable plan is produced")
        if (
            candidate_requires_transformation(candidate)
            and str(comparison_contract["transformation"]).strip().lower() == "none"
        ):
            parser.error("selected candidate requires an explicit non-none transformation in the comparison contract")
        cases = {
            case.get("case_id"): case
            for case in (plan.get("workload") or {}).get("shapes", [])
            if isinstance(case, dict)
        }
        if args.workload_case_id not in cases:
            parser.error("workload case ID is absent from the supplied plan")
        workload_case = cases[args.workload_case_id]
        if args.max_tokens != workload_case.get("output_tokens"):
            parser.error("--max-tokens does not match the selected plan workload case")
        for prompt in prompts:
            if "expected_output" not in prompt:
                parser.error("performance prompts require expected_output for fail-closed inline correctness")
            required_bounds = (
                "expected_prompt_tokens_min",
                "expected_prompt_tokens_max",
                "expected_completion_tokens_min",
                "expected_completion_tokens_max",
            )
            if any(field not in prompt for field in required_bounds):
                parser.error(f"performance prompt {prompt.get('id')} is missing token-count bounds")
            prompt_min = prompt["expected_prompt_tokens_min"]
            prompt_max = prompt["expected_prompt_tokens_max"]
            completion_min = prompt["expected_completion_tokens_min"]
            completion_max = prompt["expected_completion_tokens_max"]
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in (prompt_min, prompt_max, completion_min, completion_max)):
                parser.error("performance prompt token bounds must be integers")
            max_prompt_width = max(16, int(workload_case["input_tokens"] * 0.01))
            max_completion_width = max(4, int(workload_case["output_tokens"] * 0.01))
            if not (prompt_min <= workload_case["input_tokens"] <= prompt_max) or prompt_max - prompt_min > max_prompt_width:
                parser.error("prompt token bounds do not tightly contain the selected plan ISL")
            if not (completion_min <= workload_case["output_tokens"] <= completion_max) or completion_max - completion_min > max_completion_width:
                parser.error("completion token bounds do not tightly contain the selected plan OSL")
            if prompt_max + completion_max > comparison_contract["advertised_context_tokens"]:
                parser.error("prompt/completion bounds exceed the comparison contract advertised context")
        plan_workload = plan["workload"]
        if args.objective == "single-stream" and args.requests != plan_workload["requests_per_case"]:
            parser.error("single-stream --requests must equal plan requests_per_case")
        if args.objective == "aggregate":
            if args.concurrency not in plan_workload["concurrency"]:
                parser.error("aggregate concurrency is absent from the supplied plan")
            if args.requests < args.concurrency * plan_workload["minimum_waves_per_case"]:
                parser.error("aggregate requests do not provide the plan minimum waves")
        if args.objective == "goodput":
            if args.concurrency not in plan_workload["closed_loop_concurrency"]:
                parser.error("goodput closed-loop concurrency is absent from the supplied plan")
            if args.requests < plan_workload["minimum_completed"]:
                parser.error("goodput requests are below the plan minimum completed count")
        plan_sha256 = hashlib.sha256(raw_plan).hexdigest()
        plan_inventory_hashes = [item.get("sha256") for item in plan.get("hardware_inventories", [])]
        if not plan_inventory_hashes:
            parser.error("performance plan has no validated hardware inventories")
        if plan_inventory_hashes != comparison_contract["hardware_inventory_sha256s"]:
            parser.error("comparison contract hardware inventories do not match the supplied plan")
        if metadata.get("candidate_id") not in (None, args.candidate_id):
            parser.error("performance metadata candidate_id conflicts with --candidate-id")
        if metadata.get("plan_sha256") not in (None, plan_sha256):
            parser.error("performance metadata plan_sha256 conflicts with the supplied plan")
        metadata["candidate_id"] = args.candidate_id
        metadata["plan_sha256"] = plan_sha256
    if args.prompts is not None:
        missing_metadata = sorted(
            field for field in REQUIRED_PERFORMANCE_METADATA if metadata.get(field) in (None, "")
        )
        if missing_metadata:
            parser.error(f"performance metadata missing non-empty fields: {missing_metadata}")
        if contains_placeholder(metadata):
            parser.error("performance metadata still contains REPLACE_ template placeholders")
        if str(metadata["order_design"]).lower() not in {"abba", "randomized"}:
            parser.error("performance metadata order_design must be ABBA or randomized")
        if not isinstance(metadata["order_index"], int) or isinstance(metadata["order_index"], bool) or metadata["order_index"] < 0:
            parser.error("performance metadata order_index must be a non-negative integer")
        if metadata["served_model_name"] != args.model:
            parser.error("performance metadata served_model_name must equal --model")
    api_key = os.environ.get(args.api_key_env)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def invoke(index: int, warmup: bool = False) -> dict[str, Any]:
        prompt = prompts[index % len(prompts)]
        request_id = ("warmup" if warmup else "measured") + f"-{index}-{uuid.uuid4().hex[:8]}"
        try:
            return perform_request(
                request_id=request_id,
                prompt=prompt,
                endpoint=endpoint,
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                reasoning_effort=args.reasoning_effort,
                seed=args.seed,
                timeout=args.timeout,
                unique_prefix=args.unique_prefix,
                extra_body=extra_body,
                save_text=args.save_text,
            )
        except BaseException as exc:  # Preserve every failure in the result file.
            return failed_result(request_id, prompt.get("id"), exc)

    warmup_results = [invoke(index, warmup=True) for index in range(args.warmups)]
    if any(not result.get("success") or result.get("output_valid") is not True for result in warmup_results):
        print(json.dumps({"warmup_failed": warmup_results}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    measured_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(invoke, index) for index in range(args.requests)]
        results = [future.result() for future in futures]
    measured_wall = time.perf_counter() - measured_started

    missing_usage = any(
        result.get("success") and not isinstance(result.get("completion_tokens"), int) for result in results
    )
    if not args.allow_missing_usage and missing_usage:
        print("endpoint did not provide streaming usage token counts; refusing token-rate ranking", file=sys.stderr)

    fingerprint_payload = build_workload_fingerprint_payload(
        args,
        prompts,
        extra_body,
        comparison_contract_sha256,
        plan_sha256,
    )
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    slos = {"ttft_ms": args.ttft_slo_ms, "tpot_ms": args.tpot_slo_ms, "e2e_ms": args.e2e_slo_ms}
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "openai-chat-stream-closed-loop-microbenchmark",
        "served_model_name": args.model,
        "objective": args.objective,
        "comparison_contract": redact(comparison_contract),
        "comparison_contract_sha256": comparison_contract_sha256,
        "workload_fingerprint": fingerprint,
        "workload": fingerprint_payload,
        "metadata": metadata,
        "summary": summarize(results, measured_wall, slos),
        "warmup": warmup_results,
        "requests": results,
    }
    output = write_result(document, args.output)
    print(json.dumps({"run_id": document["run_id"], "output": str(output), "summary": document["summary"]}, ensure_ascii=False, indent=2))
    if document["summary"]["failed"]:
        return 4
    if document["summary"]["correctness_failed"]:
        return 5
    if document["summary"]["correctness_unchecked"]:
        return 6
    if not args.allow_missing_usage and missing_usage:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
