#!/usr/bin/env python3
"""Compare benchmark_openai.py results without silently mixing workloads."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


REQUIRED_PROVENANCE_FIELDS = {
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
    "candidate_id",
    "served_model_name",
}
CONFIG_IDENTITY_FIELDS = (
    "label",
    "candidate_id",
    "image_digest",
    "framework_revision",
    "server_config_sha256",
    "quality_gate_run_id",
    "quality_gate_sha256",
    "served_model_name",
)


def load_run(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") not in {"1.0", "1.1"} or "summary" not in value:
        raise ValueError(f"{path}: unsupported result schema")
    value["_path"] = path
    return value


def metric(run: dict[str, Any], objective: str) -> float | None:
    summary = run["summary"]
    if objective == "single-stream":
        distribution = (
            summary.get("single_request_completion_decode_tok_per_s_including_reasoning")
            or summary.get("single_request_decode_tok_per_s")
            or {}
        )
        return distribution.get("p50")
    if objective == "aggregate":
        return summary.get("aggregate_completion_tok_per_s_including_reasoning", summary.get("aggregate_output_tok_per_s"))
    return summary.get("goodput_per_s")


def label(run: dict[str, Any]) -> str:
    metadata = run.get("metadata") or {}
    return str(
        metadata.get("label")
        or metadata.get("stack")
        or metadata.get("candidate_id")
        or run.get("run_id")
        or run["_path"]
    )


def run_is_valid(run: dict[str, Any], value: float | None) -> bool:
    summary = run["summary"]
    return bool(
        run.get("schema_version") == "1.1"
        and value is not None
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and summary.get("failed") == 0
        and summary.get("correctness_failed", 0) == 0
        and summary.get("correctness_unchecked", 0) == 0
        and summary.get("finish_reason_failed", 0) == 0
        and summary.get("token_shape_failed", 0) == 0
        and summary.get("token_accounted") == summary.get("completed")
    )


def provenance_group_key(run: dict[str, Any]) -> tuple[str, ...]:
    metadata = run.get("metadata") or {}
    return tuple(str(metadata.get(field, "")) for field in CONFIG_IDENTITY_FIELDS)


def validate_order(runs: list[dict[str, Any]]) -> bool:
    metadata = [run.get("metadata") or {} for run in runs]
    if not all(all(item.get(field) not in (None, "") for field in REQUIRED_PROVENANCE_FIELDS) for item in metadata):
        return False
    experiment_ids = {str(item["experiment_id"]) for item in metadata}
    trial_ids = [str(item["trial_id"]) for item in metadata]
    order_indexes = [item["order_index"] for item in metadata]
    designs = {str(item["order_design"]).lower() for item in metadata}
    if (
        len(experiment_ids) != 1
        or len(trial_ids) != len(set(trial_ids))
        or not all(isinstance(index, int) and not isinstance(index, bool) and index >= 0 for index in order_indexes)
        or len(order_indexes) != len(set(order_indexes))
        or len(designs) != 1
        or not designs.issubset({"abba", "randomized"})
    ):
        return False
    ordered_indexes = sorted(order_indexes)
    if ordered_indexes != list(range(ordered_indexes[0], ordered_indexes[0] + len(ordered_indexes))):
        return False
    if designs == {"randomized"}:
        return True
    ordered_runs = [run for _, run in sorted(zip(order_indexes, runs), key=lambda pair: pair[0])]
    sequence = [provenance_group_key(run) for run in ordered_runs]
    if len(set(sequence)) != 2 or len(sequence) % 4 != 0:
        return False
    for offset in range(0, len(sequence), 4):
        a, b, c, d = sequence[offset : offset + 4]
        if not (a == d and b == c and a != b):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Result JSON files")
    parser.add_argument("--objective", choices=("single-stream", "aggregate", "goodput"))
    parser.add_argument("--allow-incomparable", action="store_true", help="Display mismatched workloads with a warning")
    parser.add_argument("--minimum-runs-per-candidate", type=int, default=3)
    parser.add_argument("--max-cv", type=float, default=0.05, help="maximum across-run coefficient of variation")
    parser.add_argument("--minimum-relative-gain", type=float, default=0.03)
    args = parser.parse_args()

    if args.minimum_runs_per_candidate < 2 or args.max_cv < 0 or args.minimum_relative_gain < 0:
        parser.error("minimum runs must be >=2 and variance/gain thresholds must be non-negative")

    runs = [load_run(path) for path in args.runs]
    run_ids = [run.get("run_id") for run in runs]
    if any(not run_id for run_id in run_ids) or len(run_ids) != len(set(run_ids)):
        print("refusing duplicate or missing run_id values; repeated files are not independent trials", file=sys.stderr)
        return 2
    objective = args.objective or runs[0].get("objective")
    if objective not in {"single-stream", "aggregate", "goodput"}:
        parser.error("objective is missing or invalid")
    objective_match = all(run.get("objective") == objective for run in runs)
    fingerprints = {run.get("workload_fingerprint") for run in runs}
    contracts = {run.get("comparison_contract_sha256") for run in runs}
    comparable_contract = len(contracts) == 1 and None not in contracts
    globally_comparable = len(fingerprints) == 1 and None not in fingerprints and comparable_contract and objective_match
    if not globally_comparable and not args.allow_incomparable:
        print("refusing to rank runs with different workload fingerprints, objectives, or missing/different structured comparison contracts; use --allow-incomparable only for inspection", file=sys.stderr)
        return 2

    global_order_ok = validate_order(runs)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for run in runs:
        value = metric(run, objective)
        grouped[provenance_group_key(run)].append({"run": run, "metric": value, "valid": run_is_valid(run, value)})

    rows = []
    for group_key, items in grouped.items():
        valid_values = [item["metric"] for item in items if item["valid"]]
        all_valid = len(valid_values) == len(items)
        median = statistics.median(valid_values) if valid_values else None
        mean = statistics.fmean(valid_values) if valid_values else None
        cv = (
            statistics.stdev(valid_values) / abs(mean)
            if mean not in (None, 0) and len(valid_values) > 1
            else (0.0 if len(valid_values) == 1 else None)
        )
        provenance_ok = all(
            all((item["run"].get("metadata") or {}).get(field) not in (None, "") for field in REQUIRED_PROVENANCE_FIELDS)
            for item in items
        )
        repeat_ok = len(items) >= args.minimum_runs_per_candidate
        variance_ok = cv is not None and cv <= args.max_cv
        eligible = globally_comparable and all_valid and repeat_ok and variance_ok and global_order_ok and provenance_ok
        summaries = [item["run"]["summary"] for item in items]
        rows.append(
            {
                "label": f"{group_key[0]} [{group_key[1]}:{group_key[4][:8]}]",
                "runs": len(items),
                "valid_runs": len(valid_values),
                "median": median,
                "minimum": min(valid_values) if valid_values else None,
                "maximum": max(valid_values) if valid_values else None,
                "cv": cv,
                "eligible": eligible,
                "failed": sum(summary.get("failed", 0) or 0 for summary in summaries),
                "correctness_failed": sum(summary.get("correctness_failed", 0) or 0 for summary in summaries),
                "order_ok": global_order_ok,
                "provenance_ok": provenance_ok,
                "repeat_ok": repeat_ok,
                "variance_ok": variance_ok,
            }
        )
    eligible_rows = sorted((row for row in rows if row["eligible"]), key=lambda row: row["median"], reverse=True)
    baseline = eligible_rows[-1]["median"] if eligible_rows else None

    print(f"objective: {objective}")
    if not globally_comparable:
        print("WARNING: workload or artifact/quality contract differs; rows are displayed but no winner is valid.")
    print("| rank | label | valid runs | median metric | observed range | CV | delta vs slowest eligible | errors/correctness | repeat/variance/order/provenance gates | eligible |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
    ordered = eligible_rows + [row for row in rows if not row["eligible"]]
    for index, row in enumerate(ordered, 1):
        delta = ((row["median"] / baseline - 1) * 100) if baseline and row["eligible"] else None
        print(
            "| {rank} | {label} | {valid_runs}/{runs} | {metric} | {range_} | {cv} | {delta} | {failed}/{correctness} | {gates} | {eligible} |".format(
                rank=index if row["eligible"] else "-",
                label=row["label"].replace("|", "\\|"),
                valid_runs=row["valid_runs"],
                runs=row["runs"],
                metric=f"{row['median']:.3f}" if isinstance(row["median"], (int, float)) else "n/a",
                range_=f"{row['minimum']:.3f}–{row['maximum']:.3f}" if isinstance(row["minimum"], (int, float)) else "n/a",
                cv=f"{row['cv']:.2%}" if isinstance(row["cv"], (int, float)) else "n/a",
                delta=f"{delta:+.1f}%" if delta is not None else "n/a",
                failed=row["failed"],
                correctness=row["correctness_failed"],
                gates="/".join("pass" if row[key] else "fail" for key in ("repeat_ok", "variance_ok", "order_ok", "provenance_ok")),
                eligible="yes" if row["eligible"] else "no",
            )
        )

    if len(eligible_rows) >= 2:
        leader, runner_up = eligible_rows[:2]
        relative_gain = leader["median"] / runner_up["median"] - 1 if runner_up["median"] else math.inf
        separated_ranges = leader["minimum"] > runner_up["maximum"]
        if relative_gain >= args.minimum_relative_gain and separated_ranges:
            print(f"\nmeasured winner for this contract: {leader['label']} (median gain {relative_gain:.1%}; observed ranges do not overlap)")
        else:
            print("\nno winner: the leading gain does not exceed the declared gain and non-overlapping-range gates")
    else:
        print("\nno winner: need at least two candidates with comparable contracts, valid results, repeated runs, low variance, and ABBA/randomized order metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
