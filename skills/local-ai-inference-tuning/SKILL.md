---
name: local-ai-inference-tuning
description: Select, launch, validate, benchmark, and tune local LLM inference stacks for a specified model and hardware, with special coverage for Qwen3.8-27B and DeepSeek-V4-Flash-0731 on DGX Spark, RTX PRO 6000, and Apple Silicon. Use when comparing vLLM, SGLang, MLX/oMLX, llama.cpp/Magnitude, Metal/CUDA kernels, quantization, speculative decoding, cache sizing, topology, or launch parameters to maximize single-stream tok/s, aggregate throughput, or SLO goodput.
---

# Local AI Inference Tuning

Treat the task as constrained empirical optimization. Do not name a winner until the same workload and correctness gates have been run on the target machine.

## Command path rule

Resolve `SKILL_ROOT` to the absolute directory containing this `SKILL.md` before invoking bundled files. Every command below is relative to that directory, not the caller's workspace root. For example, invoke `python3 "$SKILL_ROOT/scripts/collect_hardware.py"`; never assume that `scripts/...` exists in the current working directory.

## Workflow

1. Define the objective and workload.
   - Choose exactly one primary objective: single-stream completion decode tok/s, aggregate completion tok/s, or SLO goodput. State whether completion counts include hidden reasoning tokens and report visible-answer rate separately when available.
   - Record ISL/OSL distribution, concurrency or arrival rate, context ceiling, modality, reasoning mode, tools/JSON requirements, cache state, and quality tolerance.
   - If these are missing, propose a representative default and label it as an assumption.

2. Capture the target system before recommending a stack.
   - Run `python3 "$SKILL_ROOT/scripts/collect_hardware.py" --require-nvidia --pretty --storage-path MODEL_CACHE --container-image IMAGE --output hardware.json` on every NVIDIA node. Repeat `--container-image` for every locally present candidate image.
   - On a Mac, run `python3 "$SKILL_ROOT/scripts/collect_hardware.py" --require-apple-silicon --pretty --storage-path MODEL_CACHE --output hardware.json`. Require native `arm64`, exact chip/bin and GPU core count, installed unified memory, Metal support, macOS build, power source/mode, and framework versions. Reject Rosetta results.
   - Record exact GPU SKU, compute capability, host architecture, RAM/UMA, driver/CUDA, power limit, topology, network, container architecture, and free memory under the intended desktop/headless state.
   - On DGX Spark, use `/proc/meminfo`, `vmstat`, and `tegrastats`; do not infer available UMA from `nvidia-smi` VRAM fields.
   - On Apple Silicon, use `vm_stat`, swap and process/Metal/MLX footprint counters. Do not treat installed memory or `recommendedMaxWorkingSetSize` as a guaranteed model budget. A sustained swapout, compression, or page-in lane is functional-only, not performance-admissible.
   - For multiple DGX Sparks, verify the actual link and NCCL collectives. Two 128 GB systems are sharded capacity, not one coherent 256 GB pool.
   - For multiple Macs, verify the exact MLX distributed backend and collective topology. Per-host memory is sharded capacity only after the runtime proves real partitioning; Thunderbolt is not unified memory.
   - Read [apple-silicon.md](references/apple-silicon.md) for Apple hardware identity, privacy-safe telemetry, capacity formulas, and thermal gates.

3. Fingerprint the exact model artifact.
   - Pin model and tokenizer revisions. Record actual checkpoint bytes, architecture, total and active parameters, attention/SSM/MLA geometry, native context, quantization metadata, draft/MTP module, and vision tower.
   - Distinguish an official checkpoint from community quantization, pruning, expert removal, converted layouts, and fused-draft artifacts.
   - Read [model-profiles.md](references/model-profiles.md) for the two covered models.

4. Apply hard feasibility gates before tuning.
   - Require a registered architecture and kernels for the exact host architecture and accelerator target. `Blackwell`, `Apple Silicon`, or generic `Metal support` is not a sufficient compatibility claim.
   - On Apple, require a native arm64 runtime, exact MLX/oMLX/llama.cpp revision, Metal execution without CPU fallback, and support for the model architecture, quantization layout, cache, and drafter. Hardware BF16/matrix capability does not prove the framework uses it.
   - Budget resident weights + quant metadata/padding + KV/MLA/GDN/SSM state + graph/workspace + draft module + vision + runtime and OS safety margin.
   - Prefer framework cache logs and measured peak residency over bit-count estimates. Reject swap/offload for a maximum-tok/s lane unless it is explicitly the objective.
   - For MoE, use total resident parameters for capacity and active experts for token compute.
   - If the requested official artifact fails capacity, stop that plan before download, launch, or benchmark. List transformed/pruned/GGUF alternatives only as a scope decision. Do not continue until the user explicitly accepts a different artifact contract; then regenerate the plan with `--allow-alternative-artifacts`.
   - Read [framework-hardware-matrix.md](references/framework-hardware-matrix.md) before choosing a candidate.
   - When GGUF/llama.cpp is a feasible lane, use the estimator and benchmark patterns summarized in [magnitude-patterns.md](references/magnitude-patterns.md).

5. Build a short candidate list by evidence tier.
   - Prefer, in order: released upstream support; official model-specific image/recipe; pinned reproducible community image; experimental custom fork.
   - Never describe code on `main`, a custom image tag, or an unmerged patch as released upstream support.
   - Keep at least one conservative baseline with speculative decoding off.
   - If an appliance cannot run target-only, record that limitation as an evidence downgrade; do not attribute its speedup specifically to speculation.
   - Generate a starting matrix with `python3 "$SKILL_ROOT/scripts/plan_experiments.py" --model ... --hardware-profile ... --hardware-inventory hardware.json --objective ... --output plan.json`. Use `apple-silicon-1` for one Mac and repeat `--hardware-inventory` once per DGX Spark node. The planner rejects a profile whose validated accelerator/host identity does not match. Omit `--objective` only when accepting the plan's visibly labeled C1 assumption. Use `--allow-alternative-artifacts` only after explicit scope acceptance; the default plan blocks those candidates when the official artifact is infeasible.

6. Validate correctness before measuring speed.
   - Run deterministic arithmetic, code, strict JSON schema, tool-call/parser, EOS/repetition, and long-context begin/middle/end/checksum gates as applicable.
   - Re-run the same gates after every checkpoint, quantization, kernel, cache dtype, or speculative-decoding change.
   - Freeze the suite manifest and raw result hashes. Fill `templates/quality-gate.json`; every listed gate must be `passed`, and its artifact, comparison contract, image, framework revision, and server-config digest must match the performance run.
   - Stop on corruption, NaN, repeated loops, parser regression, material quality loss, silent CPU/generic-kernel fallback, or unexplained token-accounting mismatch.

7. Tune one family at a time.
   - Establish a no-speculation baseline.
   - Sweep, in order: weight/checkpoint lane; model-specific kernels; graph/eager mode; speculative method and depth; cache/state sizing; chunked prefill and scheduler limits; tensor/data/expert parallel topology.
   - Change one family per experiment. Use ABBA or randomized order and at least three repeats for finalists.
   - Capture complete launch commands, image digest, commits, environment overrides, logs, and telemetry.
   - On Apple, sweep artifact/quantization first, then target-only cache/context/batch, then MTP/draft depth, then optional ANE-prefill or distributed paths. Keep native oMLX MTP and an mlx-vlm sidecar as separate framework identities. Treat wired-memory changes as explicit privileged experiments with recorded before/after values and rollback, never as defaults.

8. Benchmark comparable lanes.
   - Use framework-native offline tools for engine throughput and one common client for online comparison.
   - Keep cold start, warm engine/cold prefix, and warm prefix as separate results.
   - Count actual usage tokens. Never count SSE chunks as tokens or target OSL as actual OSL.
   - Follow [benchmark-protocol.md](references/benchmark-protocol.md). Use `benchmark_openai.py` only as the same-client microbenchmark; retain the framework's native benchmark for deeper metrics.
   - Performance mode requires `--quality-gate-json`, `--plan-json`, a candidate and workload case, tokenizer-verified prompt bounds, and complete provenance. The client fails closed rather than accepting a free-text quality attestation.
   - On Apple, warm Metal compilation and allocator caches using a fixed protocol, then measure a sustained window at a fixed chassis, AC/battery state, power mode, display topology, and nominal starting thermal state. Run `python3 "$SKILL_ROOT/scripts/sample_apple_telemetry.py" --run-id RUN_ID --pid SERVER_PID --duration 180 --output telemetry.jsonl` beside the benchmark and require `window_health.valid=true`; report pagein/pageout, swapin/swapout, compressor, `NSProcessInfo.thermalState`, and power-source evidence. The sampler rejects non-finite or single-sample windows and exits nonzero on a failed gate. Privileged `powermetrics` is an optional same-Mac diagnostic lane and requires a separate observer-off confirmation run.

9. Select by the declared objective.
   - A candidate advances only if correctness passes and the gain exceeds run variance.
   - For single-stream, rank median hot C1 completion decode tok/s while reporting TTFT, TTFO, p95, and visible-answer rate where available.
   - For serving, rank aggregate completion tok/s or goodput at fixed SLO, not unconstrained burst throughput; never label reasoning-inclusive completion rate as visible-answer rate.
   - Reject a faster mean with unacceptable p99, errors, OOM, thermal drift, or speculative acceptance collapse on the real workload.

10. Deliver a reproducible recommendation.
    - State the winning model artifact, framework/image digest, exact launch command, workload, measured distribution, quality gates, telemetry, and confidence.
    - Include the conservative fallback and rollback command.
    - Separate confirmed observations, source-derived expectations, and unverified hypotheses.
    - If no common benchmark ran on the target hardware, say `candidate`, never `best`.

## Required output

Return a compact decision record containing:

- objective and workload contract;
- hardware and topology facts;
- model artifact and quality-equivalence boundary;
- feasible candidates with support/evidence tier;
- rejected candidates and exact stop reason;
- experiment matrix and comparable results;
- winner, exact command, fallback, remaining uncertainty.

Use [decision-workflow.md](references/decision-workflow.md) for formulas and decision gates, [troubleshooting.md](references/troubleshooting.md) for failure signatures, [apple-silicon.md](references/apple-silicon.md) for MLX/Metal-specific gates, [magnitude-patterns.md](references/magnitude-patterns.md) for the GGUF/llama.cpp planning lane, and [source-notes.md](references/source-notes.md) when a current claim needs re-verification.

## Safety and provenance rules

- Never print or scrape access tokens from shell startup files. Pass secrets through the runtime's secret mechanism and redact commands before saving.
- Hardware inventories avoid credential-like environment variables but are not anonymized; inspect or sanitize hostnames, GPU UUID/PCI data, topology, and network identifiers before external sharing.
- Do not use floating `latest` tags for a final result. Resolve tags to immutable digests.
- Do not download or convert a large checkpoint until free disk, expected bytes, revision, license, and checksum strategy are recorded.
- Do not delete global model or package caches as a tuning step.
- Do not compare transformed/pruned and official checkpoints as quality-equivalent without a dedicated evaluation.
