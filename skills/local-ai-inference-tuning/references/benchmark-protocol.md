# Benchmark protocol

## Metrics

- `TTFT`: request start to first reasoning or answer output event. If useful, report time to first visible answer separately as `TTFO`.
- `TPOT`: `(E2E - TTFT) / (actual output tokens - 1)` for a request.
- `single-stream completion decode tok/s`: inverse TPOT for the C1 request; this includes reasoning tokens when the API's completion count includes them.
- `aggregate completion tok/s`: sum of actual completion tokens divided by measured wall time, explicitly labeled reasoning-inclusive when applicable.
- `visible-answer tok/s`: completion minus reported reasoning tokens over its visible-answer timing window; unavailable when the endpoint does not expose reasoning-token accounting.
- `goodput`: successful requests satisfying every declared SLO divided by wall time.
- `actual mean concurrency`: sum of request E2E divided by wall time.

SSE events are not tokens. A chunk can contain zero, one, or many tokens under speculative decoding. Use server usage accounting or the exact tokenizer revision.

## Five lanes

1. Correctness: deterministic arithmetic/code, strict JSON, tool calling, repetition/EOS, and long-context sentinel/checksum.
2. C1: fixed 2K/512, 8K/512, 32K/512; unique prefixes; warm kernels/graphs; 5–10 repetitions.
3. Offline: direct engine, fixed shapes, requests arriving together; at least 256 prompts or a stable-duration run.
4. Online: closed-loop concurrency sweep, then Poisson rates around 50/70/85/95/110% of saturation; production trace plus synthetic shapes.
5. Stability: 30–60 minutes at the target load with power, thermals, clocks, utilization, memory/UMA, swap, errors, and restart count.

Separate cold start, warm engine/cold prefix, and warm prefix. `ignore_eos` fixed-output microbenchmarks are synthetic and must be labeled.

## Comparable experiment rules

Freeze model/tokenizer revisions, chat template, thinking/reasoning setting, sampling, stop/EOS, ISL/OSL, dataset hash, seed, cache state, power mode, and client location. Record actual rather than requested token lengths. Include errors, timeouts, and truncations in the denominator.

Use randomized or ABBA order, at least three repeats for finalists, median and p95/p99 where sample size supports them, coefficient of variation, and the full raw output. Do not call a maximum value `p99`.

## Recommended framework tools

Prefer the installed version's `--help`; CLIs evolve.

### vLLM

Use `vllm bench throughput` for direct-engine offline capacity and `vllm bench serve` for HTTP serving. `--request-rate inf` is a burst, not realistic open-loop traffic. Current references:

- https://docs.vllm.ai/en/latest/benchmarking/cli/
- https://docs.vllm.ai/en/latest/cli/bench/throughput/
- https://docs.vllm.ai/en/latest/cli/bench/serve/

### SGLang

Use `python -m sglang.benchmark.offline_throughput` and `python -m sglang.benchmark.serving`. `--flush-cache` can preserve a warm engine while clearing the warmup prefix cache. Current sources:

- https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/offline_throughput.py
- https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/serving.py

### MLX / oMLX / llama.cpp Metal

Use `mlx_lm.benchmark` only for native random-token prefill/generation/peak-memory diagnosis. It omits tokenizer, chat template, HTTP, queueing, sampling and quality. The current mlx-lm `server_benchmark.py` counts SSE events as tokens, so do not use it for ranking speculative streams.

For llama.cpp, use `llama-bench` for native pp/tg/pg and build/backend/config evidence, and `llama-batched-bench` for shared/unshared batch saturation. Neither replaces the API/quality lane. `llama-server` metrics expose draft, accepted, verification, and per-position counters; adapt them to the same-artifact MTP comparison.

oMLX's admin benchmark provides exact token IDs, unique UUIDs, JIT warmup warnings and system/peak metrics, but a final result still needs statistical repeats, full quality gates, common API timing, and the run-window telemetry below.

- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py
- https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md
- https://github.com/jundot/omlx/blob/v0.6.1/omlx/admin/benchmark.py

### Cross-framework client

Prefer AIPerf for a maintained, reasoning-aware common client: https://github.com/ai-dynamo/aiperf . GenAI-Perf is being succeeded by AIPerf, and older clients may ignore `reasoning_content`. LLMPerf is archived and is suitable only for reproducing old reports.

The bundled `scripts/benchmark_openai.py` is a dependency-free smoke/microbenchmark for OpenAI-compatible chat endpoints. It requires streaming usage counts, a tokenizer-verified `--prompts` JSONL for performance work, and a structured `--comparison-contract` JSON. The contract pins artifact/model revision, tokenizer revision, transformation, advertised context, quality contract, hardware inventory hashes, power profile, client/location, cache state, and common-denominator versus best-achievable mode. Copy `templates/comparison-contract.json` and replace every placeholder. The bundled default prompt is allowed only with `--allow-default-smoke-prompt` and is not the planned 2K/8K/32K workload.

Before a performance run, execute the broader predeclared correctness suite and fill `templates/quality-gate.json`. Its suite-manifest and raw-result SHA-256 values, passed gate map, artifact/revision, image, framework revision, server-config digest, quality-contract ID, and canonical comparison-contract digest are mandatory. Obtain the latter with `python3 "$SKILL_ROOT/scripts/benchmark_openai.py" --comparison-contract contract.json --print-comparison-contract-sha256`. Pass the completed record with `--quality-gate-json quality-gate.json`. The benchmark validates and records its canonical digest and run ID; the comparator will not mix different quality/config tuples. This is a binding record of an external suite, not a substitute for actually running that suite.

Bind a performance run to its generated plan and candidate with `--plan-json plan.json --candidate-id ... --workload-case-id isl-2048-osl-512`. Run each shape as a separate result file; the client rejects a mixed-shape aggregate. Every prompt row must provide `expected_output` plus tight `expected_prompt_tokens_min/max` and `expected_completion_tokens_min/max` bounds containing the selected plan shape. Generate and verify these with the exact tokenizer/chat template; do not type character-count approximations. C1 requests must equal the plan's per-case repeat count, aggregate runs must satisfy its concurrency/minimum-wave rule, and the closed-loop goodput screen must meet its minimum count. Use `--shared-prefix` only for the separate `warm-prefix-separate-lane`; unique early prefixes are the `warm-engine-cold-prefix` default. For goodput, declare at least one TTFT/TPOT/E2E SLO.

For every performance run, fill `templates/run-metadata.json` with the image digest, framework revision, launch/server-config digest, served name, shared experiment ID, unique trial ID, and contiguous order index. The benchmark derives the quality-gate run ID and digest from `--quality-gate-json`; do not hand-type them. Before comparing finalists, run each candidate at least three times in an actual ABBA or randomized sequence. The comparator rejects duplicate run IDs/config mixing; for ABBA it validates each four-run A-B-B-A block. Then use:

```bash
python3 "$SKILL_ROOT/scripts/compare_runs.py" results/*.json
```

The comparator groups only identical candidate/image/framework/server-config/quality tuples, rejects mismatched structured contracts/workloads, checks inline exact correctness, token-shape bounds and token accounting, applies repeat/CV/order gates, and requires both a minimum gain and non-overlapping observed ranges before printing a winner. A single point estimate or unchecked output is never a winner.

Completion-token rates are named explicitly because they can include hidden reasoning tokens. When the endpoint reports `reasoning_tokens`, the client also reports visible-answer token rate and TTFO. Missing/`length`/`content_filter` finish reasons fail by default; a fixed-length synthetic prompt must explicitly declare its allowed finish reasons in JSONL.

## Telemetry

For discrete NVIDIA GPUs, sample at least power, temperature, SM/memory clocks, utilization, memory, P-state, and throttle reason. On DGX Spark also capture `tegrastats`, `/proc/meminfo`, and `vmstat`; `nvidia-smi` VRAM can be unsupported for UMA.

Useful command:

```bash
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,memory.total,pstate --format=csv --loop-ms=200
```

Save telemetry with a run ID and monotonic time. Compute output tok/J only when power integration covers the same measurement interval.

### Apple Silicon

Fix the exact Mac/chassis, macOS build, native arm64 ABI, AC/battery state, raw power mode, display topology, and starting thermal state. Warm Metal compilation and allocator caches before the measured sustained window. Capture:

- `vm_stat` page-size-aware wired/compressor/pagein/pageout/swapin/swapout deltas;
- process `phys_footprint` and peak, not RSS/VSIZE alone;
- Metal current allocation and recommended working set;
- synchronized MLX active/cache/peak memory where the server exposes it;
- low-power and thermal state before/during/after the run.

Any sustained swapout, compressor growth, or decode-time page-in makes a lane performance-inadmissible. Invalidate `serious`/`critical` thermal trials and label `fair` trials as thermal-affected. Use at least seven C1 trials and a 30-60 minute finalist run; do not report a cold burst as sustained tok/s.

`powermetrics` commonly requires root and can perturb the workload. Use only an explicitly authorized, narrow CPU/GPU/thermal sampler as a separate same-Mac diagnostic lane, then repeat the winner with the observer off. Do not use Apple power estimates to rank different machines. See [apple-silicon.md](apple-silicon.md).

## Common false gains

- total or reasoning-inclusive completion tok/s presented as visible-answer tok/s;
- C1 decode excluding TTFT compared with aggregate throughput including TTFT;
- repeated prompts silently hitting prefix cache;
- characters or requested OSL presented as actual tokens;
- thinking enabled on only one lane;
- reasoning tokens omitted by the client;
- direct-engine result compared with HTTP result;
- errors/timeouts/truncations silently dropped;
- a one-off best run reported without distribution;
- speculative code prompts generalized to prose or tool workloads;
- Apple results compared across Rosetta/native ABI, chip bins, chassis, power modes, or thermal states;
- a stream event counted as one token under MTP, or a reply to benchmark filler accepted as task correctness;
- client CPU/tokenizer/network saturation mistaken for server saturation.
