# Decision workflow

## 1. Freeze the objective

`tok/s` is ambiguous. Choose one primary metric:

| Objective | Ranking metric | Mandatory companion metrics |
|---|---|---|
| Interactive C1 | median `(output_tokens - 1) / (E2E - TTFT)` | TTFT, p95 TPOT, quality, power |
| Offline capacity | aggregate completion tok/s from direct engine | input/total tok/s, reasoning/visible split, failures, batch |
| Online serving | goodput at fixed SLO, or aggregate completion tok/s at fixed load | TTFT/TTFO/TPOT/E2E p50/p95/p99, visible-answer rate, error rate |

Keep user-visible output tokens and reasoning tokens explicit. Record whether the endpoint reports them separately.

## 2. Feasibility gates

### Runtime compatibility

All must pass:

1. model architecture registered;
2. host image architecture matches (`linux/arm64` on DGX Spark);
3. compiled kernel/PTX supports the exact SM family;
4. quantization, attention, MoE/MLA/GDN, and draft kernels support that SM;
5. driver/runtime/container versions satisfy the chosen image;
6. requested API features and chat template are supported.

Treat `SM100`, `SM120`, and `SM121` as different targets. A generic Blackwell claim is insufficient.

### Capacity

Use this conservative budget:

```text
required resident bytes =
    measured resident weights
  + quantization metadata, padding, and converted layouts
  + KV / MLA / GDN / SSM state pool
  + CUDA graph capture pools and kernel workspaces
  + draft or MTP module
  + vision tower and processors
  + framework, driver, OS, and safety reserve
```

Checkpoint file size is an early lower bound, not resident-memory truth. Measure startup logs and the first real workload peak.

For a conventional attention layer, a rough KV estimate is:

```text
KV bytes/token = 2 * kv_heads * head_dim * cache_element_bytes * kv_layers
total KV       = KV bytes/token * live_tokens / tensor_parallel_degree
```

Do not apply that formula unchanged to MLA, hybrid attention/GDN, or Mamba. Use the framework's model-specific cache accounting.

### Compute and bandwidth

For dense batch-1 decode, a rough roofline is sustainable memory bandwidth divided by bytes streamed per target pass. It is useful for rejecting impossible claims, not predicting final tok/s. Speculative decoding can accept several tokens per target pass. MoE capacity follows total experts while decode traffic depends on active experts, routing, batching, and communication.

## 3. Evidence tiers

| Tier | Evidence | Permitted wording |
|---|---|---|
| A | Official model/config plus released framework docs/source and vendor hardware docs | supported, subject to stated version |
| B | Pinned community commit/image/model revision, runnable scripts, raw results, correctness gates | reproduced by that repo; candidate locally |
| C | README command and headline number without raw samples or canonical harness | reported claim |
| D | issue/social post/unpinned image or planned patch | hypothesis only |

Pin all four axes: model revision, tokenizer revision, framework commit/version, and container digest. A version-like repository title is not runtime provenance.

## 4. Candidate sequence

1. Conservative baseline: official or closest supported checkpoint, safe kernel, no speculative decoding, moderate context and concurrency.
2. Same checkpoint plus model-specific kernels.
3. Same checkpoint plus graph mode.
4. Speculative sweep from small to larger depth; record accepted tokens/draft tokens and workload type.
5. Memory allocation sweep: cache dtype, context ceiling, concurrency, state slots.
6. Scheduler sweep: chunk size, batch token budget, max sequences.
7. Parallelism/topology only when capacity or target load requires it.

For common-denominator framework comparison, keep checkpoint, quantization, KV dtype, sampling, template, cache state, and speculation identical. For best-achievable comparison, allow each stack's best features but re-run quality gates and disclose every difference.

When the official artifact cannot fit locally, stop the requested-artifact plan. Do not let a mechanical runner silently proceed into transformed/pruned/GGUF candidates. Continue only after explicit acceptance of that scope expansion, then establish quality equivalence against a pinned official run on adequate hardware or a sealed official-output corpus. Freeze datasets, tokenizer, sampling, scorers, feature coverage, and non-inferiority thresholds before testing. Unavailable claimed features and strict JSON/tool/context regressions fail unless excluded from the product contract in advance. Do not declare one fastest model across different artifacts; report a best-achievable Pareto frontier instead.

## 5. Stop conditions

Stop the current lane when any occurs:

- architecture, host architecture, or exact-SM mismatch;
- silent dequantization, CPU offload, generic GEMM fallback, or eager fallback that invalidates the hypothesis;
- swap activity under steady load;
- OOM, restart, CUDA/NCCL error, invalid graph capture, or unhealthy cancellation recovery;
- NaN, gibberish, repetition loop, parser/tool/JSON failure, or long-context regression;
- speculative acceptance collapses on the target workload;
- gain is within run variance after repeated ABBA trials;
- thermal/power drift makes the result nonstationary;
- network collectives dominate a multi-node lane that was chosen only for speed.

## 6. Decision record template

```markdown
Objective: ...
Workload: ISL ..., OSL ..., reasoning ..., C/rate ..., cache ...
Hardware: exact SKU/SM/arch/RAM/topology/power ...
Artifact: model@revision, tokenizer@revision, transformation boundary ...

Candidates:
- A: support tier, exact stack, hypothesis
- B: support tier, exact stack, hypothesis

Gates: correctness ..., memory ..., stability ...
Comparable result: median/p95 and repetitions ...
Winner: ... because ...
Fallback: ...
Uncertainty / next experiment: ...
```
