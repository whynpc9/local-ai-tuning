# Magnitude as a reference implementation

Snapshot reviewed: `magnitudedev/magnitude` commit `aea7e449806c4b9a14c3d46628dd306630cf3638`, 2026-08-17, Apache-2.0.

Magnitude is both a local agent and a Rust inference service built on a pinned llama.cpp stack. It is not merely an agent-quality harness. It contributes a distinct GGUF/llama.cpp execution lane and several unusually strong planning and benchmark patterns.

## Directly relevant implementation

### Exact GGUF admission planning

`icn-model-assessment` reads GGUF metadata and builds no-allocation graphs through the pinned llama.cpp `common_get_device_memory_data` and `common_fit_params` path. Inputs cover context, logical and physical prompt batch, sequence count, GPU layers, split mode/tensor split, K/V cache types, Flash Attention, KV offload, SWA/unified KV, MTP loading, and draft context type.

This is stronger than estimating `parameters × bits`: it works from exact encoded tensor storage and the same graph planner as the runtime. Its own documented boundary is equally important: the estimate can omit a projector, separate draft/MTP model, transient load allocations, other resident models, and additional OS reserve. On unified memory, CPU and accelerator reports must be deduplicated as one physical domain.

Source:

- https://github.com/magnitudedev/magnitude/blob/main/inference/crates/icn-hardware/README.md
- https://github.com/magnitudedev/magnitude/blob/main/inference/crates/icn-hardware/src/lib.rs

### Calibrated speed estimate with uncertainty

Magnitude runs short local tensor-operation calibration, then combines measured effective bytes/s and launch cost with a model-specific decode workload. The estimator accounts for:

- exact stored and executed tensor bytes;
- total versus active routed-expert traffic;
- conventional KV, MLA/compressed/sparse attention, recurrent state, and context depth;
- cross-memory-domain placement;
- fallback or unstable calibration and uncertainty bounds.

This is a useful shortlist generator, not a benchmark result. It contains policy efficiencies and uncertainty constants; missing exact calibration or complex architectures lower confidence. Always replace the estimate with an endpoint measurement before declaring a winner.

### Multi-objective recommendation

The catalog ranks concrete `(model, quantization, context)` configurations rather than model names. Four product intents weight capability, speed, fidelity, and memory differently:

| Intent | capability | speed | fidelity | memory |
|---|---:|---:|---:|---:|
| Balanced | 0.40 | 0.30 | 0.20 | 0.10 |
| Smartest | 0.60 | 0.10 | 0.30 | 0.00 |
| Fastest | 0.30 | 0.60 | 0.05 | 0.05 |
| Lightweight | 0.10 | 0.10 | 0.10 | 0.70 |

The skill adopts the principle, not these universal weights: expose the objective and tradeoffs rather than collapsing quality, speed, memory, and context into an unexplained score.

Source: https://github.com/magnitudedev/magnitude/blob/main/packages/acn/src/local-model-recommendation-policy.ts

## Coverage of the two focus models

The reviewed catalog contains:

- Qwen3.8-27B from `unsloth/Qwen3.8-27B-GGUF`, Q4/Q6/Q8, a BF16 multimodal projector, and a curated 100K serving profile. It does not currently declare a speculative draft for this model.
- DeepSeek-V4-Flash-0731 from `unsloth/DeepSeek-V4-Flash-0731-GGUF`, Q4/Q8, a GGUF DSpark Q8 draft, and a curated 100K serving profile.

These are community GGUF artifacts and product profiles. They are not the official Safetensors checkpoints used by the SGLang/vLLM recipes, and 100K is not the models' native maximum context. The publisher currently lists the DeepSeek UD-Q4_K_XL target at about 155 GB and UD-Q8_K_XL at 162 GB, before its 10.9 GB draft and runtime/cache reserve; Magnitude's cataloged Q4/Q8 lane is therefore a hard capacity rejection on one 128 GB Spark. A substantially lower-bit asymmetric artifact is a separate candidate and quality contract. Re-run quality equivalence and capacity gates. The catalog's capability scores explicitly cite publisher/model-family results rather than exact measurements of every GGUF quantization.

Sources:

- https://github.com/magnitudedev/magnitude/blob/main/inference/catalog/models.json
- https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF

## Benchmark suite worth reusing

Magnitude's endpoint-neutral `benchmark-runner` accepts generic OpenAI-compatible endpoints, so its protocol can compare Magnitude/llama.cpp with SGLang or vLLM. Its E1–E7 contrasts isolate:

- fixed request cost, prefill, sustained decode, and context-depth cost;
- prefill/decode batching and mixed-work interference;
- exact, partial, unrelated, and concurrently shared prefix reuse;
- tool/template/parser transaction cost;
- cancellation cleanup and immediate recovery.

It validates exact answer bytes and generated work, preserves raw SSE events and repetitions, disables accidental cache reuse, runs matched AB/BA blocks, and can continue until ratio confidence intervals reach a precision target. This is a better pattern than a Cartesian sweep with one noisy sample per cell.

The suite intentionally excludes startup, model quality, random production traffic, and configuration sweeps. Add separate quality, open-loop goodput, long-stability, and telemetry lanes.

Source: https://github.com/magnitudedev/magnitude/blob/main/inference/benchmark/README.md

## Adoption boundary

Use Magnitude in three ways:

1. as a llama.cpp/GGUF candidate when simplicity, local privacy, broad hardware coverage, or partial placement matters;
2. as a pre-download fit-estimator design reference for GGUF;
3. as a controlled endpoint benchmark protocol for any OpenAI-compatible server.

Do not assume its built-in lane maximizes CUDA tok/s for NVFP4/FP8, model-specific MTP/DSpark, multi-node Tensor Parallel, or very high concurrency. The specialized SGLang/vLLM/SparkInfer stacks still require direct comparison. Also pin Magnitude and its nested llama.cpp revisions: the reviewed repository is fast-moving and its package versions alone do not identify the full native runtime.
