# Apple Silicon tuning profile

Snapshot date: 2026-08-19. Re-check framework releases and Apple specifications before acting.

Use this profile for native Apple Silicon inference through MLX-family runtimes, oMLX, or llama.cpp/Magnitude Metal builds. The NVIDIA concepts of compute capability, discrete VRAM, CUDA graphs, and GPUDirect do not map one-for-one to this platform.

## Non-negotiable rules

- Run native `arm64`; reject Rosetta/x86_64 benchmark results.
- Identify the complete hardware tuple: chassis, machine model, chip, CPU/GPU core counts, installed unified memory, macOS build, AC/battery state, power mode, and display topology. A chip name alone is insufficient.
- Unified memory is shared by macOS, applications, CPU, GPU, model, runtime caches, and temporary buffers. Swap-assisted execution is not a maximum-tok/s lane.
- Prove the exact model architecture, quantization layout, cache format, and draft/MTP path in the installed runtime. Metal-family or matrix capability only describes hardware reachability.
- Do not infer an ANE gain from Neural Engine core count. Current MLX device selection is CPU/GPU; attribute ANE only when the exact runtime path and trace prove it.

## Hardware reference

| Chip | GPU cores | Maximum unified memory | Peak memory bandwidth | Metal family |
|---|---:|---:|---:|---:|
| M1 Max / Ultra | 24-32 / 48-64 | 64 / 128 GB | 400 / 800 GB/s | Apple7 |
| M2 Max / Ultra | 30-38 / 60-76 | 96 / 192 GB | 400 / 800 GB/s | Apple8 |
| M3 Max | 30 / 40 | 96 / 128 GB | 300 / 400 GB/s | Apple9 |
| M3 Ultra | 60 / 80 | 512 GB | 819 GB/s | Apple9 |
| M4 Max | 32 / 40 | 128 GB | 410 / 546 GB/s | Apple9 |
| M5 Max | up to 40 | 128 GB | 614 GB/s | Apple10 |

The same named Max chip can have different GPU cores, memory capacity, and bandwidth. MacBook Pro and Mac Studio also require separate sustained-performance profiles. Apple has not released an M4 Ultra; do not invent a profile by combining M4 Max specifications.

Primary Apple sources:

- [M1 Max and M1 Ultra Mac Studio](https://support.apple.com/en-us/111900)
- [M2 Max and M2 Ultra Mac Studio](https://support.apple.com/en-us/111835)
- [M3 Max MacBook Pro configurations](https://support.apple.com/en-us/117737)
- [M3 Ultra and 512 GB unified memory](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
- [M4 Max and M3 Ultra Mac Studio configurations](https://support.apple.com/en-us/122211)
- [M5 Max specifications](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- [Metal feature-set tables](https://developer.apple.com/metal/capabilities/)

## Inventory and privacy gate

Run the bundled collector first:

```bash
python3 "$SKILL_ROOT/scripts/collect_hardware.py" \
  --require-apple-silicon \
  --storage-path MODEL_CACHE \
  --pretty \
  --output hardware.json
```

The collector queries only an allowlisted `system_profiler` subset and never stores its raw payload. This is essential because even a minimal profile can contain serial numbers, platform UUIDs, provisioning identifiers, and user/computer names. It also avoids raw `ioreg`, network profiles, process command lines, and `sysctl -a`.

The validation tuple is native arm64 with no explicit Rosetta translation, an exact `Apple Mx [Pro|Max|Ultra]` chip, matching integrated-GPU model and positive core count, chassis/model, sane physical memory, exact macOS build, supported Metal, and captured power settings/source. Requested storage probes are also required to succeed. The inventory intentionally still contains the hostname and other local identifiers documented in its `notes`; review or further sanitize it before external sharing.

For a final run also capture, inside or beside the server process where available:

- Metal: device name, `hasUnifiedMemory`, `recommendedMaxWorkingSetSize`, `currentAllocatedSize`, supported families, maximum threadgroup memory, and threadgroup size;
- MLX after `mx.synchronize()`: active, allocator-cache, and peak memory, with the peak reset at the trial boundary;
- OS/process: `phys_footprint`, wired/compressed/swapped memory, and trial deltas for pagein/pageout and swapin/swapout;
- power/thermal: AC or battery, low-power mode, raw power-mode value, and `ProcessInfo.thermalState`.

`recommendedMaxWorkingSetSize` is an approximate no-obvious-harm threshold, not guaranteed capacity. RSS or VSIZE is not a substitute for physical footprint on UMA.

## Capacity admission

Use:

```text
effective_budget = min(
  metal_recommended_working_set,
  physical_memory
    - measured_idle_system_footprint
    - non_model_process_reserve
    - explicit_safety_reserve
)

required = resident_weights
  + quantization_metadata_and_padding
  + KV_or_recurrent_state(target_context, concurrency)
  + MTP_or_draft_residency
  + runtime_workspace_and_temporary_buffers
  + allocator_cache
  + tokenizer_and_server_footprint
```

Static arithmetic is only a prefilter. Load the maximum intended context/concurrency and reject a performance lane on any of:

- sustained swapout, page-in, or decode-time disk reads;
- compressor footprint that grows and does not settle;
- process/Metal footprint approaching the effective budget without reserve;
- Metal OOM, GPU timeout, OS kill, or framework fallback;
- an unexplained widening between MLX allocator counters and OS footprint.

MLX memory limits can exceed the recommended working set and permit swapping. Record the exact limit. `set_cache_limit`, `set_memory_limit`, and `set_wired_limit` are experiment variables, not invisible setup. A privileged `iogpu.wired_limit_mb` change requires explicit authorization, before/after capture, a rollback command, and a separate comparison contract.

## Framework lanes

| Lane | Strength | Hard boundary |
|---|---|---|
| MLX / MLX-LM | Apple-maintained native Metal core; plain current Qwen loader is a text/target-only control | it strips vision and embedded MTP; the basic server is not a production throughput reference |
| MLX-VLM | full Qwen VLM/mRoPE plus split MTP sidecar, ragged acceptance, and hybrid exact-prefix APC | pin the moving revision; structured output disables speculation and exact-prefix cache needs hit/restart/output gates |
| oMLX | Public OpenAI/Anthropic-compatible server with continuous batching, paged CoW KV, prefix cache, native MTP, and optional ANE prefill | pin the real `jundot/omlx` source/tag and its vendored dependency commits; the entry README incorrectly links an empty GitHub account |
| mlx-vlm plus MTP sidecar | inspectable alternative to native speculative integration | separate runtime identity; pin target, sidecar, tokenizer, and all source revisions; capture acceptance counters |
| llama.cpp / Magnitude | portable GGUF Metal control, exact artifact fit planning, broad quantization choices | converted artifact and different kernels; not quality-equivalent to oQ4e or official weights by default |
| MLX distributed | ring TCP, JACCL/Thunderbolt RDMA on supported systems, or MPI | each Mac keeps separate memory; prove partitioning and measure collectives before claiming combined capacity or speed |

For Qwen3.8-27B, establish these controls before promotion:

1. The exact oMLX target artifact with MTP disabled.
2. The identical artifact/runtime with MTP depth 1, 2, then 3.
3. A pinned MLX/MLX-VLM target-only lane if the exact architecture loads correctly.
4. A native arm64 llama.cpp/Magnitude GGUF Q4/Q6/Q8 control admitted by exact bytes.

For DeepSeek-V4-Flash-0731, its roughly 167 GB official repository is a hard capacity failure on 128 GB Macs. A 192 GB system is a tight measured-admission case, not an automatic pass; high-memory M3 Ultra systems still require exact runtime support and peak residency. Keep every MLX/GGUF conversion in a distinct artifact and quality contract.

## Weschera oMLX entry: what it proves and what it does not

The [Qwen3.8-27B-oMLX-MTP-Mac repository](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac) is a useful configuration lead. At commit [`0800fb5`](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac/commit/0800fb5ca9a5921a32ff32dbe4d7cb3e5d9feeac), it reports an M4 Max 128 GB, [oMLX 0.6.1](https://github.com/jundot/omlx/releases/tag/v0.6.1), `Jundot/Qwen3.8-27B-oQ4e-mtp`, ANE prefill off, and configured MTP maximum depth three. The reported means are roughly 48 tok/s for prose and 65.5 tok/s for code, versus roughly 25 tok/s without MTP. The README's `github.com/omlx` link is wrong; the public source is [jundot/omlx](https://github.com/jundot/omlx).

These are source-reported throughput leads, not winner evidence. Audit of the committed [`baseline.json`](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac/blob/0800fb5ca9a5921a32ff32dbe4d7cb3e5d9feeac/baseline.json), [`mtp2.json`](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac/blob/0800fb5ca9a5921a32ff32dbe4d7cb3e5d9feeac/mtp2.json), and [`mtp3.json`](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac/blob/0800fb5ca9a5921a32ff32dbe4d7cb3e5d9feeac/mtp3.json) finds streams that discuss repeated benchmark filler, reproduce marker/filler text, or fail to begin the requested code task. The harness does not compile code or assert task correctness. Its raw-completions path also claims thinking is off while a saved baseline starts with a thinking block, and its fallback increments once per SSE data event rather than tokenizer-counting tokens.

Therefore:

- do not reuse its 48/65.5 tok/s as a promoted default;
- use `/v1/chat/completions` with the exact chat template for reasoning-mode comparisons;
- tokenizer-verify ISL/OSL and actual completion usage;
- require deterministic task, strict JSON/tool, code execution, repetition, long-decode, and long-context gates at every MTP depth;
- obtain proposed/accepted/rejected token counters from the server, not the stream chunk count;
- keep ANE prefill as a TTFT-only A/B until a pinned source/runtime path proves otherwise.

For historical reproduction, pin oMLX v0.6.1 plus its actual dependencies: MLX 0.32.0, `mlx-lm@ab1806e`, `mlx-vlm@78b96eb`, and the declared dflash/nanobind pins in its `pyproject.toml`. For a new deployment baseline, begin with [oMLX v0.6.2](https://github.com/jundot/omlx/releases/tag/v0.6.2) and remeasure: it fixes a v0.6.1 MTP+TurboQuant KV crash. Do not mix a v0.6.1 speed record with a v0.6.2 stability result.

The reviewed Qwen oQ4e artifact revision `04dc5509...` contains about 15.81 GiB of weight shards. A 24 GB Mac's default system reserve leaves roughly the weight size before runtime/cache, so reject it for this lane. Treat 32 GB as short-context measured admission; 64 GB and above still require the requested context/concurrency peak.

oMLX MTP is default-off. Its depth-three setting is a maximum recursive chain from one MTP layer, not three heads; the controller can choose zero through the maximum and fall back to autoregressive decode when speculation loses. Grammar-constrained decode bypasses MTP. Row-wise MTP batching is opt-in and has underperformed ordinary continuous batching in published tests, so keep it out of the default serving lane and A/B it only for the real concurrency mix.

Do not conflate the historical entry with the newly public DFlash 2 path. The public oMLX v0.6.1 dependency is `jundot/dflash-mlx@2eb169f4`, whose pinned target/draft registry does not include Qwen3.8, so it still cannot reproduce the entry's original DFlash2 claim. As of 2026-08-18, however, `incoai/Qwen3.8-27B-DFlash2` revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, its GGUF variants, the `z-lab/dflash` MLX backend, and the separate `z-lab/omlx-fork` `0.6.2-dflash2` prebuilt are public. Treat that combination as a new C-tier challenger, not as a feature of the historical v0.6.1 dependency or the upstream jundot/oMLX v0.6.2 release.

The DFlash 2 BF16 draft weights are about 3.85 GB; the Q4_K_M GGUF is about 1.14 GB. These are static file sizes, not peak unified-memory cost. For quantized MLX target/draft lanes start with `block_size <= 5`, pin target and draft separately, and measure draft prefill, active/cache/peak memory, swap/compression/page-in, long-context acceptance, continuous batching, tools/JSON, and vision. The current fork build has an open report that continuous-batching benchmark rows do not run. Native MTP and DFlash remain mutually exclusive oMLX lanes and must share the same target, prompt, sampling, quality, and thermal contract.

ANE/GPU prefill is also opt-in. oMLX reports that it can improve 16K/32K prefill while adding roughly 4.15 GB peak and about 24 seconds of load/setup on an M3 Ultra; it does not accelerate decode. Use a separate TTFT contract, exact-output/logit gate, per-Mac tuner, and memory admission. Building custom kernels requires full Xcode; a published DMG can contain precompiled kernels.

## Benchmark and thermal protocol

1. Fix chassis, OS build, AC/battery state, power mode, display topology, model cache, and background-load policy.
2. Start from nominal thermal state. Warm model load, Metal compilation, allocator caches, and tokenizer using a fixed warmup; keep warm-prefix measurements separate.
3. Use tokenizer-verified prompts and fixed output bounds. Run unique-prefix cold and shared-prefix warm lanes separately.
4. Interleave candidates with ABBA or randomized order. Use at least seven C1 trials for a final median/CV decision and a sustained 30-60 minute finalist run.
5. Record TTFT/TTFO, completion decode tok/s including reasoning, visible-answer tok/s, TPOT/ITL tails, errors, finish reasons, exact output quality, MLX/Metal/process peak memory, swap/compression/page-in deltas, and thermal state.
6. Invalidate serious/critical thermal trials. Mark fair-state trials thermal-affected. A cold burst is not sustained performance.

Run `sample_apple_telemetry.py` for at least one full interval (normal finalist windows are 30-60 minutes). It rejects `NaN`/infinite timing, requires at least two samples, reads `NSProcessInfo.thermalState`, captures a sanitized power source plus an allowlisted power-mode view and full-profile hash, and exits nonzero when required probes, power-window stability, thermal state, or the zero swapin/swapout/pageout gate fails. Preserve its `window_health`, `pageins/pageouts`, `swapins/swapouts`, power-profile hash, and compression deltas with the benchmark result. When `--pid` is used, the target PID must actually appear in the privacy-bounded `top` sample; a successful command with no target row is a failure.

Optional privileged telemetry is a separate lane. Probe `powermetrics -h`, request only CPU/GPU/thermal samplers supported by that machine, avoid task/network/all samplers, and verify the winner again with the observer off. Apple describes these power values as estimates; use them only for same-Mac A/B diagnosis, not cross-machine efficiency claims.

## Multi-Mac boundary

MLX documents ring/TCP, MPI, and JACCL. JACCL requires supported macOS, Thunderbolt topology, and system RDMA setup; automatic setup may require passwordless sudo and modify interfaces, so the skill must never enable it implicitly. Validate `mlx.distributed_config`, collective bandwidth/latency, rank stability, exact sharding, and end-to-end TTFT/tok/s.

Thunderbolt bandwidth is far below a Max/Ultra chip's local 300-819 GB/s memory bandwidth. Multi-Mac tensor parallelism can solve capacity while reducing C1 tok/s. Choose it from measurements, not from the sum of installed memory.

References:

- [MLX memory management](https://ml-explore.github.io/mlx/build/html/usage/memory_management.html)
- [MLX distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [Metal recommended working set](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- [Apple power modes](https://support.apple.com/en-us/101613)
- [Apple Silicon does not support external GPUs](https://support.apple.com/en-us/102363)
