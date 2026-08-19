# Troubleshooting and falsification guide

## Failure before model load

| Signature | Likely cause | Check / action |
|---|---|---|
| no matching image manifest | missing `linux/arm64` image on DGX Spark | inspect manifest; use a multi-arch or ARM64 build |
| no kernel image / invalid device function | cubin/PTX misses exact SM target | print build arch list; rebuild for SM120/SM121 family as required |
| architecture unknown | framework registration is absent in the installed revision | confirm source and installed commit; do not assume `main` equals release |
| quantization unsupported | checkpoint layout and kernel do not match | inspect `quantization_config`; choose the exact model/SM implementation |

## Load succeeds but performance is low

1. Look for generic GEMM, dequantization, CPU offload, eager, or non-fused attention fallback in logs.
2. Check actual power mode, clocks, temperature, swap, and competing desktop processes.
3. Confirm the prompt was tokenized to the intended ISL and that prefix caching was not accidental.
4. Compare baseline without speculative decoding. Then record proposed, accepted, and rejected draft tokens by workload.
5. Inspect graph capture coverage and reserved memory; lowering a generic memory-utilization knob does not necessarily cure graph-pool OOM.
6. On dense C1, compare against the memory-bandwidth roofline. On MoE, inspect expert routing/balance and communication.

## DGX Spark specifics

- `nvidia-smi` memory unsupported is expected for UMA; use `/proc/meminfo`, `vmstat`, framework logs, and `tegrastats`.
- Require zero steady-state swap for a speed result.
- GB10 is SM121 and ARM64. An SM100 or SM120-specific image is not enough.
- Two Spark nodes use Ethernet/RoCE capacity sharding. GB10 does not provide normal GPUDirect RDMA semantics for `cudaMalloc` device memory; run real NCCL tests and measure collectives.
- Cluster setup success proves networking/SSH, not distributed inference correctness or efficiency.

## Apple Silicon specifics

- Reject `x86_64`/Rosetta. Use native arm64 Python, framework wheels, and server binaries.
- Record the exact chip bin and GPU cores. `M3 Max`/`M4 Max` alone can conceal different bandwidth, capacity, and sustained-power profiles.
- A load that enters swap is not a fast-path success. If swapout, compressor, or decode-time page-in rises, lower context, concurrency/batch, cache/state, draft residency, or artifact size.
- If MLX active memory is low but allocator cache is large, test an explicit cache limit. If MLX/Metal counters are far below `phys_footprint`, inspect IOKit/driver/in-flight/server memory and use the OS footprint for admission.
- Low GPU duty with busy CPU can indicate tokenizer, sampling, HTTP/Python scheduling, or tiny-batch overhead. High stable GPU duty plus quantization speedup is more consistent with bandwidth pressure.
- If tok/s and GPU frequency fall with thermal state, attribute the regression to chassis/power/thermals, not a framework parameter.
- ANE prefill affects TTFT, not decode, unless the exact implementation proves otherwise. Do not infer use from Neural Engine core count.
- Keep oMLX native MTP and an MLX-VLM sidecar as separate runtime identities. A same-model speed claim still needs pinned package/source and target-only controls.

## State-space / hybrid models

Do not apply ordinary Transformer KV-cache rules to GDN/Mamba/MLA state. Separate:

- persistent per-request state slots;
- speculative verification depth;
- attention KV pages;
- graph/workspace reserve.

If the framework exposes an explicit state-pool size, use its current model recipe. Do not copy a repository formula without verifying the installed framework semantics. Current Qwen/SGLang paths can use `concurrency * (state slots + draft depth)` without ReplaySSM, while a compatible ReplaySSM path can remove per-draft snapshots from that pool and use `concurrency * state slots`; its ring still consumes memory elsewhere.

## Speculative decoding

More draft tokens can reduce speed when draft cost and verification exceed accepted progress. Sweep from zero upward. Measure acceptance length/rate on code, prose, tool, and real traces separately. Stop when:

- throughput regresses;
- graph capture/OOM appears;
- structured decoding fails;
- acceptance collapses;
- output differs beyond the quality tolerance.

Never infer MTP and an external draft model are interchangeable. Confirm checkpoint structure and framework implementation.

## Multi-GPU / multi-node

- If the model and target cache fit one device, benchmark one device first.
- Confirm `nvidia-smi topo -m`, P2P behavior, NIC binding, MTU, NCCL logs, and container/model identity on every node.
- On PCIe-only RTX systems, do not assume NVLink. Disable or change custom all-reduce only after topology evidence and an A/B test.
- For MoE, compare tensor, expert, and data-parallel layouts under the target batch; the fastest capacity layout may not maximize C1.

## Correctness failures

Treat gibberish or semantic collapse as a hard failure even if kernel microtests pass. A KV/MLA layout can pass an oracle yet corrupt full-model generation. Re-run strict JSON, code, long-context sentinel/checksum, and cancellation/restart health after every low-level kernel or cache-format change.
