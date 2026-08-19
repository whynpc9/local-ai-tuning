# Focus model profiles

Snapshot date: 2026-08-19. Re-read official config/model cards before acting because these models and framework recipes are moving quickly.

## Qwen3.8-27B

### Identity and architecture

- Official IDs: `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B-FP8`.
- Native architecture: `Qwen3_5ForConditionalGeneration`, dense multimodal model.
- Approximately 27.78B stored BF16 parameters; 64 language layers: 48 gated DeltaNet/linear-attention layers and 16 full-attention layers.
- Hidden width 5120; 24 query heads, 4 KV heads, head dimension 256.
- One checkpoint MTP layer. A separate external drafter is a different speculative lane.
- Native context 262,144. YaRN can extend toward 1M, but target and draft configs, memory, quality, and stability must all be revalidated.

Official sources:

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3.8-27B-FP8

### Artifact choices

| Artifact | Approximate repository storage | Status | Use |
|---|---:|---|---|
| Official BF16 | 55.6 GB | official | quality/control if resident budget permits |
| Official block FP8 | 30.9 GB | official | strong control on supported kernels |
| RadixArk NVFP4 | 21.9 GB | community | SGLang DGX/RTX candidate |
| unsloth NVFP4 | 23.4 GB | community | vLLM/community candidate |
| Magnitude/unsloth GGUF Q4/Q6/Q8 | artifact-dependent | community | llama.cpp/Magnitude lane |

There was no accessible official `Qwen/Qwen3.8-27B-NVFP4` at review time. Record the community exporter, revision, calibration, tensor layout, and quality results. Disk bytes are not resident bytes.

### Memory model

SGLang's model recipe reports approximately:

- FP8 KV: 32.8 KiB/token;
- BF16 KV: 65.5 KiB/token;
- current upstream Qwen recipe estimate for one persistent GDN state slot: about 74.81 MiB in BF16 or 146.81 MiB in FP32. Recompute from the installed revision; older launchers report larger 78.4/153.9 MiB values.

Current upstream modes use persistent-slot counts `S=5/4/3` for `extra_buffer` / `extra_buffer_lazy` / `no_buffer`. Without ReplaySSM, speculative depth contributes per-draft snapshots and the request-pool budget depends on `C × (S + D)`. When the exact compatible ReplaySSM path moves those snapshots to its ring, use `D=0` for this pool and the expression becomes `C × S`; the ring still consumes separate memory. Derive the value from installed logs and flags—neither formula is universal across revisions/backends.

The SGLang candidate matrix must include both native MTP/EAGLE-style verification and the current external `RadixArk/Qwen3.8-27B-DSpark` lane. Sweep ReplaySSM compatibility, SSM state dtype, radix strategy, and `mamba-full-memory-ratio` as memory/performance variables. The official DGX recipe's boot matrix demonstrates configuration reachability, not quality or tok/s superiority.

The 16 full-attention layers, not all 64 hybrid layers, drive ordinary KV-token storage. Still rely on the installed framework's cache log because graph capture, state dtype, vision, and draft allocations change usable capacity.

### Framework starting lanes

#### One DGX Spark

1. Same exact official FP8 checkpoint on pinned SGLang and vLLM GB10 builds where both pass loader/exact-SM gates; this is the common-denominator control.
2. SGLang's model-specific ARM64/SM121 image, pinned by digest; community NVFP4; FP8 KV; FlashInfer; native MTP disabled baseline then depth 2/3/4 sweep.
3. The same SGLang target plus pinned external `RadixArk/Qwen3.8-27B-DSpark`; depth 3/5/7, ReplaySSM on/off where compatible, state dtype/radix/memory-ratio sweeps. Keep target-only as its baseline.
4. Pinned community vLLM GB10 image; same-quality artifact where possible; MTP off then 1/2/3; eager versus graph.
5. Magnitude/llama.cpp GGUF Q4/Q6/Q8 when easy local operation, GGUF fit planning, broader placement, or agent integration matters. Treat as a separate artifact lane.

SGLang's official cookbook validates many DGX Spark boot/serve cells but does not publish a universal DGX performance winner. Community SGLang and vLLM headline numbers use different prompts and harnesses; re-run one client.

#### One RTX PRO 6000

1. SGLang SM120/FlashInfer with official FP8 control and pinned community NVFP4; compare native MTP with the external DSpark drafter.
2. vLLM Qwen-specific build/revision after proving the required post-release GDN/MTP fixes are installed.
3. Magnitude/llama.cpp GGUF as the simpler/control lane.

The card's high memory bandwidth makes this device a much stronger C1 candidate than DGX Spark when all artifacts fit, but exact SKU/power/cooling and kernel path matter.

#### One Apple Silicon Mac

Treat the reviewed oMLX repository as a candidate generator, not a golden result:

1. Pin native arm64 oMLX and `Jundot/Qwen3.8-27B-oQ4e-mtp`; run the identical target with MTP and ANE prefill disabled.
2. Only after that target-only quality baseline passes, sweep native MTP depth 1/2/3. Obtain acceptance from server counters, not stream-event count.
3. Add a pinned MLX/MLX-VLM target-only control and, separately, any MTP sidecar whose source and target/draft revisions are inspectable.
4. Add Magnitude/llama.cpp Metal GGUF Q4/Q6/Q8 as the portable artifact/runtime control.

Use `jundot/omlx` v0.6.1 only to reproduce the entry. Start new deployment measurements from pinned v0.6.2, which fixes the reviewed MTP+TurboQuant crash. The oQ4e artifact at reviewed revision `04dc5509...` is a roughly 15.82 GiB community mixed 4/5-bit conversion: reject 24 GB, treat 32 GB as short-context measured admission, and prefer 64 GB+ for practical tuning.

`mtp_num_draft_tokens` is configured maximum recursive depth, not a fixed number accepted every cycle. Record observed depth, drafted/accepted/per-position, cycles/tok-per-cycle, and standard-decode fallback. Normal continuous batching can beat opt-in row-wise MTP at C2/C4, so select single-stream and serving profiles separately.

Plain MLX-LM is a text/target-only control because its current Qwen loader removes vision and embedded MTP. MLX-VLM preserves full VLM/mRoPE and can split the MTP sidecar; its exact-prefix APC supports hybrid cache but needs hit/restart/output/memory gates. MTPLX and mlx-dspark are additional source-inspectable challengers whose published performance artifacts still need local revision pins.

The Weschera entry reports roughly 48 prose / 65.5 code tok/s at MTP3 on an M4 Max 128 GB, versus roughly 25 tok/s target-only. Its committed outputs include replies to repeated filler/marker text and code trials that do not start the requested task, while the harness has no correctness assertion. Those means remain a Tier-C performance lead that fails this skill's promotion gate. Re-run through the chat template, verify thinking mode, compile/execute code, and check repetition/structured output before ranking.

Apple capacity and speed depend on the exact chip bin, GPU core count, chassis, installed memory, power/thermal state, and UMA pressure. Reject Rosetta and any sustained swap/compressor/page-in lane. Read [apple-silicon.md](apple-silicon.md).

### Correctness gates

- thinking off and each supported reasoning effort;
- code generation plus deterministic exact-copy output;
- strict JSON and `qwen3_coder` tool calls;
- image input if VLM support is required;
- prefix-cache reuse across hybrid state;
- mixed prefill/decode and mixed speculative batches;
- 262K and any YaRN-extended sentinel/checksum lane;
- cancellation followed by an immediate clean request.

## DeepSeek-V4-Flash-0731

### Identity and architecture

- Official ID: `deepseek-ai/DeepSeek-V4-Flash-0731`.
- Architecture: `DeepseekV4ForCausalLM`, text model, native context 1,048,576.
- 43 layers; 256 routed experts, six selected per token, plus shared-expert work.
- Official base is commonly described as 284B total / 13B active. The fused checkpoint includes the DSpark module and reports about 304.18B stored tensor elements/parameters and roughly 166.9 GB repository storage. Use actual shard bytes for fit.
- The 0731 checkpoint carries a fused DSpark head with block size five and target layers near the end. It is not the preview checkpoint's ordinary MTP lane.
- The official repository uses model-specific encoding rather than a normal Jinja chat template; use the framework's DeepSeek-V4 tokenizer, reasoning parser, and tool parser.

Official sources:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json

### Capacity boundary

The official fused checkpoint cannot fit a single 128 GB DGX Spark or one 96 GB RTX PRO 6000 before runtime/cache reserve. Two Sparks or several RTX cards are the natural full-checkpoint lane.

The same official artifact is also a hard capacity failure on any 128 GB Mac. A 192 GB M2 Ultra leaves a tight margin after roughly 167 GB of repository files and requires measured loader/runtime/cache admission. Higher-memory M3 Ultra systems remove the storage-capacity objection but do not prove that a pinned MLX/Metal runtime implements the exact architecture and kernels. Community MLX quantizations, ablations, pruning, GGUF layouts, and draft combinations remain distinct artifact contracts.

Single-Spark projects make the target fit through target-weight transformations such as expert pruning/removal, EXL3/Trellis, asymmetric GGUF, or very-low-bit experts. A compact draft or custom engine may be part of the serving tuple, but neither makes an otherwise 166.9 GB target fit by itself. These are valuable alternatives, but they are different model artifacts and require their own quality/long-context/tool evaluation.

For MoE:

- capacity follows all resident experts and fused draft tensors;
- per-token compute/traffic follows active experts, shared work, attention/MLA, routing, batch, and communication;
- `total parameters × bytes / bandwidth` is not a valid decode predictor.

### Framework starting lanes

#### Two DGX Sparks

1. Full official checkpoint on a pinned community vLLM + B12X/SparkInfer ARM64 image, TP=2, DSpark off baseline then depth 3/5/7, graph A/B, measured NCCL.
2. SGLang's released DSPARK implementation is a framework challenger, but exact two-GB10 paths have had rank-divergence/acceptance issues; use a pinned build and stress gates.

The two nodes contribute sharded capacity, not one coherent pool. Their 200 GbE/RoCE link lacks normal device-memory GPUDirect RDMA semantics on GB10, so measure collectives and expect communication to constrain C1.

#### RTX PRO 6000

- One card: full official checkpoint is infeasible.
- Two cards: community vLLM SM120 integrations exist and report strong results, but exact model layout, cache, and transformed weight format determine fit.
- Four/eight cards: vLLM/SGLang become credible full-checkpoint candidates; assume PCIe-only until topology proves otherwise. Use `fp8_ds_mla` where the exact SM120 recipe requires it; do not import SM100-only NVFP4 kernels.

#### Single DGX Spark transformed lane

Compare at least:

- SparkInfer/vLLM EXL3/Trellis or expert-transformed artifact;
- `ds4` asymmetric GGUF/DSpark engine;
- Magnitude/llama.cpp GGUF only if its exact admission planner proves the chosen artifact and draft fit.

Every transformed lane needs an actionable quality contract before it can advance:

1. Pin an official-reference run on capacity-suitable hardware, or seal reference outputs from that exact official revision.
2. Reuse the tokenizer, prompts, sampling, reasoning/tool modes, and scoring code; declare task-level non-inferiority thresholds before testing.
3. Treat corruption, strict JSON/tool failures, and claimed-context failures as zero-tolerance gates. A feature unavailable in a candidate fails unless it was explicitly excluded from the product contract before the run.
4. Prove the context ceiling of each transformed artifact independently; official 1M lineage does not grant a converted artifact 1M support.

Rank same-contract results normally. Across different transformed artifacts, report quality, speed, memory, context, and feature coverage as a Pareto comparison. A faster compressed/pruned model is not the official model's performance result.

#### Apple Silicon transformed/high-memory lane

Use three deliberately separate hypotheses:

- official fused artifact on a capacity-suitable high-memory Mac, only after exact MLX/Metal loader and resident-peak proof;
- a pinned community MLX conversion/quantization, first with speculation disabled and a published transformation manifest;
- a pinned Magnitude/llama.cpp Metal GGUF target and optional draft, admitted from exact bytes.

For current Apple research, pinned oMLX v0.6.2 is the first DeepSeek service candidate because it publishes DeepSeek-specific attention/indexer/MoE and Lightning MTP/DSpark patches. Direct llama.cpp target + separately exported MTP/DSpark draft is the portable control; mlx-vlm is a research challenger. Reviewed q4/oQ4e residency is roughly 156-161 GB and naturally targets a 256 GB Mac. A roughly 92.8 GB 2.4-bit artifact can enter a 128 GB admission test only as a new quality contract.

Do not transfer a community “49 tok/s” claim across ablation, quantization, draft, prompt, or Mac bin. A 128 GB system cannot make the official target fit through a compact draft. A swap-dependent high-memory load is functional evidence only and cannot win a tok/s comparison.

### Speculative rules

- Use `dspark`/`DSPARK` for the official 0731 checkpoint.
- Keep a target-only baseline.
- Sweep small to larger depth and capture proposed, accepted, and rejected tokens plus position-wise acceptance.
- High acceptance can be a corruption symptom. Verify exact output, logprobs/oracle where available, tools, reasoning, long context, and mixed batches.
- A draft discovered at load time does not prove it participates correctly in decoding.

### Correctness gates

- deterministic exact output and code workloads;
- strict JSON and DeepSeek-V4 tool parser;
- reasoning low/high/max, including budget/termination behavior;
- tokenizer/encoding identity;
- 8K/64K/256K and target maximum begin/middle/end/checksum retrieval;
- cold prefix, prefix reuse, mixed prefill/decode, cancellation/recovery;
- repeated DSpark acceptance and collective stability over 30–60 minutes.
