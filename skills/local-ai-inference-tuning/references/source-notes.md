# Source notes and refresh map

Research cutoff: 2026-08-19. Performance claims from community repositories were audited but not rerun on the target accelerators in this workspace.

## Official model facts

- Qwen3.8-27B model card: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen config: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- Qwen official FP8: https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- DeepSeek-V4-Flash-0731 model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- DeepSeek config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json
- Unsloth DeepSeek 0731 GGUF sizes used by the Magnitude catalog: https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF

Refresh these first for revisions, architecture, native context, quantization config, tokenizer/encoding, and speculative modules.

## Framework truth sources

- SGLang Qwen cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- SGLang DeepSeek cookbook: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
- SGLang releases: https://github.com/sgl-project/sglang/releases
- vLLM Qwen recipe: https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml
- vLLM DeepSeek recipe: https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml
- vLLM releases: https://github.com/vllm-project/vllm/releases
- vLLM supported models: https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md
- TensorRT-LLM DeepSeek guide: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v4/README.md
- TensorRT-LLM releases: https://github.com/NVIDIA/TensorRT-LLM/releases
- B12X/SparkInfer kernel source: https://github.com/local-inference-lab/b12x
- llama.cpp: https://github.com/ggml-org/llama.cpp
- MLX: https://github.com/ml-explore/mlx
- MLX-LM: https://github.com/ml-explore/mlx-lm
- MLX-VLM: https://github.com/Blaizzy/mlx-vlm
- MLX memory management: https://ml-explore.github.io/mlx/build/html/usage/memory_management.html
- MLX distributed: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- oMLX public source and releases: https://github.com/jundot/omlx and https://github.com/jundot/omlx/releases — the Weschera README incorrectly links an unrelated empty `github.com/omlx` account
- oMLX entry reproduction: v0.6.1 `b587575f3696fbc86c236b906684a48a92f8b118`, with MLX 0.32.0, `mlx-lm@ab1806e8`, `mlx-vlm@78b96eb5`, and `dflash-mlx@2eb169f4` as declared in https://github.com/jundot/omlx/blob/v0.6.1/pyproject.toml
- oMLX new deployment baseline: v0.6.2 `f2d36f3d25a7e7a2401a92eecafc28b8f8968ec7`; it fixes the v0.6.1 MTP+TurboQuant KV crash, so remeasure rather than reusing v0.6.1 numbers
- DeepSeek speculative algorithms: https://github.com/deepseek-ai/DeepSpec
- vLLM external speculators: https://github.com/vllm-project/speculators

Always distinguish formal release, upstream main, model-specific image, community fork, and open PR.

## Hardware truth sources

- DGX Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- DGX Spark clustering: https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
- DGX Spark CUDA/GDR constraints: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html
- DGX Spark known issues: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
- DGX Spark release notes: https://docs.nvidia.com/dgx/dgx-spark/release-notes.html
- CUDA compute capabilities: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- RTX PRO 6000 family: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/
- RTX PRO 6000 Server: https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
- NVIDIA DGX Spark playbooks: https://github.com/NVIDIA/dgx-spark-playbooks
- Apple Metal capability tables: https://developer.apple.com/metal/capabilities/
- Apple M1/M2 Mac Studio: https://support.apple.com/en-us/111900 and https://support.apple.com/en-us/111835
- Apple M3 Max: https://support.apple.com/en-us/117737
- Apple M3 Ultra and M4 Max Mac Studio: https://support.apple.com/en-us/122211
- Apple M3 Ultra 512 GB: https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
- Apple M5 Max: https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
- Metal working-set guidance: https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize
- Apple power modes: https://support.apple.com/en-us/101613

## Benchmark truth sources

- vLLM benchmark CLI: https://docs.vllm.ai/en/latest/benchmarking/cli/
- vLLM serving implementation: https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py
- SGLang serving benchmark: https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/serving.py
- SGLang offline benchmark: https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/offline_throughput.py
- AIPerf: https://github.com/ai-dynamo/aiperf
- MLPerf inference rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- Magnitude composite benchmark: https://github.com/magnitudedev/magnitude/blob/main/inference/benchmark/README.md
- local-inference-lab benchmark: https://github.com/local-inference-lab/llm-inference-bench

## Entry repositories and reviewed snapshots

| Repository | Reviewed commit | Evidence note |
|---|---|---|
| https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark | `077ed91903a75096d50329d35de7c3d1ffb4d9f4` | launcher + README/CHANGELOG measurements; benchmark scripts referenced in docs are absent |
| https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark | `5c65b4c54964a49eda71cc9887a2102b69044cdc` | launcher and 200–223 tok/s headline; no raw benchmark artifacts |
| https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark | `83aa4f61573c4d4fcc740e45db61f048c20ab9c5` | extensive TP2 deployment, patches, tests and results; several audit scripts/claims drift |
| https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer | `590d2172394dd83c1f36ff29f0dc9ec6032ea9e2` | transformed model `0xSero/deepseek-v4-flash-0731-spark@22f28d32b9b29b4352eaa380ff8c2c170b2847ab`; appliance image `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`; known tool/quality boundaries remain |
| https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark | `047aa237efa334fde70427629e59b6f9606ec21d` | actual runtime is `local-inference-lab/vllm@fa033bd4e1b16d9d729ad94be2d87da5a13210ce`, version `0.1.dev19043+gfa033bd4e`, not official vLLM 0.27; GHCR image manifest `sha256:abd8ea18080ae571a72e8113d80775c38f4cfe07eacd591d4fedd70e95cd1bbf`; benchmark and long-context reproducibility gaps remain |
| https://github.com/magnitudedev/magnitude | `aea7e449806c4b9a14c3d46628dd306630cf3638` | Rust/llama.cpp local inference, exact GGUF fit planning, calibrated recommendations, endpoint-neutral controlled benchmark |
| https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac | `0800fb5ca9a5921a32ff32dbe4d7cb3e5d9feeac` | exact oMLX 0.6.1/MTP settings and saved streams on M4 Max; reported 48/65.5 tok/s leads fail strict promotion because several raw outputs answer filler or miss the requested code task, and fallback token counting uses SSE events |

## High-value related repositories

### Upstream and original implementation

- https://github.com/sgl-project/sglang — Qwen/DeepSeek serving upstream.
- https://github.com/vllm-project/vllm — serving upstream.
- https://github.com/local-inference-lab/b12x — original SM120/121 kernel layer, formerly SparkInfer.
- https://github.com/local-inference-lab/blackwell-llm-docker — auditable pinned Blackwell build source.
- https://github.com/local-inference-lab/rtx6kpro — RTX PRO 6000 integration knowledge base.
- https://github.com/Anemll/dspark-vllm-gx10 — original two-GB10 vLLM/DSpark port used by several recipes.
- https://github.com/jasl/vllm-ds4-sm120-harness — SM12x oracle/correctness/performance promotion harness.
- https://github.com/antirez/ds4 and https://github.com/Entrpi/ds4 — custom single-Spark DeepSeek GGUF engine line; pin an exact reviewed commit/artifact before promoting it above a hypothesis.
- https://github.com/magnitudedev/magnitude — GGUF estimator/recommender/benchmark reference.

### Qwen measurements and adaptation

- https://github.com/0xWhiteMage/Qwen3.8-27B-SGLang-Spark — adds sealed results, quality gates, ReplaySSM and speculative A/B to the Mia launcher.
- https://github.com/malaiwah/qwen38-27b-exl3 — EXL3/KLD/quality and SM120 performance study.
- https://github.com/KyaniteLabs/qwen38-27b-strix-halo — llama.cpp/ROCm counterexample showing cache and verbosity can invert headline speed.
- https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000 — alternative vLLM/nightly recipe.
- https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090 — experimental TurboQuant/MTP lane with unmerged fixes.
- https://github.com/MiaAI-Lab/sparkDash — useful telemetry/backend metric normalization; its built-in benchmark is not canonical.
- https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac — Apple/oMLX MTP entry; use configuration leads and raw failure cases, not its headline as a winner.
- https://github.com/drowzeys/keys-MAC-oMLX-0.6.1-DualANE-Qwen3.8-27B-Abliterated-oQ4e-MTP — related oMLX/ANE recipe with a different ablated artifact; keep outside the Weschera quality contract.
- https://github.com/Layr-Labs/qwen-3.8-mtp-challenge — strongest reviewed Apple benchmark/quality protocol: pinned runtime/artifacts, hidden prompts, alternating sessions, thermal and behavioral gates; its M5 Max depth-two result also demonstrates that MTP can be neutral.
- https://github.com/youssofal/MTPLX — native Qwen MTP implementation, multiple quantization profiles, auto-tuning and extensive raw Apple tests; framework facts are inspectable, but published Qwen performance remains C-level because artifact revisions are not fixed in-repo.
- https://github.com/ARahim3/mlx-dspark — MLX external DSpark/DFlash/lookup implementation and cost-curve auto-cap; source is useful, while Qwen headline results lack pinned artifacts/raw result files.
- https://github.com/sudoingX/qwen38-mtp — llama.cpp Metal negative/low-gain MTP examples across small and Ultra Macs; benchmark token accounting/raw evidence are weaker.
- https://github.com/Alexander-Ollman/qwen3.8-on-m4max — broad MLX/Ollama context, quantization, contention, and prefix-cache experiment patterns; mostly single trials and unpinned model digest.

### DeepSeek on DGX Spark

- https://github.com/Reederey87/dgx-spark-2x-deepseek-v4-flash — maintained source-built TP2 stack and backport ledger.
- https://github.com/hazyumps/deepseek-v4-flash-gb10 — jasl-based TP2/EP deployment.
- https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark — shared-expert/acceptance and long-context work.
- https://github.com/liquidgravityai/2x-dgx-spark-deepseek-v4-flash-0731 — pinned distribution/validation packaging.
- https://github.com/Weschera/DeepSeek-V4-Flash-0731-DSpark-2x-DGX-Spark — strict target-only versus DSpark A/B.
- https://github.com/SvangenStudios/dgx-spark-agent-serving-benchmarks — real agent load and benchmark-accounting corrections.
- https://github.com/bjk110/spark_vllm_docker — broader GB10 build and patch-status lineage.
- https://github.com/r0b0tlab/deepseek-v4-flash-nvfp4-gb10-benchmark — workspace/network/driver diagnosis patterns from an earlier model line.
- https://github.com/Entrpi/ds4-on-spark — single-Spark custom engine deployment.

### DeepSeek on RTX PRO 6000

- https://github.com/ormandj/vllm-deepseek-v4-flash-sm120 — detailed two-card integration, build, patch map and concurrency matrix.
- https://github.com/jacklarmer/deepseek-v4-flash-0731-sm120 — four-card PCIe topology, correctness gates and tuning results.
- https://github.com/wonder-soft/deepseek-v4-flash-bench — raw artifacts, coding/tool/agent workload evaluation.
- https://github.com/macdad222/vllm-threadripper-duet — heterogeneous power-card production deployment.
- https://github.com/0xSero/4x-6000-pro-ds4-flash-0731 — pinned deployment sample with limited independent benchmark evidence.

### Apple Silicon

- https://github.com/ml-explore/mlx — core array/Metal/runtime truth source; reviewed MLX 0.32.0 for the oMLX reproduction chain.
- https://github.com/ml-explore/mlx-lm/commit/84946c8ad94333eef070f85f256ac41cbbcaf89a — reviewed plain MLX-LM text-baseline snapshot.
- https://github.com/Blaizzy/mlx-vlm/commit/20eec6cb5564c6a196b046d869d2081c29e3ff92 — reviewed 0.6.15 multimodal/MTP-sidecar snapshot with ragged acceptance and exact-prefix APC for hybrid cache.
- https://github.com/jundot/omlx — real oMLX upstream; v0.6.1 `b587575f` reproduces the entry and v0.6.2 `f2d36f3d` fixes the reviewed MTP+TurboQuant crash.
- https://github.com/ggml-org/llama.cpp/commit/6d05498314db1b57f81c271080018aa2d0b89be9 — reviewed b10499 Metal/GGUF/MTP/DSpark/server-metrics snapshot.
- https://github.com/ollama/ollama/commit/a5165c53acc5206ce90a684900e0d90b6fb0cb26 — reviewed native Qwen MLX wrapper; currently serialized and not a native DeepSeek-V4 MLX path.
- https://github.com/magnitudedev/magnitude/commit/7d107611680fbb06936faa259905b9c3c5d57ff9 — reviewed Apple-facing planner/control snapshot; Qwen3.8 speculative current choice remains unset.
- https://github.com/ddalcu/mlx-serve — native Zig/MLX serving with MTP/PLD/assistant and adaptive acceptance gates; exact Qwen3.8 performance was not yet established.
- https://github.com/dmitryryabkov/mtp-profiler — reusable llama.cpp log/context/acceptance/memory-pressure analysis method, demonstrated on an earlier Qwen line.
- https://github.com/magnitudedev/magnitude and https://github.com/ggml-org/llama.cpp — portable GGUF/Metal control and fit-planning lane.
- https://github.com/drowzeys/keys-Mac-DeepSeek-V4-Flash-DSpark-0731-MXFP4-MLX-Abliterated-49tps — DeepSeek Mac lead with a changed ablated/quantized artifact; require source/runtime/artifact pins, target-only control, and raw quality before promotion.

## Refresh checklist

Before applying a recipe later:

1. Fetch the current model config and repository revision.
2. Check the latest formal framework release and compare it with recipe-required commits/open PRs.
3. Resolve every container tag to a digest and inspect `linux/arm64` plus target-SM support.
4. Re-open relevant issues for corruption, graph, prefix-cache, parser, DSpark/MTP, and SM120/121 kernels.
5. Re-run the exact local benchmark; do not reuse the numbers in these notes as expected performance.
