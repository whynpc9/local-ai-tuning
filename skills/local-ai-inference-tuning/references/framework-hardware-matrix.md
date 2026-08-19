# Framework and hardware matrix

Snapshot date: 2026-08-19.

Status vocabulary:

- `released`: present in a formal framework release for the stated combination;
- `main/nightly`: implemented upstream but not yet in a formal release;
- `model image`: official or project-published special image outside the ordinary release;
- `community`: fork, patch stack, or third-party image;
- `unknown`: general feature exists, exact combination not verified.

Framework cells below describe architecture/runtime support on capacity-suitable hardware. They do not override the hardware/model admission gate; in particular, official DeepSeek support in a framework does not make its full checkpoint feasible on one Spark or one 96 GB card.

## Framework decision table

| Stack | Qwen3.8-27B | DeepSeek-V4-Flash-0731 | Best use | Main boundary |
|---|---|---|---|---|
| SGLang | model-specific image/cookbook; BF16, official FP8, community NVFP4; GDN state, MTP/EAGLE, external DSpark, and DFlash 2 on main | released DSPARK path and official model cookbook | Qwen on DGX Spark/RTX; DeepSeek on supported multi-GPU hardware | DFlash 2 entered main on 2026-08-19 but quantized target LM-head follow-ups remain open; exact GB10 performance not supplied by H200 evidence |
| vLLM | architecture released; current correct GDN/MTP path may require post-release main or Qwen image; DFlash 2 is open-PR-only at the snapshot | released architecture/DSpark; official recipes; GB10 uses community B12X image | DeepSeek full checkpoint; Qwen challenger | distinguish installed commit from version/title; do not describe DFlash 2 as released while `#52816`/`#52883` are open |
| B12X / SparkInfer | SM120/121 NVFP4/MXFP8 and related kernels via a vLLM fork | sparse/compressed MLA, MoE, collectives, FP8/NVFP4-related kernels via vLLM fork | high-performance Blackwell consumer/GB10 kernel layer | kernel library, not an API server; project states non-production/datacenter boundary |
| TensorRT-LLM | exact Qwen checkpoint has unresolved/patch-only paths | DeepSeek support is current RC/main and strongest on data-center Blackwell | later challenger on NVIDIA-supported exact model/hardware | DGX Spark beta/single-node validation and exact target gaps; do not promote by vendor name alone |
| Magnitude / llama.cpp | cataloged GGUF Q4/Q6/Q8; exact GGUF planner; no current catalog draft for Qwen | cataloged GGUF Q4/Q8 with DSpark draft; generic endpoint benchmark | simple private local agent, GGUF, partial placement, fit estimation, controlled comparison | different artifacts and kernels; cataloged DeepSeek Q4/Q8 exceed one Spark before runtime; no proof it maximizes CUDA NVFP4/FP8 or multi-node tok/s |
| MLX / MLX-LM | Qwen3.8 text backbone control; current plain loader removes vision and embedded `mtp.*` | research loader/control only | auditable native Apple Metal target-only and external-draft baseline | generic family support is not the full VLM/MTP product; basic server is not a production throughput reference |
| MLX-VLM | full Qwen3.8 VLM/mRoPE plus split MTP sidecar, ragged acceptance, and current speculative/batch server paths | DeepSeek V4 MTP split exists but is less operationally mature than oMLX | inspectable multimodal/sidecar challenger | structured output disables speculative paths; hybrid Qwen APC uses exact-prefix snapshots rather than composable block mode; pin exact commit and verify hits/restart/output |
| oMLX | public v0.6.1/0.6.2 source exposes OpenAI/Anthropic APIs, batching, cache, hybrid-model support, native MTP, and optional ANE prefill | vendored DeepSeek model/MTP patches exist, but no promoted exact official-artifact Apple lane | high single-Mac Qwen throughput challenger | entry README links the wrong account; pin `jundot/omlx` tag and vendored dependencies; source-reported Qwen raw outputs fail strict quality review |
| Ollama native MLX | Qwen3.8 alias, vision, embedded self-draft, hybrid recurrent cache | exact dispatch/runtime gate | easy local UX and smoke lane | current native MLX scheduling is effectively single-request/no continuous batching; benchmark direct engine before wrapping it |
| llama.cpp directly | rolling Qwen3.5-family/GGUF and MTP work | rolling DeepSeek-V4/GGUF; DSpark work remains evolving | CPU/Metal/CUDA portability and offload | rolling-master support, model-specific parser/speculation issues, not standard TP/EP |
| ds4 / Entrpi | not applicable | custom single-Spark asymmetric GGUF + DSpark engine | make a transformed 0731-class artifact fit one Spark | different artifact/quality boundary; smaller serving feature surface |
| EXL3/Trellis custom stacks | Qwen experimental quantization lane | single-Spark transformed DeepSeek target | very aggressive memory/performance research | quality and kernel correctness require extensive gates |

## Qwen3.8 feature matrix

| Capability | SGLang model image | vLLM Qwen build | Magnitude/llama.cpp |
|---|---|---|---|
| BF16 / official FP8 | yes | yes on validated GPU recipes | converted GGUF lane instead |
| NVFP4 | community checkpoint, SM120/121 recipe | community checkpoint, Blackwell-specific | GGUF quantizations, not the same format |
| native MTP | EAGLE-style parameters; current model recipe | `method=mtp`; installed-revision gate | upstream/rolling behavior; catalog has no Qwen3.8 draft |
| external DSpark | current DGX recipe covers pinned `RadixArk/Qwen3.8-27B-DSpark`; ReplaySSM compatibility and depth are explicit sweeps | not the default official Qwen lane | no catalog declaration |
| DFlash 2 | main supports the public `incoai/Qwen3.8-27B-DFlash2`; full/quantized target compatibility and exact SM kernels are gates | open PR only at the snapshot | open llama.cpp PR only; public Q4/Q8/BF16 GGUF draft exists |
| hybrid state cache | GDN modes `S=5/4/3`; pool is `C*(S+D)` unless compatible ReplaySSM removes draft snapshots from that pool; SSM dtype/radix/memory-ratio controls | hybrid cache; post-release fixes can matter | llama.cpp recurrent/hybrid implementation |
| prefix cache | Radix cache with hybrid-state rules | align mode/version gate | prompt cache |
| tool/reasoning parser | `qwen3`, `qwen3_coder` | `qwen3`, `qwen3_coder` | endpoint/template dependent |
| VLM | architecture/cookbook path; validate exact modality | can use full VLM or language-only | catalog includes projector; current llama.cpp supports Qwen mmproj and text MTP separately, but image-token + MTP remains unsupported/experimental |

## DeepSeek feature matrix

| Capability | SGLang | vLLM + B12X/community | TensorRT-LLM | Magnitude/llama.cpp / ds4 |
|---|---|---|---|---|
| official fused checkpoint | yes on supported multi-GPU | yes; dominant GB10/RTX community recipes | current RC/main data-center Blackwell | converted/transformed GGUF lane |
| DSpark | `DSPARK`; do not substitute EAGLE | `method=dspark`; official 0731 default, GB10 custom image | current RC fixes; exact gate | evolving GGUF support/custom engine |
| MLA/sparse attention | model-specific implementation | SparkInfer/B12X for SM120/121 | data-center Blackwell path | llama.cpp/custom representations |
| TP/DP/EP | model recipes include TP/attention-DP/EP | TP/DP/EP; GB10 commonly TP2 | TP/PP/EP/attention-DP | device split/custom batching, not equivalent semantics |
| parser | `deepseek-v4` reasoning and `deepseekv4` tool parser | DeepSeek-V4 tokenizer/reasoning/tool modes | tool formatting gaps must be checked | template/endpoint dependent |
| long context | up to checkpoint limit subject to cache | up to checkpoint limit subject to cache | exact RC limitations | product/artifact-specific; Magnitude catalog uses 100K |

## Apple Qwen feature matrix

| Capability | MLX family | oMLX | llama.cpp / Magnitude Metal |
|---|---|---|---|
| exact Qwen3.8 architecture | prove in the installed revision | demonstrated by the reviewed oMLX 0.6.1 community recipe | prove in pinned rolling-master/GGUF metadata |
| artifact | official-source conversion or pinned MLX quantization | pinned `Jundot/Qwen3.8-27B-oQ4e-mtp` in the reviewed entry | pinned GGUF Q4/Q6/Q8; projector separately when needed |
| no-speculation control | required target-only lane | same artifact with `mtp_enabled=false` | required |
| speculative decoding | sidecar/native implementation only when pinned and acceptance is exposed; public DFlash 2 MLX backend is a separate challenger | native MTP depth sweep 1/2/3; DFlash 2 only through separately pinned `z-lab/omlx-fork` at the snapshot | DFlash 2 public GGUF exists but requires the open llama.cpp PR and exact pair gate |
| ANE | do not infer use from chip capabilities | optional prefill path in a source-build lane; treat as TTFT A/B | not the ordinary Metal decode path |
| memory evidence | MLX active/cache/peak + Metal + OS footprint | server footprint + OS/Metal counters | process footprint + Metal buffers/cache |
| provenance boundary | open framework revisions and `z-lab/dflash` can be pinned | keep upstream `jundot/omlx` releases separate from `z-lab/omlx-fork@0.6.2-dflash2`; pin target/draft/runtime independently | open build and GGUF metadata, but converted-artifact quality remains separate |

## Hardware facts that change the decision

| Hardware | Exact target | Memory / bandwidth | Selection consequence |
|---|---|---|---|
| DGX Spark / GB10 | ARM64, SM121 | 128 GB coherent LPDDR5X UMA, 273 GB/s | capacity-friendly but bandwidth-limited; require ARM64+SM121 image, zero swap, UMA-aware accounting |
| 2× DGX Spark | two separate SM121 UMA nodes | 256 GB sharded total; 200 GbE/RoCE link | full DeepSeek fits by sharding, but no coherent pool and no normal device-memory GDR; measure NCCL |
| RTX PRO 6000 Blackwell Server | SM120 | 96 GB GDDR7, 1,597 GB/s, 400–600 W | much stronger dense C1 bandwidth; exact server cooling and PCIe topology matter |
| RTX PRO 6000 Workstation / Max-Q | SM120 | 96 GB, 1,792 GB/s; 600 W / 300 W | same nominal memory bandwidth but different sustained compute/power; do not call all cards one SKU |
| H100 SXM | SM90 | 80 GB HBM3, 3.35 TB/s, NVLink | FP8/BF16 path; no native Blackwell NVFP4 assumption |
| B200 SXM | SM100 | 180 GB HBM3e, up to 8 TB/s, NVLink | data-center Blackwell FP4 kernels; SM100 code is not automatically SM120/121 code |
| Apple Max family | native ARM64, Apple7-Apple10 | 64-128 GB UMA; 300-614 GB/s depending exact chip/bin | strong single-node Metal lane; exact GPU cores, chassis, power, thermal state, and zero swap matter |
| Apple Ultra family | native ARM64, Apple7-Apple9 | 128-512 GB UMA; 800-819 GB/s | high-memory full/converted-model lane; no M4 Ultra exists, and installed memory is still shared with macOS/runtime |

NVIDIA peak FP4 TOPS/PFLOPS often includes sparsity and is not a tok/s measurement.

Official hardware sources:

- https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
- https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/
- https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
- https://support.apple.com/en-us/111900
- https://support.apple.com/en-us/111835
- https://support.apple.com/en-us/122211
- https://developer.apple.com/metal/capabilities/

See [apple-silicon.md](apple-silicon.md) for exact Max/Ultra bins, privacy-safe collection, MLX/Metal capacity, thermal, and multi-Mac gates.

## Default shortlist by target

| Model + hardware | First lane | Challenger | Immediate exclusion / separate lane |
|---|---|---|---|
| Qwen + 1 Spark | pinned SGLang Qwen image, NVFP4, FP8 KV, MTP sweep | pinned community vLLM GB10 build; Magnitude GGUF control | TensorRT-LLM until exact blockers clear |
| Qwen + 1 RTX PRO 6000 | SGLang SM120, official FP8 control + NVFP4, MTP vs external DSpark | vLLM Qwen build; Magnitude GGUF | any SM100-only kernel |
| DeepSeek + 2 Sparks | pinned vLLM+B12X TP2, official artifact, DSpark sweep | pinned SGLang DSPARK TP2 | single-node full official artifact |
| DeepSeek + 4/8 RTX PRO 6000 | vLLM/SGLang exact SM120 full-checkpoint lanes | TensorRT-LLM only after exact RC validation | SM100-only NVFP4 MLA assumptions |
| DeepSeek + 1 Spark | reviewed 0xSero transformed EXL3/Trellis target served by its pinned vLLM+B12X appliance, after artifact-specific quality gates | pinned ds4/Entrpi asymmetric low-bit GGUF; Magnitude planner only for a separately identified smaller artifact | official fused checkpoint and Magnitude catalog Q4/Q8 by capacity; any unpinned artifact/runtime tuple |
| Qwen + 1 Apple Silicon Mac | same-artifact oMLX target-only control, then native MTP 1/2/3 after quality | pinned MLX/MLX-VLM control and sidecar; Magnitude/llama.cpp Metal GGUF | Rosetta, swap-assisted results, unpinned oMLX package, or the Weschera headline without new correctness results |
| DeepSeek + 1 Apple Silicon Mac | official artifact only on a high-memory exact SKU after loader/fit proof | pinned MLX or GGUF conversion as a separate quality contract | official checkpoint on any 128 GB Mac; a 192 GB configuration without measured runtime reserve |

These are starting hypotheses. A local common benchmark determines a winner only inside one exact artifact/quality contract. Across transformed artifacts, report a disclosed best-achievable Pareto comparison instead.
