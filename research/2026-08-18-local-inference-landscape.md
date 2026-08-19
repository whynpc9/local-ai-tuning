# 本地大模型推理适配调研：Qwen3.8-27B 与 DeepSeek-V4-Flash-0731

调研截止：2026-08-19

Apple Silicon 的后续入口、oMLX/MLX/Metal框架矩阵、硬件/热/统一内存门禁和新增工具已单独整理在 [Apple Silicon扩展调研](2026-08-19-apple-silicon-extension.md)。本报告保留NVIDIA主线并在结论/shortlist中纳入Apple。

## 结论先行

本次调研最重要的结论不是某个固定启动命令，而是以下七条：

1. **不存在脱离 workload 的“最高 tok/s 框架”。** 单请求 decode、离线 aggregate throughput、在线 SLO goodput 会选出不同参数，甚至不同框架。
2. **Qwen3.8-27B 在 DGX Spark 上应先对比 SGLang 专用镜像与社区 vLLM GB10 构建。** 前者当前有最完整的模型 cookbook，后者有更高的社区 headline；两边并未使用同一 benchmark，不能直接宣布赢家。
3. **官方 DeepSeek-V4-Flash-0731 checkpoint 无法装入一台 128 GB Spark。** 两台 Spark 的 TP2 是完整模型的自然起点；单 Spark 项目都通过剪专家、极低比特、EXL3/Trellis 或非对称 GGUF改变了模型合同。生成器默认在此停止，只有显式接受“换成不同 artifact”后才会解锁替代 lane。
4. **“Blackwell 支持”没有工程意义，必须精确到 SM100、SM120、SM121 和 host ABI。** B200 的 kernel、RTX PRO 6000 的 kernel、DGX Spark ARM64 kernel不能相互默认兼容。
5. **speculative decoding 是正确性功能，不只是性能开关。** 高 acceptance/high tok/s 可能来自 KV、graph、routing 或输出损坏；必须有 target-only 基线和严格输出门禁。
6. **Magnitude 是很好的方法参考和独立 GGUF/llama.cpp 候选。** 它已经实现了硬件校准、精确 GGUF fit、模型/量化/上下文联合选型和受控 endpoint benchmark，但它不是专用 CUDA NVFP4/DSpark 栈的天然性能赢家。
7. **Apple Silicon必须作为MLX/Metal独立路径。** Qwen应先在同一artifact上比较oMLX target-only与native MTP，再测MLX-VLM/MTPLX/外置DSpark和llama.cpp；Weschera的48/65.5 tok/s raw结果没有通过质量门。统一内存进入swap/持续压缩的lane不得参与最快tok/s排名。

### 当前可辩护的起始候选

| 目标 | 首选起始 lane | 必测 challenger | 关键边界 |
|---|---|---|---|
| Qwen + 1×DGX Spark | SGLang `qwen38-27b` 专用 ARM64/SM121 镜像；官方FP8 control + NVFP4；MTP与外置RadixArk DSpark/ReplaySSM分别测 | 固定 digest/commit 的社区 vLLM GB10 构建；Magnitude GGUF | 两个主要框架都不能仅按普通 release 版本判断 exact-model 支持 |
| Qwen + 1×RTX PRO 6000 | SGLang SM120，官方 FP8 control + 社区 NVFP4，native MTP vs external DSpark | vLLM Qwen build；Magnitude GGUF | `200–223 tok/s` 现有仓库缺 canonical raw benchmark，且其 state pool硬编码需按安装版ReplaySSM/mode重新求导 |
| DeepSeek + 2×DGX Spark | 固定社区 vLLM+B12X/Anemll stack，官方 checkpoint，TP2，DSpark off→K3/5/7 | 固定 SGLang DSPARK TP2 | 200 GbE/RoCE 主要解决容量；GB10 无普通 device-memory GDR，需实际测 NCCL |
| DeepSeek + 4/8×RTX PRO 6000 | exact SM120 vLLM/SGLang full-checkpoint lane | TensorRT-LLM 仅在 exact RC/功能验证后 | 默认按 PCIe-only；不能套 SM100 NVFP4 kernel |
| DeepSeek + 1×DGX Spark | 明确接受 artifact 范围变化后：0xSero 固定 revision/digest 的 EXL3/Trellis 剪专家 target + vLLM/B12X appliance | 同样需显式接受：固定 ds4/Entrpi 非对称低比特 GGUF；Magnitude planner可评估另一个明确制品 | 默认停在官方 artifact 容量门；官方和Magnitude catalog Q4/Q8均容量拒绝；其余lane都非质量等价并需独立评价 |
| Qwen + 1×Apple Silicon | 固定`jundot/omlx` v0.6.2和同一oQ4e artifact，target-only→adaptive max-depth 1/2/3 | MLX-VLM sidecar、MTPLX、mlx-dspark；Magnitude/llama.cpp Metal control | 禁止Rosetta/swap；入口结果有prompt污染和thinking/任务正确性问题 |
| DeepSeek + Apple Silicon | 256GB+优先固定oMLX DeepSeek path和pinned q4/oQ4e conversion，target-only→Lightning MTP/DSpark | direct llama.cpp DSpark；mlx-vlm research control | 官方约167GB在128GB硬拒绝；低位conversion必须另设质量合同 |

表中的“起始候选”是可检验假设，不是本机未实测的性能结论。

## 调研方法与证据等级

这次工作从用户给定的五个入口仓库向三层追踪：

1. 作者的其他相关仓库；
2. 入口仓库实际引用的 framework/kernel/image/model upstream；
3. 其他作者有源码、raw artifact、correctness gate 或独立复测的项目。

证据等级：

| 等级 | 证据 | 本报告用语 |
|---|---|---|
| A | 官方模型/config、正式 framework source/release、硬件厂商文档 | “支持”，但附版本和组合 |
| B | 固定 commit/image/model revision，有脚本、raw results、正确性检查 | “该仓库复现/报告；本机候选” |
| C | launcher + README 数字，缺 raw samples 或固定运行时 | “作者报告” |
| D | issue、截图、浮动 tag、计划中 patch | “线索/待验证” |

没有在本工作区的真实 DGX Spark/RTX PRO 6000 上复跑社区性能数字。下文所有数字都保留其来源口径。

## 五个入口仓库审计

| 仓库 | 实际技术组合 | 报告结果 | 主要价值 | 不能直接相信的地方 |
|---|---|---|---|---|
| [Mia Qwen / DGX Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | SGLang 专用镜像、RadixArk NVFP4、FP8 KV、native MTP 3/1/4、GDN state | thinking 17.2–20.5、non-thinking 21.6–22.7、tools 26–28 tok/s | 参数结构、MTP depth sweep、state pool 修正、CPU 大核绑定线索 | README 提及的 benchmark 脚本未提交；镜像 tag 浮动；prompt/raw results 不完整 |
| [Mia Qwen / RTX PRO 6000](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark) | SGLang、RadixArk NVFP4、external DSpark block 7、FP8 KV | 200–223 tok/s | 值得复测的高带宽 SM120 组合 | 无 ISL/OSL/thinking/acceptance/raw artifacts；state pool必须按安装版求导：无ReplaySSM通常`C×(S+D)`，兼容ReplaySSM才在该pool令`D=0` |
| [Mia DeepSeek / 2×Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | 官方 0731、Anemll vLLM、B12X、TP2、`nvfp4_ds_mla`、DSpark K5、regular CUDA graph | 短 prompt C1 约75–83；C6 aggregate最高约191；长冷 prefill并发显著塌缩 | 完整部署/patch/长上下文/多机材料，最接近生产 appliance | 不同结果混用 decode 与含 TTFT aggregate；audit 里有目标 token 未达、近似 token、acceptance 统计等源码问题；多条历史 runtime 路线并存 |
| [0xSero DeepSeek / 1×Spark](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer) | K216 REAP、3bpw EXL3/Trellis、K64 draft、SparkInfer/vLLM、K5；正式 KV 实际为 584 B padded FP8 | clean image 五个 code workload median 38.12，min 34.30 tok/s | 最强的 fail-closed、hash、image digest、内存组合和 kernel correctness案例 | 不是官方权重；真正 432 B NVFP4 KV 会乱码；最终 min 未过仓库35 tok/s gate；发布镜像 tool call 有已知 HTTP 500 |
| [drowzeys Qwen / vLLM](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark) | `local-inference-lab/vllm@fa033bd` 自定义开发分支、FlashInfer/B12X、unsloth mixed NVFP4/FP8、MTP3、FP8 KV | baseline 11.1→MTP3 31.7 C1；C64 aggregate 313 tok/s | vLLM challenger和 depth/concurrency线索 | 不是官方 vLLM 0.27；模型用浮动 `main`，长上下文脚本的 `--rope-scaling` 不在精确 runtime CLI，spin-wait仅有文档未进镜像；p99/1K/prefix/prefill均有口径问题 |

这些 headline 不能横向排名。例如 31.7 与 22.7 可能来自框架差异，也可能来自 prompt entropy、thinking、输出长度、镜像 revision、cache、计时窗口和模型转换差异。

## 技术关系图

```text
官方模型/config
├── SGLang
│   ├── Qwen3.8 专用 image/cookbook
│   ├── Mia Qwen launchers
│   └── 0xWhiteMage sealed A/B + quality harness
├── vLLM
│   ├── official recipes / model registration
│   ├── local-inference-lab/vLLM fork
│   │   └── B12X (原 SparkInfer，SM120/121 kernels)
│   ├── Anemll dspark-vllm-gx10
│   └── Mia / drowzeys / ormandj / liquidgravity / Reederey recipes
├── llama.cpp / GGUF
│   ├── Magnitude ICN + model planner + controlled benchmark
│   └── ds4 / Entrpi single-Spark DeepSeek line
└── TensorRT-LLM
    └── exact model/hardware support仍需逐项验证，不能因 NVIDIA 官方身份自动入选
```

## 本地推理适配涉及的技术

### 1. 模型 artifact 与量化

必须把以下合同分开：

- 官方/社区 checkpoint；
- BF16、block FP8、NVFP4、GGUF Q4/Q6/Q8、EXL3/Trellis；
- 是否做 QAT、校准、expert pruning/REAP；
- tensor storage、operation dtype、scale layout、padding；
- target 与 draft 是否同 revision/同 tokenizer；
- KV control name 与实际 physical record bytes/layout。

0xSero 是最清楚的反例：控制名叫 `nvfp4_ds_mla`，但正式可用格式是 584 B padded FP8；真正 432 B NVFP4 低层 oracle通过却端到端乱码。skill 因此同时记录“配置名、物理 layout、端到端语义”。

### 2. 模型结构专用内存

普通 Transformer KV 公式不足以描述这两个模型：

- Qwen 只有16层 full attention，另有48层 GDN state；需要分别预算 KV page、持久 state slot 与 ReplaySSM ring。当前 SGLang 模式的 `S` 常为5/4/3；无ReplaySSM时 pool随 `C×(S+D)`，兼容ReplaySSM才在该pool中消掉逐draft snapshot。
- DeepSeek 是 MoE + compressed/sparse MLA；总专家决定 resident capacity，active expert决定部分 decode工作量，MLA record决定上下文容量。
- CUDA graph、kernel workspace、draft/MTP、vision tower、allocator碎片和 OS/UMA reserve都在权重之外。

Magnitude 的 GGUF planner值得借鉴：从精确 tensor storage 和 no-allocation graph估算，而不是从参数位数猜；但它自己也明确 projector、draft、transient allocation等边界。

### 3. kernel 与 exact SM

关键路径包括：

- FlashInfer attention/MLA；
- Triton/CUTLASS/CuTe DSL；
- B12X 的 SM120/121 NVFP4/MXFP8 GEMM、MoE、sparse/compressed MLA、PCIe collective；
- graph-safe `plan → bind → run`；
- FlashInfer/Triton linear-attention verify；
- model-specific qnorm/RoPE/cache writer和router。

必须核验：host arch、image manifest、driver/CUDA、torch、FlashInfer、CUTLASS、Triton、NCCL、编译的 SM/PTX。`sm_100a`、`sm_120a`、`sm_121a` 不是同一个 target。

### 4. speculative decoding

Qwen有两条主要 lane：

- checkpoint native MTP，通过 SGLang EAGLE-style或 vLLM `method=mtp`；
- external DSpark drafter。

DeepSeek 0731 官方 artifact 是 fused DSpark，不应机械改成普通 MTP/EAGLE。调参至少记录：

- draft depth/K、steps、top-k、sample method；
- drafted/accepted/rejected token delta；
- mean accepted length及 position-wise acceptance；
- draft latency、target verify latency；
- code/prose/tool/real trace分别表现。

depth 越大不一定越快。Mia Qwen 的本机 sweep在3步达到峰值，4–6反而回退；另一些高带宽 RTX workload可能让 external DSpark胜出。

### 5. graph、scheduler 与 cache

性能敏感项包括：

- full/regular/piecewise CUDA graph与 eager；
- graph capture batch shapes和 speculative verify shape；
- chunked prefill size、max batched tokens、long-prefill threshold；
- max sequences/requests；
- prefix/radix cache及 hybrid state对齐；
- partial-prefill公平性、decode lane token reserve；
- cache/state dtype、context ceiling与实际 live token。

最大 context 能启动，不表示能并发若干条最大 context。应报告 cache总 token和 workload下 live token，而非只报 `max_num_seqs`。

### 6. 并行与互联

- 单卡能装下时先测单卡，避免通信破坏 C1。
- DeepSeek full checkpoint在双 Spark上常用 TP2；更大 GPU系统可比较 TP、attention-DP、EP和混合布局。
- RTX PRO 6000没有被官方规格保证 NVLink，应按 PCIe处理并实测 P2P。
- 双 Spark 不是一致256 GB内存池。直连是200 GbE/RoCE；GB10 UMA不支持普通 `cudaMalloc` device memory的 GPUDirect RDMA，需要 host-buffer路径。必须跑真实 NCCL collective，而不只是 `iperf3`。

### 7. provenance 与运维

最终记录至少包含：

- model/tokenizer/draft revision和 shard/hash；
- image digest、framework/kernel commit、patch hash；
- driver/CUDA/NCCL/FlashInfer/CUTLASS/Triton；
- exact launch command和环境变量；
- hardware SKU/SM/ABI/topology/power/cooling；
- allocator/cache启动日志；
- benchmark workload/hash/raw events；
- telemetry、错误、重启、回退路径。

浮动 `latest`、仓库名里的版本、或容器内被临时覆盖的 Python代码都不能充当 provenance。

## 模型、框架、硬件支持判断

### Qwen3.8-27B

官方模型是约27.78B dense VLM，64层混合结构：48 Gated DeltaNet + 16 full attention，原生262,144 context，含一个 MTP layer。官方有 BF16和FP8；当前常见 NVFP4 都是社区 artifact。

| 框架 | 当前判断 |
|---|---|
| SGLang | exact-model cookbook和专用 image最完整；DGX Spark/RTX PRO 6000/5090/H200覆盖；Qwen核心适配仍可能领先正式 release，必须 pin image |
| vLLM | 架构已注册；正确 GDN speculative修复可能在正式0.27.1之后；官方 recipe测试用 Qwen专用 build，未给官方 GB10性能 |
| Magnitude/llama.cpp | Q4/Q6/Q8 GGUF、projector、100K product profile；适合作为简单/隐私/fit-planning lane，不等同官方 NVFP4/FP8 |
| TensorRT-LLM | exact Qwen3.8-27B NVFP4尚有 patch/blocker；当前不作为首选 |

### DeepSeek-V4-Flash-0731

官方 checkpoint是1M context、256 routed experts、每 token选6、带 fused DSpark。基础常写284B/13B active；含 draft的实际 stored tensor elements/parameters约304B、仓库约167 GB。容量以实际 shard为准。

| 框架 | 当前判断 |
|---|---|
| SGLang | DSPARK已进入正式版本，官方 cookbook覆盖0731；具体2×GB10仍要 pin和压力验证 |
| vLLM | 正式架构/DSpark和官方 recipe完整；DGX Spark profile明确依赖社区 B12X image，因此GB10仍是 community lane |
| B12X/SparkInfer | 最针对SM120/121的 kernel层；不是完整 server，API/scheduler/spec来自 vLLM fork；上游声明非 production/datacenter |
| TensorRT-LLM | 当前支持集中在RC/main和数据中心 Blackwell；tool formatting等能力需单独验收 |
| Magnitude/llama.cpp | catalog含GGUF Q4/Q8+DSpark draft；Q4/Q8 target约155/162 GB，单 Spark在runtime/draft前已容量拒绝；fit/benchmark方法仍优秀，但artifact、100K profile和多机能力不是 full-checkpoint vLLM/SGLang等价物 |
| ds4 | 单 Spark的高度优化非对称GGUF/DSpark自定义引擎；容量优势明显，serving/质量合同独立 |

完整字段见 skill 的 [model profiles](../skills/local-ai-inference-tuning/references/model-profiles.md) 和 [framework matrix](../skills/local-ai-inference-tuning/references/framework-hardware-matrix.md)。

## 值得继续跟踪的项目

### 原始实现与基础设施

| 项目 | 价值 |
|---|---|
| [SGLang](https://github.com/sgl-project/sglang) / [vLLM](https://github.com/vllm-project/vllm) | 功能状态的最终框架来源 |
| [B12X](https://github.com/local-inference-lab/b12x) | SM120/121 kernel原始来源，旧名 SparkInfer |
| [blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) | 可审计的 vLLM/B12X/FlashInfer/CUTLASS/NCCL build pins |
| [Anemll dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) | 多个双 Spark recipe的原始 GB10 vLLM port |
| [jasl SM120 harness](https://github.com/jasl/vllm-ds4-sm120-harness) | oracle、logprobs、tools、reasoning、长上下文、driver-health promotion gates |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) / [vLLM speculators](https://github.com/vllm-project/speculators) | DSpark/DFlash/EAGLE/MTP算法与 draft训练/评估 |
| [DFlash 2](https://inco.ai/blog/dflash2/) / [公开 checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | Qwen3.8-27B 的 block-parallel drafter challenger；公开可复现边界与采用门见 [专题调研](2026-08-19-dflash2.md) |
| [Magnitude](https://github.com/magnitudedev/magnitude) | GGUF exact fit、硬件校准、联合推荐、受控 endpoint benchmark |

### Qwen实测补充

- [0xWhiteMage/Qwen3.8-27B-SGLang-Spark](https://github.com/0xWhiteMage/Qwen3.8-27B-SGLang-Spark)：在 Mia launcher上增加 sealed JSON、ReplaySSM、speculative A/B、GSM8K/HumanEval。
- [malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3)：EXL3/KLD/量化质量、SM120性能、context/vision gate。
- [KyaniteLabs/qwen38-27b-strix-halo](https://github.com/KyaniteLabs/qwen38-27b-strix-halo)：证明 warm prefix与输出啰嗦度会让 headline tok/s偏离任务完成时间。
- [Mia sparkDash](https://github.com/MiaAI-Lab/sparkDash)：vLLM/SGLang/ds4/llama.cpp metrics归一化与UMA telemetry值得复用；其内建 bench不够严格。
- [drowzeys 的后继 Qwen appliance](https://github.com/drowzeys/keys-vLLM.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark)：比入口仓库多了 Dockerfile、overlay、cold/warm、KV dtype、长上下文和质量对照；artifact已换成 abliterated模型，数字不能回填到入口模型。
- [drowzeys/vllm-gb10-spin-wait-fix](https://github.com/drowzeys/vllm-gb10-spin-wait-fix)：主要是多 rank/TP2 的 CPU自旋与散热线索，不是入口仓库单 Spark TP1的已应用 tok/s优化。

### DeepSeek / DGX Spark

- [Reederey87/dgx-spark-2x-deepseek-v4-flash](https://github.com/Reederey87/dgx-spark-2x-deepseek-v4-flash)：持续维护的源码构建、backport ledger、1M/观察性。
- [liquidgravityai/2x-dgx-spark-deepseek-v4-flash-0731](https://github.com/liquidgravityai/2x-dgx-spark-deepseek-v4-flash-0731)：固定分发物和 release validation。
- [tonyd2wild/DeepSeek...2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)：shared-expert loader修复使 acceptance和 tok/s大幅变化，是“先测 acceptance”的强案例。
- [SvangenStudios/dgx-spark-agent-serving-benchmarks](https://github.com/SvangenStudios/dgx-spark-agent-serving-benchmarks)：真实 agent mixed workload、fairness、SSE token计数纠错。
- [Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark)：单 Spark非对称GGUF路线。

### DeepSeek / RTX PRO 6000

- [ormandj/vllm-deepseek-v4-flash-sm120](https://github.com/ormandj/vllm-deepseek-v4-flash-sm120)：两卡 build/patch/upstream map和完整并发矩阵。
- [jacklarmer/deepseek-v4-flash-0731-sm120](https://github.com/jacklarmer/deepseek-v4-flash-0731-sm120)：四卡PCIe拓扑、10项 correctness、SSE/token与cache-bust经验。
- [wonder-soft/deepseek-v4-flash-bench](https://github.com/wonder-soft/deepseek-v4-flash-bench)：raw artifacts、coding/repair/tool/agent loop，最接近质量+性能联合评价。

完整去重后的仓库列表和刷新入口见 [source notes](../skills/local-ai-inference-tuning/references/source-notes.md)。

## Magnitude 带来的具体改进

新增的 [magnitudedev/magnitude](https://github.com/magnitudedev/magnitude) 在本任务中有四层价值。

### 直接执行 lane

它包含 Rust inference service，并固定 llama.cpp/binding revision；catalog已经有：

- Qwen3.8-27B GGUF Q4/Q6/Q8 + multimodal projector；
- DeepSeek-V4-Flash-0731 GGUF Q4/Q8 + DSpark Q8 draft。

这使它成为“易用、完全本地、GGUF、多硬件/部分 offload”候选，而不只是测试工具。但其 cataloged DeepSeek Q4/Q8 target约155/162 GB，另有约10.9 GB draft；在128 GB Spark上应由 planner直接拒绝，不能因为 llama.cpp支持offload就假定出现额外物理内存。

### fit planner

它读取 exact GGUF metadata，通过同一 pinned llama.cpp `common_fit`路径构建 no-allocation graph，并显式处理 context、batch/ubatch、sequence、GPU layer/split、KV dtype、Flash Attention、SWA/unified KV、MTP/draft。这个设计比参数位数估算可靠得多。

### tok/s估计

它先校准目标硬件的实际 tensor bytes/s/launch cost，再根据 tensor operation bytes、active experts、KV/MLA/recurrent state、context depth和跨内存域 placement生成带置信区间的 C1估计。复杂结构、fallback calibration和跨域 placement会主动降低 confidence。

### benchmark

它的 E1–E7 suite使用 exact copy和 tool contract，区分 prefill/decode/context/batching/interference/prefix reuse/cancel recovery；保存 raw SSE；做匹配 AB/BA并可按 ratio CI自适应增加重复。runner支持 generic OpenAI-compatible endpoint，因此可比较 Magnitude、SGLang和vLLM。

采用边界：Magnitude catalog的100K是产品 profile，不是模型 native上限；quality score不是每个 GGUF quant独立实测；速度估计含政策常量；它不证明 llama.cpp在 SM120/121专用 NVFP4/DSpark上超过 SGLang/vLLM。详见 [Magnitude patterns](../skills/local-ai-inference-tuning/references/magnitude-patterns.md)。

## 测试与评价协议

### 先过正确性门

至少包含：

- exact byte copy或 deterministic code/unit test；
- strict JSON schema；
- tool call、tool_choice、多轮 tool result；
- thinking/reasoning各模式；
-乱码、重复、NaN、异常 EOS；
- begin/middle/end/checksum长上下文检索；
- target-only与 speculation输出一致性/质量容差；
- cancel后立即恢复；
- cold boot至少三次，极限内存配置尤其如此。

不能把正确答案写入 JSON Schema `const` 后再声称检索/推理成功；代码输出需要编译/测试，而不只是保存hash。

### 五类性能测试

| lane | 目标 | 推荐最小设置 |
|---|---|---|
| C1 | 单流交互 decode与TTFT | 2K/512、8K/512、32K/512；unique nonce；5–10次 |
| Offline | direct engine上限 | 固定ISL/OSL；≥256请求或稳定时长 |
| Online closed-loop | 并发容量 | C=1/2/4/8/...直到吞吐不再增长或SLO崩溃 |
| Online open-loop | 生产 goodput | 饱和点50/70/85/95/110% Poisson rate；≥1000完成或5–10分钟 |
| Stability | 热、功耗、内存、错误 | 目标负载30–60分钟 |

核心定义：

```text
TPOT = (E2E - TTFT) / (actual completion tokens - 1)
C1 completion decode tok/s = 1 / TPOT
aggregate completion tok/s = sum(actual completion tokens) / measurement wall time
goodput = successful requests satisfying all SLO / wall time
actual mean concurrency = sum(request E2E) / wall time
```

若 `completion_tokens` 含隐藏 reasoning，以上两项必须标成 reasoning-inclusive；只有 endpoint 同时给出 `reasoning_tokens` 时，才能另算 visible-answer token rate 与 TTFO。

SSE chunk不是 token。必须使用 API usage或 exact tokenizer revision。冷启动、热引擎冷 prefix、热 prefix分开。错误、超时、截断必须进入分母。

### 统一比较的两种模式

`common denominator`：相同 checkpoint/quant/KV、template、thinking、sampling、ISL/OSL、cache、无 speculation；用于归因 framework。

`best achievable`：允许每个 framework使用最佳 kernel/quant/cache/speculation；用于选择整套系统。必须公开差异，并重新过同一质量门。

### 现有脚本的典型缺陷

- p99实际取 max；
- 1K指字符而非 tokenizer token；
- `prompt_tokens/TTFT` 在并发下仍叫裸 prefill；
- 相同 prompt让 prefix cache虚增；
- output target而非 actual usage作为分子；
- aggregate与单流 decode混写；
- thinking开关只在 warmup生效；
-每个 cell只跑一次；
-脚本更新后旧 raw result无法按当前命令复现；
-accepted/drafted比例和 accepted-per-draft概念混淆。

## 交付的 skill

本仓库已生成 [local-ai-inference-tuning](../skills/local-ai-inference-tuning/SKILL.md)，其工作流是：

1. 明确 objective/workload；
2. 采集 exact hardware/topology；
3. fingerprint model artifact；
4. compatibility/capacity gate；
5. 按 evidence tier生成短候选；
6. correctness first；
7. 一次调一组参数；
8. common benchmark；
9. 根据目标而非单个 headline选赢家；
10. 输出 pinned command、fallback、证据与不确定性。

附带工具：

```bash
# 目标机上采集硬件；不读取凭据，但输出含主机/设备/网络标识，外发前需审查
python3 skills/local-ai-inference-tuning/scripts/collect_hardware.py \
  --require-nvidia --storage-path MODEL_CACHE --container-image IMAGE \
  --pretty --output hardware.json

# 生成起始实验矩阵
python3 skills/local-ai-inference-tuning/scripts/plan_experiments.py \
  --model qwen3.8-27b \
  --hardware-profile dgx-spark-1 \
  --hardware-inventory hardware.json \
  --objective single-stream \
  --output plan.json

# 若且仅若已明确接受官方模型以外的 artifact，再为容量失败场景解锁替代 lane
# 在上面的 planner 命令中追加：--allow-alternative-artifacts

# 复制三个模板后，替换全部占位符；先运行广义质量套件并保存manifest/raw results的SHA-256
cp skills/local-ai-inference-tuning/templates/comparison-contract.json contract.json
cp skills/local-ai-inference-tuning/templates/run-metadata.json metadata.json
cp skills/local-ai-inference-tuning/templates/quality-gate.json quality-gate.json
python3 skills/local-ai-inference-tuning/scripts/benchmark_openai.py \
  --comparison-contract contract.json --print-comparison-contract-sha256

# 对任意 OpenAI-compatible server做相同客户端的微基准
python3 skills/local-ai-inference-tuning/scripts/benchmark_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model served-model \
  --prompts prompts-2k-512.jsonl \
  --plan-json plan.json --candidate-id qwen-sglang-sm121 \
  --workload-case-id isl-2048-osl-512 \
  --comparison-contract contract.json --metadata-json metadata.json \
  --quality-gate-json quality-gate.json \
  --requests 7 --concurrency 1 \
  --output results/candidate.json

# 每候选至少3个ABBA/randomized重复；合同、workload、provenance和方差门都通过才宣告赢家
python3 skills/local-ai-inference-tuning/scripts/compare_runs.py \
  results/*.json
```

这个微基准不是 vLLM/SGLang native benchmark或 AIPerf的替代品；它用于快速保证相同客户端、usage token计数、finish/correctness门、结构化artifact/hardware/cache合同和 workload fingerprint。最终容量与在线 SLO仍应使用 framework-native benchmark和 AIPerf，并保存 raw data/telemetry。

## 最终建议

真正可复用的自动选型器应分两阶段：

1. **planner**：像 Magnitude 一样，基于 exact artifact、硬件校准、内存拓扑和置信度快速淘汰不可能组合；
2. **empirical optimizer**：在目标机上按 correctness gate与受控实验逐步搜索 framework、kernel、graph、speculation、cache、scheduler和parallelism。

不能把社区 README 数字直接做成静态排行榜。可以把它们作为 candidate prior，但最终推荐只能来自同硬件、同 artifact合同、同 workload、同计时和同质量门的实测。
