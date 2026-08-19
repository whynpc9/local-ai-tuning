# Apple Silicon 本地推理调优扩展：Qwen3.8-27B 与 DeepSeek-V4-Flash-0731

调研截止：2026-08-19

## 结论先行

1. **Apple Silicon 不是一个硬件 profile。** 同为 M3/M4 Max，GPU 核心、内存容量、带宽和机身散热都可能不同；正式比较必须绑定完整 SKU、macOS build、原生 arm64、功率/热状态和运行窗内存压力。
2. **Qwen3.8-27B 当前最值得先测的是固定 `jundot/omlx` 的同 artifact target-only → native MTP。** 入口仓库给出了很强的速度信号，但 raw 输出已经出现 `<think>`、benchmark filler、marker 复制和未执行代码任务，因此 48/65.5 tok/s 不能直接晋级。
3. **入口 README 链接错了上游。** oMLX 不是空账号里的闭源项目；真实公开源码是 [`jundot/omlx`](https://github.com/jundot/omlx)。历史复现应固定 v0.6.1，新的部署基线应从 v0.6.2 开始并重新测量。
4. **MTP 最大深度不是实际固定 `k`，也不保证更快。** oMLX 会按 acceptance/cycle cost 自适应选择深度并可退回普通解码。独立的严格 challenge 在 M5 Max 上测得 depth 2 约 `0.994×`，说明每台机器、量化、上下文和内容域都要现场校准。
5. **ANE 是 prefill 实验，不是 decode 加速。** 它使用实验/私有 Apple runtime 路径、固定 shape 和额外内存；Weschera 的 48/65.5 tok/s 明确关闭 ANE。必须把 TTFT/prefill 与 decode 分开。
6. **统一内存“装得下”不等于性能可用。** 持续 swapout、compression churn 或 decode-time page-in 的配置只能标记 functional，不能参与最快 tok/s 排名。
7. **DeepSeek 官方 0731 在任何 128 GB Mac 上容量失败。** 256 GB M3 Ultra 是社区 q4/oQ4e 的自然起点；约 92.8 GB 的极低比特制品可以进入 128 GB admission，但已经是独立质量合同。

## Weschera 入口仓库审计

入口：[Weschera/Qwen3.8-27B-oMLX-MTP-Mac](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac)，审计 commit [`0800fb5`](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac/commit/0800fb5ca9a5921a32ff32dbe4d7cb3e5d9feeac)。

配置：

```json
{
  "qwen35_ane_prefill_enabled": false,
  "mtp_enabled": true,
  "mtp_num_draft_tokens": 3
}
```

模型是第三方混合量化 [`Jundot/Qwen3.8-27B-oQ4e-mtp@04dc5509`](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp/tree/04dc5509edd8670fc78cc8c2f74bf9b77b1f2acc)：

- 大约 15.82 GiB；
- 64 层、262,144 原生 context、一个 MTP hidden layer；
- 主体 affine q4/group 64，166 个敏感 tensor 使用 5-bit override；
- 含完整 vision tensors；
- 是由官方 lineage 转换的社区 artifact，不是官方 BF16/FP8 checkpoint；模型卡也没有固定原始 base revision。

Raw JSON 可精确重算：

| Arm | Prose mean | Code mean | 相对 target-only |
|---|---:|---:|---:|
| oMLX target-only | 24.865 | 25.921 | 1.00× |
| oMLX adaptive max-depth 2 | 43.478 | 56.325 | 1.75× / 2.17× |
| oMLX adaptive max-depth 3 | 48.004 | 65.540 | 1.93× / 2.53× |
| mlx-vlm sidecar | 43.363 | 56.740 | 1.74× / 2.19× |

这里应该写 `configured_max_depth=2/3`，而不是固定 `k=2/3`。仓库没有 server logs，无法知道实际 depth histogram、drafted/accepted、tok/cycle 或 fallback。

不能晋级的原因：

- `bench.py` 使用 raw `/v1/completions`，mlx-vlm 使用 `/v1/chat/completions`，模板/route 不同，不能归因于 engine-only 差异；
- 声称 thinking off，但保存的 baseline/MTP 输出仍出现 `<think>`；
- 多个输出在讨论重复 filler、回显 marker/filler，或没有开始用户要求的代码；
- code prompt 是“复制整段类并增加两个方法”，接近 speculation 的最佳复制场景；
- 几乎所有样本触及 319/320 token cap，未保存 finish reason 或完整输出，任务可能未完成；
- 只做 3 次，顺序固定；第一次 baseline 明显更慢，说明 16-token warmup 未稳定同 shape Metal 路径；
- usage 缺失时按 SSE data event 计 token，对 speculative stream 不成立；
- 没有 acceptance、内存、swap、热、功率、并发、长上下文、视觉、tool/JSON 或代码执行门。

因此这些数字是 **Tier C candidate prior**，不是“最好配置”或“质量无损”证据。

## oMLX 的真实上游与能力

真实源码与 release：

- [oMLX v0.6.1](https://github.com/jundot/omlx/releases/tag/v0.6.1)，commit `b587575f...`；入口声称使用此版本；
- [oMLX v0.6.2](https://github.com/jundot/omlx/releases/tag/v0.6.2)，commit `f2d36f3d...`；当前新部署起点；
- v0.6.1 精确依赖见 [`pyproject.toml`](https://github.com/jundot/omlx/blob/v0.6.1/pyproject.toml)：MLX 0.32.0、`mlx-lm@ab1806e8`、`mlx-vlm@78b96eb5`、`dflash-mlx@2eb169f4` 以及 nanobind ABI pin。

公开源码具备：

- FastAPI/OpenAI chat/completions/embeddings/rerank/models 与 Anthropic messages；
- continuous batching；
- paged copy-on-write KV、block prefix cache；
- RAM hot tier + SSD safetensors cold tier。SSD 是重复 prefix 的持久 cache，不是权重/工作内存 offload；
- VLM、tools、JSON schema；
- Qwen3.8 native embedded MTP；
- vendored DeepSeek V4/Lightning MTP/DSpark patches；
- 实验性 Qwen ANE/GPU prefill。

关键 MTP 语义：

- 默认 off；一个 MTP head 可递归形成多 token chain；
- `mtp_num_draft_tokens` 是最大深度；controller 按滚动 acceptance 与 cycle cost 选择 1..max，并可回退 AR；
- grammar-constrained decode 不走 MTP；
- MTP 与 DFlash 互斥；
- continuous batch 未对齐时可能回退普通 decode；row-wise MTP 需要显式开关，已有 batch2/4 数据弱于普通 continuous batching；
- 正式记录必须包含 configured max、实际 depth histogram、drafted/accepted、per-position、cycles、tok/cycle 和 fallback。

公开 `dflash-mlx` registry 没有 Qwen3.8 target/draft pair。入口提到的 DFlash2 是 closed/private dependency，不是公开可复现候选。

v0.6.1 存在 MTP + TurboQuant KV crash；v0.6.2 已修复。因此：

- 只在复现历史时使用 v0.6.1；
- 实际部署先固定 v0.6.2，重新跑 target-only 和全部 MTP lane；
- 不能把 v0.6.1 的速度与 v0.6.2 的稳定性拼成同一结果。

## Apple 硬件矩阵和 admission

| Chip | GPU cores | 最大统一内存 | 峰值带宽 | Metal family |
|---|---:|---:|---:|---:|
| M1 Max / Ultra | 24-32 / 48-64 | 64 / 128 GB | 400 / 800 GB/s | Apple7 |
| M2 Max / Ultra | 30-38 / 60-76 | 96 / 192 GB | 400 / 800 GB/s | Apple8 |
| M3 Max | 30 / 40 | 96 / 128 GB | 300 / 400 GB/s | Apple9 |
| M3 Ultra | 60 / 80 | 512 GB | 819 GB/s | Apple9 |
| M4 Max | 32 / 40 | 128 GB | 410 / 546 GB/s | Apple9 |
| M5 Max | up to 40 | 128 GB | 614 GB/s | Apple10 |

Apple 没有发布 M4 Ultra。同名 Max 的低/高配不能混测；MacBook Pro 与 Mac Studio 也要因散热/功率分开。

核心公式：

```text
effective_budget = min(
  Metal recommended working set,
  physical memory
    - measured idle OS footprint
    - non-model processes
    - explicit safety reserve
)

required = weights + quant metadata/padding
  + KV/GDN/MLA/state(context, concurrency)
  + draft/MTP
  + runtime workspace/temp
  + allocator cache
  + tokenizer/server
```

静态通过后必须实装目标 context/concurrency。以下任一发生就禁止进入 performance ranking：持续 swapout/pageout、compression/decompression churn、decode-time page-in/disk read、Metal OOM/timeout、OS kill、异常 footprint 增长或 CPU fallback。

Qwen oQ4e 的实用分档：

- 24 GB：权重已经接近默认系统 reserve 后的预算，拒绝；
- 32 GB：只作为短 context admission；
- 48 GB：短 context 单流的实际候选；
- 64 GB：更合理的日常起点；
- 128 GB：长 context/并发仍需实测，不能因模型声明 262K 就直接配置 262K。

DeepSeek：

- 官方 fused checkpoint 约 167 GB，128 GB 硬拒绝；
- 社区 q4/oQ4e 路线实测 residency 约 156-161 GB，256 GB 是自然起点；
- 约 92.8 GB 的 2.4-bit artifact 可以进入 128 GB admission，但量化/质量合同完全不同；
- M2 Ultra 192 GB 对官方/高位 q4 仍是 tight lane；只有实际 peak + reserve 才能放行。

硬件事实来源见 [Apple 官方规格](https://support.apple.com/en-us/122211)、[Metal capability tables](https://developer.apple.com/metal/capabilities/) 以及 skill 的 [Apple profile](../skills/local-ai-inference-tuning/references/apple-silicon.md)。

## 框架选择矩阵

| 框架 | Qwen3.8 | DeepSeek 0731 | 适合场景 | 主要边界 |
|---|---|---|---|---|
| oMLX 0.6.2 | full VLM、native MTP、自适应深度、continuous batch、cache、tools/structured output | vendored DeepSeek V4、WSDPA/indexer/MoE、Lightning MTP/DSpark patches | 当前 Apple 服务化首选候选 | 固定 tag/依赖；MTP/ANE/grammar/batching功能组合需 gate；社区 artifact质量独立 |
| plain MLX-LM | 文本 backbone；当前 loader去掉 vision和embedded MTP | 研究/target-only control | 可审计 native microbench/text baseline | 不是完整 Qwen VLM/MTP产品；基础 server不适合当吞吐冠军 |
| MLX-VLM | full VLM/mRoPE，embedded MTP拆成sidecar，ragged batch，OpenAI chat/responses | 有 DeepSeek V4 MTP split，成熟度低于 oMLX | inspectable sidecar/multimodal challenger | structured output禁 speculative；APC/hybrid cache需 exact commit实测 |
| MTPLX | Qwen native MTP kernels、AR/D1-D3 auto tuning、质量/热/长上下文 harness | 非主要路线 | Qwen性能研究 challenger | published Qwen artifacts在repo里未固定 revision，数字只能C级 |
| mlx-dspark | external DSpark/DFlash/lookup、现场 cost-curve cap | 研究实现 | external drafter challenger | headline target/draft revisions和raw results不足 |
| llama.cpp Metal | GGUF、多KV类型；Qwen full GGUF可含embedded MTP；DeepSeek target需另导MTP或DSpark draft；server acceptance/per-position metrics | DeepSeek target + separate Lightning MTP/DSpark draft | 最开放的GGUF/Metal portable control | MTP在Metal上可正可负；Qwen image-token+MTP当前会提前回退/不支持；converted artifact和MLX artifact不同 |
| Magnitude | 内嵌pinned llama.cpp、exact fit、硬件校准、受控benchmark | DSpark catalog/control | planner和product-lane comparison | catalog当前没有Qwen3.8 speculative default，不是性能engine冠军 |
| Ollama native MLX | Qwen vision/self-draft/hybrid recurrent cache | exact dispatch gate | 易用local UX | 当前单request/无continuous batching，不是multi-user throughput首选 |
| LM Studio | mlx-engine或llama.cpp包装 | UI试用 | GUI | 版本绑定、MTP sidecar/dispatch issues；先在direct engine验证 |

## 高价值相关仓库

| 仓库 | 证据 | 最值得迁移的内容 | 不能外推的部分 |
|---|---|---|---|
| [Layr-Labs/qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge) `0c90733d` | B | pinned manifest、隐藏 prompts、交替顺序、40°C thermal gate、token exactness、teacher/free-run/behavior/GPQA/semantic gates | M5 Max/特定4-bit artifact；depth2约0.994×，正是“允许AR胜出”的反例 |
| [youssofal/MTPLX](https://github.com/youssofal/MTPLX) `90d8c4b` | A功能/C+性能 | 多量化profile、AR/D1-D3、ABBA、exactness、thermal、kernel self-check、long-context、agent QA | Qwen artifact revision未在repo固定 |
| [ARahim3/mlx-dspark](https://github.com/ARahim3/mlx-dspark) `2d6a94be` | A功能/C性能 | machine/model/quant/MLX cost curves、auto cap、Race token equality、fallback | headline无checked-in raw JSON和artifact pin |
| [jundot/omlx](https://github.com/jundot/omlx) v0.6.2 | A功能/C性能 | production-like scheduler/cache/server、native MTP、ANE prefill、DeepSeek patches | 官方项目数字仍需本机统计重复/质量门 |
| [drowzeys DualANE](https://github.com/drowzeys/keys-MAC-oMLX-0.6.1-DualANE-Qwen3.8-27B-Abliterated-oQ4e-MTP) `d3e3766f` | C+ | raw/log、max-depth sweep、MLP-only ANE、activation/logit gate | M3 Ultra 256 GB + abliterated artifact；复制型code 99.5-100% acceptance不能泛化 |
| [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) `c7bc4153` | C | 53 configs的llama.cpp Metal负结果/低收益库 | SSE chunk token计数、无raw JSON |
| [Alexander-Ollman/qwen3.8-on-m4max](https://github.com/Alexander-Ollman/qwen3.8-on-m4max) `11cbfb24` | C+ | context/quant/engine/contention/prefix cache、raw scripts、drift canary | 多数单次、model digest未钉 |
| [hyunhwan-bcm/omlx-benchmarks](https://github.com/hyunhwan-bcm/omlx-benchmarks) `ea88c4d2` | C | MMLU/ARC/GSM8K/HumanEval；揭示 thinking prose 会让错误提取器把HumanEval 95.7%错报12.2% | 无harness/full generations |
| [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) `d173963d` | A功能/C性能 | adaptive prompt/acceptance break-even gate | exact Qwen3.8 raw性能未建立 |
| [dmitryryabkov/mtp-profiler](https://github.com/dmitryryabkov/mtp-profiler) `20ecd45e` | C-method | llama.cpp context/acceptance/memory pressure LOWESS分析 | 示例模型是Qwen3.6 |

这些项目形成的原始实现链：

```text
MLX / MLX-LM
├── MLX-VLM: VLM + MTP sidecar
├── oMLX: server + scheduler + cache + native MTP + ANE
├── MTPLX: native MTP kernels + auto tuning + QA
├── mlx-dspark: external DSpark/DFlash/lookup
└── Layr challenge: 严格评分和漂移/热门禁

llama.cpp MTP/DSpark
├── Magnitude: fit planner + controlled product benchmark
└── sudoingX: Apple Metal正/负实测
```

## Apple 专属调参轴

### Artifact 与量化

- 相同 MLX artifact 的 target-only/MTP 才能归因 speculation；
- 4/5-bit oQ4e、dynamic q4、q6、q8、GGUF Q4/Q6/Q8必须分 artifact合同；
- 对比不同量化/ablation只能做 best-achievable Pareto，不得称 framework A/B；
- 保存 base/model/draft/tokenizer revisions、shard hashes、conversion manifest和量化校准/下游质量。

### MTP/DSpark

先跑 AR，再按 workload sweep max depth/cap `1/2/3`，只有明显获益后再测 5/8。分 prose、真实代码生成、复制型code-edit、creative、reasoning、tools；分 2K/8K/32K/64K 和 C1/2/4/8。记录：

```text
configured_max_depth
observed_depth_histogram
drafted / accepted / rejected
accepted_per_position
verification_steps
tok_per_cycle / cycles
fallback_to_standard_decode
```

MTP只有在同 artifact target-only质量通过、median speedup超过方差/门槛时晋级。建议初始 promotion：median speedup ≥1.05，配对 bootstrap 95% CI lower bound >1.0；否则保留 AR。

### Cache、context、batch

- 记录 `cached_tokens`；cold-prefix使用早期unique nonce，warm-prefix单独合同；
- oMLX SSD cache单测restart复用，不能称offload；
- sweep MLX prefill step、batch、KV bits/group/start、max KV、prompt cache、memory/cache/wired limits；
- llama.cpp单测threads、batch/ubatch、FA、KV type、GPU layers、mmap、slots；
- continuous batch要覆盖aligned和late-join/unaligned arrival，因为speculation可能回退。

### ANE

只在可审计/pinned实现上测 `off` 与 `MLP-only`。GDN offload已有 logit cosine降至0.901并改变top token的反例，不应作为默认。每个ANE lane要检查 hidden/logit cosine、top token、完整输出hash、任务质量、额外peak、load time和TTFT；decode TPS不应变化。

### Thermal、power、UMA

- 原生 arm64，禁止 Rosetta；
- 固定chassis、AC/battery、low-power、raw power mode、display topology；
- 同shape warmup 5次起，最后3次CV≤3%、range/median≤5%，最多10次；
- 严格排名要求thermal nominal；fair标热影响，serious/critical作废；
- 30-60分钟 stability，last-quartile/first-quartile median至少0.95；
- swapout/pageout为0；compression churn相对idle baseline；未校准写unknown，不能伪造0。

## Benchmark 方法

三种 lane不能混：

1. native microbench：`mlx_lm.benchmark`、`llama-bench`/`llama-batched-bench`；诊断pp/tg/kernel/peak，不含真实tokenizer/chat/API/quality；
2. API steady/aggregate/goodput：相同chat route/client、usage token、queue/server语义；
3. cold-start/thermal stability：独立报告load/JIT/首请求和持续降频。

严禁的 fallback：SSE event、字符数或请求 `max_tokens` 当实际 token。

核心指标：

```text
client TPOT = (last response completion - first output) / (actual completion tokens - 1)
client decode tok/s = 1 / TPOT
server prefill tok/s = actually processed prompt tokens / server prompt seconds
aggregate tok/s = sum(actual completion tokens) / shared wall window
goodput requests/s = valid requests meeting every SLO / wall
goodput tokens/s = valid completion tokens meeting every SLO / wall
```

`prompt_tokens/TTFT`只能叫 `ttft_derived_prompt_rate`，因为包含tokenization、queue、Metal compile、first decode和传输。有cache时真正processed tokens是prompt减cached。

必须保存完整 answer/reasoning/tool calls、finish reason和hash；先通过exact copy、代码执行、strict JSON、tools、vision、thinking modes、repetition/EOS、long context、cancel/recovery，再测速度。Acceptance不是质量分数。

## 已扩展的 skill

主入口：[local-ai-inference-tuning skill](../skills/local-ai-inference-tuning/SKILL.md)。Apple专属细节在 [apple-silicon.md](../skills/local-ai-inference-tuning/references/apple-silicon.md)。

新增/更新能力：

- privacy-allowlisted Apple hardware collector：不保存 raw `system_profiler`/电池ID，并对macOS `df -kP`存储探针fail closed；inventory仍含hostname，外发前必须复核；
- `apple-silicon-1` planner profile，强制inventory并验证native arm64、非Rosetta、精确chip/GPU cores/chassis、Metal、统一内存、macOS和power tuple；
- Qwen oMLX target-only、native MTP、MLX-VLM、MTPLX、mlx-dspark、Magnitude/llama.cpp候选；
- DeepSeek high-memory oMLX、community quant与llama.cpp/Magnitude候选；
- Apple warmup/cache/MTP/thermal/UMA policy；
- 无特权运行窗 telemetry JSONL：拒绝非有限/单样本时间窗，以`NSProcessInfo.thermalState`执行serious/critical门禁，锁定active power source与完整`pmset` profile hash，要求swapin/swapout/pageout为零，并记录page/compressor delta与machine-readable probe health，失败非零退出；
- common OpenAI client额外记录cached prompt tokens、processed prompt rate和goodput token/s。

Apple工作流：

```bash
SKILL_ROOT="$PWD/skills/local-ai-inference-tuning"

python3 "$SKILL_ROOT/scripts/collect_hardware.py" \
  --require-apple-silicon \
  --storage-path MODEL_CACHE \
  --pretty \
  --output hardware.json

python3 "$SKILL_ROOT/scripts/plan_experiments.py" \
  --model qwen3.8-27b \
  --hardware-profile apple-silicon-1 \
  --hardware-inventory hardware.json \
  --objective single-stream \
  --output plan.json

python3 "$SKILL_ROOT/scripts/sample_apple_telemetry.py" \
  --run-id RUN_ID \
  --pid SERVER_PID \
  --duration 180 \
  --output telemetry.jsonl
```

每个 performance result仍必须绑定comparison contract、plan、quality gate、candidate、shape、artifact/runtime/config hashes，并用现有 comparator的重复/顺序/方差门才能产生winner。

## 当前可辩护的起始策略

### Qwen3.8-27B

- **64-128 GB Max/Ultra，服务化优先：** 固定 oMLX v0.6.2 + 同一 oQ4e artifact，先 target-only，再 adaptive max-depth 1/2/3；真实多并发优先普通 continuous batching，row-wise MTP只作challenger。
- **希望源码级实验：** MLX-VLM sidecar、MTPLX、mlx-dspark；每条分别固定runtime/target/draft并与同target AR比较。
- **希望GGUF/简单部署：** Magnitude/llama.cpp Metal作为portable control；MTP可能负收益，必须现场A/B。
- **长prompt TTFT：** ANE仅作独立prefill lane；off→MLP-only，记录额外内存/load和logit/output gate。
- **32 GB：** 仅短context admission；bundled 32K/serving performance plan会保持阻断，优先更小artifact/AR，MTP额外resident必须实测。

### DeepSeek-V4-Flash-0731

- **256 GB及以上：** 固定 oMLX v0.6.2 DeepSeek path + pinned q4/oQ4e conversion，先target-only，再Lightning MTP/DSpark；llama.cpp DSpark作control。
- **512 GB M3 Ultra：** 可把官方/更高精度source-conversion作为研究control，但exact runtime和peak仍是硬门。
- **128 GB：** 官方/q4硬拒绝；只有约2.4-bit或其他显著转换artifact可进入独立质量合同，不得称官方模型性能。
- **192 GB：** tight measured-admission，不因存储数字低于RAM就自动放行。

这些都是 candidate shortlist，不是未经目标机common benchmark的“best”。
