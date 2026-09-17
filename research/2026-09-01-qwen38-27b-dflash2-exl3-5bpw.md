# Qwen3.8-27B EXL3 3.5bpw + DFlash2 EXL3 5.0bpw 调研归档

> 归档截点：2026-09-01 09:14 +08:00。本文固定到目标仓库、runtime fork 和两个 Hugging Face artifact 当时的 revision，核对 README、启动脚本、服务端源码、模型卡、提交历史与上游资料。没有在 DGX Spark 或 RTX 3090/4090 上下载权重、编译 CUDA 扩展或复跑性能；所有吞吐、接受率、质量与长上下文结果均是作者报告，不是本仓库实测。

## 结论

[`MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) 是一套有价值的**低显存、单请求、文本推理实验栈**，不是可直接接入生产的 Qwen3.8 官方部署方案。

它实际绑定四个共同决定行为的组件：

1. 3.5bpw EXL3 target，14.2 GB，已经改变官方 BF16 target 的质量合同，并移除了 vision tower；
2. 内置 MTP（默认）或 5.0bpw EXL3 DFlash2 draft（可选）；
3. MiaAI-Lab 的 `exllamav3` fork，提供 GB10/aarch64、DFlash2、FP8/NVFP4 KV；
4. 一个只实现部分 OpenAI Chat Completions 语义、全局串行生成的自定义服务。

最合理的仓库定位是：

```text
qwen38 / 1x-dgx-spark-or-24gb / exllamav3-fork /
exl3-3.5bpw-target / mtp-or-dflash2-exl3-5bpw /
nvfp4-kv / text-only / batch1 / c1-experimental
```

当前建议：

- **保留为 challenger**：验证 24 GB GPU 的 262K 容量路线，以及单 DGX Spark 上 MTP / DFlash2 的 C1 速度—上下文折中。
- **不要替换既有 SGLang native-MTP 基线**：公开数字没有构成同 artifact、同 runtime、同 prompt、同热状态的可审计对照。
- **不要直接接入 `oper-v7`**：服务硬编码 `enable_thinking=True`，忽略 `response_format`，没有 schema-constrained decoding；这与 non-thinking、fail-closed strict JSON 合同冲突。
- **不要把端口直接暴露到 LAN/公网**：示例默认 `0.0.0.0:8888` 且没有 API key、身份验证、限流或队列上限。

## 调研方法与证据分级

本次用 Exa 分四个工作流检查了 40 条搜索结果：目标仓库与派生项目、EXL3/runtime 支持、模型 artifact 与许可、性能反例与运行风险。去重和过滤后，再直接克隆目标仓库与 engine fork，读取一手源码并固定 revision。

本文区分：

- **源码已核对**：固定 commit 的脚本、配置或代码直接证明；
- **作者自测**：模型卡或 README 报告了硬件结果，但没有在本工作区复跑；
- **推断**：由源码行为推出的工程后果，会明确标注；
- **未证明**：配置能表达或文字声称可用，但缺少 raw results、重复、质量门或长期运行证据。

动态数据只代表归档截点。

## 固定快照

| 对象 | 归档 revision | 状态与作用 |
|---|---|---|
| deployment kit | [`09791b8`](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/commit/09791b8a9045210cd2e90b49c13ffd3b34fb2d70) | 2026-08-24 创建；24 个 commit；截点时 32 stars、0 open issue；无 tag/release/CI；代码 MIT |
| `MiaAI-Lab/exllamav3` | [`63b32f0`](https://github.com/MiaAI-Lab/exllamav3/commit/63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed) | DFlash2、MTP、NVFP4/FP8 KV、GB10/aarch64 和自定义 server 的实际实现；MIT；deployment kit 默认没有锁到该 commit |
| EXL3 target | [`19441ac`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw/tree/19441ac874c4018295da848e250f23511361cda4) | `Qwen/Qwen3.8-27B` 的 3.5bpw 量化衍生物；14.2 GB；text-only；Apache-2.0 标注 |
| EXL3 DFlash2 draft | [`4f04362`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/tree/4f0436269bca761b071f05319e8e04a87cc633f9) | `incoai/Qwen3.8-27B-DFlash2` 的 5.0bpw 量化衍生物；约 1.4 GB；不能独立生成；Apache-2.0 标注 |

目标仓库只有启动/停止脚本、单文件服务端、配置样例和两份模型卡。没有 benchmark harness、raw results、测试目录、GitHub Actions、镜像定义、requirements lock 或 release artifact。静态核对通过：`bash -n`、`python3 -m py_compile`、`git diff --check` 和 clone integrity 均无错误；这只证明脚本/语法完整，不证明 CUDA build、模型加载或推理正确。

## 实际运行合同

### Artifact 不是官方 Qwen3.8 等价物

target 模型卡说明其校准使用 622,515 tokens 的自生成 coding + math reasoning trace，采用 EXL3 3.5bpw、6-bit head、未量化 embedding/lm-head；模型只保留 text model 和 tokenizer，没有 vision tower。这个选择有明确容量价值，但带来三个边界：

1. 它不能继承官方 BF16 Qwen3.8 的通用质量、长上下文、vision 或 tool 结论；
2. 工作负载匹配校准可能有利于 coding/math，不等于其他领域无回归；
3. DFlash2 的“lossless speculative decoding”最多表示 verifier 相对**这个量化 target 和当前 cache path**保持同一目标分布，不能把它变回官方 BF16 target。

默认 `CACHE_QUANT=nvfp4` 还会量化 target KV。NVFP4 KV 是有损状态压缩；模型卡的 cosine similarity 和少量 generation-level 对照是正向证据，但不能称为数学上的 lossless。应把 target weight quantization、KV quantization、draft quantization 分成三个变量逐步引入。

### 默认并不是 DFlash2

launcher 的默认顺序是：

| `DRAFT` | 额外 artifact | 作者定位 | 当前判断 |
|---|---|---|---|
| `mtp`（默认） | target 内置约 50 MB MTP head | 上下文/显存优先，24 GB 上目标 262K | 应作为本栈 control，但仍是量化 target + 自定义 fork |
| `dflash2` | 1.4 GB EXL3 draft + draft KV | C1 tok/s 优先 | 有测试价值；必须与同一 target 的 AR/MTP 对照 |
| `none` | 无 | 非 speculative control | 必须保留；没有它就不能归因 speculative 增益 |

仓库名称突出 DFlash2，但 `.env.example` 注释掉了 `DRAFT`，代码最终选择 `mtp`。因此“克隆后按默认启动”测到的是 MTP，不是 DFlash2。

### 服务只是 OpenAI Chat Completions 子集

源码实现：

- `GET /v1/models`、`GET /health`、`POST /v1/chat/completions`；
- streaming / non-streaming；
- Qwen XML tool-call 的自定义解析；
- 全局 `threading.Lock`，一次只生成一个请求，其他请求在 worker thread 中等待；
- temperature、top-p、top-k、seed、stop 和 max tokens 的基础映射。

它没有 Responses API、Completions API、embeddings、鉴权、限流、显式 backpressure、队列深度、请求 deadline、生成取消、structured-output grammar 或 production metrics。断开 streaming client 后，HTTP 层吞掉 `ConnectionResetError`，但 worker 没有收到取消信号；**从源码推断**，已开始或排队的生成可能继续占用唯一生成槽。

tool 支持也不是强约束：`tool_choice=required` 或指定函数只追加 system prompt，失败后做一次 greedy retry；第二次仍不调用时不会 fail closed。参数解析只按顶层 JSON Schema type 做类型转换，不检查 required、enum、嵌套 schema 或业务约束。

## 作者性能与质量证据

以下数字只按作者报告归档，不能跨表直接排名：

| 项目 | 作者报告 | 能说明什么 | 仍缺什么 |
|---|---|---|---|
| MTP C1 | HumanEval-class、T=0.6，约 2.2 accepted tokens/step、约 30 tok/s | 低显存默认 lane 有正向性能先验 | prompt、输出长度、重复、范围、热状态、raw results |
| EXL3 DFlash2 C1 | HumanEval-style、T=0.6，4.43 accepted tokens/step、47.5 tok/s | 对 coding 型 C1 有强先验 | 没有与 MTP/AR 同一次运行的完整 raw A/B |
| BF16 draft → EXL3 draft | 35.7 → 47.5 tok/s；acceptance 4.27 → 4.43 | 低比特 draft 在 GB10 的带宽假设值得验证 | 只有汇总值；没有 trial 数、CV、TTFT/TPOT、功耗、峰值内存 |
| tool-eval-bench | v2.5.1 hardmode、T=1.0、seed 42，87–88/100 | 自定义 tool path 不是完全未测 | 84 scenarios 的 raw responses、scorer、失败分类与重跑证据未发布在 kit 中 |
| 300K passkey | 10%/50% depth 通过；末尾约 30K 失败 | YaRN 能机械运行到 native context 之外 | 结果同时证明 1M 不是质量合格上下文 |

README 同时使用“DFlash2 约快 15%”“EXL3 draft 比 BF16 draft 快 33%”和 HumanEval 47.5 tok/s。这些分母与 workload 不同，不能拼成一条统一 speedup。归档时只保留各自条件，不把 47.5 tok/s 视为所有 prompt 的下界；README 对 RTX 3090/4090“应更快”的判断也是带宽推断，不是实测。

## 源码审计发现

### P0：`oper-v7` 直接不兼容

`oper-v7` 的当前合同是 non-streaming、temperature 0、thinking disabled、`max_tokens=2048`、`response_format=json_object`，并要求 strict JSON/schema fail closed。本服务：

| 合同项 | 当前支持 | 证据/后果 |
|---|---|---|
| non-streaming | 是 | 普通 JSON response path |
| temperature 0 | 是 | request 参数映射 |
| max 2048 | 是 | 可传 `max_tokens`，但服务端没有产品级上限策略 |
| thinking disabled | **否** | chat template 固定 `enable_thinking=True`，请求无法覆盖 |
| `response_format=json_object` | **否** | request parser 不读取此字段 |
| strict schema / fail closed | **否** | 无 grammar/constrained decoder，输出 JSON 只能靠 prompt |
| 并发 goodput | **否** | batch-1，全局串行；C4/C8 是排队延迟而非并行生成 |
| 鉴权/网关边界 | **否** | 默认 LAN bind、无 auth |

因此，即使 tok/s 达标，它也不能进入 `oper-v7` shadow/canary；必须先补齐 API/structured-output 合同，再重新做质量比较。

### P0：默认部署不可复现

首次启动会从浮动地址安装和下载：

- `git+https://github.com/MiaAI-Lab/exllamav3`，没有 commit/tag；
- `torch` + cu130 index，没有版本或 wheel hash；
- 两个 Hugging Face repo，没有 `revision=`；
- `aiohttp`、`huggingface_hub`，没有版本 lock。

也没有 container digest、Python/CUDA/driver matrix、artifact SHA256 或安装后的 source manifest。同一 deployment commit 在不同日期可能得到不同 engine、模型文件和依赖，无法把结果归因到仓库 commit。

### P1：网络与资源保护不足

`.env.example` 明确默认 `HOST=0.0.0.0` 且无 auth。服务又没有 rate limit、队列上限、并发 semaphore、request timeout 或 body 之外的 token budget guard。任意可达客户端都能提交长上下文/长生成并占用唯一槽位。正确边界是 `127.0.0.1` 或受保护容器网络，只通过具备身份、限流、deadline 和 body/token 限制的内部网关访问。

### P1：长上下文配置会原地漂移

当 `CONTEXT_SIZE > 262144` 时，`start.sh` 会把 `config.yarn-1m.json` 复制覆盖模型目录的 `config.json`。之后把 context 改回 262K 并不会自动还原原生配置。这个 mutation 没有备份、hash check 或 rollback；同一模型目录会随启动历史改变。实验应使用不可变的 native/yarn 两个 snapshot 或显式配置参数，不能原地覆盖已下载 artifact。

### P1：观测与恢复不足

`/health` 只暴露 busy、累计 prompt/completion tokens 和 context length。没有 queue depth、排队时间、TTFT、TPOT、acceptance、draft/verify cycle、cache occupancy、OOM、取消、错误分类、功耗或版本 identity。`stop.sh` 有端口/命令行校验和 graceful→SIGKILL 退路，是正向运维细节；但仓库没有 service manager、自动重启策略、版本化 rollback 或 soak harness。

## 对硬件的采用判断

### 单 DGX Spark / GB10

这是最匹配的实验硬件。3.5bpw target + MTP/DFlash2 + 128 GB UMA 有充分容量，C1 又可能受 273 GB/s-class 内存带宽限制，正适合验证“量化 draft 减少读带宽”的假设。仍需记录 SM121 kernel 是否走预期路径、统一内存 resident peak、swap/pageout、CUDA build flags、首次 JIT、draft/target KV、262K prefill 和热状态。

### RTX 3090 / 4090 24 GB

作者的 24 GB memory math 是有吸引力的候选资格，不是已证实的可用上限。MTP + NVFP4 KV 的 262K、DFlash2 的 200K 配置必须以实际峰值、fragmentation、CUDA graph/JIT workspace 和 OS/桌面占用复核。更高 VRAM bandwidth 不保证端到端必然更快；kernel、clock、PCIe、prefill 和 CPU 调度仍可能成为瓶颈。

### 其他平台

这是 CUDA/EXL3 fork 路线，不是 Apple MLX、llama.cpp GGUF、SGLang 或 vLLM 结果的可互换实现。不要把已有 DFlash2 H200/SGLang、oMLX 或 GGUF 数字用于本栈排名。

## 最小复现实验合同

### 1. 先冻结供应链

至少固定：

```text
deployment = 09791b8a9045210cd2e90b49c13ffd3b34fb2d70
engine     = 63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed
target_hf  = 19441ac874c4018295da848e250f23511361cda4
draft_hf   = 4f0436269bca761b071f05319e8e04a87cc633f9
```

另存 Python、PyTorch/CUDA wheels、driver、CUDA toolkit、compiler、所有模型文件 SHA256、启动参数和环境。把 engine 改为 commit-pinned URL；模型使用 `snapshot_download(revision=...)` 或预先下载的只读 snapshot。默认绑定改成 `127.0.0.1`。

### 2. 把变量逐层加入

同一 target revision、prompt、sampling、热状态下运行：

1. EXL3 target + fp16 KV + AR；
2. 同 target + fp16 KV + MTP；
3. 同 target + fp16 KV + EXL3 DFlash2；
4. 以上胜出 lane 再把 KV 改成 NVFP4；
5. 在足够硬件上用官方 BF16 target，或用封存的 BF16 output/eval corpus，做 artifact 质量对照。

不应一次同时更换 target quant、KV quant 和 drafter，再把总差异归因给 DFlash2。

### 3. 正确性先于速度

- greedy：AR / MTP / DFlash2 在同一 target + KV 上逐 token 相等；
- sampling：固定 seed 的实现回归，加重复分布检验；
- strict JSON：语法与业务 schema 100%，不能依靠 post-hoc 修补掩盖模型错误；
- tool：auto/none/required/specific、嵌套 object/array、enum、缺失 required、工具响应多轮；
- thinking on/off、EOS/repetition、取消/超时、超长 body、错误输入；
- 2K/8K/32K/128K/262K；YaRN >262K 只列实验结果，不进入质量默认值；
- vision 直接标记 unsupported，不能从官方模型能力继承。

### 4. 性能与稳定性

- C1：TTFT、TPOT/ITL、E2E、completion tok/s、acceptance length/per-position、draft/verify 时间；
- 当前 batch-1 server 的 C2/C4/C8：报告 queue wait、端到端 p50/p95/p99 和 timeout，不写成并行 aggregate throughput；
- 每个 C1 条件至少七次交错 trial，发布 raw JSONL、范围和 CV；
- ≥1,000 请求与 8 小时 soak：零 OOM、进程退出、CUDA error、swap/pageout、悬挂队列和无法取消的生成；
- 记录 resident/peak memory、context ceiling、功耗、温度和首次/热 JIT 差异。

## 晋级条件

只有全部满足后，才从 `c1-experimental` 升到 `candidate`：

1. repo、engine、dependency、target、draft 与配置全部 immutable；
2. 官方 BF16 → EXL3 target → NVFP4 KV 的质量损失分别在预设阈值内；
3. MTP/DFlash2 相对同一 target 的正确性门通过；
4. `oper-v7` thinking-off + `response_format` + strict schema fail-closed 已实现并回归；
5. 真实病例 trace 的 E2E/SLO goodput 明确优于 native MTP 基线；
6. 1,000-request、8-hour soak、取消、超时、回滚和网关鉴权门通过；
7. 没有把 1M 可分配、作者 headline 或 24 GB memory math写成已验证生产能力。

在这些门完成前，最准确的结论是：**一个针对 C1 与显存效率有强工程先验、但供应链与服务合同仍处于早期的社区实验栈。**

## 一手来源

- [目标 deployment kit，固定 commit](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/tree/09791b8a9045210cd2e90b49c13ffd3b34fb2d70)
- [目标启动脚本，固定 commit](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/blob/09791b8a9045210cd2e90b49c13ffd3b34fb2d70/start.sh)
- [OpenAI-compatible server，固定 commit](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/blob/09791b8a9045210cd2e90b49c13ffd3b34fb2d70/tools/serve_openai.py)
- [MiaAI-Lab exllamav3 fork，固定 commit](https://github.com/MiaAI-Lab/exllamav3/tree/63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed)
- [EXL3 3.5bpw target model card，固定 revision](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw/blob/19441ac874c4018295da848e250f23511361cda4/README.md)
- [EXL3 5.0bpw DFlash2 model card，固定 revision](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/blob/4f0436269bca761b071f05319e8e04a87cc633f9/README.md)
- [Qwen3.8-27B 官方模型与 Apache-2.0 LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B)
- [DFlash2 官方 BF16 draft model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- [DFlash2 工程说明](https://inco.ai/blog/dflash2/)
- [DFlash 论文](https://arxiv.org/abs/2602.06036)
- [upstream exllamav3](https://github.com/turboderp-org/exllamav3)

## 与本仓库其他归档的关系

- DFlash2 原理、H200/SGLang 公开数据与 runtime 状态见 [DFlash 2 专题调研](2026-08-19-dflash2.md)。
- Qwen3.8 NVFP4 + native MTP 的生产候选合同见 [最终实施计划](2026-08-21-qwen38-27b-nvfp4-sglang-mtp-final-plan.md)。
- DGX Spark 上不同 artifact/runtime 的总体位置见 [本地推理版图](2026-08-18-local-inference-landscape.md)。
