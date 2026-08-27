# Qwen3.8-Flash-Next / DGX Spark 三仓库调研归档

> 归档截点：2026-08-27 09:27 +08:00。本文固定到三个仓库当时的 `main` commit，并核对官方模型卡、运行时上游 PR/issue、仓库脚本与作者自测。本文没有在 DGX Spark 上复跑，因此吞吐与稳定性数据均按“作者报告”处理，不作为本仓库实测值。

## 结论

这三个仓库不是三份可以直接横向排名的同类方案，而是两条不同路线：

1. [`0xBakeer/qwen38-flash-next-spark`](https://github.com/0xBakeer/qwen38-flash-next-spark) 证明单台 128 GB DGX Spark 可以通过 llama.cpp `mmap` 将 51B n-gram/PLE 表留在 NVMe，运行 Q4 GGUF。它的价值是“容量与分层放置实验”，不是并发服务方案。
2. [`tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) 和 [`MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) 都是两台 Spark、SGLang、TP2、NVFP4、内置 MTP/NEXTN 路线。Tony 更像现场部署日志和故障手册；Mia 更接近可复用的一体化运维脚本。

当前没有一个仓库具备直接替换本项目现有服务的证据：三者都缺少 `oper-v7` 的 strict JSON、真实 trace、C1/C4/C8、1,000 请求、8 小时 soak、错误预算、回滚与网关鉴权全套门。最佳采用方式是：**0xBakeer 归档为单机容量 lane；Tony 归档为 SM121/UMA 故障知识库；Mia 作为双机实验 fork 的起点，但必须先锁版本、收紧网络与补齐正确性门。**

## 调研方法与证据分级

本次使用 Exa 检索了 6 个工作流、48 条搜索结果，再去重并过滤到仓库、官方模型卡和上游 GitHub 等第一方来源。结论按以下层级表达：

- **源码已核对**：在固定 commit 的 README、脚本、提交记录、issue/PR 中直接可见。
- **作者自测**：仓库作者报告了硬件结果，但本地未复跑；不同仓库之间不可直接排名。
- **未证明**：README 宣称或配置可表达，但没有对应 raw results、长稳或质量证据。

动态数据（star、issue 状态、PR 状态）只代表归档截点。

## 上游基线

官方模型卡给出的模型结构是：125B 主模型、每 token 激活 6B，另有 51B n-gram embedding 和 4B MTP；原生上下文 262,144，可用 YaRN 扩展到 1,000,000；模型是带视觉编码器的多模态模型。也就是说，“176B”通常是主模型加 n-gram 表、不含 MTP 的口径；含 MTP 的存储参数约为 180B。Tony README 中的“125B-A3B”与官方 6B activated 口径不一致，应以[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/README.md)为准。

模型权重使用 `qwen-community-1.0`，不是 MIT/Apache；仓库代码许可证与模型权重许可证必须分别审查，参见[官方 LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)。

截点时两条关键运行时支持仍在快速变化：

- llama.cpp 的 [`#27742`](https://github.com/ggml-org/llama.cpp/pull/27742) 仍为 open、未合并，当前 head 为 `ef9fa1ba1f0d3f11ed7ddc1da94e5db4c22ae7b6`。
- SGLang 的 Qwen3.8-Flash-Next 支持 [`#36497`](https://github.com/sgl-project/sglang/pull/36497) 仍为 open、未合并，且 PR 页面显示尚未触发完整 CI。特制 day-0 image 能运行不等于某个稳定 SGLang release 已提供相同支持。
- thinking + `qwen3_coder` tool parser 的 token-0 `!` 循环 [`#36537`](https://github.com/sgl-project/sglang/issues/36537) 仍为 open；禁用 thinking 是当时已验证的 workaround，但会改变产品能力。

## 固定快照

| 仓库 | 固定 commit | 创建/最近推送（UTC） | 归档截点状态 | 代码许可 |
|---|---|---|---|---|
| 0xBakeer | [`2ab233f`](https://github.com/0xBakeer/qwen38-flash-next-spark/commit/2ab233f53629244168ba7eb56fb4322e1a279046) | 2026-08-26 19:21 / 23:16 | 26 stars，0 issue，0 PR，0 release | MIT |
| tonyd2wild | [`c96c6a3`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/commit/c96c6a3927c4bda566c6454e51966018ddc69b83) | 2026-08-26 14:57 / 23:47 | 13 stars，0 issue，0 PR，0 release | 仓库未检测到 LICENSE |
| MiaAI-Lab | [`02cd6d9`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/commit/02cd6d9050b386644c65ebf2396324da9575e5ee) | 2026-08-26 17:15 / 23:59 | 57 stars，2 个 issue、4 个 PR open，0 release | MIT |

三个仓库都在模型发布当日附近创建，历史只有 6–9 个主分支提交，且都没有 release/tag。它们是快速演进的实验快照，不应按稳定发行版消费。

## 总览比较

| 维度 | 0xBakeer | tonyd2wild | MiaAI-Lab |
|---|---|---|---|
| 核心目标 | 单 Spark 容量突破 | 双 Spark 现场部署与极限调优 | 双 Spark 一键部署与运维 |
| 硬件 | 1× GB10 / 128 GB | 2× GB10，TP2 | 2× GB10，TP2 |
| Runtime | llama.cpp PR `#27742` + 2 patch | SGLang day-0 image + 自定义 SM121 guard image | SGLang day-0 image + 脚本内嵌 Triton QSA fallback |
| Target | Unsloth UD-Q4_K_XL GGUF，约 103.7 GiB | RadixArk NVFP4，本地约 126 GB | RadixArk NVFP4，约 135 GB |
| PLE 放置 | CPU backend + mmap/NVMe page cache | `--ple-offload-embedding`，GB10 host-pinned UMA | 自动/可配置 host offload |
| 推测解码 | `ngram-mod`，无 MTP GGUF | 内置 MTP，NEXTN 3/1/4 | 内置 MTP，NEXTN 3/1/4 |
| 并发定位 | `--parallel 1`；第二请求会 abort | `max-running-requests=6` | 默认 `max-running-requests=16` |
| API 默认暴露 | `127.0.0.1:30000` | `0.0.0.0:8000`，无 auth | `0.0.0.0:8888`，API key 可选、默认空 |
| 最强证据 | benchmark 脚本、页缓存工具、明确限制 | 现场时间线与故障收据 | doctor/download/sync/serve/status/stop/smoke 完整 |
| 主要短板 | 单流、上游未合并、模型 revision 未锁 | 无许可证、无 benchmark harness、image 未锁、公开无鉴权 | README 与执行默认值漂移、image/model 未锁、open 正确性与 NCCL 问题 |
| 建议定位 | 单用户实验 lane | 故障知识库，不直接部署 | 加固后的双机 fork 起点 |

## 1. 0xBakeer：单机 llama.cpp + NVMe PLE

### 它真正证明了什么

固定 commit 的方案使用 `unsloth/Qwen3.8-Flash-Next-GGUF` 的 `UD-Q4_K_XL`，通过：

```text
-ot per_layer_token_embd=CPU
-lm mmap
```

将约 51.2B 参数的 `per_layer_token_embd.weight` 当作稀疏查询表留在 NVMe/page cache，而让约 76.9 GiB 计算权重常驻统一内存。作者报告模型文件约 103.7 GiB、启动约 3 分 35 秒、常态总内存约 95 GiB。实现、内存算术和复现实验都在固定 commit 的 [`README`](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/2ab233f53629244168ba7eb56fb4322e1a279046/README.md)、[`how-it-works.md`](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/2ab233f53629244168ba7eb56fb4322e1a279046/docs/how-it-works.md) 和 [`run.sh`](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/2ab233f53629244168ba7eb56fb4322e1a279046/run.sh) 中可核对。

这条路线的亮点不是 22 tok/s，而是把 PLE 的稀疏访问特性真正转化成了单 Spark 可运行的存储分层方案。作者还提供：

- `mincore` page-cache 观测和顺序 warm 工具；
- 按任务新颖度区分的 benchmark；
- llama.cpp 量化时分带处理超大 PLE tensor 的 patch；
- 为 qwen4exp graph input 增加 `can_reuse()` 的 patch。

### 性能证据边界

作者报告的基础自由生成约为 22 tok/s。`ngram-mod` 在输出大量复用输入时提升明显：文件小改 52.6 tok/s（cold）到 74.6 tok/s（warm），定点 bug fix 46.0 到 51.6；新增函数约 30–32；自由 prose 约 22–23。完整表见固定 commit 的 [`bench/results.md`](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/2ab233f53629244168ba7eb56fb4322e1a279046/bench/results.md)。

这不是“代码任务普遍 3×”，而是上下文 n-gram 能复用已有 span 时的精确 speculation。对生成全新 strict JSON 或新诊断文本，应该用接近 22 tok/s 的 floor 做容量规划。

“完整 262K context works”主要证明 server 能按该窗口启动/分配；公开吞吐只测到约 19K prompt，`19K–262K` 的吞吐、page-fault 和正确性没有曲线。不要把“窗口可配置”写成“262K 长上下文已验证”。

### 阻塞项与文档漂移

- 并发 2 会触发 QSA indexer/KV cache assert，因此只能 `--parallel 1`。
- quantized KV 会 abort，只能保持 f16 KV。
- GGUF converter 不导出 MTP；外部 Qwen3.5-0.8B draft 虽有 2.88 acceptance length，但没有吞吐增益。
- 文档一处写“no speculative decoding”，另一处默认启用 `ngram-mod`。准确表述应是：**无可用 MTP/外部 model drafter，但 context n-gram speculation 可用于高复制率任务。**
- `run.sh` 固定 llama.cpp PR 到 `035e227`，而归档截点上游 PR head 已移动到 `ef9fa1...`；这是合理的 patch pin，但不是最新 upstream 状态。
- GGUF 仓库没有固定 revision，setup 还会升级未固定版本的 `huggingface_hub`。仅固定本 repo SHA 仍不足以重建完全相同的 artifact。

### 采用判断

保留为“单 Spark / 单用户 / 大模型容量与 copy-heavy coding”实验 lane。它不适合作为多租户或并发 agent 后端，也不能从 copy-heavy 数字推导 `oper-v7` strict JSON 性能。

## 2. tonyd2wild：双机 SGLang 现场部署日志

### 有价值的部分

该仓库保留了从 vLLM 架构 probe 失败、切换 SGLang、SM121 QSA kernel guard、TP2 worker-first 启动、MTP4、UMA OOM 到 `!` 循环的连续时间线。对现场排障很有价值，尤其是：

- 明确记录自定义 image 的 base digest、SGLang commit 和代表性 QSA kernel probe；
- 解释 GB10 的 OS page cache、pinned host PLE 和 CUDA allocation 共用同一 128 GB UMA；
- 记录双节点必须先完整 teardown、worker-first、再 head 的启动顺序；
- 将 1.05M KV 的激进配置回退到 `--max-total-tokens 600000 --mem-fraction-static 0.80`，换取约 23 GB headroom；
- 保留 vision，并把偶发 multimodal RoPE CUDA graph assert 的 fallback 明确为关 decode graph、峰值约从 70 降至 55。

当前实际权威入口应是固定 commit 的 [`launch-qwen38fn-sglang-tp2.sh`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/blob/c96c6a3927c4bda566c6454e51966018ddc69b83/launch-qwen38fn-sglang-tp2.sh)，不是报告中任一历史命令块。

### 作者报告的性能

作者在 TP2、MTP4 3/1/4、CUDA graph on 下报告：无 MTP 约 20 tok/s，混合 prompt 典型约 47，结构化/可预测输出峰值约 70.2；graph off 的稳定 fallback 峰值约 55。数据与故障过程见固定 commit 的 [`README`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/blob/c96c6a3927c4bda566c6454e51966018ddc69b83/README.md) 和 [`DEPLOY-REPORT`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/blob/c96c6a3927c4bda566c6454e51966018ddc69b83/DEPLOY-REPORT.md)。

仓库没有 benchmark harness、raw request/result 文件、重复/方差、并发阶梯或质量套件，不能与 0xBakeer 或 Mia 数字直接排名。“agentic output 接近峰值”是作者推断，不是 `oper-v7` 的 workload 证据。

### 阻塞项与内部不一致

- README 把模型写成 125B-A3B；官方是 125B、6B activated，另加 51B PLE 和 4B MTP。
- DEPLOY-REPORT 同时包含初始 `0.78 + graph off`、后续 `0.82 + 1.05M KV` 和 README 的 `0.80 + 600K pin`。标题中的“exact final”会误导；应按当前 launcher 解析。
- README 自述一次约 90 分钟 load 后出现 multimodal-RoPE device assert，因此不能称 24/7 已稳定。
- thinking + tools 的 `!` loop 通过 server-wide thinking off、radix off、PyTorch sampling 等组合规避；这改变了模型的 thinking 能力，且不是上游修复。
- 自定义 image 使用可变 tag `radixark/sglang-qwen38flashnext:sm121-qsa`，launcher 不校验 digest；模型目录也没有权重 hash/revision receipt。
- 仓库没有 LICENSE，不能默认继承其他项目许可。

### 网络与供应链风险

launcher 默认 `--host 0.0.0.0` 且没有 API key，README 还记录了可路由 endpoint。只能放在受控内网/防火墙后，或改为 loopback 并通过带真实鉴权的 gateway 暴露。`--trust-remote-code`、自定义 image tag、外部 NCCL `.so` 都需要 digest/hash 和来源清单后才能进入可重复部署。

### 采用判断

作为 SM121、NCCL、MTP、CUDA graph、UMA/OOM 与 agent loop 的故障知识库保存；不直接以其 launcher 建生产服务。若吸收代码，只提取经 upstream diff 核对的最小 kernel guard 和内存门，不复制现场 IP、无鉴权暴露或未锁 image。

## 3. MiaAI-Lab：双机一体化编排脚本

### 有价值的部分

三者中它的操作闭环最好。单个 `start.sh` 覆盖 doctor、权重下载与检查、head-to-worker rsync、两端 image build、worker-first TP2 launch、readiness、status、logs、smoke 和 stop；`.env.example` 把 fabric、NCCL、memory、context、API key 等关键项显式化。固定代码见 [`start.sh`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/blob/02cd6d9050b386644c65ebf2396324da9575e5ee/start.sh) 与 [`.env.example`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/blob/02cd6d9050b386644c65ebf2396324da9575e5ee/.env.example)。

作者报告 TP2/NEXTN 3/1/4 的结构化 benchmark：C1 64.4 tok/s，C2 aggregate 116.8，C4 aggregate 114.1；但仓库没有 raw benchmark artifact 或独立复现。

### 最重要的配置漂移

同一个固定 commit 内，README 和真正执行路径不一致：

| 配置 | README 宣称的高性能/长上下文值 | `start.sh` / `.env.example` 默认值 |
|---|---:|---:|
| context | 900,000 | 262,144 |
| mem fraction | 0.82 | 0.70 |
| chunked prefill | 1024 | 4096 |
| mamba full memory ratio | 0.3 | 未设置 |

Quick start 要求复制 `.env.example`，所以按文档直接运行得到的是右列，而不是 README benchmark 配置。900K 与 956,800-token pool 只能视为作者报告的另一个实验 profile，不能归因于仓库默认 recipe。归档和复跑必须显式保存完整 `.env`，不能只写 repo SHA。

### 截点时的开放问题

- [`issue #3`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/3)：另一组相同类型的双 Spark 在所有 SGLang image 上卡在 NCCL init，而 vLLM isolation test 可通过，说明 fabric/image/torch/NCCL 组合仍有环境敏感性。
- [`issue #5`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/5)：首请求 doom loop。
- [`PR #6`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/6)：贡献者把 loop 归因于 online-softmax kernel，修复后自报性能从仓库宣称的 64 降到约 31 tok/s，且直言未复现 64。这一分歧在截点仍未解决。
- [`PR #7`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/7)：继续修补 API key 与脚本自身 curl 的一致性，说明刚合入的第一版鉴权仍有边界问题。

这些信号不证明主分支一定错误，但足以否决“已生产稳定”表述。

### 网络与供应链风险

- 默认绑定 `0.0.0.0`，`API_KEY` 默认空；新合入的 key 支持是改善，但安全默认值仍是开放 LAN 服务。
- `BASE_IMAGE=lmsysorg/sglang:qwen38flashnext` 是 tag，不是 digest；`HF_REVISION` 默认空，模型跟随仓库主分支。脚本自己的 patch stamp 不能覆盖 base image/model 漂移。
- `HF_TOKEN` 被加入长运行 server container 的环境变量，可能通过容器 inspect 或同权限运维面暴露；下载完成后不应继续注入 serve container。
- `--trust-remote-code` 与自动构建 kernel patch 需要在隔离构建环境中执行，并生成 image digest、SBOM/来源清单和 patch hash。

### 采用判断

它是三者中最值得 fork 的双 Spark 起点，但应先完成：

1. 固定 model revision、base image digest、NCCL SHA256、SGLang/source commit 和 patch hash；
2. 让 loopback 或强制 API key 成为默认值，gateway 才能对外；
3. 下载容器与 serving 容器分离，移除 serving 环境里的 HF token；
4. 选择并验证唯一 kernel path，解决 PR #6 的正确性/吞吐分歧；
5. 让 README 的默认值、`.env.example` 和 benchmark metadata 由同一配置生成；
6. 保存 raw benchmark、请求体、server log、GPU/UMA telemetry、重复次数和失败样本。

## 性能数字应如何读取

| 仓库/路径 | 作者报告的 decode | 能说明什么 | 不能说明什么 |
|---|---:|---|---|
| 0xBakeer AR/novel prose | 约 22 tok/s | 单 Spark Q4 自由生成 floor | 并发、strict JSON、262K 实际长 prompt |
| 0xBakeer `ngram-mod` copy-heavy | 约 46–75 tok/s | 高输入复用编辑可获大幅精确 speculation 收益 | 新代码/新 JSON/一般 agent 吞吐 |
| Tony TP2 MTP4 graph on | 约 47 typical / 70 peak | 双 Spark 的强先验与故障边界 | 可重复分布、C8、长稳、质量 |
| Tony graph-off fallback | 峰值约 55 | vision 保留时的稳定性 fallback 先验 | 已完成 24/7 qualification |
| Mia TP2 NEXTN | C1 64.4；C2 agg 116.8 | 作者机器上的结构化 serving 结果 | 其他双 Spark 可复现；kernel 正确性；README 默认配置结果 |

这些数字使用了不同 quant、runtime、prompt、新颖度、计时脚本和并发，不能制作“最快仓库”排名。

## 对 `oper-v7` 的落地门

若后续在自有 Spark 上继续，建议建立三条彼此独立的 candidate，不把 artifact 混在同一结果表中：

1. llama.cpp GGUF `target-only` 与 `ngram-mod`；
2. SGLang NVFP4 autoregressive；
3. 同一 SGLang target 的 NEXTN 3/1/4。

每条 lane 固定：repo SHA、model revision/hash、image digest、runtime commit、patch hash、NCCL SHA、chat template、thinking、sampling、context、KV、PLE placement、CUDA graph、client commit 和完整请求体。

最低门：

- `response_format=json_object` 与真实 `oper-v7` schema，thinking off，`temperature=0`，`max_tokens=2048`；
- 同 target/config 的 AR 与 speculation 输出一致性；
- C1/C4/C8，2K/8K/32K input，真实 trace 与 copy-heavy 对照分开；
- 至少 7 次交错重复，报告 p50/p95、CV、TTFT、TPOT/ITL、E2E、goodput；
- 1,000 请求 crash/timeout/JSON/工具统计，8 小时 soak，重启与缓存冷暖分开；
- UMA resident/page cache/pinned host/CUDA pool、major fault、swap/pageout、功耗与温度；
- 断开 worker、NCCL hang、OOM、device assert、token-0 loop、API auth、回滚演练；
- 真实网关鉴权，只暴露 gateway，engine 保持 loopback/受控 fabric。

在这些门完成前，归档评级为：

- **0xBakeer：A 级容量/机制证据，C 级产品性能证据；单流实验。**
- **tonyd2wild：A 级现场故障叙事，D 级可重复发布证据；只作知识库。**
- **MiaAI-Lab：B 级部署骨架，C 级性能/稳定性证据；加固后进入双机实验。**

## 维护提示

本主题变化速度极快。再次使用本文前至少重查：三个仓库 `main`、llama.cpp `#27742`、SGLang `#36497/#36537`、Mia open PR/issue、模型 revision、day-0 image digest 和官方许可证。不要把本文的 star、open/merged 状态或 tag 当成当前值。
