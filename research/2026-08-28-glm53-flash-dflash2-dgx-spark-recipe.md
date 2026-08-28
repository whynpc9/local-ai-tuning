# GLM-5.3-Flash DFlash2 / 单 DGX Spark recipe 调研归档

> 归档截点：2026-08-28（Asia/Shanghai）。本文固定核对 [`vcruz305/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe`](https://github.com/vcruz305/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe) 的首个 `main` commit [`04134f1`](https://github.com/vcruz305/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe/commit/04134f119542f393d86c945fc198558013ed6df0)，并交叉检查目标/草稿模型卡、作者 GGUF 仓库、llama.cpp DFlash2 上游 PR 与仓库脚本。本文没有在 DGX Spark 上复跑；所有吞吐、接受率、内存和故障均按“作者报告”处理。

## 结论

这个仓库是一个**单台 128 GB DGX Spark 上运行 GLM-5.3-Flash 的容量与 speculative-decoding 实验**：用约 108.7 GiB 的社区 Q2_K GGUF target，加 Inco GLM-5.3 DFlash2 draft，在作者的 llama.cpp fork 上以 `draft-dflash`、单序列、65,536 context、Q8 KV 启动。

它当前最值得归档的不是“43.43 tok/s”，而是三件事：

1. 给出一条单 Spark 能装下 320B/18B-active GLM-5.3-Flash 的明确 artifact 路线；代价是 target 变成 2.91 BPW 的社区 Q2_K，不能与官方 BF16、双 Spark NVFP4 或 API 质量默认等价。
2. 清楚区分 DFlash2 与模型内置 MTP，并把 DFlash2 block size 8 映射为 `n_max=7`。
3. 对结果边界相对诚实：43.43 tok/s / 94.4% 是高度重复 prompt 的调参 ladder；唯一短自然 prompt 只有 17.58 tok/s / 31.41%，65K serve 也未在本次 DFlash2 实验中重新量测。

因此，本项目应把它放在“**单 Spark / 低比特 target / C1 实验 challenger**”，而不是默认部署候选。进入 `oper-v7` 前至少还缺 artifact hash/revision、同窗 AR/MTP/DFlash2、真实 strict JSON 与工具调用、长上下文、重复/方差、稳定性、资源 telemetry 和回滚证据。DFlash2 draft 的 CC BY-NC-ND 4.0 还使这条 lane 默认仅适合研究与评估，商业使用需另行授权。

## 调研方法与证据等级

本次使用 Exa 检索 4 个工作流、共审阅 40 条搜索结果，去重后优先保留仓库、模型卡、上游 PR 和作者 artifact 仓库等第一方来源。结论按以下口径表达：

- **源码已核对**：固定 commit 的 README、两个 shell 脚本、LICENSE 或上游状态可直接证明。
- **作者自测**：作者报告在一台 GB10 上实测，但本工作区未复跑，且公开仓库没有 raw log/result 文件。
- **未证明**：配置可表达或 README 提到过，但缺少同一 DFlash2 profile 的运行收据。

动态状态只代表归档截点。

## 固定快照

| 项目 | 截点状态 |
|---|---|
| Recipe | [`04134f119542f393d86c945fc198558013ed6df0`](https://github.com/vcruz305/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe/commit/04134f119542f393d86c945fc198558013ed6df0)，2026-08-28 02:32:42 UTC，unsigned |
| 仓库成熟度 | 创建约 1 分钟即首发；1 commit、4 files、0 stars、0 forks、0 issue/PR、0 release/tag |
| Recipe 代码许可 | MIT；仓库 LICENSE 明确把模型权重许可分开 |
| Target | [`vcruz305/GLM-5.3-Flash-GGUF`](https://huggingface.co/vcruz305/GLM-5.3-Flash-GGUF) 的 `GLM-5.3-Flash-Q2_K.gguf`，116,728,212,640 B，2.91 BPW |
| Target 来源 | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)，官方模型卡为 320B total / 18B active、原生多模态、混合 sparse/linear attention |
| Draft | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)；社区 BF16 GGUF 2,352,022,432 B，作者本地 Q4_K_M 697,017,248 B |
| Draft 许可 | CC BY-NC-ND 4.0，仅研究/评估；商业许可需联系 Inco |
| Runtime | `vcruz305/llama.cpp@4a06ec6`，父 commit `6f5ac9a` + aarch64 `<cmath>` 修复 |
| 上游状态 | llama.cpp [`#27342`](https://github.com/ggml-org/llama.cpp/pull/27342) 已于 2026-08-27 合并为 `4a6ad487...`；recipe 仍固定作者 fork |

README 顶部把 fork 写成 `main / glm5next-mtp`，但可重复命令实际固定 `4a06ec6`；应以后者为归档基线。仓库只有 README、LICENSE、serve 脚本与单 prompt benchmark 脚本，没有安装器、权重 manifest、checksum、CI、测试 fixture、raw results 或环境采集。

## 实际执行路径

固定脚本 [`serve_one_spark.sh`](https://github.com/vcruz305/GLM-5.3-Flash-DFlash2-DGX-Spark-recipe/blob/04134f119542f393d86c945fc198558013ed6df0/scripts/serve_one_spark.sh) 的默认服务配置是：

| 维度 | 默认值 | 判断 |
|---|---|---|
| 网络 | `127.0.0.1:8891` | 安全默认值合理；需要另设有鉴权 gateway 才能远程暴露 |
| 并发 | `-np 1 --no-kv-unified` | 明确是单流 lane，不可外推 C4/C8 |
| Context | 65,536 | 作者明确称本次未以 DFlash2 重测；不是已验证长上下文结果 |
| Speculation | `draft-dflash`, `n_max=7`, `p_min=0.30` | 与 draft block size 8 一致；不是 MTP |
| KV / attention | FA on，K/V 均 Q8_0 | 低内存 profile；质量与长上下文仍需门控 |
| Fit | off | 作者报告 fitter 在约 96K 以上不可靠，MTP 131K 曾 SIGKILL 137 |
| Target | Q2_K，全部 GPU layers | 单 Spark 容量核心，也构成最大质量变量 |
| Draft | 优先本地 Q4_K_M，不存在则 BF16 | Q4 文件并未发布到 Hub；复现者必须本地量化 |

脚本接受环境变量覆盖路径、host、port、context，并把额外 CLI 参数追加在最后。它检查文件存在，但不检查 runtime commit、模型 revision、文件大小/hash、GPU 身份、可用 UMA、swap/page cache 或端口占用；也没有 readiness、smoke、stop、status、日志留存或回滚闭环。

## 性能证据应该怎样读

| 证据 | 作者报告 | 能证明 | 不能证明 |
|---|---:|---|---|
| 唯一短 prompt：`The capital of France is`，64 tokens | 17.58 tok/s；31.41% accept | 该 binary/target/draft 能在 GB10 执行 DFlash2 | 一般 coding/chat 吞吐；与 AR/MTP 的公平增益 |
| 重复 GB10 句子，Q4 draft，n=128 | 43.43 tok/s；94.4% accept | 高重复输入下 `n_max=7` 是此 ladder 的最佳点 | 通用接受率、质量或真实 agent 工作负载 |
| 同一重复文件的 MTP-3 control | 28.23 tok/s；73.2% | 在该特定 fixture 上 DFlash2 约 1.54× | 不同 prompt/长度/并发下仍胜出 |
| BF16 + `p_min=0.30` | 42.56 tok/s；100% accept | 重复 fixture 可被完整预测 | 任何模型准确率或业务通过率 |

仓库没有 autoregressive baseline：作者记录 `llama-speculative-simple --spec-type none` 初始化失败，而不是 AR 模型本身不能运行。benchmark 脚本只有一个短 prompt、一次运行、无 raw JSON、无 TTFT/E2E/方差/质量判定；README 的 ladder 还依赖未提交的 `bench-prompt.txt`。所以不能据此给 DFlash2、MTP、AR 做普遍排名。

Inco 的官方 draft card 在 4×GB300/SGLang 上用 GSM8K、MATH-500、HumanEval、MBPP、MT-Bench 报告 DFlash2 相对 MTP/AR 的优势，但硬件、runtime、target 精度、采样和并发都不同，只能证明算法具有更广的上游证据，不能替代这份 GB10 recipe 的本地 qualification。

## 正确性、能力与许可边界

- **质量变化首先来自 Q2_K target。** Speculative decoding 理论上保持 target 分布，但只能保持这个社区 2.91 BPW target 的分布，不能恢复官方 BF16 能力；仓库没有给出 Q2_K 对 BF16/NVFP4 的质量等价证据。
- **多模态未闭环。** 官方 target 是多模态；recipe 未下载 `mmproj`、serve 脚本也未传 `--mmproj`。归档应按 text-only 处理。
- **reasoning 只做了参数提示。** README 称 jinja 只可靠接受 `low/high`，且 `max` 不退出 think；官方卡则定义 `low/high/max`。这是社区模板/runtime 行为差异，未附请求/响应 fixture。
- **tools/structured JSON 未验证。** `--jinja` 不等于工具调用、schema adherence 或 stop 行为正确；仓库没有 OpenAI API smoke 请求。
- **DFlash2 draft 不可默认商用。** CC BY-NC-ND 4.0 与 recipe MIT、target MIT 是三份独立许可；即使只加载权重也应先完成商业授权判断。

## 可复现性与供应链缺口

优点是 runtime commit 精确，服务只绑定 loopback，作者还记录了 target/draft byte size和部分 metadata。主要缺口是：

1. 两条 `hf download` 均未传 `--revision`；Hub 上同名文件未来可漂移。
2. 没有 SHA256/LFS oid manifest；“Hub tree size match (`ffa0b8d`)”不足以成为独立 artifact receipt。
3. 默认 Q4 draft 不在 Hub；本地量化命令没有记录 quantizer binary hash、输入 SHA256 或输出 SHA256。
4. runtime 是 unsigned 社区 fork，且上游 DFlash2 已合并；需要比较 `4a06ec6` 与正式 upstream merge 后的 GLM5/SM121 差异，再决定保留 fork 还是收敛 upstream。
5. 构建命令 `-j` 未限并行度，也没有保存 compiler/CUDA/CMake/driver/OS 版本和最终 binary hash。
6. 没有 clean-machine bootstrap、CI、health/readiness、长期日志或故障恢复演练。

## 与现有候选的关系

这条路线不应与双 Spark NVFP4 vLLM/SGLang recipe 合并成一个“GLM-5.3 最快方案”排行：

- 它用一台 Spark、Q2_K GGUF、llama.cpp、C1；优势是容量与简单的 loopback 服务。
- 双 Spark方案常用约 181–195 GiB 的 NVFP4 target、TP2、保留更多非专家 BF16 权重；它们的质量、上下文、并发和故障面都不同。
- DFlash2 是外部 drafter；模型 native MTP 是另一个 artifact/runtime contract。应在同一 target、同一 prompt、同一 boot window 下分别比较。

在本项目中建议登记为：`glm53 / 1x-dgx-spark / llama.cpp / community-q2k / dflash2 / text-only / c1-experimental`。

## 进入 `oper-v7` 前的最低验证门

1. 固定 repo SHA、upstream/fork SHA、target/draft Hub revision、所有 GGUF SHA256、构建环境和 binary SHA256。
2. 先用同一 Q2_K target 做 AR、native MTP、DFlash2 三路；若 AR harness 有缺陷，改用 server API 或正确的 non-spec binary，不能留空基线。
3. 分别验证 thinking off/low/high、stop、UTF-8、工具调用、`response_format` 与真实 strict JSON schema；保存原始 request/response。
4. 使用真实 `oper-v7` trace、novel coding、copy-heavy 和重复 fixture 四类，不混合报告；C1 至少 7 次交错重复，给出 TTFT、TPOT/ITL、E2E、decode、accept length、p50/p95/CV。
5. 对 2K/8K/32K/65K prompt 做正确性与 prefill/major-fault/UMA 曲线；不要把 `-c 65536` 启动成功当成长上下文通过。
6. 做至少 1,000 请求与 8 小时 soak，记录 crash、timeout、JSON/tool failure、OOM/SIGKILL、pageout/swap、温度/功耗和 cache 冷暖。
7. 对 Q2_K 与可接受的高精度 control 做领域质量门；若未证明等价，只能作为独立低比特 candidate。
8. 商用前取得 DFlash2 权重许可结论；服务继续保持 loopback，只通过带真实鉴权、限流和审计的 gateway 暴露。

## 采用建议

保留并跟踪，但不直接部署。短期最有价值的复跑是：固定全部 artifact 后，在同一 Spark 上对一个可运行的 AR baseline、MTP-3 与 DFlash2 `n=3/5/7` 做小型真实 trace A/B，同时记录可见答案质量和 UMA。若 DFlash2 只在重复 fixture 显著领先，归档为 copy-heavy specialty lane；只有在 novel coding、strict JSON 和真实 agent trace 上仍有稳定 goodput 优势，才升级为 `oper-v7` challenger。

