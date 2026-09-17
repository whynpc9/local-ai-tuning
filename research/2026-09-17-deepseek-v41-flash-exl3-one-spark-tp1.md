# DeepSeek-V4.1-Flash EXL3 / 单 DGX Spark TP1 配方登记

> 登记日期：2026-09-17。已只读克隆并核对配方及 ExLlamaV3 fork 的固定提交；未安装依赖、下载或重排模型、构建扩展、运行 GPU 或验证 API。下列性能和内存数字均为作者报告。

## 项目与固定版本

| 项目 | 内容 |
| --- | --- |
| 用户提供的入口 | [one-spark-tp1](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/tree/main/one-spark-tp1) |
| 配方提交 | [`6cd44b3bd8478922396fd623a39010ba09d4de7b`](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/commit/6cd44b3bd8478922396fd623a39010ba09d4de7b) |
| 提交时间 / 说明 | `2026-09-16T08:18:07-07:00` / `one-spark-tp1: native ExLlamaV3 TP1 recipe, measured numbers and TabbyAPI config (#20)` |
| 引擎 | `vcruz305/exllamav3`，分支 `feat/gb10-ats-load`，固定 [`954a8ca6e59d48c3e3462068ecf083fe9990f4dc`](https://github.com/vcruz305/exllamav3/commit/954a8ca6e59d48c3e3462068ecf083fe9990f4dc) |
| 模型入口 | 作者指定 `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw`；实测使用本地 64-byte 重排版本，该精确产物未发布 |
| 硬件 / 工具链 | 单 DGX Spark / GB10、128 GB 统一内存、ATS addressing mode、Linux aarch64、CUDA 13.0、`TORCH_CUDA_ARCH_LIST=12.1a` |
| 定位 | 原生 ExLlamaV3 TP1 研究候选；TabbyAPI 仅配置指导，尚未验收 |

此目录独立于同仓库 TP2/TP4 的 vLLM + vllm-exl3 路线，不使用根目录 `runtime.lock.json` 的运行时锁、Dockerfile 或集群 overlays。也不同于此前登记的 [0xBakeer 自研 PyTorch / CB3 / NVMe 项目](2026-09-14-deepseek-v41-flash-spark.md)，不能合并性能或质量结论。

## 内存机制与复现入口

ATS 让 GPU 访问进程主机虚拟地址；fork loader 可从共享只读 safetensors mmap 别名加载权重，减少额外复制。实际高性能配置仍将主模型复制到 CUDA，仅把 `mtp.*` drafter 留在可回收 page cache。作者报告主模型约 107 GiB、drafter 约 14 GiB，两者全常驻无法容纳；加载约 40 秒后 `MemAvailable` 约 5 GiB。

原始 pack 中不满足对齐要求的张量会回退复制。作者报告重排前 48.6 GiB alias、67.4 GiB copy，64-byte 重排后 text-model 张量可全部 alias；这是 alias 能力统计，不是“主模型复制 + drafter alias”最终配置的放置统计。

准备顺序：固定 fork → 安装匹配的 Python/PyTorch/CUDA 依赖并构建 aarch64 扩展 → 用 `util/align_safetensors.py` 重排权重 → 提供原生 driver → 执行配方启动器。重排示例使用 `--align 64 --min-bytes 1048576 --skip .engram.embed. --jobs 4`。未改动文件通过符号链接引用原目录，必须保留原目录，并为重写分片准备额外磁盘空间。生成的 `__align_pad__.*` 张量由 fork loader 跳过，其他 loader 不一定兼容。

| 启动参数 | 配方默认值 / 含义 |
| --- | --- |
| `EXL3_ATS_MMAP` | `1` |
| `EXL3_ATS_COPY` | `^(?!mtp\.)`，非 `mtp.*` 张量复制进 CUDA |
| `EXL3_DSPARK_CONF` / `DRAFT` | `0.7` / `1` |
| `CTX` / `CHUNK` | `6144` / `2048` |
| `PREWARM` | `1` |
| `MIN_AVAIL_KB` | `104857600`，启动前至少 100 GiB MemAvailable |
| `MODEL_DIR` / `ENTRY` | 必填：重排模型目录 / 用户提供的 Python driver |

源码确认 loader 读取 ATS 与 copy 正则、过滤 padding 张量，V4.1 MTP 读取置信阈值。启动器最终只执行 `python "$ENTRY"`；`CTX`、`CHUNK`、`DRAFT` 等变量是否被采用还取决于 driver，不能把设置环境变量当作完整运行验收。非 ATS 在脚本里只是 warning，重复进程检查也只是命令行模式匹配。

构建说明存在依赖准备缺口：示例创建空 venv 后用 `pip install --no-build-isolation --no-deps .`，没有完整依赖锁和安装步骤；固定 fork 中未见 `util/patch_exllamav3_aarch64.py`，配方允许从另一个 `vllm-exl3` 仓库取补丁，但未固定其提交，且示例用 `|| true` 忽略补丁失败。复现前需补齐并固定这些输入。

来源：[配方 README](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/blob/6cd44b3bd8478922396fd623a39010ba09d4de7b/one-spark-tp1/README.md)、[启动脚本](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/blob/6cd44b3bd8478922396fd623a39010ba09d4de7b/one-spark-tp1/scripts/run_tp1.sh)、[fork ATS 说明](https://github.com/vcruz305/exllamav3/blob/954a8ca6e59d48c3e3462068ecf083fe9990f4dc/doc/gb10_ats_loading.md)、[loader](https://github.com/vcruz305/exllamav3/blob/954a8ca6e59d48c3e3462068ecf083fe9990f4dc/exllamav3/loader/safetensors.py)、[V4.1 MTP](https://github.com/vcruz305/exllamav3/blob/954a8ca6e59d48c3e3462068ecf083fe9990f4dc/exllamav3/architecture/deepseek_v41_mtp.py)。

## 作者报告的性能

共同条件：TP1、单序列、`max_batch_size=1`、`CTX=6144`、EXL3 per-expert mixed K（K1–K6）；启用 DSpark 时 block size 5、confidence 0.7。数值来自直接驱动 ExLlamaV3，不是 HTTP 服务吞吐。

| 场景 | 作者结果 | 限定 |
| --- | --- | --- |
| 无 drafter，全部 alias | decode 13.96–14.19 tok/s | 权重可回收 |
| 无 drafter，主模型复制 | decode 15.13–15.22 tok/s | 加载 37.5 s，约 107 GiB 常驻 |
| alias + drafter | fresh prompt 中位数 11.46；repeat 16–17 tok/s | 分开保留冷热口径 |
| 主模型复制 + drafter alias | fresh 中位数 17.53、均值 19.82；repeat 20.11–24.67 tok/s | acceptance 0.889 |
| 交互聊天 | cold 11.8、warm 17.4 tok/s | acceptance 0.74；不是上行数字的重复测量 |
| warm prefill，chunk 4096 | 254–261 tok/s | 4K–6K，不能用于主模型 CUDA 常驻配置 |
| 主模型 CUDA 常驻 prefill，chunk 2048 | 154–229 tok/s | 2K–6K |

负面实验也保留：grouped MoE CUDA graph 两模式分别 9.69 / 10.43 tok/s，同 harness 基线 10.97；huge pages 约 1–2% 收益处于波动范围；强制最短 draft 更慢、early-exit 中性；`EXL3_MOE_MIXED_BSZ1=1` 虽约提升 5% warm decode，但 greedy 输出跨次不复现，不作为基线。

所读 TP1 目录只有汇总表，没有配套逐请求原始日志、完整 prompt 集、样本量及可直接复跑的测量 driver。模型 revision/hash、重排产物校验、thinking 与可见答案口径也未完整固定，故这些数字仅用于候选登记，不与其他模型/路线直接排名。超过 6144 上下文、量化 KV、混合 K CUDA graph 及中文/业务质量均未获得此配方的验收证据。

来源：[BENCHMARKS.md](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/blob/6cd44b3bd8478922396fd623a39010ba09d4de7b/one-spark-tp1/BENCHMARKS.md)。

## TabbyAPI 与交付边界

[配置](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/blob/6cd44b3bd8478922396fd623a39010ba09d4de7b/one-spark-tp1/tabbyapi/config.yml)指定 `127.0.0.1:5000`、`disable_auth: false`、FP16 KV、6144 context/cache、chunk 2048、batch 1、`tensor_parallel: false`、`vision: false`、`draft_mode: mtp`。其中路径 `/home/markus/models` 是作者环境值，`draft_num_tokens` 留空；不能认定配置已强制采用实测的 block size 5。

[TabbyAPI 说明](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/blob/6cd44b3bd8478922396fd623a39010ba09d4de7b/one-spark-tp1/tabbyapi/README.md)明确未做该 pack 在 Spark 上的端到端验证，待确认同 pack MTP 是否需要 `draft_model_name`、loader 是否保留 ATS 放置、padding 兼容与量化 KV。TabbyAPI 安装/升级可能替换自定义 ExLlamaV3，需固定服务端版本并核验实际加载的 fork。此处不登记为可用 OpenAI API 服务。

上游说明原生 ExLlamaV3 TP 仅支持单主机本地 GPU；跨 Spark 的 TP2/TP4 属于另一路 vLLM 实现，不能由此 TP1 数字外推。完整 API、工具调用、严格 JSON、长上下文与并发稳定性均需另行验证。

后续若实施：先固定模型与重排产物 hash、补丁和依赖，补齐 driver；从无 drafter 基线验证中文与业务质量，再对比 DSpark，分别采集 cold/fresh/repeat、TTFT、decode、完整请求耗时、MemAvailable 与成功率，最后单独验收 TabbyAPI。低比特量化产物不继承官方模型的质量结论。

配方声明为 [AGPL-3.0-only](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/blob/6cd44b3bd8478922396fd623a39010ba09d4de7b/LICENSE)；引擎、CUDA 和模型保留各自许可，本次未完成模型及依赖许可链核对。本登记不包含部署或本地 GPU 实测。
