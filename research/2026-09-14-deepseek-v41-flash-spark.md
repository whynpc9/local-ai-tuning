# DeepSeek-V4.1-Flash / 单 DGX Spark 项目登记

> 登记日期：2026-09-14。已只读浅克隆并核对固定提交的源码、配置、结果与限制说明；未安装依赖、下载模型、构建镜像或复跑 GPU。下文性能、内存和质量数字均为上游报告，不是本仓库实测。

## 项目与版本

| 项目 | 登记内容 |
| --- | --- |
| 仓库 | [0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark) |
| 分支 / 版本 | `main` / `VERSION=0.5.0` |
| 固定提交 | [`45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f`](https://github.com/0xBakeer/deepseek-v41-flash-spark/commit/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f) |
| 提交时间 / 说明 | `2026-09-14T09:26:51+02:00` / `Release 0.5.0` |
| 模型入口 | 下载脚本默认 `deepseek-ai/DeepSeek-V4.1-Flash`；模型 revision 尚未独立固定 |
| 硬件 / 系统 | 单 DGX Spark / GB10 / `sm_121a`，128 GB 统一内存（上游约 121 GiB 可见），Linux aarch64、DGX OS / Ubuntu 24.04、CUDA 13 驱动 |
| 推理栈 | 自研 PyTorch 引擎、Triton kernels、DSpark 草拟与验证、OpenAI-compatible HTTP 服务 |
| 登记定位 | 单 Spark 内存受限推理研究候选；默认剪枝方案需按领域单独验收 |

本项目对应 **V4.1-Flash**，与仓库既有 **V4-Flash-0731** 是不同模型，不沿用旧模型的容量、质量或多机成绩。旧研究入口见[本地推理选型报告](2026-08-18-local-inference-landscape.md)。

## 两条运行路线必须分开

| 路线 | 机制 | 评价边界 |
| --- | --- | --- |
| 原始完整专家流式路线 | 热专家以原始 FP4 常驻，未命中专家经 `O_DIRECT` 从本地 NVMe 读取；Engram 表按需取行 | 保留全部可路由专家，但严重依赖 SSD；不等于整个引擎与参考实现逐位一致 |
| 当前示例默认路线 | 保留约 39% 专家，常驻 CB3 三位行码本格式；按贡献度 `saliency` 和跨主题 `maxmin` 选择专家 | 改变专家集合、路由和数值精度，不能称为完整 FP4 模型质量等价 |

README 开头仍保留“不剪枝、不重量化”的原始介绍；实际默认值应以同提交的 `env.example`、启动器和引擎为准。DSpark 的验证针对当前 target，不能恢复剪枝前的模型质量。默认 SWA replay 也有近似行为：上游记录长于 128 token 的 prompt 可能与完整前缀计算产生不同 logits。

源码依据：[env.example](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/env.example)、[引擎](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/engine/v41_engine.py)、[路由实现](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/engine/model.py)、[LIMITATIONS.md](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/LIMITATIONS.md)。

## 当前配置与环境入口

以下是复制 `env.example` 后的主要基线，不是本地推荐最优值；`tune.sh` 可按实际空闲内存、上下文和主题重新生成配置。

| 参数 | 示例值 / 意义 |
| --- | --- |
| `MAX_SEQ` | `32768` |
| `PRUNE_KEEP` / `ARENA_GB` | `0.39` / `87`，约 6,000 个保留专家 |
| `EXPERT_FORMAT` | `cb3`，每个专家约 14.45 MB；原始 FP4 约 18.80 MB |
| `DSV41_PRUNE_SOURCE` / `DSV41_PRUNE_RANK` | `saliency` / `maxmin`；trace 必须包含对应贡献度统计 |
| 路由处置 | 引擎默认 `substitute`，在保留集合中选择替代专家；`drop` 是另一个实验选项 |
| `DSV41_DENSE_FP4` / `DSV41_HEAD_FMT` | `attn,wo_a` / `fp8`，非专家部分也有精度变更 |
| `SPEC` | `1`，启用模型自带 DSpark |
| `DEFAULT_THINKING` / `DEFAULT_EFFORT` | `off` / `75` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` |
| `MIN_FREE_GIB` | `90`，启动检查；通过该检查仍不保证长 prefill 能完成 |

上游要求本地 NVMe 至少约 600 GB 空闲；完整 checkpoint 约 510 GB、48 个 safetensors 分片，其中专家约 288.8 GB、Engram 表约 203 GB。剪枝在加载/运行时进行，当前流程仍下载完整 checkpoint，不能把内存压缩比例直接当作磁盘缩减比例。

原生入口：准备 Python 3.11/3.12 环境及 `torch==2.13.0+cu130`、配套 Triton、transformers/tokenizers/safetensors 等 → 复制 `env.example` → 下载权重 → 用 `tune.sh` 选择主题与预算 → `start.sh` 启动。`stop.sh` 管理停止。容器入口为 `run.sh setup/serve`，但本次未核实镜像发布或容器真实请求；上游测量注明来自原生运行环境。

复现注意三处源码差异：

- `scripts/download-model.sh` 支持 `MODEL_REVISION`，默认留空；已有文件的完成判断主要检查索引中的分片是否存在，不能替代 revision/hash 核验。
- `compose.yaml` 的显式 `environment` 未传入 `PRUNE_KEEP`、`EXPERT_FORMAT`、`DSV41_PRUNE_SOURCE` 等新参数，也未配置 `env_file`。因此 Compose 读取 `.env` 用于变量替换，并不意味着全部原生配置会进入容器；需核对最终容器环境与 `/health`。
- 安装依赖使用部分版本下限，尚无完整锁文件；模型、trace、主题选择与实际配置应一起固定。

来源：[安装说明](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/docs/install.md)、[下载脚本](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/scripts/download-model.sh)、[原生启动器](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/start.sh)、[Compose 配置](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/compose.yaml)。

## 上游测量及后续修正

| 日期 / 配置 | 上游结果 | 应保留的限定 |
| --- | --- | --- |
| 09-10，原始流式 FP4、73.8 GB arena、32K、thinking off | `code` 两轮各固定 512 completion tokens，decode 中位数 2.68 tok/s，TTFT 中位数 11.05 s，专家命中率 0.830，NVMe 读取 0.92 GB/token | 单机、单 workload；不是当前剪枝默认方案的成绩 |
| 09-12，旧 44% / 98 GB / CB3 | HTML 36.6、SQL 31.8、JS 28.2、Python 24.3、故事 17.1 tok/s；5,014-token prefill 约 337 tok/s | 每项单次请求；后来同配置通过加载却在首请求被内存 watchdog 杀死，不能作为稳定可交付数字 |
| 09-13，贡献度排名、40% / 89.2 GB、thinking on | 七个窄领域 profile 的严格检查共 39/61，通过数高于先前频率排名配置的 34/61 | 同时改变排名和保留比例，并非整体单变量 A/B；仍有推理重复、无答案和领域退步 |
| 09-13，40% / 89.2 GB、声明 256K | 194,797-token prefill 运行 582 s 后被杀，MemAvailable 降至 0.8 GB | 配置了 256K 或短 prompt 成功均不构成长上下文验收 |
| 09-13 后续内存记录，36% / 81 GB | 上游注明可承载填满的 256K 上下文 | 仅登记其内存运行报告，不外推长上下文质量；约 80K 以下可用 40% 的说法来自预算推算，中间区间未测 |
| 09-14，36%、saliency/maxmin、thinking on | Frontend 严格 7/10、完成 9/10；Backend 严格 3/10、完成 10/10；Data/research 严格 8/11、完成 10/11 | “完成”与“通过”分开；部分输出虽完成，但推理反复导致成本和延迟增加 |
| 09-14，同一 Frontend profile、thinking off、温度 0.6 | 单页网站任务五次中三次生成完整页面，两次仅生成样式 | 长文件生成仍非稳定交付能力；工具闭合标记修复没有消除内容退化 |

以上取自固定提交的 [RESULTS.md](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/RESULTS.md) 与 [LIMITATIONS.md](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/LIMITATIONS.md)。文档采用追加记录，早期“没有 thinking/长上下文结果”的文字已被后续记录部分更新；解读时必须带日期、配置和更正。

非英语也不能合并评价：09-14 最后更正指出，European languages 的 thinking-off 记录为 2/4，World languages 为 4/4；不能转述成所有语言均通过，更不代表中文业务质量已验证。

## API 与质量边界

服务提供 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、SSE、`reasoning_content` 和工具调用。原生及 Compose 默认只对 loopback 暴露，API 不校验 key；多请求由共享锁串行执行，不能按连续批处理服务的吞吐预期使用。图像输入被拒绝。

源码中 `response_format=json_schema` 被转入提示模板；不能据此认定具备通用严格 JSON Schema 解码。工具调用另有 DSML grammar 和闭合标记保护，默认 `DSV41_TOOL_GRAMMAR=1`，但它依赖 `xgrammar`：初始化失败会记录 warning 并回退文本解析。当前 README 原生安装命令及 Dockerfile 均未显式安装 `xgrammar`，所以“默认开关为 1”不能作为约束已生效的证据。复现需确认 grammar 初始化日志、实际工具调用及失败路径。

来源：[HTTP 实现](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/server/app.py)、[工具约束实现](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/server/tool_grammar.py)、[Dockerfile](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/Dockerfile)。

## 后续复现范围

保留为独立候选：先固定模型 revision/hash、代码提交、依赖、trace/keep-set 和配置；以完整专家路线作质量对照，再评估默认剪枝方案的实际业务 prompt。分别测短/长 prefill、长输出、中文、工具调用、严格 JSON、DSpark 开关和串行排队；同时记录 TTFT、可见答案与 reasoning tokens、峰值内存、专家命中率及 NVMe 读取量。只有请求完成且质量检查通过的结果才纳入速度比较。

仓库代码 [LICENSE](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/LICENSE) 为 MIT。上游称模型权重及参考代码也为 MIT，本次未独立核对模型仓库固定 revision 的许可与完整依赖链。

本次交付仅为源码登记与索引更新；不改变现有部署、选型默认值或 benchmark 结果。
