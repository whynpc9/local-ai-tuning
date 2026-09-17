# Qwen3.8-Flash-Next / 单 DGX Spark vLLM recipe 调研归档

> 归档截点：2026-09-04 21:59 +08:00。本文固定核对 [`MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/tree/1fbe463f20e6f279beb45d2989761c6099697051) 的仓库、启动/停止脚本、四组 patch generator、PLE packed-table builder、内存看门狗，以及目标模型镜像和官方基模元数据。本文没有在 DGX Spark 上下载约 99 GB 权重、启动容器或复跑 GPU benchmark；吞吐、质量、长上下文和主机稳定性数字均是作者报告，不是本仓库实测。

## 结论

这个仓库是目前值得保留的**单 DGX Spark / vLLM / 多模态实验候选**：它不再尝试让约 99 GB checkpoint 全部常驻统一内存，而是把约 26.8 GiB 的 NVFP4 PLE 表预打包为 mmap 文件，TP=1 下只把命中行经 CPU offload worker 送给 GPU。仓库还补了 GB10 的 PLE 同步、混合 NVFP4/FP8 PLE、MXFP8 shape fallback、FP8 QSA KV、动态内存预算、低内存 watchdog 和 graceful stop。

相对既有的单机 llama.cpp lane，它的产品形态更接近 OpenAI-compatible 服务，保留 vision、MTP、工具 parser 和多请求调度；但它依赖一个可变 vLLM image tag 和四组运行时源码 patch，模型、镜像、patch 输入与派生 PLE 表没有形成可验证的 revision/hash 闭环。公开 benchmark 原始请求与结果也不在仓库中。

因此本次登记为：

- **B 级单机部署与容量机制证据**：源码机制、内存保护和失败边界写得较完整；
- **C 级性能与质量证据**：作者提供了较丰富的数字和少量 A/B，但没有 raw artifact、独立复现或稳定性分布；
- **D 级生产证据**：默认无 API 鉴权，版本未锁，未见 strict-JSON、1,000 请求、8 小时 soak、重启/故障注入或网关验收。

保留为 `qwen38-flash-next / 1x-dgx-spark / vllm-patched / nvfp4-target / ple-mmap / mtp3 / fp8-kv / c1-c4-experimental` challenger；**不替换 `oper-v7`，也不把 1M context、FP8 KV 质量或 36.9 tok/s C1 当成已复现事实。**

## 调研方法与证据分级

- **源码已核对**：固定 commit 的脚本、配置、patch generator、模型配置和许可证中直接可见。
- **作者报告**：README 声称在一台 DGX Spark 上于 2026-09-04 测得；本地未复跑。
- **本地静态验证**：`bash -n` 通过三个 shell 文件；`python3 -m py_compile files/*.py` 通过五个 Python 文件。
- **推断/待验证**：根据源码与已有 `oper-v7` 契约判断的风险和采用边界，必须用目标机器实测关闭。

动态的 star、issue、PR、tag 和上游状态只代表归档截点。

## 固定快照与来源链

| 对象 | 固定 revision | 截点信息 | 说明 |
|---|---|---|---|
| deployment recipe | [`1fbe463`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/commit/1fbe463f20e6f279beb45d2989761c6099697051) | 2026-09-04 创建；7 commits；5 stars；0 issue、0 PR、0 release；AGPL-3.0 | 本文主要审计对象 |
| Mia checkpoint mirror | [`925d7be`](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4/tree/925d7be6c14c6c9442ef83e8f05b5a3c39304f69) | 模型卡明确写明是 mirror，不是 Mia 的量化 | launcher 默认模型 ID，但未锁 revision |
| quant source-of-truth | [`b8d739e`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/b8d739ed50127495b396d68082592eb1bd39e4a0) | 模型卡仅写 WIP；artifact/config 可见 | Mia 模型卡指向的原始量化仓库 |
| official base model | [`de4b8e4`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) | `qwen-community-1.0`；262,144 native context；多模态 | 权重许可与结构基线 |

代码仓库使用 AGPL-3.0-or-later。Mia mirror 的模型卡 metadata 写 `apache-2.0`，但它同时声明权重复制自 `local-inference-lab`，而官方基模使用 `qwen-community-1.0`。在分发或对第三方提供服务前，不能把 mirror metadata 当成已经消除基模许可义务；应单独完成权重来源与许可链审查。

## 它如何在单台 Spark 上运行

固定 commit 的 [`start.sh`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/start.sh) 按以下路径工作：

1. 只从本地 Hugging Face cache 解析模型，不负责下载；
2. 读取 `/proc/meminfo` 和 checkpoint 大小，扣除约 26.82 GiB PLE 后推导 GPU memory utilization、KV 预算和容器内存上限；
3. 拒绝活动中的 ComfyUI/GPU co-tenant，并对 port 8888 的 `comfy-h3.service` 做特殊保护；
4. 从 `vllm/vllm-openai:qwen38-flash-next` image 提取原始 Python 文件，再由仓库中的 generator 生成 patched 文件；
5. 首次运行把 128 个 PLE shard 流式合并成约 26.8 GiB、每行 90 bytes 的 `packed_u8` mmap 表；
6. 以 `--network host --ipc host` 启动 TP1 vLLM，用 `/health` 判断 ready；
7. 后台 watchdog 每秒检查 `MemAvailable`，低于默认 6 GiB 时 `docker kill`；停止时先停 watchdog，再给 vLLM 最多 30 秒 graceful shutdown，并只报告、不自动删除 `/dev/shm` 残留。

四组运行时 patch 的职责在源码中可核对：

- [`patch_ple_layer.py`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/files/patch_ple_layer.py)：支持 NVFP4 codes + FP8 scales 的 PLE 行布局及 offload buffer；
- [`patch_ple_offload.py`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/files/patch_ple_offload.py)：把 GB10 不可用的 CUDA stream-memory semaphore 改成 host shared-memory sequence handshake，并 mmap packed table、设置 `MADV_RANDOM`；
- [`patch_modelopt_mxfp8.py`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/files/patch_modelopt_mxfp8.py)：FlashInfer 不接受的 MXFP8 shape 回退到 BF16 emulation；
- [`patch_qsa_fp8_kv.py`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/files/patch_qsa_fp8_kv.py)：在 QSA Triton kernel 中传递 scale 并按 tile dequantize FP8 KV。

上游 vLLM [`#53960`](https://github.com/vllm-project/vllm/issues/53960) 记录了同一 GB10/TP1 PLE offload 在 stock image 上启动 deadlock 的可重复症状；这个 recipe 的 host handshake 正是在绕过该类路径。它证明 patch 有明确问题背景，但 open issue 不是上游已接受修复，也不能替代在固定 image 上的回归测试。

## 实际默认配置

标准 quick start 要求把 [`.env.sample`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/.env.sample) 复制为 `.env`；没有 `.env` 时脚本直接退出。因此真正的“开箱默认”应按 sample，而不是脚本中的 fallback 解读：

| 配置 | quick-start 默认 | 边界 |
|---|---:|---|
| context | 262,144，YaRN off | 官方 native window |
| YaRN profile | 524,288 | 默认不启用；ceiling 同为 524,288 |
| MTP | 3 speculative tokens | 约增加 1.49 GiB，thinking 默认开启 |
| KV target | 22 GiB | 依赖 `MADV_RANDOM` 节省的 page cache margin |
| KV dtype | FP8 | 自定义 QSA patch；是容量/质量折中，不是 stock vLLM 能力 |
| concurrency | `MAX_NUM_SEQS=4` | 不是已证明的四并发长上下文 SLO |
| API | port 8888，`0.0.0.0`，host network | 没有 `--api-key`，默认无鉴权 |

`start.sh` 开启 `qwen3` reasoning parser、`qwen3_coder` tool parser 和 auto tool choice。thinking 可由请求里的 `chat_template_kwargs.enable_thinking=false` 关闭；这使其比纯文本/硬编码 thinking 的实验 server 更接近业务候选，但 strict JSON、工具正确性和 mixed traffic 仍需单独验证。

## 作者报告的性能与质量

下面只保存可追溯口径，不做跨仓库“最快”排名。

### FP8 KV、512K YaRN profile

作者报告同一台主机、`YARN=1`、`KV_TARGET_GIB=22`：

| 指标 | BF16 KV | FP8 KV | 作者报告差异 |
|---|---:|---:|---:|
| KV pool | 779,671–796,196 tokens | 1,431,164–1,502,014 | 约 1.8–1.9× |
| 400K prefill | 1,537 tok/s | 1,495 tok/s | −2.7% |
| 32K prefill | 1,883 tok/s | 1,769 tok/s | −6.1% |
| idle prose decode | 28.3 tok/s | 27.1 tok/s | −4% |
| 11-task reasoning suite | 11/11 | 11/11 | 无可见 gross regression |
| 32K 95%-depth needle miss | 2/14 | 5/20 | Fisher `p=0.67`，样本不足 |

这组 A/B 比单次 headline 更有用，但仍不能证明 FP8 KV 与 BF16 质量等价：suite 太小且两边全通过，needle 样本少，公开仓库没有 prompt、逐请求输出或统计脚本。README 自己也引用了另一个实现的 `6/6 -> 2/6` long-reasoning regression，因此 FP8 KV 必须在目标 workload 上 fail closed。

### sparkDash prefill/decode

作者报告在 FP8 KV、512K YaRN 下使用 [`sparkDash`](https://github.com/MiaAI-Lab/sparkDash) 测得：

- prefill 从 8K 的 1,646 tok/s 上升到 32K 的 2,073 tok/s，256K 为 1,791 tok/s；
- prose decode：C1 36.9 tok/s；C2 aggregate 57.4、每流 29.7；C4 aggregate 85.9、每流 23.4；
- MTP mean acceptance length 约 2.1/4；可预测 copy 输出约 41 tok/s，dense prose 更低；
- 400,062-token prefill 的 TTFT 260.3 s，3 个 needle 单次全通过，`MemAvailable` low-water 10.97 GiB。

这些数字缺少 repo 内 benchmark scripts、raw requests/results、重复次数、服务日志和 telemetry bundle。C1 的 36.9 与 FP8/BF16 A/B 表中的 27.1/28.3 也说明 workload/运行状态很敏感，不能取最大值做容量承诺。

### 多模态

模型 config 已核对为 `is_multimodal=true`、`language_model_only=false`、native context 262,144。作者报告一张 336×336 三色图和一个 4 秒/16 帧色视频都能按顺序识别；这证明 API 路径的基本 smoke，不是图片、视频或长视频质量验收。README 明确说明 MTP 在 multimodal 请求上退化为 text-only draft，且 512K long-video 未测。

## 源码审计发现

### 1. 版本与派生 artifact 没有锁成一个可重建单元

- image 使用可变 tag `vllm/vllm-openai:qwen38-flash-next`，没有 digest；四组 generator 的输入正是从该 image 临时提取的源码。
- `TP1_MODEL_ID` 只有 repo ID，没有 revision；脚本用 `ls snapshots | head -1` 选择本地 cache 的第一个目录，而不是明确的 commit/ref。
- PLE cache 目录只含组织名与模型名，不含 snapshot revision。builder 虽把 snapshot basename 写入 JSON，但重用判断只比较 packed file 的尺寸，loader 也只校验行数、行宽和文件大小，不校验 snapshot/hash。

因此同形状的新权重 revision 可能复用旧 packed PLE 表。进入实验前应把 model revision 纳入目录/key，并校验源 shard hash、packed table hash 与 metadata；同时固定 image digest 和 patch generator SHA。

### 2. quick start 不完整

模型 cache 缺失时，`start.sh` 提示运行 `./download.sh $MODEL_ID`，README 也提示填写 `HF_TOKEN`，但固定 commit 没有 `download.sh`。使用者必须自行以固定 revision 下载并保留 hash receipt，不能把三行 quick start 视为从 fresh host 可复现。

### 3. README 与执行默认值存在多处漂移

- 顶部正确写 sample 默认 `KV_CACHE_DTYPE=fp8`，随后又把“shipped default”写成 BF16；
- FP8 节一处写 shipped default，另一处写 “off by default”；sample 与标准 quick start 的实际结果是 **FP8 on**；
- 文档说 512K decode/prefill 尚未 benchmark，但同页已经给出 512K/sparkDash 数字；
- sample 注释说 FP8 “enables a 1M context”，而脚本默认 ceiling 会拒绝大于 524,288，README 也明确 1M 从未跑过；
- 内存预算失败消息仍写 “FP8 KV is unsupported by this model”，与当前自定义 patch 和默认值相反。

复跑时应以 `.env` + 展开的 `.last_launch.sh` + image/model/patch receipts 为准，不能只引用 README profile 名称。

### 4. 默认网络边界不合格

launcher 使用 host network、绑定 `0.0.0.0`，没有 `--api-key`；`EXTRA_VLLM_ARGS` 虽可人工补参数，但默认是 LAN 上的无鉴权 OpenAI API。engine 应保持 loopback/受控服务网，通过有真实鉴权、限流、deadline 和审计的 gateway 暴露。

`HF_TOKEN` 会被展开进 `.last_launch.sh` 并注入长期 serving container；即使文件被 gitignore，它仍是磁盘和 `docker inspect` 可见的 secret residue。该脚本只使用离线 cache，下载和 serving 应分离，serve 阶段不应携带 HF token。

### 5. 内存保护有价值，但不是硬实时保证

动态预算、cgroup cap、co-tenant guard、watchdog 和 graceful stop 都是明显改进。README 同时承认 UMA 耗尽曾导致硬 hang；1 秒 userspace poller 不能保证抓住突发分配，默认 watchdog floor 6 GiB 也低于作者建议的约 10 GiB load floor。首次实验仍需远程电源/控制面、逐级 KV 预算、冷启动与长 prefill 的独立观察。

### 6. 缺少可移植测试闭环

仓库没有 CI workflow，没有 generator fixture，也不提交从目标 image 生成的 patched diff 或 benchmark bundle。当前本地只能证明 shell/Python 语法成立；无法在没有 exact image 和 DGX Spark 的环境里证明 string-rewrite patch 仍匹配、kernel 正确或服务可启动。

## 与既有单 Spark lane 的位置关系

既有 [`0xBakeer/qwen38-flash-next-spark`](2026-08-27-qwen38-flash-next-dgx-spark-repos.md) 是 llama.cpp + Q4 GGUF + NVMe PLE/page cache + `--parallel 1` 的容量/copy-heavy lane；本仓库是 vLLM + NVFP4/MXFP8 + quantized PLE mmap + MTP + multimodal + 最多四序列的 serving lane。

两者 artifact、runtime、并发和 workload 完全不同：

- llama.cpp lane 更接近最小化、单流容量证明；
- 本仓库更接近多模态 OpenAI API 实验服务，但 patch 面和供应链耦合更大；
- 22 tok/s llama.cpp prose 与 27–37 tok/s vLLM prose不能直接当作同条件 uplift；
- 本仓库没有取代之前双 Spark SGLang lane，TP1 与 TP2 仍应作为独立 candidate。

## 进入 `oper-v7` 评估前的门

1. **冻结供应链**：repo SHA、model mirror/source/base revision、每个 LFS object hash、image digest、image 内 vLLM commit、四组 generator SHA、生成后 patch diff 与 packed PLE SHA256。
2. **修复 revision-aware cache**：PLE 派生表按 model revision 分目录，并在 attach 前验证 snapshot 与源权重 manifest。
3. **收紧暴露面**：engine loopback；gateway 鉴权、限流、请求大小、deadline/cancellation、日志脱敏；serve container 不注入 HF token。
4. **建立 control**：同一 target 比较 BF16 KV / FP8 KV、MTP0 / MTP3、native 262K / YaRN 512K；禁止同时改变多个变量。
5. **正确性**：thinking off、`temperature=0`、`max_tokens=2048`；`response_format=json_object` 和真实 schema；工具调用、multimodal、sampling、token parity、异常输入、超时与取消。
6. **性能**：真实 `oper-v7` traces；C1/C4/C8；2K/8K/32K/128K/400K；TTFT、TPOT/ITL、E2E、goodput、MTP acceptance、重复与方差，copy-heavy 单列。
7. **稳定性与恢复**：1,000 请求、8 小时 soak、冷/热启动、容器 crash、watchdog kill、`/dev/shm` 清理、PLE file 损坏/换版、磁盘抖动、OOM/硬 hang 预案和回滚演练。
8. **质量**：扩大 BF16/FP8 matched A/B，尤其是长推理、长上下文事实整合、95%-depth 多次 needle、真实业务 JSON/工具输出；不能用全通过的 11 题 suite 宣布等价。

在上述门关闭前，最合理的下一步是**先修复锁版本、PLE cache key、默认鉴权与文档漂移，再做一轮单机受控复现**，而不是直接部署或与不同 runtime 的 README headline 排名。

## 维护提示

再次使用本文前至少重查：deployment repo `main`、模型 mirror/source revisions、官方基模许可证、`vllm/vllm-openai:qwen38-flash-next` 实际 digest/vLLM commit、vLLM PLE/QSA 上游状态、FP8 KV patch、sparkDash benchmark protocol，以及 README 默认值漂移。动态数据和 open issue 状态均不可沿用为当前事实。

## 来源

- [deployment recipe，固定 commit](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/tree/1fbe463f20e6f279beb45d2989761c6099697051)
- [launcher，固定 commit](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/start.sh)
- [环境配置，固定 commit](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/.env.sample)
- [PLE packed-table builder，固定 commit](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/1fbe463f20e6f279beb45d2989761c6099697051/files/build_ple_packed_table.py)
- [Mia checkpoint mirror，固定 revision](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4/tree/925d7be6c14c6c9442ef83e8f05b5a3c39304f69)
- [原始量化仓库，固定 revision](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/b8d739ed50127495b396d68082592eb1bd39e4a0)
- [官方基模，固定 revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540)
- [vLLM TP1 PLE offload deadlock issue](https://github.com/vllm-project/vllm/issues/53960)
- [FP8 KV 参考实现](https://github.com/lancelind/qwen3.8-Flash-DGX)
- [sparkDash](https://github.com/MiaAI-Lab/sparkDash)
