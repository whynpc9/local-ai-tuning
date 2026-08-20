# Qwen3.8-27B / 2× RTX PRO 6000 / SGLang + MTP 部署方案

日期：2026-08-20（Asia/Shanghai）

状态：**可执行候选方案，尚未在目标服务器上实测，不是 measured winner**

## 1. 结论

面向 `uni-icd-agent`，首发拓扑建议是：

- 模型：官方 `Qwen/Qwen3.8-27B-FP8`，固定不可变 revision；
- 推理：SGLang 官方 Qwen3.8 专用镜像，固定 image digest；
- 双卡：**每张卡一个独立 TP1 副本**，而不是把一个 27B 模型做 TP2；
- 加速：每个副本启用模型内置 MTP，使用 SGLang 推荐的 `3/1/4` 参数和 Linear ReplaySSM；
- 路由：两个独立容器前放内部 HAProxy/Nginx，健康摘除、轮询分发，不透明重试 POST；
- 初始能力边界：每副本 8 个 running request、65,536 context、2,048 chunked prefill；
- 应用协议：OpenAI-compatible `/v1`，`response_format=json_object`、`temperature=0`、thinking off；
- 上线顺序：单卡 AR 基线 → 单卡 MTP → 双副本压力 → `uni-icd-agent` shadow → 5%/25%/100% 灰度；
- 回滚：保留相同模型、镜像和参数，只移除 MTP/ReplaySSM 参数，逐副本滚动回到 AR。

Qwen3.8-27B 官方 FP8 权重约 28.5 GB，单张 RTX PRO 6000 的 96 GB 显存足够容纳完整模型、GDN state、KV cache 和运行时空间。把两张卡拆成两个副本，能直接提高病例并发、保留单卡故障后的半容量，并避免 PCIe 上的逐 token TP 通信。TP2 只作为单病例延迟不达标时的 challenger，不是默认架构。

这份方案优先保证医疗编码任务的结构化输出和稳定性。DSpark、DFlash2、NVFP4、FP8 KV、bf16 SSM 都保留为后续独立实验 lane，不混入首发变量。

## 2. 应用侧事实与目标

`uni-icd-agent` 当前 `oper-v7` 的主路径具有以下部署约束：

- 每个病例通常串行执行 3–4 次 LLM 子调用；RAG fallback 或格式修复会增加调用；
- 每个 `oper-v7` LLM 请求固定为非流式、`temperature=0`、`max_tokens=2048`；
- 请求固定带 `response_format={"type":"json_object"}`；
- 请求固定带 `chat_template_kwargs={"enable_thinking":false}`；
- 结构化结果解析失败时会重试一次；
- 单次 HTTP 调用硬编码超时为 120 秒；
- 已有双思维导图真实 smoke 的整病例耗时为 21.668 秒，但它来自旧模型/旧端点，只能作为业务形态参考，不能当作新硬件基线。

对应代码证据：`/Users/wanghongyi/Projects/uni-icd-agent/services/icd/oper_v7_service.py:379`、`/Users/wanghongyi/Projects/uni-icd-agent/scripts/oper_v7_m2_adjudicate_matching.py:302`、`/Users/wanghongyi/Projects/uni-icd-agent/config.py:41`，以及 `/Users/wanghongyi/Projects/uni-icd-agent/output/oper_v7_request_time_real_smoke.json`。这些文件只用于本次只读核对，本方案不修改 `uni-icd-agent`。

因此优化目标不是单条 completion 的最高 tok/s，而是：

> 在严格 JSON、业务质量和 120 秒硬超时不退化的前提下，最大化满足病例级 p95 SLO 的 goodput，并降低多阶段调用的尾延迟。

建议第一轮采用以下临时 SLO；正式上线前由业务方确认或替换：

| 路径 | 临时病例级 SLO | 说明 |
|---|---:|---|
| 单思维导图 KAG | p95 ≤ 20 s | 通常 3 次 LLM 调用 |
| 多思维导图 / RAG fallback | p95 ≤ 30 s | 调用数和输入更高 |
| 所有路径 | 0 次 120 s LLM timeout | 硬门 |
| 结构化输出 | 0 次引擎 crash/abort；最终 schema 通过率不低于基线 | 硬门 |

## 3. 推荐架构

```text
uni-icd-agent replicas
          |
          | OpenAI-compatible /v1, internal network only
          v
  HAProxy / Nginx :30002
       /             \
      v               v
SGLang A :30000   SGLang B :30001
GPU 0, TP1, MTP   GPU 1, TP1, MTP
Qwen3.8 FP8       Qwen3.8 FP8
```

### 为什么不是 TP2

1. 27B FP8/BF16 都能单卡装下，不存在容量必须切分的问题。
2. `oper-v7` 是多病例并发加每病例多次串行调用；第二副本主要应该扩大病例 goodput。
3. RTX PRO 6000 官方规格没有给出可依赖的 NVLink 保证，本方案按 PCIe-only 设计，并要求现场检查拓扑。
4. TP2 会失去单卡故障后的可服务能力，还会引入跨卡同步。

只有当 TP1 已通过质量门，但病例级 C1/p95 仍不达标时，才增加 TP2 对照。TP2 必须用同一模型、镜像、请求集和热状态实测，不能按理论 FLOPS 直接切换。

### 为什么用两个独立进程，而不是单进程 `--dp-size 2`

- 单个 SGLang scheduler/进程异常不会同时杀死两个服务副本；
- 可以逐卡滚动切换 MTP/AR，实现不中断回滚；
- GPU Xid、OOM 或 structured-output 异常可以被网关单独摘除；
- 代价是两份 host page cache 和两个独立 Radix cache；对当前以不同病例为主的请求可接受。

## 4. 制品冻结

### 4.1 首发候选

| 项目 | 首发选择 | 说明 |
|---|---|---|
| Target checkpoint | `Qwen/Qwen3.8-27B-FP8@<REVISION>` | 官方产物；先于社区 NVFP4 |
| MTP | checkpoint 内置 | 不额外下载 drafter |
| Container | `lmsysorg/sglang@sha256:<DIGEST>` | 先拉 `qwen38-27b`，再记录 digest |
| Attention | FlashInfer | SM120 官方建议；启动时验证 `uniform_q_len` 路径 |
| KV dtype | `auto` | 先采用模型/运行时默认并记录实际值；FP8 KV 单独过质量门 |
| SSM dtype | `float32` | checkpoint 声明精度；bf16 作为后续 challenger |
| Context | 65,536 | 覆盖现有应用上限，避免直接开放 262K |
| Speculation | native MTP, EAGLE `3/1/4` | 官方 Qwen3.8 recipe |

不要直接使用浮动的模型 `main` 或镜像 tag 进入生产。应记录：

- 模型 revision、文件清单和 SHA-256；
- tokenizer/chat-template revision 和 SHA-256；
- 镜像 RepoDigest；
- SGLang、PyTorch、CUDA、FlashInfer 版本；
- 完整启动参数的规范化文本及 SHA-256；
- 驱动版本、GPU UUID、VBIOS、功耗上限和硬件 inventory SHA-256。

镜像冻结示例：

```bash
docker pull lmsysorg/sglang:qwen38-27b
docker image inspect lmsysorg/sglang:qwen38-27b \
  --format '{{json .RepoDigests}}'
```

解析出的 `lmsysorg/sglang@sha256:...` 才能写进部署配置。镜像 digest、模型 revision 任一变化，都重新跑完整质量门。

## 5. 上线前硬件门

目标机必须先确认是 96 GB Blackwell RTX PRO 6000 的确切 SKU；Server Edition、Workstation Edition、Max-Q 的功耗和散热不能混写。

在目标机的本仓库执行：

```bash
mkdir -p results/qwen38-rtxpro6000x2

python3 skills/local-ai-inference-tuning/scripts/collect_hardware.py \
  --require-nvidia \
  --container-image lmsysorg/sglang:qwen38-27b \
  --storage-path /srv/models \
  --output results/qwen38-rtxpro6000x2/hardware.json \
  --pretty

nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,uuid,name,memory.total,pci.bus_id,power.limit,temperature.gpu,ecc.mode.current \
  --format=csv
```

停止条件：

- 任一 GPU 不是预期 96 GB 型号或 UUID 未登记；
- 驱动/CUDA 与 pinned image 不兼容；
- 目标模型盘剩余空间不足以同时保留当前与回滚制品；
- GPU 有未解释的 ECC/Xid、降频、温度或功耗限制；
- FlashInfer MTP 启动出现 `uniform_q_len` 参数/arity 错误；
- 服务日志中的 `max_running_requests` 被 GDN state pool 静默压到 8 以下。

双副本 TP1 不依赖 P2P；`nvidia-smi topo -m` 仍应保存，供 TP2 challenger 评估。

## 6. SGLang 参数基线

### 6.1 每个副本的推荐初值

| 参数 | 初值 | 理由 |
|---|---:|---|
| `--tp-size` | 1 | 每卡完整副本 |
| `--context-length` | 65536 | 覆盖应用，限制失控长上下文 |
| `--max-running-requests` | 8 | 每副本初始 admission ceiling |
| `--max-total-tokens` | 131072 | 控制活跃 KV 总量；实测后调整 |
| `--max-mamba-cache-size` | 32 | 8 requests × `extra_buffer_lazy` 的 4 state slots |
| `--mem-fraction-static` | 0.88 | 给 CUDA graph、workspace 和碎片留余量 |
| `--chunked-prefill-size` | 2048 | 降低长 prefill 对 decode 的阻塞 |
| `--mamba-ssm-dtype` | float32 | 首发质量优先 |
| `--mamba-radix-cache-strategy` | extra_buffer_lazy | 每请求 4 个 state slots |
| MTP | EAGLE 3 / top-k 1 / draft 4 | Qwen3.8 官方推荐 |
| Linear ReplaySSM | 开 | MTP 中间状态用固定 ring，避免每请求扩大 state pool |

`max-running-requests=8` 是引擎上限，不等于一开始就允许每卡 8 个在线病例。灰度初始建议每卡最多 4 个 active case，观察病例级 p95 后再提升到 6/8。

### 6.2 容器启动模板

以下命令中的 digest、revision 和路径必须在目标机替换并写入发布记录。两个容器的内部端口相同，仅 host loopback 端口不同。

GPU 0：

```bash
docker run -d \
  --name sglang-qwen38-gpu0 \
  --restart unless-stopped \
  --gpus '"device=0"' \
  --ipc=host \
  --ulimit memlock=-1 \
  -p 127.0.0.1:30000:30000 \
  -v /srv/models/Qwen3.8-27B-FP8:/models/Qwen3.8-27B-FP8:ro \
  lmsysorg/sglang@sha256:<IMAGE_DIGEST> \
  sglang serve \
    --model-path /models/Qwen3.8-27B-FP8 \
    --served-model-name Qwen3.8-27B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --context-length 65536 \
    --mem-fraction-static 0.88 \
    --max-running-requests 8 \
    --max-total-tokens 131072 \
    --max-mamba-cache-size 32 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size 2048 \
    --attention-backend flashinfer \
    --kv-cache-dtype auto \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-linear-replayssm-spec \
    --grammar-backend xgrammar \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-metrics \
    --enable-cache-report
```

GPU 1 使用相同命令，仅修改：

```text
--name sglang-qwen38-gpu1
--gpus '"device=1"'
-p 127.0.0.1:30001:30000
```

首次部署不要添加 `--api-key`。`oper-v7` 当前直接请求固定发送 `Authorization: Bearer EMPTY`，没有使用 `LLM_API_KEY`。因此两个 SGLang 端口必须只绑定 loopback，外部访问控制放在内部网关；如果需要 SGLang 自身的非 EMPTY key，应先修改并验证 `oper-v7` 的鉴权实现。

### 6.3 AR 回滚配置

AR 与 MTP 必须保持相同模型、镜像、context、KV/SSM dtype 和 cache 参数。AR 只移除：

```text
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
--enable-linear-replayssm-spec
```

不要在回滚时同时切换 BF16/FP8、KV dtype 或镜像，否则无法定位问题，也无法做公平 A/B。

## 7. 网关与应用配置

### 7.1 网关原则

- 仅暴露一个内部地址，例如 `http://sglang-gateway.internal:30002/v1`；
- `/health` 失败立即摘除对应副本；
- round-robin 起步，不基于响应时间做抖动路由；
- 对已经转发的 POST **不自动重试**，避免重复推理和负载放大；
- upstream read timeout 设为 130 秒，不早于 `oper-v7` 的 120 秒；
- 限制来源为 `uni-icd-agent` 网段/身份，禁止公网暴露；
- 不记录病例正文、prompt 或完整 response；只记录 request id、阶段、token、时延和状态。

HAProxy 最小逻辑示例：

```haproxy
backend sglang_qwen38
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    retries 0
    timeout connect 3s
    timeout server 130s
    server gpu0 127.0.0.1:30000 check
    server gpu1 127.0.0.1:30001 check
```

生产环境可在网关终止 TLS/mTLS 或接入现有 service mesh；不要把长期凭据写入 SGLang 命令行。

### 7.2 `uni-icd-agent` 环境变量

```dotenv
LLM_PROVIDER=sglang
LLM_BASE_URL=http://sglang-gateway.internal:30002/v1
LLM_MODEL_NAME=Qwen3.8-27B
VLLM_API_COMPAT_MODE=sglang
LLM_MAX_TOKENS=32768
LLM_TIMEOUT=180
DIAG_V3_LLM_MODEL_NAME=Qwen3.8-27B
```

说明：

- `oper-v7` 自己固定 `max_tokens=2048`，不会使用全局 32768；
- `oper-v3/v4/v5/v6` 中仍有 4096、8192、32768 等不同上限，65,536 server context 是兼容性边界，不代表应鼓励 32K 输出；
- `diag-v3` 有独立模型名设置，如果也切到 Qwen3.8，必须显式更新；
- `LLM_TIMEOUT=180` 适用于通用 SGLang provider，但不会覆盖 `oper-v7` 的 120 秒硬编码超时；
- 上线后调用 `/v1/models`，确认返回的 served model 精确为 `Qwen3.8-27B`。

## 8. 验证矩阵

### 8.1 最小候选集

| ID | 模型 | 推理 | 用途 |
|---|---|---|---|
| A0 | 官方 FP8 | AR | 同 artifact 正确性/性能基线与回滚目标 |
| A1 | 官方 FP8 | native MTP 3/1/4 | 首发候选 |
| A2 | 官方 BF16 | AR / MTP | 量化质量控制，不直接与 A1 混称同 artifact |
| A3 | 官方 FP8 | MTP + FP8 KV | 容量/吞吐 challenger，单独过质量门 |
| A4 | 官方 FP8 | MTP + bf16 SSM | state 容量 challenger，单独过质量门 |

TP2、NVFP4、DSpark、DFlash2 不进入这个最小矩阵。只有 A1 上线稳定后，才作为后续研究候选。

### 8.2 工作负载

必须从真实 `uni-icd-agent` 调用构造 token 化后的 JSONL，不能用字符数估算 token：

1. `oper-v7` 单思维导图 KAG；
2. `oper-v7` 多思维导图 KAG；
3. `oper-v7` RAG fallback；
4. 会触发格式修复重试的风险病例；
5. 结构化空值、长术式、多个候选编码和中英文混合病例；
6. 其他仍在线的 `oper-v5/v6`、`diag-v3` 代表请求；
7. ISL/OSL 合成形状：2K/512、8K/1K、32K/2K；
8. 并发：每副本 C1/C2/C4/C8，整机 C2/C8/C16。

病例数据必须脱敏，原始 prompt/result 留在受控结果目录，不提交 Git。

### 8.3 正确性硬门

以下任一失败都禁止进入性能排名：

- `oper-v7` 既有单元/回归测试全部通过；
- 业务风险集的 primary/adjunct code、guardrail、warning 和 fallback 决策不低于当前已接受基线；
- strict JSON 语法通过率 100%，业务 schema 最终通过率不低于 AR；
- A0 与 A1 的确定性探针精确一致；真实病例的规范化结构化结果无无法解释差异；
- thinking-off 请求不泄漏 reasoning，不返回 Markdown fenced JSON；
- tool-call/parser 探针通过，尽管 `oper-v7` 主路径当前不依赖 tool calling；
- 30K needle、EOS/重复、取消恢复和单副本故障摘除通过；
- 每个副本至少 1,000 次 `response_format=json_object` 混合并发请求，0 crash、0 scheduler abort、0 GPU worker 退出。

最后一项是关键门。SGLang 曾出现 EAGLE/NEXTN 与 `response_format` 组合导致 RTX PRO 6000 worker 崩溃的问题；不能因为 issue 已关闭就假设当前 digest 一定修复，必须在精确镜像上复现验证。

### 8.4 性能与稳定性门

- 用同一 GPU 做 A0/A1 的 ABBA；每个候选至少 4 个独立 trial；
- 每个 trial 使用独特早期前缀，区分冷 prefix 与单独的 warm-prefix lane；
- 报告病例级 E2E、子调用 TTFT/TPOT、queue time、completion tok/s，而不是只报 decode 峰值；
- A1 必须在相同临时 SLO 下将整机 goodput 提高至少 20%，或将目标路径 p95 降低至少 15%；
- 结果 CV ≤ 5%，AR/MTP 观测区间不重叠，且无 correctness/finish-reason 失败；
- 特别按 OSL 128/512/1024/2048 观察 structured-output p95 和 scheduler CPU，防止 grammar rollback 随输出长度出现非线性恶化；
- 双副本 C16 压力下，无 OOM、Xid、ECC 增长、服务重启、120 秒超时；
- 8 小时混合 soak：0 crash/abort，GPU 温度/功耗/时钟无持续异常，p99 无随时间恶化。

使用仓库工具生成初始实验计划：

```bash
python3 skills/local-ai-inference-tuning/scripts/plan_experiments.py \
  --model Qwen/Qwen3.8-27B \
  --hardware-profile rtx-pro-6000-2 \
  --objective goodput \
  --hardware-inventory results/qwen38-rtxpro6000x2/hardware.json \
  --output results/qwen38-rtxpro6000x2/plan.json
```

按 `templates/comparison-contract.json`、`templates/quality-gate.json` 和 `templates/run-metadata.json` 固定比较合同。finalist 至少完成一个完整 ABBA block，再用 `compare_runs.py` 判定；单次最好成绩不得写成 winner。

## 9. 观测与告警

### 引擎层

- 请求数、running/queued、拒绝/取消、TTFT、TPOT、E2E；
- prompt/completion/cached token；
- MTP draft/accepted token、acceptance length/rate、fallback；
- Radix cache hit、KV 使用、GDN state slots、`max_running_requests` 实际值；
- grammar 编译/执行耗时、structured-output abort；
- GPU 显存、SM/显存利用率、功耗、温度、时钟、P-state、ECC/Xid。

### 应用层

- 病例总耗时与每个 `kag_trace` 子阶段耗时；
- KAG/RAG path、思维导图数量、LLM 调用数；
- schema retry、fallback、warning、timeout；
- 最终 HTTP 状态、业务 `metadata.error`、空结果率；
- 不记录可还原患者身份的原始输入/输出。

遥测示例：

```bash
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,memory.total,pstate \
  --format=csv \
  --loop-ms=200
```

## 10. 发布与回滚

### 阶段门

1. **H0 硬件门**：inventory、拓扑、驱动、磁盘、温度/ECC 正常。
2. **R0 制品门**：模型 revision、文件 hash、image digest、server config hash 完整。
3. **R1 单卡 AR**：协议、JSON、质量、长上下文基线通过。
4. **R2 单卡 MTP**：与 AR 同 artifact，通过 1,000 次 structured-output crash gate 和 ABBA。
5. **R3 双副本**：C2/C8/C16、故障摘除、滚动重启和 8 小时 soak。
6. **R4 Shadow**：复制真实脱敏流量，不影响线上返回，对比病例结果与 p95。
7. **R5 Canary**：5% → 25% → 100%，每阶段至少覆盖一个业务高峰窗口。

任何阶段失败都停止，不绕过质量门去调更激进的 quant/cache 参数。

### 回滚条件

- 任一 engine crash、scheduler abort、GPU Xid/OOM；
- strict JSON/schema 失败率超过 AR 基线；
- primary code 或 guardrail 出现已确认回归；
- 病例 p95 连续 10 分钟超过 SLO，或出现任何 120 秒 LLM timeout；
- MTP acceptance 异常下降且 p95/功耗没有收益；
- 单副本摘除后剩余副本 queue 无法承载保护流量。

### 回滚动作

1. 网关先把新请求只发往健康副本；
2. 将另一副本用同一镜像/模型的 AR 配置重启并通过 smoke；
3. 将流量切到 AR 副本；
4. 再把第二副本滚动回 AR；
5. 保存失败窗口日志、metrics、GPU telemetry 和精确 request ids；
6. 不删除失败制品和原始结果，直到完成复盘。

## 11. 已知风险与采用边界

1. **MTP + structured output**：历史上有精确组合 crash；当前 digest 必须过 1,000 请求硬门。
2. **EAGLE + XGrammar 长输出 CPU 开销**：当前仍有 rollback token-history 复制相关的公开问题；`oper-v7` 的 2,048 输出上限降低但没有消除风险。
3. **PD disaggregation**：首发明确禁用。EAGLE/grammar/PD 曾有 double-accept abort，且本机双卡没有容量需要拆分 prefill/decode。
4. **鉴权**：`oper-v7` 固定 `Bearer EMPTY`；在代码调整前，只能依赖 loopback、内部网关、网段/身份控制，不能把 SGLang 端口暴露到公网。
5. **兼容面**：方案以 `oper-v7` 为首要 workload；其他旧 workflow 的 32K 输出和 thinking/tool path 必须单独进入回归集，不能从 `oper-v7` 结果外推。
6. **未实测**：本文没有目标服务器访问权，也没有在两张实卡上运行；所有吞吐、容量和 SLO 结论在通过实测前均为候选。

## 12. 后续优化顺序

A1 稳定上线后，按一次只改变一个变量的顺序扩展：

1. 每副本 active case 4 → 6 → 8；
2. FP8 KV；
3. bf16 SSM；
4. `mem-fraction-static` 与 `max-total-tokens`；
5. TP2 延迟 challenger；
6. NVFP4；
7. DFlash2 / DSpark 独立 lane。

DFlash2 或 DSpark 只有在同一 target artifact、相同 prompt/sampling/quality gate 下，病例级 SLO goodput 明确超过 native MTP，才有资格替换首发方案。公开 tok/s 不能直接外推到 `uni-icd-agent` 的严格 JSON、多阶段调用负载。

## 参考

- [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
- [SGLang structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs)
- [NVIDIA RTX PRO 6000 family](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/)
- [SGLang #24283: EAGLE/NEXTN with response_format crash](https://github.com/sgl-project/sglang/issues/24283)
- [SGLang #31711: EAGLE + XGrammar rollback CPU overhead](https://github.com/sgl-project/sglang/issues/31711)
- [SGLang #29110: PD disaggregation + EAGLE + grammar double accept](https://github.com/sgl-project/sglang/issues/29110)
