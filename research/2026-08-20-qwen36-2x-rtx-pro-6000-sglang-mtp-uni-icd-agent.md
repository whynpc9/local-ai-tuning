# Qwen3.6-27B / 2× RTX PRO 6000 / SGLang + MTP 部署方案

日期：2026-08-20（Asia/Shanghai）

状态：**可执行候选方案，尚未在目标双卡服务器实测，不是 measured winner**

## 1. 决策

面向当前 `uni-icd-agent`，建议把 Qwen3.6-27B 作为第一生产候选，Qwen3.8-27B 作为后续 shadow challenger：

- 模型：官方 `Qwen/Qwen3.6-27B-FP8`，固定不可变 revision；
- 推理：固定到包含 SGLang Mamba/MTP 修复提交 `1c4892d` 的镜像 revision 和 image digest；
- 双卡：每张 RTX PRO 6000 运行一个独立 TP1 副本，不使用 TP2；
- MTP：使用模型内置 MTP，EAGLE `3/1/4`；
- GDN state：首发使用 `float32` 和保守 `extra_buffer`，不使用 `extra_buffer_lazy`；
- ReplaySSM：首发关闭，基础 MTP 稳定后再作为单独 challenger；
- 路由：两个独立容器前置内部 HAProxy/Nginx，健康摘除、轮询分发，不透明重试 POST；
- 应用协议：OpenAI-compatible `/v1`、非流式、严格 JSON、`temperature=0`、thinking off；
- 回滚：保持模型、镜像、KV/SSM dtype 不变，只移除 speculative 参数，逐副本滚动回 AR。

这不是“Qwen3.6 一定优于 Qwen3.8”的判断。选择 Qwen3.6 的原因是 `uni-icd-agent` 已经验证过它的 SGLang 非流式 JSON、thinking 控制和 tool parser，业务迁移风险较低；模型能力、FP8 量化质量和 MTP 性能仍必须在目标硬件上重新验收。

## 2. 已确认事实与未确认边界

### 2.1 项目内已确认

`uni-icd-agent` 的 Qwen3.6/SGLang 验证记录已经确认：

- `LLM_PROVIDER=sglang` 能显式选择 SGLang provider；
- 非流式普通 chat 可用；
- `response_format={"type":"json_object"}` + prompt schema 可产生可解析 JSON；
- `enable_thinking=false` 时没有 reasoning 泄漏；
- `--tool-call-parser qwen3_coder` 可产生可执行 tool calls；
- 相比当时的 vLLM 0.20.1，SGLang tool calling 更稳定；
- 工具或检索失败必须由代码层 fail closed，不能允许模型自行补写编码结果。

项目证据：

- `/Users/wanghongyi/Projects/uni-icd-agent/docs/oper_v7_qwen36_clean_slate_validation.md:174`
- `/Users/wanghongyi/Projects/uni-icd-agent/services/icd/oper_v7_service.py:379`
- `/Users/wanghongyi/Projects/uni-icd-agent/services/llm_providers/sglang_provider.py:137`
- `/Users/wanghongyi/Projects/uni-icd-agent/scripts/oper_v7_m2_adjudicate_matching.py:302`
- `/Users/wanghongyi/Projects/uni-icd-agent/config.py:41`

### 2.2 不能从现有验证推导

现有记录不能证明以下精确组合已经生产可用：

- 官方 FP8 checkpoint 与此前端点是字节相同 artifact；
- 新的 pinned SGLang image 与此前运行时相同；
- MTP 与 AR 在 `oper-v7` 真实病例上质量等价；
- 两张 RTX PRO 6000 的并发、p95、功耗和 8 小时稳定性；
- 旧 workflow 的 thinking-on、工具调用和最高 32K 输出路径；
- Qwen3.6 相比 Qwen3.8 的业务准确率或病例级 goodput。

因此本方案以现有验证作为协议起点，不把它当成 FP8+MTP 的最终验收。

## 3. 目标与工作负载合同

主目标选择：

> 在严格 JSON、业务正确性和 120 秒硬超时不退化的前提下，最大化满足病例级 p95 SLO 的 goodput。

`oper-v7` 的部署约束：

- 每个病例通常串行执行 3–4 次 LLM 子调用；RAG fallback 或结构修复会增加调用；
- 每次请求固定非流式、`temperature=0`、`max_tokens=2048`；
- 每次请求固定 `response_format=json_object`；
- 每次请求固定 `enable_thinking=false`；
- schema/normalization 失败最多重试一次；
- `oper-v7` 单次 LLM HTTP 超时硬编码为 120 秒。

建议第一轮采用以下临时 SLO；正式发布前由业务方确认或替换：

| 路径 | 临时病例级 SLO | 说明 |
|---|---:|---|
| 单思维导图 KAG | p95 ≤ 20 s | 通常 3 次 LLM 调用 |
| 多思维导图 / RAG fallback | p95 ≤ 30 s | 调用数与输入更高 |
| 所有路径 | 0 次 120 s LLM timeout | 硬门 |
| structured output | 0 engine crash/abort；最终 schema 通过率不低于 AR | 硬门 |

性能报告必须区分病例级 E2E、子调用 TTFT/TPOT、queue time 和 visible answer。不能用单次 decode 峰值代替在线 goodput。

## 4. 推荐架构

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
Qwen3.6 FP8       Qwen3.6 FP8
```

### 为什么是两个独立 TP1 副本

1. 官方 FP8 仓库约 30.9 GB，单张 96 GB RTX PRO 6000 有足够容量容纳完整模型、GDN state、KV 和运行时空间。
2. `oper-v7` 的主要服务问题是多病例、多阶段调用的 goodput，而不是模型装不下。
3. 两副本能保留单卡故障后的半容量，并允许 MTP→AR 逐卡滚动回滚。
4. TP2 会引入 PCIe 同步、丢失副本隔离，且不保证密集 27B 的单病例延迟更好。

TP2 只在 TP1 质量通过、但病例级 C1/p95 仍不达标时进入 challenger 矩阵。必须按同一 artifact、镜像、请求集和热状态实测。

### 为什么不使用单进程 `--dp-size 2`

- 一个 scheduler/engine crash 不应同时终止两个副本；
- 可以独立摘除 GPU Xid/OOM/structured-output 异常；
- 可以逐副本滚动切换 AR/MTP；
- 代价是两个独立 Radix cache，但对以不同病例为主的流量可接受。

## 5. 制品与供应链冻结

### 5.1 首发候选

| 项目 | 首发选择 | 边界 |
|---|---|---|
| Target | `Qwen/Qwen3.6-27B-FP8@<REVISION>` | 官方静态 FP8 |
| MTP | checkpoint 内置 | 不增加外部 drafter |
| SGLang | `<REVISION_CONTAINING_1c4892d>` | 必须记录 framework commit |
| Container | `lmsysorg/sglang@sha256:<DIGEST>` | 不使用浮动 tag 发布 |
| Context | 65,536 | 不直接开放原生 262K |
| KV dtype | `auto` | 记录运行时实际值；FP8 KV 另过质量门 |
| SSM dtype | `float32` | 首发质量优先 |
| Cache strategy | `extra_buffer` | 不使用已出现公开问题的 lazy lane |
| ReplaySSM | off | 后续独立实验 |

必须使用官方预量化 FP8 checkpoint。不要加载 BF16 再加 `--quantization fp8` 做在线量化；公开问题显示该路径可能错误量化官方静态 FP8 明确排除的 GDN gate projections。

必须记录：

- 模型与 tokenizer revision、文件清单和 SHA-256；
- `config.json`、chat template、generation config 的 SHA-256；
- image RepoDigest、SGLang commit、PyTorch/CUDA/FlashInfer 版本；
- 完整 server command/env 的规范化文本和 SHA-256；
- GPU UUID、SKU、VBIOS、驱动、功耗上限和 hardware inventory SHA-256。

镜像冻结示例：

```bash
docker pull lmsysorg/sglang:<CANDIDATE_TAG>
docker image inspect lmsysorg/sglang:<CANDIDATE_TAG> \
  --format '{{json .RepoDigests}}'
```

发布配置只接受解析后的 `lmsysorg/sglang@sha256:...`。还要从镜像内确认 SGLang commit 包含 `1c4892d`，不能仅凭 issue 状态或 tag 名称推断。

## 6. 硬件与兼容性门

目标机先确认确切 RTX PRO 6000 SKU、96 GB 显存、SM120、驱动/CUDA、功耗和散热。Server Edition、Workstation Edition、Max-Q 的结果不能混用。

```bash
mkdir -p results/qwen36-rtxpro6000x2

python3 skills/local-ai-inference-tuning/scripts/collect_hardware.py \
  --require-nvidia \
  --container-image lmsysorg/sglang@sha256:<IMAGE_DIGEST> \
  --storage-path /srv/models \
  --output results/qwen36-rtxpro6000x2/hardware.json \
  --pretty

nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,uuid,name,memory.total,pci.bus_id,power.limit,temperature.gpu,ecc.mode.current \
  --format=csv
```

停止条件：

- 任一 GPU 不是登记的 96 GB SKU；
- pinned image 不包含目标 SGLang revision/MTP 修复；
- 驱动/CUDA/FlashInfer 与 SM120 不兼容或发生 generic/CPU fallback；
- 模型盘无法同时保留当前、候选和回滚制品；
- GPU 存在未解释的 Xid、ECC、持续降频或温度异常；
- MTP warmup 出现 `mamba_next_track_idx is None`、`NoneType`、scheduler exit；
- 服务日志的实际 `max_running_requests` 被 state/KV pool 压到 8 以下。

双 TP1 不依赖 P2P，但仍保存拓扑，为后续 TP2 challenger 提供证据。

## 7. SGLang 参数基线

### 7.1 每副本初值

| 参数 | 初值 | 理由 |
|---|---:|---|
| `--tp-size` | 1 | 每卡完整副本 |
| `--context-length` | 65536 | 应用能力边界 |
| `--max-running-requests` | 8 | engine admission ceiling |
| `--max-total-tokens` | 131072 | 限制活跃 KV 总量 |
| `--max-mamba-cache-size` | 40 | 8 requests × `extra_buffer` 的 5 slots |
| `--mem-fraction-static` | 0.85 | 为 graph/workspace/碎片保留空间 |
| `--chunked-prefill-size` | 2048 | 降低长 prefill 对 decode 的阻塞 |
| `--mamba-ssm-dtype` | float32 | checkpoint 精度优先 |
| MTP | EAGLE 3 / top-k 1 / draft 4 | 官方 Qwen3.6 recipe |
| Linear ReplaySSM | off | 先减少变量 |

每副本 8 是引擎上限，不是首日病例并发。灰度初始每副本最多 4 个 active case，观察病例 p95 和 queue 后再升到 6/8。

### 7.2 容器启动模板

GPU 0：

```bash
docker run -d \
  --name sglang-qwen36-gpu0 \
  --restart unless-stopped \
  --gpus '"device=0"' \
  --ipc=host \
  --ulimit memlock=-1 \
  -p 127.0.0.1:30000:30000 \
  -v /srv/models/Qwen3.6-27B-FP8:/models/Qwen3.6-27B-FP8:ro \
  lmsysorg/sglang@sha256:<IMAGE_DIGEST> \
  sglang serve \
    --model-path /models/Qwen3.6-27B-FP8 \
    --served-model-name Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --context-length 65536 \
    --mem-fraction-static 0.85 \
    --max-running-requests 8 \
    --max-total-tokens 131072 \
    --max-mamba-cache-size 40 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size 2048 \
    --attention-backend flashinfer \
    --kv-cache-dtype auto \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --grammar-backend xgrammar \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-metrics \
    --enable-cache-report
```

GPU 1 只修改：

```text
--name sglang-qwen36-gpu1
--gpus '"device=1"'
-p 127.0.0.1:30001:30000
```

参数名有版本边界：较旧 image 使用 `--mamba-scheduler-strategy extra_buffer`，当前 main recipe 使用 `--mamba-radix-cache-strategy extra_buffer`。部署前必须在 pinned image 中运行 `sglang serve --help`，选择实际存在的一项，禁止同时传入或静默忽略未知参数。

不要在首发命令加入：

- `--mamba-radix-cache-strategy extra_buffer_lazy`；
- `--enable-linear-replayssm-spec`；
- `--quantization fp8`；
- `--kv-cache-dtype fp8_e4m3`；
- `--dtype float16`；
- PD disaggregation 或 TP2。

这些变量可以在基础 MTP 通过后逐项实验，不能一起修改。

### 7.3 AR 基线与回滚配置

AR 保持相同模型、镜像、context、KV/SSM dtype 和 cache strategy，只移除：

```text
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

如果 MTP 不能显著改善真实 workload，AR 就是合格生产选择；不能为了“必须使用 MTP”而接受 crash、质量回归或更差 p99。

## 8. 网关和应用配置

### 8.1 网关

- 只暴露一个内部地址，例如 `http://sglang-gateway.internal:30002/v1`；
- `/health` 失败立即摘除对应副本；
- round-robin 起步；
- 已转发 POST 不自动重试，避免重复推理和负载放大；
- upstream read timeout 130 秒，不早于 `oper-v7` 的 120 秒；
- 限制来源为 `uni-icd-agent` 网段/身份，禁止公网暴露；
- 不记录病例正文、prompt 或完整 response。

```haproxy
backend sglang_qwen36
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    retries 0
    timeout connect 3s
    timeout server 130s
    server gpu0 127.0.0.1:30000 check
    server gpu1 127.0.0.1:30001 check
```

`oper-v7` 当前固定发送 `Authorization: Bearer EMPTY`，没有读取 `LLM_API_KEY`。因此 SGLang engine 只能绑定 loopback，访问控制放在内部网关。若需要 SGLang 自身的非 EMPTY key，必须先修改并验证 `oper-v7`。

### 8.2 `uni-icd-agent`

```dotenv
LLM_PROVIDER=sglang
LLM_BASE_URL=http://sglang-gateway.internal:30002/v1
LLM_MODEL_NAME=Qwen3.6-27B
VLLM_API_COMPAT_MODE=sglang
LLM_MAX_TOKENS=32768
LLM_TIMEOUT=180
DIAG_V3_LLM_MODEL_NAME=Qwen3.6-27B
```

说明：

- `oper-v7` 固定 `max_tokens=2048` 和 120 秒 HTTP timeout；全局变量不会覆盖它们；
- `oper-v3/v4/v5/v6` 仍有不同的 thinking、tool 和 4K–32K 输出路径，必须另做回归；
- 不要依赖 `thinking_budget` 作为旧 workflow 的唯一长度保护；服务端和客户端仍要设置明确 `max_tokens`；
- 上线后调用 `/v1/models`，确认精确返回 `Qwen3.6-27B`。

## 9. 实验矩阵

### 9.1 最小候选

| ID | Artifact | 推理 | 角色 |
|---|---|---|---|
| A0 | 官方 FP8 | AR + extra_buffer | 正确性/性能基线与回滚目标 |
| A1 | 官方 FP8 | MTP 3/1/4 + extra_buffer | 首发候选 |
| A2 | 官方 BF16 | AR / MTP | FP8 量化质量控制 |
| A3 | 官方 FP8 | MTP + Linear ReplaySSM | 后续 state/性能 challenger |
| A4 | 官方 FP8 | MTP + FP8 KV | 后续容量 challenger |
| A5 | 官方 FP8 | MTP + extra_buffer_lazy | 仅在公开问题关闭且精确 digest 复测后 |

AWQ、社区 text-only NVFP4、在线 FP8 量化、TP2 和 PD 分离不进入首发矩阵。

### 9.2 工作负载

从真实、脱敏的 `uni-icd-agent` 调用生成 tokenizer 验证的 JSONL：

1. `oper-v7` 单思维导图 KAG；
2. 多思维导图 KAG；
3. RAG fallback；
4. schema retry 风险病例；
5. 空值、长术式、多个候选编码、中英文混合；
6. `oper-v5/v6` thinking/tool 代表请求；
7. `diag-v3` 代表请求；
8. ISL/OSL：2K/512、8K/1K、32K/2K；
9. 每副本 C1/C2/C4/C8，整机 C2/C8/C16。

原始病例、prompt 和输出留在受控结果目录，不提交 Git。

## 10. 正确性与稳定性硬门

以下任一失败都禁止进入性能排名：

- `oper-v7` 既有单元/回归测试全部通过；
- 业务风险集的 primary/adjunct code、guardrail、warning、fallback 不低于接受基线；
- strict JSON 语法 100% 通过，最终业务 schema 通过率不低于 AR；
- A0/A1 的确定性探针精确一致，真实病例规范化结构无无法解释差异；
- thinking-off 不泄漏 reasoning 或 Markdown fenced JSON；
- tool parser、EOS/重复、30K needle、取消恢复、单副本故障摘除通过；
- 每副本至少 1,000 次 structured-output 混合并发请求，0 engine crash、0 scheduler abort、0 worker exit；
- 复现型 burst 必须混合短请求、长上下文、JSON schema 和 prefix-cache reuse；
- 日志中不出现 `mamba_next_track_idx is None`、`TARGET_VERIFY` crash、state corruption 或 NaN；
- 两副本 C16 下无 OOM、Xid、ECC 增长和 120 秒 timeout；
- 8 小时混合 soak 无 crash/restart，p99 不随时间恶化。

Qwen3.6 MTP 的 1,000 请求门不是一般性“多跑一点”：SGLang 已出现过 FP8+MTP scheduler crash，修复合并后，0.5.14 lazy-buffer lane 又报告相似的 `mamba_next_track_idx` 问题。因此必须在精确 image digest 上执行，不能用“issue 已关闭”代替复测。

## 11. 性能比较

- 在同一 GPU 上先跑 A0，再跑 A1；
- 使用完整 ABBA，每个候选至少 4 个独立 trial；
- 每个 trial 使用独特早期前缀，warm-prefix 作为单独 lane；
- 保存 prompt/completion token、TTFT、TPOT、queue、E2E、finish reason；
- 报告 MTP draft/accepted token、acceptance length/rate 和 fallback；
- A1 必须在同一 SLO 下将 goodput 提高至少 20%，或目标路径 p95 降低至少 15%；
- CV ≤ 5%，观测区间不重叠，且无 correctness/finish-reason 失败；
- OSL 128/512/1024/2048 分桶观察 XGrammar scheduler CPU 和 p99；
- 只有质量、稳定性和收益同时通过，才把 A1 标为 measured winner。

仓库的 experiment planner 目前只覆盖 Qwen3.8/DeepSeek，不应把 Qwen3.8 的 plan JSON 改名冒充 Qwen3.6。Qwen3.6 应手工建立等价 comparison contract，并使用：

- `skills/local-ai-inference-tuning/templates/comparison-contract.json`
- `skills/local-ai-inference-tuning/templates/quality-gate.json`
- `skills/local-ai-inference-tuning/templates/run-metadata.json`
- `skills/local-ai-inference-tuning/scripts/benchmark_openai.py`
- `skills/local-ai-inference-tuning/scripts/compare_runs.py`

每次 run 必须绑定 artifact revision、image digest、framework revision、server config hash、quality gate 和 workload shape。

## 12. 观测与告警

### SGLang/GPU

- running/queued/rejected/cancelled request；
- TTFT、TPOT、E2E、prompt/completion/cached token；
- MTP draft/accepted token、acceptance rate/length；
- Radix cache hit、KV 使用、GDN state slots、实际并发上限；
- grammar 时间、scheduler CPU、structured-output abort；
- GPU 显存、利用率、功耗、温度、时钟、P-state、ECC/Xid。

### 应用

- 病例总耗时和每个 `kag_trace` 子阶段耗时；
- KAG/RAG path、思维导图数量、LLM 调用数；
- schema retry、fallback、warning、timeout；
- 最终 HTTP 状态、业务 `metadata.error`、空结果率；
- 不记录可还原患者身份的输入/输出。

```bash
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,memory.total,pstate \
  --format=csv \
  --loop-ms=200
```

## 13. 发布和回滚

### 阶段门

1. **H0 硬件门**：inventory、驱动、磁盘、温度/ECC、image architecture 通过。
2. **R0 制品门**：模型 revision/hash、image digest、SGLang commit、config hash 完整。
3. **R1 单卡 AR**：协议、JSON、业务质量、长上下文基线通过。
4. **R2 单卡 MTP**：1,000 请求 crash gate、AR parity 和 ABBA 通过。
5. **R3 双副本**：C2/C8/C16、故障摘除、滚动重启和 8 小时 soak。
6. **R4 Shadow**：复制脱敏真实流量，不影响线上返回。
7. **R5 Canary**：5% → 25% → 100%，每阶段覆盖至少一个业务高峰。
8. **R6 Qwen3.8 shadow**：同一病例集和 SLO 合同评估是否值得升级。

### 回滚条件

- 任一 engine crash、scheduler abort、worker exit、GPU Xid/OOM；
- `mamba_next_track_idx`、TARGET_VERIFY 或 state corruption；
- strict JSON/schema 失败率超过 AR；
- primary code、guardrail 或 fallback 已确认回归；
- 病例 p95 连续 10 分钟超过 SLO，或出现任何 120 秒 timeout；
- MTP 没有超过方差的收益，或 p99/功耗恶化；
- 单副本摘除后剩余 queue 无法承载保护流量。

### 回滚动作

1. 网关停止向异常副本发送新请求；
2. 以同一 image/model 的 AR 配置滚动重启异常副本；
3. smoke 通过后切入 AR 副本；
4. 再把另一副本滚动回 AR；
5. 保存失败窗口的日志、metrics、GPU telemetry 和 request ids；
6. 不删除失败制品和原始结果，直到复盘完成。

## 14. 与 Qwen3.8 的采用边界

Qwen3.6 的优势是当前应用已有协议与工作流证据；Qwen3.8 的优势是当前 SGLang 有更完整的精确模型 cookbook 和专用 image。生产选择顺序建议：

1. Qwen3.6 FP8 AR 通过；
2. Qwen3.6 FP8 MTP 通过并证明收益；
3. Qwen3.6 生产灰度稳定；
4. Qwen3.8 用同一风险集、并发、SLO 和质量门 shadow；
5. 只有 Qwen3.8 业务质量不退化且病例级 goodput 明确领先，才切换模型。

不能把不同 artifact 的 tok/s 直接比较，也不能因为模型版本更高就默认业务准确率更高。

## 15. 参考

- [Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- [SGLang Qwen3.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6)
- [SGLang Qwen3.8-27B cookbook：相同 serving geometry 的当前 GDN/cache sizing 说明](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [SGLang #31249: Qwen3.6 FP8 + MTP crash](https://github.com/sgl-project/sglang/issues/31249)
- [SGLang #27998 / commit 1c4892d: Mamba spec-v2 fix](https://github.com/sgl-project/sglang/pull/27998)
- [SGLang #34786: Qwen3.6 lazy-buffer MTP crash on 0.5.14](https://github.com/sgl-project/sglang/issues/34786)
- [SGLang #30598: on-the-fly FP8 quantization and GDN projections](https://github.com/sgl-project/sglang/issues/30598)
- [SGLang #25536: Qwen3.6 thinking_budget behavior](https://github.com/sgl-project/sglang/issues/25536)
