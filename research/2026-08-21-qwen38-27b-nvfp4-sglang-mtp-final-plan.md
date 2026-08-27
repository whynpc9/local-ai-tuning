# Qwen3.8-27B NVFP4 + SGLang MTP / 2× RTX PRO 6000 最终实施计划

日期：2026-08-21（Asia/Shanghai）

状态：**最终技术路线；目标机尚未实测，当前是可执行 candidate，不是 measured winner**

## 1. 最终决策

`uni-icd-agent` 的 `oper-v7` 首发路线固定为：

- target：`RadixArk/Qwen3.8-27B-NVFP4`，下载前固定完整 40 位 revision；
- runtime：SGLang Qwen3.8 专用镜像 `lmsysorg/sglang:qwen38-27b`，部署时固定 RepoDigest；
- topology：2× RTX PRO 6000 96 GB，每张卡运行一个独立 TP1 副本；
- speculation：使用 checkpoint 内置的一层 MTP，SGLang `NEXTN 3/1/4`，不下载外部 drafter；
- cache：模型声明的 FP8 KV、FP32 GDN/SSM state、`extra_buffer_lazy`、Linear ReplaySSM；
- API：两个 SGLang 副本只绑定 loopback，前置内部 HAProxy/Nginx；
- rollout：NVFP4 AR → NVFP4 MTP → 双副本 → shadow → 5%/25%/100%；
- immediate rollback：同一 NVFP4 artifact、同一镜像和 cache 参数，仅关闭 MTP/ReplaySSM 回到 AR。

不采用 TP2 作为默认拓扑，不在首发混入 DSpark、DFlash2、PD disaggregation、HiCache、YaRN 或 BF16 SSM。这些都只能在最终路线稳定后作为独立 challenger。

这里的 NVFP4 checkpoint 是 RadixArk 使用 NVIDIA Model Optimizer 制作的**社区量化衍生物**，不是 Qwen 官方 NVFP4 权重。它进入了 SGLang 的 Qwen3.8 cookbook，但不能据此跳过量化质量验证。其模型卡说明：MLP 和 `lm_head` 使用 group-size 16 的动态 NVFP4 W4A4，attention 权重使用 FP8，MTP 与视觉张量保留 BF16；因此也不能把它简单描述成“全模型 4 bit”。

## 2. 目标与应用合同

主目标固定为：

> 在 strict JSON、业务质量和单次 120 秒超时均不退化的前提下，最大化满足病例级 p95 SLO 的 goodput。

`oper-v7` 的当前合同决定了不能用公开的单请求 tok/s 直接选型：

- 每个病例通常串行进行 3–4 次 LLM 调用，RAG fallback 或修复重试会增加调用；
- 每次 LLM 调用非流式、`temperature=0`、`max_tokens=2048`；
- 固定 `response_format={"type":"json_object"}`；
- 固定 `chat_template_kwargs={"enable_thinking":false}`；
- 结构化结果解析失败时重试一次；
- 单次 HTTP 调用硬编码超时 120 秒；
- 当前调用会发送 `Authorization: Bearer EMPTY`。

代码证据位于：

- `/Users/wanghongyi/Projects/uni-icd-agent/services/icd/oper_v7_service.py`
- `/Users/wanghongyi/Projects/uni-icd-agent/scripts/oper_v7_m2_adjudicate_matching.py`
- `/Users/wanghongyi/Projects/uni-icd-agent/config.py`

第一轮临时 SLO 如下，业务上线前必须确认：

| 路径 | 临时病例级 SLO | 硬门 |
|---|---:|---|
| 单思维导图 KAG | p95 ≤ 20 s | 无 120 s timeout |
| 多思维导图 / RAG fallback | p95 ≤ 30 s | 无 120 s timeout |
| strict JSON | 语法 100% | 0 engine crash / abort |
| 业务结果 | 不低于已接受 FP8/现网基线 | 0 critical code/guardrail regression |

## 3. 架构

```text
uni-icd-agent replicas
          |
          | OpenAI-compatible /v1, internal only
          v
 HAProxy / Nginx :30002
       /             \
      v               v
SGLang A :30000   SGLang B :30001
GPU 0, TP1        GPU 1, TP1
NVFP4 + MTP       NVFP4 + MTP
```

选择两个 TP1 副本而不是 TP2 的原因：

1. 该 artifact 单张 96 GB 卡可容纳，不需要为了 fit 做切分；
2. `oper-v7` 更需要多病例 goodput，而不是为单一病例占用两卡；
3. 双副本允许单卡故障后保留半容量，并能逐卡从 MTP 回滚到 AR；
4. 默认按 PCIe-only 设计，不假设两卡存在可用于 TP 的 NVLink；
5. 避免逐 token 跨卡同步和单进程故障同时杀死两张卡。

只有 TP1 已通过所有门、但 C1/病例 p95 仍无法满足 SLO 时，才增加同 artifact 的 TP2 对照。

## 4. 制品与精度合同

### 4.1 必须固定的四个轴

| 轴 | 生产要求 |
|---|---|
| Model | `RadixArk/Qwen3.8-27B-NVFP4@<FULL_40_CHAR_REVISION>` |
| Tokenizer/chat template | 与 model revision 一起冻结并计算 SHA-256 |
| Runtime | `lmsysorg/sglang@sha256:<REPO_DIGEST>` |
| Server config | 完整规范化参数文本及 SHA-256 |

本文编写时，Hugging Face commit history 显示权重/config 由短提交 `52d1adc` 引入，之后还有 model-card 提交。短 SHA 和浮动 `main` 都不能直接写进生产配置；在下载时解析完整 SHA，并保存文件清单、总字节数和校验值。

模型冻结示例：

```bash
MODEL_ID=RadixArk/Qwen3.8-27B-NVFP4
MODEL_REV=<FULL_40_CHAR_REVISION>
MODEL_DIR=/srv/models/Qwen3.8-27B-NVFP4-${MODEL_REV}

huggingface-cli download "${MODEL_ID}" \
  --revision "${MODEL_REV}" \
  --local-dir "${MODEL_DIR}"

find "${MODEL_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum \
  > results/qwen38-nvfp4/model-files.sha256
du -sb "${MODEL_DIR}"
```

镜像冻结示例：

```bash
docker pull lmsysorg/sglang:qwen38-27b
docker image inspect lmsysorg/sglang:qwen38-27b \
  --format '{{json .RepoDigests}}'
```

部署配置必须使用解析出的 `lmsysorg/sglang@sha256:...`。任何 model revision、image digest、FlashInfer/XGrammar 或 server config 变化，都重新执行完整质量门。

### 4.2 NVFP4 质量边界

精度合同按两个独立比较处理：

1. **量化合同**：官方 `Qwen/Qwen3.8-27B-FP8` AR 质量参考 vs NVFP4 AR；判断 NVFP4 是否适合业务。
2. **推测解码合同**：同一 NVFP4 artifact 的 AR vs MTP；判断 MTP 是否 lossless、稳定并带来 goodput 收益。

禁止用 `NVFP4+MTP` 直接对比 `FP8+AR` 后，把总差异全部归因于 MTP。

## 5. 上线前硬件与兼容性门

在两卡目标机执行：

```bash
mkdir -p results/qwen38-nvfp4

python3 skills/local-ai-inference-tuning/scripts/collect_hardware.py \
  --require-nvidia \
  --container-image lmsysorg/sglang:qwen38-27b \
  --storage-path /srv/models \
  --output results/qwen38-nvfp4/hardware.json \
  --pretty

nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,uuid,name,memory.total,pci.bus_id,power.limit,temperature.gpu,ecc.mode.current \
  --format=csv
```

兼容性检查：

- 两张卡均为登记的 RTX PRO 6000 96 GB 精确 SKU，记录 SM、功耗上限和散热形态；
- 容器架构、driver、CUDA、PyTorch、SGLang kernel 和 FlashInfer 均匹配 SM120；
- 启动日志确认加载 `modelopt` mixed-precision/NVFP4 kernel，不得静默反量化、CPU offload 或落入未知 generic GEMM；
- attention 实际使用 FlashInfer；若出现 `uniform_q_len` 参数/arity 错误，停止该镜像；
- `NEXTN 3/1/4` 确实使用 checkpoint 内置 MTP，draft/accepted 指标非零；
- `extra_buffer_lazy + NEXTN + Linear ReplaySSM` 被精确镜像接受；若不兼容，停止并切换到本文的 cache fallback，不得偷偷改参数继续；
- 服务日志中的实际 running-request ceiling 不低于 8；
- 模型磁盘同时容纳最终 NVFP4、官方 FP8 质量参考和一个上一版本回滚制品。

## 6. 每副本显存预算与初值

### 6.1 预算

不能用 `27B × 4 bit` 作为显存结论。该 artifact 是 NVFP4/FP8/BF16 混合布局，还包含量化 metadata、vision 和 BF16 MTP。

用于首启的 source-derived 预算：

- resident weights：约 16.5 GB 的量级；以目标镜像 `Load weight end` 日志为准；
- model repository：仓库 profile 约 21.9 GB；磁盘字节不等于 resident bytes；
- FP8 KV：约 32.8 KiB/token，`131072` live tokens 约 4.1 GiB；
- FP32 GDN state：`32` slots 约 4.6–4.9 GiB，精确值随 SGLang revision 变化；
- ReplaySSM：中间 speculative state 使用独立固定 ring，不应再计入每请求 state slots，但 ring 仍占额外显存；
- 另外保留 CUDA graph、kernel workspace、vision、allocator 碎片和 driver/runtime 空间。

`--mem-fraction-static 0.80` 是保守起点，不是最终最佳值。96 GB × 0.80 为约 76.8 GB static budget，足够覆盖首发的受控 token/state pool，同时给不可见 runtime 峰值留余量。启动日志和混合压力峰值才是最终容量证据。

### 6.2 推荐初值

| 参数 | 初值 | 说明 |
|---|---:|---|
| `--tp-size` | 1 | 每卡完整副本 |
| `--context-length` | 65536 | 先覆盖应用，不直接开放原生 262K |
| `--max-running-requests` | 8 | 每副本 admission ceiling |
| `--max-total-tokens` | 131072 | 约 4.1 GiB FP8 KV 上限 |
| `--max-mamba-cache-size` | 32 | 8 requests × 4 lazy state slots |
| `--mem-fraction-static` | 0.80 | 保守留出 runtime 余量 |
| `--chunked-prefill-size` | 2048 | 降低长 prefill 对 decode 的阻塞 |
| `--mamba-ssm-dtype` | float32 | 首发质量优先 |
| `--mamba-radix-cache-strategy` | extra_buffer_lazy | lazy 模式按 4 slots/request 预算 |
| `--kv-cache-dtype` | auto | 必须从日志确认实际为模型声明的 FP8 |
| MTP | NEXTN 3 / top-k 1 / draft 4 | exact artifact model card 起点 |

引擎上限 8 不等于首发立即放行 8 个在线病例。灰度从每卡 4 个 active case 起步，再扩到 6/8。

如果精确镜像拒绝 `extra_buffer_lazy + spec`，fallback 为：

```text
--mamba-radix-cache-strategy extra_buffer
--max-mamba-cache-size 40
```

即按 8 requests × 5 slots 重算。这个 fallback 是新的 server-config tuple，必须重新跑质量、容量和性能门。

## 7. SGLang 启动模板

GPU 0：

```bash
docker run -d \
  --name sglang-qwen38-nvfp4-gpu0 \
  --restart unless-stopped \
  --gpus '"device=0"' \
  --ipc=host \
  --ulimit memlock=-1 \
  -p 127.0.0.1:30000:30000 \
  -v /srv/models/Qwen3.8-27B-NVFP4-<REVISION>:/models/Qwen3.8-27B-NVFP4:ro \
  lmsysorg/sglang@sha256:<IMAGE_DIGEST> \
  sglang serve \
    --trust-remote-code \
    --model-path /models/Qwen3.8-27B-NVFP4 \
    --served-model-name Qwen3.8-27B-NVFP4 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --context-length 65536 \
    --mem-fraction-static 0.80 \
    --max-running-requests 8 \
    --max-total-tokens 131072 \
    --max-mamba-cache-size 32 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size 2048 \
    --attention-backend flashinfer \
    --kv-cache-dtype auto \
    --speculative-algorithm NEXTN \
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

GPU 1 只修改：

```text
--name sglang-qwen38-nvfp4-gpu1
--gpus '"device=1"'
-p 127.0.0.1:30001:30000
```

这里不显式传 `--quantization`：让精确 checkpoint 的 `quant_method=modelopt` metadata 驱动加载，并从日志确认 `modelopt_fp4`/NVFP4 kernel。若 pinned image 的帮助或官方 exact-artifact recipe 明确要求 `--quantization modelopt_fp4`，把它加入配置并形成新的 config hash，不能依赖自动猜测。

### AR 回滚配置

保持 model、image、KV/SSM dtype、context 和 cache 参数不变，只移除：

```text
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
--enable-linear-replayssm-spec
```

MTP 故障的第一回滚目标是同一 NVFP4 的 AR，不是切回官方 FP8。官方 FP8 是量化质量参考和二级模型回滚目标，切换它需要单独验证、预热和发布记录。

## 8. 网关与应用配置

网关规则：

- 仅暴露内部地址，例如 `http://sglang-gateway.internal:30002/v1`；
- health 失败立即摘除单副本；
- round-robin 起步；
- 对已转发 POST 不透明重试，避免重复推理和负载放大；
- upstream read timeout 130 秒，不早于 `oper-v7` 的 120 秒；
- SGLang 端口只绑定 loopback，不得公网暴露；
- access log 不记录病例正文、prompt 或完整 response。

HAProxy 最小逻辑：

```haproxy
backend sglang_qwen38_nvfp4
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    retries 0
    timeout connect 3s
    timeout server 130s
    server gpu0 127.0.0.1:30000 check
    server gpu1 127.0.0.1:30001 check
```

`uni-icd-agent`：

```dotenv
LLM_PROVIDER=sglang
LLM_BASE_URL=http://sglang-gateway.internal:30002/v1
LLM_MODEL_NAME=Qwen3.8-27B-NVFP4
VLLM_API_COMPAT_MODE=sglang
LLM_MAX_TOKENS=32768
LLM_TIMEOUT=180
DIAG_V3_LLM_MODEL_NAME=Qwen3.8-27B-NVFP4
```

注意：`oper-v7` 自身固定 `max_tokens=2048` 和 120 秒 timeout；上述全局值不会覆盖它。当前固定 `Bearer EMPTY`，所以首发鉴权必须依赖 loopback、内部网关、网段/身份控制。若要启用 SGLang 自身非空 API key，应先修改并回归应用端鉴权。

## 9. 验证矩阵

### 9.1 最小候选集

| ID | Artifact | 解码 | 目的 |
|---|---|---|---|
| Q0 | 官方 Qwen3.8-27B-FP8 | AR | 量化业务质量参考 |
| N0 | RadixArk NVFP4 | AR | NVFP4 质量基线、MTP 回滚目标 |
| N1 | RadixArk NVFP4 | NEXTN 3/1/4 | 最终生产候选 |
| N2 | RadixArk NVFP4 | MTP + `extra_buffer` | lazy 不兼容时的 cache fallback |

Q0↔N0 回答“是否接受社区量化”；N0↔N1 回答“MTP 是否正确且有收益”。首发评测不增加 DSpark/DFlash2/TP2，避免变量爆炸。

### 9.2 workload

从脱敏真实流量构造 tokenizer-verified JSONL：

1. 单思维导图 KAG；
2. 多思维导图 KAG；
3. RAG fallback；
4. 格式修复重试风险病例；
5. 空值、长术式、多个候选编码、中英文混合病例；
6. `oper-v5/v6`、`diag-v3` 仍在线的代表请求；
7. 合成 2K/512、8K/1K、32K/2K；
8. 每副本 C1/C2/C4/C8，整机 C2/C8/C16。

病例原文和结果只保存在受控结果目录，不提交 Git。

### 9.3 正确性硬门

任一失败即停止该 tuple：

- `oper-v7` 现有回归测试全部通过；
- Q0 vs N0：风险集零 critical primary-code、guardrail、fallback 回归；聚合业务指标满足预先签署的 non-inferiority 阈值；
- N0 vs N1：确定性探针 token/规范化 JSON 精确一致；任何差异必须先解释，不得直接当作正常量化噪声；
- strict JSON 语法通过率 100%，业务 schema 最终通过率不低于 N0；
- thinking off 不泄漏 `<think>`，不返回 Markdown fenced JSON；
- EOS、重复、取消后立即恢复、30K needle 和单副本故障摘除全部通过；
- 每个副本至少 1,000 次混合并发 `response_format=json_object`：0 scheduler crash、0 abort、0 worker exit；
- 检查输出 128/512/1024/2048 时 grammar CPU 和 TPOT 曲线，不得出现不可接受的非线性恶化；
- 启动与运行日志无 NaN、乱码、重复 loop、silent CPU/generic-kernel fallback。

1,000 请求 crash gate 不能省略。SGLang 曾记录 NEXTN/EAGLE 与 structured output 组合让整个 scheduler 退出；公开 issue 已关闭，不代表任意新 image digest 自动安全。

### 9.4 性能与稳定性门

- 主比较 N0 vs N1，同 GPU、同请求、同热状态，至少一个完整 ABBA block；
- finalist 每候选至少 4 trial，CV ≤ 5%，观测区间不重叠；
- 报告病例 E2E、各子调用 TTFT/TPOT/queue、completion tok/s、goodput；
- N1 必须在相同 SLO 下提升整机 goodput ≥ 20%，或目标路径 p95 降低 ≥ 15%；
- 记录 MTP proposed/draft/accepted、position-wise acceptance 和 fallback；高 acceptance 不能替代正确性；
- 双副本 C16 压力下无 OOM、Xid、ECC 增长、进程重启或 120 秒 timeout；
- 8 小时混合 soak：0 crash/abort，p99 不随时间持续恶化；
- 单副本摘除时剩余副本能承载保护流量，网关不重放已发出的 POST。

实验计划：

```bash
python3 skills/local-ai-inference-tuning/scripts/plan_experiments.py \
  --model Qwen/Qwen3.8-27B \
  --hardware-profile rtx-pro-6000-2 \
  --objective goodput \
  --hardware-inventory results/qwen38-nvfp4/hardware.json \
  --output results/qwen38-nvfp4/plan.json
```

使用 `templates/comparison-contract.json`、`templates/quality-gate.json`、`templates/run-metadata.json` 固定合同，并用 `compare_runs.py` 比较。单次最好成绩不能写成 winner。

## 10. 观测与告警

引擎层必须采集：

- running/queued/rejected/cancelled；
- TTFT、TPOT、E2E、prompt/completion/cached tokens；
- MTP draft/accepted、acceptance length/rate、fallback；
- Radix hit、FP8 KV 使用、GDN state slots、ReplaySSM ring、实际 request ceiling；
- XGrammar compile/execute/scheduler CPU、structured-output abort；
- GPU memory、SM/memory utilization、power、temperature、clocks、P-state、ECC/Xid。

应用层必须采集：

- 病例总耗时和各 `kag_trace` 子阶段耗时；
- KAG/RAG path、思维导图数、LLM 调用数；
- schema retry、fallback、warning、timeout；
- 最终 HTTP 状态、业务 error、空结果率；
- 不采集可还原患者身份的原始文本。

## 11. 发布阶段与回滚

### 阶段门

1. **H0 硬件**：inventory、SM120、driver、拓扑、磁盘、ECC/温度正常。
2. **R0 制品**：完整 model revision、文件 hash、image digest、config hash 完整。
3. **Q0 量化**：官方 FP8 AR 与 NVFP4 AR 的业务 non-inferiority 通过。
4. **R1 单卡 AR**：N0 协议、JSON、长上下文、稳定性通过。
5. **R2 单卡 MTP**：N1 通过 1,000 structured-output crash gate 和 ABBA 收益门。
6. **R3 双副本**：C2/C8/C16、故障摘除、滚动重启、8 小时 soak。
7. **R4 Shadow**：复制脱敏真实流量，不影响线上返回，比较结果与病例 p95。
8. **R5 Canary**：5% → 25% → 100%，每阶段至少覆盖一个业务高峰窗口。

### 自动停止/回滚条件

- 任一 engine crash、scheduler abort、GPU Xid/OOM；
- strict JSON/schema 低于 N0；
- 已确认的 primary code、guardrail 或 fallback 回归；
- 病例 p95 连续 10 分钟超 SLO，或任一 120 秒 LLM timeout；
- MTP acceptance 下降且 p95/goodput 无收益；
- 单副本摘除后剩余 queue 超过保护阈值；
- NVFP4 kernel、FP8 KV 或 MTP 路径没有被实际使用。

### 回滚动作

1. 网关摘除待回滚副本；
2. 使用同 NVFP4 artifact/image 的 AR 配置重启该副本；
3. smoke 通过后把流量切到 AR；
4. 第二副本滚动回 AR；
5. 保留失败窗口日志、metrics、GPU telemetry、request id 和精确制品；
6. 若确认是 NVFP4 质量而非 MTP 问题，再进入已预热、已验证的官方 FP8 二级回滚流程。

## 12. 明确不做与后续顺序

首发不做：

- TP2；
- 外部 DSpark / DFlash2；
- PD disaggregation；
- HiCache；
- 262K/YaRN；
- BF16 SSM、其他 KV dtype；
- 单进程 `--dp-size 2`；
- 任何没有 revision/digest 的社区镜像或权重。

N1 稳定后，一次只改变一个变量，顺序为：

1. 每副本 active case 4 → 6 → 8；
2. `mem-fraction-static`、`max-total-tokens` 和 chunked prefill；
3. FP32 → BF16 SSM；
4. TP2 单病例延迟 challenger；
5. 外部 DSpark；
6. DFlash2；
7. 65K → 128K/262K context。

任何后续方案都必须回到同一 strict-JSON/业务质量/goodput 合同；公开 tok/s 不得替代目标机结果。

## 13. 已知风险

1. **社区量化质量**：model card 的 GSM8K/Terminal-Bench 结果不是 ICD 手术编码证据，必须通过本地风险集。
2. **NEXTN + structured output**：存在过 scheduler crash 记录，精确 digest 必须过 1,000 请求硬门。
3. **XGrammar 长输出 CPU**：公开 issue 指出 speculative tree rollback 会复制累计 token history；`oper-v7` 的 2,048 上限只降低风险，不消除风险。
4. **cache revision drift**：GDN slot 常数和 `extra_buffer_lazy` 兼容性在演进，必须以 pinned image 日志重算。
5. **鉴权**：应用固定 `Bearer EMPTY`；代码修改前只能依靠 loopback 和内部网关。
6. **未实测**：本文没有目标服务器数据，所有显存余量、吞吐和收益阈值都是实施起点。

## 参考

- [RadixArk Qwen3.8-27B-NVFP4 model card](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- [RadixArk Qwen3.8-27B-NVFP4 config](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4/blob/main/config.json)
- [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
- [SGLang structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs)
- [NVIDIA RTX PRO 6000 family](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/)
- [SGLang #24283: NEXTN/EAGLE with response_format crash](https://github.com/sgl-project/sglang/issues/24283)
- [SGLang #31711: speculative decoding + XGrammar rollback CPU overhead](https://github.com/sgl-project/sglang/issues/31711)
- [SGLang #29110: PD + speculative decoding + grammar double accept](https://github.com/sgl-project/sglang/issues/29110)
