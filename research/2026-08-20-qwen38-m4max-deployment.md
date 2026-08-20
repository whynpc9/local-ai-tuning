# Qwen3.8-27B 在 M4 Max 128 GB 上的本机部署与调优实测

日期：2026-08-19 至 2026-08-20（Asia/Shanghai）

## 结论

这台 MacBook Pro（M4 Max 40 核 GPU、128 GB 统一内存）的当前最佳单用户交互式方案是：

- oMLX 0.6.2 原生 arm64/Metal；
- ModelScope 的 `mlx-community/Qwen3.8-27B-4bit` 目标模型；
- 同源 `mlx-community/Qwen3.8-27B-MTP-4bit` sidecar；
- VLM MTP depth 2，即 `vlm_mtp_draft_block_size=3`；
- 单并发、关闭前缀缓存、65,536-token 服务上限、safe memory guard；
- `xgrammar==0.2.3` 用于按请求执行严格 JSON schema；
- 仅绑定 `127.0.0.1`，不对局域网暴露。

2K 输入、512 输出的严格 ABBA 测试中，MTP depth 2 的单请求解码中位数为 **56.771 tok/s**，AR 为 **27.892 tok/s**，提升 **103.5%**。两者各 4 个独立 trial、每个 trial 7 个正确性校验请求，CV 分别为 0.25% 和 0.55%，观测区间不重叠。完整质量门 AR 与 MTP 均为 8/8 通过。

需要保留一个边界：32K 独特前缀压力测试可稳定完成，但在当前桌面应用共存状态下，30 分钟窗口发生 1,501 次 pageout，虽然 swap 始终为 0。后续 3 分钟稳态短负载窗口严格通过零 pageout/swap 门，说明没有持续内存泄漏；日常建议将典型输入控制在 8K–30K，并给其他桌面负载保留余量。

## 硬件与运行边界

| 项目 | 实测值 |
|---|---|
| 机器 | MacBook Pro `Mac16,5` |
| SoC | Apple M4 Max |
| CPU / GPU | 16 核 CPU / 40 核 GPU |
| 统一内存 | 128 GB |
| 架构 | 原生 arm64 |
| macOS | 26.5.2（25F84） |
| 电源 | AC，`lowpowermode=2`（High Power） |
| 睡眠 | 测试期 `sleep=0`，另用 `caffeinate -dimsu` 防止中断 |
| Metal 默认上限 | 约 107.5 GB；未修改 `iogpu.wired_limit_mb` |
| oMLX memory guard | `safe`，当时安全上限约 79.7 GB |
| 硬件清单 SHA-256 | `6039e1c09a7940d9de5c8d57fcf55672ebb728afce2e220f0bc783ae9b15507b` |

没有为了追求吞吐修改系统级 Metal wired limit，也没有关闭用户的其他应用。结论因此代表这台机器的真实桌面共存状态，而不是清空系统后的实验室峰值。

## 模型获取与供应链固定

大模型权重全部通过 ModelScope 1.22.3 下载，没有经过 Hugging Face 主站或 VPN。原计划测试的 Jundot oQ4e-MTP 产物在 ModelScope 不存在，`hf-mirror.com` 对该产物又重定向至 `huggingface.co`，因此没有让大流量走该路径，改用 ModelScope 上可完整校验的 MLX 4-bit 目标模型与 MTP sidecar。

下载命令：

```bash
/Users/wanghongyi/Library/Python/3.9/bin/modelscope download \
  --model mlx-community/Qwen3.8-27B-4bit \
  --revision master \
  --local_dir results/qwen38-m4max-20260819/models/Qwen3.8-27B-4bit

/Users/wanghongyi/Library/Python/3.9/bin/modelscope download \
  --model mlx-community/Qwen3.8-27B-MTP-4bit \
  --revision master \
  --local_dir results/qwen38-m4max-20260819/models/Qwen3.8-27B-MTP-4bit
```

ModelScope CLI 不接受该仓库的 commit hash 作为 `--revision`，因此下载前冻结文件清单及 SHA-256，下载后逐文件校验。24/24 个发布文件一致。关键摘要如下：

| 文件 | SHA-256 |
|---|---|
| target shard 1 | `6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d` |
| target shard 2 | `83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670` |
| target shard 3 | `31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a` |
| target config | `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85` |
| tokenizer | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| MTP weights | `76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6` |
| MTP config | `16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55` |

目标模型占约 15 GB，MTP sidecar 占约 267 MB。它们是 `affine` 4-bit、`group_size=64` 的社区转换产物，不应被描述为官方 BF16 权重的质量等价物。模型权重在停服后仍保留在本机，便于复现；“释放资源”指卸载统一内存并停止服务，不删除磁盘文件。

## 运行时

隔离环境位于 `results/qwen38-m4max-20260819/runtime/venv`。关键版本与固定点：

| 组件 | 版本 / 固定点 |
|---|---|
| oMLX | 0.6.2，release commit `f2d36f3d25a7e7a2401a92eecafc28b8f8968ec7` |
| oMLX wheel | SHA-256 `4be2a3ac7c3a08f2f2a828e618227d4932c04cdb0e501203e70864126a8d3c8e` |
| MLX | 0.32.0 |
| mlx-lm | 0.31.3，commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc` |
| mlx-vlm | 0.6.3，commit `78b96eb5462141447b9a6b4943ef553891da56dd` |
| mlx-embeddings | 0.1.0，commit `32981fa4e8064ed664b52071789dd18271fe4206` |
| dflash-mlx | `0.1.10+omlx.5`，commit `2eb169f461ad6b3c5b58e29025f8afb07f23cf3e` |
| xgrammar / torch | 0.2.3 / 2.13.0 |
| Transformers / HF Hub | 5.12.1 / 1.28.0 |

Python 包通过清华 PyPI 镜像安装；模型权重仍只走 ModelScope。`pip check` 最终为 `No broken requirements found`。启动日志确认 Qwen q4 MLP/linear prefill、GDN、FA256、SDPA256 等原生优化已启用，未回退到 Rosetta/CPU。

## 启动与最佳配置

启动命令：

```bash
results/qwen38-m4max-20260819/runtime/venv/bin/omlx serve \
  --model-dir /Users/wanghongyi/Projects/local-ai-tuning/results/qwen38-m4max-20260819/models \
  --base-path /Users/wanghongyi/Projects/local-ai-tuning/results/qwen38-m4max-20260819/server-state \
  --host 127.0.0.1 \
  --port 18000 \
  --max-concurrent-requests 1 \
  --memory-guard safe \
  --no-cache \
  --initial-cache-blocks 256 \
  --no-hf-cache \
  --log-level info
```

模型设置：

```json
{
  "max_context_window": 65536,
  "max_tokens": 32768,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "enable_thinking": false,
  "turboquant_kv_enabled": false,
  "specprefill_enabled": false,
  "dflash_enabled": false,
  "mtp_enabled": false,
  "vlm_mtp_enabled": true,
  "vlm_mtp_draft_model": "Qwen3.8-27B-MTP-4bit",
  "vlm_mtp_draft_block_size": 3,
  "is_default": true
}
```

OpenAI-compatible endpoint 为 `http://127.0.0.1:18000/v1`，served model 为 `Qwen3.8-27B-4bit`。服务端保留 32,768 输出上限是能力边界；普通客户端仍建议把 `max_tokens` 限制在 4,096 或业务确需的更小值，避免失控长生成。

## 调参结果

`draft_block_size` 包含一个 bonus token，因此 block size 2/3/4 分别对应 MTP draft depth 1/2/3。先用相同 5 请求 exact-copy smoke 和 7 项短质量门筛选：

| 配置 | 解码 p50 | 相对 AR | TTFT p50 | 正确性 | 汇总接受率 |
|---|---:|---:|---:|---:|---:|
| AR | 28.400 tok/s | 基线 | 1,326.8 ms | 5/5 | 不适用 |
| MTP depth 1 / block 2 | 43.286 tok/s | +52.4% | 1,370.9 ms | 5/5 | 83.5% |
| **MTP depth 2 / block 3** | **57.612 tok/s** | **+102.9%** | 1,432.6 ms | 5/5 | 76.9% |
| MTP depth 3 / block 4 | 57.137 tok/s | +101.2% | 1,389.3 ms | 5/5 | 68.3% |

depth 3 没有超过 depth 2，接受率更低且验证成本更高，因此选择 depth 2。结构化 JSON 请求会因 grammar processor 与 VLM MTP 不兼容而自动走普通 BatchGenerator，但仍由 xgrammar 严格约束并通过质量门；普通文本、推理和工具场景使用 MTP。

## 质量门

AR 与最终 MTP 配置使用同一套测试，均为 8/8 通过：

- 两次确定性算术均精确输出 `137`；
- JSON schema 精确得到 `{"name":"Li","age":42,"active":true}`；
- 工具调用正确选择 `get_weather` 并传入北京；
- thinking-on 请求正确得到 `17×19=323`，并有独立 reasoning 内容；
- EOS/重复测试正常停止，6-gram 重复率为 0；
- 30,040-token 上下文正确找回 `NEEDLE-20260819-ZEBRA`；
- temperature 0 重复结果一致。

AR 的 30K 针测试为 134.12 秒，MTP 为 139.49 秒。这个场景只输出 15 token，耗时由预填充主导，MTP 不应被期待带来收益。

## 正式性能结果

2K/512 排名使用两组完整 ABBA，共 8 个 trial、56 个测量请求。每个请求前缀唯一、缓存 token 为 0、输出必须精确匹配 512-token 参考文本。

| 配置 | trial | 解码中位数 | trial 范围 | CV | 错误/正确性失败 |
|---|---:|---:|---:|---:|---:|
| **MTP depth 2** | 4 | **56.771 tok/s** | 56.553–56.886 | 0.25% | 0/0 |
| AR | 4 | 27.892 tok/s | 27.599–27.931 | 0.55% | 0/0 |

仓库比较器在最小 4 trial、最大 CV 5%、最小增益 3%、区间不重叠和 ABBA 顺序全部通过后，判定 MTP depth 2 为 measured winner，增益 103.5%。

固定 finalist 的输入长度扩展：

| 输入 / 输出 | 请求 | 解码 p50 | TTFT p50 | E2E p50 | 正确性 |
|---|---:|---:|---:|---:|---:|
| 2K / 512 | 28 个 MTP 测量请求 | 56.771 tok/s（trial 中位数） | 约 9.1–9.4 s | 约 18.1–18.9 s | 全部通过 |
| 8K / 512 | 7 | 53.658 tok/s | 33.305 s | 42.923 s | 7/7 |
| 32K / 512 | 7 | 45.213 tok/s | 146.854 s | 158.215 s | 7/7 |

32K 时 MTP 接受率约 91.4%–93.8%，但首 token 延迟主要受约 223 prompt tok/s 的预填充限制。对于交互使用，应优先缩短/分块输入，而不是继续提高 draft depth。

## 稳定性与资源

- 140 个连续短请求：140/140 完成，零错误，解码 p50 56.050 tok/s，范围 53.368–59.872；
- 后续稳态 45 请求：45/45 完成，解码 p50 56.713 tok/s；
- 合计 185 个稳定性请求没有 correctness、finish-reason 或 token-shape 失败；
- 30 分钟混合窗口覆盖 32K 重负载和短请求：无 serious/critical 热状态，电源配置稳定，swap/swapin/swapout 为 0，但有 1,501 pageout，严格内存门失败；
- 随后的 180 秒稳态短负载窗口：65 个遥测样本，pageout/swapin/swapout 均为 0，compression 增量 0，温度只在 `nominal/fair`，全部硬门通过；
- 32K 后模型进程内存由约 25 GB 回落到约 20 GB，未随请求数线性增长。

因此，最终配置适合典型短到中长上下文。本机能跑 32K，但在其他桌面负载已经占用大量统一内存时，不应把“能完成”误写成“零页面压力”。如果生产请求长期接近 32K，应关闭不必要的高内存桌面应用后重新跑 30–60 分钟零 pageout 门，或把典型上下文下调到 16K/30K。

## 停用与复现

先卸载模型，再停止服务：

```bash
curl -fsS -X POST \
  http://127.0.0.1:18000/admin/api/models/Qwen3.8-27B-4bit/unload

# 向前台 oMLX 进程发送 Ctrl-C，然后验证：
lsof -nP -iTCP:18000 -sTCP:LISTEN
```

本次测试结束时已执行卸载与停服，并验证：模型 `loaded=false`、监听端口消失、oMLX PID 不再存在；随后停止 `caffeinate`。模型权重和忽略的原始结果仍保留在 `results/qwen38-m4max-20260819/`。

关键原始结果：

- `quality/ar-final.json` 与 `quality/mtp-depth-2-final.json`；
- `strict/isl-2048-osl-512/` 下 8 个 ABBA trial；
- `strict/isl-8192-osl-512/finalist-mtp-depth-2.json`；
- `strict/isl-32768-osl-512/finalist-mtp-depth-2.json`；
- `stability-mtp-depth-2-140.json` 与 `stability-mtp-depth-2-steady-45.json`；
- `telemetry-mtp-depth-2-final-stability.jsonl` 与 `telemetry-mtp-depth-2-steady-short.jsonl`；
- `server-state/logs/server.log`。

`results/` 默认不提交 Git；本文是可审阅的持久结论，原始文件留在本机供复查。

## 未覆盖范围

- 本轮目标是单用户文本/工具/结构化输出；虽然目标产物走 VLM engine，但没有做图像输入质量测试；
- 没有做多并发、LAN 暴露、共享前缀缓存或服务鉴权测试；
- 没有启用依赖完整 Xcode/私有源码路径的 ANE prefill；
- DFlash、SpecPrefill、TurboQuant KV 与缓存均未混入最终比较；
- 社区 4-bit 转换产物的结果不能外推为官方 BF16、其他量化或 GGUF；
- 原生模型上下文声明为 262,144，但本部署只声明并测试到 65,536 服务上限，正式性能形状最高为 32K+512。
