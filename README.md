# Local AI Tuning

这个仓库收纳一套可复现的本地大模型推理选型与调优方法。当前首批研究对象是：

- `Qwen/Qwen3.8-27B`
- `deepseek-ai/DeepSeek-V4-Flash-0731`
- DGX Spark（GB10 / SM121 / 128 GB UMA）
- RTX PRO 6000 Blackwell（SM120 / 96 GB）
- Apple Silicon（原生 arm64 / Metal / MLX-oMLX，按精确芯片 bin 和统一内存配置）

入口：

- [调研报告](research/2026-08-18-local-inference-landscape.md)
- [Apple Silicon 扩展调研](research/2026-08-19-apple-silicon-extension.md)
- [local-ai-inference-tuning skill](skills/local-ai-inference-tuning/SKILL.md)
- [硬件采集脚本](skills/local-ai-inference-tuning/scripts/collect_hardware.py)
- [Apple 运行窗 telemetry](skills/local-ai-inference-tuning/scripts/sample_apple_telemetry.py)
- [实验计划生成器](skills/local-ai-inference-tuning/scripts/plan_experiments.py)
- [统一 OpenAI API 微基准](skills/local-ai-inference-tuning/scripts/benchmark_openai.py)
- [结果比较器](skills/local-ai-inference-tuning/scripts/compare_runs.py)
- [结构化质量门模板](skills/local-ai-inference-tuning/templates/quality-gate.json)

这里的“最好”始终绑定具体目标：单请求 completion decode tok/s、聚合 completion tok/s，或满足 TTFT/TPOT SLO 的 goodput；reasoning-inclusive 与 visible-answer 指标分开。跨仓库 README 中不同工作负载的数字不会被直接排名。
