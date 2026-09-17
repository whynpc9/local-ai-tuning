# Qwen3.8-27B / 16 GB NVIDIA 一键安装项目登记

> 登记日期：2026-09-12。核对公开仓库 README、环境模板及部分配置、下载和服务源码；未安装依赖、下载模型、启动服务或复跑 GPU 测试。本次属于项目登记，容量与质量数字保留作者口径。

## 项目与固定版本

| 项目 | 登记内容 |
| --- | --- |
| 仓库 | [MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install) |
| 固定提交 | [`622c7965ed5e02a13b188c9ef21bb9857fd2ba28`](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/tree/622c7965ed5e02a13b188c9ef21bb9857fd2ba28) |
| 提交时间 | 2026-09-10 17:38:44 +03:00；`Update GPU memory requirements in README` |
| 定位 | Windows / Linux 单张消费级 NVIDIA 显卡的本地推理安装与启动套件 |
| 推理栈 | Qwen3.8-27B、EXL3 量化、exllamav3、默认使用模型内 MTP head |
| 服务形态 | OpenAI-compatible `/v1`，可配套 DeepSeek Harness 聊天界面 |
| 代码许可证 | 仓库 `LICENSE` 为 MIT；本次未核对模型权重及完整依赖许可链 |
| 登记状态 | 低显存部署实验候选；尚无本仓库运行与业务验收证据 |

项目名保留了最初的 16 GB 定位，当前 README 已列出 12 / 16 / 24 / 32 GB+ 档位。它与此前登记的 [EXL3 + DFlash2 项目](2026-09-01-qwen38-27b-dflash2-exl3-5bpw.md) 分开维护：本项目默认采用内置 MTP，不能直接继承 DFlash2 的性能结论。

## 安装要求与入口

按固定版本 README：NVIDIA compute capability 7.5+、至少 12 GB VRAM，驱动 570+、64 位 Python 3.11+；聊天界面另需 Node 22.19+。模型下载约 9.7–22.9 GB/量化档，另需 Python 环境和 PyTorch 空间。匹配预编译 wheel 时无需 CUDA Toolkit 或编译工具；无匹配 wheel 时会回退源码构建，不能把“一键安装”理解为所有平台都无需编译。

- Windows：`windows\START-HERE.bat` 安装，`windows\start.bat` 日常启动。
- Linux：`./linux/setup.sh` 安装，`./linux/start.sh` 启动；`--no-harness` 仅提供 API。
- API 默认端口 8888；聊天界面默认 loopback 3080。

以上为入口说明，本次没有执行这些命令。[来源：固定版本 README](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/blob/622c7965ed5e02a13b188c9ef21bb9857fd2ba28/README.md)。

## 显存档位与配置口径

以下为作者 README 的 profile 菜单摘要，**不是本仓库实测，也不是任意同显存显卡的容量保证**：

| 显存 | README 默认选择 | 其他选择与边界 |
| --- | --- | --- |
| 12 GB | 2.0 bpw，约 33K，纯文本 | 最低档 |
| 16 GB | 2.5 bpw，约 176K，含图像 | 3.0 bpw 约 118K 含图像；3.5 bpw 约 78K 纯文本 |
| 24 GB | 4.0 bpw，约 262K，含图像 | 5.0 bpw 约 180K 含图像；6.0 bpw 约 84K 纯文本 |
| 32 GB+ | README 列同一菜单 | 高精度档仍受作者已完成 prefill 的上限限制 |

源码 `tools/profiles.py` 保存量化档和作者测量值，并把选择写回 `.env`。2.0 bpw 来自 `Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw`；其余档来自 `turboderp/Qwen3.8-27B-exl3` 的量化分支。profile 使用 int4 KV；原始 `.env.example` 则仍是 2.0 bpw、`CONTEXT_SIZE=199936`、`CACHE_QUANT=8,4`、`GPU_MEM_GB=14.7`。两套配置不能混为一个默认基线。[来源：profiles.py](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/blob/622c7965ed5e02a13b188c9ef21bb9857fd2ba28/tools/profiles.py)、[环境模板](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/blob/622c7965ed5e02a13b188c9ef21bb9857fd2ba28/.env.example)。

README 的默认选择解释与上方表格存在局部表述冲突：解释段把 16 GB / 3.0 bpw 的 118K 写成仅文本，但表格及测量表均列为带图像；24 GB / 5.0 bpw 的解释也与约 180K 带图像的表格不一致。复现时应保存实际生成的 `.env`、vision 加载状态与完整 prefill 结果。

## 已核对的采用边界

1. **服务串行生成**：`tools/serve_openai.py` 通过 `gen_lock` 串行化生成，默认 `DRAFT_DIR="mtp"`。能接收多个请求不代表具备批量推理吞吐；多用户排队延迟需另测。
2. **API 默认无鉴权**：模板为 `HOST=0.0.0.0`；其中 `API_KEY` 注释明确说明服务端忽略该值。聊天界面的 loopback/token 边界不能当成模型 API 的鉴权。
3. **结构化输出尚未验收**：本次检查的服务文件未发现 `response_format` 或 `json_schema` 处理；OpenAI-compatible 标签不能作为 strict JSON/schema 约束已支持的证据。
4. **模型版本未固定为不可变提交**：profile 使用量化分支，2.0 bpw 的 revision 留空。复现需另记模型 commit、权重 hash、实际 engine wheel / CUDA / PyTorch 版本。模板指定 `DSH_VERSION=0.1.3-alpha.2`，同时注明不可用时可能回退 `latest`，也需记录实际解析版本。
5. **长上下文与低比特质量仍待验证**：能完成一次 prefill、作者 KL 数字和图像加载成功，均不足以证明真实业务精度、长文本检索或持续运行稳定性。

服务行为来源：[serve_openai.py，固定提交](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install/blob/622c7965ed5e02a13b188c9ef21bb9857fd2ba28/tools/serve_openai.py)。这些是登记所需的静态核对，不构成完整代码审计。

## 后续复现记录

如进入实际评估，先固定显卡型号、空闲 VRAM、系统/驱动、源码与权重版本及生成配置，再比较同一量化档 MTP 开/关；记录短/长输入的 TTFT、decode、峰值显存、图像回退、C1 与排队表现，并验证目标业务 JSON、工具调用、取消和质量。实际采用前复查上游版本；本登记不改变既有部署方案。
