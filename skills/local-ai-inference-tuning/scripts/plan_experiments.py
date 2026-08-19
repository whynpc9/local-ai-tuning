#!/usr/bin/env python3
"""Generate a conservative experiment matrix for the covered model/hardware pairs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


MODEL_ALIASES = {
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-27B": "Qwen/Qwen3.8-27B",
    "deepseek-v4-flash-0731": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "deepseek-ai/DeepSeek-V4-Flash-0731",
}

OBJECTIVES = ("single-stream", "aggregate", "goodput")

HARDWARE_PROFILES = (
    "apple-silicon-1",
    "dgx-spark-1",
    "dgx-spark-2",
    "rtx-pro-6000-1",
    "rtx-pro-6000-2",
    "rtx-pro-6000-4",
    "custom",
)


def artifact_contract(artifact_class: str) -> dict[str, str]:
    """Return fields that keep unlike model artifacts out of one ranking lane."""
    if artifact_class == "official-checkpoint":
        equivalence = "same-exact-artifact-only"
    elif artifact_class == "same-as-single-node-control":
        equivalence = "inherits-exact-control-artifact"
    else:
        equivalence = "not-assumed-requires-quality-evaluation"
    return {
        "artifact_class": artifact_class,
        "quality_equivalence": equivalence,
        "comparison_contract_rule": "assign a new contract for every exact artifact, tokenizer, transformation, and advertised context",
    }


def official_checkpoint_feasibility(model_id: str, hardware: str) -> dict[str, str]:
    if hardware == "apple-silicon-1":
        status = "inventory-and-runtime-required"
        reason = (
            "Apple Silicon capacity varies from small laptop configurations to 512 GB systems, and MLX/Metal runtimes may require "
            "a converted artifact. Admit only from the attached chip/bin, physical memory, exact artifact bytes, runtime peak, and reserve."
        )
    elif model_id == "Qwen/Qwen3.8-27B":
        status = "unknown" if hardware == "custom" else "storage-capacity-feasible"
        reason = "Official BF16/FP8 storage fits covered 96/128 GB devices; runtime reserve and requested context still require admission checks."
    elif hardware in {"dgx-spark-1", "rtx-pro-6000-1"}:
        status = "rejected"
        reason = "The roughly 166.9 GB fused checkpoint exceeds total physical memory before runtime and cache reserve."
    elif hardware == "dgx-spark-2":
        status = "sharded-capacity-feasible"
        reason = "Two nodes provide enough sharded storage capacity, subject to runtime reserve and measured collective behavior."
    elif hardware == "rtx-pro-6000-2":
        status = "conditional-tight-capacity"
        reason = "192 GB aggregate capacity is tight after layouts, cache, workspaces, and reserve; exact admission measurement is mandatory."
    elif hardware == "rtx-pro-6000-4":
        status = "sharded-capacity-feasible"
        reason = "Four cards provide storage headroom, subject to exact topology, layout, and runtime checks."
    else:
        status = "unknown"
        reason = "Custom hardware requires exact storage and resident-memory accounting."
    return {"status": status, "reason": reason}


def qwen_candidates(hardware: str) -> list[dict[str, Any]]:
    if hardware == "apple-silicon-1":
        return [
            {
                "id": "qwen-omlx-oq4e-target-only-control",
                "role": "same-artifact-no-speculation-control",
                "evidence_tier": "C",
                "evidence_detail": "The Weschera repository publishes an exact oMLX 0.6.1 configuration and raw streams, but several saved outputs fail task correctness or react to filler text.",
                "stack": "native arm64 oMLX release pinned by version or package digest, Metal backend",
                "artifact": "Jundot/Qwen3.8-27B-oQ4e-mtp pinned by immutable revision",
                **artifact_contract("community-mixed-bit-mlx"),
                "baseline": {
                    "speculative": "off",
                    "mtp_enabled": False,
                    "qwen35_ane_prefill_enabled": False,
                },
                "sweeps": {
                    "context": "admit from measured resident bytes with macOS reserve and zero swap",
                    "cache_and_batch": "one factor at a time after a fixed C1 baseline",
                },
                "caveat": "This exact target-only lane is mandatory before attributing any speedup to MTP. Use the chat endpoint and verify that thinking is actually disabled.",
            },
            {
                "id": "qwen-omlx-oq4e-native-mtp",
                "role": "speculative-challenger",
                "evidence_tier": "C",
                "evidence_detail": "Source-reported M4 Max results show a large MTP gain, but the saved benchmark outputs do not pass a strict quality gate. The README mislinks an empty account; the real public source is jundot/omlx and must be pinned separately.",
                "stack": "the identical pinned native arm64 oMLX runtime as the target-only control",
                "artifact": "the identical pinned Jundot/Qwen3.8-27B-oQ4e-mtp artifact as the target-only control",
                **artifact_contract("community-mixed-bit-mlx"),
                "comparison_group": "same exact oMLX artifact and runtime as qwen-omlx-oq4e-target-only-control",
                "baseline": {"comparison": "qwen-omlx-oq4e-target-only-control"},
                "sweeps": {
                    "mtp_num_draft_tokens": [1, 2, 3],
                    "qwen35_ane_prefill_enabled": [False, "true-only-in-an-explicit-source-build-lane"],
                },
                "caveat": "Do not promote the reported 48/65 tok/s means: raw streams include benchmark filler/repetition and wrong-task output. Reproduce on v0.6.1 only as history; start deployment trials from pinned v0.6.2 or newer after re-baselining. Require task, tool, structured-output, long-decode, and acceptance gates for every depth.",
            },
            {
                "id": "qwen-mlx-vlm-mtp-sidecar",
                "role": "open-stack-challenger",
                "evidence_tier": "C",
                "evidence_detail": "The entry repository includes a runnable sidecar benchmark and raw streams; exact upstream revisions, acceptance counters, and strict output quality still require local proof.",
                "stack": "pinned native arm64 mlx-vlm/MLX revision plus a separately pinned MTP sidecar",
                "artifact": "same exact community mixed-bit target and MTP tensors where both runtimes can load them",
                **artifact_contract("community-mixed-bit-mlx"),
                "baseline": {"speculative": "off", "endpoint": "OpenAI-compatible chat or a semantically identical client"},
                "sweeps": {"mtp_num_draft_tokens": [1, 2, 3], "metal_compile_warmup": "fixed before measured trials"},
                "caveat": "Treat sidecar and native oMLX as different framework identities even when the artifact matches; pin every source revision and verify token accounting from tokenizer/usage, never SSE chunks.",
            },
            {
                "id": "qwen-mtplx-native-mtp",
                "role": "auditable-native-mtp-challenger",
                "evidence_tier": "C",
                "evidence_detail": "MTPLX publishes implementation, raw Apple benchmark/quality/thermal artifacts, and auto-tuning, but its reported Qwen artifact revisions are not fixed in-repository.",
                "stack": "youssofal/MTPLX pinned by commit with native arm64 MLX 0.32-compatible runtime",
                "artifact": "one exact MTPLX Qwen3.8 bare/optimized/quality artifact pinned by immutable revision",
                **artifact_contract("community-mlx-quantization"),
                "baseline": {"speculative": "autoregressive", "same_artifact": True},
                "sweeps": {"decode": ["AR", "MTP-depth-1", "MTP-depth-2", "MTP-depth-3"], "domain": ["prose", "code", "reasoning", "tools"]},
                "caveat": "Do not mix its 4-bit/8-bit profiles in one comparison contract. Pin the missing artifact revision locally and preserve exactness, behavior, thermal, and kernel self-check gates.",
            },
            {
                "id": "qwen-mlx-external-dspark",
                "role": "external-drafter-challenger",
                "evidence_tier": "C",
                "evidence_detail": "mlx-dspark publishes the implementation and adaptive cost-curve method, while current Qwen3.8 performance is source-reported without pinned target/drafter revisions or checked-in raw results.",
                "stack": "ARahim3/mlx-dspark pinned by commit on native arm64 MLX",
                "artifact": "exact target plus separately pinned RadixArk/DimInfer drafter, tokenizer, and advertised context",
                **artifact_contract("community-target-plus-draft"),
                "baseline": {"speculative": "off", "same_target": True},
                "sweeps": {"method": ["AR", "external-DSpark"], "draft_cap": "machine/model/quant/domain cost-curve auto-cap", "fallbacks": ["lookup"]},
                "caveat": "Re-run token-by-token race/equality, domain-separated acceptance, and quality gates. Keep this DSpark lane separate from the now-public DFlash 2 artifact/runtime contract.",
            },
            {
                "id": "qwen-mlx-dflash2",
                "role": "public-block-parallel-drafter-challenger",
                "evidence_tier": "C",
                "evidence_detail": "The Qwen3.8 DFlash 2 checkpoint, configuration, GGUF variants, MLX inference backend, and oMLX fork are public. Official H200 results are strong, but Apple raw results, repeats, memory, prefill, long-context, batching, and quality evidence are not yet sufficient for promotion.",
                "stack": "z-lab/dflash MLX backend or z-lab/omlx-fork@0.6.2-dflash2, each pinned separately on native arm64",
                "artifact": "same exact target control plus incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4",
                **artifact_contract("public-target-plus-dflash2-draft"),
                "baseline": {"speculative": "off", "same_target": True},
                "sweeps": {
                    "method": ["AR", "native-MTP", "DFlash2"],
                    "block_size": [2, 3, 4, 5],
                    "draft_quantization": ["4-bit", "8-bit", "BF16-if-admitted"],
                    "domain": ["prose", "code", "reasoning", "tools", "strict-json"],
                },
                "caveat": "Do not attribute the fork to upstream jundot/oMLX. Pin target, draft, runtime, and quantization separately; require token equality/distribution, extra-prefill, peak-memory, zero-swap, long-context, continuous-batching, tool/JSON, vision, and thermal gates. Quantized MLX begins with block_size <= 5.",
            },
            {
                "id": "qwen-mlx-released-target-only",
                "role": "auditable-target-only-control",
                "evidence_tier": "D",
                "evidence_detail": "A framework-control hypothesis until an exact released MLX/MLX-LM/MLX-VLM revision proves this architecture and artifact path.",
                "stack": "latest pinned released native arm64 MLX-family runtime that passes the exact architecture loader gate",
                "artifact": "official-source local conversion or community MLX quantization, each in a separate comparison contract",
                **artifact_contract("converted-mlx"),
                "baseline": {"speculative": "off", "native_arm64": True},
                "sweeps": {"quantization": ["same-artifact-control", "4-bit", "6-bit", "8-bit"], "context_and_batch": "after exact fit measurement"},
                "caveat": "Do not infer Qwen3.8 support from generic Qwen support. Reject unsupported architecture, Python/Rosetta, CPU fallback, or changed chat-template semantics.",
            },
            {
                "id": "qwen-magnitude-llamacpp-metal-control",
                "role": "portable-metal-control",
                "evidence_tier": "C",
                "evidence_detail": "Magnitude supplies a reproducible GGUF planning/benchmark method; no reviewed exact-target Apple winner is assumed.",
                "stack": "pinned native arm64 Magnitude/llama.cpp Metal build",
                "artifact": "pinned Qwen3.8 GGUF Q4/Q6/Q8, plus projector when vision is required",
                **artifact_contract("converted-gguf"),
                "baseline": {"speculative": "off", "gpu_offload": "all admitted Metal layers"},
                "sweeps": {
                    "quantization": ["Q4", "Q6", "Q8"],
                    "context_batch_gpu_layers": "from Magnitude fit plan, then verified by resident memory and zero swap",
                },
                "caveat": "This is a separate artifact/runtime lane. Use it as the broad portable control, not as quality-equivalent evidence for oQ4e or official weights. Current llama.cpp Qwen MTP returns early on vision embeddings, so image+MTP is unsupported/experimental even though mmproj and text MTP work separately.",
            },
        ]
    if hardware == "dgx-spark-2":
        return [
            *qwen_candidates("dgx-spark-1"),
            {
                "id": "qwen-two-spark-capacity-lane",
                "role": "conditional-capacity-or-throughput-lane",
                "evidence_tier": "D",
                "evidence_detail": "No exact-model two-Spark performance evidence was reviewed; this is a topology hypothesis.",
                "stack": "pinned SGLang or vLLM ARM64/SM121 build, TP=2 only after one-node baseline",
                "artifact": "same pinned artifact as the one-node control",
                **artifact_contract("same-as-single-node-control"),
                "baseline": {"speculative": "off", "tensor_parallel": 2},
                "sweeps": {"enable_only_for": ["required live-token capacity", "measured aggregate-throughput gain"]},
                "caveat": "Qwen fits one Spark. Two-node TP adds 200 GbE communication without normal device-memory GDR and is unlikely to be the C1 default.",
            },
        ]
    if hardware == "dgx-spark-1":
        return [
            {
                "id": "qwen-official-fp8-cross-framework-control",
                "role": "common-denominator-control",
                "evidence_tier": "B",
                "evidence_detail": "Official model artifact, but exact GB10 runtime tuple still requires a pinned reproducible image and loader/kernel proof.",
                "stack": "same exact official FP8 artifact on pinned SGLang and vLLM ARM64/SM121 builds, each only after loader/kernel gates",
                "artifact": "official Qwen3.8-27B FP8 checkpoint pinned by revision",
                **artifact_contract("official-checkpoint"),
                "baseline": {"speculative": "off", "kv_dtype": "fp8"},
                "sweeps": {"framework": ["SGLang", "vLLM"]},
                "caveat": "This is the comparable control, not an assertion that both installed builds support the artifact. Reject a framework on loader, exact-SM kernel, or fallback failure.",
            },
            {
                "id": "qwen-sglang-sm121",
                "role": "starting-point",
                "evidence_tier": "B",
                "evidence_detail": "Official SGLang model cookbook/image plus a community NVFP4 target; local raw performance still required.",
                "stack": "SGLang model-specific ARM64/SM121 image",
                "artifact": "community NVFP4 checkpoint pinned by revision",
                **artifact_contract("community-quantization"),
                "comparison_group": "best-achievable-unless-the-identical-NVFP4-revision-loads-in-both-frameworks",
                "baseline": {"speculative": "off", "kv_dtype": "fp8_e4m3"},
                "sweeps": {
                    "speculative": ["off", "native-mtp-depth-2", "native-mtp-depth-3", "native-mtp-depth-4"],
                    "chunked_prefill": [2048, 8192],
                    "state_pool": "derive from installed SGLang: C*(S+D), or C*S only when ReplaySSM removes per-draft snapshots; never hardcode",
                    "persistent_slots_S": {"extra_buffer": 5, "extra_buffer_lazy": 4, "no_buffer": 3},
                    "ssm_state_dtype": ["bf16-control", "fp32-control"],
                    "radix_and_memory": ["installed-default", "ReplaySSM-when-compatible", "mamba-full-memory-ratio admission sweep"],
                },
                "caveat": "Official cookbook validates boot combinations, not a universal performance winner.",
            },
            {
                "id": "qwen-sglang-external-dspark-sm121",
                "role": "speculative-challenger",
                "evidence_tier": "B",
                "evidence_detail": "Current official SGLang Qwen3.8 DGX recipe boot-serves the external RadixArk draft configurations; boot coverage is not a speed verdict.",
                "stack": "pinned SGLang Qwen ARM64/SM121 image with external DSPARK and ReplaySSM only where the exact recipe enables it",
                "artifact": "same exact pinned Qwen target as its target-only SGLang control",
                "draft_artifact": "RadixArk/Qwen3.8-27B-DSpark pinned by immutable revision",
                **artifact_contract("same-as-single-node-control"),
                "baseline": {"comparison": "target-only SGLang control", "kv_dtype": "fp8_e4m3"},
                "sweeps": {
                    "dspark_block_size": [3, 5, 7],
                    "replayssm": ["off", "on-when-compatible"],
                    "state_pool": "C*(S+D) without ReplaySSM; C*S when ReplaySSM removes draft snapshots",
                    "mamba_full_memory_ratio": "admission sweep after deriving slot bytes from installed logs",
                },
                "caveat": "Pin target and draft separately. Validate tools, reasoning, long decode, radix reuse, and acceptance before attributing a gain to DSPARK.",
            },
            {
                "id": "qwen-sglang-dflash2-sm121",
                "role": "public-block-parallel-drafter-challenger",
                "evidence_tier": "C",
                "evidence_detail": "DFlash 2 is merged in SGLang main and the exact Qwen draft is public, but published performance is one H200 rather than GB10 and quantized target LM-head follow-ups were still open at the evidence cutoff.",
                "stack": "SGLang at or after merge c14312a66420b75ca9a11bf1817c4db1fa26b097 on a validated ARM64/SM121 build",
                "artifact": "same exact pinned Qwen target control plus incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4",
                **artifact_contract("same-target-plus-public-dflash2-draft"),
                "baseline": {"comparison": "target-only SGLang control", "same_target": True},
                "sweeps": {"draft_tokens": [3, 5, 7], "concurrency": [1, 8, 32], "domain": ["prose", "code", "reasoning", "tools", "strict-json"]},
                "caveat": "Reject unsupported quantized LM heads, wrong architecture dispatch, kernel fallback, or silent non-speculative execution. Measure target/draft/hidden/cache UMA residency, prefill, context ceiling, swap/pageout, acceptance by position, and E2E against native MTP.",
            },
            {
                "id": "qwen-vllm-gb10-challenger",
                "role": "challenger",
                "evidence_tier": "B",
                "evidence_detail": "Pinned community GB10 fork/image and reported benchmark; not official vLLM release behavior.",
                "stack": "pinned community vLLM ARM64/SM121 image",
                "artifact": "community NVFP4 checkpoint pinned by revision",
                **artifact_contract("community-quantization"),
                "comparison_group": "best-achievable-unless-the-identical-NVFP4-revision-loads-in-both-frameworks",
                "baseline": {"speculative": "off", "kv_dtype": "fp8"},
                "sweeps": {"speculative": ["off", "native-mtp-depth-1", "native-mtp-depth-2", "native-mtp-depth-3"], "graph": ["eager", "captured"]},
                "caveat": "Do not infer support from a vLLM-like version in the repository title; pin the real commit and image digest.",
            },
            {
                "id": "qwen-magnitude-gguf-control",
                "role": "portable-control",
                "evidence_tier": "C",
                "evidence_detail": "Pinned Magnitude planning/benchmark method, but no reviewed exact-target DGX performance run.",
                "stack": "pinned Magnitude/llama.cpp ARM64 build",
                "artifact": "pinned Qwen3.8 GGUF Q4/Q6/Q8 plus projector if vision is required",
                **artifact_contract("converted-gguf"),
                "baseline": {"speculative": "off", "context": 100000},
                "sweeps": {"quantization": ["Q4", "Q6", "Q8"], "batch_and_gpu_split": "from exact GGUF admission plan"},
                "caveat": "A separate artifact/runtime lane; use Magnitude's fit estimate to shortlist, then measure against specialized NVFP4 stacks.",
            },
        ]
    if hardware == "rtx-pro-6000-1":
        return [
            {
                "id": "qwen-sglang-sm120",
                "role": "starting-point",
                "evidence_tier": "B",
                "evidence_detail": "Official framework path combined with community NVFP4 and weak exact-repo benchmark evidence.",
                "stack": "SGLang SM120 recipe with FlashInfer",
                "artifact": "pinned NVFP4, plus official FP8 control",
                **artifact_contract("mixed-artifact-sweep-must-split"),
                "baseline": {"speculative": "off", "kv_dtype": "fp8_e4m3"},
                "sweeps": {
                    "speculative": ["off", "native-mtp-depth-2", "native-mtp-depth-3", "external-dspark-depth-7"],
                    "chunked_prefill": [2048, 4096, 8192],
                },
                "caveat": "A community 200-223 tok/s headline lacks a canonical harness and uses a potentially stale state-pool formula; remeasure.",
            },
            {
                "id": "qwen-sglang-dflash2-sm120",
                "role": "public-block-parallel-drafter-challenger",
                "evidence_tier": "C",
                "evidence_detail": "The public checkpoint and SGLang main implementation are inspectable, but official results use H200/FA3 and do not prove SM120 kernels, quantized-target compatibility, memory, long-context, or product quality.",
                "stack": "SGLang at or after merge c14312a66420b75ca9a11bf1817c4db1fa26b097 on a pinned SM120 build",
                "artifact": "same exact pinned Qwen target control plus incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4",
                **artifact_contract("same-target-plus-public-dflash2-draft"),
                "baseline": {"comparison": "qwen-sglang-sm120 target-only lane", "same_target": True},
                "sweeps": {"draft_tokens": [3, 5, 7], "concurrency": [1, 8, 32], "domain": ["prose", "code", "reasoning", "tools", "strict-json"]},
                "caveat": "Prove exact SM120 attention, top-k, convolution, and graph paths. Reject unsupported quantized LM heads or fallback; compare E2E, prefill, acceptance, VRAM/context ceiling, power, and quality against native MTP before promotion.",
            },
            {
                "id": "qwen-vllm-sm120",
                "role": "challenger",
                "evidence_tier": "B",
                "evidence_detail": "Released architecture plus exact-revision and SM120 kernel gates; exact local tuple remains community-integrated.",
                "stack": "vLLM revision with Qwen3.8 and SM120 kernels",
                "artifact": "official FP8 control and pinned community NVFP4",
                **artifact_contract("mixed-artifact-sweep-must-split"),
                "baseline": {"speculative": "off"},
                "sweeps": {"speculative": ["off", "native-mtp-depth-1", "native-mtp-depth-2", "native-mtp-depth-3"], "graph": ["eager", "captured"]},
                "caveat": "Confirm that required MTP fixes are in the installed revision, not only on upstream main.",
            },
            {
                "id": "qwen-magnitude-gguf-sm120-control",
                "role": "portable-control",
                "evidence_tier": "C",
                "evidence_detail": "Pinned Magnitude method, but no reviewed exact-target RTX PRO 6000 performance run.",
                "stack": "pinned Magnitude/llama.cpp CUDA build",
                "artifact": "pinned Qwen3.8 GGUF Q4/Q6/Q8 plus projector if required",
                **artifact_contract("converted-gguf"),
                "baseline": {"speculative": "off", "context": 100000},
                "sweeps": {"quantization": ["Q4", "Q6", "Q8"], "batch_and_gpu_split": "from exact GGUF admission plan"},
                "caveat": "Do not assume broad CUDA portability beats model-specific SM120 NVFP4/MTP kernels.",
            },
        ]
    if hardware in {"rtx-pro-6000-2", "rtx-pro-6000-4"}:
        return [
            *qwen_candidates("rtx-pro-6000-1"),
            {
                "id": f"qwen-vllm-sm120-tp{2 if hardware.endswith('-2') else 4}",
                "role": "conditional-throughput-lane",
                "evidence_tier": "B",
                "evidence_detail": "General released/community parallel recipe; exact topology and artifact run still required.",
                "stack": f"pinned vLLM/SGLang SM120 build, TP={2 if hardware.endswith('-2') else 4}",
                "artifact": "same pinned artifact as the single-card control",
                **artifact_contract("same-as-single-node-control"),
                "baseline": {"speculative": "off"},
                "sweeps": {"parallelism": ["one-card", f"TP={2 if hardware.endswith('-2') else 4}"], "objective": ["aggregate", "goodput"]},
                "caveat": "Run the one-card baseline first; dense-model C1 can regress across PCIe even when aggregate throughput rises.",
            },
        ]
    return [
        {
            "id": "qwen-generic-baseline",
            "role": "manual-baseline",
            "evidence_tier": "D",
            "evidence_detail": "No covered exact hardware/runtime tuple.",
            "stack": "latest released framework that registers the exact architecture on the exact SM",
            "artifact": "official BF16/FP8 if it fits; community quantization only with revision and quality gate",
            **artifact_contract("conditional-official-or-community"),
            "baseline": {"speculative": "off"},
            "sweeps": {"framework": ["SGLang", "vLLM", "Magnitude/llama.cpp GGUF", "TensorRT-LLM only after model validation"]},
            "caveat": "No covered, evidence-backed hardware recipe; perform compatibility and capacity gates first.",
        }
    ]


def deepseek_candidates(hardware: str) -> list[dict[str, Any]]:
    if hardware == "apple-silicon-1":
        return [
            {
                "id": "deepseek-apple-official-admission",
                "role": "conditional-official-control",
                "evidence_tier": "A",
                "evidence_detail": "Official checkpoint size is known; feasibility and framework support depend on the exact high-memory Mac and runtime tuple.",
                "stack": "pinned jundot/omlx v0.6.2-or-newer native arm64 path only after exact DeepseekV4 architecture and kernel gates",
                "artifact": "official fused DeepSeek-V4-Flash-0731 checkpoint pinned by revision",
                **artifact_contract("official-checkpoint"),
                "baseline": {"speculative": "off"},
                "sweeps": {"only_after_fit_and_loader_proof": ["context", "batch", "cache"]},
                "caveat": "Reject on systems whose total unified memory cannot hold roughly 167 GB of files plus runtime, OS, and cache reserve. A 128 GB Mac is a hard capacity failure; 192 GB is a tight measured-admission case, not an automatic pass.",
            },
            {
                "id": "deepseek-apple-community-mlx-quant",
                "role": "experimental-converted-artifact",
                "evidence_tier": "D",
                "evidence_detail": "Related community Mac recipes exist, but exact conversion, runtime, raw quality, and reproducible performance must be audited before promotion.",
                "stack": "pinned jundot/omlx v0.6.2-or-newer DeepSeek patches and exact native Metal kernels; mlx-vlm is a research control",
                "artifact": "separately pinned community quantized/converted DeepSeek-V4-Flash-0731 artifact (for example q4 on 256 GB or a much smaller low-bit lane on 128 GB)",
                **artifact_contract("converted-mlx"),
                "baseline": {"speculative": "off-if-supported"},
                "sweeps": {"only_after_quality_gate": ["draft off/on", "quantization", "context", "batch"]},
                "caveat": "Reviewed q4/oQ4e residency is roughly 156-161 GB and naturally targets 256 GB; a roughly 92.8 GB 2.4-bit artifact can enter a 128 GB admission test but has a different quality contract. Do not transfer a 49 tok/s-style headline across ablation, quantization, draft, prompt, or Mac bin.",
            },
            {
                "id": "deepseek-magnitude-llamacpp-metal-admission",
                "role": "portable-converted-control",
                "evidence_tier": "C",
                "evidence_detail": "Magnitude can enumerate GGUF sizes and construct a fit plan; exact DeepSeek artifact quality and Apple performance remain local measurements.",
                "stack": "pinned native arm64 Magnitude/llama.cpp Metal build with the exact DeepSeek DSpark implementation",
                "artifact": "pinned DeepSeek-V4-Flash-0731 GGUF target and optional draft, each admitted by exact bytes",
                **artifact_contract("converted-gguf"),
                "baseline": {"speculative": "off", "gpu_offload": "all admitted Metal layers"},
                "sweeps": {"quantization": "only variants that fit with reserve", "draft": ["off", "on-after-target-only-quality"]},
                "caveat": "High-memory Apple systems are not interchangeable. Keep transformed GGUF results outside the official-artifact comparison contract.",
            },
        ]
    if hardware == "dgx-spark-2":
        return [
            {
                "id": "deepseek-vllm-sparkinfer-tp2",
                "role": "starting-point",
                "evidence_tier": "B",
                "evidence_detail": "Pinned community two-GB10 stack with runnable deployment and results.",
                "stack": "pinned ARM64 vLLM community image with SparkInfer/B12X, TP=2",
                "artifact": "official fused DeepSeek-V4-Flash-0731 checkpoint pinned by revision",
                **artifact_contract("official-checkpoint"),
                "baseline": {"speculative": "off", "tensor_parallel": 2},
                "sweeps": {
                    "speculative": ["off", "fused-dspark-depth-3", "fused-dspark-depth-5", "fused-dspark-depth-7"],
                    "graph": ["eager-control", "regular-cudagraph", "piecewise-if-required"],
                    "max_num_seqs": [1, 2, 4, 6, 8],
                },
                "caveat": "Two Sparks solve capacity first. Measure NCCL because GB10 does not provide normal device-memory GPUDirect RDMA semantics.",
            },
            {
                "id": "deepseek-sglang-dspark-tp2",
                "role": "challenger",
                "evidence_tier": "C",
                "evidence_detail": "Released framework DSPARK support, but exact two-GB10 runtime/performance evidence is incomplete.",
                "stack": "pinned SGLang DSPARK ARM64/SM121 build, TP=2",
                "artifact": "official fused DeepSeek-V4-Flash-0731 checkpoint pinned by revision",
                **artifact_contract("official-checkpoint"),
                "baseline": {"speculative": "off", "tensor_parallel": 2},
                "sweeps": {"speculative": ["off", "DSPARK-depth-5"], "collective_stability": "required"},
                "caveat": "DSPARK is released, but exact two-GB10 stacks have had rank-divergence and acceptance issues; stress-test the pinned build.",
            },
        ]
    if hardware == "dgx-spark-1":
        return [
            {
                "id": "deepseek-sparkinfer-transformed-single",
                "role": "experimental-alternative",
                "evidence_tier": "B",
                "evidence_detail": "Pinned 0xSero transformed artifact/appliance with results and known failed gates.",
                "stack": "pinned custom SparkInfer/vLLM fork, TP=1",
                "artifact": "expert-transformed/pruned EXL3/Trellis target served by a pinned vLLM+B12X stack, plus a separately pinned compact draft",
                **artifact_contract("transformed-target"),
                "baseline": {"speculative": "target-only-required-if-supported"},
                "sweeps": {"only_after_quality_gate": ["documented fixed draft depth", "graph mode", "cache layout"]},
                "caveat": "Not quality-equivalent to the official checkpoint by default. If the appliance cannot run target-only, record that isolation failure and downgrade every speculative speed claim.",
            },
            {
                "id": "deepseek-official-single-spark",
                "role": "rejected-capacity-control",
                "evidence_tier": "A",
                "evidence_detail": "Official repository bytes versus vendor physical-memory capacity.",
                "stack": "official fused checkpoint on one 128 GB Spark",
                "artifact": "official fused DeepSeek-V4-Flash-0731 checkpoint",
                **artifact_contract("official-checkpoint"),
                "rejected": True,
                "stop_reason": "Checkpoint storage is roughly 167 GB before runtime/cache reserve; it does not fit one 128 GB UMA system.",
            },
            {
                "id": "deepseek-magnitude-gguf-admission-control",
                "role": "rejected-capacity-control",
                "evidence_tier": "B",
                "evidence_detail": "Pinned catalog plus publisher-reported Q4/Q8 file sizes; rejection is before runtime allocation.",
                "stack": "pinned Magnitude/llama.cpp ARM64 build",
                "artifact": "community DeepSeek-V4-Flash-0731 GGUF Q4/Q8 plus GGUF DSpark draft",
                **artifact_contract("converted-gguf"),
                "rejected": True,
                "stop_reason": "The cataloged Q4/Q8 target files are roughly 155/162 GB before runtime, cache, OS reserve, and draft; they do not fit one 128 GB UMA system.",
                "caveat": "Keep Magnitude's exact admission method, but choose a substantially smaller independently identified artifact rather than relabeling these Q4/Q8 entries as feasible.",
            },
            {
                "id": "deepseek-ds4-asymmetric-single",
                "role": "experimental-challenger",
                "evidence_tier": "D",
                "evidence_detail": "Related repository line only; exact source, artifact, and runtime tuple must be pinned and reviewed.",
                "stack": "pinned ds4/Entrpi custom ARM64/SM121 engine after source and artifact review",
                "artifact": "asymmetric substantially-lower-bit GGUF target, exact model and draft revisions required",
                **artifact_contract("transformed-target"),
                "baseline": {"speculative": "off-if-supported"},
                "sweeps": {"only_after_admission_and_quality": ["draft on/off", "context", "batch"]},
                "caveat": "A feasible hypothesis, not a promoted default: pin the complete tuple and prove target-only correctness, tools, and artifact-specific context first.",
            },
        ]
    if hardware in {"rtx-pro-6000-2", "rtx-pro-6000-4"}:
        tp = 2 if hardware.endswith("-2") else 4
        return [
            {
                "id": f"deepseek-vllm-sm120-tp{tp}",
                "role": "starting-point",
                "evidence_tier": "B",
                "evidence_detail": "Pinned community SM120 integrations and results; local topology/artifact proof remains required.",
                "stack": f"pinned vLLM/SparkInfer SM120 image on {tp}x PCIe RTX PRO 6000",
                "artifact": "official fused checkpoint pinned by revision",
                **artifact_contract("official-checkpoint"),
                "baseline": {"speculative": "off", "tensor_parallel": tp, "mla_cache": "fp8_ds_mla"},
                "sweeps": {"speculative": ["off", "fused-dspark-depth-5"], "all_reduce": ["framework-default", "disabled-custom-if-topology-demands"]},
                "caveat": "Assume PCIe-only until topology proves otherwise; two-card fit is tighter, and SM100-only NVFP4 kernels are not SM120 support.",
            }
        ]
    if hardware == "rtx-pro-6000-1":
        return [
            {
                "id": "deepseek-official-single-rtx",
                "role": "rejected-capacity-control",
                "evidence_tier": "A",
                "evidence_detail": "Official repository bytes versus vendor physical-memory capacity.",
                "stack": "official fused checkpoint on one 96 GB RTX PRO 6000",
                "artifact": "official fused DeepSeek-V4-Flash-0731 checkpoint",
                **artifact_contract("official-checkpoint"),
                "rejected": True,
                "stop_reason": "The full checkpoint cannot fit 96 GB before runtime/cache reserve.",
            }
        ]
    return [
        {
            "id": "deepseek-generic-baseline",
            "role": "manual-baseline",
            "evidence_tier": "D",
            "evidence_detail": "No covered exact hardware/runtime tuple.",
            "stack": "vLLM revision that registers DeepseekV4ForCausalLM and the exact hardware kernels",
            "artifact": "official fused checkpoint if capacity permits",
            **artifact_contract("official-checkpoint"),
            "baseline": {"speculative": "off"},
            "sweeps": {"speculative": ["off", "fused-dspark-small-depth"]},
            "caveat": "Perform exact capacity, architecture, MLA/MoE kernel, and topology gates before download or launch.",
        }
    ]


def workload(objective: str) -> dict[str, Any]:
    common = {
        "shapes": [
            {"case_id": "isl-2048-osl-512", "input_tokens": 2048, "output_tokens": 512},
            {"case_id": "isl-8192-osl-512", "input_tokens": 8192, "output_tokens": 512},
            {"case_id": "isl-32768-osl-512", "input_tokens": 32768, "output_tokens": 512},
        ],
        "unique_prefix": True,
        "reasoning_modes": ["off", "target-production-mode"],
        "cache_states": ["warm-engine-cold-prefix", "warm-prefix-separate-lane"],
        "quality_gate_before_performance": True,
    }
    if objective == "single-stream":
        common.update({"concurrency": [1], "requests_per_case": 7, "rank_by": "median single-stream completion decode tok/s; label reasoning inclusion and report TTFT/TTFO/p95/visible-answer rate"})
    elif objective == "aggregate":
        common.update({"concurrency": [1, 2, 4, 8, 16], "minimum_waves_per_case": 3, "rank_by": "aggregate actual completion tok/s at fixed closed-loop concurrency; report visible-answer rate separately"})
    else:
        common.update({"closed_loop_concurrency": [1, 2, 4, 8, 16], "open_loop_rates": "50/70/85/95/110% of measured saturation", "minimum_completed": 1000, "rank_by": "goodput at declared TTFT/TPOT/E2E SLO"})
    return common


def platform_policy(hardware: str) -> dict[str, Any]:
    if hardware != "apple-silicon-1":
        return {
            "execution_lanes": ["correctness", "native-offline", "api-steady", "api-aggregate", "goodput", "stability"],
            "telemetry": "accelerator, power, thermal, clocks, memory/UMA, swap, and framework counters over the measured window",
        }
    return {
        "execution_lanes": [
            "correctness",
            "native-microbench",
            "api-steady",
            "api-aggregate",
            "goodput",
            "cold-start-separate",
            "thermal-stability",
        ],
        "warmup_gate": {
            "same_shape_fingerprint": "model+ISL+OSL+batch+concurrency+dtype+prefill-step+configured-max-draft-depth",
            "minimum_attempts": 5,
            "stable_window": 3,
            "last_window_cv_max": 0.03,
            "last_window_range_over_median_max": 0.05,
            "maximum_attempts": 10,
        },
        "cache_policy": {
            "cold_prefix": "unique early nonce; record cached prompt tokens and fixed-template allowance",
            "warm_prefix": "separate comparison contract with predeclared minimum hit ratio",
            "ssd_restart": "separate oMLX cache-persistence lane; SSD cache is prefix reuse, not weight offload",
        },
        "telemetry_gate": {
            "sampler": "sample_apple_telemetry.py with matching run_id and monotonic window",
            "strict": "native arm64, identical power profile, zero swapout/pageout, nominal thermal, no sustained compression/pagein churn",
            "required_framework_fields": ["Metal current/working-set bytes", "MLX active/cache/peak bytes when available"],
        },
        "speculation_fields": [
            "configured_max_depth",
            "observed_depth_histogram",
            "drafted_tokens",
            "accepted_tokens",
            "accepted_per_position",
            "verification_steps",
            "tok_per_cycle",
            "fallback_to_standard_decode",
        ],
    }


def validate_plan_schema(plan: dict[str, Any]) -> None:
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("plan must contain at least one candidate")
    candidate_ids = [candidate.get("id") for candidate in candidates]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        raise ValueError("every candidate requires a non-empty id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ids must be unique")
    for candidate in candidates:
        if candidate.get("evidence_tier") not in {"A", "B", "C", "D"}:
            raise ValueError(f"{candidate['id']}: evidence_tier must be exactly A, B, C, or D")
        for field in ("evidence_detail", "artifact_class", "quality_equivalence", "comparison_contract_rule"):
            if not candidate.get(field):
                raise ValueError(f"{candidate['id']}: missing {field}")
        if candidate.get("rejected") and not candidate.get("stop_reason"):
            raise ValueError(f"{candidate['id']}: rejected candidate requires stop_reason")


def _apple_capacity_bytes(value: Any) -> int | None:
    match = re.fullmatch(r"\s*([\d.]+)\s*(GB|TB)\s*", str(value or ""), re.IGNORECASE)
    if not match:
        return None
    multiplier = 1024**3 if match.group(2).upper() == "GB" else 1024**4
    return int(float(match.group(1)) * multiplier)


def _apple_metal_supported(value: Any) -> bool:
    return str(value or "").strip().lower() in {"spdisplays_supported", "supported"}


def _probe_has_output(probe: Any) -> bool:
    return bool(
        isinstance(probe, dict)
        and probe.get("available") is True
        and probe.get("returncode") == 0
        and str(probe.get("stdout") or "").strip()
    )


def _apple_power_mode_probe_ok(probe: Any) -> bool:
    modes = probe.get("power_modes") if isinstance(probe, dict) else None
    return _probe_has_output(probe) and isinstance(modes, dict) and any(
        isinstance(profile, dict) and bool(profile) for profile in modes.values()
    )


def validate_hardware_inventories(hardware: str, inventories: list[dict[str, Any]]) -> None:
    if not inventories:
        return
    if hardware not in {"custom", "apple-silicon-1"} and any(
        (inventory.get("validation") or {}).get("nvidia_gpu_probe_ok") is not True
        for inventory in inventories
    ):
        raise ValueError("hardware inventory has no validated NVIDIA GPU identity")

    if hardware == "apple-silicon-1":
        if len(inventories) != 1:
            raise ValueError("apple-silicon-1 requires exactly one local hardware inventory")
        inventory = inventories[0]
        if (inventory.get("validation") or {}).get("apple_silicon_probe_ok") is not True:
            raise ValueError("Apple Silicon inventory has no validated native ARM64 + Metal identity")
        machine = str((inventory.get("host") or {}).get("machine") or "").lower()
        profile = (((inventory.get("apple_silicon") or {}).get("system_profile") or {}).get("profile") or {})
        hardware_profile = profile.get("hardware") or {}
        displays = profile.get("displays") or []
        software = profile.get("software") or {}
        probes = inventory.get("probes") or {}
        if machine != "arm64":
            raise ValueError("Apple Silicon inventory must report the native arm64 ABI, not Rosetta/x86_64")
        rosetta_value = str((probes.get("apple_rosetta_translation") or {}).get("stdout") or "").strip()
        if rosetta_value == "1":
            raise ValueError("Apple Silicon inventory reports an active Rosetta translation process")
        chip = str(hardware_profile.get("chip_type") or "").strip()
        if not re.fullmatch(r"Apple M[1-9][0-9]*(?: (?:Pro|Max|Ultra))?", chip):
            raise ValueError("Apple Silicon inventory must identify the exact Apple chip")
        if not hardware_profile.get("machine_model") or not hardware_profile.get("machine_name"):
            raise ValueError("Apple Silicon inventory must identify the exact chassis/model")
        physical_memory_bytes = _apple_capacity_bytes(hardware_profile.get("physical_memory"))
        if physical_memory_bytes is None or physical_memory_bytes < 8 * 1024**3:
            raise ValueError("Apple Silicon inventory must report sane physical unified memory")
        matching_displays = [
            display
            for display in displays
            if isinstance(display, dict) and str(display.get("sppci_model") or "").strip() == chip
        ]
        if not matching_displays:
            raise ValueError("Apple Silicon inventory must match the integrated GPU to the exact chip")
        try:
            gpu_cores_ok = any(int(str(display.get("sppci_cores") or "").strip()) > 0 for display in matching_displays)
        except ValueError:
            gpu_cores_ok = False
        if not gpu_cores_ok:
            raise ValueError("Apple Silicon inventory must report the exact GPU core count")
        if not any(_apple_metal_supported(display.get("spdisplays_metal")) for display in matching_displays):
            raise ValueError("Apple Silicon inventory must report Metal support")
        if not isinstance(software, dict) or not software.get("os_version"):
            raise ValueError("Apple Silicon inventory must report the exact macOS version")
        if not _apple_power_mode_probe_ok(probes.get("apple_power_settings")) or not _probe_has_output(probes.get("apple_power_source")):
            raise ValueError("Apple Silicon inventory must capture the exact power settings and active power source")
        return

    gpu_lines = []
    for inventory in inventories:
        probe = (inventory.get("probes") or {}).get("nvidia_gpu_query") or {}
        gpu_lines.extend(line.strip() for line in str(probe.get("stdout") or "").splitlines() if line.strip())

    if hardware.startswith("dgx-spark"):
        expected_nodes = 2 if hardware.endswith("-2") else 1
        if len(inventories) != expected_nodes:
            raise ValueError(f"{hardware} requires {expected_nodes} separately collected node inventories")
        for inventory in inventories:
            machine = str((inventory.get("host") or {}).get("machine") or "").lower()
            if machine not in {"aarch64", "arm64"}:
                raise ValueError("DGX Spark inventory must report an ARM64 host")
            probe = (inventory.get("probes") or {}).get("nvidia_gpu_query") or {}
            lines = [line for line in str(probe.get("stdout") or "").splitlines() if line.strip()]
            if not any("GB10" in line and ("12.1" in line or "121" in line) for line in lines):
                raise ValueError("DGX Spark inventory must identify GB10 compute capability 12.1 / SM121")
    elif hardware.startswith("rtx-pro-6000"):
        expected_gpus = int(hardware.rsplit("-", 1)[1])
        matching = [line for line in gpu_lines if "RTX PRO 6000" in line and ("12.0" in line or "120" in line)]
        if len(matching) < expected_gpus:
            raise ValueError(f"{hardware} requires at least {expected_gpus} RTX PRO 6000 SM120 GPUs in the attached inventories")


def apple_memory_bytes(inventory: dict[str, Any]) -> int | None:
    hardware_profile = ((((inventory.get("apple_silicon") or {}).get("system_profile") or {}).get("profile") or {}).get("hardware") or {})
    value = str(hardware_profile.get("physical_memory") or "")
    match = re.fullmatch(r"\s*([\d.]+)\s*(GB|TB)\s*", value, re.IGNORECASE)
    if not match:
        return None
    multiplier = 1024**3 if match.group(2).upper() == "GB" else 1024**4
    return int(float(match.group(1)) * multiplier)


def apply_inventory_admission(
    plan: dict[str, Any],
    hardware: str,
    inventories: list[dict[str, Any]],
    allow_alternative_artifacts: bool,
) -> None:
    if hardware != "apple-silicon-1" or len(inventories) != 1:
        return
    inventory = inventories[0]
    total_bytes = apple_memory_bytes(inventory)
    plan["hardware_admission"] = {
        "physical_unified_memory_bytes": total_bytes,
        "rule": "static capacity is only a prefilter; require measured Metal/MLX/process peak, OS reserve, zero swap, and target context/concurrency",
    }
    if total_bytes is None:
        return
    if plan.get("model") == "Qwen/Qwen3.8-27B":
        if total_bytes < 32 * 1024**3:
            reason = (
                "The focused Qwen oQ4e artifact is about 15.8 GiB before runtime, KV/cache, Metal allocations, and the macOS reserve; "
                "a sub-32-GB Mac is rejected for this candidate set. A materially smaller GGUF is a distinct artifact and scope."
            )
            plan["status"] = "focused Qwen candidate set infeasible on this unified-memory capacity"
            plan["phase_scope"] = "stopped at Apple unified-memory capacity gate"
            plan["hardware_admission"].update({"status": "rejected", "reason": reason})
            for candidate in plan["candidates"]:
                candidate["rejected"] = True
                candidate["stop_reason"] = reason
        elif total_bytes < 48 * 1024**3:
            plan["hardware_admission"].update(
                {
                    "status": "conditional-short-context-single-stream",
                    "reason": "32 GB is an admission-test lane only; require measured load headroom, short-context residency, and zero swap before benchmarking.",
                }
            )
            if plan.get("objective") == "single-stream":
                plan["status"] = "short-context capacity admission required; performance candidates are blocked"
                plan["phase_scope"] = "framework-native load/context admission only"
                for candidate in plan["candidates"]:
                    candidate["blocked_pending_capacity_admission"] = True
            else:
                reason = "A 32 GB Mac is not admitted for the bundled aggregate/goodput matrix through concurrency 16."
                plan["status"] = "serving objective infeasible for the bundled Qwen workload"
                plan["phase_scope"] = "stopped at Apple serving-capacity gate"
                for candidate in plan["candidates"]:
                    candidate["rejected"] = True
                    candidate["stop_reason"] = reason
        elif total_bytes < 64 * 1024**3:
            plan["hardware_admission"].update(
                {
                    "status": "practical-short-context-single-stream",
                    "reason": "48 GB is a practical short-context single-stream starting point, not an automatic long-context or concurrency pass.",
                }
            )
            if plan.get("objective") == "single-stream":
                plan["status"] = "short-context capacity admission required; bundled 32K performance case is blocked"
                plan["phase_scope"] = "framework-native load/context admission only"
                for candidate in plan["candidates"]:
                    candidate["blocked_pending_capacity_admission"] = True
            else:
                reason = "A 48 GB Mac is not admitted for the bundled aggregate/goodput matrix through concurrency 16."
                plan["status"] = "serving objective infeasible for the bundled Qwen workload"
                plan["phase_scope"] = "stopped at Apple serving-capacity gate"
                for candidate in plan["candidates"]:
                    candidate["rejected"] = True
                    candidate["stop_reason"] = reason
        else:
            plan["hardware_admission"].update(
                {
                    "status": "capacity-plausible-runtime-gate",
                    "reason": "Static capacity is plausible; exact context, concurrency, cache, Metal peak, macOS reserve, and zero-swap gates remain.",
                }
            )
        return
    if plan.get("model") != "deepseek-ai/DeepSeek-V4-Flash-0731":
        return
    if total_bytes < 192 * 1024**3:
        reason = "The roughly 167 GB official repository plus runtime/cache/OS reserve cannot fit this Apple unified-memory capacity."
        plan["hardware_admission"].update({"status": "official-checkpoint-rejected", "reason": reason})
        plan["artifact_boundary"]["official_checkpoint"] = {"status": "rejected", "reason": reason}
        plan["artifact_boundary"]["requested_official_artifact_plan"] = "stop-before-download-or-launch"
        plan["artifact_boundary"]["alternative_artifact_scope"] = "separate quality and performance contracts"
        plan["scope_decision"]["alternative_artifacts_require_explicit_acceptance"] = True
        plan["scope_decision"]["alternative_artifacts_accepted"] = allow_alternative_artifacts
        plan["status"] = (
            "official checkpoint infeasible; user explicitly accepted evaluation of distinct alternative artifacts"
            if allow_alternative_artifacts
            else "official checkpoint infeasible; alternative artifacts are blocked pending explicit scope acceptance"
        )
        plan["phase_scope"] = (
            "explicitly accepted alternative artifacts only"
            if allow_alternative_artifacts
            else "stopped at official capacity gate; alternatives listed but blocked"
        )
        for candidate in plan["candidates"]:
            if candidate.get("artifact_class") == "official-checkpoint":
                candidate["rejected"] = True
                candidate["stop_reason"] = reason
            else:
                candidate["requires_scope_expansion"] = True
                candidate["blocked_pending_scope_acceptance"] = not allow_alternative_artifacts
    elif total_bytes < 256 * 1024**3:
        plan["hardware_admission"].update(
            {
                "status": "conditional-tight-capacity",
                "reason": "A 192 GB-class Mac is a tight measured-admission lane after the roughly 167 GB repository; no automatic pass.",
            }
        )
        plan["artifact_boundary"]["official_checkpoint"] = {
            "status": "conditional-tight-capacity",
            "reason": "A 192 GB-class Mac is a tight measured-admission lane after the roughly 167 GB repository; no automatic pass.",
        }
    else:
        plan["hardware_admission"].update(
            {
                "status": "capacity-plausible-runtime-gate",
                "reason": "Installed memory clears the storage prefilter; exact MLX/Metal loader, peak residency, context, reserve, and zero-swap gates remain.",
            }
        )
        plan["artifact_boundary"]["official_checkpoint"] = {
            "status": "capacity-plausible-runtime-gate",
            "reason": "Installed memory clears the storage prefilter; exact MLX/Metal loader, peak residency, context, reserve, and zero-swap gates remain.",
        }


def build_plan(
    model: str,
    hardware: str,
    objective: str | None = None,
    objective_source: str | None = None,
    allow_alternative_artifacts: bool = False,
) -> dict[str, Any]:
    if model not in MODEL_ALIASES:
        raise ValueError(f"unknown model: {model}")
    if hardware not in HARDWARE_PROFILES:
        raise ValueError(f"unknown hardware profile: {hardware}")
    objective_was_missing = objective is None
    if objective_was_missing:
        objective = "single-stream"
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if objective_source is None:
        objective_source = "assumed-default" if objective_was_missing else "explicit"

    model_id = MODEL_ALIASES[model]
    candidates = qwen_candidates(hardware) if model_id.startswith("Qwen/") else deepseek_candidates(hardware)
    official_feasibility = official_checkpoint_feasibility(model_id, hardware)
    official_rejected = official_feasibility["status"] == "rejected"
    if official_rejected:
        for candidate in candidates:
            if candidate.get("artifact_class") != "official-checkpoint":
                candidate["requires_scope_expansion"] = True
                candidate["blocked_pending_scope_acceptance"] = not allow_alternative_artifacts
    plan = {
        "schema_version": "1.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": (
            "official checkpoint infeasible; alternative artifacts are blocked pending explicit scope acceptance"
            if official_rejected and not allow_alternative_artifacts
            else "official checkpoint infeasible; user explicitly accepted evaluation of distinct alternative artifacts"
            if official_rejected
            else "starting hypotheses, not a performance verdict"
        ),
        "model": model_id,
        "hardware_profile": hardware,
        "objective": objective,
        "objective_source": objective_source,
        "assumptions": [
            {
                "id": "objective-default",
                "active": objective_was_missing,
                "value": "single-stream",
                "reason": "C1 is only a representative default; replace it with aggregate or goodput when that is the deployment goal.",
            },
            {
                "id": "synthetic-workload",
                "active": True,
                "value": "2K/8K/32K input and 512 output tokens, unique prefixes, declared reasoning lanes",
                "reason": "Replace with a measured production trace before a final recommendation.",
            },
        ],
        "artifact_boundary": {
            "official_checkpoint": official_feasibility,
            "requested_official_artifact_plan": "stop-before-download-or-launch" if official_rejected else "continue-through-admission-gates",
            "alternative_artifact_scope": "separate quality and performance contracts" if official_rejected else "only when explicitly selected",
            "ranking_rule": "Only rank runs sharing one exact comparison contract; otherwise report a disclosed best-achievable Pareto comparison.",
        },
        "scope_decision": {
            "requested_model_contract": "official checkpoint",
            "alternative_artifacts_require_explicit_acceptance": official_rejected,
            "alternative_artifacts_accepted": allow_alternative_artifacts if official_rejected else False,
            "cli_gate": "--allow-alternative-artifacts",
        },
        "required_inputs": [
            "hardware inventory from every node",
            "model and tokenizer revisions",
            "container digest and framework commit",
            "real workload trace or declared synthetic workload",
            "quality tolerance and API feature requirements",
        ],
        "workload": workload(objective),
        "platform_policy": platform_policy(hardware),
        "candidates": candidates,
        "phase_scope": (
            "stopped at official capacity gate; alternatives listed but blocked"
            if official_rejected and not allow_alternative_artifacts
            else "explicitly accepted alternative artifacts only"
            if official_rejected
            else "requested artifact and explicitly listed controls"
        ),
        "phases": [
            "compatibility and capacity gates",
            "no-speculation correctness baseline",
            "same-client common-denominator benchmark",
            "one-family-at-a-time tuning",
            "best-achievable benchmark with all differences disclosed",
            "30-60 minute stability and telemetry run",
        ],
        "invariants": [
            "Never compare different model artifacts as quality-equivalent without evaluation.",
            "Never count SSE chunks or requested output length as generated tokens.",
            "Never call a source-derived recipe the best stack before a common local benchmark.",
            "Stop on corruption, parser regression, swap, silent fallback, OOM, or nonstationary thermals.",
        ],
    }
    validate_plan_schema(plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--hardware-profile", choices=HARDWARE_PROFILES, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, help="defaults to an explicitly labeled single-stream assumption")
    parser.add_argument("--hardware-inventory", action="append", default=[], help="JSON emitted by collect_hardware.py; repeat once per DGX Spark node")
    parser.add_argument("--allow-alternative-artifacts", action="store_true", help="explicitly accept evaluating transformed/converted artifacts after the official checkpoint is rejected")
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout")
    args = parser.parse_args()
    if args.hardware_profile == "apple-silicon-1" and not args.hardware_inventory:
        parser.error("apple-silicon-1 requires one collect_hardware.py inventory; chip/bin, memory, OS, power, and Metal identity are not optional")
    plan = build_plan(
        args.model,
        args.hardware_profile,
        args.objective,
        allow_alternative_artifacts=args.allow_alternative_artifacts,
    )
    if args.hardware_inventory:
        attached = []
        inventories = []
        for inventory_name in args.hardware_inventory:
            inventory_path = Path(inventory_name)
            raw_inventory = inventory_path.read_bytes()
            inventory = json.loads(raw_inventory)
            if inventory.get("schema_version") != "1.0" or "host" not in inventory or "probes" not in inventory:
                parser.error("hardware inventory is not a supported collect_hardware.py document")
            inventories.append(inventory)
            attached_inventory = {
                "sha256": hashlib.sha256(raw_inventory).hexdigest(),
                "collected_at_utc": inventory.get("collected_at_utc"),
                "host_machine": (inventory.get("host") or {}).get("machine"),
                "validation": inventory.get("validation"),
            }
            if args.hardware_profile == "apple-silicon-1":
                profile = (((inventory.get("apple_silicon") or {}).get("system_profile") or {}).get("profile") or {})
                attached_inventory["apple_profile_allowlisted"] = profile
                attached_inventory["apple_python_packages"] = (inventory.get("apple_silicon") or {}).get("python_packages") or {}
            attached.append(attached_inventory)
        try:
            validate_hardware_inventories(args.hardware_profile, inventories)
        except ValueError as exc:
            parser.error(str(exc))
        apply_inventory_admission(
            plan,
            args.hardware_profile,
            inventories,
            args.allow_alternative_artifacts,
        )
        plan["hardware_inventories"] = attached
    rendered = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        print(rendered, end="")
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
