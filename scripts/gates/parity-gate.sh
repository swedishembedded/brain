#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Cross-backend parity gate (#10): CPU == Vulkan == CUDA == NPU, and CUDA's tuned tier.
#
#   1. The gradcheck suite passes on the CPU backend (analytic grads == finite
#      differences) — CPU is correct.
#   2. The same suite passes on the Vulkan/GPU backend — GPU is correct, i.e.
#      CPU == Vulkan for the training path. This run also includes the direct
#      cross-backend forward-logit check (tests/backend_parity.rs, which compares
#      CPU vs GPU logits in one process) and the device-local codec/MSE tests.
#   3. The TTS NPU codec path matches the CPU reference (npu_stream_matches_cpu) —
#      NPU == CPU for inference. Runs on the OpenVINO CPU device, over the model store's
#      Qwen3-TTS codec, with the OpenVINO runtime of the active virtualenv.
#   4. The same gradcheck suite and the model FD suites on the CUDA backend
#      (BRAIN_BACKEND=cuda on the card `cuda-device.sh` picks; BRAIN_DEVICE=gpuN
#      chooses it) - CPU == Vulkan == CUDA. Skipped with a message where no
#      NVIDIA device exists.
#   5. The CUDA backend's native (tuned) kernels agree with the WGSL
#      reference provider on the same device - CUDA's second implementation of
#      an operator == the portable one. Skips itself where no CUDA device is
#      present, which is most machines.
#
# Usage:  scripts/gates/parity-gate.sh
set -u
cd "$(dirname "$0")/../.."
source scripts/gates/cuda-device.sh

fail=0
run() {
    local desc="$1"; shift
    echo "=== $desc ==="
    if "$@"; then echo "  PASS: $desc"; else echo "  FAIL: $desc"; fail=1; fi
    echo
}
# A GPU leg cannot run until the device seam exists; it fails, never passes on the CPU in its place.
gpu_leg() {
    echo "=== $1 ==="
    echo "  FAIL: not run: a test binary is no longer steered to a GPU by BRAIN_DEVICE/BRAIN_BACKEND (an unconfigured process runs on the CPU); this leg waits for the test-harness device seam (the brain ledger, engineering cleanup EC-4 step 2)" >&2
    echo "  FAIL: $1"; fail=1; echo
}

run "gradcheck suite — CPU backend"    env BRAIN_DEVICE=cpu    cargo test --release -q -p brain-gradcheck
# Vulkan line, one deliberate difference from the CPU line (knowledge #300):
# --test-threads=1: concurrent tests on the one shared Vulkan device give
# wrong results (cross-group leaks) and ERROR_DEVICE_LOST, at the baseline
# commit too; serial execution is what makes the kernel check meaningful.
# The shared-device race itself is tracked in .agents/roadmap/. The bf16
# training gate runs here too since the Vulkan backend reports its bf16 storage
# tier (knowledge #558).
gpu_leg "gradcheck suite - Vulkan backend (incl. CPU==GPU forward parity)"

# CUDA line: the whole suite, no skips (three tests that failed here were fixed or shown to be properties of the check;
# see .agents/roadmap/cuda.md and knowledge 408).
CUDA_DEVICE=$(cuda_device)
if [ -n "$CUDA_DEVICE" ]; then
    gpu_leg "gradcheck suite - CUDA backend on $CUDA_DEVICE (incl. CPU==CUDA forward parity)"
else
    echo "=== gradcheck suite - CUDA backend: SKIP (no NVIDIA device on this machine) ==="; echo
fi

# brain-model's FD suites (MoE block backward, ViT block backward) use the
# pooled test device, but lived OUTSIDE this gate's package scope — so the
# MoE/ViT backward ran on whichever backend the developer's box picked, never
# provably on both. Same discipline, both backends:
# backend-specific silent-zero gradients are exactly what one-backend FD runs
# cannot see.
run "model FD suites (MoE/ViT backward) — CPU backend" \
    env BRAIN_DEVICE=cpu    cargo test --release -q -p brain-model --test moe_block_gradcheck --test vit_block_gradcheck --test moe_sparse_bwd_parity
gpu_leg "model FD suites (MoE/ViT backward) - Vulkan backend"

if [ -n "$CUDA_DEVICE" ]; then
    gpu_leg "model FD suites (MoE/ViT backward) - CUDA backend"
fi

# REMAINING (audit F15, models batch): crates/moondream3, crates/nemotronasr and
# crates/vision pin their FD suites to `Gpu::new_cpu` in the test code itself,
# so adding them here would not exercise a second backend until those call
# sites move to `gpu_core::testgpu::dev`.

# int8 paged KV is the serving default (qwen::serve): translation to the CPU
# JIT is NOT execution, and the serving perf gate itself
# runs on BRAIN_DEVICE=cpu, so this must be green before the default flips.
run "qwen serve suite — CPU backend (int8 KV is the serving default)" \
                                       env BRAIN_DEVICE=cpu    cargo test --release -q -p brain-qwen3 --lib serve::

# The CUDA backend's native kernels are a SECOND implementation of
# operators the WGSL catalogue already implements, so "CPU == Vulkan" is no
# longer the whole parity question on an NVIDIA box: a tuned kernel that
# disagrees with the reference produces plausible numbers everywhere else in
# this gate. These tests compare the tuned tier against the WGSL reference
# provider on the same device, in one process, and skip themselves silently
# where there is no CUDA device - so this line is green on every machine, and
# meaningful on the ones it can be.
#
# NOT the gradcheck package (the other five lines): that would demand the full
# kernel catalogue and the backward kernels, and this backend has neither. The
# tuned kernel is held to the reference by its own gate instead, which is where
# the measured speedup floor is asserted too.
if [ -n "$CUDA_DEVICE" ]; then
    # A device binary of gpu-core: its tests run only on the card they are told.
    run "CUDA tuned kernels == WGSL reference on $CUDA_DEVICE" \
        cargo test --release -q -p brain-gpu-core --test cuda_provider_matmul -- --device "$CUDA_DEVICE" --backend cuda
else
    echo "=== CUDA tuned kernels == WGSL reference: SKIP (no NVIDIA device on this machine) ==="; echo
fi

# The codec comes from the model store's Qwen3-TTS (the test says so when there is none); OpenVINO is loaded from
# the active virtualenv (`make environment/requirements`), and its absence fails the run with that instruction.
run "TTS codec: NPU graph == CPU reference" \
    cargo test --release -q -p brain-qwen3tts npu_stream_matches_cpu -- --ignored

if [ "$fail" -eq 0 ]; then
    echo "PARITY GATE: PASS (CPU == Vulkan${CUDA_DEVICE:+ == CUDA} == NPU == CUDA tuned)"
else
    echo "PARITY GATE: FAIL"
fi
exit "$fail"
