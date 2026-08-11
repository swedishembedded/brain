#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Cross-backend parity gate (#10): CPU == Vulkan == NPU.
#
#   1. The gradcheck suite passes on the CPU backend (analytic grads == finite
#      differences) — CPU is correct.
#   2. The same suite passes on the Vulkan/GPU backend — GPU is correct, i.e.
#      CPU == Vulkan for the training path. This run also includes the direct
#      cross-backend forward-logit check (tests/backend_parity.rs, which compares
#      CPU vs GPU logits in one process) and the device-local codec/MSE tests.
#   3. The TTS NPU codec path matches the CPU reference (npu_stream_matches_cpu) —
#      NPU == CPU for inference. Runs on the OpenVINO CPU device by default (set
#      BRAIN_TTS_NPU_DEVICE=npu to exercise the real NPU); skipped without codec
#      weights.
#
# Usage:  scripts/gates/parity-gate.sh
set -u
cd "$(dirname "$0")/../.."

fail=0
run() {
    local desc="$1"; shift
    echo "=== $desc ==="
    if "$@"; then echo "  PASS: $desc"; else echo "  FAIL: $desc"; fail=1; fi
    echo
}

run "gradcheck suite — CPU backend"    env BRAIN_DEVICE=cpu    cargo test --release -q -p brain-gradcheck
run "gradcheck suite — Vulkan backend (incl. CPU==GPU forward parity)" \
                                       env BRAIN_DEVICE=vulkan cargo test --release -q -p brain-gradcheck

# brain-model's FD suites (MoE block backward, ViT block backward) use the
# pooled test device, but lived OUTSIDE this gate's package scope — so the
# MoE/ViT backward ran on whichever backend the developer's box picked, never
# provably on both. Same discipline, both backends:
# backend-specific silent-zero gradients are exactly what one-backend FD runs
# cannot see.
run "model FD suites (MoE/ViT backward) — CPU backend" \
    env BRAIN_DEVICE=cpu    cargo test --release -q -p brain-model --test moe_block_gradcheck --test vit_block_gradcheck --test moe_sparse_bwd_parity
run "model FD suites (MoE/ViT backward) — Vulkan backend" \
    env BRAIN_DEVICE=vulkan cargo test --release -q -p brain-model --test moe_block_gradcheck --test vit_block_gradcheck --test moe_sparse_bwd_parity

# REMAINING (audit F15, models batch): crates/moondream, crates/nemotron and
# crates/vision pin their FD suites to `Gpu::new_cpu` in the test code itself,
# so adding them here would not exercise a second backend until those call
# sites move to `gpu_core::testgpu::dev`.

# int8 paged KV is the serving default (qwen::serve): translation to the CPU
# JIT is NOT execution, and the serving perf gate itself
# runs on BRAIN_DEVICE=cpu, so this must be green before the default flips.
run "qwen serve suite — CPU backend (int8 KV is the serving default)" \
                                       env BRAIN_DEVICE=cpu    cargo test --release -q -p brain-qwen --lib serve::

codec="${BRAIN_CODEC_WEIGHTS:-$PWD/out/tts-1b7/codec.weights}"
if [ -f "$codec" ]; then
    run "TTS codec: NPU graph == CPU reference" \
        env BRAIN_CODEC_WEIGHTS="$codec" cargo test --release -q -p brain-tts npu_stream_matches_cpu -- --ignored
else
    echo "=== TTS NPU==CPU: SKIP (no codec weights at $codec) ==="; echo
fi

if [ "$fail" -eq 0 ]; then
    echo "PARITY GATE: PASS (CPU == Vulkan == NPU)"
else
    echo "PARITY GATE: FAIL"
fi
exit "$fail"
