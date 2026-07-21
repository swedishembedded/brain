// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolation tests for the three Kronos-specific WGSL kernels, dispatched on
//! tiny inputs via the CPU backend and checked against hand-computed values —
//! validating the parity-critical math (BSQ sign-quantize, SwiGLU, scaled
//! optionally-causal attention scores) with no model or weights.

use gpu_core::{f, Gpu};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

#[test]
fn bsq_quantize_signs_and_scales() {
    if skip() {
        return;
    }
    // k=4 -> inv_sqrt_k = 0.5. sign(z>0)=+1 else -1.
    let gpu = Gpu::new_cpu(&[("bsq_quantize", kernels::BSQ_QUANTIZE)]);
    let z = gpu.storage_init("z", &[0.7, -0.2, 0.0, 3.0]);
    let step = gpu.step(0, &[&z], &[4, f(0.5)], 4);
    gpu.submit(&[], &[step]);
    let out = gpu.read(&z, 4);
    // 0.7>0->+1*0.5; -0.2->-1*0.5; 0.0 not >0 -> -1*0.5; 3.0->+0.5
    assert_eq!(out, vec![0.5, -0.5, -0.5, 0.5]);
}

#[test]
fn silu_gate_matches_swiglu() {
    if skip() {
        return;
    }
    // out = silu(a)*b, silu(x)=x*sigmoid(x).
    let gpu = Gpu::new_cpu(&[("silu_gate", kernels::SILU_GATE)]);
    let a = gpu.storage_init("a", &[1.0, -1.0, 0.0]);
    let b = gpu.storage_init("b", &[2.0, 4.0, 5.0]);
    let out = gpu.storage(3);
    let step = gpu.step(0, &[&a, &b, &out], &[3], 3);
    gpu.submit(&[], &[step]);
    let o = gpu.read(&out, 3);
    let silu = |x: f32| x / (1.0 + (-x).exp());
    assert!((o[0] - silu(1.0) * 2.0).abs() < 1e-5, "{o:?}");
    assert!((o[1] - silu(-1.0) * 4.0).abs() < 1e-5, "{o:?}");
    assert!((o[2] - 0.0).abs() < 1e-6, "silu(0)=0, {o:?}");
}

#[test]
fn attn_scores_qk_scales_and_masks_causally() {
    if skip() {
        return;
    }
    // S=2, 1 head, head_dim=2. q=[[1,0],[0,1]], k=[[2,0],[0,2]]. scale=0.5.
    // raw q_i.k_j: [0,0]=2 [0,1]=0 ; [1,0]=0 [1,1]=2. *0.5 -> 1,0,0,1.
    // causal: j>i masked (position [0,1]) -> -inf.
    let gpu = Gpu::new_cpu(&[("attn_scores_qk", kernels::ATTN_SCORES_QK)]);
    let q = gpu.storage_init("q", &[1.0, 0.0, 0.0, 1.0]);
    let k = gpu.storage_init("k", &[2.0, 0.0, 0.0, 2.0]);
    let scores = gpu.storage(4);
    // params: bsz, n_heads, S, head_dim, qk_stride, causal, scale
    let step = gpu.step(0, &[&q, &k, &scores], &[1, 1, 2, 2, 2, 1, f(0.5)], 4);
    gpu.submit(&[], &[step]);
    let s = gpu.read(&scores, 4);
    assert!((s[0] - 1.0).abs() < 1e-6, "{s:?}");
    assert!(s[1] < -1.0e38, "causal: [0,1] must be masked, {s:?}");
    assert!((s[2] - 0.0).abs() < 1e-6, "{s:?}");
    assert!((s[3] - 1.0).abs() < 1e-6, "{s:?}");
}

#[test]
fn attn_scores_qk_non_causal_keeps_upper_triangle() {
    if skip() {
        return;
    }
    let gpu = Gpu::new_cpu(&[("attn_scores_qk", kernels::ATTN_SCORES_QK)]);
    let q = gpu.storage_init("q", &[1.0, 1.0, 1.0, 1.0]);
    let k = gpu.storage_init("k", &[1.0, 0.0, 0.0, 1.0]);
    let scores = gpu.storage(4);
    // causal=0, scale=1 -> [0,1] = q0.k1 = [1,1].[0,1] = 1 (attended, not -inf)
    let step = gpu.step(0, &[&q, &k, &scores], &[1, 1, 2, 2, 2, 0, f(1.0)], 4);
    gpu.submit(&[], &[step]);
    let s = gpu.read(&scores, 4);
    assert!((s[1] - 1.0).abs() < 1e-6, "non-causal upper triangle attended: {s:?}");
}
