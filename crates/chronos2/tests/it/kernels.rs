// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolation tests for the two Chronos-2-specific WGSL kernels, dispatched on
//! tiny inputs via the CPU backend (so they run headless) and checked against
//! hand-computed values. This validates the parity-critical kernel math —
//! **half-split RoPE** and **unscaled, non-causal attention scores** — without
//! the full model or any weights.

use gpu_core::{f, Gpu};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

#[test]
fn rope_neox_rotates_the_half_split_pair() {
    if skip() {
        return;
    }
    // two tokens; token 1 (t=1) carries q=[a0,a1,a2,a3]=[1,2,3,4]; one head,
    // head_dim=4 -> half=2, pairs (0,2) and (1,3). theta=10000.
    // freq_j = theta^(-2j/4): j=0 -> 1.0, j=1 -> 10000^-0.5 = 0.01.
    // angle_j at t=1 = freq_j.
    // out[j]   = a[j]*cos(angle_j) - a[j+2]*sin(angle_j)
    // out[j+2] = a[j+2]*cos(angle_j) + a[j]*sin(angle_j)
    let gpu = Gpu::new_cpu(&[("rope_neox", kernels::ROPE_NEOX)]);
    let q = gpu.storage_init("q", &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    // params: seq_len, n_heads, head_dim, row_stride, base_off, theta
    let step = gpu.step(0, &[&q], &[2, 1, 4, 4, 0, f(10000.0)], 2 * 2);
    gpu.submit(&[], &[step]);
    let out = gpu.read(&q, 8);
    let tok1 = &out[4..8];

    let (c0, s0) = (1.0f32.cos(), 1.0f32.sin()); // freq j=0 = 1.0
    let f1 = (10000.0f32).powf(-0.5);
    let (c1, s1) = (f1.cos(), f1.sin());
    let expect = [
        1.0 * c0 - 3.0 * s0, // out[0] = a0*c0 - a2*s0
        2.0 * c1 - 4.0 * s1, // out[1] = a1*c1 - a3*s1
        3.0 * c0 + 1.0 * s0, // out[2] = a2*c0 + a0*s0
        4.0 * c1 + 2.0 * s1, // out[3] = a3*c1 + a1*s1
    ];
    for i in 0..4 {
        assert!((tok1[i] - expect[i]).abs() < 1e-5, "rope[{i}] {} vs {}", tok1[i], expect[i]);
    }
    // token 0 (t=0) stays identity (zeros)
    assert_eq!(&out[0..4], &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn rope_neox_is_identity_at_t0() {
    if skip() {
        return;
    }
    // at t=0 the angle is 0 for every pair -> the buffer is unchanged.
    let gpu = Gpu::new_cpu(&[("rope_neox", kernels::ROPE_NEOX)]);
    let q = gpu.storage_init("q", &[5.0, -1.0, 2.0, 7.0]);
    // seq_len=1 but token index t is derived from position; use a 2-token buffer
    // and check token 0 is untouched.
    let q2 = gpu.storage_init("q2", &[5.0, -1.0, 2.0, 7.0, 9.0, 9.0, 9.0, 9.0]);
    let step = gpu.step(0, &[&q2], &[2, 1, 4, 4, 0, f(10000.0)], 2 * 2);
    gpu.submit(&[], &[step]);
    let out = gpu.read(&q2, 8);
    // token 0 (t=0) unchanged
    assert_eq!(&out[0..4], &[5.0, -1.0, 2.0, 7.0]);
    let _ = q;
}

#[test]
fn attn_scores_full_is_unscaled_with_additive_mask() {
    if skip() {
        return;
    }
    // bsz=1, n_heads=1, S=2, head_dim=2, qk_stride=2.
    // q = [[1,0],[0,1]] (rows i=0,1); k = [[1,1],[2,0]] (rows j=0,1).
    // raw scores[i,j] = q_i . k_j (UNSCALED):
    //   [0,0]=1*1+0*1=1  [0,1]=1*2+0*0=2
    //   [1,0]=0*1+1*1=1  [1,1]=0*2+1*0=0
    // key mask = [0, -10] (mask key j=1) -> add to column j.
    //   final: [1, 2-10=-8; 1, 0-10=-10]
    let gpu = Gpu::new_cpu(&[("attn_scores_full", kernels::ATTN_SCORES_FULL)]);
    let q = gpu.storage_init("q", &[1.0, 0.0, 0.0, 1.0]);
    let k = gpu.storage_init("k", &[1.0, 1.0, 2.0, 0.0]);
    let mask = gpu.storage_init("mask", &[0.0, -10.0]);
    let scores = gpu.storage(4);
    // params: bsz, n_heads, tcols(S), head_dim, qk_stride
    let step = gpu.step(0, &[&q, &k, &mask, &scores], &[1, 1, 2, 2, 2], 2 * 2);
    gpu.submit(&[], &[step]);
    let out = gpu.read(&scores, 4);
    // layout ((b*H+h)*S+i)*S+j -> [i=0,j=0],[i=0,j=1],[i=1,j=0],[i=1,j=1]
    assert!((out[0] - 1.0).abs() < 1e-6, "{out:?}");
    assert!((out[1] - (-8.0)).abs() < 1e-6, "{out:?}");
    assert!((out[2] - 1.0).abs() < 1e-6, "{out:?}");
    assert!((out[3] - (-10.0)).abs() < 1e-6, "{out:?}");
}

#[test]
fn attn_scores_full_is_not_causal() {
    if skip() {
        return;
    }
    // Unlike attn_scores.wgsl, position (i=0, j=1) is a real dot product, not
    // -inf: query 0 attends to key 1. Verified above (out[1] = -8, finite),
    // here with no mask so the upper triangle is plainly non-zero.
    let gpu = Gpu::new_cpu(&[("attn_scores_full", kernels::ATTN_SCORES_FULL)]);
    let q = gpu.storage_init("q", &[1.0, 1.0, 1.0, 1.0]);
    let k = gpu.storage_init("k", &[1.0, 0.0, 0.0, 1.0]);
    let mask = gpu.storage_init("mask", &[0.0, 0.0]);
    let scores = gpu.storage(4);
    let step = gpu.step(0, &[&q, &k, &mask, &scores], &[1, 1, 2, 2, 2], 4);
    gpu.submit(&[], &[step]);
    let out = gpu.read(&scores, 4);
    // [i=0,j=1] = q0.k1 = [1,1].[0,1] = 1  (would be -inf under causal masking)
    assert!((out[1] - 1.0).abs() < 1e-6, "upper triangle must be attended: {out:?}");
}

#[test]
fn attn_softmax_full_normalises_over_all_keys() {
    if skip() {
        return;
    }
    // one (b,h), S=2. scores row i=0 = [0, ln4] -> softmax = [1/5, 4/5].
    // (exp(0)=1, exp(ln4)=4, sum=5). Row i=1 = [ln4, 0] -> [4/5, 1/5].
    let gpu = Gpu::new_cpu(&[("attn_softmax_full", kernels::ATTN_SOFTMAX_FULL)]);
    let ln4 = 4.0f32.ln();
    let scores = gpu.storage_init("scores", &[0.0, ln4, ln4, 0.0]);
    let probs = gpu.storage(4);
    let step = gpu.step(0, &[&scores, &probs], &[1, 1, 2], 2);
    gpu.submit(&[], &[step]);
    let out = gpu.read(&probs, 4);
    assert!((out[0] - 0.2).abs() < 1e-5, "{out:?}");
    assert!((out[1] - 0.8).abs() < 1e-5, "{out:?}");
    assert!((out[2] - 0.8).abs() < 1e-5, "{out:?}");
    assert!((out[3] - 0.2).abs() < 1e-5, "{out:?}");
}

#[test]
fn attn_apply_full_weights_values_over_all_keys() {
    if skip() {
        return;
    }
    // S=2, head_dim=2, v_stride=2, d_model=2. probs row0=[0.2,0.8], row1=[0.8,0.2].
    // v = [[1,10],[3,30]] (rows j=0,1).
    // out[0] = 0.2*[1,10] + 0.8*[3,30] = [2.6, 26]
    // out[1] = 0.8*[1,10] + 0.2*[3,30] = [1.4, 14]
    let gpu = Gpu::new_cpu(&[("attn_apply_full", kernels::ATTN_APPLY_FULL)]);
    let probs = gpu.storage_init("probs", &[0.2, 0.8, 0.8, 0.2]);
    let v = gpu.storage_init("v", &[1.0, 10.0, 3.0, 30.0]);
    let out = gpu.storage(4);
    // params: bsz, n_heads, tcols, head_dim, v_stride, d_model
    let step = gpu.step(0, &[&probs, &v, &out], &[1, 1, 2, 2, 2, 2], 2 * 2);
    gpu.submit(&[], &[step]);
    let o = gpu.read(&out, 4);
    assert!((o[0] - 2.6).abs() < 1e-5, "{o:?}");
    assert!((o[1] - 26.0).abs() < 1e-4, "{o:?}");
    assert!((o[2] - 1.4).abs() < 1e-5, "{o:?}");
    assert!((o[3] - 14.0).abs() < 1e-4, "{o:?}");
}

#[test]
fn full_attention_pipeline_composes_end_to_end() {
    if skip() {
        return;
    }
    // Compose scores -> softmax -> apply for a self-attention with S=2, 1 head,
    // head_dim=2, no mask. Verify against a hand-computed reference.
    let gpu = Gpu::new_cpu(&[
        ("attn_scores_full", kernels::ATTN_SCORES_FULL),
        ("attn_softmax_full", kernels::ATTN_SOFTMAX_FULL),
        ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ]);
    // q = k = [[1,0],[0,1]], v = [[2,0],[0,3]]. mask 0.
    let q = gpu.storage_init("q", &[1.0, 0.0, 0.0, 1.0]);
    let k = gpu.storage_init("k", &[1.0, 0.0, 0.0, 1.0]);
    let v = gpu.storage_init("v", &[2.0, 0.0, 0.0, 3.0]);
    let mask = gpu.storage_init("mask", &[0.0, 0.0]);
    let scores = gpu.storage(4);
    let probs = gpu.storage(4);
    let out = gpu.storage(4);
    let s1 = gpu.step(0, &[&q, &k, &mask, &scores], &[1, 1, 2, 2, 2], 4);
    let s2 = gpu.step(1, &[&scores, &probs], &[1, 1, 2], 2);
    let s3 = gpu.step(2, &[&probs, &v, &out], &[1, 1, 2, 2, 2, 2], 4);
    gpu.submit(&[], &[s1, s2, s3]);
    let o = gpu.read(&out, 4);
    // scores: [[1,0],[0,1]]; softmax rows -> [[e/(e+1), 1/(e+1)], [1/(e+1), e/(e+1)]]
    let e = 1.0f32.exp();
    let (a, b) = (e / (e + 1.0), 1.0 / (e + 1.0));
    // out[0] = a*[2,0] + b*[0,3] = [2a, 3b]; out[1] = b*[2,0] + a*[0,3] = [2b, 3a]
    assert!((o[0] - 2.0 * a).abs() < 1e-5, "{o:?}");
    assert!((o[1] - 3.0 * b).abs() < 1e-5, "{o:?}");
    assert!((o[2] - 2.0 * b).abs() < 1e-5, "{o:?}");
    assert!((o[3] - 3.0 * a).abs() < 1e-5, "{o:?}");
}
