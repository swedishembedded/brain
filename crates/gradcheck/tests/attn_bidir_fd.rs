// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated finite-difference gradient checks for the bidirectional (encoder
//! self-attention) attention kernel family (ADR 0001 §5.1, PR-7).
//!
//! These tests do NOT build any model: they drive the WGSL kernels directly via
//! `gpu_core`, exactly as the model code will. The forward pipeline is
//!   scores_bidir -> softmax_bidir -> apply_bidir -> out [B*T, d]
//! and we define a scalar loss  L = sum(out .* g)  for a fixed random upstream
//! grad `g` (so dL/dout = g). The backward pipeline
//!   dscores_bidir -> {dq,dk,dv}_bidir into d_qkv
//! must then equal dL/d(qkv). We FD-check each qkv entry (q/k/v regions cover the
//! dq/dk/dv kernels respectively) against the analytic d_qkv.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use gpu_core::Gpu;

// Kernel order passed to Gpu::new; indices below reference these.
static KERNELS: &[(&str, &str)] = &[
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),       // 0
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),     // 1
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),         // 2
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR), // 3
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),       // 4
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),       // 5
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),       // 6
];

struct Shape {
    b: u32,
    h: u32,
    t: u32,
    hd: u32,
}
impl Shape {
    fn d(&self) -> u32 {
        self.h * self.hd
    }
    fn qkv_len(&self) -> usize {
        (self.b * self.t * 3 * self.d()) as usize
    }
    fn out_len(&self) -> usize {
        (self.b * self.t * self.d()) as usize
    }
    fn scores_len(&self) -> usize {
        (self.b * self.h * self.t * self.t) as usize
    }
}

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~[-1,1)
}

/// Run forward (scores->softmax->apply) for a given qkv vector, return out.
fn forward(gpu: &Gpu, s: &Shape, qkv: &[f32]) -> Vec<f32> {
    let d = s.d();
    let qkv_buf = gpu.storage_init("qkv", qkv);
    let scores = gpu.storage(s.scores_len() as u64);
    let probs = gpu.storage(s.scores_len() as u64);
    let out = gpu.storage(s.out_len() as u64);

    // scores: params {bsz,n_heads,T,head_dim,qkv_stride,q_off,k_off}; bufs [qkv, scores]
    let p_scores = [s.b, s.h, s.t, s.hd, 3 * d, 0, d];
    let st0 = gpu.step(0, &[&qkv_buf, &scores], &p_scores, s.scores_len() as u32);
    // softmax: params {bsz,n_heads,T}; bufs [scores, probs]
    let p_soft = [s.b, s.h, s.t];
    let st1 = gpu.step(1, &[&scores, &probs], &p_soft, s.b * s.h * s.t);
    // apply: params {bsz,n_heads,T,head_dim,qkv_stride,v_off,d_model}; bufs [probs, qkv, out]
    let p_apply = [s.b, s.h, s.t, s.hd, 3 * d, 2 * d, d];
    let st2 = gpu.step(2, &[&probs, &qkv_buf, &out], &p_apply, s.out_len() as u32);

    gpu.submit(&[], &[st0, st1, st2]);
    gpu.poll_wait();
    gpu.read(&out, s.out_len())
}

/// Run backward, return d_qkv (grad of L=sum(out.*g) wrt qkv).
fn backward(gpu: &Gpu, s: &Shape, qkv: &[f32], g: &[f32]) -> Vec<f32> {
    let d = s.d();
    let qkv_buf = gpu.storage_init("qkv", qkv);
    let scores = gpu.storage(s.scores_len() as u64);
    let probs = gpu.storage(s.scores_len() as u64);
    let d_out = gpu.storage_init("d_out", g); // dL/dout = g
    let d_scores = gpu.storage(s.scores_len() as u64);
    let d_qkv = gpu.storage(s.qkv_len() as u64);

    // recompute probs (forward of scores+softmax)
    let p_scores = [s.b, s.h, s.t, s.hd, 3 * d, 0, d];
    let st0 = gpu.step(0, &[&qkv_buf, &scores], &p_scores, s.scores_len() as u32);
    let p_soft = [s.b, s.h, s.t];
    let st1 = gpu.step(1, &[&scores, &probs], &p_soft, s.b * s.h * s.t);

    // dscores: params {bsz,n_heads,T,head_dim,qkv_stride,v_off,d_model}
    //          bufs [d_out, qkv, probs, d_scores]
    let p_dsc = [s.b, s.h, s.t, s.hd, 3 * d, 2 * d, d];
    let st2 = gpu.step(3, &[&d_out, &qkv_buf, &probs, &d_scores], &p_dsc, s.b * s.h * s.t);
    // dv: params {bsz,n_heads,T,head_dim,qkv_stride,v_off,d_model}; bufs [probs, d_out, d_qkv]
    let p_dv = [s.b, s.h, s.t, s.hd, 3 * d, 2 * d, d];
    let st3 = gpu.step(4, &[&probs, &d_out, &d_qkv], &p_dv, s.out_len() as u32);
    // dq: params {bsz,n_heads,T,head_dim,qkv_stride,q_off,k_off}; bufs [d_scores, qkv, d_qkv]
    let p_dq = [s.b, s.h, s.t, s.hd, 3 * d, 0, d];
    let st4 = gpu.step(5, &[&d_scores, &qkv_buf, &d_qkv], &p_dq, s.out_len() as u32);
    // dk: same params as dq; bufs [d_scores, qkv, d_qkv]
    let st5 = gpu.step(6, &[&d_scores, &qkv_buf, &d_qkv], &p_dq, s.out_len() as u32);

    gpu.submit(&[], &[st0, st1, st2, st3, st4, st5]);
    gpu.poll_wait();
    gpu.read(&d_qkv, s.qkv_len())
}

fn loss(out: &[f32], g: &[f32]) -> f32 {
    out.iter().zip(g).map(|(&o, &gi)| o * gi).sum()
}

#[test]
fn bidir_forward_is_deterministic() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let s = Shape { b: 2, h: 2, t: 5, hd: 4 };
    let mut st = 0x1234_5678u64;
    let qkv: Vec<f32> = (0..s.qkv_len()).map(|_| lcg(&mut st)).collect();
    let a = forward(&gpu, &s, &qkv);
    let b = forward(&gpu, &s, &qkv);
    assert_eq!(a, b, "bidir forward not deterministic");
    // softmax rows sum to 1 (non-causal: full row). Spot-check out is finite.
    assert!(a.iter().all(|x| x.is_finite()));
}

#[test]
fn bidir_softmax_rows_sum_to_one() {
    // Directly verify softmax_bidir normalises over the FULL row (non-causal).
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let s = Shape { b: 1, h: 1, t: 6, hd: 3 };
    let d = s.d();
    let mut st = 0xABCDu64;
    let qkv: Vec<f32> = (0..s.qkv_len()).map(|_| lcg(&mut st)).collect();
    let qkv_buf = gpu.storage_init("qkv", &qkv);
    let scores = gpu.storage(s.scores_len() as u64);
    let probs = gpu.storage(s.scores_len() as u64);
    let p_scores = [s.b, s.h, s.t, s.hd, 3 * d, 0, d];
    let st0 = gpu.step(0, &[&qkv_buf, &scores], &p_scores, s.scores_len() as u32);
    let st1 = gpu.step(1, &[&scores, &probs], &[s.b, s.h, s.t], s.b * s.h * s.t);
    gpu.submit(&[], &[st0, st1]);
    gpu.poll_wait();
    let pr = gpu.read(&probs, s.scores_len());
    let t = s.t as usize;
    for i in 0..t {
        let row: f32 = (0..t).map(|j| pr[i * t + j]).sum();
        assert!((row - 1.0).abs() < 1e-4, "row {i} sums to {row}, expected 1 (full non-causal row)");
    }
}

#[test]
fn bidir_backward_matches_finite_differences() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let s = Shape { b: 2, h: 2, t: 4, hd: 3 };
    let mut st = 0xDEAD_BEEFu64;
    let qkv: Vec<f32> = (0..s.qkv_len()).map(|_| lcg(&mut st)).collect();
    let g: Vec<f32> = (0..s.out_len()).map(|_| lcg(&mut st)).collect();

    let analytic = backward(&gpu, &s, &qkv, &g);

    let eps = 1e-3f32;
    let d = s.d() as usize;
    let region = |i: usize| match (i % (3 * d)) / d {
        0 => "dq",
        1 => "dk",
        _ => "dv",
    };
    let mut max_err = [0f32; 3]; // dq, dk, dv

    for i in 0..qkv.len() {
        let mut wp = qkv.clone();
        wp[i] += eps;
        let lp = loss(&forward(&gpu, &s, &wp), &g);
        let mut wm = qkv.clone();
        wm[i] -= eps;
        let lm = loss(&forward(&gpu, &s, &wm), &g);
        let num = (lp - lm) / (2.0 * eps);
        let ana = analytic[i];
        let abs_err = (num - ana).abs();
        let slot = match region(i) {
            "dq" => 0,
            "dk" => 1,
            _ => 2,
        };
        max_err[slot] = max_err[slot].max(abs_err);
    }
    println!(
        "bidir FD max abs err: dq={:.3e} dk={:.3e} dv={:.3e}",
        max_err[0], max_err[1], max_err[2]
    );
    let tol = 2e-2f32;
    assert!(max_err[0] < tol, "dq_bidir FD err {} >= {tol}", max_err[0]);
    assert!(max_err[1] < tol, "dk_bidir FD err {} >= {tol}", max_err[1]);
    assert!(max_err[2] < tol, "dv_bidir FD err {} >= {tol}", max_err[2]);
}
