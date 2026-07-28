// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated finite-difference gradient checks for the cross-attention kernel
//! family (ADR 0001 §5.1, PR-8).
//!
//! Buffer layout (the choice this PR documents):
//!   * Q lives in a DECODER fused-QKV buffer `q_dec`  [B*T_dec, 3*d], q at off 0
//!     (stride 3*d, mirroring the existing fused-QKV attention kernels). Only the
//!     q region participates in cross-attention; k/v regions are unused here.
//!   * K and V live in an ENCODER-MEMORY fused-KV buffer `kv_enc` [B*T_enc, 2*d],
//!     k at off 0, v at off d (stride 2*d).
//! Two sequence lengths T_dec (queries) x T_enc (keys/values); non-causal.
//!
//! Forward:  scores_cross -> softmax_cross -> apply_cross -> out [B*T_dec, d].
//! Loss L = sum(out .* g) for a fixed random upstream grad g (so dL/dout = g).
//! Backward: dscores_cross -> dq_cross (into q_dec q region)
//!                          -> dk_cross, dv_cross (into kv_enc k/v regions).
//! The two-buffer grad split is the subtle part: we FD-check dq against the
//! decoder buffer and dk/dv against the encoder buffer SEPARATELY.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),           // 0
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),         // 1
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),             // 2
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS), // 3
    ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),           // 4
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),           // 5
    ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),           // 6
];

struct Shape {
    b: u32,
    h: u32,
    t_dec: u32,
    t_enc: u32,
    hd: u32,
}
impl Shape {
    fn d(&self) -> u32 {
        self.h * self.hd
    }
    fn qdec_len(&self) -> usize {
        (self.b * self.t_dec * 3 * self.d()) as usize
    }
    fn kvenc_len(&self) -> usize {
        (self.b * self.t_enc * 2 * self.d()) as usize
    }
    fn out_len(&self) -> usize {
        (self.b * self.t_dec * self.d()) as usize
    }
    fn scores_len(&self) -> usize {
        (self.b * self.h * self.t_dec * self.t_enc) as usize
    }
}

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn forward(gpu: &Gpu, s: &Shape, q_dec: &[f32], kv_enc: &[f32]) -> Vec<f32> {
    let d = s.d();
    let qb = gpu.storage_init("q_dec", q_dec);
    let kvb = gpu.storage_init("kv_enc", kv_enc);
    let scores = gpu.storage(s.scores_len() as u64);
    let probs = gpu.storage(s.scores_len() as u64);
    let out = gpu.storage(s.out_len() as u64);

    // scores: {bsz,n_heads,t_dec,t_enc,head_dim,q_stride,kv_stride,q_off,k_off}
    let p_sc = [s.b, s.h, s.t_dec, s.t_enc, s.hd, 3 * d, 2 * d, 0, 0];
    let st0 = gpu.step(0, &[&qb, &kvb, &scores], &p_sc, s.scores_len() as u32);
    // softmax: {bsz,n_heads,t_dec,t_enc}
    let p_sm = [s.b, s.h, s.t_dec, s.t_enc];
    let st1 = gpu.step(1, &[&scores, &probs], &p_sm, (s.b * s.h * s.t_dec) as u32);
    // apply: {bsz,n_heads,t_dec,t_enc,head_dim,kv_stride,v_off,d_model}
    let p_ap = [s.b, s.h, s.t_dec, s.t_enc, s.hd, 2 * d, d, d];
    let st2 = gpu.step(2, &[&probs, &kvb, &out], &p_ap, s.out_len() as u32);

    gpu.submit(&[], &[st0, st1, st2]);
    gpu.poll_wait();
    gpu.read(&out, s.out_len())
}

/// Returns (d_q_dec, d_kv_enc).
fn backward(gpu: &Gpu, s: &Shape, q_dec: &[f32], kv_enc: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let d = s.d();
    let qb = gpu.storage_init("q_dec", q_dec);
    let kvb = gpu.storage_init("kv_enc", kv_enc);
    let scores = gpu.storage(s.scores_len() as u64);
    let probs = gpu.storage(s.scores_len() as u64);
    let d_out = gpu.storage_init("d_out", g);
    let d_scores = gpu.storage(s.scores_len() as u64);
    let d_q = gpu.storage(s.qdec_len() as u64); // decoder grad
    let d_kv = gpu.storage(s.kvenc_len() as u64); // encoder grad

    // recompute probs
    let p_sc = [s.b, s.h, s.t_dec, s.t_enc, s.hd, 3 * d, 2 * d, 0, 0];
    let st0 = gpu.step(0, &[&qb, &kvb, &scores], &p_sc, s.scores_len() as u32);
    let p_sm = [s.b, s.h, s.t_dec, s.t_enc];
    let st1 = gpu.step(1, &[&scores, &probs], &p_sm, (s.b * s.h * s.t_dec) as u32);

    // dscores: {bsz,n_heads,t_dec,t_enc,head_dim,kv_stride,v_off,d_model}
    //          bufs [d_out, kv, probs, d_scores]
    let p_ds = [s.b, s.h, s.t_dec, s.t_enc, s.hd, 2 * d, d, d];
    let st2 = gpu.step(3, &[&d_out, &kvb, &probs, &d_scores], &p_ds, (s.b * s.h * s.t_dec) as u32);
    // dv: {bsz,n_heads,t_dec,t_enc,head_dim,kv_stride,v_off,d_model}; bufs [probs,d_out,d_kv]
    let st3 = gpu.step(4, &[&probs, &d_out, &d_kv], &p_ds, (s.b * s.h * s.t_enc * s.hd) as u32);
    // dq: {bsz,n_heads,t_dec,t_enc,head_dim,q_stride,kv_stride,q_off,k_off}; bufs [d_scores,kv,d_q]
    let p_qk = [s.b, s.h, s.t_dec, s.t_enc, s.hd, 3 * d, 2 * d, 0, 0];
    let st4 = gpu.step(5, &[&d_scores, &kvb, &d_q], &p_qk, (s.b * s.h * s.t_dec * s.hd) as u32);
    // dk: same params; bufs [d_scores, q, d_kv]
    let st5 = gpu.step(6, &[&d_scores, &qb, &d_kv], &p_qk, (s.b * s.h * s.t_enc * s.hd) as u32);

    gpu.submit(&[], &[st0, st1, st2, st3, st4, st5]);
    gpu.poll_wait();
    (gpu.read(&d_q, s.qdec_len()), gpu.read(&d_kv, s.kvenc_len()))
}

fn loss(out: &[f32], g: &[f32]) -> f32 {
    out.iter().zip(g).map(|(&o, &gi)| o * gi).sum()
}

#[test]
fn cross_forward_is_deterministic() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let s = Shape { b: 2, h: 2, t_dec: 4, t_enc: 6, hd: 4 };
    let mut st = 0x0CADu64;
    let q_dec: Vec<f32> = (0..s.qdec_len()).map(|_| lcg(&mut st)).collect();
    let kv_enc: Vec<f32> = (0..s.kvenc_len()).map(|_| lcg(&mut st)).collect();
    let a = forward(&gpu, &s, &q_dec, &kv_enc);
    let b = forward(&gpu, &s, &q_dec, &kv_enc);
    assert_eq!(a, b, "cross forward not deterministic");
    assert!(a.iter().all(|x| x.is_finite()));
}

#[test]
fn cross_softmax_rows_sum_to_one_over_t_enc() {
    // softmax_cross normalises over the T_enc key axis (row width = T_enc != T_dec).
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let s = Shape { b: 1, h: 1, t_dec: 3, t_enc: 5, hd: 3 };
    let d = s.d();
    let mut st = 0xF00Du64;
    let q_dec: Vec<f32> = (0..s.qdec_len()).map(|_| lcg(&mut st)).collect();
    let kv_enc: Vec<f32> = (0..s.kvenc_len()).map(|_| lcg(&mut st)).collect();
    let qb = gpu.storage_init("q_dec", &q_dec);
    let kvb = gpu.storage_init("kv_enc", &kv_enc);
    let scores = gpu.storage(s.scores_len() as u64);
    let probs = gpu.storage(s.scores_len() as u64);
    let p_sc = [s.b, s.h, s.t_dec, s.t_enc, s.hd, 3 * d, 2 * d, 0, 0];
    let st0 = gpu.step(0, &[&qb, &kvb, &scores], &p_sc, s.scores_len() as u32);
    let st1 = gpu.step(1, &[&scores, &probs], &[s.b, s.h, s.t_dec, s.t_enc], (s.b * s.h * s.t_dec) as u32);
    gpu.submit(&[], &[st0, st1]);
    gpu.poll_wait();
    let pr = gpu.read(&probs, s.scores_len());
    let (tq, tk) = (s.t_dec as usize, s.t_enc as usize);
    for i in 0..tq {
        let row: f32 = (0..tk).map(|j| pr[i * tk + j]).sum();
        assert!((row - 1.0).abs() < 1e-4, "query row {i} sums to {row}, expected 1 over T_enc");
    }
}

#[test]
fn cross_backward_matches_finite_differences() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let s = Shape { b: 2, h: 2, t_dec: 3, t_enc: 5, hd: 3 };
    let mut st = 0xC0FFEEu64;
    let q_dec: Vec<f32> = (0..s.qdec_len()).map(|_| lcg(&mut st)).collect();
    let kv_enc: Vec<f32> = (0..s.kvenc_len()).map(|_| lcg(&mut st)).collect();
    let g: Vec<f32> = (0..s.out_len()).map(|_| lcg(&mut st)).collect();

    let (d_q, d_kv) = backward(&gpu, &s, &q_dec, &kv_enc, &g);

    let eps = 1e-3f32;
    let d = s.d() as usize;

    // --- dq: perturb ONLY the q region (region 0 of stride 3d) of the decoder buffer ---
    let mut max_dq = 0f32;
    for i in 0..q_dec.len() {
        if (i % (3 * d)) / d != 0 {
            continue; // skip unused k/v regions of the decoder buffer
        }
        let mut wp = q_dec.clone();
        wp[i] += eps;
        let lp = loss(&forward(&gpu, &s, &wp, &kv_enc), &g);
        let mut wm = q_dec.clone();
        wm[i] -= eps;
        let lm = loss(&forward(&gpu, &s, &wm, &kv_enc), &g);
        let num = (lp - lm) / (2.0 * eps);
        max_dq = max_dq.max((num - d_q[i]).abs());
    }

    // --- dk/dv: perturb the encoder KV buffer; region 0 = k (dk), region 1 = v (dv) ---
    let mut max_dk = 0f32;
    let mut max_dv = 0f32;
    for i in 0..kv_enc.len() {
        let mut wp = kv_enc.clone();
        wp[i] += eps;
        let lp = loss(&forward(&gpu, &s, &q_dec, &wp), &g);
        let mut wm = kv_enc.clone();
        wm[i] -= eps;
        let lm = loss(&forward(&gpu, &s, &q_dec, &wm), &g);
        let num = (lp - lm) / (2.0 * eps);
        let err = (num - d_kv[i]).abs();
        if (i % (2 * d)) / d == 0 {
            max_dk = max_dk.max(err);
        } else {
            max_dv = max_dv.max(err);
        }
    }

    println!(
        "cross FD max abs err: dq(decoder buf)={:.3e} dk(enc buf)={:.3e} dv(enc buf)={:.3e}",
        max_dq, max_dk, max_dv
    );
    let tol = 2e-2f32;
    assert!(max_dq < tol, "dq_cross FD err {max_dq} >= {tol}");
    assert!(max_dk < tol, "dk_cross FD err {max_dk} >= {tol}");
    assert!(max_dv < tol, "dv_cross FD err {max_dv} >= {tol}");
}
