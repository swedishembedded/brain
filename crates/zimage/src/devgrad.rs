// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) forward + backward for ONE S³-DiT block, driving the actual
//! training kernels — the bridge between the gradchecked host reference
//! ([`crate::grad`]) and a real device training step. Every op the host backward
//! does analytically, this does on-device with brain's pre-gradchecked kernels:
//!
//!   linear bwd     → matmul_dx_reg / matmul_dw_reg
//!   RMSNorm bwd    → rms_inv_eps / rmsnorm_dw / rmsnorm_dx_eps  (eps=1e-5 twins)
//!   attention bwd  → attn_bwd_{dscores,dv,dq,dk}_bidir on the packed qkv buffer
//!   interleaved RoPE bwd → rope_interleave_table with a negated sin table
//!   SwiGLU bwd     → silu_bwd_da / silu_bwd_db
//!   adaLN fold bwd → host (small: routes the folded-norm-weight grads back to
//!                    the raw norms, the modulation linear, and the conditioning `dc`)
//!
//! Validated against the host reference in `tests/dev_grad.rs` (same weights +
//! inputs → grads must match within fp32 tolerance). Once the device grads match
//! the gradchecked host grads, chaining this across the 34 blocks + the
//! flow-matching loss + `OffloadAdam` is a real training loop.

use gpu_core::{f, DeviceBuffer, Gpu};

use crate::grad::{Dims, Grads, Weights};

// Kernel indices into KERNELS below.
const K_RMS: usize = 0; // rmsnorm_eps (fwd)
const K_MM: usize = 1; // matmul_reg2 (fwd)
const K_ROPE: usize = 2; // rope_interleave_table (fwd + bwd via -sin)
const K_PACK: usize = 3;
const K_SCORES: usize = 4;
const K_SOFTMAX: usize = 5;
const K_APPLY: usize = 6;
const K_SILU: usize = 7;
const K_ADD: usize = 8;
const K_DX: usize = 9; // matmul_dx_reg
const K_DW: usize = 10; // matmul_dw_reg
const K_RINV: usize = 11; // rms_inv_eps
const K_RDX: usize = 12; // rmsnorm_dx_eps
const K_SDA: usize = 13; // silu_bwd_da
const K_SDB: usize = 14; // silu_bwd_db
const K_DSCORES: usize = 15;
const K_DV: usize = 16;
const K_DQ: usize = 17;
const K_DK: usize = 18;
const K_UNPACK: usize = 19;
const K_RDW: usize = 20; // rmsnorm_dw

const KERNELS: [(&str, &str); 21] = [
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("rms_inv_eps", kernels::RMS_INV_EPS),
    ("rmsnorm_dx_eps", kernels::RMSNORM_DX_EPS),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("unpack_qkv", kernels::UNPACK_QKV),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
];

const EPS: f32 = 1e-5;

fn d128(x: usize) -> u32 {
    ((x + 127) / 128) as u32
}

/// adaLN fold (host): mod = adaln_w·c + adaln_b → the four folded norm weights.
/// Returns `(an1f, an2f, fn1f, fn2f, scale_msa, gate_msa, scale_mlp, gate_mlp)`.
#[allow(clippy::type_complexity)]
fn fold(w: &Weights, d: Dims, c: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = d.dim;
    let mut m = w.adaln_b.iter().map(|&v| v as f32).collect::<Vec<f32>>();
    for (i, mi) in m.iter_mut().enumerate() {
        let mut a = *mi;
        for (j, &cj) in c.iter().enumerate() {
            a += w.adaln_w[i * d.cdim + j] as f32 * cj;
        }
        *mi = a;
    }
    let sm = m[0..dim].to_vec();
    let gm = m[dim..2 * dim].to_vec();
    let sp = m[2 * dim..3 * dim].to_vec();
    let gp = m[3 * dim..4 * dim].to_vec();
    let raw = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let (an1, an2, fn1, fn2) = (raw(&w.an1), raw(&w.an2), raw(&w.fn1), raw(&w.fn2));
    let an1f = an1.iter().zip(&sm).map(|(&r, &s)| r * (1.0 + s)).collect();
    let an2f = an2.iter().zip(&gm).map(|(&r, &g)| r * g.tanh()).collect();
    let fn1f = fn1.iter().zip(&sp).map(|(&r, &s)| r * (1.0 + s)).collect();
    let fn2f = fn2.iter().zip(&gp).map(|(&r, &g)| r * g.tanh()).collect();
    (an1f, an2f, fn1f, fn2f, sm, gm, sp, gp)
}

/// Run one block forward+backward on the GPU and return the grads (host f64),
/// for validation against [`crate::grad::backward`].
pub fn block_backward_device(d: Dims, w: &Weights, x: &[f32], c: &[f32], cos: &[f32], sin: &[f32], dout: &[f32]) -> Grads {
    let (t, dim, nh, hd, hidden, half) = (d.t, d.dim, d.nh, d.hd, d.hidden, d.half());
    let g = Gpu::new_wgpu(&KERNELS);
    let up = |data: &[f64]| {
        let b = g.storage(data.len() as u64);
        let bits: Vec<u32> = data.iter().map(|&v| (v as f32).to_bits()).collect();
        g.write(&b, &bits);
        b
    };
    let upf = |data: &[f32]| {
        let b = g.storage(data.len() as u64);
        let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        g.write(&b, &bits);
        b
    };
    let zeros = |n: usize| {
        let b = g.storage(n as u64);
        g.write(&b, &vec![0u32; n]);
        b
    };
    let buf = |n: usize| g.storage(n as u64);

    // ---- weights ----
    let (an1f, an2f, fn1f, fn2f, sm, gm, sp, gp) = fold(w, d, c);
    let (wq, wk, wv, wo) = (up(&w.wq), up(&w.wk), up(&w.wv), up(&w.wo));
    let (w1, w2, w3) = (up(&w.w1), up(&w.w2), up(&w.w3));
    let (nq, nk) = (up(&w.nq), up(&w.nk));
    let (an1b, an2b, fn1b, fn2b) = (upf(&an1f), upf(&an2f), upf(&fn1f), upf(&fn2f));

    // ---- inputs ----
    let xb = upf(x);
    let cosb = upf(cos);
    let sinb = upf(sin);
    let neg_sin: Vec<f32> = sin.iter().map(|&s| -s).collect();
    let nsinb = upf(&neg_sin);
    let doutb = upf(dout);

    // ---- forward activation buffers ----
    let (td, th, ha) = (t * dim, t * hidden, nh * t * t);
    let (n1, q, k, v) = (buf(td), buf(td), buf(td), buf(td));
    let (qn, kn, qr, kr) = (buf(td), buf(td), buf(td), buf(td));
    let qkv = buf(t * 3 * dim);
    let (scores, probs) = (buf(ha), buf(ha));
    let (ctx, attn_out, n2, x1) = (buf(td), buf(td), buf(td), buf(td));
    let (f1, gg, uu, hsw, ff, f2, outb) = (buf(td), buf(th), buf(th), buf(th), buf(td), buf(td), buf(td));

    // ---- forward ----
    let mm = |a: &DeviceBuffer, wt: &DeviceBuffer, o: &DeviceBuffer, m: usize, kk: usize, n: usize| g.step(K_MM, &[a, wt, o], &[m as u32, kk as u32, n as u32], d128(m) * d128(n) * 256);
    let rms = |x: &DeviceBuffer, wt: &DeviceBuffer, o: &DeviceBuffer, dm: usize, rows: usize| g.step(K_RMS, &[x, wt, o], &[dm as u32, rows as u32, f(EPS)], rows as u32);
    let mut s = Vec::new();
    s.push(rms(&xb, &an1b, &n1, dim, t));
    s.push(mm(&n1, &wq, &q, t, dim, dim));
    s.push(mm(&n1, &wk, &k, t, dim, dim));
    s.push(mm(&n1, &wv, &v, t, dim, dim));
    s.push(rms(&q, &nq, &qn, hd, t * nh));
    s.push(rms(&k, &nk, &kn, hd, t * nh));
    s.push(g.step(K_ROPE, &[&qn, &cosb, &sinb, &qr], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32));
    s.push(g.step(K_ROPE, &[&kn, &cosb, &sinb, &kr], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32));
    s.push(g.step(K_PACK, &[&qr, &kr, &v, &qkv], &[t as u32, dim as u32], (t * 3 * dim) as u32));
    let ap = [1u32, nh as u32, t as u32, hd as u32, (3 * dim) as u32];
    s.push(g.step(K_SCORES, &[&qkv, &scores], &[ap[0], ap[1], ap[2], ap[3], ap[4], 0, dim as u32], (nh * t * t) as u32));
    s.push(g.step(K_SOFTMAX, &[&scores, &probs], &[1, nh as u32, t as u32], (nh * t) as u32));
    s.push(g.step(K_APPLY, &[&probs, &qkv, &ctx], &[ap[0], ap[1], ap[2], ap[3], ap[4], (2 * dim) as u32, dim as u32], (nh * t * hd) as u32));
    s.push(mm(&ctx, &wo, &attn_out, t, dim, dim));
    s.push(rms(&attn_out, &an2b, &n2, dim, t));
    s.push(g.step(K_ADD, &[&xb, &n2, &x1], &[td as u32], td as u32));
    s.push(rms(&x1, &fn1b, &f1, dim, t));
    s.push(mm(&f1, &w1, &gg, t, dim, hidden));
    s.push(mm(&f1, &w3, &uu, t, dim, hidden));
    s.push(g.step(K_SILU, &[&gg, &uu, &hsw], &[th as u32], th as u32));
    s.push(mm(&hsw, &w2, &ff, t, hidden, dim));
    s.push(rms(&ff, &fn2b, &f2, dim, t));
    s.push(g.step(K_ADD, &[&x1, &f2, &outb], &[td as u32], td as u32));

    // ---- backward buffers (weight grads zeroed: rmsnorm_dw + matmul_dw_reg accumulate) ----
    let (g_wq, g_wk, g_wv, g_wo) = (zeros(dim * dim), zeros(dim * dim), zeros(dim * dim), zeros(dim * dim));
    let (g_w1, g_w2, g_w3) = (zeros(hidden * dim), zeros(dim * hidden), zeros(hidden * dim));
    let (g_nq, g_nk) = (zeros(hd), zeros(hd));
    let (d_an1f, d_an2f, d_fn1f, d_fn2f) = (zeros(dim), zeros(dim), zeros(dim), zeros(dim));
    let d_qkv = zeros(t * 3 * dim);
    // intermediates
    let (d_ff, d_hsw, d_g, d_u) = (buf(td), buf(th), buf(th), buf(th));
    let (d_f1a, d_f1b, d_f1, d_x1mlp, d_x1) = (buf(td), buf(td), buf(td), buf(td), buf(td));
    let (d_attn_out, d_ctx, d_scores) = (buf(td), buf(td), buf(ha));
    let (d_qr, d_kr, d_v, d_qn, d_kn, d_q, d_k) = (buf(td), buf(td), buf(td), buf(td), buf(td), buf(td), buf(td));
    let (d_n1q, d_n1k, d_n1v, d_n1t, d_n1, d_xattn, d_x) = (buf(td), buf(td), buf(td), buf(td), buf(td), buf(td), buf(td));
    let (inv_n1, inv_qn, inv_kn, inv_n2, inv_f1, inv_f2) = (buf(t), buf(t * nh), buf(t * nh), buf(t), buf(t), buf(t));

    // dx of a linear y=x@W^T: matmul_dx_reg(dy, W)->dx ; dw: matmul_dw_reg(dy, x)->dW.
    let lin_dx = |dy: &DeviceBuffer, wt: &DeviceBuffer, dx: &DeviceBuffer, inp: usize, out: usize| g.step(K_DX, &[dy, wt, dx], &[t as u32, inp as u32, out as u32, 0], d128(t) * d128(inp) * 256);
    let lin_dw = |dy: &DeviceBuffer, xin: &DeviceBuffer, dw: &DeviceBuffer, inp: usize, out: usize| g.step(K_DW, &[dy, xin, dw], &[t as u32, inp as u32, out as u32], d128(out) * d128(inp) * 256);
    let rinv = |x: &DeviceBuffer, inv: &DeviceBuffer, dm: usize, rows: usize| g.step(K_RINV, &[x, inv], &[dm as u32, rows as u32, f(EPS)], rows as u32);
    let rdw = |dy: &DeviceBuffer, x: &DeviceBuffer, inv: &DeviceBuffer, dw: &DeviceBuffer, dm: usize, rows: usize| g.step(K_RDW, &[dy, x, inv, dw], &[dm as u32, rows as u32], dm as u32);
    let rdx = |x: &DeviceBuffer, wt: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, dm: usize, rows: usize| g.step(K_RDX, &[x, wt, dy, dx], &[dm as u32, rows as u32, f(EPS)], rows as u32);
    let add = |a: &DeviceBuffer, b: &DeviceBuffer, o: &DeviceBuffer| g.step(K_ADD, &[a, b, o], &[td as u32], td as u32);

    // out = x1 + f2 ; f2 = rmsnorm(ff, fn2f)
    s.push(rinv(&ff, &inv_f2, dim, t));
    s.push(rdw(&doutb, &ff, &inv_f2, &d_fn2f, dim, t));
    s.push(rdx(&ff, &fn2b, &doutb, &d_ff, dim, t));
    // ff = hsw @ w2^T
    s.push(lin_dx(&d_ff, &w2, &d_hsw, hidden, dim));
    s.push(lin_dw(&d_ff, &hsw, &g_w2, hidden, dim));
    // hsw = silu(g)*u
    s.push(g.step(K_SDA, &[&gg, &uu, &d_hsw, &d_g], &[th as u32], th as u32));
    s.push(g.step(K_SDB, &[&gg, &d_hsw, &d_u], &[th as u32], th as u32));
    // g = f1@w1^T ; u = f1@w3^T
    s.push(lin_dx(&d_g, &w1, &d_f1a, dim, hidden));
    s.push(lin_dw(&d_g, &f1, &g_w1, dim, hidden));
    s.push(lin_dx(&d_u, &w3, &d_f1b, dim, hidden));
    s.push(lin_dw(&d_u, &f1, &g_w3, dim, hidden));
    s.push(add(&d_f1a, &d_f1b, &d_f1));
    // f1 = rmsnorm(x1, fn1f)
    s.push(rinv(&x1, &inv_f1, dim, t));
    s.push(rdw(&d_f1, &x1, &inv_f1, &d_fn1f, dim, t));
    s.push(rdx(&x1, &fn1b, &d_f1, &d_x1mlp, dim, t));
    // x1 = x + n2 : d_x1 = dout + d_x1mlp
    s.push(add(&doutb, &d_x1mlp, &d_x1));
    // n2 = rmsnorm(attn_out, an2f)  (d_n2 = d_x1)
    s.push(rinv(&attn_out, &inv_n2, dim, t));
    s.push(rdw(&d_x1, &attn_out, &inv_n2, &d_an2f, dim, t));
    s.push(rdx(&attn_out, &an2b, &d_x1, &d_attn_out, dim, t));
    // attn_out = ctx @ wo^T
    s.push(lin_dx(&d_attn_out, &wo, &d_ctx, dim, dim));
    s.push(lin_dw(&d_attn_out, &ctx, &g_wo, dim, dim));
    // attention backward (packed qkv)
    let pv = [1u32, nh as u32, t as u32, hd as u32, (3 * dim) as u32, (2 * dim) as u32, dim as u32]; // ..,v_off,d_model
    let pqk = [1u32, nh as u32, t as u32, hd as u32, (3 * dim) as u32, 0, dim as u32]; // ..,q_off,k_off
    s.push(g.step(K_DSCORES, &[&d_ctx, &qkv, &probs, &d_scores], &pv, (nh * t) as u32));
    s.push(g.step(K_DV, &[&probs, &d_ctx, &d_qkv], &pv, (nh * t * hd) as u32));
    s.push(g.step(K_DQ, &[&d_scores, &qkv, &d_qkv], &pqk, (nh * t * hd) as u32));
    s.push(g.step(K_DK, &[&d_scores, &qkv, &d_qkv], &pqk, (nh * t * hd) as u32));
    // unpack d_qkv -> d_qr, d_kr, d_v
    s.push(g.step(K_UNPACK, &[&d_qkv, &d_qr, &d_kr, &d_v], &[t as u32, dim as u32], (t * 3 * dim) as u32));
    // RoPE backward = forward with negated sin
    s.push(g.step(K_ROPE, &[&d_qr, &cosb, &nsinb, &d_qn], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32));
    s.push(g.step(K_ROPE, &[&d_kr, &cosb, &nsinb, &d_kn], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32));
    // QK-norm backward
    s.push(rinv(&q, &inv_qn, hd, t * nh));
    s.push(rdw(&d_qn, &q, &inv_qn, &g_nq, hd, t * nh));
    s.push(rdx(&q, &nq, &d_qn, &d_q, hd, t * nh));
    s.push(rinv(&k, &inv_kn, hd, t * nh));
    s.push(rdw(&d_kn, &k, &inv_kn, &g_nk, hd, t * nh));
    s.push(rdx(&k, &nk, &d_kn, &d_k, hd, t * nh));
    // q,k,v = n1 @ {wq,wk,wv}^T
    s.push(lin_dx(&d_q, &wq, &d_n1q, dim, dim));
    s.push(lin_dw(&d_q, &n1, &g_wq, dim, dim));
    s.push(lin_dx(&d_k, &wk, &d_n1k, dim, dim));
    s.push(lin_dw(&d_k, &n1, &g_wk, dim, dim));
    s.push(lin_dx(&d_v, &wv, &d_n1v, dim, dim));
    s.push(lin_dw(&d_v, &n1, &g_wv, dim, dim));
    s.push(add(&d_n1q, &d_n1k, &d_n1t));
    s.push(add(&d_n1t, &d_n1v, &d_n1));
    // n1 = rmsnorm(x, an1f)
    s.push(rinv(&xb, &inv_n1, dim, t));
    s.push(rdw(&d_n1, &xb, &inv_n1, &d_an1f, dim, t));
    s.push(rdx(&xb, &an1b, &d_n1, &d_xattn, dim, t));
    s.push(add(&d_x1, &d_xattn, &d_x));

    g.submit(&[], &s);
    g.poll_wait();

    let rd = |b: &DeviceBuffer, n: usize| g.read(b, n).iter().map(|&v| v as f64).collect::<Vec<f64>>();

    // ---- host: adaLN fold backward → raw norms, modulation, dc ----
    let (da1, da2, df1, df2) = (rd(&d_an1f, dim), rd(&d_an2f, dim), rd(&d_fn1f, dim), rd(&d_fn2f, dim));
    let mut gr = Grads {
        wq: rd(&g_wq, dim * dim), wk: rd(&g_wk, dim * dim), wv: rd(&g_wv, dim * dim), wo: rd(&g_wo, dim * dim),
        w1: rd(&g_w1, hidden * dim), w2: rd(&g_w2, dim * hidden), w3: rd(&g_w3, hidden * dim),
        nq: rd(&g_nq, hd), nk: rd(&g_nk, hd),
        an1: vec![0.0; dim], an2: vec![0.0; dim], fn1: vec![0.0; dim], fn2: vec![0.0; dim],
        adaln_w: vec![0.0; 4 * dim * d.cdim], adaln_b: vec![0.0; 4 * dim],
        dx: rd(&d_x, td), dc: vec![0.0; d.cdim],
    };
    let mut dmod = vec![0f64; 4 * dim];
    for cc in 0..dim {
        gr.an1[cc] = da1[cc] * (1.0 + sm[cc] as f64);
        dmod[cc] = da1[cc] * w.an1[cc];
        let tg = (gm[cc] as f64).tanh();
        gr.an2[cc] = da2[cc] * tg;
        dmod[dim + cc] = da2[cc] * w.an2[cc] * (1.0 - tg * tg);
        gr.fn1[cc] = df1[cc] * (1.0 + sp[cc] as f64);
        dmod[2 * dim + cc] = df1[cc] * w.fn1[cc];
        let tgm = (gp[cc] as f64).tanh();
        gr.fn2[cc] = df2[cc] * tgm;
        dmod[3 * dim + cc] = df2[cc] * w.fn2[cc] * (1.0 - tgm * tgm);
    }
    for i in 0..4 * dim {
        gr.adaln_b[i] = dmod[i];
        for j in 0..d.cdim {
            gr.adaln_w[i * d.cdim + j] = dmod[i] * c[j] as f64;
            gr.dc[j] += dmod[i] * w.adaln_w[i * d.cdim + j];
        }
    }
    gr
}
