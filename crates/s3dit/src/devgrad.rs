// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) forward + backward for the S³-DiT block, as a **persistent**
//! engine ([`BlockDev`]) — the compute core of the device training loop. One GPU
//! device + one set of reusable buffers (sized to the max token count) serve
//! every block: each call re-uploads that block's weights and runs the graph, so
//! a 34-block training step is 34 cheap submits, not 34 device creations.
//!
//! Every op the gradchecked host reference ([`crate::grad`]) does analytically,
//! this does on-device with brain's pre-gradchecked kernels: matmul_dx_reg/
//! matmul_dw_reg (linears), attn_bwd_{dscores,dv,dq,dk}_bidir on the packed qkv
//! buffer (attention), silu_bwd_da/db (SwiGLU), interleaved-RoPE backward (the
//! forward kernel fed a negated sin table), rms_inv_eps/rmsnorm_dw/rmsnorm_dx_eps
//! (RMSNorm at eps=1e-5). The adaLN fold + its backward stay on the host (small).
//!
//! Validated by `tests/dev_grad.rs`: device grads match the finite-difference
//! gradchecked host to fp32 (cosine 1.000000).

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

use crate::grad::{Dims, GradsF32, WeightsF32};

// Kernel indices.
const K_RMS: usize = 0;
const K_MM: usize = 1;
const K_ROPE: usize = 2;
const K_PACK: usize = 3;
const K_SCORES: usize = 4;
const K_SOFTMAX: usize = 5;
const K_APPLY: usize = 6;
const K_SILU: usize = 7;
const K_ADD: usize = 8;
const K_DX: usize = 9;
const K_DW: usize = 10;
const K_RINV: usize = 11;
const K_RDX: usize = 12;
const K_SDA: usize = 13;
const K_SDB: usize = 14;
const K_DSCORES: usize = 15;
const K_DV: usize = 16;
const K_DQ: usize = 17;
const K_DK: usize = 18;
const K_UNPACK: usize = 19;
const K_RDW: usize = 20;

const KERNELS: [(&str, &str); 21] = [
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("matmul_reg3", kernels::MATMUL_REG3),
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
    x.div_ceil(128) as u32
}

/// adaLN fold (host): `mod = adaln_w·c + adaln_b` → folded norm weights + the
/// scale/gate vectors the backward needs. Unmodulated → raw weights, zero s/g.
#[allow(clippy::type_complexity)]
fn fold(w: &WeightsF32, d: Dims, c: &[f32], modulation: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = d.dim;
    let (an1, an2, fn1, fn2) = (w.an1.clone(), w.an2.clone(), w.fn1.clone(), w.fn2.clone());
    if !modulation {
        let z = vec![0f32; dim];
        return (an1, an2, fn1, fn2, z.clone(), z.clone(), z.clone(), z);
    }
    let mut m = w.adaln_b.clone();
    for (i, mi) in m.iter_mut().enumerate() {
        let mut a = *mi;
        for (j, &cj) in c.iter().enumerate() {
            a += w.adaln_w[i * d.cdim + j] * cj;
        }
        *mi = a;
    }
    let sm = m[0..dim].to_vec();
    let gm = m[dim..2 * dim].to_vec();
    let sp = m[2 * dim..3 * dim].to_vec();
    let gp = m[3 * dim..4 * dim].to_vec();
    let an1f = an1.iter().zip(&sm).map(|(&r, &s)| r * (1.0 + s)).collect();
    let an2f = an2.iter().zip(&gm).map(|(&r, &g)| r * g.tanh()).collect();
    let fn1f = fn1.iter().zip(&sp).map(|(&r, &s)| r * (1.0 + s)).collect();
    let fn2f = fn2.iter().zip(&gp).map(|(&r, &g)| r * g.tanh()).collect();
    (an1f, an2f, fn1f, fn2f, sm, gm, sp, gp)
}

/// A persistent GPU block engine: one device, reusable buffers sized to
/// `max_t` tokens, driving any block's forward or backward.
pub struct BlockDev {
    gpu: Gpu,
    dim: usize,
    nh: usize,
    hd: usize,
    hidden: usize,
    half: usize,
    b: HashMap<&'static str, DeviceBuffer>,
}

impl BlockDev {
    pub fn new(max_t: usize, dim: usize, nh: usize) -> BlockDev {
        BlockDev::from_gpu(Gpu::new_wgpu(&KERNELS), max_t, dim, nh)
    }

    /// Build `count` engines, one per physical GPU, via a SINGLE device
    /// enumeration (`new_wgpu_multi`) — the collision-free multi-card placement
    /// the inference sharding uses. Each engine can host a different pipeline
    /// stage's layer slice on its own card.
    pub fn new_multi(count: usize, max_t: usize, dim: usize, nh: usize) -> Vec<BlockDev> {
        Gpu::new_wgpu_multi(&KERNELS, count).into_iter().map(|g| BlockDev::from_gpu(g, max_t, dim, nh)).collect()
    }

    pub fn from_gpu(gpu: Gpu, max_t: usize, dim: usize, nh: usize) -> BlockDev {
        let hd = dim / nh;
        let hidden = dim * 8 / 3;
        let mut b = HashMap::new();
        let mut mk = |name: &'static str, n: usize| {
            b.insert(name, gpu.storage(n as u64));
        };
        let td = max_t * dim;
        let th = max_t * hidden;
        // weights
        for n in ["wq", "wk", "wv", "wo"] {
            mk(n, dim * dim);
        }
        for n in ["w1", "w3"] {
            mk(n, hidden * dim);
        }
        mk("w2", dim * hidden);
        for n in ["nq", "nk"] {
            mk(n, hd);
        }
        for n in ["an1b", "an2b", "fn1b", "fn2b"] {
            mk(n, dim);
        }
        // io + acts
        for n in ["cosb", "sinb", "nsinb"] {
            mk(n, max_t * (hd / 2));
        }
        for n in [
            "xb", "doutb", "n1", "q", "k", "v", "qn", "kn", "qr", "kr", "ctx", "attn_out", "n2", "x1", "f1", "ff", "f2", "outb", "d_ff", "d_f1a", "d_f1b", "d_f1",
            "d_x1mlp", "d_x1", "d_attn_out", "d_ctx", "d_qr", "d_kr", "d_v", "d_qn", "d_kn", "d_q", "d_k", "d_n1q", "d_n1k", "d_n1v", "d_n1t", "d_n1", "d_xattn", "d_x",
        ] {
            mk(n, td);
        }
        for n in ["gg", "uu", "hsw", "d_hsw", "d_g", "d_u"] {
            mk(n, th);
        }
        mk("qkv", max_t * 3 * dim);
        mk("d_qkv", max_t * 3 * dim);
        mk("scores", nh * max_t * max_t);
        mk("probs", nh * max_t * max_t);
        mk("d_scores", nh * max_t * max_t);
        // weight grads + inv
        for n in ["g_wq", "g_wk", "g_wv", "g_wo"] {
            mk(n, dim * dim);
        }
        for n in ["g_w1", "g_w3"] {
            mk(n, hidden * dim);
        }
        mk("g_w2", dim * hidden);
        for n in ["g_nq", "g_nk"] {
            mk(n, hd);
        }
        for n in ["d_an1f", "d_an2f", "d_fn1f", "d_fn2f"] {
            mk(n, dim);
        }
        for n in ["inv_n1", "inv_n2", "inv_f1", "inv_f2"] {
            mk(n, max_t);
        }
        for n in ["inv_qn", "inv_kn"] {
            mk(n, max_t * nh);
        }
        BlockDev { gpu, dim, nh, hd, hidden, half: hd / 2, b }
    }

    fn upf(&self, name: &str, data: &[f32]) {
        let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.b[name], &bits);
    }
    fn zero(&self, name: &str, n: usize) {
        self.gpu.write(&self.b[name], &vec![0u32; n]);
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        &self.b[name]
    }
    fn rd(&self, name: &str, n: usize) -> Vec<f32> {
        self.gpu.read(&self.b[name], n)
    }

    /// Upload one block's weights + folded norms + inputs (all f32 — direct).
    fn upload(&self, w: &WeightsF32, x: &[f32], cos: &[f32], sin: &[f32], an1f: &[f32], an2f: &[f32], fn1f: &[f32], fn2f: &[f32]) {
        self.upf("wq", &w.wq);
        self.upf("wk", &w.wk);
        self.upf("wv", &w.wv);
        self.upf("wo", &w.wo);
        self.upf("w1", &w.w1);
        self.upf("w2", &w.w2);
        self.upf("w3", &w.w3);
        self.upf("nq", &w.nq);
        self.upf("nk", &w.nk);
        self.upf("an1b", an1f);
        self.upf("an2b", an2f);
        self.upf("fn1b", fn1f);
        self.upf("fn2b", fn2f);
        self.upf("xb", x);
        self.upf("cosb", cos);
        self.upf("sinb", sin);
        let nsin: Vec<f32> = sin.iter().map(|&s| -s).collect();
        self.upf("nsinb", &nsin);
    }

    /// Forward step list for `t` tokens (writes `outb`).
    fn fwd_steps(&self, t: usize) -> Vec<Step> {
        let (dim, nh, hd, hidden, half) = (self.dim, self.nh, self.hd, self.hidden, self.half);
        let g = |n: &str| self.g(n);
        let mm = |a: &str, wt: &str, o: &str, m: usize, kk: usize, n: usize| self.gpu.step(K_MM, &[g(a), g(wt), g(o)], &[m as u32, kk as u32, n as u32], d128(m) * d128(n) * 256);
        let rms = |x: &str, wt: &str, o: &str, dm: usize, rows: usize| self.gpu.step(K_RMS, &[g(x), g(wt), g(o)], &[dm as u32, rows as u32, f(EPS)], rows as u32);
        let td = t * dim;
        let ap = [1u32, nh as u32, t as u32, hd as u32, (3 * dim) as u32];
        vec![
            rms("xb", "an1b", "n1", dim, t),
            mm("n1", "wq", "q", t, dim, dim),
            mm("n1", "wk", "k", t, dim, dim),
            mm("n1", "wv", "v", t, dim, dim),
            rms("q", "nq", "qn", hd, t * nh),
            rms("k", "nk", "kn", hd, t * nh),
            self.gpu.step(K_ROPE, &[g("qn"), g("cosb"), g("sinb"), g("qr")], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32),
            self.gpu.step(K_ROPE, &[g("kn"), g("cosb"), g("sinb"), g("kr")], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32),
            self.gpu.step(K_PACK, &[g("qr"), g("kr"), g("v"), g("qkv")], &[t as u32, dim as u32], (t * 3 * dim) as u32),
            self.gpu.step(K_SCORES, &[g("qkv"), g("scores")], &[ap[0], ap[1], ap[2], ap[3], ap[4], 0, dim as u32], (nh * t * t) as u32),
            self.gpu.step(K_SOFTMAX, &[g("scores"), g("probs")], &[1, nh as u32, t as u32], (nh * t) as u32),
            self.gpu.step(K_APPLY, &[g("probs"), g("qkv"), g("ctx")], &[ap[0], ap[1], ap[2], ap[3], ap[4], (2 * dim) as u32, dim as u32], (nh * t * hd) as u32),
            mm("ctx", "wo", "attn_out", t, dim, dim),
            rms("attn_out", "an2b", "n2", dim, t),
            self.gpu.step(K_ADD, &[g("xb"), g("n2"), g("x1")], &[td as u32], td as u32),
            rms("x1", "fn1b", "f1", dim, t),
            mm("f1", "w1", "gg", t, dim, hidden),
            mm("f1", "w3", "uu", t, dim, hidden),
            self.gpu.step(K_SILU, &[g("gg"), g("uu"), g("hsw")], &[(t * hidden) as u32], (t * hidden) as u32),
            mm("hsw", "w2", "ff", t, hidden, dim),
            rms("ff", "fn2b", "f2", dim, t),
            self.gpu.step(K_ADD, &[g("x1"), g("f2"), g("outb")], &[td as u32], td as u32),
        ]
    }

    /// Backward step list for `t` tokens (reads `doutb`, writes grads + `d_x`).
    fn bwd_steps(&self, t: usize) -> Vec<Step> {
        let (dim, nh, hd, hidden, half) = (self.dim, self.nh, self.hd, self.hidden, self.half);
        let g = |n: &str| self.g(n);
        let td = t * dim;
        let th = t * hidden;
        let lin_dx = |dy: &str, wt: &str, dx: &str, inp: usize, out: usize| self.gpu.step(K_DX, &[g(dy), g(wt), g(dx)], &[t as u32, inp as u32, out as u32, 0], d128(t) * d128(inp) * 256);
        let lin_dw = |dy: &str, xin: &str, dw: &str, inp: usize, out: usize| self.gpu.step(K_DW, &[g(dy), g(xin), g(dw)], &[t as u32, inp as u32, out as u32], d128(out) * d128(inp) * 256);
        let rinv = |x: &str, inv: &str, dm: usize, rows: usize| self.gpu.step(K_RINV, &[g(x), g(inv)], &[dm as u32, rows as u32, f(EPS)], rows as u32);
        let rdw = |dy: &str, x: &str, inv: &str, dw: &str, dm: usize, rows: usize| self.gpu.step(K_RDW, &[g(dy), g(x), g(inv), g(dw)], &[dm as u32, rows as u32], dm as u32);
        let rdx = |x: &str, wt: &str, dy: &str, dx: &str, dm: usize, rows: usize| self.gpu.step(K_RDX, &[g(x), g(wt), g(dy), g(dx)], &[dm as u32, rows as u32, f(EPS)], rows as u32);
        let add = |a: &str, bb: &str, o: &str| self.gpu.step(K_ADD, &[g(a), g(bb), g(o)], &[td as u32], td as u32);
        let pv = [1u32, nh as u32, t as u32, hd as u32, (3 * dim) as u32, (2 * dim) as u32, dim as u32];
        let pqk = [1u32, nh as u32, t as u32, hd as u32, (3 * dim) as u32, 0, dim as u32];
        vec![
            // f2 = rmsnorm(ff, fn2f) ; out = x1 + f2
            rinv("ff", "inv_f2", dim, t),
            rdw("doutb", "ff", "inv_f2", "d_fn2f", dim, t),
            rdx("ff", "fn2b", "doutb", "d_ff", dim, t),
            lin_dx("d_ff", "w2", "d_hsw", hidden, dim),
            lin_dw("d_ff", "hsw", "g_w2", hidden, dim),
            self.gpu.step(K_SDA, &[g("gg"), g("uu"), g("d_hsw"), g("d_g")], &[th as u32], th as u32),
            self.gpu.step(K_SDB, &[g("gg"), g("d_hsw"), g("d_u")], &[th as u32], th as u32),
            lin_dx("d_g", "w1", "d_f1a", dim, hidden),
            lin_dw("d_g", "f1", "g_w1", dim, hidden),
            lin_dx("d_u", "w3", "d_f1b", dim, hidden),
            lin_dw("d_u", "f1", "g_w3", dim, hidden),
            add("d_f1a", "d_f1b", "d_f1"),
            rinv("x1", "inv_f1", dim, t),
            rdw("d_f1", "x1", "inv_f1", "d_fn1f", dim, t),
            rdx("x1", "fn1b", "d_f1", "d_x1mlp", dim, t),
            add("doutb", "d_x1mlp", "d_x1"),
            rinv("attn_out", "inv_n2", dim, t),
            rdw("d_x1", "attn_out", "inv_n2", "d_an2f", dim, t),
            rdx("attn_out", "an2b", "d_x1", "d_attn_out", dim, t),
            lin_dx("d_attn_out", "wo", "d_ctx", dim, dim),
            lin_dw("d_attn_out", "ctx", "g_wo", dim, dim),
            self.gpu.step(K_DSCORES, &[g("d_ctx"), g("qkv"), g("probs"), g("d_scores")], &pv, (nh * t) as u32),
            self.gpu.step(K_DV, &[g("probs"), g("d_ctx"), g("d_qkv")], &pv, (nh * t * hd) as u32),
            self.gpu.step(K_DQ, &[g("d_scores"), g("qkv"), g("d_qkv")], &pqk, (nh * t * hd) as u32),
            self.gpu.step(K_DK, &[g("d_scores"), g("qkv"), g("d_qkv")], &pqk, (nh * t * hd) as u32),
            self.gpu.step(K_UNPACK, &[g("d_qkv"), g("d_qr"), g("d_kr"), g("d_v")], &[t as u32, dim as u32], (t * 3 * dim) as u32),
            self.gpu.step(K_ROPE, &[g("d_qr"), g("cosb"), g("nsinb"), g("d_qn")], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32),
            self.gpu.step(K_ROPE, &[g("d_kr"), g("cosb"), g("nsinb"), g("d_kn")], &[t as u32, nh as u32, hd as u32, half as u32], (t * nh * half) as u32),
            rinv("q", "inv_qn", hd, t * nh),
            rdw("d_qn", "q", "inv_qn", "g_nq", hd, t * nh),
            rdx("q", "nq", "d_qn", "d_q", hd, t * nh),
            rinv("k", "inv_kn", hd, t * nh),
            rdw("d_kn", "k", "inv_kn", "g_nk", hd, t * nh),
            rdx("k", "nk", "d_kn", "d_k", hd, t * nh),
            lin_dx("d_q", "wq", "d_n1q", dim, dim),
            lin_dw("d_q", "n1", "g_wq", dim, dim),
            lin_dx("d_k", "wk", "d_n1k", dim, dim),
            lin_dw("d_k", "n1", "g_wk", dim, dim),
            lin_dx("d_v", "wv", "d_n1v", dim, dim),
            lin_dw("d_v", "n1", "g_wv", dim, dim),
            add("d_n1q", "d_n1k", "d_n1t"),
            add("d_n1t", "d_n1v", "d_n1"),
            rinv("xb", "inv_n1", dim, t),
            rdw("d_n1", "xb", "inv_n1", "d_an1f", dim, t),
            rdx("xb", "an1b", "d_n1", "d_xattn", dim, t),
            add("d_x1", "d_xattn", "d_x"),
        ]
    }

    /// Forward one block, returning its output `[t·dim]`.
    pub fn forward(&self, w: &WeightsF32, d: Dims, x: &[f32], c: &[f32], cos: &[f32], sin: &[f32], modulation: bool) -> Vec<f32> {
        let (an1f, an2f, fn1f, fn2f, ..) = fold(w, d, c, modulation);
        self.upload(w, x, cos, sin, &an1f, &an2f, &fn1f, &fn2f);
        let steps = self.fwd_steps(d.t);
        self.gpu.submit(&[], &steps);
        self.gpu.poll_wait();
        self.gpu.read(&self.b["outb"], d.t * self.dim)
    }

    /// Backward one block: recompute forward + backprop `dout`, returning the
    /// per-tensor grads (f32) and the input grad `dx`.
    pub fn backward(&self, w: &WeightsF32, d: Dims, x: &[f32], c: &[f32], cos: &[f32], sin: &[f32], modulation: bool, dout: &[f32]) -> GradsF32 {
        let (dim, hd, hidden) = (self.dim, self.hd, self.hidden);
        let t = d.t;
        let (an1f, an2f, fn1f, fn2f, sm, gm, sp, gp) = fold(w, d, c, modulation);
        self.upload(w, x, cos, sin, &an1f, &an2f, &fn1f, &fn2f);
        self.upf("doutb", dout);
        // zero the accumulating grad buffers
        for (n, sz) in [("g_wq", dim * dim), ("g_wk", dim * dim), ("g_wv", dim * dim), ("g_wo", dim * dim), ("g_w1", hidden * dim), ("g_w3", hidden * dim), ("g_w2", dim * hidden), ("g_nq", hd), ("g_nk", hd), ("d_an1f", dim), ("d_an2f", dim), ("d_fn1f", dim), ("d_fn2f", dim)] {
            self.zero(n, sz);
        }
        self.zero("d_qkv", t * 3 * dim);
        let mut steps = self.fwd_steps(t);
        steps.extend(self.bwd_steps(t));
        self.gpu.submit(&[], &steps);
        self.gpu.poll_wait();

        let (da1, da2, df1, df2) = (self.rd("d_an1f", dim), self.rd("d_an2f", dim), self.rd("d_fn1f", dim), self.rd("d_fn2f", dim));
        let mut gr = GradsF32 {
            wq: self.rd("g_wq", dim * dim), wk: self.rd("g_wk", dim * dim), wv: self.rd("g_wv", dim * dim), wo: self.rd("g_wo", dim * dim),
            w1: self.rd("g_w1", hidden * dim), w2: self.rd("g_w2", dim * hidden), w3: self.rd("g_w3", hidden * dim),
            nq: self.rd("g_nq", hd), nk: self.rd("g_nk", hd),
            an1: vec![0.0; dim], an2: vec![0.0; dim], fn1: vec![0.0; dim], fn2: vec![0.0; dim],
            adaln_w: vec![0.0; 4 * dim * d.cdim], adaln_b: vec![0.0; 4 * dim],
            dx: self.rd("d_x", t * dim), dc: vec![0.0; d.cdim],
        };
        if !modulation {
            gr.an1 = da1;
            gr.an2 = da2;
            gr.fn1 = df1;
            gr.fn2 = df2;
            return gr;
        }
        let mut dmod = vec![0f32; 4 * dim];
        for cc in 0..dim {
            gr.an1[cc] = da1[cc] * (1.0 + sm[cc]);
            dmod[cc] = da1[cc] * w.an1[cc];
            let tg = gm[cc].tanh();
            gr.an2[cc] = da2[cc] * tg;
            dmod[dim + cc] = da2[cc] * w.an2[cc] * (1.0 - tg * tg);
            gr.fn1[cc] = df1[cc] * (1.0 + sp[cc]);
            dmod[2 * dim + cc] = df1[cc] * w.fn1[cc];
            let tgm = gp[cc].tanh();
            gr.fn2[cc] = df2[cc] * tgm;
            dmod[3 * dim + cc] = df2[cc] * w.fn2[cc] * (1.0 - tgm * tgm);
        }
        for i in 0..4 * dim {
            gr.adaln_b[i] = dmod[i];
            for j in 0..d.cdim {
                gr.adaln_w[i * d.cdim + j] = dmod[i] * c[j];
                gr.dc[j] += dmod[i] * w.adaln_w[i * d.cdim + j];
            }
        }
        gr
    }
}

/// Thin wrapper: build a fresh engine and run one block's backward. Kept for the
/// standalone `tests/dev_grad.rs` parity check.
pub fn block_backward_device(d: Dims, w: &WeightsF32, x: &[f32], c: &[f32], cos: &[f32], sin: &[f32], dout: &[f32]) -> GradsF32 {
    let eng = BlockDev::new(d.t, d.dim, d.nh);
    eng.backward(w, d, x, c, cos, sin, true, dout)
}
