// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) full-model training step for the Z-Image S³-DiT. The expensive
//! 34-block DiT core runs on the GPU through the persistent [`crate::devgrad::BlockDev`]
//! engine (forward-sweep saving per-block inputs, then a reverse backward-sweep);
//! the thin wrapper — timestep MLP, image/caption embedders, adaLN final layer,
//! flow-matching loss — runs on the host (it is a handful of small linears; you
//! would not shard it anyway). Gradients from every stage are assembled into the
//! same [`ModelGradsF32`] the host reference produces.
//!
//! This is the device counterpart of [`crate::modelgrad`]: `tests/device_train.rs`
//! checks its grads match the gradchecked host reference (cosine ~1) and that it
//! overfits one batch on the GPU. The block-stack orchestration is identical for
//! the 4-block small config and the 34-block 6B — it scales by construction.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::devgrad::BlockDev;
use crate::grad::{Dims, GradsF32, WeightsF32};
use crate::modelgrad::{dsilu, layernorm, layernorm_bwd, linb, linb_bwd, patchify, rmsnorm, rmsnorm_dw, silu, timestep_embedding, Cfg, ModelGradsF32, ModelWeightsF32, TDIM, TH};

/// One training batch (host f64), mirroring the reference.
pub struct Batch {
    pub latent: Vec<f64>,
    pub cap: Vec<f64>,
    pub t: f64,
    pub img_cos: Vec<f64>,
    pub img_sin: Vec<f64>,
    pub cap_cos: Vec<f64>,
    pub cap_sin: Vec<f64>,
    pub target: Vec<f64>,
}

/// Persistent device trainer: owns the GPU block engine sized for the model.
pub struct DeviceTrainer {
    eng: BlockDev,
    cfg: Cfg,
}

fn to32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}
pub(crate) fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Saved state of the wrapper front (timestep MLP, embedders, refiners) that the
/// front backward needs. Produced by [`DeviceTrainer::front_fwd`].
pub(crate) struct Front {
    cvec: Vec<f64>,
    c32: Vec<f32>,
    te: Vec<f64>,
    h0pre: Vec<f64>,
    h0: Vec<f64>,
    patches: Vec<f64>,
    capn: Vec<f64>,
    inv_capn: Vec<f64>,
    noise_in: Vec<Vec<f32>>,
    ctx_in: Vec<Vec<f32>>,
    ic: Vec<f32>,
    is: Vec<f32>,
    cc: Vec<f32>,
    cs: Vec<f32>,
    uni_cos: Vec<f32>,
    uni_sin: Vec<f32>,
}

impl Front {
    pub(crate) fn c32(&self) -> &[f32] {
        &self.c32
    }
    pub(crate) fn cvec(&self) -> &[f64] {
        &self.cvec
    }
    pub(crate) fn uni_cos(&self) -> &[f32] {
        &self.uni_cos
    }
    pub(crate) fn uni_sin(&self) -> &[f32] {
        &self.uni_sin
    }
}

/// The unified RoPE tables `([img_cos‖cap_cos], [img_sin‖cap_sin])` as f32 — what
/// a threaded stage rebuilds from its own microbatch (they aren't in the boundary).
pub(crate) fn uni_rope(b: &Batch) -> (Vec<f32>, Vec<f32>) {
    let mut cos = to32(&b.img_cos);
    cos.extend(to32(&b.cap_cos));
    let mut sin = to32(&b.img_sin);
    sin.extend(to32(&b.cap_sin));
    (cos, sin)
}

/// Accumulate `src` into `dst` element-wise (grad accumulation across microbatches).
pub(crate) fn grad_add(dst: &mut GradsF32, src: &GradsF32) {
    let pairs: [(&mut Vec<f32>, &Vec<f32>); 15] = [
        (&mut dst.wq, &src.wq), (&mut dst.wk, &src.wk), (&mut dst.wv, &src.wv), (&mut dst.wo, &src.wo),
        (&mut dst.w1, &src.w1), (&mut dst.w2, &src.w2), (&mut dst.w3, &src.w3), (&mut dst.nq, &src.nq), (&mut dst.nk, &src.nk),
        (&mut dst.an1, &src.an1), (&mut dst.an2, &src.an2), (&mut dst.fn1, &src.fn1), (&mut dst.fn2, &src.fn2),
        (&mut dst.adaln_w, &src.adaln_w), (&mut dst.adaln_b, &src.adaln_b),
    ];
    for (d, s) in pairs {
        for (a, b) in d.iter_mut().zip(s) {
            *a += b;
        }
    }
}

fn vadd(dst: &mut [f64], src: &[f64]) {
    for (a, b) in dst.iter_mut().zip(src) {
        *a += b;
    }
}

pub(crate) fn front_grad_add(d: &mut FrontGrads, s: &FrontGrads) {
    vadd(&mut d.t0_w, &s.t0_w);
    vadd(&mut d.t0_b, &s.t0_b);
    vadd(&mut d.t2_w, &s.t2_w);
    vadd(&mut d.t2_b, &s.t2_b);
    vadd(&mut d.xemb_w, &s.xemb_w);
    vadd(&mut d.xemb_b, &s.xemb_b);
    vadd(&mut d.capn_w, &s.capn_w);
    vadd(&mut d.cap1_w, &s.cap1_w);
    vadd(&mut d.cap1_b, &s.cap1_b);
    for (a, b) in d.noise_ref.iter_mut().zip(&s.noise_ref) {
        grad_add(a, b);
    }
    for (a, b) in d.ctx_ref.iter_mut().zip(&s.ctx_ref) {
        grad_add(a, b);
    }
}

pub(crate) fn back_grad_add(d: &mut BackGrads, s: &BackGrads) {
    vadd(&mut d.fadaln_w, &s.fadaln_w);
    vadd(&mut d.fadaln_b, &s.fadaln_b);
    vadd(&mut d.flin_w, &s.flin_w);
    vadd(&mut d.flin_b, &s.flin_b);
}

/// Accumulate a slice's per-layer grads (`grad_add` element-wise).
pub(crate) fn main_grad_add(d: &mut [GradsF32], s: &[GradsF32]) {
    for (a, b) in d.iter_mut().zip(s) {
        grad_add(a, b);
    }
}

/// Saved state of the final layer, for its backward.
pub(crate) struct Back {
    silu_c: Vec<f64>,
    normed: Vec<f64>,
    inv_ln: Vec<f64>,
    scale: Vec<f64>,
    uni: Vec<f64>,
}

/// Front backward's grads (everything before the main layers).
pub(crate) struct FrontGrads {
    t0_w: Vec<f64>,
    t0_b: Vec<f64>,
    t2_w: Vec<f64>,
    t2_b: Vec<f64>,
    xemb_w: Vec<f64>,
    xemb_b: Vec<f64>,
    capn_w: Vec<f64>,
    cap1_w: Vec<f64>,
    cap1_b: Vec<f64>,
    noise_ref: Vec<GradsF32>,
    ctx_ref: Vec<GradsF32>,
}

/// Final-layer backward's grads.
pub(crate) struct BackGrads {
    fadaln_w: Vec<f64>,
    fadaln_b: Vec<f64>,
    flin_w: Vec<f64>,
    flin_b: Vec<f64>,
}

impl DeviceTrainer {
    pub fn new(cfg: Cfg) -> DeviceTrainer {
        let eng = BlockDev::new(cfg.ntot(), cfg.dim, cfg.nh);
        DeviceTrainer { eng, cfg }
    }

    /// Build over a pre-made engine (e.g. one card of a `BlockDev::new_multi`
    /// group), so a pipeline can place each stage on its own GPU.
    pub fn with_engine(cfg: Cfg, eng: BlockDev) -> DeviceTrainer {
        DeviceTrainer { eng, cfg }
    }

    fn dims(&self, t: usize) -> Dims {
        Dims::new(t, self.cfg.dim, self.cfg.nh)
    }

    /// Full forward+backward for one batch (single device). Returns `(loss, grads)`.
    pub fn grads(&self, w: &ModelWeightsF32, b: &Batch) -> (f64, ModelGradsF32) {
        let (uni, front) = self.front_fwd(w, b);
        let (uni, main_in) = self.main_fwd(&w.main, uni, &front);
        let (loss, dpred, back) = self.back_fwd(w, &uni, &front.cvec, b);
        let (d_uni, dc, bg) = self.back_bwd(w, &back, &dpred, &front.cvec);
        let (d_uni, dc, main_g) = self.main_bwd(&w.main, &main_in, &front, &d_uni, dc);
        let fg = self.front_bwd(w, b, &front, &d_uni, dc);
        (loss, assemble(fg, bg, main_g))
    }

    /// Same result as [`Self::grads`], but the main-layer stack is **cut** at
    /// `cut` and the residual crosses the split through a flat `[uni ‖ c]`
    /// boundary (forward) / `[d_uni ‖ dc]` (backward) — exactly what a pipeline
    /// stage boundary carries. Proves the boundary slab is complete and that the
    /// two halves are independent (each could stream its slice on its own card,
    /// weights in RAM — the memory-safe path for the full 6B). Bit-identical to
    /// `grads` (same ops/order; only a host round-trip is inserted at the cut).
    pub fn grads_pipelined(&self, w: &ModelWeightsF32, b: &Batch, cut: usize) -> (f64, ModelGradsF32) {
        let cdim = self.cfg.cdim();
        let (uni, front) = self.front_fwd(w, b);
        // stage 0: main[0, cut)
        let (uni, in0) = self.main_fwd(&w.main[..cut], uni, &front);
        // ---- boundary (forward): [uni ‖ c] host-staged to stage 1 ----
        let mut fwd_boundary = uni;
        fwd_boundary.extend_from_slice(&front.c32);
        let (uni, c_carry) = fwd_boundary.split_at(fwd_boundary.len() - cdim);
        let (uni, c_carry) = (uni.to_vec(), c_carry.to_vec());
        debug_assert_eq!(c_carry, front.c32, "boundary must carry c intact");
        // stage 1: main[cut, end)
        let (uni, in1) = self.main_fwd(&w.main[cut..], uni, &front);
        let (loss, dpred, back) = self.back_fwd(w, &uni, &front.cvec, b);

        // backward: head → stage 1 → boundary → stage 0 → front
        let (d_uni, dc, bg) = self.back_bwd(w, &back, &dpred, &front.cvec);
        let (d_uni, dc, mut main_g1) = self.main_bwd(&w.main[cut..], &in1, &front, &d_uni, dc);
        // ---- boundary (backward): [d_uni ‖ dc] host-staged back to stage 0 ----
        let mut bwd_boundary = d_uni;
        bwd_boundary.extend(dc.iter().map(|&x| x as f32));
        let (d_uni, dc_bytes) = bwd_boundary.split_at(bwd_boundary.len() - cdim);
        let (d_uni, dc) = (d_uni.to_vec(), dc_bytes.iter().map(|&x| x as f64).collect::<Vec<f64>>());
        let (d_uni, dc, mut main_g0) = self.main_bwd(&w.main[..cut], &in0, &front, &d_uni, dc);
        let fg = self.front_bwd(w, b, &front, &d_uni, dc);

        main_g0.append(&mut main_g1);
        (loss, assemble(fg, bg, main_g0))
    }

    // ---- phases (each stage of a pipeline runs a subset) ----

    /// Wrapper front: timestep→c, embed image/caption, refiners, unify. Returns
    /// the unified residual `[ntot·dim]` and the saved state for its backward.
    pub(crate) fn front_fwd(&self, w: &ModelWeightsF32, b: &Batch) -> (Vec<f32>, Front) {
        let c = &self.cfg;
        let (dim, cdim, pd) = (c.dim, c.cdim(), c.patch_dim());
        let (n_img, ncap) = (c.n_img(), c.ncap);
        let te = timestep_embedding(b.t * c.t_scale);
        let h0pre = linb(&te, 1, TDIM, &w.t0_w, &w.t0_b, TH);
        let h0: Vec<f64> = h0pre.iter().map(|&v| silu(v)).collect();
        let cvec = linb(&h0, 1, TH, &w.t2_w, &w.t2_b, cdim);
        let c32 = to32(&cvec);
        let patches = patchify(&b.latent, c);
        let img = linb(&patches, n_img, pd, &w.xemb_w, &w.xemb_b, dim);
        let (capn, inv_capn) = rmsnorm(&b.cap, ncap, c.cap_feat_dim, &w.capn_w);
        let capt = linb(&capn, ncap, c.cap_feat_dim, &w.cap1_w, &w.cap1_b, dim);
        let (ic, is) = (to32(&b.img_cos), to32(&b.img_sin));
        let (cc, cs) = (to32(&b.cap_cos), to32(&b.cap_sin));
        let mut img32 = to32(&img);
        let mut noise_in = Vec::new();
        for bw in &w.noise_ref {
            noise_in.push(img32.clone());
            img32 = self.eng.forward(bw, self.dims(n_img), &img32, &c32, &ic, &is, true);
        }
        let mut capt32 = to32(&capt);
        let mut ctx_in = Vec::new();
        for bw in &w.ctx_ref {
            ctx_in.push(capt32.clone());
            capt32 = self.eng.forward(bw, self.dims(ncap), &capt32, &c32, &cc, &cs, false);
        }
        let mut uni32 = img32;
        uni32.extend_from_slice(&capt32);
        let mut uni_cos = ic.clone();
        uni_cos.extend_from_slice(&cc);
        let mut uni_sin = is.clone();
        uni_sin.extend_from_slice(&cs);
        (uni32, Front { cvec, c32, te, h0pre, h0, patches, capn, inv_capn, noise_in, ctx_in, ic, is, cc, cs, uni_cos, uni_sin })
    }

    /// Run a contiguous slice of main layers on the unified residual, saving each
    /// block's input for the backward. `c32`/`uni_cos`/`uni_sin` are the shared
    /// conditioning + RoPE tables (from the front, or carried in the boundary).
    /// Returns `(uni_out, inputs)`.
    pub fn main_fwd_ctx(&self, main: &[WeightsF32], mut uni: Vec<f32>, c32: &[f32], uni_cos: &[f32], uni_sin: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let ntot = self.cfg.ntot();
        let mut inputs = Vec::with_capacity(main.len());
        for bw in main {
            inputs.push(uni.clone());
            uni = self.eng.forward(bw, self.dims(ntot), &uni, c32, uni_cos, uni_sin, true);
        }
        (uni, inputs)
    }
    fn main_fwd(&self, main: &[WeightsF32], uni: Vec<f32>, f: &Front) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.main_fwd_ctx(main, uni, &f.c32, &f.uni_cos, &f.uni_sin)
    }

    /// Final layer + flow-matching loss. `cvec` is the conditioning (f64); a
    /// threaded head takes it from the boundary `c`, a monolithic run from `Front`.
    /// Returns `(loss, dpred, Back)`.
    pub(crate) fn back_fwd(&self, w: &ModelWeightsF32, uni32: &[f32], cvec: &[f64], b: &Batch) -> (f64, Vec<f64>, Back) {
        let c = &self.cfg;
        let (dim, cdim, pd) = (c.dim, c.cdim(), c.patch_dim());
        let (n_img, ntot) = (c.n_img(), c.ntot());
        let uni = to64(uni32);
        let silu_c: Vec<f64> = cvec.iter().map(|&v| silu(v)).collect();
        let adaln = linb(&silu_c, 1, cdim, &w.fadaln_w, &w.fadaln_b, dim);
        let scale: Vec<f64> = adaln.iter().map(|&v| 1.0 + v).collect();
        let (normed, inv_ln) = layernorm(&uni, ntot, dim);
        let mut scaled = vec![0f64; ntot * dim];
        for r in 0..ntot {
            for cc2 in 0..dim {
                scaled[r * dim + cc2] = normed[r * dim + cc2] * scale[cc2];
            }
        }
        let final_out = linb(&scaled, ntot, dim, &w.flin_w, &w.flin_b, pd);
        let pred = &final_out[..n_img * pd];
        let n = pred.len() as f64;
        let mut loss = 0.0;
        let mut dpred = vec![0f64; pred.len()];
        for i in 0..pred.len() {
            let e = pred[i] - b.target[i];
            loss += e * e / n;
            dpred[i] = 2.0 * e / n;
        }
        (loss, dpred, Back { silu_c, normed, inv_ln, scale, uni })
    }

    /// Final-layer backward. `cvec` is the conditioning (f64). Returns
    /// `(d_uni, dc, back_grads)`.
    pub(crate) fn back_bwd(&self, w: &ModelWeightsF32, bk: &Back, dpred: &[f64], cvec: &[f64]) -> (Vec<f32>, Vec<f64>, BackGrads) {
        let c = &self.cfg;
        let (dim, cdim, pd) = (c.dim, c.cdim(), c.patch_dim());
        let (n_img, ntot) = (c.n_img(), c.ntot());
        let mut dc = vec![0f64; cdim];
        let mut d_final_out = vec![0f64; ntot * pd];
        d_final_out[..n_img * pd].copy_from_slice(dpred);
        let (d_scaled, g_flin_w, g_flin_b) = linb_bwd(&scaledfrom(&bk.normed, &bk.scale, ntot, dim), ntot, dim, &w.flin_w, pd, &d_final_out);
        let mut d_normed = vec![0f64; ntot * dim];
        let mut d_scale = vec![0f64; dim];
        for r in 0..ntot {
            for cc2 in 0..dim {
                d_normed[r * dim + cc2] = d_scaled[r * dim + cc2] * bk.scale[cc2];
                d_scale[cc2] += d_scaled[r * dim + cc2] * bk.normed[r * dim + cc2];
            }
        }
        let d_uni_host = layernorm_bwd(&bk.uni, ntot, dim, &bk.inv_ln, &d_normed);
        let (d_silu_c, g_fadaln_w, g_fadaln_b) = linb_bwd(&bk.silu_c, 1, cdim, &w.fadaln_w, dim, &d_scale);
        for j in 0..cdim {
            dc[j] += d_silu_c[j] * dsilu(cvec[j]);
        }
        (to32(&d_uni_host), dc, BackGrads { fadaln_w: g_fadaln_w, fadaln_b: g_fadaln_b, flin_w: g_flin_w, flin_b: g_flin_b })
    }

    /// Backward through a slice of main layers (reverse), accumulating `dc`.
    /// Returns `(d_uni_out, dc_out, grads_in_forward_order)`.
    pub fn main_bwd_ctx(&self, main: &[WeightsF32], inputs: &[Vec<f32>], c32: &[f32], uni_cos: &[f32], uni_sin: &[f32], d_uni: &[f32], mut dc: Vec<f64>) -> (Vec<f32>, Vec<f64>, Vec<GradsF32>) {
        let (cdim, ntot) = (self.cfg.cdim(), self.cfg.ntot());
        let mut d = d_uni.to_vec();
        let mut g: Vec<GradsF32> = Vec::with_capacity(main.len());
        for (bw, inp) in main.iter().zip(inputs).rev() {
            let gg = self.eng.backward(bw, self.dims(ntot), inp, c32, uni_cos, uni_sin, true, &d);
            d = gg.dx.clone();
            for (j, dcj) in dc.iter_mut().enumerate().take(cdim) {
                *dcj += gg.dc[j] as f64;
            }
            g.push(gg);
        }
        g.reverse();
        (d, dc, g)
    }
    fn main_bwd(&self, main: &[WeightsF32], inputs: &[Vec<f32>], f: &Front, d_uni: &[f32], dc: Vec<f64>) -> (Vec<f32>, Vec<f64>, Vec<GradsF32>) {
        self.main_bwd_ctx(main, inputs, &f.c32, &f.uni_cos, &f.uni_sin, d_uni, dc)
    }

    /// Wrapper-front backward: split the residual, refiners, embedders, timestep
    /// MLP (consuming the accumulated `dc`). Returns `(front_grads, noise_g, ctx_g)`.
    pub(crate) fn front_bwd(&self, w: &ModelWeightsF32, b: &Batch, f: &Front, d_uni: &[f32], mut dc: Vec<f64>) -> FrontGrads {
        let c = &self.cfg;
        let (dim, cdim, pd) = (c.dim, c.cdim(), c.patch_dim());
        let (n_img, ncap) = (c.n_img(), c.ncap);
        let mut d_img32 = d_uni[..n_img * dim].to_vec();
        let mut d_capt32 = d_uni[n_img * dim..].to_vec();
        let mut ctx_g: Vec<GradsF32> = Vec::new();
        for (bw, inp) in w.ctx_ref.iter().zip(&f.ctx_in).rev() {
            let g = self.eng.backward(bw, self.dims(ncap), inp, &f.c32, &f.cc, &f.cs, false, &d_capt32);
            d_capt32 = g.dx.clone();
            ctx_g.push(g);
        }
        ctx_g.reverse();
        let mut noise_g: Vec<GradsF32> = Vec::new();
        for (bw, inp) in w.noise_ref.iter().zip(&f.noise_in).rev() {
            let g = self.eng.backward(bw, self.dims(n_img), inp, &f.c32, &f.ic, &f.is, true, &d_img32);
            d_img32 = g.dx.clone();
            for (j, dcj) in dc.iter_mut().enumerate().take(cdim) {
                *dcj += g.dc[j] as f64;
            }
            noise_g.push(g);
        }
        noise_g.reverse();
        let (_dp, g_xemb_w, g_xemb_b) = linb_bwd(&f.patches, n_img, pd, &w.xemb_w, dim, &to64(&d_img32));
        let (d_capn, g_cap1_w, g_cap1_b) = linb_bwd(&f.capn, ncap, c.cap_feat_dim, &w.cap1_w, dim, &to64(&d_capt32));
        let g_capn_w = rmsnorm_dw(&b.cap, ncap, c.cap_feat_dim, &f.inv_capn, &d_capn);
        let (d_h0, g_t2_w, g_t2_b) = linb_bwd(&f.h0, 1, TH, &w.t2_w, cdim, &dc);
        let mut d_h0pre = vec![0f64; TH];
        for i in 0..TH {
            d_h0pre[i] = d_h0[i] * dsilu(f.h0pre[i]);
        }
        let (_dte, g_t0_w, g_t0_b) = linb_bwd(&f.te, 1, TDIM, &w.t0_w, TH, &d_h0pre);
        FrontGrads {
            t0_w: g_t0_w, t0_b: g_t0_b, t2_w: g_t2_w, t2_b: g_t2_b,
            xemb_w: g_xemb_w, xemb_b: g_xemb_b, capn_w: g_capn_w, cap1_w: g_cap1_w, cap1_b: g_cap1_b,
            noise_ref: noise_g, ctx_ref: ctx_g,
        }
    }
}

/// Recompute `scaled = normed ⊙ scale` (the final-layer input) for its backward.
fn scaledfrom(normed: &[f64], scale: &[f64], ntot: usize, dim: usize) -> Vec<f64> {
    let mut s = vec![0f64; ntot * dim];
    for r in 0..ntot {
        for cc in 0..dim {
            s[r * dim + cc] = normed[r * dim + cc] * scale[cc];
        }
    }
    s
}

/// Reassemble a [`ModelGradsF32`] from the phase grads.
pub(crate) fn assemble(fg: FrontGrads, bg: BackGrads, main_g: Vec<GradsF32>) -> ModelGradsF32 {
    ModelGradsF32 {
        t0_w: fg.t0_w, t0_b: fg.t0_b, t2_w: fg.t2_w, t2_b: fg.t2_b,
        xemb_w: fg.xemb_w, xemb_b: fg.xemb_b, capn_w: fg.capn_w, cap1_w: fg.cap1_w, cap1_b: fg.cap1_b,
        noise_ref: fg.noise_ref, ctx_ref: fg.ctx_ref, main: main_g,
        fadaln_w: bg.fadaln_w, fadaln_b: bg.fadaln_b, flin_w: bg.flin_w, flin_b: bg.flin_b,
    }
}

// ============================================================================
// impl Model — wire the Z-Image DiT into brain's generic training machinery.
//
// Once Z-Image is a `Model`, it inherits data-parallel / multi-machine / federated
// training through `model::{DdpOptimizer, NetworkCollective, federated_average}`
// with no zimage-specific code — the same seam every brain model rides. The
// adapter is thin: named-tensor views bridge `ModelWeightsF32`/`ModelGradsF32` to the
// trait's `read_weight`/`write_weight`/`read_grad`, and forward+backward delegate
// to the validated `DeviceTrainer` (GPU block stack + host wrapper).
// ============================================================================

use model::{Batch as MBatch, Model, ModelConfig};

/// The 15 trainable tensors of one block, in the canonical order used everywhere.
const BLOCK_FIELDS: [&str; 15] = ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "nq", "nk", "an1", "an2", "fn1", "fn2", "adaln_w", "adaln_b"];

fn block_size(c: &Cfg, f: &str) -> usize {
    let (dim, hd, hidden, cdim) = (c.dim, c.dim / c.nh, c.dim * 8 / 3, c.cdim());
    match f {
        "wq" | "wk" | "wv" | "wo" => dim * dim,
        "w1" | "w3" => hidden * dim,
        "w2" => dim * hidden,
        "nq" | "nk" => hd,
        "an1" | "an2" | "fn1" | "fn2" => dim,
        "adaln_w" => 4 * dim * cdim,
        "adaln_b" => 4 * dim,
        _ => unreachable!("bad block field {f}"),
    }
}

/// `(name, numel)` for every trainable tensor, in a stable canonical order:
/// timestep MLP, embedders, then noise/context/main blocks, then final layer.
fn param_layout(c: &Cfg) -> Vec<(String, usize)> {
    let (dim, cdim, pd, cf) = (c.dim, c.cdim(), c.patch_dim(), c.cap_feat_dim);
    let mut v: Vec<(String, usize)> = [
        ("t0_w", 1024 * 256), ("t0_b", 1024), ("t2_w", cdim * 1024), ("t2_b", cdim),
        ("xemb_w", dim * pd), ("xemb_b", dim), ("capn_w", cf), ("cap1_w", dim * cf), ("cap1_b", dim),
    ]
    .into_iter()
    .map(|(n, s)| (n.to_string(), s))
    .collect();
    for (grp, cnt) in [("noise_ref", c.n_refiner), ("ctx_ref", c.n_refiner), ("main", c.n_layers)] {
        for i in 0..cnt {
            for f in BLOCK_FIELDS {
                v.push((format!("{grp}.{i}.{f}"), block_size(c, f)));
            }
        }
    }
    v.extend([("fadaln_w", dim * cdim), ("fadaln_b", dim), ("flin_w", pd * dim), ("flin_b", pd)].map(|(n, s)| (n.to_string(), s)));
    v
}

// ---- hybrid name-dispatch accessors (wrapper tensors f64, blocks f32) ----

fn block_field_f32<'a>(b: &'a WeightsF32, f: &str) -> &'a Vec<f32> {
    match f {
        "wq" => &b.wq, "wk" => &b.wk, "wv" => &b.wv, "wo" => &b.wo, "w1" => &b.w1, "w2" => &b.w2, "w3" => &b.w3,
        "nq" => &b.nq, "nk" => &b.nk, "an1" => &b.an1, "an2" => &b.an2, "fn1" => &b.fn1, "fn2" => &b.fn2,
        "adaln_w" => &b.adaln_w, "adaln_b" => &b.adaln_b, _ => panic!("bad field {f}"),
    }
}
fn block_field_f32_mut<'a>(b: &'a mut WeightsF32, f: &str) -> &'a mut Vec<f32> {
    match f {
        "wq" => &mut b.wq, "wk" => &mut b.wk, "wv" => &mut b.wv, "wo" => &mut b.wo, "w1" => &mut b.w1, "w2" => &mut b.w2, "w3" => &mut b.w3,
        "nq" => &mut b.nq, "nk" => &mut b.nk, "an1" => &mut b.an1, "an2" => &mut b.an2, "fn1" => &mut b.fn1, "fn2" => &mut b.fn2,
        "adaln_w" => &mut b.adaln_w, "adaln_b" => &mut b.adaln_b, _ => panic!("bad field {f}"),
    }
}
fn grad_block_field_f32<'a>(g: &'a GradsF32, f: &str) -> &'a Vec<f32> {
    match f {
        "wq" => &g.wq, "wk" => &g.wk, "wv" => &g.wv, "wo" => &g.wo, "w1" => &g.w1, "w2" => &g.w2, "w3" => &g.w3,
        "nq" => &g.nq, "nk" => &g.nk, "an1" => &g.an1, "an2" => &g.an2, "fn1" => &g.fn1, "fn2" => &g.fn2,
        "adaln_w" => &g.adaln_w, "adaln_b" => &g.adaln_b, _ => panic!("bad field {f}"),
    }
}

fn block_group<'a>(m: &'a ModelWeightsF32, grp: &str) -> &'a [WeightsF32] {
    match grp {
        "noise_ref" => &m.noise_ref, "ctx_ref" => &m.ctx_ref, "main" => &m.main, _ => panic!("bad group {grp}"),
    }
}
fn block_group_mut<'a>(m: &'a mut ModelWeightsF32, grp: &str) -> &'a mut [WeightsF32] {
    match grp {
        "noise_ref" => &mut m.noise_ref, "ctx_ref" => &mut m.ctx_ref, "main" => &mut m.main, _ => panic!("bad group {grp}"),
    }
}
fn grad_group<'a>(g: &'a ModelGradsF32, grp: &str) -> &'a [GradsF32] {
    match grp {
        "noise_ref" => &g.noise_ref, "ctx_ref" => &g.ctx_ref, "main" => &g.main, _ => panic!("bad group {grp}"),
    }
}

/// The wrapper (f64) tensor by name, or `None` if it is a block tensor.
fn wrapper_ref<'a>(m: &'a ModelWeightsF32, n: &str) -> Option<&'a Vec<f64>> {
    Some(match n {
        "t0_w" => &m.t0_w, "t0_b" => &m.t0_b, "t2_w" => &m.t2_w, "t2_b" => &m.t2_b,
        "xemb_w" => &m.xemb_w, "xemb_b" => &m.xemb_b, "capn_w" => &m.capn_w, "cap1_w" => &m.cap1_w, "cap1_b" => &m.cap1_b,
        "fadaln_w" => &m.fadaln_w, "fadaln_b" => &m.fadaln_b, "flin_w" => &m.flin_w, "flin_b" => &m.flin_b,
        _ => return None,
    })
}
fn wrapper_mut<'a>(m: &'a mut ModelWeightsF32, n: &str) -> Option<&'a mut Vec<f64>> {
    Some(match n {
        "t0_w" => &mut m.t0_w, "t0_b" => &mut m.t0_b, "t2_w" => &mut m.t2_w, "t2_b" => &mut m.t2_b,
        "xemb_w" => &mut m.xemb_w, "xemb_b" => &mut m.xemb_b, "capn_w" => &mut m.capn_w, "cap1_w" => &mut m.cap1_w, "cap1_b" => &mut m.cap1_b,
        "fadaln_w" => &mut m.fadaln_w, "fadaln_b" => &mut m.fadaln_b, "flin_w" => &mut m.flin_w, "flin_b" => &mut m.flin_b,
        _ => return None,
    })
}
fn grad_wrapper_ref<'a>(g: &'a ModelGradsF32, n: &str) -> Option<&'a Vec<f64>> {
    Some(match n {
        "t0_w" => &g.t0_w, "t0_b" => &g.t0_b, "t2_w" => &g.t2_w, "t2_b" => &g.t2_b,
        "xemb_w" => &g.xemb_w, "xemb_b" => &g.xemb_b, "capn_w" => &g.capn_w, "cap1_w" => &g.cap1_w, "cap1_b" => &g.cap1_b,
        "fadaln_w" => &g.fadaln_w, "fadaln_b" => &g.fadaln_b, "flin_w" => &g.flin_w, "flin_b" => &g.flin_b,
        _ => return None,
    })
}

/// Read a named tensor as f32 (wrapper: f64→f32; block: direct).
fn weight_get(m: &ModelWeightsF32, name: &str) -> Vec<f32> {
    if let Some(w) = wrapper_ref(m, name) {
        return w.iter().map(|&x| x as f32).collect();
    }
    let (grp, i, field) = parse_block(name);
    block_field_f32(&block_group(m, grp)[i], field).clone()
}
/// Write a named tensor from f32 (wrapper: f32→f64; block: direct).
fn weight_set(m: &mut ModelWeightsF32, name: &str, data: &[f32]) {
    if let Some(w) = wrapper_mut(m, name) {
        *w = data.iter().map(|&x| x as f64).collect();
        return;
    }
    let (grp, i, field) = parse_block(name);
    block_field_f32_mut(&mut block_group_mut(m, grp)[i], field).copy_from_slice(data);
}
/// Read a named gradient as f32.
fn grad_get(g: &ModelGradsF32, name: &str) -> Vec<f32> {
    if let Some(w) = grad_wrapper_ref(g, name) {
        return w.iter().map(|&x| x as f32).collect();
    }
    let (grp, i, field) = parse_block(name);
    grad_block_field_f32(&grad_group(g, grp)[i], field).clone()
}
/// Parse `"grp.i.field"` → `(grp, i, field)`.
fn parse_block(name: &str) -> (&str, usize, &str) {
    let mut it = name.splitn(3, '.');
    let grp = it.next().unwrap();
    let i: usize = it.next().unwrap().parse().unwrap();
    let field = it.next().unwrap();
    (grp, i, field)
}

fn zero_block(c: &Cfg) -> WeightsF32 {
    let z = |n: usize| vec![0f32; n];
    WeightsF32 {
        wq: z(block_size(c, "wq")), wk: z(block_size(c, "wk")), wv: z(block_size(c, "wv")), wo: z(block_size(c, "wo")),
        w1: z(block_size(c, "w1")), w2: z(block_size(c, "w2")), w3: z(block_size(c, "w3")),
        nq: z(block_size(c, "nq")), nk: z(block_size(c, "nk")),
        an1: z(block_size(c, "an1")), an2: z(block_size(c, "an2")), fn1: z(block_size(c, "fn1")), fn2: z(block_size(c, "fn2")),
        adaln_w: z(block_size(c, "adaln_w")), adaln_b: z(block_size(c, "adaln_b")),
    }
}

/// All-zero weights of the right shape (filled by `write_weight` / `init`).
pub fn zero_weights(c: &Cfg) -> ModelWeightsF32 {
    let (dim, cdim, pd, cf) = (c.dim, c.cdim(), c.patch_dim(), c.cap_feat_dim);
    let z = |n: usize| vec![0f64; n];
    ModelWeightsF32 {
        t0_w: z(1024 * 256), t0_b: z(1024), t2_w: z(cdim * 1024), t2_b: z(cdim),
        xemb_w: z(dim * pd), xemb_b: z(dim), capn_w: z(cf), cap1_w: z(dim * cf), cap1_b: z(dim),
        noise_ref: (0..c.n_refiner).map(|_| zero_block(c)).collect(),
        ctx_ref: (0..c.n_refiner).map(|_| zero_block(c)).collect(),
        main: (0..c.n_layers).map(|_| zero_block(c)).collect(),
        fadaln_w: z(dim * cdim), fadaln_b: z(dim), flin_w: z(pd * dim), flin_b: z(pd),
    }
}

impl ModelConfig for Cfg {
    fn param_list(&self) -> Vec<(String, usize)> {
        param_layout(self)
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "dim": self.dim, "nh": self.nh, "n_layers": self.n_layers, "n_refiner": self.n_refiner,
            "cap_feat_dim": self.cap_feat_dim, "in_channels": self.in_channels, "patch": self.patch,
            "h": self.h, "w": self.w, "ncap": self.ncap, "t_scale": self.t_scale,
        })
    }
    fn from_json(v: &serde_json::Value) -> Self {
        let u = |k: &str| v[k].as_u64().unwrap() as usize;
        Cfg {
            dim: u("dim"), nh: u("nh"), n_layers: u("n_layers"), n_refiner: u("n_refiner"),
            cap_feat_dim: u("cap_feat_dim"), in_channels: u("in_channels"), patch: u("patch"),
            h: u("h"), w: u("w"), ncap: u("ncap"), t_scale: v["t_scale"].as_f64().unwrap(),
        }
    }
    fn vocab(&self) -> u32 {
        0
    }
    fn block_size(&self) -> u32 {
        0
    }
    fn finalize_for_dataset(self, _v: u32, _b: u32) -> Self {
        self
    }
}

/// The trainable Z-Image DiT as a brain [`Model`]: `DeviceTrainer` (GPU block
/// stack + host wrapper) behind named-tensor accessors + a gradient accumulator.
/// Load a diffusion batch with [`ZTrainModel::load_batch`]; `forward` runs the
/// full fwd+bwd on the GPU and stashes the grads, `backward` accumulates them.
pub struct ZTrainModel {
    cfg: Cfg,
    trainer: DeviceTrainer,
    weights: RefCell<ModelWeightsF32>,
    names: Vec<String>,
    idx: HashMap<String, usize>,
    grad_acc: RefCell<Vec<Vec<f32>>>,
    stash: RefCell<Option<ModelGradsF32>>,
    batch: RefCell<Option<Batch>>,
    loss: RefCell<f32>,
}

impl ZTrainModel {
    pub fn from_weights(cfg: Cfg, weights: ModelWeightsF32) -> ZTrainModel {
        let layout = param_layout(&cfg);
        let names: Vec<String> = layout.iter().map(|(n, _)| n.clone()).collect();
        let idx = names.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();
        let grad_acc = layout.iter().map(|(_, s)| vec![0f32; *s]).collect();
        ZTrainModel {
            cfg,
            trainer: DeviceTrainer::new(cfg),
            weights: RefCell::new(weights),
            names,
            idx,
            grad_acc: RefCell::new(grad_acc),
            stash: RefCell::new(None),
            batch: RefCell::new(None),
            loss: RefCell::new(0.0),
        }
    }

    /// Set the current training batch (diffusion inputs + target velocity).
    pub fn load_batch(&self, b: Batch) {
        *self.batch.borrow_mut() = Some(b);
    }
}

impl Model for ZTrainModel {
    type Config = Cfg;

    fn new(cfg: Cfg, _b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        let m = ZTrainModel::from_weights(cfg, zero_weights(&cfg));
        for (n, data) in init {
            m.write_weight(n, data);
        }
        m
    }
    fn init_weights(cfg: &Cfg, _seed: u64) -> HashMap<String, Vec<f32>> {
        param_layout(cfg).into_iter().map(|(n, s)| (n, vec![0f32; s])).collect()
    }
    fn config(&self) -> &Cfg {
        &self.cfg
    }
    fn set_batch(&self, _b: MBatch) {
        // Diffusion batches are set via `load_batch` (richer than the token/tensor
        // Batch enum); this keeps whatever was last loaded.
    }
    fn forward(&self) -> f32 {
        let b = self.batch.borrow();
        let b = b.as_ref().expect("ZTrainModel::forward before load_batch");
        let (loss, grads) = self.trainer.grads(&self.weights.borrow(), b);
        *self.stash.borrow_mut() = Some(grads);
        *self.loss.borrow_mut() = loss as f32;
        loss as f32
    }
    fn backward(&self) {
        let stash = self.stash.borrow();
        let g = stash.as_ref().expect("ZTrainModel::backward before forward");
        let mut acc = self.grad_acc.borrow_mut();
        for (name, slot) in self.names.iter().zip(acc.iter_mut()) {
            for (a, b) in slot.iter_mut().zip(grad_get(g, name)) {
                *a += b;
            }
        }
    }
    fn zero_grads(&self) {
        for slot in self.grad_acc.borrow_mut().iter_mut() {
            slot.iter_mut().for_each(|x| *x = 0.0);
        }
    }
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra: f32) {
        // The distributed path drives the optimiser via model::DdpOptimizer (grads
        // reduced through a Collective); a single-device local optimiser can be
        // added here later if needed.
    }
    fn poll_wait(&self) {}
    fn param_names(&self) -> Vec<String> {
        self.names.clone()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        weight_get(&self.weights.borrow(), name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        weight_set(&mut self.weights.borrow_mut(), name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        let i = self.idx[name];
        self.grad_acc.borrow()[i].clone()
    }
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, _path: &str) {}
    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }
}
