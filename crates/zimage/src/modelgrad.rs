// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full Z-Image S³-DiT **training** reference (host, f64): forward + analytic
//! backward for the whole model under a flow-matching (velocity-MSE) loss. This
//! is the end-to-end training loop's ground truth — the block backward
//! ([`crate::grad`]) chained across the noise/context refiners and main layers,
//! wrapped with the pieces a block doesn't have: the timestep MLP, the
//! image/caption embedders, and the adaLN final layer.
//!
//! Gradients flow into every trainable tensor and, crucially, the conditioning
//! grad `dc` is accumulated from every modulated block *and* the final layer and
//! routed back through the `t_embedder` MLP — the coupling that makes the whole
//! network train as one. Validated by `tests/model_grad.rs` (finite-difference
//! gradcheck) and `tests/model_overfit.rs` (loss → ~0).
//!
//! The loss is taken in patch space on the image tokens (velocity prediction);
//! the parameterless unpatchify reshape is omitted as it does not affect training
//! correctness. The RoPE tables are fixed inputs (position → cos/sin), exactly as
//! the block reference treats them.

use crate::grad::{backward as block_bwd, forward_m as block_fwd, Cache, Dims, Grads, Weights};

/// Minimal config for the training reference.
#[derive(Clone, Copy)]
pub struct Cfg {
    pub dim: usize,
    pub nh: usize,
    pub n_layers: usize,
    pub n_refiner: usize,
    pub cap_feat_dim: usize,
    pub in_channels: usize,
    pub patch: usize, // spatial patch size (f_patch = 1)
    pub h: usize,     // latent H, W (F = 1)
    pub w: usize,
    pub ncap: usize,
    pub t_scale: f64,
}

impl Cfg {
    pub fn cdim(&self) -> usize {
        self.dim.min(256)
    }
    pub fn n_img(&self) -> usize {
        (self.h / self.patch) * (self.w / self.patch)
    }
    pub fn patch_dim(&self) -> usize {
        self.patch * self.patch * self.in_channels
    }
    pub fn ntot(&self) -> usize {
        self.n_img() + self.ncap
    }
    fn dims(&self, t: usize) -> Dims {
        Dims::new(t, self.dim, self.nh)
    }
}

/// All trainable weights of the model (host f64).
#[derive(Clone)]
pub struct ModelWeights {
    pub t0_w: Vec<f64>, // [1024, 256]
    pub t0_b: Vec<f64>,
    pub t2_w: Vec<f64>, // [cdim, 1024]
    pub t2_b: Vec<f64>,
    pub xemb_w: Vec<f64>, // [dim, patch_dim]
    pub xemb_b: Vec<f64>,
    pub capn_w: Vec<f64>, // [cap_feat_dim] rmsnorm gain
    pub cap1_w: Vec<f64>, // [dim, cap_feat_dim]
    pub cap1_b: Vec<f64>,
    pub noise_ref: Vec<Weights>, // modulated
    pub ctx_ref: Vec<Weights>,   // UNmodulated
    pub main: Vec<Weights>,      // modulated
    pub fadaln_w: Vec<f64>,      // [dim, cdim]
    pub fadaln_b: Vec<f64>,
    pub flin_w: Vec<f64>, // [patch_dim, dim]
    pub flin_b: Vec<f64>,
}

/// Grads mirroring [`ModelWeights`].
#[derive(Clone)]
pub struct ModelGrads {
    pub t0_w: Vec<f64>,
    pub t0_b: Vec<f64>,
    pub t2_w: Vec<f64>,
    pub t2_b: Vec<f64>,
    pub xemb_w: Vec<f64>,
    pub xemb_b: Vec<f64>,
    pub capn_w: Vec<f64>,
    pub cap1_w: Vec<f64>,
    pub cap1_b: Vec<f64>,
    pub noise_ref: Vec<Grads>,
    pub ctx_ref: Vec<Grads>,
    pub main: Vec<Grads>,
    pub fadaln_w: Vec<f64>,
    pub fadaln_b: Vec<f64>,
    pub flin_w: Vec<f64>,
    pub flin_b: Vec<f64>,
}

const TDIM: usize = 256; // timestep sinusoid width
const TH: usize = 1024; // t_embedder hidden
const LN_EPS: f64 = 1e-6;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
fn dsilu(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

/// `y[r,o] = Σ_i x[r,i]·w[o,i] + b[o]`.
fn linb(x: &[f64], rows: usize, inp: usize, w: &[f64], b: &[f64], out: usize) -> Vec<f64> {
    let mut y = vec![0f64; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut a = b[o];
            for i in 0..inp {
                a += x[r * inp + i] * w[o * inp + i];
            }
            y[r * out + o] = a;
        }
    }
    y
}

/// Linear+bias backward: returns `(dx, dw, db)`.
fn linb_bwd(x: &[f64], rows: usize, inp: usize, w: &[f64], out: usize, dy: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut dx = vec![0f64; rows * inp];
    let mut dw = vec![0f64; out * inp];
    let mut db = vec![0f64; out];
    for r in 0..rows {
        for o in 0..out {
            let g = dy[r * out + o];
            db[o] += g;
            for i in 0..inp {
                dx[r * inp + i] += g * w[o * inp + i];
                dw[o * inp + i] += g * x[r * inp + i];
            }
        }
    }
    (dx, dw, db)
}

fn rmsnorm(x: &[f64], rows: usize, d: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0f64; rows * d];
    let mut inv = vec![0f64; rows];
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let ss = xr.iter().map(|v| v * v).sum::<f64>() / d as f64;
        let iv = 1.0 / (ss + 1e-5).sqrt();
        inv[r] = iv;
        for c in 0..d {
            y[r * d + c] = w[c] * xr[c] * iv;
        }
    }
    (y, inv)
}

/// RMSNorm gain grad only (input is data — no dx needed for the cap embedder).
fn rmsnorm_dw(x: &[f64], rows: usize, d: usize, inv: &[f64], dy: &[f64]) -> Vec<f64> {
    let mut dw = vec![0f64; d];
    for r in 0..rows {
        for c in 0..d {
            dw[c] += dy[r * d + c] * x[r * d + c] * inv[r];
        }
    }
    dw
}

fn timestep_embedding(t: f64) -> Vec<f64> {
    let half = TDIM / 2;
    let mut e = vec![0f64; TDIM];
    for k in 0..half {
        let freq = (-(10000f64.ln()) * k as f64 / half as f64).exp();
        let arg = t * freq;
        e[k] = arg.cos();
        e[half + k] = arg.sin();
    }
    e
}

/// `[C,F=1,H,W] -> [n_img, patch·patch·C]` (matches diffusers `_patchify_image`).
fn patchify(latent: &[f64], cfg: &Cfg) -> Vec<f64> {
    let (c, ps) = (cfg.in_channels, cfg.patch);
    let (ht, wt) = (cfg.h / ps, cfg.w / ps);
    let (fh, fw) = (cfg.h, cfg.w);
    let pd = ps * ps * c;
    let mut out = vec![0f64; ht * wt * pd];
    for hh in 0..ht {
        for ww in 0..wt {
            let tok = (hh * wt + ww) * pd;
            for ph in 0..ps {
                for pw in 0..ps {
                    for cc in 0..c {
                        let src = (cc * fh + (hh * ps + ph)) * fw + (ww * ps + pw);
                        out[tok + (ph * ps + pw) * c + cc] = latent[src];
                    }
                }
            }
        }
    }
    out
}

/// LayerNorm (no affine) forward, per row over `d`. Returns `(y, inv[rows])`.
fn layernorm(x: &[f64], rows: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0f64; rows * d];
    let mut inv = vec![0f64; rows];
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let mean = xr.iter().sum::<f64>() / d as f64;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
        let iv = 1.0 / (var + LN_EPS).sqrt();
        inv[r] = iv;
        for c in 0..d {
            y[r * d + c] = (xr[c] - mean) * iv;
        }
    }
    (y, inv)
}

/// LayerNorm (no affine) backward: `dx = inv·(dy − mean(dy) − xhat·mean(dy·xhat))`.
fn layernorm_bwd(x: &[f64], rows: usize, d: usize, inv: &[f64], dy: &[f64]) -> Vec<f64> {
    let mut dx = vec![0f64; rows * d];
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let iv = inv[r];
        let mean = xr.iter().sum::<f64>() / d as f64;
        let (mut mdy, mut mdyx) = (0.0, 0.0);
        let mut xhat = vec![0f64; d];
        for c in 0..d {
            xhat[c] = (xr[c] - mean) * iv;
            mdy += dy[r * d + c];
            mdyx += dy[r * d + c] * xhat[c];
        }
        mdy /= d as f64;
        mdyx /= d as f64;
        for c in 0..d {
            dx[r * d + c] = iv * (dy[r * d + c] - mdy - xhat[c] * mdyx);
        }
    }
    dx
}

/// Saved forward state for the backward pass.
pub struct ModelCache {
    c: Vec<f64>,       // conditioning [cdim]
    te: Vec<f64>,      // timestep sinusoid [256]
    h0pre: Vec<f64>,   // t_embedder hidden pre-activation [1024]
    h0: Vec<f64>,      // silu(h0pre) [1024]
    patches: Vec<f64>, // [n_img, patch_dim]
    cap_in: Vec<f64>,  // rmsnorm INPUT (data) [ncap, cap_feat_dim]
    capn: Vec<f64>,    // rmsnorm(cap) [ncap, cap_feat_dim]
    inv_capn: Vec<f64>,
    noise_c: Vec<Cache>,
    ctx_c: Vec<Cache>,
    main_c: Vec<Cache>,
    uni: Vec<f64>, // main-layer output [ntot, dim]
    silu_c: Vec<f64>,
    normed: Vec<f64>,
    inv_ln: Vec<f64>,
    scale: Vec<f64>, // 1 + adaln [dim]
}

/// Full forward. Returns `(pred[n_img·patch_dim], cache)`. `img_cos/img_sin`
/// size `[n_img·half]`, `cap_*` `[ncap·half]`; `half = (dim/nh)/2`.
#[allow(clippy::too_many_arguments)]
pub fn forward(cfg: &Cfg, w: &ModelWeights, latent: &[f64], cap: &[f64], t: f64, img_cos: &[f64], img_sin: &[f64], cap_cos: &[f64], cap_sin: &[f64]) -> (Vec<f64>, ModelCache) {
    let (dim, cdim) = (cfg.dim, cfg.cdim());
    let (n_img, ncap, ntot) = (cfg.n_img(), cfg.ncap, cfg.ntot());
    // timestep conditioning
    let te = timestep_embedding(t * cfg.t_scale);
    let h0pre = linb(&te, 1, TDIM, &w.t0_w, &w.t0_b, TH);
    let h0: Vec<f64> = h0pre.iter().map(|&v| silu(v)).collect();
    let c = linb(&h0, 1, TH, &w.t2_w, &w.t2_b, cdim);
    // embedders
    let patches = patchify(latent, cfg);
    let img = linb(&patches, n_img, cfg.patch_dim(), &w.xemb_w, &w.xemb_b, dim);
    let (capn, inv_capn) = rmsnorm(cap, ncap, cfg.cap_feat_dim, &w.capn_w);
    let capt = linb(&capn, ncap, cfg.cap_feat_dim, &w.cap1_w, &w.cap1_b, dim);
    // refiners
    let mut img = img;
    let mut noise_c = Vec::new();
    for bw in &w.noise_ref {
        let (o, ca) = block_fwd(cfg.dims(n_img), bw, &img, &c, img_cos, img_sin, true);
        img = o;
        noise_c.push(ca);
    }
    let mut capt = capt;
    let mut ctx_c = Vec::new();
    for bw in &w.ctx_ref {
        let (o, ca) = block_fwd(cfg.dims(ncap), bw, &capt, &c, cap_cos, cap_sin, false);
        capt = o;
        ctx_c.push(ca);
    }
    // unified [image, caption]
    let mut uni = img;
    uni.extend_from_slice(&capt);
    let mut uni_cos = img_cos.to_vec();
    uni_cos.extend_from_slice(cap_cos);
    let mut uni_sin = img_sin.to_vec();
    uni_sin.extend_from_slice(cap_sin);
    let mut main_c = Vec::new();
    for bw in &w.main {
        let (o, ca) = block_fwd(cfg.dims(ntot), bw, &uni, &c, &uni_cos, &uni_sin, true);
        uni = o;
        main_c.push(ca);
    }
    // final layer: LayerNorm(no affine) · (1 + adaLN(silu(c))) → linear
    let silu_c: Vec<f64> = c.iter().map(|&v| silu(v)).collect();
    let adaln = linb(&silu_c, 1, cdim, &w.fadaln_w, &w.fadaln_b, dim);
    let scale: Vec<f64> = adaln.iter().map(|&v| 1.0 + v).collect();
    let (normed, inv_ln) = layernorm(&uni, ntot, dim);
    let mut scaled = vec![0f64; ntot * dim];
    for r in 0..ntot {
        for cc in 0..dim {
            scaled[r * dim + cc] = normed[r * dim + cc] * scale[cc];
        }
    }
    let final_out = linb(&scaled, ntot, dim, &w.flin_w, &w.flin_b, cfg.patch_dim());
    let pred = final_out[..n_img * cfg.patch_dim()].to_vec();

    let cache = ModelCache {
        c, te, h0pre, h0, patches, cap_in: cap.to_vec(), capn, inv_capn, noise_c, ctx_c, main_c, uni, silu_c, normed, inv_ln, scale,
    };
    (pred, cache)
}

/// Velocity-MSE flow-matching loss + its `dpred`. `L = mean((pred − v)²)`.
pub fn loss(pred: &[f64], v_target: &[f64]) -> (f64, Vec<f64>) {
    let n = pred.len() as f64;
    let mut l = 0.0;
    let mut dpred = vec![0f64; pred.len()];
    for i in 0..pred.len() {
        let e = pred[i] - v_target[i];
        l += e * e / n;
        dpred[i] = 2.0 * e / n;
    }
    (l, dpred)
}

/// Full backward from `dpred` (grad of the loss w.r.t. the image patch tokens).
pub fn backward(cfg: &Cfg, w: &ModelWeights, cache: &ModelCache, dpred: &[f64]) -> ModelGrads {
    let (dim, cdim, pd) = (cfg.dim, cfg.cdim(), cfg.patch_dim());
    let (n_img, ncap, ntot) = (cfg.n_img(), cfg.ncap, cfg.ntot());
    let mut dc = vec![0f64; cdim]; // conditioning grad, accumulated everywhere

    // ---- final layer ----
    // d_final_out: image rows = dpred, caption rows = 0.
    let mut d_final_out = vec![0f64; ntot * pd];
    d_final_out[..n_img * pd].copy_from_slice(dpred);
    // final_out = scaled @ flin^T + b
    let mut scaled = vec![0f64; ntot * dim];
    for r in 0..ntot {
        for cc in 0..dim {
            scaled[r * dim + cc] = cache.normed[r * dim + cc] * cache.scale[cc];
        }
    }
    let (d_scaled, g_flin_w, g_flin_b) = linb_bwd(&scaled, ntot, dim, &w.flin_w, pd, &d_final_out);
    // scaled = normed ⊙ scale  → d_normed, d_scale
    let mut d_normed = vec![0f64; ntot * dim];
    let mut d_scale = vec![0f64; dim];
    for r in 0..ntot {
        for cc in 0..dim {
            d_normed[r * dim + cc] = d_scaled[r * dim + cc] * cache.scale[cc];
            d_scale[cc] += d_scaled[r * dim + cc] * cache.normed[r * dim + cc];
        }
    }
    // normed = layernorm(uni)  → d_uni
    let mut d_uni = layernorm_bwd(&cache.uni, ntot, dim, &cache.inv_ln, &d_normed);
    // scale = 1 + adaln(silu(c))  → d_adaln = d_scale ; linear bwd
    let (d_silu_c, g_fadaln_w, g_fadaln_b) = linb_bwd(&cache.silu_c, 1, cdim, &w.fadaln_w, dim, &d_scale);
    for j in 0..cdim {
        dc[j] += d_silu_c[j] * dsilu(cache.c[j]);
    }

    // ---- main layers (reverse) ----
    let mut main_g: Vec<Grads> = Vec::with_capacity(w.main.len());
    for (bw, ca) in w.main.iter().zip(&cache.main_c).rev() {
        let g = block_bwd(cfg.dims(ntot), bw, ca, &d_uni);
        d_uni = g.dx.clone();
        for j in 0..cdim {
            dc[j] += g.dc[j];
        }
        main_g.push(g);
    }
    main_g.reverse();
    // split d_uni → image / caption
    let mut d_img = d_uni[..n_img * dim].to_vec();
    let mut d_capt = d_uni[n_img * dim..].to_vec();

    // ---- context refiners (reverse, unmodulated → no dc) ----
    let mut ctx_g: Vec<Grads> = Vec::with_capacity(w.ctx_ref.len());
    for (bw, ca) in w.ctx_ref.iter().zip(&cache.ctx_c).rev() {
        let g = block_bwd(cfg.dims(ncap), bw, ca, &d_capt);
        d_capt = g.dx.clone();
        ctx_g.push(g);
    }
    ctx_g.reverse();
    // ---- noise refiners (reverse) ----
    let mut noise_g: Vec<Grads> = Vec::with_capacity(w.noise_ref.len());
    for (bw, ca) in w.noise_ref.iter().zip(&cache.noise_c).rev() {
        let g = block_bwd(cfg.dims(n_img), bw, ca, &d_img);
        d_img = g.dx.clone();
        for j in 0..cdim {
            dc[j] += g.dc[j];
        }
        noise_g.push(g);
    }
    noise_g.reverse();

    // ---- embedders ----
    // img = patches @ xemb^T + b  (d_patches unused: latent is data)
    let (_dp, g_xemb_w, g_xemb_b) = linb_bwd(&cache.patches, n_img, pd, &w.xemb_w, dim, &d_img);
    // capt = capn @ cap1^T + b ; capn = rmsnorm(cap)  (cap is data → only gain grad)
    let (d_capn, g_cap1_w, g_cap1_b) = linb_bwd(&cache.capn, ncap, cfg.cap_feat_dim, &w.cap1_w, dim, &d_capt);
    let g_capn_w = rmsnorm_dw(&cache.cap_in, ncap, cfg.cap_feat_dim, &cache.inv_capn, &d_capn);

    // ---- timestep MLP ----
    // c = h0 @ t2^T + b
    let (d_h0, g_t2_w, g_t2_b) = linb_bwd(&cache.h0, 1, TH, &w.t2_w, cdim, &dc);
    // h0 = silu(h0pre)
    let mut d_h0pre = vec![0f64; TH];
    for i in 0..TH {
        d_h0pre[i] = d_h0[i] * dsilu(cache.h0pre[i]);
    }
    // h0pre = te @ t0^T + b
    let (_dte, g_t0_w, g_t0_b) = linb_bwd(&cache.te, 1, TDIM, &w.t0_w, TH, &d_h0pre);

    ModelGrads {
        t0_w: g_t0_w, t0_b: g_t0_b, t2_w: g_t2_w, t2_b: g_t2_b,
        xemb_w: g_xemb_w, xemb_b: g_xemb_b, capn_w: g_capn_w, cap1_w: g_cap1_w, cap1_b: g_cap1_b,
        noise_ref: noise_g, ctx_ref: ctx_g, main: main_g,
        fadaln_w: g_fadaln_w, fadaln_b: g_fadaln_b, flin_w: g_flin_w, flin_b: g_flin_b,
    }
}
