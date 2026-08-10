// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full Z-Image S³-DiT forward (basic text-to-image mode) as host orchestration
//! over the validated [`ZImageBlock`].
//!
//! Flow (diffusers `ZImageTransformer2DModel.forward`, non-omni): patchify the
//! latent → embed image/caption/timestep → refine image (noise_refiner) and
//! caption (context_refiner) → concat `[image, caption]` → main layers →
//! FinalLayer → unpatchify.
//!
//! The transformer blocks run on the device (`ZImageBlock`); everything else —
//! the biased embedder/final-layer linears, patchify/unpatchify, the timestep
//! sinusoid, RoPE-id construction — is cheap host math done where the data
//! already lives. Tokens round-trip host↔device between blocks; a device-resident
//! chaining is a later optimization (the numerics are what this validates).

use dit::rope::{tables_for_ids, RopeConfig, RopeTables};
// Single implementation of the elementwise/normalisation math.
use model::hostmath;

use crate::block::{BlockDims, Tensors, ZImageBlock};

pub(crate) const LN_EPS: f32 = 1e-6; // FinalLayer norm_final (LayerNorm) epsilon.

/// Weight lookup on a host tensor map.
pub(crate) fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("zimage: missing {name}")).1
}

/// The wrapper tensors `preprocess`/`timestep_cond`/`postprocess` read every
/// forward — the ONLY host weights either a device-resident engine
/// (`crate::dev::HostWeights`, which stores exactly these) or the training
/// reference (`Tensors`, which stores the whole model) needs to answer.
/// Semantic accessors, not a generic string lookup, so `HostWeights` never
/// re-derives or re-matches an on-disk name string at call time — only
/// `Tensors`'s impl needs `cfg` to resolve the patch-size-qualified names;
/// `HostWeights` already resolved them once, at build time.
pub(crate) trait HostLookup {
    fn xemb(&self, cfg: &ZImageConfig) -> (&[f32], &[f32]);
    fn cap_norm(&self) -> &[f32];
    fn cap_embed(&self) -> (&[f32], &[f32]);
    fn t_embed(&self) -> (&[f32], &[f32], &[f32], &[f32]);
    fn final_layer(&self, cfg: &ZImageConfig) -> (&[f32], &[f32], &[f32], &[f32]);
}

impl HostLookup for Tensors {
    fn xemb(&self, cfg: &ZImageConfig) -> (&[f32], &[f32]) {
        let xk = format!("all_x_embedder.{}-{}", cfg.patch_size, cfg.f_patch_size);
        (tget(self, &format!("{xk}.weight")), tget(self, &format!("{xk}.bias")))
    }
    fn cap_norm(&self) -> &[f32] {
        tget(self, "cap_embedder.0.weight")
    }
    fn cap_embed(&self) -> (&[f32], &[f32]) {
        (tget(self, "cap_embedder.1.weight"), tget(self, "cap_embedder.1.bias"))
    }
    fn t_embed(&self) -> (&[f32], &[f32], &[f32], &[f32]) {
        (
            tget(self, "t_embedder.mlp.0.weight"),
            tget(self, "t_embedder.mlp.0.bias"),
            tget(self, "t_embedder.mlp.2.weight"),
            tget(self, "t_embedder.mlp.2.bias"),
        )
    }
    fn final_layer(&self, cfg: &ZImageConfig) -> (&[f32], &[f32], &[f32], &[f32]) {
        let fk = format!("all_final_layer.{}-{}", cfg.patch_size, cfg.f_patch_size);
        (
            tget(self, &format!("{fk}.adaLN_modulation.1.weight")),
            tget(self, &format!("{fk}.adaLN_modulation.1.bias")),
            tget(self, &format!("{fk}.linear.weight")),
            tget(self, &format!("{fk}.linear.bias")),
        )
    }
}

/// Z-Image transformer config (the fields the forward needs).
#[derive(Clone, Debug)]
pub struct ZImageConfig {
    pub dim: u32,
    pub n_layers: u32,
    pub n_refiner_layers: u32,
    pub n_heads: u32,
    pub cap_feat_dim: u32,
    pub in_channels: u32,
    pub patch_size: u32,
    pub f_patch_size: u32,
    pub axes_dims: Vec<u32>,
    pub axes_lens: Vec<u32>,
    pub rope_theta: f64,
    pub t_scale: f32,
    pub norm_eps: f32,
}

impl ZImageConfig {
    /// The shipped Z-Image-Turbo config (`transformer/config.json`).
    pub fn turbo() -> ZImageConfig {
        ZImageConfig {
            dim: 3840,
            n_layers: 30,
            n_refiner_layers: 2,
            n_heads: 30,
            cap_feat_dim: 2560,
            in_channels: 16,
            patch_size: 2,
            f_patch_size: 1,
            axes_dims: vec![32, 48, 48],
            axes_lens: vec![1024, 512, 512],
            rope_theta: 256.0,
            t_scale: 1000.0,
            norm_eps: 1e-5,
        }
    }
    pub(crate) fn block_dims(&self) -> BlockDims {
        BlockDims::new(self.dim, self.n_heads)
    }
    pub(crate) fn rope(&self) -> RopeConfig {
        RopeConfig {
            axes_dims: self.axes_dims.clone(),
            axes_lens: self.axes_lens.clone(),
            theta: self.rope_theta,
        }
    }
}

// ---- host math helpers ----

/// `out[r,o] = Σ_i x[r,i]·w[o,i] (+ b[o])`; `w` is `[out,in]` row-major (PyTorch).
pub(crate) fn linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], b: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..r * in_dim + in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc = b.map(|b| b[o]).unwrap_or(0.0);
            for (xi, wi) in xr.iter().zip(wr) {
                acc += xi * wi;
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
}


/// LayerNorm without affine params (FinalLayer.norm_final): per-row standardize.
pub(crate) fn layernorm_noaffine(x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..r * dim + dim];
        let mean: f32 = xr.iter().sum::<f32>() / dim as f32;
        let var: f32 = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            out[r * dim + c] = (xr[c] - mean) * inv;
        }
    }
    out
}


/// Sinusoidal timestep embedding (diffusers `TimestepEmbedder.timestep_embedding`,
/// `dim` even): `[cos(t·freq_k) ‖ sin(t·freq_k)]` — the shared
/// `model::hostmath::timestep_embedding` with `flip_sin_to_cos = false`,
/// `downscale_freq_shift = 0` (this used to be a local f32 re-derivation;
/// the shared one accumulates the angle in f64 like the references do).
pub(crate) fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    model::hostmath::timestep_embedding(t, dim, false, 0.0, max_period as f64)
}

/// Host pre-block state: timestep conditioning, embedded image/caption tokens,
/// and the per-stream RoPE tables. Shared by the reference ([`ZImageModel`]) and
/// device-resident ([`crate::dev::ZImageDit`]) forwards.
pub(crate) struct Pre {
    pub img: Vec<f32>,
    pub capt: Vec<f32>,
    pub img_rope: RopeTables,
    pub cap_rope: RopeTables,
    pub n_img: usize,
    pub ncap: usize,
}

/// Everything before the transformer blocks: t-embed, patchify + x-embed,
/// cap-embed, and RoPE-id construction (caption i → (1+i,0,0); image (f,h,w) →
/// (cap_len+1+f,h,w)).
pub(crate) fn preprocess(cfg: &ZImageConfig, w: &dyn HostLookup, latent: &[f32], f: u32, h: u32, wd: u32, cap: &[f32], cap_len: u32) -> Pre {
    let dim = cfg.dim as usize;
    let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
    let (ft, ht, wt) = (f / pf, h / ps, wd / ps);
    let n_img = (ft * ht * wt) as usize;
    let ncap = cap_len as usize;
    let patch_dim = (pf * ps * ps * cfg.in_channels) as usize;

    let patches = patchify(latent, cfg.in_channels, f, h, wd, ps, pf);
    let (xemb_w, xemb_b) = w.xemb(cfg);
    let img = linear(&patches, n_img, patch_dim, xemb_w, Some(xemb_b), dim);
    let cn = hostmath::rmsnorm_rows(cap, w.cap_norm(), ncap, cfg.cap_feat_dim as usize, cfg.norm_eps);
    let (cap1_w, cap1_b) = w.cap_embed();
    let capt = linear(&cn, ncap, cfg.cap_feat_dim as usize, cap1_w, Some(cap1_b), dim);

    let rope = cfg.rope();
    let mut img_ids = Vec::with_capacity(n_img * 3);
    for fi in 0..ft {
        for hi in 0..ht {
            for wi in 0..wt {
                img_ids.extend_from_slice(&[cap_len + 1 + fi, hi, wi]);
            }
        }
    }
    let mut cap_ids = Vec::with_capacity(ncap * 3);
    for i in 0..cap_len {
        cap_ids.extend_from_slice(&[1 + i, 0, 0]);
    }
    Pre {
        img,
        capt,
        img_rope: tables_for_ids(&rope, &img_ids, 3),
        cap_rope: tables_for_ids(&rope, &cap_ids, 3),
        n_img,
        ncap,
    }
}

/// Timestep conditioning `c = t_embedder(t·t_scale)` `[cdim]`.
pub(crate) fn timestep_cond(cfg: &ZImageConfig, w: &dyn HostLookup, t: f32) -> Vec<f32> {
    let cdim = (cfg.dim as usize).min(256);
    let te = timestep_embedding(t * cfg.t_scale, 256, 10000.0);
    let (t0_w, t0_b, t2_w, t2_b) = w.t_embed();
    let h0 = hostmath::silu_slice(&linear(&te, 1, 256, t0_w, Some(t0_b), 1024));
    linear(&h0, 1, 1024, t2_w, Some(t2_b), cdim)
}

/// FinalLayer (LayerNorm-no-affine · (1+adaLN(c)) → linear) + unpatchify the
/// image portion (first `n_img` tokens of the unified sequence).
pub(crate) fn postprocess(cfg: &ZImageConfig, w: &dyn HostLookup, uni: &[f32], cvec: &[f32], n_img: usize, f: u32, h: u32, wd: u32) -> Vec<f32> {
    let dim = cfg.dim as usize;
    let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
    let cdim = dim.min(256);
    let ntot = uni.len() / dim;
    let patch_dim = (pf * ps * ps * cfg.in_channels) as usize;
    let (fadaln_w, fadaln_b, flin_w, flin_b) = w.final_layer(cfg);
    let adaln = linear(&hostmath::silu_slice(cvec), 1, cdim, fadaln_w, Some(fadaln_b), dim);
    let scale: Vec<f32> = adaln.iter().map(|&v| 1.0 + v).collect();
    let normed = layernorm_noaffine(uni, ntot, dim, LN_EPS);
    let mut scaled = vec![0f32; ntot * dim];
    for r in 0..ntot {
        for cc in 0..dim {
            scaled[r * dim + cc] = normed[r * dim + cc] * scale[cc];
        }
    }
    let final_out = linear(&scaled, ntot, dim, flin_w, Some(flin_b), patch_dim);
    unpatchify(&final_out[..n_img * patch_dim], cfg.in_channels, f, h, wd, ps, pf)
}

/// The Z-Image DiT: config + host weights, runs a single forward.
pub struct ZImageModel {
    cfg: ZImageConfig,
    w: Tensors,
    device: Option<String>,
}

impl ZImageModel {
    pub fn new(cfg: ZImageConfig, weights: Tensors, device: Option<&str>) -> ZImageModel {
        ZImageModel { cfg, w: weights, device: device.map(|s| s.to_string()) }
    }

    /// One DiT forward (reference path: one device per block, host round-trips).
    /// `latent`: `[C·F·H·W]`; `cap`: `[cap_len·cap_feat_dim]`; `t`: timestep;
    /// `(f,h,w)`: latent spatial dims. Returns the predicted latent `[C·F·H·W]`.
    pub fn forward(&self, latent: &[f32], f: u32, h: u32, w: u32, cap: &[f32], cap_len: u32, t: f32) -> Vec<f32> {
        let c = &self.cfg;
        let dev = self.device.as_deref();
        let cvec = timestep_cond(c, &self.w, t);
        let pre = preprocess(c, &self.w, latent, f, h, w, cap, cap_len);
        let bd = c.block_dims();
        let (mut img, mut capt) = (pre.img, pre.capt);
        for l in 0..c.n_refiner_layers {
            let blk = ZImageBlock::new(&self.w, &format!("noise_refiner.{l}"), bd, pre.n_img as u32, true, dev);
            img = blk.forward(&img, &cvec, &pre.img_rope.cos, &pre.img_rope.sin);
        }
        for l in 0..c.n_refiner_layers {
            let blk = ZImageBlock::new(&self.w, &format!("context_refiner.{l}"), bd, pre.ncap as u32, false, dev);
            capt = blk.forward(&capt, &cvec, &pre.cap_rope.cos, &pre.cap_rope.sin);
        }
        // unified [image, caption]
        let mut uni = img;
        uni.extend_from_slice(&capt);
        let mut uni_cos = pre.img_rope.cos.clone();
        uni_cos.extend_from_slice(&pre.cap_rope.cos);
        let mut uni_sin = pre.img_rope.sin.clone();
        uni_sin.extend_from_slice(&pre.cap_rope.sin);
        for l in 0..c.n_layers {
            let blk = ZImageBlock::new(&self.w, &format!("layers.{l}"), bd, (pre.n_img + pre.ncap) as u32, true, dev);
            uni = blk.forward(&uni, &cvec, &uni_cos, &uni_sin);
        }
        postprocess(c, &self.w, &uni, &cvec, pre.n_img, f, h, w)
    }
}

/// `[C,F,H,W] -> [n_patches, pF·pH·pW·C]`, patch (f,h,w) row-major, inner order
/// `[pF,pH,pW,C]` (matches diffusers `_patchify_image`).
pub(crate) fn patchify(latent: &[f32], ch: u32, f: u32, h: u32, w: u32, ps: u32, pf: u32) -> Vec<f32> {
    let (c, ft, ht, wt) = (ch as usize, (f / pf) as usize, (h / ps) as usize, (w / ps) as usize);
    let (pf, ph, pw) = (pf as usize, ps as usize, ps as usize);
    let (fh, fw) = (h as usize, w as usize);
    let patch_dim = pf * ph * pw * c;
    let mut out = vec![0f32; ft * ht * wt * patch_dim];
    for ff in 0..ft {
        for hh in 0..ht {
            for ww in 0..wt {
                let tok = (ff * ht * wt + hh * wt + ww) * patch_dim;
                for pff in 0..pf {
                    for phh in 0..ph {
                        for pww in 0..pw {
                            for cc in 0..c {
                                let src = ((cc * (f as usize) + (ff * pf + pff)) * fh + (hh * ph + phh)) * fw + (ww * pw + pww);
                                let dst = tok + ((pff * ph + phh) * pw + pww) * c + cc;
                                out[dst] = latent[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`]: `[n_img, pF·pH·pW·C] -> [C,F,H,W]`.
pub(crate) fn unpatchify(tokens: &[f32], ch: u32, f: u32, h: u32, w: u32, ps: u32, pf: u32) -> Vec<f32> {
    let (c, ft, ht, wt) = (ch as usize, (f / pf) as usize, (h / ps) as usize, (w / ps) as usize);
    let (pfz, ph, pw) = (pf as usize, ps as usize, ps as usize);
    let (fh, fw) = (h as usize, w as usize);
    let patch_dim = pfz * ph * pw * c;
    let mut out = vec![0f32; c * (f as usize) * fh * fw];
    for ff in 0..ft {
        for hh in 0..ht {
            for ww in 0..wt {
                let tok = (ff * ht * wt + hh * wt + ww) * patch_dim;
                for pff in 0..pfz {
                    for phh in 0..ph {
                        for pww in 0..pw {
                            for cc in 0..c {
                                let val = tokens[tok + ((pff * ph + phh) * pw + pww) * c + cc];
                                let dst = ((cc * (f as usize) + (ff * pfz + pff)) * fh + (hh * ph + phh)) * fw + (ww * pw + pww);
                                out[dst] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
