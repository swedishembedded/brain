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

use crate::block::{BlockDims, Tensors, ZImageBlock};

pub(crate) const LN_EPS: f32 = 1e-6; // FinalLayer norm_final (LayerNorm) epsilon.

/// Weight lookup on a host tensor map.
pub(crate) fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("zimage: missing {name}")).1
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

pub(crate) fn rmsnorm(x: &[f32], rows: usize, dim: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..r * dim + dim];
        let ss: f32 = xr.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for c in 0..dim {
            out[r * dim + c] = w[c] * xr[c] * inv;
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

pub(crate) fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// Sinusoidal timestep embedding (diffusers `TimestepEmbedder.timestep_embedding`,
/// `dim` even): `[cos(t·freq_k) ‖ sin(t·freq_k)]`, `freq_k = max_period^(-k/half)`.
pub(crate) fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut e = vec![0f32; dim];
    for k in 0..half {
        let freq = (-(max_period.ln()) * k as f32 / half as f32).exp();
        let arg = t * freq;
        e[k] = arg.cos();
        e[half + k] = arg.sin();
    }
    e
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
pub(crate) fn preprocess(cfg: &ZImageConfig, w: &Tensors, latent: &[f32], f: u32, h: u32, wd: u32, cap: &[f32], cap_len: u32) -> Pre {
    let dim = cfg.dim as usize;
    let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
    let (ft, ht, wt) = (f / pf, h / ps, wd / ps);
    let n_img = (ft * ht * wt) as usize;
    let ncap = cap_len as usize;
    let patch_dim = (pf * ps * ps * cfg.in_channels) as usize;

    let patches = patchify(latent, cfg.in_channels, f, h, wd, ps, pf);
    let xk = format!("all_x_embedder.{ps}-{pf}");
    let img = linear(&patches, n_img, patch_dim, tget(w, &format!("{xk}.weight")), Some(tget(w, &format!("{xk}.bias"))), dim);
    let cn = rmsnorm(cap, ncap, cfg.cap_feat_dim as usize, tget(w, "cap_embedder.0.weight"), cfg.norm_eps);
    let capt = linear(&cn, ncap, cfg.cap_feat_dim as usize, tget(w, "cap_embedder.1.weight"), Some(tget(w, "cap_embedder.1.bias")), dim);

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
pub(crate) fn timestep_cond(cfg: &ZImageConfig, w: &Tensors, t: f32) -> Vec<f32> {
    let cdim = (cfg.dim as usize).min(256);
    let te = timestep_embedding(t * cfg.t_scale, 256, 10000.0);
    let h0 = silu(&linear(&te, 1, 256, tget(w, "t_embedder.mlp.0.weight"), Some(tget(w, "t_embedder.mlp.0.bias")), 1024));
    linear(&h0, 1, 1024, tget(w, "t_embedder.mlp.2.weight"), Some(tget(w, "t_embedder.mlp.2.bias")), cdim)
}

/// FinalLayer (LayerNorm-no-affine · (1+adaLN(c)) → linear) + unpatchify the
/// image portion (first `n_img` tokens of the unified sequence).
pub(crate) fn postprocess(cfg: &ZImageConfig, w: &Tensors, uni: &[f32], cvec: &[f32], n_img: usize, f: u32, h: u32, wd: u32) -> Vec<f32> {
    let dim = cfg.dim as usize;
    let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
    let cdim = dim.min(256);
    let ntot = uni.len() / dim;
    let patch_dim = (pf * ps * ps * cfg.in_channels) as usize;
    let fk = format!("all_final_layer.{ps}-{pf}");
    let adaln = linear(&silu(cvec), 1, cdim, tget(w, &format!("{fk}.adaLN_modulation.1.weight")), Some(tget(w, &format!("{fk}.adaLN_modulation.1.bias"))), dim);
    let scale: Vec<f32> = adaln.iter().map(|&v| 1.0 + v).collect();
    let normed = layernorm_noaffine(uni, ntot, dim, LN_EPS);
    let mut scaled = vec![0f32; ntot * dim];
    for r in 0..ntot {
        for cc in 0..dim {
            scaled[r * dim + cc] = normed[r * dim + cc] * scale[cc];
        }
    }
    let final_out = linear(&scaled, ntot, dim, tget(w, &format!("{fk}.linear.weight")), Some(tget(w, &format!("{fk}.linear.bias"))), patch_dim);
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
