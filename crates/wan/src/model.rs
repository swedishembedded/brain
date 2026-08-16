// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The full Wan DiT forward, host-orchestrated over the validated
//! [`crate::block::WanBlock`] - the parity reference.
//!
//! Flow (`wan/modules/model.py`, `WanModel.forward`): patchify the latent into
//! tokens, embed the text and the timestep, run the block stack, then the
//! modulated head and unpatchify. Everything outside the blocks is cheap host
//! math done where the data already lives; the device-resident chaining is
//! [`crate::dev`], which shares this module's pre/post helpers so the two
//! cannot drift.
//!
//! ## Two patch orderings, and they are NOT the same
//!
//! `patch_embedding` is `Conv3d(in, dim, (1,2,2), stride=(1,2,2))`. Kernel
//! equals stride and the temporal extent is 1, so it is a per-frame 2x2
//! space-to-depth - but a conv weight row is flattened `[c][kt][kh][kw]`, so
//! the token vector it consumes is **channel-outermost**. The head's output row
//! is `view(*patch_size, c)`, i.e. **channel-innermost**. Reusing one ordering
//! for both produces a shuffled latent that still looks like video.

use model::hostmath;

use crate::block::{open_device, WanBlock};
use crate::config::WanConfig;
use crate::rope::tables;

/// Host tensors by name -> `(shape, row-major f32 data)`; the same shape
/// `crate::import` produces and `checkpoint::TensorSource` is implemented for.
pub type Tensors = vae::blocks::Tensors;

pub(crate) fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("wan: missing {name}")).1
}

/// `out[r,o] = Σ_i x[r,i]·w[o,i] (+ b[o])`; `w` is `[out,in]` row-major.
pub fn linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], b: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
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

/// GELU, tanh approximation - `nn.GELU(approximate='tanh')`, matching
/// `gelu.wgsl` exactly so the host and device FFNs are the same function.
pub fn gelu_tanh(x: &mut [f32]) {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    for v in x.iter_mut() {
        let t = K * (*v + 0.044_715 * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + t.tanh());
    }
}

/// The timestep conditioning: `e` (`[dim]`, the head's modulation base) and
/// `e0` (`[6·dim]`, every block's).
///
/// `sinusoidal_embedding_1d` is `cat([cos, sin])` over `10000^(-k/half)`, which
/// is the shared `hostmath::timestep_embedding` at `flip_sin_to_cos = true`,
/// `downscale_freq_shift = 0` - upstream accumulates the angle in f64 and so
/// does that helper.
pub fn timestep_cond(cfg: &WanConfig, w: &Tensors, t: f32) -> (Vec<f32>, Vec<f32>) {
    let (dim, fd) = (cfg.dim, cfg.freq_dim);
    let te = hostmath::timestep_embedding(t, fd, true, 0.0, 10000.0);
    let h0 = linear(&te, 1, fd, tget(w, "time_embedding.0.weight"), Some(tget(w, "time_embedding.0.bias")), dim);
    let h0 = hostmath::silu_slice(&h0);
    let e = linear(&h0, 1, dim, tget(w, "time_embedding.2.weight"), Some(tget(w, "time_embedding.2.bias")), dim);
    let e_act = hostmath::silu_slice(&e);
    let e0 = linear(&e_act, 1, dim, tget(w, "time_projection.1.weight"), Some(tget(w, "time_projection.1.bias")), 6 * dim);
    (e, e0)
}

/// `[C, F, H, W]` -> `[tokens, C·pH·pW]` with the patch (f, h, w) row-major and
/// the inner order `[c, pH, pW]` - a `Conv3d` weight row's own flattening.
pub fn patchify(latent: &[f32], c: usize, f: usize, h: usize, w: usize, ph: usize, pw: usize) -> Vec<f32> {
    let (ht, wt) = (h / ph, w / pw);
    let patch = c * ph * pw;
    let mut out = vec![0f32; f * ht * wt * patch];
    for fi in 0..f {
        for hi in 0..ht {
            for wi in 0..wt {
                let tok = ((fi * ht + hi) * wt + wi) * patch;
                for ci in 0..c {
                    for a in 0..ph {
                        for b in 0..pw {
                            let src = ((ci * f + fi) * h + hi * ph + a) * w + wi * pw + b;
                            out[tok + (ci * ph + a) * pw + b] = latent[src];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of the head's row layout: `[tokens, pH·pW·C]` (channel-innermost)
/// -> `[C, F, H, W]`.
pub fn unpatchify(tokens: &[f32], c: usize, f: usize, ht: usize, wt: usize, ph: usize, pw: usize) -> Vec<f32> {
    let (h, w) = (ht * ph, wt * pw);
    let patch = ph * pw * c;
    let mut out = vec![0f32; c * f * h * w];
    for fi in 0..f {
        for hi in 0..ht {
            for wi in 0..wt {
                let tok = ((fi * ht + hi) * wt + wi) * patch;
                for a in 0..ph {
                    for b in 0..pw {
                        for ci in 0..c {
                            let v = tokens[tok + (a * pw + b) * c + ci];
                            out[((ci * f + fi) * h + hi * ph + a) * w + wi * pw + b] = v;
                        }
                    }
                }
            }
        }
    }
    out
}

/// The text encoding the cross-attention reads: zero-pad to `text_len`, then
/// `Linear -> GELU(tanh) -> Linear`.
///
/// The pad is HARD ZEROS, applied before the MLP, exactly as
/// `WanModel.forward` does with `new_zeros` - upstream's own encoder output at
/// those positions is discarded by `T5EncoderModel.__call__`'s trim, so a port
/// that carries the encoder's pad rows through instead is feeding the
/// cross-attention different keys.
pub fn text_embed(cfg: &WanConfig, w: &Tensors, context: &[f32], rows: usize) -> Vec<f32> {
    let (dim, td, tl) = (cfg.dim, cfg.text_dim, cfg.text_len);
    assert!(rows <= tl, "context has {rows} rows, text_len is {tl}");
    let mut padded = vec![0f32; tl * td];
    padded[..rows * td].copy_from_slice(&context[..rows * td]);
    let mut h = linear(&padded, tl, td, tget(w, "text_embedding.0.weight"), Some(tget(w, "text_embedding.0.bias")), dim);
    gelu_tanh(&mut h);
    linear(&h, tl, dim, tget(w, "text_embedding.2.weight"), Some(tget(w, "text_embedding.2.bias")), dim)
}

/// The head: an affine-free LayerNorm carrying `head.modulation + e`, then the
/// patch projection. Returns `[tokens, prod(patch)·out_channels]`.
pub fn head(cfg: &WanConfig, w: &Tensors, x: &[f32], e: &[f32], tokens: usize) -> Vec<f32> {
    let dim = cfg.dim;
    let m = tget(w, "head.modulation");
    assert_eq!(m.len(), 2 * dim, "head.modulation must be [1, 2, dim]");
    let shift: Vec<f32> = m[..dim].iter().zip(e).map(|(a, b)| a + b).collect();
    let gamma: Vec<f32> = m[dim..].iter().zip(e).map(|(a, b)| 1.0 + a + b).collect();
    let normed = hostmath::layernorm_rows(x, &gamma, &shift, tokens, dim, cfg.eps);
    let (pt, ph, pw) = cfg.patch_size;
    let out_dim = pt * ph * pw * cfg.out_channels;
    linear(&normed, tokens, dim, tget(w, "head.head.weight"), Some(tget(w, "head.head.bias")), out_dim)
}

/// Everything before the block stack: tokens, text encoding, RoPE tables and
/// the two timestep vectors. Shared by [`WanDit`] and [`crate::dev::WanDitDev`]
/// so the reference and the device-resident engine cannot disagree about the
/// conventions.
pub struct Pre {
    pub tokens: Vec<f32>,
    pub ctx: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub e: Vec<f32>,
    pub e0: Vec<f32>,
    pub n_tokens: usize,
    /// Patch grid `(f, h, w)` - what `unpatchify` and the RoPE ids run over.
    pub grid: (u32, u32, u32),
}

/// The patch grid `(f, h, w)` a latent extent produces.
pub fn patch_grid(cfg: &WanConfig, f: u32, h: u32, wd: u32) -> (u32, u32, u32) {
    let (pt, ph, pw) = cfg.patch_size;
    assert_eq!(pt, 1, "only a temporal patch of 1 is implemented");
    assert!(
        h.is_multiple_of(ph as u32) && wd.is_multiple_of(pw as u32),
        "latent {h}x{wd} is not a whole number of {ph}x{pw} patches"
    );
    (f / pt as u32, h / ph as u32, wd / pw as u32)
}

/// Patchify + `patch_embedding`, giving the `[tokens, dim]` slab the block
/// stack consumes.
pub fn embed_tokens(cfg: &WanConfig, w: &Tensors, latent: &[f32], f: u32, h: u32, wd: u32) -> Vec<f32> {
    let (_, ph, pw) = cfg.patch_size;
    let grid = patch_grid(cfg, f, h, wd);
    let n = (grid.0 * grid.1 * grid.2) as usize;
    let flat = patchify(latent, cfg.in_channels, f as usize, h as usize, wd as usize, ph, pw);
    linear(
        &flat,
        n,
        cfg.in_channels * ph * pw,
        tget(w, "patch_embedding.weight"),
        Some(tget(w, "patch_embedding.bias")),
        cfg.dim,
    )
}

pub fn preprocess(cfg: &WanConfig, w: &Tensors, latent: &[f32], f: u32, h: u32, wd: u32, context: &[f32], ctx_rows: usize, t: f32) -> Pre {
    let grid = patch_grid(cfg, f, h, wd);
    let n_tokens = (grid.0 * grid.1 * grid.2) as usize;
    let tokens = embed_tokens(cfg, w, latent, f, h, wd);
    let (e, e0) = timestep_cond(cfg, w, t);
    let r = tables(cfg, grid.0, grid.1, grid.2);
    Pre { tokens, ctx: text_embed(cfg, w, context, ctx_rows), cos: r.cos, sin: r.sin, e, e0, n_tokens, grid }
}

/// The head plus unpatchify - the mirror of [`preprocess`].
pub fn postprocess(cfg: &WanConfig, w: &Tensors, x: &[f32], e: &[f32], grid: (u32, u32, u32)) -> Vec<f32> {
    let n = (grid.0 * grid.1 * grid.2) as usize;
    let (pt, ph, pw) = cfg.patch_size;
    let rows = head(cfg, w, x, e, n);
    let _ = pt;
    unpatchify(&rows, cfg.out_channels, grid.0 as usize, grid.1 as usize, grid.2 as usize, ph, pw)
}

/// The Wan DiT with host weights, running one forward per call.
pub struct WanDit {
    cfg: WanConfig,
    w: Tensors,
    device: Option<String>,
}

impl WanDit {
    pub fn new(cfg: WanConfig, weights: Tensors, device: Option<&str>) -> WanDit {
        WanDit { cfg, w: weights, device: device.map(|s| s.to_string()) }
    }

    pub fn config(&self) -> &WanConfig {
        &self.cfg
    }

    /// One DiT forward. `latent`: `[C·F·H·W]` in latent space; `context`:
    /// `[ctx_rows · text_dim]` from the text encoder; `t`: the diffusion
    /// timestep on the training grid (0..1000). Returns `[C_out·F·H·W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, latent: &[f32], f: u32, h: u32, w: u32, context: &[f32], ctx_rows: usize, t: f32) -> Vec<f32> {
        let (x, _) = self.forward_taps(latent, f, h, w, context, ctx_rows, t, &[]);
        x
    }

    /// [`WanDit::forward`], also returning the output of each block named in
    /// `taps` (block indices) - what a parity test bisects with.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_taps(
        &self,
        latent: &[f32],
        f: u32,
        h: u32,
        w: u32,
        context: &[f32],
        ctx_rows: usize,
        t: f32,
        taps: &[usize],
    ) -> (Vec<f32>, Vec<(usize, Vec<f32>)>) {
        let cfg = &self.cfg;
        let pre = preprocess(cfg, &self.w, latent, f, h, w, context, ctx_rows, t);
        let gpu = open_device(self.device.as_deref());
        let mut x = pre.tokens;
        let mut out_taps = Vec::new();
        for l in 0..cfg.num_layers {
            let blk = WanBlock::on(gpu.share(), cfg, &self.w, &format!("blocks.{l}"), pre.n_tokens as u32);
            x = blk.forward(&x, &pre.e0, &pre.cos, &pre.sin, &pre.ctx);
            if taps.contains(&l) {
                out_taps.push((l, x.clone()));
            }
        }
        (postprocess(cfg, &self.w, &x, &pre.e, pre.grid), out_taps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// patchify and unpatchify use DIFFERENT inner orderings (see the module
    /// header), so they are not inverses - composing them with a transpose of
    /// the inner block is what recovers the identity, and this pins that the
    /// two orderings really do differ.
    #[test]
    fn the_two_patch_orderings_are_not_the_same() {
        let (c, f, h, w) = (2usize, 1usize, 2usize, 2usize);
        let x: Vec<f32> = (0..(c * f * h * w)).map(|i| i as f32).collect();
        let flat = patchify(&x, c, f, h, w, 2, 2);
        // Channel-outermost: [c0 p00 p01 p10 p11, c1 ...].
        assert_eq!(flat, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        // The head's layout is channel-innermost, so the same 8 values
        // unpatchify to a different tensor.
        let back = unpatchify(&flat, c, f, 1, 1, 2, 2);
        assert_ne!(back, x);
        // ...and the head layout of the SAME latent is the interleaved one.
        let head_rows = vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
        assert_eq!(unpatchify(&head_rows, c, f, 1, 1, 2, 2), x);
    }

    #[test]
    fn patchify_walks_width_fastest_like_the_rope_ids() {
        // 1 channel, 1 frame, 2x4 -> a 1x2 patch grid; token 1 must be the
        // right-hand 2x2 block, matching `rope::grid_ids`' order.
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let flat = patchify(&x, 1, 1, 2, 4, 2, 2);
        assert_eq!(&flat[..4], &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(&flat[4..], &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn gelu_tanh_matches_the_kernels_constants() {
        let mut v = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
        gelu_tanh(&mut v);
        // torch nn.GELU(approximate='tanh') at the same points.
        let want = [-0.0454, -0.1543, 0.0, 0.3457, 1.9546];
        for (g, e) in v.iter().zip(want) {
            assert!((g - e).abs() < 1e-4, "gelu {g} vs {e}");
        }
    }
}
