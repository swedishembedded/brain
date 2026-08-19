// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full Wan DiT **training** reference (host): forward + analytic backward for
//! the whole transformer under the flow-matching velocity-MSE loss.
//!
//! This chains the block reference ([`crate::grad`]) across the stack and wraps
//! it with everything a block does not have: patchify + `patch_embedding`, the
//! text embedding MLP, the whole timestep path, the modulated head and
//! unpatchify - mirroring [`crate::model`] op for op.
//!
//! ## The conditioning graph is what makes this one model
//!
//! Three couplings have to be right, and each is invisible in a forward:
//!
//! 1. **`e0` is shared by every block.** Each block's six-vector modulation grad
//!    is simultaneously `d(blocks.{l}.modulation)` and its contribution to
//!    `d e0`; the contributions are summed over the whole stack before entering
//!    `time_projection`.
//! 2. **The head reads `e`, not `e0`.** `head.modulation + e` feeds the final
//!    LayerNorm, while `e0 = time_projection(silu(e))`. So `d e` has two
//!    sources - the head site directly, and the whole block stack through
//!    `time_projection` and `silu'` - which then run back through
//!    `time_embedding` as one.
//! 3. **The embedded text context is shared by every block.** Every block's
//!    cross-attention reads the same `[text_len, dim]` slab, so the blocks'
//!    `dctx` are summed before `text_embedding`'s backward.
//!
//! Each of the three is a *folded/shared* parameter path in the sense AGENTS.md
//! warns about: dropping one leaves a gradient that is only partially wrong, so
//! `gradcheck::check_wan_conditioning` checks the tensors that sit at those
//! folds **per entry**, not just as a random contraction.
//!
//! Generic over [`Fp`] like [`crate::grad`]: the `f64` instantiation is the FD
//! gradcheck oracle, the `f32` instantiation is the trainer [`crate::finetune`]
//! drives - one implementation, no oracle/trainer drift.

use crate::config::WanConfig;
use crate::grad::{
    affine, affine_bwd, block_backward, block_forward, dgelu, dsilu, gelu, layernorm, layernorm_bwd, linear, linear_bwd, silu,
    BlockCache, BlockGrads, BlockW, Dims, Fp, Lin,
};

/// Shape of the training problem: the [`WanConfig`] fields the host path needs
/// plus the latent extent being trained on.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    pub text_len: usize,
    pub freq_dim: usize,
    /// `(t, h, w)` patch - always `(1, 2, 2)` upstream.
    pub patch: (usize, usize, usize),
    /// Latent extent `(frames, height, width)` in LATENT units (post-VAE).
    pub latent: (usize, usize, usize),
    pub eps: f64,
    /// Real components per RoPE axis, `[frame, height, width]`.
    pub rope_axes: [usize; 3],
    pub rope_theta: f64,
}

impl Cfg {
    /// Derive from a [`WanConfig`] at latent extent `(f, h, w)`.
    pub fn from_wan(c: &WanConfig, f: usize, h: usize, w: usize) -> Cfg {
        Cfg {
            dim: c.dim,
            ffn_dim: c.ffn_dim,
            n_heads: c.num_heads,
            n_layers: c.num_layers,
            in_channels: c.in_channels,
            out_channels: c.out_channels,
            text_dim: c.text_dim,
            text_len: c.text_len,
            freq_dim: c.freq_dim,
            patch: c.patch_size,
            latent: (f, h, w),
            eps: c.eps as f64,
            rope_axes: c.rope_axes_dims(),
            rope_theta: crate::rope::THETA,
        }
    }

    /// Tiny Wan-topology config for gradchecks and unit tests.
    ///
    /// **Every dimension is deliberately distinct.** Equal head counts, widths
    /// and sequence lengths hide transposition bugs, and this port has already
    /// paid for two of exactly that shape (a `T == 1` permute that made
    /// `H*W` and `T*H*W` identical, and two patch orderings that differ only
    /// when channels and patch extent disagree). So: 18 latent tokens against 5
    /// text rows, `dim` 24 against `ffn_dim` 10, 3 heads of 8, a `(3, 2, 3)`
    /// patch grid with all three axes different, and `freq_dim` 8.
    ///
    /// `in_channels == out_channels` is NOT a free choice: the flow-matching
    /// target is a velocity in the input latent's own space, so the head must
    /// predict as many channels as the input carries (upstream: 16 and 16).
    pub fn tiny() -> Cfg {
        Cfg {
            dim: 24,
            ffn_dim: 10,
            n_heads: 3,
            n_layers: 2,
            in_channels: 3,
            out_channels: 3,
            text_dim: 7,
            text_len: 5,
            freq_dim: 8,
            patch: (1, 2, 2),
            latent: (3, 4, 6),
            eps: 1e-6,
            // head_dim 8 -> 4 complex pairs -> [4-2, 1, 1] pairs -> [4,2,2] real.
            rope_axes: [4, 2, 2],
            rope_theta: 10000.0,
        }
    }

    pub const fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Patch grid `(f, h, w)` - the token order RoPE ids walk.
    pub fn grid(&self) -> (usize, usize, usize) {
        let (pt, ph, pw) = self.patch;
        let (f, h, w) = self.latent;
        assert_eq!(pt, 1, "only a temporal patch of 1 is implemented");
        assert!(h.is_multiple_of(ph) && w.is_multiple_of(pw), "latent {h}x{w} is not a whole number of patches");
        (f / pt, h / ph, w / pw)
    }

    pub fn n_tokens(&self) -> usize {
        let (f, h, w) = self.grid();
        f * h * w
    }

    /// Width of one `patch_embedding` input row (`c·pt·ph·pw`).
    pub fn patch_dim(&self) -> usize {
        let (pt, ph, pw) = self.patch;
        self.in_channels * pt * ph * pw
    }

    /// Width of one head output row (`pt·ph·pw·c_out`).
    pub fn head_dim_out(&self) -> usize {
        let (pt, ph, pw) = self.patch;
        pt * ph * pw * self.out_channels
    }

    pub fn latent_len(&self) -> usize {
        let (f, h, w) = self.latent;
        self.in_channels * f * h * w
    }

    pub fn dims(&self) -> Dims {
        Dims { t: self.n_tokens(), te: self.text_len, dim: self.dim, nh: self.n_heads, ffn: self.ffn_dim, eps: self.eps }
    }

    /// The RoPE tables this config's token grid needs.
    pub fn rope_tables(&self) -> dit::rope::RopeTables {
        let (f, h, w) = self.grid();
        let rc = dit::rope::RopeConfig {
            axes_dims: self.rope_axes.iter().map(|&d| d as u32).collect(),
            axes_lens: vec![crate::rope::MAX_SEQ_LEN; 3],
            theta: self.rope_theta,
        };
        dit::rope::tables_for_ids(&rc, &crate::rope::grid_ids(f as u32, h as u32, w as u32), 3)
    }
}

/// Every trainable tensor of the DiT, in the host training layout.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelWeights<T> {
    pub patch_embed: Lin<T>,
    pub text0: Lin<T>,
    pub text2: Lin<T>,
    pub time0: Lin<T>,
    pub time2: Lin<T>,
    pub time_proj: Lin<T>,
    pub blocks: Vec<BlockW<T>>,
    pub head: Lin<T>,
    /// `head.modulation`, `[2·dim]`: `(shift, scale)`, added to `e`.
    pub head_mod: Vec<T>,
}

/// Grads mirroring [`ModelWeights`]. The per-block grads keep their `dx`/`dctx`
/// (harmless extras, useful in tests); their modulation contributions are
/// already folded into `time_proj`/`time0`/`time2` here.
#[derive(Clone, Debug)]
pub struct ModelGrads<T> {
    pub patch_embed: Lin<T>,
    pub text0: Lin<T>,
    pub text2: Lin<T>,
    pub time0: Lin<T>,
    pub time2: Lin<T>,
    pub time_proj: Lin<T>,
    pub blocks: Vec<BlockGrads<T>>,
    pub head: Lin<T>,
    pub head_mod: Vec<T>,
}

/// `sinusoidal_embedding_1d`: `cat([cos, sin])` over `10000^(-k/half)` - the
/// generic-`T` twin of `model::hostmath::timestep_embedding` at
/// `flip_sin_to_cos = true, downscale_freq_shift = 0` (angles accumulated in
/// f64 on both, exactly as upstream does). A deliberate second implementation:
/// AGENTS.md exception 1 - it carries no parameters, so it is pure input math.
pub fn timestep_embedding<T: Fp>(t: f64, dim: usize) -> Vec<T> {
    assert!(dim.is_multiple_of(2), "timestep_embedding: dim {dim} must be even");
    let half = dim / 2;
    let mut e = vec![T::ZERO; dim];
    for k in 0..half {
        let freq = (-(10000.0f64.ln()) * k as f64 / half as f64).exp();
        let arg = t * freq;
        e[k] = T::fr(arg.cos());
        e[half + k] = T::fr(arg.sin());
    }
    e
}

/// `[C, F, H, W]` -> `[tokens, C·pH·pW]`, patch (f, h, w) row-major and the
/// inner order `[c, pH, pW]` - a `Conv3d` weight row's own flattening, i.e.
/// **channel-outermost**. The generic-`T` twin of [`crate::model::patchify`];
/// `tests::the_generic_patchify_agrees_with_the_forward_paths` pins them equal.
pub fn patchify<T: Fp>(latent: &[T], c: usize, f: usize, h: usize, w: usize, ph: usize, pw: usize) -> Vec<T> {
    let (ht, wt) = (h / ph, w / pw);
    let patch = c * ph * pw;
    let mut out = vec![T::ZERO; f * ht * wt * patch];
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

/// Inverse of the head's row layout: `[tokens, pH·pW·C]` (**channel-innermost**,
/// the OTHER of the two orderings) -> `[C, F, H, W]`.
pub fn unpatchify<T: Fp>(tokens: &[T], c: usize, f: usize, ht: usize, wt: usize, ph: usize, pw: usize) -> Vec<T> {
    let (h, w) = (ht * ph, wt * pw);
    let patch = ph * pw * c;
    let mut out = vec![T::ZERO; c * f * h * w];
    for fi in 0..f {
        for hi in 0..ht {
            for wi in 0..wt {
                let tok = ((fi * ht + hi) * wt + wi) * patch;
                for a in 0..ph {
                    for b in 0..pw {
                        for ci in 0..c {
                            out[((ci * f + fi) * h + hi * ph + a) * w + wi * pw + b] = tokens[tok + (a * pw + b) * c + ci];
                        }
                    }
                }
            }
        }
    }
    out
}

/// [`unpatchify`] backward - the same permutation read the other way.
pub(crate) fn unpatchify_bwd<T: Fp>(dout: &[T], c: usize, f: usize, ht: usize, wt: usize, ph: usize, pw: usize) -> Vec<T> {
    let (h, w) = (ht * ph, wt * pw);
    let patch = ph * pw * c;
    let mut drows = vec![T::ZERO; f * ht * wt * patch];
    for fi in 0..f {
        for hi in 0..ht {
            for wi in 0..wt {
                let tok = ((fi * ht + hi) * wt + wi) * patch;
                for a in 0..ph {
                    for b in 0..pw {
                        for ci in 0..c {
                            drows[tok + (a * pw + b) * c + ci] = dout[((ci * f + fi) * h + hi * ph + a) * w + wi * pw + b];
                        }
                    }
                }
            }
        }
    }
    drows
}

/// Saved forward state for the backward pass.
pub struct ModelCache<T> {
    te: Vec<T>,
    h0pre: Vec<T>,
    h0: Vec<T>,
    e: Vec<T>,
    eact: Vec<T>,
    ctx_in: Vec<T>,
    th: Vec<T>,
    thg: Vec<T>,
    flat: Vec<T>,
    blocks: Vec<BlockCache<T>>,
    xhat_h: Vec<T>,
    inv_h: Vec<T>,
    gamma_h: Vec<T>,
    nh: Vec<T>,
}

/// Full forward. `latent`: `[C·F·H·W]`; `ctx`: `[text_len·text_dim]` (already
/// zero-padded, as `WanModel.forward` pads with `new_zeros`); `t`: the timestep
/// on the training grid (0..1000); `cos`/`sin`: `[tokens·head_dim/2]`. Returns
/// the velocity prediction `[C_out·F·H·W]` plus the cache.
#[allow(clippy::too_many_arguments)]
pub fn forward<T: Fp>(cfg: &Cfg, w: &ModelWeights<T>, latent: &[T], ctx: &[T], t: f64, cos: &[T], sin: &[T]) -> (Vec<T>, ModelCache<T>) {
    let (dim, tl, td) = (cfg.dim, cfg.text_len, cfg.text_dim);
    let (gf, gh, gw) = cfg.grid();
    let n = cfg.n_tokens();
    let (_, ph, pw) = cfg.patch;
    assert_eq!(latent.len(), cfg.latent_len(), "latent size");
    assert_eq!(ctx.len(), tl * td, "ctx size (pad to text_len before calling)");
    assert_eq!(cos.len(), n * cfg.head_dim() / 2, "rope table size");

    // --- timestep: sinusoid -> MLP -> e, then silu -> time_projection -> e0 ---
    let te = timestep_embedding::<T>(t, cfg.freq_dim);
    let h0pre = linear(&te, 1, cfg.freq_dim, &w.time0.w, &w.time0.b, dim);
    let h0: Vec<T> = h0pre.iter().map(|&v| silu(v)).collect();
    let e = linear(&h0, 1, dim, &w.time2.w, &w.time2.b, dim);
    let eact: Vec<T> = e.iter().map(|&v| silu(v)).collect();
    let e0 = linear(&eact, 1, dim, &w.time_proj.w, &w.time_proj.b, 6 * dim);

    // --- text: Linear -> GELU(tanh) -> Linear over the padded rows ---
    let th = linear(ctx, tl, td, &w.text0.w, &w.text0.b, dim);
    let thg: Vec<T> = th.iter().map(|&v| gelu(v)).collect();
    let ctxe = linear(&thg, tl, dim, &w.text2.w, &w.text2.b, dim);

    // --- tokens ---
    let (f, h, wd) = cfg.latent;
    let flat = patchify(latent, cfg.in_channels, f, h, wd, ph, pw);
    let mut x = linear(&flat, n, cfg.patch_dim(), &w.patch_embed.w, &w.patch_embed.b, dim);

    let d = cfg.dims();
    let mut caches = Vec::with_capacity(w.blocks.len());
    for bw in &w.blocks {
        let (o, c) = block_forward(d, bw, &x, &e0, &ctxe, cos, sin);
        x = o;
        caches.push(c);
    }

    // --- head: LayerNorm carrying `head.modulation + e`, then the projection ---
    let shift_h: Vec<T> = w.head_mod[..dim].iter().zip(&e).map(|(&a, &b)| a + b).collect();
    let gamma_h: Vec<T> = w.head_mod[dim..].iter().zip(&e).map(|(&a, &b)| T::ONE + a + b).collect();
    let (xhat_h, inv_h) = layernorm(&x, n, dim, cfg.eps);
    let nh = affine(&xhat_h, &gamma_h, &shift_h, n, dim);
    let rows = linear(&nh, n, dim, &w.head.w, &w.head.b, cfg.head_dim_out());
    let pred = unpatchify(&rows, cfg.out_channels, gf, gh, gw, ph, pw);

    let cache = ModelCache {
        te, h0pre, h0, e, eact, ctx_in: ctx.to_vec(), th, thg, flat,
        blocks: caches, xhat_h, inv_h, gamma_h, nh,
    };
    (pred, cache)
}

/// Flow-matching velocity-MSE loss + its `dpred`: `L = mean((pred − v)²)`.
pub fn loss<T: Fp>(pred: &[T], v_target: &[T]) -> (f64, Vec<T>) {
    assert_eq!(pred.len(), v_target.len(), "loss: prediction/target size");
    let n = T::fr(pred.len() as f64);
    let two = T::fr(2.0);
    let mut l = 0.0;
    let mut dpred = vec![T::ZERO; pred.len()];
    for i in 0..pred.len() {
        let err = pred[i] - v_target[i];
        l += (err * err / n).f64();
        dpred[i] = two * err / n;
    }
    (l, dpred)
}

/// Full backward from `dpred` (grad of the loss w.r.t. the predicted latent).
pub fn backward<T: Fp>(cfg: &Cfg, w: &ModelWeights<T>, cache: &ModelCache<T>, dpred: &[T]) -> ModelGrads<T> {
    let (dim, tl, td) = (cfg.dim, cfg.text_len, cfg.text_dim);
    let (gf, gh, gw) = cfg.grid();
    let n = cfg.n_tokens();
    let (_, ph, pw) = cfg.patch;
    let d = cfg.dims();

    // --- head ---
    let drows = unpatchify_bwd(dpred, cfg.out_channels, gf, gh, gw, ph, pw);
    let (dnh, g_head) = linear_bwd(&cache.nh, n, dim, &w.head.w, cfg.head_dim_out(), &drows);
    let mut dgamma_h = vec![T::ZERO; dim];
    let mut dshift_h = vec![T::ZERO; dim];
    let dxhat_h = affine_bwd(&cache.xhat_h, &cache.gamma_h, n, dim, &dnh, &mut dgamma_h, &mut dshift_h);
    let mut head_mod = vec![T::ZERO; 2 * dim];
    head_mod[..dim].copy_from_slice(&dshift_h);
    head_mod[dim..].copy_from_slice(&dgamma_h);
    // `e` enters BOTH head-modulation halves additively.
    let mut de: Vec<T> = dshift_h.iter().zip(&dgamma_h).map(|(&a, &b)| a + b).collect();
    let mut dx = layernorm_bwd(&cache.xhat_h, &cache.inv_h, n, dim, &dxhat_h);

    // --- block stack (reverse), accumulating the two shared adjoints ---
    let mut de0 = vec![T::ZERO; 6 * dim];
    let mut dctx = vec![T::ZERO; tl * dim];
    let mut blocks: Vec<BlockGrads<T>> = Vec::with_capacity(w.blocks.len());
    for (bw, bc) in w.blocks.iter().zip(&cache.blocks).rev() {
        let g = block_backward(d, bw, bc, &dx);
        dx = g.dx.clone();
        for (a, b) in de0.iter_mut().zip(&g.modulation) {
            *a += *b;
        }
        for (a, b) in dctx.iter_mut().zip(&g.dctx) {
            *a += *b;
        }
        blocks.push(g);
    }
    blocks.reverse();

    // --- patch embedding (its input is data: weight grads only) ---
    let (_dflat, g_patch) = linear_bwd(&cache.flat, n, cfg.patch_dim(), &w.patch_embed.w, dim, &dx);

    // --- text embedding ---
    let (dthg, g_text2) = linear_bwd(&cache.thg, tl, dim, &w.text2.w, dim, &dctx);
    let dth: Vec<T> = dthg.iter().zip(&cache.th).map(|(&g, &v)| g * dgelu(v)).collect();
    let (_dctx_in, g_text0) = linear_bwd(&cache.ctx_in, tl, td, &w.text0.w, dim, &dth);

    // --- timestep path: e0 -> silu(e) -> e (+ the head's own `de`) -> MLP ---
    let (deact, g_time_proj) = linear_bwd(&cache.eact, 1, dim, &w.time_proj.w, 6 * dim, &de0);
    for (a, (&g, &v)) in de.iter_mut().zip(deact.iter().zip(&cache.e)) {
        *a += g * dsilu(v);
    }
    let (dh0, g_time2) = linear_bwd(&cache.h0, 1, dim, &w.time2.w, dim, &de);
    let dh0pre: Vec<T> = dh0.iter().zip(&cache.h0pre).map(|(&g, &v)| g * dsilu(v)).collect();
    let (_dte, g_time0) = linear_bwd(&cache.te, 1, cfg.freq_dim, &w.time0.w, dim, &dh0pre);

    ModelGrads {
        patch_embed: g_patch,
        text0: g_text0,
        text2: g_text2,
        time0: g_time0,
        time2: g_time2,
        time_proj: g_time_proj,
        blocks,
        head: g_head,
        head_mod,
    }
}

// ---- flow-matching batch ----

/// One training example, ready for [`forward`].
#[derive(Clone)]
pub struct Batch<T> {
    /// `x_σ`, `[C·F·H·W]`.
    pub latent: Vec<T>,
    /// The text encoding, zero-padded to `[text_len · text_dim]`.
    pub ctx: Vec<T>,
    /// Model time input on the training grid (`σ · num_train_timesteps`).
    pub t: f64,
    pub cos: Vec<T>,
    pub sin: Vec<T>,
    /// Target velocity `v = ε − x₀`.
    pub target: Vec<T>,
}

/// Timesteps the sigma grid is expressed on (`WanConfig::num_train_timesteps`).
pub const TRAIN_TIMESTEPS: f64 = 1000.0;

/// Build one flow-matching batch from a clean latent `x0` (`[C·F·H·W]`), text
/// features `ctx` (`[rows · text_dim]`, `rows <= text_len`), noise level
/// `σ ∈ (0,1]` and standard-normal `noise`:
///
/// `x_σ = (1−σ)·x₀ + σ·ε`, target `v = ε − x₀`, model time `t = σ·1000`.
///
/// That is exactly the convention [`crate::pipeline`]'s solvers invert: both
/// flow solvers recover `x₀ = x_σ − σ·model_out`, which is only consistent with
/// `model_out = ε − x₀`, and they map `σ → t = trunc(σ·1000)` for the model's
/// timestep input. The text rows are zero-padded here, the same hard zeros
/// [`crate::model::text_embed`] applies.
pub fn make_flow_batch<T: Fp>(cfg: &Cfg, x0: &[T], ctx: &[T], rows: usize, sigma: f64, noise: &[T]) -> Batch<T> {
    assert_eq!(x0.len(), cfg.latent_len(), "latent size");
    assert_eq!(noise.len(), x0.len(), "noise size");
    assert!(rows <= cfg.text_len, "context has {rows} rows, text_len is {}", cfg.text_len);
    assert_eq!(ctx.len(), rows * cfg.text_dim, "caption size");
    let s = T::fr(sigma);
    let latent: Vec<T> = x0.iter().zip(noise).map(|(&x, &e)| (T::ONE - s) * x + s * e).collect();
    let target: Vec<T> = x0.iter().zip(noise).map(|(&x, &e)| e - x).collect();
    let mut padded = vec![T::ZERO; cfg.text_len * cfg.text_dim];
    padded[..ctx.len()].copy_from_slice(ctx);
    let tables = cfg.rope_tables();
    let cast = |v: &[f32]| -> Vec<T> { v.iter().map(|&x| T::fr(x as f64)).collect() };
    Batch { latent, ctx: padded, t: sigma * TRAIN_TIMESTEPS, cos: cast(&tables.cos), sin: cast(&tables.sin), target }
}

/// One training evaluation: forward + loss + backward. The f32 instantiation is
/// the finetune trainer's step core.
pub fn grads<T: Fp>(cfg: &Cfg, w: &ModelWeights<T>, b: &Batch<T>) -> (f64, ModelGrads<T>) {
    let (pred, cache) = forward(cfg, w, &b.latent, &b.ctx, b.t, &b.cos, &b.sin);
    let (l, dpred) = loss(&pred, &b.target);
    (l, backward(cfg, w, &cache, &dpred))
}

// ---- weight construction ----

impl ModelWeights<f32> {
    /// Build host training weights from imported tensors (the upstream name
    /// space [`crate::import::dit_manifest`] defines, which is what both
    /// importers produce). Nothing is fused in a Wan checkpoint - q, k, v and
    /// the two FFN linears are all separate tensors - so this is a pure lookup,
    /// with no split to keep in sync with `model.rs`.
    pub fn from_tensors(cfg: &Cfg, ts: &crate::model::Tensors) -> Result<ModelWeights<f32>, String> {
        let get = |name: &str| -> Result<Vec<f32>, String> {
            ts.get(name).map(|(_, v)| v.clone()).ok_or_else(|| format!("from_tensors: missing {name}"))
        };
        let lin = |p: &str| -> Result<Lin<f32>, String> {
            Ok(Lin { w: get(&format!("{p}.weight"))?, b: get(&format!("{p}.bias"))? })
        };
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let b = format!("blocks.{l}");
            blocks.push(BlockW {
                modulation: get(&format!("{b}.modulation"))?,
                sq: lin(&format!("{b}.self_attn.q"))?,
                sk: lin(&format!("{b}.self_attn.k"))?,
                sv: lin(&format!("{b}.self_attn.v"))?,
                so: lin(&format!("{b}.self_attn.o"))?,
                snq: get(&format!("{b}.self_attn.norm_q.weight"))?,
                snk: get(&format!("{b}.self_attn.norm_k.weight"))?,
                cq: lin(&format!("{b}.cross_attn.q"))?,
                ck: lin(&format!("{b}.cross_attn.k"))?,
                cv: lin(&format!("{b}.cross_attn.v"))?,
                co: lin(&format!("{b}.cross_attn.o"))?,
                cnq: get(&format!("{b}.cross_attn.norm_q.weight"))?,
                cnk: get(&format!("{b}.cross_attn.norm_k.weight"))?,
                norm3_w: get(&format!("{b}.norm3.weight"))?,
                norm3_b: get(&format!("{b}.norm3.bias"))?,
                ff1: lin(&format!("{b}.ffn.0"))?,
                ff2: lin(&format!("{b}.ffn.2"))?,
            });
        }
        Ok(ModelWeights {
            patch_embed: lin("patch_embedding")?,
            text0: lin("text_embedding.0")?,
            text2: lin("text_embedding.2")?,
            time0: lin("time_embedding.0")?,
            time2: lin("time_embedding.2")?,
            time_proj: lin("time_projection.1")?,
            blocks,
            head: lin("head.head")?,
            head_mod: get("head.modulation")?,
        })
    }
}

/// Deterministic random init at any scalar type - for gradchecks and synthetic
/// training tests (real runs import a checkpoint).
pub fn init_model<T: Fp>(cfg: &Cfg, seed: u64) -> ModelWeights<T> {
    let mut rng = data::rng::Rng::new(seed);
    let mut v = |n: usize, s: f64| -> Vec<T> { (0..n).map(|_| T::fr((rng.next_f64() - 0.5) * 2.0 * s)).collect() };
    let (dim, ffn) = (cfg.dim, cfg.ffn_dim);
    // Norm gains sit near 1: a gain near 0 would zero every downstream
    // activation and make an FD check pass against an all-zero signal.
    let gain = |n: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Vec<T> {
        r(n, 0.1).iter().map(|&x| T::ONE + x).collect()
    };
    let lin = |out: usize, inn: usize, s: f64, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Lin<T> {
        Lin { w: r(out * inn, s), b: r(out, 0.05) }
    };
    let blocks = (0..cfg.n_layers)
        .map(|_| BlockW {
            modulation: v(6 * dim, 0.05),
            sq: lin(dim, dim, 0.2, &mut v),
            sk: lin(dim, dim, 0.2, &mut v),
            sv: lin(dim, dim, 0.2, &mut v),
            so: lin(dim, dim, 0.2, &mut v),
            snq: gain(dim, &mut v),
            snk: gain(dim, &mut v),
            cq: lin(dim, dim, 0.2, &mut v),
            ck: lin(dim, dim, 0.2, &mut v),
            cv: lin(dim, dim, 0.2, &mut v),
            co: lin(dim, dim, 0.2, &mut v),
            cnq: gain(dim, &mut v),
            cnk: gain(dim, &mut v),
            norm3_w: gain(dim, &mut v),
            norm3_b: v(dim, 0.05),
            ff1: lin(ffn, dim, 0.2, &mut v),
            ff2: lin(dim, ffn, 0.2, &mut v),
        })
        .collect();
    ModelWeights {
        patch_embed: lin(dim, cfg.patch_dim(), 0.2, &mut v),
        text0: lin(dim, cfg.text_dim, 0.2, &mut v),
        text2: lin(dim, dim, 0.2, &mut v),
        time0: lin(dim, cfg.freq_dim, 0.1, &mut v),
        time2: lin(dim, dim, 0.1, &mut v),
        time_proj: lin(6 * dim, dim, 0.1, &mut v),
        blocks,
        head: lin(cfg.head_dim_out(), dim, 0.2, &mut v),
        head_mod: v(2 * dim, 0.05),
    }
}

// ---- parameter enumeration (FD tests + gradcheck::check_wan) ----

/// Every trainable tensor, named exactly as [`crate::import::dit_manifest`]
/// names it, in the manifest's own order (mutable views).
pub fn params_mut<T>(w: &mut ModelWeights<T>) -> Vec<(String, &mut Vec<T>)> {
    let mut v: Vec<(String, &mut Vec<T>)> = vec![
        ("patch_embedding.weight".into(), &mut w.patch_embed.w),
        ("patch_embedding.bias".into(), &mut w.patch_embed.b),
        ("text_embedding.0.weight".into(), &mut w.text0.w),
        ("text_embedding.0.bias".into(), &mut w.text0.b),
        ("text_embedding.2.weight".into(), &mut w.text2.w),
        ("text_embedding.2.bias".into(), &mut w.text2.b),
        ("time_embedding.0.weight".into(), &mut w.time0.w),
        ("time_embedding.0.bias".into(), &mut w.time0.b),
        ("time_embedding.2.weight".into(), &mut w.time2.w),
        ("time_embedding.2.bias".into(), &mut w.time2.b),
        ("time_projection.1.weight".into(), &mut w.time_proj.w),
        ("time_projection.1.bias".into(), &mut w.time_proj.b),
    ];
    for (i, b) in w.blocks.iter_mut().enumerate() {
        let p = format!("blocks.{i}");
        v.push((format!("{p}.modulation"), &mut b.modulation));
        v.push((format!("{p}.norm3.weight"), &mut b.norm3_w));
        v.push((format!("{p}.norm3.bias"), &mut b.norm3_b));
        for (attn, q, k, vv, o, nq, nk) in [
            ("self_attn", &mut b.sq, &mut b.sk, &mut b.sv, &mut b.so, &mut b.snq, &mut b.snk),
            ("cross_attn", &mut b.cq, &mut b.ck, &mut b.cv, &mut b.co, &mut b.cnq, &mut b.cnk),
        ] {
            v.push((format!("{p}.{attn}.q.weight"), &mut q.w));
            v.push((format!("{p}.{attn}.q.bias"), &mut q.b));
            v.push((format!("{p}.{attn}.k.weight"), &mut k.w));
            v.push((format!("{p}.{attn}.k.bias"), &mut k.b));
            v.push((format!("{p}.{attn}.v.weight"), &mut vv.w));
            v.push((format!("{p}.{attn}.v.bias"), &mut vv.b));
            v.push((format!("{p}.{attn}.o.weight"), &mut o.w));
            v.push((format!("{p}.{attn}.o.bias"), &mut o.b));
            v.push((format!("{p}.{attn}.norm_q.weight"), nq));
            v.push((format!("{p}.{attn}.norm_k.weight"), nk));
        }
        v.push((format!("{p}.ffn.0.weight"), &mut b.ff1.w));
        v.push((format!("{p}.ffn.0.bias"), &mut b.ff1.b));
        v.push((format!("{p}.ffn.2.weight"), &mut b.ff2.w));
        v.push((format!("{p}.ffn.2.bias"), &mut b.ff2.b));
    }
    v.push(("head.head.weight".into(), &mut w.head.w));
    v.push(("head.head.bias".into(), &mut w.head.b));
    v.push(("head.modulation".into(), &mut w.head_mod));
    v
}

/// Gradient views in the SAME order as [`params_mut`].
pub fn grad_views<T>(g: &ModelGrads<T>) -> Vec<(String, &Vec<T>)> {
    let mut v: Vec<(String, &Vec<T>)> = vec![
        ("patch_embedding.weight".into(), &g.patch_embed.w),
        ("patch_embedding.bias".into(), &g.patch_embed.b),
        ("text_embedding.0.weight".into(), &g.text0.w),
        ("text_embedding.0.bias".into(), &g.text0.b),
        ("text_embedding.2.weight".into(), &g.text2.w),
        ("text_embedding.2.bias".into(), &g.text2.b),
        ("time_embedding.0.weight".into(), &g.time0.w),
        ("time_embedding.0.bias".into(), &g.time0.b),
        ("time_embedding.2.weight".into(), &g.time2.w),
        ("time_embedding.2.bias".into(), &g.time2.b),
        ("time_projection.1.weight".into(), &g.time_proj.w),
        ("time_projection.1.bias".into(), &g.time_proj.b),
    ];
    for (i, b) in g.blocks.iter().enumerate() {
        let p = format!("blocks.{i}");
        v.push((format!("{p}.modulation"), &b.modulation));
        v.push((format!("{p}.norm3.weight"), &b.norm3_w));
        v.push((format!("{p}.norm3.bias"), &b.norm3_b));
        for (attn, q, k, vv, o, nq, nk) in [
            ("self_attn", &b.sq, &b.sk, &b.sv, &b.so, &b.snq, &b.snk),
            ("cross_attn", &b.cq, &b.ck, &b.cv, &b.co, &b.cnq, &b.cnk),
        ] {
            v.push((format!("{p}.{attn}.q.weight"), &q.w));
            v.push((format!("{p}.{attn}.q.bias"), &q.b));
            v.push((format!("{p}.{attn}.k.weight"), &k.w));
            v.push((format!("{p}.{attn}.k.bias"), &k.b));
            v.push((format!("{p}.{attn}.v.weight"), &vv.w));
            v.push((format!("{p}.{attn}.v.bias"), &vv.b));
            v.push((format!("{p}.{attn}.o.weight"), &o.w));
            v.push((format!("{p}.{attn}.o.bias"), &o.b));
            v.push((format!("{p}.{attn}.norm_q.weight"), nq));
            v.push((format!("{p}.{attn}.norm_k.weight"), nk));
        }
        v.push((format!("{p}.ffn.0.weight"), &b.ff1.w));
        v.push((format!("{p}.ffn.0.bias"), &b.ff1.b));
        v.push((format!("{p}.ffn.2.weight"), &b.ff2.w));
        v.push((format!("{p}.ffn.2.bias"), &b.ff2.b));
    }
    v.push(("head.head.weight".into(), &g.head.w));
    v.push(("head.head.bias".into(), &g.head.b));
    v.push(("head.modulation".into(), &g.head_mod));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generic patchify/unpatchify must be the SAME permutations the
    /// forward path uses. They are two different orderings (channel-outermost
    /// in, channel-innermost out) and reusing one for both produces a shuffled
    /// latent that still looks like video - so this pins each against its f32
    /// original: `patchify` against `crate::model`'s own (Wan's genuinely
    /// different, channel-outermost ordering - not shared), `unpatchify`
    /// against `dit::patchify::unpatchify` (the shared channel-innermost
    /// ordering `crate::model::postprocess` itself calls, at a temporal patch
    /// of 1).
    #[test]
    fn the_generic_patchify_agrees_with_the_forward_paths() {
        let (c, f, h, w, ph, pw) = (3usize, 3usize, 4usize, 6usize, 2usize, 2usize);
        let x: Vec<f32> = (0..(c * f * h * w)).map(|i| (i % 29) as f32 * 0.37 - 3.0).collect();
        assert_eq!(patchify(&x, c, f, h, w, ph, pw), crate::model::patchify(&x, c, f, h, w, ph, pw));
        let rows: Vec<f32> = (0..(f * (h / ph) * (w / pw) * ph * pw * c)).map(|i| (i % 17) as f32 - 8.0).collect();
        assert_eq!(
            unpatchify(&rows, c, f, h / ph, w / pw, ph, pw),
            dit::patchify::unpatchify(&rows, c, f, h / ph, w / pw, 1, ph, pw)
        );
    }

    /// `unpatchify_bwd` is the adjoint of `unpatchify`: the permutation read
    /// the other way, so `⟨unpatchify(r), g⟩ == ⟨r, unpatchify_bwd(g)⟩`.
    #[test]
    fn unpatchify_backward_is_the_adjoint() {
        let (c, f, ht, wt, ph, pw) = (3usize, 3usize, 2usize, 3usize, 2usize, 2usize);
        let n = f * ht * wt * ph * pw * c;
        let rows: Vec<f64> = (0..n).map(|i| (i % 13) as f64 * 0.5 - 3.0).collect();
        let g: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let lhs: f64 = unpatchify(&rows, c, f, ht, wt, ph, pw).iter().zip(&g).map(|(a, b)| a * b).sum();
        let rhs: f64 = rows.iter().zip(&unpatchify_bwd(&g, c, f, ht, wt, ph, pw)).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-12, "{lhs} vs {rhs}");
    }

    /// The tiny config must not accidentally make two different quantities
    /// equal - that is what hides a transposition bug. Guard it explicitly so a
    /// later edit cannot quietly re-introduce a coincidence.
    #[test]
    fn the_tiny_config_has_no_coincidental_dimensions() {
        let c = Cfg::tiny();
        let (gf, gh, gw) = c.grid();
        let dims = [
            c.dim, c.ffn_dim, c.n_heads, c.head_dim(), c.text_len, c.text_dim, c.freq_dim,
            c.n_tokens(), c.patch_dim(), c.head_dim_out(), gf, gh, gw,
        ];
        // The genuinely-equal pairs are the ones upstream forces: in == out
        // channels, and the (h, w) RoPE axes. Everything else must differ.
        assert_eq!(c.in_channels, c.out_channels, "the velocity target lives in the input's channel space");
        assert_eq!(c.rope_axes[1], c.rope_axes[2], "height and width axes are equal upstream");
        assert_eq!(c.rope_axes.iter().sum::<usize>(), c.head_dim(), "the axes must tile head_dim");
        assert_ne!(gf, gh);
        assert_ne!(gh, gw);
        assert_ne!(c.n_tokens(), c.text_len, "token count must differ from the text row count");
        assert_ne!(c.dim, c.ffn_dim);
        assert_ne!(c.n_heads, c.head_dim());
        assert!(dims.iter().all(|&d| d > 0));
    }

    /// The batch convention must be the one the samplers invert: at σ = 1 the
    /// input is pure noise, at σ = 0 it is the clean latent, the target is
    /// always `ε − x₀`, and the model time is `σ·1000`.
    #[test]
    fn flow_batch_matches_the_sampler_convention() {
        let cfg = Cfg::tiny();
        let x0: Vec<f64> = (0..cfg.latent_len()).map(|i| i as f64 * 0.01).collect();
        let noise: Vec<f64> = (0..cfg.latent_len()).map(|_| 0.5).collect();
        let ctx = vec![0.25f64; 2 * cfg.text_dim];
        let b1 = make_flow_batch(&cfg, &x0, &ctx, 2, 1.0, &noise);
        assert_eq!(b1.latent, noise);
        assert_eq!(b1.t, 1000.0);
        let b0 = make_flow_batch(&cfg, &x0, &ctx, 2, 0.0, &noise);
        assert_eq!(b0.latent, x0);
        assert_eq!(b0.t, 0.0);
        for (i, &v) in b0.target.iter().enumerate() {
            assert!((v - (0.5 - x0[i])).abs() < 1e-12);
        }
        // The text pad is hard zeros beyond the supplied rows.
        assert_eq!(b0.ctx.len(), cfg.text_len * cfg.text_dim);
        assert!(b0.ctx[2 * cfg.text_dim..].iter().all(|&v| v == 0.0));
    }

    /// Every tensor of the manifest must appear exactly once in `params_mut`,
    /// and `grad_views` must line up name-for-name and length-for-length. This
    /// is what makes "the gradcheck covers every parameter" checkable rather
    /// than asserted.
    #[test]
    fn params_and_grads_cover_the_whole_manifest_in_the_same_order() {
        let cfg = Cfg::tiny();
        let mut w = init_model::<f64>(&cfg, 3);
        let names: Vec<String> = params_mut(&mut w).into_iter().map(|(n, _)| n).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate parameter name");
        // The manifest the importer validates against, at the same config.
        let wc = WanConfig {
            name: "tiny",
            dim: cfg.dim,
            ffn_dim: cfg.ffn_dim,
            num_heads: cfg.n_heads,
            num_layers: cfg.n_layers,
            in_channels: cfg.in_channels,
            out_channels: cfg.out_channels,
            text_dim: cfg.text_dim,
            text_len: cfg.text_len,
            freq_dim: cfg.freq_dim,
            ..WanConfig::t2v_1_3b()
        };
        let mut manifest: Vec<String> = crate::import::dit_manifest(&wc).into_iter().map(|(n, _)| n).collect();
        manifest.sort();
        assert_eq!(sorted, manifest, "params_mut must enumerate exactly the checkpoint manifest");

        // ...and the grads line up, entry for entry.
        let b = make_flow_batch(&cfg, &vec![0.1; cfg.latent_len()], &vec![0.2; cfg.text_dim], 1, 0.4, &vec![0.3; cfg.latent_len()]);
        let (_l, g) = grads(&cfg, &w, &b);
        let gv = grad_views(&g);
        let pm: Vec<(String, usize)> = params_mut(&mut w).into_iter().map(|(n, v)| (n, v.len())).collect();
        assert_eq!(gv.len(), pm.len());
        for ((gn, gvv), (pn, pl)) in gv.iter().zip(&pm) {
            assert_eq!(gn, pn, "grad_views order");
            assert_eq!(gvv.len(), *pl, "{gn}: grad length");
        }
    }
}
