// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full video-only LTX DiT **training** reference (host): forward + analytic
//! backward for the whole transformer under the flow-matching velocity-MSE
//! loss.
//!
//! This chains the block reference ([`crate::grad`]) across the stack and
//! wraps it with everything a block does not have -
//! `crate::dit::LtxDit::forward`'s host-side surrounding logic, differentiated:
//! `patchify_proj`, the keyframe absolute-position add, the per-token
//! PixArt timestep-embedding MLP -> adaLN-single raw table, and the output
//! stage (`LayerNorm(no affine)` -> per-token modulate by
//! `scale_shift_table + embedded_timestep` -> `proj_out`).
//!
//! ## `latent` is already tokens, not pixels
//!
//! Unlike `wan::modelgrad`, there is no patchify/unpatchify here: LTX's own
//! `LtxDit::forward` takes `latent: [T, in_channels]` - already the
//! per-token sequence (video-to-token patchification is a separate,
//! upstream concern outside the DiT's own math scope, see
//! `crate::patchify`) - so the training target lives directly in that same
//! `[T, out_channels]` space and [`make_flow_batch`] needs no patch geometry
//! at all.
//!
//! ## The conditioning graph is what makes this one model
//!
//! Three couplings have to be right, invisible in a forward-only test - the
//! same discipline `wan::modelgrad`'s own doc calls out, with LTX's own
//! shape:
//!
//! 1. **`adaln_shared` is shared by every block.** Each block's own
//!    `d(scale_shift_table)` is the ROW-SUM of the site gradient; the
//!    UNREDUCED site gradient (`dadaln_shared`) is this block's contribution
//!    to the shared per-token table, and [`backward`] sums those
//!    contributions over the whole stack before entering `adaln_single.linear`.
//! 2. **`embedded_timestep` (the timestep MLP's raw output, before `SiLU`)
//!    feeds TWO consumers**: `SiLU(embedded) -> adaln_single.linear ->
//!    adaln_shared` (read by every block) AND the output stage's own
//!    modulation (`scale_shift_table[shift,scale] + embedded_timestep`,
//!    directly, no linear in between - PixArt-alpha's own final-layer
//!    convention). `d(embedded_timestep)` is the SUM of both paths' gradient,
//!    exactly the T5 `rel_bias`/Wan `time_embedding.2.bias` fold shape the
//!    porting playbook warns a directional-only check can under-cover -
//!    hence `gradcheck::check_ltxv_conditioning`'s dedicated elementwise gate.
//! 3. `context` (the raw text encoding) is a genuine EXTERNAL input at this
//!    milestone (no text encoder inside this crate's DiT scope), so
//!    `BlockGrads::dctx` is exposed for block-level FD coverage but is not
//!    itself routed anywhere further by [`backward`].
//!
//! Generic over [`Fp`] like [`crate::grad`]: the `f64` instantiation is the
//! FD gradcheck oracle, the `f32` instantiation is the trainer
//! [`crate::finetune`] drives - one implementation, no oracle/trainer drift.

use crate::config::LtxDitConfig;
use crate::grad::{block_backward, block_forward, dsilu, linear, linear_bwd, silu, BlockCache, BlockGrads, BlockW, Dims, Fp, Lin};
use crate::rope::ltx_rope_tables;

/// Shape of the training problem: the [`LtxDitConfig`] fields the host path
/// needs plus the token/context extent being trained on.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub norm_eps: f64,
    pub positional_embedding_theta: f64,
    pub positional_embedding_max_pos: [u32; 3],
    pub timestep_scale_multiplier: u32,
    /// Token count (`crate::dit::LtxDit::forward`'s `t` - already the
    /// patchified sequence length, see this module's doc).
    pub t: usize,
    /// Text context rows.
    pub context_len: usize,
}

impl Cfg {
    /// Derive from an [`LtxDitConfig`] at token extent `t` / context extent
    /// `context_len`. `cross_attention_dim` must equal `inner_dim` - the
    /// same M3 assumption `crate::dit::LtxDit::new` asserts (`caption_projection:
    /// None`).
    pub fn from_ltx(c: &LtxDitConfig, t: usize, context_len: usize) -> Cfg {
        c.assert_supported();
        assert_eq!(c.cross_attention_dim, c.inner_dim, "ltxv training: M3 assumes cross_attention_dim == inner_dim");
        Cfg {
            dim: c.inner_dim as usize,
            num_heads: c.num_heads as usize,
            num_layers: c.num_layers as usize,
            in_channels: c.in_channels as usize,
            out_channels: c.out_channels as usize,
            norm_eps: c.norm_eps as f64,
            positional_embedding_theta: c.positional_embedding_theta,
            positional_embedding_max_pos: c.positional_embedding_max_pos,
            timestep_scale_multiplier: c.timestep_scale_multiplier,
            t,
            context_len,
        }
    }

    /// Tiny video-only-DiT-topology config for gradchecks and unit tests.
    ///
    /// Deliberately distinct from `LtxDitConfig::tiny`'s own golden-fixture
    /// extent (`t=8, context_len=6`, used by the host-vs-GPU forward parity
    /// test instead) so this gradcheck fixture cannot coincidentally share a
    /// dimension with anything the golden pins - `t=7`/`context_len=5`, the
    /// same distinctness discipline `wan::modelgrad::Cfg::tiny`'s own doc
    /// explains (equal dims are what hides a transposition bug).
    pub fn tiny() -> Cfg {
        Cfg::from_ltx(&LtxDitConfig::tiny(), 7, 5)
    }

    pub const fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    pub fn dims(&self) -> Dims {
        Dims { t: self.t, te: self.context_len, dim: self.dim, nh: self.num_heads, eps: self.norm_eps }
    }

    /// A simple, deterministic 3-axis position grid for `self.t` tokens: axis
    /// 0 (frame) walks `[0,1), [1,2), ...`, axes 1/2 (height/width) are
    /// pinned at the single bound `[0,1)` - a 1-D "T frames, 1x1 spatial"
    /// sequence. Valid RoPE input (the construction only reads per-axis
    /// `[start,end)` midpoints, see `crate::rope`'s doc) and enough to
    /// exercise every training path; a real (f,h,w) video grid is not
    /// needed for this milestone's synthetic training data.
    pub fn simple_positions(&self) -> Vec<f32> {
        let t = self.t;
        let mut v = vec![0f32; 3 * t * 2];
        for ti in 0..t {
            v[ti * 2] = ti as f32;
            v[ti * 2 + 1] = ti as f32 + 1.0;
            v[(t + ti) * 2 + 1] = 1.0;
            v[(2 * t + ti) * 2 + 1] = 1.0;
        }
        v
    }

    /// This config's RoPE tables (`crate::rope::ltx_rope_tables`, host f32),
    /// cast to `T` by [`make_flow_batch`] - the input adjoint is never
    /// needed (the table is fixed data, not a trainable parameter).
    pub fn rope_tables_f32(&self, positions: &[f32]) -> crate::rope::LtxRopeTables {
        ltx_rope_tables(self.dim as u32, self.num_heads as u32, self.positional_embedding_theta, &self.positional_embedding_max_pos, positions, self.t)
    }
}

/// Every trainable tensor of the DiT, in the host training layout, named as
/// `crate::dit::dit_tensor_manifest` names them.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelWeights<T> {
    pub patchify_proj: Lin<T>,
    pub keyframes_abs_pos_embedding: Vec<T>,
    pub timestep_embedder_l1: Lin<T>,
    pub timestep_embedder_l2: Lin<T>,
    pub adaln_linear: Lin<T>,
    /// The OUTPUT stage's own `[2*dim]` table - a DIFFERENT tensor from any
    /// block's own `scale_shift_table` (that one is `[9*dim]` and named
    /// `transformer_blocks.{l}.scale_shift_table`; this one is the top-level
    /// `scale_shift_table` key, see `crate::dit::dit_tensor_manifest`).
    pub output_scale_shift_table: Vec<T>,
    pub proj_out: Lin<T>,
    pub blocks: Vec<BlockW<T>>,
}

/// Grads mirroring [`ModelWeights`].
#[derive(Clone, Debug)]
pub struct ModelGrads<T> {
    pub patchify_proj: Lin<T>,
    pub keyframes_abs_pos_embedding: Vec<T>,
    pub timestep_embedder_l1: Lin<T>,
    pub timestep_embedder_l2: Lin<T>,
    pub adaln_linear: Lin<T>,
    pub output_scale_shift_table: Vec<T>,
    pub proj_out: Lin<T>,
    pub blocks: Vec<BlockGrads<T>>,
}

/// `sinusoid(t,dim) = cat([cos,sin])` over `10000^(-k/half)` - the
/// generic-`T` twin of `model::hostmath::timestep_embedding` at
/// `flip_sin_to_cos=true, downscale_freq_shift=0.0, max_period=10000.0`
/// (`dit::timestep::pixart_timestep_embed`'s own sinusoid stage). A
/// deliberate second implementation carrying no parameters - pure input math.
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

/// Saved forward state for the backward pass.
pub struct ModelCache<T> {
    te_all: Vec<T>,
    h0pre: Vec<T>,
    h0: Vec<T>,
    embedded: Vec<T>,
    embedded_silu: Vec<T>,
    flat: Vec<T>,
    x_pre_blocks: Vec<T>,
    keyframes_mask: Vec<f64>,
    blocks: Vec<BlockCache<T>>,
    xhat_out: Vec<T>,
    inv_out: Vec<T>,
    xo: Vec<T>,
}

/// Full forward. `latent`: `[t*in_channels]` (already tokens - see this
/// module's doc). `timesteps`: `[t]`, per-token `denoise_mask*sigma`
/// (RAW, matching `crate::dit::LtxDit::forward`'s own contract - the
/// `timestep_scale_multiplier` scaling happens inside). `keyframes_mask`:
/// `[t]`, non-zero marks a keyframe token. `context`: `[context_len*dim]`
/// raw text context. `cos`/`sin`: `[nh, t, hd/2]`
/// (`crate::rope::LtxRopeTables`'s layout). Returns the velocity prediction
/// `[t*out_channels]` plus the cache.
#[allow(clippy::too_many_arguments)]
pub fn forward<T: Fp>(
    cfg: &Cfg,
    w: &ModelWeights<T>,
    latent: &[T],
    timesteps: &[f64],
    keyframes_mask: &[f64],
    context: &[T],
    cos: &[T],
    sin: &[T],
) -> (Vec<T>, ModelCache<T>) {
    let (t, dim, ctx_len) = (cfg.t, cfg.dim, cfg.context_len);
    assert_eq!(latent.len(), t * cfg.in_channels, "latent size");
    assert_eq!(timesteps.len(), t, "timesteps size");
    assert_eq!(keyframes_mask.len(), t, "keyframes_mask size");
    assert_eq!(context.len(), ctx_len * dim, "ctx size");

    // --- patchify_proj + keyframes embedding ---
    let flat = latent.to_vec();
    let mut x = linear(&flat, t, cfg.in_channels, &w.patchify_proj.w, &w.patchify_proj.b, dim);
    for ti in 0..t {
        if keyframes_mask[ti] > 0.0 {
            for d in 0..dim {
                x[ti * dim + d] += w.keyframes_abs_pos_embedding[d];
            }
        }
    }
    let x_pre_blocks = x.clone();

    // --- per-token timestep embedding + adaLN-single raw table ---
    let ts_scaled: Vec<f64> = timesteps.iter().map(|&v| v * cfg.timestep_scale_multiplier as f64).collect();
    let mut te_all = vec![T::ZERO; t * 256];
    for ti in 0..t {
        let te_i: Vec<T> = timestep_embedding(ts_scaled[ti], 256);
        te_all[ti * 256..(ti + 1) * 256].copy_from_slice(&te_i);
    }
    let h0pre = linear(&te_all, t, 256, &w.timestep_embedder_l1.w, &w.timestep_embedder_l1.b, dim);
    let h0: Vec<T> = h0pre.iter().map(|&v| silu(v)).collect();
    let embedded = linear(&h0, t, dim, &w.timestep_embedder_l2.w, &w.timestep_embedder_l2.b, dim); // == embedded_timestep
    let embedded_silu: Vec<T> = embedded.iter().map(|&v| silu(v)).collect();
    let adaln_shared = linear(&embedded_silu, t, dim, &w.adaln_linear.w, &w.adaln_linear.b, 9 * dim);

    // --- block stack ---
    let d = cfg.dims();
    let mut caches = Vec::with_capacity(w.blocks.len());
    for bw in &w.blocks {
        let (o, c) = block_forward(d, bw, &x, &adaln_shared, context, cos, sin);
        x = o;
        caches.push(c);
    }

    // --- output stage: LayerNorm(no affine) -> modulate by
    // (scale_shift_table + embedded_timestep) -> proj_out ---
    let mut shift = vec![T::ZERO; t * dim];
    let mut one_plus_scale = vec![T::ZERO; t * dim];
    for ti in 0..t {
        for dd in 0..dim {
            shift[ti * dim + dd] = w.output_scale_shift_table[dd] + embedded[ti * dim + dd];
            one_plus_scale[ti * dim + dd] = T::ONE + w.output_scale_shift_table[dim + dd] + embedded[ti * dim + dd];
        }
    }
    let (xhat_out, inv_out) = layernorm(&x, t, dim, cfg.norm_eps);
    let mut xo = vec![T::ZERO; t * dim];
    for i in 0..t * dim {
        xo[i] = xhat_out[i] * one_plus_scale[i] + shift[i];
    }
    let pred = linear(&xo, t, dim, &w.proj_out.w, &w.proj_out.b, cfg.out_channels);

    let cache = ModelCache {
        te_all, h0pre, h0, embedded, embedded_silu, flat, x_pre_blocks,
        keyframes_mask: keyframes_mask.to_vec(), blocks: caches, xhat_out, inv_out, xo,
    };
    (pred, cache)
}

/// Affine-free LayerNorm over the last `d` of `[rows,d]` (population
/// variance) - `crate::dit::layernorm_noaffine`'s generic-`T` twin, the
/// output stage's `norm_out` (LayerNorm, NOT RMSNorm - see `crate::dit`'s
/// module doc). Returns `(xhat, inv)`.
fn layernorm<T: Fp>(x: &[T], rows: usize, d: usize, eps: f64) -> (Vec<T>, Vec<T>) {
    let mut y = vec![T::ZERO; rows * d];
    let mut inv = vec![T::ZERO; rows];
    let dn = T::fr(d as f64);
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let mut mean = T::ZERO;
        for &v in xr {
            mean += v;
        }
        mean = mean / dn;
        let mut var = T::ZERO;
        for &v in xr {
            var += (v - mean) * (v - mean);
        }
        var = var / dn;
        let iv = T::ONE / (var + T::fr(eps)).sqrt();
        inv[r] = iv;
        for c in 0..d {
            y[r * d + c] = (xr[c] - mean) * iv;
        }
    }
    (y, inv)
}

/// [`layernorm`] backward from the cached `xhat`.
fn layernorm_bwd<T: Fp>(xhat: &[T], inv: &[T], rows: usize, d: usize, dxhat: &[T]) -> Vec<T> {
    let mut dx = vec![T::ZERO; rows * d];
    let dn = T::fr(d as f64);
    for r in 0..rows {
        let (mut mdy, mut mdyx) = (T::ZERO, T::ZERO);
        for c in 0..d {
            mdy += dxhat[r * d + c];
            mdyx += dxhat[r * d + c] * xhat[r * d + c];
        }
        mdy = mdy / dn;
        mdyx = mdyx / dn;
        for c in 0..d {
            dx[r * d + c] = inv[r] * (dxhat[r * d + c] - mdy - xhat[r * d + c] * mdyx);
        }
    }
    dx
}

/// Flow-matching velocity-MSE loss + its `dpred`: `L = mean((pred - v)^2)`.
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
    let (t, dim, ctx_len) = (cfg.t, cfg.dim, cfg.context_len);
    let d = cfg.dims();

    // --- output stage ---
    let (dxo, g_proj_out) = linear_bwd(&cache.xo, t, dim, &w.proj_out.w, cfg.out_channels, dpred);
    let mut dshift = vec![T::ZERO; t * dim];
    let mut dscale = vec![T::ZERO; t * dim];
    let mut dxhat_out = vec![T::ZERO; t * dim];
    for i in 0..t * dim {
        // xo = xhat_out*one_plus_scale + shift
        dxhat_out[i] = {
            let one_plus_scale_i = T::ONE + w.output_scale_shift_table[dim + i % dim] + cache.embedded[i];
            one_plus_scale_i * dxo[i]
        };
        dscale[i] = cache.xhat_out[i] * dxo[i];
        dshift[i] = dxo[i];
    }
    let mut g_output_sst = vec![T::ZERO; 2 * dim];
    // embedded_timestep is added directly to BOTH shift and scale (PixArt's
    // own final-layer convention), so its gradient is their SUM - see this
    // module's doc, coupling 2.
    let mut d_embedded = vec![T::ZERO; t * dim];
    for ti in 0..t {
        for dd in 0..dim {
            g_output_sst[dd] += dshift[ti * dim + dd];
            g_output_sst[dim + dd] += dscale[ti * dim + dd];
            d_embedded[ti * dim + dd] += dshift[ti * dim + dd] + dscale[ti * dim + dd];
        }
    }
    let dx_from_out = layernorm_bwd(&cache.xhat_out, &cache.inv_out, t, dim, &dxhat_out);

    // --- block stack (reverse), accumulating the shared adjoint ---
    let mut dadaln_shared = vec![T::ZERO; t * 9 * dim];
    let mut dx = dx_from_out;
    let mut blocks: Vec<BlockGrads<T>> = Vec::with_capacity(w.blocks.len());
    for (bw, bc) in w.blocks.iter().zip(&cache.blocks).rev() {
        let g = block_backward(d, bw, bc, &dx);
        dx = g.dx.clone();
        for (acc, &gi) in dadaln_shared.iter_mut().zip(&g.dadaln_shared) {
            *acc += gi;
        }
        blocks.push(g);
    }
    blocks.reverse();

    // --- adaLN-single: adaln_shared -> SiLU(embedded) -> embedded (+ the
    // output stage's own `d_embedded`) -> timestep MLP ---
    let (d_embedded_silu, g_adaln_linear) = linear_bwd(&cache.embedded_silu, t, dim, &w.adaln_linear.w, 9 * dim, &dadaln_shared);
    for i in 0..t * dim {
        d_embedded[i] += d_embedded_silu[i] * dsilu(cache.embedded[i]);
    }
    let (dh0, g_l2) = linear_bwd(&cache.h0, t, dim, &w.timestep_embedder_l2.w, dim, &d_embedded);
    let dh0pre: Vec<T> = dh0.iter().zip(&cache.h0pre).map(|(&g, &v)| g * dsilu(v)).collect();
    let (_dte, g_l1) = linear_bwd(&cache.te_all, t, 256, &w.timestep_embedder_l1.w, dim, &dh0pre);

    // --- keyframes + patchify_proj (their inputs are data: weight grads
    // only) ---
    let mut g_keyframes = vec![T::ZERO; dim];
    for ti in 0..t {
        if cache.keyframes_mask[ti] > 0.0 {
            for dd in 0..dim {
                g_keyframes[dd] += dx[ti * dim + dd];
            }
        }
    }
    let (_dflat, g_patchify) = linear_bwd(&cache.flat, t, cfg.in_channels, &w.patchify_proj.w, dim, &dx);
    let _ = &cache.x_pre_blocks; // shape sanity only; the residual grad IS `dx` itself (add is identity).
    let _ = ctx_len;

    ModelGrads {
        patchify_proj: g_patchify,
        keyframes_abs_pos_embedding: g_keyframes,
        timestep_embedder_l1: g_l1,
        timestep_embedder_l2: g_l2,
        adaln_linear: g_adaln_linear,
        output_scale_shift_table: g_output_sst,
        proj_out: g_proj_out,
        blocks,
    }
}

// ---- flow-matching batch ----

/// One training example, ready for [`forward`].
#[derive(Clone)]
pub struct Batch<T> {
    /// `x_σ`, `[t*in_channels]`.
    pub latent: Vec<T>,
    /// The text encoding, `[context_len*dim]`.
    pub ctx: Vec<T>,
    /// Model per-token timestep input (`sigma` broadcast uniformly - see
    /// this function's doc), RAW (the `timestep_scale_multiplier` scaling
    /// happens inside [`forward`]).
    pub timesteps: Vec<f64>,
    pub keyframes_mask: Vec<f64>,
    pub cos: Vec<T>,
    pub sin: Vec<T>,
    /// Target velocity `v = ε - x0`.
    pub target: Vec<T>,
}

/// Build one flow-matching batch from a clean latent `x0` (`[t*in_channels]`,
/// already tokens - see this module's doc), text features `ctx`
/// (`[context_len*dim]`), noise level `σ ∈ (0,1]` and standard-normal `noise`:
///
/// `x_σ = (1-σ)·x0 + σ·ε`, target `v = ε - x0`, per-token model timestep
/// `σ` broadcast to every token (diffusion forcing's per-token
/// `denoise_mask*sigma` collapses to a single scalar sigma when every token
/// is denoised uniformly - `keyframes_mask` all-ones, the same "no partial
/// denoise" simplification `crate::pipeline::SingleForwardModel` already
/// makes). That is exactly the convention `crate::pipeline::to_denoised`
/// inverts: `x0 = x_σ - σ·model_out`, consistent only with
/// `model_out = ε - x0`.
pub fn make_flow_batch<T: Fp>(cfg: &Cfg, x0: &[T], ctx: &[T], sigma: f64, noise: &[T]) -> Batch<T> {
    assert_eq!(x0.len(), cfg.t * cfg.in_channels, "latent size");
    assert_eq!(noise.len(), x0.len(), "noise size");
    assert_eq!(ctx.len(), cfg.context_len * cfg.dim, "ctx size");
    let s = T::fr(sigma);
    let latent: Vec<T> = x0.iter().zip(noise).map(|(&x, &e)| (T::ONE - s) * x + s * e).collect();
    let target: Vec<T> = x0.iter().zip(noise).map(|(&x, &e)| e - x).collect();
    let positions = cfg.simple_positions();
    let tables = cfg.rope_tables_f32(&positions);
    let cast = |v: &[f32]| -> Vec<T> { v.iter().map(|&x| T::fr(x as f64)).collect() };
    Batch {
        latent,
        ctx: ctx.to_vec(),
        timesteps: vec![sigma; cfg.t],
        keyframes_mask: vec![1.0; cfg.t],
        cos: cast(&tables.cos),
        sin: cast(&tables.sin),
        target,
    }
}

/// One training evaluation: forward + loss + backward. The f32
/// instantiation is the finetune trainer's step core.
pub fn grads<T: Fp>(cfg: &Cfg, w: &ModelWeights<T>, b: &Batch<T>) -> (f64, ModelGrads<T>) {
    let (pred, cache) = forward(cfg, w, &b.latent, &b.timesteps, &b.keyframes_mask, &b.ctx, &b.cos, &b.sin);
    let (l, dpred) = loss(&pred, &b.target);
    (l, backward(cfg, w, &cache, &dpred))
}

// ---- weight construction ----

impl ModelWeights<f32> {
    /// Build host training weights from imported tensors
    /// (`crate::dit::dit_tensor_manifest`'s name space, which
    /// `crate::dit::load_tiny_weights` and any future real-checkpoint
    /// importer both produce).
    pub fn from_tensors(cfg: &Cfg, ts: &vae::blocks::Tensors) -> Result<ModelWeights<f32>, String> {
        let get = |name: &str| -> Result<Vec<f32>, String> { ts.get(name).map(|(_, v)| v.clone()).ok_or_else(|| format!("from_tensors: missing {name}")) };
        let lin = |p: &str| -> Result<Lin<f32>, String> { Ok(Lin { w: get(&format!("{p}.weight"))?, b: get(&format!("{p}.bias"))? }) };
        let lin_nb = |p: &str| -> Result<crate::grad::LinNB<f32>, String> { Ok(crate::grad::LinNB { w: get(&format!("{p}.weight"))? }) };
        let attn = |p: &str| -> Result<crate::grad::AttnW<f32>, String> {
            Ok(crate::grad::AttnW {
                q: lin(&format!("{p}.to_q"))?,
                k: lin(&format!("{p}.to_k"))?,
                v: lin(&format!("{p}.to_v"))?,
                o: lin(&format!("{p}.to_out.0"))?,
                qn: get(&format!("{p}.q_norm.weight"))?,
                kn: get(&format!("{p}.k_norm.weight"))?,
            })
        };
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{l}");
            blocks.push(BlockW {
                scale_shift_table: get(&format!("{p}.scale_shift_table"))?,
                prompt_scale_shift_table: get(&format!("{p}.prompt_scale_shift_table"))?,
                attn1: attn(&format!("{p}.attn1"))?,
                attn2: attn(&format!("{p}.attn2"))?,
                ff1: lin_nb(&format!("{p}.ff.net.0.proj"))?,
                ff2: lin_nb(&format!("{p}.ff.net.2"))?,
            });
        }
        Ok(ModelWeights {
            patchify_proj: lin("patchify_proj")?,
            keyframes_abs_pos_embedding: get("keyframes_abs_pos_embedding")?,
            timestep_embedder_l1: lin("adaln_single.emb.timestep_embedder.linear_1")?,
            timestep_embedder_l2: lin("adaln_single.emb.timestep_embedder.linear_2")?,
            adaln_linear: lin("adaln_single.linear")?,
            output_scale_shift_table: get("scale_shift_table")?,
            proj_out: lin("proj_out")?,
            blocks,
        })
    }
}

/// Deterministic random init at any scalar type - for gradchecks and
/// synthetic training tests (real runs import a checkpoint via
/// [`ModelWeights::from_tensors`]).
pub fn init_model<T: Fp>(cfg: &Cfg, seed: u64) -> ModelWeights<T> {
    let mut rng = data::rng::Rng::new(seed);
    let mut v = |n: usize, s: f64| -> Vec<T> { (0..n).map(|_| T::fr((rng.next_f64() - 0.5) * 2.0 * s)).collect() };
    // Norm gains sit near 1: a gain near 0 would zero every downstream
    // activation and make an FD check pass against an all-zero signal.
    let gain = |n: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Vec<T> { r(n, 0.1).iter().map(|&x| T::ONE + x).collect() };
    let lin = |out: usize, inn: usize, s: f64, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Lin<T> { Lin { w: r(out * inn, s), b: r(out, 0.05) } };
    let lin_nb = |out: usize, inn: usize, s: f64, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> crate::grad::LinNB<T> { crate::grad::LinNB { w: r(out * inn, s) } };
    let dim = cfg.dim;
    let attn_w = |r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> crate::grad::AttnW<T> {
        crate::grad::AttnW { q: lin(dim, dim, 0.2, r), k: lin(dim, dim, 0.2, r), v: lin(dim, dim, 0.2, r), o: lin(dim, dim, 0.2, r), qn: gain(dim, r), kn: gain(dim, r) }
    };
    let blocks = (0..cfg.num_layers)
        .map(|_| BlockW {
            scale_shift_table: v(9 * dim, 0.05),
            prompt_scale_shift_table: v(2 * dim, 0.05),
            attn1: attn_w(&mut v),
            attn2: attn_w(&mut v),
            ff1: lin_nb(4 * dim, dim, 0.2, &mut v),
            ff2: lin_nb(dim, 4 * dim, 0.2, &mut v),
        })
        .collect();
    ModelWeights {
        patchify_proj: lin(dim, cfg.in_channels, 0.2, &mut v),
        keyframes_abs_pos_embedding: v(dim, 0.1),
        timestep_embedder_l1: lin(dim, 256, 0.1, &mut v),
        timestep_embedder_l2: lin(dim, dim, 0.1, &mut v),
        adaln_linear: lin(9 * dim, dim, 0.1, &mut v),
        output_scale_shift_table: v(2 * dim, 0.05),
        proj_out: lin(cfg.out_channels, dim, 0.2, &mut v),
        blocks,
    }
}

// ---- parameter enumeration (FD tests + gradcheck::check_ltxv) ----

/// Every trainable tensor, named exactly as `crate::dit::dit_tensor_manifest`
/// names it, in that manifest's own order (mutable views).
pub fn params_mut<T>(w: &mut ModelWeights<T>) -> Vec<(String, &mut Vec<T>)> {
    let mut v: Vec<(String, &mut Vec<T>)> = vec![
        ("patchify_proj.weight".into(), &mut w.patchify_proj.w),
        ("patchify_proj.bias".into(), &mut w.patchify_proj.b),
        ("adaln_single.emb.timestep_embedder.linear_1.weight".into(), &mut w.timestep_embedder_l1.w),
        ("adaln_single.emb.timestep_embedder.linear_1.bias".into(), &mut w.timestep_embedder_l1.b),
        ("adaln_single.emb.timestep_embedder.linear_2.weight".into(), &mut w.timestep_embedder_l2.w),
        ("adaln_single.emb.timestep_embedder.linear_2.bias".into(), &mut w.timestep_embedder_l2.b),
        ("adaln_single.linear.weight".into(), &mut w.adaln_linear.w),
        ("adaln_single.linear.bias".into(), &mut w.adaln_linear.b),
        ("scale_shift_table".into(), &mut w.output_scale_shift_table),
        ("proj_out.weight".into(), &mut w.proj_out.w),
        ("proj_out.bias".into(), &mut w.proj_out.b),
        ("keyframes_abs_pos_embedding".into(), &mut w.keyframes_abs_pos_embedding),
    ];
    for (i, b) in w.blocks.iter_mut().enumerate() {
        let p = format!("transformer_blocks.{i}");
        for (attn, aw) in [("attn1", &mut b.attn1), ("attn2", &mut b.attn2)] {
            v.push((format!("{p}.{attn}.to_q.weight"), &mut aw.q.w));
            v.push((format!("{p}.{attn}.to_q.bias"), &mut aw.q.b));
            v.push((format!("{p}.{attn}.to_k.weight"), &mut aw.k.w));
            v.push((format!("{p}.{attn}.to_k.bias"), &mut aw.k.b));
            v.push((format!("{p}.{attn}.to_v.weight"), &mut aw.v.w));
            v.push((format!("{p}.{attn}.to_v.bias"), &mut aw.v.b));
            v.push((format!("{p}.{attn}.to_out.0.weight"), &mut aw.o.w));
            v.push((format!("{p}.{attn}.to_out.0.bias"), &mut aw.o.b));
            v.push((format!("{p}.{attn}.q_norm.weight"), &mut aw.qn));
            v.push((format!("{p}.{attn}.k_norm.weight"), &mut aw.kn));
        }
        v.push((format!("{p}.ff.net.0.proj.weight"), &mut b.ff1.w));
        v.push((format!("{p}.ff.net.2.weight"), &mut b.ff2.w));
        v.push((format!("{p}.scale_shift_table"), &mut b.scale_shift_table));
        v.push((format!("{p}.prompt_scale_shift_table"), &mut b.prompt_scale_shift_table));
    }
    v
}

/// Gradient views in the SAME order as [`params_mut`].
pub fn grad_views<T>(g: &ModelGrads<T>) -> Vec<(String, &Vec<T>)> {
    let mut v: Vec<(String, &Vec<T>)> = vec![
        ("patchify_proj.weight".into(), &g.patchify_proj.w),
        ("patchify_proj.bias".into(), &g.patchify_proj.b),
        ("adaln_single.emb.timestep_embedder.linear_1.weight".into(), &g.timestep_embedder_l1.w),
        ("adaln_single.emb.timestep_embedder.linear_1.bias".into(), &g.timestep_embedder_l1.b),
        ("adaln_single.emb.timestep_embedder.linear_2.weight".into(), &g.timestep_embedder_l2.w),
        ("adaln_single.emb.timestep_embedder.linear_2.bias".into(), &g.timestep_embedder_l2.b),
        ("adaln_single.linear.weight".into(), &g.adaln_linear.w),
        ("adaln_single.linear.bias".into(), &g.adaln_linear.b),
        ("scale_shift_table".into(), &g.output_scale_shift_table),
        ("proj_out.weight".into(), &g.proj_out.w),
        ("proj_out.bias".into(), &g.proj_out.b),
        ("keyframes_abs_pos_embedding".into(), &g.keyframes_abs_pos_embedding),
    ];
    for (i, b) in g.blocks.iter().enumerate() {
        let p = format!("transformer_blocks.{i}");
        for (attn, aw) in [("attn1", &b.attn1), ("attn2", &b.attn2)] {
            v.push((format!("{p}.{attn}.to_q.weight"), &aw.q.w));
            v.push((format!("{p}.{attn}.to_q.bias"), &aw.q.b));
            v.push((format!("{p}.{attn}.to_k.weight"), &aw.k.w));
            v.push((format!("{p}.{attn}.to_k.bias"), &aw.k.b));
            v.push((format!("{p}.{attn}.to_v.weight"), &aw.v.w));
            v.push((format!("{p}.{attn}.to_v.bias"), &aw.v.b));
            v.push((format!("{p}.{attn}.to_out.0.weight"), &aw.o.w));
            v.push((format!("{p}.{attn}.to_out.0.bias"), &aw.o.b));
            v.push((format!("{p}.{attn}.q_norm.weight"), &aw.qn));
            v.push((format!("{p}.{attn}.k_norm.weight"), &aw.kn));
        }
        v.push((format!("{p}.ff.net.0.proj.weight"), &b.ff1.w));
        v.push((format!("{p}.ff.net.2.weight"), &b.ff2.w));
        v.push((format!("{p}.scale_shift_table"), &b.scale_shift_table));
        v.push((format!("{p}.prompt_scale_shift_table"), &b.prompt_scale_shift_table));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny config must not accidentally make two different quantities
    /// equal - that is what hides a transposition bug.
    #[test]
    fn the_tiny_config_has_no_coincidental_dimensions() {
        let c = Cfg::tiny();
        assert_ne!(c.t, c.context_len, "token count must differ from the text row count");
        assert_ne!(c.dim, c.num_heads);
        assert_ne!(c.num_heads, c.head_dim());
        assert!(c.dim > 0 && c.num_layers > 0);
    }

    /// The batch convention must be the one `crate::pipeline::to_denoised`
    /// inverts: at σ=1 the input is pure noise, at σ=0 it is the clean
    /// latent, the target is always `ε - x0`.
    #[test]
    fn flow_batch_matches_the_sampler_convention() {
        let cfg = Cfg::tiny();
        let x0: Vec<f64> = (0..cfg.t * cfg.in_channels).map(|i| i as f64 * 0.01).collect();
        let noise: Vec<f64> = (0..x0.len()).map(|_| 0.5).collect();
        let ctx = vec![0.25f64; cfg.context_len * cfg.dim];
        let b1 = make_flow_batch(&cfg, &x0, &ctx, 1.0, &noise);
        assert_eq!(b1.latent, noise);
        let b0 = make_flow_batch(&cfg, &x0, &ctx, 0.0, &noise);
        assert_eq!(b0.latent, x0);
        for (i, &v) in b0.target.iter().enumerate() {
            assert!((v - (0.5 - x0[i])).abs() < 1e-12);
        }
        assert!(b0.timesteps.iter().all(|&t| t == 0.0));
        assert!(b1.timesteps.iter().all(|&t| t == 1.0));
    }

    /// Every tensor of the manifest must appear exactly once in
    /// [`params_mut`], and [`grad_views`] must line up name-for-name and
    /// length-for-length.
    #[test]
    fn params_and_grads_cover_the_whole_manifest_in_the_same_order() {
        let cfg = Cfg::tiny();
        let mut w = init_model::<f64>(&cfg, 3);
        let names: Vec<String> = params_mut(&mut w).into_iter().map(|(n, _)| n).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate parameter name");

        let ltx_cfg = LtxDitConfig { num_layers: cfg.num_layers as u32, ..LtxDitConfig::tiny() };
        // `dit_tensor_manifest` lists every attn's `to_gate_logits.{weight,bias}`
        // unconditionally (representable at any `apply_gated_attention` value -
        // see that function's doc), but `AttnWeights<T>` here carries no gate
        // slot at all: gated-attention BACKWARD is not implemented by this
        // training path (a tracked gap, not this test's concern - `Cfg::tiny`
        // trains the M3 ungated op sequence only). So this comparison excludes
        // `to_gate_logits` names rather than asserting a coverage this module
        // does not claim.
        let mut manifest: Vec<String> = crate::dit::dit_tensor_manifest(&ltx_cfg).into_iter().map(|(n, _)| n).filter(|n| !n.contains("to_gate_logits")).collect();
        manifest.sort();
        assert_eq!(sorted, manifest, "params_mut must enumerate exactly the checkpoint manifest (minus to_gate_logits - not yet trainable, see comment above)");

        let b = make_flow_batch(&cfg, &vec![0.1; cfg.t * cfg.in_channels], &vec![0.2; cfg.context_len * cfg.dim], 0.4, &vec![0.3; cfg.t * cfg.in_channels]);
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
