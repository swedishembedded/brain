// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full audio+video LTX DiT **training** reference (host): forward + analytic
//! backward for the whole AV transformer under the flow-matching
//! velocity-MSE loss, one term per stream.
//!
//! The AV counterpart of [`crate::modelgrad`], reusing everything that
//! module already proved rather than re-deriving it: [`crate::av_grad::
//! av_block_forward`]/`av_block_backward` chain the block stack (this
//! module's own job is everything OUTSIDE the block stack - patchify,
//! keyframes, the SIX per-token/single-row `AdaLayerNormSingle` timestep
//! MLPs the AV model carries vs. the video-only model's one, and the two
//! output stages); [`crate::modelgrad::timestep_embedding`] (the input
//! sinusoid, no parameters) and [`crate::modelgrad::layernorm`]/
//! `layernorm_bwd` (the output stage's `LayerNorm`, NOT RMSNorm) run
//! UNCHANGED.
//!
//! ## The six timestep MLPs, and why they collapse to one implementation
//!
//! `adaln_single`/`audio_adaln_single` (9-row, per-token, each stream's own
//! timestep), `av_ca_{video,audio}_scale_shift_adaln_single` (4-row,
//! per-token, each stream's own timestep - `crate::block::av_scale_shift`'s
//! doc) and `av_ca_{a2v,v2a}_gate_adaln_single` (1-row, driven by the CROSS
//! modality's SCALAR sigma, not this stream's own per-token timestep) all
//! share the exact same shape: sinusoid -> linear_1 -> SiLU -> linear_2 ->
//! (SiLU -> linear at `coeff*dim` width). [`TsMlpW`]/[`ts_mlp_forward`]/
//! [`ts_mlp_bwd`] implement this ONCE, parameterized by `(rows, dim, coeff)` -
//! `dit::ada_layer_norm_single`'s own doc names the same generalisation on
//! the device side. Four of the six calls never route their own
//! `embedded_timestep` output anywhere further (`dit.rs`'s own forward
//! discards it via `let (_, _) = ada_layer_norm_single(...)`), so
//! [`ts_mlp_bwd`]'s `d_embedded_extra` is the zero vector for those four and
//! only the two MAIN tables' calls pass the output stage's own contribution -
//! the same `embedded_timestep`-feeds-two-consumers coupling
//! `crate::modelgrad`'s own doc explains for the video-only path, now once
//! per stream.
//!
//! `to_gate_logits` (gated attention) and both embeddings connectors are NOT
//! covered - the same scope line [`crate::av_grad`]'s own doc draws
//! (`crate::config::LtxAvDitConfig::tiny`, not `tiny_gated`); neither is
//! `prompt_adaln_single`/`audio_prompt_adaln_single` (real header tensors,
//! unread by the forward at `use_prompt_adaln_single=false` - `crate::dit`'s
//! doc).
//!
//! One implementation, two instantiations, same discipline as
//! [`crate::modelgrad`]: `f64` is the finite-difference gradcheck oracle,
//! `f32` is the trainer [`crate::av_finetune`] drives.

use crate::av_grad::{av_block_backward, av_block_forward, AvBlockCache, AvBlockGrads, AvBlockW, AvCrossW, AvDims, CrossAttnGrads, CrossAttnW};
use crate::config::LtxAvDitConfig;
use crate::grad::{dsilu, linear, linear_bwd, silu, AttnGrads, AttnW, Dims, Fp, Lin, LinNB};
use crate::modelgrad::{layernorm, layernorm_bwd, timestep_embedding};
use crate::rope::{ltx_rope_tables, LtxRopeTables};

// ---- the shared timestep MLP (see this module's doc) ----

/// One `AdaLayerNormSingle`'s trainable tensors - `dit::push_adaln_group`'s
/// generic-`T` twin.
#[derive(Clone, Debug, PartialEq)]
pub struct TsMlpW<T> {
    pub l1: Lin<T>,
    pub l2: Lin<T>,
    pub lin: Lin<T>,
}

/// Everything [`ts_mlp_bwd`] needs from [`ts_mlp_forward`].
pub struct TsMlpCache<T> {
    te_all: Vec<T>,
    h0pre: Vec<T>,
    h0: Vec<T>,
    embedded: Vec<T>,
    embedded_silu: Vec<T>,
}

/// Grads mirroring [`TsMlpW`].
pub struct TsMlpGrads<T> {
    pub l1: Lin<T>,
    pub l2: Lin<T>,
    pub lin: Lin<T>,
}

/// `sinusoid(ts_scaled[r], 256) -> linear_1 -> SiLU -> linear_2` (=
/// `embedded`, returned separately since some callers route it further) `->
/// SiLU -> linear` (`coeff*dim` wide) - see this module's doc. `rows =
/// ts_scaled.len()`.
pub fn ts_mlp_forward<T: Fp>(w: &TsMlpW<T>, ts_scaled: &[f64], rows: usize, dim: usize, coeff: usize) -> (Vec<T>, Vec<T>, TsMlpCache<T>) {
    assert_eq!(ts_scaled.len(), rows, "ts_mlp_forward: ts_scaled must have `rows` entries");
    let mut te_all = vec![T::ZERO; rows * 256];
    for (r, &t) in ts_scaled.iter().enumerate() {
        let e: Vec<T> = timestep_embedding(t, 256);
        te_all[r * 256..(r + 1) * 256].copy_from_slice(&e);
    }
    let h0pre = linear(&te_all, rows, 256, &w.l1.w, &w.l1.b, dim);
    let h0: Vec<T> = h0pre.iter().map(|&v| silu(v)).collect();
    let embedded = linear(&h0, rows, dim, &w.l2.w, &w.l2.b, dim);
    let embedded_silu: Vec<T> = embedded.iter().map(|&v| silu(v)).collect();
    let table = linear(&embedded_silu, rows, dim, &w.lin.w, &w.lin.b, coeff * dim);
    (table, embedded.clone(), TsMlpCache { te_all, h0pre, h0, embedded, embedded_silu })
}

/// [`ts_mlp_forward`] backward. `dtable`: gradient of the `[rows,coeff*dim]`
/// output. `d_embedded_extra`: any gradient into `embedded` from OUTSIDE
/// this MLP (the output stage's own direct use, at the two MAIN tables only
/// - zero elsewhere, see this module's doc).
pub fn ts_mlp_bwd<T: Fp>(w: &TsMlpW<T>, c: &TsMlpCache<T>, rows: usize, dim: usize, coeff: usize, dtable: &[T], d_embedded_extra: &[T]) -> TsMlpGrads<T> {
    let (d_embedded_silu, g_lin) = linear_bwd(&c.embedded_silu, rows, dim, &w.lin.w, coeff * dim, dtable);
    let mut d_embedded = d_embedded_extra.to_vec();
    for i in 0..rows * dim {
        d_embedded[i] += d_embedded_silu[i] * dsilu(c.embedded[i]);
    }
    let (dh0, g_l2) = linear_bwd(&c.h0, rows, dim, &w.l2.w, dim, &d_embedded);
    let dh0pre: Vec<T> = dh0.iter().zip(&c.h0pre).map(|(&g, &v)| g * dsilu(v)).collect();
    let (_dte, g_l1) = linear_bwd(&c.te_all, rows, 256, &w.l1.w, dim, &dh0pre);
    TsMlpGrads { l1: g_l1, l2: g_l2, lin: g_lin }
}

// ---- the output stage (one per stream) ----

/// Everything [`output_stage_bwd`] needs from [`output_stage_fwd`].
pub struct OutputCache<T> {
    xhat_out: Vec<T>,
    inv_out: Vec<T>,
    xo: Vec<T>,
    embedded: Vec<T>,
    x: Vec<T>,
}

/// `LayerNorm(no affine) -> per-token modulate by (output_scale_shift_table +
/// embedded_timestep) -> proj_out` - `dit::output_stage`'s generic-`T` twin,
/// shared by both streams (each with its own `sst`/`proj`/`embedded`).
pub fn output_stage_fwd<T: Fp>(sst: &[T], proj: &Lin<T>, x: &[T], embedded: &[T], t: usize, dim: usize, out_channels: usize, eps: f64) -> (Vec<T>, OutputCache<T>) {
    assert_eq!(sst.len(), 2 * dim, "output_stage_fwd: sst must be [2*dim]");
    let mut shift = vec![T::ZERO; t * dim];
    let mut one_plus_scale = vec![T::ZERO; t * dim];
    for ti in 0..t {
        for dd in 0..dim {
            shift[ti * dim + dd] = sst[dd] + embedded[ti * dim + dd];
            one_plus_scale[ti * dim + dd] = T::ONE + sst[dim + dd] + embedded[ti * dim + dd];
        }
    }
    let (xhat_out, inv_out) = layernorm(x, t, dim, eps);
    let mut xo = vec![T::ZERO; t * dim];
    for i in 0..t * dim {
        xo[i] = xhat_out[i] * one_plus_scale[i] + shift[i];
    }
    let pred = linear(&xo, t, dim, &proj.w, &proj.b, out_channels);
    (pred, OutputCache { xhat_out, inv_out, xo, embedded: embedded.to_vec(), x: x.to_vec() })
}

/// [`output_stage_fwd`] backward. Returns `(dx, d_embedded, d_sst, g_proj)`.
pub fn output_stage_bwd<T: Fp>(sst: &[T], proj: &Lin<T>, c: &OutputCache<T>, t: usize, dim: usize, out_channels: usize, dpred: &[T]) -> (Vec<T>, Vec<T>, Vec<T>, Lin<T>) {
    let (dxo, g_proj) = linear_bwd(&c.xo, t, dim, &proj.w, out_channels, dpred);
    let mut dshift = vec![T::ZERO; t * dim];
    let mut dscale = vec![T::ZERO; t * dim];
    let mut dxhat_out = vec![T::ZERO; t * dim];
    for i in 0..t * dim {
        let one_plus_scale_i = T::ONE + sst[dim + i % dim] + c.embedded[i];
        dxhat_out[i] = one_plus_scale_i * dxo[i];
        dscale[i] = c.xhat_out[i] * dxo[i];
        dshift[i] = dxo[i];
    }
    let mut d_sst = vec![T::ZERO; 2 * dim];
    let mut d_embedded = vec![T::ZERO; t * dim];
    for ti in 0..t {
        for dd in 0..dim {
            d_sst[dd] += dshift[ti * dim + dd];
            d_sst[dim + dd] += dscale[ti * dim + dd];
            d_embedded[ti * dim + dd] += dshift[ti * dim + dd] + dscale[ti * dim + dd];
        }
    }
    let dx = layernorm_bwd(&c.xhat_out, &c.inv_out, t, dim, &dxhat_out);
    let _ = &c.x; // shape sanity only; the residual grad IS `dx` itself.
    (dx, d_embedded, d_sst, g_proj)
}

// ---- shape ----

/// Shape of the AV training problem: the [`LtxAvDitConfig`] fields the host
/// path needs plus each stream's own token/context extent.
#[derive(Clone, Copy, Debug)]
pub struct AvCfg {
    pub vdim: usize,
    pub vheads: usize,
    pub adim: usize,
    pub aheads: usize,
    pub num_layers: usize,
    pub v_in_channels: usize,
    pub v_out_channels: usize,
    pub a_in_channels: usize,
    pub a_out_channels: usize,
    pub norm_eps: f64,
    pub theta: f64,
    pub v_max_pos: [u32; 3],
    pub a_max_pos: [u32; 1],
    pub timestep_scale_multiplier: u32,
    pub av_ca_timestep_scale_multiplier: f32,
    pub tv: usize,
    pub v_context_len: usize,
    pub ta: usize,
    pub a_context_len: usize,
}

impl AvCfg {
    /// Derive from an [`LtxAvDitConfig`] at each stream's token/context
    /// extent - the same assumption `crate::modelgrad::Cfg::from_ltx`
    /// makes (`cross_attention_dim == inner_dim`, both streams).
    pub fn from_av(c: &LtxAvDitConfig, tv: usize, v_context_len: usize, ta: usize, a_context_len: usize) -> AvCfg {
        c.assert_supported();
        assert_eq!(c.video.cross_attention_dim, c.video.inner_dim, "ltxv AV training: assumes video.cross_attention_dim == video.inner_dim");
        assert_eq!(c.audio.cross_attention_dim, c.audio.inner_dim, "ltxv AV training: assumes audio.cross_attention_dim == audio.inner_dim");
        AvCfg {
            vdim: c.video.inner_dim as usize,
            vheads: c.video.num_heads as usize,
            adim: c.audio.inner_dim as usize,
            aheads: c.audio.num_heads as usize,
            num_layers: c.video.num_layers as usize,
            v_in_channels: c.video.in_channels as usize,
            v_out_channels: c.video.out_channels as usize,
            a_in_channels: c.audio.in_channels as usize,
            a_out_channels: c.audio.out_channels as usize,
            norm_eps: c.video.norm_eps as f64,
            theta: c.video.positional_embedding_theta,
            v_max_pos: c.video.positional_embedding_max_pos,
            a_max_pos: c.audio.positional_embedding_max_pos,
            timestep_scale_multiplier: c.video.timestep_scale_multiplier,
            av_ca_timestep_scale_multiplier: c.av_ca_timestep_scale_multiplier,
            tv,
            v_context_len,
            ta,
            a_context_len,
        }
    }

    /// Tiny AV-DiT-topology config for gradchecks and unit tests. Every
    /// count differs from every other (lesson #4): 6 video tokens, 5 video
    /// text rows, 4 audio tokens, 3 audio text rows - none coincide with
    /// each other or with [`LtxAvDitConfig::tiny`]'s own `vdim=64`/
    /// `vheads=4`/`adim=32`/`aheads=4`.
    pub fn tiny() -> AvCfg {
        AvCfg::from_av(&LtxAvDitConfig::tiny(), 6, 5, 4, 3)
    }

    pub fn dims(&self) -> AvDims {
        AvDims { v: Dims { t: self.tv, te: self.v_context_len, dim: self.vdim, nh: self.vheads, eps: self.norm_eps }, a: Dims { t: self.ta, te: self.a_context_len, dim: self.adim, nh: self.aheads, eps: self.norm_eps } }
    }

    /// `LtxAvDitConfig::cross_pe_max_pos`'s twin.
    pub fn cross_pe_max_pos(&self) -> u32 {
        self.v_max_pos[0].max(self.a_max_pos[0])
    }

    /// Video's 3-axis position grid - `crate::modelgrad::Cfg::
    /// simple_positions`'s exact twin, at `self.tv` tokens.
    pub fn simple_positions_v(&self) -> Vec<f32> {
        let t = self.tv;
        let mut v = vec![0f32; 3 * t * 2];
        for ti in 0..t {
            v[ti * 2] = ti as f32;
            v[ti * 2 + 1] = ti as f32 + 1.0;
            v[(t + ti) * 2 + 1] = 1.0;
            v[(2 * t + ti) * 2 + 1] = 1.0;
        }
        v
    }

    /// Audio's single-axis (time-only) position grid, at `self.ta` tokens -
    /// this array IS audio's own cross-modal positions too (`crate::dit::
    /// LtxAvDit::forward`'s doc: "audio's own single axis IS its own
    /// positions array").
    pub fn simple_positions_a(&self) -> Vec<f32> {
        let t = self.ta;
        let mut v = vec![0f32; t * 2];
        for ti in 0..t {
            v[ti * 2] = ti as f32;
            v[ti * 2 + 1] = ti as f32 + 1.0;
        }
        v
    }

    /// This config's four RoPE tables (host f32, `crate::dit::LtxAvDit::
    /// forward`'s exact construction): each stream's own self-attention
    /// table, plus the SHARED cross-modal (time-only) table at audio's
    /// geometry, built from each stream's own axis-0/time positions.
    pub fn rope_tables_f32(&self, v_positions: &[f32], a_positions: &[f32]) -> (LtxRopeTables, LtxRopeTables, LtxRopeTables, LtxRopeTables) {
        let v_rope = ltx_rope_tables(self.vdim as u32, self.vheads as u32, self.theta, &self.v_max_pos, v_positions, self.tv);
        let a_rope = ltx_rope_tables(self.adim as u32, self.aheads as u32, self.theta, &self.a_max_pos, a_positions, self.ta);
        let cross_max_pos = [self.cross_pe_max_pos()];
        let v_axis0 = &v_positions[0..self.tv * 2];
        let v_cross_rope = ltx_rope_tables(self.adim as u32, self.aheads as u32, self.theta, &cross_max_pos, v_axis0, self.tv);
        let a_cross_rope = ltx_rope_tables(self.adim as u32, self.aheads as u32, self.theta, &cross_max_pos, a_positions, self.ta);
        (v_rope, a_rope, v_cross_rope, a_cross_rope)
    }
}

// ---- weights ----

/// Every trainable tensor of the AV DiT, in the host training layout, named
/// as `crate::dit::av_dit_tensor_manifest` names them.
#[derive(Clone, Debug, PartialEq)]
pub struct AvModelWeights<T> {
    pub v_patchify_proj: Lin<T>,
    pub v_keyframes_abs_pos_embedding: Vec<T>,
    pub a_patchify_proj: Lin<T>,
    /// `adaln_single`.
    pub v_ts: TsMlpW<T>,
    /// `audio_adaln_single`.
    pub a_ts: TsMlpW<T>,
    /// `av_ca_video_scale_shift_adaln_single`.
    pub av_video_ss_ts: TsMlpW<T>,
    /// `av_ca_audio_scale_shift_adaln_single`.
    pub av_audio_ss_ts: TsMlpW<T>,
    /// `av_ca_a2v_gate_adaln_single`.
    pub av_a2v_gate_ts: TsMlpW<T>,
    /// `av_ca_v2a_gate_adaln_single`.
    pub av_v2a_gate_ts: TsMlpW<T>,
    /// Top-level `scale_shift_table`, `[2*vdim]`.
    pub v_output_scale_shift_table: Vec<T>,
    pub v_proj_out: Lin<T>,
    /// Top-level `audio_scale_shift_table`, `[2*adim]`.
    pub a_output_scale_shift_table: Vec<T>,
    pub a_proj_out: Lin<T>,
    pub blocks: Vec<AvBlockW<T>>,
}

/// Grads mirroring [`AvModelWeights`].
pub struct AvModelGrads<T> {
    pub v_patchify_proj: Lin<T>,
    pub v_keyframes_abs_pos_embedding: Vec<T>,
    pub a_patchify_proj: Lin<T>,
    pub v_ts: TsMlpGrads<T>,
    pub a_ts: TsMlpGrads<T>,
    pub av_video_ss_ts: TsMlpGrads<T>,
    pub av_audio_ss_ts: TsMlpGrads<T>,
    pub av_a2v_gate_ts: TsMlpGrads<T>,
    pub av_v2a_gate_ts: TsMlpGrads<T>,
    pub v_output_scale_shift_table: Vec<T>,
    pub v_proj_out: Lin<T>,
    pub a_output_scale_shift_table: Vec<T>,
    pub a_proj_out: Lin<T>,
    pub blocks: Vec<AvBlockGrads<T>>,
}

/// Saved forward state for the backward pass.
pub struct AvModelCache<T> {
    v_flat: Vec<T>,
    a_flat: Vec<T>,
    v_keyframes_mask: Vec<f64>,
    v_ts: TsMlpCache<T>,
    a_ts: TsMlpCache<T>,
    av_video_ss: TsMlpCache<T>,
    av_audio_ss: TsMlpCache<T>,
    av_a2v_gate: TsMlpCache<T>,
    av_v2a_gate: TsMlpCache<T>,
    blocks: Vec<AvBlockCache<T>>,
    v_out: OutputCache<T>,
    a_out: OutputCache<T>,
}

/// Full forward. `v_latent`/`a_latent`: `[t*in_channels]`, already tokens
/// (see `crate::modelgrad`'s doc - no patchify/unpatchify inside the DiT's
/// own math). `v_timesteps`/`a_timesteps`: `[t]` each, per-token RAW
/// timestep (the model-level `timestep_scale_multiplier` scaling happens
/// inside). `v_sigma`/`a_sigma`: each stream's SCALAR sigma - the CROSS
/// modality's own scalar drives the OTHER stream's AV gate MLP (`crate::
/// block::LtxAvBlock`'s doc). `{v,a}_cos`/`{v,a}_sin`: each stream's
/// self-attention RoPE tables; `{v,a}_cross_cos`/`{v,a}_cross_sin`: the
/// shared cross-modal tables, both at audio's head geometry.
#[allow(clippy::too_many_arguments)]
pub fn forward<T: Fp>(
    cfg: &AvCfg,
    w: &AvModelWeights<T>,
    v_latent: &[T],
    v_timesteps: &[f64],
    v_keyframes_mask: &[f64],
    v_context: &[T],
    a_latent: &[T],
    a_timesteps: &[f64],
    a_context: &[T],
    v_sigma: f64,
    a_sigma: f64,
    v_cos: &[T],
    v_sin: &[T],
    a_cos: &[T],
    a_sin: &[T],
    v_cross_cos: &[T],
    v_cross_sin: &[T],
    a_cross_cos: &[T],
    a_cross_sin: &[T],
) -> (Vec<T>, Vec<T>, AvModelCache<T>) {
    let (vdim, adim, tv, ta) = (cfg.vdim, cfg.adim, cfg.tv, cfg.ta);
    assert_eq!(v_latent.len(), tv * cfg.v_in_channels, "av forward: v_latent size");
    assert_eq!(a_latent.len(), ta * cfg.a_in_channels, "av forward: a_latent size");
    assert_eq!(v_timesteps.len(), tv, "av forward: v_timesteps size");
    assert_eq!(a_timesteps.len(), ta, "av forward: a_timesteps size");

    // --- patchify_proj + video's keyframes embedding --------------------
    let v_flat = v_latent.to_vec();
    let mut vx = linear(&v_flat, tv, cfg.v_in_channels, &w.v_patchify_proj.w, &w.v_patchify_proj.b, vdim);
    for ti in 0..tv {
        if v_keyframes_mask[ti] > 0.0 {
            for d in 0..vdim {
                vx[ti * vdim + d] += w.v_keyframes_abs_pos_embedding[d];
            }
        }
    }
    let a_flat = a_latent.to_vec();
    let ax = linear(&a_flat, ta, cfg.a_in_channels, &w.a_patchify_proj.w, &w.a_patchify_proj.b, adim);

    // --- the six per-token/single-row timestep MLPs (this module's doc) --
    let v_ts_scaled: Vec<f64> = v_timesteps.iter().map(|&x| x * cfg.timestep_scale_multiplier as f64).collect();
    let a_ts_scaled: Vec<f64> = a_timesteps.iter().map(|&x| x * cfg.timestep_scale_multiplier as f64).collect();
    let (v_adaln_shared, v_embedded, v_ts_cache) = ts_mlp_forward(&w.v_ts, &v_ts_scaled, tv, vdim, 9);
    let (a_adaln_shared, a_embedded, a_ts_cache) = ts_mlp_forward(&w.a_ts, &a_ts_scaled, ta, adim, 9);
    let (av_video_ss, _, av_video_ss_cache) = ts_mlp_forward(&w.av_video_ss_ts, &v_ts_scaled, tv, vdim, 4);
    let (av_audio_ss, _, av_audio_ss_cache) = ts_mlp_forward(&w.av_audio_ss_ts, &a_ts_scaled, ta, adim, 4);
    let a2v_gate_ts = [a_sigma * cfg.av_ca_timestep_scale_multiplier as f64];
    let v2a_gate_ts = [v_sigma * cfg.av_ca_timestep_scale_multiplier as f64];
    let (av_a2v_gate, _, av_a2v_gate_cache) = ts_mlp_forward(&w.av_a2v_gate_ts, &a2v_gate_ts, 1, vdim, 1);
    let (av_v2a_gate, _, av_v2a_gate_cache) = ts_mlp_forward(&w.av_v2a_gate_ts, &v2a_gate_ts, 1, adim, 1);

    // --- block stack ------------------------------------------------------
    let d = cfg.dims();
    let mut vxx = vx;
    let mut axx = ax;
    let mut caches = Vec::with_capacity(w.blocks.len());
    for bw in &w.blocks {
        let (vo, ao, c) = av_block_forward(
            d, bw, &vxx, &axx, &v_adaln_shared, &a_adaln_shared, v_context, a_context, v_cos, v_sin, a_cos, a_sin, v_cross_cos, v_cross_sin, a_cross_cos, a_cross_sin, &av_video_ss, &av_audio_ss,
            &av_a2v_gate, &av_v2a_gate,
        );
        vxx = vo;
        axx = ao;
        caches.push(c);
    }

    // --- output stages, one per stream ------------------------------------
    let (v_pred, v_out) = output_stage_fwd(&w.v_output_scale_shift_table, &w.v_proj_out, &vxx, &v_embedded, tv, vdim, cfg.v_out_channels, cfg.norm_eps);
    let (a_pred, a_out) = output_stage_fwd(&w.a_output_scale_shift_table, &w.a_proj_out, &axx, &a_embedded, ta, adim, cfg.a_out_channels, cfg.norm_eps);

    let cache = AvModelCache {
        v_flat,
        a_flat,
        v_keyframes_mask: v_keyframes_mask.to_vec(),
        v_ts: v_ts_cache,
        a_ts: a_ts_cache,
        av_video_ss: av_video_ss_cache,
        av_audio_ss: av_audio_ss_cache,
        av_a2v_gate: av_a2v_gate_cache,
        av_v2a_gate: av_v2a_gate_cache,
        blocks: caches,
        v_out,
        a_out,
    };
    (v_pred, a_pred, cache)
}

/// Flow-matching velocity-MSE loss, combined across BOTH streams (mean over
/// the concatenated element count - the same convention `crates/ltxv/tests/
/// av_block_grad.rs`'s own `mse` helper uses).
pub fn loss<T: Fp>(v_pred: &[T], v_target: &[T], a_pred: &[T], a_target: &[T]) -> (f64, Vec<T>, Vec<T>) {
    assert_eq!(v_pred.len(), v_target.len(), "av loss: video prediction/target size");
    assert_eq!(a_pred.len(), a_target.len(), "av loss: audio prediction/target size");
    let n = T::fr((v_pred.len() + a_pred.len()) as f64);
    let two = T::fr(2.0);
    let mut l = 0.0;
    let mut dv = vec![T::ZERO; v_pred.len()];
    for i in 0..v_pred.len() {
        let err = v_pred[i] - v_target[i];
        l += (err * err / n).f64();
        dv[i] = two * err / n;
    }
    let mut da = vec![T::ZERO; a_pred.len()];
    for i in 0..a_pred.len() {
        let err = a_pred[i] - a_target[i];
        l += (err * err / n).f64();
        da[i] = two * err / n;
    }
    (l, dv, da)
}

/// Full backward from `(dv_pred, da_pred)` (grad of the loss w.r.t. each
/// stream's predicted velocity).
pub fn backward<T: Fp>(cfg: &AvCfg, w: &AvModelWeights<T>, cache: &AvModelCache<T>, dv_pred: &[T], da_pred: &[T]) -> AvModelGrads<T> {
    let (vdim, adim, tv, ta) = (cfg.vdim, cfg.adim, cfg.tv, cfg.ta);
    let d = cfg.dims();

    // --- output stages ---
    let (dx_v_from_out, dv_embedded, dv_sst, g_v_proj) = output_stage_bwd(&w.v_output_scale_shift_table, &w.v_proj_out, &cache.v_out, tv, vdim, cfg.v_out_channels, dv_pred);
    let (dx_a_from_out, da_embedded, da_sst, g_a_proj) = output_stage_bwd(&w.a_output_scale_shift_table, &w.a_proj_out, &cache.a_out, ta, adim, cfg.a_out_channels, da_pred);

    // --- block stack (reverse), accumulating every shared adjoint ---
    let mut dv_adaln_shared = vec![T::ZERO; tv * 9 * vdim];
    let mut da_adaln_shared = vec![T::ZERO; ta * 9 * adim];
    let mut dav_video_ss = vec![T::ZERO; tv * 4 * vdim];
    let mut dav_audio_ss = vec![T::ZERO; ta * 4 * adim];
    let mut dav_a2v_gate = vec![T::ZERO; vdim];
    let mut dav_v2a_gate = vec![T::ZERO; adim];
    let mut dvx = dx_v_from_out;
    let mut dax = dx_a_from_out;
    let mut blocks: Vec<AvBlockGrads<T>> = Vec::with_capacity(w.blocks.len());
    for (bw, bc) in w.blocks.iter().zip(&cache.blocks).rev() {
        let g = av_block_backward(d, bw, bc, &dvx, &dax);
        dvx = g.dvx.clone();
        dax = g.dax.clone();
        for (acc, &gi) in dv_adaln_shared.iter_mut().zip(&g.dv_adaln_shared) {
            *acc += gi;
        }
        for (acc, &gi) in da_adaln_shared.iter_mut().zip(&g.da_adaln_shared) {
            *acc += gi;
        }
        for (acc, &gi) in dav_video_ss.iter_mut().zip(&g.dav_video_ss) {
            *acc += gi;
        }
        for (acc, &gi) in dav_audio_ss.iter_mut().zip(&g.dav_audio_ss) {
            *acc += gi;
        }
        for (acc, &gi) in dav_a2v_gate.iter_mut().zip(&g.dav_a2v_gate) {
            *acc += gi;
        }
        for (acc, &gi) in dav_v2a_gate.iter_mut().zip(&g.dav_v2a_gate) {
            *acc += gi;
        }
        blocks.push(g);
    }
    blocks.reverse();

    // --- the six timestep MLPs (reverse) ---
    let g_v_ts = ts_mlp_bwd(&w.v_ts, &cache.v_ts, tv, vdim, 9, &dv_adaln_shared, &dv_embedded);
    let g_a_ts = ts_mlp_bwd(&w.a_ts, &cache.a_ts, ta, adim, 9, &da_adaln_shared, &da_embedded);
    let g_av_video_ss_ts = ts_mlp_bwd(&w.av_video_ss_ts, &cache.av_video_ss, tv, vdim, 4, &dav_video_ss, &vec![T::ZERO; tv * vdim]);
    let g_av_audio_ss_ts = ts_mlp_bwd(&w.av_audio_ss_ts, &cache.av_audio_ss, ta, adim, 4, &dav_audio_ss, &vec![T::ZERO; ta * adim]);
    let g_av_a2v_gate_ts = ts_mlp_bwd(&w.av_a2v_gate_ts, &cache.av_a2v_gate, 1, vdim, 1, &dav_a2v_gate, &vec![T::ZERO; vdim]);
    let g_av_v2a_gate_ts = ts_mlp_bwd(&w.av_v2a_gate_ts, &cache.av_v2a_gate, 1, adim, 1, &dav_v2a_gate, &vec![T::ZERO; adim]);
    // The two gate MLPs' own scalar inputs (the CROSS modality's sigma) are
    // external batch data, not trainable - no gradient routed further for
    // them, the same convention `crate::modelgrad`'s own `timesteps` input
    // already sets.

    // --- keyframes + patchify_proj (their inputs are data: weight grads
    // only) ---
    let mut g_v_keyframes = vec![T::ZERO; vdim];
    for ti in 0..tv {
        if cache.v_keyframes_mask[ti] > 0.0 {
            for dd in 0..vdim {
                g_v_keyframes[dd] += dvx[ti * vdim + dd];
            }
        }
    }
    let (_d_v_flat, g_v_patchify) = linear_bwd(&cache.v_flat, tv, cfg.v_in_channels, &w.v_patchify_proj.w, vdim, &dvx);
    let (_d_a_flat, g_a_patchify) = linear_bwd(&cache.a_flat, ta, cfg.a_in_channels, &w.a_patchify_proj.w, adim, &dax);

    AvModelGrads {
        v_patchify_proj: g_v_patchify,
        v_keyframes_abs_pos_embedding: g_v_keyframes,
        a_patchify_proj: g_a_patchify,
        v_ts: g_v_ts,
        a_ts: g_a_ts,
        av_video_ss_ts: g_av_video_ss_ts,
        av_audio_ss_ts: g_av_audio_ss_ts,
        av_a2v_gate_ts: g_av_a2v_gate_ts,
        av_v2a_gate_ts: g_av_v2a_gate_ts,
        v_output_scale_shift_table: dv_sst,
        v_proj_out: g_v_proj,
        a_output_scale_shift_table: da_sst,
        a_proj_out: g_a_proj,
        blocks,
    }
}

// ---- flow-matching batch ----

/// One AV training example, ready for [`forward`].
#[derive(Clone)]
pub struct AvBatch<T> {
    pub v_latent: Vec<T>,
    pub a_latent: Vec<T>,
    pub v_ctx: Vec<T>,
    pub a_ctx: Vec<T>,
    pub v_timesteps: Vec<f64>,
    pub a_timesteps: Vec<f64>,
    pub v_keyframes_mask: Vec<f64>,
    pub v_sigma: f64,
    pub a_sigma: f64,
    pub v_cos: Vec<T>,
    pub v_sin: Vec<T>,
    pub a_cos: Vec<T>,
    pub a_sin: Vec<T>,
    pub v_cross_cos: Vec<T>,
    pub v_cross_sin: Vec<T>,
    pub a_cross_cos: Vec<T>,
    pub a_cross_sin: Vec<T>,
    pub v_target: Vec<T>,
    pub a_target: Vec<T>,
}

/// Build one AV flow-matching batch from clean latents `v_x0`/`a_x0`, each
/// stream's own text features, each stream's OWN noise level (diffusion
/// forcing allows the two streams to sit at different sigmas - `crate::dit::
/// AvDitBatch`'s doc) and standard-normal noise. Same `x_σ =
/// (1-σ)·x0 + σ·ε`, `v = ε - x0` convention `crate::modelgrad::
/// make_flow_batch`'s own doc explains, applied per stream.
#[allow(clippy::too_many_arguments)]
pub fn make_av_flow_batch<T: Fp>(cfg: &AvCfg, v_x0: &[T], a_x0: &[T], v_ctx: &[T], a_ctx: &[T], v_sigma: f64, a_sigma: f64, v_noise: &[T], a_noise: &[T]) -> AvBatch<T> {
    assert_eq!(v_x0.len(), cfg.tv * cfg.v_in_channels, "av flow batch: v_x0 size");
    assert_eq!(a_x0.len(), cfg.ta * cfg.a_in_channels, "av flow batch: a_x0 size");
    assert_eq!(v_noise.len(), v_x0.len(), "av flow batch: v_noise size");
    assert_eq!(a_noise.len(), a_x0.len(), "av flow batch: a_noise size");

    let mix = |x0: &[T], noise: &[T], sigma: f64| -> Vec<T> {
        let s = T::fr(sigma);
        x0.iter().zip(noise).map(|(&x, &e)| (T::ONE - s) * x + s * e).collect()
    };
    let target = |x0: &[T], noise: &[T]| -> Vec<T> { x0.iter().zip(noise).map(|(&x, &e)| e - x).collect() };

    let v_positions = cfg.simple_positions_v();
    let a_positions = cfg.simple_positions_a();
    let (v_rope, a_rope, v_cross_rope, a_cross_rope) = cfg.rope_tables_f32(&v_positions, &a_positions);
    let cast = |v: &[f32]| -> Vec<T> { v.iter().map(|&x| T::fr(x as f64)).collect() };

    AvBatch {
        v_latent: mix(v_x0, v_noise, v_sigma),
        a_latent: mix(a_x0, a_noise, a_sigma),
        v_ctx: v_ctx.to_vec(),
        a_ctx: a_ctx.to_vec(),
        v_timesteps: vec![v_sigma; cfg.tv],
        a_timesteps: vec![a_sigma; cfg.ta],
        v_keyframes_mask: vec![1.0; cfg.tv],
        v_sigma,
        a_sigma,
        v_cos: cast(&v_rope.cos),
        v_sin: cast(&v_rope.sin),
        a_cos: cast(&a_rope.cos),
        a_sin: cast(&a_rope.sin),
        v_cross_cos: cast(&v_cross_rope.cos),
        v_cross_sin: cast(&v_cross_rope.sin),
        a_cross_cos: cast(&a_cross_rope.cos),
        a_cross_sin: cast(&a_cross_rope.sin),
        v_target: target(v_x0, v_noise),
        a_target: target(a_x0, a_noise),
    }
}

/// One training evaluation: forward + loss + backward. The f32
/// instantiation is the finetune trainer's step core.
pub fn grads<T: Fp>(cfg: &AvCfg, w: &AvModelWeights<T>, b: &AvBatch<T>) -> (f64, AvModelGrads<T>) {
    let (v_pred, a_pred, cache) = forward(
        cfg, w, &b.v_latent, &b.v_timesteps, &b.v_keyframes_mask, &b.v_ctx, &b.a_latent, &b.a_timesteps, &b.a_ctx, b.v_sigma, b.a_sigma, &b.v_cos, &b.v_sin, &b.a_cos, &b.a_sin, &b.v_cross_cos,
        &b.v_cross_sin, &b.a_cross_cos, &b.a_cross_sin,
    );
    let (l, dv_pred, da_pred) = loss(&v_pred, &b.v_target, &a_pred, &b.a_target);
    (l, backward(cfg, w, &cache, &dv_pred, &da_pred))
}

// ---- weight construction ----

impl AvModelWeights<f32> {
    /// Build host training weights from imported tensors
    /// (`crate::dit::av_dit_tensor_manifest`'s name space) -
    /// `crate::modelgrad::ModelWeights::from_tensors`'s AV twin.
    pub fn from_tensors(cfg: &AvCfg, ts: &vae::blocks::Tensors) -> Result<AvModelWeights<f32>, String> {
        let get = |name: &str| -> Result<Vec<f32>, String> { ts.get(name).map(|(_, v)| v.clone()).ok_or_else(|| format!("av from_tensors: missing {name}")) };
        let lin = |p: &str| -> Result<Lin<f32>, String> { Ok(Lin { w: get(&format!("{p}.weight"))?, b: get(&format!("{p}.bias"))? }) };
        let lin_nb = |p: &str| -> Result<LinNB<f32>, String> { Ok(LinNB { w: get(&format!("{p}.weight"))? }) };
        let attn = |p: &str| -> Result<AttnW<f32>, String> {
            Ok(AttnW { q: lin(&format!("{p}.to_q"))?, k: lin(&format!("{p}.to_k"))?, v: lin(&format!("{p}.to_v"))?, o: lin(&format!("{p}.to_out.0"))?, qn: get(&format!("{p}.q_norm.weight"))?, kn: get(&format!("{p}.k_norm.weight"))? })
        };
        let cross_attn = |p: &str| -> Result<CrossAttnW<f32>, String> {
            Ok(CrossAttnW { q: lin(&format!("{p}.to_q"))?, k: lin(&format!("{p}.to_k"))?, v: lin(&format!("{p}.to_v"))?, o: lin(&format!("{p}.to_out.0"))?, qn: get(&format!("{p}.q_norm.weight"))?, kn: get(&format!("{p}.k_norm.weight"))? })
        };
        let ts_mlp = |p: &str| -> Result<TsMlpW<f32>, String> { Ok(TsMlpW { l1: lin(&format!("{p}.emb.timestep_embedder.linear_1"))?, l2: lin(&format!("{p}.emb.timestep_embedder.linear_2"))?, lin: lin(&format!("{p}.linear"))? }) };

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{l}");
            blocks.push(AvBlockW {
                v_scale_shift_table: get(&format!("{p}.scale_shift_table"))?,
                v_prompt_scale_shift_table: get(&format!("{p}.prompt_scale_shift_table"))?,
                v_attn1: attn(&format!("{p}.attn1"))?,
                v_attn2: attn(&format!("{p}.attn2"))?,
                v_ff1: lin_nb(&format!("{p}.ff.net.0.proj"))?,
                v_ff2: lin_nb(&format!("{p}.ff.net.2"))?,
                a_scale_shift_table: get(&format!("{p}.audio_scale_shift_table"))?,
                a_prompt_scale_shift_table: get(&format!("{p}.audio_prompt_scale_shift_table"))?,
                a_attn1: attn(&format!("{p}.audio_attn1"))?,
                a_attn2: attn(&format!("{p}.audio_attn2"))?,
                a_ff1: lin(&format!("{p}.audio_ff.net.0.proj"))?,
                a_ff2: lin(&format!("{p}.audio_ff.net.2"))?,
                av: AvCrossW {
                    a2v: cross_attn(&format!("{p}.audio_to_video_attn"))?,
                    v2a: cross_attn(&format!("{p}.video_to_audio_attn"))?,
                    table_video: get(&format!("{p}.scale_shift_table_a2v_ca_video"))?,
                    table_audio: get(&format!("{p}.scale_shift_table_a2v_ca_audio"))?,
                },
            });
        }
        Ok(AvModelWeights {
            v_patchify_proj: lin("patchify_proj")?,
            v_keyframes_abs_pos_embedding: get("keyframes_abs_pos_embedding")?,
            a_patchify_proj: lin("audio_patchify_proj")?,
            v_ts: ts_mlp("adaln_single")?,
            a_ts: ts_mlp("audio_adaln_single")?,
            av_video_ss_ts: ts_mlp("av_ca_video_scale_shift_adaln_single")?,
            av_audio_ss_ts: ts_mlp("av_ca_audio_scale_shift_adaln_single")?,
            av_a2v_gate_ts: ts_mlp("av_ca_a2v_gate_adaln_single")?,
            av_v2a_gate_ts: ts_mlp("av_ca_v2a_gate_adaln_single")?,
            v_output_scale_shift_table: get("scale_shift_table")?,
            v_proj_out: lin("proj_out")?,
            a_output_scale_shift_table: get("audio_scale_shift_table")?,
            a_proj_out: lin("audio_proj_out")?,
            blocks,
        })
    }
}

/// Deterministic random init at any scalar type - for gradchecks and
/// synthetic training (real runs would import a checkpoint, not yet built
/// for the AV DiT - see this crate's roadmap ledger).
pub fn init_model<T: Fp>(cfg: &AvCfg, seed: u64) -> AvModelWeights<T> {
    let mut rng = data::rng::Rng::new(seed);
    let mut v = |n: usize, s: f64| -> Vec<T> { (0..n).map(|_| T::fr((rng.next_f64() - 0.5) * 2.0 * s)).collect() };
    let gain = |n: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Vec<T> { r(n, 0.1).iter().map(|&x| T::ONE + x).collect() };
    let lin = |out: usize, inn: usize, s: f64, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Lin<T> { Lin { w: r(out * inn, s), b: r(out, 0.05) } };
    let lin_nb = |out: usize, inn: usize, s: f64, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> LinNB<T> { LinNB { w: r(out * inn, s) } };
    let attn_w = |dim: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> AttnW<T> { AttnW { q: lin(dim, dim, 0.2, r), k: lin(dim, dim, 0.2, r), v: lin(dim, dim, 0.2, r), o: lin(dim, dim, 0.2, r), qn: gain(dim, r), kn: gain(dim, r) } };
    let cross_attn_w = |q_dim: usize, kv_dim: usize, inner: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> CrossAttnW<T> {
        CrossAttnW { q: lin(inner, q_dim, 0.2, r), k: lin(inner, kv_dim, 0.2, r), v: lin(inner, kv_dim, 0.2, r), o: lin(q_dim, inner, 0.2, r), qn: gain(inner, r), kn: gain(inner, r) }
    };
    let ts_mlp = |dim: usize, coeff: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> TsMlpW<T> { TsMlpW { l1: lin(dim, 256, 0.1, r), l2: lin(dim, dim, 0.1, r), lin: lin(coeff * dim, dim, 0.1, r) } };

    let (vdim, adim) = (cfg.vdim, cfg.adim);
    let blocks = (0..cfg.num_layers)
        .map(|_| AvBlockW {
            v_scale_shift_table: v(9 * vdim, 0.05),
            v_prompt_scale_shift_table: v(2 * vdim, 0.05),
            v_attn1: attn_w(vdim, &mut v),
            v_attn2: attn_w(vdim, &mut v),
            v_ff1: lin_nb(4 * vdim, vdim, 0.2, &mut v),
            v_ff2: lin_nb(vdim, 4 * vdim, 0.2, &mut v),
            a_scale_shift_table: v(9 * adim, 0.05),
            a_prompt_scale_shift_table: v(2 * adim, 0.05),
            a_attn1: attn_w(adim, &mut v),
            a_attn2: attn_w(adim, &mut v),
            a_ff1: lin(4 * adim, adim, 0.2, &mut v),
            a_ff2: lin(adim, 4 * adim, 0.2, &mut v),
            av: AvCrossW { a2v: cross_attn_w(vdim, adim, adim, &mut v), v2a: cross_attn_w(adim, vdim, adim, &mut v), table_video: v(5 * vdim, 0.05), table_audio: v(5 * adim, 0.05) },
        })
        .collect();

    AvModelWeights {
        v_patchify_proj: lin(vdim, cfg.v_in_channels, 0.2, &mut v),
        v_keyframes_abs_pos_embedding: v(vdim, 0.1),
        a_patchify_proj: lin(adim, cfg.a_in_channels, 0.2, &mut v),
        v_ts: ts_mlp(vdim, 9, &mut v),
        a_ts: ts_mlp(adim, 9, &mut v),
        av_video_ss_ts: ts_mlp(vdim, 4, &mut v),
        av_audio_ss_ts: ts_mlp(adim, 4, &mut v),
        av_a2v_gate_ts: ts_mlp(vdim, 1, &mut v),
        av_v2a_gate_ts: ts_mlp(adim, 1, &mut v),
        v_output_scale_shift_table: v(2 * vdim, 0.05),
        v_proj_out: lin(cfg.v_out_channels, vdim, 0.2, &mut v),
        a_output_scale_shift_table: v(2 * adim, 0.05),
        a_proj_out: lin(cfg.a_out_channels, adim, 0.2, &mut v),
        blocks,
    }
}

// ---- parameter enumeration (FD tests + LoRA) ----

fn push_attn_params<'a, T>(v: &mut Vec<(String, &'a mut Vec<T>)>, p: &str, aw: &'a mut AttnW<T>) {
    v.push((format!("{p}.q_norm.weight"), &mut aw.qn));
    v.push((format!("{p}.k_norm.weight"), &mut aw.kn));
    v.push((format!("{p}.to_q.weight"), &mut aw.q.w));
    v.push((format!("{p}.to_q.bias"), &mut aw.q.b));
    v.push((format!("{p}.to_k.weight"), &mut aw.k.w));
    v.push((format!("{p}.to_k.bias"), &mut aw.k.b));
    v.push((format!("{p}.to_v.weight"), &mut aw.v.w));
    v.push((format!("{p}.to_v.bias"), &mut aw.v.b));
    v.push((format!("{p}.to_out.0.weight"), &mut aw.o.w));
    v.push((format!("{p}.to_out.0.bias"), &mut aw.o.b));
}

fn push_cross_attn_params<'a, T>(v: &mut Vec<(String, &'a mut Vec<T>)>, p: &str, aw: &'a mut CrossAttnW<T>) {
    v.push((format!("{p}.q_norm.weight"), &mut aw.qn));
    v.push((format!("{p}.k_norm.weight"), &mut aw.kn));
    v.push((format!("{p}.to_q.weight"), &mut aw.q.w));
    v.push((format!("{p}.to_q.bias"), &mut aw.q.b));
    v.push((format!("{p}.to_k.weight"), &mut aw.k.w));
    v.push((format!("{p}.to_k.bias"), &mut aw.k.b));
    v.push((format!("{p}.to_v.weight"), &mut aw.v.w));
    v.push((format!("{p}.to_v.bias"), &mut aw.v.b));
    v.push((format!("{p}.to_out.0.weight"), &mut aw.o.w));
    v.push((format!("{p}.to_out.0.bias"), &mut aw.o.b));
}

fn push_adaln_params<'a, T>(v: &mut Vec<(String, &'a mut Vec<T>)>, p: &str, m: &'a mut TsMlpW<T>) {
    v.push((format!("{p}.emb.timestep_embedder.linear_1.weight"), &mut m.l1.w));
    v.push((format!("{p}.emb.timestep_embedder.linear_1.bias"), &mut m.l1.b));
    v.push((format!("{p}.emb.timestep_embedder.linear_2.weight"), &mut m.l2.w));
    v.push((format!("{p}.emb.timestep_embedder.linear_2.bias"), &mut m.l2.b));
    v.push((format!("{p}.linear.weight"), &mut m.lin.w));
    v.push((format!("{p}.linear.bias"), &mut m.lin.b));
}

/// Every trainable tensor, named exactly as `crate::dit::
/// av_dit_tensor_manifest` names it (minus `to_gate_logits`, both
/// embeddings connectors, and `prompt_adaln_single`/
/// `audio_prompt_adaln_single` - see this module's doc), in that manifest's
/// own relative order (mutable views).
pub fn params_mut<T>(w: &mut AvModelWeights<T>) -> Vec<(String, &mut Vec<T>)> {
    let mut v: Vec<(String, &mut Vec<T>)> = vec![
        ("patchify_proj.weight".into(), &mut w.v_patchify_proj.w),
        ("patchify_proj.bias".into(), &mut w.v_patchify_proj.b),
        ("audio_patchify_proj.weight".into(), &mut w.a_patchify_proj.w),
        ("audio_patchify_proj.bias".into(), &mut w.a_patchify_proj.b),
        ("proj_out.weight".into(), &mut w.v_proj_out.w),
        ("proj_out.bias".into(), &mut w.v_proj_out.b),
        ("audio_proj_out.weight".into(), &mut w.a_proj_out.w),
        ("audio_proj_out.bias".into(), &mut w.a_proj_out.b),
        ("scale_shift_table".into(), &mut w.v_output_scale_shift_table),
        ("audio_scale_shift_table".into(), &mut w.a_output_scale_shift_table),
        ("keyframes_abs_pos_embedding".into(), &mut w.v_keyframes_abs_pos_embedding),
    ];
    push_adaln_params(&mut v, "adaln_single", &mut w.v_ts);
    push_adaln_params(&mut v, "audio_adaln_single", &mut w.a_ts);
    push_adaln_params(&mut v, "av_ca_video_scale_shift_adaln_single", &mut w.av_video_ss_ts);
    push_adaln_params(&mut v, "av_ca_audio_scale_shift_adaln_single", &mut w.av_audio_ss_ts);
    push_adaln_params(&mut v, "av_ca_a2v_gate_adaln_single", &mut w.av_a2v_gate_ts);
    push_adaln_params(&mut v, "av_ca_v2a_gate_adaln_single", &mut w.av_v2a_gate_ts);

    for (i, b) in w.blocks.iter_mut().enumerate() {
        let p = format!("transformer_blocks.{i}");
        push_attn_params(&mut v, &format!("{p}.attn1"), &mut b.v_attn1);
        push_attn_params(&mut v, &format!("{p}.attn2"), &mut b.v_attn2);
        push_attn_params(&mut v, &format!("{p}.audio_attn1"), &mut b.a_attn1);
        push_attn_params(&mut v, &format!("{p}.audio_attn2"), &mut b.a_attn2);
        push_cross_attn_params(&mut v, &format!("{p}.audio_to_video_attn"), &mut b.av.a2v);
        push_cross_attn_params(&mut v, &format!("{p}.video_to_audio_attn"), &mut b.av.v2a);
        v.push((format!("{p}.ff.net.0.proj.weight"), &mut b.v_ff1.w));
        v.push((format!("{p}.ff.net.2.weight"), &mut b.v_ff2.w));
        v.push((format!("{p}.audio_ff.net.0.proj.weight"), &mut b.a_ff1.w));
        v.push((format!("{p}.audio_ff.net.0.proj.bias"), &mut b.a_ff1.b));
        v.push((format!("{p}.audio_ff.net.2.weight"), &mut b.a_ff2.w));
        v.push((format!("{p}.audio_ff.net.2.bias"), &mut b.a_ff2.b));
        v.push((format!("{p}.scale_shift_table"), &mut b.v_scale_shift_table));
        v.push((format!("{p}.prompt_scale_shift_table"), &mut b.v_prompt_scale_shift_table));
        v.push((format!("{p}.audio_scale_shift_table"), &mut b.a_scale_shift_table));
        v.push((format!("{p}.audio_prompt_scale_shift_table"), &mut b.a_prompt_scale_shift_table));
        v.push((format!("{p}.scale_shift_table_a2v_ca_video"), &mut b.av.table_video));
        v.push((format!("{p}.scale_shift_table_a2v_ca_audio"), &mut b.av.table_audio));
    }
    v
}

fn push_attn_grads<'a, T>(v: &mut Vec<(String, &'a Vec<T>)>, p: &str, g: &'a AttnGrads<T>) {
    v.push((format!("{p}.q_norm.weight"), &g.qn));
    v.push((format!("{p}.k_norm.weight"), &g.kn));
    v.push((format!("{p}.to_q.weight"), &g.q.w));
    v.push((format!("{p}.to_q.bias"), &g.q.b));
    v.push((format!("{p}.to_k.weight"), &g.k.w));
    v.push((format!("{p}.to_k.bias"), &g.k.b));
    v.push((format!("{p}.to_v.weight"), &g.v.w));
    v.push((format!("{p}.to_v.bias"), &g.v.b));
    v.push((format!("{p}.to_out.0.weight"), &g.o.w));
    v.push((format!("{p}.to_out.0.bias"), &g.o.b));
}

fn push_cross_attn_grads<'a, T>(v: &mut Vec<(String, &'a Vec<T>)>, p: &str, g: &'a CrossAttnGrads<T>) {
    v.push((format!("{p}.q_norm.weight"), &g.qn));
    v.push((format!("{p}.k_norm.weight"), &g.kn));
    v.push((format!("{p}.to_q.weight"), &g.q.w));
    v.push((format!("{p}.to_q.bias"), &g.q.b));
    v.push((format!("{p}.to_k.weight"), &g.k.w));
    v.push((format!("{p}.to_k.bias"), &g.k.b));
    v.push((format!("{p}.to_v.weight"), &g.v.w));
    v.push((format!("{p}.to_v.bias"), &g.v.b));
    v.push((format!("{p}.to_out.0.weight"), &g.o.w));
    v.push((format!("{p}.to_out.0.bias"), &g.o.b));
}

fn push_adaln_grads<'a, T>(v: &mut Vec<(String, &'a Vec<T>)>, p: &str, g: &'a TsMlpGrads<T>) {
    v.push((format!("{p}.emb.timestep_embedder.linear_1.weight"), &g.l1.w));
    v.push((format!("{p}.emb.timestep_embedder.linear_1.bias"), &g.l1.b));
    v.push((format!("{p}.emb.timestep_embedder.linear_2.weight"), &g.l2.w));
    v.push((format!("{p}.emb.timestep_embedder.linear_2.bias"), &g.l2.b));
    v.push((format!("{p}.linear.weight"), &g.lin.w));
    v.push((format!("{p}.linear.bias"), &g.lin.b));
}

/// Gradient views in the SAME order as [`params_mut`].
pub fn grad_views<T>(g: &AvModelGrads<T>) -> Vec<(String, &Vec<T>)> {
    let mut v: Vec<(String, &Vec<T>)> = vec![
        ("patchify_proj.weight".into(), &g.v_patchify_proj.w),
        ("patchify_proj.bias".into(), &g.v_patchify_proj.b),
        ("audio_patchify_proj.weight".into(), &g.a_patchify_proj.w),
        ("audio_patchify_proj.bias".into(), &g.a_patchify_proj.b),
        ("proj_out.weight".into(), &g.v_proj_out.w),
        ("proj_out.bias".into(), &g.v_proj_out.b),
        ("audio_proj_out.weight".into(), &g.a_proj_out.w),
        ("audio_proj_out.bias".into(), &g.a_proj_out.b),
        ("scale_shift_table".into(), &g.v_output_scale_shift_table),
        ("audio_scale_shift_table".into(), &g.a_output_scale_shift_table),
        ("keyframes_abs_pos_embedding".into(), &g.v_keyframes_abs_pos_embedding),
    ];
    push_adaln_grads(&mut v, "adaln_single", &g.v_ts);
    push_adaln_grads(&mut v, "audio_adaln_single", &g.a_ts);
    push_adaln_grads(&mut v, "av_ca_video_scale_shift_adaln_single", &g.av_video_ss_ts);
    push_adaln_grads(&mut v, "av_ca_audio_scale_shift_adaln_single", &g.av_audio_ss_ts);
    push_adaln_grads(&mut v, "av_ca_a2v_gate_adaln_single", &g.av_a2v_gate_ts);
    push_adaln_grads(&mut v, "av_ca_v2a_gate_adaln_single", &g.av_v2a_gate_ts);

    for (i, b) in g.blocks.iter().enumerate() {
        let p = format!("transformer_blocks.{i}");
        push_attn_grads(&mut v, &format!("{p}.attn1"), &b.v_attn1);
        push_attn_grads(&mut v, &format!("{p}.attn2"), &b.v_attn2);
        push_attn_grads(&mut v, &format!("{p}.audio_attn1"), &b.a_attn1);
        push_attn_grads(&mut v, &format!("{p}.audio_attn2"), &b.a_attn2);
        push_cross_attn_grads(&mut v, &format!("{p}.audio_to_video_attn"), &b.av.a2v);
        push_cross_attn_grads(&mut v, &format!("{p}.video_to_audio_attn"), &b.av.v2a);
        v.push((format!("{p}.ff.net.0.proj.weight"), &b.v_ff1.w));
        v.push((format!("{p}.ff.net.2.weight"), &b.v_ff2.w));
        v.push((format!("{p}.audio_ff.net.0.proj.weight"), &b.a_ff1.w));
        v.push((format!("{p}.audio_ff.net.0.proj.bias"), &b.a_ff1.b));
        v.push((format!("{p}.audio_ff.net.2.weight"), &b.a_ff2.w));
        v.push((format!("{p}.audio_ff.net.2.bias"), &b.a_ff2.b));
        v.push((format!("{p}.scale_shift_table"), &b.v_scale_shift_table));
        v.push((format!("{p}.prompt_scale_shift_table"), &b.v_prompt_scale_shift_table));
        v.push((format!("{p}.audio_scale_shift_table"), &b.a_scale_shift_table));
        v.push((format!("{p}.audio_prompt_scale_shift_table"), &b.a_prompt_scale_shift_table));
        v.push((format!("{p}.scale_shift_table_a2v_ca_video"), &b.av.table_video));
        v.push((format!("{p}.scale_shift_table_a2v_ca_audio"), &b.av.table_audio));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny config must not accidentally make two different quantities
    /// equal - lesson #4.
    #[test]
    fn the_tiny_config_has_no_coincidental_dimensions() {
        let c = AvCfg::tiny();
        assert_ne!(c.tv, c.ta);
        assert_ne!(c.v_context_len, c.a_context_len);
        assert_ne!(c.vdim, c.adim);
        assert_ne!(c.tv, c.v_context_len);
        assert_ne!(c.ta, c.a_context_len);
        assert!(c.vdim > 0 && c.adim > 0 && c.num_layers > 0);
    }

    /// The batch convention must be the one `crate::pipeline::to_denoised`
    /// inverts (the same assertion `crate::modelgrad`'s own test makes),
    /// checked independently per stream since the two sigmas can differ.
    #[test]
    fn flow_batch_matches_the_sampler_convention_per_stream() {
        let cfg = AvCfg::tiny();
        let v_x0: Vec<f64> = (0..cfg.tv * cfg.v_in_channels).map(|i| i as f64 * 0.01).collect();
        let a_x0: Vec<f64> = (0..cfg.ta * cfg.a_in_channels).map(|i| i as f64 * 0.02).collect();
        let v_noise: Vec<f64> = (0..v_x0.len()).map(|_| 0.5).collect();
        let a_noise: Vec<f64> = (0..a_x0.len()).map(|_| 0.4).collect();
        let v_ctx = vec![0.25f64; cfg.v_context_len * cfg.vdim];
        let a_ctx = vec![0.15f64; cfg.a_context_len * cfg.adim];

        let b1 = make_av_flow_batch(&cfg, &v_x0, &a_x0, &v_ctx, &a_ctx, 1.0, 0.0, &v_noise, &a_noise);
        assert_eq!(b1.v_latent, v_noise, "video at sigma=1 must be pure noise");
        assert_eq!(b1.a_latent, a_x0, "audio at sigma=0 must be the clean latent");
        assert!(b1.v_timesteps.iter().all(|&t| t == 1.0));
        assert!(b1.a_timesteps.iter().all(|&t| t == 0.0));
    }

    /// Every tensor of the manifest must appear exactly once in
    /// [`params_mut`], and [`grad_views`] must line up name-for-name and
    /// length-for-length - `crate::modelgrad`'s own coverage test, extended
    /// to the AV manifest.
    #[test]
    fn params_and_grads_cover_the_whole_manifest_minus_the_untrained_gap(){
        let cfg = AvCfg::tiny();
        let mut w = init_model::<f64>(&cfg, 3);
        let names: Vec<String> = params_mut(&mut w).into_iter().map(|(n, _)| n).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate parameter name");

        let av_cfg = LtxAvDitConfig { video: crate::config::LtxDitConfig { num_layers: cfg.num_layers as u32, ..LtxAvDitConfig::tiny().video }, ..LtxAvDitConfig::tiny() };
        let mut manifest: Vec<String> = crate::dit::av_dit_tensor_manifest(&av_cfg)
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| {
                !n.contains("to_gate_logits")
                    && !n.starts_with("video_embeddings_connector")
                    && !n.starts_with("audio_embeddings_connector")
                    && !n.starts_with("prompt_adaln_single")
                    && !n.starts_with("audio_prompt_adaln_single")
            })
            .collect();
        manifest.sort();
        assert_eq!(sorted, manifest, "params_mut must enumerate exactly the checkpoint manifest minus the untrained gap (to_gate_logits, both connectors, the two unused prompt_adaln_single groups)");

        let b = make_av_flow_batch(
            &cfg,
            &vec![0.1; cfg.tv * cfg.v_in_channels],
            &vec![0.1; cfg.ta * cfg.a_in_channels],
            &vec![0.2; cfg.v_context_len * cfg.vdim],
            &vec![0.2; cfg.a_context_len * cfg.adim],
            0.4,
            0.6,
            &vec![0.3; cfg.tv * cfg.v_in_channels],
            &vec![0.3; cfg.ta * cfg.a_in_channels],
        );
        let (_l, g) = grads(&cfg, &w, &b);
        let gv = grad_views(&g);
        let pm: Vec<(String, usize)> = params_mut(&mut w).into_iter().map(|(n, val)| (n, val.len())).collect();
        assert_eq!(gv.len(), pm.len());
        for ((gn, gvv), (pn, pl)) in gv.iter().zip(&pm) {
            assert_eq!(gn, pn, "grad_views order");
            assert_eq!(gvv.len(), *pl, "{gn}: grad length");
        }
    }
}
