// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5's gated, doubled-`q_proj` GQA mixer internals: the per-head
//! [value|gate] de-interleave, QK-norm, partial M-RoPE, attention, and the
//! sigmoid output gate - everything between a layer's `q_proj`/`k_proj`/
//! `v_proj` and its `o_proj`. Both `qwen35` and `qwen35moe` run
//! byte-identical code here (verified by `crates/model/tests/
//! gdn_mixer_equivalence.rs`); only the projections around this differ per
//! model - see [`crate::gdn_mixer`]'s own doc for why this boundary matches
//! [`crate::block`]'s "linear projections stay in the model" principle.
//!
//! Distinct from [`crate::block::gqa_attn_sublayer_fwd`]: that helper's own
//! `GqaAttnWeights` embeds a PLAIN (non-doubled, non-gated) `q_proj`
//! directly, a different, simpler GQA shape (no output gate, no LoRA/int8
//! seam), not reusable for Qwen3.5's mixer.
//!
//! [`gqa_mixer_fwd`] takes the layer's ALREADY-projected `q_full` (the
//! doubled `[value|gate]` `q_proj` output), `k`, `v` and returns `ctx_gated`,
//! ready for the caller's own `o_proj`. [`gqa_mixer_bwd`] is the exact
//! mirror: it takes `d_ctx_gated` (the caller's own `o_proj` backward output)
//! and returns `(d_q_full, d_k, d_v)` for the caller's own `q_proj`/`k_proj`/
//! `v_proj` backward.

use gpu_core::{DeviceBuffer, Gpu};

use crate::block::{gqa_bwd, gqa_fwd, rmsnorm_bwd, rmsnorm_fwd, rope2d_partial_bwd, rope2d_partial_fwd, Gqa, KernelIds};

/// Kernel-pipeline indices [`gqa_mixer_fwd`]/[`gqa_mixer_bwd`] dispatch,
/// beyond `kernels`. Resolved by the calling model against its own
/// registered pipeline list, same convention as [`crate::gdn_mixer::GdnMixerIds`].
#[derive(Clone, Copy)]
pub struct GqaMixerIds {
    /// `rmsnorm` (QK-norm) + `gqa_scores`/`attn_softmax`/`gqa_apply`
    /// (attention) + their backward twins.
    pub kernels: KernelIds,
    pub concat_split: usize,
    pub concat2: usize,
    pub sigmoid: usize,
    pub sigmoid_bwd: usize,
    pub mul: usize,
    /// Partial M-RoPE (`rope2d_partial.wgsl`) - see [`rope2d_partial_fwd`]'s doc.
    pub rope2d_partial: usize,
}

/// The mixer's shape.
#[derive(Clone, Copy)]
pub struct GqaMixerShape {
    pub b: u32,
    pub t: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// `config.rotary_dim() / 2` - the M-RoPE table width (partial rotary
    /// factor already folded in).
    pub rotary_half: u32,
}

impl GqaMixerShape {
    pub fn qd(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    pub fn kvd(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
}

/// The mixer's non-projection weights - never a LoRA target, never
/// quantized. `cos`/`sin` are the model-level `[block_size, rotary_half]`
/// M-RoPE tables (shared by every layer), not per-layer weights.
pub struct GqaMixerWeights<'a> {
    pub q_norm: &'a DeviceBuffer,
    pub k_norm: &'a DeviceBuffer,
    pub cos: &'a DeviceBuffer,
    pub sin: &'a DeviceBuffer,
}

/// [`GqaMixerWeights`]'s gradient buffers, for [`gqa_mixer_bwd`]. `None`
/// when the corresponding weight is Frozen.
pub struct GqaMixerGrads<'a> {
    pub q_norm: Option<&'a DeviceBuffer>,
    pub k_norm: Option<&'a DeviceBuffer>,
}

/// Everything [`gqa_mixer_bwd`] needs beyond what it recomputes fresh. The
/// caller's own `ctx_gated` (this function's forward return value) is NOT
/// included here - the caller already owns it.
pub struct GqaMixerActs {
    pub q_normed: DeviceBuffer, // post QK-norm AND post-RoPE (rope is in-place; gqa_bwd's own `q`)
    pub k_normed: DeviceBuffer, // post QK-norm AND post-RoPE (gqa_bwd's own `kbuf`)
    pub v: DeviceBuffer,        // raw v projection (gqa_bwd's own `v`)
    pub q_value: DeviceBuffer,  // pre q_norm (q_norm's rmsnorm_bwd `x`)
    pub k: DeviceBuffer,        // pre k_norm (k_norm's rmsnorm_bwd `x`)
    pub q_gate: DeviceBuffer,   // pre-sigmoid
    pub probs: DeviceBuffer,
    pub ctx: DeviceBuffer,
    pub gate: DeviceBuffer, // post-sigmoid (mul_bwd's other operand)
}

/// `q_full = q_proj(xn1)` (the doubled `[value|gate]` projection), `k =
/// k_proj(xn1)`, `v = v_proj(xn1)` -> `ctx_gated`, ready for the caller's own
/// `o_proj`. `n` is the row count (`b*t`); `is_train` gates whether the
/// activations [`gqa_mixer_bwd`] needs are saved.
pub fn gqa_mixer_fwd(g: &Gpu, ids: &GqaMixerIds, shape: &GqaMixerShape, w: &GqaMixerWeights, q_full: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, n: u32, is_train: bool) -> (DeviceBuffer, Option<GqaMixerActs>) {
    let (nh, nkv, hd) = (shape.n_heads, shape.n_kv_heads, shape.head_dim);
    let (qd, kvd) = (shape.qd(), shape.kvd());

    // Per-head de-interleaved split of q_full's [query|gate] halves - NOT a
    // whole-row split. Fold n_heads into concat_split's own N so each head's
    // 2*head_dim block splits into its own first/second half.
    let q_value = g.storage((n * qd) as u64);
    let q_gate = g.storage((n * qd) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.concat_split, &[q_full, &q_value], &[n * nh, 2 * hd, hd, 0, 1, 1], n * nh * hd),
            g.step(ids.concat_split, &[q_full, &q_gate], &[n * nh, 2 * hd, hd, hd, 1, 1], n * nh * hd),
        ],
    );

    let q_normed = g.storage((n * qd) as u64);
    let k_normed = g.storage((n * kvd) as u64);
    g.submit(
        &[],
        &[
            rmsnorm_fwd(g, &ids.kernels, &q_value, w.q_norm, &q_normed, hd, n * nh),
            rmsnorm_fwd(g, &ids.kernels, k, w.k_norm, &k_normed, hd, n * nkv),
        ],
    );

    let half = shape.rotary_half;
    g.submit(
        &[],
        &[
            rope2d_partial_fwd(g, ids.rope2d_partial, &q_normed, w.cos, w.sin, n, nh, half, qd, 0, hd),
            rope2d_partial_fwd(g, ids.rope2d_partial, &k_normed, w.cos, w.sin, n, nkv, half, kvd, 0, hd),
        ],
    );

    let scores = g.storage(shape.b as u64 * nh as u64 * shape.t as u64 * shape.t as u64);
    let probs = g.storage(shape.b as u64 * nh as u64 * shape.t as u64 * shape.t as u64);
    let ctx = g.storage((n * qd) as u64);
    let ga = Gqa { b: shape.b, t: shape.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
    g.submit(&[], &gqa_fwd(g, &ids.kernels, &ga, &q_normed, &k_normed, v, &scores, &probs, &ctx));

    let gate = g.storage((n * qd) as u64);
    let ctx_gated = g.storage((n * qd) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.sigmoid, &[&q_gate, &gate], &[n * qd], n * qd),
            g.step(ids.mul, &[&ctx, &gate, &ctx_gated], &[n * qd], n * qd),
        ],
    );

    let acts = is_train.then(|| GqaMixerActs { q_normed, k_normed, v: v.clone(), q_value, k: k.clone(), q_gate, probs, ctx, gate });
    (ctx_gated, acts)
}

/// Reverse of [`gqa_mixer_fwd`]: `d_ctx_gated` (the caller's own `o_proj`
/// backward output) -> `(d_q_full, d_k, d_v)`, for the caller's own
/// `q_proj`/`k_proj`/`v_proj` backward. `n` must match the forward call's own.
pub fn gqa_mixer_bwd(g: &Gpu, ids: &GqaMixerIds, shape: &GqaMixerShape, w: &GqaMixerWeights, gw: &GqaMixerGrads, la: &GqaMixerActs, d_ctx_gated: &DeviceBuffer, n: u32) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let (nh, nkv, hd) = (shape.n_heads, shape.n_kv_heads, shape.head_dim);
    let (qd, kvd) = (shape.qd(), shape.kvd());

    // ---- ctx*gate backward, sigmoid backward ----
    let d_ctx = g.storage((n * qd) as u64);
    let d_gate = g.storage((n * qd) as u64);
    let d_q_gate = g.storage((n * qd) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.mul, &[d_ctx_gated, &la.gate, &d_ctx], &[n * qd], n * qd),
            g.step(ids.mul, &[d_ctx_gated, &la.ctx, &d_gate], &[n * qd], n * qd),
            g.step(ids.sigmoid_bwd, &[&la.q_gate, &d_gate, &d_q_gate], &[n * qd], n * qd),
        ],
    );

    // ---- gqa_bwd ----
    let ga = Gqa { b: shape.b, t: shape.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
    let d_scores = g.storage(shape.b as u64 * nh as u64 * shape.t as u64 * shape.t as u64);
    let d_q_normed = g.storage((n * qd) as u64);
    let d_k_normed = g.storage((n * kvd) as u64);
    let d_v = g.storage((n * kvd) as u64);
    g.submit(&[], &gqa_bwd(g, &ids.kernels, &ga, &la.q_normed, &la.k_normed, &la.v, &la.probs, &d_ctx, &d_scores, &d_q_normed, &d_k_normed, &d_v));

    // ---- RoPE backward (in place, sign=-1) ----
    let half = shape.rotary_half;
    g.submit(
        &[],
        &[
            rope2d_partial_bwd(g, ids.rope2d_partial, &d_q_normed, w.cos, w.sin, n, nh, half, qd, 0, hd),
            rope2d_partial_bwd(g, ids.rope2d_partial, &d_k_normed, w.cos, w.sin, n, nkv, half, kvd, 0, hd),
        ],
    );

    // ---- per-head QK-norm backward ----
    let d_q_value = g.storage((n * qd) as u64);
    let d_k = g.storage((n * kvd) as u64);
    {
        let inv_q = g.storage((n * nh) as u64);
        let inv_k = g.storage((n * nkv) as u64);
        let mut s = Vec::new();
        s.extend(rmsnorm_bwd(g, &ids.kernels, &la.q_value, w.q_norm, &d_q_normed, &d_q_value, &inv_q, gw.q_norm, hd, n * nh));
        s.extend(rmsnorm_bwd(g, &ids.kernels, &la.k, w.k_norm, &d_k_normed, &d_k, &inv_k, gw.k_norm, hd, n * nkv));
        g.submit(&[], &s);
    }

    // ---- q_full [value|gate] split backward (concat2, per-head interleaved) ----
    let qpd = shape.n_heads * shape.head_dim * 2;
    let d_q_full = g.storage((n * qpd) as u64);
    g.submit(&[], &[g.step(ids.concat2, &[&d_q_value, &d_q_gate, &d_q_full], &[n * nh, hd, hd, 1, 1], n * nh * 2 * hd)]);

    (d_q_full, d_k, d_v)
}
