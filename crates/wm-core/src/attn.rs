// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Biased / QK-normalized attention host helpers — the ST-transformer
//! attention primitive shared by GenieRedux's tokenizer and dynamics.
//!
//! Two features distinguish it from the stock `1/sqrt(head_dim)` softmax
//! attention already in the kernel set, and both are wired here:
//!
//! - an additive per-head score bias `[H,T,T]` (spatial ContinuousPositionBias
//!   or temporal ALiBi), and a CONSTANT scale (GenieRedux uses 8);
//! - QK-normalization: q and k are L2-normalized over `head_dim` and multiplied
//!   by a learnable per-dim scale before the dot product.
//!
//! Layout matches the rest of the attention zoo: q,k,v live in a fused
//! `[B, T, 3C]` buffer (`q_off=0, k_off=C, v_off=2C`); scores/probs are
//! `[B,H,T,T]`; the caller supplies `bias` as `[H,T,T]` shared across the batch.
//! The QK-norm operates on q (or k) viewed as `[B*T*H, head_dim]`, which is the
//! natural per-head slice of the fused buffer's q/k region when `C = H*hd`.
//!
//! Every method RETURNS a [`Step`] (it does not submit), so the model batches
//! the whole block into one graph. The softmax / apply / dscores / dv stages
//! reuse the existing bidir kernels: the causal mask is carried by the `-1e30`
//! the causal scores kernel writes for `j>i`, which softmax turns into a zero
//! probability, so no separate causal softmax/apply is needed.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Kernel-table indices for the biased/QK-norm attention pipeline.
#[derive(Clone, Copy, Debug)]
pub struct BiasedAttn {
    pub l2norm: usize,
    pub l2norm_dx: usize,
    pub l2norm_dg: usize,
    pub scores_bidir: usize,
    pub scores_causal: usize,
    pub softmax: usize,
    pub apply: usize,
    pub dscores: usize,
    pub dv: usize,
    pub dq: usize,
    pub dk: usize,
    pub dbias: usize,
}

impl BiasedAttn {
    /// `(name, source)` pairs for `Gpu::new`, in the order [`BiasedAttn::seq`]
    /// assigns indices.
    pub fn kernel_sources() -> [(&'static str, &'static str); 12] {
        [
            ("l2norm_scale", kernels::L2NORM_SCALE),
            ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
            ("l2norm_scale_dg", kernels::L2NORM_SCALE_DG),
            ("attn_scores_bidir_bias", kernels::ATTN_SCORES_BIDIR_BIAS),
            ("attn_scores_causal_bias", kernels::ATTN_SCORES_CAUSAL_BIAS),
            ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
            ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
            ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
            ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
            ("attn_bwd_dq_bias", kernels::ATTN_BWD_DQ_BIAS),
            ("attn_bwd_dk_bias", kernels::ATTN_BWD_DK_BIAS),
            ("attn_bwd_dbias", kernels::ATTN_BWD_DBIAS),
        ]
    }
    /// Indices matching [`BiasedAttn::kernel_sources`] loaded at offset 0.
    pub fn seq() -> BiasedAttn {
        BiasedAttn {
            l2norm: 0, l2norm_dx: 1, l2norm_dg: 2, scores_bidir: 3, scores_causal: 4,
            softmax: 5, apply: 6, dscores: 7, dv: 8, dq: 9, dk: 10, dbias: 11,
        }
    }

    // ---- QK-norm (apply to q or k viewed as [rows, head_dim]) ----

    /// `y[n,d] = x[n,d] * rsqrt(sum_k x[n,k]^2 + eps) * g[d]`, rows `= B*T*H`,
    /// dim `= head_dim`. `g` is the learnable per-dim scale `[head_dim]`.
    pub fn step_l2norm(
        &self, gpu: &Gpu, rows: u32, head_dim: u32, eps: f32,
        x: &DeviceBuffer, g: &DeviceBuffer, y: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.l2norm, &[x, g, y], &[rows, head_dim, eps.to_bits()], rows * head_dim)
    }
    pub fn step_l2norm_dx(
        &self, gpu: &Gpu, rows: u32, head_dim: u32, eps: f32,
        x: &DeviceBuffer, g: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.l2norm_dx, &[x, g, dy, dx], &[rows, head_dim, eps.to_bits()], rows * head_dim)
    }
    pub fn step_l2norm_dg(
        &self, gpu: &Gpu, rows: u32, head_dim: u32, eps: f32,
        x: &DeviceBuffer, dy: &DeviceBuffer, dg: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.l2norm_dg, &[x, dy, dg], &[rows, head_dim, eps.to_bits()], head_dim)
    }

    // ---- forward: scores -> softmax -> apply ----

    /// Fused-qkv score params: `[B, H, T, hd, 3C, q_off=0, k_off=C, scale]`.
    fn score_params(b: u32, heads: u32, t: u32, hd: u32, scale: f32) -> [u32; 8] {
        let c = heads * hd;
        [b, heads, t, hd, 3 * c, 0, c, scale.to_bits()]
    }

    /// `scores[b,h,i,j] = (q.k)*scale + bias[h,i,j]`. `causal` selects the
    /// masked (`j>i` -> -1e30) variant.
    pub fn step_scores(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32, hd: u32, scale: f32, causal: bool,
        qkv: &DeviceBuffer, bias: &DeviceBuffer, scores: &DeviceBuffer,
    ) -> Step {
        let k = if causal { self.scores_causal } else { self.scores_bidir };
        gpu.step(k, &[qkv, bias, scores], &Self::score_params(b, heads, t, hd, scale), b * heads * t * t)
    }
    /// Row-softmax over the last axis (params `[B,H,T]`); handles the causal
    /// `-1e30` entries as probability 0.
    pub fn step_softmax(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32,
        scores: &DeviceBuffer, probs: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.softmax, &[scores, probs], &[b, heads, t], b * heads * t)
    }
    /// `out[b,i,h,d] = sum_j probs[b,h,i,j] * v[b,j,h,d]`, v from the fused
    /// buffer (`v_off = 2C`). params `[B,H,T,hd,3C,2C,C]`.
    pub fn step_apply(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32, hd: u32,
        probs: &DeviceBuffer, qkv: &DeviceBuffer, out: &DeviceBuffer,
    ) -> Step {
        let c = heads * hd;
        gpu.step(self.apply, &[probs, qkv, out], &[b, heads, t, hd, 3 * c, 2 * c, c], b * heads * t * hd)
    }

    // ---- backward ----

    fn ap(b: u32, heads: u32, t: u32, hd: u32) -> [u32; 7] {
        let c = heads * hd;
        [b, heads, t, hd, 3 * c, 2 * c, c]
    }
    fn dqk_params(b: u32, heads: u32, t: u32, hd: u32, scale: f32, causal: bool) -> [u32; 9] {
        let c = heads * hd;
        [b, heads, t, hd, 3 * c, 0, c, scale.to_bits(), causal as u32]
    }

    /// Softmax-jacobian backward: `d_scores` (pre-softmax) from `d_out`.
    pub fn step_dscores(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32, hd: u32,
        d_out: &DeviceBuffer, qkv: &DeviceBuffer, probs: &DeviceBuffer, d_scores: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.dscores, &[d_out, qkv, probs, d_scores], &Self::ap(b, heads, t, hd), b * heads * t)
    }
    /// `d_v` into the v region of `d_qkv`.
    pub fn step_dv(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32, hd: u32,
        probs: &DeviceBuffer, d_out: &DeviceBuffer, d_qkv: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.dv, &[probs, d_out, d_qkv], &Self::ap(b, heads, t, hd), b * heads * t * hd)
    }
    /// `d_q` into the q region of `d_qkv` (configurable scale + causal range).
    pub fn step_dq(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32, hd: u32, scale: f32, causal: bool,
        d_scores: &DeviceBuffer, qkv: &DeviceBuffer, d_qkv: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.dq, &[d_scores, qkv, d_qkv], &Self::dqk_params(b, heads, t, hd, scale, causal), b * heads * t * hd)
    }
    /// `d_k` into the k region of `d_qkv`.
    pub fn step_dk(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32, hd: u32, scale: f32, causal: bool,
        d_scores: &DeviceBuffer, qkv: &DeviceBuffer, d_qkv: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.dk, &[d_scores, qkv, d_qkv], &Self::dqk_params(b, heads, t, hd, scale, causal), b * heads * t * hd)
    }
    /// `d_bias[h,i,j] = sum_b d_scores` (params `[B,H,T,causal]`).
    pub fn step_dbias(
        &self, gpu: &Gpu, b: u32, heads: u32, t: u32, causal: bool,
        d_scores: &DeviceBuffer, d_bias: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.dbias, &[d_scores, d_bias], &[b, heads, t, causal as u32], heads * t * t)
    }
}
