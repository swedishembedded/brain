// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dispatch emitters over **row ranges of one buffer**.
//!
//! A family of models in this workspace builds the same shape of graph: linear /
//! LayerNorm / cross-attention chains where a "concatenation" is not a copy but
//! two producers writing into different row offsets of a single buffer. The
//! IP-Adapter lineage in particular — PuLID's `IDFormer`, InstantID's
//! `Resampler`, and the `PerceiverAttention` both are built from — is exactly
//! this: `cat(x, latents)` is one buffer whose first rows are `x` and whose rest
//! are the latents, and `to_kv` runs over the whole thing.
//!
//! This module is that emitter, hoisted out of `crates/pulid` so the second such
//! model does not carry a second copy. Writing one is easy and getting it subtly
//! different is easier — the row arithmetic, the fused-`kv` strides and the
//! `region_copy` aliasing rule are each a place two copies drift.
//!
//! # Why row ranges rather than separate buffers
//!
//! `step_sliced` binds a *view* of a buffer, and `region_copy` copies at equal
//! indices **within the two bound views** — so two slices at different row
//! offsets are exactly a row-range copy, with no kernel that knows about
//! offsets. The same trick makes a concatenation free: the producing linear
//! writes straight into the destination rows.
//!
//! Storage bindings must respect the 256-byte `min_storage_buffer_offset_
//! alignment` (= 64 floats), so a caller slicing at a row offset needs
//! `r0 * d` to be a multiple of 64. That is a property of the *caller's* dims,
//! not of this module, and it is why `crates/model/src/vit.rs` passes offsets in
//! kernel `Params` where its windows are ragged.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::block::{gemm_variant, ln_variant, GemmVariants};

/// A `[.., d]` row range as a `(float offset, float length)` binding slice.
pub fn rows(r0: usize, n: usize, d: usize) -> (u64, u64) {
    ((r0 * d) as u64, (n * d) as u64)
}

/// An f32 `Params` word. Kernel params are `u32`; floats travel bit-cast.
pub fn fbits(x: f32) -> u32 {
    x.to_bits()
}

/// The kernel indices [`RowEmit`] dispatches, as a model's own pipeline
/// positions.
///
/// Resolved by name against the list the [`Gpu`] was built with rather than
/// hardcoded, because a wrong index is silently wrong output rather than a
/// crash (`.agents/rules/kernels.md`).
#[derive(Clone, Copy, Debug)]
pub struct RowKernels {
    pub ln: usize,
    pub ln_rows: usize,
    pub mm: usize,
    pub mm_reg: usize,
    pub mm_gemv: usize,
    pub bias: usize,
    pub gelu: usize,
    pub xscores: usize,
    pub xsoftmax: usize,
    pub xapply: usize,
    pub copy: usize,
}

/// The kernels [`RowKernels::resolve`] requires, so a model can concatenate them
/// into its own `PIPELINES` list without transcribing the names.
pub const REQUIRED: [&str; 11] = [
    "layernorm",
    "layernorm_rows",
    "matmul",
    "matmul_reg3",
    "matmul_gemv",
    "bias_add",
    "gelu_erf",
    "attn_scores_cross",
    "attn_softmax_cross",
    "attn_apply_cross",
    "region_copy",
];

impl RowKernels {
    /// Resolve against the pipeline list the [`Gpu`] was constructed with.
    /// Panics naming the kernel if one is absent.
    pub fn resolve(who: &str, names: &[(&str, &str)]) -> RowKernels {
        let f = |k: &str| {
            names
                .iter()
                .position(|(n, _)| *n == k)
                .unwrap_or_else(|| panic!("{who}: the Gpu was built without the `{k}` kernel"))
        };
        RowKernels {
            ln: f("layernorm"),
            ln_rows: f("layernorm_rows"),
            mm: f("matmul"),
            mm_reg: f("matmul_reg3"),
            mm_gemv: f("matmul_gemv"),
            bias: f("bias_add"),
            gelu: f("gelu_erf"),
            xscores: f("attn_scores_cross"),
            xsoftmax: f("attn_softmax_cross"),
            xapply: f("attn_apply_cross"),
            copy: f("region_copy"),
        }
    }

    /// The fp32 GEMM tier to dispatch on this device — the ONE rule `flux1`,
    /// `flux2` and the IP-Adapter models share.
    ///
    /// Gated on the queried `DeviceCaps::workgroup_reductions`: both fast
    /// kernels cooperate across a workgroup, so a device without that capability
    /// keeps the naive one-thread-per-output reference (which `backend-cpu`
    /// routes to its AVX2 GEMM anyway).
    pub fn tier(&self, g: &Gpu) -> GemmVariants {
        if g.caps().workgroup_reductions {
            GemmVariants::Fast { gemv: Some(self.mm_gemv), tiled: self.mm_reg }
        } else {
            GemmVariants::Reference(self.mm)
        }
    }
}

/// Records dispatches over row ranges. Cheap to construct; hold one per graph
/// build.
pub struct RowEmit<'a> {
    pub g: &'a Gpu,
    pub k: RowKernels,
    pub eps: f32,
    tier: GemmVariants,
}

impl<'a> RowEmit<'a> {
    pub fn new(g: &'a Gpu, k: RowKernels, eps: f32) -> RowEmit<'a> {
        RowEmit { g, k, eps, tier: k.tier(g) }
    }

    /// The GEMM tier this emitter resolved, for a caller dispatching its own
    /// matmuls under the same rule.
    pub fn tier(&self) -> GemmVariants {
        self.tier
    }

    /// `y[yr0..yr0+m] = LayerNorm(x[xr0..xr0+m]) * gamma + beta`.
    #[allow(clippy::too_many_arguments)]
    pub fn ln(
        &self,
        s: &mut Vec<Step>,
        x: &DeviceBuffer,
        xr0: usize,
        gamma: &DeviceBuffer,
        beta: &DeviceBuffer,
        y: &DeviceBuffer,
        yr0: usize,
        m: usize,
        d: usize,
    ) {
        let (kind, threads) = ln_variant(self.g, self.k.ln, Some(self.k.ln_rows), m as u32, d as u32);
        s.push(self.g.step_sliced(
            kind,
            &[x, gamma, beta, y],
            &[rows(xr0, m, d), (0, 0), (0, 0), rows(yr0, m, d)],
            &[d as u32, m as u32, fbits(self.eps)],
            threads,
        ));
    }

    /// `y[yr0..] = x[xr0..xr0+m] @ Wᵀ` (+ `bias` when given), `W` = `[n, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        s: &mut Vec<Step>,
        x: &DeviceBuffer,
        xr0: usize,
        w: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        y: &DeviceBuffer,
        yr0: usize,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let (kind, threads) = gemm_variant(self.tier, m as u32, n as u32);
        s.push(self.g.step_sliced(
            kind,
            &[x, w, y],
            &[rows(xr0, m, k), (0, 0), rows(yr0, m, n)],
            &[m as u32, k as u32, n as u32],
            threads,
        ));
        if let Some(b) = bias {
            s.push(self.g.step_sliced(
                self.k.bias,
                &[y, b],
                &[rows(yr0, m, n), (0, 0)],
                &[m as u32, n as u32],
                (m * n) as u32,
            ));
        }
    }

    /// `dst[r1..r1+m] = src[r0..r0+m]`, both `[.., d]`. `region_copy` copies at
    /// equal indices *within the two bound views*, so two slices at different
    /// row offsets are exactly a row-range copy.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows(
        &self,
        s: &mut Vec<Step>,
        src: &DeviceBuffer,
        r0: usize,
        dst: &DeviceBuffer,
        r1: usize,
        m: usize,
        d: usize,
    ) {
        s.push(self.g.step_sliced(
            self.k.copy,
            &[src, dst],
            &[rows(r0, m, d), rows(r1, m, d)],
            &[m as u32, d as u32, d as u32, 0],
            (m * d) as u32,
        ));
    }

    /// Cross-attention: `[t_dec, inner]` queries at row `q_r0` against a fused
    /// `[t_enc, 2·inner]` kv buffer, context out as `[t_dec, inner]`.
    ///
    /// The fused-kv layout is the load-bearing detail: `k` occupies each row's
    /// first `inner` floats and `v` the second, so the kv stride is `2*inner`
    /// and `v_off` is `inner`. Passing `inner` as the stride — the natural
    /// mistake when the buffer is not fused — reads `v` as `k` and still runs.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_attn(
        &self,
        s: &mut Vec<Step>,
        q: &DeviceBuffer,
        q_r0: usize,
        kv: &DeviceBuffer,
        scores: &DeviceBuffer,
        probs: &DeviceBuffer,
        ctx: &DeviceBuffer,
        heads: usize,
        head_dim: usize,
        t_dec: usize,
        t_enc: usize,
    ) {
        let inner = (heads * head_dim) as u32;
        let (h, hd, td, te) = (heads as u32, head_dim as u32, t_dec as u32, t_enc as u32);
        // `attn_scores_cross` Params:
        //   [bsz, n_heads, t_dec, t_enc, head_dim, q_stride, kv_stride, q_off, k_off]
        s.push(self.g.step_sliced(
            self.k.xscores,
            &[q, kv, scores],
            &[rows(q_r0, t_dec, inner as usize), (0, 0), (0, 0)],
            &[1, h, td, te, hd, inner, 2 * inner, 0, 0],
            h * td * te,
        ));
        // `attn_softmax_cross` Params: [bsz, n_heads, t_dec, t_enc]
        s.push(self.g.step(self.k.xsoftmax, &[scores, probs], &[1, h, td, te], h * td));
        // `attn_apply_cross` Params:
        //   [bsz, n_heads, t_dec, t_enc, head_dim, kv_stride, v_off, d_model]
        s.push(self.g.step(
            self.k.xapply,
            &[probs, kv, ctx],
            &[1, h, td, te, hd, 2 * inner, inner, inner],
            h * td * hd,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_is_a_float_offset_and_length() {
        assert_eq!(rows(0, 4, 8), (0, 32));
        assert_eq!(rows(2, 4, 8), (16, 32));
    }

    #[test]
    fn required_lists_exactly_what_resolve_looks_up() {
        // If a kernel is added to `resolve` and not to `REQUIRED`, a model that
        // built its PIPELINES from `REQUIRED` would panic at construction —
        // late, and in a way that reads as the model's fault.
        let names: Vec<(&str, &str)> = REQUIRED.iter().map(|n| (*n, "")).collect();
        let _ = RowKernels::resolve("test", &names);
    }
}
