// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-rank fine-tuning of mixture-of-experts models whose routed experts stay in their
// published quantized encoding for its clients. If your team needs expertise in training adapters over frozen expert
// banks then you can procure our services by sending an email to info@swedishembedded.com.

//! Low-rank adapters over a stacked expert bank, on the row-compacted training path ([`super::experts_fwd_train_adapted`],
//! [`super::moe_layer_bwd_adapted`]).
//!
//! One adapter covers one projection of every expert at once: `A [n_experts, rank, k]` and `B [n_experts, n, rank]`,
//! expert `e`'s pair at element offsets `e * rank * k` and `e * n * rank`. On the rows routed to expert `e` the
//! projection's output gains `scale * B_e (A_e x)`; the rows of other experts never read or touch expert `e`'s pair, so
//! an expert no row routes to keeps its gradients exactly as the caller left them.
//!
//! The products are the gated MoE linears every dense expert bank already trains through (`moe_linear_gated` and its
//! `_dx` / `_dw`, which take an element offset into a stacked bank and skip rows whose gate for the expert is zero), the
//! scaled add is `axpy` and the scale of the low-rank intermediates `grad_scale`: no kernel of its own. The backward
//! recomputes `A_e x` from the input the expert path already holds (the gathered `x` rows, or the saved activation `h`
//! of the down projection), so the adapter saves no activation.

use gpu_core::{DeviceBuffer, Gpu, Step};

use super::MoeShape;

/// The kernels an adapted expert path dispatches beside [`super::COMPACT_TRAIN_KERNELS`], resolved by name. A model that
/// trains stacked-expert adapters registers these too (a name it already lists is not needed twice).
pub const EXPERT_LORA_KERNELS: &[(&str, &str)] = &[
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("moe_linear_gated_dx", kernels::MOE_LINEAR_GATED_DX),
    ("moe_linear_gated_dw", kernels::MOE_LINEAR_GATED_DW),
    ("axpy", kernels::AXPY),
    ("grad_scale", kernels::GRAD_SCALE),
];

/// A low-rank adapter over one projection of a stacked expert bank (module doc): `a` is `A [n_experts, rank, k]`, `b`
/// is `B [n_experts, n, rank]`, row-major, and `scale` is the factor the product is multiplied by (`alpha / rank`, times
/// any strength). `k` and `n` are the projection's own (the bank's `[n_experts, n, k]`).
#[derive(Clone, Copy)]
pub struct ExpertLora<'a> {
    pub a: &'a DeviceBuffer,
    pub b: &'a DeviceBuffer,
    pub rank: u32,
    pub scale: f32,
}

/// Where the backward adds an [`ExpertLora`]'s gradients: `a` is `dA [n_experts, rank, k]`, `b` is `dB [n_experts, n,
/// rank]`. The backward ADDS each routed expert's gradient into its own range (the caller zeroes them, or accumulates
/// over micro-batches); the range of an expert no row routes to is not written.
#[derive(Clone, Copy)]
pub struct ExpertLoraGrads<'a> {
    pub a: &'a DeviceBuffer,
    pub b: &'a DeviceBuffer,
}

/// The adapters of one MoE layer's expert projections: `gate_up` corrects the gate projection (the separate layout) or
/// the fused gate/up projection (in its declared row layout), `up` the separate layout's up projection, `down` the down
/// projection. Each is optional; the default adapts nothing.
#[derive(Clone, Copy, Default)]
pub struct ExpertAdapters<'a> {
    pub gate_up: Option<ExpertLora<'a>>,
    pub up: Option<ExpertLora<'a>>,
    pub down: Option<ExpertLora<'a>>,
}

/// The gradient buffers of [`ExpertAdapters`], projection by projection. An adapter without gradient buffers is held
/// frozen (its share of the input gradient still flows); gradient buffers for an absent adapter are refused.
#[derive(Clone, Copy, Default)]
pub struct ExpertAdapterGrads<'a> {
    pub gate_up: Option<ExpertLoraGrads<'a>>,
    pub up: Option<ExpertLoraGrads<'a>>,
    pub down: Option<ExpertLoraGrads<'a>>,
}

/// One adapted projection of a layer: its adapter and its `(n, k)`.
pub(super) struct Projection<'a> {
    pub lora: ExpertLora<'a>,
    pub n: u32,
    pub k: u32,
}

impl<'a> ExpertAdapters<'a> {
    pub fn is_empty(&self) -> bool {
        self.gate_up.is_none() && self.up.is_none() && self.down.is_none()
    }

    /// The adapted projections of a layer of `shape` (`fused`: one gate/up projection `2 * moe_ff` wide), each checked
    /// ([`Projection::checked`]); an `up` adapter is only for the separate layout.
    pub(super) fn projections(&self, shape: &MoeShape, fused: bool) -> Result<[Option<Projection<'a>>; 3], String> {
        let (d, ff) = (shape.d_model, shape.moe_ff);
        if fused && self.up.is_some() {
            return Err("expert adapters: an up adapter needs the separate gate/up layout; a fused bank's adapter is gate_up".into());
        }
        let gate_rows = if fused { 2 * ff } else { ff };
        Ok([
            Projection::checked("gate_up", self.gate_up, gate_rows, d, shape.n_experts)?,
            Projection::checked("up", self.up, ff, d, shape.n_experts)?,
            Projection::checked("down", self.down, d, ff, shape.n_experts)?,
        ])
    }

    /// The widest rank among the adapters: what the low-rank scratch is sized for.
    pub(super) fn max_rank(&self) -> u32 {
        [self.gate_up, self.up, self.down].iter().flatten().map(|l| l.rank).max().unwrap_or(0)
    }
}

impl ExpertAdapterGrads<'_> {
    /// Gradient buffers only where an adapter is.
    pub(super) fn check(&self, adapters: &ExpertAdapters) -> Result<(), String> {
        for (name, grads, lora) in [("gate_up", self.gate_up.is_some(), adapters.gate_up.is_some()), ("up", self.up.is_some(), adapters.up.is_some()), ("down", self.down.is_some(), adapters.down.is_some())] {
            if grads && !lora {
                return Err(format!("expert adapters: gradient buffers for the {name} projection, which has no adapter"));
            }
        }
        Ok(())
    }
}

/// The pipeline indices of [`EXPERT_LORA_KERNELS`] on one device.
#[derive(Clone, Copy)]
pub(super) struct LoraKernels {
    linear: usize,
    dx: usize,
    dw: usize,
    axpy: usize,
    scale: usize,
}

impl LoraKernels {
    pub fn resolve(g: &Gpu) -> Result<LoraKernels, String> {
        let find = |name: &str| g.kernel_index(name).ok_or_else(|| format!("expert adapters need the `{name}` kernel registered (model::moe::EXPERT_LORA_KERNELS)"));
        Ok(LoraKernels { linear: find("moe_linear_gated")?, dx: find("moe_linear_gated_dx")?, dw: find("moe_linear_gated_dw")?, axpy: find("axpy")?, scale: find("grad_scale")? })
    }
}

/// The low-rank scratch of one layer's adapted experts: `t` and `dt` hold `[rows, rank]`, `u` holds `[rows, n]` of the
/// widest adapted projection.
pub(super) struct LoraScratch {
    t: DeviceBuffer,
    dt: DeviceBuffer,
    u: DeviceBuffer,
}

impl LoraScratch {
    pub fn new(g: &Gpu, rows: u64, rank: u32, widest: u32) -> LoraScratch {
        let r = rows * u64::from(rank.max(1));
        LoraScratch { t: g.storage(r), dt: g.storage(r), u: g.storage(rows * u64::from(widest.max(1))) }
    }
}

impl<'a> Projection<'a> {
    /// The projection `name` of `[ne, n, k]` banks, when it is adapted: a rank of at least 1, a finite scale, and every
    /// expert's factor offsets inside the 32-bit element range the kernels address.
    fn checked(name: &'static str, lora: Option<ExpertLora<'a>>, n: u32, k: u32, ne: u32) -> Result<Option<Projection<'a>>, String> {
        let Some(lora) = lora else { return Ok(None) };
        if lora.rank == 0 {
            return Err(format!("expert adapters: the {name} adapter has rank 0"));
        }
        if !lora.scale.is_finite() {
            return Err(format!("expert adapters: the {name} adapter's scale {} is not finite", lora.scale));
        }
        let words = |rows: u32, cols: u32| u64::from(ne) * u64::from(rows) * u64::from(cols);
        if words(lora.rank, k).max(words(n, lora.rank)) > u64::from(u32::MAX) {
            return Err(format!("expert adapters: the {name} adapter's {ne} x rank-{} factors over [{n}, {k}] exceed the 32-bit element offsets of the expert kernels", lora.rank));
        }
        Ok(Some(Projection { lora, n, k }))
    }

    /// `out [rows, n] += scale * B_e (A_e x)` on expert `e`'s rows (`x` `[rows, k]`, `gate_c` the rows' gates).
    pub fn forward(&self, g: &Gpu, kn: &LoraKernels, s: &LoraScratch, sub: &MoeShape, e: u32, x: &DeviceBuffer, gate_c: &DeviceBuffer, out: &DeviceBuffer) -> [Step; 3] {
        let (m, ne, r, n, k) = (sub.rows, sub.n_experts, self.lora.rank, self.n, self.k);
        let (a_off, b_off) = (e * r * k, e * n * r);
        [
            g.step(kn.linear, &[x, self.lora.a, gate_c, &s.t], &[m, k, r, ne, e, a_off], m * r),
            g.step(kn.linear, &[&s.t, self.lora.b, gate_c, &s.u], &[m, r, n, ne, e, b_off], m * n),
            g.step(kn.axpy, &[out, &s.u], &[m * n, gpu_core::f(self.lora.scale)], m * n),
        ]
    }

    /// The backward of [`Self::forward`] on expert `e`'s rows from `d_out [rows, n]`: `dB_e += scale * d_out^T (A_e x)`
    /// and `dA_e += scale * (d_out B_e)^T x` when `grads` are given, and the adapter's share of the input gradient
    /// `scale * (d_out B_e) A_e` added into `d_in [rows, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(&self, g: &Gpu, kn: &LoraKernels, s: &LoraScratch, grads: Option<ExpertLoraGrads>, sub: &MoeShape, e: u32, x: &DeviceBuffer, gate_c: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer, steps: &mut Vec<Step>) {
        let (m, ne, r, n, k) = (sub.rows, sub.n_experts, self.lora.rank, self.n, self.k);
        let (a_off, b_off) = (e * r * k, e * n * r);
        let scale = gpu_core::f(self.lora.scale);
        if let Some(gr) = grads {
            // `t = scale * A_e x`, then `dB_e += d_out^T t`.
            steps.push(g.step(kn.linear, &[x, self.lora.a, gate_c, &s.t], &[m, k, r, ne, e, a_off], m * r));
            steps.push(g.step(kn.scale, &[&s.t], &[m * r, scale], m * r));
            steps.push(g.step(kn.dw, &[d_out, &s.t, gate_c, gr.b], &[m, r, n, ne, e, b_off], n * r));
        }
        // `dt = scale * d_out B_e`.
        steps.push(g.step(kn.dx, &[d_out, self.lora.b, gate_c, &s.dt], &[m, r, n, ne, e, 0, b_off], m * r));
        steps.push(g.step(kn.scale, &[&s.dt], &[m * r, scale], m * r));
        if let Some(gr) = grads {
            steps.push(g.step(kn.dw, &[&s.dt, x, gate_c, gr.a], &[m, k, r, ne, e, a_off], r * k));
        }
        steps.push(g.step(kn.dx, &[&s.dt, self.lora.a, gate_c, d_in], &[m, k, r, ne, e, 1, a_off], m * k));
    }
}

/// A layer's adapted projections with their kernels and scratch, indexed `0` gate/up, `1` up, `2` down: what the
/// compacted forward and backward call per expert. Empty (and dispatching nothing) without adapters.
pub(super) struct AdaptedProjections<'a> {
    proj: [Option<Projection<'a>>; 3],
    run: Option<(LoraKernels, LoraScratch)>,
}

impl<'a> AdaptedProjections<'a> {
    /// `fused` is the bank's gate/up layout, `dense` whether its weights are a trainable [`super::ExpertBank::Dense`]
    /// (refused with adapters), `most` the largest routed row count of an expert.
    pub fn new(g: &Gpu, adapters: &ExpertAdapters<'a>, shape: &MoeShape, fused: bool, dense: bool, most: u64) -> Result<AdaptedProjections<'a>, String> {
        if adapters.is_empty() {
            return Ok(AdaptedProjections { proj: [None, None, None], run: None });
        }
        if dense {
            return Err("expert adapters ride beside a frozen microscaled bank (ExpertBank::Micro); a dense bank trains its own weights".into());
        }
        let proj = adapters.projections(shape, fused)?;
        let widest = proj.iter().flatten().map(|p| p.n).max().unwrap_or(1);
        let run = (LoraKernels::resolve(g)?, LoraScratch::new(g, most, adapters.max_rank(), widest));
        Ok(AdaptedProjections { proj, run: Some(run) })
    }

    /// [`Projection::forward`] of projection `i`, when it is adapted.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, g: &Gpu, i: usize, sub: &MoeShape, e: u32, x: &DeviceBuffer, gate_c: &DeviceBuffer, out: &DeviceBuffer, steps: &mut Vec<Step>) {
        if let (Some(p), Some((kn, s))) = (&self.proj[i], &self.run) {
            steps.extend(p.forward(g, kn, s, sub, e, x, gate_c, out));
        }
    }

    /// [`Projection::backward`] of projection `i`, when it is adapted.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(&self, g: &Gpu, i: usize, grads: Option<ExpertLoraGrads>, sub: &MoeShape, e: u32, x: &DeviceBuffer, gate_c: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer, steps: &mut Vec<Step>) {
        if let (Some(p), Some((kn, s))) = (&self.proj[i], &self.run) {
            p.backward(g, kn, s, grads, sub, e, x, gate_c, d_out, d_in, steps);
        }
    }
}
