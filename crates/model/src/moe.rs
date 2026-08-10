// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse top-k MoE expert FFN — router weights in, combined output out,
//! without evaluating every expert densely.
//!
//! `crates/glm` (today's only HF-importable MoE forward) evaluates **every**
//! expert over the **whole** row batch and discards non-selected rows by
//! multiplying by a zero gate weight afterward (`Mlp::Moe` in
//! `crates/glm/src/model.rs`, combining with `scale_add.wgsl`) — numerically
//! exact (`router_gate.wgsl`'s own doc comment proves it), but `n_experts`x
//! the FLOPs of an actual top-k dispatch. At 128 experts / top-8
//! (Qwen3-Omni's Thinker) that is 16x wasted work, which motivated this
//! module (`docs/lessons.md`).
//!
//! The fix here is deliberately the smallest one that removes the FLOPs
//! without adding new failure modes: [`moe_linear_gated`] is `matmul.wgsl`
//! with one extra check — a row whose gate weight for this expert is zero
//! writes 0 and returns *before* the K-reduction, instead of computing it and
//! discarding it downstream. Composing three of those (gate/up/down
//! projection) plus the existing `silu_mul`/`scale_add` kernels reproduces
//! [`expert_fwd`]'s per-expert step exactly, at a cost proportional to the
//! number of rows actually routed to that expert.
//!
//! [`expert_fwd_i8`] is the same trick over `model::int8`'s packed weights,
//! via a NEW naive (non-tiled) DP4A kernel rather than gating the existing
//! `matmul_i8_dyn`/`matmul_i8_gemv` — see `moe_linear_gated_i8.wgsl`'s doc for
//! why: those stage rows into workgroup-shared memory across a barrier, and a
//! per-thread early return ahead of a `workgroupBarrier()` that not every
//! thread in the workgroup reaches is undefined behaviour in WGSL. Row-level
//! gating safely at that tier needs compaction, which this workstream already
//! deferred for the fp32 tier for the same atomics-forbidden reason.
//!
//! What this module deliberately does NOT do (left for a follow-up): no
//! row-compaction/gather-scatter (WGSL kernels here may not use atomics, so a
//! parallel stream-compaction would need a separate prefix-sum pass; the
//! per-row early-exit already removes the FLOPs, just not the thread
//! *launches* — see `docs/models/omni/status.md` M2 for the follow-up plan),
//! no TILED int8 GEMM tier (both int8 and fp32 are naive-dispatch today; a
//! future tiled+gated kernel is one change, not two, once compaction lands).
//!
//! **Backward** (this session's addition) is a hoist, not a from-scratch
//! derivation: two complete, gradient-checked MoE backwards already existed
//! — `crates/moe/src/train.rs`'s softmax-router training loop and
//! `crates/glm/src/model.rs`'s sigmoid `noaux_tc` router MLP arm — and
//! comparing them line for line shows the expert half (SwiGLU backward,
//! `scale_add_dexp`/`scale_add_dgate` combine) is IDENTICAL; only the router
//! half differs. [`RouterKind`] parameterises that one difference so there is
//! one router entry point, not two implementations. See [`moe_layer_bwd`]'s
//! doc for the phase-ordering contract every caller must follow.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Kernel indices this module dispatches, resolved by the calling model crate
/// against its own registered pipeline list (same pattern as
/// [`crate::block::KernelIds`]).
#[derive(Clone, Copy)]
pub struct MoeIds {
    /// `router_gate.wgsl` (or `router_gate_train.wgsl` if probs are wanted) —
    /// softmax -> top-k -> renormalise into a dense `[rows, n_experts]` gate.
    pub router_gate: usize,
    /// `moe_linear_gated.wgsl` — `matmul.wgsl` with a per-row gate early-exit.
    pub linear_gated: usize,
    /// `silu_mul.wgsl` — shared with every other SwiGLU MLP in the engine.
    pub silu_mul: usize,
    /// `scale_add.wgsl` — the same combine step `crates/glm` already uses.
    pub scale_add: usize,
}

/// The shape one call to [`router_fwd`]/[`expert_fwd`] operates over.
#[derive(Clone, Copy)]
pub struct MoeShape {
    pub rows: u32,
    pub d_model: u32,
    pub moe_ff: u32,
    pub n_experts: u32,
    pub top_k: u32,
}

/// Router forward: `logits [rows, n_experts] -> gate [rows, n_experts]`
/// (dense, nonzero only at the `top_k` selected experts, renormalised so they
/// sum to 1 per row). Plain softmax top-k — Qwen3-Omni's Thinker and Talker
/// routers both use this (no aux-loss-free sigmoid/bias/group-limiting; that
/// variant is `router_gate_sigmoid.wgsl`, already in use by `crates/glm`).
pub fn router_fwd(g: &Gpu, ids: &MoeIds, shape: &MoeShape, logits: &DeviceBuffer, gate: &DeviceBuffer) -> Step {
    g.step(ids.router_gate, &[logits, gate], &[shape.rows, shape.n_experts, shape.top_k], shape.rows)
}

/// Which router computes an MoE layer's per-token expert weights. The two
/// variants existing kernels support today — parameterising the difference
/// here means one router entry point ([`router_fwd_kind`]/[`router_bwd`]),
/// not two hand-duplicated implementations (confirmed by comparing
/// `crates/glm/src/model.rs:915-935` against `crates/moe/src/train.rs:466-486`
/// line for line: the expert halves are identical, only the router half
/// differs).
#[derive(Clone, Copy)]
pub enum RouterKind {
    /// `router_gate.wgsl` (fwd) + `router_bwd.wgsl` (bwd) — BOTH array-free
    /// (no expert-count cap): the backward since the #35-recurrence fix, the
    /// forward since the audit-F4 rewrite (its `array<f32,128>` scratch was
    /// the same silent-OOB literal one expert higher). Qwen3-Omni's
    /// Thinker/Talker.
    Softmax { aux_coef: f32, z_coef: f32 },
    /// `router_gate_sigmoid.wgsl` (fwd) + `router_bwd_sigmoid.wgsl` (bwd,
    /// already unbounded). `crates/glm`'s GLM-5.2/DeepSeek-V3 "noaux_tc"
    /// router: per-expert selection bias, group-limited top-k, optional
    /// renormalisation. `router_gate_sigmoid.wgsl`'s forward hard-caps at 64
    /// experts (`MAX_E`, fixed-size array scratch) — [`router_fwd_kind`]
    /// asserts this loudly rather than let it silently corrupt (the same
    /// stopgap `crates/glm/src/model.rs::new_impl_on` already applies; a
    /// real array-free top-k rewrite is separate kernel work, not a literal
    /// bump — see that assert's own doc for why).
    SigmoidNoAuxTc { n_group: u32, topk_group: u32, norm_topk_prob: bool, routed_scaling: f32 },
}

/// Router forward, dispatching whichever kernel `kind` selects. `bias`/
/// `probs` are REQUIRED (`Some`) for `SigmoidNoAuxTc` — `router_gate_sigmoid
/// .wgsl`'s mandatory 3rd/5th bindings (`probs` is written but never read
/// back by `router_bwd_sigmoid.wgsl`, which recomputes `sigmoid(logits)`
/// inline rather than reading a saved probability; the buffer must still
/// exist because the shader's own interface requires it) — and unused
/// (`None`, ignored) for `Softmax`, which has no bias/probs bindings at all.
pub fn router_fwd_kind(
    g: &Gpu,
    ids: &MoeIds,
    kind: RouterKind,
    shape: &MoeShape,
    logits: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    gate: &DeviceBuffer,
    probs: Option<&DeviceBuffer>,
) -> Step {
    match kind {
        RouterKind::Softmax { .. } => router_fwd(g, ids, shape, logits, gate),
        RouterKind::SigmoidNoAuxTc { n_group, topk_group, norm_topk_prob, routed_scaling } => {
            assert!(
                shape.n_experts <= 64,
                "RouterKind::SigmoidNoAuxTc: router_gate_sigmoid.wgsl hard-caps at 64 experts \
                 (fixed-size array scratch), got {} -- see this variant's own doc",
                shape.n_experts
            );
            let bias = bias.expect("RouterKind::SigmoidNoAuxTc requires a selection-bias buffer (router_gate_sigmoid.wgsl's 3rd binding)");
            let probs = probs.expect("RouterKind::SigmoidNoAuxTc requires a probs scratch buffer (router_gate_sigmoid.wgsl's mandatory 5th binding, unread by backward)");
            g.step(
                ids.router_gate,
                &[logits, bias, gate, probs],
                &[shape.rows, shape.n_experts, shape.top_k, n_group, topk_group, norm_topk_prob as u32, gpu_core::f(routed_scaling)],
                shape.rows,
            )
        }
    }
}

/// Kernel indices [`router_bwd`] dispatches, resolved by the calling model
/// against its own registered pipeline list.
#[derive(Clone, Copy)]
pub struct RouterBwdIds {
    /// `router_bwd.wgsl` (Softmax) or `router_bwd_sigmoid.wgsl` (SigmoidNoAuxTc).
    pub router_bwd: usize,
    /// `expert_counts.wgsl` — Softmax's aux-loss load-balancing fractions.
    /// `None` for SigmoidNoAuxTc: DeepSeek-V3's selection bias is a
    /// forward-only load-balancing heuristic, never backprop'd (matches
    /// `crates/glm/src/model.rs`'s own note keeping `moe.router.bias`
    /// `Role::Frozen`, out of the optimiser).
    pub expert_counts: Option<usize>,
}

/// Router backward: gradient w.r.t. the router logits, dispatching whichever
/// kernel `kind` selects. Returns the full step list — `Softmax` needs an
/// extra `expert_counts` dispatch first (its aux-loss term needs per-expert
/// usage fractions); `SigmoidNoAuxTc` does not (no aux loss).
///
/// Does NOT touch `d_x` or the router weight's own gradient — those are an
/// ordinary dense-linear backward over `d_router_logits` (this fn's own
/// output), which is not MoE-specific math, so this module does not own its
/// GEMM-kernel choice (`crates/glm`'s adaptive `pick_gemm` vs plain
/// `MATMUL_DW`/`MATMUL_DX`). See [`moe_layer_bwd`]'s doc for where that step
/// belongs in the required ordering.
pub fn router_bwd(
    g: &Gpu,
    ids: &RouterBwdIds,
    kind: RouterKind,
    shape: &MoeShape,
    logits: &DeviceBuffer,
    gate: &DeviceBuffer,
    d_gate: &DeviceBuffer,
    dlogits: &DeviceBuffer,
    fe: Option<&DeviceBuffer>,
) -> Vec<Step> {
    let (rows, e, top_k) = (shape.rows, shape.n_experts, shape.top_k);
    match kind {
        RouterKind::Softmax { aux_coef, z_coef } => {
            let fe = fe.expect("RouterKind::Softmax requires an fe (expert-usage) scratch buffer for the aux-loss term");
            let ec = ids.expert_counts.expect("RouterKind::Softmax requires RouterBwdIds::expert_counts");
            vec![
                g.step(ec, &[gate, fe], &[rows, e, top_k], e),
                g.step(ids.router_bwd, &[logits, gate, d_gate, fe, dlogits], &[rows, e, top_k, 0, gpu_core::f(aux_coef), gpu_core::f(z_coef)], rows),
            ]
        }
        RouterKind::SigmoidNoAuxTc { norm_topk_prob, routed_scaling, n_group, .. } => {
            assert!(n_group <= 64, "RouterKind::SigmoidNoAuxTc: router_gate_sigmoid.wgsl hard-caps at 64 experts, n_group must be <=64 (got {n_group})");
            vec![g.step(ids.router_bwd, &[logits, gate, d_gate, dlogits], &[rows, e, top_k, norm_topk_prob as u32, gpu_core::f(routed_scaling)], rows)]
        }
    }
}

/// Scratch buffers for one expert's FFN step, sized once by the caller and
/// reused across every expert (they are fully overwritten each call, so
/// nothing needs clearing between experts).
pub struct ExpertScratch<'a> {
    pub gate_pre: &'a DeviceBuffer,
    pub up: &'a DeviceBuffer,
    pub h: &'a DeviceBuffer,
    pub expert_out: &'a DeviceBuffer,
}

/// One expert's gated SwiGLU FFN step, combined into `acc` — the sparse
/// replacement for `crates/glm`'s dense per-expert loop body. `x` is the
/// (already normed) hidden state shared by every expert; `gate_w`/`up_w`/
/// `down_w` are expert `e_idx`'s own weights. `accumulate` is `false` only
/// for the very first expert in the layer's loop (matching `scale_add.wgsl`'s
/// own set-vs-add contract).
#[allow(clippy::too_many_arguments)]
pub fn expert_fwd(
    g: &Gpu,
    ids: &MoeIds,
    shape: &MoeShape,
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    gate_w: &DeviceBuffer,
    up_w: &DeviceBuffer,
    down_w: &DeviceBuffer,
    scratch: &ExpertScratch,
    acc: &DeviceBuffer,
    e_idx: u32,
    accumulate: bool,
) -> [Step; 5] {
    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);
    let lin = |x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32| {
        g.step(ids.linear_gated, &[x, w, gate, out], &[m, k, n, e, e_idx], m * n)
    };
    [
        lin(x, gate_w, scratch.gate_pre, d, ff),
        lin(x, up_w, scratch.up, d, ff),
        g.step(ids.silu_mul, &[scratch.gate_pre, scratch.up, scratch.h], &[m * ff], m * ff),
        lin(scratch.h, down_w, scratch.expert_out, ff, d),
        g.step(
            ids.scale_add,
            &[gate, scratch.expert_out, acc],
            &[m, d, e, e_idx, accumulate as u32],
            m * d,
        ),
    ]
}

/// Saved per-expert activations for backward — exactly [`ExpertScratch`]'s
/// four tensors, but OWNED per-expert rather than one shared scratch set
/// reused across experts. [`ExpertScratch`]'s forward-time contract (fully
/// overwritten each call, safe to alias across experts) does not hold for
/// training: backward needs EVERY expert's own `gate_pre`/`up`/`h`/
/// `expert_out`, not just whichever expert ran last. Allocates exactly what
/// `crates/glm/src/model.rs`'s `Mlp::Moe` already allocates per layer
/// (`model.rs:463-466`) — memory-neutral, not a reduction. Recompute-in-
/// backward and true row compaction are follow-ups that compose over this
/// API unchanged; NOT part of this item.
pub struct MoeActs {
    gate_pre: Vec<DeviceBuffer>,
    up: Vec<DeviceBuffer>,
    h: Vec<DeviceBuffer>,
    expert_out: Vec<DeviceBuffer>,
}

impl MoeActs {
    pub fn new(g: &Gpu, shape: &MoeShape) -> MoeActs {
        let n = shape.n_experts as usize;
        let (rows, ff, d) = (shape.rows as u64, shape.moe_ff as u64, shape.d_model as u64);
        MoeActs {
            gate_pre: (0..n).map(|_| g.storage(rows * ff)).collect(),
            up: (0..n).map(|_| g.storage(rows * ff)).collect(),
            h: (0..n).map(|_| g.storage(rows * ff)).collect(),
            expert_out: (0..n).map(|_| g.storage(rows * d)).collect(),
        }
    }

    /// Expert `e`'s saved activations, as an [`ExpertScratch`] — the same
    /// value forward AND backward read for that expert (forward writes it
    /// during the layer's forward pass; backward reads it here unchanged).
    pub fn at(&self, e: usize) -> ExpertScratch<'_> {
        ExpertScratch { gate_pre: &self.gate_pre[e], up: &self.up[e], h: &self.h[e], expert_out: &self.expert_out[e] }
    }
}

/// Kernel indices [`expert_dgate`]/[`expert_bwd`] dispatch, resolved by the
/// calling model against its own registered pipeline list.
#[derive(Clone, Copy)]
pub struct MoeIdsBwd {
    pub scale_add_dexp: usize,
    pub scale_add_dgate: usize,
    pub silu_da: usize,
    pub silu_db: usize,
    /// `moe_linear_gated_dx.wgsl` when `linear_gated`, else `matmul_dx.wgsl`
    /// (or a tiled sibling with the same 4-buffer/4-param dense contract).
    pub linear_dx: usize,
    /// `moe_linear_gated_dw.wgsl` when `linear_gated`, else `matmul_dw.wgsl`
    /// (or a tiled sibling with the same 3-buffer/3-param dense contract).
    pub linear_dw: usize,
    /// Selects the gated kernels (5 bindings incl. `gate`, skip non-routed
    /// rows — bit-identical to the dense kernels over the same `dy`, since a
    /// non-routed row's `dy` is already exactly 0.0 there; see
    /// `moe_linear_gated_dx.wgsl`'s own doc) vs the dense ones (measure
    /// before committing either way, per `docs/kernel-checklist.md` §F).
    pub linear_gated: bool,
}

/// One expert's weight gradients — `None` skips that weight's dW entirely
/// (a frozen weight, or a caller that only wants dX for this expert).
#[derive(Clone, Copy)]
pub struct ExpertGrads<'a> {
    pub gate_w: Option<&'a DeviceBuffer>,
    pub up_w: Option<&'a DeviceBuffer>,
    pub down_w: Option<&'a DeviceBuffer>,
}

/// Scratch for one expert's backward pass, sized once by the caller and
/// reused across experts (fully overwritten each call).
pub struct ExpertBwdScratch<'a> {
    pub d_expert_out: &'a DeviceBuffer,
    pub d_h: &'a DeviceBuffer,
    pub d_gate_pre: &'a DeviceBuffer,
    pub d_up: &'a DeviceBuffer,
}

/// Phase A of an MoE layer's backward: gradient w.r.t. one expert's gate
/// weight (`scale_add_dgate.wgsl`), writing column `e_idx` of `d_gate` from
/// that expert's SAVED output (`saved.expert_out`, from [`MoeActs::at`]).
/// Must run for EVERY expert BEFORE [`router_bwd`] — the router needs the
/// WHOLE `d_gate` row, and this call only ever writes one column of it. See
/// [`moe_layer_bwd`]'s doc for the full ordering contract.
pub fn expert_dgate(g: &Gpu, ids: &MoeIdsBwd, shape: &MoeShape, saved: &ExpertScratch, d_acc: &DeviceBuffer, d_gate: &DeviceBuffer, e_idx: u32) -> Step {
    let (rows, d, e) = (shape.rows, shape.d_model, shape.n_experts);
    g.step(ids.scale_add_dgate, &[saved.expert_out, d_acc, d_gate], &[rows, d, e, e_idx], rows)
}

/// Phase C of an MoE layer's backward: one expert's SwiGLU backward,
/// accumulating dX into `d_x` (the shared input gradient every expert
/// contributes to). Must run AFTER [`router_bwd`] (see [`moe_layer_bwd`]'s
/// ordering contract) — mirrors `crates/moe/src/train.rs`'s Phase C loop and
/// `crates/glm/src/model.rs`'s per-expert MLP backward arm exactly (both
/// already gradient-checked); this hoists that sequence, it does not
/// re-derive it.
///
/// `accumulate` governs only the FIRST touch to `d_x` within this call (the
/// up-projection's dX write) — mirroring [`expert_fwd`]'s own `accumulate`
/// semantics exactly (`false` only when nothing has written `d_x` yet for
/// this row, e.g. a frozen/non-dispatched router). The gate-projection's dX
/// write that follows always accumulates on top of whatever the up
/// projection just wrote, since both happen within this one expert's call.
/// Every current caller passes `true` (the router's own backward, run first
/// by [`moe_layer_bwd`], already established `d_x`'s base value).
#[allow(clippy::too_many_arguments)]
pub fn expert_bwd(
    g: &Gpu,
    ids: &MoeIdsBwd,
    shape: &MoeShape,
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    gate_w: &DeviceBuffer,
    up_w: &DeviceBuffer,
    down_w: &DeviceBuffer,
    gr: &ExpertGrads,
    saved: &ExpertScratch,
    sb: &ExpertBwdScratch,
    d_acc: &DeviceBuffer,
    d_x: &DeviceBuffer,
    e_idx: u32,
    accumulate: bool,
    steps: &mut Vec<Step>,
) {
    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);

    let dx = |g: &Gpu, dy: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32, acc: u32, steps: &mut Vec<Step>| {
        if ids.linear_gated {
            steps.push(g.step(ids.linear_dx, &[dy, w, gate, out], &[m, k, n, e, e_idx, acc], m * k));
        } else {
            steps.push(g.step(ids.linear_dx, &[dy, w, out], &[m, k, n, acc], m * k));
        }
    };
    let dw = |g: &Gpu, dy: &DeviceBuffer, xin: &DeviceBuffer, dwbuf: &DeviceBuffer, k: u32, n: u32, steps: &mut Vec<Step>| {
        if ids.linear_gated {
            steps.push(g.step(ids.linear_dw, &[dy, xin, gate, dwbuf], &[m, k, n, e, e_idx], n * k));
        } else {
            steps.push(g.step(ids.linear_dw, &[dy, xin, dwbuf], &[m, k, n], n * k));
        }
    };

    // d_expert_out_e = gate[:,e] * d_moe_acc  (scale_add's own backward half 1)
    steps.push(g.step(ids.scale_add_dexp, &[gate, d_acc, sb.d_expert_out], &[m, d, e, e_idx], m * d));

    // down projection: dW (if trainable) + dX -> d_h (fresh per expert, accumulate=0)
    if let Some(w) = gr.down_w {
        dw(g, sb.d_expert_out, saved.h, w, ff, d, steps);
    }
    dx(g, sb.d_expert_out, down_w, sb.d_h, ff, d, 0, steps);

    // SwiGLU backward: d_h -> d_gate_pre, d_up (mirrors block::swiglu_bwd's
    // two-step body; not called directly to avoid pulling in block::KernelIds
    // for two lines this module already has its own ids for).
    steps.push(g.step(ids.silu_da, &[saved.gate_pre, saved.up, sb.d_h, sb.d_gate_pre], &[m * ff], m * ff));
    steps.push(g.step(ids.silu_db, &[saved.gate_pre, sb.d_h, sb.d_up], &[m * ff], m * ff));

    // up projection: dW (if trainable) + dX -> d_x (first touch: caller's accumulate)
    if let Some(w) = gr.up_w {
        dw(g, sb.d_up, x, w, d, ff, steps);
    }
    dx(g, sb.d_up, up_w, d_x, d, ff, accumulate as u32, steps);

    // gate projection: dW (if trainable) + dX -> d_x (always accumulates on
    // top of the up-projection's write above, within this same expert call)
    if let Some(w) = gr.gate_w {
        dw(g, sb.d_gate_pre, x, w, d, ff, steps);
    }
    dx(g, sb.d_gate_pre, gate_w, d_x, d, ff, 1, steps);
}

/// One MoE layer's full backward pass: Phase A (every expert's `d_gate`
/// column, [`expert_dgate`]) -> Phase B ([`router_bwd`]'s kernel-level
/// router backward, THEN the router weight's own dense-linear backward,
/// `router_weight_bwd` — caller-supplied since that GEMM's kernel/tiling
/// choice is not MoE-specific math this module should own, e.g.
/// `crates/glm`'s adaptive `pick_gemm` vs plain `MATMUL_DW`/`MATMUL_DX`) ->
/// Phase C (every expert's SwiGLU backward, [`expert_bwd`], accumulating
/// into `d_x`).
///
/// This exact order is REQUIRED, not incidental: [`router_bwd`] needs the
/// WHOLE `d_gate` row (every expert's column written first, Phase A), and
/// running Phase C before `router_weight_bwd` would accumulate expert
/// gradients into `d_x` before the router's own base value exists there
/// (`router_weight_bwd`'s dX write must be the FIRST touch to `d_x`,
/// accumulate=0). Every caller should use this wrapper rather than the
/// three phases' primitives directly, so the ordering cannot silently drift
/// per call site — the exact failure class `gradcheck`'s own doc warns
/// about: a partial gradient that a scalar check alone can pass by
/// coincidence (the T5 `rel_bias` case: a 33% error `directional_check`
/// alone reported as `rel_err = 6.2e-4`).
///
/// `router_weight_bwd`'s steps must reference the SAME `d_router_logits`
/// buffer [`router_bwd`] writes — `Step`s are plain dispatch descriptors
/// (kernel index + buffer handles + params), not lazily evaluated, so the
/// caller may build them before this call returns; only their POSITION in
/// the returned step list (after `router_bwd`'s steps, before Phase C)
/// matters, which this function fixes.
#[allow(clippy::too_many_arguments)]
pub fn moe_layer_bwd(
    g: &Gpu,
    router_bwd_ids: &RouterBwdIds,
    expert_bwd_ids: &MoeIdsBwd,
    kind: RouterKind,
    shape: &MoeShape,
    logits: &DeviceBuffer,
    gate: &DeviceBuffer,
    fe: Option<&DeviceBuffer>,
    d_gate: &DeviceBuffer,
    d_router_logits: &DeviceBuffer,
    router_weight_bwd: &[Step],
    x: &DeviceBuffer,
    expert_weights: &[(DeviceBuffer, DeviceBuffer, DeviceBuffer)],
    expert_grads: &[ExpertGrads],
    acts: &MoeActs,
    sb: &ExpertBwdScratch,
    d_moe_acc: &DeviceBuffer,
    d_x: &DeviceBuffer,
) -> Vec<Step> {
    assert_eq!(expert_weights.len(), shape.n_experts as usize, "moe_layer_bwd: expert_weights.len() must equal shape.n_experts");
    assert_eq!(expert_grads.len(), shape.n_experts as usize, "moe_layer_bwd: expert_grads.len() must equal shape.n_experts");

    let mut steps = Vec::new();

    // Phase A: every expert's d_gate column.
    for e_idx in 0..shape.n_experts as usize {
        steps.push(expert_dgate(g, expert_bwd_ids, shape, &acts.at(e_idx), d_moe_acc, d_gate, e_idx as u32));
    }

    // Phase B: router backward (kernel-level), then the router weight's own
    // dense-linear backward (caller-supplied -- see this fn's own doc).
    steps.extend(router_bwd(g, router_bwd_ids, kind, shape, logits, gate, d_gate, d_router_logits, fe));
    steps.extend_from_slice(router_weight_bwd);

    // Phase C: every expert's SwiGLU backward, accumulating into d_x (whose
    // base value router_weight_bwd's own dX write already established above).
    for (e_idx, (gate_w, up_w, down_w)) in expert_weights.iter().enumerate() {
        expert_bwd(g, expert_bwd_ids, shape, x, gate, gate_w, up_w, down_w, &expert_grads[e_idx], &acts.at(e_idx), sb, d_moe_acc, d_x, e_idx as u32, true, &mut steps);
    }

    steps
}

/// int8 kernel indices, parallel to [`MoeIds`]. `max_abs_row`/`quant_pack` are
/// `model::int8`'s shared dynamic-activation-quantization pair (the same ones
/// every DP4A model in the engine uses), needed here because `h` (the
/// post-SiLU hidden, the down-projection's input) is expert-specific and so
/// cannot be quantized once per layer the way the shared input `x` can.
#[derive(Clone, Copy)]
pub struct MoeIds8 {
    /// `moe_linear_gated_i8.wgsl`.
    pub linear_gated_i8: usize,
    pub silu_mul: usize,
    pub scale_add: usize,
    /// `crate::int8::quant_rows_steps`'s `[max_abs_row, quant_pack]` pair.
    pub quant: [usize; 2],
}

/// One int8-quantized expert linear's weight, in `crate::int8::quantize_weight`'s
/// packed layout: `wq` is `[n, k/4]` u32, `sw` is `[n]` f32 per-channel scale.
#[derive(Clone, Copy)]
pub struct Lin8<'a> {
    pub wq: &'a DeviceBuffer,
    pub sw: &'a DeviceBuffer,
}

/// Scratch for one expert's int8 FFN step. `gate_pre`/`up`/`h` stay fp32 (the
/// activation functions and quantization math are fp32 arithmetic, per the
/// engine-wide invariant — only storage is int8); `hq`/`sh` are `h` quantized
/// fresh each call, since `h` is a different tensor for every expert.
pub struct ExpertScratch8<'a> {
    pub gate_pre: &'a DeviceBuffer,
    pub up: &'a DeviceBuffer,
    pub h: &'a DeviceBuffer,
    pub hq: &'a DeviceBuffer,
    pub sh: &'a DeviceBuffer,
    pub expert_out: &'a DeviceBuffer,
}

/// int8 counterpart of [`expert_fwd`]. `xq`/`sx` are the shared input `x`,
/// ALREADY quantized once by the caller (via `crate::int8::quant_rows_steps`)
/// before the expert loop starts — every expert reads the same quantized
/// activation, so quantizing it 128 times would be pure waste.
#[allow(clippy::too_many_arguments)]
pub fn expert_fwd_i8(
    g: &Gpu,
    ids: &MoeIds8,
    shape: &MoeShape,
    xq: &DeviceBuffer,
    sx: &DeviceBuffer,
    gate: &DeviceBuffer,
    gate_w: Lin8,
    up_w: Lin8,
    down_w: Lin8,
    scratch: &ExpertScratch8,
    acc: &DeviceBuffer,
    e_idx: u32,
    accumulate: bool,
) -> Vec<Step> {
    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);
    let lin = |xq: &DeviceBuffer, sx: &DeviceBuffer, w: Lin8, out: &DeviceBuffer, kg: u32, n: u32| {
        g.step(ids.linear_gated_i8, &[xq, w.wq, sx, w.sw, gate, out], &[m, kg, n, e, e_idx], m * n)
    };
    let quant_h = crate::int8::quant_rows_steps(
        g,
        crate::int8::QuantRows { kernels: ids.quant, x: scratch.h, sx: scratch.sh, xq: scratch.hq },
        0,
        m,
        ff,
    );
    let mut steps = vec![
        lin(xq, sx, gate_w, scratch.gate_pre, d / 4, ff),
        lin(xq, sx, up_w, scratch.up, d / 4, ff),
        g.step(ids.silu_mul, &[scratch.gate_pre, scratch.up, scratch.h], &[m * ff], m * ff),
    ];
    steps.extend(quant_h);
    steps.push(lin(scratch.hq, scratch.sh, down_w, scratch.expert_out, ff / 4, d));
    steps.push(g.step(ids.scale_add, &[gate, scratch.expert_out, acc], &[m, d, e, e_idx, accumulate as u32], m * d));
    steps
}

/// Kernel indices for [`shared_expert_fwd`] -- an always-active (non-gated)
/// dense SwiGLU, so it dispatches the plain `matmul.wgsl` rather than
/// [`MoeIds::linear_gated`]'s row-skipping variant.
#[derive(Clone, Copy)]
pub struct SharedExpertIds {
    pub matmul: usize,
    pub silu_mul: usize,
    /// `sigmoid.wgsl`.
    pub sigmoid: usize,
    /// `scale_row.wgsl` (`y[i] = s[i / m] * x[i]`).
    pub scale_row: usize,
    pub add2: usize,
}

/// Scratch for [`shared_expert_fwd`]'s dense SwiGLU + sigmoid gate, sized
/// once by the caller. `shared_ff` (the shared expert's own intermediate
/// width) is generally different from the routed experts' `moe_ff` -- e.g.
/// Qwen3-Omni's Talker: 768 vs 384 -- so these buffers are NOT the same size
/// as [`ExpertScratch`]'s and must not be reused across the two.
pub struct SharedExpertScratch<'a> {
    pub gate_pre: &'a DeviceBuffer,    // [rows, shared_ff]
    pub up: &'a DeviceBuffer,          // [rows, shared_ff]
    pub h: &'a DeviceBuffer,           // [rows, shared_ff]
    pub mlp_out: &'a DeviceBuffer,     // [rows, d_model]
    pub gate_logits: &'a DeviceBuffer, // [rows, 1]
    pub gate_scalar: &'a DeviceBuffer, // [rows, 1]
    pub scaled: &'a DeviceBuffer,      // [rows, d_model]
}

/// An MoE block's "always-active" shared expert: a dense SwiGLU MLP (own
/// gate/up/down weights, own intermediate width `shared_ff`) applied to
/// EVERY row (no top-k gating), added to `acc` (the routed [`expert_fwd`]
/// loop's output) into a fresh `out` buffer -- never in place on `acc`,
/// matching `add2.wgsl`'s own out-of-place convention.
///
/// `shared_gate_w` selects which of two real architectures this is:
/// - `Some(w)`: scaled per-row by `sigmoid(x @ w^T)` first. Matches HF
///   `Qwen3OmniMoeTalkerTextSparseMoeBlock.forward`'s `expert_output +
///   sigmoid(shared_expert_gate(x)) * shared_expert(x)` exactly (the routed
///   and shared paths read the SAME `x`, not the shared path reading the
///   routed output) -- Qwen3-Omni's Talker.
/// - `None`: added UNWEIGHTED, no gate at all. Matches `crates/glm`'s shared
///   expert (`model.rs:794`, GLM-5.2/DeepSeek-V3's architecture) exactly --
///   a distinct real design, not a degenerate case of the gated one.
///
/// FORWARD-ONLY (audit F18): there is no `shared_expert_bwd` — the GLM/Omni
/// MoE trainers will need the adjoint of this exact composition (dense
/// SwiGLU backward + the sigmoid-gate product rule for the `Some` arm) and
/// it does not exist yet. Implement it WITH its gradcheck when training
/// first needs it; do not assume the routed experts' backward covers it.
#[allow(clippy::too_many_arguments)]
pub fn shared_expert_fwd(
    g: &Gpu,
    ids: &SharedExpertIds,
    rows: u32,
    d_model: u32,
    shared_ff: u32,
    x: &DeviceBuffer,
    gate_w: &DeviceBuffer,
    up_w: &DeviceBuffer,
    down_w: &DeviceBuffer,
    shared_gate_w: Option<&DeviceBuffer>,
    scratch: &SharedExpertScratch,
    acc: &DeviceBuffer,
    out: &DeviceBuffer,
) -> Vec<Step> {
    let mut steps = vec![
        g.step(ids.matmul, &[x, gate_w, scratch.gate_pre], &[rows, d_model, shared_ff], rows * shared_ff),
        g.step(ids.matmul, &[x, up_w, scratch.up], &[rows, d_model, shared_ff], rows * shared_ff),
        g.step(ids.silu_mul, &[scratch.gate_pre, scratch.up, scratch.h], &[rows * shared_ff], rows * shared_ff),
        g.step(ids.matmul, &[scratch.h, down_w, scratch.mlp_out], &[rows, shared_ff, d_model], rows * d_model),
    ];
    match shared_gate_w {
        Some(shared_gate_w) => {
            steps.push(g.step(ids.matmul, &[x, shared_gate_w, scratch.gate_logits], &[rows, d_model, 1], rows));
            steps.push(g.step(ids.sigmoid, &[scratch.gate_logits, scratch.gate_scalar], &[rows], rows));
            steps.push(g.step(ids.scale_row, &[scratch.mlp_out, scratch.gate_scalar, scratch.scaled], &[rows * d_model, d_model], rows * d_model));
            steps.push(g.step(ids.add2, &[acc, scratch.scaled, out], &[rows * d_model], rows * d_model));
        }
        None => {
            steps.push(g.step(ids.add2, &[acc, scratch.mlp_out, out], &[rows * d_model], rows * d_model));
        }
    }
    steps
}

// ---- row-compacted sparse expert forward (tiled, not naive) ---------------
//
// `expert_fwd` above removes the redundant FLOPs of evaluating every expert
// densely, but stays naive-tier (one thread per output element, no tiling) --
// measured 6.51x SLOWER than GLM's existing dense TILED path at GLM-5.2's
// real shape (`docs/models/glm/status.md`, `crates/glm/examples/
// moe_migration_bench.rs`), because the naive kernel's per-FLOP inefficiency
// at ~64 rows/expert swamps the 32x FLOP-count win sparsity promises. This
// section is the real fix: gather each expert's routed rows into a dense
// sub-batch, run the SAME
// `model::block::pick_gemm`-selected tiled GEMM the dense path already uses
// (unchanged, no new GEMM kernel), scatter the scaled result back.
//
// KNOWN COST (audit F9, deliberate remainder): the per-expert call shape
// below does one host scan over `host_gate`, one index upload, and one
// `submit` PER EXPERT — at GLM-5.2 scale (~128 experts x ~48 MoE layers)
// that is ~6100 submits and small uploads per forward. Every per-expert
// count is knowable host-side from ONE pass over `host_gate`, and dispatch
// ordering within a submit already guarantees the shared scratch is safe to
// reuse across experts in a single batched submission — so a layer-level
// entry point (bucket rows for ALL experts in one pass, per-expert index
// regions, one submit per layer) removes the storm with no new kernel. That
// API change lands WITH its call-site migration (crates/glm, crates/omni) so
// the faster sibling is never an unconsumed second path (docs/lessons.md #8);
// this crate alone cannot do it without stranding the current callers.

/// Kernel indices [`expert_fwd_compact`] dispatches, resolved by the calling
/// model against its own registered pipeline list.
#[derive(Clone, Copy)]
pub struct CompactExpertFwdIds {
    /// `embed.wgsl`, reused unchanged as a generic row-gather-by-index
    /// kernel: `x_compact[i, :] = x[idx[i], :]` is exactly
    /// `emb_row[t, :] = emb[token[t], :]` with `idx` standing in for
    /// `token`.
    pub gather: usize,
    /// The naive reference GEMM (`matmul.wgsl`'s `{x,w,out}`/`{m,k,n}`
    /// contract) -- [`model::block::pick_gemm`] falls back to this for a
    /// compacted batch too small to fill a tile (a lightly-routed expert).
    pub gemm_naive: usize,
    /// The tiled GEMM (`matmul_reg3.wgsl`, SAME contract as `gemm_naive`) --
    /// `pick_gemm` selects this once the compacted row count clears its
    /// tiling threshold.
    pub gemm_tiled: usize,
    pub silu_mul: usize,
    /// `moe_scatter_scaled_add.wgsl`.
    pub scatter: usize,
}

/// Per-expert scratch for [`expert_fwd_compact`], sized ONCE by the caller
/// for `capacity` rows and reused across every expert in a layer (fully
/// overwritten each call — the tail past this call's `count` is simply
/// unused, never read, so leftover data from a larger previous expert is
/// harmless). `capacity_for` returns the only value that makes
/// [`expert_fwd_compact`]'s capacity panic unreachable: every row could in
/// principle route to one expert.
pub struct CompactExpertScratch {
    capacity: u32,
    idx: DeviceBuffer,
    x_compact: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    expert_out: DeviceBuffer,
}

impl CompactExpertScratch {
    /// `capacity` rows' worth of scratch — pass `shape.rows` for the
    /// unconditionally-safe bound, or a smaller measured high-watermark if
    /// the caller already knows routing is more balanced than that.
    pub fn new(g: &Gpu, shape: &MoeShape, capacity: u32) -> CompactExpertScratch {
        let (d, ff) = (shape.d_model as u64, shape.moe_ff as u64);
        let cap = capacity as u64;
        CompactExpertScratch {
            capacity,
            idx: g.storage(cap),
            x_compact: g.storage(cap * d),
            gate_pre: g.storage(cap * ff),
            up: g.storage(cap * ff),
            h: g.storage(cap * ff),
            expert_out: g.storage(cap * d),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

/// Row-compacted sparse expert forward — see this section's header doc.
///
/// `host_gate` MUST be the CURRENT contents of `gate` (`[shape.rows,
/// shape.n_experts]`), already read back by the caller (`Gpu::read`). This
/// function does NOT perform that readback itself, so the caller controls
/// exactly when the host/device round trip happens — once per LAYER (read
/// `gate` once, call this once per expert against the same `host_gate`), not
/// once per expert. Consequently this is NOT a pure step-builder like
/// [`expert_fwd`]: each expert's compacted row count is a value only the
/// HOST can know (discovered from `host_gate`), so this function builds AND
/// submits its own steps rather than returning a `Vec<Step>` for the caller
/// to batch — every other step-list builder in this engine always knows its
/// dispatch shapes before building a `Step`, which a data-dependent row
/// count breaks by construction. This is therefore a real synchronisation
/// point per expert (the `submit` below), acceptable for a BATCHED forward
/// (training/eval), not intended for a per-token decode loop.
///
/// Panics if expert `e_idx` routes more rows than `scratch`'s capacity —
/// size `scratch` via [`CompactExpertScratch::new`] with `shape.rows` to
/// make this impossible, rather than silently truncating which rows this
/// expert's output reaches.
///
/// Returns how many rows were routed to this expert, for the caller's own
/// diagnostics (e.g. confirming a benchmark's routing landed near its
/// expected average).
#[allow(clippy::too_many_arguments)]
pub fn expert_fwd_compact(
    g: &Gpu,
    ids: &CompactExpertFwdIds,
    shape: &MoeShape,
    host_gate: &[f32],
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    gate_w: &DeviceBuffer,
    up_w: &DeviceBuffer,
    down_w: &DeviceBuffer,
    scratch: &CompactExpertScratch,
    acc: &DeviceBuffer,
    e_idx: u32,
    accumulate: bool,
) -> usize {
    let (d, ff, e) = (shape.d_model, shape.moe_ff, shape.n_experts);
    // A short readback here previously panicked with a bare index message in
    // the filter below, while the capacity precondition next door got a
    // detailed assert — same rigor for both preconditions.
    assert_eq!(
        host_gate.len(),
        (shape.rows * e) as usize,
        "expert_fwd_compact: host_gate has {} elements, want rows*n_experts = {}x{} -- \
         pass the CURRENT full readback of `gate`",
        host_gate.len(),
        shape.rows,
        e
    );
    let rows: Vec<u32> = (0..shape.rows).filter(|&r| host_gate[(r * e + e_idx) as usize] > 0.0).collect();
    let count = rows.len() as u32;
    if count == 0 {
        // Nothing routed to this expert this call — still honour
        // `accumulate`'s set-vs-add contract: a zero-row expert must not
        // silently skip zeroing `acc` if it happens to be the first expert
        // in the layer's loop.
        if !accumulate {
            g.submit(&[acc], &[]);
        }
        return 0;
    }
    assert!(
        count <= scratch.capacity,
        "expert_fwd_compact: expert {e_idx} routed {count} rows, exceeding scratch capacity {} -- \
         size CompactExpertScratch::new with shape.rows to make this impossible",
        scratch.capacity
    );
    g.write(&scratch.idx, &rows);
    let lin = |x_in: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32| {
        let (kid, threads) = crate::block::pick_gemm(count as usize, n as usize, ids.gemm_naive, ids.gemm_tiled, false);
        g.step(kid, &[x_in, w, out], &[count, k, n], threads)
    };
    let steps = vec![
        g.step(ids.gather, &[&scratch.idx, x, &scratch.x_compact], &[d, count], count * d),
        lin(&scratch.x_compact, gate_w, &scratch.gate_pre, d, ff),
        lin(&scratch.x_compact, up_w, &scratch.up, d, ff),
        g.step(ids.silu_mul, &[&scratch.gate_pre, &scratch.up, &scratch.h], &[count * ff], count * ff),
        lin(&scratch.h, down_w, &scratch.expert_out, ff, d),
        g.step(
            ids.scatter,
            &[&scratch.idx, gate, &scratch.expert_out, acc],
            &[count, d, e, e_idx, accumulate as u32],
            count * d,
        ),
    ];
    g.submit(&[], &steps);
    count as usize
}
