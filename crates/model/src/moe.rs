// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse top-k MoE expert FFN - router weights in, combined output out,
//! without evaluating every expert densely.
//!
//! `crates/glm` (today's only HF-importable MoE forward) evaluates **every**
//! expert over the **whole** row batch and discards non-selected rows by
//! multiplying by a zero gate weight afterward (`Mlp::Moe` in
//! `crates/glm/src/model.rs`, combining with `scale_add.wgsl`) - numerically
//! exact (`router_gate.wgsl`'s own doc comment proves it), but `n_experts`
//! times the FLOPs of an actual top-k dispatch. At 128 experts / top-8
//! (Qwen3-Omni's Thinker) that is sixteen times the necessary work, which
//! motivated this module.
//!
//! The fix here is deliberately the smallest one that removes the FLOPs
//! without adding new failure modes: [`moe_linear_gated`] is `matmul.wgsl`
//! with one extra check - a row whose gate weight for this expert is zero
//! writes 0 and returns *before* the K-reduction, instead of computing it and
//! discarding it downstream. Composing three of those (gate/up/down
//! projection) plus the existing `silu_mul`/`scale_add` kernels reproduces
//! [`expert_fwd`]'s per-expert step exactly, at a cost proportional to the
//! number of rows actually routed to that expert.
//!
//! [`expert_fwd_i8`] is the same trick over `model::int8`'s packed weights,
//! via a NEW naive (non-tiled) DP4A kernel rather than gating the existing
//! `matmul_i8_dyn`/`matmul_i8_gemv` - see `moe_linear_gated_i8.wgsl`'s doc for
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
//! *launches*), no TILED int8 GEMM tier (both int8 and fp32 are naive-dispatch today; a
//! future tiled+gated kernel is one change, not two, once compaction lands).
//!
//! **Backward** (this session's addition) is a hoist, not a from-scratch
//! derivation: two complete, gradient-checked MoE backwards already existed -
//! `crates/moe/src/train.rs`'s softmax-router training loop and
//! `crates/glm/src/model.rs`'s sigmoid `noaux_tc` router MLP arm - and
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
    /// `router_gate.wgsl` (or `router_gate_train.wgsl` if probs are wanted) -
    /// softmax -> top-k -> renormalise into a dense `[rows, n_experts]` gate.
    pub router_gate: usize,
    /// `moe_linear_gated.wgsl` - `matmul.wgsl` with a per-row gate early-exit.
    pub linear_gated: usize,
    /// `silu_mul.wgsl` - shared with every other SwiGLU MLP in the engine.
    pub silu_mul: usize,
    /// `scale_add.wgsl` - the same combine step `crates/glm` already uses.
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
/// (dense, nonzero only at the `top_k` selected experts). Plain softmax top-k -
/// Qwen3-Omni's Thinker and Talker routers both use this (no aux-loss-free
/// sigmoid/bias/group-limiting; that variant is `router_gate_sigmoid.wgsl`,
/// already in use by `crates/glm`).
///
/// `norm_topk_prob` renormalises the selected probabilities to sum to 1 per row
/// (Switch/Mixtral, and what every caller in this workspace wants);
/// `routed_scaling` is DeepSeek's `routed_scaling_factor` applied on top
/// (`1.0` = none). Both are spelled at every call site rather than defaulted:
/// the pair is exactly the tail `RouterKind::SigmoidNoAuxTc` already carries,
/// and a forward that silently defaults one of them is a gradient the backward
/// cannot check - `router_bwd` must be handed the SAME two values.
pub fn router_fwd(
    g: &Gpu,
    ids: &MoeIds,
    shape: &MoeShape,
    logits: &DeviceBuffer,
    gate: &DeviceBuffer,
    norm_topk_prob: bool,
    routed_scaling: f32,
) -> Step {
    g.step(
        ids.router_gate,
        &[logits, gate],
        &[shape.rows, shape.n_experts, shape.top_k, norm_topk_prob as u32, gpu_core::f(routed_scaling)],
        shape.rows,
    )
}

/// Which router computes an MoE layer's per-token expert weights. The two
/// variants existing kernels support today - parameterising the difference
/// here means one router entry point ([`router_fwd_kind`]/[`router_bwd`]),
/// not two hand-duplicated implementations (confirmed by comparing
/// `crates/glm/src/model.rs:915-935` against `crates/moe/src/train.rs:466-486`
/// line for line: the expert halves are identical, only the router half
/// differs).
#[derive(Clone, Copy)]
pub enum RouterKind {
    /// `router_gate.wgsl` (fwd) + `router_bwd.wgsl` (bwd) - BOTH array-free
    /// (no expert-count cap): the backward since the #35-recurrence fix, the
    /// forward since the audit-F4 rewrite (its `array<f32,128>` scratch was
    /// the same silent-OOB literal one expert higher). Qwen3-Omni's
    /// Thinker/Talker.
    ///
    /// `norm_topk_prob`/`routed_scaling` are the SAME two knobs
    /// [`RouterKind::SigmoidNoAuxTc`] carries, now expressed on this family
    /// too: `true`/`1.0` is Switch/Mixtral (and what every caller in this
    /// workspace uses today), `false` keeps the RAW top-k softmax
    /// probabilities as combine weights. The backward has a genuinely
    /// different (simpler) form for `false` - see `router_bwd.wgsl`'s header.
    Softmax { aux_coef: f32, z_coef: f32, norm_topk_prob: bool, routed_scaling: f32 },
    /// `router_gate_sigmoid.wgsl` (fwd) + `router_bwd_sigmoid.wgsl` (bwd,
    /// already unbounded). `crates/glm`'s GLM-5.2/DeepSeek-V3 "noaux_tc"
    /// router: per-expert selection bias, group-limited top-k, optional
    /// renormalisation. `router_gate_sigmoid.wgsl`'s forward hard-caps at 64
    /// experts (`MAX_E`, fixed-size array scratch) - [`router_fwd_kind`]
    /// asserts this loudly rather than let it silently corrupt (the same
    /// stopgap `crates/glm/src/model.rs::new_impl_on` already applies; a
    /// real array-free top-k rewrite is separate kernel work, not a literal
    /// bump - see that assert's own doc for why).
    SigmoidNoAuxTc { n_group: u32, topk_group: u32, norm_topk_prob: bool, routed_scaling: f32 },
}

/// Router forward, dispatching whichever kernel `kind` selects. `bias`/
/// `probs` are REQUIRED (`Some`) for `SigmoidNoAuxTc` - `router_gate_sigmoid
/// .wgsl`'s mandatory 3rd/5th bindings (`probs` is written but never read
/// back by `router_bwd_sigmoid.wgsl`, which recomputes `sigmoid(logits)`
/// inline rather than reading a saved probability; the buffer must still
/// exist because the shader's own interface requires it) - and unused
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
        RouterKind::Softmax { norm_topk_prob, routed_scaling, .. } => router_fwd(g, ids, shape, logits, gate, norm_topk_prob, routed_scaling),
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
    /// `expert_counts.wgsl` - Softmax's aux-loss load-balancing fractions.
    /// `None` for SigmoidNoAuxTc: DeepSeek-V3's selection bias is a
    /// forward-only load-balancing heuristic, never backprop'd (matches
    /// `crates/glm/src/model.rs`'s own note keeping `moe.router.bias`
    /// `Role::Frozen`, out of the optimiser).
    pub expert_counts: Option<usize>,
}

/// Router backward: gradient w.r.t. the router logits, dispatching whichever
/// kernel `kind` selects. Returns the full step list - `Softmax` needs an
/// extra `expert_counts` dispatch first (its aux-loss term needs per-expert
/// usage fractions); `SigmoidNoAuxTc` does not (no aux loss).
///
/// Does NOT touch `d_x` or the router weight's own gradient - those are an
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
        RouterKind::Softmax { aux_coef, z_coef, norm_topk_prob, routed_scaling } => {
            let fe = fe.expect("RouterKind::Softmax requires an fe (expert-usage) scratch buffer for the aux-loss term");
            let ec = ids.expert_counts.expect("RouterKind::Softmax requires RouterBwdIds::expert_counts");
            vec![
                g.step(ec, &[gate, fe], &[rows, e, top_k], e),
                g.step(
                    ids.router_bwd,
                    &[logits, gate, d_gate, fe, dlogits],
                    &[rows, e, top_k, 0, gpu_core::f(aux_coef), gpu_core::f(z_coef), norm_topk_prob as u32, gpu_core::f(routed_scaling)],
                    rows,
                ),
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

/// One expert's gated SwiGLU FFN step, combined into `acc` - the sparse
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

/// Saved per-expert activations for backward - exactly [`ExpertScratch`]'s
/// four tensors, but OWNED per-expert rather than one shared scratch set
/// reused across experts. [`ExpertScratch`]'s forward-time contract (fully
/// overwritten each call, safe to alias across experts) does not hold for
/// training: backward needs EVERY expert's own `gate_pre`/`up`/`h`/
/// `expert_out`, not just whichever expert ran last. Allocates exactly what
/// `crates/glm/src/model.rs`'s `Mlp::Moe` already allocates per layer
/// (`model.rs:463-466`) - memory-neutral, not a reduction. Recompute-in-
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

    /// Expert `e`'s saved activations, as an [`ExpertScratch`] - the same
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
    /// rows - bit-identical to the dense kernels over the same `dy`, since a
    /// non-routed row's `dy` is already exactly 0.0 there; see
    /// `moe_linear_gated_dx.wgsl`'s own doc) vs the dense ones (measure
    /// before committing either way).
    pub linear_gated: bool,
}

/// One expert's weight gradients - `None` skips that weight's dW entirely
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
/// Must run for EVERY expert BEFORE [`router_bwd`] - the router needs the
/// WHOLE `d_gate` row, and this call only ever writes one column of it. See
/// [`moe_layer_bwd`]'s doc for the full ordering contract.
pub fn expert_dgate(g: &Gpu, ids: &MoeIdsBwd, shape: &MoeShape, saved: &ExpertScratch, d_acc: &DeviceBuffer, d_gate: &DeviceBuffer, e_idx: u32) -> Step {
    let (rows, d, e) = (shape.rows, shape.d_model, shape.n_experts);
    g.step(ids.scale_add_dgate, &[saved.expert_out, d_acc, d_gate], &[rows, d, e, e_idx], rows)
}

/// Phase C of an MoE layer's backward: one expert's SwiGLU backward,
/// accumulating dX into `d_x` (the shared input gradient every expert
/// contributes to). Must run AFTER [`router_bwd`] (see [`moe_layer_bwd`]'s
/// ordering contract) - mirrors `crates/moe/src/train.rs`'s Phase C loop and
/// `crates/glm/src/model.rs`'s per-expert MLP backward arm exactly (both
/// already gradient-checked); this hoists that sequence, it does not
/// re-derive it.
///
/// `accumulate` governs only the FIRST touch to `d_x` within this call (the
/// up-projection's dX write) - mirroring [`expert_fwd`]'s own `accumulate`
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
/// `router_weight_bwd` - caller-supplied since that GEMM's kernel/tiling
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
/// per call site - the exact failure class `gradcheck`'s own doc warns
/// about: a partial gradient that a scalar check alone can pass by
/// coincidence (the T5 `rel_bias` case: an error of a third of the value,
/// which `directional_check` alone reported as `rel_err = 6.2e-4`).
///
/// `router_weight_bwd`'s steps must reference the SAME `d_router_logits`
/// buffer [`router_bwd`] writes - `Step`s are plain dispatch descriptors
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
/// packed layout: `wq` is `[n, k/4]` u32, `sw` is `[n, k/32]` f32 group scale
/// (`crate::int8::GROUP`).
#[derive(Clone, Copy)]
pub struct Lin8<'a> {
    pub wq: &'a DeviceBuffer,
    pub sw: &'a DeviceBuffer,
}

/// Scratch for one expert's int8 FFN step. `gate_pre`/`up`/`h` stay fp32 (the
/// activation functions and quantization math are fp32 arithmetic, per the
/// engine-wide invariant - only storage is int8); `hq`/`sh` are `h` quantized
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
/// before the expert loop starts - every expert reads the same quantized
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
/// Its adjoint is [`shared_expert_bwd`], which covers BOTH arms.
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

/// Kernel indices [`shared_expert_bwd`] dispatches, resolved by the calling
/// model against its own registered pipeline list. The last three are needed
/// ONLY by the sigmoid-gated (`Some`) arm; the unweighted arm never reads them
/// and may leave them at any value.
#[derive(Clone, Copy)]
pub struct SharedExpertBwdIds {
    /// `matmul_dx.wgsl` (`{d_out, w, dx}` / `{m, k, n, acc}`).
    pub linear_dx: usize,
    /// `matmul_dw.wgsl` (`{d_out, x, dw}` / `{m, k, n}`).
    pub linear_dw: usize,
    pub silu_da: usize,
    pub silu_db: usize,
    /// `scale_row.wgsl` - its OWN backward w.r.t. `x` (`dx = scale_row(dy, s)`;
    /// see that kernel's header). Gated arm only.
    pub scale_row: usize,
    /// `row_dot.wgsl` - the row-scale factor's gradient
    /// `ds[row] = sum_d dy[row,d]*x[row,d]`. Gated arm only.
    pub row_dot: usize,
    /// `sigmoid_bwd.wgsl`. Gated arm only.
    pub sigmoid_bwd: usize,
}

/// The forward activations [`shared_expert_bwd`] reads back.
///
/// A SUBSET of [`SharedExpertScratch`], not a copy of it: the backward never
/// reads the forward's `scaled` buffer, and the three gated-arm tensors are
/// `Option` because the unweighted arm's forward never produces them (
/// `crates/glm` allocates no sigmoid-gate buffers at all). Build it from a
/// forward scratch with [`SharedExpertScratch::acts`], or by hand.
#[derive(Clone, Copy)]
pub struct SharedExpertActs<'a> {
    pub gate_pre: &'a DeviceBuffer, // [rows, shared_ff]
    pub up: &'a DeviceBuffer,       // [rows, shared_ff]
    pub h: &'a DeviceBuffer,        // [rows, shared_ff]
    /// The SwiGLU's output BEFORE the sigmoid scale. Gated arm only.
    pub mlp_out: Option<&'a DeviceBuffer>, // [rows, d_model]
    pub gate_logits: Option<&'a DeviceBuffer>, // [rows]
    pub gate_scalar: Option<&'a DeviceBuffer>, // [rows]
}

impl<'a> SharedExpertScratch<'a> {
    /// The subset of this forward scratch [`shared_expert_bwd`] reads. Always
    /// fills the gated arm's three `Option`s - a caller that ran the forward
    /// through this struct ran the gate projection into these buffers, so they
    /// are valid whenever the forward was called with `Some(shared_gate_w)`
    /// and simply unread otherwise.
    pub fn acts(&self) -> SharedExpertActs<'a> {
        SharedExpertActs {
            gate_pre: self.gate_pre,
            up: self.up,
            h: self.h,
            mlp_out: Some(self.mlp_out),
            gate_logits: Some(self.gate_logits),
            gate_scalar: Some(self.gate_scalar),
        }
    }
}

/// The shared expert's weight gradients - `None` skips that weight's dW
/// entirely (a frozen weight under a LoRA build, or a caller that only wants
/// dX). Same contract as [`ExpertGrads`], plus the sigmoid-gate projection.
#[derive(Clone, Copy, Default)]
pub struct SharedExpertGrads<'a> {
    pub gate_w: Option<&'a DeviceBuffer>,
    pub up_w: Option<&'a DeviceBuffer>,
    pub down_w: Option<&'a DeviceBuffer>,
    /// The `d_model -> 1` sigmoid-gate projection's dW. Gated arm only.
    pub shared_gate_w: Option<&'a DeviceBuffer>,
}

/// Scratch for [`shared_expert_bwd`], sized once by the caller. The last three
/// are required by the gated (`Some`) arm and unused by the unweighted one,
/// where `d_out` IS the gradient w.r.t. the SwiGLU's output.
pub struct SharedExpertBwdScratch<'a> {
    pub d_h: &'a DeviceBuffer,          // [rows, shared_ff]
    pub d_gate_pre: &'a DeviceBuffer,   // [rows, shared_ff]
    pub d_up: &'a DeviceBuffer,         // [rows, shared_ff]
    pub d_mlp_out: Option<&'a DeviceBuffer>,    // [rows, d_model]
    pub d_gate_scalar: Option<&'a DeviceBuffer>, // [rows]
    pub d_gate_logits: Option<&'a DeviceBuffer>, // [rows]
}

/// Adjoint of [`shared_expert_fwd`], for BOTH of its arms.
///
/// `d_out` is the gradient w.r.t. that function's `out`. Because the forward's
/// last step is `out = acc + (gated) mlp_out`, `d_out` is ALSO the gradient
/// w.r.t. `acc` unchanged - the routed experts' own backward consumes the same
/// buffer, and no kernel is needed to split it. That is why this function takes
/// no `d_acc` output.
///
/// `shared_gate_w` selects the arm, exactly as in the forward:
/// - `None` - unweighted. This is the composition `crates/glm` hand-wrote
///   inline (`Mlp::Moe`'s shared-expert block) and now calls; the step sequence
///   below IS that code, moved, not re-derived.
/// - `Some(w)` - per-row `sigmoid(x @ wᵀ)` gate. `scale_row` is its own
///   backward w.r.t. `x`, `row_dot` gives the row-scale factor's gradient, and
///   `sigmoid_bwd` closes it. `crates/qwen35moe`'s `moe_sublayer_bwd` derived
///   this arm first; it is reproduced here so both real architectures have one
///   home.
///
/// `accumulate` governs only the FIRST touch to `d_x` (whichever step that is
/// in the selected arm), mirroring [`expert_bwd`]'s own semantics: `false` when
/// nothing has written `d_x` yet, `true` when the routed-MoE backward already
/// established its base value. Every later `d_x` write in this call accumulates.
///
/// ORDERING: this must run AFTER the routed [`moe_layer_bwd`] when both share
/// `d_x`, since that phase owns `d_x`'s first write in every current caller.
#[allow(clippy::too_many_arguments)]
pub fn shared_expert_bwd(
    g: &Gpu,
    ids: &SharedExpertBwdIds,
    rows: u32,
    d_model: u32,
    shared_ff: u32,
    x: &DeviceBuffer,
    gate_w: &DeviceBuffer,
    up_w: &DeviceBuffer,
    down_w: &DeviceBuffer,
    shared_gate_w: Option<&DeviceBuffer>,
    gr: &SharedExpertGrads,
    saved: &SharedExpertActs,
    sb: &SharedExpertBwdScratch,
    d_out: &DeviceBuffer,
    d_x: &DeviceBuffer,
    accumulate: bool,
) -> Vec<Step> {
    let (m, d, ff) = (rows, d_model, shared_ff);
    let mut steps: Vec<Step> = Vec::new();
    // Backward of `y = x·Wᵀ`: weight grad (when trainable) then input grad -
    // the exact pair `crates/glm`'s `mm_bwd` emits, in that order.
    let mm_bwd = |steps: &mut Vec<Step>, d_y: &DeviceBuffer, xin: &DeviceBuffer, w: &DeviceBuffer, dw: Option<&DeviceBuffer>, dx: &DeviceBuffer, k: u32, n: u32, acc: u32| {
        if let Some(dw) = dw {
            steps.push(g.step(ids.linear_dw, &[d_y, xin, dw], &[m, k, n], n * k));
        }
        steps.push(g.step(ids.linear_dx, &[d_y, w, dx], &[m, k, n, acc], m * k));
    };

    // Which gradient the SwiGLU's down-projection sees, and whether `d_x`'s
    // first touch has already happened by the time the SwiGLU runs.
    let (d_mlp_out, first_touch) = match shared_gate_w {
        None => (d_out, accumulate as u32),
        Some(sgw) => {
            let d_mlp_out = sb.d_mlp_out.expect("shared_expert_bwd: the gated arm needs SharedExpertBwdScratch::d_mlp_out");
            let d_gate_scalar = sb.d_gate_scalar.expect("shared_expert_bwd: the gated arm needs SharedExpertBwdScratch::d_gate_scalar");
            let d_gate_logits = sb.d_gate_logits.expect("shared_expert_bwd: the gated arm needs SharedExpertBwdScratch::d_gate_logits");
            // scaled = mlp_out * gate_scalar : d_mlp_out = d_out * gate_scalar
            // (scale_row is its own backward w.r.t. `x`), and the row-scale
            // factor's own gradient is the row dot with the forward's mlp_out.
            let gate_scalar = saved.gate_scalar.expect("shared_expert_bwd: the gated arm needs SharedExpertActs::gate_scalar");
            let saved_mlp_out = saved.mlp_out.expect("shared_expert_bwd: the gated arm needs SharedExpertActs::mlp_out");
            let gate_logits = saved.gate_logits.expect("shared_expert_bwd: the gated arm needs SharedExpertActs::gate_logits");
            steps.push(g.step(ids.scale_row, &[d_out, gate_scalar, d_mlp_out], &[m * d, d], m * d));
            steps.push(g.step(ids.row_dot, &[d_out, saved_mlp_out, d_gate_scalar], &[m, d, 0, 0, gpu_core::f(1.0)], m));
            steps.push(g.step(ids.sigmoid_bwd, &[gate_logits, d_gate_scalar, d_gate_logits], &[m], m));
            // gate projection (d_model -> 1) -- the gated arm's FIRST d_x touch.
            mm_bwd(&mut steps, d_gate_logits, x, sgw, gr.shared_gate_w, d_x, d, 1, accumulate as u32);
            (d_mlp_out as &DeviceBuffer, 1u32)
        }
    };

    // down projection: dW + dX -> d_h (fresh per call, acc=0)
    mm_bwd(&mut steps, d_mlp_out, saved.h, down_w, gr.down_w, sb.d_h, ff, d, 0);
    // SwiGLU backward: d_h -> d_gate_pre, d_up
    steps.push(g.step(ids.silu_da, &[saved.gate_pre, saved.up, sb.d_h, sb.d_gate_pre], &[m * ff], m * ff));
    steps.push(g.step(ids.silu_db, &[saved.gate_pre, sb.d_h, sb.d_up], &[m * ff], m * ff));
    // up then gate projection, both into d_x
    mm_bwd(&mut steps, sb.d_up, x, up_w, gr.up_w, d_x, d, ff, first_touch);
    // always on top of the up projection's write above, within this one call
    mm_bwd(&mut steps, sb.d_gate_pre, x, gate_w, gr.gate_w, d_x, d, ff, 1);
    steps
}

/// Kernel indices [`shared_expert_fwd_i8`] dispatches. The gate/up/down
/// SwiGLU linears go through `matmul_i8_dyn` (`crate::int8`'s group-wise
/// weight scale + per-token dynamic activation scale - the SAME kernel
/// `qwen3::q8::Q8::mm8` uses), NOT [`expert_fwd_i8`]'s GATED
/// `moe_linear_gated_i8`: the shared expert applies to every row
/// unconditionally, so it has no per-row route/skip to express through a
/// gate array - using the gated kernel here would be dispatching the wrong
/// shape, not just a slower one. The tiny sigmoid-gate projection
/// (`d_model -> 1`) stays fp32 via the plain `matmul`, matching
/// [`expert_fwd_i8`]'s own "not worth quantizing a rank-1 output" scope.
#[derive(Clone, Copy)]
pub struct SharedExpertIds8 {
    pub matmul_i8: usize,
    pub matmul: usize,
    pub silu_mul: usize,
    pub sigmoid: usize,
    pub scale_row: usize,
    pub add2: usize,
    /// `crate::int8::quant_rows_steps`'s `[max_abs_row, quant_pack]` pair -
    /// used to quantize the intermediate `h` (the shared expert's OWN
    /// SwiGLU output, a different width than any routed expert's, so it
    /// cannot reuse `MoeIds8::quant`'s call site, only its kernel ids).
    pub quant: [usize; 2],
}

/// Scratch for [`shared_expert_fwd_i8`]. `hq`/`sh` are `h` quantized fresh
/// (the shared expert's own intermediate, `shared_ff` wide - not the same
/// buffer any routed expert's `ExpertScratch8::hq` uses).
pub struct SharedExpertScratch8<'a> {
    pub gate_pre: &'a DeviceBuffer,
    pub up: &'a DeviceBuffer,
    pub h: &'a DeviceBuffer,
    pub hq: &'a DeviceBuffer,
    pub sh: &'a DeviceBuffer,
    pub mlp_out: &'a DeviceBuffer,
    pub gate_logits: &'a DeviceBuffer,
    pub gate_scalar: &'a DeviceBuffer,
    pub scaled: &'a DeviceBuffer,
}

/// int8 counterpart of [`shared_expert_fwd`]. `xq`/`sx` are the SAME
/// quantized input the routed [`expert_fwd_i8`] loop already produced once
/// (the shared expert reads the identical `x` every routed expert does) -
/// reused here, not requantized. `x_fp32` is that same activation's
/// original fp32 form, needed only for the (unquantized) sigmoid-gate
/// projection when `shared_gate_w` is `Some`. Same two real architectures
/// as [`shared_expert_fwd`] (`Some`/`None` gate) - see that function's doc.
#[allow(clippy::too_many_arguments)]
pub fn shared_expert_fwd_i8(
    g: &Gpu,
    ids: &SharedExpertIds8,
    rows: u32,
    d_model: u32,
    shared_ff: u32,
    xq: &DeviceBuffer,
    sx: &DeviceBuffer,
    x_fp32: &DeviceBuffer,
    gate_w: Lin8,
    up_w: Lin8,
    down_w: Lin8,
    shared_gate_w: Option<&DeviceBuffer>,
    scratch: &SharedExpertScratch8,
    acc: &DeviceBuffer,
    out: &DeviceBuffer,
) -> Vec<Step> {
    let mm8 = |xq: &DeviceBuffer, sx: &DeviceBuffer, w: Lin8, out: &DeviceBuffer, kg: u32, n: u32| {
        g.step(ids.matmul_i8, &[xq, w.wq, sx, w.sw, out], &[rows, kg, n], rows.div_ceil(128) * n.div_ceil(128) * 256)
    };
    let mut steps = vec![
        mm8(xq, sx, gate_w, scratch.gate_pre, d_model / 4, shared_ff),
        mm8(xq, sx, up_w, scratch.up, d_model / 4, shared_ff),
        g.step(ids.silu_mul, &[scratch.gate_pre, scratch.up, scratch.h], &[rows * shared_ff], rows * shared_ff),
    ];
    steps.extend(crate::int8::quant_rows_steps(g, crate::int8::QuantRows { kernels: ids.quant, x: scratch.h, sx: scratch.sh, xq: scratch.hq }, 0, rows, shared_ff));
    steps.push(mm8(scratch.hq, scratch.sh, down_w, scratch.mlp_out, shared_ff / 4, d_model));
    match shared_gate_w {
        Some(shared_gate_w) => {
            steps.push(g.step(ids.matmul, &[x_fp32, shared_gate_w, scratch.gate_logits], &[rows, d_model, 1], rows));
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
// measured several times SLOWER than GLM's existing dense TILED path at
// GLM-5.2's real shape (`crates/glm/examples/moe_migration_bench.rs`), because
// the naive kernel's per-FLOP inefficiency at ~64 rows/expert swamps the
// FLOP-count win sparsity promises. This
// section is the real fix: gather each expert's routed rows into a dense
// sub-batch, run the SAME
// `model::block::pick_gemm`-selected tiled GEMM the dense path already uses
// (unchanged, no new GEMM kernel), scatter the scaled result back.
//
// FIXED (was "audit F9, deliberate remainder"): [`expert_fwd_compact`] below
// does one host scan over `host_gate`, one index upload, and one `submit`
// PER EXPERT - at GLM-5.2 scale (~128 experts x ~48 MoE layers) that is
// ~6100 submits and small uploads per forward. Every per-expert count is
// knowable host-side from ONE pass over `host_gate`, and dispatch ordering
// within a submit already guarantees the shared scratch is safe to reuse
// across experts in a single batched submission, so [`expert_fwd_compact_layer`]
// below buckets rows for ALL experts in one pass, uploads one combined index
// list, and returns one `Vec<Step>` for the caller's existing per-layer
// submit to carry - no new kernel, no per-expert host round trip.
//
// This module's own comment previously named the call-site migration target
// as "crates/glm, crates/omni" - neither directory exists in this tree
// (checked, not assumed): the one real caller of [`expert_fwd_compact`] is
// `crates/glmdsa::model::Glm::forward_compact`, migrated onto
// [`expert_fwd_compact_layer`] in the same change that added it.
// [`expert_fwd_compact`] itself stays - `crates/model/tests/moe_compact_parity.rs`
// still exercises it directly as the simpler, still-correct per-expert
// primitive the layer version is built from.

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

/// Per-expert scratch for [`expert_fwd_compact`]/[`expert_fwd_compact_layer`],
/// sized ONCE by the caller for `capacity` rows and reused across every
/// expert in a layer (fully overwritten each call - the tail past this
/// call's `count` is simply unused, never read, so leftover data from a
/// larger previous expert is harmless). Pass `shape.rows` for the
/// unconditionally-safe `capacity` bound: every row could in principle route
/// to one expert.
///
/// `idx` alone is sized larger than `capacity`: [`expert_fwd_compact_layer`]
/// holds every expert's routed-row list in it SIMULTANEOUSLY (one shared
/// buffer, per-expert offset regions, one upload for the whole layer), so it
/// needs room for `shape.rows * shape.top_k` entries (the exact total across
/// all experts - every row selects exactly `top_k` of them) PLUS up to 63
/// wasted words per expert region (`model::block::pad64`'s padding: each
/// region's `step_sliced` offset must satisfy wgpu's
/// `min_storage_buffer_offset_alignment`, 256B = 64 words on the near-
/// universal case - a real validation failure caught by
/// `logits_all_compact_matches_logits_all`, not a theoretical concern), not
/// just one expert's worst case.
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
    /// `capacity` rows' worth of scratch - pass `shape.rows` for the
    /// unconditionally-safe bound, or a smaller measured high-watermark if
    /// the caller already knows routing is more balanced than that.
    pub fn new(g: &Gpu, shape: &MoeShape, capacity: u32) -> CompactExpertScratch {
        let (d, ff) = (shape.d_model as u64, shape.moe_ff as u64);
        let cap = capacity as u64;
        let idx_cap = cap.max(shape.rows as u64 * shape.top_k as u64 + shape.n_experts as u64 * 64);
        CompactExpertScratch {
            capacity,
            idx: g.storage(idx_cap),
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

/// Row-compacted sparse expert forward - see this section's header doc.
///
/// `host_gate` MUST be the CURRENT contents of `gate` (`[shape.rows,
/// shape.n_experts]`), already read back by the caller (`Gpu::read`). This
/// function does NOT perform that readback itself, so the caller controls
/// exactly when the host/device round trip happens - once per LAYER (read
/// `gate` once, call this once per expert against the same `host_gate`), not
/// once per expert. Consequently this is NOT a pure step-builder like
/// [`expert_fwd`]: each expert's compacted row count is a value only the
/// HOST can know (discovered from `host_gate`), so this function builds AND
/// submits its own steps rather than returning a `Vec<Step>` for the caller
/// to batch - every other step-list builder in this engine always knows its
/// dispatch shapes before building a `Step`, which a data-dependent row
/// count breaks by construction. This is therefore a real synchronisation
/// point per expert (the `submit` below), acceptable for a BATCHED forward
/// (training/eval), not intended for a per-token decode loop.
///
/// Panics if expert `e_idx` routes more rows than `scratch`'s capacity -
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
    // detailed assert - same rigor for both preconditions.
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
        // Nothing routed to this expert this call - still honour
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

/// One MoE layer's ENTIRE row-compacted sparse expert forward, as a SINGLE
/// `Vec<Step>` for the caller to fold into its own per-layer (or per-forward)
/// submit - the real fix this section's header comment names. Unlike
/// [`expert_fwd_compact`] (one host-driven `submit` PER expert), this
/// function does the routing decision for every expert up front from the
/// SAME `host_gate` readback, uploads every expert's routed-row list into
/// `scratch`'s shared `idx` buffer ONCE (per-expert regions, addressed via
/// [`Gpu::step_sliced`] offsets - never overlapping, since each row appears
/// in exactly `top_k` experts' regions and every region is a disjoint slice
/// of the same upload), and returns every expert's gather/GEMM/GEMM/
/// silu/GEMM/scatter steps as one list. Building these steps performs no
/// device submission at all (`Gpu::step`'s own doc: "RECORDED... only sent
/// to GPU on next read/write/poll"), so the ONLY new work this function adds
/// relative to zero is the one `Gpu::write` that uploads the combined index
/// list - not one `submit` per expert.
///
/// `acc` MUST already be zeroed by the caller (e.g. via `g.submit(&[acc], &[])`,
/// the same explicit pre-zero `crates/glmdsa::model::Glm::forward_compact`
/// already performs before its expert loop) - every expert here always
/// ACCUMULATES (`scale_add`'s `accumulate=true`), mirroring that caller's own
/// convention exactly. This is a narrower contract than
/// [`expert_fwd_compact`]'s per-call `accumulate` flag (which can also
/// perform the initial "set" itself): the narrower form is sufficient because
/// zeroing `acc` once per layer, before ANY expert's dispatch, is strictly
/// simpler than picking which per-expert call gets to zero it - the exact
/// bug class the [`expert_fwd_compact`] caller's own comment describes
/// avoiding ("a row whose top-k experts happen to exclude expert 0 would
/// never get `moe_acc` zeroed").
///
/// `host_gate` MUST be the CURRENT contents of `gate` (`[shape.rows,
/// shape.n_experts]`), already read back by the caller - see
/// [`expert_fwd_compact`]'s own doc for why this function cannot do that
/// readback itself. `expert_weights[e]` is expert `e`'s own
/// `(gate_w, up_w, down_w)`.
///
/// Panics if any expert routes more rows than `scratch`'s (per-expert, NOT
/// layer-total) capacity - same precondition as [`expert_fwd_compact`], see
/// [`CompactExpertScratch::new`]'s doc for the safe bound and the SEPARATE,
/// larger bound its shared `idx` buffer needs for this function specifically.
///
/// Returns `(steps, counts)`: `counts[e]` is how many rows expert `e`
/// routed, for the caller's own diagnostics - the same value
/// [`expert_fwd_compact`] already returns per call, gathered here for every
/// expert at once.
#[allow(clippy::too_many_arguments)]
pub fn expert_fwd_compact_layer(
    g: &Gpu,
    ids: &CompactExpertFwdIds,
    shape: &MoeShape,
    host_gate: &[f32],
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    expert_weights: &[(DeviceBuffer, DeviceBuffer, DeviceBuffer)],
    scratch: &CompactExpertScratch,
    acc: &DeviceBuffer,
) -> (Vec<Step>, Vec<usize>) {
    let (d, ff, e) = (shape.d_model, shape.moe_ff, shape.n_experts);
    assert_eq!(
        host_gate.len(),
        (shape.rows * e) as usize,
        "expert_fwd_compact_layer: host_gate has {} elements, want rows*n_experts = {}x{} -- \
         pass the CURRENT full readback of `gate`",
        host_gate.len(),
        shape.rows,
        e
    );
    assert_eq!(
        expert_weights.len(),
        e as usize,
        "expert_fwd_compact_layer: expert_weights.len() ({}) must equal shape.n_experts ({e})",
        expert_weights.len()
    );

    // One host pass buckets every row's routed experts; `combined` is the
    // WHOLE layer's index upload (every expert's region back to back) and
    // `regions[e]` is expert e's own `(start, count)` slice of it. Each
    // region's `start` is padded up to `model::block::pad64`'s 64-word
    // (256B) grain - `step_sliced`'s storage-buffer offsets must satisfy
    // wgpu's `min_storage_buffer_offset_alignment`, exactly the same
    // constraint `gemm_bidir_fwd` already pads its own per-head strides for.
    // The padding words themselves are never bound by any step (every
    // region's own `(start, count)` slice stops at its real row count), so
    // their content is irrelevant - `resize` fills them with 0 only because
    // `Vec` cannot have gaps.
    let mut per_expert_rows: Vec<Vec<u32>> = vec![Vec::new(); e as usize];
    for r in 0..shape.rows {
        for ei in 0..e {
            if host_gate[(r * e + ei) as usize] > 0.0 {
                per_expert_rows[ei as usize].push(r);
            }
        }
    }
    let mut combined: Vec<u32> = Vec::with_capacity((shape.rows * shape.top_k) as usize);
    let regions: Vec<(u32, u32)> = per_expert_rows
        .iter()
        .map(|rows| {
            let start = crate::block::pad64(combined.len() as u64) as u32;
            combined.resize(start as usize, 0);
            combined.extend_from_slice(rows);
            (start, rows.len() as u32)
        })
        .collect();
    g.write(&scratch.idx, &combined);

    let full = (0u64, 0u64);
    let mut steps = Vec::new();
    let mut counts = Vec::with_capacity(e as usize);
    for (ei, &(start, count)) in regions.iter().enumerate() {
        counts.push(count as usize);
        if count == 0 {
            continue;
        }
        assert!(
            count <= scratch.capacity,
            "expert_fwd_compact_layer: expert {ei} routed {count} rows, exceeding scratch capacity {} -- \
             size CompactExpertScratch::new with shape.rows to make this impossible",
            scratch.capacity
        );
        let idx_slice = (start as u64, count as u64);
        let (gate_w, up_w, down_w) = &expert_weights[ei];
        steps.push(g.step_sliced(ids.gather, &[&scratch.idx, x, &scratch.x_compact], &[idx_slice, full, full], &[d, count], count * d));
        let lin = |x_in: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32, steps: &mut Vec<Step>| {
            let (kid, threads) = crate::block::pick_gemm(count as usize, n as usize, ids.gemm_naive, ids.gemm_tiled, false);
            steps.push(g.step(kid, &[x_in, w, out], &[count, k, n], threads));
        };
        lin(&scratch.x_compact, gate_w, &scratch.gate_pre, d, ff, &mut steps);
        lin(&scratch.x_compact, up_w, &scratch.up, d, ff, &mut steps);
        steps.push(g.step(ids.silu_mul, &[&scratch.gate_pre, &scratch.up, &scratch.h], &[count * ff], count * ff));
        lin(&scratch.h, down_w, &scratch.expert_out, ff, d, &mut steps);
        steps.push(g.step_sliced(
            ids.scatter,
            &[&scratch.idx, gate, &scratch.expert_out, acc],
            &[idx_slice, full, full, full],
            &[count, d, e, ei as u32, 1],
            count * d,
        ));
    }
    (steps, counts)
}
