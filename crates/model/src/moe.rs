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
//! What this module deliberately does NOT do (left for a follow-up): no
//! row-compaction/gather-scatter (WGSL kernels here may not use atomics, so a
//! parallel stream-compaction would need a separate prefix-sum pass; the
//! per-row early-exit already removes the FLOPs, just not the thread
//! *launches* — see `docs/models/omni/status.md` M2 for the follow-up plan),
//! no int8 grouped GEMM tier, and no backward pass yet.

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
