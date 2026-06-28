// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reusable transformer-block Step-builders — the composable layer new
//! architectures build on instead of re-hand-rolling dispatch sequences.
//!
//! Each model maps its own PIPELINE kernel indices into [`KernelIds`] (so no
//! model has to reorder its pipeline list), then composes the forward/backward
//! graph from these helpers. They are pure dispatch assembly — no WGSL, no
//! ParamStore, no buffer ownership — so they stay decoupled from any one model
//! and are validated by each caller's gradient check.
//!
//! Covered today (the Qwen/RMSNorm family): RMSNorm fwd/bwd, half-split RoPE
//! fwd/bwd, grouped-query attention fwd/bwd, and the SwiGLU activation fwd/bwd.
//! Linear projections stay in the model (they carry model-specific concerns such
//! as LoRA adapters and bias). MoE/GPT/PID are not yet ported.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Kernel-pipeline indices a model supplies from its own PIPELINES list. Only
/// the kernels a given helper uses need valid indices.
#[derive(Clone, Copy)]
pub struct KernelIds {
    pub rmsnorm: usize,
    pub rms_inv: usize,
    pub rmsnorm_dx: usize,
    pub rmsnorm_dw: usize,
    pub rope: usize,
    pub rope_bwd: usize,
    pub gqa_scores: usize,
    pub gqa_apply: usize,
    pub attn_softmax: usize,
    pub gqa_dscores: usize,
    pub gqa_dv: usize,
    pub gqa_dq: usize,
    pub gqa_dk: usize,
    pub silu_mul: usize,
    pub silu_da: usize,
    pub silu_db: usize,
}

/// Grouped-query attention shape (MHA is the special case `n_kv_heads == n_heads`).
#[derive(Clone, Copy)]
pub struct Gqa {
    pub b: u32,
    pub t: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

impl Gqa {
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }
    fn params(&self) -> [u32; 6] {
        [self.b, self.n_heads, self.n_kv_heads, self.t, self.head_dim, self.group()]
    }
}

/// RMSNorm forward: `out = (x / rms(x)) * w` over the last `dim` axis, one row
/// per invocation (`rows` total).
pub fn rmsnorm_fwd(g: &Gpu, k: &KernelIds, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32) -> Step {
    g.step(k.rmsnorm, &[x, w, out], &[dim, rows], rows)
}

/// RMSNorm backward: always the input grad (`dx`); the gain grad (`gw`, needing
/// the per-row inverse `inv`) only when `gw` is `Some` (trainable gain).
pub fn rmsnorm_bwd(
    g: &Gpu,
    k: &KernelIds,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    dy: &DeviceBuffer,
    dx: &DeviceBuffer,
    inv: &DeviceBuffer,
    gw: Option<&DeviceBuffer>,
    dim: u32,
    rows: u32,
) -> Vec<Step> {
    let mut s = Vec::new();
    if let Some(gw) = gw {
        s.push(g.step(k.rms_inv, &[x, inv], &[dim, rows], rows));
        s.push(g.step(k.rmsnorm_dw, &[dy, x, inv, gw], &[dim, rows], dim));
    }
    s.push(g.step(k.rmsnorm_dx, &[x, w, dy, dx], &[dim, rows], rows));
    s
}

/// Half-split RoPE (forward) in place on a contiguous q/k buffer (one head-group
/// per row). `row_stride` is the per-row width; `theta` the rotary base.
pub fn rope_fwd(g: &Gpu, k: &KernelIds, buf: &DeviceBuffer, n: u32, n_heads: u32, head_dim: u32, row_stride: u32, t: u32, theta: f32) -> Step {
    let half = head_dim / 2;
    g.step(k.rope, &[buf], &[n, n_heads, head_dim, row_stride, 0, t, f(theta)], n * n_heads * half)
}

/// Half-split RoPE backward (in place on the grad buffer).
pub fn rope_bwd(g: &Gpu, k: &KernelIds, buf: &DeviceBuffer, n: u32, n_heads: u32, head_dim: u32, row_stride: u32, t: u32, theta: f32) -> Step {
    let half = head_dim / 2;
    g.step(k.rope_bwd, &[buf], &[n, n_heads, head_dim, row_stride, 0, t, f(theta)], n * n_heads * half)
}

/// GQA attention forward: `scores = qkᵀ/√d (+causal)`, `probs = softmax(scores)`,
/// `ctx = probs·v`. `q`/`ctx` are `[B*T, n_heads*head_dim]`; `k`/`v` are
/// `[B*T, n_kv_heads*head_dim]`; `scores`/`probs` are `[B*n_heads*T*T]`.
pub fn gqa_fwd(
    g: &Gpu,
    k: &KernelIds,
    a: &Gqa,
    q: &DeviceBuffer,
    kbuf: &DeviceBuffer,
    v: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Vec<Step> {
    let p = a.params();
    vec![
        g.step(k.gqa_scores, &[q, kbuf, scores], &p, a.b * a.n_heads * a.t * a.t),
        g.step(k.attn_softmax, &[scores, probs], &[a.b, a.n_heads, a.t], a.b * a.n_heads * a.t),
        g.step(k.gqa_apply, &[probs, v, ctx], &p, a.b * a.n_heads * a.t * a.head_dim),
    ]
}

/// GQA attention backward: produces `d_scores`, `d_v`, `d_q`, `d_k` from the
/// context grad `d_ctx` and the cached `q`/`k`/`v`/`probs`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_bwd(
    g: &Gpu,
    k: &KernelIds,
    a: &Gqa,
    q: &DeviceBuffer,
    kbuf: &DeviceBuffer,
    v: &DeviceBuffer,
    probs: &DeviceBuffer,
    d_ctx: &DeviceBuffer,
    d_scores: &DeviceBuffer,
    d_q: &DeviceBuffer,
    d_k: &DeviceBuffer,
    d_v: &DeviceBuffer,
) -> Vec<Step> {
    let p = a.params();
    vec![
        g.step(k.gqa_dscores, &[d_ctx, v, probs, d_scores], &p, a.b * a.n_heads * a.t),
        g.step(k.gqa_dv, &[probs, d_ctx, d_v], &p, a.b * a.n_kv_heads * a.t * a.head_dim),
        g.step(k.gqa_dq, &[d_scores, kbuf, d_q], &p, a.b * a.n_heads * a.t * a.head_dim),
        g.step(k.gqa_dk, &[d_scores, q, d_k], &p, a.b * a.n_kv_heads * a.t * a.head_dim),
    ]
}

/// SwiGLU activation forward: `h = SiLU(gate) * up`, elementwise over `total`.
pub fn swiglu_fwd(g: &Gpu, k: &KernelIds, gate: &DeviceBuffer, up: &DeviceBuffer, h: &DeviceBuffer, total: u32) -> Step {
    g.step(k.silu_mul, &[gate, up, h], &[total], total)
}

/// SwiGLU backward: grads w.r.t. the gate pre-activation and the up projection.
pub fn swiglu_bwd(
    g: &Gpu,
    k: &KernelIds,
    gate: &DeviceBuffer,
    up: &DeviceBuffer,
    d_h: &DeviceBuffer,
    d_gate: &DeviceBuffer,
    d_up: &DeviceBuffer,
    total: u32,
) -> Vec<Step> {
    vec![
        g.step(k.silu_da, &[gate, up, d_h, d_gate], &[total], total),
        g.step(k.silu_db, &[gate, d_h, d_up], &[total], total),
    ]
}
