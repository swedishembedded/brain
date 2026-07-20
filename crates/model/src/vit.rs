// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reusable ViT-block Step-builders (the bidirectional/vision sibling of
//! [`crate::block`]): pre-LN transformer block with optional per-head QK
//! LayerNorm, optional table-driven 2D RoPE, and optional LayerScale — the
//! superset covering DINOv2 blocks (LayerScale only), the WorldMirror trunk
//! (all hooks on), and plain camera-head blocks.
//!
//! Attention is **span + query-chunked** over a fused `[rows, 3C]` qkv buffer
//! using the stride/offset-parameterized cross-attention trio: each span
//! (row0, len) self-attends independently (a frame, or the whole token set),
//! and queries are dispatched in chunks so the materialized `[H, chunk, len]`
//! score slab stays inside a fixed budget. Same-buffer q/kv views are bound
//! via `step_sliced` (row offsets are 256B-aligned because row strides are).
//!
//! Pure dispatch assembly: no WGSL, no ParamStore, no buffer ownership.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Kernel-pipeline indices a model supplies from its own PIPELINES list.
/// Only the kernels a given configuration dispatches need valid indices.
#[derive(Clone, Copy)]
pub struct VitKernelIds {
    pub layernorm: usize,
    pub matmul: usize,
    pub bias_add: usize,
    pub gelu_erf: usize,
    pub scale_chan: usize,
    pub add2: usize,
    pub attn_scores_cross: usize,
    pub attn_softmax_cross: usize,
    pub attn_apply_cross: usize,
    pub ln_head: usize,
    pub rope2d: usize,
}

/// Block shape: `dim` (C), `heads`, `mlp` hidden width, LayerNorm eps.
#[derive(Clone, Copy)]
pub struct VitShape {
    pub dim: u32,
    pub heads: u32,
    pub mlp: u32,
    pub eps: f32,
}

impl VitShape {
    pub fn head_dim(&self) -> u32 {
        self.dim / self.heads
    }
}

/// Per-head QK LayerNorm weights (applied in place on the q/k regions of the
/// fused qkv buffer, BEFORE RoPE — WorldMirror order).
pub struct QkNorm<'a> {
    pub q_w: &'a DeviceBuffer,
    pub q_b: &'a DeviceBuffer,
    pub k_w: &'a DeviceBuffer,
    pub k_b: &'a DeviceBuffer,
}

/// Host-precomputed 2D-RoPE tables `[tmod, head_dim/2]`; token row -> table
/// row = `row % tmod` (per-frame positions repeat across frames).
pub struct RopeTables<'a> {
    pub cos: &'a DeviceBuffer,
    pub sin: &'a DeviceBuffer,
    pub tmod: u32,
}

/// One block's weights (PyTorch layouts: Linear `[out,in]`, biases `[out]`).
pub struct VitBlockWeights<'a> {
    pub norm1_w: &'a DeviceBuffer,
    pub norm1_b: &'a DeviceBuffer,
    pub qkv_w: &'a DeviceBuffer,
    pub qkv_b: &'a DeviceBuffer,
    pub qk_norm: Option<QkNorm<'a>>,
    pub rope: Option<RopeTables<'a>>,
    pub proj_w: &'a DeviceBuffer,
    pub proj_b: &'a DeviceBuffer,
    /// LayerScale gammas; `None` = identity (no scaling).
    pub ls1: Option<&'a DeviceBuffer>,
    pub norm2_w: &'a DeviceBuffer,
    pub norm2_b: &'a DeviceBuffer,
    pub fc1_w: &'a DeviceBuffer,
    pub fc1_b: &'a DeviceBuffer,
    pub fc2_w: &'a DeviceBuffer,
    pub fc2_b: &'a DeviceBuffer,
    pub ls2: Option<&'a DeviceBuffer>,
}

/// Model-owned scratch, reused across every block invocation. Sizes (f32):
/// `ln,ctx,res: rows*C`, `qkv: rows*3C`, `h,h2: rows*mlp`,
/// `scores,probs: heads * chunk * max_span_len` (see [`attn_chunk_for`]).
pub struct VitScratch {
    pub ln: DeviceBuffer,
    pub qkv: DeviceBuffer,
    pub ctx: DeviceBuffer,
    pub h: DeviceBuffer,
    pub h2: DeviceBuffer,
    pub res: DeviceBuffer,
    pub scores: DeviceBuffer,
    pub probs: DeviceBuffer,
}

impl VitScratch {
    /// Allocate for up to `rows` tokens with score slabs sized for
    /// `chunk`-query dispatches against spans up to `max_span` keys.
    pub fn new(gpu: &Gpu, sh: &VitShape, rows: u32, chunk: u32, max_span: u32) -> VitScratch {
        let rc = rows as u64 * sh.dim as u64;
        let slab = sh.heads as u64 * chunk as u64 * max_span as u64;
        VitScratch {
            ln: gpu.storage(rc),
            qkv: gpu.storage(3 * rc),
            ctx: gpu.storage(rc),
            h: gpu.storage(rows as u64 * sh.mlp as u64),
            h2: gpu.storage(rows as u64 * sh.mlp as u64),
            res: gpu.storage(rc),
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
        }
    }
}

/// Largest query-chunk size whose `[heads, chunk, max_span]` f32 score slab
/// fits in `budget_bytes` (min 64 rows so tiny budgets still work).
pub fn attn_chunk_for(sh: &VitShape, max_span: u32, budget_bytes: u64) -> u32 {
    let per_row = sh.heads as u64 * max_span as u64 * 4;
    ((budget_bytes / per_row.max(1)) as u32).clamp(64, 4096)
}

/// Span + chunked self-attention over the fused qkv buffer: for each span
/// `(row0, len)`, queries attend to that span's keys/values; results land in
/// `ctx` at the same absolute rows. `chunk` bounds the score slab.
#[allow(clippy::too_many_arguments)]
pub fn chunked_attn_fwd(
    g: &Gpu,
    k: &VitKernelIds,
    sh: &VitShape,
    qkv: &DeviceBuffer,
    ctx: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    spans: &[(u32, u32)],
    chunk: u32,
    steps: &mut Vec<Step>,
) {
    let c = sh.dim;
    let stride = 3 * c;
    for &(row0, len) in spans {
        let mut q0 = 0u32;
        while q0 < len {
            let qn = chunk.min(len - q0);
            // q view: rows [row0+q0 ..); kv view + ctx view: rows [row0 ..).
            let q_off = (row0 + q0) as u64 * stride as u64;
            let kv_off = row0 as u64 * stride as u64;
            let ctx_off = (row0 + q0) as u64 * c as u64;
            steps.push(g.step_sliced(
                k.attn_scores_cross,
                &[qkv, qkv, scores],
                &[(q_off, 0), (kv_off, 0), (0, 0)],
                &[1, sh.heads, qn, len, sh.head_dim(), stride, stride, 0, c],
                sh.heads * qn * len,
            ));
            steps.push(g.step(
                k.attn_softmax_cross,
                &[scores, probs],
                &[1, sh.heads, qn, len],
                sh.heads * qn,
            ));
            steps.push(g.step_sliced(
                k.attn_apply_cross,
                &[probs, qkv, ctx],
                &[(0, 0), (kv_off, 0), (ctx_off, 0)],
                &[1, sh.heads, qn, len, sh.head_dim(), stride, 2 * c, c],
                sh.heads * qn * sh.head_dim(),
            ));
            q0 += qn;
        }
    }
}

/// Backward kernel ids (only needed by [`vit_block_bwd`]).
#[derive(Clone, Copy)]
pub struct VitBwdIds {
    pub layernorm_dx: usize,
    pub ln_dgamma: usize,
    pub ln_dbeta: usize,
    pub matmul_dx: usize,
    pub matmul_dw: usize,
    pub bias_grad: usize,
    pub gelu_erf_bwd: usize,
    pub scale_chan_dg: usize,
    pub ln_head_dx: usize,
    pub ln_head_dgb: usize,
    pub attn_bwd_dscores_cross: usize,
    pub attn_bwd_dv_cross: usize,
    pub attn_bwd_dq_cross: usize,
    pub attn_bwd_dk_cross: usize,
    pub ln_stats: usize,
    pub mul: usize,
    pub axpy: usize,
}

/// Forward-side caches the backward needs (SSA: fresh buffers per block).
/// The training-mode forward must run with `cache` so pre-QK-norm qkv and
/// post-rope qkv survive (inference normalizes/rotates in place).
pub struct VitBlockCache {
    pub x_in: DeviceBuffer,     // block input
    pub ln1: DeviceBuffer,      // LN1 output
    pub qkv_pre: DeviceBuffer,  // qkv before qk-norm/rope
    pub qkv: DeviceBuffer,      // qkv after qk-norm+rope (attention input)
    pub probs: DeviceBuffer,    // softmax probs (per span, chunk == span!)
    pub ctx: DeviceBuffer,
    pub attn_proj: DeviceBuffer, // proj output (pre-LayerScale)
    pub res_mid: DeviceBuffer,  // after attention residual
    pub ln2: DeviceBuffer,
    pub h: DeviceBuffer,        // fc1 out (pre-gelu)
    pub h2: DeviceBuffer,       // gelu out
    pub mlp_out: DeviceBuffer,  // fc2 out (pre-LayerScale)
}

impl VitBlockCache {
    pub fn new(gpu: &Gpu, sh: &VitShape, rows: u32, max_span: u32) -> VitBlockCache {
        let rc = rows as u64 * sh.dim as u64;
        VitBlockCache {
            x_in: gpu.storage(rc),
            ln1: gpu.storage(rc),
            qkv_pre: gpu.storage(3 * rc),
            qkv: gpu.storage(3 * rc),
            probs: gpu.storage(sh.heads as u64 * rows as u64 * max_span as u64),
            ctx: gpu.storage(rc),
            attn_proj: gpu.storage(rc),
            res_mid: gpu.storage(rc),
            ln2: gpu.storage(rc),
            h: gpu.storage(rows as u64 * sh.mlp as u64),
            h2: gpu.storage(rows as u64 * sh.mlp as u64),
            mlp_out: gpu.storage(rc),
        }
    }
}

/// Record one pre-LN ViT block, in place on `x` (`[rows, C]`):
///   x += ls1 ∘ proj(attn(qk_norm/rope(qkv(LN1(x)))));
///   x += ls2 ∘ fc2(gelu_erf(fc1(LN2(x))))
#[allow(clippy::too_many_arguments)]
pub fn vit_block_fwd(
    g: &Gpu,
    k: &VitKernelIds,
    sh: &VitShape,
    w: &VitBlockWeights,
    x: &DeviceBuffer,
    rows: u32,
    spans: &[(u32, u32)],
    chunk: u32,
    scr: &VitScratch,
    steps: &mut Vec<Step>,
) {
    let c = sh.dim;
    let hd = sh.head_dim();
    let stride = 3 * c;

    // ---- attention half ----
    steps.push(g.step(k.layernorm, &[x, w.norm1_w, w.norm1_b, &scr.ln], &[c, rows, f(sh.eps)], rows));
    steps.push(g.step(k.matmul, &[&scr.ln, w.qkv_w, &scr.qkv], &[rows, c, stride], rows * stride));
    steps.push(g.step(k.bias_add, &[&scr.qkv, w.qkv_b], &[rows, stride], rows * stride));
    if let Some(qk) = &w.qk_norm {
        steps.push(g.step(k.ln_head, &[&scr.qkv, qk.q_w, qk.q_b], &[rows, sh.heads, hd, stride, 0, f(sh.eps)], rows * sh.heads));
        steps.push(g.step(k.ln_head, &[&scr.qkv, qk.k_w, qk.k_b], &[rows, sh.heads, hd, stride, c, f(sh.eps)], rows * sh.heads));
    }
    if let Some(r) = &w.rope {
        let half = hd / 2;
        steps.push(g.step(k.rope2d, &[&scr.qkv, r.cos, r.sin], &[rows, sh.heads, half, stride, 0, r.tmod, f(1.0)], rows * sh.heads * half));
        steps.push(g.step(k.rope2d, &[&scr.qkv, r.cos, r.sin], &[rows, sh.heads, half, stride, c, r.tmod, f(1.0)], rows * sh.heads * half));
    }
    chunked_attn_fwd(g, k, sh, &scr.qkv, &scr.ctx, &scr.scores, &scr.probs, spans, chunk, steps);
    steps.push(g.step(k.matmul, &[&scr.ctx, w.proj_w, &scr.ln], &[rows, c, c], rows * c));
    steps.push(g.step(k.bias_add, &[&scr.ln, w.proj_b], &[rows, c], rows * c));
    let branch: &DeviceBuffer = if let Some(ls1) = w.ls1 {
        steps.push(g.step(k.scale_chan, &[&scr.ln, ls1, &scr.ctx], &[rows * c, c, 1], rows * c));
        &scr.ctx
    } else {
        &scr.ln
    };
    steps.push(g.step(k.add2, &[x, branch, &scr.res], &[rows * c], rows * c));

    // ---- MLP half ----
    steps.push(g.step(k.layernorm, &[&scr.res, w.norm2_w, w.norm2_b, &scr.ln], &[c, rows, f(sh.eps)], rows));
    steps.push(g.step(k.matmul, &[&scr.ln, w.fc1_w, &scr.h], &[rows, c, sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.bias_add, &[&scr.h, w.fc1_b], &[rows, sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.gelu_erf, &[&scr.h, &scr.h2], &[rows * sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.matmul, &[&scr.h2, w.fc2_w, &scr.ln], &[rows, sh.mlp, c], rows * c));
    steps.push(g.step(k.bias_add, &[&scr.ln, w.fc2_b], &[rows, c], rows * c));
    let branch: &DeviceBuffer = if let Some(ls2) = w.ls2 {
        steps.push(g.step(k.scale_chan, &[&scr.ln, ls2, &scr.ctx], &[rows * c, c, 1], rows * c));
        &scr.ctx
    } else {
        &scr.ln
    };
    steps.push(g.step(k.add2, &[&scr.res, branch, x], &[rows * c], rows * c));
}
