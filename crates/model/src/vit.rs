// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reusable ViT-block Step-builders (the bidirectional/vision sibling of
//! [`crate::block`]): pre-LN transformer block with optional per-head QK
//! LayerNorm, optional table-driven 2D RoPE, and optional LayerScale - the
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

// See `block.rs`: these are dispatch builders whose arity is the kernel's
// buffer + Params list, not a design smell.
#![allow(clippy::too_many_arguments)]

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Kernel-pipeline indices a model supplies from its own PIPELINES list.
/// Only the kernels a given configuration dispatches need valid indices.
#[derive(Clone, Copy)]
pub struct VitKernelIds {
    pub layernorm: usize,
    pub matmul: usize,
    /// 8-row-blocked matmul (bit-identical to `matmul`, 8× less weight
    /// traffic) - used for the large forward linears.
    pub matmul_rows: usize,
    pub bias_add: usize,
    /// The MLP's pointwise activation. ANY `(x)->(y)` elementwise kernel with
    /// the `Params { total: u32 }` + `(x, out)` signature fits this slot -
    /// `gelu_erf`, `gelu` (tanh), `quick_gelu` and `silu` are all binary-
    /// compatible here, so the field names the ROLE, not one kernel.
    pub mlp_act: usize,
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
/// fused qkv buffer, BEFORE RoPE - WorldMirror order).
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
    // Delegates to the layout-generic core in `crate::block` (vit's fused
    // layout: [q|k|v], stride 3c, offsets 0/c/2c, ctx width c).
    let c = sh.dim;
    let ids = crate::block::CrossIds {
        scores: k.attn_scores_cross,
        softmax: k.attn_softmax_cross,
        apply: k.attn_apply_cross,
    };
    crate::block::chunked_bidir_fwd(
        g, &ids, sh.heads, sh.head_dim(), c, qkv, 3 * c, 0, c, 2 * c, ctx, scores, probs, spans, chunk, None, steps,
    );
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
    /// Backward of whatever [`VitKernelIds::mlp_act`] holds - same role-not-
    /// kernel rule: `Params { total: u32 }` + `(x, dout, dx)`. It MUST be the
    /// adjoint of the forward slot's kernel; `gelu_bwd` standing in for
    /// `gelu_erf_bwd` is a real, gradcheck-invisible bug (see
    /// `crates/gradcheck/tests/gelu_erf_fd.rs`).
    pub mlp_act_bwd: usize,
    pub scale_chan_dg: usize,
    pub ln_head_dx: usize,
    pub ln_head_dgb: usize,
    pub attn_bwd_dscores_cross: usize,
    pub attn_bwd_dv_cross: usize,
    pub attn_bwd_dq_cross: usize,
    pub attn_bwd_dk_cross: usize,
    pub ln_stats: usize,
    pub region_copy: usize,
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
    /// Softmax probs, one `[heads, len, len]` slab per span at
    /// [`probs_offsets`] (chunk == span). Slabs are PADDED to the storage
    /// binding alignment, so the buffer is sized for the padding too.
    pub probs: DeviceBuffer,
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
        // Σ heads·lenᵢ² ≤ heads·max_span·Σlenᵢ = heads·max_span·rows, plus at
        // most `BIND_ALIGN - 1` padding floats per span and at most `rows`
        // spans (every span has len ≥ 1). Exact sizing: `probs_len(spans)`.
        let probs = sh.heads as u64 * rows as u64 * max_span as u64 + BIND_ALIGN * rows as u64;
        VitBlockCache {
            x_in: gpu.storage(rc),
            ln1: gpu.storage(rc),
            qkv_pre: gpu.storage(3 * rc),
            qkv: gpu.storage(3 * rc),
            probs: gpu.storage(probs),
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
///   x += ls2 ∘ fc2(mlp_act(fc1(LN2(x))))
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
    // The coalesced workgroup-per-row LayerNorm where the model registered
    // it (2.3-9.1x on a P40); reference otherwise. See `block::LayerNormIds`.
    let ln = crate::block::LayerNormIds::resolve_fwd(g, k.layernorm);
    let hd = sh.head_dim();
    let stride = 3 * c;

    // ---- attention half ----
    steps.push(crate::block::layernorm_fwd(g, &ln, x, w.norm1_w, w.norm1_b, &scr.ln, c, rows, sh.eps));
    steps.push(g.step(k.matmul_rows, &[&scr.ln, w.qkv_w, &scr.qkv], &[rows, c, stride], rows.div_ceil(8) * stride));
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
    steps.push(g.step(k.matmul_rows, &[&scr.ctx, w.proj_w, &scr.ln], &[rows, c, c], rows.div_ceil(8) * c));
    steps.push(g.step(k.bias_add, &[&scr.ln, w.proj_b], &[rows, c], rows * c));
    let branch: &DeviceBuffer = if let Some(ls1) = w.ls1 {
        steps.push(g.step(k.scale_chan, &[&scr.ln, ls1, &scr.ctx], &[rows * c, c, 1], rows * c));
        &scr.ctx
    } else {
        &scr.ln
    };
    steps.push(g.step(k.add2, &[x, branch, &scr.res], &[rows * c], rows * c));

    // ---- MLP half ----
    steps.push(crate::block::layernorm_fwd(g, &ln, &scr.res, w.norm2_w, w.norm2_b, &scr.ln, c, rows, sh.eps));
    steps.push(g.step(k.matmul_rows, &[&scr.ln, w.fc1_w, &scr.h], &[rows, c, sh.mlp], rows.div_ceil(8) * sh.mlp));
    steps.push(g.step(k.bias_add, &[&scr.h, w.fc1_b], &[rows, sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.mlp_act, &[&scr.h, &scr.h2], &[rows * sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.matmul_rows, &[&scr.h2, w.fc2_w, &scr.ln], &[rows, sh.mlp, c], rows.div_ceil(8) * c));
    steps.push(g.step(k.bias_add, &[&scr.ln, w.fc2_b], &[rows, c], rows * c));
    let branch: &DeviceBuffer = if let Some(ls2) = w.ls2 {
        steps.push(g.step(k.scale_chan, &[&scr.ln, ls2, &scr.ctx], &[rows * c, c, 1], rows * c));
        &scr.ctx
    } else {
        &scr.ln
    };
    steps.push(g.step(k.add2, &[&scr.res, branch, x], &[rows * c], rows * c));
}

/// Gradient buffers for one block's parameters (accumulated `+=`; zero them
/// before the backward like `ParamStore::zero_grads` does).
pub struct VitBlockGrads<'a> {
    pub norm1_w: &'a DeviceBuffer,
    pub norm1_b: &'a DeviceBuffer,
    pub qkv_w: &'a DeviceBuffer,
    pub qkv_b: &'a DeviceBuffer,
    pub q_norm_w: Option<&'a DeviceBuffer>,
    pub q_norm_b: Option<&'a DeviceBuffer>,
    pub k_norm_w: Option<&'a DeviceBuffer>,
    pub k_norm_b: Option<&'a DeviceBuffer>,
    pub proj_w: &'a DeviceBuffer,
    pub proj_b: &'a DeviceBuffer,
    pub ls1: Option<&'a DeviceBuffer>,
    pub norm2_w: &'a DeviceBuffer,
    pub norm2_b: &'a DeviceBuffer,
    pub fc1_w: &'a DeviceBuffer,
    pub fc1_b: &'a DeviceBuffer,
    pub fc2_w: &'a DeviceBuffer,
    pub fc2_b: &'a DeviceBuffer,
    pub ls2: Option<&'a DeviceBuffer>,
}

/// Backward scratch, reusable across blocks (per-block state lives in the
/// caches). Sizes: `[rows,C]` ×4, `[rows,3C]` ×2, `[rows,M]` ×2, per-row LN
/// stats ×2, and a `[heads, span, span]` dscores slab.
pub struct VitBwdScratch {
    pub d_res: DeviceBuffer,
    pub d_branch: DeviceBuffer,
    pub d_ln: DeviceBuffer,
    pub tmp: DeviceBuffer,
    pub d_qkv: DeviceBuffer,
    pub d_qkv_pre: DeviceBuffer,
    pub d_ctx: DeviceBuffer,
    pub d_h: DeviceBuffer,
    pub d_h2: DeviceBuffer,
    pub mean: DeviceBuffer,
    pub inv: DeviceBuffer,
    pub dscores: DeviceBuffer,
}

impl VitBwdScratch {
    pub fn new(gpu: &Gpu, sh: &VitShape, rows: u32, max_span: u32) -> VitBwdScratch {
        let rc = rows as u64 * sh.dim as u64;
        VitBwdScratch {
            d_res: gpu.storage(rc),
            d_branch: gpu.storage(rc),
            d_ln: gpu.storage(rc),
            tmp: gpu.storage(rc),
            d_qkv: gpu.storage(3 * rc),
            d_qkv_pre: gpu.storage(3 * rc),
            d_ctx: gpu.storage(rc),
            d_h: gpu.storage(rows as u64 * sh.mlp as u64),
            d_h2: gpu.storage(rows as u64 * sh.mlp as u64),
            mean: gpu.storage(rows as u64),
            inv: gpu.storage(rows as u64),
            dscores: gpu.storage(sh.heads as u64 * max_span as u64 * max_span as u64),
        }
    }
}

/// Training-mode forward: like [`vit_block_fwd`] but every stage lands in
/// the block's [`VitBlockCache`] for the backward. Input = `cache.x_in`
/// (caller-filled), output → `x_out`. Attention runs ONE dispatch per span
/// (chunk == span) through [`cross_q_fwd`], caching each span's probs at
/// [`probs_offsets`] - a running, binding-aligned prefix, so `spans` may be
/// RAGGED. `cache.qkv` must be in the submit clears list (axpy-copied).
pub fn vit_block_fwd_cached(
    g: &Gpu,
    k: &VitKernelIds,
    kb: &VitBwdIds,
    sh: &VitShape,
    w: &VitBlockWeights,
    cache: &VitBlockCache,
    x_out: &DeviceBuffer,
    rows: u32,
    spans: &[(u32, u32)],
    scr_tmp: &DeviceBuffer, // [rows, C] LayerScale staging
    scores: &DeviceBuffer,  // [heads, max_span, max_span] transient
    steps: &mut Vec<Step>,
) {
    let c = sh.dim;
    let ln = crate::block::LayerNormIds::resolve(g, k.layernorm, kb.ln_stats, kb.layernorm_dx);
    let hd = sh.head_dim();
    let stride = 3 * c;

    steps.push(crate::block::layernorm_fwd(g, &ln, &cache.x_in, w.norm1_w, w.norm1_b, &cache.ln1, c, rows, sh.eps));
    steps.push(g.step(k.matmul, &[&cache.ln1, w.qkv_w, &cache.qkv_pre], &[rows, c, stride], rows * stride));
    steps.push(g.step(k.bias_add, &[&cache.qkv_pre, w.qkv_b], &[rows, stride], rows * stride));
    steps.push(g.step(kb.axpy, &[&cache.qkv, &cache.qkv_pre], &[rows * stride, f(1.0)], rows * stride));
    if let Some(qk) = &w.qk_norm {
        steps.push(g.step(k.ln_head, &[&cache.qkv, qk.q_w, qk.q_b], &[rows, sh.heads, hd, stride, 0, f(sh.eps)], rows * sh.heads));
        steps.push(g.step(k.ln_head, &[&cache.qkv, qk.k_w, qk.k_b], &[rows, sh.heads, hd, stride, c, f(sh.eps)], rows * sh.heads));
    }
    if let Some(r) = &w.rope {
        let half = hd / 2;
        steps.push(g.step(k.rope2d, &[&cache.qkv, r.cos, r.sin], &[rows, sh.heads, half, stride, 0, r.tmod, f(1.0)], rows * sh.heads * half));
        steps.push(g.step(k.rope2d, &[&cache.qkv, r.cos, r.sin], &[rows, sh.heads, half, stride, c, r.tmod, f(1.0)], rows * sh.heads * half));
    }
    // Self-attention is the `q0 == k0`, `qn == kn` case of [`cross_q_fwd`] -
    // ONE implementation of the per-span cross trio, shared with Hiera's pooled
    // query. Its running (and binding-aligned) probs offsets are what make a
    // RAGGED span list - a border window, Swin's shifted partition - bindable;
    // `si * heads * len * len` was both wrong for ragged spans and unbindable.
    let cross = crate::block::CrossIds {
        scores: k.attn_scores_cross,
        softmax: k.attn_softmax_cross,
        apply: k.attn_apply_cross,
    };
    let att: Vec<AttnSpan> = spans.iter().map(|&(row0, len)| AttnSpan::span(row0, len)).collect();
    cross_q_fwd(
        g, &cross, sh, &cache.qkv, stride, 0, &cache.qkv, stride, c, 2 * c, &cache.ctx, scores, &cache.probs, &att,
        steps,
    );
    steps.push(g.step(k.matmul, &[&cache.ctx, w.proj_w, &cache.attn_proj], &[rows, c, c], rows * c));
    steps.push(g.step(k.bias_add, &[&cache.attn_proj, w.proj_b], &[rows, c], rows * c));
    let branch: &DeviceBuffer = if let Some(ls1) = w.ls1 {
        steps.push(g.step(k.scale_chan, &[&cache.attn_proj, ls1, scr_tmp], &[rows * c, c, 1], rows * c));
        scr_tmp
    } else {
        &cache.attn_proj
    };
    steps.push(g.step(k.add2, &[&cache.x_in, branch, &cache.res_mid], &[rows * c], rows * c));

    steps.push(crate::block::layernorm_fwd(g, &ln, &cache.res_mid, w.norm2_w, w.norm2_b, &cache.ln2, c, rows, sh.eps));
    steps.push(g.step(k.matmul, &[&cache.ln2, w.fc1_w, &cache.h], &[rows, c, sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.bias_add, &[&cache.h, w.fc1_b], &[rows, sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.mlp_act, &[&cache.h, &cache.h2], &[rows * sh.mlp], rows * sh.mlp));
    steps.push(g.step(k.matmul, &[&cache.h2, w.fc2_w, &cache.mlp_out], &[rows, sh.mlp, c], rows * c));
    steps.push(g.step(k.bias_add, &[&cache.mlp_out, w.fc2_b], &[rows, c], rows * c));
    let branch: &DeviceBuffer = if let Some(ls2) = w.ls2 {
        steps.push(g.step(k.scale_chan, &[&cache.mlp_out, ls2, scr_tmp], &[rows * c, c, 1], rows * c));
        scr_tmp
    } else {
        &cache.mlp_out
    };
    steps.push(g.step(k.add2, &[&cache.res_mid, branch, x_out], &[rows * c], rows * c));
}

/// Backward through one cached block: upstream `d_out` → `d_x_in`, parameter
/// grads accumulated into `gr`. Same span discipline as the cached forward
/// (chunk == span; the cross-attention backward kernels ASSIGN, so spans
/// must cover disjoint rows - which frame spans and the single global span
/// both satisfy). `sb.d_qkv`/`sb.d_qkv_pre` must be zero-cleared by the
/// caller's submit for each block.
pub fn vit_block_bwd(
    g: &Gpu,
    k: &VitKernelIds,
    kb: &VitBwdIds,
    sh: &VitShape,
    w: &VitBlockWeights,
    gr: &VitBlockGrads,
    cache: &VitBlockCache,
    d_out: &DeviceBuffer,
    d_x_in: &DeviceBuffer,
    rows: u32,
    spans: &[(u32, u32)],
    sb: &VitBwdScratch,
    steps: &mut Vec<Step>,
) {
    let c = sh.dim;
    let ln = crate::block::LayerNormIds::resolve(g, k.layernorm, kb.ln_stats, kb.layernorm_dx);
    let hd = sh.head_dim();
    let stride = 3 * c;
    let m = sh.mlp;

    // ---- MLP half (upstream d_out) ----
    let d_mlp: &DeviceBuffer = if let Some(ls2) = w.ls2 {
        steps.push(g.step(kb.scale_chan_dg, &[&cache.mlp_out, d_out, gr.ls2.expect("ls2 grad")], &[rows * c, c, 1], c));
        steps.push(g.step(k.scale_chan, &[d_out, ls2, &sb.d_branch], &[rows * c, c, 1], rows * c));
        &sb.d_branch
    } else {
        d_out
    };
    steps.push(g.step(kb.matmul_dx, &[d_mlp, w.fc2_w, &sb.d_h2], &[rows, m, c, 0], rows * m));
    steps.push(g.step(kb.matmul_dw, &[d_mlp, &cache.h2, gr.fc2_w], &[rows, m, c], c * m));
    steps.push(g.step(kb.bias_grad, &[d_mlp, gr.fc2_b], &[rows, c], c));
    steps.push(g.step(kb.mlp_act_bwd, &[&cache.h, &sb.d_h2, &sb.d_h], &[rows * m], rows * m));
    steps.push(g.step(kb.matmul_dx, &[&sb.d_h, w.fc1_w, &sb.d_ln], &[rows, c, m, 0], rows * c));
    steps.push(g.step(kb.matmul_dw, &[&sb.d_h, &cache.ln2, gr.fc1_w], &[rows, c, m], m * c));
    steps.push(g.step(kb.bias_grad, &[&sb.d_h, gr.fc1_b], &[rows, m], m));
    steps.push(crate::block::ln_stats_fwd(g, &ln, &cache.res_mid, &sb.mean, &sb.inv, c, rows, sh.eps));
    steps.push(g.step(kb.ln_dgamma, &[&sb.d_ln, &cache.res_mid, &sb.mean, &sb.inv, gr.norm2_w], &[c, rows], c));
    steps.push(g.step(kb.ln_dbeta, &[&sb.d_ln, gr.norm2_b], &[c, rows], c));
    steps.push(crate::block::layernorm_dx_bwd(g, &ln, &cache.res_mid, w.norm2_w, &sb.d_ln, &sb.tmp, c, rows, sh.eps));
    steps.push(g.step(k.add2, &[d_out, &sb.tmp, &sb.d_res], &[rows * c], rows * c));

    // ---- attention half (upstream sb.d_res) ----
    let d_attn: &DeviceBuffer = if let Some(ls1) = w.ls1 {
        steps.push(g.step(kb.scale_chan_dg, &[&cache.attn_proj, &sb.d_res, gr.ls1.expect("ls1 grad")], &[rows * c, c, 1], c));
        steps.push(g.step(k.scale_chan, &[&sb.d_res, ls1, &sb.d_branch], &[rows * c, c, 1], rows * c));
        &sb.d_branch
    } else {
        &sb.d_res
    };
    steps.push(g.step(kb.matmul_dx, &[d_attn, w.proj_w, &sb.d_ctx], &[rows, c, c, 0], rows * c));
    steps.push(g.step(kb.matmul_dw, &[d_attn, &cache.ctx, gr.proj_w], &[rows, c, c], c * c));
    steps.push(g.step(kb.bias_grad, &[d_attn, gr.proj_b], &[rows, c], c));

    // The adjoint of the cached forward's attention, through the SAME builder -
    // so the two can never drift in how they address a span's cached softmax.
    let att: Vec<AttnSpan> = spans.iter().map(|&(row0, len)| AttnSpan::span(row0, len)).collect();
    cross_q_bwd(
        g,
        kb,
        sh,
        &cache.qkv,
        stride,
        0,
        &cache.qkv,
        stride,
        c,
        2 * c,
        &cache.probs,
        &sb.d_ctx,
        &sb.d_qkv,
        &sb.d_qkv,
        &sb.dscores,
        &att,
        steps,
    );
    if let Some(r) = &w.rope {
        let half = hd / 2;
        steps.push(g.step(k.rope2d, &[&sb.d_qkv, r.cos, r.sin], &[rows, sh.heads, half, stride, 0, r.tmod, f(-1.0)], rows * sh.heads * half));
        steps.push(g.step(k.rope2d, &[&sb.d_qkv, r.cos, r.sin], &[rows, sh.heads, half, stride, c, r.tmod, f(-1.0)], rows * sh.heads * half));
    }
    let d_lin: &DeviceBuffer = if let Some(qk) = &w.qk_norm {
        steps.push(g.step(kb.ln_head_dgb, &[&cache.qkv_pre, &sb.d_qkv, gr.q_norm_w.expect("qnw"), gr.q_norm_b.expect("qnb")], &[rows, sh.heads, hd, stride, 0, f(sh.eps)], hd));
        steps.push(g.step(kb.ln_head_dgb, &[&cache.qkv_pre, &sb.d_qkv, gr.k_norm_w.expect("knw"), gr.k_norm_b.expect("knb")], &[rows, sh.heads, hd, stride, c, f(sh.eps)], hd));
        steps.push(g.step(kb.ln_head_dx, &[&cache.qkv_pre, qk.q_w, &sb.d_qkv, &sb.d_qkv_pre], &[rows, sh.heads, hd, stride, 0, f(sh.eps)], rows * sh.heads));
        steps.push(g.step(kb.ln_head_dx, &[&cache.qkv_pre, qk.k_w, &sb.d_qkv, &sb.d_qkv_pre], &[rows, sh.heads, hd, stride, c, f(sh.eps)], rows * sh.heads));
        steps.push(g.step(kb.region_copy, &[&sb.d_qkv, &sb.d_qkv_pre], &[rows, c, stride, 2 * c], rows * c));
        &sb.d_qkv_pre
    } else {
        &sb.d_qkv
    };
    steps.push(g.step(kb.matmul_dx, &[d_lin, w.qkv_w, &sb.d_ln], &[rows, c, stride, 0], rows * c));
    steps.push(g.step(kb.matmul_dw, &[d_lin, &cache.ln1, gr.qkv_w], &[rows, c, stride], stride * c));
    steps.push(g.step(kb.bias_grad, &[d_lin, gr.qkv_b], &[rows, stride], stride));
    steps.push(crate::block::ln_stats_fwd(g, &ln, &cache.x_in, &sb.mean, &sb.inv, c, rows, sh.eps));
    steps.push(g.step(kb.ln_dgamma, &[&sb.d_ln, &cache.x_in, &sb.mean, &sb.inv, gr.norm1_w], &[c, rows], c));
    steps.push(g.step(kb.ln_dbeta, &[&sb.d_ln, gr.norm1_b], &[c, rows], c));
    steps.push(crate::block::layernorm_dx_bwd(g, &ln, &cache.x_in, w.norm1_w, &sb.d_ln, &sb.tmp, c, rows, sh.eps));
    steps.push(g.step(k.add2, &[&sb.d_res, &sb.tmp, d_x_in], &[rows * c], rows * c));
}

// ===========================================================================
// Windowed attention (Hiera / Swin / DaViT) and Hiera's `q_pool`
// ===========================================================================
//
// COMPOSITION, NOT NEW KERNELS. Two observations do all the work:
//
//  1. **A disjoint window IS a disjoint span** - once the token rows are in
//     window-major order. Window partitioning is therefore a PERMUTATION OF
//     ROWS, and brain already has both halves of a row permutation: `embed`
//     (gather, `dst[r] = src[idx[r]]`) and `row_scatter` (scatter,
//     `dst[idx[r]] = src[r]`). For a permutation those two are exact inverses
//     AND exact adjoints of each other, so the backward of a partition is the
//     forward of the reverse - no new kernel, and no new gradient form.
//
//  2. **Every stage of a ViT block except attention is row-wise**, so the whole
//     block COMMUTES with a row permutation:
//         windowed_block(x) == unpermute(vit_block_fwd(permute(x), win_spans))
//     LayerNorm, the qkv/proj/fc linears, `bias_add`, `ln_head` (per row+head),
//     GELU, LayerScale and the residual adds are all per-row; attention is
//     per-span. So `vit_block_fwd`/`vit_block_fwd_cached`/`vit_block_bwd` need
//     NO window parameter at all: permute once, run N blocks with window spans,
//     unpermute once.
//
//     THE ONE EXCEPTION IS `rope2d`, which indexes its table by `row % tmod` -
//     an absolute-position op, not a row-wise one. Windowed attention composes
//     with `rope: None` (Hiera and Swin both use additive/relative position
//     encodings, not RoPE), or with RoPE tables the caller has ALREADY permuted
//     into window-major order (legal when `tmod == rows`). A model that leaves
//     unpermuted tables on and windows anyway gets silently wrong positions.
//
// Hiera's `q_pool` is the one place the shapes genuinely diverge: the query is
// max-pooled 2x2 inside attention while keys/values stay at full resolution, so
// q and kv have different token counts. The cross-attention kernels
// (`attn_scores_cross`/`attn_softmax_cross`/`attn_apply_cross` and their four
// backward kernels) ALREADY take two lengths, two buffers and independent
// strides/offsets - verified against every kernel header. What they did not
// have was a *builder*: `block::chunked_bidir_fwd` binds one buffer for q and
// kv and ties the query rows to the key rows. [`cross_q_fwd`] / [`cross_q_bwd`]
// below are that builder, and [`vit_block_fwd_cached`] / [`vit_block_bwd`] run
// their own self-attention through it too, so there is exactly one per-span
// cross-attention dispatch sequence in this file.
//
// RAGGED SPANS HAVE A BINDING PRECONDITION - read this before claiming Swin.
// Swin's shifted-window attention is expressible as ragged spans with no mask
// (see [`axis_cuts`]), and the arithmetic is right for any partition. But the
// cached-attention path binds two buffers at a per-span offset, and a storage
// binding offset must be a multiple of `min_storage_buffer_offset_alignment`
// (256 B = 64 f32) or wgpu rejects the bind group outright:
//   * `probs` - SOLVED here: [`probs_offsets`] pads every slab to
//     [`BIND_ALIGN`], so any span list binds. (Unpadded, `heads*qn*kn` is not a
//     multiple of 64 for most real shapes - 16 heads x 729^2 is not - so this
//     bit already for UNIFORM spans.)
//   * `q` / `kv` / `d_q` / `d_kv` - SOLVED here: the row offset rides in the
//     kernels' own `q_off`/`k_off`/`v_off` Params and the buffers bind whole.
//   * `ctx` / `d_ctx` - NOT solvable without a kernel ABI change:
//     `attn_apply_cross` writes `out[(b*Tq + i)*d_model + ...]` and has no
//     output-offset Param, so the binding must carry `q0*C` and that offset
//     must be 64-aligned. [`WindowPlan::ctx_bindable`] answers this for a plan;
//     [`aligned_ctx`] is the loud failure otherwise.
// So: shifted/ragged windows run whenever every span's `q0*C` is 64-aligned -
// always true for `C % 64 == 0`, and for many smaller `C`/window combinations.
// It is NOT unconditional. Lifting the last case means adding an `out_off`
// Param to `attn_apply_cross`/`attn_bwd_dscores_cross`/`attn_bwd_dv_cross`,
// which is an ABI change shared with `seq2seq` and `fastvlm` - a deliberate
// kernel task, not something to slip in here.

/// The two row-permutation kernels. `embed` gathers, `row_scatter` scatters;
/// for a permutation index vector they are exact inverses and exact adjoints.
#[derive(Clone, Copy)]
pub struct VitPermuteIds {
    pub embed: usize,
    pub row_scatter: usize,
}

/// `dst[r, :] = src[idx[r], :]`, `n` rows of `d` floats (`embed`).
/// Params: `[d_model, seq_len] = [d, n]`, threads `n*d`.
pub fn gather_rows(
    g: &Gpu,
    ids: &VitPermuteIds,
    idx: &DeviceBuffer,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    n: u32,
    d: u32,
) -> Step {
    g.step(ids.embed, &[idx, src, dst], &[d, n], n * d)
}

/// `dst[idx[r], :] = src[r, :]`, `n` rows of `d` floats into a `n_rows_out`-row
/// destination (`row_scatter`). Rows of `dst` that no index names are left
/// UNTOUCHED - for a permutation every row is named, so nothing needs clearing;
/// for the q-region scatter of [`q_pool_bwd`] the k/v regions are deliberately
/// left for `attn_bwd_dk/dv_cross` to write.
/// Params: `[n_idx, d, n_rows_out]`, threads `n*d`.
pub fn scatter_rows(
    g: &Gpu,
    ids: &VitPermuteIds,
    idx: &DeviceBuffer,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    n: u32,
    d: u32,
    n_rows_out: u32,
) -> Step {
    g.step(ids.row_scatter, &[idx, src, dst], &[n, d, n_rows_out], n * d)
}

/// Upload a row-index vector as the `u32` index buffer [`gather_rows`] /
/// [`scatter_rows`] bind. Build once per partition, reuse for every block.
pub fn row_index_buffer(g: &Gpu, label: &str, idx: &[u32]) -> DeviceBuffer {
    let b = g.buffer(label, idx.len() as u64 * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
    g.write(&b, idx);
    b
}

/// Row indices selecting one region of a fused `[rows, n_regions*d]` buffer,
/// viewed as `[rows*n_regions, d]`: `idx[t] = t*n_regions + region`.
/// `region_index(rows, 3, 0)` gathers the q region of a `[rows, 3C]` qkv buffer
/// into a compact `[rows, C]` buffer with [`gather_rows`], and scatters the
/// gradient back with [`scatter_rows`] (`n_rows_out = rows*n_regions`).
pub fn region_index(rows: u32, n_regions: u32, region: u32) -> Vec<u32> {
    assert!(region < n_regions, "region {region} out of range for {n_regions} regions");
    (0..rows).map(|t| t * n_regions + region).collect()
}

/// Cut points of one grid axis under a Swin-style cyclic shift.
///
/// Swin rolls by `-shift` and then masks the softmax so tokens that wrapped
/// never attend across the seam. That masked partition is EXACTLY the ragged
/// partition returned here - `[0, shift)`, then full windows, then the
/// remainder - so brain expresses shifted-window attention with **variable-size
/// spans and no mask at all**, which is why no masked-softmax kernel appears in
/// this file. `shift == 0` is the plain partition (Hiera, Swin's even layers).
/// A grid length that is not a multiple of `win` yields a short final window
/// (Hiera instead zero-pads the token grid upstream - that is the caller's
/// `pad2d`, not this planner's business).
fn axis_cuts(len: u32, win: u32, shift: u32) -> Vec<(u32, u32)> {
    assert!(win > 0, "window size must be > 0");
    assert!(shift < win, "shift {shift} must be < window {win}");
    let mut v = Vec::new();
    let mut p = 0u32;
    if shift > 0 {
        let n = shift.min(len);
        v.push((0, n));
        p = n;
    }
    while p < len {
        let n = win.min(len - p);
        v.push((p, n));
        p += n;
    }
    v
}

/// A rectangular token grid partitioned into disjoint attention windows,
/// expressed as a row permutation plus a span list.
///
/// The token buffer is `[grid_h*grid_w, C]` in row-major grid order. `perm()`
/// reorders it into window-major order, where window `m` occupies the
/// contiguous rows `spans()[m]` - so the existing span-chunked attention runs
/// windowed attention unchanged. `inv()` reverses it.
///
/// Three partitions live here, all reusing the same permutation/span machinery:
/// [`WindowPlan::new`] (exact division), [`WindowPlan::shifted`] (Swin's
/// shift, expressed as RAGGED windows instead of roll + mask) and
/// [`WindowPlan::padded`] (SAM-1/ViTDet's zero-pad to a uniform grid). They are
/// three real reference designs, not one policy with knobs - see each
/// constructor's own doc.
#[derive(Clone, Debug)]
pub struct WindowPlan {
    pub grid_h: u32,
    pub grid_w: u32,
    pub win_h: u32,
    pub win_w: u32,
    pub shift_h: u32,
    pub shift_w: u32,
    /// Grid the windows actually tile. Equal to `grid_h`/`grid_w` for every
    /// partition except [`WindowPlan::padded`]'s, where each is rounded up to
    /// a multiple of the window.
    pad_h: u32,
    pad_w: u32,
    perm: Vec<u32>,
    inv: Vec<u32>,
    spans: Vec<(u32, u32)>,
    max_span: u32,
    uniform: bool,
}

impl WindowPlan {
    /// Plain (unshifted) partition - Hiera, Swin's even layers, DaViT's local
    /// window stage.
    pub fn new(grid_h: u32, grid_w: u32, win_h: u32, win_w: u32) -> WindowPlan {
        WindowPlan::shifted(grid_h, grid_w, win_h, win_w, 0, 0)
    }

    /// Swin's shifted partition (`shift_h`/`shift_w` typically `win/2`),
    /// realized as ragged windows rather than roll + attention mask.
    pub fn shifted(grid_h: u32, grid_w: u32, win_h: u32, win_w: u32, shift_h: u32, shift_w: u32) -> WindowPlan {
        let hs = axis_cuts(grid_h, win_h, shift_h);
        let ws = axis_cuts(grid_w, win_w, shift_w);
        let rows = (grid_h * grid_w) as usize;
        let mut perm = Vec::with_capacity(rows);
        let mut spans = Vec::with_capacity(hs.len() * ws.len());
        let mut cursor = 0u32;
        let mut max_span = 0u32;
        // "Uniform" must mean every window is exactly `win_h x win_w`, not
        // merely that the spans are equal LENGTH: a grid of 4 with window 4 and
        // shift 2 splits into two equal 2-row bands, and calling that uniform
        // would hand `QPoolPlan::per_window` the wrong (h, w).
        let mut uniform = true;
        for &(h0, hn) in &hs {
            for &(w0, wn) in &ws {
                uniform &= hn == win_h && wn == win_w;
                for dh in 0..hn {
                    for dw in 0..wn {
                        perm.push((h0 + dh) * grid_w + (w0 + dw));
                    }
                }
                let len = hn * wn;
                spans.push((cursor, len));
                cursor += len;
                max_span = max_span.max(len);
            }
        }
        debug_assert_eq!(perm.len(), rows);
        let mut inv = vec![0u32; rows];
        for (dstrow, &srcrow) in perm.iter().enumerate() {
            inv[srcrow as usize] = dstrow as u32;
        }
        WindowPlan { grid_h, grid_w, win_h, win_w, shift_h, shift_w, pad_h: grid_h, pad_w: grid_w, perm, inv, spans, max_span, uniform }
    }

    /// SAM-1 / ViTDet's partition: zero-pad the grid bottom/right to the next
    /// multiple of the window so EVERY window is exactly `win_h x win_w`.
    ///
    /// The third real variant next to [`WindowPlan::new`] (exact division only)
    /// and [`WindowPlan::shifted`] (Hiera/Swin's ragged final window). It is not
    /// a degenerate case of either: a ragged plan gives border windows a
    /// SHORTER key set, while this one gives them a full-length key set whose
    /// tail rows are the padding.
    ///
    /// **Out-of-grid positions are the sentinel row `grid_h*grid_w`** - one past
    /// the last real row. The caller therefore allocates its per-block row
    /// buffers with `rows()+1` rows, zeroes row `rows()` ONCE, and the existing
    /// [`gather_rows`] pulls that zero row into every padded slot for free. No
    /// masked kernel and no new kernel appear anywhere in this file.
    ///
    /// **Where the pad must happen matters**, and it is the caller's job, not
    /// this function's: SAM pads AFTER `norm1` and BEFORE the `qkv` projection,
    /// so a padded position's k/v is `qkv_bias` (the projection of a zero input
    /// is its bias, NOT zero) and it participates in the real tokens' softmax
    /// as an extra key/value. Padding after `qkv` instead would feed exact
    /// zeros as keys - a different model. Run the projection over `rows()+1`
    /// rows with row `rows()` zeroed and this falls out automatically.
    ///
    /// [`Self::inv`] keeps length `rows()` (real rows only), so reversing a
    /// padded windowed attention with `gather_rows(inv)` drops the pad rows with
    /// no extra bookkeeping - the pad exists only inside the windowed buffer.
    ///
    /// When the grid already divides evenly this is EXACTLY
    /// [`WindowPlan::new`]: same `perm`, same `spans`, no sentinel.
    pub fn padded(grid_h: u32, grid_w: u32, win_h: u32, win_w: u32) -> WindowPlan {
        assert!(win_h > 0 && win_w > 0, "window size must be > 0");
        let pad_h = grid_h.div_ceil(win_h) * win_h;
        let pad_w = grid_w.div_ceil(win_w) * win_w;
        let sentinel = grid_h * grid_w;
        let mut perm = Vec::with_capacity((pad_h * pad_w) as usize);
        let mut spans = Vec::with_capacity(((pad_h / win_h) * (pad_w / win_w)) as usize);
        let mut cursor = 0u32;
        for h0 in (0..pad_h).step_by(win_h as usize) {
            for w0 in (0..pad_w).step_by(win_w as usize) {
                for dh in 0..win_h {
                    for dw in 0..win_w {
                        let (r, c) = (h0 + dh, w0 + dw);
                        perm.push(if r < grid_h && c < grid_w { r * grid_w + c } else { sentinel });
                    }
                }
                spans.push((cursor, win_h * win_w));
                cursor += win_h * win_w;
            }
        }
        // Real rows only - the pad has no home in grid order, and that is what
        // makes `window_reverse` drop it without a mask.
        let mut inv = vec![0u32; sentinel as usize];
        for (dstrow, &srcrow) in perm.iter().enumerate() {
            if srcrow != sentinel {
                inv[srcrow as usize] = dstrow as u32;
            }
        }
        WindowPlan {
            grid_h,
            grid_w,
            win_h,
            win_w,
            shift_h: 0,
            shift_w: 0,
            pad_h,
            pad_w,
            perm,
            inv,
            spans,
            max_span: win_h * win_w,
            uniform: true,
        }
    }

    /// window-major row -> grid row (the [`gather_rows`] index: partition).
    pub fn perm(&self) -> &[u32] {
        &self.perm
    }
    /// grid row -> window-major row (the [`gather_rows`] index: reverse; also
    /// the [`scatter_rows`] index for the partition).
    pub fn inv(&self) -> &[u32] {
        &self.inv
    }
    /// `(row0, len)` spans over the WINDOW-MAJOR buffer, one per window - feed
    /// straight to [`vit_block_fwd`] / [`vit_block_fwd_cached`].
    pub fn spans(&self) -> &[(u32, u32)] {
        &self.spans
    }
    pub fn rows(&self) -> u32 {
        self.grid_h * self.grid_w
    }
    /// Rows of the WINDOW-MAJOR buffer - `pad_h*pad_w`, i.e. [`Self::rows`] for
    /// every partition except [`WindowPlan::padded`]'s, where it is larger by
    /// exactly the pad. This, not [`Self::rows`], sizes the windowed buffer and
    /// the [`window_partition`] gather.
    pub fn win_rows(&self) -> u32 {
        self.perm.len() as u32
    }
    /// The padded grid `(pad_h, pad_w)` the windows tile - `(grid_h, grid_w)`
    /// unless this is a [`WindowPlan::padded`] plan.
    pub fn padded_grid(&self) -> (u32, u32) {
        (self.pad_h, self.pad_w)
    }
    /// The out-of-grid row index [`Self::perm`] uses for padded positions
    /// (`rows()`, one past the last real row), or `None` when this partition
    /// has no pad. `Some` means the caller must allocate `rows()+1` rows and
    /// zero the last one before the qkv projection - see [`WindowPlan::padded`].
    pub fn sentinel(&self) -> Option<u32> {
        (self.win_rows() > self.rows()).then(|| self.rows())
    }
    /// Longest window, for [`VitScratch::new`] / [`VitBlockCache::new`].
    pub fn max_span(&self) -> u32 {
        self.max_span
    }
    pub fn n_windows(&self) -> u32 {
        self.spans.len() as u32
    }
    /// True when every window is `win_h*win_w` (no shift, grid divisible).
    /// [`QPoolPlan::per_window`] requires it: a batched max-pool has one shape.
    pub fn is_uniform(&self) -> bool {
        self.uniform
    }

    /// Can this partition's spans be run through the CACHED attention path
    /// ([`vit_block_fwd_cached`] / [`vit_block_bwd`] / [`cross_q_fwd`]) at
    /// channel width `dim`?
    ///
    /// `ctx` is the one binding those still slice per span (the apply kernel
    /// has no output-offset Param), so every span's `row0*dim` must be
    /// 64-float aligned. Always true for `dim % 64 == 0`; check it before
    /// building a shifted plan rather than meeting [`aligned_ctx`] at
    /// step-build time. The unchunked inference path [`vit_block_fwd`] does not
    /// slice `ctx` per span and is unaffected.
    pub fn ctx_bindable(&self, dim: u32) -> bool {
        self.spans.iter().all(|&(row0, _)| (row0 as u64 * dim as u64).is_multiple_of(BIND_ALIGN))
    }
}

/// The two device index buffers of a [`WindowPlan`], built once and reused by
/// every block that shares the partition.
pub struct WindowIndex {
    /// window-major row -> grid row.
    pub fwd: DeviceBuffer,
    /// grid row -> window-major row.
    pub inv: DeviceBuffer,
    /// Real grid rows ([`WindowPlan::rows`]) - what [`window_reverse`] emits.
    pub rows: u32,
    /// Window-major rows ([`WindowPlan::win_rows`]) - what [`window_partition`]
    /// emits. Larger than `rows` only for a [`WindowPlan::padded`] plan.
    pub win_rows: u32,
}

impl WindowIndex {
    pub fn new(g: &Gpu, plan: &WindowPlan) -> WindowIndex {
        WindowIndex {
            fwd: row_index_buffer(g, "win_perm", plan.perm()),
            inv: row_index_buffer(g, "win_perm_inv", plan.inv()),
            rows: plan.rows(),
            win_rows: plan.win_rows(),
        }
    }
}

/// Grid order -> window-major order (`window_partition`), `[win_rows, c]`.
/// Adjoint/inverse: [`window_reverse`] with the same [`WindowIndex`].
///
/// For a [`WindowPlan::padded`] plan `src` must carry `rows+1` rows with the
/// last one zeroed (the sentinel the pad indexes) and `dst` `win_rows` - see
/// that constructor's doc. Every other plan has `win_rows == rows`.
pub fn window_partition(g: &Gpu, ids: &VitPermuteIds, w: &WindowIndex, src: &DeviceBuffer, dst: &DeviceBuffer, c: u32) -> Step {
    gather_rows(g, ids, &w.fwd, src, dst, w.win_rows, c)
}

/// Window-major order -> grid order (`window_reverse`), `[rows, c]`.
///
/// Uses the gather (`embed`) with the INVERSE index rather than the scatter, so
/// both directions have the same coalescing on the write side; `scatter_rows`
/// with `w.fwd` computes the identical result and is the form to use when the
/// destination is a gradient buffer whose other rows must survive.
pub fn window_reverse(g: &Gpu, ids: &VitPermuteIds, w: &WindowIndex, src: &DeviceBuffer, dst: &DeviceBuffer, c: u32) -> Step {
    gather_rows(g, ids, &w.inv, src, dst, w.rows, c)
}

// ---------------------------------------------------------------------------
// Hiera q_pool
// ---------------------------------------------------------------------------

/// Kernel ids for [`q_pool_fwd`] / [`q_pool_bwd`]. All five already exist; the
/// max-pool pair landed with the imaging workstream's phase 0.
#[derive(Clone, Copy)]
pub struct VitQPoolIds {
    pub permute: VitPermuteIds,
    /// `nlc_nchw`: `[N, L, C] -> [N, C, H, W]`.
    pub nlc_nchw: usize,
    /// `nchw_nlc`: `[N, C, H, W] -> [N, L, C]`. Exact inverse AND adjoint of
    /// `nlc_nchw`, which is why the backward needs no extra kernel.
    pub nchw_nlc: usize,
    pub maxpool2d: usize,
    pub maxpool2d_dx: usize,
}

/// Hiera `q_pool`: a spatial `MaxPool2d` over the query token grid, applied
/// INSIDE attention while keys/values stay at full resolution.
///
/// `n` is the number of independent grids pooled in one dispatch. Hiera pools
/// per WINDOW (`window_partition` runs before the attention), so `n` is the
/// window count and `h`/`w` the window extent; `n = 1`, `h/w = grid` is the
/// unwindowed (global-attention stage) case.
#[derive(Clone, Copy, Debug)]
pub struct QPoolPlan {
    pub n: u32,
    pub h: u32,
    pub w: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
}

impl QPoolPlan {
    /// Hiera's `MaxPool2d(kernel_size=2, stride=2, padding=0)` over each window
    /// of a UNIFORM [`WindowPlan`]. Ragged (shifted) windows are rejected: a
    /// batched pool has one `(h, w)`, and Hiera does not shift anyway.
    pub fn per_window(plan: &WindowPlan, k: u32, stride: u32, pad: u32) -> QPoolPlan {
        assert!(plan.is_uniform(), "q_pool needs uniform windows; this plan is ragged (shift or indivisible grid)");
        QPoolPlan { n: plan.n_windows(), h: plan.win_h, w: plan.win_w, k, stride, pad }
    }
    /// Whole-grid pooling (no windowing).
    pub fn global(grid_h: u32, grid_w: u32, k: u32, stride: u32, pad: u32) -> QPoolPlan {
        QPoolPlan { n: 1, h: grid_h, w: grid_w, k, stride, pad }
    }
    /// Both extents are checked: `h + 2*pad < k` underflows `u32`, which panics
    /// in debug but WRAPS in release into an enormous `ho` that the max-pool
    /// then happily dispatches against.
    pub fn ho(&self) -> u32 {
        assert!(self.stride > 0 && self.h + 2 * self.pad >= self.k, "q_pool: kernel {} does not fit h {} + 2*pad {}", self.k, self.h, self.pad);
        (self.h + 2 * self.pad - self.k) / self.stride + 1
    }
    pub fn wo(&self) -> u32 {
        assert!(self.stride > 0 && self.w + 2 * self.pad >= self.k, "q_pool: kernel {} does not fit w {} + 2*pad {}", self.k, self.w, self.pad);
        (self.w + 2 * self.pad - self.k) / self.stride + 1
    }
    /// Query rows entering the pool.
    pub fn rows_in(&self) -> u32 {
        self.n * self.h * self.w
    }
    /// Query rows leaving it - the `qn` of every [`AttnSpan`].
    pub fn rows_out(&self) -> u32 {
        self.n * self.ho() * self.wo()
    }
    /// Pooled rows per window (`ho*wo`).
    pub fn win_rows_out(&self) -> u32 {
        self.ho() * self.wo()
    }
}

/// SSA cache for one block's `q_pool` stage: every intermediate lands in its
/// own buffer, and `argmax` is the one the backward genuinely needs (the
/// max-pool adjoint is a gather through the recorded winner).
pub struct QPoolCache {
    /// `[rows_in, c]` - q region gathered out of the fused qkv (NLC).
    pub q_c: DeviceBuffer,
    /// `[n, c, h, w]`.
    pub q_nchw: DeviceBuffer,
    /// `[n, c, ho, wo]`.
    pub qp_nchw: DeviceBuffer,
    /// `[n, c, ho, wo]` - winning INPUT flat index per output, as f32.
    pub argmax: DeviceBuffer,
    /// `[rows_out, c]` - the attention query buffer (NLC, stride `c`).
    pub q_pooled: DeviceBuffer,
    /// `[n, c, ho, wo]` backward staging.
    pub d_qp_nchw: DeviceBuffer,
    /// `[n, c, h, w]` backward staging.
    pub d_q_nchw: DeviceBuffer,
    /// `[rows_in, c]` backward staging.
    pub d_q_c: DeviceBuffer,
}

impl QPoolCache {
    pub fn new(gpu: &Gpu, plan: &QPoolPlan, c: u32) -> QPoolCache {
        let ni = plan.rows_in() as u64 * c as u64;
        let no = plan.rows_out() as u64 * c as u64;
        // `maxpool2d` stores the winner's input flat index in an f32 - exact
        // only while N*C*H*W < 2^24. Beyond that the pool silently routes the
        // gradient to a NEIGHBOURING pixel, which no test would notice.
        assert!(ni < (1u64 << 24), "q_pool argmax needs n*c*h*w = {ni} < 2^24 (f32 index exactness)");
        QPoolCache {
            q_c: gpu.storage(ni),
            q_nchw: gpu.storage(ni),
            qp_nchw: gpu.storage(no),
            argmax: gpu.storage(no),
            q_pooled: gpu.storage(no),
            d_qp_nchw: gpu.storage(no),
            d_q_nchw: gpu.storage(ni),
            d_q_c: gpu.storage(ni),
        }
    }
}

/// Pool the query: fused `qkv[rows_in, 3c]` -> `cache.q_pooled[rows_out, c]`.
///
/// Four dispatches, no new kernel:
///   `embed` (q region -> compact NLC) -> `nlc_nchw` -> `maxpool2d` ->
///   `nchw_nlc`. `q_idx` is [`region_index`]`(rows_in, 3, 0)` uploaded with
///   [`row_index_buffer`].
///
/// Kernel Params, in order (a mismatched list here is silently wrong):
///   * `embed`      `[c, rows_in]`, bufs `[q_idx, qkv, q_c]`
///   * `nlc_nchw`   `[n*c*h*w, c, h*w]`, bufs `[q_c, q_nchw]`
///   * `maxpool2d`  `[n, c, h, w, K, stride, pad, ho, wo]`,
///     bufs `[q_nchw, qp_nchw, argmax]` - note `stride` sits BEFORE `pad`
///   * `nchw_nlc`   `[n*c*ho*wo, c, ho*wo]`, bufs `[qp_nchw, q_pooled]`
pub fn q_pool_fwd(
    g: &Gpu,
    ids: &VitQPoolIds,
    plan: &QPoolPlan,
    c: u32,
    qkv: &DeviceBuffer,
    q_idx: &DeviceBuffer,
    cache: &QPoolCache,
    steps: &mut Vec<Step>,
) {
    let (ho, wo) = (plan.ho(), plan.wo());
    let ti = plan.n * c * plan.h * plan.w;
    let to = plan.n * c * ho * wo;
    steps.push(gather_rows(g, &ids.permute, q_idx, qkv, &cache.q_c, plan.rows_in(), c));
    steps.push(g.step(ids.nlc_nchw, &[&cache.q_c, &cache.q_nchw], &[ti, c, plan.h * plan.w], ti));
    steps.push(g.step(
        ids.maxpool2d,
        &[&cache.q_nchw, &cache.qp_nchw, &cache.argmax],
        &[plan.n, c, plan.h, plan.w, plan.k, plan.stride, plan.pad, ho, wo],
        to,
    ));
    steps.push(g.step(ids.nchw_nlc, &[&cache.qp_nchw, &cache.q_pooled], &[to, c, ho * wo], to));
}

/// Adjoint of [`q_pool_fwd`]: `d_q_pooled[rows_out, c]` -> the q region of
/// `d_qkv[rows_in, 3c]`. The four steps are the four forward steps reversed,
/// each replaced by its own adjoint - `nchw_nlc` <-> `nlc_nchw` are each
/// other's, `maxpool2d_dx` gathers through `cache.argmax`, and `row_scatter` is
/// `embed`'s. Nothing here is a new gradient form.
///
/// `d_qkv`'s k/v regions are left untouched (the cross-attention `dk`/`dv`
/// kernels own them), so it does NOT need pre-zeroing for this call.
///
/// Kernel Params:
///   * `nlc_nchw`     `[n*c*ho*wo, c, ho*wo]`, bufs `[d_q_pooled, d_qp_nchw]`
///   * `maxpool2d_dx` `[n, c, h, w, K, stride, pad, ho, wo]`,
///     bufs `[d_qp_nchw, argmax, d_q_nchw]`, threads `n*c*h*w`
///   * `nchw_nlc`     `[n*c*h*w, c, h*w]`, bufs `[d_q_nchw, d_q_c]`
///   * `row_scatter`  `[rows_in, c, 3*rows_in]`, bufs `[q_idx, d_q_c, d_qkv]`
pub fn q_pool_bwd(
    g: &Gpu,
    ids: &VitQPoolIds,
    plan: &QPoolPlan,
    c: u32,
    d_q_pooled: &DeviceBuffer,
    q_idx: &DeviceBuffer,
    d_qkv: &DeviceBuffer,
    cache: &QPoolCache,
    steps: &mut Vec<Step>,
) {
    let (ho, wo) = (plan.ho(), plan.wo());
    let ti = plan.n * c * plan.h * plan.w;
    let to = plan.n * c * ho * wo;
    steps.push(g.step(ids.nlc_nchw, &[d_q_pooled, &cache.d_qp_nchw], &[to, c, ho * wo], to));
    steps.push(g.step(
        ids.maxpool2d_dx,
        &[&cache.d_qp_nchw, &cache.argmax, &cache.d_q_nchw],
        &[plan.n, c, plan.h, plan.w, plan.k, plan.stride, plan.pad, ho, wo],
        ti,
    ));
    steps.push(g.step(ids.nchw_nlc, &[&cache.d_q_nchw, &cache.d_q_c], &[ti, c, plan.h * plan.w], ti));
    steps.push(scatter_rows(g, &ids.permute, q_idx, &cache.d_q_c, d_qkv, plan.rows_in(), c, 3 * plan.rows_in()));
}

// ---------------------------------------------------------------------------
// Attention with a separate (pooled) query buffer
// ---------------------------------------------------------------------------

/// One attention group with INDEPENDENT query and key/value extents: `qn` query
/// rows from `q0` attend `kn` key/value rows from `k0`. Hiera's pooled window
/// `m` is `{ q0: m*ho*wo, qn: ho*wo, k0: m*win, kn: win }`.
///
/// `block::chunked_bidir_fwd`'s `(row0, len)` span is the special case
/// `q0 == k0`, `qn == kn`.
#[derive(Clone, Copy, Debug)]
pub struct AttnSpan {
    pub q0: u32,
    pub qn: u32,
    pub k0: u32,
    pub kn: u32,
}

impl AttnSpan {
    /// Self-attention span (`q == kv`), i.e. a `(row0, len)` pair.
    pub fn span(row0: u32, len: u32) -> AttnSpan {
        AttnSpan { q0: row0, qn: len, k0: row0, kn: len }
    }
    /// Every window of `plan` with its query rows pooled by `pool`.
    pub fn pooled_windows(plan: &WindowPlan, pool: &QPoolPlan) -> Vec<AttnSpan> {
        assert!(plan.is_uniform(), "pooled windows need a uniform partition");
        assert_eq!(plan.n_windows(), pool.n, "pool.n must be the window count");
        let qw = pool.win_rows_out();
        plan.spans()
            .iter()
            .enumerate()
            .map(|(m, &(row0, len))| AttnSpan { q0: m as u32 * qw, qn: qw, k0: row0, kn: len })
            .collect()
    }
}

/// A storage-binding offset must be a multiple of the adapter's
/// `min_storage_buffer_offset_alignment` - 256 bytes = 64 f32 on every backend
/// brain runs on, and wgpu REJECTS the bind group otherwise ("Buffer offset N
/// does not respect ... limit 256"). That is a hard constraint on every
/// `step_sliced` offset below, not a style preference.
const BIND_ALIGN: u64 = 64;

fn align_up(n: u64) -> u64 {
    n.div_ceil(BIND_ALIGN) * BIND_ALIGN
}

/// Base offsets of each span's `[heads, qn, kn]` softmax slab in a cached
/// `probs` buffer.
///
/// Running prefix sum with each slab PADDED to [`BIND_ALIGN`]. Both halves
/// matter: the running sum is what makes RAGGED spans (a shifted Swin
/// partition, a border window) address the right slab, and the padding is what
/// makes the resulting offset bindable at all. `heads*qn*kn` is very often not
/// a multiple of 64 - 16 heads × 729² is not - so an unpadded prefix sum is a
/// driver validation failure at submit for ragged spans AND for perfectly
/// uniform ones.
pub fn probs_offsets(spans: &[AttnSpan], heads: u32) -> Vec<u64> {
    let mut off = 0u64;
    spans
        .iter()
        .map(|s| {
            let at = off;
            off += align_up(heads as u64 * s.qn as u64 * s.kn as u64);
            at
        })
        .collect()
}

/// Floats a cached `probs` buffer needs for `spans`, padding included.
pub fn probs_len(spans: &[AttnSpan], heads: u32) -> u64 {
    spans.iter().map(|s| align_up(heads as u64 * s.qn as u64 * s.kn as u64)).sum()
}

/// Largest `heads*qn*kn` over `spans` - the transient `scores`/`dscores` slab.
/// Bound at offset 0 every dispatch, so it needs no padding.
pub fn max_slab(spans: &[AttnSpan], heads: u32) -> u64 {
    spans.iter().map(|s| heads as u64 * s.qn as u64 * s.kn as u64).max().unwrap_or(0)
}

/// A row offset folded into a kernel's own `q_off`/`k_off`/`v_off` Param.
/// Params are `u32` and the kernels index `array<f32>` with `u32`, so this is
/// the same ceiling the kernel already has - asserted where the shape that
/// produced it is still in scope.
fn row_param(rows: u32, stride: u32, region: u32) -> u32 {
    let v = rows as u64 * stride as u64 + region as u64;
    assert!(v <= u32::MAX as u64, "attention row offset {v} overflows the kernels' u32 addressing");
    v as u32
}

/// `ctx`/`d_ctx` is the ONE buffer of the cross trio with no offset Param -
/// `attn_apply_cross` writes `out[(b*Tq + i)*d_model + …]` - so it must be
/// bound sliced, and its offset must satisfy [`BIND_ALIGN`]. Checked here, with
/// the shape that produced it, because the driver's complaint arrives at submit
/// with no hint of which span caused it.
fn aligned_ctx(q0: u32, d_out: u32) -> u64 {
    let off = q0 as u64 * d_out as u64;
    assert!(
        off.is_multiple_of(BIND_ALIGN),
        "ctx binding offset q0*C = {q0}*{d_out} = {off} floats is not 64-float (256B) aligned; \
         a ragged/shifted partition needs C to be a multiple of 64, or every span's q0 to be a \
         multiple of {}",
        BIND_ALIGN / num_gcd(d_out as u64, BIND_ALIGN)
    );
    off
}

/// Greatest common divisor - used only to phrase the [`aligned_ctx`] message.
fn num_gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a.max(1)
    } else {
        num_gcd(b, a % b)
    }
}

/// Bidirectional attention where the QUERY lives in its own buffer with its own
/// length - Hiera's `q_pool`, and the general two-length case.
///
/// One dispatch per span, caching the full `[heads, qn, kn]` softmax at
/// `probs_offsets(spans, heads)[i]`, exactly like [`vit_block_fwd_cached`]'s
/// attention. There is no query chunking: pooled-q attention is window-local
/// (a Hiera window is at most 14x14 keys), so the slab is already small. A
/// future model that pools q over a whole-image span wants the chunk loop from
/// `block::chunked_bidir_fwd` added HERE, not a second copy of this function.
///
/// This is the ONE per-span cross-attention forward in `vit`:
/// [`vit_block_fwd_cached`] calls it with `AttnSpan::span(row0, len)` for plain
/// self-attention. It also generalizes `block::chunked_bidir_fwd` (which binds
/// one buffer for q and kv, ties `qn` to `kn`, and keeps `probs` as a single
/// reused scratch slab instead of caching one per span, so that one is a
/// forward-only inference path and NOT a duplicate of this).
///
/// ROW OFFSETS RIDE IN THE PARAMS, NOT IN THE BINDING. `q`/`kv` are bound whole
/// and the span's row offset is folded into the kernels' own `q_off`/`k_off`/
/// `v_off`. Slicing them instead - `(q0*q_stride, 0)` - imposes a 256-byte
/// alignment on `row0 * stride` that a ragged partition does not satisfy, which
/// is a hard wgpu validation failure, not a slow path. `ctx` has no offset
/// Param and is the one binding still sliced (see [`aligned_ctx`]).
///
/// Kernel Params (`CrossIds`), per span:
///   * `attn_scores_cross`  `[1, heads, qn, kn, hd, q_stride, kv_stride,
///     q0*q_stride + q_off, k0*kv_stride + k_off]`, bufs `[q, kv, scores]`
///     bound whole, threads `heads*qn*kn`
///   * `attn_softmax_cross` `[1, heads, qn, kn]`, bufs `[scores, probs]`,
///     slices `[(0,0), (probs_at,0)]`, threads `heads*qn`
///   * `attn_apply_cross`   `[1, heads, qn, kn, hd, kv_stride,
///     k0*kv_stride + v_off, d_out]`, bufs `[probs, kv, ctx]`,
///     slices `[(probs_at,0), (0,0), (q0*d_out,0)]`, threads `heads*qn*hd`
pub fn cross_q_fwd(
    g: &Gpu,
    ids: &crate::block::CrossIds,
    sh: &VitShape,
    q: &DeviceBuffer,
    q_stride: u32,
    q_off: u32,
    kv: &DeviceBuffer,
    kv_stride: u32,
    k_off: u32,
    v_off: u32,
    ctx: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    spans: &[AttnSpan],
    steps: &mut Vec<Step>,
) {
    let (heads, hd, d_out) = (sh.heads, sh.head_dim(), sh.dim);
    let at = probs_offsets(spans, heads);
    for (i, s) in spans.iter().enumerate() {
        let qo = row_param(s.q0, q_stride, q_off);
        let ko = row_param(s.k0, kv_stride, k_off);
        let vo = row_param(s.k0, kv_stride, v_off);
        let cs = aligned_ctx(s.q0, d_out);
        let ps = at[i];
        steps.push(g.step(
            ids.scores,
            &[q, kv, scores],
            &[1, heads, s.qn, s.kn, hd, q_stride, kv_stride, qo, ko],
            heads * s.qn * s.kn,
        ));
        steps.push(g.step_sliced(ids.softmax, &[scores, probs], &[(0, 0), (ps, 0)], &[1, heads, s.qn, s.kn], heads * s.qn));
        steps.push(g.step_sliced(
            ids.apply,
            &[probs, kv, ctx],
            &[(ps, 0), (0, 0), (cs, 0)],
            &[1, heads, s.qn, s.kn, hd, kv_stride, vo, d_out],
            heads * s.qn * hd,
        ));
    }
}

/// Backward of [`cross_q_fwd`] from the cached `probs` (no recompute), writing
/// `d_q` in the query buffer's layout and `d_k`/`d_v` into the kv buffer's
/// regions of `d_kv`. Spans must cover DISJOINT q rows and DISJOINT kv rows -
/// which windows do - because these four kernels ASSIGN.
///
/// `d_q`/`d_kv` are fully written for the rows the spans cover, so neither
/// needs pre-zeroing when the spans partition the rows.
///
/// Row offsets ride in the Params exactly as in [`cross_q_fwd`], so `q`, `kv`,
/// `d_q` and `d_kv` are all bound whole and only `d_ctx` (no offset Param) and
/// `probs` are sliced.
///
/// Kernel Params (`VitBwdIds`), per span, with `vo = k0*kv_stride + v_off`,
/// `ko = k0*kv_stride + k_off`, `qo = q0*q_stride + q_off`:
///   * `attn_bwd_dscores_cross` `[1, heads, qn, kn, hd, kv_stride, vo, d_out]`
///     bufs `[d_ctx, kv, probs, dscores]`, threads `heads*qn`
///   * `attn_bwd_dv_cross`      same 8 words,
///     bufs `[probs, d_ctx, d_kv]`, threads `heads*kn*hd`
///   * `attn_bwd_dq_cross`      `[1, heads, qn, kn, hd, q_stride, kv_stride, qo, ko]`
///     bufs `[dscores, kv, d_q]`, threads `heads*qn*hd`
///   * `attn_bwd_dk_cross`      same 9 words,
///     bufs `[dscores, q, d_kv]`, threads `heads*kn*hd`
pub fn cross_q_bwd(
    g: &Gpu,
    kb: &VitBwdIds,
    sh: &VitShape,
    q: &DeviceBuffer,
    q_stride: u32,
    q_off: u32,
    kv: &DeviceBuffer,
    kv_stride: u32,
    k_off: u32,
    v_off: u32,
    probs: &DeviceBuffer,
    d_ctx: &DeviceBuffer,
    d_q: &DeviceBuffer,
    d_kv: &DeviceBuffer,
    dscores: &DeviceBuffer,
    spans: &[AttnSpan],
    steps: &mut Vec<Step>,
) {
    let (heads, hd, d_out) = (sh.heads, sh.head_dim(), sh.dim);
    let at = probs_offsets(spans, heads);
    for (i, s) in spans.iter().enumerate() {
        let qo = row_param(s.q0, q_stride, q_off);
        let ko = row_param(s.k0, kv_stride, k_off);
        let vo = row_param(s.k0, kv_stride, v_off);
        let cs = aligned_ctx(s.q0, d_out);
        let ps = at[i];
        let p_v = [1, heads, s.qn, s.kn, hd, kv_stride, vo, d_out];
        let p_qk = [1, heads, s.qn, s.kn, hd, q_stride, kv_stride, qo, ko];
        steps.push(g.step_sliced(
            kb.attn_bwd_dscores_cross,
            &[d_ctx, kv, probs, dscores],
            &[(cs, 0), (0, 0), (ps, 0), (0, 0)],
            &p_v,
            heads * s.qn,
        ));
        steps.push(g.step_sliced(kb.attn_bwd_dv_cross, &[probs, d_ctx, d_kv], &[(ps, 0), (cs, 0), (0, 0)], &p_v, heads * s.kn * hd));
        steps.push(g.step(kb.attn_bwd_dq_cross, &[dscores, kv, d_q], &p_qk, heads * s.qn * hd));
        steps.push(g.step(kb.attn_bwd_dk_cross, &[dscores, q, d_kv], &p_qk, heads * s.kn * hd));
    }
}

// ===========================================================================
// Decomposed relative position bias (SAM ViT-B / ViTDet)
// ===========================================================================
//
// SAM's windowed attention adds a learned bias that DECOMPOSES over the two
// grid axes - `add_decomposed_rel_pos` in the reference:
//
//   scores[(qh,qw), (kh,kw)] = scale * (q . k)
//                            + q[(qh,qw),:] . Rh[qh,kh,:]
//                            + q[(qh,qw),:] . Rw[qw,kw,:]
//
// with q UNSCALED in the two bias terms (`attn_scores_cross` applies
// `1/sqrt(head_dim)` itself, so nothing pre-scales the fused buffer's q
// region - the layout is already what the bias wants).
//
// THREE OBSERVATIONS SHAPE THE IMPLEMENTATION, and each one removes work:
//
//  1. **The bias is a function of q, so it is never materialised.** Both terms
//     factor through the small intermediates `rel_h[h, i, kh]` and
//     `rel_w[h, i, kw]` (`attn_relpos_qr`), which are then FOLDED into the
//     already-computed score slab in place (`attn_relpos_add`). At SAM ViT-B's
//     global-attention shape a materialised `[heads, T, T]` bias would be
//     805 MB; the two intermediates are 12.6 MB each.
//
//  2. **The row -> grid map is arithmetic, not an index buffer.**
//     [`WindowPlan`] emits window-local ROW-MAJOR order, so a span-local query
//     row `i` sits at `(i / qw, i % qw)`. A query-CHUNKED dispatch therefore
//     needs exactly one new parameter, `q0` - the chunk's span-local first
//     query row.
//
//  3. **`get_rel_pos`'s table resample and its shifted gather are both linear
//     and data-INDEPENDENT**, so their composition is a fixed 2-tap weighted
//     gather. [`rel_pos_gather`] computes the indices and weights on the host
//     (pure integer/float arithmetic, no device), and the dense table is then
//     built and differentiated with kernels this repo already has -
//     `embed` + `scale_row` + `add2` forward, `scale_row` + `emb_bwd` back.
//     No kernel was added for the interpolation.
//
// Two dense-table LAYOUTS are kept, because the forward and the backward want
// opposite fastest axes and a table is at most `[64, 64, 64]` at real scale:
// `r` is `[q_ext, k_ext, head_dim]` (what the gather naturally produces, what
// `attn_relpos_dq`/`_dr` want, and what the `emb_bwd` chain consumes) and
// `r_t = nlc_nchw(r)` is `[q_ext, head_dim, k_ext]` (what `attn_relpos_qr`
// wants). Because the backward writes `r`'s layout DIRECTLY, the transpose is
// forward-only and needs no adjoint.

/// `get_rel_pos`'s 2-tap gather, host-side: for each `(qi, kj)` pair, the two
/// learned-table rows it reads and their weights.
///
/// The reference does this in two steps - resample the `[L, head_dim]` table to
/// `2*max(q,k)-1` rows with `F.interpolate(..., mode="linear")`, then gather
/// row `floor(qi*max(k/q,1) - kj*max(q/k,1) + (k-1)*max(q/k,1))` - and both are
/// linear maps that depend only on `(q_size, k_size, table_len)`. Composed,
/// every output row is `w0*T[idx0] + w1*T[idx1]`.
///
/// The interpolation is PyTorch's `align_corners=False` half-pixel rule:
/// `src = (dst + 0.5)*(L/max_rel) - 0.5`, clamped at 0, `idx0 = floor(src)`,
/// `idx1 = idx0 + 1` unless `idx0` is the last row (then `idx1 == idx0`, which
/// is why the two weights still sum to 1 and the pair still reproduces
/// `T[idx0]`). `table_len == 2*max(q,k)-1` short-circuits to the identity, so
/// the no-resample case is exact rather than round-tripped through floats.
#[derive(Clone, Debug)]
pub struct RelPosGather {
    /// `[q_size * k_size]`, row-major in `(qi, kj)`.
    pub idx0: Vec<u32>,
    pub idx1: Vec<u32>,
    pub w0: Vec<f32>,
    pub w1: Vec<f32>,
}

impl RelPosGather {
    pub fn len(&self) -> usize {
        self.idx0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.idx0.is_empty()
    }
}

pub fn rel_pos_gather(q_size: u32, k_size: u32, table_len: u32) -> RelPosGather {
    assert!(q_size > 0 && k_size > 0, "rel_pos_gather: empty grid {q_size}x{k_size}");
    assert!(table_len > 0, "rel_pos_gather: empty table");
    let max_rel = 2 * q_size.max(k_size) - 1;

    // ---- step 1: the resample, as (row, row, weight) per resampled row ----
    let mut t0 = vec![0u32; max_rel as usize];
    let mut t1 = vec![0u32; max_rel as usize];
    let mut tf = vec![0f32; max_rel as usize];
    if table_len == max_rel {
        for d in 0..max_rel as usize {
            t0[d] = d as u32;
            t1[d] = d as u32;
        }
    } else {
        // `scale = in/out` is what PyTorch computes when no scale is passed.
        let scale = table_len as f64 / max_rel as f64;
        for d in 0..max_rel as usize {
            let src = ((d as f64 + 0.5) * scale - 0.5).max(0.0);
            let lo = (src.floor() as u32).min(table_len - 1);
            let hi = if lo + 1 < table_len { lo + 1 } else { lo };
            t0[d] = lo;
            t1[d] = hi;
            tf[d] = (src - lo as f64) as f32;
        }
    }

    // ---- step 2: the shifted gather, composed onto step 1 ----
    let sq = (k_size as f64 / q_size as f64).max(1.0);
    let sk = (q_size as f64 / k_size as f64).max(1.0);
    let n = (q_size * k_size) as usize;
    let mut g = RelPosGather {
        idx0: Vec::with_capacity(n),
        idx1: Vec::with_capacity(n),
        w0: Vec::with_capacity(n),
        w1: Vec::with_capacity(n),
    };
    for qi in 0..q_size {
        for kj in 0..k_size {
            let rel = qi as f64 * sq - kj as f64 * sk + (k_size as f64 - 1.0) * sk;
            // `.long()` on a non-negative value; the +1e-9 absorbs the f64
            // round-off of an exact integer product (e.g. 3*(4/3.0)).
            let d = (rel + 1e-9) as usize;
            assert!(d < max_rel as usize, "rel_pos_gather: index {d} out of range {max_rel}");
            g.idx0.push(t0[d]);
            g.idx1.push(t1[d]);
            g.w0.push(1.0 - tf[d]);
            g.w1.push(tf[d]);
        }
    }
    g
}

/// The five EXISTING kernels the dense-table build and its adjoint compose
/// from. No `attn_relpos_*` kernel appears here on purpose - see the section
/// header: the interpolation added none.
#[derive(Clone, Copy)]
pub struct RelPosTableIds {
    /// `embed` - the 2-tap gather's two lookups.
    pub embed: usize,
    /// `scale_row` - the per-(qi,kj) interpolation weight.
    pub scale_row: usize,
    pub add2: usize,
    /// `nlc_nchw` - `r` -> `r_t`, the forward's fastest-axis transpose.
    pub nlc_nchw: usize,
    /// `emb_bwd` - the table scatter; ACCUMULATES, which is what makes the two
    /// taps (and repeated gather indices) sum correctly.
    pub emb_bwd: usize,
}

/// One axis's device-side gather, uploaded once and reused by every forward.
pub struct RelPosAxis {
    pub idx0: DeviceBuffer,
    pub idx1: DeviceBuffer,
    pub w0: DeviceBuffer,
    pub w1: DeviceBuffer,
    /// `[q_ext * k_ext, head_dim]` - the dense table, natural layout.
    pub r: DeviceBuffer,
    /// `[q_ext, head_dim, k_ext]` - the same table, `k` fastest.
    pub r_t: DeviceBuffer,
    /// `[q_ext * k_ext, head_dim]` staging for the two weighted taps.
    a: DeviceBuffer,
    b: DeviceBuffer,
    a2: DeviceBuffer,
    b2: DeviceBuffer,
    pub q_ext: u32,
    pub k_ext: u32,
    pub head_dim: u32,
    pub table_len: u32,
}

impl RelPosAxis {
    /// Build one axis from its geometry and the learned table's row count.
    pub fn new(g: &Gpu, q_ext: u32, k_ext: u32, head_dim: u32, table_len: u32) -> RelPosAxis {
        let gather = rel_pos_gather(q_ext, k_ext, table_len);
        let n = gather.len() as u64;
        let dense = n * head_dim as u64;
        RelPosAxis {
            idx0: row_index_buffer(g, "relpos_idx0", &gather.idx0),
            idx1: row_index_buffer(g, "relpos_idx1", &gather.idx1),
            w0: g.storage_init("relpos_w0", &gather.w0),
            w1: g.storage_init("relpos_w1", &gather.w1),
            r: g.storage(dense),
            r_t: g.storage(dense),
            a: g.storage(dense),
            b: g.storage(dense),
            a2: g.storage(dense),
            b2: g.storage(dense),
            q_ext,
            k_ext,
            head_dim,
            table_len,
        }
    }

    fn n(&self) -> u32 {
        self.q_ext * self.k_ext
    }

    /// `table[table_len, head_dim]` -> `r` and `r_t`. Six dispatches, all of
    /// existing kernels. Re-run whenever the learned table changes (i.e. every
    /// training step; an inference-only model builds it once).
    pub fn build_fwd(&self, g: &Gpu, k: &RelPosTableIds, table: &DeviceBuffer, steps: &mut Vec<Step>) {
        let (n, hd) = (self.n(), self.head_dim);
        let total = n * hd;
        steps.push(g.step(k.embed, &[&self.idx0, table, &self.a], &[hd, n], total));
        steps.push(g.step(k.embed, &[&self.idx1, table, &self.b], &[hd, n], total));
        steps.push(g.step(k.scale_row, &[&self.a, &self.w0, &self.a2], &[total, hd], total));
        steps.push(g.step(k.scale_row, &[&self.b, &self.w1, &self.b2], &[total, hd], total));
        steps.push(g.step(k.add2, &[&self.a2, &self.b2, &self.r], &[total], total));
        steps.push(g.step(k.nlc_nchw, &[&self.r, &self.r_t], &[total, hd, self.k_ext], total));
    }

    /// `d_r` -> `d_table`, ACCUMULATING (`emb_bwd` adds, so `d_table` must be
    /// zero-cleared once per step by the caller's submit, exactly like any
    /// other embedding gradient). `scratch` is `[q_ext*k_ext, head_dim]`.
    ///
    /// The forward's `nlc_nchw` needs no adjoint here: `attn_relpos_dr` writes
    /// `d_r` in `r`'s own layout, so `r_t` is a forward-only convenience copy.
    pub fn build_bwd(
        &self,
        g: &Gpu,
        k: &RelPosTableIds,
        d_r: &DeviceBuffer,
        d_table: &DeviceBuffer,
        scratch: &DeviceBuffer,
        steps: &mut Vec<Step>,
    ) {
        let (n, hd) = (self.n(), self.head_dim);
        let total = n * hd;
        steps.push(g.step(k.scale_row, &[d_r, &self.w0, scratch], &[total, hd], total));
        steps.push(g.step(k.emb_bwd, &[&self.idx0, scratch, d_table], &[n, hd, self.table_len], self.table_len * hd));
        steps.push(g.step(k.scale_row, &[d_r, &self.w1, scratch], &[total, hd], total));
        steps.push(g.step(k.emb_bwd, &[&self.idx1, scratch, d_table], &[n, hd, self.table_len], self.table_len * hd));
    }
}

/// Kernel-pipeline indices for the six `attn_relpos_*` kernels. Only the ids a
/// given direction dispatches need to be valid - an inference-only model may
/// leave the four backward slots unset.
#[derive(Clone, Copy)]
pub struct RelPosIds {
    /// `attn_relpos_qr` - the q·R hoist (dispatched per axis).
    pub qr: usize,
    /// `attn_relpos_add` - fold into the score slab, in place.
    pub add: usize,
    pub drh: usize,
    pub drw: usize,
    /// `attn_relpos_dq` - ACCUMULATES onto the q region of `d_qkv`.
    pub dq: usize,
    /// `attn_relpos_dr` - the dense-table adjoint (`acc` flag).
    pub dr: usize,
}

/// The backward-only half of [`RelPos`].
pub struct RelPosBwd<'a> {
    /// `[qh*kh, head_dim]` / `[qw*kw, head_dim]` - the NATURAL-layout tables.
    pub rh: &'a DeviceBuffer,
    pub rw: &'a DeviceBuffer,
    /// Adjoints of those, same shapes. Under `acc0 == false` the first span
    /// ASSIGNS them, so they need no zero-clear.
    pub d_rh: &'a DeviceBuffer,
    pub d_rw: &'a DeviceBuffer,
    /// `[heads, qh*qw, kh]` / `[heads, qh*qw, kw]` - adjoints of the hoisted
    /// intermediates. Separate buffers from `rel_h`/`rel_w`, which the
    /// per-chunk score RECOMPUTE is still reading.
    pub d_rel_h: &'a DeviceBuffer,
    pub d_rel_w: &'a DeviceBuffer,
    /// Accumulate into `d_rh`/`d_rw` from the FIRST span too - for a caller
    /// that folds several blocks/batches into one table gradient.
    pub acc0: bool,
}

/// One block's decomposed relative-position bias: the geometry every span
/// shares, plus the device buffers the six kernels bind.
///
/// SAM's windows are uniform (the token grid is zero-padded to a multiple of
/// the window before partitioning, and cropped after), and its global blocks
/// are one span over the whole grid - so ONE `(qh, qw, kh, kw)` covers every
/// span of a block. A ragged partition (a short border window) would need one
/// dense table per distinct window shape and is rejected loudly by
/// [`RelPos::check_span`] rather than silently mis-indexed.
pub struct RelPos<'a> {
    pub ids: RelPosIds,
    /// Query grid extent of every span.
    pub qh: u32,
    pub qw: u32,
    /// Key grid extent of every span. SAM always has `qh == kh`, `qw == kw`.
    pub kh: u32,
    pub kw: u32,
    /// `[qh, head_dim, kh]` / `[qw, head_dim, kw]` - the TRANSPOSED tables
    /// `attn_relpos_qr` reads.
    pub rh_t: &'a DeviceBuffer,
    pub rw_t: &'a DeviceBuffer,
    /// `[heads, qh*qw, kh]` / `[heads, qh*qw, kw]` - the hoisted q·R, rebuilt
    /// per span.
    pub rel_h: &'a DeviceBuffer,
    pub rel_w: &'a DeviceBuffer,
    pub bwd: Option<RelPosBwd<'a>>,
}

impl RelPos<'_> {
    /// Keys per invocation in `attn_relpos_add.wgsl`'s dispatch - see
    /// [`Self::add_step`]. Must equal that kernel's own `const JB`.
    pub const ADD_JB: u32 = 8;

    /// Query rows per span (`qh*qw`) - the row stride of `rel_h`/`rel_w`.
    pub fn span_qn(&self) -> u32 {
        self.qh * self.qw
    }
    /// Key count per span (`kh*kw`).
    pub fn span_kn(&self) -> u32 {
        self.kh * self.kw
    }

    /// Both extents must match the span the chunk loop is running, or every
    /// index below is quietly wrong. Checked once per span, with the numbers
    /// still in scope.
    pub fn check_span(&self, len: u32) {
        assert_eq!(
            len,
            self.span_qn(),
            "rel-pos span length {len} != qh*qw = {}x{}; a ragged/shifted partition needs one \
             dense table per window shape, which this bias does not model (SAM pads the grid \
             to a multiple of the window instead)",
            self.qh,
            self.qw
        );
        assert_eq!(len, self.span_kn(), "rel-pos self-attention needs qh*qw == kh*kw");
    }

    /// Invocations per output of `attn_relpos_drh`'s cooperative segment sum:
    /// the smallest power of two that covers `kw`, capped at the workgroup.
    fn drh_seg(&self) -> u32 {
        self.kw.next_power_of_two().min(64)
    }

    /// The two `attn_relpos_qr` dispatches for one span: `rel_h` (axis 0) and
    /// `rel_w` (axis 1). `q_off` is the ABSOLUTE float offset of this span's q
    /// region (`row0*stride + region offset`) - the buffer binds whole, so a
    /// ragged row offset never has to meet the 256 B binding alignment.
    pub fn qr_steps(
        &self,
        g: &Gpu,
        heads: u32,
        head_dim: u32,
        q: &DeviceBuffer,
        stride: u32,
        q_off: u32,
        steps: &mut Vec<Step>,
    ) {
        for (axis, r_t, k_ext, rel) in
            [(0u32, self.rh_t, self.kh, self.rel_h), (1, self.rw_t, self.kw, self.rel_w)]
        {
            let p = [heads, self.qh, self.qw, k_ext, head_dim, stride, q_off, axis];
            steps.push(g.step(self.ids.qr, &[q, r_t, rel], &p, self.qr_threads(heads, axis, k_ext)));
        }
    }

    /// Workgroups × 64 for `attn_relpos_qr` / `attn_relpos_dq` (same tiling;
    /// only the tiled axis differs - `k_ext` forward, `head_dim` backward).
    fn qr_threads(&self, heads: u32, axis: u32, k_ext: u32) -> u32 {
        self.tile_threads(heads, axis, k_ext.div_ceil(64))
    }
    fn dq_threads(&self, heads: u32, head_dim: u32, axis: u32) -> u32 {
        self.tile_threads(heads, axis, head_dim.div_ceil(64))
    }
    fn tile_threads(&self, heads: u32, axis: u32, outer_tiles: u32) -> u32 {
        let (panel_ext, group_len) = if axis == 0 { (self.qh, self.qw) } else { (self.qw, self.qh) };
        heads * panel_ext * group_len.div_ceil(8) * outer_tiles * 64
    }

    /// `attn_relpos_add` for one query chunk: `scores += rel_h + rel_w`, in
    /// place, between the scores kernel and the softmax.
    ///
    /// Dispatch width is `heads*qn*ceil(kn/ADD_JB)`, NOT `heads*qn*kn` -
    /// `attn_relpos_add.wgsl` blocks [`Self::ADD_JB`] keys per invocation
    /// (`crates/kernels/wgsl/attn_relpos_add.wgsl`'s own `const JB` must stay
    /// equal to this), which keeps every SAM-1 shape this repo defines clear
    /// of `backend_api::MAX_GROUPS_PER_DIM`'s 65 535-workgroup 2D-tiling
    /// threshold (the original one-thread-per-score shape crossed it at
    /// DeepSeek-OCR's real global-attention chunk shape). See that kernel's
    /// header comment: this is a mitigation for an intermittent wgpu defect
    /// this dispatch is implicated in, not a confirmed fix for it - the SAM-1
    /// tower still corrupts unrelated device buffers on wgpu some fraction of
    /// runs at production shape, and `crates/cli/src/resident_deepseekocr.rs`
    /// still pins the CPU backend for that reason.
    pub fn add_step(&self, g: &Gpu, heads: u32, qn: u32, kn: u32, q0: u32, scores: &DeviceBuffer) -> Step {
        let p = [heads, qn, kn, q0, self.span_qn(), self.kh, self.kw];
        let kn_blocks = kn.div_ceil(Self::ADD_JB);
        g.step(self.ids.add, &[self.rel_h, self.rel_w, scores], &p, heads * qn * kn_blocks)
    }

    /// `attn_relpos_drh` + `attn_relpos_drw` for one query chunk. Both ASSIGN
    /// the chunk's own rows of `d_rel_h`/`d_rel_w` - chunks partition the query
    /// rows, so nothing accumulates here.
    pub fn drel_steps(&self, g: &Gpu, heads: u32, qn: u32, kn: u32, q0: u32, d_scores: &DeviceBuffer, steps: &mut Vec<Step>) {
        let b = self.bwd.as_ref().expect("rel-pos backward buffers");
        let seg = self.drh_seg();
        let per = 64 / seg;
        let p_h = [heads, qn, kn, q0, self.span_qn(), self.kh, self.kw, seg];
        steps.push(g.step(self.ids.drh, &[d_scores, b.d_rel_h], &p_h, (heads * qn * self.kh).div_ceil(per) * 64));
        let p_w = [heads, qn, kn, q0, self.span_qn(), self.kh, self.kw];
        steps.push(g.step(self.ids.drw, &[d_scores, b.d_rel_w], &p_w, heads * qn * self.kw));
    }

    /// `attn_relpos_dq` (both axes) and `attn_relpos_dr` (both axes) for one
    /// span, dispatched AFTER the chunk loop has filled every row of
    /// `d_rel_h`/`d_rel_w`.
    ///
    /// `dq` ACCUMULATES onto the q region of `d_qkv`, which the plain
    /// `attn_bwd_dq_cross` has already assigned - an ordering precondition, not
    /// a zero-clear. `dr` accumulates when `acc` (any span after a block's
    /// first, or `RelPosBwd::acc0`).
    pub fn span_bwd_steps(
        &self,
        g: &Gpu,
        heads: u32,
        head_dim: u32,
        q: &DeviceBuffer,
        d_qkv: &DeviceBuffer,
        stride: u32,
        q_off: u32,
        acc: bool,
        steps: &mut Vec<Step>,
    ) {
        let b = self.bwd.as_ref().expect("rel-pos backward buffers");
        for (axis, k_ext, d_rel, r, d_r) in [
            (0u32, self.kh, b.d_rel_h, b.rh, b.d_rh),
            (1, self.kw, b.d_rel_w, b.rw, b.d_rw),
        ] {
            let p = [heads, self.qh, self.qw, k_ext, head_dim, stride, q_off, axis];
            steps.push(g.step(self.ids.dq, &[d_rel, r, d_qkv], &p, self.dq_threads(heads, head_dim, axis)));
            let panel_ext = if axis == 0 { self.qh } else { self.qw };
            let p_r = [heads, self.qh, self.qw, k_ext, head_dim, stride, q_off, axis, u32::from(acc)];
            let threads = panel_ext * k_ext.div_ceil(8) * head_dim.div_ceil(64) * 64;
            steps.push(g.step(self.ids.dr, &[d_rel, q, d_r], &p_r, threads));
        }
    }
}
