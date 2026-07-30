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

/// [`gqa_fwd`] with an additive per-key mask on the scores (the
/// `gqa_scores_kmask` kernel): `kmask[j]` is 0 for live keys, -3.4e38 for
/// excluded ones — right-padded encoder batches where pad tokens are queries
/// but must not be attended as keys. The kmask pipeline id is passed
/// explicitly so [`KernelIds`] (a struct literal at every call site in the
/// workspace) stays unchanged for models that never mask.
#[allow(clippy::too_many_arguments)]
pub fn gqa_fwd_kmask(
    g: &Gpu,
    kmask_kernel: usize,
    k: &KernelIds,
    a: &Gqa,
    q: &DeviceBuffer,
    kbuf: &DeviceBuffer,
    kmask: &DeviceBuffer,
    v: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Vec<Step> {
    let p = a.params();
    vec![
        g.step(kmask_kernel, &[q, kbuf, kmask, scores], &p, a.b * a.n_heads * a.t * a.t),
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

/// Bidirectional (encoder self-)attention shape over a fused qkv buffer
/// `[b*t, stride]` whose q/k/v regions each hold `n_heads*head_dim` floats at
/// `q_off`/`k_off`/`v_off`. MHA by construction — GQA projections are widened
/// first with [`kv_expand_fwd`] (group replication), which is what makes these
/// builders serve GQA encoders (LFM2.5) and plain MHA encoders (seq2seq) alike.
#[derive(Clone, Copy)]
pub struct Bidir {
    pub b: u32,
    pub t: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    /// Fused row width (typically `3*d_model`).
    pub stride: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
}

/// Kernel-pipeline indices for the bidirectional attention family.
#[derive(Clone, Copy)]
pub struct BidirIds {
    pub scores: usize,
    pub softmax: usize,
    pub apply: usize,
    pub dscores: usize,
    pub dv: usize,
    pub dq: usize,
    pub dk: usize,
}

impl Bidir {
    fn d_model(&self) -> u32 {
        self.n_heads * self.head_dim
    }
}

/// Bidirectional attention forward: `scores = qkᵀ/√hd` (all j), `probs =
/// softmax(scores)`, `ctx = probs·v`. `ctx` is contiguous `[b*t, d_model]`;
/// `scores`/`probs` are `[b*n_heads*t*t]`.
pub fn bidir_fwd(
    g: &Gpu,
    k: &BidirIds,
    a: &Bidir,
    qkv: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Vec<Step> {
    let (b, h, t, hd) = (a.b, a.n_heads, a.t, a.head_dim);
    vec![
        g.step(k.scores, &[qkv, scores], &[b, h, t, hd, a.stride, a.q_off, a.k_off], b * h * t * t),
        g.step(k.softmax, &[scores, probs], &[b, h, t], b * h * t),
        g.step(k.apply, &[probs, qkv, ctx], &[b, h, t, hd, a.stride, a.v_off, a.d_model()], b * h * t * hd),
    ]
}

/// Bidirectional attention backward: `d_scores` from the context grad `d_ctx`
/// (softmax jacobian folded in), then `d_q`/`d_k`/`d_v` written into their
/// regions of the fused `d_qkv` (disjoint assigns — no accumulation).
pub fn bidir_bwd(
    g: &Gpu,
    k: &BidirIds,
    a: &Bidir,
    qkv: &DeviceBuffer,
    probs: &DeviceBuffer,
    d_ctx: &DeviceBuffer,
    d_scores: &DeviceBuffer,
    d_qkv: &DeviceBuffer,
) -> Vec<Step> {
    let (b, h, t, hd) = (a.b, a.n_heads, a.t, a.head_dim);
    let pv = [b, h, t, hd, a.stride, a.v_off, a.d_model()];
    let pqk = [b, h, t, hd, a.stride, a.q_off, a.k_off];
    vec![
        g.step(k.dscores, &[d_ctx, qkv, probs, d_scores], &pv, b * h * t),
        g.step(k.dv, &[probs, d_ctx, d_qkv], &pv, b * h * t * hd),
        g.step(k.dq, &[d_scores, qkv, d_qkv], &pqk, b * h * t * hd),
        g.step(k.dk, &[d_scores, qkv, d_qkv], &pqk, b * h * t * hd),
    ]
}

/// Kernel-pipeline indices for the cross-attention trio (two lengths +
/// independent strides/offsets) — the substrate of query-chunked attention.
#[derive(Clone, Copy)]
pub struct CrossIds {
    pub scores: usize,
    pub softmax: usize,
    pub apply: usize,
}

/// Span + query-chunked bidirectional self-attention over a fused qkv buffer:
/// for each span `(row0, len)`, queries attend that span's keys/values
/// (non-causal); results land in `ctx` at the same absolute rows. `chunk`
/// bounds the materialized score slab to `[heads, chunk, max_span]` — the
/// mechanism that keeps long-context (8k+) attention inside the per-binding
/// budget. Layout-generic: `stride` is the fused row width, `q/k/v_off` the
/// region offsets, `d_out` the context width (`heads*head_dim`).
#[allow(clippy::too_many_arguments)]
pub fn chunked_bidir_fwd(
    g: &Gpu,
    k: &CrossIds,
    heads: u32,
    head_dim: u32,
    d_out: u32,
    qkv: &DeviceBuffer,
    stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    ctx: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    spans: &[(u32, u32)],
    chunk: u32,
    steps: &mut Vec<Step>,
) {
    for &(row0, len) in spans {
        let mut q0 = 0u32;
        while q0 < len {
            let qn = chunk.min(len - q0);
            // q view: rows [row0+q0 ..); kv view + ctx view: rows [row0 ..).
            let q_row_off = (row0 + q0) as u64 * stride as u64;
            let kv_row_off = row0 as u64 * stride as u64;
            let ctx_off = (row0 + q0) as u64 * d_out as u64;
            steps.push(g.step_sliced(
                k.scores,
                &[qkv, qkv, scores],
                &[(q_row_off, 0), (kv_row_off, 0), (0, 0)],
                &[1, heads, qn, len, head_dim, stride, stride, q_off, k_off],
                heads * qn * len,
            ));
            steps.push(g.step(k.softmax, &[scores, probs], &[1, heads, qn, len], heads * qn));
            steps.push(g.step_sliced(
                k.apply,
                &[probs, qkv, ctx],
                &[(0, 0), (kv_row_off, 0), (ctx_off, 0)],
                &[1, heads, qn, len, head_dim, stride, v_off, d_out],
                heads * qn * head_dim,
            ));
            q0 += qn;
        }
    }
}

/// The two interchangeable bidirectional flash-attention kernels, as a model's
/// own pipeline indices. `split` is optional (`None` = the model only registered
/// the baseline), which keeps this additive for callers that have not adopted it.
///
/// `flash_attn_bidir_split` computes the same thing as `flash_attn_bidir` to
/// cosine 1.00000000 and is faster at every head_dim measured on a P40
/// (29× at hd=128, 4.4× at hd=32 — see the kernel header for the table),
/// because the baseline's per-thread `q[128]`/`o[128]` arrays cannot live in
/// registers and its inner loop therefore runs at local-memory bandwidth. The
/// split kernel needs `@workgroup_size(256)`, so selection is gated on the
/// device's queried `max_workgroup_size`, never assumed.
#[derive(Clone, Copy)]
pub struct FlashIds {
    pub bidir: usize,
    pub split: Option<usize>,
}

/// The flash variant to dispatch on this device: `(kernel index, workgroup
/// size)`. Pure in its inputs — `caps` comes from `DeviceCaps`, so no backend
/// name is consulted.
pub fn flash_bidir_variant(ids: FlashIds, caps: &gpu_core::DeviceCaps) -> (usize, u32) {
    match ids.split {
        Some(i) if caps.max_workgroup_size >= 256 => (i, 256),
        _ => (ids.bidir, 64),
    }
}

/// One fused bidirectional flash-attention dispatch over `t` rows of a packed
/// qkv slab — the variant chosen by [`flash_bidir_variant`]. Both kernels take
/// the SAME Params and produce the SAME output layout, so only the pipeline
/// index and the per-workgroup thread count differ; the workgroup still owns
/// BR = 64 query rows in both.
#[allow(clippy::too_many_arguments)]
pub fn flash_bidir_step(
    g: &Gpu,
    ids: FlashIds,
    heads: u32,
    t: u32,
    head_dim: u32,
    d_model: u32,
    qkv: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Step {
    assert!(head_dim <= 128, "flash_attn_bidir: head_dim {head_dim} > 128");
    const BR: u32 = 64; // query rows per workgroup — the same in both kernels
    let (kind, ws) = flash_bidir_variant(ids, &g.caps());
    let nwg = heads * t.div_ceil(BR);
    g.step(
        kind,
        &[qkv, ctx],
        &[1, heads, t, head_dim, 3 * d_model, 0, d_model, 2 * d_model, d_model],
        nwg * ws,
    )
}

/// Span-wise fused flash attention: one dispatch per span replaces the whole
/// scores/softmax/apply chain with an online-softmax tiled kernel — O(t·hd)
/// memory AND the tuned inner loops, where the chunked cross trio materializes
/// `[H, chunk, t]` slabs through naive kernels. Picks the kernel through
/// [`flash_bidir_variant`], so a caller that registers `flash_attn_bidir_split`
/// gets it here too. Forward-only and workgroup-cooperative: callers MUST gate
/// on `DeviceCaps::workgroup_reductions` (false on the CPU JIT) and fall back
/// to [`chunked_bidir_fwd`]. `head_dim` ≤ 128.
#[allow(clippy::too_many_arguments)]
pub fn flash_bidir_fwd(
    g: &Gpu,
    ids: FlashIds,
    heads: u32,
    head_dim: u32,
    d_out: u32,
    qkv: &DeviceBuffer,
    stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    ctx: &DeviceBuffer,
    spans: &[(u32, u32)],
    steps: &mut Vec<Step>,
) {
    assert!(head_dim <= 128, "flash_attn_bidir: head_dim {head_dim} > 128");
    const BR: u32 = 64; // query rows per workgroup — the same in both kernels
    let (kind, ws) = flash_bidir_variant(ids, &g.caps());
    for &(row0, len) in spans {
        let nwg = heads * len.div_ceil(BR);
        steps.push(g.step_sliced(
            kind,
            &[qkv, ctx],
            &[(row0 as u64 * stride as u64, 0), (row0 as u64 * d_out as u64, 0)],
            &[1, heads, len, head_dim, stride, q_off, k_off, v_off, d_out],
            nwg * ws,
        ));
    }
}

/// Kernel indices for GEMM attention (see [`gemm_bidir_fwd`]).
#[derive(Clone, Copy)]
pub struct GemmAttnIds {
    pub head_pack: usize,
    pub head_pack_t: usize,
    pub head_unpack: usize,
    /// `softmax_rows` — workgroup-per-row softmax over the `[H·chunk, len]`
    /// slab (GPU-only; the GEMM path is already gated on cooperative devices).
    pub softmax_rows: usize,
    pub matmul: usize,
    pub matmul_reg2: usize,
}

/// Query-chunked bidirectional attention as REAL GEMMs: per-head packed
/// operands drive the register-tiled matmul instead of the naive
/// one-thread-per-score kernels. Measured motivation: at t=8192 the naive
/// trio (and the fused flash kernel — a memory escape hatch, not a fast
/// path) left a P40 at ~2% of peak; the same insight already made the CPU
/// fast paths 7× (they route these kernels to the native GEMM).
///
/// Layout: `packs` holds the three per-head-contiguous operands per span —
/// q (scaled by 1/√hd) at 0, k at `len·d_out`, vᵀ at `2·len·d_out` — with
/// GQA replication folded into the pack (`group` reads the NARROW k/v
/// projections; no expanded buffer exists). Scores/probs slabs stay
/// `[H, chunk, len]`; `ctx_pack` collects per-head context, unpacked into
/// the row-major `[rows, d_out]` `ctx` at the end of each span.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bidir_fwd(
    g: &Gpu,
    k: &GemmAttnIds,
    heads: u32,
    head_dim: u32,
    group: u32,
    q: &DeviceBuffer,
    q_stride: u32,
    kv: (&DeviceBuffer, &DeviceBuffer),
    kv_stride: u32,
    ctx: &DeviceBuffer,
    d_out: u32,
    packs: &DeviceBuffer,
    ctx_pack: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    spans: &[(u32, u32)],
    chunk: u32,
    force_naive: bool,
    steps: &mut Vec<Step>,
) {
    let hd = head_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let (kbuf, vbuf) = kv;
    for &(row0, len) in spans {
        let seg = len as u64 * d_out as u64; // one pack region, f32 words
        let total = heads * len * hd;
        // Pack q (scale folded), k, vᵀ for this span.
        steps.push(g.step_sliced(
            k.head_pack,
            &[q, packs],
            &[(row0 as u64 * q_stride as u64, 0), (0, 0)],
            &[len, heads, 1, hd, q_stride, 0, f(scale)],
            total,
        ));
        steps.push(g.step_sliced(
            k.head_pack,
            &[kbuf, packs],
            &[(row0 as u64 * kv_stride as u64, 0), (seg, 0)],
            &[len, heads, group, hd, kv_stride, 0, f(1.0)],
            total,
        ));
        steps.push(g.step_sliced(
            k.head_pack_t,
            &[vbuf, packs],
            &[(row0 as u64 * kv_stride as u64, 0), (2 * seg, 0)],
            &[len, heads, group, hd, kv_stride, 0, f(1.0)],
            total,
        ));
        let mut q0 = 0u32;
        while q0 < len {
            let qn = chunk.min(len - q0);
            for h in 0..heads {
                let (mk, mt) = pick_gemm(qn as usize, len as usize, k.matmul, k.matmul_reg2, force_naive);
                // scores[h] = q_pack[h][q0..q0+qn] · k_pack[h]ᵀ   ([qn,hd]·[len,hd]ᵀ)
                steps.push(g.step_sliced(
                    mk,
                    &[packs, packs, scores],
                    &[((h * len + q0) as u64 * hd as u64, 0), (seg + h as u64 * len as u64 * hd as u64, 0), (h as u64 * qn as u64 * len as u64, 0)],
                    &[qn, hd, len],
                    mt,
                ));
            }
            steps.push(g.step(k.softmax_rows, &[scores, probs], &[heads * qn, len], heads * qn * 64));
            for h in 0..heads {
                let (mk, mt) = pick_gemm(qn as usize, hd as usize, k.matmul, k.matmul_reg2, force_naive);
                // ctx_pack[h][q0..] = probs[h] · V[h]   (A·Bᵀ with B = vᵀ[hd,len])
                steps.push(g.step_sliced(
                    mk,
                    &[probs, packs, ctx_pack],
                    &[(h as u64 * qn as u64 * len as u64, 0), (2 * seg + h as u64 * hd as u64 * len as u64, 0), ((h * len + q0) as u64 * hd as u64, 0)],
                    &[qn, len, hd],
                    mt,
                ));
            }
            q0 += qn;
        }
        // Scatter the span's per-head context back to row-major [len, d_out].
        steps.push(g.step_sliced(
            k.head_unpack,
            &[ctx_pack, ctx],
            &[(0, 0), (row0 as u64 * d_out as u64, 0)],
            &[len, heads, hd, d_out, 0],
            total,
        ));
    }
}

/// Kernel indices for the query-chunked bidirectional backward: the cross
/// forward pair recomputes each chunk's scores/probs (nothing T×T is cached),
/// `dscores`/`dq` assign chunk-local rows, and the ACCUMULATING `dk_acc`/
/// `dv_acc` twins sum each chunk's partial contribution (their `acc_flag`
/// uniform assigns on a span's first chunk — no zero-clears to forget).
#[derive(Clone, Copy)]
pub struct CrossBwdIds {
    pub dscores: usize,
    pub dq: usize,
    pub dk_acc: usize,
    pub dv_acc: usize,
}

/// Backward of [`chunked_bidir_fwd`] with per-chunk score/softmax recompute —
/// what makes long-context (8k) training fit the per-binding budget: the
/// transient slabs stay `[heads, chunk, max_span]`. Writes `d_q`/`d_k`/`d_v`
/// into their regions of the fused `d_qkv`.
#[allow(clippy::too_many_arguments)]
pub fn chunked_bidir_bwd(
    g: &Gpu,
    fwd: &CrossIds,
    bwd: &CrossBwdIds,
    heads: u32,
    head_dim: u32,
    d_out: u32,
    qkv: &DeviceBuffer,
    stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    d_ctx: &DeviceBuffer,
    d_qkv: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    d_scores: &DeviceBuffer,
    spans: &[(u32, u32)],
    chunk: u32,
    steps: &mut Vec<Step>,
) {
    for &(row0, len) in spans {
        let mut q0 = 0u32;
        while q0 < len {
            let qn = chunk.min(len - q0);
            let q_row_off = (row0 + q0) as u64 * stride as u64;
            let kv_row_off = row0 as u64 * stride as u64;
            let dc_off = (row0 + q0) as u64 * d_out as u64;
            let p_qk = [1, heads, qn, len, head_dim, stride, stride, q_off, k_off];
            let p_v = [1, heads, qn, len, head_dim, stride, v_off, d_out];
            // Recompute this chunk's scores + probs from the cached qkv.
            steps.push(g.step_sliced(fwd.scores, &[qkv, qkv, scores], &[(q_row_off, 0), (kv_row_off, 0), (0, 0)], &p_qk, heads * qn * len));
            steps.push(g.step(fwd.softmax, &[scores, probs], &[1, heads, qn, len], heads * qn));
            // Softmax jacobian → d_scores (chunk-local).
            steps.push(g.step_sliced(
                bwd.dscores,
                &[d_ctx, qkv, probs, d_scores],
                &[(dc_off, 0), (kv_row_off, 0), (0, 0), (0, 0)],
                &p_v,
                heads * qn,
            ));
            // d_q: chunk rows only (disjoint — plain assign into the q region).
            steps.push(g.step_sliced(
                bwd.dq,
                &[d_scores, qkv, d_qkv],
                &[(0, 0), (kv_row_off, 0), (q_row_off, 0)],
                &p_qk,
                heads * qn * head_dim,
            ));
            // d_k / d_v: sums over ALL queries — accumulate across chunks.
            let acc = u32::from(q0 > 0);
            let mut p_qk_acc = [0u32; 10];
            p_qk_acc[..9].copy_from_slice(&p_qk);
            p_qk_acc[9] = acc;
            steps.push(g.step_sliced(
                bwd.dk_acc,
                &[d_scores, qkv, d_qkv],
                &[(0, 0), (q_row_off, 0), (kv_row_off, 0)],
                &p_qk_acc,
                heads * len * head_dim,
            ));
            let mut p_v_acc = [0u32; 9];
            p_v_acc[..8].copy_from_slice(&p_v);
            p_v_acc[8] = acc;
            steps.push(g.step_sliced(
                bwd.dv_acc,
                &[probs, d_ctx, d_qkv],
                &[(0, 0), (dc_off, 0), (kv_row_off, 0)],
                &p_v_acc,
                heads * len * head_dim,
            ));
            q0 += qn;
        }
    }
}

/// GQA→MHA head replication into a fused-buffer region: dst head `ho` copies
/// src head `ho/group` (`repeat_kv` layout). `group == 1` is a strided copy —
/// the same dispatch places q. `src` is `[rows, (heads_out/group)*hd]`.
#[allow(clippy::too_many_arguments)]
pub fn kv_expand_fwd(
    g: &Gpu,
    idx: usize,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    rows: u32,
    heads_out: u32,
    group: u32,
    hd: u32,
    dst_stride: u32,
    dst_off: u32,
) -> Step {
    let src_stride = heads_out / group * hd;
    g.step(idx, &[src, dst], &[rows, heads_out, group, hd, src_stride, dst_stride, dst_off], rows * heads_out * hd)
}

/// Adjoint of [`kv_expand_fwd`]: group-sums the region grad back to the
/// narrow projection grad (overwrites `d_src`).
#[allow(clippy::too_many_arguments)]
pub fn kv_expand_bwd(
    g: &Gpu,
    idx: usize,
    d_dst: &DeviceBuffer,
    d_src: &DeviceBuffer,
    rows: u32,
    heads_out: u32,
    group: u32,
    hd: u32,
    dst_stride: u32,
    dst_off: u32,
) -> Step {
    let src_stride = heads_out / group * hd;
    g.step(idx, &[d_dst, d_src], &[rows, heads_out, group, hd, src_stride, dst_stride, dst_off], rows * (heads_out / group) * hd)
}

/// RMSNorm forward with a runtime epsilon (`rmsnorm_eps`); the fixed-eps
/// [`rmsnorm_fwd`] covers the 1e-6 family (Qwen/GLM), this one models whose
/// checkpoints carry a different eps (LFM2.5: 1e-5).
pub fn rmsnorm_eps_fwd(g: &Gpu, idx: usize, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) -> Step {
    g.step(idx, &[x, w, out], &[dim, rows, f(eps)], rows)
}

/// RMSNorm backward with runtime epsilon: input grad always (`rmsnorm_dx_eps`),
/// gain grad only when `gw` is `Some` (`rms_inv_eps` + `rmsnorm_dw`; the dw
/// kernel is eps-free — eps enters through the per-row inverse).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_eps_bwd(
    g: &Gpu,
    inv_idx: usize,
    dw_idx: usize,
    dx_idx: usize,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    dy: &DeviceBuffer,
    dx: &DeviceBuffer,
    inv: &DeviceBuffer,
    gw: Option<&DeviceBuffer>,
    dim: u32,
    rows: u32,
    eps: f32,
) -> Vec<Step> {
    let mut s = Vec::new();
    if let Some(gw) = gw {
        s.push(g.step(inv_idx, &[x, inv], &[dim, rows, f(eps)], rows));
        s.push(g.step(dw_idx, &[dy, x, inv, gw], &[dim, rows], dim));
    }
    s.push(g.step(dx_idx, &[x, w, dy, dx], &[dim, rows, f(eps)], rows));
    s
}

/// LayerNorm kernel indices, with the workgroup-per-row variants **optional**
/// so a model can adopt them one at a time (and a model that has not registered
/// them keeps working unchanged). Added alongside [`KernelIds`] rather than
/// inside it, so no existing struct literal has to change.
///
/// The per-element kernels (`layernorm`, `ln_stats`, `layernorm_dx`) give
/// thread *t* row *t*: a warp's 32 loads are `d_model` floats apart, so each
/// 32-byte sector fetched serves one useful float. `layernorm_dx` walks its row
/// four times that way. The `*_rows` kernels give a whole 64-thread workgroup
/// to one row and are coalesced by construction — measured 2.3-9.1x on a P40
/// across d_model 768-3072 x 512-2048 rows (`brain-gpu-core`'s
/// `bench_layernorm`), winning at every shape including the 1-row decode case.
#[derive(Clone, Copy)]
pub struct LayerNormIds {
    pub layernorm: usize,
    pub layernorm_rows: Option<usize>,
    pub ln_stats: usize,
    pub ln_stats_rows: Option<usize>,
    pub layernorm_dx: usize,
    pub layernorm_dx_rows: Option<usize>,
}

impl LayerNormIds {
    /// Reference indices supplied by the model; the `*_rows` variants resolved
    /// **by name** from this handle's pipeline list.
    ///
    /// This is how a model adopts the coalesced kernels without every
    /// `KernelIds`-style struct literal in the workspace growing a field: add
    /// `layernorm_rows` / `ln_stats_rows` / `layernorm_dx_rows` to its
    /// PIPELINES and it is opted in; leave them out and it keeps the reference
    /// kernels. A model with its own fixed indices should build the struct
    /// literally instead (cheaper, and the indices stay greppable).
    pub fn resolve(g: &Gpu, layernorm: usize, ln_stats: usize, layernorm_dx: usize) -> LayerNormIds {
        LayerNormIds {
            layernorm,
            layernorm_rows: g.kernel_index("layernorm_rows"),
            ln_stats,
            ln_stats_rows: g.kernel_index("ln_stats_rows"),
            layernorm_dx,
            layernorm_dx_rows: g.kernel_index("layernorm_dx_rows"),
        }
    }

    /// [`LayerNormIds::resolve`] for a forward-only path: the LN backward
    /// helpers are never dispatched, so their slots mirror the forward kernel.
    pub fn resolve_fwd(g: &Gpu, layernorm: usize) -> LayerNormIds {
        LayerNormIds {
            layernorm,
            layernorm_rows: g.kernel_index("layernorm_rows"),
            ln_stats: layernorm,
            ln_stats_rows: None,
            layernorm_dx: layernorm,
            layernorm_dx_rows: None,
        }
    }
}

/// `(kernel index, dispatch threads)` for one LayerNorm-family op: the
/// cooperative kernel where the model registered it and the device can run a
/// workgroup reduction, else the reference.
///
/// The policy itself lives in `backend_api::select` (`Op::LayerNorm`) and is
/// keyed on `DeviceCaps`, never a backend name. The `*_rows` kernels are
/// `@workgroup_size(64)` — at or below the WebGPU floor of 256 — so no
/// `max_workgroup_size` gate is needed on top of it.
fn ln_variant(g: &Gpu, reference: usize, coop: Option<usize>, rows: u32, d: u32) -> (usize, u32) {
    use gpu_core::select::{Dtype, KernelSelector, KernelVariant, Op, OpShape};
    let shape = OpShape { m: rows, n: d, k: 0, dtype: Dtype::F32 };
    match coop {
        Some(i)
            if gpu_core::select::DefaultSelector.select(Op::LayerNorm, shape, &g.caps())
                == KernelVariant::WorkgroupPerOutput =>
        {
            (i, rows * 64)
        }
        _ => (reference, rows),
    }
}

/// LayerNorm forward: `y = (x-mean)/sqrt(var+eps) * gamma + beta` over `rows`
/// rows of `d` elements. Same math and Params either variant.
#[allow(clippy::too_many_arguments)]
pub fn layernorm_fwd(
    g: &Gpu,
    k: &LayerNormIds,
    x: &DeviceBuffer,
    gamma: &DeviceBuffer,
    beta: &DeviceBuffer,
    out: &DeviceBuffer,
    d: u32,
    rows: u32,
    eps: f32,
) -> Step {
    let (kind, threads) = ln_variant(g, k.layernorm, k.layernorm_rows, rows, d);
    g.step(kind, &[x, gamma, beta, out], &[d, rows, f(eps)], threads)
}

/// Per-row `mean` + `1/sqrt(var+eps)` (feeds `layernorm_dgamma`).
pub fn ln_stats_fwd(
    g: &Gpu,
    k: &LayerNormIds,
    x: &DeviceBuffer,
    mean: &DeviceBuffer,
    inv: &DeviceBuffer,
    d: u32,
    rows: u32,
    eps: f32,
) -> Step {
    let (kind, threads) = ln_variant(g, k.ln_stats, k.ln_stats_rows, rows, d);
    g.step(kind, &[x, mean, inv], &[d, rows, f(eps)], threads)
}

/// LayerNorm backward w.r.t. `x` (mean/inv recomputed from `x`).
#[allow(clippy::too_many_arguments)]
pub fn layernorm_dx_bwd(
    g: &Gpu,
    k: &LayerNormIds,
    x: &DeviceBuffer,
    gamma: &DeviceBuffer,
    dy: &DeviceBuffer,
    dx: &DeviceBuffer,
    d: u32,
    rows: u32,
    eps: f32,
) -> Step {
    let (kind, threads) = ln_variant(g, k.layernorm_dx, k.layernorm_dx_rows, rows, d);
    g.step(kind, &[x, gamma, dy, dx], &[d, rows, f(eps)], threads)
}

/// Per-binding budget (f32 words) for tiling an embedding / lm_head over vocab,
/// so each storage binding stays under a backend's max-binding size (e.g. 128MB
/// on Mesa-GL). ~96 MiB; small models collapse to one tile.
/// `BRAIN_TILE_BUDGET_WORDS` overrides it (e.g. tiny, to force tiling in tests).
pub const TILE_BUDGET_WORDS: u64 = 24 * 1024 * 1024;

pub fn tile_budget_words() -> u64 {
    std::env::var("BRAIN_TILE_BUDGET_WORDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(TILE_BUDGET_WORDS)
}

/// Vocab tiles `(v0, count)` sized so a `[count, d_model]` weight slice stays
/// within the per-binding budget. Small vocabularies yield a single tile.
pub fn vocab_tiles(vocab: u64, d_model: u64) -> Vec<(u32, u32)> {
    let rows = (tile_budget_words() / d_model.max(1)).max(1);
    let mut out = Vec::new();
    let mut v0 = 0u64;
    while v0 < vocab {
        let cnt = rows.min(vocab - v0);
        out.push((v0 as u32, cnt as u32));
        v0 += cnt;
    }
    out
}

/// Pick the forward GEMM kernel + dispatch thread count for `[m,k]·[n,k]ᵀ`:
/// the register-tiled kernel (128×128 tile, 256 threads) once both output dims
/// fill a tile, else the naive one-thread-per-output kernel. Same math either
/// way — this only changes speed. `force_naive` lets models keep an env escape.
pub fn pick_gemm(m: usize, n: usize, naive: usize, reg2: usize, force_naive: bool) -> (usize, u32) {
    if force_naive || m < 128 || n < 128 {
        (naive, (m * n) as u32)
    } else {
        (reg2, (m.div_ceil(128) * n.div_ceil(128) * 256) as u32)
    }
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
