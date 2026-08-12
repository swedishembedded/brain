// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reusable transformer-block Step-builders - the composable layer new
//! architectures build on instead of re-hand-rolling dispatch sequences.
//!
//! Each model maps its own PIPELINE kernel indices into [`KernelIds`] (so no
//! model has to reorder its pipeline list), then composes the forward/backward
//! graph from these helpers. They are pure dispatch assembly - no WGSL, no
//! ParamStore, no buffer ownership - so they stay decoupled from any one model
//! and are validated by each caller's gradient check.
//!
//! Covered today (the Qwen/RMSNorm family): RMSNorm fwd/bwd, half-split RoPE
//! fwd/bwd, grouped-query attention fwd/bwd, and the SwiGLU activation fwd/bwd.
//! Linear projections stay in the model (they carry model-specific concerns such
//! as LoRA adapters and bias). MoE/GPT/PID are not yet ported.

// Every builder here takes a device, a kernel-id set, the kernel's buffer
// bindings and its Params fields - so the arity IS the WGSL kernel's binding
// list. Packing those into a struct would put a second, drifting description
// of a kernel's signature next to the authoritative one in the .wgsl, which
// is a failure mode worth preventing.
#![allow(clippy::too_many_arguments)]

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

/// Table-driven interleaved M-RoPE (forward), in place on a contiguous q/k
/// buffer: the same half-split rotation `rope_fwd` applies, but the per-token
/// angle comes from a precomputed `[rows, head_dim/2]` `cos`/`sin` table
/// (`qwenvl::mrope::mrope_tables`) instead of a single scalar position - the
/// seam that lets a caller feed genuinely divergent per-axis (text/image/
/// video/audio) positions, or the degenerate all-axes-equal case (which
/// `qwenvl::mrope`'s own test proves collapses to identical output). `qwen3::
/// Qwen::rope2d_step` already dispatches this exact kernel for Qwen3-VL;
/// hoisted here so a second model (`omni::thinker`) doesn't re-wire it.
///
/// `kernel` is `kernels::ROPE2D`'s pipeline index in the caller's own table -
/// not a [`KernelIds`] field, since it pairs with two extra buffer bindings
/// (`cos`/`sin`) `rope_fwd` doesn't have; folding it in would grow every
/// other model's `KernelIds` literal for a kernel most never dispatch (same
/// reasoning as `model::moe`'s separate `MoeIds`).
#[allow(clippy::too_many_arguments)]
pub fn rope2d_fwd(g: &Gpu, kernel: usize, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, rows: u32, n_heads: u32, head_dim: u32, row_stride: u32) -> Step {
    let half = head_dim / 2;
    // tmod = rows: an exact per-token table, no frame-repeat (Omni's
    // get_rope_index already assigns one position per token, video included).
    g.step(kernel, &[buf, cos, sin], &[rows, n_heads, half, row_stride, 0, rows, f(1.0)], rows * n_heads * half)
}

/// Table-driven interleaved M-RoPE (forward), **partial**: rotates only the
/// first `rot_dim = 2*half` channels of each `head_dim`-wide head, leaving
/// `head_dim - rot_dim` channels untouched (Qwen3.5's
/// `partial_rotary_factor`). Dispatches `kernels::ROPE2D_PARTIAL`
/// (`rope2d_partial.wgsl`) - unlike `rope2d_fwd`, the per-head stride in the
/// buffer is the FULL `head_dim`, not `2*half`, so the two kernels are not
/// interchangeable at a partial rotary factor (see the kernel's header for
/// why a plain `rope2d` dispatch would corrupt every head after the first).
/// `half` is the table width (`rot_dim/2`), built by
/// `qwenvl::mrope::mrope_tables` called with `head_dim = rot_dim`.
#[allow(clippy::too_many_arguments)]
pub fn rope2d_partial_fwd(
    g: &Gpu,
    kernel: usize,
    buf: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    rows: u32,
    n_heads: u32,
    half: u32,
    row_stride: u32,
    off: u32,
    head_dim: u32,
) -> Step {
    g.step(
        kernel,
        &[buf, cos, sin],
        &[rows, n_heads, half, row_stride, off, rows, f(1.0), head_dim],
        rows * n_heads * half,
    )
}

/// The exact inverse of [`rope2d_partial_fwd`] (`sign = -1`), for backward.
#[allow(clippy::too_many_arguments)]
pub fn rope2d_partial_bwd(
    g: &Gpu,
    kernel: usize,
    buf: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    rows: u32,
    n_heads: u32,
    half: u32,
    row_stride: u32,
    off: u32,
    head_dim: u32,
) -> Step {
    g.step(
        kernel,
        &[buf, cos, sin],
        &[rows, n_heads, half, row_stride, off, rows, f(-1.0), head_dim],
        rows * n_heads * half,
    )
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
/// excluded ones - right-padded encoder batches where pad tokens are queries
/// but must not be attended as keys. The kmask pipeline id is passed
/// explicitly so [`KernelIds`] (a struct literal at every call site in the
/// workspace) stays unchanged for models that never mask.
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

/// [`gqa_fwd`] with a sliding-window causal mask (the `gqa_scores_win`
/// kernel): key `j` is live only for `i-window < j <= i`. `window >= a.t`
/// degenerates to `gqa_fwd`'s plain causal mask exactly (see the kernel's own
/// doc), so a caller with no window requirement may always use this entry
/// point instead of keeping two call sites. The kernel id is passed
/// explicitly, same convention as [`gqa_fwd_kmask`], so [`KernelIds`] stays
/// unchanged for models that never window.
#[allow(clippy::too_many_arguments)]
pub fn gqa_fwd_win(
    g: &Gpu,
    win_kernel: usize,
    k: &KernelIds,
    a: &Gqa,
    window: u32,
    q: &DeviceBuffer,
    kbuf: &DeviceBuffer,
    v: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Vec<Step> {
    let mut p = a.params().to_vec();
    p.push(window);
    vec![
        g.step(win_kernel, &[q, kbuf, scores], &p, a.b * a.n_heads * a.t * a.t),
        g.step(k.attn_softmax, &[scores, probs], &[a.b, a.n_heads, a.t], a.b * a.n_heads * a.t),
        g.step(k.gqa_apply, &[probs, v, ctx], &p[..6], a.b * a.n_heads * a.t * a.head_dim),
    ]
}

/// Kernel-pipeline indices for incremental KV-cache decode attention - the
/// O(cached length) twin of [`gqa_fwd`]'s O(T²) full recompute. Hoisted from
/// `qwen3::Qwen`'s `decode_steps` (`crates/qwen3/src/model.rs`) so a second
/// model (`omni::thinker`, a 48-layer MoE decoder) reuses the exact same
/// dispatch sequence instead of re-deriving it - the "one implementation,
/// migrate existing users" rule this crate exists to enforce.
#[derive(Clone, Copy)]
pub struct GqaDecodeIds {
    pub kv_append: usize,
    pub attn_decode_scores: usize,
    pub decode_softmax: usize,
    pub attn_decode_apply: usize,
}

/// One incremental decode step of GQA attention: appends the new token's
/// (already QK-normed + RoPE'd) `k_new`/`v_new` into the persistent per-layer
/// `kcache`/`vcache` at row `pos`, then attends `q` (the same new token) against
/// all `pos+1` cached positions - O(cached length), not O(cached length)²,
/// and implicitly causal (only ever reads cache rows `0..=pos`, never later
/// ones, since none exist yet).
///
/// `q` is `[n_heads*head_dim]`; `k_new`/`v_new` are `[n_kv_heads*head_dim]`
/// (a SINGLE new token's row, batch-of-one - this is a decode primitive, not
/// a batched one). `kcache`/`vcache` are the persistent `[cap, n_kv_heads*
/// head_dim]` per-layer buffers the caller sized upfront for the whole
/// generation (`cap` = max sequence length, prompt + max_new_tokens);
/// `scores`/`probs` are `[n_heads, cap]`-strided scratch; `ctx` is
/// `[n_heads*head_dim]`, this step's attention output (feed straight into the
/// output projection, exactly like [`gqa_fwd`]'s `ctx`).
///
/// A batched prefill can reuse the SAME cache buffers without a per-token
/// loop: after a normal [`gqa_fwd`] pass over the prompt's `n` positions, bulk
/// -copy the resulting `k`/`v` (`[n, n_kv_heads*head_dim]`, contiguous, same
/// per-row layout as the cache) into `kcache`/`vcache` rows `0..n` with one
/// `kv_append` dispatch each (`width = n*n_kv_heads*head_dim, row = 0` - a
/// flat prefix copy), then decode steps continue from `pos = n`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_decode_step(
    g: &Gpu,
    k: &GqaDecodeIds,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    cap: u32,
    q: &DeviceBuffer,
    k_new: &DeviceBuffer,
    v_new: &DeviceBuffer,
    kcache: &DeviceBuffer,
    vcache: &DeviceBuffer,
    scores: &DeviceBuffer,
    probs: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Vec<Step> {
    let group = n_heads / n_kv_heads;
    let hkv = n_kv_heads * head_dim;
    let t = pos + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();
    vec![
        g.step(k.kv_append, &[k_new, kcache], &[hkv, pos], hkv),
        g.step(k.kv_append, &[v_new, vcache], &[hkv, pos], hkv),
        g.step(k.attn_decode_scores, &[q, kcache, scores], &[n_heads, group, head_dim, t, cap, hkv, f(scale)], n_heads * t),
        g.step(k.decode_softmax, &[scores, probs], &[n_heads, t, cap], n_heads),
        g.step(k.attn_decode_apply, &[probs, vcache, ctx], &[n_heads, group, head_dim, t, cap, hkv], n_heads * head_dim),
    ]
}

/// Bulk-fill a KV cache's rows `0..n` from a batched prefill's contiguous
/// `k`/`v` output - see [`gqa_decode_step`]'s doc for why one `kv_append`
/// dispatch suffices (a flat prefix copy, since the cache and the batched
/// buffer share the same per-row `n_kv_heads*head_dim` stride).
pub fn kv_cache_fill(g: &Gpu, kv_append: usize, src: &DeviceBuffer, cache: &DeviceBuffer, n: u32, n_kv_heads: u32, head_dim: u32) -> Step {
    let width = n * n_kv_heads * head_dim;
    g.step(kv_append, &[src, cache], &[width, 0], width)
}

// ---- the full GQA attention SUBLAYER (norm -> QKV -> QK-norm -> RoPE ->
// attend -> out-proj -> residual), hoisted from omni::thinker/omni::talker ---
//
// The two omni decoders carried byte-identical copies of this whole sequence
// (batched AND decode-step variants), and the copy already cost a real
// regression: the Thinker copy got the accumulated-single-submit MoE fix
// while the Talker copy kept submitting per expert. Per the hoist-and-migrate
// policy, the sublayer lives here ONCE; a model that wants it supplies its
// kernel indices via [`GqaAttnIds`] and its dims via [`GqaAttnDims`].

/// Kernel indices for [`gqa_attn_sublayer_fwd`]/[`gqa_attn_sublayer_decode_step`],
/// resolved by the calling model against its own registered pipeline list.
#[derive(Clone, Copy)]
pub struct GqaAttnIds {
    /// The shared forward ids (`rmsnorm`, `gqa_scores`/`attn_softmax`/
    /// `gqa_apply`); backward slots are never dispatched by the sublayer.
    pub kernels: KernelIds,
    /// Naive matmul (`matmul.wgsl`'s `{x,w,out}`/`{m,k,n}` contract).
    pub matmul: usize,
    pub add2: usize,
    /// Table-driven M-RoPE (`rope2d.wgsl`) - see `rope2d_fwd`.
    pub rope2d: usize,
    /// `kv_append` for [`kv_cache_fill`] (prefill) and [`GqaDecodeIds`] (decode).
    pub kv_append: usize,
    /// The decode-step ids ([`gqa_decode_step`]).
    pub decode: GqaDecodeIds,
}

/// The attention sublayer's shape parameters.
#[derive(Clone, Copy)]
pub struct GqaAttnDims {
    pub hidden: u32,
    pub head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    /// Qwen3-style per-head RMSNorm on Q/K after the projections.
    pub use_qk_norm: bool,
}

/// The sublayer's weights: pre-attention RMSNorm, QKV/out projections, and
/// (when `use_qk_norm`) the per-head Q/K norm weights.
pub struct GqaAttnWeights<'a> {
    pub ln1: &'a DeviceBuffer,
    pub wq: &'a DeviceBuffer,
    pub wk: &'a DeviceBuffer,
    pub wv: &'a DeviceBuffer,
    pub wo: &'a DeviceBuffer,
    pub q_norm: &'a DeviceBuffer,
    pub k_norm: &'a DeviceBuffer,
}

/// Shared head of both sublayer variants: RMSNorm -> QKV projections ->
/// optional per-head QK-norm -> RoPE, submitted as one batch. Returns the
/// post-RoPE `(q, k, v)` buffers (`[n, n_heads*head_dim]` /
/// `[n, n_kv_heads*head_dim]`).
fn gqa_attn_qkv(g: &Gpu, ids: &GqaAttnIds, dims: &GqaAttnDims, w: &GqaAttnWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let (d, hd, nh, nkv) = (dims.hidden, dims.head_dim, dims.n_heads, dims.n_kv_heads);
    let (hq, hkv) = (nh * hd, nkv * hd);

    let xn1 = g.storage((n * d) as u64);
    let mut steps = vec![rmsnorm_fwd(g, &ids.kernels, x, w.ln1, &xn1, d, n)];

    let q_pre = g.storage((n * hq) as u64);
    let k_pre = g.storage((n * hkv) as u64);
    let v = g.storage((n * hkv) as u64);
    steps.push(g.step(ids.matmul, &[&xn1, w.wq, &q_pre], &[n, d, hq], n * hq));
    steps.push(g.step(ids.matmul, &[&xn1, w.wk, &k_pre], &[n, d, hkv], n * hkv));
    steps.push(g.step(ids.matmul, &[&xn1, w.wv, &v], &[n, d, hkv], n * hkv));

    let (q, k) = if dims.use_qk_norm {
        let q = g.storage((n * hq) as u64);
        let k = g.storage((n * hkv) as u64);
        steps.push(rmsnorm_fwd(g, &ids.kernels, &q_pre, w.q_norm, &q, hd, n * nh));
        steps.push(rmsnorm_fwd(g, &ids.kernels, &k_pre, w.k_norm, &k, hd, n * nkv));
        (q, k)
    } else {
        (q_pre, k_pre)
    };
    steps.push(rope2d_fwd(g, ids.rope2d, &q, cos, sin, n, nh, hd, hq));
    steps.push(rope2d_fwd(g, ids.rope2d, &k, cos, sin, n, nkv, hd, hkv));
    g.submit(&[], &steps);
    (q, k, v)
}

/// Shared tail of both sublayer variants: output projection + residual add.
/// Returns `xmid = x + ctx @ wo` (`[n, d]`).
fn gqa_attn_out(g: &Gpu, ids: &GqaAttnIds, dims: &GqaAttnDims, w: &GqaAttnWeights, x: &DeviceBuffer, ctx: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let (d, hq) = (dims.hidden, dims.n_heads * dims.head_dim);
    let proj = g.storage((n * d) as u64);
    let xmid = g.storage((n * d) as u64);
    g.submit(&[], &[g.step(ids.matmul, &[ctx, w.wo, &proj], &[n, hq, d], n * d)]);
    g.submit(&[], &[g.step(ids.add2, &[x, &proj, &xmid], &[n * d], n * d)]);
    xmid
}

/// The full batched GQA attention sublayer: `x [n, d] -> xmid [n, d]`
/// (post-attention residual, ready for the caller's FFN/MoE sublayer).
/// `cos`/`sin` are the `[n, head_dim/2]` RoPE tables. `kv_cache`, when
/// `Some((kcache, vcache))`, bulk-fills the persistent per-layer cache with
/// the `n` post-RoPE key/value rows ([`kv_cache_fill`]) - purely additive,
/// `xmid` is identical either way.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attn_sublayer_fwd(g: &Gpu, ids: &GqaAttnIds, dims: &GqaAttnDims, w: &GqaAttnWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32, kv_cache: Option<(&DeviceBuffer, &DeviceBuffer)>) -> DeviceBuffer {
    let (hd, nh, nkv) = (dims.head_dim, dims.n_heads, dims.n_kv_heads);
    let (q, k, v) = gqa_attn_qkv(g, ids, dims, w, x, cos, sin, n);

    if let Some((kcache, vcache)) = kv_cache {
        g.submit(&[], &[kv_cache_fill(g, ids.kv_append, &k, kcache, n, nkv, hd), kv_cache_fill(g, ids.kv_append, &v, vcache, n, nkv, hd)]);
    }

    let scores = g.storage((nh * n * n) as u64);
    let probs = g.storage((nh * n * n) as u64);
    let ctx = g.storage((n * nh * hd) as u64);
    let ga = Gqa { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
    g.submit(&[], &gqa_fwd(g, &ids.kernels, &ga, &q, &k, &v, &scores, &probs, &ctx));

    gqa_attn_out(g, ids, dims, w, x, &ctx, n)
}

/// The single-token incremental-decode twin of [`gqa_attn_sublayer_fwd`]:
/// one new token's row `x [1, d]` -> `xmid [1, d]`, attending against
/// `(kcache, vcache)`'s `pos+1` valid positions via [`gqa_decode_step`].
/// `cos`/`sin` are the 1-row RoPE table for this token's absolute position;
/// `cap` is the cache's allocated capacity.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attn_sublayer_decode_step(g: &Gpu, ids: &GqaAttnIds, dims: &GqaAttnDims, w: &GqaAttnWeights, kv_cache: (&DeviceBuffer, &DeviceBuffer), x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, pos: u32, cap: u32) -> DeviceBuffer {
    let (hd, nh, nkv) = (dims.head_dim, dims.n_heads, dims.n_kv_heads);
    let (q, k, v) = gqa_attn_qkv(g, ids, dims, w, x, cos, sin, 1);

    let scores = g.storage((nh * cap) as u64);
    let probs = g.storage((nh * cap) as u64);
    let ctx = g.storage((nh * hd) as u64);
    let (kcache, vcache) = kv_cache;
    g.submit(&[], &gqa_decode_step(g, &ids.decode, nh, nkv, hd, pos, cap, &q, &k, &v, kcache, vcache, &scores, &probs, &ctx));

    gqa_attn_out(g, ids, dims, w, x, &ctx, 1)
}

/// GQA attention backward: produces `d_scores`, `d_v`, `d_q`, `d_k` from the
/// context grad `d_ctx` and the cached `q`/`k`/`v`/`probs`.
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
/// `q_off`/`k_off`/`v_off`. MHA by construction - GQA projections are widened
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
/// regions of the fused `d_qkv` (disjoint assigns - no accumulation).
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
/// independent strides/offsets) - the substrate of query-chunked attention.
#[derive(Clone, Copy)]
pub struct CrossIds {
    pub scores: usize,
    pub softmax: usize,
    pub apply: usize,
}

/// Span + query-chunked bidirectional self-attention over a fused qkv buffer:
/// for each span `(row0, len)`, queries attend that span's keys/values
/// (non-causal); results land in `ctx` at the same absolute rows. `chunk`
/// bounds the materialized score slab to `[heads, chunk, max_span]` - the
/// mechanism that keeps long-context (8k+) attention inside the per-binding
/// budget. Layout-generic: `stride` is the fused row width, `q/k/v_off` the
/// region offsets, `d_out` the context width (`heads*head_dim`).
///
/// `rel` adds SAM's decomposed relative-position bias
/// (`crate::vit::RelPos` - two hoisted `q·R` dispatches per span, then an
/// in-place fold into each chunk's score slab before the softmax). `None`
/// leaves the dispatch sequence byte-for-byte what it was; there is deliberately
/// no second copy of this loop for the biased case.
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
    rel: Option<&crate::vit::RelPos>,
    steps: &mut Vec<Step>,
) {
    for &(row0, len) in spans {
        if let Some(r) = rel {
            r.check_span(len);
            // The hoist covers the WHOLE span; the chunk loop only folds the
            // rows it owns, addressed by `q0`.
            r.qr_steps(g, heads, head_dim, qkv, stride, row0 * stride + q_off, steps);
        }
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
            if let Some(r) = rel {
                steps.push(r.add_step(g, heads, qn, len, q0, scores));
            }
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
/// (29× at hd=128, 4.4× at hd=32 - see the kernel header for the table),
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
/// size)`. Pure in its inputs - `caps` comes from `DeviceCaps`, so no backend
/// name is consulted.
pub fn flash_bidir_variant(ids: FlashIds, caps: &gpu_core::DeviceCaps) -> (usize, u32) {
    match ids.split {
        Some(i) if caps.max_workgroup_size >= 256 => (i, 256),
        _ => (ids.bidir, 64),
    }
}

/// One fused bidirectional flash-attention dispatch over `bsz` samples of `t`
/// rows each in a packed qkv slab - the variant chosen by
/// [`flash_bidir_variant`]. Both kernels take the SAME Params and produce the
/// SAME output layout, so only the pipeline index and the per-workgroup thread
/// count differ; the workgroup still owns BR = 64 query rows in both.
///
/// `bsz > 1` is a **sample-major** slab: sample `b` occupies rows
/// `[b·t, (b+1)·t)` of `qkv` (`[bsz·t, 3·d_model]`) and of `ctx`
/// (`[bsz·t, d_model]`). One workgroup owns one `(b, head, query-tile)`, so
/// samples never mix and the per-query arithmetic is unchanged - a batched
/// dispatch is bit-identical to `bsz` separate ones.
pub fn flash_bidir_step(
    g: &Gpu,
    ids: FlashIds,
    bsz: u32,
    heads: u32,
    t: u32,
    head_dim: u32,
    d_model: u32,
    qkv: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Step {
    assert!(head_dim <= 128, "flash_attn_bidir: head_dim {head_dim} > 128");
    const BR: u32 = 64; // query rows per workgroup - the same in both kernels
    let (kind, ws) = flash_bidir_variant(ids, &g.caps());
    let nwg = bsz * heads * t.div_ceil(BR);
    g.step(
        kind,
        &[qkv, ctx],
        &[bsz, heads, t, head_dim, 3 * d_model, 0, d_model, 2 * d_model, d_model],
        nwg * ws,
    )
}

/// Span-wise fused flash attention: one dispatch per span replaces the whole
/// scores/softmax/apply chain with an online-softmax tiled kernel - O(t·hd)
/// memory AND the tuned inner loops, where the chunked cross trio materializes
/// `[H, chunk, t]` slabs through naive kernels. Picks the kernel through
/// [`flash_bidir_variant`], so a caller that registers `flash_attn_bidir_split`
/// gets it here too. Forward-only and workgroup-cooperative: callers MUST gate
/// on `DeviceCaps::workgroup_reductions` (false on the CPU JIT) and fall back
/// to [`chunked_bidir_fwd`]. `head_dim` ≤ 128.
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
    const BR: u32 = 64; // query rows per workgroup - the same in both kernels
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

/// Round up to a 64-word (256B) boundary - `step_sliced`'s storage-buffer
/// `BufferBinding` offsets must satisfy `min_storage_buffer_offset_alignment`
/// (256B on the near-universal case), so every per-head/per-segment stride
/// used as an offset multiplier in [`gemm_bidir_fwd`] is padded to this grain.
pub fn pad64(words: u64) -> u64 {
    words.div_ceil(64) * 64
}

/// Kernel indices for GEMM attention (see [`gemm_bidir_fwd`]).
#[derive(Clone, Copy)]
pub struct GemmAttnIds {
    pub head_pack: usize,
    pub head_pack_t: usize,
    pub head_unpack: usize,
    /// `softmax_rows` - workgroup-per-row softmax over the `[H·chunk, len]`
    /// slab (GPU-only; the GEMM path is already gated on cooperative devices).
    pub softmax_rows: usize,
    pub matmul: usize,
    pub matmul_reg: usize,
}

/// Query-chunked bidirectional attention as REAL GEMMs: per-head packed
/// operands drive the register-tiled matmul instead of the naive
/// one-thread-per-score kernels. Measured motivation: at t=8192 the naive
/// trio (and the fused flash kernel - a memory escape hatch, not a fast
/// path) left a P40 at ~2% of peak; the same insight already made the CPU
/// fast paths 7× (they route these kernels to the native GEMM).
///
/// Layout: `packs` holds the three per-head-contiguous operands per span -
/// q (scaled by 1/√hd) at 0, k at `len·d_out`, vᵀ at `2·len·d_out` - with
/// GQA replication folded into the pack (`group` reads the NARROW k/v
/// projections; no expanded buffer exists). Scores/probs slabs stay
/// `[H, chunk, len]`; `ctx_pack` collects per-head context, unpacked into
/// the row-major `[rows, d_out]` `ctx` at the end of each span.
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
        let hstride = pad64(len as u64 * hd as u64); // padded per-head stride: keeps every h*hstride offset 256B-aligned
        let seg = heads as u64 * hstride; // one pack region, f32 words
        let total = heads * len * hd;
        // Pack q (scale folded), k, vᵀ for this span.
        steps.push(g.step_sliced(
            k.head_pack,
            &[q, packs],
            &[(row0 as u64 * q_stride as u64, 0), (0, 0)],
            &[len, heads, 1, hd, q_stride, 0, f(scale), hstride as u32],
            total,
        ));
        steps.push(g.step_sliced(
            k.head_pack,
            &[kbuf, packs],
            &[(row0 as u64 * kv_stride as u64, 0), (seg, 0)],
            &[len, heads, group, hd, kv_stride, 0, f(1.0), hstride as u32],
            total,
        ));
        steps.push(g.step_sliced(
            k.head_pack_t,
            &[vbuf, packs],
            &[(row0 as u64 * kv_stride as u64, 0), (2 * seg, 0)],
            &[len, heads, group, hd, kv_stride, 0, f(1.0), hstride as u32],
            total,
        ));
        let mut q0 = 0u32;
        while q0 < len {
            let qn = chunk.min(len - q0);
            let sp_stride = pad64(qn as u64 * len as u64); // padded per-head scores/probs stride, same alignment reason
            for h in 0..heads {
                let (mk, mt) = pick_gemm(qn as usize, len as usize, k.matmul, k.matmul_reg, force_naive);
                // scores[h] = q_pack[h][q0..q0+qn] · k_pack[h]ᵀ   ([qn,hd]·[len,hd]ᵀ)
                steps.push(g.step_sliced(
                    mk,
                    &[packs, packs, scores],
                    &[(h as u64 * hstride + q0 as u64 * hd as u64, 0), (seg + h as u64 * hstride, 0), (h as u64 * sp_stride, 0)],
                    &[qn, hd, len],
                    mt,
                ));
            }
            // Per-head softmax: the shared row-softmax kernel only ever sees its own
            // head's contiguous [qn,len] sub-range, so the head-to-head padding gap
            // (needed for the matmul writes above) stays invisible to it.
            for h in 0..heads {
                steps.push(g.step_sliced(k.softmax_rows, &[scores, probs], &[(h as u64 * sp_stride, 0), (h as u64 * sp_stride, 0)], &[qn, len], qn * 64));
            }
            for h in 0..heads {
                let (mk, mt) = pick_gemm(qn as usize, hd as usize, k.matmul, k.matmul_reg, force_naive);
                // ctx_pack[h][q0..] = probs[h] · V[h]   (A·Bᵀ with B = vᵀ[hd,len])
                steps.push(g.step_sliced(
                    mk,
                    &[probs, packs, ctx_pack],
                    &[(h as u64 * sp_stride, 0), (2 * seg + h as u64 * hstride, 0), (h as u64 * hstride + q0 as u64 * hd as u64, 0)],
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
            &[len, heads, hd, d_out, 0, hstride as u32],
            total,
        ));
    }
}

/// Kernel indices for the query-chunked bidirectional backward: the cross
/// forward pair recomputes each chunk's scores/probs (nothing T×T is cached),
/// `dscores`/`dq` assign chunk-local rows, and the ACCUMULATING `dk_acc`/
/// `dv_acc` twins sum each chunk's partial contribution (their `acc_flag`
/// uniform assigns on a span's first chunk - no zero-clears to forget).
#[derive(Clone, Copy)]
pub struct CrossBwdIds {
    pub dscores: usize,
    pub dq: usize,
    pub dk_acc: usize,
    pub dv_acc: usize,
}

/// Backward of [`chunked_bidir_fwd`] with per-chunk score/softmax recompute -
/// what makes long-context (8k) training fit the per-binding budget: the
/// transient slabs stay `[heads, chunk, max_span]`. Writes `d_q`/`d_k`/`d_v`
/// into their regions of the fused `d_qkv`.
///
/// `rel` is the adjoint of [`chunked_bidir_fwd`]'s own `rel` - SAME `Option`,
/// SAME loop. Three things about it are load-bearing:
///  * the per-chunk score RECOMPUTE must re-apply the bias, or the softmax it
///    feeds is not the one the forward took;
///  * `d_rel_h`/`d_rel_w` are filled chunk by chunk (chunks partition the query
///    rows, so those two ASSIGN), and only then does the span-level pass run;
///  * that pass ACCUMULATES `dq` onto what `bwd.dq` already assigned, and
///    accumulates the dense-table adjoint across spans - every window of a
///    windowed block shares one table.
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
    rel: Option<&crate::vit::RelPos>,
    steps: &mut Vec<Step>,
) {
    for (si, &(row0, len)) in spans.iter().enumerate() {
        if let Some(r) = rel {
            r.check_span(len);
            r.qr_steps(g, heads, head_dim, qkv, stride, row0 * stride + q_off, steps);
        }
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
            if let Some(r) = rel {
                steps.push(r.add_step(g, heads, qn, len, q0, scores));
            }
            steps.push(g.step(fwd.softmax, &[scores, probs], &[1, heads, qn, len], heads * qn));
            // Softmax jacobian → d_scores (chunk-local).
            steps.push(g.step_sliced(
                bwd.dscores,
                &[d_ctx, qkv, probs, d_scores],
                &[(dc_off, 0), (kv_row_off, 0), (0, 0), (0, 0)],
                &p_v,
                heads * qn,
            ));
            // d_q: chunk rows only (disjoint - plain assign into the q region).
            steps.push(g.step_sliced(
                bwd.dq,
                &[d_scores, qkv, d_qkv],
                &[(0, 0), (kv_row_off, 0), (q_row_off, 0)],
                &p_qk,
                heads * qn * head_dim,
            ));
            // d_k / d_v: sums over ALL queries - accumulate across chunks.
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
            // The bias is purely ADDITIVE, so d(bias) == d_scores exactly: the
            // two intermediates' adjoints are partial sums of this same slab.
            if let Some(r) = rel {
                r.drel_steps(g, heads, qn, len, q0, d_scores, steps);
            }
            q0 += qn;
        }
        // Span-level: dq (accumulating onto what `bwd.dq` assigned above) and
        // the dense-table adjoint (accumulating across spans).
        if let Some(r) = rel {
            let acc = si > 0 || r.bwd.as_ref().is_some_and(|b| b.acc0);
            r.span_bwd_steps(g, heads, head_dim, qkv, d_qkv, stride, row0 * stride + q_off, acc, steps);
        }
    }
}

/// GQA→MHA head replication into a fused-buffer region: dst head `ho` copies
/// src head `ho/group` (`repeat_kv` layout). `group == 1` is a strided copy -
/// the same dispatch places q. `src` is `[rows, (heads_out/group)*hd]`.
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

/// `(kernel index, dispatch threads)` for one RMSNorm - the RMSNorm twin of
/// `ln_variant`: the cooperative workgroup-per-row kernel (`rmsnorm_rows`,
/// measured 19.4x on a P40 and a win at *every* row width because the
/// per-element kernel's one-thread-per-row layout is uncoalesced) where the
/// model registered it and the device can run a workgroup reduction, else the
/// per-element reference (`rmsnorm` / `rmsnorm_eps`).
///
/// Both variants take the same buffers `[x, w, out]` and the same Params
/// `[d, rows, eps]`, so only the index and the thread count change - which is
/// why this returns them instead of building the `Step`: callers bind whole
/// buffers (`Gpu::step`) or slices (`Gpu::step_sliced`) and both must share one
/// selection rule. The policy itself lives in `backend_api::select`
/// (`Op::RmsNorm`) keyed on `DeviceCaps`, never on a backend name; the `*_rows`
/// kernels are `@workgroup_size(64)`, at or below the WebGPU floor of 256, so
/// no `max_workgroup_size` gate is needed on top of it.
pub fn rms_variant(g: &Gpu, reference: usize, coop: Option<usize>, rows: u32, d: u32) -> (usize, u32) {
    use gpu_core::select::{Dtype, KernelSelector, KernelVariant, Op, OpShape};
    let shape = OpShape { m: rows, n: d, k: 0, dtype: Dtype::F32 };
    match coop {
        Some(i)
            if gpu_core::select::DefaultSelector.select(Op::RmsNorm, shape, &g.caps())
                == KernelVariant::WorkgroupPerOutput =>
        {
            (i, rows * 64)
        }
        _ => (reference, rows),
    }
}

/// RMSNorm backward with runtime epsilon: input grad always (`rmsnorm_dx_eps`),
/// gain grad only when `gw` is `Some` (`rms_inv_eps` + `rmsnorm_dw`; the dw
/// kernel is eps-free - eps enters through the per-row inverse).
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
/// to one row and are coalesced by construction - measured 2.3-9.1x on a P40
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
/// `@workgroup_size(64)` - at or below the WebGPU floor of 256 - so no
/// `max_workgroup_size` gate is needed on top of it.
///
/// Public for the same reason [`gemm_variant`] returns indices rather than a
/// `Step`: [`layernorm_fwd`] and friends bind whole buffers (`Gpu::step`), while
/// the DiT forwards normalise a ROW RANGE of a joint slab and must bind
/// sub-ranges (`Gpu::step_sliced`). Both shapes have to share ONE selection
/// rule - a second copy is a place a model silently keeps the slow kernel,
/// the most expensive class of defect there is.
pub fn ln_variant(g: &Gpu, reference: usize, coop: Option<usize>, rows: u32, d: u32) -> (usize, u32) {
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

/// Fallback per-binding budget (f32 words) for tiling an embedding / lm_head
/// over vocab, used only when the device's real limit is not available.
///
/// ~96 MiB, chosen for the smallest limit brain has met (128 MB on Mesa-GL).
/// `BRAIN_TILE_BUDGET_WORDS` overrides everything (e.g. tiny, to force tiling in
/// tests).
pub const TILE_BUDGET_WORDS: u64 = 24 * 1024 * 1024;

/// Safety margin on the device's reported binding limit. The tiling rule sizes
/// the *weight* slice; the same dispatch also binds the logits and the hidden
/// state, so it does not spend the whole limit on one operand.
const TILE_BUDGET_FRACTION: u64 = 2;

pub fn tile_budget_words() -> u64 {
    std::env::var("BRAIN_TILE_BUDGET_WORDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(TILE_BUDGET_WORDS)
}

/// The budget for `gpu`, from its **queried** max storage-binding size.
///
/// The constant above is a floor for an unknown device, and using it on a known
/// one was expensive: a P40 reports 2047 MiB, so the fixed ~96 MiB tiled 21×
/// more finely than necessary. On Qwen3-0.6B that split the tied 151936×1024
/// head into **7 tiles**, and the caller only routes to the register-tiled GEMM
/// when the vocab collapses to ONE (a tiled head has to use `matmul_tile`, the
/// naive kernel). Measured: that head was **90.8% of a T=512 prefill at 0.4% of
/// the compute roof**, and letting it collapse took the whole pass
/// **4375.51 → 361.36 ms (12.1×)**, 117 → 1417 tok/s.
///
/// An explicit `BRAIN_TILE_BUDGET_WORDS` still wins, so tests can force tiling.
pub fn tile_budget_words_for(gpu: &Gpu) -> u64 {
    if let Some(w) =
        std::env::var("BRAIN_TILE_BUDGET_WORDS").ok().and_then(|s| s.parse::<u64>().ok()).filter(|&w| w > 0)
    {
        return w;
    }
    // Queried, never assumed - and never smaller than the portable floor, so a
    // backend that under-reports cannot make the tiling worse than it was.
    let bytes = gpu.max_storage_binding_bytes() / TILE_BUDGET_FRACTION;
    (bytes / 4).max(TILE_BUDGET_WORDS)
}

/// Vocab tiles `(v0, count)` sized so a `[count, d_model]` weight slice stays
/// within the per-binding budget. Small vocabularies yield a single tile.
///
/// Prefer [`vocab_tiles_on`] wherever a device is in hand: this form has to
/// assume the portable floor, and assuming it on a card that reports 20× more
/// is what made the Qwen3-0.6B head 90% of its own prefill.
pub fn vocab_tiles(vocab: u64, d_model: u64) -> Vec<(u32, u32)> {
    tiles_with_budget(vocab, d_model, tile_budget_words())
}

/// [`vocab_tiles`] against the device's own queried binding limit.
pub fn vocab_tiles_on(gpu: &Gpu, vocab: u64, d_model: u64) -> Vec<(u32, u32)> {
    tiles_with_budget(vocab, d_model, tile_budget_words_for(gpu))
}

fn tiles_with_budget(vocab: u64, d_model: u64, budget: u64) -> Vec<(u32, u32)> {
    let rows = (budget / d_model.max(1)).max(1);
    let mut out = Vec::new();
    let mut v0 = 0u64;
    while v0 < vocab {
        let cnt = rows.min(vocab - v0);
        out.push((v0 as u32, cnt as u32));
        v0 += cnt;
    }
    out
}

/// Pick the GEMM kernel + dispatch thread count for `[m,k]·[n,k]ᵀ`: the
/// register-tiled kernel (128×128 tile, 256 threads) when there is enough work
/// to fill tiles, else the naive one-thread-per-output kernel. Same math either
/// way - every variant is bit-identical to the naive reference (measured,
/// `max|Δ| = 0`), so this only changes speed. `force_naive` is a model's env
/// escape.
///
/// # `M` is NOT required to fill a tile, and requiring it cost 22x
///
/// The rule used to be `m < 128 || n < 128 -> naive`, on the reading that a
/// partial tile is wasted. The tiled kernel bounds-guards its tile, so a short
/// `M` costs only the unused rows - while the naive kernel gives one thread per
/// output element, each walking `k` serially, which collapses on a wide `N`.
///
/// SDXL's cross-attention `kv` projection is `[77, 2048, 2560]`: 77 text tokens
/// is under the old threshold, so 60 of those per forward took the naive path at
/// **43 GFLOP/s - 0.4% of a P40's 11.76 TFLOP/s peak, and 49% of the entire UNet
/// forward**. Measured on a P40, `k = 2048`, `n = 2560`:
///
/// | m | naive | tiled |
/// |---|---|---|
/// | 1 | **0.19 ms** | 0.48 ms |
/// | 2 | **0.37** | 0.73 |
/// | 4 | **0.43** | 0.77 |
/// | 8 | 0.89 | **0.77** |
/// | 12 | 1.08 | **0.78** |
/// | 77 | 18.67 | **0.84**  (22x) |
///
/// So the crossover is `m = 8`, not 128. Below it the tile genuinely is mostly
/// idle and naive wins; at `m = 1` the right kernel is neither - it is
/// `matmul_gemv` (one workgroup per output column), which `gemm_variant`
/// selects for models that register it.
pub fn pick_gemm(m: usize, n: usize, naive: usize, reg2: usize, force_naive: bool) -> (usize, u32) {
    if force_naive || m < 8 || n < 128 {
        (naive, (m * n) as u32)
    } else {
        (reg2, (m.div_ceil(128) * n.div_ceil(128) * 256) as u32)
    }
}

/// Which forward-GEMM kernels a model registered, for [`gemm_variant`].
///
/// `pick_gemm` above answers the *training-shaped* question ("is this output
/// big enough to fill a 128×128 tile?"). This answers the *inference-shaped*
/// one the DiT models ask: the graph is a fixed list of linears whose M is
/// either a whole token slab or a single conditioning row, and the tier
/// (naive reference vs. the fast kernels) is a property of the model, not of
/// the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmVariants {
    /// One thread per output element. The portable reference: no workgroup
    /// reduction, no tiling, runs anywhere. Selected when the model is in its
    /// reference tier (`fast == false`).
    Reference(usize),
    /// The fast tier.
    Fast {
        /// Skinny-M GEMV - one WORKGROUP per output COLUMN, reading each weight
        /// row once and applying it to all M rows from registers. `None` when
        /// the model did not register it. **Requires `m <= 32`** (the kernel
        /// says so); `gemm_variant` enforces that, so a model may pass `Some`
        /// unconditionally.
        gemv: Option<usize>,
        /// 128×128 register-tiled GEMM, `@workgroup_size(256)`.
        tiled: usize,
    },
}

/// Pick the forward GEMM kernel + dispatch thread count for `[m,k]·[n,k]ᵀ`.
///
/// This exists because the rule was written twice and the two copies **drifted
/// apart in the useful direction**: `flux1` learned that a register-tiled GEMM
/// at M=1 wastes 127/128 of every tile and routed skinny-M to the GEMV kernels;
/// `flux2`, written first, never did, so every one of its per-token modulation
/// mat-vecs paid the full tile. That is a fast kernel a later model never
/// learned about, and the answer is to put the fix in *selection*, in one
/// place, not in a second copy.
///
/// The same rule serves the fp32 and the int8 (DP4A) tiers: the two families
/// take different buffers but identical dispatch geometry, so int8 callers pass
/// their own `matmul_i8_gemv` / `matmul_i8_dyn` indices and always the `Fast`
/// arm (the DP4A path is GPU-only and has no naive sibling).
///
/// Returns `(kernel index, invocation count)` rather than a `Step` for the same
/// reason [`rms_variant`] does: callers bind whole buffers (`Gpu::step`) or
/// sub-ranges (`Gpu::step_sliced`) and both must share one selection rule.
/// Every arm computes the same math - this only changes speed.
pub fn gemm_variant(v: GemmVariants, m: u32, n: u32) -> (usize, u32) {
    match v {
        GemmVariants::Reference(k) => (k, m * n),
        GemmVariants::Fast { gemv: Some(g), .. } if m <= 32 => (g, n * 64),
        GemmVariants::Fast { tiled, .. } => (tiled, m.div_ceil(128) * n.div_ceil(128) * 256),
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

/// [`gqa_decode_step`], called once per position `0..t`, must reproduce
/// [`gqa_fwd`]'s causal batched output exactly at every row - `gqa_scores.wgsl`
/// already masks `j > i` to `-inf` (see its header), so [`gqa_fwd`]'s row `i`
/// already only attends keys `0..=i`, the same set a decode step at `pos = i`
/// sees from the cache. This is the `model::block` twin of `qwen3::Qwen`'s own
/// `cache_matches_full_recompute` test, proving the hoisted primitive (not
/// just qwen's original inline copy) is algebraically exact before a second
/// model (`omni::thinker`) builds on it.
#[cfg(test)]
mod kv_cache_tests {
    use super::*;
    use gpu_core::Gpu;

    #[test]
    fn decode_step_matches_causal_batched_attention_at_every_position() {
        let (t, n_heads, n_kv_heads, head_dim) = (5u32, 4u32, 2u32, 8u32);
        let (hq, hkv) = (n_heads * head_dim, n_kv_heads * head_dim);

        let gpu = Gpu::new_cpu(&[
            ("gqa_scores", kernels::GQA_SCORES),
            ("attn_softmax", kernels::ATTN_SOFTMAX),
            ("gqa_apply", kernels::GQA_APPLY),
            ("kv_append", kernels::KV_APPEND),
            ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
            ("decode_softmax", kernels::DECODE_SOFTMAX),
            ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
        ]);
        let ids = KernelIds {
            rmsnorm: 0,
            rms_inv: 0,
            rmsnorm_dx: 0,
            rmsnorm_dw: 0,
            rope: 0,
            rope_bwd: 0,
            gqa_scores: 0,
            gqa_apply: 2,
            attn_softmax: 1,
            gqa_dscores: 0,
            gqa_dv: 0,
            gqa_dq: 0,
            gqa_dk: 0,
            silu_mul: 0,
            silu_da: 0,
            silu_db: 0,
        };
        let decode_ids = GqaDecodeIds { kv_append: 3, attn_decode_scores: 4, decode_softmax: 5, attn_decode_apply: 6 };

        // Deterministic pseudo-random q/k/v -- fixed formula, no RNG (engine convention).
        let mk = |n: u32, seed: f32| (0..n).map(|i| (i as f32 * 0.7 + seed).sin()).collect::<Vec<f32>>();
        let q_host = mk(t * hq, 0.1);
        let k_host = mk(t * hkv, 0.2);
        let v_host = mk(t * hkv, 0.3);

        // Full batched causal attention (the reference).
        let q = gpu.storage_init("q", &q_host);
        let k = gpu.storage_init("k", &k_host);
        let v = gpu.storage_init("v", &v_host);
        let scores_full = gpu.storage((n_heads * t * t) as u64);
        let probs_full = gpu.storage((n_heads * t * t) as u64);
        let ctx_full = gpu.storage((t * hq) as u64);
        let ga = Gqa { b: 1, t, n_heads, n_kv_heads, head_dim };
        gpu.submit(&[], &gqa_fwd(&gpu, &ids, &ga, &q, &k, &v, &scores_full, &probs_full, &ctx_full));
        let want = gpu.read(&ctx_full, (t * hq) as usize);

        // Incremental decode: append one position at a time, compare each
        // step's ctx row against the batched reference's same row.
        let cap = t;
        let kcache = gpu.storage((cap * hkv) as u64);
        let vcache = gpu.storage((cap * hkv) as u64);
        let scores = gpu.storage((n_heads * cap) as u64);
        let probs = gpu.storage((n_heads * cap) as u64);
        for pos in 0..t {
            let q_row = gpu.storage_init("q_row", &q_host[(pos * hq) as usize..((pos + 1) * hq) as usize]);
            let k_row = gpu.storage_init("k_row", &k_host[(pos * hkv) as usize..((pos + 1) * hkv) as usize]);
            let v_row = gpu.storage_init("v_row", &v_host[(pos * hkv) as usize..((pos + 1) * hkv) as usize]);
            let ctx = gpu.storage(hq as u64);
            gpu.submit(
                &[],
                &gqa_decode_step(&gpu, &decode_ids, n_heads, n_kv_heads, head_dim, pos, cap, &q_row, &k_row, &v_row, &kcache, &vcache, &scores, &probs, &ctx),
            );
            let got = gpu.read(&ctx, hq as usize);
            let want_row = &want[(pos * hq) as usize..((pos + 1) * hq) as usize];
            for (i, (g, w)) in got.iter().zip(want_row).enumerate() {
                assert!((g - w).abs() < 1e-4, "pos {pos} elem {i}: got {g}, want {w}");
            }
        }
    }

    /// [`kv_cache_fill`]'s bulk-copy path (batched prefill -> cache) must land
    /// the exact same bytes a per-row [`gqa_decode_step`] loop would have
    /// appended -- proven by filling one cache via the bulk path and a second
    /// via `kv_append` called once per row, then comparing.
    #[test]
    fn bulk_fill_matches_per_row_append() {
        let (n, n_kv_heads, head_dim) = (4u32, 2u32, 8u32);
        let hkv = n_kv_heads * head_dim;
        let gpu = Gpu::new_cpu(&[("kv_append", kernels::KV_APPEND)]);
        let src_host: Vec<f32> = (0..n * hkv).map(|i| i as f32 * 0.5).collect();
        let src = gpu.storage_init("src", &src_host);

        let bulk = gpu.storage((n * hkv) as u64);
        gpu.submit(&[], &[kv_cache_fill(&gpu, 0, &src, &bulk, n, n_kv_heads, head_dim)]);

        let per_row = gpu.storage((n * hkv) as u64);
        for row in 0..n {
            let row_buf = gpu.storage_init("row", &src_host[(row * hkv) as usize..((row + 1) * hkv) as usize]);
            gpu.submit(&[], &[gpu.step(0, &[&row_buf, &per_row], &[hkv, row], hkv)]);
        }

        assert_eq!(gpu.read(&bulk, (n * hkv) as usize), gpu.read(&per_row, (n * hkv) as usize));
    }
}

#[cfg(test)]
mod tests {
    use super::{gemm_variant, tiles_with_budget, GemmVariants, TILE_BUDGET_FRACTION, TILE_BUDGET_WORDS};

    /// Pins the three arms and, in particular, the `m <= 32` precondition the
    /// GEMV kernels state in their headers: violating it is silently wrong
    /// output, not a crash, so the bound belongs in the selector and nowhere
    /// else. Thread counts are pinned too - they are the kernels' documented
    /// dispatch geometry (one workgroup per output column; one 256-thread
    /// workgroup per 128x128 output tile), not free parameters.
    #[test]
    fn gemm_variant_routes_skinny_m_to_the_gemv_kernel() {
        let fast = GemmVariants::Fast { gemv: Some(7), tiled: 9 };
        assert_eq!(gemm_variant(fast, 1, 3072), (7, 3072 * 64));
        assert_eq!(gemm_variant(fast, 32, 3072), (7, 3072 * 64));
        // One row past the kernel's stated limit: the tile takes over.
        assert_eq!(gemm_variant(fast, 33, 3072), (9, 24 * 256));
        assert_eq!(gemm_variant(fast, 512, 3072), (9, 4 * 24 * 256));

        // A model that never registered the GEMV kernel keeps the tiled arm at
        // every M - this is what makes the migration of an existing user
        // provably behaviour-preserving before the kernel is added.
        let no_gemv = GemmVariants::Fast { gemv: None, tiled: 9 };
        assert_eq!(gemm_variant(no_gemv, 1, 3072), (9, 24 * 256));
        assert_eq!(gemm_variant(no_gemv, 512, 3072), (9, 4 * 24 * 256));

        // The reference tier ignores both fast kernels at every shape.
        assert_eq!(gemm_variant(GemmVariants::Reference(2), 1, 3072), (2, 3072));
        assert_eq!(gemm_variant(GemmVariants::Reference(2), 512, 3072), (2, 512 * 3072));
    }

    /// The tiling rule, against the budget rather than against a device - the
    /// device half is one `max_storage_binding_bytes()` call.
    ///
    /// This exists because the fixed ~96 MiB budget was 21x smaller than what a
    /// P40 reports, which split Qwen3-0.6B's tied 151936x1024 head into 7 tiles;
    /// the caller only routes to the register-tiled GEMM when the vocab
    /// collapses to ONE tile, so the head ran the naive kernel at 0.4% of the
    /// compute roof and was 90.8% of a T=512 prefill (4375.51 -> 361.36 ms once
    /// it collapsed).
    #[test]
    fn a_real_lm_head_collapses_to_one_tile_on_a_device_that_reports_its_limit() {
        let (vocab, d) = (151936u64, 1024u64);

        // The portable floor still tiles it 7 ways - the behaviour that cost 12x.
        assert_eq!(tiles_with_budget(vocab, d, TILE_BUDGET_WORDS).len(), 7);

        // Half of a P40's reported 2047 MiB, in f32 words.
        let p40 = (2047 * 1024 * 1024 / TILE_BUDGET_FRACTION) / 4;
        assert_eq!(tiles_with_budget(vocab, d, p40).len(), 1);

        // A model too large for one binding still tiles, and every tile fits
        // inside the real limit with the safety fraction to spare: Qwen3-8B's
        // head is 151936x4096 = 2.49 GB, past the 2047 MiB binding ceiling.
        let big = tiles_with_budget(vocab, 4096, p40);
        assert!(big.len() > 1, "a 2.49 GB head must not become one binding");
        for (_, cnt) in &big {
            let bytes = *cnt as u64 * 4096 * 4;
            assert!(bytes <= 2047 * 1024 * 1024, "tile of {bytes} B exceeds the binding limit");
        }

        // Tiles cover the vocab exactly, with no gap and no overlap.
        let mut next = 0u32;
        for (v0, cnt) in &big {
            assert_eq!(*v0, next);
            next += cnt;
        }
        assert_eq!(next as u64, vocab);
    }

    /// A small budget must still force the tiled path, which is what
    /// `crates/t5/tests/smoke.rs` relies on to exercise the sliced-binding
    /// `embed_tile` branch at toy size (it sets `BRAIN_TILE_BUDGET_WORDS=4096`,
    /// and `tile_budget_words_for` honours that ahead of any device limit).
    #[test]
    fn a_small_budget_still_forces_the_tiled_path() {
        // 512 words / d_model 8 = 64 rows per tile over a 4096 vocab.
        let t = tiles_with_budget(4096, 8, 512);
        assert_eq!(t.len(), 64);
        assert_eq!(t[0], (0, 64));

        // A budget smaller than one row still yields whole rows, never zero -
        // a zero-row tile would loop forever.
        assert_eq!(tiles_with_budget(3, 1024, 1).len(), 3);
    }
}
