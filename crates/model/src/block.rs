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

use gpu_core::select::{self, KernelVariant};
use gpu_core::{f, DeviceBuffer, DeviceCaps, DeviceClass, Gpu, Step};

/// The "this model registered no kernel for this slot" sentinel, for every
/// index set in this module - the `model::block` twin of
/// [`crate::vit::UNREGISTERED`], and the same value.
///
/// A slot holding it must never be dispatched. The builders that can do
/// without a slot check for it; the ones that cannot fail loudly on an
/// out-of-range pipeline index instead of silently running whatever kernel
/// happens to sit at the index that was written there.
///
/// **Filling an unused slot with `0` instead is a silent-corruption defect,
/// not a harmless placeholder.** Index 0 is a real, registered kernel in every
/// PIPELINES list in this workspace, so a misroute through such a slot runs
/// that kernel with another kernel's bindings and uniform. On a GPU backend the
/// binding check turns it into a panic; on `backend-cpu` there is no
/// buffer-count or uniform-size check at dispatch, so it is an out-of-bounds
/// read that no unit test on that backend can see.
pub const UNREGISTERED: usize = usize::MAX;

/// Kernel-pipeline indices a model supplies from its own PIPELINES list. Only
/// the kernels a given helper uses need valid indices; every other slot is
/// [`UNREGISTERED`].
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
    /// The COALESCED RMSNorm forward twin (`rmsnorm_rows`), or [`UNREGISTERED`]
    /// when the model has not registered that pipeline. Purely a performance
    /// slot: [`rmsnorm_fwd`] selects between it and `rmsnorm` through
    /// [`rms_variant`], and a model that leaves it unregistered keeps
    /// dispatching the per-element reference exactly as before.
    ///
    /// It is a slot on [`KernelIds`] rather than an argument because the
    /// builders that pay the most for the naive kernel are the SHARED ones -
    /// [`gqa_attn_qkv`]'s three norms, [`crate::gqa_mixer::gqa_mixer_fwd`]'s
    /// QK-norms, [`crate::gdn_mixer::gdn_mixer_fwd`]'s gated norm - which no
    /// model can reach from its own call sites. Registering one pipeline and
    /// filling one slot switches every RMSNorm a model composes, including
    /// the ones inside those builders.
    pub rmsnorm_rows: usize,
}

impl KernelIds {
    /// Every slot, paired with its field name - the enumeration a model's own
    /// "no unused slot is dispatchable" gate walks. It lives here, next to the
    /// struct, so adding a field cannot leave a gate silently checking 16 of 17
    /// slots.
    pub fn slots(&self) -> [(&'static str, usize); 17] {
        [
            ("rmsnorm", self.rmsnorm),
            ("rmsnorm_rows", self.rmsnorm_rows),
            ("rms_inv", self.rms_inv),
            ("rmsnorm_dx", self.rmsnorm_dx),
            ("rmsnorm_dw", self.rmsnorm_dw),
            ("rope", self.rope),
            ("rope_bwd", self.rope_bwd),
            ("gqa_scores", self.gqa_scores),
            ("gqa_apply", self.gqa_apply),
            ("attn_softmax", self.attn_softmax),
            ("gqa_dscores", self.gqa_dscores),
            ("gqa_dv", self.gqa_dv),
            ("gqa_dq", self.gqa_dq),
            ("gqa_dk", self.gqa_dk),
            ("silu_mul", self.silu_mul),
            ("silu_da", self.silu_da),
            ("silu_db", self.silu_db),
        ]
    }
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

/// The epsilon `rmsnorm.wgsl` hardcodes. [`rmsnorm_fwd`] has to pass it
/// explicitly because its two variants share one `Params` layout and the
/// cooperative one reads a third `eps` field; a two-field list would hand it
/// whatever the uniform happened to hold.
pub const RMSNORM_EPS: f32 = 1e-6;

/// RMSNorm forward: `out = (x / rms(x)) * w` over the last `dim` axis, one row
/// per invocation (`rows` total).
///
/// Dispatches the coalesced `rmsnorm_rows` when the model registered it
/// ([`KernelIds::rmsnorm_rows`]) and the device can run a workgroup reduction,
/// the per-element `rmsnorm` otherwise - the choice is [`rms_variant`]'s, i.e.
/// `backend_api::select`'s, never a backend name's.
///
/// Why the seam is here and not only at model call sites: `rmsnorm.wgsl` gives
/// thread `t` row `t`, so a warp's 32 loads are `dim` floats apart and every
/// 32-byte sector fetched serves ONE useful float. That penalty is worst at
/// the `rows = 1` of a decode step - measured on a real two-card Qwen3.8-27B
/// it was 48% of all device time per token, more than the whole int8 weight
/// stream underneath it - but the swept comparison in `rmsnorm_rows.wgsl` wins
/// at every row width. Models compose this builder from inside
/// [`gqa_attn_qkv`], [`crate::gqa_mixer`] and [`crate::gdn_mixer`], so a
/// per-model fix cannot reach those norms at all.
///
/// NOT a bit-identical swap: 64 partial sums fold in a different order,
/// agreeing to ~3e-6 max_abs. That is why it lives behind a registration a
/// model opts into, and why every adopting model gates it with a
/// variant-agreement test against a HOST reference.
pub fn rmsnorm_fwd(g: &Gpu, k: &KernelIds, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32) -> Step {
    let coop = (k.rmsnorm_rows != UNREGISTERED).then_some(k.rmsnorm_rows);
    let (kind, threads) = rms_variant(g, k.rmsnorm, coop, rows, dim);
    g.step(kind, &[x, w, out], &[dim, rows, f(RMSNORM_EPS)], threads)
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
/// (`qwen3vl::mrope::mrope_tables`) instead of a single scalar position - the
/// seam that lets a caller feed genuinely divergent per-axis (text/image/
/// video/audio) positions, or the degenerate all-axes-equal case (which
/// `qwen3vl::mrope`'s own test proves collapses to identical output). `qwen3::
/// Qwen::rope2d_step` already dispatches this exact kernel for Qwen3-VL;
/// hoisted here so a second model (`qwen3omnimoe::thinker`) doesn't re-wire it.
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
/// `qwen3vl::mrope::mrope_tables` called with `head_dim = rot_dim`.
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
/// model (`qwen3omnimoe::thinker`, a 48-layer MoE decoder) reuses the exact same
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
// attend -> out-proj -> residual), hoisted from qwen3omnimoe::thinker/qwen3omnimoe::talker ---
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
    /// `flash_attn_causal_gqa.wgsl`'s pipeline index, when a caller has
    /// registered it - `None` (every caller before this field existed)
    /// keeps [`gqa_attn_sublayer_fwd`] on the original `gqa_fwd` path
    /// (materialized `[H,T,T]` scores/probs), byte-for-byte unchanged.
    /// `Some` switches to the O(T*head_dim)-memory flash-attention kernel
    /// instead - see that kernel's doc for the real
    /// `ERROR_OUT_OF_DEVICE_MEMORY` (a real agent's long system prompt +
    /// tool schemas, thousands of tokens) this closes.
    pub flash_causal_gqa: Option<usize>,
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

    let ga = Gqa { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
    let ctx = g.storage((n * nh * hd) as u64);
    match ids.flash_causal_gqa {
        // Same gate every other `flash_bidir_fwd` caller in this codebase uses
        // (flux1/flux2/lfm/unet): shared memory + `workgroupBarrier` need real
        // workgroup cooperation, which the CPU JIT backend (and any other
        // backend that reports it) does not have -- dispatching there panics
        // instead of computing a wrong answer, so this falls back to the
        // materialized path rather than assuming every registered kernel id
        // is safe to run on every device. The kernel is also lane-split at a
        // fixed `@workgroup_size(256)` (see its own doc), so a device capped
        // below that - `flash_bidir_variant`'s own check for its split kernel -
        // falls back the same way rather than dispatching a workgroup size
        // the device cannot run.
        Some(flash) if g.caps().workgroup_reductions && g.caps().max_workgroup_size >= 256 => {
            g.submit(&[], &[flash_gqa_causal_fwd(g, flash, &ga, &q, &k, &v, &ctx)])
        }
        _ => {
            let scores = g.storage((nh * n * n) as u64);
            let probs = g.storage((nh * n * n) as u64);
            g.submit(&[], &gqa_fwd(g, &ids.kernels, &ga, &q, &k, &v, &scores, &probs, &ctx));
        }
    }

    gqa_attn_out(g, ids, dims, w, x, &ctx, n)
}

/// One fused causal GQA flash-attention dispatch: [`gqa_fwd`]'s
/// scores -> softmax -> apply chain, FUSED into one tiled online-softmax
/// kernel (`crates/kernels/wgsl/flash_attn_causal_gqa.wgsl`) so the dense
/// `[H,T,T]` scores/probs slabs are never materialized - peak attention
/// memory is O(T*head_dim) instead of O(T*T). Same separate-q/k/v-buffer,
/// GQA-head-group layout `gqa_fwd` uses (unlike [`flash_bidir_fwd`]'s fused-
/// qkv, non-causal, plain-MHA shape) so it drops into
/// [`gqa_attn_sublayer_fwd`] with no upstream layout change.
///
/// Lane-split across `head_dim` (`flash_attn_bidir_split`'s own fix for the
/// same real Pascal register-spill bug, worth well over an order of magnitude
/// at head_dim=128 - see that kernel's header and this one's own doc for the
/// account of the register-per-thread design that was tried first and measured
/// unusably slow). The workgroup still owns `BR = 64` query rows, same as
/// [`flash_bidir_step`]'s convention, but each row is now split across 4
/// lanes, so the workgroup's real thread count is `256`, not `BR`.
///
/// `Gqa::params()`'s field order (`[b, n_heads, n_kv_heads, t, head_dim,
/// group]`) already matches the kernel's own `Params` struct exactly - same
/// contract `gqa_fwd`/`gqa_scores.wgsl`/`gqa_apply.wgsl` share.
pub fn flash_gqa_causal_fwd(g: &Gpu, kernel: usize, a: &Gqa, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, ctx: &DeviceBuffer) -> Step {
    const BR: u32 = 64; // query rows per workgroup - matches the kernel's own BR
    const WS: u32 = 256; // threads per workgroup - matches the kernel's own workgroup_size(256) (BR * LANES)
    let nwg = a.b * a.n_heads * a.t.div_ceil(BR);
    g.step(kernel, &[q, k, v, ctx], &a.params(), nwg * WS)
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

/// The coalesced replacement for the `attn_scores_cross` dispatch: `kv_k_headt`
/// transposes a span's K region into the key-minor `[d_model, kn]` scratch
/// `kt`, and `attn_scores_cross_kt` reads that instead of the fused KV slab.
///
/// `attn_scores_cross` parallelises over the KEY index and reduces over
/// `head_dim`, so consecutive lanes land `kv_stride` floats apart and every
/// lane costs its own memory transaction. Its twin `attn_apply_cross` moves the
/// same bytes with a contiguous thread index and is several times faster on the
/// same device, which is what identifies this as a layout defect rather than a
/// roofline. The transpose is O(kn·d_model) and the sweep it feeds is
/// O(qn·kn·d_model), so it pays for itself as soon as a span has more than a
/// couple of query rows.
///
/// `None` at every call site leaves the fused-KV dispatch byte for byte what it
/// was, which is what lets a model adopt this one at a time against its own
/// parity numbers.
#[derive(Clone, Copy)]
pub struct KeyMinor<'a> {
    /// `kv_k_headt` pipeline index.
    pub transpose: usize,
    /// `attn_scores_cross_kt` pipeline index.
    pub scores: usize,
    /// `[d_model, max_kn]` scratch, `d_model = heads*head_dim`. Sized for the
    /// LONGEST span the caller will pass; it is rewritten per span.
    pub kt: &'a DeviceBuffer,
}

/// One score-slab dispatch, through whichever of the two paths the caller
/// enabled. Same output buffer, same `((h*qn)+i)*kn + j` layout, same values -
/// only where K is read from differs.
///
/// The `kt` path needs [`key_minor_step`] to have run for this span first, and
/// that dispatch must be HOISTED out of any query-chunk loop: K does not vary
/// with the query chunk, and re-transposing per chunk throws away the reuse the
/// transpose exists to buy.
///
/// `q_slice`/`kv_slice` are the binding-level `(offset, len)` in floats, for
/// callers that slice their row window rather than folding it into `q_off`/
/// `k_off`; `(0, 0)` binds whole.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cross_scores_step(
    g: &Gpu,
    k: &CrossIds,
    km: Option<&KeyMinor>,
    a: CrossScoreArgs,
    q: &DeviceBuffer,
    q_slice: (u64, u64),
    kv: &DeviceBuffer,
    kv_slice: (u64, u64),
    scores: &DeviceBuffer,
) -> Step {
    let CrossScoreArgs { heads, head_dim, q_stride, q_off, kv_stride, k_off, qn, kn } = a;
    match km {
        Some(km) => g.step_sliced(
            km.scores,
            &[q, km.kt, scores],
            &[q_slice, (0, 0), (0, 0)],
            &[1, heads, qn, kn, head_dim, q_stride, q_off],
            heads * qn * kn,
        ),
        None => g.step_sliced(
            k.scores,
            &[q, kv, scores],
            &[q_slice, kv_slice, (0, 0)],
            &[1, heads, qn, kn, head_dim, q_stride, kv_stride, q_off, k_off],
            heads * qn * kn,
        ),
    }
}

/// The nine numbers [`cross_scores_step`] needs that are not buffers - grouped
/// so the two layouts' parameter lists stay one argument list rather than two.
#[derive(Clone, Copy)]
pub(crate) struct CrossScoreArgs {
    pub heads: u32,
    pub head_dim: u32,
    pub q_stride: u32,
    pub q_off: u32,
    pub kv_stride: u32,
    pub k_off: u32,
    pub qn: u32,
    pub kn: u32,
}

/// The `kv_k_headt` dispatch that fills [`KeyMinor::kt`] with this span's `kn`
/// key rows. `k_off` is the region offset with the span's first row folded in
/// exactly as the scores kernel's own `k_off` carries it, unless the caller
/// slices `kv` instead (`kv_slice`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn key_minor_step(
    g: &Gpu,
    km: &KeyMinor,
    d_model: u32,
    kv: &DeviceBuffer,
    kv_slice: (u64, u64),
    kv_stride: u32,
    k_off: u32,
    kn: u32,
) -> Step {
    g.step_sliced(km.transpose, &[kv, km.kt], &[kv_slice, (0, 0)], &[kn, d_model, kv_stride, k_off], d_model * kn)
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
///
/// `km` swaps the score dispatch for the coalesced [`KeyMinor`] pair. This loop
/// is the best case for it: the transpose is hoisted to once per SPAN while the
/// score sweep runs once per query CHUNK, so a span of `len` rows at chunk size
/// `c` amortises one `len·d_out` transpose over `len/c` sweeps of `c·len·d_out`
/// reads each.
pub fn chunked_bidir_fwd(
    g: &Gpu,
    k: &CrossIds,
    km: Option<&KeyMinor>,
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
        let kv_row_off = row0 as u64 * stride as u64;
        if let Some(km) = km {
            // Once per span, outside the chunk loop - see [`KeyMinor`].
            steps.push(key_minor_step(g, km, d_out, qkv, (kv_row_off, 0), stride, k_off, len));
        }
        let mut q0 = 0u32;
        while q0 < len {
            let qn = chunk.min(len - q0);
            // q view: rows [row0+q0 ..); kv view + ctx view: rows [row0 ..).
            let q_row_off = (row0 + q0) as u64 * stride as u64;
            let ctx_off = (row0 + q0) as u64 * d_out as u64;
            let sa = CrossScoreArgs { heads, head_dim, q_stride: stride, q_off, kv_stride: stride, k_off, qn, kn: len };
            steps.push(cross_scores_step(g, k, km, sa, qkv, (q_row_off, 0), qkv, (kv_row_off, 0), scores));
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

/// The workspace-wide OUTER gate for asking any flash-attention family for a
/// dispatch at all - the check every caller must make BEFORE it may even call
/// [`flash_bidir_variant`] or [`flash_cross_supported`], both of which pick a
/// *rung* by shape/shared-memory fit and do not themselves read
/// `workgroup_reductions`. That field is a CORRECTNESS gate, not a tuning
/// input: false means the Cranelift CPU JIT's split-at-one-barrier execution
/// model cannot run these multi-barrier kernels at all (see the field's own
/// doc on [`gpu_core::DeviceCaps`]).
///
/// Before this function existed, four sites each reimplemented a different
/// subset of the same `workgroup_reductions` check: `wan::block::attn_mode`
/// (the check alone), `lfm2::Model::flash_selectable` (plus "the ladder
/// actually beat the materialised baseline rung"), `sdxlunet::Rec`'s
/// self-attention (plus "not a training/gradient-recording pass"), and
/// `ltxv::block::flash_self_attn`/`flash_cross_attn` (plus `head_dim <=
/// 128`) - lesson #78's exact shape ("a selection seam only reaches callers
/// that opt in"), just for a gate instead of a kernel: a future change to the
/// base check (say, a driver quirk that also needs excluding) would have had
/// to be hunted down and reapplied in four places, and silently missed in
/// whichever one nobody remembered.
///
/// `extra` is that call site's OWN additional requirement, kept an explicit
/// argument rather than folded into a config enum here: it is genuinely
/// different per site (train-mode exclusion, a measured "beats the baseline"
/// check, a `head_dim` ceiling) and forcing it into one shared type would
/// only move the duplication into deciding which variant of the enum each
/// site needs. A site with no extra condition of its own (`wan`) passes
/// `true`. The cross-attention family additionally requires the stricter,
/// already-centralised [`flash_cross_supported`] (shared memory + workgroup
/// size, on top of the same `workgroup_reductions` bit this function reads),
/// so a caller ANDs this with that rather than this function trying to guess
/// which ladder it is gating.
///
/// Pure in its inputs: `caps` comes from `DeviceCaps`, so no backend name is
/// consulted.
pub fn flash_gate(caps: &gpu_core::DeviceCaps, extra: bool) -> bool {
    caps.workgroup_reductions && extra
}

/// The interchangeable bidirectional flash-attention kernels, as a model's own
/// pipeline indices. Every field past `bidir` is optional (`None` = the model
/// has not registered that kernel), which keeps adoption additive.
///
/// All four compute the same thing to cosine 1.000000000 of each other and
/// differ only in how the inner loops are scheduled. They form a ladder, and
/// [`flash_bidir_variant`] walks it from the top:
///
/// | field | kernel | workgroup | BR | shared | needs |
/// |---|---|---|---|---|---|
/// | `reg2` | `flash_attn_bidir_reg2` | 256 | 128 | 48 KiB | `max_workgroup_size ≥ 256`, `workgroup_mem_bytes ≥ 49152` |
/// | `reg` | `flash_attn_bidir_reg` | 256 | 64 | 16 KiB | `max_workgroup_size ≥ 256` |
/// | `split` | `flash_attn_bidir_split` | 256 | 64 | 16 KiB | `max_workgroup_size ≥ 256` |
/// | `bidir` | `flash_attn_bidir` | 64 | 64 | 16 KiB | - |
///
/// What separates them is the ratio of shared-memory loads to fused
/// multiply-adds in the inner loops, which is what a Pascal-class SM is
/// actually limited by here: it issues an FFMA warp-instruction every clock
/// but retires a shared load only every fourth, so a 1:1 mix cannot exceed a
/// quarter of the card's fp32 rate however well it is laid out. `bidir` spills
/// its per-thread `q[128]`/`o[128]` to local memory and runs at local-memory
/// bandwidth; `split` fixed the spill but left the mix at 1:1; `reg` vectorises
/// the tile reads to 1:4; `reg2` adds a second query row per thread and a
/// software-pipelined tile for ~1:7, which also halves the kernel's global K/V
/// traffic because a workgroup owns twice the query rows.
///
/// Measured at Wan 1.3B's self-attention (T=14040, 12 heads, head_dim 128)
/// against the device's own measured fp32 roof (`brain flops`): `bidir` is off
/// the scale, `split` reaches a fifth of the roof, `reg` a little more, and
/// `reg2` close to what the register-tiled GEMM `matmul_reg3` itself reaches
/// on the same card - which is the point, since these ARE GEMMs. `reg2` wins
/// at every T measured from 256 to 14040, so the ladder is walked on device
/// caps alone and never on shape.
///
/// Both new kernels need `@workgroup_size(256)` and `reg2` additionally needs
/// 48 KiB of workgroup memory - four times the 16 KiB a Vulkan implementation
/// is only *required* to offer - so both are gated on queried `DeviceCaps`,
/// never assumed.
#[derive(Clone, Copy)]
pub struct FlashIds {
    pub bidir: usize,
    pub split: Option<usize>,
    pub reg: Option<usize>,
    pub reg2: Option<usize>,
}

/// Workgroup memory `flash_attn_bidir_reg2` needs: `ksh` + `vsh` (BC·HD·4 B
/// each) + `part0` + `part1` (BC·RG·4·4 B each) at its BC=16, RG=64, HD=128.
const FLASH_REG2_SHARED: u32 = 49152;

/// The flash variant to dispatch on this device: `(kernel index, workgroup
/// size, query rows per workgroup)`. The third element is NOT a constant across
/// the family - `flash_attn_bidir_reg2` owns 128 query rows where the others
/// own 64 - so a caller must size its grid from this and never from a BR of its
/// own. Pure in its inputs: `caps` comes from `DeviceCaps`, so no backend name
/// is consulted.
pub fn flash_bidir_variant(ids: FlashIds, caps: &gpu_core::DeviceCaps) -> (usize, u32, u32) {
    let wide = caps.max_workgroup_size >= 256;
    match (ids.reg2, ids.reg, ids.split) {
        (Some(i), _, _) if wide && caps.workgroup_mem_bytes >= FLASH_REG2_SHARED => (i, 256, 128),
        (_, Some(i), _) if wide => (i, 256, 64),
        (_, _, Some(i)) if wide => (i, 256, 64),
        _ => (ids.bidir, 64, 64),
    }
}

/// One fused bidirectional flash-attention dispatch over `bsz` samples of `t`
/// rows each in a packed qkv slab - the variant chosen by
/// [`flash_bidir_variant`]. Every kernel in the family takes the SAME Params
/// and produces the SAME output layout, so only the pipeline index, the
/// per-workgroup thread count and the query rows per workgroup differ - and
/// all three come from the selector, which is why BR is not a constant here.
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
    let (kind, ws, br) = flash_bidir_variant(ids, &g.caps());
    let nwg = bsz * heads * t.div_ceil(br);
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
    let (kind, ws, br) = flash_bidir_variant(ids, &g.caps());
    for &(row0, len) in spans {
        let nwg = heads * len.div_ceil(br);
        steps.push(g.step_sliced(
            kind,
            &[qkv, ctx],
            &[(row0 as u64 * stride as u64, 0), (row0 as u64 * d_out as u64, 0)],
            &[1, heads, len, head_dim, stride, q_off, k_off, v_off, d_out],
            nwg * ws,
        ));
    }
}

/// Workgroup memory `flash_attn_cross_reg2` needs - the same tiles, hence the
/// same figure, as [`FLASH_REG2_SHARED`]; named separately so a future rung of
/// the cross ladder with different tiles cannot silently inherit this one.
const FLASH_CROSS_REG2_SHARED: u32 = 49152;

/// Query rows one `flash_attn_cross_reg2` workgroup owns (its `BR`), and its
/// thread count. A caller must size its grid from these and never from a `BR`
/// of its own - the same contract [`flash_bidir_variant`] states by returning
/// them.
const FLASH_CROSS_REG2_BR: u32 = 128;
const FLASH_CROSS_REG2_WS: u32 = 256;

/// Whether this device can run [`flash_cross_step`] at all.
///
/// Unlike the bidirectional family there is no ladder yet: `flash_attn_cross_
/// reg2` is the only cross rung in the tree, so the answer is a bool, not a
/// choice. A device that says no keeps whatever materialized
/// scores/softmax/apply trio the caller already had - which is the branch the
/// Cranelift CPU JIT takes (`workgroup_reductions == false`; the kernel needs
/// two top-level barriers where that JIT supports one) and therefore stays the
/// reference definition of the math.
///
/// Pure in its inputs: `caps` comes from `DeviceCaps`, so no backend name is
/// consulted.
pub fn flash_cross_supported(caps: &gpu_core::DeviceCaps) -> bool {
    !no_flash_cross() && caps.workgroup_reductions && caps.max_workgroup_size >= FLASH_CROSS_REG2_WS && caps.workgroup_mem_bytes >= FLASH_CROSS_REG2_SHARED
}

/// `BRAIN_NO_FLASH_CROSS=1` pins cross-attention to the caller's materialized
/// scores/softmax/apply trio - the A/B switch this kernel's speedup was
/// measured with, the way a correctness gate reaches the reference arm on a
/// device that would otherwise always take the fused one, and the fallback if
/// a driver ever mishandles a 48 KiB workgroup allocation. Same shape as
/// `backend_api::select`'s `BRAIN_NO_COOP_LN`, and read once for the same
/// reason: the policy must stay a pure function of its inputs for a given
/// process.
///
/// Without a switch here an A/B on a capable device compares the fused path
/// against itself and reports a meaningless parity - which looks like evidence
/// and is not.
fn no_flash_cross() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_NO_FLASH_CROSS").map(|v| v != "0").unwrap_or(false))
}

/// The seven layout numbers [`flash_cross_step`] needs beyond the buffers -
/// grouped because q, k and v are three independent operands here, each with
/// its own row width and region offset, and seven more positional `u32`s at
/// the call site is how a stride ends up in an offset slot.
#[derive(Clone, Copy)]
pub struct FlashCrossLayout {
    pub q_stride: u32,
    pub q_off: u32,
    pub k_stride: u32,
    pub k_off: u32,
    pub v_stride: u32,
    pub v_off: u32,
    /// Row width of the output context (`heads*head_dim` for an unsliced
    /// caller).
    pub d_out: u32,
}

/// One fused cross-attention dispatch: `nq` query rows attend `nk` key/value
/// rows out of three SEPARATE buffers, through an online softmax that never
/// materializes the `[heads, nq, nk]` score slab or its probabilities twin.
///
/// This is the cross-attention counterpart of [`flash_bidir_fwd`], and it
/// exists because neither of that family's members can express the shape:
/// they take one fused `[t, 3*d_model]` slab at a single `tcols`, and
/// `flash_attn_causal_gqa` additionally masks `j > i`. A caller with a decoder
/// stream and an encoder memory has two row counts and three buffers.
///
/// Callers MUST gate on [`flash_cross_supported`] and keep their materialized
/// trio as the fallback; `head_dim` ≤ 128. `bsz > 1` is sample-major in all
/// four buffers (q/out at `nq` rows per sample, k/v at `nk`).
///
/// The output is the same `[bsz*nq, d_out]` row-major context the trio's
/// `attn_apply_cross` writes, with the same `1/√head_dim` score scale, so
/// everything downstream is untouched by the choice. It is NOT bit-identical:
/// an online softmax reassociates both the score sum and the value
/// accumulation, so a caller's gate needs a tolerance (and, since the change
/// is a reassociation rather than a rescale, it needs relative L2 as well as
/// cosine - cosine is scale-invariant).
#[allow(clippy::too_many_arguments)]
pub fn flash_cross_step(
    g: &Gpu,
    kind: usize,
    bsz: u32,
    heads: u32,
    head_dim: u32,
    nq: u32,
    nk: u32,
    lay: FlashCrossLayout,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    ctx: &DeviceBuffer,
) -> Step {
    assert!(head_dim <= 128, "flash_attn_cross: head_dim {head_dim} > 128");
    let nwg = bsz * heads * nq.div_ceil(FLASH_CROSS_REG2_BR);
    g.step(
        kind,
        &[q, k, v, ctx],
        &[bsz, heads, nq, nk, head_dim, lay.q_stride, lay.q_off, lay.k_stride, lay.k_off, lay.v_stride, lay.v_off, lay.d_out],
        nwg * FLASH_CROSS_REG2_WS,
    )
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
/// path) left the card at a low single-digit percent of peak; the same insight
/// already bought the CPU fast paths a large multiple (they route these
/// kernels to the native GEMM).
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
    km: Option<&KeyMinor>,
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
        let kv_row_off = row0 as u64 * stride as u64;
        if let Some(km) = km {
            // Once per span, outside the chunk loop - see [`KeyMinor`].
            steps.push(key_minor_step(g, km, d_out, qkv, (kv_row_off, 0), stride, k_off, len));
        }
        let mut q0 = 0u32;
        while q0 < len {
            let qn = chunk.min(len - q0);
            let q_row_off = (row0 + q0) as u64 * stride as u64;
            let dc_off = (row0 + q0) as u64 * d_out as u64;
            let p_qk = [1, heads, qn, len, head_dim, stride, stride, q_off, k_off];
            let p_v = [1, heads, qn, len, head_dim, stride, v_off, d_out];
            // Recompute this chunk's scores + probs from the cached qkv.
            let sa = CrossScoreArgs { heads, head_dim, q_stride: stride, q_off, kv_stride: stride, k_off, qn, kn: len };
            steps.push(cross_scores_step(g, fwd, km, sa, qkv, (q_row_off, 0), qkv, (kv_row_off, 0), scores));
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
/// measured a win at *every* row width, widest where rows are narrow, because
/// the per-element kernel's one-thread-per-row layout is uncoalesced) where the
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

/// Gate for a model that has just registered [`KernelIds::rmsnorm_rows`]:
/// assert the seam computes the reference normalization at every `(rows, dim)`
/// its own tapes dispatch one at.
///
/// Adopting `rmsnorm_rows` is a THROUGHPUT change to a kernel that is not
/// bit-identical to the one it replaces (64 partial sums fold in a different
/// order, agreeing to ~3e-6), so every adopting model owes a numerical gate.
/// The reference is a HOST one - [`crate::hostmath::rmsnorm_rows`], written to
/// match `rmsnorm.wgsl`'s reduction and epsilon placement exactly. Comparing
/// the two device kernels to each other would pass if both were wrong the same
/// way.
///
/// It lives here, next to the seam, because five models adopted it in one pass
/// and five byte-identical copies of this comparison is how a tolerance drifts.
/// Same "a production module can carry the one test helper for the thing it
/// implements" convention as `gpu_core::testgpu`.
///
/// `shapes` are `(rows, dim, what)`; `what` names the dispatch site so a
/// failure says which tape broke, not just which number.
pub fn assert_rmsnorm_variant_agrees(g: &Gpu, ids: &KernelIds, shapes: &[(u32, u32, &str)]) {
    // The kernels differ only in reduction ORDER over the same `dim` squares,
    // so the error is O(sqrt(dim) * eps) on a sum whose scale is `dim`;
    // `rmsnorm_rows`'s own header records 3.3e-6 max_abs over a wide sweep.
    // TIGHT on purpose: a real defect (a wrong eps, a missed tail element, a
    // mis-strided row) moves the answer by orders of magnitude more.
    const TOL: f32 = 2e-5;
    assert!(!shapes.is_empty(), "no shapes given, so this gate would pass vacuously");

    for &(rows, dim, what) in shapes {
        let (rows_u, dim_u) = (rows as usize, dim as usize);
        // Deterministic, no RNG (engine convention), and scaled well away from
        // 1.0 so a dropped `eps` or a wrong element count cannot hide inside a
        // coincidentally-unit normalization.
        let x: Vec<f32> = (0..rows_u * dim_u).map(|i| 3.0 * (i as f32 * 0.7 + 0.1).sin()).collect();
        let w: Vec<f32> = (0..dim_u).map(|i| 0.5 * (i as f32 * 0.31 + 0.2).cos()).collect();
        let want = crate::hostmath::rmsnorm_rows(&x, &w, rows_u, dim_u, RMSNORM_EPS);

        let xb = g.storage_init("rms_agree_x", &x);
        let wb = g.storage_init("rms_agree_w", &w);
        let ob = g.storage((rows_u * dim_u) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, ids, &xb, &wb, &ob, dim, rows)]);
        let got = g.read(&ob, rows_u * dim_u);

        assert!(got.iter().all(|v| v.is_finite()), "{what} ({rows}x{dim}): produced a non-finite value");
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let e = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
        assert!(e <= TOL, "{what} ({rows}x{dim}): relative error {e:e} exceeds {TOL:e}");
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
/// to one row and are coalesced by construction - measured faster across
/// d_model 768-3072 x 512-2048 rows (`brain-gpu-core`'s `bench_layernorm`,
/// which is what to re-run on another card), winning at every shape including
/// the 1-row decode case.
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

/// Which softmax kernel to dispatch (`softmax_rows` vs a caller-supplied
/// reference - `attn_softmax`/`decode_softmax`/`attn_softmax_{cross,bidir}`,
/// whichever shape the caller needs) and the resulting thread count. Same
/// shape as [`rms_variant`]/[`ln_variant`]: `coop` is `None` when the caller
/// never registered `softmax_rows` at all (walks straight to `reference`,
/// same as a model with no GEMV kernel falling through `Op::MatMul`'s head),
/// and the policy itself lives in `backend_api::select` (`Op::Softmax`),
/// keyed on `DeviceCaps`, never a backend name or a hand-rolled
/// `caps.workgroup_reductions` check at the call site - two of which
/// (`wan::block::Sel::new`, `ltxv::block::attn_softmax`) had independently
/// reimplemented this exact rule before this seam existed.
///
/// `rows` is the caller's own row count (`n_heads * query_positions`, or
/// `batch * n_heads` for a decode step - whatever the reference kernel's
/// own `Params` already uses), `cols` is the key axis being softmaxed over.
/// `softmax_rows.wgsl` is `@workgroup_size(64)`, at or below the WebGPU
/// floor, so no `max_workgroup_size` gate is needed on top of the seam.
pub fn softmax_variant(g: &Gpu, reference: usize, coop: Option<usize>, rows: u32, cols: u32) -> (usize, u32) {
    use gpu_core::select::{Dtype, KernelSelector, KernelVariant, Op, OpShape};
    let shape = OpShape { m: rows, n: cols, k: 0, dtype: Dtype::F32 };
    match coop {
        Some(i)
            if gpu_core::select::DefaultSelector.select(Op::Softmax, shape, &g.caps())
                == KernelVariant::WorkgroupPerOutput =>
        {
            (i, rows * 64)
        }
        _ => (reference, rows),
    }
}

/// Scores one `paged_decode_scores_wg` workgroup computes - `64 / LPS` in
/// that kernel's own header. A kernel-contract constant, not a model choice:
/// [`paged_scores_variant`] is the only caller and must match the WGSL
/// exactly, or the dispatch covers the wrong number of scores.
pub const PAGED_SCORES_PER_WORKGROUP: u32 = 16;

/// Which paged-attention decode SCORES kernel to dispatch
/// (`paged_decode_scores_wg` vs a caller-supplied reference -
/// `paged_decode_scores_batched`) and the resulting thread count. `coop` is
/// `None` when the caller never registered `paged_decode_scores_wg` at all
/// (walks straight to `reference`), same convention as
/// [`rms_variant`]/[`ln_variant`]/[`softmax_variant`] - but the thread-count
/// formula differs from those three: the cooperative kernel does not own
/// one row per workgroup, it owns [`PAGED_SCORES_PER_WORKGROUP`] scores per
/// workgroup, so this takes the TOTAL score count (`batch * n_heads`) and
/// the key axis (`cap`) rather than a plain row count, and computes the
/// dispatch accordingly. The policy itself lives in `backend_api::select`
/// (`Op::PagedAttention`), keyed on `DeviceCaps`, never a hand-rolled
/// `caps.workgroup_reductions` check at the call site -
/// `qwen3::serve::Engine::run_batched`'s own scores dispatch had
/// independently reimplemented exactly this rule before this seam existed.
///
/// Only the F32 tier is expressed here (this Op's `Dtype::I8` arm has no
/// cooperative sibling at all, and no capability gate either - see
/// `Op::PagedAttention`'s doc: unlike `Op::MatMul`'s DP4A-bound packed
/// GEMMs, the packed-int8 KV kernels are plain scalar WGSL, portable to
/// every backend); a caller dispatching the packed-int8 KV trio always
/// uses it unconditionally, with no `*_variant` seam needed on that side.
pub fn paged_scores_variant(g: &Gpu, reference: usize, coop: Option<usize>, batch_heads: u32, cap: u32) -> (usize, u32) {
    use gpu_core::select::{Dtype, KernelSelector, KernelVariant, Op, OpShape};
    let shape = OpShape { m: batch_heads, n: cap, k: 0, dtype: Dtype::F32 };
    let total = batch_heads.saturating_mul(cap);
    match coop {
        Some(i)
            if gpu_core::select::DefaultSelector.select(Op::PagedAttention, shape, &g.caps())
                == KernelVariant::WorkgroupPerOutput =>
        {
            (i, total.div_ceil(PAGED_SCORES_PER_WORKGROUP) * 64)
        }
        _ => (reference, total),
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
/// naive kernel). Measured: that head was nearly the entire T=512 prefill and
/// ran at a fraction of one percent of the compute roof, and letting it
/// collapse cut the whole pass by an order of magnitude.
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

/// A device with every capability [`pick_gemm`]/[`gemm_variant`] can ever act
/// on already reported present - `workgroup_reductions: true`, from
/// [`DeviceCaps::portable_baseline`]'s WebGPU-conformant floor.
///
/// Neither function takes a device/caps parameter (their signatures predate
/// `backend_api::select` and are preserved as-is this phase), so they were
/// always device-BLIND: every caller got the fast tiers regardless of what
/// actually ran the dispatch. That was never a correctness bug because it
/// never actually mattered on the one backend it theoretically could -
/// `backend-cpu`'s own `dispatch` (`crates/backend-cpu/src/lib.rs`)
/// special-cases `matmul`/`matmul_reg`/`matmul_reg2`/`matmul_reg3` BY KERNEL
/// IDENTITY and routes all of them to the same native AVX2 GEMM regardless of
/// the WGSL kernel's nominal `workgroupBarrier()` - so filtering
/// `RegisterTiled` out on a real CPU JIT caps struct (`workgroup_reductions:
/// false`) would silently CHANGE behaviour these two functions never had, not
/// preserve it. Querying `select::candidates` against this always-capable
/// caps reproduces the two functions' historical device-blind behaviour
/// exactly.
fn fast_tier_caps() -> DeviceCaps {
    DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu)
}

/// Pick the GEMM kernel + dispatch thread count for `[m,k]·[n,k]ᵀ`: the
/// register-tiled kernel (128×128 tile, 256 threads) when there is enough work
/// to fill tiles, else the naive one-thread-per-output kernel. Same math either
/// way - every variant is bit-identical to the naive reference (measured,
/// `max|Δ| = 0`), so this only changes speed. `force_naive` is a model's env
/// escape.
///
/// A thin adapter over `backend_api::select::candidates` (B2) - the crossover
/// constants (`GEMM_TILE_MIN_ROWS`/`GEMM_TILE_MIN_COLS`, with the measured P40
/// table that justifies them) now live there, as the one source every GEMM
/// picker in the workspace reads. This function's own callers never register a
/// GEMV kernel (there is no such parameter here), so the adapter skips
/// `WorkgroupPerOutput` in the candidate list and takes the first variant it
/// CAN express - `RegisterTiled` maps to `reg2`, anything else (`Reference`,
/// or every candidate filtered out) maps to `naive`.
pub fn pick_gemm(m: usize, n: usize, naive: usize, reg2: usize, force_naive: bool) -> (usize, u32) {
    if force_naive {
        return (naive, (m * n) as u32);
    }
    let shape = select::OpShape { m: m as u32, n: n as u32, k: 0, dtype: select::Dtype::F32 };
    let chosen = select::candidates(select::Op::MatMul, shape, &fast_tier_caps())
        .into_iter()
        .find(|v| *v != KernelVariant::WorkgroupPerOutput);
    match chosen {
        Some(KernelVariant::RegisterTiled) => (reg2, (m.div_ceil(128) * n.div_ceil(128) * 256) as u32),
        _ => (naive, (m * n) as u32),
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
///
/// A thin adapter over `backend_api::select::candidates` (B2), delegating the
/// decode-regime cutoff (`m <= DECODE_REGIME_MAX_ROWS`) to the SAME rule
/// `pick_gemm` and `qwen3::serve::Engine::mm8`'s int8 path use, instead of its
/// own `m <= 32` copy. `GemmVariants::Fast` has no naive/reference kernel slot
/// at all (a model on this tier committed to the fast kernels), so unlike
/// `pick_gemm` there is nothing to fall back to but `tiled` - the adapter only
/// ever takes the GEMV branch when `select::candidates`'s head is actually
/// `WorkgroupPerOutput` AND the model registered a GEMV kernel; every other
/// case (no GEMV registered, or `m` past the decode regime, regardless of `n`)
/// uses `tiled`, exactly as before.
pub fn gemm_variant(v: GemmVariants, m: u32, n: u32) -> (usize, u32) {
    match v {
        GemmVariants::Reference(k) => (k, m * n),
        GemmVariants::Fast { gemv, tiled } => {
            let shape = select::OpShape { m, n, k: 0, dtype: select::Dtype::F32 };
            let head = select::candidates(select::Op::MatMul, shape, &fast_tier_caps())
                .into_iter()
                .next();
            match (gemv, head) {
                (Some(g), Some(KernelVariant::WorkgroupPerOutput)) => (g, n * 64),
                _ => (tiled, m.div_ceil(128) * n.div_ceil(128) * 256),
            }
        }
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
/// model (`qwen3omnimoe::thinker`) builds on it.
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
            rmsnorm_rows: UNREGISTERED,
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

    /// [`flash_gqa_causal_fwd`] (O(T*head_dim) memory, tiled online softmax)
    /// must reproduce [`gqa_fwd`]'s materialized-`[H,T,T]` causal output --
    /// the real correctness bar for the fix that closed a real
    /// `ERROR_OUT_OF_DEVICE_MEMORY` (see `flash_attn_causal_gqa.wgsl`'s doc):
    /// a memory-cheaper kernel that answers differently is not a fix, it is a
    /// different bug. `t=100` deliberately exceeds both the kernel's query
    /// tile (BR=64) and key/value tile (BC=16) sizes, so this exercises the
    /// multi-tile loop in both dimensions, not just a single-tile shortcut;
    /// `n_kv_heads < n_heads` exercises the GQA head-group mapping the same
    /// way `gqa_scores.wgsl`'s own `hkv = h / group` does.
    ///
    /// Runs on the pooled *wgpu* test device (`gpu_core::testgpu::dev`), not
    /// `Gpu::new_cpu`: the CPU backend only JIT-compiles a fixed set of
    /// natively-recognized workgroup kernels, and this one -- a fresh shared-
    /// memory/`workgroupBarrier` kernel with no hand-written CPU fast path --
    /// is not in that set (`wgsl-cpu: ... unsupported work-group structure`).
    /// wgpu compiles arbitrary WGSL for real, on a software rasterizer when no
    /// GPU is present, so it is the correct portable device for a kernel this
    /// new.
    #[test]
    fn flash_causal_gqa_matches_materialized_gqa_fwd() {
        let (t, n_heads, n_kv_heads, head_dim) = (100u32, 4u32, 2u32, 8u32);
        let (hq, hkv) = (n_heads * head_dim, n_kv_heads * head_dim);

        let gpu = gpu_core::testgpu::dev(&[
            ("gqa_scores", kernels::GQA_SCORES),
            ("attn_softmax", kernels::ATTN_SOFTMAX),
            ("gqa_apply", kernels::GQA_APPLY),
            ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA),
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
            rmsnorm_rows: UNREGISTERED,
        };

        // Deterministic pseudo-random q/k/v -- same fixed-formula convention
        // `decode_step_matches_causal_batched_attention_at_every_position` uses.
        let mk = |n: u32, seed: f32| (0..n).map(|i| (i as f32 * 0.7 + seed).sin()).collect::<Vec<f32>>();
        let q = gpu.storage_init("q", &mk(t * hq, 0.1));
        let k = gpu.storage_init("k", &mk(t * hkv, 0.2));
        let v = gpu.storage_init("v", &mk(t * hkv, 0.3));
        let ga = Gqa { b: 1, t, n_heads, n_kv_heads, head_dim };

        let scores = gpu.storage((n_heads * t * t) as u64);
        let probs = gpu.storage((n_heads * t * t) as u64);
        let ctx_ref = gpu.storage((t * hq) as u64);
        gpu.submit(&[], &gqa_fwd(&gpu, &ids, &ga, &q, &k, &v, &scores, &probs, &ctx_ref));
        let want = gpu.read(&ctx_ref, (t * hq) as usize);

        let ctx_flash = gpu.storage((t * hq) as u64);
        gpu.submit(&[], &[flash_gqa_causal_fwd(&gpu, 3, &ga, &q, &k, &v, &ctx_flash)]);
        let got = gpu.read(&ctx_flash, (t * hq) as usize);

        let mut worst = 0.0f32;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let diff = (g - w).abs();
            worst = worst.max(diff);
            assert!(diff < 1e-3, "elem {i}: flash={g}, reference={w}, diff={diff}");
        }
        assert!(worst > 0.0, "sanity: q/k/v are not all-zero, so a real match should not be a trivial 0==0");
        println!("flash_causal_gqa vs gqa_fwd: worst abs diff = {worst:e}");
    }
}

#[cfg(test)]
mod tests {
    use super::{flash_gate, gemm_variant, pick_gemm, tiles_with_budget, GemmVariants, TILE_BUDGET_FRACTION, TILE_BUDGET_WORDS};
    use gpu_core::{DeviceCaps, DeviceClass};

    /// The shared outer gate is exactly `workgroup_reductions AND extra` - no
    /// more, no less - at all four truth-table points. This is what makes it
    /// safe for `wan`/`lfm2`/`sdxlunet`/`ltxv` to each hand it their own
    /// `extra` condition and trust the device half is applied identically:
    /// `caps.workgroup_reductions == false` must refuse a dispatch regardless
    /// of `extra` (the Cranelift CPU JIT correctness gate, not a preference),
    /// and a capable device must still refuse when the caller's own extra
    /// condition (train-mode, "beats the baseline", `head_dim <= 128`) fails.
    #[test]
    fn flash_gate_is_workgroup_reductions_and_the_callers_extra_condition() {
        let coop = DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu);
        let non_coop = DeviceCaps { workgroup_reductions: false, ..DeviceCaps::portable_baseline(DeviceClass::DiscreteGpu) };

        assert!(flash_gate(&coop, true), "capable device, no extra condition: must dispatch");
        assert!(!flash_gate(&coop, false), "capable device but the caller's extra condition failed: must not dispatch");
        assert!(!flash_gate(&non_coop, true), "CPU-JIT-shaped device: must never dispatch regardless of extra");
        assert!(!flash_gate(&non_coop, false), "neither half holds");
    }

    /// Pins `pick_gemm`'s measured crossover (now delegated to
    /// `backend_api::select::candidates` - see that module's
    /// `GEMM_TILE_MIN_ROWS`/`GEMM_TILE_MIN_COLS`, B2) at the table's own
    /// (m, n) points, `force_naive`, and the narrow-`n` case the row count
    /// alone does not cover. This function had no dedicated test before B2 -
    /// its crossover was only pinned indirectly by whatever shapes each
    /// caller happened to exercise.
    #[test]
    fn pick_gemm_routes_by_the_measured_crossover() {
        let (naive, reg2) = (2usize, 9usize);
        // Below GEMM_TILE_MIN_ROWS (8): naive, at the table's own m values.
        for m in [1usize, 2, 4, 7] {
            assert_eq!(pick_gemm(m, 2560, naive, reg2, false), (naive, (m * 2560) as u32), "m={m}");
        }
        // At and above it: the tile.
        for m in [8usize, 12, 32, 33, 77, 512] {
            let want_threads = (m.div_ceil(128) * 2560usize.div_ceil(128) * 256) as u32;
            assert_eq!(pick_gemm(m, 2560, naive, reg2, false), (reg2, want_threads), "m={m}");
        }
        // A narrow n keeps the naive kernel even at a large m.
        assert_eq!(pick_gemm(512, 64, naive, reg2, false), (naive, (512 * 64) as u32));
        // `force_naive` overrides the shape entirely.
        assert_eq!(pick_gemm(512, 2560, naive, reg2, true), (naive, (512 * 2560) as u32));
    }

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
    /// This exists because the fixed ~96 MiB budget is far smaller than what a
    /// P40 reports, which split Qwen3-0.6B's tied 151936x1024 head into 7 tiles;
    /// the caller only routes to the register-tiled GEMM when the vocab
    /// collapses to ONE tile, so the head ran the naive kernel at a fraction of
    /// one percent of the compute roof and was nearly the entire T=512 prefill
    /// (which collapsing it cut by an order of magnitude).
    #[test]
    fn a_real_lm_head_collapses_to_one_tile_on_a_device_that_reports_its_limit() {
        let (vocab, d) = (151936u64, 1024u64);

        // The portable floor still tiles it 7 ways - the behaviour that cost an
        // order of magnitude on the prefill.
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
