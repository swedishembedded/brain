// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gated DeltaNet (GDN) chunked-parallel linear-attention recurrence —
//! Qwen3.5-35B-A3B's "linear attention" layer. Transcribed step-for-step from
//! HuggingFace's `torch_chunk_gated_delta_rule`
//! (`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`) into a sequence
//! of device [`Step`]s. Lives in `brain-model` next to [`crate::moe`] and
//! [`crate::block`], not in a model-specific crate, because a future SSM /
//! linear-attention architecture should reuse this recurrence unchanged.
//!
//! **Forward only.** See "Backward" at the bottom of this doc.
//!
//! ## The reference algorithm, in this module's terms
//!
//! Given per-`(batch,head)` `query,key: [T,Dk]`, `value: [T,Dv]`, a raw
//! per-token log-decay `g: [T]` and a sigmoid gate `beta: [T]` (both already
//! computed by the caller — this module does not compute them), chunked into
//! `n_chunks = T/C`:
//!
//! 1. `query *= 1/sqrt(Dk)` (folded into every kernel call that reads
//!    `query`, via `bmm`'s `alpha` or `gdn_row_scale_off`'s `alpha` — `query`
//!    itself is never mutated).
//! 2. `v_beta = value*beta`, `k_beta = key*beta` (row-broadcast, whole
//!    tensor — `scale_row.wgsl`, reused unmodified).
//! 3. (chunking is just how the flat buffer is addressed — see "Layout"
//!    below; no kernel work.)
//! 4. `g_cs = cumsum(g)` per chunk (`gdn_chunk_cumsum_step.wgsl`, one host
//!    dispatch per row index).
//! 5. `decay_mask[i,j] = exp(g_cs[i]-g_cs[j])` for `j<=i`, else 0
//!    (`gdn_decay_mask.wgsl`).
//! 6. `attn0 = -(k_beta @ key^T) * decay_mask`, masked to `j<i`
//!    (`bmm.wgsl` with `alpha=-1,trans_b=1` then `gdn_mask_strict_lower.wgsl`).
//! 7. UT-transform: `T_mat = (I - attn0)^-1` computed by forward substitution
//!    (`gdn_ut_step.wgsl`, one host dispatch per row) then `+= I`
//!    (`gdn_add_identity.wgsl`).
//! 8. `u = T_mat @ v_beta` (`bmm.wgsl`).
//! 9. `w = T_mat @ (k_beta * exp(g_cs))` (`exp.wgsl` + `scale_row.wgsl` +
//!    `bmm.wgsl`).
//! 10. Sequential across-chunk loop (state carries chunk-to-chunk): per
//!     chunk, `v_prime = w_c @ state`, `v_new = u_c - v_prime` (the
//!     reference REASSIGNS its `value` variable to `u` in step 8 —
//!     `torch_chunk_gated_delta_rule`'s `value = attn @ v_beta` shadows the
//!     function's own `value` PARAMETER, so every later `value`/`v_c` in the
//!     reference, including this one, means `u`'s chunk slice, never the
//!     original raw `value` tensor — easy to miss, worth re-checking against
//!     the actual reference source rather than trusting this paraphrase),
//!     `attn_inter = (q_c*exp(g_cs_c)) @ state`,
//!     `core_out_c = attn_inter + (q_c.k_c*decay_mask_c) @ v_new`,
//!     `state = state*exp(g_cs_c[-1]) + (k_c*exp(g_cs_c[-1]-g_cs_c))^T @ v_new`.
//! 11. `core_out` concatenated over chunks is the `[T,Dv]` output.
//!
//! `intra_scores = (q_c.k_c)*decay_mask_c` does not depend on `state`, so it
//! is precomputed for EVERY chunk in one whole-tensor pass before the
//! sequential loop starts (reusing `decay_mask` from step 5), not recomputed
//! per chunk-iteration.
//!
//! ## Layout — CHUNK-MAJOR, a deliberate departure from `[B,H,T,D]`
//!
//! The reference (and this doc, above) describes shapes in PyTorch's
//! `[B,H,T,D]` axis order. This module instead requires every per-token
//! buffer (`query`,`key`,`value`,`raw_g`,`beta`,`out`) to be laid out as if
//! shaped `[n_chunks, B, H, C, D]` row-major — i.e. flat index
//! `bhc = chunk*(B*H) + b*H + h` is the OUTERMOST enumeration, not
//! `(b*H+h)*n_chunks + chunk` as a literal `[B,H,T,D]` reshape would give.
//!
//! This is why: [`gdn_chunk_fwd`]'s step-10 loop is genuinely SEQUENTIAL
//! across chunks (state feeds forward), but fully PARALLEL across
//! `(batch,head)` within one chunk. Dispatching one chunk's worth of work
//! therefore needs "every `(b,h)`, one fixed chunk" as a single batch range
//! for [`bmm`]/[`bmm_acc`] and the offset-taking elementwise kernels
//! (`gdn_row_scale_off.wgsl`, `sub.wgsl`, `gdn_decay_scale.wgsl`,
//! `gdn_state_decay.wgsl`). With chunk OUTERMOST, that range is the
//! CONTIGUOUS slice `[chunk*(B*H), (chunk+1)*(B*H))` of the buffer's own
//! flat batch axis — addressable with a plain `Params` element offset
//! (`off_of` below), never a bound byte-offset slice (which
//! `docs/kernel-checklist.md` requires to be 256-byte aligned — this
//! module's own gradcheck-style test at deliberately tiny, non-aligned dims
//! would fail under that scheme). With `(b,h)` outermost instead (a literal
//! `[B,H,T,D]` reshape), the same per-chunk slice is STRIDED
//! (`stride = n_chunks * C * D`), which `bmm`/`bmm_acc` do not support (by
//! design — see `bmm.wgsl`'s own header for why offset-only was chosen).
//!
//! Consequently: **the caller (a future `qwen35` wiring, out of scope here)
//! must produce `query`/`key`/`value`/`raw_g`/`beta` in this chunk-major
//! order** (or insert a permute step) before calling [`gdn_chunk_fwd`], and
//! must consume `out` the same way. `initial_state`/`final_state` have no
//! chunk axis (`[B,H,Dk,Dv]`, ordinary `(b,h)`-major) and need no such care.
//!
//! Every OTHER buffer ([`GdnScratch`]'s fields) is internal and chosen to
//! match this same chunk-major convention throughout, so every per-chunk
//! kernel call in [`gdn_chunk_fwd`] uses a plain element offset.
//!
//! ## `exp(g_cs)`: materialise once, or recompute inline? Both, by consumer
//!
//! `exp_g_cs` (all of `g_cs`, exponentiated once via `exp.wgsl`) is reused by
//! the two consumers that want `exp(g_cs)` ALONE: `k_cumdecay`'s row-scale
//! (step 9) and `attn_inter`'s row-scale (step 10). The state update's decay
//! terms (`gdn_decay_scale.wgsl`, `gdn_state_decay.wgsl`) instead recompute
//! `exp` INLINE on raw `g_cs`, because they need `exp(a-b)` for two cumulative
//! sums `a,b` that can each be very negative over a whole chunk — computing
//! that as `exp(a)/exp(b)` from the materialised buffer risks dividing by an
//! UNDERFLOWED near-zero denominator that the direct `exp(a-b)` form never
//! hits. Recomputing costs one redundant transcendental per element in
//! exchange for not needing a second scratch buffer AND not risking that
//! numerical trap — a deliberate choice, not an oversight.
//!
//! ## What this module does NOT do
//!
//! Per the porting task this exists for: no GQA head-repeat (`H` here is
//! already `num_v_heads`), no L2-norm, no gated RMSNorm, no depthwise conv,
//! no decay-gate computation (`raw_g`/`beta` arrive ready-made), no T-padding
//! (caller must pass `t` already a multiple of `chunk` — [`GdnShape::n_chunks`]
//! asserts it), no incremental/decode (single-token) path — chunked/prefill
//! (steps 1-11) only.
//!
//! ## Backward
//!
//! **Not implemented.** Deriving and gradient-checking a hand-written
//! reverse-mode pass through all eleven steps (including the UT-transform's
//! reverse sequential sweep) is real, separate work this pass did not have
//! budget to get right with confidence — and `docs/porting-playbook.md`'s own
//! rule is to ship a correct forward + an honest gap over a rushed backward
//! that is subtly wrong. `crates/model/tests/gdn_chunk_bwd.rs` is a stub
//! naming exactly this as a `#[ignore]`d TODO; do not remove that file's
//! intent without implementing the real gradcheck it describes.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Kernel indices [`gdn_chunk_fwd`] dispatches, resolved by the calling model
/// against its own registered pipeline list (same pattern as
/// [`crate::moe::MoeIds`]).
#[derive(Clone, Copy)]
pub struct GdnIds {
    /// `bmm.wgsl` — batched matmul, overwrite.
    pub bmm: usize,
    /// `bmm_acc.wgsl` — batched matmul, accumulate.
    pub bmm_acc: usize,
    /// `gdn_chunk_cumsum_step.wgsl`.
    pub cumsum_step: usize,
    /// `gdn_decay_mask.wgsl`.
    pub decay_mask: usize,
    /// `gdn_mask_strict_lower.wgsl` — finishes step 6's `attn0` from
    /// `bmm.wgsl`'s raw `-(k_beta @ key^T)`.
    pub mask_strict_lower: usize,
    /// `gdn_ut_step.wgsl`.
    pub ut_step: usize,
    /// `gdn_add_identity.wgsl`.
    pub add_identity: usize,
    /// `scale_row.wgsl` (existing, unmodified) — every WHOLE-TENSOR row-scale
    /// (`v_beta`, `k_beta`, `k_cumdecay`'s `k_beta*exp(g_cs)`).
    pub row_scale: usize,
    /// `gdn_row_scale_off.wgsl` (new) — the two per-chunk row-scales in the
    /// step-10 loop (`q_c*exp(g_cs_c)`, `key_c*decay_scale`) that read a
    /// chunk's slice out of a larger buffer.
    pub row_scale_off: usize,
    /// `gdn_decay_scale.wgsl`.
    pub decay_scale: usize,
    /// `gdn_state_decay.wgsl`.
    pub state_decay: usize,
    /// `exp.wgsl`.
    pub exp: usize,
    /// `sub.wgsl`.
    pub sub: usize,
    /// `mul.wgsl` (existing, unmodified) — `intra_scores = raw_qk * decay_mask`.
    pub mul: usize,
    /// `region_copy.wgsl` (existing, unmodified) — `g_cs = copy(raw_g)` and
    /// `final_state = copy(initial_state)` (the loop's working buffer).
    pub region_copy: usize,
}

/// The shape one call to [`gdn_chunk_fwd`] operates over. `b`/`h` are the
/// ALREADY-repeated (`num_v_heads`) batch/head counts — see this module's
/// doc for what is explicitly out of scope.
#[derive(Clone, Copy)]
pub struct GdnShape {
    pub b: u32,
    pub h: u32,
    pub t: u32,
    pub dk: u32,
    pub dv: u32,
    pub chunk: u32,
}

impl GdnShape {
    /// `t / chunk`. Panics if `t` is not an exact multiple of `chunk` —
    /// padding `t` is the CALLER's job (this module's doc, "What this module
    /// does NOT do"), not something silently rounded here.
    pub fn n_chunks(&self) -> u32 {
        assert_eq!(
            self.t % self.chunk,
            0,
            "GdnShape: t={} must be an exact multiple of chunk={} -- pad T on the host before calling gdn_chunk_fwd",
            self.t,
            self.chunk
        );
        self.t / self.chunk
    }

    /// `B*H` — the recurrent state's own batch count (no chunk axis).
    pub fn bh(&self) -> u32 {
        self.b * self.h
    }

    /// `B*H*n_chunks` — the batch count for every WHOLE-TENSOR (all chunks
    /// at once) dispatch in steps 1-9.
    pub fn bhc(&self) -> u32 {
        self.bh() * self.n_chunks()
    }
}

/// Every intermediate buffer [`gdn_chunk_fwd`] needs, sized once by the
/// caller. All follow this module's chunk-major convention (see the module
/// doc's "Layout" section) except where noted. `bhc = shape.bhc()`,
/// `bh = shape.bh()`, `c = shape.chunk`.
pub struct GdnScratch<'a> {
    /// `[bhc, c]` — cumulative per-chunk log-decay (copy of `raw_g`, then
    /// cumsum'd in place).
    pub g_cs: &'a DeviceBuffer,
    /// `[bhc, c]` — `exp(g_cs)`, materialised once (see module doc).
    pub exp_g_cs: &'a DeviceBuffer,
    /// `[bhc, c, dk]` — `key * beta`.
    pub k_beta: &'a DeviceBuffer,
    /// `[bhc, c, dv]` — `value * beta`.
    pub v_beta: &'a DeviceBuffer,
    /// `[bhc, c, dk]` — `k_beta * exp(g_cs)` (step 9's `k_cumdecay` input).
    pub k_beta_decay: &'a DeviceBuffer,
    /// `[bhc, c, c]` — the causal decay mask (step 5), reused by step 6 AND
    /// by the step-10 `intra_scores` precompute.
    pub decay_mask: &'a DeviceBuffer,
    /// `[bhc, c, c]` — `-(k_beta @ key^T)`, before masking.
    pub raw_attn0: &'a DeviceBuffer,
    /// `[bhc, c, c]` — masked `attn0`, frozen after step 6 (read-only input
    /// to every `gdn_ut_step.wgsl` dispatch — see that kernel's header for
    /// why it must not alias the evolving `t_mat`).
    pub attn0: &'a DeviceBuffer,
    /// `[bhc, c, c]` — the evolving `T_mat`. MUST be zeroed by the caller
    /// (pass it in `Gpu::submit`'s `clears` list) before submitting the
    /// steps [`gdn_chunk_fwd`] returns — see that function's own doc.
    pub t_mat: &'a DeviceBuffer,
    /// `[bhc, c, dv]` — `T_mat @ v_beta`.
    pub u: &'a DeviceBuffer,
    /// `[bhc, c, dk]` — `T_mat @ k_cumdecay`.
    pub w: &'a DeviceBuffer,
    /// `[bhc, c, c]` — `query @ key^T` (scaled by `1/sqrt(dk)`), before the
    /// decay-mask multiply.
    pub raw_intra: &'a DeviceBuffer,
    /// `[bhc, c, c]` — `raw_intra * decay_mask`, precomputed for every chunk
    /// before the sequential loop (chunk-independent, see module doc).
    pub intra_scores: &'a DeviceBuffer,
    /// `[bh, c, dk]` — one chunk's `query * exp(g_cs) * scale`, recomputed
    /// (overwritten) every loop iteration.
    pub q_scaled: &'a DeviceBuffer,
    /// `[bh, c]` — one chunk's `exp(g_cs_last - g_cs)`, recomputed every
    /// iteration.
    pub decay_scale: &'a DeviceBuffer,
    /// `[bh, c, dk]` — one chunk's `key * decay_scale`, recomputed every
    /// iteration.
    pub decayed_k: &'a DeviceBuffer,
    /// `[bh, c, dv]` — one chunk's `w_c @ state`, recomputed every iteration.
    pub v_prime: &'a DeviceBuffer,
    /// `[bh, c, dv]` — one chunk's `v_c - v_prime`, recomputed every
    /// iteration.
    pub v_new: &'a DeviceBuffer,
}

/// One `bmm.wgsl`/`bmm_acc.wgsl` dispatch. Public because a batched matmul
/// with offset-addressable batch slices is a general primitive, not a
/// GDN-only detail — a future caller assembling its own batched-matmul step
/// sequence can use this instead of re-deriving the `Params` order.
#[allow(clippy::too_many_arguments)]
pub fn bmm_step(
    g: &Gpu,
    kernel: usize,
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    a: &DeviceBuffer,
    a_off: u32,
    b: &DeviceBuffer,
    b_off: u32,
    out: &DeviceBuffer,
    out_off: u32,
) -> Step {
    g.step(
        kernel,
        &[a, b, out],
        &[batch, m, k, n, trans_a as u32, trans_b as u32, f(alpha), a_off, b_off, out_off],
        batch * m * n,
    )
}

/// The full Gated DeltaNet chunked-parallel forward — steps 1-11 of this
/// module's doc — as a step list. `initial_state`/`final_state` are
/// `[B,H,Dk,Dv]` (see module doc); `final_state` is both the OUTPUT recurrent
/// state (for the caller to persist / feed a later incremental call) and,
/// internally, the loop's own working buffer (initialised as a copy of
/// `initial_state` by this function's first two steps, then updated in
/// place per chunk) — pass a zeroed buffer as `initial_state` for "no prior
/// state".
///
/// **Caller MUST zero `scratch.t_mat` at submit time**: this function
/// returns a plain `Vec<Step>` (no clear primitive exists at the `Step`
/// level in this engine — only `Backend::submit`'s own `clears` list does),
/// so `scratch.t_mat` needs `g.submit(&[scratch.t_mat], &steps)`, not
/// `g.submit(&[], &steps)`. Every other scratch buffer is fully overwritten
/// by the steps that use it and needs no clear.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_fwd(
    g: &Gpu,
    ids: &GdnIds,
    shape: &GdnShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    raw_g: &DeviceBuffer,
    beta: &DeviceBuffer,
    initial_state: &DeviceBuffer,
    scratch: &GdnScratch,
    out: &DeviceBuffer,
    final_state: &DeviceBuffer,
) -> Vec<Step> {
    let (dk, dv, c) = (shape.dk, shape.dv, shape.chunk);
    let n_chunks = shape.n_chunks();
    let bh = shape.bh();
    let bhc = shape.bhc();
    let scale = 1.0f32 / (dk as f32).sqrt();

    let bmm = |kernel: usize, batch: u32, m: u32, k: u32, n: u32, ta: bool, tb: bool, alpha: f32, a: &DeviceBuffer, a_off: u32, b: &DeviceBuffer, b_off: u32, o: &DeviceBuffer, o_off: u32| {
        bmm_step(g, kernel, batch, m, k, n, ta, tb, alpha, a, a_off, b, b_off, o, o_off)
    };

    let mut steps = Vec::new();

    // ---- steps 1-2: v_beta = value*beta, k_beta = key*beta (whole tensor) ----
    steps.push(g.step(ids.row_scale, &[value, beta, scratch.v_beta], &[bhc * c * dv, dv], bhc * c * dv));
    steps.push(g.step(ids.row_scale, &[key, beta, scratch.k_beta], &[bhc * c * dk, dk], bhc * c * dk));

    // ---- step 4: g_cs = copy(raw_g), then the sequential per-chunk cumsum ----
    steps.push(g.step(ids.region_copy, &[raw_g, scratch.g_cs], &[1, bhc * c, bhc * c, 0], bhc * c));
    for i in 1..c {
        steps.push(g.step(ids.cumsum_step, &[scratch.g_cs], &[bhc, c, i], bhc));
    }

    // ---- step 5: decay_mask ----
    steps.push(g.step(ids.decay_mask, &[scratch.g_cs, scratch.decay_mask], &[bhc, c], bhc * c * c));

    // ---- step 6: attn0 = -(k_beta @ key^T), strictly-lower masked by decay_mask ----
    steps.push(bmm(ids.bmm, bhc, c, dk, c, false, true, -1.0, scratch.k_beta, 0, key, 0, scratch.raw_attn0, 0));
    steps.push(g.step(ids.mask_strict_lower, &[scratch.raw_attn0, scratch.decay_mask, scratch.attn0], &[bhc, c], bhc * c * c));

    // ---- step 7: UT-transform (forward substitution, then += I) ----
    for i in 1..c {
        steps.push(g.step(ids.ut_step, &[scratch.attn0, scratch.t_mat], &[bhc, c, i], bhc * i));
    }
    steps.push(g.step(ids.add_identity, &[scratch.t_mat], &[bhc, c], bhc * c));

    // ---- step 8: u = T_mat @ v_beta ----
    steps.push(bmm(ids.bmm, bhc, c, c, dv, false, false, 1.0, scratch.t_mat, 0, scratch.v_beta, 0, scratch.u, 0));

    // ---- step 9: w = T_mat @ (k_beta * exp(g_cs)) ----
    steps.push(g.step(ids.exp, &[scratch.g_cs, scratch.exp_g_cs], &[bhc * c], bhc * c));
    steps.push(g.step(ids.row_scale, &[scratch.k_beta, scratch.exp_g_cs, scratch.k_beta_decay], &[bhc * c * dk, dk], bhc * c * dk));
    steps.push(bmm(ids.bmm, bhc, c, c, dk, false, false, 1.0, scratch.t_mat, 0, scratch.k_beta_decay, 0, scratch.w, 0));

    // ---- intra_scores, precomputed for every chunk (state-independent) ----
    steps.push(bmm(ids.bmm, bhc, c, dk, c, false, true, scale, query, 0, key, 0, scratch.raw_intra, 0));
    steps.push(g.step(ids.mul, &[scratch.raw_intra, scratch.decay_mask, scratch.intra_scores], &[bhc * c * c], bhc * c * c));

    // ---- step 10: sequential across-chunk loop ----
    // `final_state` is the loop's own working buffer, seeded from `initial_state`.
    steps.push(g.step(ids.region_copy, &[initial_state, final_state], &[1, bh * dk * dv, bh * dk * dv, 0], bh * dk * dv));

    for ci in 0..n_chunks {
        // Chunk `ci`'s flat element offset into a `[n_chunks, bh, C, D]`
        // (or `[n_chunks, bh, C]`/`[n_chunks, bh, C, C]`) chunk-major buffer
        // — see this module's "Layout" doc for why this is a plain offset.
        let off_d = |d: u32| ci * bh * c * d;
        let off_g = ci * bh * c;
        let off_cc = ci * bh * c * c;

        // v_prime = w_c @ state  (state BEFORE this chunk's decay/update)
        steps.push(bmm(ids.bmm, bh, c, dk, dv, false, false, 1.0, scratch.w, off_d(dk), final_state, 0, scratch.v_prime, 0));
        // v_new = u_c - v_prime -- NOTE: the reference reassigns its "value"
        // variable to `u = T_mat @ v_beta` (step 8) BEFORE this line, so "v_c"
        // here means `u`'s chunk slice, NOT the original raw `value` tensor
        // (`torch_chunk_gated_delta_rule`'s `value = attn @ v_beta` shadows the
        // function's own `value` parameter; easy to miss, called out explicitly
        // in this module's doc and worth double-checking against the reference
        // source, not just this comment).
        steps.push(g.step(ids.sub, &[scratch.u, scratch.v_prime, scratch.v_new], &[bh * c * dv, off_d(dv), 0], bh * c * dv));
        // decay_scale[i] = exp(g_cs_c[-1] - g_cs_c[i])
        steps.push(g.step(ids.decay_scale, &[scratch.g_cs, scratch.decay_scale], &[bh, c, off_g], bh * c));
        // decayed_k = key_c * decay_scale
        steps.push(g.step(ids.row_scale_off, &[key, scratch.decay_scale, scratch.decayed_k], &[bh * c * dk, dk, off_d(dk), 0, f(1.0)], bh * c * dk));
        // q_scaled = query_c * exp(g_cs_c) * (1/sqrt(dk))
        steps.push(g.step(ids.row_scale_off, &[query, scratch.exp_g_cs, scratch.q_scaled], &[bh * c * dk, dk, off_d(dk), off_g, f(scale)], bh * c * dk));
        // out[chunk ci] = attn_inter = q_scaled @ state  (still the OLD state)
        steps.push(bmm(ids.bmm, bh, c, dk, dv, false, false, 1.0, scratch.q_scaled, 0, final_state, 0, out, off_d(dv)));
        // out[chunk ci] += intra_scores_c @ v_new
        steps.push(bmm(ids.bmm_acc, bh, c, c, dv, false, false, 1.0, scratch.intra_scores, off_cc, scratch.v_new, 0, out, off_d(dv)));
        // state *= exp(g_cs_c[-1])
        steps.push(g.step(ids.state_decay, &[scratch.g_cs, final_state], &[bh, dk, dv, c, off_g], bh * dk * dv));
        // state += decayed_k^T @ v_new
        steps.push(bmm(ids.bmm_acc, bh, dk, c, dv, true, false, 1.0, scratch.decayed_k, 0, scratch.v_new, 0, final_state, 0));
    }

    steps
}
