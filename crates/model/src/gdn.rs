// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gated DeltaNet (GDN) chunked-parallel linear-attention recurrence -
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
//! computed by the caller - this module does not compute them), chunked into
//! `n_chunks = T/C`:
//!
//! 1. `query *= 1/sqrt(Dk)` (folded into every kernel call that reads
//!    `query`, via `bmm`'s `alpha` or `gdn_row_scale_off`'s `alpha` - `query`
//!    itself is never mutated).
//! 2. `v_beta = value*beta`, `k_beta = key*beta` (row-broadcast, whole
//!    tensor - `scale_row.wgsl`, reused unmodified).
//! 3. (chunking is just how the flat buffer is addressed - see "Layout"
//!    below; no kernel work.)
//! 4. `g_cs = cumsum(g)` per chunk (`gdn_chunk_cumsum_step.wgsl`, one fused
//!    dispatch - a serial per-row loop, not one dispatch per row index).
//! 5. `decay_mask[i,j] = exp(g_cs[i]-g_cs[j])` for `j<=i`, else 0
//!    (`gdn_decay_mask.wgsl`).
//! 6. `attn0 = -(k_beta @ key^T) * decay_mask`, masked to `j<i`
//!    (`bmm.wgsl` with `alpha=-1,trans_b=1` then `gdn_mask_strict_lower.wgsl`).
//! 7. UT-transform: `T_mat = (I - attn0)^-1` by forward substitution
//!    (`gdn_ut_step.wgsl`, one host dispatch per row index) then `+= I`
//!    (`gdn_add_identity.wgsl`). `attn0`'s nilpotency also makes a
//!    repeated-squaring Neumann series exact and much cheaper in dispatches,
//!    but NOT numerically usable at real weights - see
//!    `gdn_chunk_fwd_prefix`'s own comment at this step for the conditioning
//!    argument and the test that pins it.
//! 8. `u = T_mat @ v_beta` (`bmm.wgsl`).
//! 9. `w = T_mat @ (k_beta * exp(g_cs))` (`exp.wgsl` + `scale_row.wgsl` +
//!    `bmm.wgsl`).
//! 10. Sequential across-chunk loop (state carries chunk-to-chunk): per
//!     chunk, `v_prime = w_c @ state`, `v_new = u_c - v_prime` (the
//!     reference REASSIGNS its `value` variable to `u` in step 8 -
//!     `torch_chunk_gated_delta_rule`'s `value = attn @ v_beta` shadows the
//!     function's own `value` PARAMETER, so every later `value`/`v_c` in the
//!     reference, including this one, means `u`'s chunk slice, never the
//!     original raw `value` tensor - easy to miss, worth re-checking against
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
//! ## Layout - CHUNK-MAJOR, a deliberate departure from `[B,H,T,D]`
//!
//! The reference (and this doc, above) describes shapes in PyTorch's
//! `[B,H,T,D]` axis order. This module instead requires every per-token
//! buffer (`query`,`key`,`value`,`raw_g`,`beta`,`out`) to be laid out as if
//! shaped `[n_chunks, B, H, C, D]` row-major - i.e. flat index
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
//! flat batch axis - addressable with a plain `Params` element offset
//! (`off_of` below), never a bound byte-offset slice (which is required to
//! be 256-byte aligned - this module's own gradcheck-style test at
//! deliberately tiny, non-aligned dims would fail under that scheme). With
//! `(b,h)` outermost instead (a literal
//! `[B,H,T,D]` reshape), the same per-chunk slice is STRIDED
//! (`stride = n_chunks * C * D`), which `bmm`/`bmm_acc` do not support (by
//! design - see `bmm.wgsl`'s own header for why offset-only was chosen).
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
//! sums `a,b` that can each be very negative over a whole chunk - computing
//! that as `exp(a)/exp(b)` from the materialised buffer risks dividing by an
//! UNDERFLOWED near-zero denominator that the direct `exp(a-b)` form never
//! hits. Recomputing costs one redundant transcendental per element in
//! exchange for not needing a second scratch buffer AND not risking that
//! numerical trap - a deliberate choice, not an oversight.
//!
//! ## What this module does NOT do
//!
//! Per the porting task this exists for: no GQA head-repeat (`H` here is
//! already `num_v_heads`), no L2-norm, no gated RMSNorm, no
//! decay-gate computation (`raw_g`/`beta` arrive ready-made), no T-padding
//! (caller must pass `t` already a multiple of `chunk` - [`GdnShape::n_chunks`]
//! asserts it). [`gdn_chunk_fwd`] ITSELF remains chunked/prefill (steps 1-11)
//! only, unchanged - the single-token recurrence and the causal depthwise
//! conv decode step this paragraph used to say were out of scope are now
//! provided as SEPARATE, decode-only entry points below: [`gdn_recurrent_step`]
//! (the per-token state update, validated against this module's own
//! [`gdn_chunk_fwd`] at `chunk=1` - see that function's doc) and
//! [`gdn_causal_conv1d_step`] (the streaming ring-buffer sibling of
//! `audio::conv::conv1d_fwd`'s whole-sequence causal conv, validated against
//! it the same way).
//!
//! ## Backward
//!
//! **Implemented** - [`gdn_chunk_bwd`], gradient-checked at this module's own
//! tiny shape by `crates/model/tests/gdn_chunk_bwd.rs` (finite-difference,
//! f64 host oracle, both backends). It needs every chunk's OWN version of the
//! five small per-chunk buffers ([`GdnScratch::q_scaled`] and friends) that
//! [`gdn_chunk_fwd`] overwrites each loop iteration, plus the full recurrent
//! state history (one [`gdn_chunk_fwd`] only keeps the latest) - so a
//! training run does NOT call [`gdn_chunk_fwd`] at all: it calls
//! [`gdn_chunk_fwd_train`], the training-mode sibling that SAVES what
//! backward needs (same math, same `out`/`final_state`, see that function's
//! own doc for the exact promotion from `[bh,c,d]` overwritten-in-place to
//! `[bhc,c,d]` saved-per-chunk). [`gdn_chunk_fwd`] itself is UNCHANGED -
//! still the inference-only entry point, still what
//! `crates/model/tests/gdn_chunk_fwd.rs` gates.
//!
//! [`gdn_chunk_bwd`]'s own doc covers the two-phase structure (a reverse
//! sweep over chunks threading the recurrent state's gradient backward, then
//! a reverse-forward-step-order pass over the whole-tensor precompute) at a
//! summary level; the full per-step derivation lived in the porting task this
//! module was written against and is reproduced in this function's doc only
//! at the level a future maintainer needs to extend it, not line-for-line.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Kernel indices [`gdn_chunk_fwd`] dispatches, resolved by the calling model
/// against its own registered pipeline list (same pattern as
/// [`crate::moe::MoeIds`]).
#[derive(Clone, Copy)]
pub struct GdnIds {
    /// `bmm.wgsl` - batched matmul, overwrite.
    pub bmm: usize,
    /// `bmm_acc.wgsl` - batched matmul, accumulate.
    pub bmm_acc: usize,
    /// `gdn_chunk_cumsum_step.wgsl`.
    pub cumsum_step: usize,
    /// `gdn_decay_mask.wgsl`.
    pub decay_mask: usize,
    /// `gdn_mask_strict_lower.wgsl` - finishes step 6's `attn0` from
    /// `bmm.wgsl`'s raw `-(k_beta @ key^T)`.
    pub mask_strict_lower: usize,
    /// `gdn_ut_step.wgsl`.
    pub ut_step: usize,
    /// `gdn_add_identity.wgsl`.
    pub add_identity: usize,
    /// `scale_row.wgsl` (existing, unmodified) - every WHOLE-TENSOR row-scale
    /// (`v_beta`, `k_beta`, `k_cumdecay`'s `k_beta*exp(g_cs)`).
    pub row_scale: usize,
    /// `gdn_row_scale_off.wgsl` (new) - the two per-chunk row-scales in the
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
    /// `mul.wgsl` (existing, unmodified) - `intra_scores = raw_qk * decay_mask`.
    pub mul: usize,
    /// `region_copy.wgsl` (existing, unmodified) - `g_cs = copy(raw_g)` and
    /// `final_state = copy(initial_state)` (the loop's working buffer).
    pub region_copy: usize,
    /// Faster, contract-equivalent kernels for the prefill-shaped forward, for
    /// a caller that registered them. `None` is always valid; whether a device
    /// actually takes them is [`use_fast_kernels`]'s decision, not the
    /// caller's.
    pub fast: Option<GdnFastIds>,
}

/// Kernels that replace a run of dispatches in [`gdn_chunk_fwd`] with fewer,
/// bigger ones while computing the same numbers. Registered by a caller that
/// has a device to run them on; see [`use_fast_kernels`].
#[derive(Clone, Copy)]
pub struct GdnFastIds {
    /// `gdn_ut_fwd.wgsl` - the whole UT transform (step 7 and its identity) in
    /// one workgroup-per-matrix dispatch instead of `chunk - 1` row launches.
    pub ut_fwd: usize,
    /// `bmm_tiled.wgsl` - the 64x64-tile, register-blocked batched matmul that
    /// serves both `bmm` and `bmm_acc` (bit-identical to them).
    pub bmm_tiled: usize,
    /// Whether the fused tensor-core prefix (`gdn_chunk_prep_tc`, see
    /// [`native_chunk_prep`]) may replace the whole-tensor steps 1-9 on a device
    /// that offers it. Unlike the other fast kernels it is NOT bit-identical to
    /// the portable path (three-pass TF32 products, agreement about 1e-6
    /// relative), so a caller that needs the portable numbers - or a parity
    /// test that wants the reference - leaves it `false`.
    pub tensor_core: bool,
}

/// Whether `g` takes [`GdnIds::fast`] kernels.
///
/// They are workgroup-cooperative WGSL kernels, bit-identical to the loops
/// they replace (`tests/gdn_chunk_fwd_fast.rs`, `tests/gdn_ut_fwd.rs`), and
/// faster on every CUDA generation with `dp4a` (compute capability 6.1, where
/// the portable sequence's batched matmuls were a third of a prefill round)
/// and newer; every other device keeps the portable dispatch sequence it has
/// always run. The gate is the queried compute capability (never a device
/// name), and `BRAIN_NO_GDN_FAST=1` switches it off for A/B measurement.
pub fn use_fast_kernels(g: &Gpu) -> bool {
    std::env::var_os("BRAIN_NO_GDN_FAST").is_none() && g.caps().arch.compute_capability.is_some_and(|cc| cc >= (6, 1))
}

/// The shape one call to [`gdn_chunk_fwd`] operates over. `b`/`h` are the
/// ALREADY-repeated (`num_v_heads`) batch/head counts - see this module's
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
    /// `t / chunk`. Panics if `t` is not an exact multiple of `chunk` -
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

    /// `B*H` - the recurrent state's own batch count (no chunk axis).
    pub fn bh(&self) -> u32 {
        self.b * self.h
    }

    /// `B*H*n_chunks` - the batch count for every WHOLE-TENSOR (all chunks
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
    /// `[bhc, c]` - cumulative per-chunk log-decay (copy of `raw_g`, then
    /// cumsum'd in place).
    pub g_cs: &'a DeviceBuffer,
    /// `[bhc, c]` - `exp(g_cs)`, materialised once (see module doc).
    pub exp_g_cs: &'a DeviceBuffer,
    /// `[bhc, c, dk]` - `key * beta`.
    pub k_beta: &'a DeviceBuffer,
    /// `[bhc, c, dv]` - `value * beta`.
    pub v_beta: &'a DeviceBuffer,
    /// `[bhc, c, dk]` - `k_beta * exp(g_cs)` (step 9's `k_cumdecay` input).
    pub k_beta_decay: &'a DeviceBuffer,
    /// `[bhc, c, c]` - the causal decay mask (step 5), reused by step 6 AND
    /// by the step-10 `intra_scores` precompute.
    pub decay_mask: &'a DeviceBuffer,
    /// `[bhc, c, c]` - `-(k_beta @ key^T)`, before masking.
    pub raw_attn0: &'a DeviceBuffer,
    /// `[bhc, c, c]` - masked `attn0`, frozen after step 6 (read-only input
    /// to every `gdn_ut_step.wgsl` dispatch - see that kernel's header for
    /// why it must not alias the evolving `t_mat`).
    pub attn0: &'a DeviceBuffer,
    /// `[bhc, c, c]` - the evolving `T_mat`. MUST be zeroed by the caller
    /// (pass it in `Gpu::submit`'s `clears` list) before submitting the
    /// steps [`gdn_chunk_fwd`] returns - see that function's own doc.
    pub t_mat: &'a DeviceBuffer,
    /// `[bhc, c, dv]` - `T_mat @ v_beta`.
    pub u: &'a DeviceBuffer,
    /// `[bhc, c, dk]` - `T_mat @ k_cumdecay`.
    pub w: &'a DeviceBuffer,
    /// `[bhc, c, c]` - `query @ key^T` (scaled by `1/sqrt(dk)`), before the
    /// decay-mask multiply.
    pub raw_intra: &'a DeviceBuffer,
    /// `[bhc, c, c]` - `raw_intra * decay_mask`, precomputed for every chunk
    /// before the sequential loop (chunk-independent, see module doc).
    pub intra_scores: &'a DeviceBuffer,
    /// `[bh, c, dk]` - one chunk's `query * exp(g_cs) * scale`, recomputed
    /// (overwritten) every loop iteration.
    pub q_scaled: &'a DeviceBuffer,
    /// `[bh, c]` - one chunk's `exp(g_cs_last - g_cs)`, recomputed every
    /// iteration.
    pub decay_scale: &'a DeviceBuffer,
    /// `[bh, c, dk]` - one chunk's `key * decay_scale`, recomputed every
    /// iteration.
    pub decayed_k: &'a DeviceBuffer,
    /// `[bh, c, dv]` - one chunk's `w_c @ state`, recomputed every iteration.
    pub v_prime: &'a DeviceBuffer,
    /// `[bh, c, dv]` - one chunk's `v_c - v_prime`, recomputed every
    /// iteration.
    pub v_new: &'a DeviceBuffer,
}

/// Every buffer [`gdn_chunk_fwd_train`] needs - the training-mode sibling of
/// [`GdnScratch`] that additionally SAVES what [`gdn_chunk_bwd`] needs: every
/// chunk's own `decay_scale` and the FULL recurrent-state history rather than
/// just the latest chunk's state. The other four per-chunk working buffers
/// are recomputed by backward, not saved. See
/// [`gdn_chunk_fwd_train`]'s own doc for exactly which steps write which
/// field. `bhc = shape.bhc()`, `bh = shape.bh()`, `c = shape.chunk`,
/// `n_chunks = shape.n_chunks()`.
///
/// **The caller MUST zero `decay_scale_hist` and `state_history`** (pass
/// them in [`Gpu::submit`]'s `clears` list, alongside `t_mat` - see
/// [`GdnScratch::t_mat`]'s own doc for why this engine has no clear
/// primitive at the `Step` level) - every one of them is populated via
/// `splice_add.wgsl`'s `dst[base+i] += src[i]`, so a non-zero starting value
/// would silently corrupt the saved history with garbage. The 5 small
/// `q_scaled`/`decay_scale`/`decayed_k`/`v_prime`/`v_new` working buffers and
/// `t_mat` need exactly [`GdnScratch`]'s own clearing contract (`t_mat`
/// only); the working buffers are fully overwritten every iteration by the
/// SAME kernels [`gdn_chunk_fwd`] uses, unchanged.
pub struct GdnScratchTrain<'a> {
    // ---- the 13 whole-tensor buffers, identical role to `GdnScratch`'s own ----
    pub g_cs: &'a DeviceBuffer,
    pub exp_g_cs: &'a DeviceBuffer,
    pub k_beta: &'a DeviceBuffer,
    pub v_beta: &'a DeviceBuffer,
    pub k_beta_decay: &'a DeviceBuffer,
    pub decay_mask: &'a DeviceBuffer,
    pub raw_attn0: &'a DeviceBuffer,
    pub attn0: &'a DeviceBuffer,
    pub t_mat: &'a DeviceBuffer,
    pub u: &'a DeviceBuffer,
    pub w: &'a DeviceBuffer,
    pub raw_intra: &'a DeviceBuffer,
    pub intra_scores: &'a DeviceBuffer,
    // ---- the 5 small `[bh,c,d]` per-iteration working buffers, identical
    // role (and identical kernel calls) to `GdnScratch`'s own same-named
    // fields -- overwritten every chunk, NOT what backward reads back.
    pub q_scaled: &'a DeviceBuffer,
    pub decay_scale: &'a DeviceBuffer,
    pub decayed_k: &'a DeviceBuffer,
    pub v_prime: &'a DeviceBuffer,
    pub v_new: &'a DeviceBuffer,
    // ---- SAVED per-chunk history: `[bhc, c]`, chunk `ci`'s slice holding
    // `decay_scale` as it was during chunk `ci`'s own iteration. It is the
    // only per-chunk value kept (one float per token per head): backward
    // recomputes `q_scaled`, `decayed_k`, `v_prime` and `v_new` from it and
    // from `state_history`, because those four are `dk`/`dv` wide (about 25K
    // floats per token at the real width) and cost one cheap kernel each.
    pub decay_scale_hist: &'a DeviceBuffer,
    // ---- full recurrent-state history: `[n_chunks+1, bh, dk, dv]`.
    // `state_history[0]` = a saved copy of `initial_state`; `state_history[ci+1]`
    // = the state AFTER chunk `ci`'s decay+update (what `gdn_chunk_fwd`'s own
    // `final_state` holds only the LATEST version of). `gdn_chunk_bwd` reads
    // `state_history[ci]` as chunk `ci`'s `state_in`.
    pub state_history: &'a DeviceBuffer,
}

/// The Gated DeltaNet chunk size a forward uses. The reference default is 64
/// (`torch_chunk_gated_delta_rule`'s `chunk_size=64`); this module is
/// prefill-only and asserts `t % chunk == 0` itself (no padding support), so
/// rather than force every caller's `t` to be a multiple of 64, this picks
/// the LARGEST candidate in `[64,32,16,8,4,2,1]` that divides `t` exactly -
/// landing on exactly 64 at real hybrid-decoder scale (`block_size=4096`,
/// `4096%64==0`) while still giving a tiny test config (`block_size=24`) a
/// genuinely multi-chunk exercise (`24%8==0` -> chunk 8, 3 chunks) instead of
/// silently collapsing to one giant chunk covering the whole sequence (which
/// `t%t==0` would always pick first and would never exercise the
/// cross-chunk recurrence at all).
pub fn gdn_chunk_size(t: u32) -> u32 {
    for c in [64, 32, 16, 8, 4, 2, 1] {
        if t.is_multiple_of(c) {
            return c;
        }
    }
    1
}

/// Owned scratch buffers for one [`gdn_chunk_fwd`] call - see [`GdnScratch`]'s
/// own doc for what each buffer holds. Freshly allocated per Gated-DeltaNet
/// layer call (never shared/reused across layers or across forward passes),
/// so every buffer here is zero-fresh - `t_mat` is still passed through
/// `Gpu::submit`'s `clears` list at the call site per [`gdn_chunk_fwd`]'s own
/// documented contract, defensively.
pub struct GdnScratchBufs {
    g_cs: DeviceBuffer,
    exp_g_cs: DeviceBuffer,
    k_beta: DeviceBuffer,
    v_beta: DeviceBuffer,
    k_beta_decay: DeviceBuffer,
    decay_mask: DeviceBuffer,
    raw_attn0: DeviceBuffer,
    attn0: DeviceBuffer,
    t_mat: DeviceBuffer,
    u: DeviceBuffer,
    w: DeviceBuffer,
    raw_intra: DeviceBuffer,
    intra_scores: DeviceBuffer,
    q_scaled: DeviceBuffer,
    decay_scale: DeviceBuffer,
    decayed_k: DeviceBuffer,
    v_prime: DeviceBuffer,
    v_new: DeviceBuffer,
}

impl GdnScratchBufs {
    pub fn new(g: &Gpu, shape: &GdnShape) -> GdnScratchBufs {
        let bhc = shape.bhc() as u64;
        let bh = shape.bh() as u64;
        let c = shape.chunk as u64;
        let dk = shape.dk as u64;
        let dv = shape.dv as u64;
        GdnScratchBufs {
            g_cs: g.storage(bhc * c),
            exp_g_cs: g.storage(bhc * c),
            k_beta: g.storage(bhc * c * dk),
            v_beta: g.storage(bhc * c * dv),
            k_beta_decay: g.storage(bhc * c * dk),
            decay_mask: g.storage(bhc * c * c),
            raw_attn0: g.storage(bhc * c * c),
            attn0: g.storage(bhc * c * c),
            t_mat: g.storage(bhc * c * c),
            u: g.storage(bhc * c * dv),
            w: g.storage(bhc * c * dk),
            raw_intra: g.storage(bhc * c * c),
            intra_scores: g.storage(bhc * c * c),
            q_scaled: g.storage(bh * c * dk),
            decay_scale: g.storage(bh * c),
            decayed_k: g.storage(bh * c * dk),
            v_prime: g.storage(bh * c * dv),
            v_new: g.storage(bh * c * dv),
        }
    }

    pub fn as_ref(&self) -> GdnScratch<'_> {
        GdnScratch {
            g_cs: &self.g_cs,
            exp_g_cs: &self.exp_g_cs,
            k_beta: &self.k_beta,
            v_beta: &self.v_beta,
            k_beta_decay: &self.k_beta_decay,
            decay_mask: &self.decay_mask,
            raw_attn0: &self.raw_attn0,
            attn0: &self.attn0,
            t_mat: &self.t_mat,
            u: &self.u,
            w: &self.w,
            raw_intra: &self.raw_intra,
            intra_scores: &self.intra_scores,
            q_scaled: &self.q_scaled,
            decay_scale: &self.decay_scale,
            decayed_k: &self.decayed_k,
            v_prime: &self.v_prime,
            v_new: &self.v_new,
        }
    }

    /// Buffers the caller MUST zero (pass in [`Gpu::submit`]'s `clears` list)
    /// before submitting [`gdn_chunk_fwd`]'s steps - just `t_mat`, see
    /// [`GdnScratch::t_mat`]'s own doc for why.
    pub fn clears(&self) -> Vec<&DeviceBuffer> {
        vec![&self.t_mat]
    }
}

/// The scratch [`gdn_chunk_fwd_tokens`] reads and writes: the five buffers its prep and recurrence kernels pass between
/// them. The prep kernel writes every word of each before the recurrence reads it, so none is cleared, and the
/// thirteen buffers of [`GdnScratch`] that the portable sequence needs are not allocated at all.
pub struct GdnTokScratch<'a> {
    pub g_cs: &'a DeviceBuffer,
    pub exp_g_cs: &'a DeviceBuffer,
    pub u: &'a DeviceBuffer,
    pub w: &'a DeviceBuffer,
    pub intra_scores: &'a DeviceBuffer,
}

/// Owner of a [`GdnTokScratch`].
pub struct GdnTokScratchBufs {
    g_cs: DeviceBuffer,
    exp_g_cs: DeviceBuffer,
    u: DeviceBuffer,
    w: DeviceBuffer,
    intra_scores: DeviceBuffer,
}

impl GdnTokScratchBufs {
    pub fn new(g: &Gpu, shape: &GdnShape) -> GdnTokScratchBufs {
        let (bhc, c, dk, dv) = (shape.bhc() as u64, shape.chunk as u64, shape.dk as u64, shape.dv as u64);
        GdnTokScratchBufs {
            g_cs: g.storage_uninit(bhc * c),
            exp_g_cs: g.storage_uninit(bhc * c),
            u: g.storage_uninit(bhc * c * dv),
            w: g.storage_uninit(bhc * c * dk),
            intra_scores: g.storage_uninit(bhc * c * c),
        }
    }

    pub fn as_ref(&self) -> GdnTokScratch<'_> {
        GdnTokScratch { g_cs: &self.g_cs, exp_g_cs: &self.exp_g_cs, u: &self.u, w: &self.w, intra_scores: &self.intra_scores }
    }
}

/// Owned scratch for one [`gdn_chunk_fwd_train`] call - the training-mode
/// sibling of [`GdnScratchBufs`] that additionally saves the per-chunk
/// history [`gdn_chunk_bwd`] reads back (see [`GdnScratchTrain`]'s own doc for
/// exactly which field is which). `clears()` lists every field
/// [`gdn_chunk_fwd_train`]'s own doc says MUST be zeroed before submit
/// (`t_mat`, `decay_scale_hist`, `state_history`).
pub struct GdnScratchTrainBufs {
    g_cs: DeviceBuffer,
    exp_g_cs: DeviceBuffer,
    k_beta: DeviceBuffer,
    v_beta: DeviceBuffer,
    k_beta_decay: DeviceBuffer,
    decay_mask: DeviceBuffer,
    raw_attn0: DeviceBuffer,
    attn0: DeviceBuffer,
    t_mat: DeviceBuffer,
    u: DeviceBuffer,
    w: DeviceBuffer,
    raw_intra: DeviceBuffer,
    intra_scores: DeviceBuffer,
    q_scaled: DeviceBuffer,
    decay_scale: DeviceBuffer,
    decayed_k: DeviceBuffer,
    v_prime: DeviceBuffer,
    v_new: DeviceBuffer,
    decay_scale_hist: DeviceBuffer,
    state_history: DeviceBuffer,
}

impl GdnScratchTrainBufs {
    /// No field is cleared at allocation: a step of the chunk forward writes each before it is read, except the
    /// three `clears()` names, which the caller zeroes at submit (a debug build poisons the blocks).
    pub fn new(g: &Gpu, shape: &GdnShape) -> GdnScratchTrainBufs {
        let bhc = shape.bhc() as u64;
        let bh = shape.bh() as u64;
        let c = shape.chunk as u64;
        let dk = shape.dk as u64;
        let dv = shape.dv as u64;
        let n_chunks = shape.n_chunks() as u64;
        GdnScratchTrainBufs {
            g_cs: g.storage_uninit(bhc * c),
            exp_g_cs: g.storage_uninit(bhc * c),
            k_beta: g.storage_uninit(bhc * c * dk),
            v_beta: g.storage_uninit(bhc * c * dv),
            k_beta_decay: g.storage_uninit(bhc * c * dk),
            decay_mask: g.storage_uninit(bhc * c * c),
            raw_attn0: g.storage_uninit(bhc * c * c),
            attn0: g.storage_uninit(bhc * c * c),
            t_mat: g.storage_uninit(bhc * c * c),
            u: g.storage_uninit(bhc * c * dv),
            w: g.storage_uninit(bhc * c * dk),
            raw_intra: g.storage_uninit(bhc * c * c),
            intra_scores: g.storage_uninit(bhc * c * c),
            q_scaled: g.storage_uninit(bh * c * dk),
            decay_scale: g.storage_uninit(bh * c),
            decayed_k: g.storage_uninit(bh * c * dk),
            v_prime: g.storage_uninit(bh * c * dv),
            v_new: g.storage_uninit(bh * c * dv),
            decay_scale_hist: g.storage_uninit(bhc * c),
            state_history: g.storage_uninit((n_chunks + 1) * bh * dk * dv),
        }
    }

    pub fn as_ref(&self) -> GdnScratchTrain<'_> {
        GdnScratchTrain {
            g_cs: &self.g_cs,
            exp_g_cs: &self.exp_g_cs,
            k_beta: &self.k_beta,
            v_beta: &self.v_beta,
            k_beta_decay: &self.k_beta_decay,
            decay_mask: &self.decay_mask,
            raw_attn0: &self.raw_attn0,
            attn0: &self.attn0,
            t_mat: &self.t_mat,
            u: &self.u,
            w: &self.w,
            raw_intra: &self.raw_intra,
            intra_scores: &self.intra_scores,
            q_scaled: &self.q_scaled,
            decay_scale: &self.decay_scale,
            decayed_k: &self.decayed_k,
            v_prime: &self.v_prime,
            v_new: &self.v_new,
            decay_scale_hist: &self.decay_scale_hist,
            state_history: &self.state_history,
        }
    }

    pub fn clears(&self) -> Vec<&DeviceBuffer> {
        vec![
            &self.t_mat,
            &self.decay_scale_hist,
            &self.state_history,
        ]
    }
}

/// Owned scratch for one [`gdn_chunk_bwd`] call - see [`GdnBwdScratch`]'s own
/// doc for what each field holds and which MUST be zeroed (`clears()` below:
/// `d_g_cs`, `d_exp_g_cs`, `d_u`, `d_decay_mask`).
pub struct GdnBwdScratchBufs {
    d_decayed_k: DeviceBuffer,
    d_q_scaled: DeviceBuffer,
    d_v_new: DeviceBuffer,
    d_decay_scale: DeviceBuffer,
    v_prime_all: DeviceBuffer,
    v_new_all: DeviceBuffer,
    decayed_k_all: DeviceBuffer,
    q_scaled_all: DeviceBuffer,
    state_a: DeviceBuffer,
    d_raw_intra: DeviceBuffer,
    d_k_beta_decay: DeviceBuffer,
    d_v_beta: DeviceBuffer,
    d_raw_attn0: DeviceBuffer,
    d_attn0: DeviceBuffer,
    d_g_cs: DeviceBuffer,
    d_exp_g_cs: DeviceBuffer,
    d_t_mat: DeviceBuffer,
    d_u: DeviceBuffer,
    d_w: DeviceBuffer,
    d_intra_scores: DeviceBuffer,
    d_decay_mask: DeviceBuffer,
    d_k_beta: DeviceBuffer,
    dot_scratch: DeviceBuffer,
    mul_scratch: DeviceBuffer,
    mul_scratch_cc: DeviceBuffer,
    d_state_hist: Option<DeviceBuffer>,
}

impl GdnBwdScratchBufs {
    /// The scratch of the portable backward and of the fast kernels that keep its dispatch sequence.
    /// Nothing is cleared at allocation; the four `clears()` names are zeroed by the caller at submit, and a
    /// debug build poisons the rest.
    pub fn new(g: &Gpu, shape: &GdnShape) -> GdnBwdScratchBufs {
        let bhc = shape.bhc() as u64;
        let bh = shape.bh() as u64;
        let c = shape.chunk as u64;
        let dk = shape.dk as u64;
        let dv = shape.dv as u64;
        GdnBwdScratchBufs {
            d_decayed_k: g.storage_uninit(bhc * c * dk),
            d_q_scaled: g.storage_uninit(bhc * c * dk),
            d_v_new: g.storage_uninit(bhc * c * dv),
            d_decay_scale: g.storage_uninit(bhc * c),
            v_prime_all: g.storage_uninit(bhc * c * dv),
            v_new_all: g.storage_uninit(bhc * c * dv),
            decayed_k_all: g.storage_uninit(bhc * c * dk),
            q_scaled_all: g.storage_uninit(bhc * c * dk),
            state_a: g.storage_uninit(bh * dk * dv),
            d_raw_intra: g.storage_uninit(bhc * c * c),
            d_k_beta_decay: g.storage_uninit(bhc * c * dk),
            d_v_beta: g.storage_uninit(bhc * c * dv),
            d_raw_attn0: g.storage_uninit(bhc * c * c),
            d_attn0: g.storage_uninit(bhc * c * c),
            d_g_cs: g.storage_uninit(bhc * c),
            d_exp_g_cs: g.storage_uninit(bhc * c),
            d_t_mat: g.storage_uninit(bhc * c * c),
            d_u: g.storage_uninit(bhc * c * dv),
            d_w: g.storage_uninit(bhc * c * dk),
            d_intra_scores: g.storage_uninit(bhc * c * c),
            d_decay_mask: g.storage_uninit(bhc * c * c),
            d_k_beta: g.storage_uninit(bhc * c * dk),
            dot_scratch: g.storage_uninit(bhc * c),
            mul_scratch: g.storage_uninit(bhc * c),
            mul_scratch_cc: g.storage_uninit(bhc * c * c),
            d_state_hist: None,
        }
    }

    /// [`Self::new`] plus the state-gradient history the tensor-core backward needs, allocated only
    /// when `ids` opts in to the tensor-core kernels on a device that takes them and the shape is
    /// one they serve ([`tensor_core_backward_offered`]). A call that is later declined by the
    /// backend simply leaves the buffer unused.
    pub fn new_for(g: &Gpu, ids: &GdnIds, shape: &GdnShape) -> GdnBwdScratchBufs {
        let mut bufs = Self::new(g, shape);
        if tensor_core_backward_offered(g, ids, shape) {
            bufs.d_state_hist = Some(g.storage_uninit(shape.bhc() as u64 * shape.dk as u64 * shape.dv as u64));
        }
        bufs
    }

    pub fn as_ref(&self) -> GdnBwdScratch<'_> {
        GdnBwdScratch {
            d_decayed_k: &self.d_decayed_k,
            d_q_scaled: &self.d_q_scaled,
            d_v_new: &self.d_v_new,
            d_decay_scale: &self.d_decay_scale,
            v_prime_all: &self.v_prime_all,
            v_new_all: &self.v_new_all,
            decayed_k_all: &self.decayed_k_all,
            q_scaled_all: &self.q_scaled_all,
            state_a: &self.state_a,
            d_raw_intra: &self.d_raw_intra,
            d_k_beta_decay: &self.d_k_beta_decay,
            d_v_beta: &self.d_v_beta,
            d_raw_attn0: &self.d_raw_attn0,
            d_attn0: &self.d_attn0,
            d_g_cs: &self.d_g_cs,
            d_exp_g_cs: &self.d_exp_g_cs,
            d_t_mat: &self.d_t_mat,
            d_u: &self.d_u,
            d_w: &self.d_w,
            d_intra_scores: &self.d_intra_scores,
            d_decay_mask: &self.d_decay_mask,
            d_k_beta: &self.d_k_beta,
            dot_scratch: &self.dot_scratch,
            mul_scratch: &self.mul_scratch,
            mul_scratch_cc: &self.mul_scratch_cc,
            d_state_hist: self.d_state_hist.as_ref(),
        }
    }

    pub fn clears(&self) -> Vec<&DeviceBuffer> {
        vec![&self.d_g_cs, &self.d_exp_g_cs, &self.d_u, &self.d_decay_mask]
    }
}

/// [`gpu_core::provider::cuda::gdn_chunk_loop_step`] where a native backend
/// exists at all; `None` (the portable chunk loop) on wasm.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop(g: &Gpu, bufs: &[&DeviceBuffer; 10], bh: u32, c: u32, dk: u32, dv: u32, n_chunks: u32, scale: f32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_loop_step(g, bufs, bh, c, dk, dv, n_chunks, scale)
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop(_g: &Gpu, _bufs: &[&DeviceBuffer; 10], _bh: u32, _c: u32, _dk: u32, _dv: u32, _n_chunks: u32, _scale: f32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_chunk_loop_train_step`] where a native
/// backend exists at all; `None` (the portable chunk loop) on wasm.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_train(g: &Gpu, bufs: &[&DeviceBuffer; 12], bh: u32, c: u32, dk: u32, dv: u32, n_chunks: u32, scale: f32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_loop_train_step(g, bufs, bh, c, dk, dv, n_chunks, scale)
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_train(_g: &Gpu, _bufs: &[&DeviceBuffer; 12], _bh: u32, _c: u32, _dk: u32, _dv: u32, _n_chunks: u32, _scale: f32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_bwd_sweep_step`] where a native backend
/// exists at all; `None` (the host walk of the reverse sweep) on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn native_bwd_sweep(g: &Gpu, bufs: &[&DeviceBuffer; 12], bh: u32, c: u32, dk: u32, dv: u32, n_chunks: u32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_bwd_sweep_step(g, bufs, bh, c, dk, dv, n_chunks)
}

#[cfg(target_arch = "wasm32")]
fn native_bwd_sweep(_g: &Gpu, _bufs: &[&DeviceBuffer; 12], _bh: u32, _c: u32, _dk: u32, _dv: u32, _n_chunks: u32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_bwd_sweep_tc_step`] where a native backend exists at all; `None` on wasm.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn native_bwd_sweep_tc(g: &Gpu, bufs: &[&DeviceBuffer; 12], bh: u32, c: u32, dk: u32, dv: u32, n_chunks: u32, scale: f32, heads_tok: u32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_bwd_sweep_tc_step(g, bufs, bh, c, dk, dv, n_chunks, scale, heads_tok)
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn native_bwd_sweep_tc(_g: &Gpu, _bufs: &[&DeviceBuffer; 12], _bh: u32, _c: u32, _dk: u32, _dv: u32, _n_chunks: u32, _scale: f32, _heads_tok: u32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_chunk_bwd_post_tc_step`] where a native backend exists at all; `None` on wasm.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn native_bwd_post_tc(g: &Gpu, bufs: &[&DeviceBuffer; 21], bhc: u32, c: u32, dk: u32, dv: u32, bh: u32, scale: f32, heads_tok: u32, n_chunks: u32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_bwd_post_tc_step(g, bufs, bhc, c, dk, dv, bh, scale, heads_tok, n_chunks)
}

/// [`gpu_core::provider::cuda::gdn_qk_bwd_tok_step`] where a native backend exists at all; `None` on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn native_qk_bwd_tok(g: &Gpu, bufs: &[&DeviceBuffer; 6], shape: &GdnShape, group: u32, l2_eps: f32) -> Option<Step> {
    let layout = gpu_core::provider::cuda::GdnQkBwdLayout {
        tokens: shape.b * shape.t,
        kv_heads: shape.h / group,
        group,
        heads: shape.h,
        c_len: shape.chunk,
        seq_len: shape.t,
        batch: shape.b,
        eps: l2_eps,
    };
    gpu_core::provider::cuda::gdn_qk_bwd_tok_step(g, bufs, layout)
}

#[cfg(target_arch = "wasm32")]
fn native_qk_bwd_tok(_g: &Gpu, _bufs: &[&DeviceBuffer; 6], _shape: &GdnShape, _group: u32, _l2_eps: f32) -> Option<Step> {
    None
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn native_bwd_post_tc(_g: &Gpu, _bufs: &[&DeviceBuffer; 21], _bhc: u32, _c: u32, _dk: u32, _dv: u32, _bh: u32, _scale: f32, _heads_tok: u32, _n_chunks: u32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_chunk_prep_step`] where a native backend
/// exists at all; `None` (the portable whole-tensor prefix) on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn native_chunk_prep(g: &Gpu, bufs: &[&DeviceBuffer; 10], bhc: u32, c: u32, dk: u32, dv: u32, scale: f32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_prep_step(g, bufs, bhc, c, dk, dv, scale)
}

#[cfg(target_arch = "wasm32")]
fn native_chunk_prep(_g: &Gpu, _bufs: &[&DeviceBuffer; 10], _bhc: u32, _c: u32, _dk: u32, _dv: u32, _scale: f32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_chunk_prep_train_step`] where a native
/// backend exists at all; `None` on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn native_chunk_prep_train(g: &Gpu, bufs: &[&DeviceBuffer; 17], bhc: u32, c: u32, dk: u32, dv: u32, scale: f32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_prep_train_step(g, bufs, bhc, c, dk, dv, scale)
}

#[cfg(target_arch = "wasm32")]
fn native_chunk_prep_train(_g: &Gpu, _bufs: &[&DeviceBuffer; 17], _bhc: u32, _c: u32, _dk: u32, _dv: u32, _scale: f32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_chunk_loop_tc_step`] where a native backend
/// exists at all; `None` on wasm.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_tc(g: &Gpu, bufs: &[&DeviceBuffer; 10], bh: u32, c: u32, dk: u32, dv: u32, n_chunks: u32, scale: f32, heads_tok: u32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_loop_tc_step(g, bufs, bh, c, dk, dv, n_chunks, scale, heads_tok)
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_tc(_g: &Gpu, _bufs: &[&DeviceBuffer; 10], _bh: u32, _c: u32, _dk: u32, _dv: u32, _n_chunks: u32, _scale: f32, _heads_tok: u32) -> Option<Step> {
    None
}

/// [`gpu_core::provider::cuda::gdn_chunk_loop_train_tc_step`] where a native
/// backend exists at all; `None` on wasm.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_train_tc(g: &Gpu, bufs: &[&DeviceBuffer; 12], bh: u32, c: u32, dk: u32, dv: u32, n_chunks: u32, scale: f32, heads_tok: u32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_loop_train_tc_step(g, bufs, bh, c, dk, dv, n_chunks, scale, heads_tok)
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_train_tc(_g: &Gpu, _bufs: &[&DeviceBuffer; 12], _bh: u32, _c: u32, _dk: u32, _dv: u32, _n_chunks: u32, _scale: f32, _heads_tok: u32) -> Option<Step> {
    None
}

/// Whether the caller opted in to the tensor-core kernels (the fused prefix and the
/// recurrence) and this device takes the fast kernels at all.
pub(crate) fn tensor_core_enabled(g: &Gpu, ids: &GdnIds) -> bool {
    ids.fast.is_some_and(|f| f.tensor_core) && use_fast_kernels(g)
}

/// Whether [`gdn_chunk_bwd`] can be expected to run as the tensor-core sweep and post kernels for this
/// `shape` (a `true` is not a promise: the backend may still decline the launch, and the portable
/// backward then runs).
pub fn tensor_core_backward_offered(g: &Gpu, ids: &GdnIds, shape: &GdnShape) -> bool {
    tensor_core_enabled(g, ids) && shape.dk == 128 && shape.dv == 128 && shape.chunk != 0 && shape.chunk <= 64
}

/// Largest chunk `gdn_ut_fwd.wgsl` serves: one thread per column and a packed
/// triangle sized for 64 rows (its header derives both).
const GDN_UT_FWD_MAX_CHUNK: u32 = 64;

/// One `bmm.wgsl`/`bmm_acc.wgsl` dispatch. Public because a batched matmul
/// with offset-addressable batch slices is a general primitive, not a
/// GDN-only detail - a future caller assembling its own batched-matmul step
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

/// [`bmm_step`] for the FORWARD chunk path: the tiled kernel when `ids` carries
/// one and the device takes it ([`use_fast_kernels`]), else the one-thread-per-
/// output pair. `kernel` is `ids.bmm` or `ids.bmm_acc`; the tiled kernel folds
/// the two behind a flag and reproduces either exactly.
#[allow(clippy::too_many_arguments)]
fn bmm_auto(
    g: &Gpu,
    ids: &GdnIds,
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
    let Some(fast) = ids.fast.filter(|_| use_fast_kernels(g)) else {
        return bmm_step(g, kernel, batch, m, k, n, trans_a, trans_b, alpha, a, a_off, b, b_off, out, out_off);
    };
    let acc = kernel == ids.bmm_acc;
    let tiles = batch * m.div_ceil(BMM_TILED_TILE) * n.div_ceil(BMM_TILED_TILE);
    g.dispatch(
        fast.bmm_tiled,
        &[a, b, out],
        &[batch, m, k, n, trans_a as u32, trans_b as u32, f(alpha), a_off, b_off, out_off, acc as u32],
        gpu_core::Dispatch::Workgroups(tiles),
    )
}

/// Rows and columns one `bmm_tiled.wgsl` workgroup covers.
const BMM_TILED_TILE: u32 = 64;

/// The 13 whole-tensor (chunk-independent-dispatch) scratch buffers steps 1-9
/// and the pre-loop `intra_scores` precompute need - the fields [`GdnScratch`]
/// and [`GdnScratchTrain`] have IN COMMON. Not part of either struct's public
/// API (both expose these same 13 buffers under the same names at their own
/// top level, matching this module's existing flat-field style rather than
/// nesting); this is purely an internal parameter bundle so
/// [`gdn_chunk_fwd_prefix`] can be written once and called from both
/// [`gdn_chunk_fwd`] and [`gdn_chunk_fwd_train`] - see that function's doc.
struct GdnWholeScratch<'a> {
    g_cs: &'a DeviceBuffer,
    exp_g_cs: &'a DeviceBuffer,
    k_beta: &'a DeviceBuffer,
    v_beta: &'a DeviceBuffer,
    k_beta_decay: &'a DeviceBuffer,
    decay_mask: &'a DeviceBuffer,
    raw_attn0: &'a DeviceBuffer,
    attn0: &'a DeviceBuffer,
    t_mat: &'a DeviceBuffer,
    u: &'a DeviceBuffer,
    w: &'a DeviceBuffer,
    raw_intra: &'a DeviceBuffer,
    intra_scores: &'a DeviceBuffer,
}

/// Steps 1-9 of this module's doc, plus the pre-loop `intra_scores`
/// precompute - every WHOLE-TENSOR (state-independent) step, shared
/// byte-for-byte between [`gdn_chunk_fwd`] (inference) and
/// [`gdn_chunk_fwd_train`] (training, which additionally saves what
/// [`gdn_chunk_bwd`] needs). Extracted so the two forward entry points cannot
/// silently drift apart on this shared half - only step 10's per-chunk loop
/// differs between them (overwrite-in-place vs. save-every-chunk).
#[allow(clippy::too_many_arguments)]
fn gdn_chunk_fwd_prefix(
    g: &Gpu,
    ids: &GdnIds,
    shape: &GdnShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    raw_g: &DeviceBuffer,
    beta: &DeviceBuffer,
    s: &GdnWholeScratch,
) -> Vec<Step> {
    let (dk, dv, c) = (shape.dk, shape.dv, shape.chunk);
    let bhc = shape.bhc();
    let scale = 1.0f32 / (dk as f32).sqrt();

    let bmm = |kernel: usize, batch: u32, m: u32, k: u32, n: u32, ta: bool, tb: bool, alpha: f32, a: &DeviceBuffer, a_off: u32, b: &DeviceBuffer, b_off: u32, o: &DeviceBuffer, o_off: u32| {
        bmm_auto(g, ids, kernel, batch, m, k, n, ta, tb, alpha, a, a_off, b, b_off, o, o_off)
    };

    // ---- steps 1-2: v_beta = value*beta, k_beta = key*beta (whole tensor) ----
    // ---- step 4: g_cs = copy(raw_g), then the per-chunk cumsum (one fused dispatch) ----
    // ---- step 5: decay_mask ----
    // ---- step 6: attn0 = -(k_beta @ key^T), strictly-lower masked by decay_mask ----
    let mut steps = vec![
        g.step(ids.row_scale, &[value, beta, s.v_beta], &[bhc * c * dv, dv], bhc * c * dv),
        g.step(ids.row_scale, &[key, beta, s.k_beta], &[bhc * c * dk, dk], bhc * c * dk),
        g.step(ids.region_copy, &[raw_g, s.g_cs], &[1, bhc * c, bhc * c, 0], bhc * c),
        g.step(ids.cumsum_step, &[s.g_cs], &[bhc, c], bhc),
        g.step(ids.decay_mask, &[s.g_cs, s.decay_mask], &[bhc, c], bhc * c * c),
        bmm(ids.bmm, bhc, c, dk, c, false, true, -1.0, s.k_beta, 0, key, 0, s.raw_attn0, 0),
        g.step(ids.mask_strict_lower, &[s.raw_attn0, s.decay_mask, s.attn0], &[bhc, c], bhc * c * c),
    ];

    // ---- step 7: UT-transform, T_mat = (I - attn0)^-1, by forward substitution ----
    //
    // One `gdn_ut_step.wgsl` dispatch per row index (that kernel's own header
    // derives the per-element recurrence and explains why `attn0` and the
    // evolving `t_mat` must be physically separate buffers), then `+= I`
    // (`gdn_add_identity.wgsl`) once the whole strictly-lower part is
    // finalised. `c = 1` needs no substitution at all: `attn0` is then a 1x1
    // zero matrix and T_mat is plain `I`, which the `+= I` alone produces.
    //
    // **Forward substitution specifically, and not a Neumann series.**
    // `attn0` is strictly lower triangular and therefore nilpotent at `c`
    // (`attn0^c = 0`), which makes `T_mat = sum_{k=0}^{c-1} attn0^k =
    // prod_{m} (I + attn0^(2^m))` an exact identity, and evaluating it by
    // repeated squaring is far cheaper in dispatches (17 vs. `c` at
    // `c = 64`). It is also numerically unusable at real weights, which is
    // why this is the loop and not that product:
    //
    // `attn0[i,j] = -beta_i * (k_i.k_j) * exp(g_cs_i - g_cs_j)`. The keys are
    // L2-normalized by the caller and `beta` is a sigmoid, so every entry is
    // bounded by 1 - but how close it gets to 1 is entirely a property of the
    // CONTENT. Independent keys in `Dk` dimensions are near-orthogonal
    // (`k_i.k_j ~ 1/sqrt(Dk)`), leaving `|attn0| << 1`, where the powers decay
    // and both evaluations agree to fp32 round-off. Repetitive text drives
    // `k_i.k_j` towards 1 for most pairs, and at `|attn0| -> 1` the
    // intermediate powers behave like the strictly-lower all-ones matrix,
    // whose entries are binomial coefficients: `attn0^(c/2)` reaches
    // `~C(c-2, c/2-1) ~ 1e17` at `c = 64`, while the alternating sum they
    // telescope into stays `O(1)`. Recovering an `O(1)` result from `1e17`
    // intermediates needs ~24 significant digits; fp32 has ~7, so the
    // cancellation is total and `T_mat` comes back with `1e10` entries. (It
    // is not an fp32-only problem - f64 fails the same way, just further
    // along the correlation axis; the instability is in the reformulation,
    // not in the precision it is evaluated at.) Forward substitution never
    // forms a power of `attn0` at all: every intermediate it holds is already
    // an entry of the bounded answer, so it is accurate to ~1e-7 across the
    // whole range. `crates/model/tests/gdn_chunk_fwd.rs`'s
    // `gdn_chunk_fwd_matches_host_oracle_with_near_parallel_keys` pins this.
    match ids.fast.filter(|_| c <= GDN_UT_FWD_MAX_CHUNK && use_fast_kernels(g)) {
        // One workgroup per matrix; the kernel writes the whole `t_mat`
        // (identity included), so no `add_identity` follows.
        Some(fast) => steps.push(g.dispatch(fast.ut_fwd, &[s.attn0, s.t_mat], &[bhc, c], gpu_core::Dispatch::Workgroups(bhc))),
        None => {
            for i in 1..c {
                steps.push(g.step(ids.ut_step, &[s.attn0, s.t_mat], &[bhc, c, i], bhc * i));
            }
            steps.push(g.step(ids.add_identity, &[s.t_mat], &[bhc, c], bhc * c));
        }
    }

    // ---- step 8: u = T_mat @ v_beta ----
    steps.push(bmm(ids.bmm, bhc, c, c, dv, false, false, 1.0, s.t_mat, 0, s.v_beta, 0, s.u, 0));

    // ---- step 9: w = T_mat @ (k_beta * exp(g_cs)) ----
    steps.push(g.step(ids.exp, &[s.g_cs, s.exp_g_cs], &[bhc * c], bhc * c));
    steps.push(g.step(ids.row_scale, &[s.k_beta, s.exp_g_cs, s.k_beta_decay], &[bhc * c * dk, dk], bhc * c * dk));
    steps.push(bmm(ids.bmm, bhc, c, c, dk, false, false, 1.0, s.t_mat, 0, s.k_beta_decay, 0, s.w, 0));

    // ---- intra_scores, precomputed for every chunk (state-independent) ----
    steps.push(bmm(ids.bmm, bhc, c, dk, c, false, true, scale, query, 0, key, 0, s.raw_intra, 0));
    steps.push(g.step(ids.mul, &[s.raw_intra, s.decay_mask, s.intra_scores], &[bhc * c * c], bhc * c * c));

    steps
}

/// The full Gated DeltaNet chunked-parallel forward - steps 1-11 of this
/// module's doc - as a step list. `initial_state`/`final_state` are
/// `[B,H,Dk,Dv]` (see module doc); `final_state` is both the OUTPUT recurrent
/// state (for the caller to persist / feed a later incremental call) and,
/// internally, the loop's own working buffer (initialised as a copy of
/// `initial_state` by this function's first two steps, then updated in
/// place per chunk) - pass a zeroed buffer as `initial_state` for "no prior
/// state".
///
/// **Caller MUST zero `scratch.t_mat` at submit time**: this function
/// returns a plain `Vec<Step>` (no clear primitive exists at the `Step`
/// level in this engine - only `Backend::submit`'s own `clears` list does),
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
    let scale = 1.0f32 / (dk as f32).sqrt();

    let bmm = |kernel: usize, batch: u32, m: u32, k: u32, n: u32, ta: bool, tb: bool, alpha: f32, a: &DeviceBuffer, a_off: u32, b: &DeviceBuffer, b_off: u32, o: &DeviceBuffer, o_off: u32| {
        bmm_auto(g, ids, kernel, batch, m, k, n, ta, tb, alpha, a, a_off, b, b_off, o, o_off)
    };

    let whole = GdnWholeScratch {
        g_cs: scratch.g_cs,
        exp_g_cs: scratch.exp_g_cs,
        k_beta: scratch.k_beta,
        v_beta: scratch.v_beta,
        k_beta_decay: scratch.k_beta_decay,
        decay_mask: scratch.decay_mask,
        raw_attn0: scratch.raw_attn0,
        attn0: scratch.attn0,
        t_mat: scratch.t_mat,
        u: scratch.u,
        w: scratch.w,
        raw_intra: scratch.raw_intra,
        intra_scores: scratch.intra_scores,
    };
    // Steps 1-9 and the intra-chunk scores as one tensor-core launch where the caller and the
    // device offer it; the portable whole-tensor sequence otherwise.
    let fused = if tensor_core_enabled(g, ids) {
        let bufs = [query, key, value, raw_g, beta, scratch.g_cs, scratch.exp_g_cs, scratch.u, scratch.w, scratch.intra_scores];
        native_chunk_prep(g, &bufs, shape.bhc(), c, dk, dv, scale)
    } else {
        None
    };
    let mut steps = match fused {
        Some(step) => vec![step],
        None => gdn_chunk_fwd_prefix(g, ids, shape, query, key, value, raw_g, beta, &whole),
    };

    // ---- step 10 in one launch, where the device has a kernel for it ----
    // The kernel reads `initial_state` and writes `final_state` itself, so the
    // state copy and the whole chunk loop below are what it replaces.
    if ids.fast.is_some() && use_fast_kernels(g) {
        let bufs = [key, query, scratch.w, scratch.u, scratch.g_cs, scratch.exp_g_cs, scratch.intra_scores, initial_state, out, final_state];
        let tc = if tensor_core_enabled(g, ids) { native_chunk_loop_tc(g, &bufs, bh, c, dk, dv, n_chunks, scale, 0) } else { None };
        if let Some(step) = tc.or_else(|| native_chunk_loop(g, &bufs, bh, c, dk, dv, n_chunks, scale)) {
            steps.push(step);
            return steps;
        }
    }

    // ---- step 10: sequential across-chunk loop ----
    // `final_state` is the loop's own working buffer, seeded from `initial_state`.
    steps.push(g.step(ids.region_copy, &[initial_state, final_state], &[1, bh * dk * dv, bh * dk * dv, 0], bh * dk * dv));

    for ci in 0..n_chunks {
        // Chunk `ci`'s flat element offset into a `[n_chunks, bh, C, D]`
        // (or `[n_chunks, bh, C]`/`[n_chunks, bh, C, C]`) chunk-major buffer
        // - see this module's "Layout" doc for why this is a plain offset.
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

/// Kernel indices [`gdn_chunk_fwd_train`] and [`gdn_chunk_bwd`] dispatch,
/// beyond [`GdnIds`]. Kept as a SEPARATE struct rather than new fields on
/// `GdnIds` so that struct's existing field set (and
/// `crates/model/tests/gdn_chunk_fwd.rs`'s own `GdnIds { ... }` literal)
/// keeps compiling unmodified - the same "forward ids vs backward ids" split
/// [`crate::moe`]'s `MoeIds`/`MoeIdsBwd` already established.
#[derive(Clone, Copy)]
pub struct GdnBwdIds {
    /// `splice_add.wgsl` (existing, unmodified) - `dst[base+i] += src[i]`,
    /// the commit primitive both [`gdn_chunk_fwd_train`]'s saves and
    /// [`gdn_chunk_bwd`]'s per-chunk gradient writes use to place a densely
    /// computed small result into its own slice of a bigger
    /// `[bhc,...]`/history buffer.
    pub splice_add: usize,
    /// `row_dot.wgsl` - generic per-row dot product, every row-scale
    /// gradient (`d_exp_g_cs`, `d_decay_scale`, `d_beta`).
    pub row_dot: usize,
    /// `scale_add.wgsl` (existing, unmodified) - row-scale with an
    /// overwrite/accumulate flag (`n_experts=1,e_idx=0` degenerates it to a
    /// plain per-row scale), reused for `d_key`/`d_k_beta`/`d_value`'s
    /// whole-tensor row-scale-shaped contributions.
    pub scale_add: usize,
    /// `gdn_chunk_reverse_cumsum_step.wgsl`.
    pub reverse_cumsum_step: usize,
    /// `gdn_ut_bwd_dattn0.wgsl` - superseded on [`gdn_chunk_bwd`]'s own call
    /// path by M5.11's closed-form `bmm`-pair (`dL/dattn0 = T_mat^T @
    /// d_t_mat @ T_mat^T`); kept registered/tested, same precedent as
    /// [`GdnIds::ut_step`] after M5.9.
    pub ut_bwd_dattn0: usize,
    /// `gdn_ut_bwd_dtmat.wgsl` - same M5.11 status as
    /// [`GdnBwdIds::ut_bwd_dattn0`].
    pub ut_bwd_dtmat: usize,
    /// `gdn_mask_strict_lower_bwd.wgsl`.
    pub mask_strict_lower_bwd: usize,
    /// `gdn_decay_mask_bwd.wgsl` (dispatched twice - `mode=0` then `mode=1`).
    pub decay_mask_bwd: usize,
    /// `gdn_decay_scale_bwd.wgsl`.
    pub decay_scale_bwd: usize,
    /// `gdn_decay_scale_bwd_last.wgsl`.
    pub decay_scale_bwd_last: usize,
    /// `gdn_state_decay_bwd_dscale.wgsl`.
    pub state_decay_bwd_dscale: usize,
}

/// The training-mode sibling of [`gdn_chunk_fwd`]: byte-identical forward
/// math (same steps 1-9 via the shared [`gdn_chunk_fwd_prefix`], same
/// per-chunk step-10 kernel calls, so this function's `out`/`final_state`
/// outputs are IDENTICAL to what [`gdn_chunk_fwd`] would produce for the same
/// inputs - a cheap cross-check worth running before trusting a backward
/// gradcheck), but the per-chunk loop additionally SAVES what
/// [`gdn_chunk_bwd`]'s reverse sweep needs: [`gdn_chunk_fwd`]'s own loop
/// overwrites its five small per-chunk buffers every iteration and evolves
/// ONE `final_state` in place, so backward - which runs the loop in REVERSE -
/// needs every chunk's OWN version, not just the latest. Same precedent as
/// `router_gate.wgsl` vs `router_gate_train.wgsl` (same forward math, the
/// training variant additionally persists what backward needs).
///
/// Every working buffer (`scratch.q_scaled` and its four siblings) is
/// computed EXACTLY as [`gdn_chunk_fwd`] computes it (same kernel, same
/// params) - the only addition is one extra `splice_add.wgsl` dispatch per
/// chunk, committing `decay_scale` into its own slice of `decay_scale_hist`
/// (the other four are recomputed by [`gdn_chunk_bwd`]). The
/// recurrent state gets the same treatment: `final_state` evolves in place
/// exactly as in `gdn_chunk_fwd`, and after each chunk's update a
/// `splice_add` additionally snapshots it into `state_history[ci+1]`;
/// `state_history[0]` is committed from `initial_state` before the loop
/// starts.
///
/// **Caller MUST zero [`GdnScratchTrain`]'s `decay_scale_hist`,
/// `state_history`, and `t_mat`** - see [`GdnScratchTrain`]'s own doc.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_fwd_train(
    g: &Gpu,
    ids: &GdnIds,
    bwd_ids: &GdnBwdIds,
    shape: &GdnShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    raw_g: &DeviceBuffer,
    beta: &DeviceBuffer,
    initial_state: &DeviceBuffer,
    scratch: &GdnScratchTrain,
    out: &DeviceBuffer,
    final_state: &DeviceBuffer,
) -> Vec<Step> {
    let (dk, dv, c) = (shape.dk, shape.dv, shape.chunk);
    let scale = 1.0f32 / (dk as f32).sqrt();

    let whole = GdnWholeScratch {
        g_cs: scratch.g_cs,
        exp_g_cs: scratch.exp_g_cs,
        k_beta: scratch.k_beta,
        v_beta: scratch.v_beta,
        k_beta_decay: scratch.k_beta_decay,
        decay_mask: scratch.decay_mask,
        raw_attn0: scratch.raw_attn0,
        attn0: scratch.attn0,
        t_mat: scratch.t_mat,
        u: scratch.u,
        w: scratch.w,
        raw_intra: scratch.raw_intra,
        intra_scores: scratch.intra_scores,
    };
    // Steps 1-9 and the intra-chunk scores as one tensor-core launch (it also writes everything
    // the backward reads) where the caller and the device offer it; the portable sequence
    // otherwise. The fused path leaves `attn0` unwritten: nothing downstream reads it.
    let fused = if tensor_core_enabled(g, ids) {
        let bufs = [
            query,
            key,
            value,
            raw_g,
            beta,
            scratch.g_cs,
            scratch.exp_g_cs,
            scratch.u,
            scratch.w,
            scratch.intra_scores,
            scratch.k_beta,
            scratch.v_beta,
            scratch.k_beta_decay,
            scratch.decay_mask,
            scratch.raw_attn0,
            scratch.t_mat,
            scratch.raw_intra,
        ];
        native_chunk_prep_train(g, &bufs, shape.bhc(), c, dk, dv, scale)
    } else {
        None
    };
    let steps = match fused {
        Some(step) => vec![step],
        None => gdn_chunk_fwd_prefix(g, ids, shape, query, key, value, raw_g, beta, &whole),
    };

    gdn_chunk_loop_train(g, ids, bwd_ids, shape, query, key, initial_state, scratch, out, final_state, steps)
}

/// Step 10 of [`gdn_chunk_fwd_train`]: the across-chunk recurrence with the state and decay-scale
/// histories saved, as one native launch where the device offers it (the tensor-core kernel when the
/// caller opted in, else the fp32 one) and as the per-chunk dispatch walk otherwise. `steps` holds the
/// prefix already built; the finished list is returned. Shared with [`gdn_chunk_fwd_train_tokens`].
#[allow(clippy::too_many_arguments)]
fn gdn_chunk_loop_train(
    g: &Gpu,
    ids: &GdnIds,
    bwd_ids: &GdnBwdIds,
    shape: &GdnShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    initial_state: &DeviceBuffer,
    scratch: &GdnScratchTrain,
    out: &DeviceBuffer,
    final_state: &DeviceBuffer,
    mut steps: Vec<Step>,
) -> Vec<Step> {
    let (dk, dv, c) = (shape.dk, shape.dv, shape.chunk);
    let n_chunks = shape.n_chunks();
    let bh = shape.bh();
    let scale = 1.0f32 / (dk as f32).sqrt();

    let bmm = |kernel: usize, batch: u32, m: u32, k: u32, n: u32, ta: bool, tb: bool, alpha: f32, a: &DeviceBuffer, a_off: u32, b: &DeviceBuffer, b_off: u32, o: &DeviceBuffer, o_off: u32| {
        bmm_auto(g, ids, kernel, batch, m, k, n, ta, tb, alpha, a, a_off, b, b_off, o, o_off)
    };
    let splice = |src: &DeviceBuffer, dst: &DeviceBuffer, base: u32, n: u32| g.step(bwd_ids.splice_add, &[src, dst], &[n, base], n);

    // ---- the whole recurrence in one launch, where the device has a kernel for it ----
    // It reads `initial_state`, writes `final_state` and records both histories
    // itself: the state copy, the snapshots and the eleven dispatches per chunk
    // below are what it replaces.
    if ids.fast.is_some() && use_fast_kernels(g) {
        let bufs = [
            key,
            query,
            scratch.w,
            scratch.u,
            scratch.g_cs,
            scratch.exp_g_cs,
            scratch.intra_scores,
            initial_state,
            out,
            final_state,
            scratch.state_history,
            scratch.decay_scale_hist,
        ];
        let tc = if tensor_core_enabled(g, ids) { native_chunk_loop_train_tc(g, &bufs, bh, c, dk, dv, n_chunks, scale, 0) } else { None };
        if let Some(step) = tc.or_else(|| native_chunk_loop_train(g, &bufs, bh, c, dk, dv, n_chunks, scale)) {
            steps.push(step);
            return steps;
        }
    }

    // ---- step 10: sequential across-chunk loop, saving every chunk's version ----
    steps.push(g.step(ids.region_copy, &[initial_state, final_state], &[1, bh * dk * dv, bh * dk * dv, 0], bh * dk * dv));
    steps.push(splice(initial_state, scratch.state_history, 0, bh * dk * dv));

    for ci in 0..n_chunks {
        let off_d = |d: u32| ci * bh * c * d;
        let off_g = ci * bh * c;
        let off_cc = ci * bh * c * c;

        steps.push(bmm(ids.bmm, bh, c, dk, dv, false, false, 1.0, scratch.w, off_d(dk), final_state, 0, scratch.v_prime, 0));
        steps.push(g.step(ids.sub, &[scratch.u, scratch.v_prime, scratch.v_new], &[bh * c * dv, off_d(dv), 0], bh * c * dv));
        steps.push(g.step(ids.decay_scale, &[scratch.g_cs, scratch.decay_scale], &[bh, c, off_g], bh * c));
        steps.push(g.step(ids.row_scale_off, &[key, scratch.decay_scale, scratch.decayed_k], &[bh * c * dk, dk, off_d(dk), 0, f(1.0)], bh * c * dk));
        steps.push(g.step(ids.row_scale_off, &[query, scratch.exp_g_cs, scratch.q_scaled], &[bh * c * dk, dk, off_d(dk), off_g, f(scale)], bh * c * dk));
        steps.push(bmm(ids.bmm, bh, c, dk, dv, false, false, 1.0, scratch.q_scaled, 0, final_state, 0, out, off_d(dv)));
        steps.push(bmm(ids.bmm_acc, bh, c, c, dv, false, false, 1.0, scratch.intra_scores, off_cc, scratch.v_new, 0, out, off_d(dv)));

        // ---- save this chunk's working-buffer values into their history slice ----
        steps.push(splice(scratch.decay_scale, scratch.decay_scale_hist, off_g, bh * c));

        steps.push(g.step(ids.state_decay, &[scratch.g_cs, final_state], &[bh, dk, dv, c, off_g], bh * dk * dv));
        steps.push(bmm(ids.bmm_acc, bh, dk, c, dv, true, false, 1.0, scratch.decayed_k, 0, scratch.v_new, 0, final_state, 0));

        // ---- save the state AFTER this chunk's decay+update ----
        steps.push(splice(final_state, scratch.state_history, (ci + 1) * bh * dk * dv, bh * dk * dv));
    }

    steps
}

/// The mixer's token-major operands of one chunked training forward, for
/// [`gdn_chunk_fwd_train_tokens`]. `query` and `key` are the raw (pre-normalisation) narrow
/// `[tokens, heads / group, dk]` projections; `value` is `[tokens, heads, dv]`; `raw_g` and `beta`
/// are `[tokens, heads]`.
pub struct GdnTokenInputs<'a> {
    pub query: &'a DeviceBuffer,
    pub key: &'a DeviceBuffer,
    pub value: &'a DeviceBuffer,
    pub raw_g: &'a DeviceBuffer,
    pub beta: &'a DeviceBuffer,
    /// Value heads per key head (the repeat factor the mixer would otherwise apply).
    pub group: u32,
    /// Epsilon of the query/key L2 normalisation the mixer would otherwise apply.
    pub l2_eps: f32,
}

/// The chunk-major tensors [`gdn_chunk_fwd_train_tokens`] produces for the backward (and the recurrence):
/// the normalised, head-repeated query and key, the value and beta.
pub struct GdnChunkMajorOut<'a> {
    pub query_cm: &'a DeviceBuffer,
    pub key_cm: &'a DeviceBuffer,
    pub value_cm: &'a DeviceBuffer,
    pub beta_cm: &'a DeviceBuffer,
}

/// Whether [`gdn_chunk_fwd_train_tokens`] can be expected to serve a call of this `shape` and `group`,
/// so a caller can skip allocating its chunk-major outputs when it cannot. A `true` is not a promise
/// (the backend may still decline the kernel); the function's own `None` is the authority.
pub fn tokens_prefix_offered(g: &Gpu, ids: &GdnIds, shape: &GdnShape, group: u32) -> bool {
    tensor_core_enabled(g, ids) && shape.dk == 128 && shape.dv == 128 && shape.chunk <= 64 && group != 0 && shape.h.is_multiple_of(group)
}

/// [`gdn_chunk_fwd_train`] fed with the mixer's TOKEN-MAJOR tensors: the query/key L2 normalisation, the
/// head repeat and the token-major to chunk-major permutes of all five operands happen inside the
/// tensor-core prefix kernel's loads (see `gdn_chunk_prep_tc.cu`), and `out_chunk_major.*` receive the
/// chunk-major tensors [`gdn_chunk_bwd`] reads. Everything after the prefix is [`gdn_chunk_fwd_train`]'s.
///
/// Returns `None` when this device or this call is not served - the caller did not opt in
/// ([`GdnFastIds::tensor_core`]), the device has no tensor cores, the head widths are not 128, the
/// chunk is longer than 64 rows, or the backend declined the kernel - and the caller then runs the
/// normalisation, repeat, permutes and [`gdn_chunk_fwd_train`] it always ran. Callers must not have
/// touched any output buffer in that case: nothing is written until the returned steps are submitted.
///
/// With `out_token_major` the tensor-core recurrence writes `out` in the token-major layout `[tokens, heads,
/// dv]` (saving the caller its chunk-major to token-major permute); the returned flag says whether it did, and
/// when it is false `out` is chunk-major as always.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_fwd_train_tokens(
    g: &Gpu,
    ids: &GdnIds,
    bwd_ids: &GdnBwdIds,
    shape: &GdnShape,
    tok: &GdnTokenInputs,
    cm: &GdnChunkMajorOut,
    initial_state: &DeviceBuffer,
    scratch: &GdnScratchTrain,
    out: &DeviceBuffer,
    final_state: &DeviceBuffer,
    out_token_major: bool,
) -> Option<(Vec<Step>, bool)> {
    if !tensor_core_enabled(g, ids) || shape.dk != shape.dv {
        return None;
    }
    let layout = tok_layout(shape, tok);
    let bufs = [
        tok.query,
        tok.key,
        tok.value,
        tok.raw_g,
        tok.beta,
        scratch.g_cs,
        scratch.exp_g_cs,
        scratch.u,
        scratch.w,
        scratch.intra_scores,
        scratch.k_beta,
        scratch.v_beta,
        scratch.k_beta_decay,
        scratch.decay_mask,
        scratch.raw_attn0,
        scratch.t_mat,
        scratch.raw_intra,
        cm.query_cm,
        cm.key_cm,
        cm.value_cm,
        cm.beta_cm,
    ];
    let prep = native_chunk_prep_tok_train(g, &bufs, layout)?;
    // The recurrence's output rows in the mixer's token-major layout straight from the tensor-core loop, when
    // asked for and served; otherwise the chunk-major `out` the portable loops write.
    if out_token_major {
        let (bh, c, n_chunks) = (shape.bh(), shape.chunk, shape.n_chunks());
        let loop_bufs = [
            cm.key_cm,
            cm.query_cm,
            scratch.w,
            scratch.u,
            scratch.g_cs,
            scratch.exp_g_cs,
            scratch.intra_scores,
            initial_state,
            out,
            final_state,
            scratch.state_history,
            scratch.decay_scale_hist,
        ];
        if let Some(recurrence) = native_chunk_loop_train_tc(g, &loop_bufs, bh, c, shape.dk, shape.dv, n_chunks, 1.0f32 / (shape.dk as f32).sqrt(), shape.h) {
            return Some((vec![prep, recurrence], true));
        }
    }
    Some((gdn_chunk_loop_train(g, ids, bwd_ids, shape, cm.query_cm, cm.key_cm, initial_state, scratch, out, final_state, vec![prep]), false))
}

/// [`gdn_chunk_fwd`] fed with the mixer's token-major tensors, the inference sibling of
/// [`gdn_chunk_fwd_train_tokens`]: the same folded loads, and only `query_cm` and `key_cm` (what the
/// recurrence reads) are written. `None` under the same conditions, with the same meaning.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_fwd_tokens(
    g: &Gpu,
    ids: &GdnIds,
    shape: &GdnShape,
    tok: &GdnTokenInputs,
    query_cm: &DeviceBuffer,
    key_cm: &DeviceBuffer,
    initial_state: &DeviceBuffer,
    scratch: &GdnTokScratch,
    out: &DeviceBuffer,
    final_state: &DeviceBuffer,
    out_token_major: bool,
) -> Option<(Vec<Step>, bool)> {
    if !tensor_core_enabled(g, ids) || shape.dk != shape.dv {
        return None;
    }
    let bufs = [
        tok.query,
        tok.key,
        tok.value,
        tok.raw_g,
        tok.beta,
        scratch.g_cs,
        scratch.exp_g_cs,
        scratch.u,
        scratch.w,
        scratch.intra_scores,
        query_cm,
        key_cm,
    ];
    let (dk, dv, c) = (shape.dk, shape.dv, shape.chunk);
    let (bh, n_chunks) = (shape.bh(), shape.n_chunks());
    let scale = 1.0f32 / (dk as f32).sqrt();
    let loop_bufs = [key_cm, query_cm, scratch.w, scratch.u, scratch.g_cs, scratch.exp_g_cs, scratch.intra_scores, initial_state, out, final_state];
    // The bf16 form of the prefill (prep writes the loop's operands as bf16, the loop multiplies them on bf16 tensor
    // cores with an fp32 state) where the device's measured selection chose it; the exact TF32 form otherwise.
    if out_token_major && bf16_prefill_selected(g) {
        if let (Some(prep), Some(recurrence)) = (native_chunk_prep_tok_bf16(g, &bufs, tok_layout(shape, tok)), native_chunk_loop_bf16(g, &loop_bufs, bh, c, dk, dv, n_chunks, scale, shape.h)) {
            return Some((vec![prep, recurrence], true));
        }
    }
    let prep = native_chunk_prep_tok(g, &bufs, tok_layout(shape, tok))?;
    // The recurrence: one tensor-core launch, else the fp32 one. A device that took the prefix above
    // takes at least the fp32 loop (same compute-capability floor family); if neither is offered the
    // whole call declines so the caller runs its portable sequence from the start.
    if out_token_major {
        if let Some(recurrence) = native_chunk_loop_tc(g, &loop_bufs, bh, c, dk, dv, n_chunks, scale, shape.h) {
            return Some((vec![prep, recurrence], true));
        }
    }
    let recurrence = native_chunk_loop_tc(g, &loop_bufs, bh, c, dk, dv, n_chunks, scale, 0).or_else(|| native_chunk_loop(g, &loop_bufs, bh, c, dk, dv, n_chunks, scale))?;
    Some((vec![prep, recurrence], false))
}

#[cfg(not(target_arch = "wasm32"))]
type TokLayout = gpu_core::provider::cuda::GdnTokenLayout;

#[cfg(target_arch = "wasm32")]
type TokLayout = ();

#[cfg(not(target_arch = "wasm32"))]
fn tok_layout(shape: &GdnShape, tok: &GdnTokenInputs) -> TokLayout {
    gpu_core::provider::cuda::GdnTokenLayout {
        bhc: shape.bhc(),
        c_len: shape.chunk,
        n_chunks: shape.n_chunks(),
        heads: shape.h,
        group: tok.group,
        head_dim: shape.dk,
        scale: 1.0f32 / (shape.dk as f32).sqrt(),
        eps: tok.l2_eps,
    }
}

#[cfg(target_arch = "wasm32")]
fn tok_layout(_shape: &GdnShape, _tok: &GdnTokenInputs) -> TokLayout {}

/// Whether the measured selection of this device chose the bf16 prefill form ([`gpu_core::provider::cuda::gdn_bf16_prefill_selected`]).
#[cfg(not(target_arch = "wasm32"))]
fn bf16_prefill_selected(g: &Gpu) -> bool {
    gpu_core::provider::cuda::gdn_bf16_prefill_selected(g)
}

#[cfg(target_arch = "wasm32")]
fn bf16_prefill_selected(_g: &Gpu) -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
fn native_chunk_prep_tok_bf16(g: &Gpu, bufs: &[&DeviceBuffer; 12], layout: TokLayout) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_prep_tok_bf16_step(g, bufs, layout)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_bf16(g: &Gpu, bufs: &[&DeviceBuffer; 10], bh: u32, c: u32, dk: u32, dv: u32, n_chunks: u32, scale: f32, heads_tok: u32) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_loop_bf16_step(g, bufs, bh, c, dk, dv, n_chunks, scale, heads_tok)
}

#[cfg(target_arch = "wasm32")]
fn native_chunk_prep_tok_bf16(_g: &Gpu, _bufs: &[&DeviceBuffer; 12], _layout: TokLayout) -> Option<Step> {
    None
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn native_chunk_loop_bf16(_g: &Gpu, _bufs: &[&DeviceBuffer; 10], _bh: u32, _c: u32, _dk: u32, _dv: u32, _n_chunks: u32, _scale: f32, _heads_tok: u32) -> Option<Step> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn native_chunk_prep_tok_train(g: &Gpu, bufs: &[&DeviceBuffer; 21], layout: TokLayout) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_prep_tok_train_step(g, bufs, layout)
}

#[cfg(not(target_arch = "wasm32"))]
fn native_chunk_prep_tok(g: &Gpu, bufs: &[&DeviceBuffer; 12], layout: TokLayout) -> Option<Step> {
    gpu_core::provider::cuda::gdn_chunk_prep_tok_step(g, bufs, layout)
}

#[cfg(target_arch = "wasm32")]
fn native_chunk_prep_tok(_g: &Gpu, _bufs: &[&DeviceBuffer; 12], _layout: TokLayout) -> Option<Step> {
    None
}

#[cfg(target_arch = "wasm32")]
fn native_chunk_prep_tok_train(_g: &Gpu, _bufs: &[&DeviceBuffer; 21], _layout: TokLayout) -> Option<Step> {
    None
}

/// Every scratch buffer [`gdn_chunk_bwd`] needs, beyond the forward-saved
/// [`GdnScratchTrain`] it reads from. `bhc = shape.bhc()`, `bh = shape.bh()`,
/// `c = shape.chunk`.
///
/// **The caller MUST zero** `d_g_cs`, `d_exp_g_cs`, `d_u`, and `d_decay_mask`
/// (pass them in [`Gpu::submit`]'s `clears` list) - every one of them is a
/// genuine multi-source accumulator or a `splice_add.wgsl` commit target
/// starting from zero; see [`gdn_chunk_bwd`]'s own doc for exactly which
/// steps write which field and why. Every other field here is a dedicated
/// single-producer buffer, fully overwritten before it is ever read, and
/// needs no clearing.
pub struct GdnBwdScratch<'a> {
    // ---- Phase 1 working buffers, whole-tensor `[bhc,...]`-shaped: everything
    // that does not depend on the running `d_state` is computed for all chunks
    // at once, so the reverse sweep keeps only what does. ----
    /// `[bhc,c,dk]` - each chunk's `d_decayed_k` (written by the sweep).
    pub d_decayed_k: &'a DeviceBuffer,
    /// `[bhc,c,dk]` - `d_q_scaled`, from `d_out` and the saved states alone.
    pub d_q_scaled: &'a DeviceBuffer,
    /// `[bhc,c,dv]` - `d_v_new`: seeded with the intra-chunk term, completed by
    /// the sweep, then read by the whole-tensor `d_u` and `d_w` products.
    pub d_v_new: &'a DeviceBuffer,
    /// `[bhc,c]` - `d_decay_scale`.
    pub d_decay_scale: &'a DeviceBuffer,
    /// `[bhc,c,dv]` - `w @ state_in`, the recomputed `v_prime`.
    pub v_prime_all: &'a DeviceBuffer,
    /// `[bhc,c,dv]` - the recomputed `v_new = u - v_prime`.
    pub v_new_all: &'a DeviceBuffer,
    /// `[bhc,c,dk]` - the recomputed `decayed_k = key * decay_scale`.
    pub decayed_k_all: &'a DeviceBuffer,
    /// `[bhc,c,dk]` - the recomputed `q_scaled = query * exp_g_cs * scale`.
    pub q_scaled_all: &'a DeviceBuffer,
    /// `[bh,dk,dv]` - the running `d_state` gradient the reverse chunk sweep threads
    /// backward, updated in place chunk by chunk.
    pub state_a: &'a DeviceBuffer,

    // ---- whole-tensor dedicated single-producer buffers, `[bhc,...]`-shaped ----
    pub d_raw_intra: &'a DeviceBuffer,
    pub d_k_beta_decay: &'a DeviceBuffer,
    pub d_v_beta: &'a DeviceBuffer,
    pub d_raw_attn0: &'a DeviceBuffer,
    pub d_attn0: &'a DeviceBuffer,

    // ---- whole-tensor accumulators, `[bhc,...]`-shaped ----
    /// `[bhc,c]` - 4 sources (state-decay's scalar, `decay_scale`'s two
    /// halves, `decay_mask`'s row/column sums, `exp_g_cs`'s backward). MUST
    /// be zeroed.
    pub d_g_cs: &'a DeviceBuffer,
    /// `[bhc,c]` - 2 sources (`q_scaled`'s per-chunk row-scale,
    /// `k_cumdecay`'s whole-tensor row-scale). MUST be zeroed.
    pub d_exp_g_cs: &'a DeviceBuffer,
    /// `[bhc,c,c]` - 2 sources: the two direct linear uses of `t_mat` (`u`'s
    /// and `w`'s own backward, the first a plain `bmm` overwrite so no zero
    /// is needed before it). Before M5.11's closed form this also had a
    /// third source, the UT-transform's own internal recurrence scatter
    /// (`gdn_ut_bwd_dtmat.wgsl`) - the closed form only READS `d_t_mat` (as
    /// a `bmm` operand) rather than scattering further contributions into
    /// it, so that third source is gone. Needs no explicit zero: the first
    /// producer (`w`'s backward, item 10) is a plain overwrite.
    pub d_t_mat: &'a DeviceBuffer,
    /// `[bhc,c,dv]` - single producer (`v_new`'s identity pass-through,
    /// chunk-by-chunk via `splice_add.wgsl`). MUST be zeroed (a `splice_add`
    /// commit, not a native-offset overwrite).
    pub d_u: &'a DeviceBuffer,
    /// `[bhc,c,dk]` - single producer (`v_prime`'s backward, a plain `bmm`
    /// with a native chunk-offset write). Needs no zero.
    pub d_w: &'a DeviceBuffer,
    /// `[bhc,c,c]` - single producer (`out`'s intra-chunk term backward, a
    /// plain `bmm` with a native chunk-offset write). Needs no zero.
    pub d_intra_scores: &'a DeviceBuffer,
    /// `[bhc,c,c]` - 2 sources (`intra_scores`'s and `attn0`'s own
    /// backward). MUST be zeroed.
    pub d_decay_mask: &'a DeviceBuffer,
    /// `[bhc,c,dk]` - 2 sources (`k_cumdecay`'s row-scale via `scale_add`
    /// overwrite, then `attn0`'s backward accumulate). Needs no zero: the
    /// first producer uses `scale_add.wgsl`'s own `accumulate=0` mode.
    pub d_k_beta: &'a DeviceBuffer,

    // ---- dense scratch for the "row_dot/mul, then splice_add" commit
    // pattern -- reused across every call site whose lifetime does not
    // overlap (each result is consumed by the very next step, before the
    // next producer of the same scratch runs). Sized for the LARGEST use.
    /// `[bhc*c]` - every `row_dot.wgsl` output.
    pub dot_scratch: &'a DeviceBuffer,
    /// `[bhc*c]` - `mul.wgsl`'s `exp_g_cs` backward output (item 13).
    pub mul_scratch: &'a DeviceBuffer,
    /// `[bhc*c*c]` - `mul.wgsl`'s `decay_mask` backward output (item 14),
    /// then reused (its item-14 result is fully committed into
    /// `d_decay_mask` by the time item 15 starts) as item 15's `T_mat^T @
    /// d_t_mat` intermediate (M5.11's closed-form UT-transform backward).
    pub mul_scratch_cc: &'a DeviceBuffer,

    // ---- tensor-core backward only ----
    /// `[bhc, dk, dv]` - the state gradient entering each chunk's reverse step, written by
    /// the tensor-core sweep and read by the tensor-core post kernel. `None` selects the portable
    /// backward (the tensor-core one needs this buffer; see [`GdnBwdScratchBufs::new_for`]).
    pub d_state_hist: Option<&'a DeviceBuffer>,
}

/// The tensor-core backward as its two launches (the sweep, then the per-(head, chunk) kernel), or `None`
/// when the call is not served: no state-gradient history in `bwd`, the caller did not opt in, the device has
/// no tensor cores, the head widths or the chunk are outside what the kernels take, or the backend declined
/// a launch. Nothing is written until the returned steps are submitted.
///
/// `chunk_major` is `[query, key, value, beta]`, `grads` is `[d_query, d_key, d_value, d_raw_g, d_beta]`.
/// With `heads_tok == 0` every tensor is chunk-major, as for [`gdn_chunk_bwd`]. With `heads_tok` set to the
/// number of value heads, `d_out` is read token-major and `d_value`, `d_raw_g`, `d_beta` are written
/// token-major (`[tokens, heads, dv]`, `[tokens, heads]`) - the chunk-major permutes folded into the kernels'
/// loads and stores; `d_query` and `d_key` stay chunk-major.
#[allow(clippy::too_many_arguments)]
fn bwd_tensor_core(
    g: &Gpu,
    ids: &GdnIds,
    shape: &GdnShape,
    chunk_major: [&DeviceBuffer; 4],
    saved: &GdnScratchTrain,
    d_out: &DeviceBuffer,
    d_final_state: &DeviceBuffer,
    bwd: &GdnBwdScratch,
    grads: [&DeviceBuffer; 5],
    d_initial_state: &DeviceBuffer,
    heads_tok: u32,
) -> Option<Vec<Step>> {
    let d_state_hist = bwd.d_state_hist.filter(|_| tensor_core_backward_offered(g, ids, shape))?;
    let [query, key, value, beta] = chunk_major;
    let [d_query, d_key, d_value, d_raw_g, d_beta] = grads;
    let (dk, dv, c) = (shape.dk, shape.dv, shape.chunk);
    let (bh, bhc, n_chunks) = (shape.bh(), shape.bhc(), shape.n_chunks());
    let scale = 1.0f32 / (dk as f32).sqrt();
    let sweep_bufs = [
        key,
        query,
        saved.w,
        d_out,
        saved.intra_scores,
        saved.g_cs,
        saved.exp_g_cs,
        saved.decay_scale_hist,
        d_final_state,
        bwd.d_v_new,
        d_state_hist,
        d_initial_state,
    ];
    let post_bufs = [
        query,
        key,
        value,
        beta,
        saved.w,
        saved.u,
        saved.g_cs,
        saved.exp_g_cs,
        saved.decay_scale_hist,
        saved.state_history,
        saved.t_mat,
        saved.raw_intra,
        saved.raw_attn0,
        d_out,
        d_state_hist,
        bwd.d_v_new,
        d_query,
        d_key,
        d_value,
        d_raw_g,
        d_beta,
    ];
    let sweep = native_bwd_sweep_tc(g, &sweep_bufs, bh, c, dk, dv, n_chunks, scale, heads_tok)?;
    let post = native_bwd_post_tc(g, &post_bufs, bhc, c, dk, dv, bh, scale, heads_tok, n_chunks)?;
    Some(vec![sweep, post])
}

/// [`gdn_chunk_bwd`] fed with, and producing, the mixer's TOKEN-MAJOR tensors where it can: `d_out_tok`
/// (`[tokens, heads, dv]`) is read in place of the chunk-major `d_out`, and `d_value_tok` (`[tokens, heads,
/// dv]`), `d_raw_g_tok` and `d_beta_tok` (`[tokens, heads]`) are written token-major, so the mixer drops four
/// of the layout permutes of the portable backward. `d_query_cm` and `d_key_cm` are chunk-major
/// (`[bhc, chunk, dk]`); [`gdn_qk_bwd_tokens`] finishes them. Every output element is overwritten.
///
/// Returns `None` when the call is not served - the caller did not opt in ([`GdnFastIds::tensor_core`]), the
/// device has no tensor cores, `bwd` has no state-gradient history ([`GdnBwdScratchBufs::new_for`]), the head
/// widths are not 128, the chunk is longer than 64 rows, or the backend declined a launch - and the caller
/// then runs the permutes and [`gdn_chunk_bwd`] it always ran. Nothing is written in that case.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_bwd_tokens(
    g: &Gpu,
    ids: &GdnIds,
    shape: &GdnShape,
    chunk_major: &GdnChunkMajorOut,
    saved: &GdnScratchTrain,
    d_out_tok: &DeviceBuffer,
    d_final_state: &DeviceBuffer,
    bwd: &GdnBwdScratch,
    d_query_cm: &DeviceBuffer,
    d_key_cm: &DeviceBuffer,
    d_value_tok: &DeviceBuffer,
    d_raw_g_tok: &DeviceBuffer,
    d_beta_tok: &DeviceBuffer,
    d_initial_state: &DeviceBuffer,
) -> Option<Vec<Step>> {
    bwd_tensor_core(
        g,
        ids,
        shape,
        [chunk_major.query_cm, chunk_major.key_cm, chunk_major.value_cm, chunk_major.beta_cm],
        saved,
        d_out_tok,
        d_final_state,
        bwd,
        [d_query_cm, d_key_cm, d_value_tok, d_raw_g_tok, d_beta_tok],
        d_initial_state,
        shape.h,
    )
}

/// The last leg of the mixer backward for query and key as one launch: `d_query_cm` / `d_key_cm`
/// (chunk-major, per value head, from [`gdn_chunk_bwd_tokens`]) are summed over the `group` value heads
/// that share a key head and taken back through the L2 normalisation of the forward, landing in
/// `d_query` / `d_key` token-major (`[tokens, heads / group, dk]`) against the raw projections `query`
/// and `key`. Replaces two permutes, two group sums and two normalisation backwards. `None` when the
/// device or shape is not served, as for [`gdn_chunk_bwd_tokens`].
#[allow(clippy::too_many_arguments)]
pub fn gdn_qk_bwd_tokens(
    g: &Gpu,
    ids: &GdnIds,
    shape: &GdnShape,
    group: u32,
    l2_eps: f32,
    d_query_cm: &DeviceBuffer,
    d_key_cm: &DeviceBuffer,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    d_query: &DeviceBuffer,
    d_key: &DeviceBuffer,
) -> Option<Step> {
    if !tokens_prefix_offered(g, ids, shape, group) {
        return None;
    }
    native_qk_bwd_tok(g, &[d_query_cm, d_key_cm, query, key, d_query, d_key], shape, group, l2_eps)
}

/// The full Gated DeltaNet chunked-parallel BACKWARD - reverse-mode gradients
/// through all 11 forward steps, including the UT-transform's own reverse
/// sequential sweep and the across-chunk recurrent-state gradient thread.
/// Reads [`GdnScratchTrain`] (produced by [`gdn_chunk_fwd_train`] - NOT
/// [`gdn_chunk_fwd`], which does not save what this function needs) as its
/// forward-saved activations, plus the ORIGINAL `query`/`key`/`value`/`beta`
/// inputs (`raw_g`'s own VALUE is never read here - only its gradient,
/// [`GdnScratchTrain::g_cs`] already carries everything backward needs from
/// it). `d_out` is chunk-major, same shape as forward's own `out`;
/// `d_final_state` is `[bh,dk,dv]` (zero if the caller does not use
/// `final_state` downstream).
///
/// ## Structure
///
/// **Phase 1** - a REVERSE sweep over chunks (`ci` from `n_chunks-1` down to
/// `0`), threading a running `d_state` gradient backward through the
/// recurrence exactly as far as the forward loop threaded `state` forward
/// (just in the opposite direction), and writing 9 forward-step
/// contributions' worth of gradient per chunk: `decayed_k`/`v_new`'s
/// backward (state-update accumulate), `state_in`'s decay-scale backward
/// (reusing `gdn_state_decay.wgsl` itself as its own adjoint for the
/// tensor half, plus a dedicated reduction kernel for the scalar-decay
/// half), `intra_scores`/`v_new`'s backward (the intra-chunk output term),
/// `q_scaled`/`state_in`'s backward (the inter-chunk output term), then
/// `q_scaled`'s and `decayed_k`'s OWN row-scale backward into `d_query`/
/// `d_key` (committed into their own chunk slice via `splice_add.wgsl`,
/// since the row-scale kernels have no output offset), `decay_scale`'s
/// backward into `d_g_cs`, and finally `v_new`/`w`'s backward completing
/// `d_state_in` (which becomes next iteration's `d_state`, or
/// `d_initial_state` after `ci=0`).
///
/// **Phase 2** - the rest, in REVERSE FORWARD-STEP order, over the
/// WHOLE TENSOR (no chunk loop: steps 8-9's `bmm`s and everything upstream
/// of them ran once over every chunk at once, so their backward does too):
/// `w`/`u`'s producing `bmm`s backward into `d_t_mat` (two contributions,
/// the second `accumulate`s onto the first's plain overwrite) and
/// `d_k_beta_decay`/`d_v_beta`; `k_beta_decay`'s row-scale backward into
/// `d_k_beta`/`d_exp_g_cs`; `exp_g_cs`'s backward into `d_g_cs`; the
/// `intra_scores` precompute's backward into `d_raw_intra`/`d_decay_mask`
/// and (accumulating) `d_query`/`d_key`; the UT-transform's OWN backward,
/// `dL/dattn0 = T_mat^T @ d_t_mat @ T_mat^T` (M5.11's closed form, two
/// batched `bmm`s, replacing a former `2*(c-1)`-dispatch reverse sweep -
/// `gdn_ut_bwd_dattn0.wgsl`/`gdn_ut_bwd_dtmat.wgsl`, still registered/tested
/// but no longer on this call path, kernel-performance.md M5.11) completing
/// `d_attn0`; `attn0`'s mask-multiply backward into
/// `d_raw_attn0`/`d_decay_mask`; `raw_attn0`'s producing `bmm` backward
/// (accumulating into `d_k_beta`/`d_key`); `decay_mask`'s backward (a
/// row-sum AND a column-sum over the same `[bhc,c,c]` tensor, since
/// `g_cs[i]` and `g_cs[j]` both feed every masked cell) completing `d_g_cs`;
/// the per-chunk cumsum's backward (a REVERSE cumsum / suffix sum) producing
/// `d_raw_g`; and finally `v_beta`/`k_beta`'s row-scale backward completing
/// `d_value`/`d_key`/`d_beta`.
///
/// Every output with more than one contributing forward use is explicitly
/// zeroed by the caller (see [`GdnBwdScratch`]'s own doc for the complete
/// list) and every contribution below ACCUMULATES into it rather than
/// overwriting - `d_query` (2 sources: this chunk-loop's row-scale, the
/// `intra_scores` precompute's `bmm`), `d_key` (4 sources: the chunk-loop's
/// row-scale, the `intra_scores` precompute, `raw_attn0`'s producing `bmm`,
/// `k_beta`'s own row-scale), `d_beta` (2 sources: `v_beta`'s and `k_beta`'s
/// row-scales), and `d_g_cs` (4 sources, see [`GdnBwdScratch::d_g_cs`]).
///
/// `d_query`/`d_key`/`d_value` are chunk-major `[bhc,c,dk-or-dv]`, matching
/// forward's own `query`/`key`/`value` layout; `d_raw_g`/`d_beta` are
/// chunk-major `[bhc,c]`; `d_initial_state` is `[bh,dk,dv]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_bwd(
    g: &Gpu,
    ids: &GdnIds,
    bwd_ids: &GdnBwdIds,
    shape: &GdnShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    beta: &DeviceBuffer,
    saved: &GdnScratchTrain,
    d_out: &DeviceBuffer,
    d_final_state: &DeviceBuffer,
    bwd: &GdnBwdScratch,
    d_query: &DeviceBuffer,
    d_key: &DeviceBuffer,
    d_value: &DeviceBuffer,
    d_raw_g: &DeviceBuffer,
    d_beta: &DeviceBuffer,
    d_initial_state: &DeviceBuffer,
) -> Vec<Step> {
    let (dk, dv, c) = (shape.dk, shape.dv, shape.chunk);
    let n_chunks = shape.n_chunks();
    let bh = shape.bh();
    let bhc = shape.bhc();
    let scale = 1.0f32 / (dk as f32).sqrt();

    // Tensor cores: the reverse sweep as one launch and everything else as a second, one block per
    // (head, chunk). Declined (no scratch, no device support, shape) means the sequence below.
    if let Some(steps) = bwd_tensor_core(g, ids, shape, [query, key, value, beta], saved, d_out, d_final_state, bwd, [d_query, d_key, d_value, d_raw_g, d_beta], d_initial_state, 0) {
        return steps;
    }

    let bmm = |kernel: usize, batch: u32, m: u32, k: u32, n: u32, ta: bool, tb: bool, alpha: f32, a: &DeviceBuffer, a_off: u32, b: &DeviceBuffer, b_off: u32, o: &DeviceBuffer, o_off: u32| {
        bmm_auto(g, ids, kernel, batch, m, k, n, ta, tb, alpha, a, a_off, b, b_off, o, o_off)
    };
    let splice = |src: &DeviceBuffer, dst: &DeviceBuffer, base: u32, n: u32| g.step(bwd_ids.splice_add, &[src, dst], &[n, base], n);
    let row_dot = |a: &DeviceBuffer, a_off: u32, b: &DeviceBuffer, b_off: u32, rows: u32, d: u32, alpha: f32, out: &DeviceBuffer| {
        g.step(bwd_ids.row_dot, &[a, b, out], &[rows, d, a_off, b_off, f(alpha)], rows)
    };
    let row_scale_off = |x: &DeviceBuffer, x_off: u32, s: &DeviceBuffer, s_off: u32, alpha: f32, out: &DeviceBuffer, total: u32, m: u32| {
        g.step(ids.row_scale_off, &[x, s, out], &[total, m, x_off, s_off, f(alpha)], total)
    };
    let scale_add = |gate: &DeviceBuffer, src: &DeviceBuffer, acc: &DeviceBuffer, seq_len: u32, d_model: u32, accumulate: bool| {
        g.step(bwd_ids.scale_add, &[gate, src, acc], &[seq_len, d_model, 1, 0, accumulate as u32], seq_len * d_model)
    };

    let mut steps = Vec::new();

    // ==================== Phase 1: reverse sweep over chunks ====================
    // Only the running `d_state` is sequential. Everything the sweep needs that
    // does not depend on it - the recomputed forward values of every chunk, the
    // intra-chunk and `state_in` terms of the gradients, and the row-scale
    // gradients of `query` - is computed for ALL chunks at once, in a handful of
    // whole-tensor dispatches (the saved `state_history[0 .. n_chunks]` is a
    // `[bhc, dk, dv]` batch of each chunk's `state_in`). The sweep then spends
    // seven dispatches per chunk instead of twenty-nine.
    let (vd, kd) = (bhc * c * dv, bhc * c * dk);

    // ---- recompute every chunk's forward working values (not saved) ----
    steps.push(bmm(ids.bmm, bhc, c, dk, dv, false, false, 1.0, saved.w, 0, saved.state_history, 0, bwd.v_prime_all, 0));
    steps.push(g.step(ids.sub, &[saved.u, bwd.v_prime_all, bwd.v_new_all], &[vd, 0, 0], vd));
    steps.push(row_scale_off(key, 0, saved.decay_scale_hist, 0, 1.0, bwd.decayed_k_all, kd, dk));
    steps.push(row_scale_off(query, 0, saved.exp_g_cs, 0, scale, bwd.q_scaled_all, kd, dk));

    // ---- item 3: out_c += intra_scores_c @ v_new (bmm_acc) backward, `d_state`-free halves ----
    // d_intra_scores[i,j] = sum_q d_out[i,q]*v_new[j,q]  (single producer)
    steps.push(bmm(ids.bmm, bhc, c, dv, c, false, true, 1.0, d_out, 0, bwd.v_new_all, 0, bwd.d_intra_scores, 0));
    // d_v_new[j,q] = sum_i intra_scores[i,j]*d_out[i,q]  (seeds the sum the sweep completes)
    steps.push(bmm(ids.bmm, bhc, c, c, dv, true, false, 1.0, saved.intra_scores, 0, d_out, 0, bwd.d_v_new, 0));

    // ---- item 4: out_c = q_scaled @ state_in (bmm) backward, the `d_out`-only half ----
    // d_q_scaled[i,p] = sum_q d_out[i,q]*state_in[p,q]
    steps.push(bmm(ids.bmm, bhc, c, dv, dk, false, true, 1.0, d_out, 0, saved.state_history, 0, bwd.d_q_scaled, 0));

    // ---- item 5: q_scaled = query_c * exp(g_cs_c) * scale backward ----
    // d_query (FIRST of two contributions - an overwrite)
    steps.push(row_scale_off(bwd.d_q_scaled, 0, saved.exp_g_cs, 0, scale, d_query, kd, dk));
    // d_exp_g_cs += scale * sum_d d_q_scaled[row,d]*query[row,d]  (FIRST of two, via splice)
    steps.push(row_dot(bwd.d_q_scaled, 0, query, 0, bhc * c, dk, scale, bwd.dot_scratch));
    steps.push(splice(bwd.dot_scratch, bwd.d_exp_g_cs, 0, bhc * c));

    // ---- the sequential part: the running d_state, newest chunk first ----
    // One native launch (the state gradient stays in shared memory across all chunks)
    // where the device offers it, else the host walk below.
    let native_sweep = native_bwd_sweep(
        g,
        &[
            bwd.v_new_all,
            bwd.decayed_k_all,
            bwd.q_scaled_all,
            saved.w,
            d_out,
            saved.state_history,
            saved.g_cs,
            d_final_state,
            bwd.d_decayed_k,
            bwd.d_v_new,
            bwd.d_g_cs,
            d_initial_state,
        ],
        bh,
        c,
        dk,
        dv,
        n_chunks,
    );
    if let Some(step) = native_sweep {
        steps.push(step);
    } else {
        // Seed the running d_state from the caller's d_final_state; the sweep updates it in place.
        let cur = bwd.state_a;
        steps.push(g.step(ids.region_copy, &[d_final_state, cur], &[1, bh * dk * dv, bh * dk * dv, 0], bh * dk * dv));

        for step_idx in 0..n_chunks {
            let ci = n_chunks - 1 - step_idx;
            let off_d = |d: u32| ci * bh * c * d;
            let off_g = ci * bh * c;
            let state_in_off = ci * bh * dk * dv;

            // ---- item 1: state_out += decayed_k^T @ v_new (bmm_acc) backward ----
            // d_decayed_k[c,p] = sum_q v_new[c,q]*d_state[p,q]
            steps.push(bmm(ids.bmm, bh, c, dv, dk, false, true, 1.0, bwd.v_new_all, off_d(dv), cur, 0, bwd.d_decayed_k, off_d(dk)));
            // d_v_new[c,q] += sum_p decayed_k[c,p]*d_state[p,q]  (the other half is already there)
            steps.push(bmm(ids.bmm_acc, bh, c, dk, dv, false, false, 1.0, bwd.decayed_k_all, off_d(dk), cur, 0, bwd.d_v_new, off_d(dv)));

            // ---- item 2: state_out = state_in * decay_last (gdn_state_decay) backward ----
            // d_decay_last = sum_{p,q} d_state[p,q]*state_in[p,q] -> d_g_cs[last] += ... * decay_last
            // (reads the d_state of the chunk after this one, so it runs before the scaling below)
            steps.push(g.step(
                bwd_ids.state_decay_bwd_dscale,
                &[cur, saved.state_history, saved.g_cs, bwd.d_g_cs],
                &[bh, dk, dv, c, off_g, state_in_off],
                bh,
            ));
            // d_state_in += d_state * decay_last -- gdn_state_decay.wgsl is its own
            // backward here (a scalar scale is self-adjoint), applied in place with the
            // SAME g_cs_off; the other two terms then accumulate onto the scaled state.
            steps.push(g.step(ids.state_decay, &[saved.g_cs, cur], &[bh, dk, dv, c, off_g], bh * dk * dv));

            // ---- item 4: d_state_in[p,q] += sum_i q_scaled[i,p]*d_out[i,q]  (SECOND of three) ----
            steps.push(bmm(ids.bmm_acc, bh, dk, c, dv, true, false, 1.0, bwd.q_scaled_all, off_d(dk), d_out, off_d(dv), cur, 0));

            // ---- item 9: v_prime = w_c @ state_in (bmm) backward ----
            // d_state_in[p,q] += -sum_c w[c,p]*d_v_new[c,q]  (THIRD and final)
            steps.push(bmm(ids.bmm_acc, bh, dk, c, dv, true, false, -1.0, saved.w, off_d(dk), bwd.d_v_new, off_d(dv), cur, 0));
        }
        // After processing ci=0, `cur` holds the completed d_initial_state.
        steps.push(g.step(ids.region_copy, &[cur, d_initial_state], &[1, bh * dk * dv, bh * dk * dv, 0], bh * dk * dv));
    }

    // ---- item 6: decayed_k = key_c * decay_scale_c backward, all chunks ----
    // d_key (FIRST of four contributions - an overwrite)
    steps.push(row_scale_off(bwd.d_decayed_k, 0, saved.decay_scale_hist, 0, 1.0, d_key, kd, dk));
    // d_decay_scale[row] = sum_d d_decayed_k[row,d]*key[row,d]
    steps.push(row_dot(bwd.d_decayed_k, 0, key, 0, bhc * c, dk, 1.0, bwd.d_decay_scale));

    // ---- item 7: decay_scale[i] = exp(g_last - g_cs[i]) backward ----
    steps.push(g.step(bwd_ids.decay_scale_bwd, &[bwd.d_decay_scale, saved.decay_scale_hist, bwd.d_g_cs], &[bhc, c, 0], bhc * c));
    steps.push(g.step(bwd_ids.decay_scale_bwd_last, &[bwd.d_decay_scale, saved.decay_scale_hist, bwd.d_g_cs], &[bhc, c, 0], bhc));

    // ---- item 8: v_new = u_c - v_prime (sub) backward ----
    // d_u += d_v_new (identity pass-through; single producer, via splice)
    steps.push(splice(bwd.d_v_new, bwd.d_u, 0, vd));

    // ---- item 9: v_prime = w_c @ state_in (bmm) backward, the `d_w` half ----
    // d_w[c,p] = -sum_q d_v_new[c,q]*state_in[p,q]  (single producer)
    steps.push(bmm(ids.bmm, bhc, c, dv, dk, false, true, -1.0, bwd.d_v_new, 0, saved.state_history, 0, bwd.d_w, 0));

    // ==================== Phase 2: whole-tensor, reverse forward-step order ====================

    // ---- item 10: w = t_mat @ k_beta_decay (bmm) backward ----
    steps.push(bmm(ids.bmm, bhc, c, dk, c, false, true, 1.0, bwd.d_w, 0, saved.k_beta_decay, 0, bwd.d_t_mat, 0));
    steps.push(bmm(ids.bmm, bhc, c, c, dk, true, false, 1.0, saved.t_mat, 0, bwd.d_w, 0, bwd.d_k_beta_decay, 0));

    // ---- item 11: k_beta_decay = k_beta * exp_g_cs backward ----
    steps.push(scale_add(saved.exp_g_cs, bwd.d_k_beta_decay, bwd.d_k_beta, bhc * c, dk, false));
    steps.push(row_dot(bwd.d_k_beta_decay, 0, saved.k_beta, 0, bhc * c, dk, 1.0, bwd.dot_scratch));
    steps.push(splice(bwd.dot_scratch, bwd.d_exp_g_cs, 0, bhc * c));

    // ---- item 12: u = t_mat @ v_beta (bmm) backward ----
    steps.push(bmm(ids.bmm_acc, bhc, c, dv, c, false, true, 1.0, bwd.d_u, 0, saved.v_beta, 0, bwd.d_t_mat, 0));
    steps.push(bmm(ids.bmm, bhc, c, c, dv, true, false, 1.0, saved.t_mat, 0, bwd.d_u, 0, bwd.d_v_beta, 0));

    // ---- item 13: exp_g_cs = exp(g_cs) backward ----
    steps.push(g.step(ids.mul, &[bwd.d_exp_g_cs, saved.exp_g_cs, bwd.mul_scratch], &[bhc * c], bhc * c));
    steps.push(splice(bwd.mul_scratch, bwd.d_g_cs, 0, bhc * c));

    // ---- item 14: intra_scores = raw_intra * decay_mask, then raw_intra = scale*(q@k^T) backward ----
    steps.push(g.step(ids.mul, &[bwd.d_intra_scores, saved.decay_mask, bwd.d_raw_intra], &[bhc * c * c], bhc * c * c));
    steps.push(g.step(ids.mul, &[bwd.d_intra_scores, saved.raw_intra, bwd.mul_scratch_cc], &[bhc * c * c], bhc * c * c));
    steps.push(splice(bwd.mul_scratch_cc, bwd.d_decay_mask, 0, bhc * c * c));
    steps.push(bmm(ids.bmm_acc, bhc, c, c, dk, false, false, scale, bwd.d_raw_intra, 0, key, 0, d_query, 0));
    steps.push(bmm(ids.bmm_acc, bhc, c, c, dk, true, false, scale, bwd.d_raw_intra, 0, query, 0, d_key, 0));

    // ---- item 15: UT-transform backward -- closed form (M5.11) ----
    // T_mat = (I - attn0)^-1, so its reverse-mode adjoint is the standard
    // matrix-inverse adjoint dL/dM = -Y^T @ dL/dY @ Y^T (Y = M^-1), and
    // M = I - attn0 is affine with Jacobian -I, so dL/dattn0 = -dL/dM =
    // T_mat^T @ d_t_mat @ T_mat^T -- two batched GEMMs (`bmm.wgsl`,
    // `trans_a`/`trans_b` picking out each transpose) replacing the former
    // `2*(c-1)`-dispatch reverse sweep (`i` from `c-1` downto `1`, one
    // `gdn_ut_bwd_dattn0.wgsl` + one `gdn_ut_bwd_dtmat.wgsl` per `i` -- both
    // kernels remain registered/tested, just off this call path now, same
    // precedent as `gdn_ut_step.wgsl` after M5.9). `bwd.mul_scratch_cc` is
    // free here (its last producer/consumer pair was item 14's `mul.wgsl` +
    // `splice_add.wgsl` into `d_decay_mask`, both already committed above)
    // and holds the intermediate `T_mat^T @ d_t_mat`.
    //
    // No explicit re-mask of the dense `[c,c]` product to strictly-lower is
    // applied here, DELIBERATELY: `d_t_mat` itself is dense (the `w`/`u`
    // `bmm` backwards above write/accumulate into every cell, including the
    // diagonal and strict upper, which are structurally-constant `1`/`0` in
    // `t_mat`'s forward value but still carry a well-defined local partial
    // derivative that the correct closed-form input must include), and the
    // resulting `d_attn0`'s own diagonal/strict-upper cells are simply
    // un-physical -- `attn0` has no freedom there, so nothing may ever read
    // them back out. Item 16 (`gdn_mask_strict_lower_bwd.wgsl`, right below)
    // is the ONLY consumer of `d_attn0`, and it already gates every read on
    // `j < i` on its own (seeded from `attn0`'s own strict-lower mask, not
    // from anything this step produced) -- confirmed empirically, not
    // assumed: `gdn_chunk_bwd_gradcheck` is green with no re-mask kernel
    // inserted here (see kernel-performance.md M5.11).
    steps.push(bmm(ids.bmm, bhc, c, c, c, true, false, 1.0, saved.t_mat, 0, bwd.d_t_mat, 0, bwd.mul_scratch_cc, 0));
    steps.push(bmm(ids.bmm, bhc, c, c, c, false, true, 1.0, bwd.mul_scratch_cc, 0, saved.t_mat, 0, bwd.d_attn0, 0));

    // ---- item 16: attn0 = raw_attn0 * decay_mask, masked j<i, backward ----
    steps.push(g.step(
        bwd_ids.mask_strict_lower_bwd,
        &[bwd.d_attn0, saved.raw_attn0, saved.decay_mask, bwd.d_raw_attn0, bwd.d_decay_mask],
        &[bhc, c],
        bhc * c * c,
    ));

    // ---- item 17: raw_attn0 = -1*(k_beta @ key^T) backward ----
    steps.push(bmm(ids.bmm_acc, bhc, c, c, dk, false, false, -1.0, bwd.d_raw_attn0, 0, key, 0, bwd.d_k_beta, 0));
    steps.push(bmm(ids.bmm_acc, bhc, c, c, dk, true, false, -1.0, bwd.d_raw_attn0, 0, saved.k_beta, 0, d_key, 0));

    // ---- item 18: decay_mask[i,j] = exp(g_cs[i]-g_cs[j]) backward (row-sum, then column-sum) ----
    steps.push(g.step(bwd_ids.decay_mask_bwd, &[bwd.d_decay_mask, saved.decay_mask, bwd.d_g_cs], &[bhc, c, 0], bhc * c));
    steps.push(g.step(bwd_ids.decay_mask_bwd, &[bwd.d_decay_mask, saved.decay_mask, bwd.d_g_cs], &[bhc, c, 1], bhc * c));

    // ---- item 19: g_cs = cumsum(raw_g) backward (reverse cumsum / suffix sum, one fused dispatch) ----
    steps.push(g.step(ids.region_copy, &[bwd.d_g_cs, d_raw_g], &[1, bhc * c, bhc * c, 0], bhc * c));
    steps.push(g.step(bwd_ids.reverse_cumsum_step, &[d_raw_g], &[bhc, c], bhc));

    // ---- item 20: v_beta = value*beta, k_beta = key*beta backward ----
    steps.push(g.step(ids.row_scale, &[bwd.d_v_beta, beta, d_value], &[bhc * c * dv, dv], bhc * c * dv));
    steps.push(scale_add(beta, bwd.d_k_beta, d_key, bhc * c, dk, true));
    steps.push(row_dot(bwd.d_v_beta, 0, value, 0, bhc * c, dv, 1.0, bwd.dot_scratch));
    steps.push(splice(bwd.dot_scratch, d_beta, 0, bhc * c));
    steps.push(row_dot(bwd.d_k_beta, 0, key, 0, bhc * c, dk, 1.0, bwd.dot_scratch));
    steps.push(splice(bwd.dot_scratch, d_beta, 0, bhc * c));

    steps
}

// =============================================================================
// Decode: the single-token recurrent state update -- gdn_recurrent_step
// =============================================================================
//
// Everything above is [`gdn_chunk_fwd`]'s prefill machinery (whole sequence at
// once, chunk-parallel). Incremental decode needs the OTHER form of the same
// recurrence: given one new token and the recurrent state persisted from the
// previous token, produce this token's output and the updated state, with no
// chunk/UT-transform machinery at all. Transcribed directly from HuggingFace's
// `torch_recurrent_gated_delta_rule` (`transformers/models/qwen3_5_moe/
// modeling_qwen3_5_moe.py` lines 332-381) -- read that function, not this
// paraphrase, before changing anything here. Per-`(batch,head)`, per-token
// `query,key: [Dk]`, `value: [Dv]`, scalar raw log-decay `g` and scalar sigmoid
// gate `beta` (both already computed by the caller, exactly as
// [`gdn_chunk_fwd`] expects, and `query` UNSCALED -- see below):
//
//   state = state * exp(g)                     -- decay, state: [Dk,Dv]
//   kv_mem[Dv]  = sum_Dk state[Dk,Dv] * key[Dk] -- kv_mem = key^T @ state
//   delta[Dv]   = (value[Dv] - kv_mem[Dv]) * beta
//   state[Dk,Dv] += key[Dk] * delta[Dv]         -- rank-1 outer-product accumulate
//   output[Dv]  = sum_Dk state[Dk,Dv] * query[Dk] * scale -- POST-update state
//
// `scale = 1/sqrt(Dk)`. The reference's own `torch_recurrent_gated_delta_rule`
// pre-scales `query` ONCE, outside its per-token loop; this module instead
// folds `scale` into the one `bmm` that ever consumes `query` (its `alpha`),
// matching [`gdn_chunk_fwd`]'s own convention (see this file's top-of-module
// doc, step 1) of never mutating `query` itself. `key`/`value`/`state` are
// otherwise identical to [`gdn_chunk_fwd`]'s own per-chunk state-update math
// (steps 10's `v_prime`/`v_new`/state-decay/state-accumulate) -- unsurprising,
// since [`gdn_chunk_fwd`] AT `chunk=1` degenerates to exactly this recurrence
// (`T_mat` collapses to the 1x1 identity, `decay_mask`'s only cell is `1`, so
// `u = v_beta`, `w = k_beta*exp(g)`, and the chunk loop's `v_prime`/`v_new`
// become precisely this function's `kv_mem`/`delta`) -- which is the oracle
// `crates/model/tests/gdn_recurrent_step.rs` checks this function against,
// rather than re-deriving a second independent host reference: running
// [`gdn_chunk_fwd`] with `chunk=1` over T one-token "chunks" and running this
// function T times in a row, state threaded between calls, must agree to
// fp32 tolerance.
//
// ## Why every step composes from EXISTING kernels
//
// * `state *= exp(g)`: [`GdnIds::state_decay`] (`gdn_state_decay.wgsl`)
//   already computes exactly `state[bh,dk,dv] *= exp(g_cs[g_cs_off +
//   bh*c_len + c_len-1])` in place -- passing `c_len=1, g_cs_off=0` and
//   `raw_g` itself (a single value needs no cumsum) degenerates that to
//   `state *= exp(raw_g[bh])`, this function's decay step, VERBATIM. This is
//   a better fit than `exp.wgsl` + `scale_row.wgsl` (the composition this
//   module's own porting task sketched): `scale_row.wgsl` is not an in-place
//   kernel (separate `x`/`y` bindings), and `Gpu::step`'s own
//   `assert_no_output_alias` FORBIDS binding one buffer as both an input and
//   the output slot of a single dispatch -- so decaying `state` through
//   `scale_row.wgsl` would need a second `[bh,dk,dv]`-sized scratch buffer
//   for no benefit, when `gdn_state_decay.wgsl` (already in [`GdnIds`], no
//   new kernel, no extra scratch) does the identical arithmetic in place.
// * `kv_mem = key^T @ state`: [`bmm_step`] with `batch=bh, m=1, k=dk, n=dv,
//   trans_a=false, trans_b=false` -- `key` addressed as `[bh,1,dk]` (its own
//   natural `[bh,dk]` layout, unchanged), `state` as `[bh,dk,dv]`, output into
//   a dedicated `[bh,dv]` scratch buffer (never `state` itself -- see the
//   `assert_no_output_alias` note above; every bmm output here is a buffer
//   distinct from all of that call's inputs).
// * `delta = (value - kv_mem) * beta`: [`GdnIds::sub`] into a SECOND small
//   scratch buffer (again distinct from `value`/`kv_mem`), then
//   [`GdnIds::row_scale`] (`scale_row.wgsl`, `m=dv`) writing back into the
//   FIRST scratch buffer -- legal because by the time `row_scale` runs, that
//   buffer's `kv_mem` value has already been consumed by `sub` and nothing
//   else needs it; the two scratch buffers simply ping-pong rather than one
//   being reused as a `scale_row.wgsl` self-alias (which, again, `Gpu::step`
//   would reject).
// * `state += key ⊗ delta`: [`bmm_step`] with [`GdnIds::bmm_acc`], `batch=bh,
//   m=dk, k=1, n=dv` -- `key` reinterpreted as `[bh,dk,1]` (SAME physical
//   bytes as the `kv_mem` bmm's `[bh,1,dk]` view: with `k=1` the address
//   expression collapses to the same `a_base + i` either way, so no separate
//   transposed copy of `key` is needed), `delta` as `[bh,1,dv]` (its own
//   natural `[bh,dv]` layout), accumulating into `state` (the one dispatch
//   here where `state` legitimately appears ONLY as the output binding, which
//   `bmm_acc.wgsl` is specifically designed to read-modify-write).
// * `output = query^T @ state(updated) * scale`: [`bmm_step`] with
//   [`GdnIds::bmm`], the same `batch=bh, m=1, k=dk, n=dv` shape as `kv_mem`'s
//   own bmm with `query` in place of `key` and `alpha=scale`, run AFTER the
//   state update above (`state` at this point already holds the new value).
//
// No new kernel, no new [`GdnIds`] field -- every kernel this function
// dispatches is already a [`GdnIds`] member [`gdn_chunk_fwd`] itself uses.

/// Small per-token scratch [`gdn_recurrent_step`] needs, both `[bh,Dv]`
/// (`bh = shape.bh()`). Ping-ponged across the call (see this section's doc,
/// "Why every step composes from existing kernels") rather than being two
/// fixed-role buffers: `kv_mem` first holds `key^T @ state`, then (after
/// `sub`) is overwritten with the beta-scaled `delta` that `scale`
/// (initially holding `value - kv_mem`) has just been used to produce.
pub struct GdnRecurrentScratch<'a> {
    /// `[bh, Dv]` -- `key^T @ state`, then (after the `sub`+`row_scale` pair)
    /// the final `delta = (value - kv_mem) * beta` fed to the state's
    /// rank-1 accumulate.
    pub kv_mem: &'a DeviceBuffer,
    /// `[bh, Dv]` -- `value - kv_mem`, consumed immediately by the
    /// `row_scale` call that produces `kv_mem`'s final `delta` value above.
    pub sub_out: &'a DeviceBuffer,
}

/// The single-token Gated DeltaNet recurrent state update -- see this
/// section's doc (above [`GdnRecurrentScratch`]) for the exact formula and
/// why it needs no kernel beyond [`GdnIds`]'s existing members.
///
/// **Processes exactly ONE token per call** (`shape.t` is not read by this
/// function at all -- every buffer here is `[bh, ...]`-shaped with NO time
/// axis, unlike [`gdn_chunk_fwd`]'s chunk-major `[bhc, c, ...]` convention).
/// A caller decoding `N` tokens sequentially calls this `N` times, threading
/// the SAME `state` buffer from one call to the next -- the same shape as
/// `crates/qwen3/src/model.rs`'s `decode_at`/`decode_steps` calling one
/// incremental step per token rather than taking a token count. This was the
/// simpler of the two options this function's porting task left open (loop
/// inside vs. outside): it composes directly with a host decode loop that
/// also has to do other per-token work (embedding lookup, GQA-layer KV
/// append, sampling) between GDN layers, none of which this module knows
/// about or should.
///
/// `state` is both the input recurrent state AND the in-place-updated output
/// -- the same "one buffer, evolved in place, also the return value"
/// convention [`gdn_chunk_fwd`]'s own `final_state` uses (seed it with zeros
/// for a fresh sequence's first token). `query`/`key`/`value`/`raw_g`/`beta`
/// are `[bh,Dk]`/`[bh,Dk]`/`[bh,Dv]`/`[bh]`/`[bh]` respectively -- this
/// token's row alone, no chunk or time axis; `query` is UNSCALED (see this
/// section's doc for where `1/sqrt(Dk)` is folded in). `out` is `[bh,Dv]`,
/// overwritten (never accumulated).
pub fn gdn_recurrent_step(
    g: &Gpu,
    ids: &GdnIds,
    shape: &GdnShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    raw_g: &DeviceBuffer,
    beta: &DeviceBuffer,
    state: &DeviceBuffer,
    scratch: &GdnRecurrentScratch,
    out: &DeviceBuffer,
) -> Vec<Step> {
    let (dk, dv) = (shape.dk, shape.dv);
    let bh = shape.bh();
    let scale = 1.0f32 / (dk as f32).sqrt();

    vec![
        // state *= exp(raw_g) -- gdn_state_decay.wgsl's own in-place contract,
        // degenerated to a length-1 window (see this section's doc).
        g.step(ids.state_decay, &[raw_g, state], &[bh, dk, dv, 1, 0], bh * dk * dv),
        // kv_mem = key^T @ state (state already decayed).
        bmm_step(g, ids.bmm, bh, 1, dk, dv, false, false, 1.0, key, 0, state, 0, scratch.kv_mem, 0),
        // sub_out = value - kv_mem.
        g.step(ids.sub, &[value, scratch.kv_mem, scratch.sub_out], &[bh * dv, 0, 0], bh * dv),
        // kv_mem <- sub_out * beta == delta (ping-pong: kv_mem's old value was
        // fully consumed by the sub above).
        g.step(ids.row_scale, &[scratch.sub_out, beta, scratch.kv_mem], &[bh * dv, dv], bh * dv),
        // state += key ⊗ delta.
        bmm_step(g, ids.bmm_acc, bh, dk, 1, dv, false, false, 1.0, key, 0, scratch.kv_mem, 0, state, 0),
        // out = query^T @ state(updated), scaled by 1/sqrt(Dk).
        bmm_step(g, ids.bmm, bh, 1, dk, dv, false, false, scale, query, 0, state, 0, out, 0),
    ]
}

// =============================================================================
// Decode: the causal depthwise Conv1d streaming step -- gdn_causal_conv1d_step
// =============================================================================
//
// Qwen3.5's GDN layer runs `in_proj_qkv`'s output through a causal depthwise
// `Conv1d` (`kernel_size=4`, `groups=conv_dim`, i.e. one independent K-tap FIR
// filter per channel) before the recurrence above ever sees `query`/`key`/
// `value`. `audio::conv::conv1d_fwd` (`conv1d.wgsl`) already computes this for
// a WHOLE sequence at once (causal expressed as a left `pad=K-1`); decode
// needs the streaming, one-token-at-a-time sibling, since re-running the
// whole-sequence kernel over a growing sequence every step would be O(T^2).
//
// [`gdn_causal_conv1d_step`] is a thin `Step` wrapper around the new
// `causal_conv1d_step.wgsl` kernel (see that file's own header for the exact
// per-`(n,c)` math and why NO existing kernel fit without an `N`x memory
// cost before adding it). Kept in this module (rather than `audio::conv`,
// where `conv1d_fwd` itself lives)
// because its only caller is GDN-shaped decode and its state -- a per-`(n,c)`
// history ring buffer -- is exactly the kind of persisted-across-calls decode
// state [`gdn_recurrent_step`]'s `state` argument already is; a future
// non-GDN streaming-conv caller can still reuse the kernel directly without
// this wrapper.
//
// `hist` is the persistent "conv state": `[N, C, K-1]`, threaded across calls
// the same way [`gdn_recurrent_step`]'s `state` is, and must start ZEROED for
// a fresh sequence (matching `conv1d_fwd`'s own implicit left zero-pad for a
// whole sequence's first `K-1` tokens). Validated against `conv1d_fwd` by
// `crates/model/tests/causal_conv1d_step.rs`: running `conv1d_fwd` once over a
// short sequence and running this function once per token over the same
// input (with the SAME zeroed `hist` start) must agree to fp32 tolerance.

/// Kernel index [`gdn_causal_conv1d_step`] dispatches.
#[derive(Clone, Copy)]
pub struct GdnConvIds {
    /// `causal_conv1d_step.wgsl`.
    pub causal_conv1d_step: usize,
}

/// The shape one call to [`gdn_causal_conv1d_step`] operates over: `n`
/// sequences decoding in parallel (the decode batch), `c` channels
/// (`conv_dim = key_dim*2 + value_dim` for Qwen3.5's `in_proj_qkv`), `k` the
/// causal conv kernel size (4 for Qwen3.5-35B-A3B's GDN layers).
#[derive(Clone, Copy)]
pub struct GdnConvShape {
    pub n: u32,
    pub c: u32,
    pub k: u32,
}

impl GdnConvShape {
    /// `N*C*(K-1)` -- the `hist` ring-buffer's element count. Panics if
    /// `k == 0` (not a real conv kernel size); `k == 1` is legal (an empty
    /// history, a pointwise "conv") and gives `0`.
    pub fn hist_len(&self) -> u32 {
        assert!(self.k > 0, "GdnConvShape: k must be >= 1");
        self.n * self.c * (self.k - 1)
    }
}

/// One `causal_conv1d_step.wgsl` dispatch -- see this section's doc and that
/// kernel's own header for the exact per-`(n,c)` math, the `hist`
/// zero-initialisation requirement, and why this composes no other kernel.
/// `x`/`y`: `[N,C]` (this token's per-channel input/output, one row per
/// sequence); `w`: `[C,K]` (the depthwise conv weight, shared across every
/// sequence and every decode step); `hist`: `[N,C,K-1]`, read AND
/// written in place (the persisted conv state -- see this section's doc).
pub fn gdn_causal_conv1d_step(
    g: &Gpu,
    ids: &GdnConvIds,
    shape: &GdnConvShape,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    hist: &DeviceBuffer,
    y: &DeviceBuffer,
) -> Step {
    g.step(ids.causal_conv1d_step, &[x, w, hist, y], &[shape.n, shape.c, shape.k], shape.n * shape.c)
}

// =============================================================================
// Serving: one sequence's persistent per-layer state -- RecurrentSlot
// =============================================================================
//
// A resident sequence's [`gdn_recurrent_step`] `state` and
// [`gdn_causal_conv1d_step`] `hist` are not scratch (unlike every buffer
// above this section) -- they are threaded across every decode call for as
// long as the sequence is admitted, one pair per GDN layer, with a size-1
// dummy pair at every non-GDN (full-attention) layer index so a caller can
// index by plain layer number without a second lookup. [`RecurrentSlot`] is
// that per-sequence bundle, hoisted here (rather than left duplicated) from
// `qwen35::serve::GdnSlot`/`qwen35moe::serve::GdnSlot`, which built this
// exact struct byte-for-byte identically -- confirmed by direct comparison,
// not assumed, while doing that hoist. Both crates key a private
// `HashMap<u32, RecurrentSlot>` by paged-KV physical block id (their own
// engine's own serving concern, kept there, not here); this module only
// owns the one slot's own allocation and cost accounting.
//
// A future `DECODE_WINDOW_CAPACITY > 1` (speculative decode) will need this
// slot's `state`/`hist` to be snapshotted and restored, the same way
// `model::paged::BlockTable::fork`/`unshare_tail` let the KV side branch --
// that seam is NOT built here (a separate, larger milestone), just noted as
// the natural extension point.

/// Plain-data shape describing one sequence's persistent Gated-DeltaNet
/// state -- exactly the handful of dims [`RecurrentSlot::new`]/
/// [`RecurrentSlot::bytes`] need, deliberately NOT either model crate's own
/// full `Qwen35Config`: this crate must not depend on `qwen35`/`qwen35moe`
/// (the dependency would run the wrong way), so each adopts this shape from
/// its own config instead.
#[derive(Clone, Debug)]
pub struct RecurrentSlotShape {
    /// Recurrent-state element count for ONE GDN layer:
    /// `linear_num_value_heads * linear_key_head_dim * linear_value_head_dim`
    /// (`[bh, Dk, Dv]` flattened, matching [`GdnShape::bh`]'s own per-layer
    /// state).
    pub state_len: u64,
    /// Causal-conv history element count for ONE GDN layer:
    /// `linear_conv_dim * (linear_conv_kernel_dim - 1)` (matching
    /// [`GdnConvShape::hist_len`]'s own formula).
    pub hist_len: u64,
    /// One entry per decoder layer, in layer order: `true` for a GDN
    /// (linear-attention) layer, `false` for a full-attention layer. Plain
    /// `bool` rather than either crate's own `LayerType` enum, for the same
    /// reason `state_len`/`hist_len` are plain integers above.
    pub is_recurrent: Vec<bool>,
}

/// One admitted sequence's persistent Gated-DeltaNet resources: recurrent
/// `state` + causal-conv `hist`, one pair per layer (a size-1 dummy at
/// full-attention layer indices). See this section's doc for why this lives
/// here rather than in either adopting crate.
pub struct RecurrentSlot {
    pub state: Vec<DeviceBuffer>,
    pub hist: Vec<DeviceBuffer>,
}

impl RecurrentSlot {
    /// Allocate and explicitly ZERO every layer's buffers for a fresh
    /// sequence. [`Gpu::storage`] does not guarantee zero-initialised
    /// memory, and a fresh sequence's recurrent state / conv history MUST
    /// start at zero ([`gdn_recurrent_step`]/[`gdn_causal_conv1d_step`]'s
    /// own docs).
    pub fn new(gpu: &Gpu, shape: &RecurrentSlotShape) -> RecurrentSlot {
        let mut state = Vec::with_capacity(shape.is_recurrent.len());
        let mut hist = Vec::with_capacity(shape.is_recurrent.len());
        for &recurrent in &shape.is_recurrent {
            if recurrent {
                state.push(gpu.storage(shape.state_len));
                hist.push(gpu.storage(shape.hist_len));
            } else {
                state.push(gpu.storage(1));
                hist.push(gpu.storage(1));
            }
        }
        let clears: Vec<&DeviceBuffer> = state.iter().chain(hist.iter()).collect();
        gpu.submit(&clears, &[]);
        RecurrentSlot { state, hist }
    }

    /// Device bytes one slot costs: every GDN layer's `state` + `hist`, fp32.
    /// The one place this arithmetic lives -- [`RecurrentSlot::new`]'s
    /// allocation loop and every caller's own pool-sizing estimate both
    /// depend on agreeing with it.
    pub fn bytes(shape: &RecurrentSlotShape) -> u64 {
        let n_gdn = shape.is_recurrent.iter().filter(|&&r| r).count() as u64;
        n_gdn * (shape.state_len + shape.hist_len) * 4
    }
}
