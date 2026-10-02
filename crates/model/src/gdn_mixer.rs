// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Gated DeltaNet mixer's LoRA/dtype-agnostic internals: depthwise causal
//! conv1d, the query/key/value split, L2-normalize, decay-gate expansion,
//! chunked recurrence ([`crate::gdn::gdn_chunk_fwd`]/[`crate::gdn::
//! gdn_chunk_bwd`]), and the gated RMSNorm - everything between a layer's
//! `in_proj_*` projections and its `out_proj`. Both `qwen35` and `qwen35moe`
//! run byte-identical code here (verified by `crates/model/tests/
//! gdn_mixer_equivalence.rs`); only the projections around this differ per
//! model (LoRA adapters, and for `qwen35moe`, `model::ops::Weight` int8
//! dispatch) - [`crate::block`]'s own module doc states the reason: "Linear
//! projections stay in the model (they carry model-specific concerns such as
//! LoRA adapters and bias)".
//!
//! [`gdn_mixer_fwd`] takes the layer's ALREADY-projected `mixed_qkv`/`bproj`/
//! `aproj`/`z` (the caller's own `in_proj_qkv`/`in_proj_b`/`in_proj_a`/
//! `in_proj_z` outputs) and returns `gated`, ready for the caller's own
//! `out_proj`. [`gdn_mixer_bwd`] is the exact mirror: it takes `d_gated` (the
//! caller's own `out_proj` backward output) and returns `(d_mixed_qkv,
//! d_bproj, d_aproj, d_z)` for the caller's own `in_proj_*` backward.

use gpu_core::{f, DeviceBuffer, Gpu};

use audio::conv::{conv1d_bwd, conv1d_fwd, Conv1d, ConvKernels};

use crate::block::{kv_expand_bwd, kv_expand_fwd, rmsnorm_bwd, rmsnorm_fwd, KernelIds};
use crate::gdn::{
    gdn_chunk_bwd, gdn_chunk_fwd, gdn_chunk_fwd_train, gdn_recurrent_step, GdnBwdIds, GdnBwdScratchBufs, GdnIds, GdnRecurrentScratch, GdnScratchBufs,
    GdnScratchTrainBufs, GdnShape,
};

/// Kernel-pipeline indices [`gdn_mixer_fwd`]/[`gdn_mixer_bwd`] dispatch,
/// beyond the sub-Ids they bundle. Resolved by the calling model against its
/// own registered pipeline list, same convention as [`crate::block::
/// GqaAttnIds`].
#[derive(Clone, Copy)]
pub struct GdnMixerIds {
    /// `rmsnorm`/`rms_inv`/`rmsnorm_dx`/`rmsnorm_dw` for the gated RMSNorm.
    pub kernels: KernelIds,
    /// Depthwise causal conv1d fwd/dx/dw (`crates/audio/src/conv.rs`).
    pub conv: ConvKernels,
    /// [`crate::gdn::gdn_chunk_fwd`]/[`crate::gdn::gdn_chunk_fwd_train`]'s own ids.
    pub chunk: GdnIds,
    /// [`crate::gdn::gdn_chunk_bwd`]'s own ids, beyond `chunk` - unused by
    /// [`gdn_mixer_fwd`], only [`gdn_mixer_bwd`] reads this field.
    pub chunk_bwd: GdnBwdIds,
    pub nlc_nchw: usize,
    pub nchw_nlc: usize,
    pub silu: usize,
    pub silu_bwd: usize,
    pub concat_split: usize,
    pub concat2: usize,
    pub l2norm_scale: usize,
    pub l2norm_scale_dx: usize,
    pub sigmoid: usize,
    pub sigmoid_bwd: usize,
    pub gdn_decay_gate: usize,
    pub gdn_decay_gate_bwd: usize,
    pub kv_expand: usize,
    pub kv_expand_bwd: usize,
    pub gdn_layout_permute: usize,
    pub mul: usize,
    pub bias_grad: usize,
}

/// The mixer's shape, beyond the pure chunked-recurrence [`GdnShape`] it
/// wraps (`gdn.h`/`gdn.dk`/`gdn.dv` are `linear_num_value_heads`/
/// `linear_key_head_dim`/`linear_value_head_dim`). `nkh` (`linear_num_key_heads`)
/// and `conv_kernel` (`linear_conv_kernel_dim`) are the only two dims this
/// layer needs beyond what feeds the chunked recurrence directly - every
/// other width (`key_dim`/`value_dim`/`conv_dim`/`group`) is derived below,
/// mirroring [`GdnShape`]'s own `n_chunks`/`bh`/`bhc` computed-method
/// convention rather than storing a redundant field a caller could pass out
/// of sync with the others.
#[derive(Clone, Copy)]
pub struct GdnMixerShape {
    pub gdn: GdnShape,
    pub nkh: u32,
    pub conv_kernel: u32,
    /// The gated output RMSNorm's epsilon (the checkpoint's `rms_norm_eps`).
    pub rms_eps: f32,
}

impl GdnMixerShape {
    pub fn key_dim(&self) -> u32 {
        self.nkh * self.gdn.dk
    }
    pub fn value_dim(&self) -> u32 {
        self.gdn.h * self.gdn.dv
    }
    pub fn conv_dim(&self) -> u32 {
        2 * self.key_dim() + self.value_dim()
    }
    /// GQA-style repeat factor for the linear-attention heads
    /// (`num_v_heads / num_k_heads`).
    pub fn group(&self) -> u32 {
        self.gdn.h / self.nkh
    }
}

/// The mixer's non-projection weights - never a LoRA target, never quantized
/// (see `qwen35moe::q8`'s own module doc: "Norms/RoPE/`A_log`/`dt_bias`/
/// conv1d: not matmuls, untouched either way"), so always plain fp32 buffers
/// regardless of which dtype tier the caller's projections use.
pub struct GdnMixerWeights<'a> {
    pub conv1d_weight: &'a DeviceBuffer,
    pub a_log: &'a DeviceBuffer,
    pub dt_bias: &'a DeviceBuffer,
    pub norm_weight: &'a DeviceBuffer,
    /// `[linear_key_head_dim]` all-ones buffer bound as `l2norm_scale.wgsl`'s
    /// per-dim scale (query/key L2-norm has no learnable gain).
    pub ones_khd: &'a DeviceBuffer,
}

/// [`GdnMixerWeights`]'s gradient buffers, for [`gdn_mixer_bwd`]. `None` when
/// the corresponding weight is Frozen (inference, or a non-LoRA-targeted
/// weight under a LoRA build - none of these four are ever a LoRA target).
pub struct GdnMixerGrads<'a> {
    pub conv1d_weight: Option<&'a DeviceBuffer>,
    pub a_log: Option<&'a DeviceBuffer>,
    pub dt_bias: Option<&'a DeviceBuffer>,
    pub norm_weight: Option<&'a DeviceBuffer>,
}

/// Everything [`gdn_mixer_bwd`] needs beyond what it recomputes fresh -
/// exactly the forward's own internal activations, saved only when the
/// caller is a training build (mirrors every other model crate's own
/// `is_train`-gated Acts pattern). The caller's own `gated` (this function's
/// forward return value) is NOT included here - the caller already owns it.
pub struct GdnMixerActs {
    pub shape: GdnShape,
    // conv1d: `x` (dw needs it) and pre-SiLU output (silu_bwd needs it).
    pub ncl_in: DeviceBuffer,
    pub ncl_out: DeviceBuffer,
    // pre-L2-norm query/key, and the (post-SiLU, post-split, un-permuted)
    // value - `gdn_chunk_bwd` never needs `value` token-major (only
    // `value_cm`, below), so this is diagnostic-only: parity-debugging
    // introspection against `tools/goldens/qwen35_gguf_reference_forward.py`
    // (see `qwen35::model::Qwen35::debug_gdn_trace`) needs a token-major `v`
    // it can compare 1:1 against the script's own `v`, and re-deriving it
    // from `value_cm`'s chunk-major layout at the call site is needless
    // work when the un-permuted buffer already exists here for free.
    pub query: DeviceBuffer,
    pub key: DeviceBuffer,
    pub value: DeviceBuffer,
    // bproj (pre-sigmoid), aproj (for gdn_decay_gate_bwd), g_decay
    // (gdn_decay_gate's own output - needed for d_A_log = bias_grad(d_g_decay
    // * g_decay), see that gradient's own derivation in
    // `gdn_decay_gate_bwd.wgsl`'s header).
    pub bproj: DeviceBuffer,
    pub aproj: DeviceBuffer,
    pub g_decay: DeviceBuffer,
    // chunk-major inputs gdn_chunk_bwd itself reads.
    pub query_cm: DeviceBuffer,
    pub key_cm: DeviceBuffer,
    pub value_cm: DeviceBuffer,
    pub beta_cm: DeviceBuffer,
    // gdn_chunk_fwd_train's saved history.
    pub scratch_train: GdnScratchTrainBufs,
    // token-major output (gated RMSNorm's `x`).
    pub out_tok: DeviceBuffer,
    // gated RMSNorm ("norm before gate").
    pub normed: DeviceBuffer,
    pub z: DeviceBuffer,
    pub z_silu: DeviceBuffer,
}

/// One sequence's PERSISTENT Gated-DeltaNet stream state, threaded across
/// successive [`gdn_mixer_stream_fwd`] calls so a prompt can be consumed in
/// several bounded rounds instead of one whole-sequence forward.
///
/// Both buffers are read as this round's INPUT state and overwritten in place
/// with its OUTPUT state, the same convention
/// [`crate::gdn::gdn_recurrent_step`]'s `state` and
/// [`crate::gdn::gdn_causal_conv1d_step`]'s `hist` already use at `n = 1` -
/// and they are bit-for-bit the SAME two buffers those decode primitives
/// thread, so a chunked prefill and a per-token decode can be interleaved on
/// one sequence.
pub struct GdnStream<'a> {
    /// `[bh, dk, dv]` recurrent state: [`crate::gdn::gdn_chunk_fwd`]'s
    /// `initial_state` on the way in, its `final_state` on the way out. Zero
    /// for a fresh sequence.
    pub state: &'a DeviceBuffer,
    /// `[conv_dim, conv_kernel-1]` causal-conv history, channel-major, oldest
    /// tap first - `gdn_causal_conv1d_step`'s own `hist` layout. Zero for a
    /// fresh sequence; after a round it holds the LAST `conv_kernel-1` rows of
    /// that round's `mixed_qkv` (the conv's own input, NOT its output), which
    /// is exactly the window the next round's first token needs.
    pub hist: &'a DeviceBuffer,
}

/// `mixed_qkv = in_proj_qkv(xn1)`, `bproj = in_proj_b(xn1)`, `aproj =
/// in_proj_a(xn1)`, `z = in_proj_z(xn1)` -> `gated`, ready for the caller's
/// own `out_proj`. `n` is the row count (`b*t`); `is_train` gates whether the
/// activations [`gdn_mixer_bwd`] needs are saved.
///
/// Always starts from a FRESH sequence: a zero recurrent state and a zero
/// conv history (`conv1d_fwd`'s own left `pad = K-1`). That is what a training
/// forward wants; a serving prefill that consumes a long prompt in rounds
/// wants [`gdn_mixer_stream_fwd`] instead.
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_fwd(
    g: &Gpu,
    ids: &GdnMixerIds,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    n: u32,
    is_train: bool,
) -> (DeviceBuffer, Option<GdnMixerActs>) {
    gdn_mixer_stream_fwd(g, ids, shape, w, mixed_qkv, bproj, aproj, z, n, is_train, None)
}

/// [`gdn_mixer_fwd`] with an optional persistent [`GdnStream`] - the seam a
/// CHUNKED prefill needs, where round `r > 0` must continue from round
/// `r-1`'s recurrent state and conv tail rather than from zero.
///
/// `None` is [`gdn_mixer_fwd`] exactly (same dispatches, same buffers, same
/// numbers). `Some` changes exactly two things and nothing else:
///
/// 1. **The conv.** `conv1d_fwd`'s left `pad = K-1` IS the zero history, so it
///    cannot express "continue from these `K-1` real values". Instead this
///    prepends the history rows to the round's own `mixed_qkv` and convolves
///    the `K-1+t` extended input with `pad = 0`, which yields exactly `t`
///    outputs whose first `K-1` now see real left context. The new history is
///    the extended input's own last `K-1` rows.
/// 2. **The recurrence.** `gdn_chunk_fwd` already takes `initial_state` and
///    writes `final_state` explicitly; this binds the caller's persistent
///    buffer to the first and copies the result back into it, instead of
///    letting both default to a fresh (zeroed) allocation.
///
/// Requires `shape.gdn.b == 1` (the stream state is per sequence) and
/// `!is_train` (a chunked forward saves no whole-sequence backward history).
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_stream_fwd(
    g: &Gpu,
    ids: &GdnMixerIds,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    n: u32,
    is_train: bool,
    cont: Option<GdnStream>,
) -> (DeviceBuffer, Option<GdnMixerActs>) {
    let gdn = shape.gdn;
    let (conv_dim, key_dim, value_dim, group) = (shape.conv_dim(), shape.key_dim(), shape.value_dim(), shape.group());
    let (nkh, nvh, khd, vhd, kw) = (shape.nkh, gdn.h, gdn.dk, gdn.dv, shape.conv_kernel);
    let (b, t, chunk) = (gdn.b, gdn.t, gdn.chunk);
    let n_chunks = t / chunk;
    if cont.is_some() {
        assert_eq!(b, 1, "gdn_mixer_stream_fwd: a persistent GdnStream is per SEQUENCE, so b must be 1 (got {b})");
        assert!(!is_train, "gdn_mixer_stream_fwd: a streaming (chunked) forward saves no backward history");
    }

    // Depthwise causal conv1d + SiLU (activation AFTER the conv).
    // conv1d.wgsl is NCL ([N,Cin,L]); mixed_qkv is token-major ([B,T,C]).
    let ncl_out = g.storage((n * conv_dim) as u64);
    let ncl_in = match &cont {
        None => {
            let ncl_in = g.storage((n * conv_dim) as u64);
            g.submit(&[], &[g.step(ids.nlc_nchw, &[mixed_qkv, &ncl_in], &[n * conv_dim, conv_dim, t], n * conv_dim)]);
            let conv_shape = Conv1d { n: b, cin: conv_dim, l: t, cout: conv_dim, k: kw, stride: 1, pad: kw - 1, dilation: 1, groups: conv_dim, lo: t };
            g.submit(&[], &[conv1d_fwd(g, &ids.conv, &conv_shape, &ncl_in, w.conv1d_weight, &ncl_out)]);
            ncl_in
        }
        Some(st) => {
            // The K-1 history rows, channel-major -> token-major, so they can
            // be prepended to this round's own token-major rows.
            let kw1 = kw - 1;
            let ext_rows = kw1 + t;
            let hist_tok = g.storage((kw1 * conv_dim) as u64);
            g.submit(&[], &[g.step(ids.nchw_nlc, &[st.hist, &hist_tok], &[kw1 * conv_dim, conv_dim, kw1], kw1 * conv_dim)]);
            // [history | this round], a flat row concatenation (N=H=W=1).
            let ext_tok = g.storage((ext_rows * conv_dim) as u64);
            g.submit(&[], &[g.step(ids.concat2, &[&hist_tok, mixed_qkv, &ext_tok], &[1, kw1 * conv_dim, t * conv_dim, 1, 1], ext_rows * conv_dim)]);
            // The NEXT round's history: the extended input's own last K-1
            // rows (correct even when t < K-1, where some of them are still
            // this round's inherited history). Read before `st.hist` is
            // overwritten below - separate dispatches, submitted in order.
            let tail_tok = g.storage((kw1 * conv_dim) as u64);
            g.submit(
                &[],
                &[
                    g.step(ids.concat_split, &[&ext_tok, &tail_tok], &[1, ext_rows * conv_dim, kw1 * conv_dim, t * conv_dim, 1, 1], kw1 * conv_dim),
                    g.step(ids.nlc_nchw, &[&tail_tok, st.hist], &[kw1 * conv_dim, conv_dim, kw1], kw1 * conv_dim),
                ],
            );
            // pad = 0 over K-1+t inputs gives exactly t outputs, the causal
            // window of each already filled by real left context.
            let ncl_in = g.storage((ext_rows * conv_dim) as u64);
            g.submit(&[], &[g.step(ids.nlc_nchw, &[&ext_tok, &ncl_in], &[ext_rows * conv_dim, conv_dim, ext_rows], ext_rows * conv_dim)]);
            let conv_shape = Conv1d { n: 1, cin: conv_dim, l: ext_rows, cout: conv_dim, k: kw, stride: 1, pad: 0, dilation: 1, groups: conv_dim, lo: t };
            g.submit(&[], &[conv1d_fwd(g, &ids.conv, &conv_shape, &ncl_in, w.conv1d_weight, &ncl_out)]);
            ncl_in
        }
    };
    let ncl_act = g.storage((n * conv_dim) as u64);
    g.submit(&[], &[g.step(ids.silu, &[&ncl_out, &ncl_act], &[n * conv_dim], n * conv_dim)]);
    let mixed_act = g.storage((n * conv_dim) as u64);
    g.submit(&[], &[g.step(ids.nchw_nlc, &[&ncl_act, &mixed_act], &[n * conv_dim, conv_dim, t], n * conv_dim)]);

    // Split into query/key/value - ONE whole-row contiguous split.
    let query = g.storage((n * key_dim) as u64);
    let key = g.storage((n * key_dim) as u64);
    let value = g.storage((n * value_dim) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.concat_split, &[&mixed_act, &query], &[n, conv_dim, key_dim, 0, 1, 1], n * key_dim),
            g.step(ids.concat_split, &[&mixed_act, &key], &[n, conv_dim, key_dim, key_dim, 1, 1], n * key_dim),
            g.step(ids.concat_split, &[&mixed_act, &value], &[n, conv_dim, value_dim, 2 * key_dim, 1, 1], n * value_dim),
        ],
    );

    // L2-normalize query/key - bare l2norm (no learnable scale).
    let query_n = g.storage((n * key_dim) as u64);
    let key_n = g.storage((n * key_dim) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.l2norm_scale, &[&query, w.ones_khd, &query_n], &[n * nkh, khd, f(1e-6)], n * key_dim),
            g.step(ids.l2norm_scale, &[&key, w.ones_khd, &key_n], &[n * nkh, khd, f(1e-6)], n * key_dim),
        ],
    );

    // beta = sigmoid(bproj); g = -exp(A_log)*softplus(aproj+dt_bias).
    let beta = g.storage((n * nvh) as u64);
    let g_decay = g.storage((n * nvh) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.sigmoid, &[bproj, &beta], &[n * nvh], n * nvh),
            g.step(ids.gdn_decay_gate, &[aproj, w.a_log, w.dt_bias, &g_decay], &[n, nvh], n * nvh),
        ],
    );

    // Repeat query/key from linear_num_key_heads to linear_num_value_heads.
    let query_w = g.storage((n * nvh * khd) as u64);
    let key_w = g.storage((n * nvh * khd) as u64);
    g.submit(
        &[],
        &[
            kv_expand_fwd(g, ids.kv_expand, &query_n, &query_w, n, nvh, group, khd, nvh * khd, 0),
            kv_expand_fwd(g, ids.kv_expand, &key_n, &key_w, n, nvh, group, khd, nvh * khd, 0),
        ],
    );

    // Chunk-major permute (token-major -> chunk-major) for gdn_chunk_fwd.
    let permute_fwd = |src: &DeviceBuffer, dim: u32| -> DeviceBuffer {
        let dst = g.storage(b as u64 * nvh as u64 * n_chunks as u64 * chunk as u64 * dim as u64);
        g.submit(
            &[],
            &[g.step(ids.gdn_layout_permute, &[src, &dst], &[b, nvh, n_chunks, chunk, dim, 1], b * nvh * n_chunks * chunk * dim)],
        );
        dst
    };
    let query_cm = permute_fwd(&query_w, khd);
    let key_cm = permute_fwd(&key_w, khd);
    let value_cm = permute_fwd(&value, vhd);
    let g_cm = permute_fwd(&g_decay, 1);
    let beta_cm = permute_fwd(&beta, 1);

    // gdn_chunk_fwd - the chunked-recurrence forward itself. Training builds
    // use gdn_chunk_fwd_train instead: bit-identical out/final_state but
    // additionally saves the per-chunk history gdn_chunk_bwd needs.
    let bh = gdn.bh() as u64;
    let state_len = bh * khd as u64 * vhd as u64;
    // Fresh sequence -> a zero `initial_state` (every `Gpu::storage` here is
    // zero-fresh); a continued one -> the caller's own persistent buffer.
    // `gdn_chunk_fwd` reads `initial_state` exactly once (its first step
    // copies it into `final_state`) and never writes it, so binding the
    // caller's buffer directly is safe.
    let owned_initial = cont.is_none().then(|| g.storage(state_len));
    let initial_state = match (&cont, &owned_initial) {
        (Some(st), _) => st.state,
        (None, Some(zeroed)) => zeroed,
        (None, None) => unreachable!("owned_initial is Some whenever cont is None"),
    };
    let final_state = g.storage(state_len);
    let out_cm = g.storage(gdn.bhc() as u64 * chunk as u64 * vhd as u64);
    let scratch_train = if is_train { Some(GdnScratchTrainBufs::new(g, &gdn)) } else { None };
    if let Some(strain) = &scratch_train {
        let steps = gdn_chunk_fwd_train(
            g,
            &ids.chunk,
            &ids.chunk_bwd,
            &gdn,
            &query_cm,
            &key_cm,
            &value_cm,
            &g_cm,
            &beta_cm,
            initial_state,
            &strain.as_ref(),
            &out_cm,
            &final_state,
        );
        g.submit(&strain.clears(), &steps);
    } else {
        let scratch = GdnScratchBufs::new(g, &gdn);
        let steps = gdn_chunk_fwd(g, &ids.chunk, &gdn, &query_cm, &key_cm, &value_cm, &g_cm, &beta_cm, initial_state, &scratch.as_ref(), &out_cm, &final_state);
        g.submit(&scratch.clears(), &steps);
    }
    // Persist this round's end state for the next one. Done after the loop
    // rather than by aliasing `final_state` to the caller's buffer, so
    // `gdn_chunk_fwd`'s own seeding copy never has the same buffer on both
    // sides of a dispatch.
    if let Some(st) = &cont {
        let len = state_len as u32;
        g.submit(&[], &[g.step(ids.chunk.region_copy, &[&final_state, st.state], &[1, len, len, 0], len)]);
    }

    // Permute back to token-major.
    let out_tok = g.storage((n * value_dim) as u64);
    g.submit(&[], &[g.step(ids.gdn_layout_permute, &[&out_cm, &out_tok], &[b, nvh, n_chunks, chunk, vhd, 0], b * nvh * n_chunks * chunk * vhd)]);

    // Gated RMSNorm ("norm before gate"): normed = RMSNorm(out_tok)*weight,
    // THEN * SiLU(z).
    let normed = g.storage((n * value_dim) as u64);
    let z_silu = g.storage((n * value_dim) as u64);
    let gated = g.storage((n * value_dim) as u64);
    g.submit(
        &[],
        &[
            rmsnorm_fwd(g, &ids.kernels, &out_tok, w.norm_weight, &normed, vhd, n * nvh, shape.rms_eps),
            g.step(ids.silu, &[z, &z_silu], &[n * value_dim], n * value_dim),
            g.step(ids.mul, &[&normed, &z_silu, &gated], &[n * value_dim], n * value_dim),
        ],
    );

    let acts = scratch_train.map(|scratch_train| GdnMixerActs {
        shape: gdn,
        ncl_in,
        ncl_out,
        query,
        key,
        value: value.clone(),
        bproj: bproj.clone(),
        aproj: aproj.clone(),
        g_decay,
        query_cm,
        key_cm,
        value_cm,
        beta_cm,
        scratch_train,
        out_tok,
        normed,
        z: z.clone(),
        z_silu,
    });
    (gated, acts)
}

/// Kernel-pipeline indices the DECODE entry point
/// ([`gdn_mixer_decode_fwd`]) needs beyond [`GdnMixerIds`]. Kept separate
/// rather than folded into that struct because the whole-sequence forward and
/// its backward - every other caller of [`GdnMixerIds`] - dispatch neither of
/// these, and a slot a caller must fill but can never use is exactly the
/// [`crate::block::UNREGISTERED`] hazard.
#[derive(Clone, Copy)]
pub struct GdnMixerDecodeIds {
    /// `causal_conv1d_step.wgsl` - the streaming, one-token-per-sequence conv
    /// that replaces the whole-sequence `conv1d_fwd` in decode.
    pub conv: crate::gdn::GdnConvIds,
    /// `splice.wgsl` - stages one sequence's persistent recurrent state / conv
    /// history into its row of the batch slab, and writes the updated row back
    /// (see [`gdn_mixer_decode_fwd`]'s own doc for why a batch needs staging at
    /// all). Never dispatched at `b == 1`.
    pub splice: usize,
}

/// A serving engine's recurrent state for the whole resident set: one row per
/// sequence of two per-layer pools - the recurrent state and the conv history -
/// so staging a batch is one gather and one scatter whatever its size.
pub struct GdnPoolRows<'a> {
    /// `[rows, state_len]` recurrent states of this layer.
    pub state: &'a DeviceBuffer,
    /// `[rows, hist_len]` conv histories of this layer.
    pub hist: &'a DeviceBuffer,
    /// `[b]` u32: the pool row of each batch row, distinct.
    pub rows: &'a DeviceBuffer,
    /// `pool_rows_gather2.wgsl` and `pool_rows_scatter2.wgsl`.
    pub gather: usize,
    pub scatter: usize,
    /// Run a one-sequence step as the single native launch
    /// ([`gdn_mixer_decode_fused`]) where the device is offered it. Off, the
    /// step is always the nineteen-kernel chain it is gated against.
    pub fuse: bool,
}

/// Where [`gdn_mixer_decode_state_fwd`] finds each batch row's state.
pub enum GdnDecodeState<'a> {
    /// One buffer pair per sequence, in batch-row order. A batch of more than
    /// one is staged by a copy per sequence in and out.
    Streams(&'a [GdnStream<'a>]),
    /// Rows of a pool, staged by a single dispatch in and out.
    Pool(GdnPoolRows<'a>),
}

/// ONE decode token for each of `shape.gdn.b` INDEPENDENT sequences, in one
/// set of dispatches - the batched, decode-shaped sibling of
/// [`gdn_mixer_stream_fwd`], and the Gated-DeltaNet half of a hybrid decoder's
/// batched decode step.
///
/// Takes the caller's already-projected `mixed_qkv`/`bproj`/`aproj`/`z`
/// (`[b, conv_dim]`/`[b, h]`/`[b, h]`/`[b, value_dim]`, one row per sequence)
/// and returns `gated` (`[b, value_dim]`), ready for the caller's own
/// `out_proj` - the same contract [`gdn_mixer_fwd`] has, at `t = 1` and `b`
/// genuinely distinct requests instead of one batch of training rows.
///
/// `streams` holds one [`GdnStream`] per batch row, in batch-row order, and
/// every one is read as this token's input state and OVERWRITTEN with its
/// output state - the same in-place convention the single-sequence primitives
/// use, so a caller may freely mix batched steps, single steps and
/// [`gdn_mixer_stream_fwd`] prefill rounds on the same buffers.
///
/// **Why the state is staged.** Both stateful kernels
/// ([`crate::gdn::gdn_causal_conv1d_step`] and
/// [`crate::gdn::gdn_recurrent_step`]) address the batch through one flat
/// axis - `[N, C, K-1]` and `[b*h, dk, dv]` - so they need the batch's state
/// CONTIGUOUS, while a serving engine owns one buffer per resident sequence and
/// gets an arbitrary subset of them in an arbitrary order each step. This
/// gathers the rows in, runs the batch, and scatters them back. It is real
/// traffic, but it is `b * (state + hist)` words each way against a step that
/// reads every projection weight in the layer, and the alternative - a
/// slot-indirected state binding - is a change to five kernels for a saving
/// that does not show up next to the weight reads.
///
/// At `b == 1` there is nothing to gather: the caller's own buffers are bound
/// directly and the staging dispatches do not exist, so a single-sequence
/// decode step through this function is dispatch-for-dispatch what it was
/// before the function was batched.
///
/// Requires `shape.gdn.t == 1` (one token per sequence) and
/// `streams.len() == shape.gdn.b`.
pub fn gdn_mixer_decode_fwd(
    g: &Gpu,
    ids: &GdnMixerIds,
    dec: &GdnMixerDecodeIds,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    streams: &[GdnStream],
) -> DeviceBuffer {
    gdn_mixer_decode_state_fwd(g, ids, dec, shape, w, mixed_qkv, bproj, aproj, z, &GdnDecodeState::Streams(streams))
}

/// [`gdn_mixer_decode_fwd`] over either way of holding the batch's state
/// ([`GdnDecodeState`]). A pooled batch is staged in ONE dispatch whatever its
/// size - at 32 sequences that is the difference between 2 and 128 dispatches
/// per layer, each a serial node of the step's graph.
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_decode_state_fwd(
    g: &Gpu,
    ids: &GdnMixerIds,
    dec: &GdnMixerDecodeIds,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    batch_state: &GdnDecodeState,
) -> DeviceBuffer {
    let gdn = shape.gdn;
    let (conv_dim, key_dim, value_dim, group) = (shape.conv_dim(), shape.key_dim(), shape.value_dim(), shape.group());
    let (nkh, nvh, khd, vhd, kw) = (shape.nkh, gdn.h, gdn.dk, gdn.dv, shape.conv_kernel);
    let b = gdn.b;
    assert_eq!(gdn.t, 1, "gdn_mixer_decode_fwd is a DECODE step: exactly one token per sequence (got t={})", gdn.t);
    if let GdnDecodeState::Streams(streams) = batch_state {
        assert_eq!(streams.len(), b as usize, "gdn_mixer_decode_fwd: {} GdnStreams for a batch of {b}", streams.len());
    }
    let state_len = nvh * khd * vhd;
    let hist_len = conv_dim * (kw - 1);

    // A pooled batch the native kernel serves runs in place on its rows: no
    // staging in or out, and the whole step is one launch.
    if let GdnDecodeState::Pool(pool) = batch_state {
        if pool.fuse {
            if let Some(gated) = gdn_mixer_decode_pooled_fused(g, shape, w, mixed_qkv, bproj, aproj, z, pool) {
                return gated;
            }
        }
    }

    // Stage the batch's persistent state contiguously - see this function's
    // own doc. One sequence over its own buffers binds them directly instead.
    let (state, hist) = match batch_state {
        GdnDecodeState::Pool(pool) => {
            let st = g.storage((b * state_len) as u64);
            let hi = g.storage((b * hist_len).max(1) as u64);
            g.submit(&[], &[g.step(pool.gather, &[pool.state, pool.hist, pool.rows, &st, &hi], &[b, state_len, hist_len], b * (state_len + hist_len))]);
            (st, hi)
        }
        GdnDecodeState::Streams(streams) if b > 1 => {
            let st = g.storage((b * state_len) as u64);
            let hi = g.storage((b * hist_len).max(1) as u64);
            let mut s = Vec::with_capacity(2 * b as usize);
            for (i, sm) in streams.iter().enumerate() {
                let row = i as u32;
                s.push(g.step(dec.splice, &[sm.state, &st], &[state_len, row * state_len], state_len));
                if hist_len > 0 {
                    s.push(g.step(dec.splice, &[sm.hist, &hi], &[hist_len, row * hist_len], hist_len));
                }
            }
            g.submit(&[], &s);
            (st, hi)
        }
        GdnDecodeState::Streams(streams) => (streams[0].state.clone(), streams[0].hist.clone()),
    };

    // 1. Streaming causal conv1d + SiLU (activation after the conv). No
    // NLC/NCHW round trip: `gdn_causal_conv1d_step`'s x/y are `[N, C]`, which
    // is already `mixed_qkv`'s own layout at one token per sequence.
    let conv_out = g.storage((b * conv_dim) as u64);
    let conv_shape = crate::gdn::GdnConvShape { n: b, c: conv_dim, k: kw };
    g.submit(&[], &[crate::gdn::gdn_causal_conv1d_step(g, &dec.conv, &conv_shape, mixed_qkv, w.conv1d_weight, &hist, &conv_out)]);
    let mixed_act = g.storage((b * conv_dim) as u64);
    g.submit(&[], &[g.step(ids.silu, &[&conv_out, &mixed_act], &[b * conv_dim], b * conv_dim)]);

    // 2. Split each row into query/key/value - a whole-row split, so the row
    // count is the only thing the batch changes.
    let query = g.storage((b * key_dim) as u64);
    let key = g.storage((b * key_dim) as u64);
    let value = g.storage((b * value_dim) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.concat_split, &[&mixed_act, &query], &[b, conv_dim, key_dim, 0, 1, 1], b * key_dim),
            g.step(ids.concat_split, &[&mixed_act, &key], &[b, conv_dim, key_dim, key_dim, 1, 1], b * key_dim),
            g.step(ids.concat_split, &[&mixed_act, &value], &[b, conv_dim, value_dim, 2 * key_dim, 1, 1], b * value_dim),
        ],
    );

    // 3. Per-head L2-normalize query/key.
    let query_n = g.storage((b * key_dim) as u64);
    let key_n = g.storage((b * key_dim) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.l2norm_scale, &[&query, w.ones_khd, &query_n], &[b * nkh, khd, f(1e-6)], b * key_dim),
            g.step(ids.l2norm_scale, &[&key, w.ones_khd, &key_n], &[b * nkh, khd, f(1e-6)], b * key_dim),
        ],
    );

    // 4. beta = sigmoid(bproj); g_decay = decay-gate(aproj).
    let beta = g.storage((b * nvh) as u64);
    let g_decay = g.storage((b * nvh) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.sigmoid, &[bproj, &beta], &[b * nvh], b * nvh),
            g.step(ids.gdn_decay_gate, &[aproj, w.a_log, w.dt_bias, &g_decay], &[b, nvh], b * nvh),
        ],
    );

    // 5. Repeat query/key from the key-head count to the value-head count.
    let query_w = g.storage((b * nvh * khd) as u64);
    let key_w = g.storage((b * nvh * khd) as u64);
    g.submit(
        &[],
        &[
            kv_expand_fwd(g, ids.kv_expand, &query_n, &query_w, b, nvh, group, khd, nvh * khd, 0),
            kv_expand_fwd(g, ids.kv_expand, &key_n, &key_w, b, nvh, group, khd, nvh * khd, 0),
        ],
    );

    // 6. The recurrent state update. `gdn_recurrent_step` consumes only
    // `bh = b*h`, so a batch row is indistinguishable from an extra head to
    // every kernel it dispatches - which is exactly why the staging above has
    // to get the row order right, and why this module's own decode test
    // compares per-sequence state, not just the layer's output.
    let bh = gdn.bh();
    let kv_mem = g.storage((bh * vhd) as u64);
    let sub_out = g.storage((bh * vhd) as u64);
    let scratch = GdnRecurrentScratch { kv_mem: &kv_mem, sub_out: &sub_out };
    let out_bh = g.storage((bh * vhd) as u64);
    g.submit(&[], &gdn_recurrent_step(g, &ids.chunk, &gdn, &query_w, &key_w, &value, &g_decay, &beta, &state, &scratch, &out_bh));

    // 7. Gated RMSNorm (norm before gate).
    let normed = g.storage((b * value_dim) as u64);
    let z_silu = g.storage((b * value_dim) as u64);
    let gated = g.storage((b * value_dim) as u64);
    g.submit(
        &[],
        &[
            rmsnorm_fwd(g, &ids.kernels, &out_bh, w.norm_weight, &normed, vhd, bh, shape.rms_eps),
            g.step(ids.silu, &[z, &z_silu], &[b * value_dim], b * value_dim),
            g.step(ids.mul, &[&normed, &z_silu, &gated], &[b * value_dim], b * value_dim),
        ],
    );

    // Return each sequence's evolved state to where it lives.
    match batch_state {
        GdnDecodeState::Pool(pool) => {
            g.submit(&[], &[g.step(pool.scatter, &[&state, &hist, pool.rows, pool.state, pool.hist], &[b, state_len, hist_len], b * (state_len + hist_len))]);
        }
        GdnDecodeState::Streams(streams) if b > 1 => {
            let mut s = Vec::with_capacity(2 * b as usize);
            for (i, sm) in streams.iter().enumerate() {
                let row = i as u32;
                s.push(g.step(ids.concat_split, &[&state, sm.state], &[1, b * state_len, state_len, row * state_len, 1, 1], state_len));
                if hist_len > 0 {
                    s.push(g.step(ids.concat_split, &[&hist, sm.hist], &[1, b * hist_len, hist_len, row * hist_len, 1, 1], hist_len));
                }
            }
            g.submit(&[], &s);
        }
        GdnDecodeState::Streams(_) => {}
    }
    gated
}

/// [`gdn_mixer_decode_fwd`] for ONE sequence as a single native launch - the
/// `gdn_decode` kernel, which does the conv, the gates, the delta-rule state
/// update and the gated norm that function dispatches nineteen kernels for, and
/// leaves `stream`'s state and conv window updated exactly as it does. Returns
/// `gated` (`[1, value_dim]`).
///
/// `None` - nothing submitted, `stream` untouched - where the device is not
/// offered the kernel, or the shape is outside its block shape: one sequence,
/// 128-wide key and value heads, a 4-tap conv, and at most three value heads
/// per key head. The caller then runs [`gdn_mixer_decode_fwd`], which is what
/// this is gated byte-for-byte against (`tests/gdn_decode_native.rs`).
pub fn gdn_mixer_decode_fused(
    g: &Gpu,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    stream: &GdnStream,
) -> Option<DeviceBuffer> {
    gdn_mixer_rows_fused(g, shape, w, mixed_qkv, bproj, aproj, z, stream, 1)
}

/// Whether `shape` (one sequence, any `t`) is one the native recurrent kernel's
/// block shape serves: 128-wide key and value heads, a 4-tap conv, and at most
/// three value heads per key head. A caller that must commit to the kernel for
/// a whole pass up front asks this and [`Gpu::has_fused`] before starting.
fn rows_fused_shape_ok(shape: &GdnMixerShape) -> bool {
    let gdn = shape.gdn;
    gdn.b == 1 && gdn.dk == 128 && gdn.dv == 128 && shape.conv_kernel == 4 && shape.nkh != 0 && gdn.h.is_multiple_of(shape.nkh) && (1..=3).contains(&shape.group())
}

/// Whether [`gdn_mixer_rows_fused`] will be taken for this `shape` on `g`: the
/// device is offered the kernel and the shape is one it serves.
pub fn rows_fused_offered(g: &Gpu, shape: &GdnMixerShape) -> bool {
    g.has_fused(gpu_core::Fused::GdnDecode) && rows_fused_shape_ok(shape)
}

/// [`gdn_mixer_decode_fused`] for `rows` CONSECUTIVE tokens of one sequence in a
/// single launch: the rows are taken in order inside the kernel, each from the
/// state and conv window the previous one left, so the outputs, the state and
/// the window are exactly what `rows` single steps produce. `shape.gdn.t` must
/// be `rows`; the inputs and the returned `gated` are `[rows, ..]`.
///
/// This is the recurrent half of a speculative verify round: the rows are
/// dependent, so they cannot be a batch, and running them through the chunked
/// parallel form would verify on different arithmetic than plain decode runs.
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_rows_fused(
    g: &Gpu,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    stream: &GdnStream,
    rows: u32,
) -> Option<DeviceBuffer> {
    let gdn = shape.gdn;
    if rows == 0 || gdn.t != rows || !rows_fused_shape_ok(shape) {
        return None;
    }
    let gated = g.storage(rows as u64 * shape.value_dim() as u64);
    let params = [shape.nkh, gdn.h, shape.group(), f(1e-6), f(shape.rms_eps), f(1.0f32 / (gdn.dk as f32).sqrt()), rows, 0];
    let step = g.fused_step(
        gpu_core::Fused::GdnDecode,
        &[mixed_qkv, w.conv1d_weight, stream.hist, bproj, aproj, w.a_log, w.dt_bias, stream.state, z, w.norm_weight, &gated],
        &params,
    )?;
    g.submit(&[], &[step]);
    Some(gated)
}

/// [`gdn_mixer_decode_state_fwd`] for a BATCH over pool rows as a single native
/// launch - the `gdn_decode_pool` kernel, which runs [`gdn_mixer_decode_fused`]'s
/// body for every (key head, sequence) on that sequence's pool row of `state` and
/// `hist`, in place. Returns `gated` (`[b, value_dim]`); each row is what
/// [`gdn_mixer_decode_fused`] and the nineteen-kernel chain produce for it.
///
/// `None` - nothing submitted, the pools untouched - where the device is not
/// offered the kernel or the shape is outside its block shape (128-wide key and
/// value heads, a 4-tap conv, at most three value heads per key head).
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_decode_pooled_fused(
    g: &Gpu,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    pool: &GdnPoolRows,
) -> Option<DeviceBuffer> {
    let gdn = shape.gdn;
    if gdn.t != 1 || gdn.dk != 128 || gdn.dv != 128 || shape.conv_kernel != 4 || shape.nkh == 0 || gdn.h % shape.nkh != 0 {
        return None;
    }
    let gated = g.storage(gdn.b as u64 * shape.value_dim() as u64);
    let params = [shape.nkh, gdn.h, shape.group(), f(1e-6), f(shape.rms_eps), f(1.0f32 / (gdn.dk as f32).sqrt()), gdn.b, 0];
    let step = g.fused_step(
        gpu_core::Fused::GdnDecodePool,
        &[mixed_qkv, w.conv1d_weight, pool.hist, bproj, aproj, w.a_log, w.dt_bias, pool.state, z, w.norm_weight, &gated, pool.rows],
        &params,
    )?;
    g.submit(&[], &[step]);
    Some(gated)
}

/// Reverse of [`gdn_mixer_fwd`]: `d_gated` (the caller's own `out_proj`
/// backward output) -> `(d_mixed_qkv, d_bproj, d_aproj, d_z)`, for the
/// caller's own `in_proj_*` backward. `n` must match the forward call's own.
pub fn gdn_mixer_bwd(g: &Gpu, ids: &GdnMixerIds, shape: &GdnMixerShape, w: &GdnMixerWeights, gw: &GdnMixerGrads, la: &GdnMixerActs, d_gated: &DeviceBuffer, n: u32) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let gdn = shape.gdn;
    let (conv_dim, key_dim, value_dim, group) = (shape.conv_dim(), shape.key_dim(), shape.value_dim(), shape.group());
    let (nkh, nvh, khd, vhd, kw) = (shape.nkh, gdn.h, gdn.dk, gdn.dv, shape.conv_kernel);
    let (b, t, chunk) = (gdn.b, gdn.t, gdn.chunk);
    let n_chunks = t / chunk;

    // ---- gated RMSNorm backward: gated = normed*z_silu; z_silu = silu(z); normed = rmsnorm(out_tok) ----
    let d_normed = g.storage((n * value_dim) as u64);
    let d_z_silu = g.storage((n * value_dim) as u64);
    let d_z = g.storage((n * value_dim) as u64);
    let d_out_tok = g.storage((n * value_dim) as u64);
    {
        let inv = g.storage((n * nvh) as u64);
        let mut s = vec![
            g.step(ids.mul, &[d_gated, &la.z_silu, &d_normed], &[n * value_dim], n * value_dim),
            g.step(ids.mul, &[d_gated, &la.normed, &d_z_silu], &[n * value_dim], n * value_dim),
            g.step(ids.silu_bwd, &[&la.z, &d_z_silu, &d_z], &[n * value_dim], n * value_dim),
        ];
        s.extend(rmsnorm_bwd(g, &ids.kernels, &la.out_tok, w.norm_weight, &d_normed, &d_out_tok, &inv, gw.norm_weight, vhd, n * nvh, shape.rms_eps));
        g.submit(&[], &s);
    }

    // ---- permute back to chunk-major (forward used to_chunk_major=0; backward flips it) ----
    let d_out_cm = g.storage(gdn.bhc() as u64 * gdn.chunk as u64 * vhd as u64);
    g.submit(&[], &[g.step(ids.gdn_layout_permute, &[&d_out_tok, &d_out_cm], &[b, nvh, n_chunks, chunk, vhd, 1], b * nvh * n_chunks * chunk * vhd)]);

    // ---- gdn_chunk_bwd - the chunked-recurrence backward itself ----
    let bh = gdn.bh() as u64;
    let bhc = gdn.bhc() as u64;
    let cw = gdn.chunk as u64;
    let dk = gdn.dk as u64;
    let dv = gdn.dv as u64;
    let d_final_state = g.storage(bh * dk * dv); // no incremental decode continuation -> zero
    let d_initial_state = g.storage(bh * dk * dv); // discarded (no earlier chunk upstream)
    let d_query_cm = g.storage(bhc * cw * dk);
    let d_key_cm = g.storage(bhc * cw * dk);
    let d_value_cm = g.storage(bhc * cw * dv);
    let d_g_cm = g.storage(bhc * cw);
    let d_beta_cm = g.storage(bhc * cw);
    let bwd_scratch = GdnBwdScratchBufs::new(g, &gdn);
    {
        let steps = gdn_chunk_bwd(
            g,
            &ids.chunk,
            &ids.chunk_bwd,
            &gdn,
            &la.query_cm,
            &la.key_cm,
            &la.value_cm,
            &la.beta_cm,
            &la.scratch_train.as_ref(),
            &d_out_cm,
            &d_final_state,
            &bwd_scratch.as_ref(),
            &d_query_cm,
            &d_key_cm,
            &d_value_cm,
            &d_g_cm,
            &d_beta_cm,
            &d_initial_state,
        );
        // Every output with more than one contributing forward use is
        // explicitly zeroed by the caller (see gdn_chunk_bwd's own doc):
        // `d_final_state` (external gradient, none), plus d_query/d_key/d_beta.
        let mut clears = bwd_scratch.clears();
        clears.push(&d_final_state);
        clears.push(&d_query_cm);
        clears.push(&d_key_cm);
        clears.push(&d_beta_cm);
        g.submit(&clears, &steps);
    }

    // ---- permute back to token-major (forward used to_chunk_major=1; backward flips it) ----
    let permute_bwd = |src_cm: &DeviceBuffer, dim: u32| -> DeviceBuffer {
        let dst = g.storage(n as u64 * nvh as u64 * dim as u64);
        g.submit(&[], &[g.step(ids.gdn_layout_permute, &[src_cm, &dst], &[b, nvh, n_chunks, chunk, dim, 0], b * nvh * n_chunks * chunk * dim)]);
        dst
    };
    let d_query_w = permute_bwd(&d_query_cm, khd);
    let d_key_w = permute_bwd(&d_key_cm, khd);
    let d_value = permute_bwd(&d_value_cm, vhd);
    let d_g_decay = permute_bwd(&d_g_cm, 1);
    let d_beta = permute_bwd(&d_beta_cm, 1);

    // ---- kv_expand backward (group-sum, overwrite - no accumulate needed) ----
    let d_query_n = g.storage((n * key_dim) as u64);
    let d_key_n = g.storage((n * key_dim) as u64);
    g.submit(
        &[],
        &[
            kv_expand_bwd(g, ids.kv_expand_bwd, &d_query_w, &d_query_n, n, nvh, group, khd, nvh * khd, 0),
            kv_expand_bwd(g, ids.kv_expand_bwd, &d_key_w, &d_key_n, n, nvh, group, khd, nvh * khd, 0),
        ],
    );

    // ---- beta/g_decay backward into bproj/aproj, A_log/dt_bias reductions ----
    let d_bproj = g.storage((n * nvh) as u64);
    let d_aproj = g.storage((n * nvh) as u64);
    {
        let mut s = vec![
            g.step(ids.sigmoid_bwd, &[&la.bproj, &d_beta, &d_bproj], &[n * nvh], n * nvh),
            g.step(ids.gdn_decay_gate_bwd, &[&la.aproj, w.a_log, w.dt_bias, &d_g_decay, &d_aproj], &[n, nvh], n * nvh),
        ];
        // d_A_log[h] = sum_row d_g_decay[row,h]*g_decay[row,h]; d_dt_bias[h] = sum_row d_aproj[row,h].
        // Neither is ever a LoRA target - Frozen under a LoRA build, same as
        // any other non-targeted weight - so skip these reductions entirely
        // when frozen (no grad buffer to write into).
        let mul_tmp = g.storage((n * nvh) as u64);
        s.push(g.step(ids.mul, &[&d_g_decay, &la.g_decay, &mul_tmp], &[n * nvh], n * nvh));
        if let Some(ga) = gw.a_log {
            s.push(g.step(ids.bias_grad, &[&mul_tmp, ga], &[n, nvh], nvh));
        }
        if let Some(gdt) = gw.dt_bias {
            s.push(g.step(ids.bias_grad, &[&d_aproj, gdt], &[n, nvh], nvh));
        }
        g.submit(&[], &s);
    }

    // ---- L2-norm backward ----
    let d_query = g.storage((n * key_dim) as u64);
    let d_key = g.storage((n * key_dim) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.l2norm_scale_dx, &[&la.query, w.ones_khd, &d_query_n, &d_query], &[n * nkh, khd, f(1e-6)], n * key_dim),
            g.step(ids.l2norm_scale_dx, &[&la.key, w.ones_khd, &d_key_n, &d_key], &[n * nkh, khd, f(1e-6)], n * key_dim),
        ],
    );

    // ---- qkv split backward (concat2 x2: the 3-way split's adjoint) ----
    let d_qk = g.storage((n * 2 * key_dim) as u64);
    let d_mixed_act = g.storage((n * conv_dim) as u64);
    g.submit(
        &[],
        &[
            g.step(ids.concat2, &[&d_query, &d_key, &d_qk], &[n, key_dim, key_dim, 1, 1], n * 2 * key_dim),
            g.step(ids.concat2, &[&d_qk, &d_value, &d_mixed_act], &[n, 2 * key_dim, value_dim, 1, 1], n * conv_dim),
        ],
    );

    // ---- conv1d + SiLU backward ----
    let d_ncl_act = g.storage((n * conv_dim) as u64);
    let d_ncl_out = g.storage((n * conv_dim) as u64);
    let d_ncl_in = g.storage((n * conv_dim) as u64);
    let d_mixed_qkv = g.storage((n * conv_dim) as u64);
    let conv_shape = Conv1d { n: b, cin: conv_dim, l: t, cout: conv_dim, k: kw, stride: 1, pad: kw - 1, dilation: 1, groups: conv_dim, lo: t };
    {
        let mut s = vec![
            g.step(ids.nlc_nchw, &[&d_mixed_act, &d_ncl_act], &[n * conv_dim, conv_dim, t], n * conv_dim),
            g.step(ids.silu_bwd, &[&la.ncl_out, &d_ncl_act, &d_ncl_out], &[n * conv_dim], n * conv_dim),
        ];
        s.extend(conv1d_bwd(g, &ids.conv, &conv_shape, &d_ncl_out, &la.ncl_in, w.conv1d_weight, Some(&d_ncl_in), gw.conv1d_weight));
        s.push(g.step(ids.nchw_nlc, &[&d_ncl_in, &d_mixed_qkv], &[n * conv_dim, conv_dim, t], n * conv_dim));
        g.submit(&[], &s);
    }

    (d_mixed_qkv, d_bproj, d_aproj, d_z)
}
