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
    gdn_chunk_bwd, gdn_chunk_fwd, gdn_chunk_fwd_train, GdnBwdIds, GdnBwdScratchBufs, GdnIds, GdnScratchBufs, GdnScratchTrainBufs, GdnShape,
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
    // pre-L2-norm query/key.
    pub query: DeviceBuffer,
    pub key: DeviceBuffer,
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

/// `mixed_qkv = in_proj_qkv(xn1)`, `bproj = in_proj_b(xn1)`, `aproj =
/// in_proj_a(xn1)`, `z = in_proj_z(xn1)` -> `gated`, ready for the caller's
/// own `out_proj`. `n` is the row count (`b*t`); `is_train` gates whether the
/// activations [`gdn_mixer_bwd`] needs are saved.
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
    let gdn = shape.gdn;
    let (conv_dim, key_dim, value_dim, group) = (shape.conv_dim(), shape.key_dim(), shape.value_dim(), shape.group());
    let (nkh, nvh, khd, vhd, kw) = (shape.nkh, gdn.h, gdn.dk, gdn.dv, shape.conv_kernel);
    let (b, t, chunk) = (gdn.b, gdn.t, gdn.chunk);
    let n_chunks = t / chunk;

    // Depthwise causal conv1d + SiLU (activation AFTER the conv).
    // conv1d.wgsl is NCL ([N,Cin,L]); mixed_qkv is token-major ([B,T,C]).
    let ncl_in = g.storage((n * conv_dim) as u64);
    g.submit(&[], &[g.step(ids.nlc_nchw, &[mixed_qkv, &ncl_in], &[n * conv_dim, conv_dim, t], n * conv_dim)]);
    let conv_shape = Conv1d { n: b, cin: conv_dim, l: t, cout: conv_dim, k: kw, stride: 1, pad: kw - 1, dilation: 1, groups: conv_dim, lo: t };
    let ncl_out = g.storage((n * conv_dim) as u64);
    g.submit(&[], &[conv1d_fwd(g, &ids.conv, &conv_shape, &ncl_in, w.conv1d_weight, &ncl_out)]);
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
    let initial_state = g.storage(bh * khd as u64 * vhd as u64);
    let final_state = g.storage(bh * khd as u64 * vhd as u64);
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
            &initial_state,
            &strain.as_ref(),
            &out_cm,
            &final_state,
        );
        g.submit(&strain.clears(), &steps);
    } else {
        let scratch = GdnScratchBufs::new(g, &gdn);
        let steps = gdn_chunk_fwd(g, &ids.chunk, &gdn, &query_cm, &key_cm, &value_cm, &g_cm, &beta_cm, &initial_state, &scratch.as_ref(), &out_cm, &final_state);
        g.submit(&scratch.clears(), &steps);
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
            rmsnorm_fwd(g, &ids.kernels, &out_tok, w.norm_weight, &normed, vhd, n * nvh),
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
        s.extend(rmsnorm_bwd(g, &ids.kernels, &la.out_tok, w.norm_weight, &d_normed, &d_out_tok, &inv, gw.norm_weight, vhd, n * nvh));
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
