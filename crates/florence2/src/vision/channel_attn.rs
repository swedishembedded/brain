// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DaViT's `ChannelBlock` channel attention: channel-GROUPS attend to each
//! other, contracting over the spatial-token axis `N` - the inverse of
//! ordinary attention (tokens attending to tokens, contracting over
//! `head_dim`). Not a `model::vit::chunked_attn_fwd` reuse: that engine's
//! fused-qkv buffer interleaves q/k/v per row (one token's row holds all
//! its channels), while this shape needs them in disjoint per-segment
//! blocks instead - built from generic primitives directly:
//!
//! One whole-buffer `nlc_nchw`-style transpose turns the fused `[N,3C]` qkv
//! into `[3C,N]`, which makes every group's `Cg = C/groups` rows of Q, K,
//! and V a plain CONTIGUOUS range (`step_sliced`, no per-group striding).
//! `model::block`'s generic `matmul` computes `out[m,n] = sum_k A[m,k] *
//! B[n,k]` (confirmed from `arcface::train`'s cosine-similarity matmul) -
//! `A @ B^T` - which maps directly onto both of channel attention's GEMMs
//! once operands are in the right transposed/untransposed form (see the
//! per-group comments in [`ChannelAttn::forward`]). Softmax uses
//! `attn_softmax_cross`, NOT the plain `attn_softmax` the name would
//! suggest - `attn_softmax`'s own WGSL source hardcodes causal masking
//! (`for j in 0..=i`, no toggle at all despite param name confusion
//! elsewhere in this codebase's comments) and silently zeroed roughly half
//! of every `[Cg,Cg]` score matrix during development, costing a real
//! numerical-parity failure (cosine 0.987, not the expected >=0.999) before
//! being caught. `attn_softmax_cross` is genuinely non-causal (full-row
//! softmax, `bsz`/`n_heads`/`t_dec`/`t_enc` params) - the right general
//! row-wise softmax for this and any other non-causal small-matrix need.
//!
//! The reference's `1/sqrt(N)` scale (`ChannelAttention.forward`: `q = q *
//! N**-0.5`, scaling by the spatial TOKEN COUNT, not `head_dim` the way
//! every other attention variant in this repo scales) is folded into the
//! qkv weight's Q-ROWS at import time instead of a runtime scale kernel -
//! `N` is a compile-time-known constant per DaViT stage. **The caller
//! building this module's `ParamStore` source MUST pre-scale
//! `{prefix}.fn.qkv.weight`'s first `C` rows and `{prefix}.fn.qkv.bias`'s
//! first `C` entries by `N^-0.5` before construction** - see
//! [`ChannelAttn::new`]'s doc.

use gpu_core::{DeviceBuffer, Gpu, Step};

pub struct ChannelAttnKernelIds {
    pub layernorm: usize,
    pub matmul_rows: usize,
    pub matmul: usize,
    pub bias_add: usize,
    pub nlc_nchw: usize,
    pub nchw_nlc: usize,
    pub attn_softmax_cross: usize,
    pub add2: usize,
}

impl ChannelAttnKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> ChannelAttnKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        ChannelAttnKernelIds {
            layernorm: k("layernorm"),
            matmul_rows: k("matmul_rows"),
            matmul: k("matmul"),
            bias_add: k("bias_add"),
            nlc_nchw: k("nlc_nchw"),
            nchw_nlc: k("nchw_nlc"),
            attn_softmax_cross: k("attn_softmax_cross"),
            add2: k("add2"),
        }
    }
}

pub struct ChannelAttn {
    prefix: String,
    dim: u32,
    groups: u32,
    cg: u32,
    n: u32,
    eps: f32,
    normed: DeviceBuffer,
    qkv: DeviceBuffer,
    qkv_t: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    v_small: DeviceBuffer,
    ctx_t: DeviceBuffer,
    ctx: DeviceBuffer,
    proj_out: DeviceBuffer,
    out: DeviceBuffer,
}

impl ChannelAttn {
    /// `n`: spatial token count (`grid_h * grid_w`) - both the attention
    /// contraction length AND the scale `1/sqrt(n)` the caller must have
    /// already folded into `{prefix}.fn.qkv.weight`/`.bias`'s first `dim`
    /// rows/entries (the Q portion) before building the `ParamStore` this
    /// reads from. `dim` must be an exact multiple of `groups`.
    pub fn new(gpu: &Gpu, prefix: &str, dim: u32, groups: u32, n: u32, eps: f32) -> ChannelAttn {
        assert!(dim.is_multiple_of(groups), "ChannelAttn: dim {dim} must be an exact multiple of groups {groups}");
        let cg = dim / groups;
        ChannelAttn {
            prefix: prefix.to_string(),
            dim,
            groups,
            cg,
            n,
            eps,
            normed: gpu.storage((n * dim) as u64),
            qkv: gpu.storage(3 * (n * dim) as u64),
            qkv_t: gpu.storage(3 * (n * dim) as u64),
            scores: gpu.storage((cg * cg) as u64),
            probs: gpu.storage((cg * cg) as u64),
            v_small: gpu.storage((n * cg) as u64),
            ctx_t: gpu.storage((dim * n) as u64),
            ctx: gpu.storage((n * dim) as u64),
            proj_out: gpu.storage((n * dim) as u64),
            out: gpu.storage((n * dim) as u64),
        }
    }

    pub fn forward(&self, gpu: &Gpu, k: &ChannelAttnKernelIds, ps: &paramstore::ParamStore, x_in: &DeviceBuffer) -> &DeviceBuffer {
        let (c, n, cg, groups) = (self.dim, self.n, self.cg, self.groups);

        // PreNorm's `norm` is a sibling of `fn`, same layout window_attn.rs
        // found against the real checkpoint's own tensor names.
        let ln = model::block::LayerNormIds::resolve_fwd(gpu, k.layernorm);
        let norm_w = ps.w(&format!("{}.norm.weight", self.prefix));
        let norm_b = ps.w(&format!("{}.norm.bias", self.prefix));
        let qkv_w = ps.w(&format!("{}.fn.qkv.weight", self.prefix));
        let qkv_b = ps.w(&format!("{}.fn.qkv.bias", self.prefix));
        let proj_w = ps.w(&format!("{}.fn.proj.weight", self.prefix));
        let proj_b = ps.w(&format!("{}.fn.proj.bias", self.prefix));

        // LN -> fused qkv [N,3C] (Q's weight rows already N^-0.5-scaled by
        // the caller) -> one whole-buffer transpose to [3C,N].
        let s1: Vec<Step> = vec![
            model::block::layernorm_fwd(gpu, &ln, x_in, norm_w, norm_b, &self.normed, c, n, self.eps),
            gpu.step(k.matmul_rows, &[&self.normed, qkv_w, &self.qkv], &[n, c, 3 * c], n.div_ceil(8) * 3 * c),
            gpu.step(k.bias_add, &[&self.qkv, qkv_b], &[n, 3 * c], n * 3 * c),
            gpu.step(k.nlc_nchw, &[&self.qkv, &self.qkv_t], &[n * 3 * c, 3 * c, n], n * 3 * c),
        ];
        gpu.submit(&[], &s1);

        // Per group: contiguous [Cg,N] slices of qkv_t (rows g*Cg..(g+1)*Cg
        // within the Q/K/V third), two GEMMs via matmul's confirmed A@B^T
        // convention, one small V-untranspose, writing straight into the
        // group's contiguous row range of ctx_t ([C,N], not [N,C] - avoids
        // a strided write, one final transpose covers every group at once).
        let mut s2: Vec<Step> = Vec::new();
        for g in 0..groups {
            let q_off = (g * cg * n) as u64;
            let k_off = ((c + g * cg) * n) as u64;
            let v_off = ((2 * c + g * cg) * n) as u64;
            let slice_len = (cg * n) as u64;

            // scores[Cg,Cg] = Q_g_t[Cg,N] @ K_g_t[Cg,N]^T = Q_g^T @ K_g
            // (Q_g_t already carries the N^-0.5 scale via its pre-scaled weight).
            s2.push(gpu.step_sliced(
                k.matmul,
                &[&self.qkv_t, &self.qkv_t, &self.scores],
                &[(q_off, slice_len), (k_off, slice_len), (0, (cg * cg) as u64)],
                &[cg, n, cg],
                cg * cg,
            ));
            // Row-wise softmax over [1 head, Cg rows, Cg cols], genuinely
            // non-causal (attn_softmax_cross, not attn_softmax - see the
            // module doc).
            s2.push(gpu.step(k.attn_softmax_cross, &[&self.scores, &self.probs], &[1, 1, cg, cg], cg));
            // V_g = transpose(V_g_t[Cg,N]) -> [N,Cg] (untransposed - needed
            // as matmul's second operand below, see the module doc).
            s2.push(gpu.step_sliced(k.nchw_nlc, &[&self.qkv_t, &self.v_small], &[(v_off, slice_len), (0, (n * cg) as u64)], &[n * cg, cg, n], n * cg));
            // ctx_g_t[Cg,N] = probs[Cg,Cg] @ V_g[N,Cg]^T = probs @ V_g^T,
            // written directly into ctx_t's contiguous rows for this group.
            s2.push(gpu.step_sliced(k.matmul, &[&self.probs, &self.v_small, &self.ctx_t], &[(0, (cg * cg) as u64), (0, (n * cg) as u64), (q_off, slice_len)], &[cg, cg, n], cg * n));
        }
        gpu.submit(&[], &s2);

        // One transpose back [C,N] -> [N,C], then proj + bias + residual -
        // identical shape to window_attn.rs's tail.
        let s3: Vec<Step> = vec![
            gpu.step(k.nchw_nlc, &[&self.ctx_t, &self.ctx], &[n * c, c, n], n * c),
            gpu.step(k.matmul_rows, &[&self.ctx, proj_w, &self.proj_out], &[n, c, c], n.div_ceil(8) * c),
            gpu.step(k.bias_add, &[&self.proj_out, proj_b], &[n, c], n * c),
            gpu.step(k.add2, &[x_in, &self.proj_out, &self.out], &[n * c], n * c),
        ];
        gpu.submit(&[], &s3);
        &self.out
    }
}
