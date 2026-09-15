// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One shared attention block for all three flavors Florence-2's BART text
//! model needs (`Florence2Attention`, `Florence2EncoderLayer.self_attn`,
//! `Florence2DecoderLayer.{self_attn,encoder_attn}`): separate `q_proj`/
//! `k_proj`/`v_proj`/`out_proj` `nn.Linear` (with bias), scale folded into
//! the score kernel itself (`attn_scores_cross`'s own `inverseSqrt(head_dim)`
//! - NOT reapplied here, which would double the reference's `q * scaling`).
//!
//! Built entirely from the existing generic cross-attention kernels
//! (`attn_scores_cross`/`attn_softmax{,_cross}`/`attn_apply_cross`), never a
//! FUSED qkv buffer: BART ships separate q/k/v weights, and these kernels'
//! `q_stride`/`kv_stride`/`*_off` params are already general enough to read
//! three independent `[T,d_model]` buffers directly - `attn_scores_cross`
//! only ever touches its `kv` argument's K region, `attn_apply_cross` only
//! ever touches its `kv` argument's V region, so passing K and V as two
//! DIFFERENT physical buffers to that one generic slot (once per kernel
//! call) needs no fused layout at all. `causal` swaps only the softmax
//! kernel (`attn_softmax`, unconditionally causal - see
//! `crate::vision::channel_attn`'s module doc for why that kernel's name is
//! misleading elsewhere in this codebase) vs `attn_softmax_cross`
//! (genuinely non-causal); scores/apply are identical either way.
//!
//! No KV cache: every call recomputes Q/K/V (and, for cross-attention,
//! re-projects the FIXED encoder memory) from scratch over the CURRENT
//! prefix length. Buffers are preallocated at `max_tq`/`max_tkv` (the same
//! "allocate for the ceiling, dispatch over the live length" pattern
//! `model::vit`'s `VitScratch`/`VitBlockCache` use for their `max_span`) so
//! a growing decode prefix never reallocates - only the per-call `tq`/`tkv`
//! passed to [`BartAttn::forward`] changes. Florence-2's grounding outputs
//! are short (a handful of `<loc_N>` tokens plus a phrase), so recomputing
//! the whole prefix every step is cheap enough that a KV cache is a real,
//! tracked optimization gap (Kronos's own prefill/decode-step pattern is
//! the template to follow when it's worth doing) rather than a correctness
//! requirement.

use gpu_core::{DeviceBuffer, Gpu};

pub struct BartAttnKernelIds {
    pub matmul_rows: usize,
    pub bias_add: usize,
    pub scores: usize,
    pub softmax_causal: usize,
    pub softmax: usize,
    pub apply: usize,
}

impl BartAttnKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> BartAttnKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        BartAttnKernelIds {
            matmul_rows: k("matmul_rows"),
            bias_add: k("bias_add"),
            scores: k("attn_scores_cross"),
            softmax_causal: k("attn_softmax"),
            softmax: k("attn_softmax_cross"),
            apply: k("attn_apply_cross"),
        }
    }
}

pub struct BartAttn {
    heads: u32,
    head_dim: u32,
    d_model: u32,
    causal: bool,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    out: DeviceBuffer,
}

impl BartAttn {
    /// `causal`: true for a decoder's `self_attn`, false for an encoder's
    /// `self_attn` and every `encoder_attn` (cross-attention is never
    /// causal - the reference never masks it). `max_tq`/`max_tkv`: the
    /// ceiling this instance's scratch is sized for (self-attention passes
    /// the same bound for both; cross-attention passes the decoder's max
    /// length for `max_tq` and the encoder's FIXED length for `max_tkv`).
    pub fn new(gpu: &Gpu, heads: u32, d_model: u32, causal: bool, max_tq: u32, max_tkv: u32) -> BartAttn {
        let head_dim = d_model / heads;
        BartAttn {
            heads,
            head_dim,
            d_model,
            causal,
            q: gpu.storage((max_tq * d_model) as u64),
            k: gpu.storage((max_tkv * d_model) as u64),
            v: gpu.storage((max_tkv * d_model) as u64),
            scores: gpu.storage((heads * max_tq * max_tkv) as u64),
            probs: gpu.storage((heads * max_tq * max_tkv) as u64),
            ctx: gpu.storage((max_tq * d_model) as u64),
            out: gpu.storage((max_tq * d_model) as u64),
        }
    }

    /// `prefix`: e.g. `"language_model.model.encoder.layers.0.self_attn"`
    /// (its `{q,k,v,out}_proj.{weight,bias}` are read from `ps`). `x_q`:
    /// query-side hidden states, `[tq,d_model]`. `x_kv`: key/value-side
    /// hidden states - pass `x_q` itself for self-attention, or the fixed
    /// encoder output for cross-attention. Returns the projected attention
    /// output, `[tq,d_model]` - NOT residual-added or normed (the caller's
    /// layer does that, matching the reference's own module boundary).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, gpu: &Gpu, k: &BartAttnKernelIds, ps: &paramstore::ParamStore, prefix: &str, x_q: &DeviceBuffer, x_kv: &DeviceBuffer, tq: u32, tkv: u32) -> &DeviceBuffer {
        let d = self.d_model;
        let qw = ps.w(&format!("{prefix}.q_proj.weight"));
        let qb = ps.w(&format!("{prefix}.q_proj.bias"));
        let kw = ps.w(&format!("{prefix}.k_proj.weight"));
        let kb = ps.w(&format!("{prefix}.k_proj.bias"));
        let vw = ps.w(&format!("{prefix}.v_proj.weight"));
        let vb = ps.w(&format!("{prefix}.v_proj.bias"));
        let ow = ps.w(&format!("{prefix}.out_proj.weight"));
        let ob = ps.w(&format!("{prefix}.out_proj.bias"));

        let s1 = vec![
            gpu.step(k.matmul_rows, &[x_q, qw, &self.q], &[tq, d, d], tq.div_ceil(8) * d),
            gpu.step(k.bias_add, &[&self.q, qb], &[tq, d], tq * d),
            gpu.step(k.matmul_rows, &[x_kv, kw, &self.k], &[tkv, d, d], tkv.div_ceil(8) * d),
            gpu.step(k.bias_add, &[&self.k, kb], &[tkv, d], tkv * d),
            gpu.step(k.matmul_rows, &[x_kv, vw, &self.v], &[tkv, d, d], tkv.div_ceil(8) * d),
            gpu.step(k.bias_add, &[&self.v, vb], &[tkv, d], tkv * d),
        ];
        gpu.submit(&[], &s1);

        let sp = [1, self.heads, tq, tkv, self.head_dim, d, d, 0, 0];
        let softmax_kind = if self.causal { k.softmax_causal } else { k.softmax };
        let softmax_params: Vec<u32> = if self.causal { vec![1, self.heads, tq] } else { vec![1, self.heads, tq, tkv] };
        let s2 = vec![
            gpu.step(k.scores, &[&self.q, &self.k, &self.scores], &sp, self.heads * tq * tkv),
            gpu.step(softmax_kind, &[&self.scores, &self.probs], &softmax_params, self.heads * tq),
            gpu.step(k.apply, &[&self.probs, &self.v, &self.ctx], &[1, self.heads, tq, tkv, self.head_dim, d, 0, d], self.heads * tq * self.head_dim),
        ];
        gpu.submit(&[], &s2);

        let s3 = vec![
            gpu.step(k.matmul_rows, &[&self.ctx, ow, &self.out], &[tq, d, d], tq.div_ceil(8) * d),
            gpu.step(k.bias_add, &[&self.out, ob], &[tq, d], tq * d),
        ];
        gpu.submit(&[], &s3);

        &self.out
    }
}
