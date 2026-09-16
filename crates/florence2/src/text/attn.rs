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

use super::lora::{lora_bwd, lora_fwd, trainable, LoraCtx};

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

    // ---- training (M6): additive to the inference path above, which stays
    // byte-for-byte the graph M3's real-checkpoint parity tests gate. ----

    /// Same math as [`Self::forward`], with an optional LoRA delta fused
    /// onto each of the four projections (`lora.is_some()` implies every
    /// one of `q_proj`/`k_proj`/`v_proj`/`out_proj` under `prefix` carries a
    /// `.lora_a`/`.lora_b` pair - the caller's `ParamStore` guarantees this
    /// by construction, see `crate::train`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_train(&self, gpu: &Gpu, k: &BartAttnKernelIds, ps: &paramstore::ParamStore, prefix: &str, x_q: &DeviceBuffer, x_kv: &DeviceBuffer, tq: u32, tkv: u32, lora: Option<&LoraCtx>) -> &DeviceBuffer {
        let d = self.d_model;
        let qw = ps.w(&format!("{prefix}.q_proj.weight"));
        let qb = ps.w(&format!("{prefix}.q_proj.bias"));
        let kw = ps.w(&format!("{prefix}.k_proj.weight"));
        let kb = ps.w(&format!("{prefix}.k_proj.bias"));
        let vw = ps.w(&format!("{prefix}.v_proj.weight"));
        let vb = ps.w(&format!("{prefix}.v_proj.bias"));
        let ow = ps.w(&format!("{prefix}.out_proj.weight"));
        let ob = ps.w(&format!("{prefix}.out_proj.bias"));

        let mut s1 = vec![
            gpu.step(k.matmul_rows, &[x_q, qw, &self.q], &[tq, d, d], tq.div_ceil(8) * d),
            gpu.step(k.bias_add, &[&self.q, qb], &[tq, d], tq * d),
        ];
        if let Some(l) = lora {
            lora_fwd(&mut s1, gpu, l.ids, ps, &format!("{prefix}.q_proj.weight"), l.cfg, l.scratch, x_q, &self.q, tq, d, d);
        }
        s1.push(gpu.step(k.matmul_rows, &[x_kv, kw, &self.k], &[tkv, d, d], tkv.div_ceil(8) * d));
        s1.push(gpu.step(k.bias_add, &[&self.k, kb], &[tkv, d], tkv * d));
        if let Some(l) = lora {
            lora_fwd(&mut s1, gpu, l.ids, ps, &format!("{prefix}.k_proj.weight"), l.cfg, l.scratch, x_kv, &self.k, tkv, d, d);
        }
        s1.push(gpu.step(k.matmul_rows, &[x_kv, vw, &self.v], &[tkv, d, d], tkv.div_ceil(8) * d));
        s1.push(gpu.step(k.bias_add, &[&self.v, vb], &[tkv, d], tkv * d));
        if let Some(l) = lora {
            lora_fwd(&mut s1, gpu, l.ids, ps, &format!("{prefix}.v_proj.weight"), l.cfg, l.scratch, x_kv, &self.v, tkv, d, d);
        }
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

        let mut s3 = vec![gpu.step(k.matmul_rows, &[&self.ctx, ow, &self.out], &[tq, d, d], tq.div_ceil(8) * d)];
        if let Some(l) = lora {
            lora_fwd(&mut s3, gpu, l.ids, ps, &format!("{prefix}.out_proj.weight"), l.cfg, l.scratch, &self.ctx, &self.out, tq, d, d);
        }
        s3.push(gpu.step(k.bias_add, &[&self.out, ob], &[tq, d], tq * d));
        gpu.submit(&[], &s3);

        &self.out
    }

    /// Backward of [`Self::forward_train`]. `d_out`: grad of this block's
    /// returned output. ASSIGNS `d_x_q`/`d_x_kv` (the caller accumulates
    /// them into its own residual-stream grad, exactly as
    /// [`Self::forward_train`]'s caller adds this block's output into its
    /// own residual sum) and, when the relevant base tensor is trainable,
    /// ACCUMULATES the four projections' weight/bias grads (plus the LoRA
    /// adapter grads whenever `lora` is `Some`). Reads `probs`/`ctx` -
    /// [`Self::forward_train`] must have been the last call on `self`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(&self, gpu: &Gpu, bwd: &BartAttnBwdIds, ps: &paramstore::ParamStore, prefix: &str, x_q: &DeviceBuffer, x_kv: &DeviceBuffer, d_out: &DeviceBuffer, d_x_q: &DeviceBuffer, d_x_kv: &DeviceBuffer, scratch: &BartAttnBwdScratch, tq: u32, tkv: u32, lora: Option<&LoraCtx>) {
        let d = self.d_model;
        let hd = self.head_dim;
        let qw_n = format!("{prefix}.q_proj.weight");
        let kw_n = format!("{prefix}.k_proj.weight");
        let vw_n = format!("{prefix}.v_proj.weight");
        let ow_n = format!("{prefix}.out_proj.weight");

        // ---- out_proj ----
        let mut s1 = Vec::new();
        if trainable(ps, &format!("{prefix}.out_proj.bias")) {
            s1.push(gpu.step(bwd.bias_grad, &[d_out, ps.g(&format!("{prefix}.out_proj.bias"))], &[tq, d], d));
        }
        if trainable(ps, &ow_n) {
            s1.push(gpu.step(bwd.matmul_dw, &[d_out, &self.ctx, ps.g(&ow_n)], &[tq, d, d], d * d));
        }
        s1.push(gpu.step(bwd.matmul_dx, &[d_out, ps.w(&ow_n), &scratch.d_ctx], &[tq, d, d, 0], tq * d));
        if let Some(l) = lora {
            lora_bwd(&mut s1, gpu, l.ids, ps, &ow_n, l.cfg, l.scratch, d_out, &self.ctx, &scratch.d_ctx, tq, d, d);
        }
        gpu.submit(&[], &s1);

        // ---- attention backward: d_ctx -> d_scores -> d_q/d_k/d_v ----
        // Causal (self-attn, tq==tkv) and cross (encoder self-attn /
        // decoder cross-attn) kernels have DIFFERENT `Params` struct
        // layouts (causal has no separate t_dec/t_enc split) - not just
        // different kernel indices, so the param arrays cannot be shared.
        let s2 = if self.causal {
            let p_v = [1, self.heads, tq, hd, d, 0, d];
            let p_qk = [1, self.heads, tq, hd, d, 0, 0];
            vec![
                gpu.step(bwd.dscores_causal, &[&scratch.d_ctx, &self.v, &self.probs, &scratch.d_scores], &p_v, self.heads * tq),
                gpu.step(bwd.dv_causal, &[&self.probs, &scratch.d_ctx, &scratch.d_v], &p_v, self.heads * tkv * hd),
                gpu.step(bwd.dq_causal, &[&scratch.d_scores, &self.k, &scratch.d_q], &p_qk, self.heads * tq * hd),
                gpu.step(bwd.dk_causal, &[&scratch.d_scores, &self.q, &scratch.d_k], &p_qk, self.heads * tkv * hd),
            ]
        } else {
            let p_v = [1, self.heads, tq, tkv, hd, d, 0, d];
            let p_qk = [1, self.heads, tq, tkv, hd, d, d, 0, 0];
            vec![
                gpu.step(bwd.dscores_cross, &[&scratch.d_ctx, &self.v, &self.probs, &scratch.d_scores], &p_v, self.heads * tq),
                gpu.step(bwd.dv_cross, &[&self.probs, &scratch.d_ctx, &scratch.d_v], &p_v, self.heads * tkv * hd),
                gpu.step(bwd.dq_cross, &[&scratch.d_scores, &self.k, &scratch.d_q], &p_qk, self.heads * tq * hd),
                gpu.step(bwd.dk_cross, &[&scratch.d_scores, &self.q, &scratch.d_k], &p_qk, self.heads * tkv * hd),
            ]
        };
        gpu.submit(&[], &s2);

        // ---- q/k/v projections ----
        let mut s3 = Vec::new();
        if trainable(ps, &format!("{prefix}.q_proj.bias")) {
            s3.push(gpu.step(bwd.bias_grad, &[&scratch.d_q, ps.g(&format!("{prefix}.q_proj.bias"))], &[tq, d], d));
        }
        if trainable(ps, &qw_n) {
            s3.push(gpu.step(bwd.matmul_dw, &[&scratch.d_q, x_q, ps.g(&qw_n)], &[tq, d, d], d * d));
        }
        s3.push(gpu.step(bwd.matmul_dx, &[&scratch.d_q, ps.w(&qw_n), d_x_q], &[tq, d, d, 0], tq * d));
        if let Some(l) = lora {
            lora_bwd(&mut s3, gpu, l.ids, ps, &qw_n, l.cfg, l.scratch, &scratch.d_q, x_q, d_x_q, tq, d, d);
        }

        if trainable(ps, &format!("{prefix}.k_proj.bias")) {
            s3.push(gpu.step(bwd.bias_grad, &[&scratch.d_k, ps.g(&format!("{prefix}.k_proj.bias"))], &[tkv, d], d));
        }
        if trainable(ps, &kw_n) {
            s3.push(gpu.step(bwd.matmul_dw, &[&scratch.d_k, x_kv, ps.g(&kw_n)], &[tkv, d, d], d * d));
        }
        s3.push(gpu.step(bwd.matmul_dx, &[&scratch.d_k, ps.w(&kw_n), d_x_kv], &[tkv, d, d, 0], tkv * d));
        if let Some(l) = lora {
            lora_bwd(&mut s3, gpu, l.ids, ps, &kw_n, l.cfg, l.scratch, &scratch.d_k, x_kv, d_x_kv, tkv, d, d);
        }
        gpu.submit(&[], &s3);

        // v's dx ACCUMULATES onto d_x_kv (k's dx just assigned it above) -
        // its own submit so d_x_kv's write from the k-projection above is
        // ordered before this accumulate.
        let mut s4 = Vec::new();
        if trainable(ps, &format!("{prefix}.v_proj.bias")) {
            s4.push(gpu.step(bwd.bias_grad, &[&scratch.d_v, ps.g(&format!("{prefix}.v_proj.bias"))], &[tkv, d], d));
        }
        if trainable(ps, &vw_n) {
            s4.push(gpu.step(bwd.matmul_dw, &[&scratch.d_v, x_kv, ps.g(&vw_n)], &[tkv, d, d], d * d));
        }
        s4.push(gpu.step(bwd.matmul_dx, &[&scratch.d_v, ps.w(&vw_n), d_x_kv], &[tkv, d, d, 1], tkv * d));
        if let Some(l) = lora {
            lora_bwd(&mut s4, gpu, l.ids, ps, &vw_n, l.cfg, l.scratch, &scratch.d_v, x_kv, d_x_kv, tkv, d, d);
        }
        gpu.submit(&[], &s4);
    }
}

/// Backward-kernel indices for [`BartAttn::backward`]: `causal`/`cross`
/// mirror `forward`'s own `softmax_causal`/`softmax` split - the causal
/// four cover a decoder's `self_attn`, the cross four cover an encoder's
/// `self_attn` and every `encoder_attn` (non-causal either way).
pub struct BartAttnBwdIds {
    pub dscores_causal: usize,
    pub dq_causal: usize,
    pub dk_causal: usize,
    pub dv_causal: usize,
    pub dscores_cross: usize,
    pub dq_cross: usize,
    pub dk_cross: usize,
    pub dv_cross: usize,
    pub matmul_dx: usize,
    pub matmul_dw: usize,
    pub bias_grad: usize,
}

impl BartAttnBwdIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> BartAttnBwdIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        BartAttnBwdIds {
            dscores_causal: k("attn_bwd_dscores"),
            dq_causal: k("attn_bwd_dq"),
            dk_causal: k("attn_bwd_dk"),
            dv_causal: k("attn_bwd_dv"),
            dscores_cross: k("attn_bwd_dscores_cross"),
            dq_cross: k("attn_bwd_dq_cross"),
            dk_cross: k("attn_bwd_dk_cross"),
            dv_cross: k("attn_bwd_dv_cross"),
            matmul_dx: k("matmul_dx"),
            matmul_dw: k("matmul_dw"),
            bias_grad: k("bias_grad"),
        }
    }
}

/// Backward scratch for one [`BartAttn::backward`] call, reused
/// sequentially across every attention instance in the model - same reuse
/// discipline as `text::lora::LoraScratch`. Sized at the largest
/// `(tq, tkv)` any instance in the model uses.
pub struct BartAttnBwdScratch {
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_ctx: DeviceBuffer,
}

impl BartAttnBwdScratch {
    pub fn new(gpu: &Gpu, heads: u32, d_model: u32, max_tq: u32, max_tkv: u32) -> BartAttnBwdScratch {
        BartAttnBwdScratch {
            d_q: gpu.storage((max_tq * d_model) as u64),
            d_k: gpu.storage((max_tkv * d_model) as u64),
            d_v: gpu.storage((max_tkv * d_model) as u64),
            d_scores: gpu.storage((heads * max_tq * max_tkv) as u64),
            d_ctx: gpu.storage((max_tq * d_model) as u64),
        }
    }
}
