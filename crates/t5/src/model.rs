// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The T5 encoder forward graph — pure dispatch assembly over shared kernels.
//!
//! ```text
//! x0    = embed(ids)                                        [B*T, D]
//! bias  = permute(embed(bucket_ids, rel_bias))              [H, T, T]   (once)
//! per block l:
//!   xn   = RMSNorm(x_l)                                     (no bias, no mean)
//!   qkv  = xn @ Wqkv^T          (fused [3*inner, D] — fused at import)
//!   scr  = q.k^T * 1.0 + bias   (NO 1/sqrt(d_kv) — see below)
//!   ctx  = softmax_bidir(scr) . v
//!   res  = x_l + ctx @ Wo^T                                 (residual UNSCALED)
//!   fn_  = RMSNorm(res)
//!   h    = gelu_new(fn_ @ Wi0^T) * (fn_ @ Wi1^T)            (gated GELU)
//!   x_l+1= res + h @ Wo_ff^T
//! hidden = RMSNorm(x_L)                                     = last_hidden_state
//! ```
//!
//! The four ways T5 differs from every decoder in this workspace, and what each
//! costs if it is assumed instead of checked (each was verified against the
//! reference by `tools/goldens/t5_dump_reference.py`, which records the numbers in its
//! manifest):
//!
//! 1. **RMSNorm, no bias, no residual rescale.** `rmsnorm_eps` is exactly
//!    `T5LayerNorm` (`w * x * rsqrt(mean(x²)+eps)`), so no new kernel and no
//!    host copy. The residual is a plain `add2`.
//! 2. **Relative position bias, not RoPE.** A learned `[num_buckets, heads]`
//!    table gathered through a bucketing of `key - query`, computed once in
//!    block 0 and shared by all 24. There is no RoPE and no absolute position
//!    embedding, so no rotary kernel is dispatched at all. The bucket table is
//!    host integer math ([`crate::hostbias`]); the gather is the `embed`
//!    kernel and the `(q,k,H) -> (H,q,k)` permutation is `nlc_nchw`.
//!    brain's `rel_shift` is Transformer-XL's *shift* of an existing score
//!    slab — a different mechanism, not reusable here (see `hostbias`).
//! 3. **No attention scaling.** T5 folds `1/sqrt(d_kv)` into its
//!    initialisation, so the scores are a bare `q.k^T`. This is why the
//!    attention uses `attn_scores_bidir_bias` (which takes the multiplier as a
//!    Param) with `scale = 1.0` rather than `attn_scores_bidir` (which hardcodes
//!    `1/sqrt(head_dim)`). Getting it wrong is silent: the reference dump
//!    measures the wrong variant at max|d| 7.0e+01, not a crash.
//! 4. **Gated GELU FFN**, `gelu_new(wi_0(x)) * wi_1(x)`. `gelu_new` is HF's
//!    NewGELUActivation — the tanh form brain's `gelu.wgsl` already computes —
//!    and the gate is `mul`, whose header documents exactly this composition.
//!
//! **Attention mask.** diffusers' `FluxPipeline._get_t5_prompt_embeds` calls the
//! encoder with NO `attention_mask`, so right-pad positions are attended as
//! ordinary keys; that is the contract implemented here and the run the goldens
//! gate. It is not a no-op the way CLIP's causal isolation is — the dumper
//! measures 4.5 max|d| on *content* rows between the masked and unmasked runs —
//! so a masked variant is a real (unimplemented) feature, not a rounding
//! difference. See the crate docs.
//!
//! SSA: every stage writes a fresh buffer, which doubles as the activation
//! cache the deferred backward needs. The two shared score slabs are the
//! documented exception, at their allocation site.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::T5Config;

const K_EMBED_TILE: usize = 0;
const K_EMBED: usize = 1;
const K_NLC_NCHW: usize = 2;
const K_RMSNORM: usize = 3;
const K_MATMUL: usize = 4;
const K_MATMUL_REG3: usize = 5;
const K_SCORES: usize = 6;
const K_SOFTMAX: usize = 7;
const K_APPLY: usize = 8;
const K_ADD2: usize = 9;
const K_GELU: usize = 10;
const K_MUL: usize = 11;
const K_RMSNORM_ROWS: usize = 12;

/// The encoder's kernels. Forward only — the backward workstream appends its
/// own, which is why nothing here may be reordered (every `K_*` above is a
/// position in this list).
pub const PIPELINES: &[(&str, &str)] = &[
    ("embed_tile", kernels::EMBED_TILE),
    ("embed", kernels::EMBED),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("attn_scores_bidir_bias", kernels::ATTN_SCORES_BIDIR_BIAS),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("add2", kernels::ADD2),
    ("gelu", kernels::GELU),
    ("mul", kernels::MUL),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
];

/// One block's SSA activations.
pub struct BlockBufs {
    pub attn_norm: DeviceBuffer,
    /// Fused `[N, 3*inner]` — q at 0, k at `inner`, v at `2*inner`.
    pub qkv: DeviceBuffer,
    pub ctx: DeviceBuffer,
    pub attn_out: DeviceBuffer,
    pub res: DeviceBuffer,
    pub ff_norm: DeviceBuffer,
    pub wi0: DeviceBuffer,
    pub wi1: DeviceBuffer,
    /// `gelu_new(wi0)` — kept separate from `gated` so the deferred backward
    /// has both the pre-activation (`wi0`) and the gate factor.
    pub act: DeviceBuffer,
    pub gated: DeviceBuffer,
    pub ff_out: DeviceBuffer,
}

/// Per-block taps the parity test replays (block 0 in the goldens).
#[derive(Clone, Copy, Debug)]
pub enum Tap {
    AttnNorm,
    /// Fused `[N, 3*inner]`: q at `[.., 0..inner]`, k at `[.., inner..2*inner]`,
    /// v at `[.., 2*inner..]`. Head-major within each region, matching T5's
    /// `view(B, T, heads, d_kv)`.
    Qkv,
    Ctx,
    AttnOut,
    AttnRes,
    FfNorm,
    Wi0,
    Wi1,
    Gated,
    FfOut,
}

pub struct T5Encoder {
    pub gpu: Gpu,
    pub cfg: T5Config,
    pub ps: ParamStore,
    b: u32,
    t: u32,
    tokens: DeviceBuffer,
    /// `[T*T]` u32 relative-position bucket ids — uploaded once at build time.
    buckets: DeviceBuffer,
    /// `[T*T, heads]` gather of `rel_bias.weight`, before the permute.
    bias_gather: DeviceBuffer,
    /// `[heads, T, T]` additive attention bias, shared by every block.
    bias: DeviceBuffer,
    /// `x[0]` = embedding output; `x[i+1]` = output of block `i`.
    x: Vec<DeviceBuffer>,
    blocks: Vec<BlockBufs>,
    /// Shared score/probability slabs — the only non-SSA buffers here, and
    /// deliberately so: one `[B, 64, T, T]` f32 slab is 8.4 MB at T=128 and
    /// 134 MB at FLUX's T=512, so a per-block softmax cache would be 3.2 GB
    /// for a 24-block tower next to 19 GB of fp32 weights. A backward should
    /// recompute per chunk (`block::chunked_bidir_bwd`) rather than cache these.
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    hidden: DeviceBuffer,
    steps: Vec<Step>,
}

impl T5Encoder {
    /// Build an inference encoder on an existing device (tests pass
    /// `gpu_core::testgpu::dev`). Every parameter is `Frozen`: no gradient
    /// buffers, no reverse step list.
    pub fn new_on(
        gpu: Gpu,
        cfg: T5Config,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> T5Encoder {
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = b as u64 * t as u64;
        let d = cfg.d_model as u64;
        let inner = cfg.inner() as u64;
        let ff = cfg.d_ff as u64;
        let tt = t as u64 * t as u64;
        let slab = b as u64 * cfg.heads as u64 * tt;
        let tokens = gpu.buffer(
            "t5_tokens",
            n * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let buckets = gpu.buffer(
            "t5_buckets",
            tt * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        gpu.write(&buckets, &crate::hostbias::buckets(t, cfg.rel_buckets, cfg.rel_max_distance));
        let blocks: Vec<BlockBufs> = (0..cfg.layers)
            .map(|_| BlockBufs {
                attn_norm: gpu.storage(n * d),
                qkv: gpu.storage(n * 3 * inner),
                ctx: gpu.storage(n * inner),
                attn_out: gpu.storage(n * d),
                res: gpu.storage(n * d),
                ff_norm: gpu.storage(n * d),
                wi0: gpu.storage(n * ff),
                wi1: gpu.storage(n * ff),
                act: gpu.storage(n * ff),
                gated: gpu.storage(n * ff),
                ff_out: gpu.storage(n * d),
            })
            .collect();
        let mut m = T5Encoder {
            bias_gather: gpu.storage(tt * cfg.heads as u64),
            bias: gpu.storage(cfg.heads as u64 * tt),
            x: (0..=cfg.layers).map(|_| gpu.storage(n * d)).collect(),
            blocks,
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            hidden: gpu.storage(n * d),
            tokens,
            buckets,
            gpu,
            cfg,
            ps,
            b,
            t,
            steps: Vec::new(),
        };
        m.steps = m.build_steps();
        m
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG3, false)
    }

    /// RMSNorm with the coalesced workgroup-per-row kernel wherever the device
    /// supports workgroup reductions (19.4x on a P40 — see `rmsnorm_rows.wgsl`),
    /// else the per-element reference. Same Params and same math either way;
    /// the selection is `model::block::rms_variant`'s (the RMSNorm twin of
    /// `ln_variant`), so the policy lives in `backend_api::select`
    /// (`Op::RmsNorm`) and there is one copy of it for every model.
    fn rmsnorm(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, rows: u32) -> Step {
        let d = self.cfg.d_model;
        let (kind, threads) =
            block::rms_variant(&self.gpu, K_RMSNORM, Some(K_RMSNORM_ROWS), rows, d);
        // `rmsnorm_eps` / `rmsnorm_rows` Params: [d_model, rows, eps(f32 bits)].
        self.gpu.step(kind, &[x, w, out], &[d, rows, f(self.cfg.eps)], threads)
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let (b, t) = (self.b, self.t);
        let n = b * t;
        let d = c.d_model;
        let ff = c.d_ff;
        let inner = c.inner();
        let heads = c.heads;
        let hd = c.d_kv;
        let tt = t * t;
        let mut s = Vec::new();

        // ---- token embedding, tiled over vocab so each `shared.weight`
        // binding stays under the backend's max-binding size (GL: 128 MB; the
        // XXL table is 526 MB). `embed_tile` Params: [d_model, seq_len, v0,
        // v_count]; bufs [tokens(u32), emb tile, out].
        let dw = d as u64;
        for (v0, cnt) in block::vocab_tiles_on(g, c.vocab as u64, dw) {
            s.push(g.step_sliced(
                K_EMBED_TILE,
                &[&self.tokens, self.w("shared.weight"), &self.x[0]],
                &[(0, 0), (v0 as u64 * dw, cnt as u64 * dw), (0, 0)],
                &[d, n, v0, cnt],
                n * d,
            ));
        }

        // ---- relative-position bias, computed ONCE and shared by every block.
        // `embed` Params: [d_model, seq_len]; here the "tokens" are the bucket
        // ids and the "d_model" is the head count, so the gather yields
        // [T*T, heads].
        s.push(g.step(
            K_EMBED,
            &[&self.buckets, self.w("rel_bias.weight"), &self.bias_gather],
            &[heads, tt],
            tt * heads,
        ));
        // `nlc_nchw` Params: [total, c, hw] — y[(ch*hw)+l] = x[l*c + ch], i.e.
        // exactly the reference's `.permute(2, 0, 1)` from (q, k, H) to
        // (H, q, k). The score kernel indexes bias[(h*T + i)*T + j].
        s.push(g.step(K_NLC_NCHW, &[&self.bias_gather, &self.bias], &[heads * tt, heads, tt], heads * tt));

        for l in 0..c.layers as usize {
            let bb = &self.blocks[l];
            let p = format!("blocks.{l}");
            s.push(self.rmsnorm(&self.x[l], self.w(&format!("{p}.attn_norm.weight")), &bb.attn_norm, n));

            let (mk, mt) = self.gemm(n, 3 * inner);
            s.push(g.step(
                mk,
                &[&bb.attn_norm, self.w(&format!("{p}.qkv.weight")), &bb.qkv],
                &[n, d, 3 * inner],
                mt,
            ));

            // Bidirectional attention with the learned bias and NO scaling.
            // `attn_scores_bidir_bias` Params:
            //   [bsz, n_heads, tcols, head_dim, qkv_stride, q_off, k_off, scale(f32 bits)]
            s.push(g.step(
                K_SCORES,
                &[&bb.qkv, &self.bias, &self.scores],
                &[b, heads, t, hd, 3 * inner, 0, inner, f(1.0)],
                b * heads * t * t,
            ));
            // `attn_softmax_bidir` Params: [bsz, n_heads, tcols].
            s.push(g.step(K_SOFTMAX, &[&self.scores, &self.probs], &[b, heads, t], b * heads * t));
            // `attn_apply_bidir` Params:
            //   [bsz, n_heads, tcols, head_dim, qkv_stride, v_off, d_model]
            // where `d_model` is the CONTEXT width (heads*d_kv), not `cfg.d_model`.
            s.push(g.step(
                K_APPLY,
                &[&self.probs, &bb.qkv, &bb.ctx],
                &[b, heads, t, hd, 3 * inner, 2 * inner, inner],
                b * heads * t * hd,
            ));

            let (mk, mt) = self.gemm(n, d);
            s.push(g.step(mk, &[&bb.ctx, self.w(&format!("{p}.o.weight")), &bb.attn_out], &[n, inner, d], mt));
            // `add2` Params: a single `total`. The residual is UNSCALED — T5
            // does not rescale the stream at a block boundary.
            s.push(g.step(K_ADD2, &[&self.x[l], &bb.attn_out, &bb.res], &[n * d], n * d));

            s.push(self.rmsnorm(&bb.res, self.w(&format!("{p}.ff_norm.weight")), &bb.ff_norm, n));
            let (mk, mt) = self.gemm(n, ff);
            s.push(g.step(mk, &[&bb.ff_norm, self.w(&format!("{p}.wi_0.weight")), &bb.wi0], &[n, d, ff], mt));
            let (mk, mt) = self.gemm(n, ff);
            s.push(g.step(mk, &[&bb.ff_norm, self.w(&format!("{p}.wi_1.weight")), &bb.wi1], &[n, d, ff], mt));
            // `gelu` Params: a single `total` — the tanh form, == HF `gelu_new`.
            s.push(g.step(K_GELU, &[&bb.wi0, &bb.act], &[n * ff], n * ff));
            // `mul` Params: a single `n`. GEGLU is documented in its header as
            // exactly this composition (gelu into a fresh SSA buffer, then mul).
            s.push(g.step(K_MUL, &[&bb.act, &bb.wi1, &bb.gated], &[n * ff], n * ff));
            let (mk, mt) = self.gemm(n, d);
            s.push(g.step(mk, &[&bb.gated, self.w(&format!("{p}.wo.weight")), &bb.ff_out], &[n, ff, d], mt));
            s.push(g.step(K_ADD2, &[&bb.res, &bb.ff_out, &self.x[l + 1]], &[n * d], n * d));
        }

        s.push(self.rmsnorm(&self.x[c.layers as usize], self.w("final_norm.weight"), &self.hidden, n));
        s
    }

    /// Set the token ids, `[B*T]` row-major (right-padded with the T5 pad id 0,
    /// the way `FluxPipeline` tokenizes).
    pub fn set_tokens(&self, ids: &[u32]) {
        assert_eq!(ids.len(), (self.b * self.t) as usize, "token count");
        assert!(
            ids.iter().all(|&i| i < self.cfg.vocab),
            "token id >= vocab {}",
            self.cfg.vocab
        );
        self.gpu.write(&self.tokens, ids);
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    // ---- parity / inference taps ----
    fn n(&self) -> usize {
        (self.b * self.t) as usize
    }

    /// The `[heads, T, T]` additive attention bias (the reference's
    /// `position_bias[0]`).
    pub fn read_position_bias(&self) -> Vec<f32> {
        self.gpu.read(&self.bias, (self.cfg.heads * self.t * self.t) as usize)
    }

    /// `x[0]` = embedding output; `x[i+1]` = output of block `i`.
    pub fn read_x(&self, i: usize) -> Vec<f32> {
        self.gpu.read(&self.x[i], self.n() * self.cfg.d_model as usize)
    }

    pub fn read_block_tap(&self, l: usize, tap: Tap) -> Vec<f32> {
        let bb = &self.blocks[l];
        let c = &self.cfg;
        let (buf, w) = match tap {
            Tap::AttnNorm => (&bb.attn_norm, c.d_model),
            Tap::Qkv => (&bb.qkv, 3 * c.inner()),
            Tap::Ctx => (&bb.ctx, c.inner()),
            Tap::AttnOut => (&bb.attn_out, c.d_model),
            Tap::AttnRes => (&bb.res, c.d_model),
            Tap::FfNorm => (&bb.ff_norm, c.d_model),
            Tap::Wi0 => (&bb.wi0, c.d_ff),
            Tap::Wi1 => (&bb.wi1, c.d_ff),
            Tap::Gated => (&bb.gated, c.d_ff),
            Tap::FfOut => (&bb.ff_out, c.d_model),
        };
        self.gpu.read(buf, self.n() * w as usize)
    }

    /// `final_layer_norm(x_L)` — transformers' `last_hidden_state`, and the
    /// tensor FLUX conditions on.
    pub fn read_hidden(&self) -> Vec<f32> {
        self.gpu.read(&self.hidden, self.n() * self.cfg.d_model as usize)
    }
}
