// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The T5 encoder **training** graph: SSA forward + the hand-written reverse,
//! gated by `gradcheck::check_t5`.
//!
//! # What is trainable and what is frozen
//!
//! **Every** tensor in [`T5Config::tensor_manifest`] is `Role::Trainable` here —
//! all 7 per-block matrices/gains, the token table `shared.weight`, the final
//! norm gain, and (the one that is easy to forget) **`rel_bias.weight`**: T5's
//! relative-position bias is a learned `[num_buckets, heads]` **embedding**, not
//! a constant, and it is the only parameter in the graph whose gradient arrives
//! from *every* block at once. Nothing is frozen. The gradcheck therefore walks
//! the whole manifest.
//!
//! # The four T5-specific things the reverse has to get right
//!
//! 1. **No attention scale.** The forward dispatches
//!    `attn_scores_bidir_bias` with `scale = 1.0` (T5 folds `1/√d_kv` into its
//!    initialisation). The backward pair `attn_bwd_d{q,k}_bias` takes the same
//!    `scale` as a Param and must be handed the *same* `1.0`; passing the
//!    conventional `1/√d_kv` scales `d_q`/`d_k` by `1/8` at `d_kv = 64` and is
//!    invisible to anything except a finite-difference check.
//! 2. **The relative-position bias gradient is shared across the stack.**
//!    `attn_bwd_dbias` **ASSIGNS** `d_bias[h,i,j] = Σ_b d_scores[b,h,i,j]` — it
//!    does not accumulate — so a per-block dispatch straight into one buffer
//!    would keep only block 0's contribution (the last one written in the
//!    reverse walk) and silently train the bias on 1/24 of its gradient. Each
//!    block writes a scratch `d_bias_blk` and `axpy` folds it into
//!    `d_bias_acc`, which the submit's clear list zeroes once per step.
//! 3. **The bias table's adjoint is a permute + a scatter.** The forward is
//!    `bias_gather = embed(buckets, rel_bias)` `[T·T, heads]` then
//!    `bias = nlc_nchw(bias_gather)` `[heads, T·T]`. `nlc_nchw`'s adjoint is
//!    exactly `nchw_nlc` with the same Params (its own header states the
//!    inner-product identity), and `embed`'s adjoint is `emb_bwd` with the
//!    bucket ids as the "tokens", `heads` as the "d_model" and `rel_buckets` as
//!    the "vocab". Both are gathers; there are no atomics.
//! 4. **RMSNorm with a runtime epsilon.** T5 is `eps = 1e-6`, and the backward
//!    recomputes `r = 1/√(mean(x²)+eps)` — so it goes through
//!    `block::rmsnorm_eps_bwd` (`rms_inv_eps` + `rmsnorm_dw` + `rmsnorm_dx_eps`),
//!    never the eps-hardcoded `rmsnorm_dx`. All three are barrier-free
//!    per-row/per-channel gathers, which is why they behave identically on
//!    `backend-cpu` (which reports `workgroup_reductions == false`) and on the
//!    P40; only the *forward* norm consults `block::rms_variant`.
//!
//! # Cache policy — the one place this graph differs from [`crate::model`]
//!
//! The inference graph shares ONE `[B, heads, T, T]` probability slab across all
//! 24 blocks (documented at its allocation site: 134 MB at FLUX's T = 512, so a
//! per-block cache would be 3.2 GB). The backward reads `probs`, so training
//! mode allocates it **per block** — the SSA discipline AGENTS.md requires, and
//! affordable at the shapes a gradcheck and a small finetune run at. At T = 512
//! this is the term that has to become `block::chunked_bidir_bwd`'s per-chunk
//! recompute; that is recorded in `.agents/roadmap/t5.md` as the remaining
//! scaling work, not papered over here. `scores` stays a single scratch slab
//! because nothing in the reverse reads it.
//!
//! # Why the forward is recorded again here
//!
//! [`crate::model::T5Encoder`] holds its SSA buffers privately and its
//! `build_steps` is private, so a sibling module cannot drive its reverse. The
//! step list below is the same dispatch sequence with the same Params — it is
//! duplicated, and that is a real cost: **the two must be kept in lockstep**.
//! `tests::trainer_forward_matches_inference_graph` (at the foot of this file)
//! asserts they agree BITWISE on the encoder output and on the position bias
//! for a fixed batch, so a drift is a test failure rather than a silent one.
//! The right fix is to hoist the recorder into `model.rs` behind a
//! `train: bool` (what `clip::model::ClipText::build` does); that is a
//! `model.rs` change and is left to the owner of that file.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::T5Config;

// ---- forward kernels (same set, same order, as `crate::model::PIPELINES`) ----
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
// ---- backward, APPENDED so every index above is unchanged ----
const K_RMS_INV_EPS: usize = 13;
const K_RMSNORM_DW: usize = 14;
const K_RMSNORM_DX_EPS: usize = 15;
const K_MATMUL_DX: usize = 16;
const K_MATMUL_DW: usize = 17;
const K_MATMUL_DX_REG: usize = 18;
const K_MATMUL_DW_REG: usize = 19;
const K_DSCORES: usize = 20;
const K_DV: usize = 21;
const K_DQ: usize = 22;
const K_DK: usize = 23;
const K_DBIAS: usize = 24;
const K_GELU_BWD: usize = 25;
const K_EMB_BWD: usize = 26;
const K_NCHW_NLC: usize = 27;
const K_AXPY: usize = 28;

/// Forward **and** backward kernels in one list, so a trainer is one device
/// handle. Every name appears exactly once: registering a kernel name twice in
/// one set is rejected outright by the CPU backend's JIT
/// (`DuplicateDefinition`), which is the reason the backward half is appended
/// to a copy of the forward list rather than concatenated with it.
pub const TRAIN_PIPELINES: &[(&str, &str)] = &[
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
    ("rms_inv_eps", kernels::RMS_INV_EPS),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rmsnorm_dx_eps", kernels::RMSNORM_DX_EPS),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bias", kernels::ATTN_BWD_DQ_BIAS),
    ("attn_bwd_dk_bias", kernels::ATTN_BWD_DK_BIAS),
    ("attn_bwd_dbias", kernels::ATTN_BWD_DBIAS),
    ("gelu_bwd", kernels::GELU_BWD),
    ("emb_bwd", kernels::EMB_BWD),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("axpy", kernels::AXPY),
];

/// One block's SSA activations. Identical to [`crate::model::BlockBufs`] except
/// for `probs`, which is per block here — see the module header.
struct TrainBlock {
    attn_norm: DeviceBuffer,
    /// Fused `[N, 3*inner]` — q at 0, k at `inner`, v at `2*inner`.
    qkv: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    res: DeviceBuffer,
    ff_norm: DeviceBuffer,
    wi0: DeviceBuffer,
    wi1: DeviceBuffer,
    act: DeviceBuffer,
    gated: DeviceBuffer,
    ff_out: DeviceBuffer,
}

/// Reverse-pass scratch. Every entry is a *gradient*; the activations the
/// reverse reads are the forward's own SSA buffers. Only `dx` is per layer —
/// the reverse walk holds one block's intermediates live at a time.
struct Bwd {
    /// `dx[0]` = embedding-output grad, `dx[i+1]` = grad of block `i`'s output.
    dx: Vec<DeviceBuffer>,
    /// Objective grad on `hidden`, uploaded by the caller.
    seed_hidden: DeviceBuffer,
    d_res: DeviceBuffer,
    d_branch: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_scores: DeviceBuffer,
    /// One block's `[heads, T, T]` bias grad, ASSIGNED by `attn_bwd_dbias`.
    d_bias_blk: DeviceBuffer,
    /// The sum over blocks. Zeroed by the submit's clear list.
    d_bias_acc: DeviceBuffer,
    /// `[T*T, heads]` — the permuted bias grad the `emb_bwd` scatter consumes.
    d_bias_gather: DeviceBuffer,
    d_gated: DeviceBuffer,
    d_act: DeviceBuffer,
    d_wi0: DeviceBuffer,
    d_wi1: DeviceBuffer,
    /// Per-row inverse RMS, recomputed per use (`[rows]`, cheaper than caching).
    inv: DeviceBuffer,
    steps: Vec<Step>,
}

/// A trainable T5 encoder: the SSA forward plus the recorded reverse pass.
pub struct T5Trainer {
    pub gpu: Gpu,
    pub cfg: T5Config,
    pub ps: ParamStore,
    b: u32,
    t: u32,
    tokens: DeviceBuffer,
    buckets: DeviceBuffer,
    bias_gather: DeviceBuffer,
    bias: DeviceBuffer,
    x: Vec<DeviceBuffer>,
    blocks: Vec<TrainBlock>,
    /// Pre-softmax scratch. The one non-SSA forward buffer, and sound: nothing
    /// in the reverse reads it (the softmax backward works from `probs`).
    scores: DeviceBuffer,
    hidden: DeviceBuffer,
    steps: Vec<Step>,
    bwd: Bwd,
}

impl T5Trainer {
    /// Build on an existing device (tests pass `gpu_core::testgpu::dev`).
    pub fn new_on(
        gpu: Gpu,
        cfg: T5Config,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> T5Trainer {
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Trainable))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = b as u64 * t as u64;
        let d = cfg.d_model as u64;
        let inner = cfg.inner() as u64;
        let ff = cfg.d_ff as u64;
        let tt = t as u64 * t as u64;
        let slab = b as u64 * cfg.heads as u64 * tt;
        let tokens = gpu.buffer(
            "t5_train_tokens",
            n * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let buckets = gpu.buffer(
            "t5_train_buckets",
            tt * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        gpu.write(&buckets, &crate::hostbias::buckets(t, cfg.rel_buckets, cfg.rel_max_distance));
        let blocks: Vec<TrainBlock> = (0..cfg.layers)
            .map(|_| TrainBlock {
                attn_norm: gpu.storage(n * d),
                qkv: gpu.storage(n * 3 * inner),
                probs: gpu.storage(slab),
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
        let heads_tt = cfg.heads as u64 * tt;
        let bwd = Bwd {
            dx: (0..=cfg.layers).map(|_| gpu.storage(n * d)).collect(),
            seed_hidden: gpu.storage(n * d),
            d_res: gpu.storage(n * d),
            d_branch: gpu.storage(n * d),
            d_tmp: gpu.storage(n * d),
            d_ctx: gpu.storage(n * inner),
            d_qkv: gpu.storage(n * 3 * inner),
            d_scores: gpu.storage(slab),
            d_bias_blk: gpu.storage(heads_tt),
            d_bias_acc: gpu.storage(heads_tt),
            d_bias_gather: gpu.storage(heads_tt),
            d_gated: gpu.storage(n * ff),
            d_act: gpu.storage(n * ff),
            d_wi0: gpu.storage(n * ff),
            d_wi1: gpu.storage(n * ff),
            inv: gpu.storage(n),
            steps: Vec::new(),
        };
        let mut m = T5Trainer {
            bias_gather: gpu.storage(tt * cfg.heads as u64),
            bias: gpu.storage(heads_tt),
            x: (0..=cfg.layers).map(|_| gpu.storage(n * d)).collect(),
            blocks,
            scores: gpu.storage(slab),
            hidden: gpu.storage(n * d),
            tokens,
            buckets,
            gpu,
            cfg,
            ps,
            b,
            t,
            steps: Vec::new(),
            bwd,
        };
        m.steps = m.build_steps();
        m.bwd.steps = m.build_bwd_steps();
        m
    }

    /// Trainable encoder on its own device handle. Prefer [`T5Trainer::new_on`]
    /// in a process that already holds a device.
    pub fn new(cfg: T5Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> T5Trainer {
        T5Trainer::new_on(Gpu::new(TRAIN_PIPELINES), cfg, b, t, init)
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG3, false)
    }

    /// Backward-GEMM kernel + threads, picked on the OUTPUT dims — the same
    /// policy `block::pick_gemm` implements for the forward.
    fn bwd_gemm(&self, rows: u32, cols: u32, naive: usize, reg: usize) -> (usize, u32) {
        block::pick_gemm(rows as usize, cols as usize, naive, reg, false)
    }

    /// The forward RMSNorm, with the coalesced workgroup-per-row variant
    /// wherever the device reports workgroup reductions. Identical to
    /// [`crate::model::T5Encoder`]'s.
    fn rmsnorm(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, rows: u32) -> Step {
        let d = self.cfg.d_model;
        let (kind, threads) =
            block::rms_variant(&self.gpu, K_RMSNORM, Some(K_RMSNORM_ROWS), rows, d);
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
        s.push(g.step(
            K_EMBED,
            &[&self.buckets, self.w("rel_bias.weight"), &self.bias_gather],
            &[heads, tt],
            tt * heads,
        ));
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
            // `attn_scores_bidir_bias` Params:
            //   [bsz, n_heads, tcols, head_dim, qkv_stride, q_off, k_off, scale(f32 bits)]
            // scale = 1.0: T5 has NO 1/sqrt(d_kv).
            s.push(g.step(
                K_SCORES,
                &[&bb.qkv, &self.bias, &self.scores],
                &[b, heads, t, hd, 3 * inner, 0, inner, f(1.0)],
                b * heads * t * t,
            ));
            s.push(g.step(K_SOFTMAX, &[&self.scores, &bb.probs], &[b, heads, t], b * heads * t));
            s.push(g.step(
                K_APPLY,
                &[&bb.probs, &bb.qkv, &bb.ctx],
                &[b, heads, t, hd, 3 * inner, 2 * inner, inner],
                b * heads * t * hd,
            ));

            let (mk, mt) = self.gemm(n, d);
            s.push(g.step(mk, &[&bb.ctx, self.w(&format!("{p}.o.weight")), &bb.attn_out], &[n, inner, d], mt));
            s.push(g.step(K_ADD2, &[&self.x[l], &bb.attn_out, &bb.res], &[n * d], n * d));

            s.push(self.rmsnorm(&bb.res, self.w(&format!("{p}.ff_norm.weight")), &bb.ff_norm, n));
            let (mk, mt) = self.gemm(n, ff);
            s.push(g.step(mk, &[&bb.ff_norm, self.w(&format!("{p}.wi_0.weight")), &bb.wi0], &[n, d, ff], mt));
            let (mk, mt) = self.gemm(n, ff);
            s.push(g.step(mk, &[&bb.ff_norm, self.w(&format!("{p}.wi_1.weight")), &bb.wi1], &[n, d, ff], mt));
            s.push(g.step(K_GELU, &[&bb.wi0, &bb.act], &[n * ff], n * ff));
            s.push(g.step(K_MUL, &[&bb.act, &bb.wi1, &bb.gated], &[n * ff], n * ff));
            let (mk, mt) = self.gemm(n, d);
            s.push(g.step(mk, &[&bb.gated, self.w(&format!("{p}.wo.weight")), &bb.ff_out], &[n, ff, d], mt));
            s.push(g.step(K_ADD2, &[&bb.res, &bb.ff_out, &self.x[l + 1]], &[n * d], n * d));
        }

        s.push(self.rmsnorm(&self.x[c.layers as usize], self.w("final_norm.weight"), &self.hidden, n));
        s
    }

    /// The reverse pass, recorded once at build time — the exact adjoint of
    /// [`T5Trainer::build_steps`], walked bottom-up.
    ///
    /// Clears: only `d_bias_acc` (see the module header). `matmul_dx` runs with
    /// `accumulate = 0` on its first use per site, `add2` / `mul` / `gelu_bwd` /
    /// `rmsnorm_dx_eps` / the four `attn_bwd_*` kernels all ASSIGN. The
    /// PARAMETER grads accumulate and are zeroed exactly once per step by
    /// [`T5Trainer::zero_grads`], so they must never enter a submit's clear
    /// list.
    fn build_bwd_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let bw = &self.bwd;
        let (b, t) = (self.b, self.t);
        let n = b * t;
        let d = c.d_model;
        let ff = c.d_ff;
        let inner = c.inner();
        let heads = c.heads;
        let hd = c.d_kv;
        let tt = t * t;
        let gr = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::new();

        // ---- final RMSNorm ----
        let last = c.layers as usize;
        s.extend(block::rmsnorm_eps_bwd(
            g,
            K_RMS_INV_EPS,
            K_RMSNORM_DW,
            K_RMSNORM_DX_EPS,
            &self.x[last],
            self.w("final_norm.weight"),
            &bw.seed_hidden,
            &bw.dx[last],
            &bw.inv,
            Some(gr("final_norm.weight")),
            d,
            n,
            c.eps,
        ));

        for l in (0..c.layers as usize).rev() {
            let bb = &self.blocks[l];
            let p = format!("blocks.{l}");
            let d_out = &bw.dx[l + 1];

            // ---- FFN branch: x_{l+1} = res + gated @ Wo^T ----
            // `matmul_dw` Params: [m, k, n]; bufs [dy, x, dw] — ACCUMULATES.
            let (dw, dwt) = self.bwd_gemm(d, ff, K_MATMUL_DW, K_MATMUL_DW_REG);
            s.push(g.step(dw, &[d_out, &bb.gated, gr(&format!("{p}.wo.weight"))], &[n, ff, d], dwt));
            // `matmul_dx` Params: [m, k, n, accumulate]; bufs [dy, w, dx].
            let (dx, dxt) = self.bwd_gemm(n, ff, K_MATMUL_DX, K_MATMUL_DX_REG);
            s.push(g.step(dx, &[d_out, self.w(&format!("{p}.wo.weight")), &bw.d_gated], &[n, ff, d, 0], dxt));

            // gated = act * wi1  ->  d_act = d_gated*wi1, d_wi1 = d_gated*act.
            // `mul` Params: a single `n`; the mul backward IS `mul` (its header).
            s.push(g.step(K_MUL, &[&bw.d_gated, &bb.wi1, &bw.d_act], &[n * ff], n * ff));
            s.push(g.step(K_MUL, &[&bw.d_gated, &bb.act, &bw.d_wi1], &[n * ff], n * ff));
            // act = gelu_new(wi0): `gelu_bwd` Params: [total]; bufs [x, dout, dx],
            // and `x` is the PRE-activation.
            s.push(g.step(K_GELU_BWD, &[&bb.wi0, &bw.d_act, &bw.d_wi0], &[n * ff], n * ff));

            let (dw, dwt) = self.bwd_gemm(ff, d, K_MATMUL_DW, K_MATMUL_DW_REG);
            s.push(g.step(dw, &[&bw.d_wi0, &bb.ff_norm, gr(&format!("{p}.wi_0.weight"))], &[n, d, ff], dwt));
            s.push(g.step(dw, &[&bw.d_wi1, &bb.ff_norm, gr(&format!("{p}.wi_1.weight"))], &[n, d, ff], dwt));
            // Both projections read the SAME `ff_norm`, so its grad is a sum:
            // wi_0 assigns (accumulate = 0), wi_1 adds (accumulate = 1).
            let (dx, dxt) = self.bwd_gemm(n, d, K_MATMUL_DX, K_MATMUL_DX_REG);
            s.push(g.step(dx, &[&bw.d_wi0, self.w(&format!("{p}.wi_0.weight")), &bw.d_branch], &[n, d, ff, 0], dxt));
            s.push(g.step(dx, &[&bw.d_wi1, self.w(&format!("{p}.wi_1.weight")), &bw.d_branch], &[n, d, ff, 1], dxt));

            s.extend(block::rmsnorm_eps_bwd(
                g,
                K_RMS_INV_EPS,
                K_RMSNORM_DW,
                K_RMSNORM_DX_EPS,
                &bb.res,
                self.w(&format!("{p}.ff_norm.weight")),
                &bw.d_branch,
                &bw.d_tmp,
                &bw.inv,
                Some(gr(&format!("{p}.ff_norm.weight"))),
                d,
                n,
                c.eps,
            ));
            // residual re-join: d_res = d_out (pass-through) + branch grad.
            s.push(g.step(K_ADD2, &[d_out, &bw.d_tmp, &bw.d_res], &[n * d], n * d));

            // ---- attention branch: res = x_l + ctx @ Wo^T ----
            let (dw, dwt) = self.bwd_gemm(d, inner, K_MATMUL_DW, K_MATMUL_DW_REG);
            s.push(g.step(dw, &[&bw.d_res, &bb.ctx, gr(&format!("{p}.o.weight"))], &[n, inner, d], dwt));
            let (dx, dxt) = self.bwd_gemm(n, inner, K_MATMUL_DX, K_MATMUL_DX_REG);
            s.push(g.step(dx, &[&bw.d_res, self.w(&format!("{p}.o.weight")), &bw.d_ctx], &[n, inner, d, 0], dxt));

            // `attn_bwd_dscores_bidir` / `attn_bwd_dv_bidir` Params:
            //   [bsz, n_heads, tcols, head_dim, qkv_stride, v_off, d_model]
            // where `d_model` is the CONTEXT width (heads*d_kv).
            let pv = [b, heads, t, hd, 3 * inner, 2 * inner, inner];
            // `attn_bwd_d{q,k}_bias` Params:
            //   [bsz, n_heads, tcols, head_dim, qkv_stride, q_off, k_off, scale(bits), causal]
            // scale MUST be the forward's 1.0; causal = 0 (bidirectional).
            let pqk = [b, heads, t, hd, 3 * inner, 0, inner, f(1.0), 0];
            s.push(g.step(K_DSCORES, &[&bw.d_ctx, &bb.qkv, &bb.probs, &bw.d_scores], &pv, b * heads * t));
            s.push(g.step(K_DV, &[&bb.probs, &bw.d_ctx, &bw.d_qkv], &pv, b * heads * t * hd));
            s.push(g.step(K_DQ, &[&bw.d_scores, &bb.qkv, &bw.d_qkv], &pqk, b * heads * t * hd));
            s.push(g.step(K_DK, &[&bw.d_scores, &bb.qkv, &bw.d_qkv], &pqk, b * heads * t * hd));
            // `attn_bwd_dbias` Params: [bsz, n_heads, tcols, causal]; it ASSIGNS,
            // so the per-block result is folded in with `axpy` (Params [n, s]).
            s.push(g.step(K_DBIAS, &[&bw.d_scores, &bw.d_bias_blk], &[b, heads, t, 0], heads * tt));
            s.push(g.step(K_AXPY, &[&bw.d_bias_acc, &bw.d_bias_blk], &[heads * tt, f(1.0)], heads * tt));

            let (dw, dwt) = self.bwd_gemm(3 * inner, d, K_MATMUL_DW, K_MATMUL_DW_REG);
            s.push(g.step(dw, &[&bw.d_qkv, &bb.attn_norm, gr(&format!("{p}.qkv.weight"))], &[n, d, 3 * inner], dwt));
            let (dx, dxt) = self.bwd_gemm(n, d, K_MATMUL_DX, K_MATMUL_DX_REG);
            s.push(g.step(dx, &[&bw.d_qkv, self.w(&format!("{p}.qkv.weight")), &bw.d_branch], &[n, d, 3 * inner, 0], dxt));

            s.extend(block::rmsnorm_eps_bwd(
                g,
                K_RMS_INV_EPS,
                K_RMSNORM_DW,
                K_RMSNORM_DX_EPS,
                &self.x[l],
                self.w(&format!("{p}.attn_norm.weight")),
                &bw.d_branch,
                &bw.d_tmp,
                &bw.inv,
                Some(gr(&format!("{p}.attn_norm.weight"))),
                d,
                n,
                c.eps,
            ));
            s.push(g.step(K_ADD2, &[&bw.d_res, &bw.d_tmp, &bw.dx[l]], &[n * d], n * d));
        }

        // ---- the shared relative-position bias table ----
        // `nlc_nchw`'s adjoint is `nchw_nlc` with the SAME Params [total, c, hw].
        s.push(g.step(
            K_NCHW_NLC,
            &[&bw.d_bias_acc, &bw.d_bias_gather],
            &[heads * tt, heads, tt],
            heads * tt,
        ));
        // `emb_bwd` Params: [n_rows, d_model, vocab]; bufs [ids(u32), d_x, grad].
        // Here the "tokens" are the bucket ids, "d_model" is the head count and
        // the "vocab" is the bucket count.
        s.push(g.step(
            K_EMB_BWD,
            &[&self.buckets, &bw.d_bias_gather, gr("rel_bias.weight")],
            &[tt, heads, c.rel_buckets],
            c.rel_buckets * heads,
        ));

        // ---- token embedding ----
        s.push(g.step(K_EMB_BWD, &[&self.tokens, &bw.dx[0], gr("shared.weight")], &[n, d, c.vocab], c.vocab * d));
        s
    }

    /// Set the token ids, `[B*T]` row-major.
    pub fn set_tokens(&self, ids: &[u32]) {
        assert_eq!(ids.len(), (self.b * self.t) as usize, "token count");
        assert!(ids.iter().all(|&i| i < self.cfg.vocab), "token id >= vocab {}", self.cfg.vocab);
        self.gpu.write(&self.tokens, ids);
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    /// Zero every parameter gradient. Call once per training step, BEFORE
    /// [`T5Trainer::backward`] — the reverse pass accumulates into them.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Run the reverse pass for the objective whose gradient w.r.t. the
    /// encoder output is `d_hidden` (`[B*T, D]`, the grad of
    /// `final_layer_norm(x_L)` — what FLUX conditions on).
    ///
    /// The forward must already have run on the current tokens/weights: the
    /// backward reads the SSA activation buffers it left behind.
    pub fn backward(&self, d_hidden: &[f32]) {
        let n = (self.b * self.t) as usize;
        let d = self.cfg.d_model as usize;
        assert_eq!(d_hidden.len(), n * d, "d_hidden must be [B*T, D]");
        self.gpu.write_f32(&self.bwd.seed_hidden, d_hidden);
        // `d_bias_acc` is the ONLY buffer that must start at zero — see the
        // module header on `attn_bwd_dbias`.
        self.gpu.submit(&[&self.bwd.d_bias_acc], &self.bwd.steps);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.w(name), data);
    }

    /// `final_layer_norm(x_L)` — the encoder output.
    pub fn read_hidden(&self) -> Vec<f32> {
        self.gpu.read(&self.hidden, (self.b * self.t) as usize * self.cfg.d_model as usize)
    }
    /// The `[heads, T, T]` additive attention bias.
    pub fn read_position_bias(&self) -> Vec<f32> {
        self.gpu.read(&self.bias, (self.cfg.heads * self.t * self.t) as usize)
    }
}

/// A gradcheck-scale T5 config. `heads * d_kv != d_model` and `heads != d_kv`
/// on purpose: at XXL those three numbers are all equal (64 * 64 = 4096), so a
/// transposed or swapped index is invisible there — the same reason
/// `crates/t5/tests/` carries a `tiny_ref` gate at distinct dims.
pub fn tiny_config() -> T5Config {
    T5Config {
        vocab: 23,
        d_model: 16,
        d_ff: 32,
        d_kv: 6,
        layers: 2,
        heads: 3,
        rel_buckets: 8,
        rel_max_distance: 6,
        eps: 1e-6,
    }
}

/// Random weights for `cfg`, deterministic for a fixed `seed`.
///
/// For tests and gradient checks only — a real T5-XXL encoder is always
/// imported ([`crate::import`]). The scheme is `crates/clip/src/init.rs`'s and
/// exists for the same reason: T5's own initialiser uses deviations so small
/// that at a 16-channel config every activation sits in the linear regime of
/// `gelu` and the softmax, and the FD comparison would test almost nothing.
///
/// `rel_bias.weight` is deliberately NOT small: the bias is added to raw
/// (unscaled — T5 has no `1/√d_kv`) scores, and a near-zero table would make
/// the softmax uniform and its gradient nearly degenerate.
pub fn init_weights(cfg: &T5Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    let normal = |n: usize, s: f32, rng: &mut Rng| -> Vec<f32> {
        (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
    };
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        // fan_in is the LAST axis of every 2-D tensor in the manifest
        // (`[out, in]`, matching `matmul`'s `[N, K]` weight layout).
        let fan_in = *shape.last().expect("non-empty shape");
        let v: Vec<f32> = if name.ends_with("norm.weight") {
            // RMSNorm gain: 1 + jitter, so `rmsnorm_dw` is not evaluated at a
            // point where every gain is identical.
            normal(numel, 0.1, &mut rng).iter().map(|x| 1.0 + x).collect()
        } else if name == "shared.weight" || name == "rel_bias.weight" {
            // Both are lookup tables read at full magnitude, not through a
            // fan-in-scaled GEMM: the token table feeds the residual stream
            // directly, and the bias table is added to raw (unscaled — T5 has
            // no 1/sqrt(d_kv)) scores, where a near-zero entry would make the
            // softmax uniform and its gradient nearly degenerate.
            normal(numel, 0.5, &mut rng)
        } else {
            normal(numel, 1.0 / (fan_in as f32).sqrt(), &mut rng)
        };
        assert_eq!(v.len(), numel, "{name}: init size");
        w.insert(name, v);
    }
    w
}

/// A fixed token batch: deterministic, every id `< cfg.vocab`, and with the
/// right-padding (id 0) T5 actually sees — pad rows are attended as ordinary
/// keys, which is the unmasked FLUX contract [`crate::model`] documents.
pub fn fixed_tokens(cfg: &T5Config, b: u32, t: u32) -> Vec<u32> {
    let mut ids = vec![0u32; (b * t) as usize];
    for s in 0..b {
        let row = &mut ids[(s * t) as usize..((s + 1) * t) as usize];
        let content = (t - s).min(t);
        for (i, slot) in row.iter_mut().enumerate().take(content as usize) {
            *slot = (i as u32 * 7 + s * 5 + 1) % cfg.vocab;
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trainer's forward is the inference graph's forward.
    ///
    /// This is the lockstep gate the module header promises: the two step
    /// builders are separate code (see "Why the forward is recorded again
    /// here"), so nothing but a test stops them drifting. Both graphs run the
    /// same kernels with the same Params in the same order on the same weights
    /// and the same tokens, so the comparison is **exact**, not a tolerance —
    /// a single reordered dispatch or a changed Param shows up as a non-zero
    /// difference.
    ///
    /// The one deliberate difference (per-block `probs` in training mode) is
    /// invisible here: it changes which buffer the softmax writes, not what it
    /// computes.
    #[test]
    fn trainer_forward_matches_inference_graph() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = tiny_config();
        let (b, t) = (2u32, 6u32);
        let init = init_weights(&cfg, 11);
        let ids = fixed_tokens(&cfg, b, t);

        let inf = crate::model::T5Encoder::new_on(
            gpu_core::testgpu::dev(crate::model::PIPELINES),
            cfg.clone(),
            b,
            t,
            &init,
        );
        inf.set_tokens(&ids);
        inf.forward();
        let (want_hidden, want_bias) = (inf.read_hidden(), inf.read_position_bias());

        let tr = T5Trainer::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg, b, t, &init);
        tr.set_tokens(&ids);
        tr.forward();
        let (got_hidden, got_bias) = (tr.read_hidden(), tr.read_position_bias());

        assert_eq!(got_bias, want_bias, "position bias differs from the inference graph");
        assert_eq!(got_hidden, want_hidden, "encoder output differs from the inference graph");
    }
}
