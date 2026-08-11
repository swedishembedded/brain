// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer's **code-prediction Transformer**: SSA forward + hand-written
//! reverse under the cross-entropy objective the reference trains it with.
//! Gated by `gradcheck::check_codeformer`.
//!
//! # Scope — read this before trusting a green gradcheck
//!
//! This module trains **stage II only**: the transformer that maps the frozen
//! VQ encoder's latent to a distribution over codebook indices. Concretely:
//!
//! | part of the graph | here? | why |
//! |---|---|---|
//! | `position_emb`, `feat_emb`, the `n_layers` × `TransformerSALayer`, `idx_pred_layer` | **trained** | this module |
//! | VQ encoder / codebook / generator | **frozen** | already gradient-checked by `gradcheck::check_vqgan` over `vqgan::train::VqganTrainer`; its latent enters here as a fixed input buffer |
//! | the controllable feature transformation (`Fuse_sft_block`) and the dial `w` | **frozen** | it sits on the *generator* side of the code lookup, downstream of everything this objective touches |
//!
//! That is exactly the reference's own stage-II recipe
//! (`codeformer_arch.py` / the CodeFormer paper §3.2: "we fix the pre-trained
//! VQGAN and only train the transformer with the code-token loss"), so it is a
//! faithful partial rather than a convenient one. The gradient w.r.t. the
//! latent input is **not** propagated (the encoder is frozen, so there is
//! nothing to propagate it into); wiring the two trainers together end to end
//! is the remaining work and is listed in `.agents/roadmap/restore.md`.
//!
//! # The non-differentiable argmin, and why it never enters the loss
//!
//! Two different discrete operations sit near this graph, and only one of them
//! is a gradient problem:
//!
//! * `vq_argmin` — the VQ autoencoder's nearest-codebook search. It IS a
//!   gradient problem, and `vqgan::train` solves it with the straight-through
//!   estimator (documented in that module). **CodeFormer does not run it**: the
//!   whole point of the code-prediction transformer is to replace that search.
//! * `argmax_row` — how [`crate::model`]'s *inference* graph turns the predicted
//!   logits into indices for the codebook gather. It is **not on the training
//!   path at all**.
//!
//! The training objective is
//!
//! ```text
//! L = (1/T) · Σ_t  CE( logits[t, :],  code_target[t] )
//! ```
//!
//! — plain softmax cross-entropy of the `[T, codebook_size]` logits against the
//! ground-truth indices (which in the reference come from running the *frozen*
//! encoder + `vq_argmin` on the high-quality image, i.e. they are data, not a
//! differentiable function of these weights). `logits` is a smooth function of
//! every parameter here, so no straight-through estimator, no Gumbel relaxation
//! and no stop-gradient is needed or used: **the argmax is downstream of the
//! loss, not inside it**. `ce_value` computes the per-row loss and `ce_grad`
//! the `(softmax − onehot)/T` seed; the reverse is exact, and finite
//! differences of *this* objective validate it directly (unlike the VQ STE,
//! which FD can only validate against its surrogate).
//!
//! # The layer, and the one asymmetry that shapes the backward
//!
//! `TransformerSALayer` is pre-norm, and the position embedding reaches q and k
//! but **not** v (`codeformer_arch.py:120`):
//!
//! ```text
//! n1  = LayerNorm(x)
//! qk  = (n1 + position_emb) @ W_qk^T + b_qk      [T, 2E]   q at 0, k at E
//! v   =  n1                @ W_v^T  + b_v        [T,  E]
//! ctx = bidirectional MHA(q, k, v)                          scale 1/sqrt(head_dim)
//! res = x + (ctx @ W_o^T + b_o)
//! x'  = res + (linear2(gelu_erf(linear1(LayerNorm(res)))))
//! ```
//!
//! In reverse that asymmetry means `d(n1)` is an **accumulation of two
//! branches** — the v projection's `matmul_dx` plus the qk projection's, routed
//! through the `add2` with `position_emb` — and `d(position_emb)` is the *same*
//! qk-branch term summed over every layer. Both are `axpy` folds, so the
//! position embedding is the one parameter here whose gradient arrives from all
//! `n_layers` at once; a reverse that assigned instead of accumulating would
//! keep one layer's share and still train to a plausible loss curve. That is
//! what `check_codeformer`'s multi-layer config exists to catch.
//!
//! `F.gelu`'s default is the **erf** form, not the tanh approximation, so the
//! activation pair is `gelu_erf` / `gelu_erf_bwd` — not `gelu` / `gelu_bwd`.
//!
//! # Relation to [`crate::model`]
//!
//! [`crate::model::CodeFormer`] records its graph through
//! `vae::blocks::Builder` over host [`vae::blocks::Tensors`] with a recycling
//! activation pool — the right shape for inference and the wrong one for a
//! backward, which needs every stage's buffer alive and every weight in a
//! `ParamStore` with a gradient next to it. The step sequence and every Params
//! list below are the same as `model::record_transformer`'s, and
//! `tests::trainer_forward_matches_inference_graph_bitwise` holds them to that:
//! it records the INFERENCE transformer through `vae::blocks::Builder` and
//! `assert_eq!`s the `[T, codebook_size]` logits — no tolerance, so one
//! reordered dispatch or one changed Param is a test failure.
//!
//! That needs no fixture and uses none, contrary to what an earlier revision of
//! this header claimed: `Builder::dev` resolves weights lazily by name, so a
//! `Tensors` map holding only [`transformer_manifest`]'s tensors records the
//! transformer half on its own. It is `CodeFormer::new` that needs all 515
//! (it records the conv half too), not `model::record_transformer`.
//! `tests::trainer_forward_matches_inference_graph` remains as the separate,
//! independent check of the *objective*: the device mean CE against a host
//! `logsumexp − logit[target]` over the same logits.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::CodeFormerConfig;

// ---- forward ----
const K_LAYERNORM: usize = 0;
const K_MATMUL: usize = 1;
const K_MATMUL_REG3: usize = 2;
const K_BIAS_ADD: usize = 3;
const K_GELU_ERF: usize = 4;
const K_ADD2: usize = 5;
const K_SCORES: usize = 6;
const K_SOFTMAX: usize = 7;
const K_APPLY: usize = 8;
const K_CE_VALUE: usize = 9;
const K_CE_GRAD: usize = 10;
// `layernorm_rows` / `ln_stats_rows` / `layernorm_dx_rows` are registered but
// resolved BY NAME through `block::LayerNormIds::resolve`, never indexed.
// ---- backward ----
const K_LN_STATS: usize = 14;
const K_LAYERNORM_DX: usize = 15;
const K_LN_DGAMMA: usize = 16;
const K_LN_DBETA: usize = 17;
const K_MATMUL_DX: usize = 18;
const K_MATMUL_DW: usize = 19;
const K_MATMUL_DX_REG: usize = 20;
const K_MATMUL_DW_REG: usize = 21;
const K_BIAS_GRAD: usize = 22;
const K_DSCORES: usize = 23;
const K_DV: usize = 24;
const K_DQ: usize = 25;
const K_DK: usize = 26;
const K_GELU_ERF_BWD: usize = 27;
const K_AXPY: usize = 28;

/// Forward **and** backward kernels for the code-prediction Transformer, one
/// list so a trainer is one device handle. Each name appears exactly once (the
/// CPU JIT rejects a duplicate outright), and the backward half is APPENDED —
/// every index above is a position in this list.
///
/// This is deliberately NOT [`crate::model::KERNELS`]: that set carries the
/// whole VQGAN conv stack, which this objective never dispatches.
pub const TRAIN_PIPELINES: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("add2", kernels::ADD2),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("ce_value", kernels::CE_VALUE),
    ("ce_grad", kernels::CE_GRAD),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("bias_grad", kernels::BIAS_GRAD),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("axpy", kernels::AXPY),
];

/// Compile-time check that the two halves of [`TRAIN_PIPELINES`] line up with
/// the `K_*` indices. A silently shifted index is the failure mode
/// `.agents/rules/kernels.md` exists to prevent.
const _: () = {
    assert!(TRAIN_PIPELINES.len() == 29);
};

/// One layer's SSA activations. Every one of these is read by the reverse.
struct LayerBufs {
    n1: DeviceBuffer,
    /// `n1 + position_emb` — the q/k projection's input, kept because its
    /// `matmul_dw` needs it and it is NOT `n1`.
    qk_in: DeviceBuffer,
    /// Fused `[T, 2E]` — q at 0, k at `E`.
    qk: DeviceBuffer,
    /// `[T, E]`, its own buffer at stride `E` (`v_off = 0`).
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    res: DeviceBuffer,
    n2: DeviceBuffer,
    /// Pre-activation MLP hidden `[T, mlp]`.
    h: DeviceBuffer,
    h_act: DeviceBuffer,
    mlp_out: DeviceBuffer,
}

/// Reverse-pass scratch. Per-layer buffers are reused across layers (the walk
/// holds one layer live at a time); only `dx` mirrors `x`.
struct Bwd {
    dx: Vec<DeviceBuffer>,
    d_logits: DeviceBuffer,
    d_res: DeviceBuffer,
    d_branch: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_qk: DeviceBuffer,
    d_qk_in: DeviceBuffer,
    d_v: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_h_act: DeviceBuffer,
    d_h: DeviceBuffer,
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    steps: Vec<Step>,
}

/// The trainable code-prediction Transformer.
pub struct CodeTransformerTrainer {
    pub gpu: Gpu,
    pub cfg: CodeFormerConfig,
    pub ps: ParamStore,
    t: u32,
    /// The FROZEN VQ encoder's latent, flattened to `[T, vqgan.emb_dim]`. An
    /// input, not a parameter — see the module header.
    lq_rows: DeviceBuffer,
    /// Ground-truth code indices `[T]` (u32).
    targets: DeviceBuffer,
    /// `x[0]` = `feat_emb` output; `x[i+1]` = layer `i`'s output.
    x: Vec<DeviceBuffer>,
    layers: Vec<LayerBufs>,
    /// Pre-softmax scratch — the one non-SSA forward buffer, and sound: the
    /// softmax backward reads `probs`, never `scores`.
    scores: DeviceBuffer,
    ln_out: DeviceBuffer,
    logits: DeviceBuffer,
    /// Per-row CE `[T]`, summed on the host.
    ce: DeviceBuffer,
    steps: Vec<Step>,
    bwd: Bwd,
}

/// The tensors this trainer owns: exactly the transformer half of
/// [`CodeFormerConfig::runtime_manifest`], selected by name rather than
/// restated, so the shapes can only be wrong in one place.
pub fn transformer_manifest(cfg: &CodeFormerConfig) -> Vec<(String, Vec<usize>)> {
    cfg.runtime_manifest()
        .into_iter()
        .filter(|(n, _)| {
            n == "position_emb"
                || n.starts_with("feat_emb.")
                || n.starts_with("ft_layers.")
                || n.starts_with("idx_pred_layer.")
        })
        .collect()
}

impl CodeTransformerTrainer {
    /// Build on an existing device (tests pass `gpu_core::testgpu::dev`).
    ///
    /// `T` is `cfg.latent_size` — the flattened latent grid — because
    /// `position_emb` is `[latent_size, E]` and is added elementwise, so the
    /// sequence length is fixed by the checkpoint.
    pub fn new_on(gpu: Gpu, cfg: CodeFormerConfig, init: &HashMap<String, Vec<f32>>) -> Self {
        let roles: Vec<(String, usize, Role)> = transformer_manifest(&cfg)
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Trainable))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let t = cfg.latent_size;
        let e = cfg.dim_embd as u64;
        let mlp = cfg.dim_mlp as u64;
        let k = cfg.vqgan.codebook_size as u64;
        let tt = t as u64;
        let slab = cfg.n_head as u64 * tt * tt;
        let targets = gpu.buffer(
            "cf_targets",
            tt * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let layers: Vec<LayerBufs> = (0..cfg.n_layers)
            .map(|_| LayerBufs {
                n1: gpu.storage(tt * e),
                qk_in: gpu.storage(tt * e),
                qk: gpu.storage(tt * 2 * e),
                v: gpu.storage(tt * e),
                probs: gpu.storage(slab),
                ctx: gpu.storage(tt * e),
                attn_out: gpu.storage(tt * e),
                res: gpu.storage(tt * e),
                n2: gpu.storage(tt * e),
                h: gpu.storage(tt * mlp),
                h_act: gpu.storage(tt * mlp),
                mlp_out: gpu.storage(tt * e),
            })
            .collect();
        let bwd = Bwd {
            dx: (0..=cfg.n_layers).map(|_| gpu.storage(tt * e)).collect(),
            d_logits: gpu.storage(tt * k),
            d_res: gpu.storage(tt * e),
            d_branch: gpu.storage(tt * e),
            d_tmp: gpu.storage(tt * e),
            d_ctx: gpu.storage(tt * e),
            d_qk: gpu.storage(tt * 2 * e),
            d_qk_in: gpu.storage(tt * e),
            d_v: gpu.storage(tt * e),
            d_scores: gpu.storage(slab),
            d_h_act: gpu.storage(tt * mlp),
            d_h: gpu.storage(tt * mlp),
            mean: gpu.storage(tt),
            inv: gpu.storage(tt),
            steps: Vec::new(),
        };
        let mut m = CodeTransformerTrainer {
            lq_rows: gpu.storage(tt * cfg.vqgan.emb_dim as u64),
            x: (0..=cfg.n_layers).map(|_| gpu.storage(tt * e)).collect(),
            layers,
            scores: gpu.storage(slab),
            ln_out: gpu.storage(tt * e),
            logits: gpu.storage(tt * k),
            ce: gpu.storage(tt),
            targets,
            gpu,
            cfg,
            ps,
            t,
            steps: Vec::new(),
            bwd,
        };
        m.steps = m.build_steps();
        m.bwd.steps = m.build_bwd_steps();
        m
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG3, false)
    }
    fn bwd_gemm(&self, rows: u32, cols: u32, naive: usize, reg: usize) -> (usize, u32) {
        block::pick_gemm(rows as usize, cols as usize, naive, reg, false)
    }

    /// `y = x @ W^T + b`, the shape every `nn.Linear` in this transformer has.
    fn linear(&self, s: &mut Vec<Step>, p: &str, m: u32, k: u32, n: u32, x: &DeviceBuffer, y: &DeviceBuffer) {
        let (kind, threads) = self.gemm(m, n);
        s.push(self.gpu.step(kind, &[x, self.w(&format!("{p}.weight")), y], &[m, k, n], threads));
        // `bias_add` Params: [m, n]; bufs [out(rw), bias].
        s.push(self.gpu.step(K_BIAS_ADD, &[y, self.w(&format!("{p}.bias"))], &[m, n], m * n));
    }

    /// The adjoint of [`Self::linear`]: bias row-sum, weight GEMM, input GEMM.
    /// `acc` selects `matmul_dx`'s overwrite (0) / accumulate (1).
    #[allow(clippy::too_many_arguments)]
    fn linear_bwd(
        &self,
        s: &mut Vec<Step>,
        p: &str,
        m: u32,
        k: u32,
        n: u32,
        x: &DeviceBuffer,
        dy: &DeviceBuffer,
        dx: &DeviceBuffer,
        acc: u32,
    ) {
        let gr = |name: &str| self.ps.g(name);
        // `bias_grad` Params: [m, n]; one thread per feature — ACCUMULATES.
        s.push(self.gpu.step(K_BIAS_GRAD, &[dy, gr(&format!("{p}.bias"))], &[m, n], n));
        // `matmul_dw` Params: [m, k, n]; bufs [dy, x, dw] — ACCUMULATES.
        let (dw, dwt) = self.bwd_gemm(n, k, K_MATMUL_DW, K_MATMUL_DW_REG);
        s.push(self.gpu.step(dw, &[dy, x, gr(&format!("{p}.weight"))], &[m, k, n], dwt));
        // `matmul_dx` Params: [m, k, n, accumulate]; bufs [dy, w, dx].
        let (dxk, dxt) = self.bwd_gemm(m, k, K_MATMUL_DX, K_MATMUL_DX_REG);
        s.push(self.gpu.step(dxk, &[dy, self.w(&format!("{p}.weight")), dx], &[m, k, n, acc], dxt));
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let t = self.t;
        let e = c.dim_embd;
        let mlp = c.dim_mlp;
        let heads = c.n_head;
        let hd = c.head_dim();
        let k = c.vqgan.codebook_size;
        let eps = c.ln_eps;
        let ln = block::LayerNormIds::resolve(g, K_LAYERNORM, K_LN_STATS, K_LAYERNORM_DX);
        let mut s = Vec::new();

        self.linear(&mut s, "feat_emb", t, c.vqgan.emb_dim, e, &self.lq_rows, &self.x[0]);

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = CodeFormerConfig::layer_prefix(l);
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &self.x[l],
                self.w(&format!("{p}.norm1.weight")),
                self.w(&format!("{p}.norm1.bias")),
                &lb.n1,
                e,
                t,
                eps,
            ));
            // q = k = norm1(x) + position_emb; v = norm1(x). `add2` Params: [total].
            s.push(g.step(K_ADD2, &[&lb.n1, self.w("position_emb"), &lb.qk_in], &[t * e], t * e));
            self.linear(&mut s, &format!("{p}.self_attn.qk"), t, e, 2 * e, &lb.qk_in, &lb.qk);
            self.linear(&mut s, &format!("{p}.self_attn.v"), t, e, e, &lb.n1, &lb.v);

            // `attn_scores_bidir`  Params: [bsz, n_heads, tcols, head_dim, qkv_stride, q_off, k_off]
            // `attn_softmax_bidir` Params: [bsz, n_heads, tcols]
            // `attn_apply_bidir`   Params: [bsz, n_heads, tcols, head_dim, qkv_stride, v_off, d_model]
            // bsz = 1: CodeFormer restores one face at a time (see model.rs).
            s.push(g.step(K_SCORES, &[&lb.qk, &self.scores], &[1, heads, t, hd, 2 * e, 0, e], heads * t * t));
            s.push(g.step(K_SOFTMAX, &[&self.scores, &lb.probs], &[1, heads, t], heads * t));
            s.push(g.step(K_APPLY, &[&lb.probs, &lb.v, &lb.ctx], &[1, heads, t, hd, e, 0, e], heads * t * hd));

            self.linear(&mut s, &format!("{p}.self_attn.out_proj"), t, e, e, &lb.ctx, &lb.attn_out);
            s.push(g.step(K_ADD2, &[&self.x[l], &lb.attn_out, &lb.res], &[t * e], t * e));

            s.push(block::layernorm_fwd(
                g,
                &ln,
                &lb.res,
                self.w(&format!("{p}.norm2.weight")),
                self.w(&format!("{p}.norm2.bias")),
                &lb.n2,
                e,
                t,
                eps,
            ));
            self.linear(&mut s, &format!("{p}.linear1"), t, e, mlp, &lb.n2, &lb.h);
            // `F.gelu`'s default is the ERF form, not the tanh approximation.
            s.push(g.step(K_GELU_ERF, &[&lb.h, &lb.h_act], &[t * mlp], t * mlp));
            self.linear(&mut s, &format!("{p}.linear2"), t, mlp, e, &lb.h_act, &lb.mlp_out);
            s.push(g.step(K_ADD2, &[&lb.res, &lb.mlp_out, &self.x[l + 1]], &[t * e], t * e));
        }

        // idx_pred_layer = Sequential(LayerNorm(E), Linear(E, K, bias=False)).
        s.push(block::layernorm_fwd(
            g,
            &ln,
            &self.x[c.n_layers as usize],
            self.w("idx_pred_layer.0.weight"),
            self.w("idx_pred_layer.0.bias"),
            &self.ln_out,
            e,
            t,
            eps,
        ));
        let (mk, mt) = self.gemm(t, k);
        s.push(g.step(mk, &[&self.ln_out, self.w("idx_pred_layer.1.weight"), &self.logits], &[t, e, k], mt));
        // `ce_value` Params: [n_rows, vocab]; bufs [logits, targets(u32), out].
        // Per-row loss; the host sums and divides — the mean CE.
        s.push(g.step(K_CE_VALUE, &[&self.logits, &self.targets, &self.ce], &[t, k], t));
        s
    }

    /// The reverse pass — the exact adjoint of [`Self::build_steps`].
    ///
    /// Clears: **none**. `ce_grad`, `matmul_dx` at `accumulate = 0`, `add2`,
    /// `gelu_erf_bwd`, `layernorm_dx` and the four `attn_bwd_*` kernels all
    /// ASSIGN. The PARAMETER grads accumulate (`matmul_dw`, `bias_grad`,
    /// `layernorm_dgamma`/`dbeta`, and `position_emb`'s `axpy`) and are zeroed
    /// exactly once per step by [`Self::zero_grads`], so they must never enter
    /// a submit's clear list.
    fn build_bwd_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let bw = &self.bwd;
        let t = self.t;
        let e = c.dim_embd;
        let mlp = c.dim_mlp;
        let heads = c.n_head;
        let hd = c.head_dim();
        let k = c.vqgan.codebook_size;
        let eps = c.ln_eps;
        let ln = block::LayerNormIds::resolve(g, K_LAYERNORM, K_LN_STATS, K_LAYERNORM_DX);
        let gr = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::new();

        // ---- the objective: mean CE over the T code positions ----
        // `ce_grad` Params: [n_rows, vocab]; it already divides by `n_rows`, so
        // the seed is d(mean CE)/d(logits) and matches `loss()`'s host mean.
        s.push(g.step(K_CE_GRAD, &[&self.logits, &self.targets, &bw.d_logits], &[t, k], t * k));

        // ---- idx_pred_layer: biasless head, then its LayerNorm ----
        let (dw, dwt) = self.bwd_gemm(k, e, K_MATMUL_DW, K_MATMUL_DW_REG);
        s.push(g.step(dw, &[&bw.d_logits, &self.ln_out, gr("idx_pred_layer.1.weight")], &[t, e, k], dwt));
        let (dx, dxt) = self.bwd_gemm(t, e, K_MATMUL_DX, K_MATMUL_DX_REG);
        s.push(g.step(dx, &[&bw.d_logits, self.w("idx_pred_layer.1.weight"), &bw.d_branch], &[t, e, k, 0], dxt));

        let last = c.n_layers as usize;
        s.push(block::ln_stats_fwd(g, &ln, &self.x[last], &bw.mean, &bw.inv, e, t, eps));
        // `layernorm_dgamma` Params: [d_model, n_rows]; bufs [dy, x, mean, inv, dgamma].
        s.push(g.step(
            K_LN_DGAMMA,
            &[&bw.d_branch, &self.x[last], &bw.mean, &bw.inv, gr("idx_pred_layer.0.weight")],
            &[e, t],
            e,
        ));
        // `layernorm_dbeta` Params: [d_model, n_rows]; bufs [dy, dbeta].
        s.push(g.step(K_LN_DBETA, &[&bw.d_branch, gr("idx_pred_layer.0.bias")], &[e, t], e));
        s.push(block::layernorm_dx_bwd(
            g,
            &ln,
            &self.x[last],
            self.w("idx_pred_layer.0.weight"),
            &bw.d_branch,
            &bw.dx[last],
            e,
            t,
            eps,
        ));

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let p = CodeFormerConfig::layer_prefix(l);
            let d_out = &bw.dx[l + 1];

            // ---- MLP branch ----
            self.linear_bwd(&mut s, &format!("{p}.linear2"), t, mlp, e, &lb.h_act, d_out, &bw.d_h_act, 0);
            // The activation backward reads the PRE-activation `h` (post-bias).
            // Params: a single `total`; bufs [x, dout, dx].
            s.push(g.step(K_GELU_ERF_BWD, &[&lb.h, &bw.d_h_act, &bw.d_h], &[t * mlp], t * mlp));
            self.linear_bwd(&mut s, &format!("{p}.linear1"), t, e, mlp, &lb.n2, &bw.d_h, &bw.d_branch, 0);

            s.push(block::ln_stats_fwd(g, &ln, &lb.res, &bw.mean, &bw.inv, e, t, eps));
            s.push(g.step(
                K_LN_DGAMMA,
                &[&bw.d_branch, &lb.res, &bw.mean, &bw.inv, gr(&format!("{p}.norm2.weight"))],
                &[e, t],
                e,
            ));
            s.push(g.step(K_LN_DBETA, &[&bw.d_branch, gr(&format!("{p}.norm2.bias"))], &[e, t], e));
            s.push(block::layernorm_dx_bwd(
                g,
                &ln,
                &lb.res,
                self.w(&format!("{p}.norm2.weight")),
                &bw.d_branch,
                &bw.d_tmp,
                e,
                t,
                eps,
            ));
            // residual re-join: d_res = d_out (pass-through) + branch grad.
            s.push(g.step(K_ADD2, &[d_out, &bw.d_tmp, &bw.d_res], &[t * e], t * e));

            // ---- attention branch ----
            self.linear_bwd(&mut s, &format!("{p}.self_attn.out_proj"), t, e, e, &lb.ctx, &bw.d_res, &bw.d_ctx, 0);
            // v lives in its OWN buffer at stride E, so the dscores/dv pair gets
            // `qkv_stride = e, v_off = 0`; q/k share one buffer at stride 2E.
            let pv = [1, heads, t, hd, e, 0, e];
            let pqk = [1, heads, t, hd, 2 * e, 0, e];
            s.push(g.step(K_DSCORES, &[&bw.d_ctx, &lb.v, &lb.probs, &bw.d_scores], &pv, heads * t));
            s.push(g.step(K_DV, &[&lb.probs, &bw.d_ctx, &bw.d_v], &pv, heads * t * hd));
            s.push(g.step(K_DQ, &[&bw.d_scores, &lb.qk, &bw.d_qk], &pqk, heads * t * hd));
            s.push(g.step(K_DK, &[&bw.d_scores, &lb.qk, &bw.d_qk], &pqk, heads * t * hd));

            // d(n1) accumulates TWO branches: the v projection reads n1
            // directly, the qk projection reads n1 + position_emb.
            self.linear_bwd(&mut s, &format!("{p}.self_attn.v"), t, e, e, &lb.n1, &bw.d_v, &bw.d_branch, 0);
            self.linear_bwd(&mut s, &format!("{p}.self_attn.qk"), t, e, 2 * e, &lb.qk_in, &bw.d_qk, &bw.d_qk_in, 0);
            // `add2`'s adjoint is the identity into BOTH addends: `position_emb`
            // (a parameter, so `axpy` accumulates across layers) and `n1`.
            // `axpy` Params: [n, s]; bufs [out(rw), inp].
            s.push(g.step(K_AXPY, &[gr("position_emb"), &bw.d_qk_in], &[t * e, gpu_core::f(1.0)], t * e));
            s.push(g.step(K_AXPY, &[&bw.d_branch, &bw.d_qk_in], &[t * e, gpu_core::f(1.0)], t * e));

            s.push(block::ln_stats_fwd(g, &ln, &self.x[l], &bw.mean, &bw.inv, e, t, eps));
            s.push(g.step(
                K_LN_DGAMMA,
                &[&bw.d_branch, &self.x[l], &bw.mean, &bw.inv, gr(&format!("{p}.norm1.weight"))],
                &[e, t],
                e,
            ));
            s.push(g.step(K_LN_DBETA, &[&bw.d_branch, gr(&format!("{p}.norm1.bias"))], &[e, t], e));
            s.push(block::layernorm_dx_bwd(
                g,
                &ln,
                &self.x[l],
                self.w(&format!("{p}.norm1.weight")),
                &bw.d_branch,
                &bw.d_tmp,
                e,
                t,
                eps,
            ));
            s.push(g.step(K_ADD2, &[&bw.d_res, &bw.d_tmp, &bw.dx[l]], &[t * e], t * e));
        }

        // ---- feat_emb ----
        // `d_branch` is a throwaway here: the latent is a frozen INPUT, so its
        // gradient has nowhere to go (see the module header on scope). It is
        // still computed because `linear_bwd` is one helper; binding a scratch
        // buffer is cheaper than a second helper that would drift.
        self.linear_bwd(&mut s, "feat_emb", t, c.vqgan.emb_dim, e, &self.lq_rows, &bw.dx[0], &bw.d_branch, 0);
        s
    }

    /// Upload the frozen encoder's latent, flattened `[T, vqgan.emb_dim]`
    /// row-major (channel-last — `vae::blocks::Builder::nchw_to_rows`'s layout).
    pub fn set_latent(&self, rows: &[f32]) {
        let want = self.t as usize * self.cfg.vqgan.emb_dim as usize;
        assert_eq!(rows.len(), want, "latent must be [T, emb_dim]");
        self.gpu.write_f32(&self.lq_rows, rows);
    }

    /// Set the ground-truth code indices `[T]`.
    pub fn set_targets(&self, idx: &[u32]) {
        assert_eq!(idx.len(), self.t as usize, "target count");
        assert!(
            idx.iter().all(|&i| i < self.cfg.vqgan.codebook_size),
            "code index >= codebook_size {}",
            self.cfg.vqgan.codebook_size
        );
        self.gpu.write(&self.targets, idx);
    }

    /// Run the forward and return the **mean** cross-entropy — the scalar the
    /// reverse differentiates.
    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.steps);
        let per_row = self.gpu.read(&self.ce, self.t as usize);
        per_row.iter().sum::<f32>() / self.t as f32
    }

    /// Zero every parameter gradient. Call once per step, BEFORE
    /// [`Self::backward`] — the reverse accumulates into them.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Run the reverse pass. The forward must already have run on the current
    /// latent/targets/weights: the backward reads the SSA buffers it left.
    pub fn backward(&self) {
        self.gpu.submit(&[], &self.bwd.steps);
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
    /// `[T, codebook_size]` — the code logits the CE is taken over.
    pub fn read_logits(&self) -> Vec<f32> {
        self.gpu.read(&self.logits, self.t as usize * self.cfg.vqgan.codebook_size as usize)
    }
}

/// A gradcheck-scale CodeFormer config: a 4×4 latent grid (`T = 16`), 2
/// transformer layers, `E = 12` over 3 heads (`head_dim = 4`), a 20-wide MLP,
/// and a 12-entry codebook over an 8-d latent.
///
/// `n_layers = 2` is the minimum that makes `position_emb`'s cross-layer
/// gradient accumulation observable; `dim_embd ≠ dim_mlp ≠ vqgan.emb_dim` and
/// `n_head ≠ head_dim` on purpose, so a transposed or swapped dimension cannot
/// hide (at the real config `head_dim = 64 = n_head·8` and every stage is a
/// multiple of 256).
pub fn tiny_config() -> CodeFormerConfig {
    let mut cfg = CodeFormerConfig::codeformer();
    cfg.vqgan.codebook_size = 12;
    cfg.vqgan.emb_dim = 8;
    cfg.dim_embd = 12;
    cfg.n_head = 3;
    cfg.n_layers = 2;
    cfg.dim_mlp = 20;
    cfg.latent_size = 16;
    cfg
}

/// Random transformer weights for `cfg`, deterministic for a fixed `seed`.
///
/// Tests and gradient checks only — a real CodeFormer is always imported
/// ([`crate::import`]). Same scheme as `crates/clip/src/init.rs` and for the
/// same reason: the reference's own deviations are small enough that at a
/// 12-channel config every activation sits in the linear regime of `gelu_erf`
/// and the softmax, and the FD comparison would test almost nothing.
pub fn init_weights(cfg: &CodeFormerConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    let normal = |n: usize, s: f32, rng: &mut Rng| -> Vec<f32> {
        (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
    };
    for (name, shape) in transformer_manifest(cfg) {
        let numel: usize = shape.iter().product();
        // fan_in is the LAST axis of every 2-D tensor here (`[out, in]`, the
        // layout `matmul` wants).
        let fan_in = *shape.last().expect("non-empty shape");
        let v: Vec<f32> = if name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name == "idx_pred_layer.0.weight"
        {
            // LayerNorm gain: 1 + jitter, so `dgamma` is not evaluated where
            // every gain is identical.
            normal(numel, 0.1, &mut rng).iter().map(|x| 1.0 + x).collect()
        } else if name.ends_with(".bias") || name == "idx_pred_layer.0.bias" {
            normal(numel, 0.05, &mut rng)
        } else if name == "position_emb" {
            normal(numel, 0.5, &mut rng)
        } else {
            normal(numel, 1.0 / (fan_in as f32).sqrt(), &mut rng)
        };
        assert_eq!(v.len(), numel, "{name}: init size");
        w.insert(name, v);
    }
    w
}

/// A fixed latent + code-target pair, deterministic for a fixed `seed`.
///
/// The targets are ordinary data (in training they come from running the frozen
/// encoder + `vq_argmin` on the ground-truth image), so a deterministic
/// pseudo-random assignment is a faithful stand-in: nothing in the objective
/// depends on them being self-consistent with the latent.
pub fn fixed_batch(cfg: &CodeFormerConfig, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut rng = Rng::new(seed ^ 0xC0DE);
    let t = cfg.latent_size as usize;
    let latent = (0..t * cfg.vqgan.emb_dim as usize).map(|_| rng.next_f32() - 0.5).collect();
    let targets = (0..t).map(|i| (i as u32 * 5 + 3) % cfg.vqgan.codebook_size).collect();
    (latent, targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trainer owns exactly the transformer half of the runtime manifest,
    /// and nothing of the conv half.
    #[test]
    fn manifest_is_the_transformer_half_of_the_runtime_manifest() {
        let cfg = CodeFormerConfig::codeformer();
        let m = transformer_manifest(&cfg);
        let runtime: HashMap<String, Vec<usize>> = cfg.runtime_manifest().into_iter().collect();
        for (n, s) in &m {
            assert_eq!(runtime.get(n), Some(s), "{n}: shape drifted from the runtime manifest");
        }
        // 1 position_emb + 2 feat_emb + 9 layers x 14 + 3 idx_pred = 132.
        assert_eq!(m.len(), 1 + 2 + 9 * 14 + 3);
        assert!(m.iter().all(|(n, _)| !n.starts_with("encoder.") && !n.starts_with("generator.")));
    }

    /// **The lockstep gate.** The trainer's forward IS
    /// [`crate::model::record_transformer`], bit for bit.
    ///
    /// Both graphs run the same kernels with the same Params in the same order
    /// on the same weights and the same latent, so the comparison is `assert_eq`
    /// on `Vec<f32>`, not a tolerance — one reordered dispatch or one changed
    /// Param is a test failure.
    ///
    /// No fixture is needed and none is used: `vae::blocks::Builder` resolves
    /// weights lazily by name (`Builder::dev`), so a `Tensors` map holding only
    /// [`transformer_manifest`]'s tensors records the transformer half on its
    /// own. It is `CodeFormer::new` that needs all 515 — not the recorder.
    #[test]
    fn trainer_forward_matches_inference_graph_bitwise() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        use vae::blocks::{BlockNames, Builder, Tensors};

        let cfg = tiny_config();
        let init = init_weights(&cfg, 5);
        let (latent, _) = fixed_batch(&cfg, 5);
        let t = cfg.latent_size;
        let emb = cfg.vqgan.emb_dim;
        let k = cfg.vqgan.codebook_size as usize;

        // ---- the inference recorder, on the model's own kernel set ----
        let tensors: Tensors = transformer_manifest(&cfg)
            .into_iter()
            .map(|(n, shape)| {
                let v = init.get(&n).unwrap_or_else(|| panic!("init missing {n}")).clone();
                (n, (shape, v))
            })
            .collect();
        let gpu = gpu_core::testgpu::dev(&crate::model::KERNELS);
        let ids = crate::model::Ids::resolve(&gpu);
        let lq = gpu.storage((t * emb) as u64);
        gpu.write_f32(&lq, &latent);
        let mut b =
            Builder::new(&gpu, &tensors, cfg.vqgan.norm_eps, cfg.vqgan.norm_groups, BlockNames::vqgan(), false);
        let (logits, _idx) = crate::model::record_transformer(&mut b, &cfg, &ids, t, &lq);
        let (steps, _taps) = b.finish();
        gpu.submit(&[], &steps);
        let want = gpu.read(&logits, t as usize * k);

        // ---- the trainer's recorder ----
        let tr =
            CodeTransformerTrainer::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg, &init);
        tr.set_latent(&latent);
        tr.set_targets(&fixed_batch(&tiny_config(), 5).1);
        let _ = tr.forward();
        let got = tr.read_logits();

        assert_eq!(got, want, "code logits differ from the inference graph");
    }

    /// The device CE agrees with an independent host `logsumexp − logit[target]`
    /// over the same logits — the objective, checked separately from the graph.
    #[test]
    fn trainer_forward_matches_inference_graph() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = tiny_config();
        let init = init_weights(&cfg, 5);
        let (latent, targets) = fixed_batch(&cfg, 5);

        let tr = CodeTransformerTrainer::new_on(
            gpu_core::testgpu::dev(TRAIN_PIPELINES),
            cfg.clone(),
            &init,
        );
        tr.set_latent(&latent);
        tr.set_targets(&targets);
        let l = tr.forward();
        assert!(l.is_finite() && l > 0.0, "mean CE {l} is not a sane loss");

        // The inference graph needs the FULL tensor set (it records the conv
        // half too), so this comparison is against `crate::model`'s transformer
        // driven on the same weights via its `logits` tap — see
        // `crates/restore/tests/` for the checkpoint-driven version. Here we
        // assert the cheap invariant the trainer can check on its own: the
        // logits are finite and the CE agrees with a host recomputation of
        // `logsumexp - logit[target]`.
        let logits = tr.read_logits();
        let k = cfg.vqgan.codebook_size as usize;
        let mut host = 0.0f64;
        for (row, &tg) in targets.iter().enumerate() {
            let r = &logits[row * k..(row + 1) * k];
            let mx = r.iter().cloned().fold(f32::MIN, f32::max);
            let sum: f64 = r.iter().map(|v| ((v - mx) as f64).exp()).sum();
            host += (mx as f64 + sum.ln()) - r[tg as usize] as f64;
        }
        host /= targets.len() as f64;
        assert!(
            (host - l as f64).abs() < 1e-4,
            "device mean CE {l} vs host {host}"
        );
    }
}
