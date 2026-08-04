// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The CLIP-family forward graphs: one config-driven **text** tower serving
//! CLIP-L and OpenCLIP-bigG, and the EVA02-CLIP **image** tower.
//!
//! Both are pure dispatch assembly over shared kernels and the shared
//! Step-builders in `model::block` — no host pixel/tensor loops, no private
//! copies of norm/attention/activation math. SSA: every stage writes a fresh
//! buffer, which is also the activation cache the deferred backward will need.
//! The two documented exceptions are named at their allocation sites.
//!
//! ## Text tower (`ClipText`)
//! ```text
//! x0   = pos_add(embed(ids))                                  [B*T, H]
//! per layer i:
//!   ln1  = LayerNorm(x_i)
//!   qkv  = ln1 @ Wqkv^T + b        (fused [3H, H] — split in the checkpoint)
//!   ctx  = CAUSAL MHA(qkv)         (attn_scores/softmax/apply; 1/sqrt(hd))
//!   res  = x_i + (ctx @ Wo^T + bo)
//!   h    = act(LN2(res) @ W1^T + b1)   act = quick_gelu (CLIP-L) | gelu_erf (bigG)
//!   x_i+1= res + (h @ W2^T + b2)
//! hidden      = final_layer_norm(x_L)
//! penultimate = x_{L-1}                (NOT layer-normed — see the config)
//! pooled      = hidden[b*T + eos_index[b]]
//! text_embeds = pooled @ Wproj^T       (bigG only)
//! ```
//! Attention is **causal-mask-only**: SDXL passes `attention_mask=None`, and pad
//! rows are causally isolated, so no key-padding mask kernel is dispatched.
//!
//! ## Image tower (`EvaVision`)
//! EVA02 is **not** a vanilla ViT, so it does not go through
//! `model::vit::vit_block_fwd` — that builder has no hook for the `inner_attn_ln`
//! between the attention context and `proj`, and its MLP is a single
//! GELU-activated pair where EVA's is a SwiGLU with an interior `ffn_ln`.
//! Extending `VitBlockWeights` would change a struct literal in every ViT model
//! in the workspace for two optional hooks; the block is composed here from the
//! same *primitives* instead — `block::layernorm_fwd` for every LayerNorm,
//! `block::bidir_fwd` for the whole attention trio, `block::pick_gemm` for every
//! linear, and the `rope2d` / `silu_mul` kernels dispatched directly — so no
//! math is re-implemented. (`block::swiglu_fwd` is NOT used: it is a one-line
//! `silu_mul` dispatch behind a 16-field `block::KernelIds`, 15 of whose slots
//! this forward-only tower has no kernel for.)
//! ```text
//! x0 = [cls ; conv2d_patch(pixels)] + pos_embed                [B*577, W]
//! per block i:
//!   n1   = LayerNorm(x_i, eps=1e-6)
//!   qkv  = n1 @ Wqkv^T + b     (fused; k's bias third is 0 — see import.rs)
//!   rope2d on the q and k regions, TOKENS 1.. ONLY (cls excluded)
//!   ctx  = BIDIRECTIONAL MHA(qkv)
//!   res  = x_i + (LayerNorm(ctx, inner_ln) @ Wproj^T + b)
//!   h    = LayerNorm(SiLU(n2 @ W1^T + b1) * (n2 @ W2^T + b2), ffn_ln)
//!   x_i+1= res + (h @ W3^T + b3)
//! norm_out = LayerNorm(x_L);  head_out = norm_out[0] @ Whead^T + bhead
//! ```

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::{ClipTextConfig, EvaVisionConfig, TextAct};

// ---------------------------------------------------------------------------
// text tower
// ---------------------------------------------------------------------------

const T_EMBED: usize = 0;
const T_REGION_COPY: usize = 1;
const T_POS_ADD: usize = 2;
const T_LAYERNORM: usize = 3;
const T_MATMUL: usize = 4;
const T_MATMUL_REG2: usize = 5;
const T_BIAS_ADD: usize = 6;
const T_SCORES: usize = 7;
const T_SOFTMAX: usize = 8;
const T_APPLY: usize = 9;
const T_ADD2: usize = 10;
const T_QUICK_GELU: usize = 11;
const T_GELU_ERF: usize = 12;
// `layernorm_rows` is 13 — resolved BY NAME, never indexed directly.
// ---- backward (appended, so every index above is unchanged) ----
const T_LN_STATS: usize = 14;
const T_LAYERNORM_DX: usize = 16;
const T_LN_DGAMMA: usize = 18;
const T_LN_DBETA: usize = 19;
const T_MATMUL_DX: usize = 20;
const T_MATMUL_DW: usize = 21;
const T_MATMUL_DX_REG: usize = 22;
const T_MATMUL_DW_REG: usize = 23;
const T_BIAS_GRAD: usize = 24;
const T_ATTN_DSCORES: usize = 25;
const T_ATTN_DV: usize = 26;
const T_ATTN_DQ: usize = 27;
const T_ATTN_DK: usize = 28;
const T_QUICK_GELU_BWD: usize = 29;
const T_GELU_ERF_BWD: usize = 30;
const T_POS_BWD: usize = 31;
const T_EMB_BWD: usize = 32;

/// Text-tower kernels — forward AND backward, one list, so an inference build
/// and a training build share a device handle (`gpu_core::testgpu::dev` keys on
/// the slice address). `layernorm_rows` / `ln_stats_rows` / `layernorm_dx_rows`
/// are registered but never indexed directly: `block::LayerNormIds::resolve`
/// picks them up BY NAME wherever the device supports workgroup reductions
/// (2.3-9.1x on a P40).
///
/// The backward half is appended, never interleaved: every `T_*` index above is
/// a position in this list and reordering it is silently wrong.
pub const TEXT_PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("region_copy", kernels::REGION_COPY),
    ("pos_add", kernels::POS_ADD),
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("bias_add", kernels::BIAS_ADD),
    ("attn_scores", kernels::ATTN_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("attn_apply", kernels::ATTN_APPLY),
    ("add2", kernels::ADD2),
    ("quick_gelu", kernels::QUICK_GELU),
    ("gelu_erf", kernels::GELU_ERF),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats", kernels::LN_STATS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("bias_grad", kernels::BIAS_GRAD),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("attn_bwd_dq", kernels::ATTN_BWD_DQ),
    ("attn_bwd_dk", kernels::ATTN_BWD_DK),
    ("quick_gelu_bwd", kernels::QUICK_GELU_BWD),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("pos_bwd", kernels::POS_BWD),
    ("emb_bwd", kernels::EMB_BWD),
];

/// One text layer's SSA activations.
pub struct TextLayerBufs {
    pub ln1: DeviceBuffer,
    /// Fused `[N, 3H]` — q at 0, k at H, v at 2H.
    pub qkv: DeviceBuffer,
    /// Softmax probabilities `[B, heads, T, T]` — per layer, because the
    /// backward needs them and at T=77 the whole set is only a few MB.
    pub probs: DeviceBuffer,
    pub ctx: DeviceBuffer,
    pub attn_out: DeviceBuffer,
    pub res: DeviceBuffer,
    pub ln2: DeviceBuffer,
    pub h: DeviceBuffer,
    pub h_act: DeviceBuffer,
    pub mlp_out: DeviceBuffer,
}

/// Reverse-pass buffers + the recorded backward step list. Allocated ONLY by
/// [`ClipText::new_train_on`]; an inference build carries `None` and is
/// byte-for-byte the graph the parity ladder gates.
///
/// Every entry is a *gradient*; the activations the backward reads are the
/// forward's own SSA buffers ([`TextLayerBufs`], `x`, `hidden`, `pooled`) —
/// nothing is recomputed and nothing is aliased. The per-layer scratch is
/// reused across layers because the reverse walk holds exactly one layer's
/// intermediate grads live at a time; only `dx` is per layer, mirroring `x`.
struct TextBwd {
    /// `dx[i]` is the grad of `x[i]`: `dx[0]` = embedding-output grad,
    /// `dx[i+1]` = grad of layer `i`'s output.
    dx: Vec<DeviceBuffer>,
    /// Objective grad on `hidden` (`final_layer_norm` output), uploaded by the
    /// caller; `d_hidden` starts as a copy of it and then accumulates the
    /// EOS-pooling scatter.
    seed_hidden: DeviceBuffer,
    /// Objective grad on the tower OUTPUT — `text_embeds` when the config has a
    /// projection, else `pooled`.
    seed_out: DeviceBuffer,
    d_hidden: DeviceBuffer,
    d_pooled: DeviceBuffer,
    /// Grad of the post-attention residual `res` (the MLP branch's input).
    d_res: DeviceBuffer,
    /// Grad flowing out of a normalized branch, before the residual re-join.
    d_branch: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_scores: DeviceBuffer,
    /// Grad of the POST-activation MLP hidden (`h_act`).
    d_h_act: DeviceBuffer,
    /// Grad of the PRE-activation MLP hidden (`h`).
    d_h: DeviceBuffer,
    /// Per-row LayerNorm mean / inverse-std, recomputed per use (they are
    /// `[rows]` — cheaper to recompute than to cache, as `model::vit` does).
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    steps: Vec<Step>,
}

pub struct ClipText {
    pub gpu: Gpu,
    pub cfg: ClipTextConfig,
    pub ps: ParamStore,
    b: u32,
    t: u32,
    tokens: DeviceBuffer,
    eos_rows: DeviceBuffer,
    tok_embed: DeviceBuffer,
    /// `x[0]` = embeddings output, `x[i+1]` = output of layer `i`.
    x: Vec<DeviceBuffer>,
    layers: Vec<TextLayerBufs>,
    /// Transient score slab (pre-softmax) — the one non-SSA buffer here, and it
    /// is sound: the backward reads `probs` (cached per layer), never `scores`.
    scores: DeviceBuffer,
    hidden: DeviceBuffer,
    pooled: DeviceBuffer,
    text_embeds: Option<DeviceBuffer>,
    steps: Vec<Step>,
    bwd: Option<TextBwd>,
}

impl ClipText {
    /// Build on an existing device (tests pass `gpu_core::testgpu::dev`).
    /// **Inference**: every parameter `Frozen`, no gradient buffers, no reverse
    /// step list — the graph the forward-parity ladder gates.
    pub fn new_on(
        gpu: Gpu,
        cfg: ClipTextConfig,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> ClipText {
        ClipText::build(gpu, cfg, b, t, init, false)
    }

    /// Build a **trainable** tower on an existing device: every parameter
    /// `Role::Trainable` (grad + AdamW moments) plus the reverse step list.
    ///
    /// The forward is the SAME `build_steps` an inference build records — CLIP's
    /// text tower is already SSA (`ln1`/`qkv`/`probs`/`ctx`/`res`/`ln2`/`h`/
    /// `h_act` each own a buffer), so there is no `fwd` vs `fwd_cached` split to
    /// make here and no parity risk from the training path existing.
    pub fn new_train_on(
        gpu: Gpu,
        cfg: ClipTextConfig,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> ClipText {
        ClipText::build(gpu, cfg, b, t, init, true)
    }

    /// Trainable tower on its own device handle (`Gpu::new`), mirroring
    /// `Gpt::new` / `Seq2Seq::new`. Prefer [`ClipText::new_train_on`] in a
    /// process that already holds a device.
    pub fn new_train(
        cfg: ClipTextConfig,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> ClipText {
        ClipText::new_train_on(Gpu::new(TEXT_PIPELINES), cfg, b, t, init)
    }

    fn build(
        gpu: Gpu,
        cfg: ClipTextConfig,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
        train: bool,
    ) -> ClipText {
        assert!(t <= cfg.max_positions, "seq len {t} > max_positions {}", cfg.max_positions);
        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), role))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = (b * t) as u64;
        let h = cfg.hidden as u64;
        let i = cfg.intermediate as u64;
        let slab = b as u64 * cfg.heads as u64 * t as u64 * t as u64;
        let tokens =
            gpu.buffer("tokens", n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
        let eos_rows = gpu.buffer(
            "eos_rows",
            b as u64 * 4,
            gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
        );
        let layers: Vec<TextLayerBufs> = (0..cfg.layers)
            .map(|_| TextLayerBufs {
                ln1: gpu.storage(n * h),
                qkv: gpu.storage(n * 3 * h),
                probs: gpu.storage(slab),
                ctx: gpu.storage(n * h),
                attn_out: gpu.storage(n * h),
                res: gpu.storage(n * h),
                ln2: gpu.storage(n * h),
                h: gpu.storage(n * i),
                h_act: gpu.storage(n * i),
                mlp_out: gpu.storage(n * h),
            })
            .collect();
        let mut m = ClipText {
            tok_embed: gpu.storage(n * h),
            x: (0..=cfg.layers).map(|_| gpu.storage(n * h)).collect(),
            layers,
            scores: gpu.storage(slab),
            hidden: gpu.storage(n * h),
            pooled: gpu.storage(b as u64 * h),
            text_embeds: cfg.projection.map(|p| gpu.storage(b as u64 * p as u64)),
            tokens,
            eos_rows,
            gpu,
            cfg,
            ps,
            b,
            t,
            steps: Vec::new(),
            bwd: None,
        };
        m.steps = m.build_steps();
        if train {
            let g = &m.gpu;
            let out_w = m.cfg.projection.unwrap_or(m.cfg.hidden) as u64;
            m.bwd = Some(TextBwd {
                dx: (0..=m.cfg.layers).map(|_| g.storage(n * h)).collect(),
                seed_hidden: g.storage(n * h),
                seed_out: g.storage(b as u64 * out_w),
                d_hidden: g.storage(n * h),
                d_pooled: g.storage(b as u64 * h),
                d_res: g.storage(n * h),
                d_branch: g.storage(n * h),
                d_tmp: g.storage(n * h),
                d_ctx: g.storage(n * h),
                d_qkv: g.storage(n * 3 * h),
                d_scores: g.storage(slab),
                d_h_act: g.storage(n * i),
                d_h: g.storage(n * i),
                mean: g.storage(n),
                inv: g.storage(n),
                steps: Vec::new(),
            });
            let steps = m.build_bwd_steps();
            m.bwd.as_mut().expect("bwd allocated").steps = steps;
        }
        m
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, T_MATMUL, T_MATMUL_REG2, false)
    }

    /// Backward-GEMM kernel + dispatch threads, picked on the OUTPUT dims — the
    /// same policy `block::pick_gemm` implements for the forward (the tiled
    /// `matmul_{dx,dw}_reg` share `matmul_reg2`'s 128x128 / 256-thread shape and
    /// are bit-compatible with the naive kernels). `matmul_dw` writes `[n,k]`
    /// and `matmul_dx` writes `[m,k]`, so each passes its own output dims.
    fn bwd_gemm(&self, rows: u32, cols: u32, naive: usize, reg: usize) -> (usize, u32) {
        block::pick_gemm(rows as usize, cols as usize, naive, reg, false)
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let (b, t) = (self.b, self.t);
        let n = b * t;
        let h = c.hidden;
        let inter = c.intermediate;
        let hd = c.head_dim();
        // Same handle the backward uses; for the forward only `layernorm` /
        // `layernorm_rows` are consulted, so this is identical to the
        // `resolve_fwd` an inference-only tower would build.
        let ln = block::LayerNormIds::resolve(g, T_LAYERNORM, T_LN_STATS, T_LAYERNORM_DX);
        let act = match c.act {
            TextAct::QuickGelu => T_QUICK_GELU,
            TextAct::GeluErf => T_GELU_ERF,
        };
        let mut s = Vec::new();

        // ---- embeddings ----
        // `embed` Params: [d_model, seq_len]; bufs [tokens(u32), table, out].
        s.push(g.step(T_EMBED, &[&self.tokens, self.w("tok.weight"), &self.tok_embed], &[h, n], n * h));
        // `region_copy` Params: [rows, width, row_stride, off] — a whole-buffer
        // copy at off=0, so `pos_add` (in place) leaves the tok_embed tap intact.
        s.push(g.step(T_REGION_COPY, &[&self.tok_embed, &self.x[0]], &[n, h, h, 0], n * h));
        // `pos_add` Params: [total, d_model, t]; bufs [x(rw), pos].
        s.push(g.step(T_POS_ADD, &[&self.x[0], self.w("pos.weight")], &[n * h, h, t], n * h));

        for l in 0..c.layers as usize {
            let lb = &self.layers[l];
            let p = format!("blocks.{l}");
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &self.x[l],
                self.w(&format!("{p}.ln1.weight")),
                self.w(&format!("{p}.ln1.bias")),
                &lb.ln1,
                h,
                n,
                c.eps,
            ));
            let (mk, mt) = self.gemm(n, 3 * h);
            s.push(g.step(mk, &[&lb.ln1, self.w(&format!("{p}.qkv.weight")), &lb.qkv], &[n, h, 3 * h], mt));
            // `bias_add` Params: [m, n]; bufs [out(rw), bias].
            s.push(g.step(T_BIAS_ADD, &[&lb.qkv, self.w(&format!("{p}.qkv.bias"))], &[n, 3 * h], n * 3 * h));

            // Causal MHA over the fused qkv.
            // `attn_scores`  Params: [bsz, n_heads, tcols, head_dim, qkv_stride, q_off, k_off]
            // `attn_softmax` Params: [bsz, n_heads, tcols]
            // `attn_apply`   Params: [bsz, n_heads, tcols, head_dim, qkv_stride, v_off, d_model]
            s.push(g.step(T_SCORES, &[&lb.qkv, &self.scores], &[b, c.heads, t, hd, 3 * h, 0, h], b * c.heads * t * t));
            s.push(g.step(T_SOFTMAX, &[&self.scores, &lb.probs], &[b, c.heads, t], b * c.heads * t));
            s.push(g.step(T_APPLY, &[&lb.probs, &lb.qkv, &lb.ctx], &[b, c.heads, t, hd, 3 * h, 2 * h, h], b * c.heads * t * hd));

            let (mk, mt) = self.gemm(n, h);
            s.push(g.step(mk, &[&lb.ctx, self.w(&format!("{p}.proj.weight")), &lb.attn_out], &[n, h, h], mt));
            s.push(g.step(T_BIAS_ADD, &[&lb.attn_out, self.w(&format!("{p}.proj.bias"))], &[n, h], n * h));
            s.push(g.step(T_ADD2, &[&self.x[l], &lb.attn_out, &lb.res], &[n * h], n * h));

            s.push(block::layernorm_fwd(
                g,
                &ln,
                &lb.res,
                self.w(&format!("{p}.ln2.weight")),
                self.w(&format!("{p}.ln2.bias")),
                &lb.ln2,
                h,
                n,
                c.eps,
            ));
            let (mk, mt) = self.gemm(n, inter);
            s.push(g.step(mk, &[&lb.ln2, self.w(&format!("{p}.fc1.weight")), &lb.h], &[n, h, inter], mt));
            s.push(g.step(T_BIAS_ADD, &[&lb.h, self.w(&format!("{p}.fc1.bias"))], &[n, inter], n * inter));
            s.push(g.step(act, &[&lb.h, &lb.h_act], &[n * inter], n * inter));
            let (mk, mt) = self.gemm(n, h);
            s.push(g.step(mk, &[&lb.h_act, self.w(&format!("{p}.fc2.weight")), &lb.mlp_out], &[n, inter, h], mt));
            s.push(g.step(T_BIAS_ADD, &[&lb.mlp_out, self.w(&format!("{p}.fc2.bias"))], &[n, h], n * h));
            s.push(g.step(T_ADD2, &[&lb.res, &lb.mlp_out, &self.x[l + 1]], &[n * h], n * h));
        }

        s.push(block::layernorm_fwd(
            g,
            &ln,
            &self.x[c.layers as usize],
            self.w("final_norm.weight"),
            self.w("final_norm.bias"),
            &self.hidden,
            h,
            n,
            c.eps,
        ));
        // EOS pooling as a row gather: `embed` with the ABSOLUTE row indices
        // `b*T + eos_index[b]` and the hidden states as the "table".
        s.push(g.step(T_EMBED, &[&self.eos_rows, &self.hidden, &self.pooled], &[h, b], b * h));
        if let Some(te) = &self.text_embeds {
            let p = c.projection.expect("text_embeds without projection dim");
            let (mk, mt) = self.gemm(b, p);
            s.push(g.step(mk, &[&self.pooled, self.w("text_projection.weight"), te], &[b, h, p], mt));
        }
        s
    }

    /// The reverse pass, recorded once at build time — the exact adjoint of
    /// [`ClipText::build_steps`], walked bottom-up.
    ///
    /// Seeds. The objective is supplied as two output gradients (see
    /// [`ClipText::backward`]): `seed_hidden` on `final_layer_norm(x_L)` and
    /// `seed_out` on the tower output (`text_embeds` if the config projects,
    /// else `pooled`). That covers both consumers CLIP actually has — SDXL's
    /// sequence conditioning and its pooled/projected embedding — and makes the
    /// EOS gather's adjoint a *second* contribution into `d_hidden`, which is
    /// the accumulation the gradcheck has to see.
    ///
    /// Every kernel here is gather-based (one invocation per element of the
    /// output it writes) and there are no atomics: `emb_bwd` / `pos_bwd` loop
    /// the source rows inside the invocation that owns the destination row.
    ///
    /// Clears. NOTHING in this list needs a pre-zeroed transient: `matmul_dx`
    /// runs with `accumulate = 0`, `layernorm_dx` / `add2` / the activation
    /// backwards / the four `attn_bwd_*` kernels all ASSIGN, and `region_copy`
    /// seeds `d_hidden` before `emb_bwd` adds to it. The PARAMETER grads do
    /// accumulate (`matmul_dw`, `bias_grad`, `layernorm_dgamma/dbeta`,
    /// `pos_bwd`, `emb_bwd`) — they are cleared exactly once per step by
    /// `ParamStore::zero_grads`, so they must NOT appear in a submit's clear
    /// list or every earlier contribution is dropped.
    fn build_bwd_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let bw = self.bwd.as_ref().expect("build_bwd_steps in training mode only");
        let (b, t) = (self.b, self.t);
        let n = b * t;
        let h = c.hidden;
        let inter = c.intermediate;
        let hd = c.head_dim();
        let ln = block::LayerNormIds::resolve(g, T_LAYERNORM, T_LN_STATS, T_LAYERNORM_DX);
        let act_bwd = match c.act {
            TextAct::QuickGelu => T_QUICK_GELU_BWD,
            TextAct::GeluErf => T_GELU_ERF_BWD,
        };
        let gr = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::new();

        // ---- head: optional projection, then EOS pooling ----
        if let Some(te_dim) = c.projection {
            // `matmul_dw` Params: [m, k, n]; bufs [dy, x, dw] — ACCUMULATES.
            // Forward was `text_embeds[b, p] = pooled[b, h] @ Wproj[p, h]^T`.
            let (dw, dwt) = self.bwd_gemm(te_dim, h, T_MATMUL_DW, T_MATMUL_DW_REG);
            s.push(g.step(dw, &[&bw.seed_out, &self.pooled, gr("text_projection.weight")], &[b, h, te_dim], dwt));
            // `matmul_dx` Params: [m, k, n, accumulate]; bufs [dy, w, dx].
            let (dx, dxt) = self.bwd_gemm(b, h, T_MATMUL_DX, T_MATMUL_DX_REG);
            s.push(g.step(dx, &[&bw.seed_out, self.w("text_projection.weight"), &bw.d_pooled], &[b, h, te_dim, 0], dxt));
        } else {
            // No projection: the tower output IS `pooled`.
            // `region_copy` Params: [rows, width, row_stride, off] — whole buffer.
            s.push(g.step(T_REGION_COPY, &[&bw.seed_out, &bw.d_pooled], &[b, h, h, 0], b * h));
        }
        // `d_hidden` starts at the sequence seed, then the EOS gather's adjoint
        // scatters `d_pooled` onto the pooled rows. The forward gather is
        // `embed(eos_rows, hidden -> pooled)`, whose adjoint is `emb_bwd` with
        // the SAME index buffer, `hidden`'s row count as the "vocab", and
        // `d_pooled` as the row grads. It accumulates, which is why the seed
        // copy has to come first.
        s.push(g.step(T_REGION_COPY, &[&bw.seed_hidden, &bw.d_hidden], &[n, h, h, 0], n * h));
        // `emb_bwd` Params: [n_rows, d_model, vocab]; bufs [tokens(u32), d_x, grad_emb].
        s.push(g.step(T_EMB_BWD, &[&self.eos_rows, &bw.d_pooled, &bw.d_hidden], &[b, h, n], n * h));

        // ---- final LayerNorm ----
        let last = c.layers as usize;
        s.push(block::ln_stats_fwd(g, &ln, &self.x[last], &bw.mean, &bw.inv, h, n, c.eps));
        // `layernorm_dgamma` Params: [d_model, n_rows]; bufs [dy, x, mean, inv, dgamma].
        s.push(g.step(T_LN_DGAMMA, &[&bw.d_hidden, &self.x[last], &bw.mean, &bw.inv, gr("final_norm.weight")], &[h, n], h));
        // `layernorm_dbeta` Params: [d_model, n_rows]; bufs [dy, dbeta].
        s.push(g.step(T_LN_DBETA, &[&bw.d_hidden, gr("final_norm.bias")], &[h, n], h));
        s.push(block::layernorm_dx_bwd(g, &ln, &self.x[last], self.w("final_norm.weight"), &bw.d_hidden, &bw.dx[last], h, n, c.eps));

        for l in (0..c.layers as usize).rev() {
            let lb = &self.layers[l];
            let p = format!("blocks.{l}");
            let d_out = &bw.dx[l + 1]; // grad of this layer's output

            // ---- MLP branch ----
            // `bias_grad` Params: [m, n]; bufs [dy, dbias] — one thread per feature.
            s.push(g.step(T_BIAS_GRAD, &[d_out, gr(&format!("{p}.fc2.bias"))], &[n, h], h));
            let (dw, dwt) = self.bwd_gemm(h, inter, T_MATMUL_DW, T_MATMUL_DW_REG);
            s.push(g.step(dw, &[d_out, &lb.h_act, gr(&format!("{p}.fc2.weight"))], &[n, inter, h], dwt));
            let (dx, dxt) = self.bwd_gemm(n, inter, T_MATMUL_DX, T_MATMUL_DX_REG);
            s.push(g.step(dx, &[d_out, self.w(&format!("{p}.fc2.weight")), &bw.d_h_act], &[n, inter, h, 0], dxt));
            // The activation backward reads the PRE-activation `h` (post-bias),
            // never `h_act`. Params: a single `total`; bufs [x, dout, dx].
            s.push(g.step(act_bwd, &[&lb.h, &bw.d_h_act, &bw.d_h], &[n * inter], n * inter));
            s.push(g.step(T_BIAS_GRAD, &[&bw.d_h, gr(&format!("{p}.fc1.bias"))], &[n, inter], inter));
            let (dw, dwt) = self.bwd_gemm(inter, h, T_MATMUL_DW, T_MATMUL_DW_REG);
            s.push(g.step(dw, &[&bw.d_h, &lb.ln2, gr(&format!("{p}.fc1.weight"))], &[n, h, inter], dwt));
            let (dx, dxt) = self.bwd_gemm(n, h, T_MATMUL_DX, T_MATMUL_DX_REG);
            s.push(g.step(dx, &[&bw.d_h, self.w(&format!("{p}.fc1.weight")), &bw.d_branch], &[n, h, inter, 0], dxt));
            s.push(block::ln_stats_fwd(g, &ln, &lb.res, &bw.mean, &bw.inv, h, n, c.eps));
            s.push(g.step(T_LN_DGAMMA, &[&bw.d_branch, &lb.res, &bw.mean, &bw.inv, gr(&format!("{p}.ln2.weight"))], &[h, n], h));
            s.push(g.step(T_LN_DBETA, &[&bw.d_branch, gr(&format!("{p}.ln2.bias"))], &[h, n], h));
            s.push(block::layernorm_dx_bwd(g, &ln, &lb.res, self.w(&format!("{p}.ln2.weight")), &bw.d_branch, &bw.d_tmp, h, n, c.eps));
            // residual re-join: d_res = d_out (pass-through) + branch grad.
            s.push(g.step(T_ADD2, &[d_out, &bw.d_tmp, &bw.d_res], &[n * h], n * h));

            // ---- attention branch ----
            s.push(g.step(T_BIAS_GRAD, &[&bw.d_res, gr(&format!("{p}.proj.bias"))], &[n, h], h));
            let (dw, dwt) = self.bwd_gemm(h, h, T_MATMUL_DW, T_MATMUL_DW_REG);
            s.push(g.step(dw, &[&bw.d_res, &lb.ctx, gr(&format!("{p}.proj.weight"))], &[n, h, h], dwt));
            let (dx, dxt) = self.bwd_gemm(n, h, T_MATMUL_DX, T_MATMUL_DX_REG);
            s.push(g.step(dx, &[&bw.d_res, self.w(&format!("{p}.proj.weight")), &bw.d_ctx], &[n, h, h, 0], dxt));
            // Causal attention backward. The dscores/dv pair carries the v-region
            // params `[.., v_off, d_model]`, the dq/dk pair the q/k-region params
            // `[.., q_off, k_off]` — the same split `gpt::model` dispatches. All
            // four ASSIGN their region of `d_qkv`, which is why it needs no clear.
            // Note they read `probs` (per-layer cache), NOT `scores`.
            let pv = [b, c.heads, t, hd, 3 * h, 2 * h, h];
            let pqk = [b, c.heads, t, hd, 3 * h, 0, h];
            s.push(g.step(T_ATTN_DSCORES, &[&bw.d_ctx, &lb.qkv, &lb.probs, &bw.d_scores], &pv, b * c.heads * t));
            s.push(g.step(T_ATTN_DV, &[&lb.probs, &bw.d_ctx, &bw.d_qkv], &pv, b * c.heads * t * hd));
            s.push(g.step(T_ATTN_DQ, &[&bw.d_scores, &lb.qkv, &bw.d_qkv], &pqk, b * c.heads * t * hd));
            s.push(g.step(T_ATTN_DK, &[&bw.d_scores, &lb.qkv, &bw.d_qkv], &pqk, b * c.heads * t * hd));
            s.push(g.step(T_BIAS_GRAD, &[&bw.d_qkv, gr(&format!("{p}.qkv.bias"))], &[n, 3 * h], 3 * h));
            let (dw, dwt) = self.bwd_gemm(3 * h, h, T_MATMUL_DW, T_MATMUL_DW_REG);
            s.push(g.step(dw, &[&bw.d_qkv, &lb.ln1, gr(&format!("{p}.qkv.weight"))], &[n, h, 3 * h], dwt));
            let (dx, dxt) = self.bwd_gemm(n, h, T_MATMUL_DX, T_MATMUL_DX_REG);
            s.push(g.step(dx, &[&bw.d_qkv, self.w(&format!("{p}.qkv.weight")), &bw.d_branch], &[n, h, 3 * h, 0], dxt));
            s.push(block::ln_stats_fwd(g, &ln, &self.x[l], &bw.mean, &bw.inv, h, n, c.eps));
            s.push(g.step(T_LN_DGAMMA, &[&bw.d_branch, &self.x[l], &bw.mean, &bw.inv, gr(&format!("{p}.ln1.weight"))], &[h, n], h));
            s.push(g.step(T_LN_DBETA, &[&bw.d_branch, gr(&format!("{p}.ln1.bias"))], &[h, n], h));
            s.push(block::layernorm_dx_bwd(g, &ln, &self.x[l], self.w(&format!("{p}.ln1.weight")), &bw.d_branch, &bw.d_tmp, h, n, c.eps));
            s.push(g.step(T_ADD2, &[&bw.d_res, &bw.d_tmp, &bw.dx[l]], &[n * h], n * h));
        }

        // ---- embeddings ----
        // Forward: `x0 = region_copy(embed(ids)); pos_add(x0, pos)` — the
        // positional add is in place, so `dx[0]` is BOTH the positional grad
        // source and (through the identity region_copy) the token-embedding
        // grad source.
        // `pos_bwd` Params: [b, t, d_model]; bufs [d_x, dpos] — ACCUMULATES, and
        // writes only the first `t` of `max_positions` rows.
        s.push(g.step(T_POS_BWD, &[&bw.dx[0], gr("pos.weight")], &[b, t, h], t * h));
        s.push(g.step(T_EMB_BWD, &[&self.tokens, &bw.dx[0], gr("tok.weight")], &[n, h, c.vocab], c.vocab * h));
        s
    }

    /// Zero every parameter gradient. Call once per training step, BEFORE
    /// [`ClipText::backward`] — the reverse pass accumulates into them.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Run the reverse pass for the objective whose gradients w.r.t. the two
    /// tower outputs are `d_hidden` (`[B*T, H]`, the grad of
    /// `final_layer_norm(x_L)`) and `d_out` (`[B, P]` when the config has a
    /// projection, else `[B, H]` on `pooled`).
    ///
    /// The forward must already have run on the current tokens/weights — the
    /// backward reads the SSA activation buffers it left behind.
    pub fn backward(&self, d_hidden: &[f32], d_out: &[f32]) {
        let bw = self.bwd.as_ref().expect("backward() needs ClipText::new_train_on");
        let n = self.n();
        let h = self.cfg.hidden as usize;
        let out_w = self.cfg.projection.unwrap_or(self.cfg.hidden) as usize;
        assert_eq!(d_hidden.len(), n * h, "d_hidden must be [B*T, H]");
        assert_eq!(d_out.len(), self.b as usize * out_w, "d_out must be [B, P]");
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        self.gpu.write(&bw.seed_hidden, &bits(d_hidden));
        self.gpu.write(&bw.seed_out, &bits(d_out));
        self.gpu.submit(&[], &bw.steps);
    }

    /// Whether this tower was built trainable (`new_train_on`).
    pub fn is_trainable(&self) -> bool {
        self.bwd.is_some()
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(self.w(name), &bits);
    }
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// Set the token ids (`[B*T]`, row-major) and derive the EOS pooling rows.
    ///
    /// The pooling index is `argmax(ids)` per row — transformers' legacy
    /// `eos_token_id == 2` branch, which for a CLIP vocabulary is exactly the
    /// FIRST occurrence of `<|endoftext|>` (49407, the largest id in the
    /// vocabulary). Asserted here rather than assumed: a row whose argmax is
    /// not the eos id would silently pool the wrong token.
    pub fn set_tokens(&self, ids: &[u32]) {
        assert_eq!(ids.len(), (self.b * self.t) as usize, "token count");
        self.gpu.write(&self.tokens, ids);
        let rows: Vec<u32> = (0..self.b as usize)
            .map(|s| {
                let row = &ids[s * self.t as usize..(s + 1) * self.t as usize];
                // FIRST argmax, like torch — NOT `Iterator::max_by_key`, which
                // returns the LAST maximal element. With CLIP-L's pad token
                // (`<|endoftext|>` == the eos id == the largest id in the
                // vocabulary) every pad slot ties the max, so the last-wins
                // rule silently pools row T-1 instead of the real EOS.
                let mut k = 0usize;
                for (i, &v) in row.iter().enumerate() {
                    if v > row[k] {
                        k = i;
                    }
                }
                assert_eq!(
                    row[k], self.cfg.eos_id,
                    "sample {s}: argmax token {} is not eos {}",
                    row[k], self.cfg.eos_id
                );
                s as u32 * self.t + k as u32
            })
            .collect();
        self.gpu.write(&self.eos_rows, &rows);
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    // ---- parity / inference taps ----
    fn n(&self) -> usize {
        (self.b * self.t) as usize
    }
    pub fn read_tok_embed(&self) -> Vec<f32> {
        self.gpu.read(&self.tok_embed, self.n() * self.cfg.hidden as usize)
    }
    /// `x[0]` = embeddings output; `x[i+1]` = output of encoder layer `i`.
    pub fn read_x(&self, i: usize) -> Vec<f32> {
        self.gpu.read(&self.x[i], self.n() * self.cfg.hidden as usize)
    }
    pub fn read_layer_tap(&self, l: usize, tap: TextTap) -> Vec<f32> {
        let lb = &self.layers[l];
        let (buf, w) = match tap {
            TextTap::Ln1 => (&lb.ln1, self.cfg.hidden),
            TextTap::Qkv => (&lb.qkv, 3 * self.cfg.hidden),
            TextTap::AttnOut => (&lb.attn_out, self.cfg.hidden),
            TextTap::Ln2 => (&lb.ln2, self.cfg.hidden),
            TextTap::Fc1 => (&lb.h, self.cfg.intermediate),
            TextTap::MlpOut => (&lb.mlp_out, self.cfg.hidden),
        };
        self.gpu.read(buf, self.n() * w as usize)
    }
    /// `final_layer_norm(x_L)` — transformers' `last_hidden_state`.
    pub fn read_hidden(&self) -> Vec<f32> {
        self.gpu.read(&self.hidden, self.n() * self.cfg.hidden as usize)
    }
    /// diffusers' `hidden_states[-2]`: the output of layer `layers-2`, NOT
    /// layer-normed. This is what SDXL concatenates into `prompt_embeds`.
    pub fn read_penultimate(&self) -> Vec<f32> {
        self.read_x(self.cfg.penultimate_layer() as usize + 1)
    }
    pub fn read_pooled(&self) -> Vec<f32> {
        self.gpu.read(&self.pooled, self.b as usize * self.cfg.hidden as usize)
    }
    /// `text_projection(pooled)` — bigG only; SDXL's `pooled_prompt_embeds`.
    pub fn read_text_embeds(&self) -> Option<Vec<f32>> {
        let te = self.text_embeds.as_ref()?;
        let p = self.cfg.projection? as usize;
        Some(self.gpu.read(te, self.b as usize * p))
    }
}

/// Per-layer taps the parity test replays.
#[derive(Clone, Copy, Debug)]
pub enum TextTap {
    Ln1,
    /// Fused `[N, 3H]`: q at `[.., 0..H]`, k at `[.., H..2H]`, v at `[.., 2H..]`.
    Qkv,
    AttnOut,
    Ln2,
    Fc1,
    MlpOut,
}

// ---------------------------------------------------------------------------
// EVA02 image tower
// ---------------------------------------------------------------------------

const V_CONV2D: usize = 0;
const V_NCHW_NLC: usize = 1;
const V_REGION_COPY: usize = 2;
const V_BIAS_ADD: usize = 3;
const V_ADD2: usize = 4;
const V_LAYERNORM: usize = 5;
const V_MATMUL: usize = 6;
const V_MATMUL_REG2: usize = 7;
const V_ROPE2D: usize = 8;
const V_SCORES: usize = 9;
const V_SOFTMAX: usize = 10;
const V_APPLY: usize = 11;
const V_SILU_MUL: usize = 12;
const V_L2NORM: usize = 13;
const V_POS_ADD: usize = 14;

pub const VISION_PIPELINES: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("region_copy", kernels::REGION_COPY),
    ("bias_add", kernels::BIAS_ADD),
    ("add2", kernels::ADD2),
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("rope2d", kernels::ROPE2D),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("pos_add", kernels::POS_ADD),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
];

/// One EVA block's SSA activations.
pub struct EvaBlockBufs {
    pub norm1: DeviceBuffer,
    pub qkv: DeviceBuffer,
    pub ctx: DeviceBuffer,
    pub inner_ln: DeviceBuffer,
    pub attn_proj: DeviceBuffer,
    pub res: DeviceBuffer,
    pub norm2: DeviceBuffer,
    pub w1: DeviceBuffer,
    pub w2: DeviceBuffer,
    pub swiglu: DeviceBuffer,
    pub ffn_ln: DeviceBuffer,
    pub mlp_out: DeviceBuffer,
}

pub struct EvaVision {
    pub gpu: Gpu,
    pub cfg: EvaVisionConfig,
    pub ps: ParamStore,
    b: u32,
    pixels: DeviceBuffer,
    rope_cos: DeviceBuffer,
    rope_sin: DeviceBuffer,
    /// All-ones gain for `l2norm_scale` (the kernel is the QK-norm form with a
    /// learnable per-dim scale; a plain L2 normalize is `g = 1`).
    l2_ones: DeviceBuffer,
    patch_nchw: DeviceBuffer,
    x: Vec<DeviceBuffer>,
    blocks: Vec<EvaBlockBufs>,
    /// Shared attention slabs. NOT SSA, deliberately: at T=577 one
    /// `[B, 16, 577, 577]` f32 slab is 21 MB, so a per-block softmax cache would
    /// be ~512 MB for a 24-block tower. The backward should use
    /// `block::chunked_bidir_bwd`'s per-chunk recompute rather than cache these.
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    norm_out: DeviceBuffer,
    head_out: DeviceBuffer,
    cls_l2: DeviceBuffer,
    steps: Vec<Step>,
}

impl EvaVision {
    pub fn new_on(
        gpu: Gpu,
        cfg: EvaVisionConfig,
        b: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> EvaVision {
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let w = cfg.width as u64;
        let m = cfg.mlp_hidden as u64;
        let seq = cfg.seq_len() as u64;
        let n = b as u64 * seq;
        let np = b as u64 * cfg.num_patches() as u64;
        let slab = b as u64 * cfg.heads as u64 * seq * seq;
        let (cos, sin) = cfg.rope_tables();
        let blocks: Vec<EvaBlockBufs> = (0..cfg.layers)
            .map(|_| EvaBlockBufs {
                norm1: gpu.storage(n * w),
                qkv: gpu.storage(n * 3 * w),
                ctx: gpu.storage(n * w),
                inner_ln: gpu.storage(n * w),
                attn_proj: gpu.storage(n * w),
                res: gpu.storage(n * w),
                norm2: gpu.storage(n * w),
                w1: gpu.storage(n * m),
                w2: gpu.storage(n * m),
                swiglu: gpu.storage(n * m),
                ffn_ln: gpu.storage(n * m),
                mlp_out: gpu.storage(n * w),
            })
            .collect();
        let px = b as u64 * 3 * cfg.image_size as u64 * cfg.image_size as u64;
        let mut v = EvaVision {
            pixels: gpu.storage(px),
            rope_cos: gpu.storage_init("eva_rope_cos", &cos),
            rope_sin: gpu.storage_init("eva_rope_sin", &sin),
            l2_ones: gpu.storage_init("eva_l2_ones", &vec![1.0f32; cfg.embed_dim as usize]),
            patch_nchw: gpu.storage(np * w),
            x: (0..=cfg.layers).map(|_| gpu.storage(n * w)).collect(),
            blocks,
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            norm_out: gpu.storage(n * w),
            head_out: gpu.storage(b as u64 * cfg.embed_dim as u64),
            cls_l2: gpu.storage(b as u64 * cfg.embed_dim as u64),
            gpu,
            cfg,
            ps,
            b,
            steps: Vec::new(),
        };
        v.steps = v.build_steps();
        v
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, V_MATMUL, V_MATMUL_REG2, false)
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let b = self.b;
        let w = c.width;
        let m = c.mlp_hidden;
        let seq = c.seq_len();
        let n = b * seq;
        let npatch = c.num_patches();
        let grid = c.grid();
        let hd = c.head_dim();
        let half = c.rope_half();
        let ln = block::LayerNormIds::resolve_fwd(g, V_LAYERNORM);
        // The four backward slots are deliberately unregistered in this
        // forward-only tower. `usize::MAX` makes any future `block::bidir_bwd`
        // call panic on the pipeline lookup instead of silently dispatching some
        // other kernel; the backward workstream registers the real indices.
        const UNREGISTERED: usize = usize::MAX;
        let bidir_ids = block::BidirIds {
            scores: V_SCORES,
            softmax: V_SOFTMAX,
            apply: V_APPLY,
            dscores: UNREGISTERED,
            dv: UNREGISTERED,
            dq: UNREGISTERED,
            dk: UNREGISTERED,
        };
        let bidir = block::Bidir {
            b,
            t: seq,
            n_heads: c.heads,
            head_dim: hd,
            stride: 3 * w,
            q_off: 0,
            k_off: w,
            v_off: 2 * w,
        };
        let mut s = Vec::new();

        // ---- stem ----
        // `conv2d` Params: [N, Cin, H, W, Cout, K, stride, pad, Ho, Wo].
        s.push(g.step(
            V_CONV2D,
            &[&self.pixels, self.w("patch.weight"), &self.patch_nchw],
            &[b, 3, c.image_size, c.image_size, w, c.patch, c.patch, 0, grid, grid],
            b * w * grid * grid,
        ));
        // Per sample: NCHW -> NLC straight into rows 1.. of x[0], cls into row 0,
        // then the patch bias on the same rows. Row offsets are multiples of
        // `w` (1024), hence of the 64-float binding alignment.
        for si in 0..b as u64 {
            let src = si * w as u64 * npatch as u64;
            let dst = (si * seq as u64 + 1) * w as u64;
            // `nchw_nlc` Params: [total, c, hw].
            s.push(g.step_sliced(
                V_NCHW_NLC,
                &[&self.patch_nchw, &self.x[0]],
                &[(src, 0), (dst, 0)],
                &[w * npatch, w, npatch],
                w * npatch,
            ));
            s.push(g.step_sliced(
                V_BIAS_ADD,
                &[&self.x[0], self.w("patch.bias")],
                &[(dst, 0), (0, 0)],
                &[npatch, w],
                npatch * w,
            ));
            // `region_copy` Params: [rows, width, row_stride, off] — one row.
            s.push(g.step_sliced(
                V_REGION_COPY,
                &[self.w("cls_token"), &self.x[0]],
                &[(0, 0), (si * seq as u64 * w as u64, 0)],
                &[1, w, w, 0],
                w,
            ));
        }
        // pos_embed is [seq, W] and the batch is [b*seq, W]; `pos_add`'s row
        // modulo handles the broadcast. Params: [total, d_model, t].
        s.push(g.step(V_POS_ADD, &[&self.x[0], self.w("pos_embed")], &[n * w, w, seq], n * w));

        for l in 0..c.layers as usize {
            let bb = &self.blocks[l];
            let p = format!("blocks.{l}");
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &self.x[l],
                self.w(&format!("{p}.norm1.weight")),
                self.w(&format!("{p}.norm1.bias")),
                &bb.norm1,
                w,
                n,
                c.eps,
            ));
            let (mk, mt) = self.gemm(n, 3 * w);
            s.push(g.step(mk, &[&bb.norm1, self.w(&format!("{p}.qkv.weight")), &bb.qkv], &[n, w, 3 * w], mt));
            s.push(g.step(V_BIAS_ADD, &[&bb.qkv, self.w(&format!("{p}.qkv.bias"))], &[n, 3 * w], n * 3 * w));

            // 2D RoPE on q and k, EXCLUDING the cls token: bind the fused qkv at
            // row 1 of each sample. `rope2d` Params:
            //   [rows, heads, half, row_stride, off, tmod, sign(f32 bits)]
            for si in 0..b as u64 {
                let off = (si * seq as u64 + 1) * 3 * w as u64;
                // region offsets within the fused row: q at 0, k at W.
                for roff in [0u32, w] {
                    s.push(g.step_sliced(
                        V_ROPE2D,
                        &[&bb.qkv, &self.rope_cos, &self.rope_sin],
                        &[(off, 0), (0, 0), (0, 0)],
                        &[npatch, c.heads, half, 3 * w, roff, npatch, f(1.0)],
                        npatch * c.heads * half,
                    ));
                }
            }

            // Bidirectional MHA (non-causal), through the SHARED builder so the
            // seven-field `attn_*_bidir` param lists have exactly one home.
            s.extend(block::bidir_fwd(g, &bidir_ids, &bidir, &bb.qkv, &self.scores, &self.probs, &bb.ctx));

            // subln: the attention context is LayerNorm'd BEFORE `proj`.
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &bb.ctx,
                self.w(&format!("{p}.inner_ln.weight")),
                self.w(&format!("{p}.inner_ln.bias")),
                &bb.inner_ln,
                w,
                n,
                c.eps,
            ));
            let (mk, mt) = self.gemm(n, w);
            s.push(g.step(mk, &[&bb.inner_ln, self.w(&format!("{p}.proj.weight")), &bb.attn_proj], &[n, w, w], mt));
            s.push(g.step(V_BIAS_ADD, &[&bb.attn_proj, self.w(&format!("{p}.proj.bias"))], &[n, w], n * w));
            s.push(g.step(V_ADD2, &[&self.x[l], &bb.attn_proj, &bb.res], &[n * w], n * w));

            // ---- naive SwiGLU MLP with the interior ffn_ln ----
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &bb.res,
                self.w(&format!("{p}.norm2.weight")),
                self.w(&format!("{p}.norm2.bias")),
                &bb.norm2,
                w,
                n,
                c.eps,
            ));
            let (mk, mt) = self.gemm(n, m);
            s.push(g.step(mk, &[&bb.norm2, self.w(&format!("{p}.w1.weight")), &bb.w1], &[n, w, m], mt));
            s.push(g.step(V_BIAS_ADD, &[&bb.w1, self.w(&format!("{p}.w1.bias"))], &[n, m], n * m));
            let (mk, mt) = self.gemm(n, m);
            s.push(g.step(mk, &[&bb.norm2, self.w(&format!("{p}.w2.weight")), &bb.w2], &[n, w, m], mt));
            s.push(g.step(V_BIAS_ADD, &[&bb.w2, self.w(&format!("{p}.w2.bias"))], &[n, m], n * m));
            // `silu_mul` Params: a SINGLE `total` (not [rows, cols]).
            s.push(g.step(V_SILU_MUL, &[&bb.w1, &bb.w2, &bb.swiglu], &[n * m], n * m));
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &bb.swiglu,
                self.w(&format!("{p}.ffn_ln.weight")),
                self.w(&format!("{p}.ffn_ln.bias")),
                &bb.ffn_ln,
                m,
                n,
                c.eps,
            ));
            let (mk, mt) = self.gemm(n, w);
            s.push(g.step(mk, &[&bb.ffn_ln, self.w(&format!("{p}.w3.weight")), &bb.mlp_out], &[n, m, w], mt));
            s.push(g.step(V_BIAS_ADD, &[&bb.mlp_out, self.w(&format!("{p}.w3.bias"))], &[n, w], n * w));
            s.push(g.step(V_ADD2, &[&bb.res, &bb.mlp_out, &self.x[l + 1]], &[n * w], n * w));
        }

        s.push(block::layernorm_fwd(
            g,
            &ln,
            &self.x[c.layers as usize],
            self.w("norm.weight"),
            self.w("norm.bias"),
            &self.norm_out,
            w,
            n,
            c.eps,
        ));
        // head over the cls row of each sample: `use_mean_pooling=False` ->
        // `norm(x)[:, 0]`. Row `si*seq` is a multiple of 64 floats (w = 1024).
        for si in 0..b as u64 {
            let (mk, mt) = self.gemm(1, c.embed_dim);
            s.push(g.step_sliced(
                mk,
                &[&self.norm_out, self.w("head.weight"), &self.head_out],
                &[(si * seq as u64 * w as u64, 0), (0, 0), (si * c.embed_dim as u64, 0)],
                &[1, w, c.embed_dim],
                mt,
            ));
        }
        s.push(g.step(V_BIAS_ADD, &[&self.head_out, self.w("head.bias")], &[b, c.embed_dim], b * c.embed_dim));
        // PuLID's `id_cond_vit`: the L2-normalized cls embedding.
        // `l2norm_scale` Params: [n, d, eps(f32 bits)].
        s.push(g.step(
            V_L2NORM,
            &[&self.head_out, &self.l2_ones, &self.cls_l2],
            &[b, c.embed_dim, f(1e-12)],
            b * c.embed_dim,
        ));
        s
    }

    /// Upload preprocessed pixels, `[B, 3, S, S]` NCHW (already mean/std
    /// normalized — `crates/imaging` owns decode/resize/normalize).
    pub fn set_pixels(&self, px: &[f32]) {
        let want = self.b as usize * 3 * (self.cfg.image_size * self.cfg.image_size) as usize;
        assert_eq!(px.len(), want, "pixel count");
        let bits: Vec<u32> = px.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.pixels, &bits);
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    fn n(&self) -> usize {
        (self.b * self.cfg.seq_len()) as usize
    }
    /// Patch tokens in NLC order, `[B*num_patches, W]` — pre-bias, pre-cls.
    pub fn read_patch_nchw(&self) -> Vec<f32> {
        self.gpu
            .read(&self.patch_nchw, self.b as usize * (self.cfg.width * self.cfg.num_patches()) as usize)
    }
    /// `x[0]` = block input (cls ‖ patches + pos); `x[i+1]` = block `i` output.
    pub fn read_x(&self, i: usize) -> Vec<f32> {
        self.gpu.read(&self.x[i], self.n() * self.cfg.width as usize)
    }
    pub fn read_block_tap(&self, l: usize, tap: VisionTap) -> Vec<f32> {
        let bb = &self.blocks[l];
        let (buf, w) = match tap {
            VisionTap::Norm1 => (&bb.norm1, self.cfg.width),
            VisionTap::Qkv => (&bb.qkv, 3 * self.cfg.width),
            VisionTap::Ctx => (&bb.ctx, self.cfg.width),
            VisionTap::InnerLn => (&bb.inner_ln, self.cfg.width),
            VisionTap::AttnProj => (&bb.attn_proj, self.cfg.width),
            VisionTap::Norm2 => (&bb.norm2, self.cfg.width),
            VisionTap::W1 => (&bb.w1, self.cfg.mlp_hidden),
            VisionTap::W2 => (&bb.w2, self.cfg.mlp_hidden),
            VisionTap::FfnLn => (&bb.ffn_ln, self.cfg.mlp_hidden),
            VisionTap::MlpOut => (&bb.mlp_out, self.cfg.width),
        };
        self.gpu.read(buf, self.n() * w as usize)
    }
    pub fn read_norm_out(&self) -> Vec<f32> {
        self.gpu.read(&self.norm_out, self.n() * self.cfg.width as usize)
    }
    pub fn read_head_out(&self) -> Vec<f32> {
        self.gpu.read(&self.head_out, self.b as usize * self.cfg.embed_dim as usize)
    }
    pub fn read_cls_embed_l2norm(&self) -> Vec<f32> {
        self.gpu.read(&self.cls_l2, self.b as usize * self.cfg.embed_dim as usize)
    }
}

/// Per-block taps the parity test replays.
#[derive(Clone, Copy, Debug)]
pub enum VisionTap {
    Norm1,
    /// Fused `[N, 3W]`, q/k in brain's PERMUTED head-channel order — see
    /// `EvaVisionConfig::head_perm`.
    Qkv,
    Ctx,
    InnerLn,
    AttnProj,
    Norm2,
    W1,
    W2,
    FfnLn,
    MlpOut,
}
