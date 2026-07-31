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

/// Text-tower kernels. `layernorm_rows` is registered but never indexed
/// directly: `block::LayerNormIds::resolve_fwd` picks it up BY NAME wherever
/// the device supports workgroup reductions (2.3-9.1x on a P40).
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
    /// Transient score slab (pre-softmax) — the one non-SSA buffer here; the
    /// backward recomputes scores from `qkv` rather than caching two slabs.
    scores: DeviceBuffer,
    hidden: DeviceBuffer,
    pooled: DeviceBuffer,
    text_embeds: Option<DeviceBuffer>,
    steps: Vec<Step>,
}

impl ClipText {
    /// Build on an existing device (tests pass `gpu_core::testgpu::dev`).
    pub fn new_on(
        gpu: Gpu,
        cfg: ClipTextConfig,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> ClipText {
        assert!(t <= cfg.max_positions, "seq len {t} > max_positions {}", cfg.max_positions);
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
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
        };
        m.steps = m.build_steps();
        m
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, T_MATMUL, T_MATMUL_REG2, false)
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let (b, t) = (self.b, self.t);
        let n = b * t;
        let h = c.hidden;
        let inter = c.intermediate;
        let hd = c.head_dim();
        let ln = block::LayerNormIds::resolve_fwd(g, T_LAYERNORM);
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
