// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream text decoder pieces. Built up incrementally; today: the sparse-MoE
//! FFN (GeGLU-shift experts + top-k router), which mirrors `crates/moe`'s dense-
//! over-all-experts FFN but swaps SwiGLU for Moondream's GeGLU-with-+1-shift
//! (`geglu_shift`) and a single fc1 split into its `h`/`g` halves (`w_h`/`w_g`).

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Decoder kernel pipeline (indices used below).
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("matmul", kernels::MATMUL),                     // 0
        ("router_gate", kernels::ROUTER_GATE),           // 1
        ("geglu_shift", kernels::GEGLU_SHIFT),           // 2
        ("scale_add", kernels::SCALE_ADD),               // 3
        ("layernorm", kernels::LAYERNORM),               // 4
        ("gelu", kernels::GELU),                         // 5 (tanh gelu_approx)
        ("bias_add", kernels::BIAS_ADD),                 // 6
        ("add2", kernels::ADD2),                         // 7
        ("rope_partial", kernels::ROPE_PARTIAL),         // 8
        ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR), // 9
        ("attn_prefix_mask", kernels::ATTN_PREFIX_MASK), // 10
        ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR), // 11
        ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR), // 12
        ("embed", kernels::EMBED),                       // 13
        ("splice", kernels::SPLICE),                     // 14
        ("ce_value", kernels::CE_VALUE_MASKED),          // 15
        ("gelu_erf", kernels::GELU_ERF),                 // 16 (tau tok_feat: erf GELU)
        ("tau_scale", kernels::TAU_SCALE),               // 17
        // --- backward ---
        ("matmul_dx", kernels::MATMUL_DX),               // 18
        ("matmul_dw", kernels::MATMUL_DW),               // 19
        ("bias_grad", kernels::BIAS_GRAD),               // 20
        ("gelu_bwd", kernels::GELU_BWD),                 // 21 (tanh gelu bwd)
        ("layernorm_dx", kernels::LAYERNORM_DX),         // 22
        ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA), // 23
        ("layernorm_dbeta", kernels::LAYERNORM_DBETA),   // 24
        ("ln_stats", kernels::LN_STATS),                 // 25
        ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS), // 26
        ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS), // 27
        ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS), // 28
        ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS), // 29
        ("rope_partial_bwd", kernels::ROPE_PARTIAL_BWD), // 30
        ("ce_grad_masked", kernels::CE_GRAD_MASKED),     // 31
        ("emb_bwd", kernels::EMB_BWD),                   // 32
        ("splice_bwd", kernels::SPLICE_BWD),             // 33
        ("gelu_erf_bwd", kernels::GELU_ERF_BWD),         // 34 (tau tok_feat bwd)
        ("tau_scale_ds", kernels::TAU_SCALE_DS),         // 35 (tau scale grad)
        ("geglu_shift_da", kernels::GEGLU_SHIFT_DA),     // 36 (MoE expert dh)
        ("geglu_shift_db", kernels::GEGLU_SHIFT_DB),     // 37 (MoE expert dg)
        ("scale_add_dexp", kernels::SCALE_ADD_DEXP),     // 38 (MoE combine → d_expert)
        ("scale_add_dgate", kernels::SCALE_ADD_DGATE),   // 39 (MoE combine → d_gate)
        ("router_bwd", kernels::ROUTER_BWD),             // 40 (top-k softmax gate bwd)
    ]
}

// Backward kernel pipeline indices.
const K_MATMUL_DX: usize = 18;
const K_MATMUL_DW: usize = 19;
const K_BIAS_GRAD: usize = 20;
const K_GELU_BWD: usize = 21;
const K_LN_DX: usize = 22;
const K_LN_DGAMMA: usize = 23;
const K_LN_DBETA: usize = 24;
const K_LN_STATS: usize = 25;
const K_ATTN_DSCORES: usize = 26;
const K_ATTN_DV: usize = 27;
const K_ATTN_DQ: usize = 28;
const K_ATTN_DK: usize = 29;
const K_ROPE_PARTIAL_BWD: usize = 30;
const K_CE_GRAD: usize = 31;
const K_EMB_BWD: usize = 32;
const K_SPLICE_BWD: usize = 33;
const K_GELU_ERF_BWD: usize = 34;
const K_TAU_SCALE_DS: usize = 35;
const K_GEGLU_DA: usize = 36;
const K_GEGLU_DB: usize = 37;
const K_SCALE_ADD_DEXP: usize = 38;
const K_SCALE_ADD_DGATE: usize = 39;
const K_ROUTER_BWD: usize = 40;

/// Masked cross-entropy ignore index (matches the loaders' `-1 i32` as `u32`).
pub const IGNORE: u32 = 0xFFFF_FFFF;

/// Logistic sigmoid (host, for the tau position term).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The Moondream text decoder: token embedding → splice the projected image tokens
/// into the prefix rows → a stack of [`MoondreamBlock`]s (dense 0..3, MoE 4..23) →
/// post-LN → lm_head → masked cross-entropy. Image tokens occupy rows
/// `[1, 1+n_img)` (after the bos), a positional prefix (no placeholder token).
pub struct MoondreamDecoder<'g> {
    gpu: &'g Gpu,
    blocks: Vec<MoondreamBlock<'g>>,
    w: HashMap<String, DeviceBuffer>,
    d: u32,
    vocab: u32,
    t: u32,
    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    res: DeviceBuffer,
    normed: DeviceBuffer,
    logits: DeviceBuffer,
    ce: DeviceBuffer,
    n_img: u32,
}

impl<'g> MoondreamDecoder<'g> {
    /// Build from per-layer prefixed weights (`blocks.{l}.…`) plus `tok.weight`
    /// `[vocab,d]`, `post_ln.weight`/`.bias`, `lm_head.weight` `[vocab,d]`/`.bias`.
    /// `moe_layers` marks which layers use the MoE FFN (their MoE weights are under
    /// `blocks.{l}.moe.…`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, blocks: Vec<MoondreamBlock<'g>>, t: u32, d: u32, vocab: u32, n_img: u32) -> MoondreamDecoder<'g> {
        let w = weights
            .iter()
            .filter(|(k, _)| !k.starts_with("blocks."))
            .map(|(k, v)| (k.clone(), gpu.storage_init(k, v)))
            .collect();
        MoondreamDecoder {
            gpu,
            blocks,
            w,
            d,
            vocab,
            t,
            tokens: gpu.storage(t as u64),
            targets: gpu.storage(t as u64),
            res: gpu.storage((t * d) as u64),
            normed: gpu.storage((t * d) as u64),
            logits: gpu.storage((t * vocab) as u64),
            ce: gpu.storage(t as u64),
            n_img,
        }
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("decoder weight missing: {n}"))
    }
    /// Forward → mean masked cross-entropy. `tokens`/`targets` length `t`
    /// (targets IGNORE at the image + non-supervised rows); `image_embeds` is the
    /// `[n_img, d]` connector output spliced at rows `[1, 1+n_img)`.
    pub fn forward(&self, tokens: &[u32], targets: &[u32], image_embeds: &[f32]) -> f32 {
        let g = self.gpu;
        let (t, d, v) = (self.t, self.d, self.vocab);
        g.write(&self.tokens, tokens);
        g.write(&self.targets, targets);
        let img = g.storage_init("md.img", image_embeds);
        // embed → res, then splice image tokens at rows [1, 1+n_img) (base = 1·d).
        g.submit(
            &[],
            &[
                g.step(13, &[&self.tokens, self.wb("tok.weight"), &self.res], &[d, t], t * d),
                g.step(14, &[&img, &self.res], &[self.n_img * d, d], self.n_img * d),
            ],
        );
        // Block stack (each returns its own output buffer).
        let mut cur: &DeviceBuffer = &self.res;
        for b in &self.blocks {
            cur = b.forward(cur);
        }
        // post-LN → lm_head → CE.
        g.submit(
            &[],
            &[
                g.step(4, &[cur, self.wb("post_ln.weight"), self.wb("post_ln.bias"), &self.normed], &[d, t, f(LN_EPS)], t),
                g.step(0, &[&self.normed, self.wb("lm_head.weight"), &self.logits], &[t, d, v], t * v),
                g.step(6, &[&self.logits, self.wb("lm_head.bias")], &[t, v], t * v),
                g.step(15, &[&self.logits, &self.targets, &self.ce], &[t, v, IGNORE], t),
            ],
        );
        let ce = g.read(&self.ce, t as usize);
        let count = targets.iter().filter(|&&x| x != IGNORE).count().max(1) as f32;
        ce.iter().sum::<f32>() / count
    }

    /// Decoder backward (dense blocks): from the cached forward (call `forward`
    /// first), fill every grad in `gr`. Chain: CE → lm_head → post-LN → blocks in
    /// reverse (each `MoondreamBlock::backward`, threading the residual-stream grad)
    /// → splice (image rows → `d_image_embeds`, zeroed in the residual grad) →
    /// embedding (text rows → `tok.weight`). Requires all blocks dense/no-tau.
    pub fn backward(&self, targets: &[u32], gr: &MoondreamDecoderGrads) {
        let g = self.gpu;
        let (t, d, v) = (self.t, self.d, self.vocab);
        let count = targets.iter().filter(|&&x| x != IGNORE).count().max(1) as f32;
        let d_logits = g.storage((t * v) as u64);
        let d_normed = g.storage((t * d) as u64);
        let d_last = g.storage((t * d) as u64);
        let mean = g.storage(t as u64);
        let inv = g.storage(t as u64);
        let last_out = self.blocks.last().map(|b| b.output()).unwrap_or(&self.res);

        // CE → d_logits; lm_head (bias/weight/input); post-LN → d_last.
        g.submit(
            &[],
            &[
                g.step(K_CE_GRAD, &[&self.logits, &self.targets, &d_logits], &[t, v, IGNORE, f(count)], t * v),
                g.step(K_BIAS_GRAD, &[&d_logits, &gr.lm_head_b], &[t, v], v),
                g.step(K_MATMUL_DW, &[&d_logits, &self.normed, &gr.lm_head_w], &[t, d, v], v * d),
                g.step(K_MATMUL_DX, &[&d_logits, self.wb("lm_head.weight"), &d_normed], &[t, d, v, 0], t * d),
                g.step(K_LN_STATS, &[last_out, &mean, &inv], &[d, t, f(LN_EPS)], t),
                g.step(K_LN_DGAMMA, &[&d_normed, last_out, &mean, &inv, &gr.post_ln_w], &[d, t], d),
                g.step(K_LN_DBETA, &[&d_normed, &gr.post_ln_b], &[d, t], d),
                g.step(K_LN_DX, &[last_out, self.wb("post_ln.weight"), &d_normed, &d_last], &[d, t, f(LN_EPS)], t),
            ],
        );

        // Blocks in reverse, threading the residual-stream grad (each submits itself).
        let n = self.blocks.len();
        let mut d_cur = d_last;
        for i in (0..n).rev() {
            let x_in = if i == 0 { &self.res } else { self.blocks[i - 1].output() };
            let d_in = g.storage((t * d) as u64);
            self.blocks[i].backward(x_in, &d_cur, &gr.blocks[i], &d_in);
            d_cur = d_in;
        }
        // d_cur is now the grad of `res`. Route image rows → d_image_embeds (and zero
        // them in d_cur), then scatter the text rows into tok.weight.
        g.submit(
            &[],
            &[
                g.step(K_SPLICE_BWD, &[&d_cur, &gr.d_image_embeds], &[self.n_img * d, d], self.n_img * d),
                g.step(K_EMB_BWD, &[&self.tokens, &d_cur, &gr.tok_w], &[t, d, v], v * d),
            ],
        );
    }
}

/// All gradient buffers for a dense [`MoondreamDecoder`] (per-block grads + the
/// decoder-level embedding/head grads + the spliced image-embedding grad).
pub struct MoondreamDecoderGrads {
    pub blocks: Vec<MoondreamBlockGrads>,
    pub tok_w: DeviceBuffer,
    pub post_ln_w: DeviceBuffer,
    pub post_ln_b: DeviceBuffer,
    pub lm_head_w: DeviceBuffer,
    pub lm_head_b: DeviceBuffer,
    /// Grad w.r.t. the spliced image embeddings `[n_img, d]` (the connector output).
    pub d_image_embeds: DeviceBuffer,
}

impl MoondreamDecoderGrads {
    /// Allocate zeroed grads for a dense decoder of the given shape.
    pub fn new(g: &Gpu, n_layers: u32, d: u32, ff: u32, vocab: u32, n_img: u32) -> MoondreamDecoderGrads {
        let z = |n: u32| g.storage_init("md.dg", &vec![0.0f32; n as usize]);
        MoondreamDecoderGrads {
            blocks: (0..n_layers).map(|_| MoondreamBlockGrads::new(g, d, ff)).collect(),
            tok_w: z(vocab * d),
            post_ln_w: z(d),
            post_ln_b: z(d),
            lm_head_w: z(vocab * d),
            lm_head_b: z(vocab),
            d_image_embeds: z(n_img * d),
        }
    }
}

const LN_EPS: f32 = 1e-5;

/// One Moondream decoder block — the PARALLEL attn+MLP form: a single shared
/// LayerNorm feeds BOTH the attention and the FFN, and `x = x + l_attn + l_mlp`
/// (a 3-way residual). The attention is full MHA with **partial RoPE** and the
/// **prefix-LM mask** (image prefix bidirectional, else causal); the FFN here is
/// the dense variant (layers 0..3). (The tau temperature and the MoE FFN variant
/// for layers 4..23 are added on top; backward + gradcheck are a follow-up.)
/// Weight keys: `ln.weight`/`ln.bias`, `attn.qkv.weight` `[3d, d]`,
/// `attn.proj.weight` `[d,d]`/`attn.proj.bias`, `mlp.fc1.weight` `[ff,d]`/`.bias`,
/// `mlp.fc2.weight` `[d,ff]`/`.bias`.
pub struct MoondreamBlock<'g> {
    gpu: &'g Gpu,
    w: HashMap<String, DeviceBuffer>,
    d: u32,
    n_heads: u32,
    head_dim: u32,
    ff: u32,
    t: u32,
    prefix: u32,
    rot_dim: u32,
    theta: f32,
    /// True when `attn.tau.*` weights are present (per-head attention temperature).
    tau: bool,
    // scratch
    l_in: DeviceBuffer,
    qkv: DeviceBuffer,
    qkv2: DeviceBuffer,   // tau-scaled qkv (q,v scaled; k passthrough)
    tok_feat: DeviceBuffer, // gelu_erf(qkv) [t, 3d]
    tqr: DeviceBuffer,    // tok_feat·wqᵀ [t, nh]
    tvr: DeviceBuffer,    // tok_feat·wvᵀ [t, nh]
    s3: DeviceBuffer,     // [3·nh, t] per-(head,token) scale (q | k=1 | v)
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    l_attn: DeviceBuffer,
    h: DeviceBuffer,
    h2: DeviceBuffer,
    l_mlp: DeviceBuffer,
    mid: DeviceBuffer,
    out: DeviceBuffer,
    /// `Some` for the MoE layers (4..23); `None` uses the dense FFN.
    moe: Option<MoeFfn<'g>>,
}

impl<'g> MoondreamBlock<'g> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, n_heads: u32, head_dim: u32, ff: u32, prefix: u32, rot_dim: u32, theta: f32) -> MoondreamBlock<'g> {
        let w: HashMap<String, DeviceBuffer> = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        let slab = (n_heads * t * t) as u64;
        let tau = w.contains_key("attn.tau.wq");
        MoondreamBlock {
            gpu,
            w,
            d,
            n_heads,
            head_dim,
            ff,
            t,
            prefix,
            rot_dim,
            theta,
            tau,
            l_in: gpu.storage((t * d) as u64),
            qkv: gpu.storage((t * 3 * d) as u64),
            qkv2: gpu.storage((t * 3 * d) as u64),
            tok_feat: gpu.storage((t * 3 * d) as u64),
            tqr: gpu.storage((t * n_heads) as u64),
            tvr: gpu.storage((t * n_heads) as u64),
            s3: gpu.storage((3 * n_heads * t) as u64),
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            ctx: gpu.storage((t * d) as u64),
            l_attn: gpu.storage((t * d) as u64),
            h: gpu.storage((t * ff) as u64),
            h2: gpu.storage((t * ff) as u64),
            l_mlp: gpu.storage((t * d) as u64),
            mid: gpu.storage((t * d) as u64),
            out: gpu.storage((t * d) as u64),
            moe: None,
        }
    }
    /// Attach an MoE FFN (replaces the dense FFN branch) for a deep layer.
    pub fn with_moe(mut self, moe: MoeFfn<'g>) -> Self {
        self.moe = Some(moe);
        self
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("block weight missing: {n}"))
    }
    pub fn forward(&self, x: &DeviceBuffer) -> &DeviceBuffer {
        let g = self.gpu;
        let (t, d, nh, hd, ff) = (self.t, self.d, self.n_heads, self.head_dim, self.ff);
        let stride3 = 3 * d;
        let mut s: Vec<Step> = Vec::new();

        // Shared LayerNorm (with bias).
        s.push(g.step(4, &[x, self.wb("ln.weight"), self.wb("ln.bias"), &self.l_in], &[d, t, f(LN_EPS)], t));
        // --- attention branch ---
        // fused qkv = l_in · wqkv^T  ([t, 3d])
        s.push(g.step(0, &[&self.l_in, self.wb("attn.qkv.weight"), &self.qkv], &[t, d, stride3], t * stride3));
        // Per-head attention temperature (tau): scale q and v (NOT k) by a per-
        // (head,token) scalar computed from the raw qkv, BEFORE RoPE. tok_feat is
        // erf-GELU over the full 3d projection; tok_q/tok_v = tanh(tok_feat·w{q,v}ᵀ);
        // tau_pos = 0.5+sigmoid(alpha·ln(pos+1)) folds on host (positions = row).
        // Scalar scaling commutes with the RoPE rotation, so applying it here (pre-
        // RoPE, matching the reference) is equivalent up to that rotation. The tiny
        // tanh+tau_pos assembly into the [3·nh, t] scale (q | k=1 | v) folds on host.
        let qkv = if self.tau {
            s.push(g.step(16, &[&self.qkv, &self.tok_feat], &[t * stride3], t * stride3));
            s.push(g.step(0, &[&self.tok_feat, self.wb("attn.tau.wq"), &self.tqr], &[t, stride3, nh], t * nh));
            s.push(g.step(0, &[&self.tok_feat, self.wb("attn.tau.wv"), &self.tvr], &[t, stride3, nh], t * nh));
            g.submit(&[], &s);
            s = Vec::new();
            let tqr = g.read(&self.tqr, (t * nh) as usize);
            let tvr = g.read(&self.tvr, (t * nh) as usize);
            let alpha = g.read(self.wb("attn.tau.alpha"), nh as usize);
            let mut s3 = vec![1.0f32; (3 * nh * t) as usize];
            for h in 0..nh as usize {
                for row in 0..t as usize {
                    let tau_pos = 0.5 + sigmoid(alpha[h] * ((row + 1) as f32).ln());
                    s3[h * t as usize + row] = tqr[row * nh as usize + h].tanh() + tau_pos;
                    s3[(2 * nh as usize + h) * t as usize + row] = tvr[row * nh as usize + h].tanh() + tau_pos;
                }
            }
            let packed: Vec<u32> = s3.iter().map(|&x| f(x)).collect();
            g.write(&self.s3, &packed);
            // Treat qkv as [t, 3·nh, hd]: scale q-heads by s_q, k-heads by 1, v by s_v.
            s.push(g.step(17, &[&self.qkv, &self.s3, &self.qkv2], &[t, 3 * nh, hd], t * stride3));
            &self.qkv2
        } else {
            &self.qkv
        };
        // partial RoPE on q (off 0) and k (off d)
        let half = self.rot_dim / 2;
        s.push(g.step(8, &[qkv], &[t, nh, hd, stride3, 0, t, f(self.theta), self.rot_dim], t * nh * half));
        s.push(g.step(8, &[qkv], &[t, nh, hd, stride3, d, t, f(self.theta), self.rot_dim], t * nh * half));
        // bidir scores → prefix-LM mask → bidir softmax → bidir apply
        s.push(g.step(9, &[qkv, &self.scores], &[1, nh, t, hd, stride3, 0, d], nh * t * t));
        s.push(g.step(10, &[&self.scores], &[1, nh, t, self.prefix], nh * t * t));
        s.push(g.step(11, &[&self.scores, &self.probs], &[1, nh, t], nh * t));
        s.push(g.step(12, &[&self.probs, qkv, &self.ctx], &[1, nh, t, hd, stride3, 2 * d, d], nh * t * hd));
        // proj + bias → l_attn. Submit phase 1 (LN + attention) so l_in/l_attn are
        // ready before the FFN (MoE submits internally).
        s.push(g.step(0, &[&self.ctx, self.wb("attn.proj.weight"), &self.l_attn], &[t, d, d], t * d));
        s.push(g.step(6, &[&self.l_attn, self.wb("attn.proj.bias")], &[t, d], t * d));
        g.submit(&[], &s);

        // --- FFN branch on the SAME l_in: MoE (layers 4..23) or dense. ---
        let l_mlp: &DeviceBuffer = if let Some(moe) = &self.moe {
            moe.forward(&self.l_in)
        } else {
            g.submit(
                &[],
                &[
                    g.step(0, &[&self.l_in, self.wb("mlp.fc1.weight"), &self.h], &[t, d, ff], t * ff),
                    g.step(6, &[&self.h, self.wb("mlp.fc1.bias")], &[t, ff], t * ff),
                    g.step(5, &[&self.h, &self.h2], &[t * ff], t * ff), // tanh GELU
                    g.step(0, &[&self.h2, self.wb("mlp.fc2.weight"), &self.l_mlp], &[t, ff, d], t * d),
                    g.step(6, &[&self.l_mlp, self.wb("mlp.fc2.bias")], &[t, d], t * d),
                ],
            );
            &self.l_mlp
        };
        // --- 3-way residual: out = x + l_attn + l_mlp ---
        g.submit(
            &[],
            &[
                g.step(7, &[x, &self.l_attn, &self.mid], &[t * d], t * d),
                g.step(7, &[&self.mid, l_mlp, &self.out], &[t * d], t * d),
            ],
        );
        &self.out
    }
    pub fn numel(&self) -> usize {
        (self.t * self.d) as usize
    }

    /// The block's output buffer (the residual-stream slice it produced), valid
    /// after `forward`. Used as the next block's cached input during backward.
    pub fn output(&self) -> &DeviceBuffer {
        &self.out
    }

    /// Dense-block backward (no tau, no MoE): from the output grad `d_out`, fill the
    /// weight grads `gr` and the block-input grad `d_x_in`. Reuses the forward
    /// scratch as the SSA cache (valid immediately after `forward`). The two branches
    /// feed the SAME shared LayerNorm, so their input-grads accumulate into `d_ln`
    /// (the MLP `matmul_dx` writes it, the attention `matmul_dx` adds) before one
    /// `layernorm_dx` — the shared-activation pattern. The masked-bidir attention
    /// backward reuses the ViT `_cross` kernels: the cached `probs` already carry the
    /// prefix mask (masked positions have prob≈0 → contribute 0). d_x_in = d_out
    /// (the 3-way residual's identity path) + the LayerNorm input grad.
    pub fn backward(&self, x: &DeviceBuffer, d_out: &DeviceBuffer, gr: &MoondreamBlockGrads, d_x_in: &DeviceBuffer) {
        let g = self.gpu;
        let (t, d, nh, hd, ff) = (self.t, self.d, self.n_heads, self.head_dim, self.ff);
        let stride3 = 3 * d;
        let half = self.rot_dim / 2;
        // The attention ran on qkv2 (post-tau) when tau is on, else the raw qkv.
        let attn_qkv = if self.tau { &self.qkv2 } else { &self.qkv };
        let d_ln = g.storage((t * d) as u64);
        let d_h = g.storage((t * ff) as u64);
        let d_h2 = g.storage((t * ff) as u64);
        let d_ctx = g.storage((t * d) as u64);
        let d_qkv = g.storage((t * 3 * d) as u64);
        let dscores = g.storage((nh * t * t) as u64);
        let mean = g.storage(t as u64);
        let inv = g.storage(t as u64);
        let d_xln = g.storage((t * d) as u64);
        let mut s: Vec<Step> = Vec::new();

        // --- FFN branch → d_ln (overwrite): dense MLP or MoE experts. ---
        if let Some(moe) = &self.moe {
            moe.backward(&self.l_in, d_out, gr.moe.as_ref().expect("moe grads required for a MoE block"), &d_ln);
        } else {
            s.push(g.step(K_MATMUL_DX, &[d_out, self.wb("mlp.fc2.weight"), &d_h2], &[t, ff, d, 0], t * ff));
            s.push(g.step(K_MATMUL_DW, &[d_out, &self.h2, &gr.fc2_w], &[t, ff, d], d * ff));
            s.push(g.step(K_BIAS_GRAD, &[d_out, &gr.fc2_b], &[t, d], d));
            s.push(g.step(K_GELU_BWD, &[&self.h, &d_h2, &d_h], &[t * ff], t * ff)); // tanh gelu
            s.push(g.step(K_MATMUL_DX, &[&d_h, self.wb("mlp.fc1.weight"), &d_ln], &[t, d, ff, 0], t * d));
            s.push(g.step(K_MATMUL_DW, &[&d_h, &self.l_in, &gr.fc1_w], &[t, d, ff], ff * d));
            s.push(g.step(K_BIAS_GRAD, &[&d_h, &gr.fc1_b], &[t, ff], ff));
        }

        // --- attention branch → d_qkv (grad of attn_qkv, pre-RoPE) ---
        s.push(g.step(K_MATMUL_DX, &[d_out, self.wb("attn.proj.weight"), &d_ctx], &[t, d, d, 0], t * d));
        s.push(g.step(K_MATMUL_DW, &[d_out, &self.ctx, &gr.proj_w], &[t, d, d], d * d));
        s.push(g.step(K_BIAS_GRAD, &[d_out, &gr.proj_b], &[t, d], d));
        s.push(g.step(K_ATTN_DSCORES, &[&d_ctx, attn_qkv, &self.probs, &dscores], &[1, nh, t, t, hd, stride3, 2 * d, d], nh * t * t));
        s.push(g.step(K_ATTN_DV, &[&self.probs, &d_ctx, &d_qkv], &[1, nh, t, t, hd, stride3, 2 * d, d], nh * t * hd));
        s.push(g.step(K_ATTN_DQ, &[&dscores, attn_qkv, &d_qkv], &[1, nh, t, t, hd, stride3, stride3, 0, d], nh * t * hd));
        s.push(g.step(K_ATTN_DK, &[&dscores, attn_qkv, &d_qkv], &[1, nh, t, t, hd, stride3, stride3, 0, d], nh * t * hd));
        // Rotate d_q (off 0) and d_k (off d) back through the partial RoPE (−angle).
        s.push(g.step(K_ROPE_PARTIAL_BWD, &[&d_qkv], &[t, nh, hd, stride3, 0, t, f(self.theta), self.rot_dim], t * nh * half));
        s.push(g.step(K_ROPE_PARTIAL_BWD, &[&d_qkv], &[t, nh, hd, stride3, d, t, f(self.theta), self.rot_dim], t * nh * half));

        // Route d_qkv into the shared-LN grad. Dense: qkv matmul bwd directly. Tau:
        // d_qkv is the grad of qkv2 = tau_scale(qkv_raw, s3), so it flows back through
        // tau_scale (in-grad + the tok_feat/wq/wv/alpha chain) to the raw qkv first.
        let d_qkv_raw = if self.tau {
            let tg = gr.tau.as_ref().expect("tau grads required for a tau block");
            let d_s3 = g.storage((3 * nh * t) as u64);
            // ds[h,row] = Σ_d d_qkv[row,h,d]·qkv_raw[row,h,d] over the 3·nh heads.
            s.push(g.step(K_TAU_SCALE_DS, &[&d_qkv, &self.qkv, &d_s3], &[t, 3 * nh, hd], 3 * nh * t));
            g.submit(&[], &s);
            s = Vec::new();
            // Host: tanh′ and the alpha (tau_pos) grad; positions = row index.
            let ds = g.read(&d_s3, (3 * nh * t) as usize);
            let tqr = g.read(&self.tqr, (t * nh) as usize);
            let tvr = g.read(&self.tvr, (t * nh) as usize);
            let alpha = g.read(self.wb("attn.tau.alpha"), nh as usize);
            let (mut d_tqr, mut d_tvr) = (vec![0.0f32; (t * nh) as usize], vec![0.0f32; (t * nh) as usize]);
            let mut d_alpha = vec![0.0f32; nh as usize];
            for h in 0..nh as usize {
                for row in 0..t as usize {
                    let lp = ((row + 1) as f32).ln();
                    let sg = sigmoid(alpha[h] * lp);
                    let (ds_q, ds_v) = (ds[h * t as usize + row], ds[(2 * nh as usize + h) * t as usize + row]);
                    let (thq, thv) = (tqr[row * nh as usize + h].tanh(), tvr[row * nh as usize + h].tanh());
                    d_tqr[row * nh as usize + h] = ds_q * (1.0 - thq * thq);
                    d_tvr[row * nh as usize + h] = ds_v * (1.0 - thv * thv);
                    d_alpha[h] += (ds_q + ds_v) * sg * (1.0 - sg) * lp; // dtau_pos/dα
                }
            }
            let dtqr_b = g.storage_init("md.dtqr", &d_tqr);
            let dtvr_b = g.storage_init("md.dtvr", &d_tvr);
            g.write(&tg.alpha, &d_alpha.iter().map(|&v| f(v)).collect::<Vec<u32>>());
            let d_qkv_in = g.storage((t * stride3) as u64); // in-grad through tau_scale
            let d_tok_feat = g.storage((t * stride3) as u64);
            let d_qraw2 = g.storage((t * stride3) as u64);
            let d_qkv_raw = g.storage((t * stride3) as u64);
            // in-grad: d_qkv_in = d_qkv · s3 (same tau_scale kernel).
            s.push(g.step(17, &[&d_qkv, &self.s3, &d_qkv_in], &[t, 3 * nh, hd], t * stride3));
            // wq/wv weight grads + d_tok_feat = d_tqr·wq + d_tvr·wv.
            s.push(g.step(K_MATMUL_DW, &[&dtqr_b, &self.tok_feat, &tg.wq], &[t, stride3, nh], nh * stride3));
            s.push(g.step(K_MATMUL_DX, &[&dtqr_b, self.wb("attn.tau.wq"), &d_tok_feat], &[t, stride3, nh, 0], t * stride3));
            s.push(g.step(K_MATMUL_DW, &[&dtvr_b, &self.tok_feat, &tg.wv], &[t, stride3, nh], nh * stride3));
            s.push(g.step(K_MATMUL_DX, &[&dtvr_b, self.wb("attn.tau.wv"), &d_tok_feat], &[t, stride3, nh, 1], t * stride3)); // accumulate
            // tok_feat = gelu_erf(qkv_raw): d_qraw2 = d_tok_feat · gelu_erf′(qkv_raw).
            s.push(g.step(K_GELU_ERF_BWD, &[&self.qkv, &d_tok_feat, &d_qraw2], &[t * stride3], t * stride3));
            // Total raw-qkv grad = tau_scale in-grad + tok_feat path.
            s.push(g.step(7, &[&d_qkv_in, &d_qraw2, &d_qkv_raw], &[t * stride3], t * stride3));
            d_qkv_raw
        } else {
            d_qkv
        };
        s.push(g.step(K_MATMUL_DX, &[&d_qkv_raw, self.wb("attn.qkv.weight"), &d_ln], &[t, d, stride3, 1], t * d)); // accumulate
        s.push(g.step(K_MATMUL_DW, &[&d_qkv_raw, &self.l_in, &gr.qkv_w], &[t, d, stride3], stride3 * d));

        // --- shared LayerNorm backward: d_ln → ln grads + d_xln ---
        s.push(g.step(K_LN_STATS, &[x, &mean, &inv], &[d, t, f(LN_EPS)], t));
        s.push(g.step(K_LN_DGAMMA, &[&d_ln, x, &mean, &inv, &gr.ln_w], &[d, t], d));
        s.push(g.step(K_LN_DBETA, &[&d_ln, &gr.ln_b], &[d, t], d));
        s.push(g.step(K_LN_DX, &[x, self.wb("ln.weight"), &d_ln, &d_xln], &[d, t, f(LN_EPS)], t));
        // d_x_in = d_out (residual identity) + LayerNorm input grad.
        s.push(g.step(7, &[d_out, &d_xln, d_x_in], &[t * d], t * d)); // add2
        g.submit(&[], &s);
    }
}

/// Per-weight gradient buffers for a dense [`MoondreamBlock`] (zeroed on build; the
/// accumulating bwd kernels add into them).
pub struct MoondreamBlockGrads {
    pub ln_w: DeviceBuffer,
    pub ln_b: DeviceBuffer,
    pub qkv_w: DeviceBuffer,
    pub proj_w: DeviceBuffer,
    pub proj_b: DeviceBuffer,
    pub fc1_w: DeviceBuffer,
    pub fc1_b: DeviceBuffer,
    pub fc2_w: DeviceBuffer,
    pub fc2_b: DeviceBuffer,
    /// Present for tau blocks: grads of the attention-temperature head.
    pub tau: Option<TauGrads>,
    /// Present for MoE blocks (replaces the dense fc1/fc2 grads).
    pub moe: Option<MoeGrads>,
}

/// Gradient buffers for the per-head attention-temperature head (`attn.tau.*`).
pub struct TauGrads {
    pub wq: DeviceBuffer,
    pub wv: DeviceBuffer,
    pub alpha: DeviceBuffer,
}

impl MoondreamBlockGrads {
    /// Allocate zeroed grad buffers matching a dense block of the given shape.
    pub fn new(g: &Gpu, d: u32, ff: u32) -> MoondreamBlockGrads {
        let z = |n: u32| g.storage_init("md.g", &vec![0.0f32; n as usize]);
        MoondreamBlockGrads {
            ln_w: z(d),
            ln_b: z(d),
            qkv_w: z(3 * d * d),
            proj_w: z(d * d),
            proj_b: z(d),
            fc1_w: z(ff * d),
            fc1_b: z(ff),
            fc2_w: z(d * ff),
            fc2_b: z(d),
            tau: None,
            moe: None,
        }
    }

    /// Add zeroed tau grads (`wq`/`wv` `[nh, 3·d]`, `alpha` `[nh]`) for a tau block.
    pub fn with_tau(mut self, g: &Gpu, nh: u32, d: u32) -> MoondreamBlockGrads {
        let z = |n: u32| g.storage_init("md.tg", &vec![0.0f32; n as usize]);
        self.tau = Some(TauGrads { wq: z(nh * 3 * d), wv: z(nh * 3 * d), alpha: z(nh) });
        self
    }

    /// Add zeroed MoE grads (router + `e` experts) for a MoE block.
    pub fn with_moe(mut self, g: &Gpu, d: u32, inner: u32, e: u32) -> MoondreamBlockGrads {
        self.moe = Some(MoeGrads::new(g, d, inner, e));
        self
    }
}

/// Sparse-MoE FFN: `router → for each expert (w_h, w_g → geglu_shift → w_down) →
/// gate-weighted accumulate`. Returns the mixed output `[t, d]` (no residual — the
/// parallel block owns the 3-way residual). Weight keys: `router.weight` `[e, d]`,
/// and per expert `experts.{e}.{w_h,w_g}.weight` `[inner, d]`, `w_down.weight`
/// `[d, inner]`.
pub struct MoeFfn<'g> {
    gpu: &'g Gpu,
    w: HashMap<String, DeviceBuffer>,
    e: u32,
    top_k: u32,
    d: u32,
    inner: u32,
    // scratch
    logits: DeviceBuffer,
    gate: DeviceBuffer,
    h: DeviceBuffer,
    g: DeviceBuffer,
    act: DeviceBuffer,
    eout: DeviceBuffer,
    acc: DeviceBuffer,
    t: u32,
}

impl<'g> MoeFfn<'g> {
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, inner: u32, e: u32, top_k: u32) -> MoeFfn<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        MoeFfn {
            gpu,
            w,
            e,
            top_k,
            d,
            inner,
            logits: gpu.storage((t * e) as u64),
            gate: gpu.storage((t * e) as u64),
            h: gpu.storage((t * inner) as u64),
            g: gpu.storage((t * inner) as u64),
            act: gpu.storage((t * inner) as u64),
            eout: gpu.storage((t * d) as u64),
            acc: gpu.storage((t * d) as u64),
            t,
        }
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("moe weight missing: {n}"))
    }
    pub fn forward(&self, xn: &DeviceBuffer) -> &DeviceBuffer {
        let (t, d, inner, e) = (self.t, self.d, self.inner, self.e);
        let mut s: Vec<Step> = Vec::new();
        // Router: logits = xn·router.weight^T, then top-k softmax gate.
        s.push(self.gpu.step(0, &[xn, self.wb("router.weight"), &self.logits], &[t, d, e], t * e));
        s.push(self.gpu.step(1, &[&self.logits, &self.gate], &[t, e, self.top_k], t));
        for ei in 0..e {
            let ep = |leaf: &str| self.wb(&format!("experts.{ei}.{leaf}"));
            s.push(self.gpu.step(0, &[xn, ep("w_h.weight"), &self.h], &[t, d, inner], t * inner));
            s.push(self.gpu.step(0, &[xn, ep("w_g.weight"), &self.g], &[t, d, inner], t * inner));
            s.push(self.gpu.step(2, &[&self.h, &self.g, &self.act], &[t * inner], t * inner)); // gelu(h)·(g+1)
            s.push(self.gpu.step(0, &[&self.act, ep("w_down.weight"), &self.eout], &[t, inner, d], t * d));
            let acc = if ei == 0 { 0u32 } else { 1u32 };
            s.push(self.gpu.step(3, &[&self.gate, &self.eout, &self.acc], &[t, d, e, ei, acc], t * d));
        }
        self.gpu.submit(&[], &s);
        &self.acc
    }
    /// Number of output elements (`t·d`).
    pub fn numel(&self) -> usize {
        (self.t * self.d) as usize
    }

    /// MoE backward: from the mixed-output grad `d_out` (= the block's `d_out`, since
    /// the MoE output is a residual branch), fill `gr` and write the input grad into
    /// `d_xn` (the shared-LN branch grad — its first write overwrites, so `d_xn` need
    /// not be pre-zeroed). Per expert the forward is recomputed (the scratch isn't
    /// cached per-expert), then: combine bwd (`scale_add_dexp/dgate`) → `w_down` →
    /// `geglu_shift_da/db` → `w_h`/`w_g`. Finally the router (`router_bwd`, no aux/z
    /// loss) → `router.weight`. Assumes `forward` ran (for `logits`/`gate`).
    pub fn backward(&self, xn: &DeviceBuffer, d_out: &DeviceBuffer, gr: &MoeGrads, d_xn: &DeviceBuffer) {
        let (t, d, inner, e) = (self.t, self.d, self.inner, self.e);
        let g = self.gpu;
        let d_gate = g.storage((t * e) as u64);
        for ei in 0..e {
            let ep = |leaf: &str| self.wb(&format!("experts.{ei}.{leaf}"));
            let eg = &gr.experts[ei as usize];
            let d_eout = g.storage((t * d) as u64);
            let d_act = g.storage((t * inner) as u64);
            let d_h = g.storage((t * inner) as u64);
            let d_g = g.storage((t * inner) as u64);
            let acc_first = if ei == 0 { 0 } else { 1 }; // first expert's w_h dx overwrites d_xn
            g.submit(
                &[],
                &[
                    // Recompute this expert's forward (h, g, act, eout).
                    g.step(0, &[xn, ep("w_h.weight"), &self.h], &[t, d, inner], t * inner),
                    g.step(0, &[xn, ep("w_g.weight"), &self.g], &[t, d, inner], t * inner),
                    g.step(2, &[&self.h, &self.g, &self.act], &[t * inner], t * inner),
                    g.step(0, &[&self.act, ep("w_down.weight"), &self.eout], &[t, inner, d], t * d),
                    // Combine bwd: d_eout = gate[:,ei]·d_out; d_gate[:,ei] = Σ_c eout·d_out.
                    g.step(K_SCALE_ADD_DEXP, &[&self.gate, d_out, &d_eout], &[t, d, e, ei], t * d),
                    g.step(K_SCALE_ADD_DGATE, &[&self.eout, d_out, &d_gate], &[t, d, e, ei], t),
                    // w_down bwd.
                    g.step(K_MATMUL_DX, &[&d_eout, ep("w_down.weight"), &d_act], &[t, inner, d, 0], t * inner),
                    g.step(K_MATMUL_DW, &[&d_eout, &self.act, &eg.w_down], &[t, inner, d], d * inner),
                    // geglu_shift bwd: dh = d_act·(g+1)·gelu′(h); dg = d_act·gelu_erf(h).
                    g.step(K_GEGLU_DA, &[&d_act, &self.g, &self.h, &d_h], &[t * inner], t * inner),
                    g.step(K_GEGLU_DB, &[&d_act, &self.h, &d_g], &[t * inner], t * inner),
                    // w_h / w_g bwd; d_xn accumulates over experts.
                    g.step(K_MATMUL_DX, &[&d_h, ep("w_h.weight"), d_xn], &[t, d, inner, acc_first], t * d),
                    g.step(K_MATMUL_DW, &[&d_h, xn, &eg.w_h], &[t, d, inner], inner * d),
                    g.step(K_MATMUL_DX, &[&d_g, ep("w_g.weight"), d_xn], &[t, d, inner, 1], t * d),
                    g.step(K_MATMUL_DW, &[&d_g, xn, &eg.w_g], &[t, d, inner], inner * d),
                ],
            );
        }
        // Router bwd (no load-balance / z-loss for finetune): d_gate → d_logits → xn.
        let d_logits = g.storage((t * e) as u64);
        let fe = g.storage_init("md.fe", &vec![0.0f32; e as usize]);
        g.submit(
            &[],
            &[
                g.step(K_ROUTER_BWD, &[&self.logits, &self.gate, &d_gate, &fe, &d_logits], &[t, e, self.top_k, 0, f(0.0), f(0.0)], t),
                g.step(K_MATMUL_DX, &[&d_logits, self.wb("router.weight"), d_xn], &[t, d, e, 1], t * d),
                g.step(K_MATMUL_DW, &[&d_logits, xn, &gr.router], &[t, d, e], e * d),
            ],
        );
    }
}

/// Gradient buffers for one MoE expert (`w_h`/`w_g` `[inner,d]`, `w_down` `[d,inner]`).
pub struct MoeExpertGrads {
    pub w_h: DeviceBuffer,
    pub w_g: DeviceBuffer,
    pub w_down: DeviceBuffer,
}

/// Gradient buffers for an [`MoeFfn`] (router + per-expert), zeroed on build.
pub struct MoeGrads {
    pub router: DeviceBuffer,
    pub experts: Vec<MoeExpertGrads>,
}

impl MoeGrads {
    pub fn new(g: &Gpu, d: u32, inner: u32, e: u32) -> MoeGrads {
        let z = |n: u32| g.storage_init("md.moeg", &vec![0.0f32; n as usize]);
        MoeGrads {
            router: z(e * d),
            experts: (0..e).map(|_| MoeExpertGrads { w_h: z(inner * d), w_g: z(inner * d), w_down: z(d * inner) }).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    #[test]
    fn parallel_block_runs() {
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot) = (6u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32);
        let mut rng = Rng::new(6);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("ln.weight".into(), vec![1.0; d as usize]);
        w.insert("ln.bias".into(), r(d as usize));
        w.insert("attn.qkv.weight".into(), r((3 * d * d) as usize));
        w.insert("attn.proj.weight".into(), r((d * d) as usize));
        w.insert("attn.proj.bias".into(), r(d as usize));
        w.insert("mlp.fc1.weight".into(), r((ff * d) as usize));
        w.insert("mlp.fc1.bias".into(), r(ff as usize));
        w.insert("mlp.fc2.weight".into(), r((d * ff) as usize));
        w.insert("mlp.fc2.bias".into(), r(d as usize));
        let blk = MoondreamBlock::new(&gpu, &w, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let x = gpu.storage_init("x", &(0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>());
        let out = gpu.read(blk.forward(&x), blk.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn tau_temperature_changes_block_output() {
        // A block with attn.tau.* present applies per-head temperature to q,v and
        // must differ from the same weights without tau.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot) = (6u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32);
        let mut rng = Rng::new(11);
        let base = block_weights(d, ff, &mut rng);
        let x: Vec<f32> = (0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect();

        let blk = MoondreamBlock::new(&gpu, &base, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let xb = gpu.storage_init("x", &x);
        assert!(!blk.tau);
        let plain = gpu.read(blk.forward(&xb), blk.numel());

        let mut tw = base.clone();
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        tw.insert("attn.tau.wq".into(), r((nh * 3 * d) as usize));
        tw.insert("attn.tau.wv".into(), r((nh * 3 * d) as usize));
        tw.insert("attn.tau.alpha".into(), r(nh as usize));
        let tblk = MoondreamBlock::new(&gpu, &tw, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let xb2 = gpu.storage_init("x2", &x);
        assert!(tblk.tau);
        let tau = gpu.read(tblk.forward(&xb2), tblk.numel());

        assert!(tau.iter().all(|v| v.is_finite()));
        let diff: f32 = plain.iter().zip(&tau).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "tau must change the block output, Σ|Δ|={diff}");
    }

    #[test]
    fn dense_block_backward_matches_finite_diff() {
        // Directional finite-diff gradcheck of the dense parallel-block backward:
        // the input grad exercises the whole reverse chain (residual → MLP+attn →
        // shared-LN accumulation → LN dx); a weight grad covers the accumulating path.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot, theta) = (5u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32, 1.5e6f32);
        let mut rng = Rng::new(7);
        let w = block_weights(d, ff, &mut rng);
        let n = (t * d) as usize;
        let x_host: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();

        // Analytic grads (forward populates the SSA cache, then backward).
        let blk = MoondreamBlock::new(&gpu, &w, t, d, nh, hd, ff, prefix, rot, theta);
        let xb = gpu.storage_init("x", &x_host);
        let _ = blk.forward(&xb);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = MoondreamBlockGrads::new(&gpu, d, ff);
        let d_x = gpu.storage((t * d) as u64);
        blk.backward(&xb, &d_out, &gr, &d_x);
        let dx = gpu.read(&d_x, n);
        let g_ln_w = gpu.read(&gr.ln_w, d as usize);

        // L(w, x) = Σ block.forward(x) (matches d_out = ones).
        let loss = |wm: &HashMap<String, Vec<f32>>, xh: &[f32]| -> f32 {
            let b = MoondreamBlock::new(&gpu, wm, t, d, nh, hd, ff, prefix, rot, theta);
            let xbb = gpu.storage_init("x", xh);
            gpu.read(b.forward(&xbb), n).iter().sum::<f32>()
        };
        let eps = 1e-3f32;
        let ok = |a: f32, num: f32| (a - num).abs() <= 4e-3 + 8e-2 * num.abs();

        // Input-gradient check on a sample of positions.
        for &i in &[0usize, 7, 13, 21, 33, 44] {
            let (mut xp, mut xm) = (x_host.clone(), x_host.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let num = (loss(&w, &xp) - loss(&w, &xm)) / (2.0 * eps);
            assert!(ok(dx[i], num), "d_x[{i}]: analytic {} vs numeric {}", dx[i], num);
        }
        // Shared-LN weight-gradient check.
        for &j in &[0usize, 5, 11] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("ln.weight").unwrap()[j] += eps;
            wm2.get_mut("ln.weight").unwrap()[j] -= eps;
            let num = (loss(&wp, &x_host) - loss(&wm2, &x_host)) / (2.0 * eps);
            assert!(ok(g_ln_w[j], num), "d ln.w[{j}]: analytic {} vs numeric {}", g_ln_w[j], num);
        }
    }

    #[test]
    fn tau_block_backward_matches_finite_diff() {
        // Gradcheck the tau path: input grad exercises the full tau chain (tau_scale
        // in-grad + tok_feat/wq/wv/alpha); the alpha and wq grads cover the tau head.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot, theta) = (5u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32, 1.5e6f32);
        let mut rng = Rng::new(21);
        let mut w = block_weights(d, ff, &mut rng);
        {
            let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
            w.insert("attn.tau.wq".into(), r((nh * 3 * d) as usize));
            w.insert("attn.tau.wv".into(), r((nh * 3 * d) as usize));
            w.insert("attn.tau.alpha".into(), r(nh as usize));
        }
        let n = (t * d) as usize;
        let x_host: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();

        let blk = MoondreamBlock::new(&gpu, &w, t, d, nh, hd, ff, prefix, rot, theta);
        assert!(blk.tau);
        let xb = gpu.storage_init("x", &x_host);
        let _ = blk.forward(&xb);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = MoondreamBlockGrads::new(&gpu, d, ff).with_tau(&gpu, nh, d);
        let d_x = gpu.storage((t * d) as u64);
        blk.backward(&xb, &d_out, &gr, &d_x);
        let dx = gpu.read(&d_x, n);
        let tg = gr.tau.as_ref().unwrap();
        let g_alpha = gpu.read(&tg.alpha, nh as usize);
        let g_wq = gpu.read(&tg.wq, (nh * 3 * d) as usize);

        let loss = |wm: &HashMap<String, Vec<f32>>, xh: &[f32]| -> f32 {
            let b = MoondreamBlock::new(&gpu, wm, t, d, nh, hd, ff, prefix, rot, theta);
            let xbb = gpu.storage_init("x", xh);
            gpu.read(b.forward(&xbb), n).iter().sum::<f32>()
        };
        let eps = 1e-3f32;
        let ok = |a: f32, num: f32| (a - num).abs() <= 4e-3 + 8e-2 * num.abs();

        for &i in &[0usize, 7, 13, 21, 33, 44] {
            let (mut xp, mut xm) = (x_host.clone(), x_host.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let num = (loss(&w, &xp) - loss(&w, &xm)) / (2.0 * eps);
            assert!(ok(dx[i], num), "d_x[{i}]: analytic {} vs numeric {}", dx[i], num);
        }
        for &h in &[0usize, 1] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("attn.tau.alpha").unwrap()[h] += eps;
            wm2.get_mut("attn.tau.alpha").unwrap()[h] -= eps;
            let num = (loss(&wp, &x_host) - loss(&wm2, &x_host)) / (2.0 * eps);
            assert!(ok(g_alpha[h], num), "d tau.alpha[{h}]: analytic {} vs numeric {}", g_alpha[h], num);
        }
        for &j in &[0usize, 17, 40] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("attn.tau.wq").unwrap()[j] += eps;
            wm2.get_mut("attn.tau.wq").unwrap()[j] -= eps;
            let num = (loss(&wp, &x_host) - loss(&wm2, &x_host)) / (2.0 * eps);
            assert!(ok(g_wq[j], num), "d tau.wq[{j}]: analytic {} vs numeric {}", g_wq[j], num);
        }
    }

    #[test]
    fn moe_block_backward_matches_finite_diff() {
        // Gradcheck the MoE FFN branch backward inside the parallel block: input grad
        // exercises experts (geglu_shift + w_h/w_g/w_down) + router; an expert weight
        // and a router weight cover those paths.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot, theta) = (5u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32, 1.5e6f32);
        let (inner, e, top_k) = (6u32, 3u32, 2u32);
        let mut rng = Rng::new(31);
        let bw = block_weights(d, ff, &mut rng);
        let mut mw = HashMap::new();
        {
            let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect::<Vec<f32>>();
            mw.insert("router.weight".to_string(), r((e * d) as usize));
            for ei in 0..e {
                mw.insert(format!("experts.{ei}.w_h.weight"), r((inner * d) as usize));
                mw.insert(format!("experts.{ei}.w_g.weight"), r((inner * d) as usize));
                mw.insert(format!("experts.{ei}.w_down.weight"), r((d * inner) as usize));
            }
        }
        let n = (t * d) as usize;
        let x_host: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();

        let build = |bwm: &HashMap<String, Vec<f32>>, mwm: &HashMap<String, Vec<f32>>| -> MoondreamBlock {
            let moe = MoeFfn::new(&gpu, mwm, t, d, inner, e, top_k);
            MoondreamBlock::new(&gpu, bwm, t, d, nh, hd, ff, prefix, rot, theta).with_moe(moe)
        };
        let blk = build(&bw, &mw);
        let xb = gpu.storage_init("x", &x_host);
        let _ = blk.forward(&xb);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = MoondreamBlockGrads::new(&gpu, d, ff).with_moe(&gpu, d, inner, e);
        let d_x = gpu.storage((t * d) as u64);
        blk.backward(&xb, &d_out, &gr, &d_x);
        let dx = gpu.read(&d_x, n);
        let mg = gr.moe.as_ref().unwrap();
        let g_wh0 = gpu.read(&mg.experts[0].w_h, (inner * d) as usize);
        let g_router = gpu.read(&mg.router, (e * d) as usize);

        let loss = |mwm: &HashMap<String, Vec<f32>>, xh: &[f32]| -> f32 {
            let b = build(&bw, mwm);
            let xbb = gpu.storage_init("x", xh);
            gpu.read(b.forward(&xbb), n).iter().sum::<f32>()
        };
        let eps = 1e-3f32;
        let ok = |a: f32, num: f32| (a - num).abs() <= 4e-3 + 8e-2 * num.abs();

        for &i in &[0usize, 7, 13, 21, 33, 44] {
            let (mut xp, mut xm) = (x_host.clone(), x_host.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let num = (loss(&mw, &xp) - loss(&mw, &xm)) / (2.0 * eps);
            assert!(ok(dx[i], num), "d_x[{i}]: analytic {} vs numeric {}", dx[i], num);
        }
        for &j in &[0usize, 17, 40] {
            let (mut mp, mut mm) = (mw.clone(), mw.clone());
            mp.get_mut("experts.0.w_h.weight").unwrap()[j] += eps;
            mm.get_mut("experts.0.w_h.weight").unwrap()[j] -= eps;
            let num = (loss(&mp, &x_host) - loss(&mm, &x_host)) / (2.0 * eps);
            assert!(ok(g_wh0[j], num), "d expert0.w_h[{j}]: analytic {} vs numeric {}", g_wh0[j], num);
        }
        for &j in &[0usize, 20, 40] {
            let (mut mp, mut mm) = (mw.clone(), mw.clone());
            mp.get_mut("router.weight").unwrap()[j] += eps;
            mm.get_mut("router.weight").unwrap()[j] -= eps;
            let num = (loss(&mp, &x_host) - loss(&mm, &x_host)) / (2.0 * eps);
            assert!(ok(g_router[j], num), "d router[{j}]: analytic {} vs numeric {}", g_router[j], num);
        }
    }

    #[test]
    fn check_moondream_dense_decoder_backward() {
        // End-to-end gradcheck of the dense decoder backward: loss = mean masked CE.
        // The image-embed grad exercises splice_bwd → the full block chain → head;
        // lm_head.bias and tok.weight grads cover the head + embedding-scatter paths.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, vocab, n_img, prefix, rot, theta, nl) = (7u32, 16u32, 2u32, 8u32, 32u32, 19u32, 3u32, 4u32, 4u32, 1.5e6f32, 2u32);
        let mut rng = Rng::new(15);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.3).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("tok.weight".to_string(), r((vocab * d) as usize));
        w.insert("post_ln.weight".to_string(), vec![1.0; d as usize]);
        w.insert("post_ln.bias".to_string(), r(d as usize));
        w.insert("lm_head.weight".to_string(), r((vocab * d) as usize));
        w.insert("lm_head.bias".to_string(), r(vocab as usize));
        let img: Vec<f32> = r((n_img * d) as usize); // last use of `r` before rng is reborrowed
        drop(r);
        for l in 0..nl {
            for (k, v) in block_weights(d, ff, &mut rng) {
                w.insert(format!("blocks.{l}.{k}"), v);
            }
        }
        let tokens = vec![0u32, 5, 5, 5, 7, 9, 11]; // bos + 3 image + 3 text
        let mut targets = vec![5u32, 0, 0, 0, 9, 11, 13];
        for tg in targets.iter_mut().take(1 + n_img as usize).skip(1) {
            *tg = IGNORE; // image rows unsupervised
        }

        let build = |wm: &HashMap<String, Vec<f32>>| -> MoondreamDecoder {
            let blocks = (0..nl)
                .map(|l| {
                    let bw: HashMap<String, Vec<f32>> = wm.iter().filter_map(|(k, v)| k.strip_prefix(&format!("blocks.{l}.")).map(|s| (s.to_string(), v.clone()))).collect();
                    MoondreamBlock::new(&gpu, &bw, t, d, nh, hd, ff, prefix, rot, theta)
                })
                .collect();
            MoondreamDecoder::new(&gpu, wm, blocks, t, d, vocab, n_img)
        };

        // Analytic grads.
        let dec = build(&w);
        let _ = dec.forward(&tokens, &targets, &img);
        let gr = MoondreamDecoderGrads::new(&gpu, nl, d, ff, vocab, n_img);
        dec.backward(&targets, &gr);
        let d_img = gpu.read(&gr.d_image_embeds, (n_img * d) as usize);
        let g_lmb = gpu.read(&gr.lm_head_b, vocab as usize);
        let g_tok = gpu.read(&gr.tok_w, (vocab * d) as usize);

        let loss = |wm: &HashMap<String, Vec<f32>>, im: &[f32]| -> f32 { build(wm).forward(&tokens, &targets, im) };
        let eps = 1e-3f32;
        let ok = |a: f32, num: f32| (a - num).abs() <= 4e-3 + 8e-2 * num.abs();

        // Image-embedding grad (splice → blocks → head).
        for &i in &[0usize, 7, 13, 20, 33, 44] {
            let (mut ip, mut im) = (img.clone(), img.clone());
            ip[i] += eps;
            im[i] -= eps;
            let num = (loss(&w, &ip) - loss(&w, &im)) / (2.0 * eps);
            assert!(ok(d_img[i], num), "d_image_embeds[{i}]: analytic {} vs numeric {}", d_img[i], num);
        }
        // lm_head.bias grad.
        for &j in &[0usize, 9, 13] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("lm_head.bias").unwrap()[j] += eps;
            wm2.get_mut("lm_head.bias").unwrap()[j] -= eps;
            let num = (loss(&wp, &img) - loss(&wm2, &img)) / (2.0 * eps);
            assert!(ok(g_lmb[j], num), "d lm_head.bias[{j}]: analytic {} vs numeric {}", g_lmb[j], num);
        }
        // tok.weight grad on a supervised text token's row (token 7 at position 4).
        for &c in &[0usize, 5, 11] {
            let j = 7 * d as usize + c;
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("tok.weight").unwrap()[j] += eps;
            wm2.get_mut("tok.weight").unwrap()[j] -= eps;
            let num = (loss(&wp, &img) - loss(&wm2, &img)) / (2.0 * eps);
            assert!(ok(g_tok[j], num), "d tok.weight[{j}]: analytic {} vs numeric {}", g_tok[j], num);
        }
    }

    fn block_weights(d: u32, ff: u32, rng: &mut Rng) -> HashMap<String, Vec<f32>> {
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("ln.weight".into(), vec![1.0; d as usize]);
        w.insert("ln.bias".into(), r(d as usize));
        w.insert("attn.qkv.weight".into(), r((3 * d * d) as usize));
        w.insert("attn.proj.weight".into(), r((d * d) as usize));
        w.insert("attn.proj.bias".into(), r(d as usize));
        w.insert("mlp.fc1.weight".into(), r((ff * d) as usize));
        w.insert("mlp.fc1.bias".into(), r(ff as usize));
        w.insert("mlp.fc2.weight".into(), r((d * ff) as usize));
        w.insert("mlp.fc2.bias".into(), r(d as usize));
        w
    }

    #[test]
    fn full_decoder_forward_is_finite() {
        // t=8 stream: bos + 4 image + 3 text (prefix=5). 2 dense layers.
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, vocab, n_img, prefix, rot) = (8u32, 16u32, 2u32, 8u32, 32u32, 23u32, 4u32, 5u32, 4u32);
        let mut rng = Rng::new(9);
        let bw0 = block_weights(d, ff, &mut rng);
        let bw1 = block_weights(d, ff, &mut rng);
        let blocks = vec![
            MoondreamBlock::new(&gpu, &bw0, t, d, nh, hd, ff, prefix, rot, 1.5e6),
            MoondreamBlock::new(&gpu, &bw1, t, d, nh, hd, ff, prefix, rot, 1.5e6),
        ];
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut dw = HashMap::new();
        dw.insert("tok.weight".into(), r((vocab * d) as usize));
        dw.insert("post_ln.weight".into(), vec![1.0; d as usize]);
        dw.insert("post_ln.bias".into(), r(d as usize));
        dw.insert("lm_head.weight".into(), r((vocab * d) as usize));
        dw.insert("lm_head.bias".into(), r(vocab as usize));
        let dec = MoondreamDecoder::new(&gpu, &dw, blocks, t, d, vocab, n_img);

        let tokens = vec![0u32, 5, 5, 5, 5, 7, 9, 11]; // bos + image + text
        let mut targets = vec![5u32, 0, 0, 0, 0, 9, 11, 13];
        for tg in targets.iter_mut().take(5).skip(1) {
            *tg = IGNORE;
        }
        let img: Vec<f32> = r((n_img * d) as usize);
        let loss = dec.forward(&tokens, &targets, &img);
        assert!(loss.is_finite() && loss > 0.0, "moondream decoder loss must be finite+positive, got {loss}");
    }

    #[test]
    fn parallel_block_with_moe_runs() {
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff, prefix, rot) = (6u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32);
        let (inner, e, top_k) = (4u32, 3u32, 2u32);
        let mut rng = Rng::new(8);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut bw = HashMap::new();
        bw.insert("ln.weight".into(), vec![1.0; d as usize]);
        bw.insert("ln.bias".into(), r(d as usize));
        bw.insert("attn.qkv.weight".into(), r((3 * d * d) as usize));
        bw.insert("attn.proj.weight".into(), r((d * d) as usize));
        bw.insert("attn.proj.bias".into(), r(d as usize));
        // dense fc weights present but unused when MoE is attached.
        bw.insert("mlp.fc1.weight".into(), r((ff * d) as usize));
        bw.insert("mlp.fc1.bias".into(), r(ff as usize));
        bw.insert("mlp.fc2.weight".into(), r((d * ff) as usize));
        bw.insert("mlp.fc2.bias".into(), r(d as usize));
        let mut mw = HashMap::new();
        mw.insert("router.weight".into(), r((e * d) as usize));
        for ei in 0..e {
            mw.insert(format!("experts.{ei}.w_h.weight"), r((inner * d) as usize));
            mw.insert(format!("experts.{ei}.w_g.weight"), r((inner * d) as usize));
            mw.insert(format!("experts.{ei}.w_down.weight"), r((d * inner) as usize));
        }
        let moe = MoeFfn::new(&gpu, &mw, t, d, inner, e, top_k);
        let blk = MoondreamBlock::new(&gpu, &bw, t, d, nh, hd, ff, prefix, rot, 1.5e6).with_moe(moe);
        let x = gpu.storage_init("x", &(0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>());
        let out = gpu.read(blk.forward(&x), blk.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn moe_ffn_geglu_runs() {
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, inner, e, top_k) = (4u32, 8u32, 4u32, 3u32, 2u32);
        let mut rng = Rng::new(5);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("router.weight".into(), r((e * d) as usize));
        for ei in 0..e {
            w.insert(format!("experts.{ei}.w_h.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_g.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_down.weight"), r((d * inner) as usize));
        }
        let ffn = MoeFfn::new(&gpu, &w, t, d, inner, e, top_k);
        let xn = gpu.storage_init("xn", &(0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>());
        let out = gpu.read(ffn.forward(&xn), ffn.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-9));
    }
}
