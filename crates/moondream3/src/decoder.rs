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
        // --- int8 expert tier (inference only; see `MoeFfn8`) ---
        ("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8), // 41
        ("max_abs_row", kernels::MAX_ABS_ROW),           // 42 (activation scale)
        ("quant_pack", kernels::QUANT_PACK),             // 43 (activation pack)
        // --- incremental decode tier (inference only; see `KvCache`) ---
        ("kv_append", kernels::KV_APPEND),               // 44
        ("attn_decode_scores", kernels::ATTN_DECODE_SCORES), // 45
        ("decode_softmax", kernels::DECODE_SOFTMAX),     // 46
        ("attn_decode_apply", kernels::ATTN_DECODE_APPLY), // 47
        ("rope_partial_at", kernels::ROPE_PARTIAL_AT),   // 48 (partial RoPE at an explicit pos)
    ]
}

/// `model::block::GqaDecodeIds` over this crate's slots. Moondream's attention
/// is full MHA, so `n_kv_heads == n_heads` and the shared GQA decode primitive
/// runs at `group = 1` - which is plain MHA, not an approximation of it.
/// `rope_partial_at`: partial RoPE at an EXPLICIT position - what a single
/// decode row needs, since `rope_partial` derives its position from the row
/// index and a one-row call is therefore always position 0.
const K_ROPE_PARTIAL_AT: usize = 48;

fn decode_ids() -> model::block::GqaDecodeIds {
    model::block::GqaDecodeIds { kv_append: 44, attn_decode_scores: 45, decode_softmax: 46, attn_decode_apply: 47 }
}

/// `moe_linear_gated_i8`: one expert linear over int8 weights and dynamically
/// quantized activations, skipping rows this expert is not routed to.
const K_MOE_LIN_I8: usize = 41;
/// `model::int8::quant_rows_steps`'s `[max_abs_row, quant_pack]` pair.
const K_QUANT: [usize; 2] = [42, 43];

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
pub struct MoondreamDecoder {
    blocks: Vec<MoondreamBlock>,
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
    /// One activation set shared by every block, when this decoder was built
    /// for inference. `None` on a training decoder, where each block owns its
    /// own (its forward IS the backprop cache). See [`BlockScratch`].
    shared: Option<BlockScratch>,
}

impl MoondreamDecoder {
    /// Build from per-layer prefixed weights (`blocks.{l}.…`) plus `tok.weight`
    /// `[vocab,d]`, `post_ln.weight`/`.bias`, `lm_head.weight` `[vocab,d]`/`.bias`.
    /// `moe_layers` marks which layers use the MoE FFN (their MoE weights are under
    /// `blocks.{l}.moe.…`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &Gpu, weights: &HashMap<String, Vec<f32>>, blocks: Vec<MoondreamBlock>, t: u32, d: u32, vocab: u32, n_img: u32) -> MoondreamDecoder {
        let w = weights
            .iter()
            .filter(|(k, _)| !k.starts_with("blocks."))
            .map(|(k, v)| (k.clone(), gpu.storage_init(k, v)))
            .collect();
        MoondreamDecoder {
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
            shared: None,
        }
    }

    /// Move this decoder's blocks onto ONE shared activation set.
    ///
    /// Inference only: every block drops its own set, so `backward` on any of
    /// them refuses by name. At the released config this is the difference
    /// between ~10.3 GiB of block scratch and ~0.6 GiB - see [`BlockScratch`].
    pub fn share_scratch(mut self, gpu: &Gpu, n_heads: u32, ff: u32) -> MoondreamDecoder {
        self.shared = Some(BlockScratch::new(gpu, self.t, self.d, n_heads, ff));
        self.blocks = self.blocks.into_iter().map(|b| b.without_scratch()).collect();
        self
    }

    /// Run the block stack over `cur`, on the shared scratch when there is one.
    fn run_blocks<'a>(&'a self, g: &Gpu, mut cur: &'a DeviceBuffer) -> &'a DeviceBuffer {
        for b in &self.blocks {
            cur = match &self.shared {
                Some(sc) => b.forward_on(g, sc, cur),
                None => b.forward(g, cur),
            };
        }
        cur
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("decoder weight missing: {n}"))
    }
    /// Forward → mean masked cross-entropy. `tokens`/`targets` length `t`
    /// (targets IGNORE at the image + non-supervised rows); `image_embeds` is the
    /// `[n_img, d]` connector output spliced at rows `[1, 1+n_img)`.
    pub fn forward(&self, g: &Gpu, tokens: &[u32], targets: &[u32], image_embeds: &[f32]) -> f32 {
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
        let cur: &DeviceBuffer = self.run_blocks(g, &self.res);
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

    /// Run the forward through the LM head and return the `[t, vocab]` logits
    /// (embed → splice image tokens → blocks → post-LN → lm_head), skipping the CE.
    /// Used for real-weight reference parity.
    pub fn logits_all(&self, g: &Gpu, tokens: &[u32], image_embeds: &[f32]) -> Vec<f32> {
        let (t, d, v) = (self.t, self.d, self.vocab);
        g.write(&self.tokens, tokens);
        let img = g.storage_init("md.img", image_embeds);
        let mut steps = vec![g.step(13, &[&self.tokens, self.wb("tok.weight"), &self.res], &[d, t], t * d)];
        if self.n_img > 0 {
            steps.push(g.step(14, &[&img, &self.res], &[self.n_img * d, d], self.n_img * d));
        }
        g.submit(&[], &steps);
        let cur: &DeviceBuffer = self.run_blocks(g, &self.res);
        g.submit(
            &[],
            &[
                g.step(4, &[cur, self.wb("post_ln.weight"), self.wb("post_ln.bias"), &self.normed], &[d, t, f(LN_EPS)], t),
                g.step(0, &[&self.normed, self.wb("lm_head.weight"), &self.logits], &[t, d, v], t * v),
                g.step(6, &[&self.logits, self.wb("lm_head.bias")], &[t, v], t * v),
            ],
        );
        g.read(&self.logits, (t * v) as usize)
    }

    /// Allocate one [`KvCache`] per layer, sized for the built context.
    pub fn new_kv_caches(&self, g: &Gpu, n_heads: u32, head_dim: u32, ff: u32) -> Vec<KvCache> {
        (0..self.blocks.len()).map(|_| KvCache::new(g, self.t, self.d, n_heads, head_dim, ff)).collect()
    }

    /// Prefill: one batched masked forward over `tokens[..n]`, seeding every
    /// layer's cache, and returning the hidden state of row `n - 1`.
    ///
    /// The batched pass is what makes the image prefix BIDIRECTIONAL - the
    /// decode steps that follow are causal-only and could not produce it. The
    /// cost is one `O(n²)` forward, paid once, instead of one per token.
    pub fn prefill(&self, g: &Gpu, tokens: &[u32], image_embeds: &[f32], caches: &[KvCache], n: u32) -> Vec<f32> {
        let (t, d) = (self.t, self.d);
        assert_eq!(caches.len(), self.blocks.len(), "one cache per layer");
        assert!(n <= t, "prefill: {n} rows exceeds the built context {t}");
        g.write(&self.tokens, tokens);
        let img = g.storage_init("md.img", image_embeds);
        let mut steps = vec![g.step(13, &[&self.tokens, self.wb("tok.weight"), &self.res], &[d, t], t * d)];
        if self.n_img > 0 {
            steps.push(g.step(14, &[&img, &self.res], &[self.n_img * d, d], self.n_img * d));
        }
        g.submit(&[], &steps);

        let mut cur: &DeviceBuffer = &self.res;
        for (i, b) in self.blocks.iter().enumerate() {
            cur = match &self.shared {
                Some(sc) => {
                    let out = b.forward_on(g, sc, cur);
                    b.fill_kv(g, sc, &caches[i], n);
                    out
                }
                None => {
                    let out = b.forward(g, cur);
                    b.fill_kv(g, b.own.as_ref().expect("owned scratch"), &caches[i], n);
                    out
                }
            };
        }
        // The last prompt row's hidden state is what the first decode step
        // continues from.
        let all = g.read(cur, (t * d) as usize);
        all[((n - 1) * d) as usize..(n * d) as usize].to_vec()
    }

    /// Project one hidden row through post-LN and the LM head.
    pub fn head(&self, g: &Gpu, hidden: &[f32]) -> Vec<f32> {
        let (d, v) = (self.d, self.vocab);
        let x = g.storage_init("md.head_in", hidden);
        let normed = g.storage(d as u64);
        let logits = g.storage(v as u64);
        g.submit(
            &[],
            &[
                g.step(4, &[&x, self.wb("post_ln.weight"), self.wb("post_ln.bias"), &normed], &[d, 1, f(LN_EPS)], 1),
                g.step(0, &[&normed, self.wb("lm_head.weight"), &logits], &[1, d, v], v),
                g.step(6, &[&logits, self.wb("lm_head.bias")], &[1, v], v),
            ],
        );
        g.read(&logits, v as usize)
    }

    /// One incremental decode step across every layer: a token embedding in, the
    /// next hidden row out. `pos` is the new token's absolute position.
    pub fn decode_step(&self, g: &Gpu, caches: &[KvCache], token: u32, pos: u32) -> Vec<f32> {
        let d = self.d;
        let row = g.storage(d as u64);
        // `embed`'s token buffer is `array<u32>` - RAW ids, not f32 bits. Writing
        // it with `storage_init` (which stores f32) makes the gather index
        // garbage and reads off the end of the embedding table.
        let tok = g.storage(1);
        g.write(&tok, &[token]);
        g.submit(&[], &[g.step(13, &[&tok, self.wb("tok.weight"), &row], &[d, 1], d)]);
        // Each layer has its OWN `KvCache`, so layer i's `out` is a distinct
        // buffer that layer i+1 reads immediately and nothing else touches
        // until the next token, by which point it has already been consumed.
        // The hidden state can therefore stay on the device across all 24
        // layers - an earlier version round-tripped it to the host per layer,
        // which is 48 transfers per token and 24 forced syncs for nothing.
        let mut cur: &DeviceBuffer = &row;
        for (i, b) in self.blocks.iter().enumerate() {
            cur = b.decode_step(g, &caches[i], cur, pos);
        }
        g.read(cur, d as usize)
    }

    /// Decoder backward (dense blocks): from the cached forward (call `forward`
    /// first), fill every grad in `gr`. Chain: CE → lm_head → post-LN → blocks in
    /// reverse (each `MoondreamBlock::backward`, threading the residual-stream grad)
    /// → splice (image rows → `d_image_embeds`, zeroed in the residual grad) →
    /// embedding (text rows → `tok.weight`). Requires all blocks dense/no-tau.
    pub fn backward(&self, g: &Gpu, targets: &[u32], gr: &MoondreamDecoderGrads) {
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
            self.blocks[i].backward(g, x_in, &d_cur, &gr.blocks[i], &d_in);
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
    /// Allocate zeroed grads for a dense decoder of the given shape (dense blocks).
    pub fn new(g: &Gpu, n_layers: u32, d: u32, ff: u32, vocab: u32, n_img: u32) -> MoondreamDecoderGrads {
        Self::from_blocks(g, (0..n_layers).map(|_| MoondreamBlockGrads::new(g, d, ff)).collect(), d, vocab, n_img)
    }

    /// Allocate zeroed decoder-level grads around caller-built per-block grads (so
    /// tau/MoE blocks can supply `with_tau`/`with_moe` grads).
    pub fn from_blocks(g: &Gpu, blocks: Vec<MoondreamBlockGrads>, d: u32, vocab: u32, n_img: u32) -> MoondreamDecoderGrads {
        let z = |n: u32| g.storage_init("md.dg", &vec![0.0f32; n as usize]);
        MoondreamDecoderGrads {
            blocks,
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

/// One Moondream decoder block - the PARALLEL attn+MLP form: a single shared
/// LayerNorm feeds BOTH the attention and the FFN, and `x = x + l_attn + l_mlp`
/// (a 3-way residual). The attention is full MHA with **partial RoPE** and the
/// **prefix-LM mask** (image prefix bidirectional, else causal); the FFN is the
/// dense variant (layers 0..3) or the MoE variant (layers 4..23, via `with_moe`).
/// The optional tau temperature and the MoE FFN both have gradient-checked backwards.
/// Every activation buffer one [`MoondreamBlock`] forward needs, except its
/// output.
///
/// # Why this is a separate object
///
/// A block used to own its whole scratch set, which is correct for TRAINING -
/// the forward IS the backprop cache, so each block's activations must survive
/// until its own `backward` reads them. For INFERENCE nothing reads them again
/// once the block has produced its output, and at the released config that
/// distinction is the difference between running and not: the set is ~441 MiB
/// per block, dominated by the two `n_heads·t²` attention slabs, so 24 blocks
/// hold ~10.3 GiB of buffers that are live one at a time.
///
/// One shared set plus a per-block `out` is a small fraction of that, and
/// the largest single saving available to this model after int8 weights.
///
/// **Inference only, and structurally so.** Sharing this under the existing
/// backward would have every block differentiate against the LAST block's
/// activations - a plausible, finite, entirely wrong gradient. That is why
/// [`MoondreamBlock::backward`] reads the block's OWN set and a block built
/// without one refuses to run backward at all.
pub struct BlockScratch {
    l_in: DeviceBuffer,
    qkv: DeviceBuffer,
    qkv2: DeviceBuffer,
    tok_feat: DeviceBuffer,
    tqr: DeviceBuffer,
    tvr: DeviceBuffer,
    s3: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    l_attn: DeviceBuffer,
    h: DeviceBuffer,
    h2: DeviceBuffer,
    l_mlp: DeviceBuffer,
    mid: DeviceBuffer,
}

impl BlockScratch {
    /// Sized for one block at `(t, d, n_heads, ff)`. Every block in a stack
    /// shares these dims, which is what makes one set reusable.
    pub fn new(gpu: &Gpu, t: u32, d: u32, n_heads: u32, ff: u32) -> BlockScratch {
        let slab = (n_heads * t * t) as u64;
        BlockScratch {
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
        }
    }
}

/// One decoder layer's persistent K/V cache, plus the single-row scratch an
/// incremental step needs.
///
/// # Why the prefix-LM mask does not appear here
///
/// The batched forward masks with `allow(i, j) = (i < P && j < P) || (j <= i)`:
/// the `P` bos+image rows are bidirectional, everything after is causal. A
/// DECODE row is always at `pos >= P`, so its own mask row is purely causal -
/// and `model::block::gqa_decode_step` reads cache rows `0..=pos` and no
/// others, because no later rows exist yet. The causality is structural rather
/// than applied, which is why no mask kernel is dispatched on this path.
///
/// The bidirectional prefix is still honoured: it is baked into the cached
/// K/V, which the PROMPT's batched forward produced under the full mask.
pub struct KvCache {
    /// `[cap, n_heads·head_dim]` each.
    k: DeviceBuffer,
    v: DeviceBuffer,
    /// `[n_heads, cap]` decode scores/probs.
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    /// Single-row working buffers: `[1, d]`, `[1, 3d]`, `[1, n_heads]`.
    x1: DeviceBuffer,
    l_in: DeviceBuffer,
    qkv: DeviceBuffer,
    qkv2: DeviceBuffer,
    tok_feat: DeviceBuffer,
    tqr: DeviceBuffer,
    tvr: DeviceBuffer,
    s3: DeviceBuffer,
    q: DeviceBuffer,
    k_new: DeviceBuffer,
    v_new: DeviceBuffer,
    ctx: DeviceBuffer,
    l_attn: DeviceBuffer,
    h: DeviceBuffer,
    h2: DeviceBuffer,
    l_mlp: DeviceBuffer,
    out: DeviceBuffer,
    cap: u32,
}

impl KvCache {
    /// Sized for a whole generation: `cap` is the maximum sequence length
    /// (prompt + generated), which is the built context.
    pub fn new(gpu: &Gpu, cap: u32, d: u32, n_heads: u32, head_dim: u32, ff: u32) -> KvCache {
        let hk = n_heads * head_dim;
        KvCache {
            k: gpu.storage((cap * hk) as u64),
            v: gpu.storage((cap * hk) as u64),
            scores: gpu.storage((n_heads * cap) as u64),
            probs: gpu.storage((n_heads * cap) as u64),
            x1: gpu.storage(d as u64),
            l_in: gpu.storage(d as u64),
            qkv: gpu.storage((3 * d) as u64),
            qkv2: gpu.storage((3 * d) as u64),
            tok_feat: gpu.storage((3 * d) as u64),
            tqr: gpu.storage(n_heads as u64),
            tvr: gpu.storage(n_heads as u64),
            s3: gpu.storage((3 * n_heads) as u64),
            q: gpu.storage(hk as u64),
            k_new: gpu.storage(hk as u64),
            v_new: gpu.storage(hk as u64),
            ctx: gpu.storage(hk as u64),
            l_attn: gpu.storage(d as u64),
            h: gpu.storage(ff as u64),
            h2: gpu.storage(ff as u64),
            l_mlp: gpu.storage(d as u64),
            out: gpu.storage(d as u64),
            cap,
        }
    }

    /// This layer's cached-K buffer, for the prompt-time bulk fill.
    pub fn k(&self) -> &DeviceBuffer {
        &self.k
    }
    pub fn v(&self) -> &DeviceBuffer {
        &self.v
    }
    pub fn cap(&self) -> u32 {
        self.cap
    }
}

/// Weight keys: `ln.weight`/`ln.bias`, `attn.qkv.weight` `[3d, d]`,
/// `attn.proj.weight` `[d,d]`/`attn.proj.bias`, `mlp.fc1.weight` `[ff,d]`/`.bias`,
/// `mlp.fc2.weight` `[d,ff]`/`.bias`.
pub struct MoondreamBlock {
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
    /// True when `attn.qkv.bias` is present (the real checkpoint's fused-qkv Linear
    /// has a bias; the synthetic gradcheck weights do not).
    qkv_bias: bool,
    /// This block's OWN activation set. `Some` for a block that will be
    /// differentiated or run standalone; `None` for one in a stack that shares
    /// a single [`BlockScratch`] (see that type's doc for the 16.7x reason).
    own: Option<BlockScratch>,
    out: DeviceBuffer,
    /// `Some` for the MoE layers (4..23); `None` uses the dense FFN.
    moe: Option<MoeFfn>,
    /// The int8 expert tier, when this block was built for INFERENCE at int8
    /// precision. Mutually exclusive with `moe` (a block has one FFN), kept as
    /// a separate field rather than an enum because only `moe` participates in
    /// `backward` - the differentiated code path stays free of a precision
    /// branch it can never take.
    moe8: Option<MoeFfn8>,
}

impl MoondreamBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, n_heads: u32, head_dim: u32, ff: u32, prefix: u32, rot_dim: u32, theta: f32) -> MoondreamBlock {
        let w: HashMap<String, DeviceBuffer> = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        let tau = w.contains_key("attn.tau.wq");
        let qkv_bias = w.contains_key("attn.qkv.bias");
        MoondreamBlock {
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
            qkv_bias,
            own: Some(BlockScratch::new(gpu, t, d, n_heads, ff)),
            out: gpu.storage((t * d) as u64),
            moe: None,
            moe8: None,
        }
    }
    /// Drop this block's own activation set, so it runs on a shared
    /// [`BlockScratch`] via [`MoondreamBlock::forward_on`].
    ///
    /// Inference only, and enforced: [`MoondreamBlock::backward`] refuses a
    /// block with no own set rather than differentiating against whatever the
    /// shared buffers happen to hold.
    pub fn without_scratch(mut self) -> Self {
        self.own = None;
        self
    }

    /// Attach an MoE FFN (replaces the dense FFN branch) for a deep layer.
    pub fn with_moe(mut self, moe: MoeFfn) -> Self {
        assert!(self.moe8.is_none(), "a block has one FFN: with_moe after with_moe8");
        self.moe = Some(moe);
        self
    }

    /// Attach the INT8 expert FFN. Inference only - a block built this way has
    /// no `backward` (it would need fp32 expert weights to recompute from), and
    /// [`MoondreamBlock::backward`] says so by name rather than silently
    /// producing a dense-FFN gradient.
    pub fn with_moe8(mut self, moe: MoeFfn8) -> Self {
        assert!(self.moe.is_none(), "a block has one FFN: with_moe8 after with_moe");
        self.moe8 = Some(moe);
        self
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("block weight missing: {n}"))
    }
    /// Forward using this block's OWN scratch. Requires a block built by
    /// [`MoondreamBlock::new`] (which allocates one); a stack-shared block
    /// built with [`MoondreamBlock::without_scratch`] must use
    /// [`MoondreamBlock::forward_on`] instead.
    pub fn forward(&self, g: &Gpu, x: &DeviceBuffer) -> &DeviceBuffer {
        let sc = self.own.as_ref().expect("MoondreamBlock::forward: this block shares a BlockScratch - call forward_on");
        self.forward_on(g, sc, x)
    }

    /// Forward using a CALLER-OWNED scratch set, so a stack of blocks can share
    /// one. See [`BlockScratch`] for why that is worth doing and why it is
    /// inference-only.
    pub fn forward_on(&self, g: &Gpu, sc: &BlockScratch, x: &DeviceBuffer) -> &DeviceBuffer {
        let (t, d, nh, hd, ff) = (self.t, self.d, self.n_heads, self.head_dim, self.ff);
        let stride3 = 3 * d;
        let mut s: Vec<Step> = Vec::new();

        // Shared LayerNorm (with bias).
        s.push(g.step(4, &[x, self.wb("ln.weight"), self.wb("ln.bias"), &sc.l_in], &[d, t, f(LN_EPS)], t));
        // --- attention branch ---
        // fused qkv = l_in · wqkv^T  ([t, 3d]) (+ bias on the real checkpoint). The
        // bias must land before tau (whose tok_feat = gelu(qkv)) and RoPE/attention.
        s.push(g.step(0, &[&sc.l_in, self.wb("attn.qkv.weight"), &sc.qkv], &[t, d, stride3], t * stride3));
        if self.qkv_bias {
            s.push(g.step(6, &[&sc.qkv, self.wb("attn.qkv.bias")], &[t, stride3], t * stride3));
        }
        // Per-head attention temperature (tau): scale q and v (NOT k) by a per-
        // (head,token) scalar computed from the raw qkv, BEFORE RoPE. tok_feat is
        // erf-GELU over the full 3d projection; tok_q/tok_v = tanh(tok_feat·w{q,v}ᵀ);
        // tau_pos = 0.5+sigmoid(alpha·ln(pos+1)) folds on host (positions = row).
        // Scalar scaling commutes with the RoPE rotation, so applying it here (pre-
        // RoPE, matching the reference) is equivalent up to that rotation. The tiny
        // tanh+tau_pos assembly into the [3·nh, t] scale (q | k=1 | v) folds on host.
        let qkv = if self.tau {
            s.push(g.step(16, &[&sc.qkv, &sc.tok_feat], &[t * stride3], t * stride3));
            s.push(g.step(0, &[&sc.tok_feat, self.wb("attn.tau.wq"), &sc.tqr], &[t, stride3, nh], t * nh));
            s.push(g.step(0, &[&sc.tok_feat, self.wb("attn.tau.wv"), &sc.tvr], &[t, stride3, nh], t * nh));
            g.submit(&[], &s);
            s = Vec::new();
            let tqr = g.read(&sc.tqr, (t * nh) as usize);
            let tvr = g.read(&sc.tvr, (t * nh) as usize);
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
            g.write(&sc.s3, &packed);
            // Treat qkv as [t, 3·nh, hd]: scale q-heads by s_q, k-heads by 1, v by s_v.
            s.push(g.step(17, &[&sc.qkv, &sc.s3, &sc.qkv2], &[t, 3 * nh, hd], t * stride3));
            &sc.qkv2
        } else {
            &sc.qkv
        };
        // partial RoPE on q (off 0) and k (off d)
        let half = self.rot_dim / 2;
        s.push(g.step(8, &[qkv], &[t, nh, hd, stride3, 0, t, f(self.theta), self.rot_dim], t * nh * half));
        s.push(g.step(8, &[qkv], &[t, nh, hd, stride3, d, t, f(self.theta), self.rot_dim], t * nh * half));
        // bidir scores → prefix-LM mask → bidir softmax → bidir apply
        s.push(g.step(9, &[qkv, &sc.scores], &[1, nh, t, hd, stride3, 0, d], nh * t * t));
        s.push(g.step(10, &[&sc.scores], &[1, nh, t, self.prefix], nh * t * t));
        s.push(g.step(11, &[&sc.scores, &sc.probs], &[1, nh, t], nh * t));
        s.push(g.step(12, &[&sc.probs, qkv, &sc.ctx], &[1, nh, t, hd, stride3, 2 * d, d], nh * t * hd));
        // proj + bias → l_attn. Submit phase 1 (LN + attention) so l_in/l_attn are
        // ready before the FFN (MoE submits internally).
        s.push(g.step(0, &[&sc.ctx, self.wb("attn.proj.weight"), &sc.l_attn], &[t, d, d], t * d));
        s.push(g.step(6, &[&sc.l_attn, self.wb("attn.proj.bias")], &[t, d], t * d));
        g.submit(&[], &s);

        // --- FFN branch on the SAME l_in: MoE (layers 4..23) or dense. ---
        let l_mlp: &DeviceBuffer = if let Some(moe) = &self.moe {
            moe.forward(g, &sc.l_in)
        } else if let Some(moe) = &self.moe8 {
            moe.forward(g, &sc.l_in)
        } else {
            g.submit(
                &[],
                &[
                    g.step(0, &[&sc.l_in, self.wb("mlp.fc1.weight"), &sc.h], &[t, d, ff], t * ff),
                    g.step(6, &[&sc.h, self.wb("mlp.fc1.bias")], &[t, ff], t * ff),
                    g.step(5, &[&sc.h, &sc.h2], &[t * ff], t * ff), // tanh GELU
                    g.step(0, &[&sc.h2, self.wb("mlp.fc2.weight"), &sc.l_mlp], &[t, ff, d], t * d),
                    g.step(6, &[&sc.l_mlp, self.wb("mlp.fc2.bias")], &[t, d], t * d),
                ],
            );
            &sc.l_mlp
        };
        // --- 3-way residual: out = x + l_attn + l_mlp ---
        g.submit(
            &[],
            &[
                g.step(7, &[x, &sc.l_attn, &sc.mid], &[t * d], t * d),
                g.step(7, &[&sc.mid, l_mlp, &self.out], &[t * d], t * d),
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
    /// `layernorm_dx` - the shared-activation pattern. The masked-bidir attention
    /// backward reuses the ViT `_cross` kernels: the cached `probs` already carry the
    /// prefix mask (masked positions have prob≈0 → contribute 0). d_x_in = d_out
    /// (the 3-way residual's identity path) + the LayerNorm input grad.
    /// Copy this block's post-tau, post-RoPE K and V for the first `n` prompt
    /// rows out of the batched forward's scratch and into `kv`.
    ///
    /// Must be called immediately after this block's `forward_on`, and BEFORE
    /// the next block runs: on the shared-scratch path every block writes the
    /// same `qkv` buffer, so afterwards only the last block's survives. That is
    /// the whole reason prefill interleaves the copy with the forward instead
    /// of doing one pass at the end.
    ///
    /// The copy goes through the host because the source is STRIDED - the
    /// batched buffer is `[t, 3d]` rows and the cache wants a contiguous
    /// `[cap, d]` - so `kv_cache_fill`'s flat prefix copy does not apply. It is
    /// once per request, not per token, against a forward that just ran 24 MoE
    /// layers over the same rows.
    pub fn fill_kv(&self, g: &Gpu, sc: &BlockScratch, kv: &KvCache, n: u32) {
        let d = self.d as usize;
        let src = if self.tau { &sc.qkv2 } else { &sc.qkv };
        let rows = g.read(src, (n as usize) * 3 * d);
        let mut kbuf = Vec::with_capacity(n as usize * d);
        let mut vbuf = Vec::with_capacity(n as usize * d);
        for r in 0..n as usize {
            kbuf.extend_from_slice(&rows[r * 3 * d + d..r * 3 * d + 2 * d]);
            vbuf.extend_from_slice(&rows[r * 3 * d + 2 * d..(r + 1) * 3 * d]);
        }
        g.write_f32(&kv.k, &kbuf);
        g.write_f32(&kv.v, &vbuf);
    }

    /// ONE incremental decode step: a single row `x_row` at absolute position
    /// `pos`, attending over this layer's KV cache.
    ///
    /// `O(pos)` per token instead of the batched forward's `O(pos²)` over the
    /// whole grown sequence - the same recompute/incremental pair `crates/gpt2`,
    /// `crates/qwen3` and `crates/deepseek2` each keep.
    ///
    /// The three things that differ from the batched path, and why:
    ///
    /// * **No mask.** A decode row is at `pos >= prefix_attn`, so its mask row
    ///   is purely causal, and `gqa_decode_step` reads cache rows `0..=pos` and
    ///   no others. The bidirectional image prefix is still honoured - it is
    ///   baked into the K/V the PROMPT's masked forward produced.
    /// * **`rope_partial_at`, not `rope_partial`.** The latter takes its
    ///   position from the row index, so a one-row call is always position 0.
    /// * **The tau scale is computed for this row alone**, from this row's own
    ///   `tok_feat`, with `tau_pos` folded on the host at `pos` - the batched
    ///   path does the same per row, just `t` of them at once.
    pub fn decode_step<'a>(&self, g: &Gpu, kv: &'a KvCache, x_row: &DeviceBuffer, pos: u32) -> &'a DeviceBuffer {
        let (d, nh, hd, ff) = (self.d, self.n_heads, self.head_dim, self.ff);
        let stride3 = 3 * d;
        let mut s: Vec<Step> = Vec::new();

        s.push(g.step(4, &[x_row, self.wb("ln.weight"), self.wb("ln.bias"), &kv.l_in], &[d, 1, f(LN_EPS)], 1));
        s.push(g.step(0, &[&kv.l_in, self.wb("attn.qkv.weight"), &kv.qkv], &[1, d, stride3], stride3));
        if self.qkv_bias {
            s.push(g.step(6, &[&kv.qkv, self.wb("attn.qkv.bias")], &[1, stride3], stride3));
        }
        let qkv = if self.tau {
            s.push(g.step(16, &[&kv.qkv, &kv.tok_feat], &[stride3], stride3));
            s.push(g.step(0, &[&kv.tok_feat, self.wb("attn.tau.wq"), &kv.tqr], &[1, stride3, nh], nh));
            s.push(g.step(0, &[&kv.tok_feat, self.wb("attn.tau.wv"), &kv.tvr], &[1, stride3, nh], nh));
            g.submit(&[], &s);
            s = Vec::new();
            let tqr = g.read(&kv.tqr, nh as usize);
            let tvr = g.read(&kv.tvr, nh as usize);
            let alpha = g.read(self.wb("attn.tau.alpha"), nh as usize);
            // `tau_pos` uses the ABSOLUTE position, matching the batched path's
            // `row + 1` at row = pos.
            let mut s3 = vec![1.0f32; (3 * nh) as usize];
            for h in 0..nh as usize {
                let tau_pos = 0.5 + sigmoid(alpha[h] * ((pos + 1) as f32).ln());
                s3[h] = tqr[h].tanh() + tau_pos;
                s3[2 * nh as usize + h] = tvr[h].tanh() + tau_pos;
            }
            g.write(&kv.s3, &s3.iter().map(|&x| f(x)).collect::<Vec<u32>>());
            s.push(g.step(17, &[&kv.qkv, &kv.s3, &kv.qkv2], &[1, 3 * nh, hd], stride3));
            &kv.qkv2
        } else {
            &kv.qkv
        };
        // Partial RoPE at the ABSOLUTE position, on q (offset 0) and k (offset d).
        let half = self.rot_dim / 2;
        s.push(g.step(K_ROPE_PARTIAL_AT, &[qkv], &[1, nh, hd, stride3, 0, pos, f(self.theta), self.rot_dim], nh * half));
        s.push(g.step(K_ROPE_PARTIAL_AT, &[qkv], &[1, nh, hd, stride3, d, pos, f(self.theta), self.rot_dim], nh * half));
        g.submit(&[], &s);

        // Split the fused row into the three contiguous buffers
        // `gqa_decode_step` binds. This round-trips through the host, and that
        // is a KNOWN cost rather than an oversight: the shared primitive takes
        // q/k/v as separate buffers, so binding sub-ranges of the fused row
        // would mean either duplicating its four dispatches here with
        // `Gpu::step_sliced` (against the one-implementation rule) or widening
        // `model::block`'s decode API for one caller.
        //
        // Left as-is deliberately. This layer ALREADY syncs once per token when
        // tau is on (the `tqr`/`tvr` readback above, which the batched path
        // does too), so the split adds a second sync rather than a first, and
        // no machine here can run the real weights to say what either costs.
        // Measure before restructuring - on this engine the profile has been
        // right and the confident hypothesis has not.
        let row = g.read(qkv, stride3 as usize);
        g.write_f32(&kv.q, &row[..d as usize]);
        g.write_f32(&kv.k_new, &row[d as usize..2 * d as usize]);
        g.write_f32(&kv.v_new, &row[2 * d as usize..]);

        let steps = model::block::gqa_decode_step(
            g,
            &decode_ids(),
            nh,
            nh, // full MHA: n_kv_heads == n_heads, so the shared GQA primitive runs at group = 1
            hd,
            pos,
            kv.cap,
            &kv.q,
            &kv.k_new,
            &kv.v_new,
            &kv.k,
            &kv.v,
            &kv.scores,
            &kv.probs,
            &kv.ctx,
        );
        g.submit(&[], &steps);

        let mut s: Vec<Step> = vec![
            g.step(0, &[&kv.ctx, self.wb("attn.proj.weight"), &kv.l_attn], &[1, d, d], d),
            g.step(6, &[&kv.l_attn, self.wb("attn.proj.bias")], &[1, d], d),
        ];
        g.submit(&[], &s);

        // FFN on the SAME l_in, one row.
        let l_mlp: &DeviceBuffer = if let Some(moe) = &self.moe {
            moe.forward_rows(g, &kv.l_in, 1)
        } else if let Some(moe) = &self.moe8 {
            moe.forward_rows(g, &kv.l_in, 1)
        } else {
            s = vec![
                g.step(0, &[&kv.l_in, self.wb("mlp.fc1.weight"), &kv.h], &[1, d, ff], ff),
                g.step(6, &[&kv.h, self.wb("mlp.fc1.bias")], &[1, ff], ff),
                g.step(5, &[&kv.h, &kv.h2], &[ff], ff),
                g.step(0, &[&kv.h2, self.wb("mlp.fc2.weight"), &kv.l_mlp], &[1, ff, d], d),
                g.step(6, &[&kv.l_mlp, self.wb("mlp.fc2.bias")], &[1, d], d),
            ];
            g.submit(&[], &s);
            &kv.l_mlp
        };
        // 3-way residual, as the batched path.
        g.submit(
            &[],
            &[g.step(7, &[x_row, &kv.l_attn, &kv.x1], &[d], d), g.step(7, &[&kv.x1, l_mlp, &kv.out], &[d], d)],
        );
        &kv.out
    }

    pub fn backward(&self, g: &Gpu, x: &DeviceBuffer, d_out: &DeviceBuffer, gr: &MoondreamBlockGrads, d_x_in: &DeviceBuffer) {
        // An int8 block has no fp32 expert weights to recompute from, and the
        // dense-FFN arm below would happily run and produce a gradient for an
        // FFN this block does not have. Refuse by name instead.
        assert!(self.moe8.is_none(), "MoondreamBlock::backward: this block was built with the int8 expert tier, which is inference-only");
        // The block's OWN activations are the backprop cache - a shared
        // `BlockScratch` would by now hold the LAST block's values, so this
        // path deliberately cannot reach one (see `BlockScratch`'s doc).
        let sc = self.own.as_ref().expect("MoondreamBlock::backward: this block shares a BlockScratch, so its forward activations are gone");
        let (t, d, nh, hd, ff) = (self.t, self.d, self.n_heads, self.head_dim, self.ff);
        let stride3 = 3 * d;
        let half = self.rot_dim / 2;
        // The attention ran on qkv2 (post-tau) when tau is on, else the raw qkv.
        let attn_qkv = if self.tau { &sc.qkv2 } else { &sc.qkv };
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
            moe.backward(g, &sc.l_in, d_out, gr.moe.as_ref().expect("moe grads required for a MoE block"), &d_ln);
        } else {
            s.push(g.step(K_MATMUL_DX, &[d_out, self.wb("mlp.fc2.weight"), &d_h2], &[t, ff, d, 0], t * ff));
            s.push(g.step(K_MATMUL_DW, &[d_out, &sc.h2, &gr.fc2_w], &[t, ff, d], d * ff));
            s.push(g.step(K_BIAS_GRAD, &[d_out, &gr.fc2_b], &[t, d], d));
            s.push(g.step(K_GELU_BWD, &[&sc.h, &d_h2, &d_h], &[t * ff], t * ff)); // tanh gelu
            s.push(g.step(K_MATMUL_DX, &[&d_h, self.wb("mlp.fc1.weight"), &d_ln], &[t, d, ff, 0], t * d));
            s.push(g.step(K_MATMUL_DW, &[&d_h, &sc.l_in, &gr.fc1_w], &[t, d, ff], ff * d));
            s.push(g.step(K_BIAS_GRAD, &[&d_h, &gr.fc1_b], &[t, ff], ff));
        }

        // --- attention branch → d_qkv (grad of attn_qkv, pre-RoPE) ---
        s.push(g.step(K_MATMUL_DX, &[d_out, self.wb("attn.proj.weight"), &d_ctx], &[t, d, d, 0], t * d));
        s.push(g.step(K_MATMUL_DW, &[d_out, &sc.ctx, &gr.proj_w], &[t, d, d], d * d));
        s.push(g.step(K_BIAS_GRAD, &[d_out, &gr.proj_b], &[t, d], d));
        s.push(g.step(K_ATTN_DSCORES, &[&d_ctx, attn_qkv, &sc.probs, &dscores], &[1, nh, t, t, hd, stride3, 2 * d, d], nh * t * t));
        s.push(g.step(K_ATTN_DV, &[&sc.probs, &d_ctx, &d_qkv], &[1, nh, t, t, hd, stride3, 2 * d, d], nh * t * hd));
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
            s.push(g.step(K_TAU_SCALE_DS, &[&d_qkv, &sc.qkv, &d_s3], &[t, 3 * nh, hd], 3 * nh * t));
            g.submit(&[], &s);
            s = Vec::new();
            // Host: tanh′ and the alpha (tau_pos) grad; positions = row index.
            let ds = g.read(&d_s3, (3 * nh * t) as usize);
            let tqr = g.read(&sc.tqr, (t * nh) as usize);
            let tvr = g.read(&sc.tvr, (t * nh) as usize);
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
            s.push(g.step(17, &[&d_qkv, &sc.s3, &d_qkv_in], &[t, 3 * nh, hd], t * stride3));
            // wq/wv weight grads + d_tok_feat = d_tqr·wq + d_tvr·wv.
            s.push(g.step(K_MATMUL_DW, &[&dtqr_b, &sc.tok_feat, &tg.wq], &[t, stride3, nh], nh * stride3));
            s.push(g.step(K_MATMUL_DX, &[&dtqr_b, self.wb("attn.tau.wq"), &d_tok_feat], &[t, stride3, nh, 0], t * stride3));
            s.push(g.step(K_MATMUL_DW, &[&dtvr_b, &sc.tok_feat, &tg.wv], &[t, stride3, nh], nh * stride3));
            s.push(g.step(K_MATMUL_DX, &[&dtvr_b, self.wb("attn.tau.wv"), &d_tok_feat], &[t, stride3, nh, 1], t * stride3)); // accumulate
            // tok_feat = gelu_erf(qkv_raw): d_qraw2 = d_tok_feat · gelu_erf′(qkv_raw).
            s.push(g.step(K_GELU_ERF_BWD, &[&sc.qkv, &d_tok_feat, &d_qraw2], &[t * stride3], t * stride3));
            // Total raw-qkv grad = tau_scale in-grad + tok_feat path.
            s.push(g.step(7, &[&d_qkv_in, &d_qraw2, &d_qkv_raw], &[t * stride3], t * stride3));
            d_qkv_raw
        } else {
            d_qkv
        };
        s.push(g.step(K_MATMUL_DX, &[&d_qkv_raw, self.wb("attn.qkv.weight"), &d_ln], &[t, d, stride3, 1], t * d)); // accumulate
        s.push(g.step(K_MATMUL_DW, &[&d_qkv_raw, &sc.l_in, &gr.qkv_w], &[t, d, stride3], stride3 * d));
        // Fused-qkv bias grad = Σ_rows d_qkv_raw (bias is additive on the raw qkv).
        if let Some(qb) = &gr.qkv_b {
            s.push(g.step(K_BIAS_GRAD, &[&d_qkv_raw, qb], &[t, stride3], stride3));
        }

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
    /// Present when the block has a fused-qkv bias (`attn.qkv.bias`); grad `[3·d]`.
    pub qkv_b: Option<DeviceBuffer>,
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
            qkv_b: None,
        }
    }

    /// Add zeroed tau grads (`wq`/`wv` `[nh, 3·d]`, `alpha` `[nh]`) for a tau block.
    pub fn with_tau(mut self, g: &Gpu, nh: u32, d: u32) -> MoondreamBlockGrads {
        let z = |n: u32| g.storage_init("md.tg", &vec![0.0f32; n as usize]);
        self.tau = Some(TauGrads { wq: z(nh * 3 * d), wv: z(nh * 3 * d), alpha: z(nh) });
        self
    }

    /// Add a zeroed fused-qkv bias grad `[3·d]` for a block with a qkv bias.
    pub fn with_qkv_bias(mut self, g: &Gpu, d: u32) -> MoondreamBlockGrads {
        self.qkv_b = Some(g.storage_init("md.qb", &vec![0.0f32; (3 * d) as usize]));
        self
    }

    /// Add zeroed MoE grads (router + `e` experts) for a MoE block.
    pub fn with_moe(mut self, g: &Gpu, d: u32, inner: u32, e: u32) -> MoondreamBlockGrads {
        self.moe = Some(MoeGrads::new(g, d, inner, e));
        self
    }
}

/// One expert's three linears, int8-packed: `[n, k/4]` u32 words plus the
/// `[n, k/32]` f32 group scale [`model::int8::quantize_weight`] wrote.
struct Expert8 {
    w_h: (DeviceBuffer, DeviceBuffer),
    w_g: (DeviceBuffer, DeviceBuffer),
    w_down: (DeviceBuffer, DeviceBuffer),
}

/// [`MoeFfn`]'s int8 twin: the same graph, with the EXPERT weights stored
/// per-channel-quantized and dispatched through `moe_linear_gated_i8`.
///
/// # Why this model needed one
///
/// At the released config the decoder is 20 MoE layers of 64 experts, each
/// expert three `[1024, 2048]`-ish matrices - 8.05 B parameters, ~32 GiB of
/// fp32, on top of ~10 GiB of activation scratch. There is no machine that
/// runs. int8 weights take the parameter half to ~8.2 GiB, which is what makes
/// the rest of the serving surface worth having.
///
/// # Why it is a separate type rather than a mode on [`MoeFfn`]
///
/// `MoeFfn` is the TRAINING path: its `backward` recomputes each expert's
/// forward from the same fp32 weights the forward read, and a quantized weight
/// has no gradient of its own. Bolting a precision flag onto it would put a
/// branch inside the differentiated code for a tier that is never
/// differentiated. So this is inference-only, and says so.
///
/// # Two things it does that the fp32 tier does not
///
/// * **The input is quantized ONCE per layer**, not once per expert - every
///   expert reads the same `xq`/`sx`. `model::moe::expert_fwd_i8` makes the
///   same point for the SwiGLU shape.
/// * **Rows this expert is not routed to are skipped**, because
///   `moe_linear_gated_i8` takes the gate and returns early on a zero row.
///   With `top_k = 8` of 64 that is 56 of every 64 experts' row work the fp32 tier
///   (which evaluates every expert densely and gates afterwards) still does.
///   The kernel is the naive one-thread-per-output tier deliberately - a tiled
///   kernel stages rows across a `workgroupBarrier()`, which a per-row early
///   return would make undefined; see its own header.
///
/// The router stays fp32: it is one `[e, d]` matrix, and quantizing the thing
/// that DECIDES the routing to save 6 MB would be a poor trade.
pub struct MoeFfn8 {
    router: DeviceBuffer,
    experts: Vec<Expert8>,
    e: u32,
    top_k: u32,
    d: u32,
    inner: u32,
    t: u32,
    // scratch
    logits: DeviceBuffer,
    gate: DeviceBuffer,
    /// The layer input, quantized once and shared by every expert.
    xq: DeviceBuffer,
    sx: DeviceBuffer,
    h: DeviceBuffer,
    g: DeviceBuffer,
    act: DeviceBuffer,
    /// `act` quantized - a different tensor per expert, so re-quantized each time.
    aq: DeviceBuffer,
    sa: DeviceBuffer,
    eout: DeviceBuffer,
    acc: DeviceBuffer,
}

impl MoeFfn8 {
    /// Quantize `weights` (the same fp32 map [`MoeFfn::new`] takes) and upload
    /// the packed form. `d` and `inner` must both be multiples of
    /// `model::int8::GROUP` (32) - the weight scale is per 32-element group of
    /// the contraction axis, which `quantize_weight` asserts.
    pub fn new(gpu: &Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, inner: u32, e: u32, top_k: u32) -> MoeFfn8 {
        let get = |n: &str| weights.get(n).unwrap_or_else(|| panic!("moe weight missing: {n}"));
        let pack = |name: &str, n: u32, k: u32| -> (DeviceBuffer, DeviceBuffer) {
            let (packed, sw) = model::int8::quantize_weight(get(name), n as usize, k as usize);
            let pb = gpu.storage(packed.len() as u64);
            gpu.write(&pb, &packed);
            // Reclaim staging between weights: a 64-expert layer packs 192
            // tensors, and `qwen3::q8` learned the same lesson (see paramstore).
            gpu.poll_wait();
            let sb = gpu.storage(sw.len() as u64);
            gpu.write_f32(&sb, &sw);
            gpu.poll_wait();
            (pb, sb)
        };
        let experts = (0..e)
            .map(|ei| Expert8 {
                w_h: pack(&format!("experts.{ei}.w_h.weight"), inner, d),
                w_g: pack(&format!("experts.{ei}.w_g.weight"), inner, d),
                w_down: pack(&format!("experts.{ei}.w_down.weight"), d, inner),
            })
            .collect();
        MoeFfn8 {
            router: gpu.storage_init("router.weight", get("router.weight")),
            experts,
            e,
            top_k,
            d,
            inner,
            t,
            logits: gpu.storage((t * e) as u64),
            gate: gpu.storage((t * e) as u64),
            xq: gpu.storage((t * d / 4) as u64),
            sx: gpu.storage(t as u64),
            h: gpu.storage((t * inner) as u64),
            g: gpu.storage((t * inner) as u64),
            act: gpu.storage((t * inner) as u64),
            aq: gpu.storage((t * inner / 4) as u64),
            sa: gpu.storage(t as u64),
            eout: gpu.storage((t * d) as u64),
            acc: gpu.storage((t * d) as u64),
        }
    }

    /// The mixed expert output `[t, d]`. Same contract as [`MoeFfn::forward`].
    pub fn forward(&self, g: &Gpu, xn: &DeviceBuffer) -> &DeviceBuffer {
        self.forward_rows(g, xn, self.t)
    }

    /// [`Self::forward`] over the FIRST `rows` rows only - what an incremental
    /// decode step needs (`rows = 1`).
    ///
    /// The scratch stays sized for the built `t`; every kernel here takes the
    /// row count as its `m` parameter, so a short run simply touches the first
    /// `rows` rows of buffers that are larger than it needs. Sizing a second
    /// set for decode would mean a second copy of the expert weights, which is
    /// the one thing this tier exists to avoid.
    pub fn forward_rows(&self, g: &Gpu, xn: &DeviceBuffer, rows: u32) -> &DeviceBuffer {
        let (t, d, inner, e) = (rows, self.d, self.inner, self.e);
        assert!(rows <= self.t, "MoeFfn8: {rows} rows exceeds the built context {}", self.t);
        let mut s: Vec<Step> = Vec::new();
        // Router in fp32, exactly as the fp32 tier does it.
        s.push(g.step(0, &[xn, &self.router, &self.logits], &[t, d, e], t * e));
        s.push(g.step(1, &[&self.logits, &self.gate], &[t, e, self.top_k, 1, f(1.0)], t));
        // ONE activation quantization for the whole layer - every expert reads
        // the same `xq`/`sx`, so quantizing per expert would be 64x waste.
        for st in model::int8::quant_rows_steps(
            g,
            model::int8::QuantRows { kernels: K_QUANT, x: xn, sx: &self.sx, xq: &self.xq },
            0,
            t,
            d,
        ) {
            s.push(st);
        }
        // `moe_linear_gated_i8` Params: [m, kg, n, n_experts, e_idx];
        // bufs [xq, wq, sx, sw, gate, out].
        let lin = |s: &mut Vec<Step>, xq: &DeviceBuffer, sx: &DeviceBuffer, w: &(DeviceBuffer, DeviceBuffer), out: &DeviceBuffer, kg: u32, n: u32, ei: u32| {
            s.push(g.step(K_MOE_LIN_I8, &[xq, &w.0, sx, &w.1, &self.gate, out], &[t, kg, n, e, ei], t * n));
        };
        for ei in 0..e {
            let ex = &self.experts[ei as usize];
            lin(&mut s, &self.xq, &self.sx, &ex.w_h, &self.h, d / 4, inner, ei);
            lin(&mut s, &self.xq, &self.sx, &ex.w_g, &self.g, d / 4, inner, ei);
            s.push(g.step(2, &[&self.h, &self.g, &self.act], &[t * inner], t * inner)); // gelu(h)·(g+1)
            // `act` is this expert's own tensor, so it needs its own scale.
            for st in model::int8::quant_rows_steps(
                g,
                model::int8::QuantRows { kernels: K_QUANT, x: &self.act, sx: &self.sa, xq: &self.aq },
                0,
                t,
                inner,
            ) {
                s.push(st);
            }
            lin(&mut s, &self.aq, &self.sa, &ex.w_down, &self.eout, inner / 4, d, ei);
            let acc = if ei == 0 { 0u32 } else { 1u32 };
            s.push(g.step(3, &[&self.gate, &self.eout, &self.acc], &[t, d, e, ei, acc], t * d));
        }
        g.submit(&[], &s);
        &self.acc
    }

    /// Number of output elements (`t·d`).
    pub fn numel(&self) -> usize {
        (self.t * self.d) as usize
    }
}

/// Sparse-MoE FFN: `router → for each expert (w_h, w_g → geglu_shift → w_down) →
/// gate-weighted accumulate`. Returns the mixed output `[t, d]` (no residual - the
/// parallel block owns the 3-way residual). Weight keys: `router.weight` `[e, d]`,
/// and per expert `experts.{e}.{w_h,w_g}.weight` `[inner, d]`, `w_down.weight`
/// `[d, inner]`.
pub struct MoeFfn {
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

impl MoeFfn {
    pub fn new(gpu: &Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, inner: u32, e: u32, top_k: u32) -> MoeFfn {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        MoeFfn {
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
    pub fn forward(&self, g: &Gpu, xn: &DeviceBuffer) -> &DeviceBuffer {
        self.forward_rows(g, xn, self.t)
    }

    /// [`Self::forward`] over the FIRST `rows` rows only - see
    /// [`MoeFfn8::forward_rows`] for why the scratch stays full-sized.
    pub fn forward_rows(&self, g: &Gpu, xn: &DeviceBuffer, rows: u32) -> &DeviceBuffer {
        let (t, d, inner, e) = (rows, self.d, self.inner, self.e);
        assert!(rows <= self.t, "MoeFfn: {rows} rows exceeds the built context {}", self.t);
        let mut s: Vec<Step> = Vec::new();
        // Router: logits = xn·router.weight^T, then top-k softmax gate.
        s.push(g.step(0, &[xn, self.wb("router.weight"), &self.logits], &[t, d, e], t * e));
        // norm=1, scale=1.0: `router_gate.wgsl`'s renormalised top-k gate
        // (Moondream 3's router), spelled rather than implied.
        s.push(g.step(1, &[&self.logits, &self.gate], &[t, e, self.top_k, 1, f(1.0)], t));
        for ei in 0..e {
            let ep = |leaf: &str| self.wb(&format!("experts.{ei}.{leaf}"));
            s.push(g.step(0, &[xn, ep("w_h.weight"), &self.h], &[t, d, inner], t * inner));
            s.push(g.step(0, &[xn, ep("w_g.weight"), &self.g], &[t, d, inner], t * inner));
            s.push(g.step(2, &[&self.h, &self.g, &self.act], &[t * inner], t * inner)); // gelu(h)·(g+1)
            s.push(g.step(0, &[&self.act, ep("w_down.weight"), &self.eout], &[t, inner, d], t * d));
            let acc = if ei == 0 { 0u32 } else { 1u32 };
            s.push(g.step(3, &[&self.gate, &self.eout, &self.acc], &[t, d, e, ei, acc], t * d));
        }
        g.submit(&[], &s);
        &self.acc
    }
    /// Number of output elements (`t·d`).
    pub fn numel(&self) -> usize {
        (self.t * self.d) as usize
    }

    /// MoE backward: from the mixed-output grad `d_out` (= the block's `d_out`, since
    /// the MoE output is a residual branch), fill `gr` and write the input grad into
    /// `d_xn` (the shared-LN branch grad - its first write overwrites, so `d_xn` need
    /// not be pre-zeroed). Per expert the forward is recomputed (the scratch isn't
    /// cached per-expert), then: combine bwd (`scale_add_dexp/dgate`) → `w_down` →
    /// `geglu_shift_da/db` → `w_h`/`w_g`. Finally the router (`router_bwd`, no aux/z
    /// loss) → `router.weight`. Assumes `forward` ran (for `logits`/`gate`).
    pub fn backward(&self, g: &Gpu, xn: &DeviceBuffer, d_out: &DeviceBuffer, gr: &MoeGrads, d_xn: &DeviceBuffer) {
        let (t, d, inner, e) = (self.t, self.d, self.inner, self.e);
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
                g.step(K_ROUTER_BWD, &[&self.logits, &self.gate, &d_gate, &fe, &d_logits], &[t, e, self.top_k, 0, f(0.0), f(0.0), 1, f(1.0)], t),
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
        let gpu = gpu_core::testgpu::dev(pipelines());
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
        let out = gpu.read(blk.forward(&gpu, &x), blk.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn tau_temperature_changes_block_output() {
        // A block with attn.tau.* present applies per-head temperature to q,v and
        // must differ from the same weights without tau.
        let gpu = gpu_core::testgpu::dev(pipelines());
        let (t, d, nh, hd, ff, prefix, rot) = (6u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32);
        let mut rng = Rng::new(11);
        let base = block_weights(d, ff, &mut rng);
        let x: Vec<f32> = (0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect();

        let blk = MoondreamBlock::new(&gpu, &base, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let xb = gpu.storage_init("x", &x);
        assert!(!blk.tau);
        let plain = gpu.read(blk.forward(&gpu, &xb), blk.numel());

        let mut tw = base.clone();
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        tw.insert("attn.tau.wq".into(), r((nh * 3 * d) as usize));
        tw.insert("attn.tau.wv".into(), r((nh * 3 * d) as usize));
        tw.insert("attn.tau.alpha".into(), r(nh as usize));
        let tblk = MoondreamBlock::new(&gpu, &tw, t, d, nh, hd, ff, prefix, rot, 1.5e6);
        let xb2 = gpu.storage_init("x2", &x);
        assert!(tblk.tau);
        let tau = gpu.read(tblk.forward(&gpu, &xb2), tblk.numel());

        assert!(tau.iter().all(|v| v.is_finite()));
        let diff: f32 = plain.iter().zip(&tau).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "tau must change the block output, Σ|Δ|={diff}");
    }

    #[test]
    fn qkv_bias_block_backward_matches_finite_diff() {
        // With the real checkpoint's fused-qkv bias present, the block applies it
        // (before tau/RoPE) and its grad = Σ_rows d_qkv_raw. Gradcheck the bias grad.
        let gpu = gpu_core::testgpu::dev(pipelines());
        let (t, d, nh, hd, ff, prefix, rot, theta) = (5u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32, 1.5e6f32);
        let mut rng = Rng::new(23);
        let mut w = block_weights(d, ff, &mut rng);
        w.insert("attn.qkv.bias".into(), (0..(3 * d) as usize).map(|_| (rng.next_f32() - 0.5) * 0.3).collect());
        let n = (t * d) as usize;
        let x_host: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();

        let blk = MoondreamBlock::new(&gpu, &w, t, d, nh, hd, ff, prefix, rot, theta);
        assert!(blk.qkv_bias);
        let xb = gpu.storage_init("x", &x_host);
        let _ = blk.forward(&gpu, &xb);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = MoondreamBlockGrads::new(&gpu, d, ff).with_qkv_bias(&gpu, d);
        let d_x = gpu.storage((t * d) as u64);
        blk.backward(&gpu, &xb, &d_out, &gr, &d_x);
        let g_qb = gpu.read(gr.qkv_b.as_ref().unwrap(), (3 * d) as usize);

        let loss = |wm: &HashMap<String, Vec<f32>>| -> f32 {
            let b = MoondreamBlock::new(&gpu, wm, t, d, nh, hd, ff, prefix, rot, theta);
            gpu.read(b.forward(&gpu, &gpu.storage_init("x", &x_host)), n).iter().sum::<f32>()
        };
        let eps = 1e-3f32;
        for &j in &[0usize, 17, 40] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("attn.qkv.bias").unwrap()[j] += eps;
            wm2.get_mut("attn.qkv.bias").unwrap()[j] -= eps;
            let num = (loss(&wp) - loss(&wm2)) / (2.0 * eps);
            assert!((g_qb[j] - num).abs() <= 4e-3 + 8e-2 * num.abs(), "d qkv.bias[{j}]: analytic {} vs numeric {}", g_qb[j], num);
        }
    }

    #[test]
    fn dense_block_backward_matches_finite_diff() {
        // Directional finite-diff gradcheck of the dense parallel-block backward:
        // the input grad exercises the whole reverse chain (residual → MLP+attn →
        // shared-LN accumulation → LN dx); a weight grad covers the accumulating path.
        let gpu = gpu_core::testgpu::dev(pipelines());
        let (t, d, nh, hd, ff, prefix, rot, theta) = (5u32, 16u32, 2u32, 8u32, 32u32, 3u32, 4u32, 1.5e6f32);
        let mut rng = Rng::new(7);
        let w = block_weights(d, ff, &mut rng);
        let n = (t * d) as usize;
        let x_host: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();

        // Analytic grads (forward populates the SSA cache, then backward).
        let blk = MoondreamBlock::new(&gpu, &w, t, d, nh, hd, ff, prefix, rot, theta);
        let xb = gpu.storage_init("x", &x_host);
        let _ = blk.forward(&gpu, &xb);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = MoondreamBlockGrads::new(&gpu, d, ff);
        let d_x = gpu.storage((t * d) as u64);
        blk.backward(&gpu, &xb, &d_out, &gr, &d_x);
        let dx = gpu.read(&d_x, n);
        let g_ln_w = gpu.read(&gr.ln_w, d as usize);

        // L(w, x) = Σ block.forward(x) (matches d_out = ones).
        let loss = |wm: &HashMap<String, Vec<f32>>, xh: &[f32]| -> f32 {
            let b = MoondreamBlock::new(&gpu, wm, t, d, nh, hd, ff, prefix, rot, theta);
            let xbb = gpu.storage_init("x", xh);
            gpu.read(b.forward(&gpu, &xbb), n).iter().sum::<f32>()
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
        let gpu = gpu_core::testgpu::dev(pipelines());
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
        let _ = blk.forward(&gpu, &xb);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = MoondreamBlockGrads::new(&gpu, d, ff).with_tau(&gpu, nh, d);
        let d_x = gpu.storage((t * d) as u64);
        blk.backward(&gpu, &xb, &d_out, &gr, &d_x);
        let dx = gpu.read(&d_x, n);
        let tg = gr.tau.as_ref().unwrap();
        let g_alpha = gpu.read(&tg.alpha, nh as usize);
        let g_wq = gpu.read(&tg.wq, (nh * 3 * d) as usize);

        let loss = |wm: &HashMap<String, Vec<f32>>, xh: &[f32]| -> f32 {
            let b = MoondreamBlock::new(&gpu, wm, t, d, nh, hd, ff, prefix, rot, theta);
            let xbb = gpu.storage_init("x", xh);
            gpu.read(b.forward(&gpu, &xbb), n).iter().sum::<f32>()
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
        let gpu = gpu_core::testgpu::dev(pipelines());
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
        let _ = blk.forward(&gpu, &xb);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = MoondreamBlockGrads::new(&gpu, d, ff).with_moe(&gpu, d, inner, e);
        let d_x = gpu.storage((t * d) as u64);
        blk.backward(&gpu, &xb, &d_out, &gr, &d_x);
        let dx = gpu.read(&d_x, n);
        let mg = gr.moe.as_ref().unwrap();
        let g_wh0 = gpu.read(&mg.experts[0].w_h, (inner * d) as usize);
        let g_router = gpu.read(&mg.router, (e * d) as usize);

        let loss = |mwm: &HashMap<String, Vec<f32>>, xh: &[f32]| -> f32 {
            let b = build(&bw, mwm);
            let xbb = gpu.storage_init("x", xh);
            gpu.read(b.forward(&gpu, &xbb), n).iter().sum::<f32>()
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
        let gpu = gpu_core::testgpu::dev(pipelines());
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
        let _ = dec.forward(&gpu, &tokens, &targets, &img);
        let gr = MoondreamDecoderGrads::new(&gpu, nl, d, ff, vocab, n_img);
        dec.backward(&gpu, &targets, &gr);
        let d_img = gpu.read(&gr.d_image_embeds, (n_img * d) as usize);
        let g_lmb = gpu.read(&gr.lm_head_b, vocab as usize);
        let g_tok = gpu.read(&gr.tok_w, (vocab * d) as usize);

        let loss = |wm: &HashMap<String, Vec<f32>>, im: &[f32]| -> f32 { build(wm).forward(&gpu, &tokens, &targets, im) };
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

    #[test]
    fn check_moondream_full_decoder_backward() {
        // End-to-end gradcheck of the REAL architecture: layer 0 = tau + dense FFN,
        // layer 1 = tau + MoE FFN. Proves the decoder backward composes the tau and
        // MoE block backwards through the residual-stream chain + splice + head.
        let gpu = gpu_core::testgpu::dev(pipelines());
        let (t, d, nh, hd, ff, vocab, n_img, prefix, rot, theta) = (7u32, 16u32, 2u32, 8u32, 32u32, 19u32, 3u32, 4u32, 4u32, 1.5e6f32);
        let (inner, e, top_k) = (6u32, 3u32, 2u32);
        let mut rng = Rng::new(41);
        let mut w = HashMap::new();
        {
            let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.3).collect::<Vec<f32>>();
            w.insert("tok.weight".to_string(), r((vocab * d) as usize));
            w.insert("post_ln.weight".to_string(), vec![1.0; d as usize]);
            w.insert("post_ln.bias".to_string(), r(d as usize));
            w.insert("lm_head.weight".to_string(), r((vocab * d) as usize));
            w.insert("lm_head.bias".to_string(), r(vocab as usize));
        }
        for l in 0..2u32 {
            for (k, v) in block_weights(d, ff, &mut rng) {
                w.insert(format!("blocks.{l}.{k}"), v);
            }
            let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
            w.insert(format!("blocks.{l}.attn.tau.wq"), r((nh * 3 * d) as usize));
            w.insert(format!("blocks.{l}.attn.tau.wv"), r((nh * 3 * d) as usize));
            w.insert(format!("blocks.{l}.attn.tau.alpha"), r(nh as usize));
        }
        // Layer 1 is MoE.
        {
            let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect::<Vec<f32>>();
            w.insert("blocks.1.moe.router.weight".to_string(), r((e * d) as usize));
            for ei in 0..e {
                w.insert(format!("blocks.1.moe.experts.{ei}.w_h.weight"), r((inner * d) as usize));
                w.insert(format!("blocks.1.moe.experts.{ei}.w_g.weight"), r((inner * d) as usize));
                w.insert(format!("blocks.1.moe.experts.{ei}.w_down.weight"), r((d * inner) as usize));
            }
        }
        let img: Vec<f32> = (0..(n_img * d) as usize).map(|_| (rng.next_f32() - 0.5) * 0.3).collect();
        let tokens = vec![0u32, 5, 5, 5, 7, 9, 11];
        let mut targets = vec![5u32, 0, 0, 0, 9, 11, 13];
        for tg in targets.iter_mut().take(1 + n_img as usize).skip(1) {
            *tg = IGNORE;
        }

        let sub = |wm: &HashMap<String, Vec<f32>>, pre: &str| -> HashMap<String, Vec<f32>> { wm.iter().filter_map(|(k, v)| k.strip_prefix(pre).map(|s| (s.to_string(), v.clone()))).collect() };
        let build = |wm: &HashMap<String, Vec<f32>>| -> MoondreamDecoder {
            let b0w: HashMap<String, Vec<f32>> = sub(wm, "blocks.0.").into_iter().filter(|(k, _)| !k.starts_with("moe.")).collect();
            let b1w: HashMap<String, Vec<f32>> = sub(wm, "blocks.1.").into_iter().filter(|(k, _)| !k.starts_with("moe.")).collect();
            let moe = MoeFfn::new(&gpu, &sub(wm, "blocks.1.moe."), t, d, inner, e, top_k);
            let blocks = vec![
                MoondreamBlock::new(&gpu, &b0w, t, d, nh, hd, ff, prefix, rot, theta),
                MoondreamBlock::new(&gpu, &b1w, t, d, nh, hd, ff, prefix, rot, theta).with_moe(moe),
            ];
            MoondreamDecoder::new(&gpu, wm, blocks, t, d, vocab, n_img)
        };

        let dec = build(&w);
        let _ = dec.forward(&gpu, &tokens, &targets, &img);
        let grads = MoondreamDecoderGrads::from_blocks(
            &gpu,
            vec![
                MoondreamBlockGrads::new(&gpu, d, ff).with_tau(&gpu, nh, d),
                MoondreamBlockGrads::new(&gpu, d, ff).with_tau(&gpu, nh, d).with_moe(&gpu, d, inner, e),
            ],
            d,
            vocab,
            n_img,
        );
        dec.backward(&gpu, &targets, &grads);
        let d_img = gpu.read(&grads.d_image_embeds, (n_img * d) as usize);
        let g_alpha0 = gpu.read(&grads.blocks[0].tau.as_ref().unwrap().alpha, nh as usize);
        let g_wh1 = gpu.read(&grads.blocks[1].moe.as_ref().unwrap().experts[0].w_h, (inner * d) as usize);

        let loss = |wm: &HashMap<String, Vec<f32>>, im: &[f32]| -> f32 { build(wm).forward(&gpu, &tokens, &targets, im) };
        let eps = 1e-3f32;
        let ok = |a: f32, num: f32| (a - num).abs() <= 5e-3 + 8e-2 * num.abs();

        for &i in &[0usize, 7, 13, 20, 33, 44] {
            let (mut ip, mut im) = (img.clone(), img.clone());
            ip[i] += eps;
            im[i] -= eps;
            let num = (loss(&w, &ip) - loss(&w, &im)) / (2.0 * eps);
            assert!(ok(d_img[i], num), "d_image_embeds[{i}]: analytic {} vs numeric {}", d_img[i], num);
        }
        for &h in &[0usize, 1] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("blocks.0.attn.tau.alpha").unwrap()[h] += eps;
            wm2.get_mut("blocks.0.attn.tau.alpha").unwrap()[h] -= eps;
            let num = (loss(&wp, &img) - loss(&wm2, &img)) / (2.0 * eps);
            assert!(ok(g_alpha0[h], num), "d blocks.0.tau.alpha[{h}]: analytic {} vs numeric {}", g_alpha0[h], num);
        }
        for &j in &[0usize, 17, 40] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("blocks.1.moe.experts.0.w_h.weight").unwrap()[j] += eps;
            wm2.get_mut("blocks.1.moe.experts.0.w_h.weight").unwrap()[j] -= eps;
            let num = (loss(&wp, &img) - loss(&wm2, &img)) / (2.0 * eps);
            assert!(ok(g_wh1[j], num), "d blocks.1.moe.w_h[{j}]: analytic {} vs numeric {}", g_wh1[j], num);
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
        let gpu = gpu_core::testgpu::dev(pipelines());
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
        let loss = dec.forward(&gpu, &tokens, &targets, &img);
        assert!(loss.is_finite() && loss > 0.0, "moondream decoder loss must be finite+positive, got {loss}");
    }

    #[test]
    fn parallel_block_with_moe_runs() {
        let gpu = gpu_core::testgpu::dev(pipelines());
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
        let out = gpu.read(blk.forward(&gpu, &x), blk.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn moe_ffn_geglu_runs() {
        let gpu = gpu_core::testgpu::dev(pipelines());
        let (t, d, inner, e, top_k) = (4u32, 64u32, 32u32, 3u32, 2u32);
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
        let out = gpu.read(ffn.forward(&gpu, &xn), ffn.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-9));
    }

    /// `(weights, input)` for a small MoE at `(t, d, inner, e)`, deterministic
    /// for a fixed seed. `d` and `inner` are multiples of `model::int8::GROUP`
    /// (32) - the weight scale is per 32-element group of the contraction axis
    /// and `quantize_weight` asserts it.
    fn moe_fixture(t: u32, d: u32, inner: u32, e: u32, seed: u64) -> (HashMap<String, Vec<f32>>, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("router.weight".into(), r((e * d) as usize));
        for ei in 0..e {
            w.insert(format!("experts.{ei}.w_h.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_g.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_down.weight"), r((d * inner) as usize));
        }
        let x = (0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect();
        (w, x)
    }

    /// THE GATE FOR THE int8 TIER: it must compute the SAME function as the
    /// fp32 tier, to within quantization error.
    ///
    /// This is what a shape-only smoke test cannot see. `MoeFfn8` differs from
    /// `MoeFfn` in three ways that all still produce finite, plausible numbers
    /// when wrong: the weights are per-output-channel quantized (a transposed
    /// `(n, k)` would rescale whole rows), the activations are dynamically
    /// quantized once per layer (quantizing per expert, or forgetting to
    /// re-quantize `act`, changes the scale silently), and
    /// `moe_linear_gated_i8` SKIPS unrouted rows (an off-by-one in `e_idx`
    /// would mix the wrong expert's output into the wrong rows). Cosine against
    /// the fp32 tier catches all three; `is_finite` catches none of them.
    #[test]
    fn int8_experts_agree_with_the_fp32_tier() {
        let gpu = gpu_core::testgpu::dev(pipelines());
        let (t, d, inner, e, top_k) = (4u32, 64u32, 32u32, 3u32, 2u32);
        let (w, x) = moe_fixture(t, d, inner, e, 5);
        let xn = gpu.storage_init("xn", &x);

        let f32_out = {
            let ffn = MoeFfn::new(&gpu, &w, t, d, inner, e, top_k);
            gpu.read(ffn.forward(&gpu, &xn), ffn.numel())
        };
        let i8_out = {
            let ffn = MoeFfn8::new(&gpu, &w, t, d, inner, e, top_k);
            gpu.read(ffn.forward(&gpu, &xn), ffn.numel())
        };
        assert_eq!(i8_out.len(), f32_out.len());
        assert!(i8_out.iter().all(|v| v.is_finite()), "int8 tier produced a non-finite value");
        assert!(f32_out.iter().any(|&v| v.abs() > 1e-9), "the fp32 reference is all-zero - the comparison would be vacuous");

        let dot: f64 = f32_out.iter().zip(&i8_out).map(|(&a, &b)| a as f64 * b as f64).sum();
        let na: f64 = f32_out.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = i8_out.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb).max(1e-30);
        // int8 is a lossy tier: this floor only has to catch a BROKEN port (a
        // transposed pack, a wrong expert index, a missing re-quantization),
        // all of which land far below it. It is not a claim about accuracy on
        // real weights.
        assert!(cos > 0.99, "int8 MoE diverges from fp32: cosine {cos:.6}");
    }

    /// The int8 tier is deterministic - two calls on one instance agree
    /// bit-for-bit. Quantization happens once at construction; only the
    /// activation scales are per-call, and those are a pure function of the
    /// input.
    #[test]
    fn int8_experts_are_deterministic() {
        let gpu = gpu_core::testgpu::dev(pipelines());
        let (t, d, inner, e, top_k) = (4u32, 64u32, 32u32, 3u32, 2u32);
        let (w, x) = moe_fixture(t, d, inner, e, 9);
        let xn = gpu.storage_init("xn", &x);
        let ffn = MoeFfn8::new(&gpu, &w, t, d, inner, e, top_k);
        let a = gpu.read(ffn.forward(&gpu, &xn), ffn.numel());
        let b = gpu.read(ffn.forward(&gpu, &xn), ffn.numel());
        assert_eq!(a, b, "two forwards on one int8 instance must be bit-identical");
    }
}
