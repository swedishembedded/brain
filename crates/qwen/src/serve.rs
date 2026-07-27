// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Concurrent serving engine for the Qwen3 decoder: a **paged** KV cache shared by
//! many sequences + **batched** decode that advances every active sequence by one
//! token per iteration. Each sequence's KV grows a block at a time from a shared
//! pool (no per-sequence worst-case reservation), and one batched forward serves
//! the whole running set — so more sequences stay resident and decode together.
//!
//! Self-contained: it owns its `Gpu` (with the batched paged kernels), a
//! `ParamStore` of the decoder weights, per-layer block pools, and the block
//! allocator. The forward math is the shared [`model::block`] Qwen3 block; only
//! the attention is paged + ragged-batched.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{self, KernelIds};
use model::paged::{BlockAllocator, BlockTable};
use paramstore::ParamStore;

use crate::config::QwenConfig;

const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const RMS_INV: usize = 3;
const SILU_MUL: usize = 4;
const ADD2: usize = 5;
const ROPE_PAGED: usize = 6;
const KV_APPEND_B: usize = 7;
const SCORES_B: usize = 8;
const SOFTMAX_B: usize = 9;
const APPLY_B: usize = 10;
// Causal attention over a whole prompt (batched prefill).
const GQA_SCORES: usize = 11;
const ATTN_SOFTMAX: usize = 12;
const GQA_APPLY: usize = 13;
// int8 paged KV (dequant on read).
const APPEND_I8: usize = 14;
const SCORES_I8: usize = 15;
const APPLY_I8: usize = 16;

const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("rope_paged", kernels::ROPE_PAGED),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("paged_kv_append_i8_batched", kernels::PAGED_KV_APPEND_I8_BATCHED),
    ("paged_decode_scores_i8_batched", kernels::PAGED_DECODE_SCORES_I8_BATCHED),
    ("paged_decode_apply_i8_batched", kernels::PAGED_DECODE_APPLY_I8_BATCHED),
];

fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        silu_mul: SILU_MUL,
        // unused on the forward decode path:
        rmsnorm_dx: RMSNORM,
        rmsnorm_dw: RMSNORM,
        rope: ROPE_PAGED,
        rope_bwd: ROPE_PAGED,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: 0,
        gqa_dv: 0,
        gqa_dq: 0,
        gqa_dk: 0,
        silu_da: SILU_MUL,
        silu_db: SILU_MUL,
    }
}

fn fb(x: f32) -> u32 {
    x.to_bits()
}

/// A batched-forward input: token ids (embedded via `tok.weight`) or ready-made
/// per-row embeddings written straight into the residual stream (the tts Talker
/// feeds codec/text-conditioned embeddings rather than ids).
pub enum Input<'a> {
    Tokens(&'a [u32]),
    Embeds(&'a [f32]),
}

/// One decoder-param leaf name → element count (mirrors the decode weight set).
fn decoder_param_list(cfg: &QwenConfig) -> Vec<(String, usize)> {
    let (d, ff) = (cfg.d_model as usize, cfg.d_ff as usize);
    let (hq, hkv, hd) = (cfg.q_dim() as usize, cfg.kv_dim() as usize, cfg.head_dim as usize);
    let mut out = Vec::new();
    for l in 0..cfg.n_layers {
        let p = |s: &str| format!("blocks.{l}.{s}");
        out.push((p("ln1.weight"), d));
        out.push((p("attn.wq.weight"), hq * d));
        out.push((p("attn.wk.weight"), hkv * d));
        out.push((p("attn.wv.weight"), hkv * d));
        out.push((p("attn.q_norm.weight"), hd));
        out.push((p("attn.k_norm.weight"), hd));
        out.push((p("attn.wo.weight"), d * hq));
        out.push((p("ln2.weight"), d));
        out.push((p("mlp.gate.weight"), ff * d));
        out.push((p("mlp.up.weight"), ff * d));
        out.push((p("mlp.down.weight"), d * ff));
    }
    out.push(("norm.weight".to_string(), d));
    out.push(("tok.weight".to_string(), cfg.vocab as usize * d)); // embedding gather
    out
}

/// A running sequence: its block table, generated tokens, and completion flag.
struct Seq {
    table: BlockTable,
    generated: Vec<u32>,
    done: bool,
}

/// Batched scratch (sized for `max_batch` rows), reused every iteration.
struct Scratch {
    res: Vec<DeviceBuffer>, // n_layers+1, each [B*d]
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn_final: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    // per-step metadata (uploaded each iteration)
    tok_buf: DeviceBuffer,
    pos_buf: DeviceBuffer,
    seqlen_buf: DeviceBuffer,
    blk_buf: DeviceBuffer,
    off_buf: DeviceBuffer,
    bt_buf: DeviceBuffer,
}

/// Paged, batched Qwen3 serving engine.
pub struct Engine {
    cfg: QwenConfig,
    gpu: Gpu,
    ps: ParamStore,
    block_size: u32,
    max_batch: u32,
    max_blocks_per_seq: u32,
    max_prefill: u32,
    cap: u32,
    alloc: BlockAllocator,
    pool_k: Vec<DeviceBuffer>,
    pool_v: Vec<DeviceBuffer>,
    // int8 KV: pools hold packed int8 (4/u32, ~4x smaller) + per-(token,kv-head)
    // dequant scales. Empty when kv_int8 is false (fp32 pools).
    kv_int8: bool,
    scales_k: Vec<DeviceBuffer>,
    scales_v: Vec<DeviceBuffer>,
    sc: Scratch,
    head: Vec<f32>, // [vocab, d] tied/untied head, host-applied
}

impl Engine {
    /// Build from an in-memory decoder weight map (tests / embedded weights).
    /// `num_blocks` physical blocks of `block_size` tokens, up to `max_batch`
    /// concurrent sequences of at most `max_blocks_per_seq * block_size` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn from_map(cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool) -> Engine {
        let gpu = Gpu::new(PIPELINES);
        let roles = decoder_param_list(&cfg).into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, weights);
        let head = weights.get(cfg.head_weight()).cloned().unwrap_or_else(|| weights.get("tok.weight").cloned().expect("head weight"));

        let (d, ff) = (cfg.d_model as u64, cfg.d_ff as u64);
        let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
        // Scratch rows serve both decode (max_batch sequences) and prefill (a whole
        // prompt of up to max_prefill tokens processed in one forward).
        let b = max_batch.max(max_prefill) as u64;
        let cap = max_blocks_per_seq * block_size;
        let nh = cfg.n_heads as u64;
        // scores/probs hold decode [rows,nh,cap] OR prefill causal [nh,N,N].
        let bcap = (b * nh * cap as u64).max(max_prefill as u64 * max_prefill as u64 * nh);
        let st = |x: u64| gpu.storage(x);

        let mut res = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(b * d));
        }
        let n_kv = cfg.n_kv_heads as u64;
        let slots = num_blocks as u64 * block_size as u64;
        // int8 pools pack 4 values/u32 (÷4 words) + a scale per (token slot, kv-head).
        let pool_words = if kv_int8 { slots * hkv / 4 } else { slots * hkv };
        let mut pool_k = Vec::new();
        let mut pool_v = Vec::new();
        let mut scales_k = Vec::new();
        let mut scales_v = Vec::new();
        for _ in 0..cfg.n_layers {
            pool_k.push(st(pool_words));
            pool_v.push(st(pool_words));
            if kv_int8 {
                scales_k.push(st(slots * n_kv));
                scales_v.push(st(slots * n_kv));
            }
        }
        let sc = Scratch {
            res,
            xn1: st(b * d),
            q_pre: st(b * hq),
            q: st(b * hq),
            k_pre: st(b * hkv),
            k: st(b * hkv),
            v: st(b * hkv),
            ctx: st(b * hq),
            xmid: st(b * d),
            xn2: st(b * d),
            gate_pre: st(b * ff),
            up: st(b * ff),
            h: st(b * ff),
            proj: st(b * d),
            mlp_out: st(b * d),
            xn_final: st(b * d),
            scores: st(bcap),
            probs: st(bcap),
            tok_buf: st(b),
            pos_buf: st(b),
            seqlen_buf: st(b),
            blk_buf: st(b),
            off_buf: st(b),
            bt_buf: st(b * max_blocks_per_seq as u64),
        };
        Engine {
            cfg,
            gpu,
            ps,
            block_size,
            max_batch,
            max_blocks_per_seq,
            max_prefill,
            cap,
            alloc: BlockAllocator::new(num_blocks, block_size),
            pool_k,
            pool_v,
            kv_int8,
            scales_k,
            scales_v,
            sc,
            head,
        }
    }

    /// Load a serving engine from a brain Qwen checkpoint (fp32 decode weights).
    #[allow(clippy::too_many_arguments)]
    pub fn load(path: &str, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool) -> Engine {
        let c = checkpoint::load(path);
        let cfg = QwenConfig::from_json(&c.header["config"]);
        let mut map = HashMap::new();
        for (name, _) in decoder_param_list(&cfg) {
            let t = c.find(&name, "").cloned().unwrap_or_else(|| panic!("serve: checkpoint missing tensor {name}"));
            map.insert(name, t);
        }
        let hw = cfg.head_weight();
        if !map.contains_key(hw) {
            let h = c.find(hw, "").cloned().unwrap_or_else(|| panic!("serve: checkpoint missing head {hw}"));
            map.insert(hw.to_string(), h);
        }
        Engine::from_map(cfg, &map, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8)
    }

    /// The model's vocabulary size (for a caller doing its own sampling).
    pub fn vocab(&self) -> usize {
        self.cfg.vocab as usize
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    /// Append one slot per sequence and gather the batched-forward metadata.
    fn append_meta(&mut self, tables: &mut [&mut BlockTable]) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let mbt = self.max_blocks_per_seq as usize;
        let bsz = tables.len() as u32;
        assert!(bsz <= self.max_batch);
        let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut bt = vec![0u32; bsz as usize * mbt];
        for (i, table) in tables.iter_mut().enumerate() {
            let pos = table.len();
            let (block, offset) = table.append(&mut self.alloc).expect("KV pool exhausted");
            positions.push(pos);
            seqlens.push(pos + 1);
            blocks.push(block);
            offsets.push(offset);
            for (lb, &phys) in table.blocks().iter().enumerate() {
                bt[i * mbt + lb] = phys;
            }
        }
        (bsz, positions, seqlens, blocks, offsets, bt)
    }

    /// Advance every sequence by one token (decode).
    pub(crate) fn forward_batched(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<f32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Advance every sequence by one token from a ready-made embedding per sequence
    /// (`[bsz, d_model]`) — the tts Talker multi-stream path: concurrent voice
    /// streams decode together on the shared paged pool.
    pub fn forward_batched_embed(&mut self, tables: &mut [&mut BlockTable], embeds: &[f32]) -> Vec<f32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        assert_eq!(embeds.len(), bsz as usize * self.cfg.d_model as usize);
        self.run_batched(bsz, Input::Embeds(embeds), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Run one batched forward over `bsz` rows given fully-computed metadata:
    /// `positions[i]` RoPE position, `seqlens[i]` the cached length row i attends
    /// (row i's query attends `j < seqlens[i]` — set to start+i+1 for causal
    /// prefill), `(blocks[i], offsets[i])` the pool slot to write row i's K/V, and
    /// `bt` the per-row block tables (`bsz * max_blocks_per_seq`). Serves decode
    /// (one new token per sequence) and prefill chunks alike.
    #[allow(clippy::too_many_arguments)]
    fn run_batched(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let group = nh / nkv;
        let half = hd / 2;
        let bs = self.block_size;
        let cap = self.cap;
        let mbt = self.max_blocks_per_seq;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let theta = c.rope_theta;
        let g = &self.gpu;
        g.write(&self.sc.pos_buf, positions);
        g.write(&self.sc.seqlen_buf, seqlens);
        g.write(&self.sc.blk_buf, blocks);
        g.write(&self.sc.off_buf, offsets);
        g.write(&self.sc.bt_buf, bt);
        let kids = ids();
        let sc = &self.sc;
        let w = |name: &str| self.ps.w(name);
        let b = bsz;
        let mut s: Vec<Step> = Vec::new();
        match input {
            Input::Tokens(t) => {
                g.write(&sc.tok_buf, t);
                s.push(g.step(EMBED, &[&sc.tok_buf, w("tok.weight"), &sc.res[0]], &[d, b], d * b));
            }
            Input::Embeds(e) => {
                g.write(&sc.res[0], bytemuck::cast_slice(e));
            }
        }
        for l in 0..c.n_layers as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            s.push(block::rmsnorm_fwd(g, &kids, &sc.res[l], w(&p("ln1.weight")), &sc.xn1, d, b));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wq.weight")), &sc.q_pre], &[b, d, hq], b * hq));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wk.weight")), &sc.k_pre], &[b, d, hkv], b * hkv));
            s.push(g.step(MATMUL, &[&sc.xn1, w(&p("attn.wv.weight")), &sc.v], &[b, d, hkv], b * hkv));
            s.push(block::rmsnorm_fwd(g, &kids, &sc.q_pre, w(&p("attn.q_norm.weight")), &sc.q, hd, b * nh));
            s.push(block::rmsnorm_fwd(g, &kids, &sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, hd, b * nkv));
            s.push(g.step(ROPE_PAGED, &[&sc.q, &sc.pos_buf], &[b, nh, hd, hq, fb(theta)], b * nh * half));
            s.push(g.step(ROPE_PAGED, &[&sc.k, &sc.pos_buf], &[b, nkv, hd, hkv, fb(theta)], b * nkv * half));
            if self.kv_int8 {
                s.push(g.step(APPEND_I8, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.pool_k[l], &self.scales_k[l]], &[b, hkv, bs, hd], b * nkv));
                s.push(g.step(APPEND_I8, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.pool_v[l], &self.scales_v[l]], &[b, hkv, bs, hd], b * nkv));
                s.push(g.step(SCORES_I8, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_k[l], &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], b * nh * cap));
                s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                s.push(g.step(APPLY_I8, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_v[l], &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
            } else {
                s.push(g.step(KV_APPEND_B, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.pool_k[l]], &[b, hkv, bs], b * hkv));
                s.push(g.step(KV_APPEND_B, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.pool_v[l]], &[b, hkv, bs], b * hkv));
                s.push(g.step(SCORES_B, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], b * nh * cap));
                s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                s.push(g.step(APPLY_B, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
            }
            s.push(g.step(MATMUL, &[&sc.ctx, w(&p("attn.wo.weight")), &sc.proj], &[b, hq, d], b * d));
            s.push(g.step(ADD2, &[&sc.res[l], &sc.proj, &sc.xmid], &[b * d], b * d));
            s.push(block::rmsnorm_fwd(g, &kids, &sc.xmid, w(&p("ln2.weight")), &sc.xn2, d, b));
            s.push(g.step(MATMUL, &[&sc.xn2, w(&p("mlp.gate.weight")), &sc.gate_pre], &[b, d, ff], b * ff));
            s.push(g.step(MATMUL, &[&sc.xn2, w(&p("mlp.up.weight")), &sc.up], &[b, d, ff], b * ff));
            s.push(block::swiglu_fwd(g, &kids, &sc.gate_pre, &sc.up, &sc.h, b * ff));
            s.push(g.step(MATMUL, &[&sc.h, w(&p("mlp.down.weight")), &sc.mlp_out], &[b, ff, d], b * d));
            s.push(g.step(ADD2, &[&sc.xmid, &sc.mlp_out, &sc.res[l + 1]], &[b * d], b * d));
        }
        let last = c.n_layers as usize;
        s.push(block::rmsnorm_fwd(g, &kids, &sc.res[last], w("norm.weight"), &sc.xn_final, d, b));
        g.submit(&[], &s);
        g.read(&sc.xn_final, (b * d) as usize)
    }

    /// **Chunked prefill**: process the prompt in chunks of up to `max_prefill`
    /// tokens. Each chunk is a batched forward whose C queries attend the paged
    /// prefix + the causal chunk (seqlens[i] = start+i+1), scattering K/V into the
    /// pool for the decode phase. One chunk == whole-prompt prefill; larger prompts
    /// stream through without a giant single forward. Returns the last token's
    /// final-norm hidden `[d_model]`.
    pub(crate) fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "prefill expects a fresh sequence");
        let d = self.cfg.d_model as usize;
        let bs = self.block_size;
        let mbt = self.max_blocks_per_seq as usize;
        let n = prompt.len() as u32;
        let chunk = self.max_prefill.max(1);
        let mut last = Vec::new();
        let mut start = 0u32;
        while start < n {
            let cc = (n - start).min(chunk);
            table.reserve(cc, &mut self.alloc).expect("KV pool exhausted");
            let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut bt = vec![0u32; cc as usize * mbt];
            for i in 0..cc {
                let pos = start + i;
                let (bl, off) = table.locate(pos, bs);
                positions.push(pos);
                seqlens.push(pos + 1); // causal: query i attends positions 0..=pos
                blocks.push(bl);
                offsets.push(off);
                for (lb, &phys) in table.blocks().iter().enumerate() {
                    bt[i as usize * mbt + lb] = phys;
                }
            }
            let hidden = self.run_batched(cc, Input::Tokens(&prompt[start as usize..(start + cc) as usize]), &positions, &seqlens, &blocks, &offsets, &bt);
            let cu = cc as usize;
            last = hidden[(cu - 1) * d..cu * d].to_vec();
            start += cc;
        }
        last
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (d, v) = (self.cfg.d_model as usize, self.cfg.vocab as usize);
        (0..v).map(|o| self.head[o * d..o * d + d].iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
    }

    /// Physical KV blocks currently free in the pool.
    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    /// Blocks a sequence of `tokens` length occupies (for admission checks).
    pub fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }

    fn argmax(s: &[f32]) -> u32 {
        let mut bi = 0;
        for i in 1..s.len() {
            if s[i] > s[bi] {
                bi = i;
            }
        }
        bi as u32
    }

    /// Greedy generation of `max_new` tokens for each prompt, run with a **paged
    /// KV cache** and **batched decode** across all prompts concurrently. Prompts
    /// are prefilled per-sequence (one token per step), then every still-running
    /// sequence advances together each decode iteration. Returns the generated
    /// tokens per prompt. `eos` (when set) stops a sequence early.
    pub fn generate_greedy(&mut self, prompts: &[Vec<u32>], max_new: usize, eos: Option<u32>) -> Vec<Vec<u32>> {
        let mut seqs: Vec<Seq> = prompts.iter().map(|_| Seq { table: BlockTable::new(), generated: Vec::new(), done: false }).collect();

        // Prefill each sequence and sample its first token.
        for (i, prompt) in prompts.iter().enumerate() {
            assert!(!prompt.is_empty(), "empty prompt");
            let hidden = self.prefill(&mut seqs[i].table, prompt);
            let first = Self::argmax(&self.logits(&hidden));
            seqs[i].generated.push(first);
            if Some(first) == eos {
                seqs[i].done = true;
            }
        }

        // Batched decode: feed each running sequence its last token together.
        for _ in 1..max_new {
            let active: Vec<usize> = (0..seqs.len()).filter(|&i| !seqs[i].done).collect();
            if active.is_empty() {
                break;
            }
            let inputs: Vec<u32> = active.iter().map(|&i| *seqs[i].generated.last().unwrap()).collect();
            // Borrow the active sequences' block tables mutably for the batched step.
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = Vec::new();
                for (idx, seq) in seqs.iter_mut().enumerate() {
                    if active.contains(&idx) {
                        refs.push(&mut seq.table);
                    }
                }
                self.forward_batched(&mut refs, &inputs)
            };
            let d = self.cfg.d_model as usize;
            for (bi, &si) in active.iter().enumerate() {
                let next = Self::argmax(&self.logits(&hidden[bi * d..(bi + 1) * d]));
                seqs[si].generated.push(next);
                if Some(next) == eos {
                    seqs[si].done = true;
                }
            }
        }
        for s in seqs.iter_mut() {
            self.release_table(&mut s.table);
        }
        seqs.into_iter().map(|s| s.generated).collect()
    }

    /// **Speculative decoding** (greedy): a `draft` proposes up to `k` tokens from
    /// the running context; the target verifies them in ONE batched forward,
    /// accepting the longest correct prefix plus a bonus/correction token, and
    /// rolling the paged cache back over any rejected tokens. The output is
    /// identical to plain greedy target decoding — the win is fewer (expensive)
    /// target forwards when the draft guesses well. `draft(ctx, want) -> tokens`.
    /// Returns `(generated_tokens, target_forward_count)`.
    pub fn spec_decode<D: FnMut(&[u32], u32) -> Vec<u32>>(&mut self, prompt: &[u32], max_new: usize, k: u32, mut draft: D) -> (Vec<u32>, usize) {
        assert!(!prompt.is_empty() && k >= 1);
        let d = self.cfg.d_model as usize;
        let bs = self.block_size;
        let mbt = self.max_blocks_per_seq as usize;
        let mut table = BlockTable::new();
        // Prefill all but the last prompt token; the last is the first `pending`.
        if prompt.len() > 1 {
            self.prefill(&mut table, &prompt[..prompt.len() - 1]);
        }
        let mut pending = *prompt.last().unwrap();
        let mut ctx: Vec<u32> = prompt.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut forwards = 0usize;

        while generated.len() < max_new {
            let want = ((max_new - generated.len()) as u32).min(k);
            let mut props = draft(&ctx, want);
            props.truncate(want as usize);
            let kk = props.len() as u32;

            // Verify forward over [pending, props...] at positions base..=base+kk.
            let base = table.len();
            let inputs: Vec<u32> = std::iter::once(pending).chain(props.iter().copied()).collect();
            let rows = kk + 1;
            table.reserve(rows, &mut self.alloc).expect("KV pool exhausted");
            let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut bt = vec![0u32; rows as usize * mbt];
            for i in 0..rows {
                let pos = base + i;
                let (bl, off) = table.locate(pos, bs);
                positions.push(pos);
                seqlens.push(pos + 1);
                blocks.push(bl);
                offsets.push(off);
                for (lb, &phys) in table.blocks().iter().enumerate() {
                    bt[i as usize * mbt + lb] = phys;
                }
            }
            let hidden = self.run_batched(rows, Input::Tokens(&inputs), &positions, &seqlens, &blocks, &offsets, &bt);
            forwards += 1;

            // hidden[j] gives the target distribution that should have produced
            // props[j]; accept while it matches, else take the target's own token.
            let mut accepted = 0usize;
            let correction;
            loop {
                if accepted < kk as usize {
                    let pred = Self::argmax(&self.logits(&hidden[accepted * d..(accepted + 1) * d]));
                    if pred == props[accepted] {
                        accepted += 1;
                        continue;
                    }
                    correction = pred;
                    break;
                }
                // All drafts accepted → the bonus token from the last position.
                correction = Self::argmax(&self.logits(&hidden[kk as usize * d..(kk as usize + 1) * d]));
                break;
            }
            for prop in props.iter().take(accepted) {
                generated.push(*prop);
                ctx.push(*prop);
            }
            generated.push(correction);
            ctx.push(correction);
            // Commit pending + accepted drafts; the correction is the next pending.
            table.truncate(base + accepted as u32 + 1, &mut self.alloc);
            pending = correction;
        }
        generated.truncate(max_new);
        table.release(&mut self.alloc);
        (generated, forwards)
    }
}

/// A submitted generation request.
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub eos: Option<u32>,
}

/// A sequence the scheduler is actively decoding.
struct Running {
    id: u64,
    table: BlockTable,
    generated: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
    next_input: u32,
    done: bool,
}

/// **Continuous-batching scheduler.** Requests are submitted at any time, admitted
/// when the KV pool + batch have room (prefilled + first token sampled), then every
/// running sequence advances together in one batched decode step per iteration.
/// Finished sequences return their blocks immediately, so newly submitted requests
/// can be admitted mid-flight — the batch composition changes each iteration to keep
/// as much useful work resident as possible.
pub struct Scheduler {
    eng: Engine,
    waiting: std::collections::VecDeque<(u64, Request)>,
    running: Vec<Running>,
    next_id: u64,
    max_running: usize,
}

impl Scheduler {
    pub fn new(eng: Engine, max_running: usize) -> Scheduler {
        Scheduler { eng, waiting: std::collections::VecDeque::new(), running: Vec::new(), next_id: 0, max_running }
    }

    /// Enqueue a request; returns its id (results come back keyed by it).
    pub fn submit(&mut self, req: Request) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.waiting.push_back((id, req));
        id
    }

    /// True while any request is waiting or running.
    pub fn pending(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    fn finish_check(r: &mut Running) {
        if Some(*r.generated.last().unwrap()) == r.eos || r.generated.len() >= r.max_new {
            r.done = true;
        }
    }

    /// One scheduler iteration: admit waiting requests that fit (prefill + sample
    /// first token), run one batched decode step over all running sequences, then
    /// reap completed ones. Returns the `(id, tokens)` of requests finished here.
    pub fn step(&mut self) -> Vec<(u64, Vec<u32>)> {
        let d = self.eng.cfg.d_model as usize;

        // 1. Admit while there's batch room and enough free blocks for the prompt.
        while self.running.len() < self.max_running {
            let fits = match self.waiting.front() {
                Some((_, req)) => self.eng.free_blocks() >= self.eng.blocks_for(req.prompt.len() as u32 + 1),
                None => false,
            };
            if !fits {
                break;
            }
            let (id, req) = self.waiting.pop_front().unwrap();
            let mut table = BlockTable::new();
            let hidden = self.eng.prefill(&mut table, &req.prompt);
            let first = Engine::argmax(&self.eng.logits(&hidden));
            let mut r = Running { id, table, generated: vec![first], max_new: req.max_new, eos: req.eos, next_input: first, done: false };
            Self::finish_check(&mut r);
            self.running.push(r);
        }

        // 2. Batched decode over every running (not-done) sequence.
        let active: Vec<usize> = (0..self.running.len()).filter(|&i| !self.running[i].done).collect();
        if !active.is_empty() {
            let inputs: Vec<u32> = active.iter().map(|&i| self.running[i].next_input).collect();
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = Vec::new();
                for (idx, r) in self.running.iter_mut().enumerate() {
                    if active.contains(&idx) {
                        refs.push(&mut r.table);
                    }
                }
                self.eng.forward_batched(&mut refs, &inputs)
            };
            for (bi, &si) in active.iter().enumerate() {
                let next = Engine::argmax(&self.eng.logits(&hidden[bi * d..(bi + 1) * d]));
                let r = &mut self.running[si];
                r.generated.push(next);
                r.next_input = next;
                Self::finish_check(r);
            }
        }

        // 3. Reap completed sequences, returning their blocks to the pool.
        let mut completed = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].done {
                let mut r = self.running.remove(i);
                self.eng.release_table(&mut r.table);
                completed.push((r.id, r.generated));
            } else {
                i += 1;
            }
        }
        completed
    }

    /// Drive to completion, returning every request's tokens keyed by id.
    pub fn run(&mut self) -> HashMap<u64, Vec<u32>> {
        let mut out = HashMap::new();
        while self.pending() {
            for (id, toks) in self.step() {
                out.insert(id, toks);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Qwen;
    use data::rng::Rng;

    fn tiny_weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(1);
        let mut map = HashMap::new();
        for (name, count) in cfg.param_list() {
            let v = if name.contains("norm") { vec![1.0f32; count] } else { (0..count).map(|_| rng.next_gaussian() as f32 * 0.05).collect() };
            map.insert(name, v);
        }
        map
    }

    /// Single-sequence paged/batched serving must match the reference contiguous
    /// KV generation (`Qwen::generate_kv`) token-for-token, and a two-sequence
    /// batch must equal running each prompt on its own — proving batched paged
    /// decode is exact.
    #[test]
    fn batched_serving_matches_reference() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let bs = 4;
        let (num_blocks, max_batch, mbt) = (64u32, 4u32, 8u32);

        // Reference: the committed single-sequence KV generation.
        let model = Qwen::new(cfg.clone(), 1, 64, &map);
        let p0 = vec![1u32, 5, 3, 9];
        let p1 = vec![7u32, 2, 4];
        let mut r0 = Rng::new(0);
        let mut r1 = Rng::new(0);
        let ref0 = crate::sample::generate_kv(&model, &p0, 12, 0.0, 0, None, &mut r0);
        let ref1 = crate::sample::generate_kv(&model, &p1, 12, 0.0, 0, None, &mut r1);

        // Engine: run both prompts concurrently (batched paged).
        let mut eng = Engine::from_map(cfg, &map, bs, num_blocks, max_batch, mbt, 32, false);
        let out = eng.generate_greedy(&[p0.clone(), p1.clone()], 12, None);

        assert_eq!(out[0], ref0, "seq0 batched paged != reference");
        assert_eq!(out[1], ref1, "seq1 batched paged != reference");
    }

    /// Continuous batching: requests submitted at DIFFERENT times (one mid-flight)
    /// must each produce the same tokens as run alone — the scheduler admits,
    /// batches, completes, and frees dynamically without changing any output.
    #[test]
    fn scheduler_dynamic_admission_matches_reference() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let model = Qwen::new(cfg.clone(), 1, 64, &map);

        let prompts = [vec![1u32, 5, 3, 9], vec![7u32, 2, 4], vec![3u32, 3, 8, 1, 6]];
        let maxn = [10usize, 6, 8];
        let refs: Vec<Vec<u32>> = prompts
            .iter()
            .zip(maxn)
            .map(|(p, n)| {
                let mut r = Rng::new(0);
                crate::sample::generate_kv(&model, p, n, 0.0, 0, None, &mut r)
            })
            .collect();

        let eng = Engine::from_map(cfg, &map, 4, 64, 4, 8, 32, false);
        let mut sched = Scheduler::new(eng, 4);
        let mut out: HashMap<u64, Vec<u32>> = HashMap::new();

        let id0 = sched.submit(Request { prompt: prompts[0].clone(), max_new: maxn[0], eos: None });
        let id1 = sched.submit(Request { prompt: prompts[1].clone(), max_new: maxn[1], eos: None });
        // Run two iterations with only the first two requests active...
        for _ in 0..2 {
            for (id, t) in sched.step() {
                out.insert(id, t);
            }
        }
        // ...then submit a third mid-flight; it must batch in and still be correct.
        let id2 = sched.submit(Request { prompt: prompts[2].clone(), max_new: maxn[2], eos: None });
        while sched.pending() {
            for (id, t) in sched.step() {
                out.insert(id, t);
            }
        }

        assert_eq!(out[&id0], refs[0], "req0 under continuous batching != reference");
        assert_eq!(out[&id1], refs[1], "req1 under continuous batching != reference");
        assert_eq!(out[&id2], refs[2], "mid-flight req2 != reference");
    }

    /// int8 paged KV stays close to fp32 through prefill + decode (both read the
    /// quantised cache) — a ~4× smaller KV pool for a small, bounded error.
    #[test]
    fn int8_kv_close_to_fp32() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2];
        let run = |int8: bool| -> Vec<f32> {
            let mut e = Engine::from_map(cfg.clone(), &map, 4, 64, 1, 8, 32, int8);
            let mut t = BlockTable::new();
            let mut hidden = e.prefill(&mut t, &prompt);
            for _ in 0..6 {
                let next = Engine::argmax(&e.logits(&hidden));
                let mut one = [&mut t];
                hidden = e.forward_batched(&mut one, &[next]);
            }
            hidden
        };
        let h32 = run(false);
        let h8 = run(true);
        let err = h32.iter().zip(&h8).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let mag = h32.iter().fold(0f32, |m, &x| m.max(x.abs()));
        println!("int8 KV vs fp32 (prefill + 6 decode) maxabs={err:e} (mag {mag:e})");
        assert!(err < 0.2 * mag + 1e-3, "int8 diverges too far: {err} vs mag {mag}");
    }

    /// Chunked prefill (small chunk) must produce the same hidden as whole-prompt
    /// prefill — the prompt streams through in pieces attending the paged prefix.
    #[test]
    fn chunked_prefill_matches_whole() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2, 7, 4, 8];
        let prefill_last = |max_prefill: u32| -> Vec<f32> {
            let mut e = Engine::from_map(cfg.clone(), &map, 4, 64, 1, 8, max_prefill, false);
            let mut t = BlockTable::new();
            e.prefill(&mut t, &prompt)
        };
        let whole = prefill_last(16); // one chunk
        let chunked = prefill_last(2); // 4 chunks of 2
        let err = whole.iter().zip(&chunked).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("chunked (2) vs whole prefill: maxabs={err:e}");
        assert!(err < 1e-4, "chunked prefill != whole prefill: {err}");
    }

    /// Speculative decoding output equals plain greedy — with a good (oracle)
    /// draft it takes far fewer target forwards; with a bad draft it falls back to
    /// ~one token per forward. Either way the tokens are identical.
    #[test]
    fn spec_decode_matches_greedy() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9];
        let max_new = 20usize;

        let mut e_ref = Engine::from_map(cfg.clone(), &map, 4, 64, 1, 8, 32, false);
        let greedy = e_ref.generate_greedy(&[prompt.clone()], max_new, None)[0].clone();
        let full: Vec<u32> = prompt.iter().copied().chain(greedy.iter().copied()).collect();

        // Oracle draft: proposes the true continuation → all accepted.
        let mut e1 = Engine::from_map(cfg.clone(), &map, 4, 64, 1, 8, 32, false);
        let (out_oracle, fwd_oracle) = e1.spec_decode(&prompt, max_new, 4, |ctx, want| {
            (0..want as usize).map(|i| full.get(ctx.len() + i).copied().unwrap_or(0)).collect()
        });
        // Bad draft: always proposes token 0 → mostly rejected.
        let mut e2 = Engine::from_map(cfg, &map, 4, 64, 1, 8, 32, false);
        let (out_bad, fwd_bad) = e2.spec_decode(&prompt, max_new, 4, |_ctx, want| vec![0u32; want as usize]);

        println!("spec decode: greedy={max_new} tokens | oracle-draft {fwd_oracle} target-forwards | bad-draft {fwd_bad} forwards");
        assert_eq!(out_oracle, greedy, "spec (oracle draft) != greedy");
        assert_eq!(out_bad, greedy, "spec (bad draft) != greedy");
        assert!(fwd_oracle < max_new, "oracle draft should cut target forwards ({fwd_oracle} vs {max_new})");
        assert!(fwd_bad >= fwd_oracle, "bad draft should need more forwards");
    }

    /// tts multi-stream: N Talker streams (embedding inputs) decoded together on
    /// the shared paged pool must match each stream decoded alone — bit-for-bit.
    /// (The Talker is the same Qwen3 block, so the tiny config stands in for it.)
    #[test]
    fn tts_multistream_embed_matches_per_stream() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let d = cfg.d_model as usize;
        let (n_streams, steps) = (3usize, 5usize);
        let mut rng = Rng::new(42);
        let embs: Vec<Vec<Vec<f32>>> = (0..n_streams)
            .map(|_| (0..steps).map(|_| (0..d).map(|_| rng.next_gaussian() as f32).collect()).collect())
            .collect();

        // Batched: all streams advance together each step.
        let mut e = Engine::from_map(cfg.clone(), &map, 4, 64, n_streams as u32, 8, 4, false);
        let mut tables: Vec<BlockTable> = (0..n_streams).map(|_| BlockTable::new()).collect();
        let mut batched: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
        for s in 0..steps {
            let flat: Vec<f32> = (0..n_streams).flat_map(|i| embs[i][s].clone()).collect();
            let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
            let h = e.forward_batched_embed(&mut refs, &flat);
            for (i, b) in batched.iter_mut().enumerate() {
                b.push(h[i * d..(i + 1) * d].to_vec());
            }
        }

        // Per-stream reference.
        let mut worst = 0f32;
        for (i, se) in embs.iter().enumerate() {
            let mut e1 = Engine::from_map(cfg.clone(), &map, 4, 64, 1, 8, 4, false);
            let mut t = BlockTable::new();
            for (s, emb) in se.iter().enumerate() {
                let mut refs = [&mut t];
                let h = e1.forward_batched_embed(&mut refs, emb);
                worst = worst.max(h.iter().zip(&batched[i][s]).fold(0f32, |m, (a, b)| m.max((a - b).abs())));
            }
        }
        println!("tts multi-stream (embed) vs per-stream: worst maxabs = {worst:e}");
        assert!(worst < 1e-6, "batched embed decode != per-stream: {worst}");
    }

    fn medium_cfg() -> QwenConfig {
        let mut c = QwenConfig::tiny();
        c.n_layers = 8;
        c.d_model = 256;
        c.head_dim = 64;
        c.n_heads = 8;
        c.n_kv_heads = 4;
        c.d_ff = 1024;
        c.vocab = 256;
        c
    }

    /// Throughput: N concurrent requests served with continuous batching vs run one
    /// at a time. Batched decode should give higher aggregate tokens/sec.
    ///   cargo test -p brain-qwen --lib serve_throughput -- --ignored --nocapture
    #[test]
    #[ignore]
    fn serve_throughput() {
        let cfg = medium_cfg();
        let (dm, nl) = (cfg.d_model, cfg.n_layers);
        let map = tiny_weights(&cfg);
        let n_req = 8usize;
        let max_new = 48usize;
        let prompts: Vec<Vec<u32>> = (0..n_req).map(|i| vec![(i as u32 % 200) + 1, 5, 3, 9, 2]).collect();

        // Sequential: one request at a time (fresh reuse of one engine's pool).
        let mut eng_seq = Engine::from_map(cfg.clone(), &map, 16, 512, n_req as u32, 16, 32, false);
        let t0 = std::time::Instant::now();
        for p in &prompts {
            eng_seq.generate_greedy(&[p.clone()], max_new, None);
        }
        let seq_s = t0.elapsed().as_secs_f64();

        // Continuous batching: all requests admitted + decoded together.
        let eng = Engine::from_map(cfg, &map, 16, 512, n_req as u32, 16, 32, false);
        let mut sched = Scheduler::new(eng, n_req);
        for p in &prompts {
            sched.submit(Request { prompt: p.clone(), max_new, eos: None });
        }
        let t1 = std::time::Instant::now();
        let out = sched.run();
        let batch_s = t1.elapsed().as_secs_f64();

        let total_tokens = (n_req * max_new) as f64;
        assert_eq!(out.len(), n_req);
        println!(
            "serve throughput ({n_req} reqs x {max_new} tok, d{dm} L{nl}): sequential {:.1} tok/s ({seq_s:.2}s) | continuous-batched {:.1} tok/s ({batch_s:.2}s) | {:.1}x",
            total_tokens / seq_s,
            total_tokens / batch_s,
            seq_s / batch_s,
        );
    }
}
