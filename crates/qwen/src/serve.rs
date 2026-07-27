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
        gqa_scores: 0,
        gqa_apply: 0,
        attn_softmax: 0,
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
    cap: u32,
    alloc: BlockAllocator,
    pool_k: Vec<DeviceBuffer>,
    pool_v: Vec<DeviceBuffer>,
    sc: Scratch,
    head: Vec<f32>, // [vocab, d] tied/untied head, host-applied
}

impl Engine {
    /// Build from an in-memory decoder weight map (tests / embedded weights).
    /// `num_blocks` physical blocks of `block_size` tokens, up to `max_batch`
    /// concurrent sequences of at most `max_blocks_per_seq * block_size` tokens.
    pub fn from_map(cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32) -> Engine {
        let gpu = Gpu::new(PIPELINES);
        let roles = decoder_param_list(&cfg).into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, weights);
        let head = weights.get(cfg.head_weight()).cloned().unwrap_or_else(|| weights.get("tok.weight").cloned().expect("head weight"));

        let (d, ff) = (cfg.d_model as u64, cfg.d_ff as u64);
        let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
        let b = max_batch as u64;
        let cap = max_blocks_per_seq * block_size;
        let nh = cfg.n_heads as u64;
        let bcap = b * nh * cap as u64;
        let st = |x: u64| gpu.storage(x);

        let mut res = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(b * d));
        }
        let mut pool_k = Vec::new();
        let mut pool_v = Vec::new();
        for _ in 0..cfg.n_layers {
            pool_k.push(st(num_blocks as u64 * block_size as u64 * hkv));
            pool_v.push(st(num_blocks as u64 * block_size as u64 * hkv));
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
            cap,
            alloc: BlockAllocator::new(num_blocks, block_size),
            pool_k,
            pool_v,
            sc,
            head,
        }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    /// Advance `seqs` (all active) by feeding one `input` token each; appends its
    /// K/V into each sequence's paged cache and returns the per-sequence final-norm
    /// hidden state `[B, d_model]`. Metadata (positions, block tables, append
    /// slots) is derived from each sequence's current block table.
    fn forward_batched(&mut self, seqs: &mut [&mut Seq], inputs: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let group = nh / nkv;
        let half = hd / 2;
        let bsz = seqs.len() as u32;
        let bs = self.block_size;
        let cap = self.cap;
        let mbt = self.max_blocks_per_seq;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let theta = c.rope_theta;
        assert!(bsz <= self.max_batch);

        // Host metadata: append a slot for each sequence's new token.
        let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut bt = vec![0u32; bsz as usize * mbt as usize];
        for (i, seq) in seqs.iter_mut().enumerate() {
            let pos = seq.table.len();
            let (block, offset) = seq.table.append(&mut self.alloc).expect("KV pool exhausted");
            positions.push(pos);
            seqlens.push(pos + 1);
            blocks.push(block);
            offsets.push(offset);
            for (lb, &phys) in seq.table.blocks().iter().enumerate() {
                bt[i * mbt as usize + lb] = phys;
            }
        }
        let g = &self.gpu;
        g.write(&self.sc.tok_buf, inputs);
        g.write(&self.sc.pos_buf, &positions);
        g.write(&self.sc.seqlen_buf, &seqlens);
        g.write(&self.sc.blk_buf, &blocks);
        g.write(&self.sc.off_buf, &offsets);
        g.write(&self.sc.bt_buf, &bt);

        let kids = ids();
        let sc = &self.sc;
        let w = |name: &str| self.ps.w(name);
        let b = bsz;
        let mut s: Vec<Step> = Vec::new();
        s.push(g.step(EMBED, &[&sc.tok_buf, w("tok.weight"), &sc.res[0]], &[d, b], d * b));
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
            s.push(g.step(KV_APPEND_B, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.pool_k[l]], &[b, hkv, bs], b * hkv));
            s.push(g.step(KV_APPEND_B, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.pool_v[l]], &[b, hkv, bs], b * hkv));
            s.push(g.step(SCORES_B, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], b * nh * cap));
            s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
            s.push(g.step(APPLY_B, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
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

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (d, v) = (self.cfg.d_model as usize, self.cfg.vocab as usize);
        (0..v).map(|o| self.head[o * d..o * d + d].iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
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
            let mut hidden = Vec::new();
            for &tok in prompt {
                let mut one = [&mut seqs[i]];
                hidden = self.forward_batched(&mut one, &[tok]);
            }
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
            // Borrow the active sequences mutably for the batched step.
            let mut refs: Vec<&mut Seq> = Vec::new();
            for (idx, seq) in seqs.iter_mut().enumerate() {
                if active.contains(&idx) {
                    refs.push(seq);
                }
            }
            let hidden = self.forward_batched(&mut refs, &inputs);
            let d = self.cfg.d_model as usize;
            for (bi, &si) in active.iter().enumerate() {
                let next = Self::argmax(&self.logits(&hidden[bi * d..(bi + 1) * d]));
                seqs[si].generated.push(next);
                if Some(next) == eos {
                    seqs[si].done = true;
                }
            }
        }
        seqs.into_iter().map(|s| s.generated).collect()
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
        let mut eng = Engine::from_map(cfg, &map, bs, num_blocks, max_batch, mbt);
        let out = eng.generate_greedy(&[p0.clone(), p1.clone()], 12, None);

        assert_eq!(out[0], ref0, "seq0 batched paged != reference");
        assert_eq!(out[1], ref1, "seq1 batched paged != reference");
    }
}
