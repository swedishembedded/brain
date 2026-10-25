// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching serving for recurrent-depth
// language models for its clients. If your team needs expertise in paged KV
// pools where every recurrent pass owns its own history, or in serving a looped
// model behind one continuous-batching engine, then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The Ouro serving engine.
//!
//! One shared parameter set, `T` attention histories per sequence. The pool is
//! paged exactly like a conventional decoder's, except there are `T` pools per
//! layer (one per recurrent pass) and a sequence owns `T` block tables. The
//! scheduler only sees the first table; its length is the sequence length and
//! the engine allocates the other `T-1` tables alongside it from the same
//! allocator, so admission's block accounting (`passes × blocks_for`) matches
//! what is actually consumed.
//!
//! Prefill reads the prompt one token at a time through the same paged decode
//! step that serves generation, which keeps one code path for both and needs no
//! variable-shape graph. A chunked prefill is a later optimization; the gate on
//! this path is that it reproduces the full-sequence forward exactly.
//!
//! Not enabled for the looped path yet, by design rather than omission:
//! prefix-cache sharing, host-KV swap and runtime adapters. Each needs a
//! pass-aware identity the shared bookkeeping does not have.

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use ::model::paged::{BlockAllocator, BlockTable};
use ::model::serve::PagedDecoder;
use paramstore::ParamStore;

use crate::config::OuroConfig;
use crate::model;

struct Scratch {
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xn1_2: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    mlp_h: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn2_2: DeviceBuffer,
}

struct Row {
    /// Block tables for passes `1..T`; pass 0 is the scheduler's table.
    extra: Vec<BlockTable>,
}

pub struct Engine {
    gpu: Gpu,
    cfg: OuroConfig,
    ps: ParamStore,
    alloc: BlockAllocator,
    block_size: u32,
    max_bt: u32,
    /// Per-sequence logit-addressable capacity (`max_bt * block_size`).
    cap: u32,
    /// `[pass][layer]` K/V planes over the shared block pool.
    k_pool: Vec<Vec<DeviceBuffer>>,
    v_pool: Vec<Vec<DeviceBuffer>>,
    rows: HashMap<u32, Row>,
    next_row: u32,

    token: DeviceBuffer,
    h: [DeviceBuffer; 2],
    hs: Vec<DeviceBuffer>,
    gates: Vec<DeviceBuffer>,
    logits: DeviceBuffer,
    hrow: DeviceBuffer,
    s: Scratch,
    idx_blocks: DeviceBuffer,
    idx_offsets: DeviceBuffer,
    idx_tables: DeviceBuffer,
    idx_lens: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    last_pass: Cell<usize>,
}

impl Engine {
    /// Build an engine over a checkpoint directory (HF layout) or a brain
    /// checkpoint file, with `max_tokens` of shared KV capacity and
    /// `block_size` tokens per page.
    pub fn open(path: &str, max_tokens: u32, block_size: u32) -> Result<Engine, String> {
        let gpu = Gpu::new(model::PIPELINES);
        let (cfg, ps) = if std::path::Path::new(path).is_dir() {
            let (cfg, src) = crate::import::open_dir(std::path::Path::new(path))?;
            let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
            let ps = ParamStore::new_with_roles_src(&gpu, roles, &src);
            (cfg, ps)
        } else {
            let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("{path}: {e}"))?;
            let cfg = OuroConfig::from_json(&reader.config())?;
            let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
            let ps = ParamStore::new_with_roles_src(&gpu, roles, &reader);
            (cfg, ps)
        };
        Ok(Engine::build(gpu, cfg, ps, max_tokens, block_size))
    }

    /// Build from an in-memory weight source (tests use `init::init_weights`).
    pub fn from_source(cfg: OuroConfig, src: &dyn checkpoint::TensorSource, max_tokens: u32, block_size: u32) -> Engine {
        let gpu = Gpu::new(model::PIPELINES);
        let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        Engine::build(gpu, cfg, ps, max_tokens, block_size)
    }

    fn build(gpu: Gpu, cfg: OuroConfig, ps: ParamStore, max_tokens: u32, block_size: u32) -> Engine {
        assert!(block_size > 0 && max_tokens > 0 && max_tokens % block_size == 0);
        let num_blocks = max_tokens / block_size;
        let max_bt = num_blocks;
        let cap = num_blocks * block_size;
        let alloc = BlockAllocator::new(num_blocks, block_size);
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
        let nh = cfg.n_heads as u64;
        let st = |x: u64| gpu.storage(x);

        let mut k_pool = Vec::new();
        let mut v_pool = Vec::new();
        for _ in 0..cfg.total_ut_steps {
            let mut pk = Vec::new();
            let mut pv = Vec::new();
            for _ in 0..cfg.n_layers {
                pk.push(st(max_tokens as u64 * hkv));
                pv.push(st(max_tokens as u64 * hkv));
            }
            k_pool.push(pk);
            v_pool.push(pv);
        }
        let token = gpu.buffer("ouro_token", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let h = [st(d), st(d)];
        let hs = (0..cfg.total_ut_steps).map(|_| st(d)).collect();
        let gates = (0..cfg.total_ut_steps).map(|_| st(1)).collect();
        let logits = st(cfg.vocab as u64);
        let hrow = st(d);
        let s = Scratch {
            xn1: st(d),
            q: st(hq),
            k: st(hkv),
            v: st(hkv),
            ctx: st(hq),
            proj: st(d),
            xn1_2: st(d),
            xmid: st(d),
            xn2: st(d),
            gate_pre: st(ff),
            up: st(ff),
            mlp_h: st(ff),
            mlp_out: st(d),
            xn2_2: st(d),
        };
        let idx_blocks = gpu.buffer("ouro_blocks", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let idx_offsets = gpu.buffer("ouro_offsets", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let idx_tables = gpu.buffer("ouro_tables", max_bt as u64 * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let idx_lens = gpu.buffer("ouro_lens", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let scores = st(nh * cap as u64);
        let probs = st(nh * cap as u64);

        Engine {
            gpu,
            cfg,
            ps,
            alloc,
            block_size,
            max_bt,
            cap,
            k_pool,
            v_pool,
            rows: HashMap::new(),
            next_row: 0,
            token,
            h,
            hs,
            gates,
            logits,
            hrow,
            s,
            idx_blocks,
            idx_offsets,
            idx_tables,
            idx_lens,
            scores,
            probs,
            last_pass: Cell::new(0),
        }
    }

    fn ensure_row(&mut self, table: &mut BlockTable) -> Result<u32, String> {
        if let Some(id) = table.state_row() {
            return Ok(id);
        }
        let id = self.next_row;
        self.next_row += 1;
        table.set_state_row(id);
        let extra = (1..self.cfg.total_ut_steps).map(|_| BlockTable::new()).collect();
        self.rows.insert(id, Row { extra });
        Ok(id)
    }

    /// One token through every pass, appending its K/V to each pass's table.
    /// Returns the logits of the selected pass.
    fn advance(&mut self, table: &mut BlockTable, row: u32, token: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        let gpu = &self.gpu;
        let ids = model::kernel_ids();
        let dec = model::decode_batched_ids();
        let s = &self.s;
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let t = cfg.total_ut_steps as usize;
        let l_count = cfg.n_layers as usize;

        gpu.write(&self.token, &[token]);
        let mut steps = vec![gpu.step(model::EMBED_ID, &[&self.token, self.ps.w("tok.weight"), &self.h[0]], &[d, 1], d)];
        for p in 0..t {
            let pos = if p == 0 { table.len() } else { self.rows[&row].extra[p - 1].len() };
            // Pass 0 appends into the scheduler's table; the other passes into
            // their own tables, allocated from the same pool so admission's
            // `passes × blocks_for` is what is actually consumed.
            let appended = if p == 0 {
                table.append(&mut self.alloc)
            } else {
                let ex = &mut self.rows.get_mut(&row).expect("row").extra[p - 1];
                ex.append(&mut self.alloc)
            };
            let (blk, off) = appended.expect("out of KV blocks");
            gpu.write(&self.idx_blocks, &[blk]);
            gpu.write(&self.idx_offsets, &[off]);
            gpu.write(&self.idx_lens, &[pos + 1]);
            let mut bt = vec![0u32; self.max_bt as usize];
            let blocks = if p == 0 { table.blocks() } else { self.rows[&row].extra[p - 1].blocks() };
            for (i, b) in blocks.iter().enumerate() {
                bt[i] = *b;
            }
            gpu.write(&self.idx_tables, &bt);

            let mut prev: &DeviceBuffer = if p == 0 { &self.h[0] } else { &self.hs[p - 1] };
            for l in 0..l_count {
                let out: &DeviceBuffer = if p == 0 { &self.h[(l + 1) % 2] } else { &self.h[l % 2] };
                let pfx = |name: &str| format!("blocks.{l}.{name}");
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, prev, self.ps.w(&pfx("ln1.weight")), &s.xn1, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn1, self.ps.w(&pfx("attn.wq.weight")), &s.q], &[1, d, hq], hq));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn1, self.ps.w(&pfx("attn.wk.weight")), &s.k], &[1, d, hkv], hkv));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn1, self.ps.w(&pfx("attn.wv.weight")), &s.v], &[1, d, hkv], hkv));
                steps.push(rope_at(gpu, &s.q, nh, hd, hq, pos, cfg.rope_theta));
                steps.push(rope_at(gpu, &s.k, nkv, hd, hkv, pos, cfg.rope_theta));
                steps.extend(::model::block::gqa_decode_batched_step(
                    gpu,
                    &dec,
                    nh,
                    nkv,
                    hd,
                    1,
                    self.block_size,
                    self.max_bt,
                    self.cap,
                    &s.q,
                    &s.k,
                    &s.v,
                    &self.k_pool[p][l],
                    &self.v_pool[p][l],
                    &self.idx_blocks,
                    &self.idx_offsets,
                    &self.idx_tables,
                    &self.idx_lens,
                    &self.scores,
                    &self.probs,
                    &s.ctx,
                ));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.ctx, self.ps.w(&pfx("attn.wo.weight")), &s.proj], &[1, hq, d], d));
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &s.proj, self.ps.w(&pfx("ln1_2.weight")), &s.xn1_2, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::ADD2_ID, &[prev, &s.xn1_2, &s.xmid], &[d], d));
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &s.xmid, self.ps.w(&pfx("ln2.weight")), &s.xn2, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn2, self.ps.w(&pfx("mlp.gate.weight")), &s.gate_pre], &[1, d, ff], ff));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn2, self.ps.w(&pfx("mlp.up.weight")), &s.up], &[1, d, ff], ff));
                steps.push(::model::block::swiglu_fwd(gpu, &ids, &s.gate_pre, &s.up, &s.mlp_h, ff));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.mlp_h, self.ps.w(&pfx("mlp.down.weight")), &s.mlp_out], &[1, ff, d], d));
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &s.mlp_out, self.ps.w(&pfx("ln2_2.weight")), &s.xn2_2, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::ADD2_ID, &[&s.xmid, &s.xn2_2, out], &[d], d));
                prev = out;
            }
            steps.push(::model::block::rmsnorm_fwd(gpu, &ids, prev, self.ps.w("norm.weight"), &self.hs[p], d, 1, cfg.norm_eps));
            steps.push(gpu.step(model::MATMUL_ID, &[&self.hs[p], self.ps.w("gate.weight"), &self.gates[p]], &[1, d, 1], 1));
            steps.push(gpu.step(model::BIAS_ADD_ID, &[&self.gates[p], self.ps.w("gate.bias")], &[1, 1], 1));
        }
        gpu.submit(&[], &steps);
        let pass = self.choose_exit();
        let head = vec![gpu.step(model::MATMUL_ID, &[&self.hs[pass], self.ps.w("lm_head.weight"), &self.logits], &[1, d, cfg.vocab], cfg.vocab)];
        gpu.submit(&[], &head);
        self.last_pass.set(pass);
        gpu.read(&self.logits, cfg.vocab as usize)
    }

    fn choose_exit(&self) -> usize {
        let t = self.cfg.total_ut_steps;
        let mut remaining = 1.0f64;
        let mut cdf = 0.0f64;
        let mut out = (t - 1) as usize;
        for pass in 0..t {
            let gate = self.gpu.read(&self.gates[pass as usize], 1)[0] as f64;
            let lambda = 1.0 / (1.0 + (-gate).exp());
            let p = if pass + 1 < t {
                let v = lambda * remaining;
                remaining *= 1.0 - lambda;
                v
            } else {
                remaining
            };
            cdf += p;
            if cdf >= self.cfg.early_exit_threshold as f64 {
                out = pass as usize;
            }
        }
        out
    }

    fn release_extras(&mut self, table: &mut BlockTable) {
        if let Some(id) = table.take_state_row() {
            if let Some(mut row) = self.rows.remove(&id) {
                for t in row.extra.iter_mut() {
                    t.release(&mut self.alloc);
                }
            }
        }
    }
}

impl PagedDecoder for Engine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }

    fn max_prefill_tokens(&self) -> u32 {
        self.cap
    }

    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }

    fn max_seq_len(&self) -> usize {
        self.cap as usize
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab as usize
    }

    /// A looped sequence consumes one block per pass per page of context.
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size) * self.cfg.total_ut_steps
    }

    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }

    fn release_table(&mut self, t: &mut BlockTable) {
        self.release_extras(t);
        t.release(&mut self.alloc);
    }

    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        let row = self.ensure_row(table).expect("row");
        let mut hidden = Vec::new();
        for tok in prompt {
            self.advance(table, row, *tok);
            hidden = self.gpu.read(&self.hs[self.last_pass.get()], self.cfg.d_model as usize);
        }
        hidden
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.gpu.write_f32(&self.hrow, hidden);
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        let steps = vec![self.gpu.step(model::MATMUL_ID, &[&self.hrow, self.ps.w("lm_head.weight"), &self.logits], &[1, d, v], v)];
        self.gpu.submit(&[], &steps);
        self.gpu.read(&self.logits, v as usize)
    }

    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(tables.len());
        for (t, input) in tables.iter_mut().zip(inputs) {
            let t: &mut BlockTable = t;
            let row = self.ensure_row(t).expect("row");
            let logits = self.advance(t, row, *input);
            out.push(argmax(&logits));
        }
        out
    }

    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); tables.len()];
        let mut next = inputs.to_vec();
        for _ in 0..k.max(1) {
            let chosen = self.forward_batched_greedy(tables, &next);
            for (row, tok) in out.iter_mut().zip(&chosen) {
                row.push(*tok);
            }
            next = chosen;
        }
        out
    }

    fn decode_window_capacity(&self) -> usize {
        1
    }

    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let mut out = Vec::with_capacity(tables.len());
        for (t, input) in tables.iter_mut().zip(inputs) {
            let t: &mut BlockTable = t;
            let row = self.ensure_row(t).expect("row");
            let logits = self.advance(t, row, *input);
            let mut c: Vec<(u32, f32)> = logits.into_iter().enumerate().map(|(i, v)| (i as u32, v)).collect();
            c.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            c.truncate(k.max(1));
            out.push(c);
        }
        out
    }

    fn topk_capacity(&self) -> usize {
        8
    }

    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }

    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.gpu.stats()
    }

    fn kv_pool_bytes(&self) -> u64 {
        let per_plane = self.cap as u64 * self.cfg.kv_dim() as u64 * 4;
        per_plane * 2 * self.cfg.n_layers as u64 * self.cfg.total_ut_steps as u64
    }

    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.cap as u64
    }
}

fn argmax(s: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in s.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

fn rope_at(g: &Gpu, buf: &DeviceBuffer, n_heads: u32, head_dim: u32, row_stride: u32, pos: u32, theta: f32) -> Step {
    g.step(model::ROPE_AT_ID, &[buf], &[1, n_heads, head_dim, row_stride, 0, pos, gpu_core::f(theta)], n_heads * (head_dim / 2))
}
