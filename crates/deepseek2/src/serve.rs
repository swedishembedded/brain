// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in continuous
// batching of mixture-of-experts decoders then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The paged, batched serving engine of the DeepSeek-V2 MHA decoder: a
//! [`model::serve::PagedDecoder`], so `model::serve::Scheduler` gives it
//! admission, continuous batching, chunked prefill, cancellation and the
//! step timeline without any scheduling code here.
//!
//! # One weight pass per step for every running sequence
//!
//! A decode step carries one row per running sequence. The projections are
//! `[rows, d] x [d, *]` GEMMs, the attention is one paged batched dispatch set
//! (`model::block`'s batched decode over a block-table-addressed pool), the
//! router and the grouped routed experts run over all rows at once
//! ([`DeepseekV2::mlp_rows`], the same MLP half the single-sequence round
//! uses: the MLP is row-wise, so it does not care whose rows they are), and the
//! head is one `[rows, d] x [d, vocab]` GEMM followed by a device argmax or
//! top-k. A sequence joins or leaves between steps; nothing is padded to the
//! longest sequence.
//!
//! # Prefill
//!
//! A prompt is read in chunks of `chunk` rows ([`EngineConfig::chunk`]): one
//! forward per chunk with the chunk's K/V appended through the block table
//! and its queries attending the whole cache so far. Positions of the prompt
//! may be embedding rows from an encoder ([`model::serve::PromptEmbeds`], the
//! vision-language splice): the chunk's token embedding is gathered, then each
//! contiguous run of rows overwrites its positions, exactly as the
//! single-sequence path splices an image block. Only the last chunk of a
//! prompt runs the head, and only on its last row.
//!
//! # Prefix reuse
//!
//! None, and declared so (`prefix_stats` is `(0, 0, 0)`): a document-image
//! prompt starts with the image rows, which differ for every page, so there is
//! no shared prefix to reuse; and with none there is nothing that could serve
//! a cached prefix across differing embedding rows.
//!
//! # What is not here
//!
//! Adapters, speculative decoding, KV offload to host memory: the defaults of
//! [`PagedDecoder`] (none of them applies to the OCR workload).

use checkpoint::TensorSource;
use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{self, ChunkPaging};
use model::kv_tier::{self, KvAppend, KvKernels, KvLayer, KvTier};
use model::ops::PagedDecodeShape;
use model::paged::{BlockAllocator, BlockTable};
use model::readout::{self, ReadoutKernels};
use model::serve::{PagedDecoder, PrefillProgress, PromptEmbeds};

use crate::config::DeepseekV2Config;
use crate::model::{DeepseekV2, MlpScratch, Sizes, ADD2, CONCAT_SPLIT, DECODE_SOFTMAX_BATCHED, EMBED, PIPELINES, ROPE_AT};

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// The shape of one engine: how long a sequence may be, how many tokens of KV
/// the pool holds across all of them, and the widths a forward is sized for.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    /// The most tokens (prompt plus generated) one sequence may hold.
    pub max_seq: u32,
    /// Tokens of K/V the pool holds across every sequence; rounded down to whole blocks.
    pub pool_tokens: u32,
    /// Rows a decode step carries: the most sequences decoded together.
    pub max_batch: u32,
    /// Rows a prefill forward carries.
    pub chunk: u32,
    /// Tokens per KV block.
    pub block_size: u32,
    /// How the pool stores K and V.
    pub kv: KvTier,
    /// Candidates a top-k step can return (the no-repeat filter and sampling read them).
    pub topk: u32,
}

impl EngineConfig {
    /// The shape the OCR decoder is served at: 8192-token sequences (the
    /// checkpoint's own context), 64-token blocks, 512-row prefill chunks.
    pub fn serving(max_batch: u32, pool_tokens: u32) -> EngineConfig {
        EngineConfig { max_seq: 8192, pool_tokens, max_batch, chunk: 512, block_size: 64, kv: KvTier::F32, topk: 64 }
    }
}

/// The pipelines an engine's [`Gpu`] must be built with.
pub fn pipelines() -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&'static str, &'static str)> = PIPELINES.to_vec();
    v.push(("rope_paged", kernels::ROPE_PAGED));
    v.push(("region_copy", kernels::REGION_COPY));
    v.push(("argmax_part", kernels::ARGMAX_PART));
    v.push(("argmax_final", kernels::ARGMAX_FINAL));
    v.push(("topk_extract_step", kernels::TOPK_EXTRACT_STEP));
    let tiers = kv_tier::kernel_list(&v);
    v.extend(tiers);
    v
}

/// Device buffers one forward writes: the per-row indices, the activations and
/// the attention score slabs. Sized once for the widest forward (`rows`).
struct Scratch {
    tok_ids: DeviceBuffer,
    positions: DeviceBuffer,
    blocks: DeviceBuffer,
    offsets: DeviceBuffer,
    seq_lens: DeviceBuffer,
    tables: DeviceBuffer,
    /// The residual stream ping-pongs between two buffers: layers run one after another.
    res: [DeviceBuffer; 2],
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xmid: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    mlp: MlpScratch,
    last_row: DeviceBuffer,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
    /// The prompt chunk's embedding rows, laid out like the chunk's residual.
    emb_stage: DeviceBuffer,
}

/// The paged batched serving engine. See the module doc.
pub struct Engine {
    m: DeepseekV2,
    ec: EngineConfig,
    kvk: KvKernels,
    kv: Vec<KvLayer>,
    alloc: BlockAllocator,
    max_bt: u32,
    sc: Scratch,
    ro: ReadoutKernels,
    rope_paged: usize,
    region_copy: usize,
}

fn u32_buf(g: &Gpu, label: &str, n: u64) -> DeviceBuffer {
    g.buffer(label, n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
}

impl Engine {
    /// Build the engine on `gpu` (made from [`pipelines`]) from a checkpoint's decoder tensors.
    pub fn new(gpu: Gpu, cfg: DeepseekV2Config, src: &dyn TensorSource, ec: EngineConfig) -> Result<Engine, String> {
        let c = cfg.clone();
        let (d, nh, hd, vocab) = (c.d_model(), c.n_heads(), c.head_dim(), c.vocab());
        if ec.max_seq == 0 || ec.max_batch == 0 || ec.chunk == 0 || ec.block_size == 0 || ec.topk == 0 {
            return Err(format!("deepseek2 engine: every size must be positive: {ec:?}"));
        }
        let num_blocks = ec.pool_tokens / ec.block_size;
        if num_blocks == 0 {
            return Err(format!("deepseek2 engine: a pool of {} tokens holds no {}-token block", ec.pool_tokens, ec.block_size));
        }
        if !KvTier::fits_addressing(num_blocks as u64 * ec.block_size as u64, d as u64) {
            return Err(format!("deepseek2 engine: a pool of {} tokens is beyond what one K/V plane addresses", ec.pool_tokens));
        }
        let max_bt = ec.max_seq.div_ceil(ec.block_size);
        // The weights, and the config/kernel handle every step is built from. A
        // 1-row, 1-token build: the engine owns its own K/V pool and scratch.
        let m = DeepseekV2::new_sized(gpu, cfg, Sizes { b: 1, t: 1, ctx: 1, chunk: 1, batched: false }, src, false);
        let g = &m.gpu;
        let kvk = KvKernels::resolve(g, ec.kv)?;
        let rows = ec.max_batch.max(ec.chunk);
        let st = |n: u64| g.storage(n);
        let (r, du) = (rows as u64, d as u64);
        let score_elems = r * nh as u64 * ec.max_seq as u64;
        let sc = Scratch {
            tok_ids: u32_buf(g, "ds2_tok_ids", r),
            positions: u32_buf(g, "ds2_positions", r),
            blocks: u32_buf(g, "ds2_blocks", r),
            offsets: u32_buf(g, "ds2_offsets", r),
            seq_lens: u32_buf(g, "ds2_seq_lens", r),
            tables: u32_buf(g, "ds2_tables", r * max_bt as u64),
            res: [st(r * du), st(r * du)],
            xn1: st(r * du),
            q: st(r * du),
            k: st(r * du),
            v: st(r * du),
            ctx: st(r * du),
            proj: st(r * du),
            xmid: st(r * du),
            scores: st(score_elems),
            probs: st(score_elems),
            mlp: MlpScratch::new(g, &c, rows),
            last_row: st(du),
            xn_final: st(ec.max_batch as u64 * du),
            logits: st(ec.max_batch as u64 * vocab as u64),
            emb_stage: st(ec.chunk as u64 * du),
        };
        let pool_rows = num_blocks as u64 * ec.block_size as u64;
        let kv = (0..c.n_layers()).map(|_| KvLayer::new(g, ec.kv, pool_rows, d as u64, hd as u64)).collect();
        let find = |name: &str| g.kernel_index(name).ok_or_else(|| format!("deepseek2 engine: kernel '{name}' is not registered (build the Gpu from deepseek2::serve::pipelines)"));
        let (rope_paged, region_copy) = (find("rope_paged")?, find("region_copy")?);
        let ro = ReadoutKernels::resolve(g)?;
        Ok(Engine { ec, kvk, kv, alloc: BlockAllocator::new(num_blocks, ec.block_size), max_bt, sc, ro, rope_paged, region_copy, m })
    }

    /// Dispatch the layer stack over the `n` rows already staged in `sc.res[0]`
    /// and return which residual buffer holds the output.
    ///
    /// `attend` records one layer's attention over `q`/`k`/`v` (K/V append
    /// included) into `ctx`; everything else - norms, projections, the MLP - is
    /// the same for a decode step (one row per sequence) and a prefill chunk (one
    /// sequence's rows).
    fn layers(&self, s: &mut Vec<Step>, n: u32, rope: &dyn Fn(&Engine, &mut Vec<Step>, &DeviceBuffer), attend: &dyn Fn(&Engine, &mut Vec<Step>, usize)) -> usize {
        let (c, g, sc) = (&self.m.cfg, &self.m.gpu, &self.sc);
        let d = c.d_model();
        let mut cur = 0usize;
        for l in 0..c.n_layers() as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            self.m.norm_fwd(s, &sc.res[cur], &p("ln1.weight"), &sc.xn1, d, n);
            self.m.mm(s, &sc.xn1, &p("self_attn.q_proj.weight"), &sc.q, n, d, d);
            self.m.mm(s, &sc.xn1, &p("self_attn.k_proj.weight"), &sc.k, n, d, d);
            self.m.mm(s, &sc.xn1, &p("self_attn.v_proj.weight"), &sc.v, n, d, d);
            rope(self, s, &sc.q);
            rope(self, s, &sc.k);
            attend(self, s, l);
            self.m.mm(s, &sc.ctx, &p("self_attn.o_proj.weight"), &sc.proj, n, d, d);
            s.push(g.step(ADD2, &[&sc.res[cur], &sc.proj, &sc.xmid], &[n * d], n * d));
            self.m.mlp_rows(s, l, n, &sc.xmid, &sc.mlp, &sc.res[1 - cur]);
            cur = 1 - cur;
        }
        cur
    }

    /// Final norm and head over `rows` rows of `from` into `logits` (`[rows, vocab]`).
    fn head(&self, s: &mut Vec<Step>, from: &DeviceBuffer, rows: u32) {
        let c = &self.m.cfg;
        self.m.norm_fwd(s, from, "norm.weight", &self.sc.xn_final, c.d_model(), rows);
        self.m.mm(s, &self.sc.xn_final, c.head_weight(), &self.sc.logits, rows, c.d_model(), c.vocab());
    }

    /// One decode step: append each sequence's next token (`inputs[i]`) and leave every
    /// row's logits in `sc.logits` (`[rows, vocab]`).
    fn decode_batch(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) {
        let b = tables.len() as u32;
        assert!(b >= 1 && b <= self.ec.max_batch, "deepseek2 decode: {b} rows exceeds the {}-row batch", self.ec.max_batch);
        assert_eq!(tables.len(), inputs.len(), "deepseek2 decode: tables/inputs length mismatch");
        let bt = self.max_bt as usize;
        let (mut blocks, mut offsets, mut positions, mut seq_lens) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut host_tables = vec![0u32; b as usize * bt];
        for (i, t) in tables.iter_mut().enumerate() {
            let (block, offset) = t.append(&mut self.alloc).expect("the scheduler let a decode step run past the pool");
            blocks.push(block);
            offsets.push(offset);
            positions.push(t.len() - 1);
            seq_lens.push(t.len());
            host_tables[i * bt..i * bt + t.blocks().len()].copy_from_slice(t.blocks());
        }
        let cap = *seq_lens.iter().max().expect("a non-empty batch");
        let (g, sc, c) = (&self.m.gpu, &self.sc, &self.m.cfg);
        g.write(&sc.tok_ids, inputs);
        g.write(&sc.positions, &positions);
        g.write(&sc.blocks, &blocks);
        g.write(&sc.offsets, &offsets);
        g.write(&sc.seq_lens, &seq_lens);
        g.write(&sc.tables, &host_tables);
        let (d, nh, hd) = (c.d_model(), c.n_heads(), c.head_dim());
        let theta = c.rope_theta();
        let mut s: Vec<Step> = Vec::new();
        s.push(g.step(EMBED, &[&sc.tok_ids, self.m.w("tok.weight"), &sc.res[0]], &[d, b], b * d));
        let rope = |e: &Engine, s: &mut Vec<Step>, buf: &DeviceBuffer| {
            s.push(e.m.gpu.step(e.rope_paged, &[buf, &e.sc.positions], &[b, nh, hd, d, f(theta)], b * nh * (hd / 2)));
        };
        let shape = PagedDecodeShape { batch: b, n_heads: nh, group: 1, head_dim: hd, block_size: self.ec.block_size, kv_stride: d, cap, max_bt: self.max_bt, scale: 1.0 / (hd as f32).sqrt() };
        let attend = |e: &Engine, s: &mut Vec<Step>, l: usize| {
            let (g, sc) = (&e.m.gpu, &e.sc);
            let append = KvAppend { batch: b, kv_stride: d, block_size: e.ec.block_size, head_dim: hd };
            s.push(e.kvk.append(g, &e.kv[l].k, &sc.k, &sc.blocks, &sc.offsets, append));
            s.push(e.kvk.append(g, &e.kv[l].v, &sc.v, &sc.blocks, &sc.offsets, append));
            s.push(e.kvk.scores(g, &sc.q, &e.kv[l].k, &sc.tables, &sc.seq_lens, &sc.scores, shape));
            s.push(g.step(DECODE_SOFTMAX_BATCHED, &[&sc.scores, &sc.seq_lens, &sc.probs], &[b, nh, cap], b * nh));
            s.push(e.kvk.apply(g, &sc.probs, &e.kv[l].v, &sc.tables, &sc.seq_lens, &sc.ctx, shape));
        };
        let out = self.layers(&mut s, b, &rope, &attend);
        self.head(&mut s, &self.sc.res[out], b);
        self.m.gpu.submit(&[], &s);
    }

    /// Read the next slice of `prompt` (at most `budget` rows) into `table`; when it is
    /// the last, return the final-normed hidden state of the prompt's last row.
    fn prefill_slice(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: Option<&PromptEmbeds>, budget: usize) -> PrefillProgress {
        let start = table.len();
        let total = prompt.len() as u32;
        assert!(start < total, "deepseek2 prefill: the prompt is already in");
        let n = (budget.min(self.ec.chunk as usize) as u32).min(total - start).max(1);
        table.reserve(n, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        let (bs, bt) = (self.ec.block_size, self.max_bt as usize);
        let (mut blocks, mut offsets, mut seq_lens) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let pos = start + i;
            blocks.push(table.blocks()[(pos / bs) as usize]);
            offsets.push(pos % bs);
            seq_lens.push(pos + 1);
        }
        let mut host_tables = vec![0u32; n as usize * bt];
        for i in 0..n as usize {
            host_tables[i * bt..i * bt + table.blocks().len()].copy_from_slice(table.blocks());
        }
        let (g, sc, c) = (&self.m.gpu, &self.sc, &self.m.cfg);
        g.write(&sc.tok_ids, &prompt[start as usize..(start + n) as usize]);
        g.write(&sc.blocks, &blocks);
        g.write(&sc.offsets, &offsets);
        g.write(&sc.seq_lens, &seq_lens);
        g.write(&sc.tables, &host_tables);
        let (d, nh, hd) = (c.d_model(), c.n_heads(), c.head_dim());
        let theta = c.rope_theta();
        let mut s: Vec<Step> = Vec::new();
        s.push(g.step(EMBED, &[&sc.tok_ids, self.m.w("tok.weight"), &sc.res[0]], &[d, n], n * d));
        // Each contiguous run of embedding rows overwrites its positions of the token embedding.
        if let Some(e) = embeds {
            let mut run: Option<(u32, u32)> = None; // (first row in chunk, rows)
            let mut runs: Vec<(u32, u32)> = Vec::new();
            for (pos, row) in e.rows_in(start, start + n) {
                g.write_f32_at(&sc.emb_stage, ((pos - start) * d) as u64, row);
                run = match run {
                    Some((r0, m)) if r0 + m == pos - start => Some((r0, m + 1)),
                    other => {
                        runs.extend(other);
                        Some((pos - start, 1))
                    }
                };
            }
            runs.extend(run);
            for (r0, m) in runs {
                s.push(g.step(self.region_copy, &[&sc.emb_stage, &sc.res[0]], &[1, m * d, m * d, r0 * d], m * d));
            }
        }
        let rope = |e: &Engine, s: &mut Vec<Step>, buf: &DeviceBuffer| {
            s.push(e.m.gpu.step(ROPE_AT, &[buf], &[n, nh, hd, d, 0, start, f(theta)], n * nh * (hd / 2)));
        };
        let attend = |e: &Engine, s: &mut Vec<Step>, l: usize| {
            let (g, sc) = (&e.m.gpu, &e.sc);
            let paging = ChunkPaging { block_size: e.ec.block_size, max_bt: e.max_bt, blocks: &sc.blocks, offsets: &sc.offsets, tables: &sc.tables };
            s.extend(block::gqa_chunk_step_kv_paged(g, &e.kvk, DECODE_SOFTMAX_BATCHED, nh, nh, hd, start, n, &sc.q, &sc.k, &sc.v, &e.kv[l], &paging, &sc.seq_lens, &sc.ctx));
        };
        let out = self.layers(&mut s, n, &rope, &attend);
        if start + n < total {
            self.m.gpu.submit(&[], &s);
            return PrefillProgress::Partial { consumed: n as usize };
        }
        // The prompt's last row, through the final norm; the head runs when the first token is picked.
        s.push(g.step(CONCAT_SPLIT, &[&self.sc.res[out], &self.sc.last_row], &[1, n * d, d, (n - 1) * d, 1, 1], d));
        self.m.norm_fwd(&mut s, &self.sc.last_row, "norm.weight", &self.sc.xn_final, d, 1);
        self.m.gpu.submit(&[], &s);
        PrefillProgress::Done(self.m.gpu.read(&self.sc.xn_final, d as usize))
    }

    /// Head over one already-normed hidden row into `sc.logits` row 0.
    fn head_of_hidden(&self, hidden: &[f32]) {
        let (g, c) = (&self.m.gpu, &self.m.cfg);
        assert_eq!(hidden.len(), c.d_model() as usize, "deepseek2: a hidden row is d_model floats");
        g.write_f32(&self.sc.xn_final, hidden);
        let mut s: Vec<Step> = Vec::new();
        self.m.mm(&mut s, &self.sc.xn_final, c.head_weight(), &self.sc.logits, 1, c.d_model(), c.vocab());
        g.submit(&[], &s);
    }

    /// Tokens of K/V the pool holds.
    pub fn pool_tokens(&self) -> u32 {
        self.alloc.num_blocks() * self.ec.block_size
    }
}

impl PagedDecoder for Engine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        self.ec.chunk
    }
    fn max_batch(&self) -> Option<usize> {
        Some(self.ec.max_batch as usize)
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        self.ec.max_seq as usize
    }
    fn vocab(&self) -> usize {
        self.m.cfg.vocab() as usize
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.ec.block_size)
    }
    /// There is no prefix cache (see the module doc), so nothing to reclaim.
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(h) = self.prefill_slice(table, prompt, None, usize::MAX) {
                return h;
            }
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(self.ec.chunk as usize)
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        self.prefill_slice(table, prompt, None, budget)
    }
    fn embed_width(&self) -> Option<usize> {
        Some(self.m.cfg.d_model() as usize)
    }
    fn prefill_chunk_embeds(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: &PromptEmbeds, budget: usize) -> PrefillProgress {
        self.prefill_slice(table, prompt, Some(embeds), budget)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.head_of_hidden(hidden);
        self.m.gpu.read(&self.sc.logits, self.m.cfg.vocab() as usize)
    }
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        self.head_of_hidden(hidden);
        readout::argmax_rows(&self.m.gpu, self.ro, &self.sc.logits, 1, self.m.cfg.vocab())[0]
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.head_of_hidden(hidden);
        let k = k.clamp(1, self.ec.topk as usize) as u32;
        readout::topk_rows(&self.m.gpu, self.ro, &self.sc.logits, 1, self.m.cfg.vocab(), k).remove(0)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        if tables.is_empty() {
            return Vec::new();
        }
        self.decode_batch(tables, inputs);
        readout::argmax_rows(&self.m.gpu, self.ro, &self.sc.logits, tables.len() as u32, self.m.cfg.vocab())
    }
    /// No on-device multi-step schedule: the window is [`Self::decode_window_capacity`] = 1.
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        let mut next = inputs.to_vec();
        let mut out = vec![Vec::new(); tables.len()];
        for _ in 0..k.max(1) {
            next = self.forward_batched_greedy(tables, &next);
            for (o, &t) in out.iter_mut().zip(&next) {
                o.push(t);
            }
        }
        out
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        if tables.is_empty() {
            return Vec::new();
        }
        self.decode_batch(tables, inputs);
        let k = k.clamp(1, self.ec.topk as usize) as u32;
        readout::topk_rows(&self.m.gpu, self.ro, &self.sc.logits, tables.len() as u32, self.m.cfg.vocab(), k)
    }
    fn topk_capacity(&self) -> usize {
        self.ec.topk as usize
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.m.gpu.stats()
    }
    fn kv_pool_bytes(&self) -> u64 {
        let rows = self.pool_tokens() as u64;
        let d = self.m.cfg.d_model() as u64;
        2 * self.m.cfg.n_layers() as u64 * self.ec.kv.plane_bytes(rows, d, self.m.cfg.head_dim() as u64)
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.pool_tokens() as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
}
