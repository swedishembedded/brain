// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving engines for looped and hybrid decoders for its clients. If your team needs
// expertise in continuous batching for models that revisit their layers then you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec of `model::serve::visit_state` through a tiny looped decoder built on it: 2 prelude + 2 core x 2 passes + 1
//! coda layers, odd layers attending their whole history and even ones a window of [`WINDOW`] keys, every visit with
//! three short convolution states (on K, on V and on the attention output), hidden-conditioned attention sinks,
//! random weights. The decoder exists only here: it is the conformance subject of the building block.
//!
//! - it passes the generic serving conformance suite (batched == alone, cancel frees state, a prefix hit prefills only
//!   the suffix and answers as a cold prefill, full rows, keyed draws), with prefills sliced;
//! - every logit it serves, prefilled in slices and decoded token by token far past the window, equals one
//!   whole-sequence forward of the same weights that keeps no state (contiguous windowed attention, convolutions
//!   without history): each ring holds exactly its window and each convolution state its taps;
//! - a prefix hit restores every visit's ring and convolution state and shares the full visits' blocks: what it then
//!   prefills and decodes equals a cold run, and a repeat of the cached prompt computes nothing yet decodes the same;
//! - a speculative verify (the proposals run as one chunk past the state row's checkpoint) that keeps only some of its
//!   rows leaves every ring and convolution state as decoding the kept rows alone does: the undo row restores what the
//!   verify overwrote (ring rows a window back, convolution inputs) and the kept rows are recomputed, so what the
//!   sequence serves next equals the stateless forward, and speculative greedy decoding equals plain greedy decoding.

use std::sync::OnceLock;

use data::rng::{random_seed, Lcg};
use gpu_core::{DeviceBuffer, Gpu};
use model::flash_ext::{flash_ext_fwd, AttnInputs, AttnOptions, Contiguous, HeadShape};
use model::hostmath::{argmax_row, linear_rows};
use model::kv_tier::KvTier;
use model::looped::{Attention, LoopShape, VisitPlan};
use model::paged::{BlockAllocator, BlockTable, KvNamespace, BASE_KV_NAMESPACE};
use model::serve::conformance::{self, Conformance};
use model::serve::visit_state::{ConvState, VisitLayout, VisitPool, VisitPoolSizing, VisitStateShape};
use model::serve::{AdapterId, PagedDecoder, PrefillProgress};
use model::short_conv::{conv_state_tail_step, short_conv_fwd, short_conv_step, BaseLayout, ShortConvIds, ShortConvShape};

const VOCAB: usize = 48;
const D: usize = 16;
const NH: u32 = 2;
const NKV: u32 = 1;
const HD: u32 = 8;
const QD: usize = (NH * HD) as usize;
const KVD: usize = (NKV * HD) as usize;
const FF: usize = 24;
const WINDOW: u32 = 5;
const TAPS: u32 = 4;
/// `(conv(x) + x) * CONV_SCALE`, the published short convolutions' residual form.
const CONV_SCALE: f32 = 0.447_213_6;
const BLOCK: u32 = 8;
const MAX_SEQ: usize = 96;
/// Rows of one prefill forward.
const ROUND: usize = 32;

fn shape() -> LoopShape {
    LoopShape { prelude: 2, core: 2, coda: 1, passes: 2 }
}

fn attention(layer: usize) -> Attention {
    if layer % 2 == 1 { Attention::Full } else { Attention::Window(WINDOW) }
}

/// The convolution states of every visit, by index: K, V, the attention output.
fn conv_states() -> Vec<ConvState> {
    vec![ConvState { channels: KVD as u32, taps: TAPS }, ConvState { channels: KVD as u32, taps: TAPS }, ConvState { channels: D as u32, taps: TAPS }]
}

struct Layer {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    ws: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    /// Channel-major `[c, taps]` bases of the K, V and output convolutions.
    conv: [Vec<f32>; 3],
}

struct Weights {
    embed: Vec<f32>,
    head: Vec<f32>,
    layers: Vec<Layer>,
}

/// One set of weights per process, from a random seed printed for a failure to reproduce: every decoder the suite
/// builds must be the same model.
fn weights() -> &'static Weights {
    static W: OnceLock<Weights> = OnceLock::new();
    W.get_or_init(|| {
        let seed = random_seed();
        eprintln!("serve_visit_state weights seed: {seed}");
        let mut rng = Lcg::new(seed);
        let mut m = |rows: usize, cols: usize| rng.vec_scaled(rows * cols, 1.5 / (cols as f32).sqrt());
        let embed = m(VOCAB, D);
        let head = m(VOCAB, D);
        let layers = (0..shape().layers())
            .map(|_| Layer { wq: m(QD, D), wk: m(KVD, D), wv: m(KVD, D), ws: m(NH as usize, D), wo: m(D, QD), w1: m(FF, D), w2: m(D, FF), conv: [m(KVD, TAPS as usize), m(KVD, TAPS as usize), m(D, TAPS as usize)] })
            .collect();
        Weights { embed, head, layers }
    })
}

fn kernels() -> &'static [(&'static str, &'static str)] {
    static K: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    K.get_or_init(VisitPool::<Vec<f32>>::pipelines).as_slice()
}

fn heads() -> HeadShape {
    HeadShape { n_heads: NH, n_kv_heads: NKV, head_dim: HD }
}

fn options(attention: Attention) -> AttnOptions {
    let window = match attention {
        Attention::Full => None,
        Attention::Window(n) => Some(n),
    };
    AttnOptions { window, sink: true, ..AttnOptions::plain(1.0 / (HD as f32).sqrt()) }
}

/// Which rows a forward runs and where their state lives.
enum Rows<'a> {
    /// One prompt chunk of one served sequence, at positions `start ..`.
    Chunk { table: &'a BlockTable, start: u32 },
    /// One new token of each served sequence.
    Decode { tables: &'a [&'a BlockTable] },
    /// A whole sequence from position 0 with no state at all: the reference.
    Whole,
}

/// The tiny looped decoder over a [`VisitPool`].
struct Looped {
    g: Gpu,
    conv_ids: ShortConvIds,
    plan: VisitPlan,
    /// `[layer][state]` device copies of the convolution bases.
    bases: Vec<[DeviceBuffer; 3]>,
    pool: VisitPool<Vec<f32>>,
    interleave: Option<usize>,
    /// The verify awaiting `commit_verified`: each sequence's rows and first position.
    pending: Vec<(Vec<u32>, u32)>,
}

impl Looped {
    fn new(interleave: Option<usize>) -> Looped {
        Self::on(KvTier::F32, interleave)
    }

    /// K/V (paged planes and rings) held at `kv_tier`.
    fn on(kv_tier: KvTier, interleave: Option<usize>) -> Looped {
        let sizing = VisitPoolSizing { block_size: BLOCK, num_blocks: 64, live_rows: 4, snapshot_rows: 4, undo_rows: true };
        let g = gpu_core::testgpu::dev(kernels());
        let plan = VisitPlan::new(shape(), attention).expect("a valid plan");
        let layout = VisitLayout::new(plan.clone(), VisitStateShape { n_kv_heads: NKV, head_dim: HD, kv_tier, conv: conv_states() }).expect("a valid layout");
        let pool = VisitPool::new(&g, layout, sizing).expect("the pool fits").with_min_cached_tokens(8);
        let bases = weights().layers.iter().map(|l| [0, 1, 2].map(|i| g.storage_init("conv.base", &l.conv[i]))).collect();
        Looped { conv_ids: ShortConvIds::resolve(&g), g, plan, bases, pool, interleave, pending: Vec::new() }
    }

    fn upload(&self, x: &[f32]) -> DeviceBuffer {
        self.g.storage_init("rows", x)
    }

    /// Convolution `state` of visit `visit` over `n` rows `x` of `c` channels, carrying the rows' state.
    fn conv(&self, visit: usize, state: usize, rows: &Rows, x: &DeviceBuffer, c: u32, n: u32) -> DeviceBuffer {
        let base = &self.bases[self.plan.visits()[visit].layer][state];
        let y = self.g.storage(u64::from(n * c));
        let g = &self.g;
        match rows {
            Rows::Whole => {
                let s = ShortConvShape::static_taps(n, c, TAPS, BaseLayout::ChannelMajor).residual_scaled(CONV_SCALE);
                g.submit(&[], &[short_conv_fwd(g, &self.conv_ids, &s, x, None, base, None, None, &y)]);
            }
            Rows::Chunk { table, .. } => {
                let hist = self.pool.conv_rows(g, visit, state, &[table]);
                let s = ShortConvShape::static_taps(n, c, TAPS, BaseLayout::ChannelMajor).residual_scaled(CONV_SCALE).with_history();
                let tail = g.storage(u64::from(c * (TAPS - 1)));
                g.submit(&[], &[short_conv_fwd(g, &self.conv_ids, &s, x, None, base, None, Some(&hist), &y), conv_state_tail_step(g, &self.conv_ids, n, c, TAPS, x, &hist, &tail)]);
                self.pool.store_conv_rows(g, visit, state, &[table], &tail);
            }
            Rows::Decode { tables } => {
                let hist = self.pool.conv_rows(g, visit, state, tables);
                let s = ShortConvShape::static_taps(n, c, TAPS, BaseLayout::ChannelMajor).residual_scaled(CONV_SCALE);
                g.submit(&[], &[short_conv_step(g, &self.conv_ids, &s, x, None, base, &hist, &y)]);
                self.pool.store_conv_rows(g, visit, state, tables, &hist);
            }
        }
        y
    }

    /// Every visit over `n` rows `x` (`[n, D]`), in place.
    fn layers(&self, rows: &Rows, x: &mut [f32], n: usize) {
        let w = weights();
        let g = &self.g;
        for visit in self.plan.visits() {
            let l = &w.layers[visit.layer];
            let q = self.upload(&linear_rows(x, &l.wq, n, D, QD));
            let k = self.conv(visit.index, 0, rows, &self.upload(&linear_rows(x, &l.wk, n, D, KVD)), KVD as u32, n as u32);
            let v = self.conv(visit.index, 1, rows, &self.upload(&linear_rows(x, &l.wv, n, D, KVD)), KVD as u32, n as u32);
            let sink = self.upload(&linear_rows(x, &l.ws, n, D, NH as usize));
            let (opts, inputs) = (options(visit.attention), AttnInputs { sink: Some(&sink), ..AttnInputs::default() });
            let ctx = match rows {
                Rows::Chunk { table, start } => self.pool.attend_chunk(g, visit.index, table, *start, &heads(), &opts, &inputs, &q, &k, &v, n as u32),
                Rows::Decode { tables } => self.pool.attend_decode(g, visit.index, tables, &heads(), &opts, &inputs, &q, &k, &v),
                Rows::Whole => {
                    let (ctx, lse) = (g.storage((n * QD) as u64), g.storage((n * NH as usize) as u64));
                    g.submit(&[], &flash_ext_fwd(g, &self.pool_flash(), &heads(), &opts, &inputs, Contiguous { b: 1, t: n as u32, pos0: 0 }, &q, &k, &v, &ctx, &lse));
                    ctx
                }
            };
            let o = linear_rows(&g.read(&ctx, n * QD), &l.wo, n, QD, D);
            let o = g.read(&self.conv(visit.index, 2, rows, &self.upload(&o), D as u32, n as u32), n * D);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            let h: Vec<f32> = linear_rows(x, &l.w1, n, D, FF).into_iter().map(f32::tanh).collect();
            for (xi, mi) in x.iter_mut().zip(linear_rows(&h, &l.w2, n, FF, D)) {
                *xi += mi;
            }
        }
    }

    fn pool_flash(&self) -> model::flash_ext::FlashExtIds {
        model::flash_ext::FlashExtIds::resolve(&self.g)
    }

    fn embed(tokens: &[u32]) -> Vec<f32> {
        tokens.iter().flat_map(|&t| weights().embed[t as usize * D..(t as usize + 1) * D].iter().copied()).collect()
    }

    fn head(x: &[f32], n: usize) -> Vec<Vec<f32>> {
        linear_rows(x, &weights().head, n, D, VOCAB).chunks(VOCAB).map(<[f32]>::to_vec).collect()
    }

    /// The final hidden row of every position of `tokens` run as one whole sequence with no state: the reference.
    fn whole(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let mut x = Self::embed(tokens);
        self.layers(&Rows::Whole, &mut x, tokens.len());
        x.chunks(D).map(<[f32]>::to_vec).collect()
    }

    /// One decode step of every sequence: the logits rows.
    fn decode(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        assert_eq!(tables.len(), inputs.len());
        for t in tables.iter_mut() {
            self.pool.append(t);
        }
        let refs: Vec<&BlockTable> = tables.iter().map(|t| &**t).collect();
        let mut x = Self::embed(inputs);
        self.layers(&Rows::Decode { tables: &refs }, &mut x, inputs.len());
        Self::head(&x, inputs.len())
    }

    /// The rows of one chunk of a served sequence at positions `start ..` (backed), returning its final hidden rows.
    fn chunk(&self, table: &BlockTable, start: u32, tokens: &[u32]) -> Vec<f32> {
        let mut x = Self::embed(tokens);
        self.layers(&Rows::Chunk { table, start }, &mut x, tokens.len());
        x
    }

    /// A verify: each sequence's rows run as one chunk past its checkpointed state row.
    fn verify(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>]) -> Vec<Vec<Vec<f32>>> {
        let refs: Vec<&BlockTable> = tables.iter().map(|t| &**t).collect();
        self.pool.checkpoint(&self.g, &refs);
        self.pending.clear();
        let mut out = Vec::with_capacity(rows.len());
        for (t, r) in tables.iter_mut().zip(rows) {
            let start = self.pool.reserve(t, r.len() as u32);
            out.push(Self::head(&self.chunk(t, start, r), r.len()));
            self.pending.push((r.clone(), start));
        }
        out
    }
}

fn namespace(adapter: Option<AdapterId>) -> KvNamespace {
    adapter.map_or(BASE_KV_NAMESPACE, AdapterId::kv_namespace)
}

fn top_k(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut c: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    c.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    c.truncate(k);
    c
}

impl PagedDecoder for Looped {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        self.pool.alloc_mut()
    }
    fn max_prefill_tokens(&self) -> u32 {
        ROUND as u32
    }
    fn max_batch(&self) -> Option<usize> {
        Some(self.pool.max_batch())
    }
    fn free_blocks(&self) -> u32 {
        self.pool.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        MAX_SEQ
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        self.pool.blocks_for(tokens)
    }
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        self.pool.cached_prefix_blocks(namespace(adapter), prompt)
    }
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        self.pool.reclaim(want)
    }
    fn reclaim_prefix_keeping(&mut self, want: u32, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        self.pool.reclaim_keeping(want, namespace(adapter), prompt)
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        self.pool.release(t);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(hidden) = self.prefill_chunk(table, prompt, usize::MAX) {
                return hidden;
            }
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        self.interleave
    }
    /// The first slice takes a state row and resumes from the prefix cache; the slice that ends the prompt files it.
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        assert!(!prompt.is_empty() && prompt.len() <= MAX_SEQ && prompt.iter().all(|&t| (t as usize) < VOCAB), "a prompt the scheduler admitted");
        if table.state_row().is_none() {
            if let Some(hidden) = self.pool.begin(&self.g, table, prompt).exact {
                return PrefillProgress::Done(hidden);
            }
        }
        let done = table.len() as usize;
        let n = budget.clamp(1, ROUND).min(prompt.len() - done);
        let start = self.pool.reserve(table, n as u32);
        let mut x = Self::embed(&prompt[done..done + n]);
        self.layers(&Rows::Chunk { table, start }, &mut x, n);
        if done + n < prompt.len() {
            return PrefillProgress::Partial { consumed: n };
        }
        let hidden = x[(n - 1) * D..].to_vec();
        self.pool.file_prefix(&self.g, table, prompt, hidden.clone());
        PrefillProgress::Done(hidden)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        Self::head(hidden, 1).remove(0)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.decode(tables, inputs).iter().map(|r| argmax_row(r)).collect()
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        assert!((1..=self.decode_window_capacity()).contains(&k));
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); inputs.len()];
        let mut next = inputs.to_vec();
        for _ in 0..k {
            next = self.forward_batched_greedy(tables, &next);
            for (o, &t) in out.iter_mut().zip(&next) {
                o.push(t);
            }
        }
        out
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, self.topk_capacity());
        self.decode(tables, inputs).iter().map(|r| top_k(r, k)).collect()
    }
    fn full_row_readback(&self) -> bool {
        true
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        self.decode(tables, inputs)
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        self.pool.prefix_stats()
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.g.stats()
    }
    fn kv_pool_bytes(&self) -> u64 {
        self.pool.pool_bytes()
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.pool.capacity_tokens()
    }
    fn decode_window_capacity(&self) -> usize {
        4
    }
    fn topk_capacity(&self) -> usize {
        16
    }
    fn spec_max_draft(&self) -> usize {
        4
    }
    fn forward_batched_verify(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>], cands: usize) -> Vec<Vec<Vec<(u32, f32)>>> {
        let k = cands.clamp(1, self.topk_capacity());
        self.verify(tables, rows).iter().map(|seq| seq.iter().map(|r| top_k(r, k)).collect()).collect()
    }
    fn forward_batched_verify_logit_rows(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>]) -> Vec<Vec<Vec<f32>>> {
        self.verify(tables, rows)
    }
    /// A sequence that keeps every row is settled; one that keeps fewer restores its state row and recomputes the
    /// kept rows over it.
    fn commit_verified(&mut self, tables: &mut [&mut BlockTable], keep: &[usize]) {
        let pending = std::mem::take(&mut self.pending);
        for ((t, &kept), (rows, start)) in tables.iter().zip(keep).zip(&pending) {
            if kept < rows.len() {
                self.pool.rollback(&self.g, t);
                self.chunk(t, *start, &rows[..kept]);
            }
        }
    }
}

/// Prompts in slices of seven tokens, so admission, preemption and prefix reuse all meet partial prefills.
fn sliced() -> Looped {
    Looped::new(Some(7))
}

#[test]
fn the_looped_decoder_passes_the_serving_conformance_suite() {
    conformance::run_all(&Conformance { make: sliced, prefix_reuse: true, prefix_prompt_tokens: 20, load_adapter: None });
}

/// The same with every K/V row rounded to bf16 where it is stored (the paged planes and the rings).
#[test]
fn the_looped_decoder_passes_the_serving_conformance_suite_over_bf16_kv() {
    conformance::run_all(&Conformance { make: || Looped::on(KvTier::BF16, Some(7)), prefix_reuse: true, prefix_prompt_tokens: 20, load_adapter: None });
}

/// Fixed in-vocabulary tokens for a sequence of `n`.
fn sequence(n: usize, salt: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 11 + salt * 5 + i * i) % (VOCAB - 1) + 1) as u32).collect()
}

/// Prefill `prompt` (in slices of `slice`) on a fresh sequence, then decode `rest` teacher-forced: the logits after
/// the prompt and after each decoded token.
fn served(e: &mut Looped, table: &mut BlockTable, prompt: &[u32], slice: usize, rest: &[u32]) -> Vec<Vec<f32>> {
    let hidden = loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk(table, prompt, slice) {
            break h;
        }
    };
    let mut out = vec![e.logits(&hidden)];
    for &t in rest {
        out.push(e.decode(&mut [table], &[t]).remove(0));
    }
    out
}

/// Spec: a sequence prefilled in slices and decoded token by token far past the window serves, at every position,
/// exactly the logits one stateless whole-sequence forward gives: each ring of [`WINDOW`] rows and each convolution
/// state carried every key and input the reference reads, and a state row another sequence used first starts clean.
/// The rings stay the window's size however long it runs.
#[test]
fn served_logits_equal_the_stateless_whole_sequence_forward() {
    let tokens = sequence(29, 1);
    let (prompt, rest) = tokens.split_at(11);
    for slice in [3, 11] {
        let mut e = Looped::new(None);
        assert!(e.pool.layout().ring_rows().iter().all(|&r| r == WINDOW), "every windowed visit keeps exactly its window");
        let whole = e.whole(&tokens);
        // Another sequence first: the target then runs on the state row it left (rings and convolutions written).
        let mut other = BlockTable::new();
        served(&mut e, &mut other, &sequence(13, 9), 4, &sequence(6, 8));
        e.release_table(&mut other);
        let mut table = BlockTable::new();
        let got = served(&mut e, &mut table, prompt, slice, &rest[..rest.len() - 1]);
        for (i, row) in got.iter().enumerate() {
            let p = prompt.len() - 1 + i;
            assert_eq!(row, &e.logits(&whole[p]), "slices of {slice}: logits at position {p} differ from the whole-sequence forward");
        }
        e.release_table(&mut table);
        assert_eq!(e.free_blocks(), 64 - e.prefix_stats().2 as u32, "a released sequence holds nothing; the cache holds its entry's blocks");
    }
}

/// Spec: a prompt extending a cached one resumes from it (whole blocks shared, the partly filled tail copied, every
/// ring and convolution state restored) and serves what a cold run serves; a repeat of the cached prompt computes
/// nothing and decodes the same as a cold run, twice (the entry survives the sequences that copied its tail block and
/// wrote past it).
#[test]
fn a_prefix_hit_restores_every_visits_state_exactly() {
    let tokens = sequence(34, 2);
    let (cached, tail) = tokens.split_at(20);
    let (extension, rest) = tail.split_at(6);

    let cold = |prompt: &[u32], rest: &[u32]| {
        let mut e = Looped::new(None);
        let mut t = BlockTable::new();
        served(&mut e, &mut t, prompt, 7, rest)
    };
    let mut warm = Looped::new(None);
    let mut first = BlockTable::new();
    served(&mut warm, &mut first, cached, 7, &[]);
    warm.release_table(&mut first);
    assert_eq!(warm.prefix_stats().2, 3, "the entry holds the cached prompt's 20 rows: three blocks of every full visit");

    for round in 0..2 {
        let mut table = BlockTable::new();
        let got = served(&mut warm, &mut table, cached, 7, extension);
        assert_eq!(table.cached_rows(), 20);
        assert_eq!(got, cold(cached, extension), "repeat {round} of the cached prompt decodes as a cold run");
        warm.release_table(&mut table);
    }

    let extended: Vec<u32> = cached.iter().chain(extension).copied().collect();
    let mut table = BlockTable::new();
    let got = served(&mut warm, &mut table, &extended, 7, rest);
    assert_eq!(table.cached_rows(), 20, "the extension resumed from the cached prompt");
    assert_eq!(got, cold(&extended, rest), "a prefix hit serves what a cold run serves");
    warm.release_table(&mut table);
    let (hits, _, _) = warm.prefix_stats();
    assert_eq!(hits, 60, "three hits of 20 tokens");
}

/// Spec: a verify of five rows that keeps `kept` of them leaves the sequence where decoding the kept rows alone would:
/// the verify wrapped every ring (window [`WINDOW`]) and moved every convolution state past what is kept, and the
/// logits served afterwards, decoded far on, equal the stateless whole-sequence forward at every position.
#[test]
fn a_verify_that_keeps_part_of_its_rows_leaves_every_ring_and_convolution_as_decoding_them() {
    let tokens = sequence(30, 3);
    let (prompt, rest) = tokens.split_at(9);
    for kept in [1, 3, 5] {
        let mut e = Looped::new(None);
        let whole = e.whole(&tokens);
        let mut table = BlockTable::new();
        served(&mut e, &mut table, prompt, 4, &[]);
        // The kept rows are the sequence's next tokens; the rest are wrong guesses.
        let mut rows: Vec<u32> = rest[..5].to_vec();
        for r in rows.iter_mut().skip(kept) {
            *r = (*r % (VOCAB as u32 - 1)) + 1;
        }
        let verified = e.verify(&mut [&mut table], &[rows]).remove(0);
        for (j, row) in verified.iter().take(kept).enumerate() {
            assert_eq!(row, &e.logits(&whole[prompt.len() + j]), "kept {kept}: verify row {j}");
        }
        e.commit_verified(&mut [&mut table], &[kept]);
        let len = prompt.len() as u32 + kept as u32;
        table.truncate(len, e.pool.alloc_mut());
        for (i, &t) in rest[kept..rest.len() - 1].iter().enumerate() {
            let p = prompt.len() + kept + i;
            let got = e.decode(&mut [&mut table], &[t]).remove(0);
            assert_eq!(got, e.logits(&whole[p]), "kept {kept}: position {p} after the verify differs from the stateless forward");
        }
        e.release_table(&mut table);
    }
}
