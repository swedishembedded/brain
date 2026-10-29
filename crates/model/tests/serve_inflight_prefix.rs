// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for LLM serving that prefills a
// shared document once for every concurrent request instead of once each, for
// its clients. If your team needs expertise in prefix-aware continuous
// batching, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: requests that arrive together and share a long prefix prefill it once.
//! A waiting request whose prompt shares at least two prefill slices with a
//! sequence still prefilling them is held back while the decoder agrees to make
//! that prefix resumable ([`PagedDecoder::hint_shared_prefix`]); once the
//! prefill has passed it, the request is admitted and resumes from it. A
//! decoder that does not agree keeps the old behaviour: nothing is held back.
//! Either way every request produces the tokens it produces alone.
//!
//! The decoder here keeps real KV contents and shares cached blocks through
//! `model::paged::PrefixCache`, so a wrongly adopted prefix yields different
//! tokens than a cold run.

use std::collections::HashMap;

use model::paged::{BlockAllocator, BlockTable, PrefixCache};
use model::serve::{AdapterId, PagedDecoder, PrefillProgress, Request, Scheduler};

const VOCAB: usize = 64;
const BLOCK: u32 = 4;
const SLICE: usize = 8;

struct Mock {
    alloc: BlockAllocator,
    prefix: PrefixCache,
    kv: HashMap<u32, Vec<u32>>,
    hit_tokens: u64,
    /// Whether this decoder honours [`PagedDecoder::hint_shared_prefix`].
    hints: bool,
    /// Whether several sequences prefill per iteration (so requests arriving together are admitted together, as on
    /// a decoder that prefills in the background) or one prompt at a time.
    batched: bool,
    /// Prefixes (namespace, tokens) to cache as soon as a prefill has written them.
    wanted: Vec<(u32, Vec<u32>)>,
}

impl Mock {
    fn new(hints: bool, batched: bool) -> Mock {
        Mock { alloc: BlockAllocator::new(512, BLOCK), prefix: PrefixCache::new(), kv: HashMap::new(), hit_tokens: 0, hints, batched, wanted: Vec::new() }
    }

    fn write(&mut self, block: u32, offset: u32, token: u32) {
        let slots = self.kv.entry(block).or_default();
        slots.truncate(offset as usize);
        slots.push(token);
    }

    fn history(&self, table: &BlockTable) -> Vec<u32> {
        (0..table.len()).map(|i| self.kv[&table.blocks()[(i / BLOCK) as usize]][(i % BLOCK) as usize]).collect()
    }

    /// Index the whole blocks of a wanted prefix this table has now written.
    fn file_wanted(&mut self, table: &BlockTable, prompt: &[u32]) {
        let ns = table.kv_namespace();
        let written = table.len() as usize;
        let done: Vec<usize> = (0..self.wanted.len()).filter(|&i| self.wanted[i].0 == ns && self.wanted[i].1.len() <= written && prompt.starts_with(&self.wanted[i].1)).collect();
        for &i in done.iter().rev() {
            let (_, tokens) = self.wanted.remove(i);
            let full = tokens.len() / BLOCK as usize;
            let from = (table.cached_rows() / BLOCK) as usize;
            if full > from {
                self.prefix.insert_chain_in(ns, &tokens[..full * BLOCK as usize], &table.blocks()[..full], from, &mut self.alloc);
            }
        }
    }
}

fn digest(history: &[u32]) -> u32 {
    let h = history.iter().fold(7u64, |h, &t| h.wrapping_mul(1_000_003) ^ (t as u64 + 1));
    (h >> 5) as u32 % VOCAB as u32
}

fn namespace(adapter: Option<AdapterId>) -> u32 {
    adapter.map_or(model::paged::BASE_KV_NAMESPACE, |a| a.kv_namespace())
}

impl PagedDecoder for Mock {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        SLICE as u32
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        4096
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        self.prefix.peek_in(namespace(adapter), prompt, BLOCK, prompt.len().saturating_sub(1)).len() as u32
    }
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        self.prefix.evict(want, &mut self.alloc)
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(SLICE)
    }
    fn prefill_batch_tokens(&self) -> Option<usize> {
        self.batched.then_some(4 * SLICE)
    }
    fn hint_shared_prefix(&mut self, table: &mut BlockTable, prompt: &[u32], at: usize) -> bool {
        if !self.hints || table.len() as usize >= at {
            return false;
        }
        self.wanted.push((table.kv_namespace(), prompt[..at].to_vec()));
        true
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        if table.is_empty() {
            let hits = self.prefix.lookup_in(table.kv_namespace(), prompt, BLOCK, prompt.len() - 1);
            self.hit_tokens += hits.len() as u64 * BLOCK as u64;
            if !hits.is_empty() {
                table.adopt_prefix(&hits, &mut self.alloc);
            }
        }
        let from = table.len() as usize;
        let to = (from + budget.max(1)).min(prompt.len());
        let start = table.reserve((to - from) as u32, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        for (i, &t) in prompt[from..to].iter().enumerate() {
            let pos = start + i as u32;
            self.write(table.blocks()[(pos / BLOCK) as usize], pos % BLOCK, t);
        }
        self.file_wanted(table, prompt);
        if to == prompt.len() {
            PrefillProgress::Done(vec![digest(&self.history(table)) as f32])
        } else {
            PrefillProgress::Partial { consumed: to - from }
        }
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(h) = self.prefill_chunk(table, prompt, usize::MAX) {
                return h;
            }
        }
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[hidden[0] as usize % VOCAB] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &input)| {
                let (block, offset) = t.append(&mut self.alloc).expect("pool");
                self.write(block, offset, input);
                digest(&self.history(t))
            })
            .collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("these requests decode greedily")
    }
    fn topk_capacity(&self) -> usize {
        1
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (self.hit_tokens, 0, self.prefix.len())
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        None
    }
    fn kv_pool_bytes(&self) -> u64 {
        0
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.alloc.num_blocks() as u64 * BLOCK as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
}

/// Four questions about one 64-token document (eight slices of it), all submitted at once.
fn questions() -> Vec<Vec<u32>> {
    let doc: Vec<u32> = (0..64u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    (1..=4u32).map(|q| doc.iter().copied().chain((0..8).map(|i| (q * 11 + i * 5) % VOCAB as u32)).collect()).collect()
}

fn run(hints: bool, batched: bool) -> (Vec<Vec<u32>>, u64) {
    let mut sched = Scheduler::new(Mock::new(hints, batched), 4);
    let ids: Vec<u64> = questions().into_iter().map(|prompt| sched.submit(Request { prompt, max_new: 6, stop: Vec::new() })).collect();
    let mut done = HashMap::new();
    let mut steps = 0;
    while sched.pending() {
        done.extend(sched.step());
        steps += 1;
        assert!(steps < 1_000, "the scheduler stopped making progress");
    }
    (ids.iter().map(|id| done.remove(id).expect("finished")).collect(), sched.prefix_stats().0)
}

fn alone(prompt: Vec<u32>) -> Vec<u32> {
    let mut sched = Scheduler::new(Mock::new(false, false), 1);
    let id = sched.submit(Request { prompt, max_new: 6, stop: Vec::new() });
    sched.run().remove(&id).expect("finished")
}

#[test]
fn requests_sharing_a_document_prefill_it_once() {
    for batched in [true, false] {
        let (got, hits) = run(true, batched);
        assert_eq!(hits, 3 * 64, "batched {batched}: the three later questions resume from the first one's document");
        for (toks, prompt) in got.iter().zip(questions()) {
            assert_eq!(toks, &alone(prompt), "a request resumed from an in-flight prefix produces its solo tokens");
        }
    }
}

#[test]
fn a_decoder_that_declines_the_hint_holds_nobody_back() {
    let (got, hits) = run(false, true);
    assert_eq!(hits, 0, "admitted together, nothing is cached yet for any of them");
    for (toks, prompt) in got.iter().zip(questions()) {
        assert_eq!(toks, &alone(prompt));
    }
}
