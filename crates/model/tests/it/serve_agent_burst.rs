// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements LLM serving that keeps a card's batch wide under bursts of long agent conversations.
// If your team needs expertise in paged-KV continuous batching for agent fleets, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a burst of agent conversations (contexts of 5-30k tokens resent every turn, a long shared system prompt,
//! closed-loop turns) against a KV pool the size of a production Qwen3-4B's (196k tokens in 16-token blocks) on a card
//! with room for every sequence the pool holds never runs the pool dry. The blocks a prompt was admitted to adopt from
//! the prefix cache are not reclaimed out from under it by the admissions and the decode room that follow it, so the
//! pool's own accounting holds, no sequence is sent back to be recomputed for want of a block admission promised it,
//! and the batch stays as wide as the pool holds instead of collapsing to a handful.
//!
//! Observed 2026-11-27 on GH200 servers (knowledge #613): "paged KV pool exhausted" over and over, each time read as
//! the card's memory limit, the running width halved to 25, 11, 8, 5 while the pool read almost entirely free.
//!
//! The decoder is host bookkeeping only: the real allocator and prefix cache, prompts adopted from the cache at the
//! start of their prefill exactly as the Qwen3 engine does, and nothing else that could run out.

use model::paged::{BlockAllocator, BlockTable, PrefixCache};
use model::serve::{PagedDecoder, Request, Scheduler};

const VOCAB: usize = 1 << 20;
const BLOCK: u32 = 16;
/// The pool of the Qwen3-4B server observed: 12261 blocks of 16 tokens (`kv_total_tokens` 196176).
const POOL_BLOCKS: u32 = 12_261;
const MAX_BATCH: usize = 64;
const PREFILL_ROUND: u32 = 4096;
const SYSTEM_PROMPT: usize = 2048;

/// A decoder whose only limit is its KV pool, with Qwen3's prefix adoption.
struct Agentic {
    alloc: BlockAllocator,
    prefix: PrefixCache,
    /// Allocations the pool could not back: each one sends the sequences of the call back to the queue.
    dry: usize,
    /// Prompt rows computed (not adopted from the cache).
    prefill_rows: u64,
    /// Tokens decoded.
    decoded: u64,
}

impl Agentic {
    fn new() -> Agentic {
        Agentic { alloc: BlockAllocator::new(POOL_BLOCKS, BLOCK), prefix: PrefixCache::new(), dry: 0, prefill_rows: 0, decoded: 0 }
    }
}

impl PagedDecoder for Agentic {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        PREFILL_ROUND
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        40_960
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<model::serve::AdapterId>) -> u32 {
        let ns = adapter.map_or(model::paged::BASE_KV_NAMESPACE, |a| a.kv_namespace());
        self.prefix.peek_in(ns, prompt, BLOCK, prompt.len().saturating_sub(1)).len() as u32
    }
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        self.prefix.evict(want, &mut self.alloc)
    }
    fn reclaim_prefix_keeping(&mut self, want: u32, prompt: &[u32], adapter: Option<model::serve::AdapterId>) -> u32 {
        let ns = adapter.map_or(model::paged::BASE_KV_NAMESPACE, |a| a.kv_namespace());
        let keep = self.prefix.peek_in(ns, prompt, BLOCK, prompt.len().saturating_sub(1));
        self.prefix.evict_keeping(want, &keep, &mut self.alloc)
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        let ns = table.kv_namespace();
        let hits = self.prefix.lookup_in(ns, prompt, BLOCK, prompt.len() - 1);
        if !hits.is_empty() {
            table.adopt_prefix(&hits, &mut self.alloc);
        }
        let rest = prompt.len() as u32 - table.len();
        if self.alloc.free_blocks() < (table.len() + rest).div_ceil(BLOCK) - table.blocks().len() as u32 {
            self.dry += 1;
        }
        table.reserve_or_raise(rest, &mut self.alloc);
        self.prefill_rows += u64::from(rest);
        self.prefix.insert_chain_in(ns, prompt, table.blocks(), hits.len(), &mut self.alloc);
        vec![*prompt.last().expect("a prompt") as f32]
    }
    fn logits(&self, _hidden: &[f32]) -> Vec<f32> {
        unreachable!("admission is answered by admit_greedy")
    }
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        (hidden[0] as u32 + 1) % VOCAB as u32
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        let need: u32 = tables.iter().filter(|t| t.len() % BLOCK == 0).count() as u32;
        if self.alloc.free_blocks() < need {
            self.dry += 1;
        }
        for t in tables.iter_mut() {
            t.append_or_raise(&mut self.alloc);
        }
        self.decoded += inputs.len() as u64;
        inputs.iter().map(|&i| (i + 1) % VOCAB as u32).collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, self.prefix.len())
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        None
    }
    fn kv_pool_bytes(&self) -> u64 {
        0
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        u64::from(POOL_BLOCKS * BLOCK)
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("greedy only")
    }
    fn topk_capacity(&self) -> usize {
        1
    }
}

/// One agent trial: a conversation that resends its whole history every turn.
struct Trial {
    id: u32,
    history: Vec<u32>,
    turns_left: u32,
    /// Tokens written into the history so far beyond the system prompt, which keeps every conversation's own tokens
    /// distinct from every other's.
    written: u32,
}

impl Trial {
    fn extend(&mut self, n: usize) {
        for _ in 0..n {
            self.written += 1;
            let mixed = (u64::from(self.id) << 32 | u64::from(self.written)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            self.history.push((mixed >> 44) as u32 % VOCAB as u32);
        }
    }
}

/// What a burst measured: the running width sampled every step, and the work done.
struct Burst {
    widths: Vec<usize>,
    steps: usize,
    dry: usize,
    decoded: u64,
    prefill_rows: u64,
    /// Running width the scheduler admits at the end ([`Scheduler::admitted_width`]).
    admitted_end: usize,
}

/// 96 trials of 8 turns each against one scheduler: every trial's next turn is submitted when its last reply arrives
/// (closed loop), with 500-3000 tokens of tool output appended, so contexts run from about 5k to 30k tokens.
fn run_burst(rng: &mut data::rng::Lcg) -> Burst {
    let mut sched = Scheduler::new(Agentic::new(), MAX_BATCH);
    let system: Vec<u32> = (1..=SYSTEM_PROMPT as u32).collect();
    let mut trials: Vec<Trial> = (0..96)
        .map(|id| {
            let mut t = Trial { id, history: system.clone(), turns_left: 8, written: 0 };
            t.extend(3000 + rng.next_u32() as usize % 3000);
            t
        })
        .collect();
    let mut pending: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for (i, t) in trials.iter_mut().enumerate() {
        let max_new = 64 + rng.next_u32() as usize % 192;
        pending.insert(sched.submit(Request { prompt: t.history.clone(), max_new, stop: Vec::new() }), i);
    }
    let mut widths = Vec::new();
    let mut steps = 0;
    while !pending.is_empty() {
        assert!(steps < 200_000, "the burst did not finish: {} trials pending", pending.len());
        let report = sched.step_report();
        steps += 1;
        widths.push(sched.running_len());
        assert!(report.rejected.is_empty(), "no request is refused: {:?}", report.rejected.first());
        for (id, reply) in report.completed {
            let i = pending.remove(&id).expect("a pending turn");
            let t = &mut trials[i];
            t.history.extend_from_slice(&reply);
            t.turns_left -= 1;
            if t.turns_left > 0 {
                t.extend(500 + rng.next_u32() as usize % 2500);
                let max_new = 64 + rng.next_u32() as usize % 192;
                pending.insert(sched.submit(Request { prompt: t.history.clone(), max_new, stop: Vec::new() }), i);
            }
        }
    }
    let admitted_end = sched.admitted_width();
    let dec = sched.decoder_mut();
    Burst { widths, steps, dry: dec.dry, decoded: dec.decoded, prefill_rows: dec.prefill_rows, admitted_end }
}

#[test]
fn a_burst_of_agent_conversations_keeps_the_batch_as_wide_as_the_pool_holds() {
    let mut rng = data::rng::Lcg::random();
    let b = run_burst(&mut rng);
    let mean = b.widths.iter().sum::<usize>() as f64 / b.steps as f64;
    let deciles: Vec<usize> = (0..10).map(|d| b.widths[d * b.steps / 10..(d + 1) * b.steps / 10].iter().sum::<usize>() / (b.steps / 10).max(1)).collect();
    eprintln!(
        "agent burst: {} steps, mean running width {mean:.1}, width by tenth of the run {deciles:?}, {} decoded, {} prompt rows computed, {} dry-pool failures, admits {} at the end",
        b.steps, b.decoded, b.prefill_rows, b.dry, b.admitted_end
    );
    assert_eq!(b.dry, 0, "admission's promises hold: the pool never runs dry under a sequence it admitted");
    assert_eq!(b.admitted_end, MAX_BATCH, "nothing ran the card out of memory, so the scheduler admits its whole batch");
}
