// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context LLM serving that
// reuses conversation history instead of paying for it twice, for its
// clients. If your team needs expertise in paged-KV admission and prefix
// caching, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: admission charges a request only for the KV blocks it will allocate.
//! Blocks a prompt adopts from the prefix cache are already resident, so the
//! next turn of a conversation whose history fills most of the pool is
//! admitted - reusing that history - as soon as the pool holds its UNCACHED
//! part, rather than waiting for room for the whole prompt (which it never
//! gets while its own history occupies the pool).
//!
//! The decoder here keeps real KV contents and shares cached blocks through
//! `model::paged::PrefixCache`, so a wrongly adopted prefix yields different
//! tokens than a cold run.

use std::collections::HashMap;

use model::paged::{BlockAllocator, BlockTable, PrefixCache};
use model::serve::{AdapterId, PagedDecoder, Request, Scheduler};

const VOCAB: usize = 64;
const BLOCK: u32 = 4;

struct Mock {
    alloc: BlockAllocator,
    prefix: PrefixCache,
    /// Token stored at each slot of each physical block.
    kv: HashMap<u32, Vec<u32>>,
    hit_tokens: u64,
    lookup_tokens: u64,
}

impl Mock {
    fn new(blocks: u32) -> Mock {
        Mock { alloc: BlockAllocator::new(blocks, BLOCK), prefix: PrefixCache::new(), kv: HashMap::new(), hit_tokens: 0, lookup_tokens: 0 }
    }

    fn write(&mut self, block: u32, offset: u32, token: u32) {
        let slots = self.kv.entry(block).or_default();
        slots.truncate(offset as usize);
        slots.push(token);
    }

    fn history(&self, table: &BlockTable) -> Vec<u32> {
        (0..table.len()).map(|i| self.kv[&table.blocks()[(i / BLOCK) as usize]][(i % BLOCK) as usize]).collect()
    }
}

/// A function of the whole cache, so the next token depends on every slot.
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
        64
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
    fn reclaim_prefix_keeping(&mut self, want: u32, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        let keep = self.prefix.peek_in(namespace(adapter), prompt, BLOCK, prompt.len().saturating_sub(1));
        self.prefix.evict_keeping(want, &keep, &mut self.alloc)
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        let history = self.history(t);
        let from = (t.cached_rows() / BLOCK) as usize;
        self.prefix.insert_chain_in(t.kv_namespace(), &history, t.blocks(), from, &mut self.alloc);
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        // Always compute the last token: its hidden state is what the caller samples from.
        let hits = self.prefix.lookup_in(table.kv_namespace(), prompt, BLOCK, prompt.len() - 1);
        self.lookup_tokens += prompt.len() as u64;
        self.hit_tokens += hits.len() as u64 * BLOCK as u64;
        if !hits.is_empty() {
            table.adopt_prefix(&hits, &mut self.alloc);
        }
        let cached = table.len() as usize;
        let start = table.reserve((prompt.len() - cached) as u32, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        for (i, &t) in prompt[cached..].iter().enumerate() {
            let pos = start + i as u32;
            self.write(table.blocks()[(pos / BLOCK) as usize], pos % BLOCK, t);
        }
        vec![digest(&self.history(table)) as f32]
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
                let (block, offset) = t.append(&mut self.alloc).expect("the scheduler let a decode step run past the pool");
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
        (self.hit_tokens, self.lookup_tokens, self.prefix.len())
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

fn tokens(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (seed * 7 + i * 3) % VOCAB as u32).collect()
}

/// Run every submitted request to completion, failing instead of hanging.
fn drain(sched: &mut Scheduler<Mock>) -> HashMap<u64, Vec<u32>> {
    let mut done = HashMap::new();
    let mut steps = 0;
    while sched.pending() {
        done.extend(sched.step());
        steps += 1;
        assert!(steps < 1_000, "the scheduler stopped making progress");
    }
    done
}

/// The pool is 16 blocks (64 tokens). Turn one leaves 10 full blocks of
/// history in the prefix cache; turn two is that history plus a new message,
/// 13 blocks with its first token. Only 3 of them are uncached, and 6 are free.
#[test]
fn the_next_turn_of_a_conversation_filling_most_of_the_pool_is_admitted_on_its_cached_history() {
    const POOL: u32 = 16;
    let mut sched = Scheduler::new(Mock::new(POOL), 4);
    let first = tokens(1, 40);
    let id = sched.submit(Request { prompt: first.clone(), max_new: 4, stop: Vec::new() });
    let reply = drain(&mut sched).remove(&id).expect("turn one finished");
    // The conversation so far, as the cache holds it: the prompt and the replies the decoder consumed.
    let mut second = first;
    second.extend_from_slice(&reply[..reply.len() - 1]);
    second.extend(tokens(9, 52 - second.len()));
    assert_eq!(sched.free_blocks(), POOL - 10, "turn one's history stays cached");

    let (hits_before, _, _) = sched.prefix_stats();
    let id = sched.submit(Request { prompt: second.clone(), max_new: 4, stop: Vec::new() });
    let got = drain(&mut sched).remove(&id).expect("turn two finished");
    let (hits_after, _, _) = sched.prefix_stats();
    assert_eq!(hits_after - hits_before, 40, "turn two resumes from its 10 cached blocks instead of evicting them");

    let mut cold = Scheduler::new(Mock::new(256), 1);
    let cold_id = cold.submit(Request { prompt: second, max_new: 4, stop: Vec::new() });
    assert_eq!(got, drain(&mut cold).remove(&cold_id).expect("cold run finished"), "the reused history yields the cold run's tokens");
}
