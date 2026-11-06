// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements LLM serving that keeps every admitted request through a full card instead of failing it. If
// your team needs expertise in serving large models at the memory limit of a card, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a device allocation that fails (activation scratch and recurrent state live beside the KV pool, so the
//! pool's own accounting cannot see them coming) never fails a request the server admitted. The sequences the failed
//! call carried go back to the head of the queue, the decoder drops its prefill work, and the scheduler admits fewer
//! sequences at once until the card has shown it can hold more. Every request completes with exactly the tokens it
//! produces on a card that never ran out. Only a request that fails again while it is alone on the card, after the
//! prefix cache has been given back, is refused, with a device-out-of-memory reason a client can retry on.

use std::collections::HashMap;

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{PagedDecoder, RejectReason, Request, Scheduler};

const VOCAB: usize = 64;
const BLOCK: u32 = 8;

/// A decoder whose forwards fail while more sequences hold the card than it has room for, the way scratch and
/// recurrent state outside the KV pool do.
struct Toy {
    alloc: BlockAllocator,
    abandoned: usize,
    /// Sequences the card holds scratch and state for at once; a call that would hold more runs it out of memory.
    slots: usize,
    /// Sequences holding blocks now.
    live: usize,
    /// A prompt this long never fits the card, alone or not.
    too_long: usize,
    /// Failures raised so far.
    failures: usize,
    prefix_reclaims: usize,
}

impl Toy {
    fn new(blocks: u32, slots: usize) -> Toy {
        Toy { alloc: BlockAllocator::new(blocks, BLOCK), abandoned: 0, slots, live: 0, too_long: usize::MAX, failures: 0, prefix_reclaims: 0 }
    }

    fn run_out(&mut self, bytes: u64) -> ! {
        self.failures += 1;
        std::panic::resume_unwind(Box::new(gpu_core::DeviceOutOfMemory { bytes, what: "storage".into(), detail: "out of memory".into() }))
    }
}

impl PagedDecoder for Toy {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        256
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        1024
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        self.prefix_reclaims += 1;
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if !t.blocks().is_empty() {
            self.live -= 1;
        }
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        if prompt.len() >= self.too_long || self.live + 1 > self.slots {
            self.run_out(48 << 20);
        }
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        self.live += 1;
        vec![*prompt.last().unwrap() as f32]
    }
    fn abandon_prefill(&mut self) {
        self.abandoned += 1;
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[(hidden[0] as usize + 1) % VOCAB] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        if tables.len() > self.slots {
            self.run_out(252 << 20);
        }
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|&i| (i + 1) % VOCAB as u32).collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        None
    }
    fn kv_pool_bytes(&self) -> u64 {
        0
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        4096
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

fn prompt(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| i % VOCAB as u32).collect()
}

/// The tokens a request of `prompt_len` tokens produces: each is the previous one plus one.
fn expected(prompt_len: usize, max_new: usize) -> Vec<u32> {
    let last = (prompt_len as u32 - 1) % VOCAB as u32;
    (1..=max_new as u32).map(|i| (last + i) % VOCAB as u32).collect()
}

/// What a run ended with: every completion by id, and every refusal.
type Outcome = (HashMap<u64, Vec<u32>>, Vec<(u64, RejectReason)>);

/// Drive `sched` until it is idle.
fn drain(sched: &mut Scheduler<Toy>) -> Outcome {
    let (mut done, mut rejected) = (HashMap::new(), Vec::new());
    for _ in 0..10_000 {
        if !sched.pending() {
            return (done, rejected);
        }
        let report = sched.step_report();
        rejected.extend(report.rejected);
        done.extend(report.completed);
    }
    panic!("the scheduler did not go idle: {} waiting, {} running", sched.waiting_len(), sched.running_len());
}

#[test]
fn prompts_that_overfill_the_card_wait_for_each_other_instead_of_failing() {
    // Eight prompts in one burst, room for three at a time: the first prefill round runs the card out of memory.
    let mut sched = Scheduler::new(Toy::new(512, 3), 8);
    let free = sched.free_blocks();
    let ids: Vec<(u64, usize)> = (0..8).map(|i| (sched.submit(Request { prompt: prompt(10 + i), max_new: 6, stop: Vec::new() }), 10 + i)).collect();
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "no admitted request is refused for memory the scheduler could learn it lacked: {rejected:?}");
    for (id, len) in ids {
        assert_eq!(done[&id], expected(len, 6), "request {id} gets the tokens a card with room gives it");
    }
    assert!(sched.decoder_mut().failures >= 1, "the burst did run the card out of memory");
    assert!(sched.decoder_mut().abandoned >= 1, "the decoder is told to drop its prefill work");
    assert_eq!(sched.free_blocks(), free, "every block is back in the pool");
}

#[test]
fn a_decode_step_that_runs_the_card_out_of_memory_loses_no_sequence() {
    // Both sequences decode together on a card with room for one: the step that carries both fails.
    let mut sched = Scheduler::new(Toy::new(512, 1), 4);
    let free = sched.free_blocks();
    sched.decoder_mut().slots = 2;
    let a = sched.submit(Request { prompt: prompt(10), max_new: 20, stop: Vec::new() });
    let b = sched.submit(Request { prompt: prompt(12), max_new: 20, stop: Vec::new() });
    sched.step();
    sched.step();
    sched.decoder_mut().slots = 1;
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "{rejected:?}");
    assert_eq!(done[&a], expected(10, 20));
    assert_eq!(done[&b], expected(12, 20), "the sequence sent back to be recomputed continues with the tokens it would have made");
    assert!(sched.decoder_mut().failures >= 1);
    assert_eq!(sched.free_blocks(), free);
}

#[test]
fn a_prompt_that_cannot_fit_even_alone_is_refused_with_a_retryable_reason_after_the_cache_is_given_back() {
    let mut toy = Toy::new(512, 4);
    toy.too_long = 100;
    let mut sched = Scheduler::new(toy, 4);
    let free = sched.free_blocks();
    let doomed = sched.submit(Request { prompt: prompt(100), max_new: 4, stop: Vec::new() });
    let (done, rejected) = drain(&mut sched);
    assert!(done.is_empty());
    assert!(matches!(rejected.as_slice(), [(id, RejectReason::DeviceOutOfMemory { bytes })] if *id == doomed && *bytes == 48 << 20), "{rejected:?}");
    assert!(sched.decoder_mut().prefix_reclaims >= 1, "the prefix cache is evicted before the request is refused");
    assert!(sched.decoder_mut().failures <= 4, "the retries are bounded: {}", sched.decoder_mut().failures);
    assert_eq!(sched.free_blocks(), free);

    // The scheduler serves on, and a request after the refusal is answered.
    let later = sched.submit(Request { prompt: prompt(12), max_new: 3, stop: Vec::new() });
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty() && done[&later] == expected(12, 3), "{rejected:?}");
}

#[test]
fn a_refused_request_never_takes_the_sequences_beside_it_down() {
    let mut toy = Toy::new(512, 4);
    toy.too_long = 100;
    let mut sched = Scheduler::new(toy, 4);
    let good: Vec<u64> = (0..3).map(|i| sched.submit(Request { prompt: prompt(10 + i), max_new: 8, stop: Vec::new() })).collect();
    sched.step();
    let doomed = sched.submit(Request { prompt: prompt(100), max_new: 4, stop: Vec::new() });
    let (done, rejected) = drain(&mut sched);
    assert!(matches!(rejected.as_slice(), [(id, _)] if *id == doomed), "only the prompt that cannot fit is refused: {rejected:?}");
    for id in good {
        assert_eq!(done[&id].len(), 8);
    }
}

#[test]
fn the_batch_grows_back_once_the_card_has_held_the_smaller_one_for_a_while() {
    let mut sched = Scheduler::new(Toy::new(4096, 2), 8);
    // A burst that overfills the card sends the scheduler to a batch of one or two ...
    for i in 0..6 {
        sched.submit(Request { prompt: prompt(10 + i), max_new: 4, stop: Vec::new() });
    }
    drain(&mut sched);
    // ... and later traffic, on a card that now has room for eight, is not held to that for ever.
    sched.decoder_mut().slots = 8;
    let failures = sched.decoder_mut().failures;
    let ids: Vec<u64> = (0..8).map(|i| sched.submit(Request { prompt: prompt(10 + i), max_new: 900, stop: Vec::new() })).collect();
    let (mut done, mut widest) = (HashMap::new(), 0);
    for _ in 0..3_000 {
        let report = sched.step_report();
        assert!(report.rejected.is_empty(), "{:?}", report.rejected);
        done.extend(report.completed);
        widest = widest.max(sched.running_len());
    }
    assert_eq!(widest, 8, "all eight run together again");
    assert!(ids.iter().all(|id| done.contains_key(id)));
    assert_eq!(sched.decoder_mut().failures, failures, "no further failure on the way back");
}

#[test]
fn a_hundred_long_conversations_on_a_small_pool_and_a_small_card_all_complete() {
    // 100 requests of 40..140 tokens, 40 tokens of reply each, in one burst: the pool holds about eight prompts at
    // once (KV blocks), the card about five (scratch and state), and the batch is allowed 48. No request may fail.
    let mut sched = Scheduler::new(Toy::new(160, 5), 48);
    let free = sched.free_blocks();
    let ids: Vec<(u64, usize)> = (0..100).map(|i| (sched.submit(Request { prompt: prompt(40 + i), max_new: 40, stop: Vec::new() }), 40 + i)).collect();
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "only completions: {:?}", rejected.iter().take(3).collect::<Vec<_>>());
    for (id, len) in ids {
        assert_eq!(done[&id], expected(len, 40), "request {id}");
    }
    let failures = sched.decoder_mut().failures;
    assert!(failures <= 40, "the scheduler learns the card's limit instead of retrying into it: {failures} failures for 100 requests");
    assert_eq!(sched.free_blocks(), free);
}
