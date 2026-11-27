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
use model::serve::{PagedDecoder, PressureCause, RejectReason, Request, Scheduler, THROUGHPUT_WINDOW_SECS};

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
    /// Blocks the scheduler is told are free beyond the pool's own: the pool accounting it admitted against was
    /// wrong (a cached prefix it counted on was evicted before the prefill adopted it).
    phantom_free: u32,
}

impl Toy {
    fn new(blocks: u32, slots: usize) -> Toy {
        Toy { alloc: BlockAllocator::new(blocks, BLOCK), abandoned: 0, slots, live: 0, too_long: usize::MAX, failures: 0, prefix_reclaims: 0, phantom_free: 0 }
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
        self.alloc.free_blocks() + self.phantom_free
    }
    fn device_free_bytes(&self) -> Option<u64> {
        // A GiB of scratch and state a sequence: what the card could still give.
        Some((self.slots.saturating_sub(self.live) as u64) << 30)
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
        // Counted before the blocks are taken: a pool that runs dry leaves the blocks it did take in the table, which release gives back.
        self.live += 1;
        table.reserve_or_raise(prompt.len() as u32, &mut self.alloc);
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
            t.append_or_raise(&mut self.alloc);
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
fn the_batch_grows_back_once_the_card_has_held_the_smaller_one_for_a_while_and_says_what_it_admits() {
    let mut sched = Scheduler::new(Toy::new(4096, 2), 8);
    // A burst that overfills the card sends the scheduler to a batch of one or two ...
    for i in 0..6 {
        sched.submit(Request { prompt: prompt(10 + i), max_new: 4, stop: Vec::new() });
    }
    assert_eq!(sched.admitted_width(), 8, "before any failure the scheduler admits its whole batch");
    drain(&mut sched);
    // It says so, so a server advertises the width it admits now rather than its nominal batch.
    assert!(sched.admitted_width() <= 2, "the card held two at most: {}", sched.admitted_width());
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
    assert_eq!(sched.admitted_width(), 8, "and it admits its whole batch again");
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

/// A pool that cannot back a prefill the scheduler admitted (its accounting was off: a prefix it counted on was
/// evicted before the prefill adopted it) is a full device like any other: the sequences go back to the queue and
/// complete once the pool has room, with the tokens a roomy pool gives. It never unwinds the lane (a panic here took
/// every request in flight with it).
#[test]
fn a_kv_pool_that_cannot_back_an_admitted_prefill_queues_the_request_instead_of_panicking() {
    let mut toy = Toy::new(8, 8);
    toy.phantom_free = 100;
    let mut sched = Scheduler::new(toy, 4);
    let ids: Vec<(u64, usize)> = (0..4).map(|i| (sched.submit(Request { prompt: prompt(40 + i), max_new: 6, stop: Vec::new() }), 40 + i)).collect();
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "a request that fits the pool alone is never refused: {rejected:?}");
    for (id, len) in ids {
        assert_eq!(done[&id], expected(len, 6), "request {id} gets the tokens a roomy pool gives it");
    }
    assert!(sched.decoder_mut().abandoned >= 1, "the exhausted pool sent the prefilling sequences back");
    assert_eq!(sched.decoder_mut().alloc.free_blocks(), 8, "every block is back in the pool");
}

/// Conformance (knowledge #613): with work queued and the card's room back, the running width climbs back to the whole
/// batch within a bounded number of iterations after an out-of-memory failure, not one sequence every few dozen. On
/// 2026-11-27 servers sat at 5-25 of 64 for minutes with 28-48 requests queued and the card half empty.
#[test]
fn after_a_card_runs_out_the_batch_is_whole_again_within_a_few_dozen_iterations_once_it_has_room() {
    let mut sched = Scheduler::new(Toy::new(4096, 2), 32);
    let ids: Vec<u64> = (0..48).map(|i| sched.submit(Request { prompt: prompt(10 + i % 7), max_new: 400, stop: Vec::new() })).collect();
    // The card holds two: the failures narrow the batch to that.
    for _ in 0..64 {
        sched.step();
    }
    assert!(sched.decoder_mut().failures > 0 && sched.admitted_width() <= 3, "the failures narrowed the batch: {}", sched.admitted_width());
    // The card has room for the whole batch again (whatever held it is gone).
    sched.decoder_mut().slots = 32;
    let mut iterations = 0;
    while sched.running_len() < 32 {
        sched.step();
        iterations += 1;
        assert!(iterations <= 48, "the batch climbs back to 32 within 48 iterations; at {iterations} it runs {}", sched.running_len());
    }
    assert_eq!(sched.admitted_width(), 32, "and says so to whoever routes work to it");
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "{rejected:?}");
    assert!(ids.iter().all(|id| done.contains_key(id)) || sched.decoder_mut().failures > 0);
}

/// A KV pool that runs out of free blocks once (its accounting was short) is not the card's memory limit: the
/// sequences go back to the queue and the batch is not narrowed (knowledge #613).
#[test]
fn a_kv_pool_that_ran_dry_once_does_not_narrow_the_batch() {
    let mut toy = Toy::new(16, 16);
    toy.phantom_free = 40;
    let mut sched = Scheduler::new(toy, 8);
    for i in 0..8 {
        sched.submit(Request { prompt: prompt(40 + i), max_new: 4, stop: Vec::new() });
    }
    sched.step();
    assert!(sched.decoder_mut().abandoned >= 1, "the pool ran dry under the admitted prefills");
    sched.decoder_mut().phantom_free = 0;
    assert_eq!(sched.admitted_width(), 8, "a dry pool is the scheduler's accounting, not the card: the batch stays whole");
    let (_, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "{rejected:?}");
}

/// Conformance (knowledge #613): under a steady stream of arrivals of varied lengths, the running width tracks what
/// the card holds as it changes: down when room is taken (another resident grows), back up to the whole batch within a
/// bounded number of iterations once the room is given back, re-judged every iteration while requests keep coming.
#[test]
fn under_a_stream_of_arrivals_the_running_width_tracks_the_room_the_card_has() {
    let mut rng = data::rng::Lcg::random();
    let mut sched = Scheduler::new(Toy::new(8192, 16), 16);
    let mut wide_again = Vec::new();
    for step in 0..900 {
        // Room for 16, then for 4 while something else holds the card, then for 16 again.
        sched.decoder_mut().slots = if (300..450).contains(&step) { 4 } else { 16 };
        // Two arrivals an iteration, prompts of 8-80 tokens, replies of 20-120: the queue never runs empty.
        for _ in 0..2 {
            if sched.waiting_len() < 64 {
                let len = 8 + rng.next_u32() as usize % 72;
                sched.submit(Request { prompt: prompt(len), max_new: 20 + rng.next_u32() as usize % 100, stop: Vec::new() });
            }
        }
        let report = sched.step_report();
        assert!(report.rejected.is_empty(), "{:?}", report.rejected);
        if (400..450).contains(&step) {
            assert!(sched.running_len() <= 4, "the card holds four at step {step}: {} run", sched.running_len());
        }
        if step >= 500 {
            wide_again.push(sched.running_len());
        }
    }
    // Sequences that finish leave their rows empty until the next iteration admits into them, so "whole" is the batch
    // less those few.
    let least = *wide_again.iter().min().unwrap();
    let mean = wide_again.iter().sum::<usize>() as f64 / wide_again.len() as f64;
    assert!(least >= 12 && mean >= 14.5, "from 50 iterations after the room came back the batch stays whole: mean {mean:.1}, least {least}");
    assert_eq!(sched.admitted_width(), 16);
}

// What a scheduler tells an observer about itself (`Scheduler::observe`): the back-off a device failure or a dry pool
// set, its cause and time and how far it has recovered, and the tokens it decoded and prefilled, with their rate over
// a window. A server publishes it per replica in its metrics document.

/// Milliseconds since the Unix epoch now, for bracketing a recorded wall-clock time.
fn unix_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64
}

#[test]
fn a_scheduler_says_why_and_since_when_it_admits_fewer_and_how_far_it_has_recovered() {
    let mut sched = Scheduler::new(Toy::new(4096, 2), 8);
    let calm = sched.observe();
    assert_eq!((calm.backoff.limit, calm.backoff.cause, calm.backoff.last_device_oom_unix_ms, calm.backoff.last_pool_dry_unix_ms), (None, None, None, None), "a scheduler that never failed holds no back-off");
    assert_eq!((calm.admitted_width, calm.max_width), (8, 8));
    let before = unix_ms();
    for i in 0..6 {
        sched.submit(Request { prompt: prompt(10 + i), max_new: 400, stop: Vec::new() });
    }
    while sched.decoder_mut().failures == 0 {
        sched.step();
    }
    let narrowed = sched.observe();
    let after = unix_ms();
    let backoff = narrowed.backoff;
    assert_eq!(backoff.cause, Some(PressureCause::DeviceOutOfMemory), "a refused allocation is the card's limit");
    assert!(backoff.limit.is_some_and(|l| l < 8), "the width is held under the batch: {backoff:?}");
    assert_eq!(Some(narrowed.admitted_width), backoff.limit, "the admitted width is the limit");
    assert!(backoff.failed_at_width.is_some());
    assert!(backoff.last_device_oom_unix_ms.is_some_and(|t| (before..=after).contains(&t)), "the failure's wall-clock time: {backoff:?}");
    assert_eq!(backoff.last_pool_dry_unix_ms, None);
    let grows = backoff.grows_after_iterations.expect("a held limit says when it next grows");
    assert!(backoff.calm_iterations < grows, "{backoff:?}");
    assert!(backoff.next_limit.is_some_and(|n| n > backoff.limit.unwrap()), "the next limit is wider: {backoff:?}");
    // Recovery progress moves with calm iterations, and a card with room gives the whole batch back.
    sched.decoder_mut().slots = 8;
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty() && done.len() == 6);
    let recovered = sched.observe().backoff;
    assert_eq!((recovered.limit, recovered.grows_after_iterations, recovered.next_limit), (None, None, None), "a drained scheduler holds no limit: {recovered:?}");
    assert_eq!(recovered.last_device_oom_unix_ms, backoff.last_device_oom_unix_ms, "the time of the last failure stays readable");
}

#[test]
fn a_dry_kv_pool_is_reported_as_its_own_cause_and_narrows_nothing() {
    let mut toy = Toy::new(16, 16);
    toy.phantom_free = 40;
    let mut sched = Scheduler::new(toy, 8);
    for i in 0..8 {
        sched.submit(Request { prompt: prompt(40 + i), max_new: 4, stop: Vec::new() });
    }
    sched.step();
    let backoff = sched.observe().backoff;
    assert!(backoff.last_pool_dry_unix_ms.is_some(), "the dry pool is recorded: {backoff:?}");
    assert_eq!(backoff.cause, Some(PressureCause::KvPoolDry));
    assert_eq!((backoff.limit, backoff.last_device_oom_unix_ms), (None, None), "and narrows nothing: {backoff:?}");
}

#[test]
fn a_scheduler_counts_the_tokens_it_decodes_and_prefills_and_their_rate_over_a_window() {
    let now = std::sync::Arc::new(std::sync::Mutex::new(0.0f64));
    let mut sched = Scheduler::new(Toy::new(4096, 8), 8);
    let clock = now.clone();
    sched.set_clock(Box::new(move || *clock.lock().unwrap()));
    // Four requests of 10 prompt tokens and 6 reply tokens: 40 prompt tokens prefilled, 24 tokens decoded.
    for _ in 0..4 {
        sched.submit(Request { prompt: prompt(10), max_new: 6, stop: Vec::new() });
    }
    while sched.pending() {
        *now.lock().unwrap() += 0.5;
        sched.step();
    }
    let busy = sched.observe();
    assert_eq!((busy.decoded_tokens, busy.prefilled_tokens), (24, 40));
    let secs = *now.lock().unwrap();
    assert!(secs <= THROUGHPUT_WINDOW_SECS, "the run fits one window: {secs} s");
    // Everything happened inside the window: the rate is the tokens over the time the scheduler has existed.
    assert!((busy.decode_tokens_per_s - 24.0 / secs).abs() < 1e-9, "{}", busy.decode_tokens_per_s);
    assert!((busy.prefill_tokens_per_s - 40.0 / secs).abs() < 1e-9, "{}", busy.prefill_tokens_per_s);
    assert_eq!((busy.running, busy.prefilling, busy.waiting, busy.swapped), (0, 0, 0, 0));
    // A window later nothing was decoded in it: the rate is zero, the totals stay.
    *now.lock().unwrap() += THROUGHPUT_WINDOW_SECS + 1.0;
    let idle = sched.observe();
    assert_eq!((idle.decode_tokens_per_s, idle.prefill_tokens_per_s), (0.0, 0.0));
    assert_eq!((idle.decoded_tokens, idle.prefilled_tokens), (24, 40));
    assert_eq!(idle.kv.block_tokens, BLOCK);
    assert_eq!(idle.kv.total_blocks, 4096);
    assert_eq!(idle.kv.free_blocks, 4096, "every block is back");
}
