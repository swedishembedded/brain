// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for LLM serving that stays up under
// memory pressure for its clients. If your team needs expertise in paged-KV
// schedulers that degrade by queueing instead of crashing, then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a request the scheduler admitted never takes the engine down because
//! the KV pool ran out. Admission only proves that a PROMPT fits; the sequences
//! it lets in can still outgrow the pool together while they decode. Whether
//! host-RAM offload is off, or its budget too small for the overflow, every
//! request still completes with exactly the tokens it produces alone - no
//! token lost, none repeated - and a request that cannot fit the whole pool is
//! refused rather than queued forever.
//!
//! The decoder here keeps real KV contents (the token written at each cache
//! slot) and derives every next token from the whole cache, so a sequence
//! whose cache is rebuilt wrongly after preemption yields different tokens.

use std::collections::HashMap;

use model::kv_offload::{KvOffloadError, OffloadStats};
use model::paged::{BlockAllocator, BlockTable};
use model::serve::{PagedDecoder, RejectReason, Request, SampleParams, Scheduler};

const VOCAB: usize = 64;
const BLOCK: u32 = 4;

/// Tokens of one sequence's host-RAM copy.
struct HostCopy {
    namespace: u32,
    len: u32,
    blocks: Vec<Vec<u32>>,
}

struct Mock {
    alloc: BlockAllocator,
    /// Token stored at each slot of each physical block.
    kv: HashMap<u32, Vec<u32>>,
    /// Host budget in blocks; `0` is "offload unsupported".
    host_budget: u32,
    host: HashMap<u64, HostCopy>,
    stats: OffloadStats,
    /// How often the scheduler asked for the vocabulary (it validates prompts against it).
    vocab_calls: std::sync::atomic::AtomicUsize,
}

impl Mock {
    fn new(blocks: u32, host_budget: u32) -> Mock {
        Mock { alloc: BlockAllocator::new(blocks, BLOCK), kv: HashMap::new(), host_budget, host: HashMap::new(), stats: OffloadStats::default(), vocab_calls: Default::default() }
    }

    fn write(&mut self, block: u32, offset: u32, token: u32) {
        let slots = self.kv.entry(block).or_default();
        slots.truncate(offset as usize);
        slots.push(token);
    }

    fn history(&self, table: &BlockTable) -> Vec<u32> {
        (0..table.len()).map(|i| self.kv[&table.blocks()[(i / BLOCK) as usize]][(i % BLOCK) as usize]).collect()
    }

    fn host_blocks_used(&self) -> u32 {
        self.host.values().map(|h| h.blocks.len() as u32).sum()
    }
}

/// A function of the whole cache, so the next token depends on every slot.
fn digest(history: &[u32]) -> u32 {
    let h = history.iter().fold(7u64, |h, &t| h.wrapping_mul(1_000_003) ^ (t as u64 + 1));
    (h >> 5) as u32 & 0xFF_FFFF
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
        self.vocab_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        let start = table.reserve(prompt.len() as u32, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        for (i, &t) in prompt.iter().enumerate() {
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
                digest(&self.history(t)) % VOCAB as u32
            })
            .collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], _k: usize) -> Vec<Vec<(u32, f32)>> {
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &input)| {
                let (block, offset) = t.append(&mut self.alloc).expect("the scheduler let a decode step run past the pool");
                self.write(block, offset, input);
                let d = digest(&self.history(t));
                (0..3u32).map(|j| ((d + 17 * j) % VOCAB as u32, 1.0 - 0.1 * j as f32)).collect()
            })
            .collect()
    }
    fn topk_capacity(&self) -> usize {
        3
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
        self.alloc.num_blocks() as u64 * BLOCK as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }

    fn kv_offload_bytes(&self) -> u64 {
        self.host_budget as u64
    }
    fn demote_sequence(&mut self, key: u64, table: &mut BlockTable) -> Result<u32, KvOffloadError> {
        let need = table.blocks().len() as u32;
        if self.host_budget == 0 {
            return Err(KvOffloadError::Unsupported);
        }
        let free = self.host_budget - self.host_blocks_used();
        if need > free {
            self.stats.refused += 1;
            return Err(KvOffloadError::HostPoolFull { need_bytes: need as u64, free_bytes: free as u64 });
        }
        let blocks = table.blocks().iter().map(|b| self.kv[b].clone()).collect();
        self.host.insert(key, HostCopy { namespace: table.kv_namespace(), len: table.len(), blocks });
        table.release(&mut self.alloc);
        self.stats.demotions += 1;
        Ok(need)
    }
    fn promote_sequence(&mut self, key: u64) -> Result<BlockTable, KvOffloadError> {
        let copy = self.host.get(&key).ok_or(KvOffloadError::NotOffloaded { key })?;
        let need = copy.blocks.len() as u32;
        if self.alloc.free_blocks() < need {
            return Err(KvOffloadError::DevicePoolExhausted { need_blocks: need, free_blocks: self.alloc.free_blocks() });
        }
        let copy = self.host.remove(&key).expect("just found");
        let mut ids = Vec::new();
        for contents in copy.blocks {
            let b = self.alloc.alloc().expect("room was checked");
            self.kv.insert(b, contents);
            ids.push(b);
        }
        self.stats.promotions += 1;
        Ok(BlockTable::restore(copy.namespace, ids, copy.len))
    }
    fn offloaded_blocks(&self, key: u64) -> Option<u32> {
        self.host.get(&key).map(|h| h.blocks.len() as u32)
    }
    fn discard_offloaded(&mut self, key: u64) -> bool {
        self.host.remove(&key).is_some()
    }
    fn offload_stats(&self) -> OffloadStats {
        self.stats
    }
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (seed * 7 + i) % VOCAB as u32).collect()
}

fn request(seed: u32) -> Request {
    Request { prompt: prompt(seed, 10), max_new: 16, stop: Vec::new() }
}

/// The tokens a request produces with the whole (ample) pool to itself.
fn alone(seed: u32, sample: SampleParams, rng_seed: u64) -> Vec<u32> {
    let mut sched = Scheduler::new(Mock::new(256, 0), 1);
    let id = sched.submit_sampled(request(seed), sample, rng_seed);
    sched.run().remove(&id).expect("finished")
}

const POOL_BLOCKS: u32 = 24;
// seed: the prompt seeds of the six requests; the alone and oversubscribed runs must build the same requests
const SEEDS: [u32; 6] = [1, 2, 3, 4, 5, 6];

/// Six requests whose prompts (3 blocks each) fit together, but whose full
/// 26-token lives (7 blocks each) do not fit a 24-block pool at once.
fn run_oversubscribed(host_budget: u32, sample: SampleParams) -> (Vec<Vec<u32>>, Scheduler<Mock>) {
    let mut sched = Scheduler::new(Mock::new(POOL_BLOCKS, host_budget), 8);
    let ids: Vec<u64> = SEEDS.iter().map(|&s| sched.submit_sampled(request(s), sample, 100 + s as u64)).collect();
    let mut done = HashMap::new();
    let mut guard = 0;
    while sched.pending() {
        for (id, toks) in sched.step() {
            assert!(done.insert(id, toks).is_none(), "request {id} completed twice");
        }
        guard += 1;
        assert!(guard < 10_000, "the scheduler stopped making progress");
    }
    assert_eq!(done.len(), ids.len(), "no request may be lost");
    (ids.iter().map(|id| done.remove(id).unwrap()).collect(), sched)
}

fn assert_each_matches_running_alone(got: &[Vec<u32>], sample: SampleParams) {
    for (&seed, toks) in SEEDS.iter().zip(got) {
        assert_eq!(toks, &alone(seed, sample, 100 + seed as u64), "request {seed} diverged from its solo run");
    }
}

#[test]
fn sequences_outgrowing_the_pool_without_offload_finish_with_their_solo_tokens() {
    let (got, sched) = run_oversubscribed(0, SampleParams::greedy());
    assert_each_matches_running_alone(&got, SampleParams::greedy());
    assert_eq!(sched.free_blocks(), POOL_BLOCKS, "every block is back once all are done");
}

#[test]
fn sequences_outgrowing_the_pool_and_a_too_small_host_budget_finish_with_their_solo_tokens() {
    // Room on the host for one sequence's cache, far less than the overflow.
    let (got, sched) = run_oversubscribed(4, SampleParams::greedy());
    assert_each_matches_running_alone(&got, SampleParams::greedy());
    assert_eq!(sched.free_blocks(), POOL_BLOCKS);
    assert!(sched.offload_stats().refused > 0, "the host budget must actually have been too small for the test to mean anything");
}

#[test]
fn a_preempted_sampled_sequence_keeps_its_seeded_stream() {
    let sample = SampleParams { temp: 0.8, top_k: 3, top_p: 1.0, ..SampleParams::greedy() };
    let (got, _) = run_oversubscribed(0, sample);
    assert_each_matches_running_alone(&got, sample);
}

/// Every request's `(tokens, log-probabilities)`, run to the end.
fn run_with_logprobs(sched: &mut Scheduler<Mock>) -> HashMap<u64, (Vec<u32>, Vec<f32>)> {
    let mut done = HashMap::new();
    while sched.pending() {
        let report = sched.step_report();
        for ((id, toks), (lid, logprobs)) in report.completed.into_iter().zip(report.completed_logprobs) {
            assert_eq!(id, lid);
            done.insert(id, (toks, logprobs));
        }
    }
    done
}

#[test]
fn a_preempted_sampled_sequence_reports_the_log_probabilities_of_its_solo_run() {
    let sample = SampleParams { temp: 0.8, top_k: 3, top_p: 1.0, ..SampleParams::greedy() };
    let mut sched = Scheduler::new(Mock::new(POOL_BLOCKS, 0), 8);
    let ids: Vec<u64> = SEEDS.iter().map(|&s| sched.submit_sampled(request(s), sample, 100 + s as u64)).collect();
    let mut together = run_with_logprobs(&mut sched);
    for (&seed, id) in SEEDS.iter().zip(ids) {
        let mut solo = Scheduler::new(Mock::new(256, 0), 1);
        let solo_id = solo.submit_sampled(request(seed), sample, 100 + seed as u64);
        let want = run_with_logprobs(&mut solo).remove(&solo_id).expect("finished");
        let got = together.remove(&id).expect("finished");
        assert_eq!(got.1.len(), got.0.len(), "request {seed}: one log-probability per token");
        assert_eq!(got, want, "request {seed} diverged from its solo run");
    }
}

#[test]
fn a_request_that_cannot_fit_the_whole_pool_is_rejected_not_queued_forever() {
    let mut sched = Scheduler::new(Mock::new(POOL_BLOCKS, 0), 4);
    // 96 tokens of pool; this request alone needs 10 + 100.
    let id = sched.submit(Request { prompt: prompt(1, 10), max_new: 100, stop: Vec::new() });
    let report = sched.step_report();
    assert_eq!(report.rejected, vec![(id, RejectReason::ExceedsCapacity { need: 110, capacity: 96 })]);
    assert!(!sched.pending());
}

#[test]
fn cancelling_a_preempted_request_returns_its_progress_and_frees_everything() {
    let mut sched = Scheduler::new(Mock::new(POOL_BLOCKS, 0), 8);
    let ids: Vec<u64> = SEEDS.iter().map(|&s| sched.submit(request(s))).collect();
    let victim = loop {
        let report = sched.step_report();
        if let Some(&id) = report.recomputed.first() {
            break id;
        }
        assert!(sched.pending(), "the pool pressure must preempt someone before the work ends");
    };
    let progress = sched.tokens_of(victim).expect("a preempted request stays readable for a streaming caller").to_vec();
    assert!(!progress.is_empty());
    assert_eq!(sched.cancel(victim), Some(progress));
    for id in ids.into_iter().filter(|&id| id != victim) {
        sched.cancel(id);
    }
    assert!(!sched.pending());
    assert_eq!(sched.free_blocks(), POOL_BLOCKS);
}

/// Every sequence a streaming caller can read is in `streams()`, with the tokens `tokens_of` gives it,
/// whether it is running, preempted by recompute, or demoted to host RAM; and nothing else is.
#[test]
fn streams_agree_with_tokens_of_through_preemption() {
    for host_budget in [0, 6] {
        let mut sched = Scheduler::new(Mock::new(POOL_BLOCKS, host_budget), 8);
        let ids: Vec<u64> = SEEDS.iter().map(|&s| sched.submit(request(s))).collect();
        let mut preempted_seen = false;
        while sched.pending() {
            let report = sched.step_report();
            preempted_seen |= !report.recomputed.is_empty() || !report.demoted.is_empty();
            let streams = sched.streams();
            for &id in &ids {
                assert_eq!(streams.get(&id).copied(), sched.tokens_of(id), "request {id}");
            }
            assert!(streams.len() <= ids.len());
        }
        assert!(preempted_seen, "the pool pressure must preempt someone for the test to mean anything");
        assert!(sched.streams().is_empty());
    }
}

/// A request that waits for the pool is validated once, when it is queued, not again on every iteration
/// it waits: the vocabulary is not consulted per step per waiting prompt.
#[test]
fn a_waiting_prompt_is_validated_once_not_every_iteration() {
    let mut sched = Scheduler::new(Mock::new(POOL_BLOCKS, 0), 8);
    // Two requests that each need most of the pool: the second waits for the first to finish.
    let long = |seed| Request { prompt: prompt(seed, 60), max_new: 30, stop: Vec::new() };
    sched.submit(long(1));
    sched.submit(long(2));
    let mut waited_steps = 0;
    while sched.pending() {
        sched.step();
        if sched.waiting_len() > 0 {
            waited_steps += 1;
        }
    }
    assert!(waited_steps >= 20, "the second request must wait many iterations ({waited_steps}) for this to mean anything");
    let calls = sched.decoder_mut().vocab_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert!(calls < 10, "{calls} vocabulary lookups over {waited_steps} iterations of a waiting request");
}
