// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for low-latency multi-tenant LLM
// serving for its clients. If your team needs expertise in continuous batching
// that does not let one long prompt stall everyone else's decode, then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec of continuous batching under a stream of long prompts: a short request
//! is answered at its own step count and its own time, not when the queue of
//! prompts behind it has been prefilled.
//!
//! The toy decoder charges a virtual clock: one unit per prompt token
//! prefilled and ten per decode step. Without interleaving, every scheduler
//! iteration admits a whole prompt (100 units) before the decode step, so a
//! request that needs five tokens waits five whole prompts. With a prefill
//! budget per iteration, decode runs every iteration and a prompt is prefilled
//! over several of them.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{PagedDecoder, PrefillJob, PrefillProgress, Request, Scheduler};

const VOCAB: usize = 64;

struct Toy {
    alloc: BlockAllocator,
    clock: Rc<Cell<u64>>,
    /// Prompt tokens already prefilled per first block.
    progress: HashMap<u32, usize>,
    chunked: bool,
    /// Rows per `prefill_chunks` call this toy takes across sequences (`None`: one sequence per call).
    batch_rows: Option<usize>,
    /// The (jobs, rows) of every `prefill_chunks` call, for the batching spec.
    calls: Rc<RefCell<Vec<(usize, usize)>>>,
}

impl Toy {
    fn new(clock: Rc<Cell<u64>>, chunked: bool) -> Toy {
        Toy { alloc: BlockAllocator::new(512, 8), clock, progress: HashMap::new(), chunked, batch_rows: None, calls: Rc::default() }
    }
}

impl PagedDecoder for Toy {
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
        1024
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(8)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(b) = t.blocks().first() {
            self.progress.remove(b);
        }
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        self.clock.set(self.clock.get() + prompt.len() as u64);
        vec![*prompt.last().unwrap() as f32]
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        self.chunked.then_some(16)
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        if !self.chunked {
            return PrefillProgress::Done(self.prefill(table, prompt));
        }
        if table.is_empty() {
            table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        }
        let slot = table.blocks()[0];
        let done = self.progress.entry(slot).or_insert(0);
        let n = budget.min(prompt.len() - *done);
        *done += n;
        self.clock.set(self.clock.get() + n as u64);
        if *done == prompt.len() {
            self.progress.remove(&slot);
            PrefillProgress::Done(vec![*prompt.last().unwrap() as f32])
        } else {
            PrefillProgress::Partial { consumed: n }
        }
    }
    fn prefill_batch_tokens(&self) -> Option<usize> {
        self.batch_rows
    }
    /// One forward over every job's rows: the clock is charged once per call (a fixed
    /// 40 units standing for the weight pass) plus one unit per row.
    fn prefill_chunks(&mut self, jobs: &mut [PrefillJob<'_>]) -> Vec<PrefillProgress> {
        let before = self.clock.get();
        let out: Vec<PrefillProgress> = jobs.iter_mut().map(|j| self.prefill_chunk(j.table, j.prompt, j.budget)).collect();
        let rows = (self.clock.get() - before) as usize;
        self.clock.set(before + 40 + rows as u64);
        self.calls.borrow_mut().push((jobs.len(), rows));
        out
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[(hidden[0] as usize + 1) % VOCAB] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.clock.set(self.clock.get() + 10);
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

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (seed + i) % VOCAB as u32).collect()
}

/// Virtual time at which each request finished, for one long runaway and eight
/// short answers arriving together behind long prompts.
fn finish_times(chunked: bool) -> (u64, Vec<u64>, Vec<Vec<u32>>) {
    let clock = Rc::new(Cell::new(0));
    let mut sched = Scheduler::new(Toy::new(clock.clone(), chunked), 16);
    let long = sched.submit(Request { prompt: prompt(1, 100), max_new: 400, stop: Vec::new() });
    let shorts: Vec<u64> = (0..8).map(|i| sched.submit(Request { prompt: prompt(10 + i, 100), max_new: 5, stop: Vec::new() })).collect();
    let mut at: HashMap<u64, u64> = HashMap::new();
    let mut tokens: HashMap<u64, Vec<u32>> = HashMap::new();
    while sched.pending() {
        for (id, toks) in sched.step() {
            at.insert(id, clock.get());
            tokens.insert(id, toks);
        }
    }
    (at[&long], shorts.iter().map(|id| at[id]).collect(), shorts.iter().map(|id| tokens[id].clone()).collect())
}

#[test]
fn short_requests_finish_at_their_own_time_not_after_the_queue_of_prompts_behind_them() {
    let (_, whole, whole_tokens) = finish_times(false);
    let (long_done, chunked, chunked_tokens) = finish_times(true);
    assert_eq!(whole_tokens, chunked_tokens, "interleaving prefill must not change any answer");
    // Whole-prompt admission: the first short answer waits for five more prompts' worth of prefill.
    assert!(whole[0] >= 500, "the baseline should be slow for the test to mean anything: {}", whole[0]);
    // Interleaved: the first short answer is back after the runaway's and its own prompt (16-token
    // slices, 26 units an iteration) and five decode steps - well before the whole-prompt baseline.
    assert!(chunked[0] < whole[0] * 3 / 4, "first short answer at {} against {} without interleaving", chunked[0], whole[0]);
    assert!(chunked.iter().all(|&t| t < long_done / 2), "every short answer is out long before the runaway: {chunked:?} vs {long_done}");
    assert!(chunked.windows(2).all(|w| w[0] <= w[1]), "and they leave in arrival order");
}

#[test]
fn a_cancelled_half_prefilled_request_returns_its_blocks() {
    let clock = Rc::new(Cell::new(0));
    let mut sched = Scheduler::new(Toy::new(clock, true), 4);
    let free = sched.free_blocks();
    let id = sched.submit(Request { prompt: prompt(3, 100), max_new: 5, stop: Vec::new() });
    sched.step();
    assert!(sched.free_blocks() < free, "the prompt's blocks are held while it is being prefilled");
    assert_eq!(sched.cancel(id), Some(Vec::new()));
    assert_eq!(sched.free_blocks(), free);
    assert!(!sched.pending());
}

/// Eight 100-token prompts arrive together; the decoder takes 64 rows per forward and a
/// sequence gets at most 16 of them while others wait.
fn run_prefill_batched(batch_rows: Option<usize>) -> (Vec<(usize, usize)>, Vec<Vec<u32>>, u64) {
    let clock = Rc::new(Cell::new(0));
    let mut toy = Toy::new(clock.clone(), true);
    toy.batch_rows = batch_rows;
    let calls = toy.calls.clone();
    let mut sched = Scheduler::new(toy, 16);
    let ids: Vec<u64> = (0..8).map(|i| sched.submit(Request { prompt: prompt(10 + i, 100), max_new: 5, stop: Vec::new() })).collect();
    let mut tokens: HashMap<u64, Vec<u32>> = HashMap::new();
    while sched.pending() {
        for (id, toks) in sched.step() {
            tokens.insert(id, toks);
        }
    }
    let calls = calls.borrow().clone();
    (calls, ids.iter().map(|id| tokens[id].clone()).collect(), clock.get())
}

#[test]
fn a_decoder_that_takes_rows_across_sequences_is_handed_several_prompts_slices_in_one_call() {
    let (calls, batched_tokens, batched_time) = run_prefill_batched(Some(64));
    let (_, one_by_one_tokens, _) = run_prefill_batched(None);
    assert_eq!(batched_tokens, one_by_one_tokens, "batching the prefill must not change any answer");
    assert!(!calls.is_empty(), "the scheduler never called prefill_chunks");
    assert!(calls.iter().all(|&(_, rows)| rows <= 64), "no call carries more than the decoder's row limit: {calls:?}");
    assert!(calls.iter().any(|&(jobs, rows)| jobs == 4 && rows == 64), "full calls gather four 16-row slices: {calls:?}");
    // 800 prompt tokens at 64 rows a call is 13 calls, not the 50 one-slice calls of 16 rows.
    assert!(calls.len() <= 14, "{} calls for 800 rows", calls.len());
    assert_eq!(calls.iter().map(|c| c.1).sum::<usize>(), 800, "every prompt token goes in exactly once");
    assert!(batched_time > 0);
}

#[test]
fn a_lone_prefilling_sequence_may_use_the_whole_row_budget() {
    let clock = Rc::new(Cell::new(0));
    let mut toy = Toy::new(clock, true);
    toy.batch_rows = Some(64);
    let calls = toy.calls.clone();
    let mut sched = Scheduler::new(toy, 4);
    sched.submit(Request { prompt: prompt(3, 100), max_new: 2, stop: Vec::new() });
    sched.run();
    let calls = calls.borrow().clone();
    assert_eq!(calls.iter().map(|c| c.0).max(), Some(1));
    assert_eq!(calls[0].1, 64, "alone, the sequence takes the whole budget in one call, not its 16-row share: {calls:?}");
}

#[test]
fn every_generated_token_has_a_logprob_reported_with_its_completion() {
    let clock = Rc::new(Cell::new(0));
    let mut sched = Scheduler::new(Toy::new(clock, false), 4);
    let id = sched.submit(Request { prompt: prompt(2, 10), max_new: 6, stop: Vec::new() });
    let mut tokens = Vec::new();
    let mut logprobs = Vec::new();
    while sched.pending() {
        let report = sched.step_report();
        for (done, toks) in report.completed {
            assert_eq!(done, id);
            tokens = toks;
        }
        for (done, lps) in report.completed_logprobs {
            assert_eq!(done, id);
            logprobs = lps;
        }
    }
    assert_eq!(tokens.len(), 6);
    // Greedy decoding is deterministic: each token had probability one.
    assert_eq!(logprobs, vec![0.0; 6]);
}

#[test]
fn a_cancelled_request_reports_the_logprobs_of_the_tokens_it_produced() {
    let clock = Rc::new(Cell::new(0));
    let mut sched = Scheduler::new(Toy::new(clock, false), 4);
    let id = sched.submit(Request { prompt: prompt(2, 10), max_new: 50, stop: Vec::new() });
    for _ in 0..3 {
        sched.step();
    }
    let (tokens, logprobs) = sched.cancel_detailed(id).expect("a running request");
    assert!(!tokens.is_empty());
    assert_eq!(tokens.len(), logprobs.len());
}
