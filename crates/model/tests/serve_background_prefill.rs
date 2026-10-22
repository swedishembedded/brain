// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for low-latency multi-tenant LLM
// serving for its clients. If your team needs expertise in serving that keeps
// every running conversation generating while new prompts are read in, then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of background prefill: a decoder that reads prompts in on its own
//! (another stream of the device) while the scheduler keeps decoding.
//!
//! The toy advances its prompt by a few tokens whenever it is asked about it
//! during an iteration that also decodes - the work the device did while the
//! step ran - and finishes it outright when the scheduler says nothing else
//! runs. The spec: a running sequence gets a token every iteration however
//! many prompts are behind it; the scheduler hands over every prompt still
//! being read, oldest first; it never spins on a decoder that is told nothing
//! else runs; and the tokens are the ones a foreground prefill produces.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{PagedDecoder, PrefillJob, PrefillProgress, Request, Scheduler};

const VOCAB: usize = 64;
/// Prompt tokens the toy reads in per iteration that also decodes.
const BACKGROUND_RATE: usize = 8;

#[derive(Default)]
struct Log {
    /// Per `prefill_background` call: (jobs handed over, whether told to wait).
    calls: Vec<(usize, bool)>,
    /// Per decode step: how many sequences decoded.
    decodes: Vec<usize>,
    /// Per scheduler iteration: whether the decoder was told to wait, and
    /// whether a sequence was already running when the iteration began.
    iterations: Vec<(Option<bool>, bool)>,
}

/// One scheduler iteration, logging whether it waited for prefill and whether a sequence was running.
fn step(s: &mut Scheduler<Toy>, log: &Rc<RefCell<Log>>) -> Vec<(u64, Vec<u32>)> {
    let calls = log.borrow().calls.len();
    let running = s.running_len() > 0;
    let out = s.step();
    let mut l = log.borrow_mut();
    let waited = l.calls.get(calls).map(|&(_, wait)| wait);
    l.iterations.push((waited, running));
    out
}

struct Toy {
    alloc: BlockAllocator,
    /// Prompt tokens read in so far, by the sequence's first block.
    progress: HashMap<u32, usize>,
    log: Rc<RefCell<Log>>,
}

impl Toy {
    fn new(log: Rc<RefCell<Log>>) -> Toy {
        Toy { alloc: BlockAllocator::new(512, 8), progress: HashMap::new(), log }
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
        vec![*prompt.last().unwrap() as f32]
    }
    fn background_prefill(&self) -> bool {
        true
    }
    /// One prompt at a time, the oldest: it advances by [`BACKGROUND_RATE`]
    /// tokens, or to its end when told nothing else runs.
    fn prefill_background(&mut self, jobs: &mut [PrefillJob<'_>], wait: bool) -> Vec<PrefillProgress> {
        self.log.borrow_mut().calls.push((jobs.len(), wait));
        let mut out = Vec::new();
        let mut active = true;
        for j in jobs.iter_mut() {
            if j.table.is_empty() {
                j.table.reserve(j.prompt.len() as u32, &mut self.alloc).unwrap();
            }
            if !active {
                out.push(PrefillProgress::Partial { consumed: 0 });
                continue;
            }
            active = false;
            let slot = j.table.blocks()[0];
            let done = self.progress.entry(slot).or_insert(0);
            let n = if wait { j.prompt.len() - *done } else { BACKGROUND_RATE.min(j.prompt.len() - *done) };
            assert!(n <= j.budget, "the toy read past the budget it was given");
            *done += n;
            if *done == j.prompt.len() {
                self.progress.remove(&slot);
                out.push(PrefillProgress::Done(vec![*j.prompt.last().unwrap() as f32]));
            } else {
                out.push(PrefillProgress::Partial { consumed: n });
            }
        }
        out
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[(hidden[0] as usize + 1) % VOCAB] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.log.borrow_mut().decodes.push(tables.len());
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
    (0..len as u32).map(|i| (seed * 7 + i) % VOCAB as u32).collect()
}

#[test]
fn a_running_sequence_gets_a_token_every_iteration_while_prompts_are_read_in_behind_it() {
    let log = Rc::new(RefCell::new(Log::default()));
    let mut s = Scheduler::new(Toy::new(log.clone()), 8);
    // A short request first, then three long prompts arrive while it decodes.
    let short = s.submit(Request { prompt: prompt(1, 4), max_new: 30, stop: vec![] });
    while s.running_len() == 0 {
        step(&mut s, &log);
    }
    let long: Vec<u64> = (2..5).map(|i| s.submit(Request { prompt: prompt(i, 100), max_new: 3, stop: vec![] })).collect();
    let mut short_tokens_per_iteration = Vec::new();
    let mut done = HashMap::new();
    while done.len() < 4 {
        let before = log.borrow().decodes.len();
        for (id, toks) in step(&mut s, &log) {
            done.insert(id, toks);
        }
        if !done.contains_key(&short) {
            short_tokens_per_iteration.push(log.borrow().decodes.len() - before);
        }
    }
    assert!(short_tokens_per_iteration.iter().all(|&n| n == 1), "the running sequence skipped a decode step while prompts were read in: {short_tokens_per_iteration:?}");
    assert_eq!(done[&short].len(), 30);
    for (i, id) in long.iter().enumerate() {
        let p = prompt(i as u32 + 2, 100);
        let first = (*p.last().unwrap() + 1) % VOCAB as u32;
        assert_eq!(done[id], vec![first, (first + 1) % VOCAB as u32, (first + 2) % VOCAB as u32], "request {id}");
    }
    let log = log.borrow();
    // Every prompt still being read is handed over each iteration.
    assert!(log.calls.iter().any(|&(jobs, _)| jobs == 3), "the three waiting prompts were never handed over together: {:?}", log.calls);
    // Told to wait exactly in the iterations that begin with nothing running.
    for &(waited, running) in &log.iterations {
        if let Some(waited) = waited {
            assert_eq!(waited, !running, "an iteration with {} running told the decoder to wait={waited}: {:?}", if running { "a sequence" } else { "nothing" }, log.iterations);
        }
    }
}

#[test]
fn with_nothing_decoding_the_decoder_is_told_to_wait_and_is_never_spun() {
    let log = Rc::new(RefCell::new(Log::default()));
    let mut s = Scheduler::new(Toy::new(log.clone()), 8);
    let ids: Vec<u64> = (0..2).map(|i| s.submit(Request { prompt: prompt(i, 100), max_new: 2, stop: vec![] })).collect();
    let mut done = HashMap::new();
    let mut iterations = 0;
    while done.len() < ids.len() {
        iterations += 1;
        assert!(iterations < 50, "the scheduler spun without finishing");
        done.extend(step(&mut s, &log));
    }
    let log = log.borrow();
    assert!(log.calls.first().is_some_and(|&(_, wait)| wait), "the first iteration decodes nothing and must wait: {:?}", log.calls);
}
