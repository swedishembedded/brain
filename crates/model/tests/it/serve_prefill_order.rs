// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for LLM serving that keeps the first token of every conversation early
// while long prompts are read in beside it, for its clients. If your team needs expertise in continuous-batching
// schedulers then you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of the ONE order the scheduler reads prompts in, whatever path a decoder takes (whole prompts per call,
//! slices of one prompt per call, slices of several prompts per call):
//!
//! * the prompt with the fewest rows left is read first, so a short prompt is not queued behind a long one that
//!   arrived just before it (the first tokens come in order of prompt length);
//! * the oldest prompt is never starved: it earns rows every iteration it is passed over, and is read once they
//!   cover it (a decoder that reads whole prompts) or takes them as a slice (one that reads prompts in parts), so a
//!   stream of short prompts arriving every iteration delays it by a bounded number of iterations.

use std::collections::HashMap;

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{PagedDecoder, PrefillJob, PrefillProgress, Request, Scheduler};

const VOCAB: usize = 61;
const BLOCK: u32 = 8;
/// Rows a decoder that reads prompts in parts takes per call, and its per-iteration budget.
const SLICE: usize = 64;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    /// No slicing: every prefill call reads the whole prompt.
    Whole,
    /// One prompt's slice per call.
    Sliced,
    /// Slices of several prompts per call.
    Batched,
}

struct Toy {
    alloc: BlockAllocator,
    history: HashMap<u32, Vec<u32>>,
    mode: Mode,
}

impl Toy {
    fn digest(history: &[u32]) -> u32 {
        (history.iter().fold(7u64, |h, &t| h.wrapping_mul(1_000_003) ^ (t as u64 + 1)) >> 5) as u32 % VOCAB as u32
    }

    fn slice(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        let from = table.len() as usize;
        let to = (from + budget.max(1)).min(prompt.len());
        table.reserve((to - from) as u32, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        self.history.entry(table.blocks()[0]).or_default().extend_from_slice(&prompt[from..to]);
        if to == prompt.len() {
            PrefillProgress::Done(vec![Self::digest(&self.history[&table.blocks()[0]]) as f32])
        } else {
            PrefillProgress::Partial { consumed: to - from }
        }
    }
}

impl PagedDecoder for Toy {
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
        8192
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(b) = t.blocks().first() {
            self.history.remove(b);
        }
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        match self.slice(table, prompt, prompt.len()) {
            PrefillProgress::Done(h) => h,
            PrefillProgress::Partial { .. } => unreachable!("the whole prompt"),
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        (self.mode != Mode::Whole).then_some(SLICE)
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        match self.mode {
            Mode::Whole => self.slice(table, prompt, prompt.len()),
            _ => self.slice(table, prompt, budget),
        }
    }
    fn prefill_batch_tokens(&self) -> Option<usize> {
        (self.mode == Mode::Batched).then_some(2 * SLICE)
    }
    fn prefill_chunks(&mut self, jobs: &mut [PrefillJob<'_>]) -> Vec<PrefillProgress> {
        jobs.iter_mut().map(|j| self.slice(j.table, j.prompt, j.budget)).collect()
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[hidden[0] as usize] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &input)| {
                t.append(&mut self.alloc).unwrap();
                let h = self.history.get_mut(&t.blocks()[0]).unwrap();
                h.push(input);
                Self::digest(h)
            })
            .collect()
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
        u64::from(self.alloc.num_blocks()) * u64::from(BLOCK)
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

fn scheduler(mode: Mode) -> Scheduler<Toy> {
    Scheduler::new(Toy { alloc: BlockAllocator::new(8192, BLOCK), history: HashMap::new(), mode }, 64)
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (seed * 13 + i * 5 + i / 7) % VOCAB as u32).collect()
}

/// The iteration each request's first token came out in, for requests `(arrival iteration, prompt length)`.
fn first_tokens(mode: Mode, load: &[(usize, usize)]) -> Vec<usize> {
    let mut s = scheduler(mode);
    let mut ids = Vec::new();
    let mut first = vec![None; load.len()];
    for iteration in 0..100_000 {
        for (n, &(at, len)) in load.iter().enumerate() {
            if at == iteration {
                ids.push((n, s.submit(Request { prompt: prompt(n as u32 + 1, len), max_new: 4, stop: vec![] })));
            }
        }
        s.step();
        for &(n, id) in &ids {
            if first[n].is_none() && s.tokens_of(id).is_some_and(|t| !t.is_empty()) {
                first[n] = Some(iteration);
            }
        }
        if first.iter().all(Option::is_some) {
            return first.into_iter().map(Option::unwrap).collect();
        }
    }
    panic!("not every request produced a token");
}

#[test]
fn the_prompt_with_the_fewest_rows_left_is_read_first_on_every_path() {
    // Arrived together, the longest first.
    let load = [(0, 40 * SLICE), (0, 3 * SLICE), (0, 12 * SLICE), (0, SLICE / 2)];
    for mode in [Mode::Whole, Mode::Sliced, Mode::Batched] {
        let first = first_tokens(mode, &load);
        let mut by_time: Vec<usize> = (0..load.len()).collect();
        by_time.sort_by_key(|&n| (first[n], n));
        assert_eq!(by_time, vec![3, 1, 2, 0], "{mode:?}: the first tokens come in order of prompt length: {first:?}");
    }
}

#[test]
fn a_long_prompt_behind_a_stream_of_short_ones_is_not_starved() {
    // A long prompt, then a short one arriving every iteration for longer than the long one could ever wait.
    let long = 20 * SLICE;
    let mut load = vec![(0, long)];
    load.extend((0..400).map(|i| (i, SLICE / 4)));
    for mode in [Mode::Whole, Mode::Sliced, Mode::Batched] {
        let first = first_tokens(mode, &load);
        // It earns a slice's rows each iteration it is passed over: read within its length in slices, and a few more.
        assert!(first[0] <= 2 * long / SLICE + 4, "{mode:?}: the long prompt waited {} iterations", first[0]);
    }
}
