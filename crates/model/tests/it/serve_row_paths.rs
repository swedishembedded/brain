// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sampling that stays exact at the full
// vocabulary for its clients. If your team needs expertise in decoding
// strategies for serving language models then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Every place the scheduler draws a token takes a decoder that reads its rows back from the whole row, not
//! from its candidate list: speculative verification, rollouts and mixed iterations, beside the plain decode
//! step that [`serve_sampling`](super::serve_sampling) covers.
//!
//! The toy decoder answers every position with the same flat 300-token row and offers a candidate list of only 4,
//! so a request with `top_p = 0.9` has a 270-token nucleus no list could hold. Its candidate entry points
//! panic: a path that still reads a list does not finish, and a path that reads the row draws tokens past the
//! list's width and none past the nucleus.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use data::rng::Rng;
use model::paged::{BlockAllocator, BlockTable};
use model::rollout::{PagedRollout, Rollout, RolloutParams};
use model::serve::{DraftQuery, Drafter, MixedLimits, PagedDecoder, PrefillProgress, Request, SampleParams, Scheduler};

const VOCAB: usize = 300;
const NUCLEUS: u32 = 270;
const LIST_WIDTH: usize = 4;
const BLOCK: u32 = 8;

#[derive(Default)]
struct Calls {
    /// Verify steps read back as whole rows.
    verify_rows: AtomicUsize,
    /// Decode steps read back as whole rows.
    decode_rows: AtomicUsize,
    /// Decode steps that followed a prompt slice in the same iteration: the mixed iteration.
    with_a_prompt_slice: AtomicUsize,
    /// A prompt slice went in and no decode step has followed it yet.
    slice_pending: AtomicUsize,
}

struct Flat {
    alloc: BlockAllocator,
    speculates: bool,
    mixed: bool,
    calls: Arc<Calls>,
}

impl Flat {
    fn new(speculates: bool, mixed: bool) -> (Flat, Arc<Calls>) {
        let calls = Arc::new(Calls::default());
        (Flat { alloc: BlockAllocator::new(512, BLOCK), speculates, mixed, calls: calls.clone() }, calls)
    }
}

impl PagedDecoder for Flat {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        4096
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
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        vec![0.0]
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        let (from, to) = (table.len() as usize, (table.len() as usize + budget.max(1)).min(prompt.len()));
        table.reserve((to - from) as u32, &mut self.alloc).unwrap();
        self.calls.slice_pending.store(1, Ordering::Relaxed);
        if to == prompt.len() {
            PrefillProgress::Done(vec![0.0])
        } else {
            PrefillProgress::Partial { consumed: to - from }
        }
    }
    fn logits(&self, _hidden: &[f32]) -> Vec<f32> {
        vec![1.0; VOCAB]
    }
    fn forward_batched_greedy(&mut self, _: &mut [&mut BlockTable], _: &[u32]) -> Vec<u32> {
        unreachable!("every request here samples")
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
        512 * BLOCK as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        panic!("a request wider than the candidate list was sampled from the list");
    }
    fn topk_capacity(&self) -> usize {
        LIST_WIDTH
    }
    fn admit_topk(&self, _: &[f32], _: usize) -> Vec<(u32, f32)> {
        panic!("a request wider than the candidate list took its first token from the list");
    }
    fn full_row_readback(&self) -> bool {
        true
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        self.calls.decode_rows.fetch_add(1, Ordering::Relaxed);
        if self.calls.slice_pending.swap(0, Ordering::Relaxed) == 1 {
            self.calls.with_a_prompt_slice.fetch_add(1, Ordering::Relaxed);
        }
        assert_eq!(tables.len(), inputs.len());
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|_| vec![1.0; VOCAB]).collect()
    }
    fn mixed_step(&self) -> Option<MixedLimits> {
        self.mixed.then_some(MixedLimits { max_rows: 64, min_prompt_rows: 8, row_quantum: 1 })
    }
    fn spec_max_draft(&self) -> usize {
        if self.speculates {
            4
        } else {
            0
        }
    }
    fn forward_batched_verify(&mut self, _: &mut [&mut BlockTable], _: &[Vec<u32>], _: usize) -> Vec<Vec<Vec<(u32, f32)>>> {
        panic!("a request wider than the candidate list was verified against the list");
    }
    fn forward_batched_verify_logit_rows(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>]) -> Vec<Vec<Vec<f32>>> {
        self.calls.verify_rows.fetch_add(1, Ordering::Relaxed);
        if self.calls.slice_pending.swap(0, Ordering::Relaxed) == 1 {
            self.calls.with_a_prompt_slice.fetch_add(1, Ordering::Relaxed);
        }
        tables
            .iter_mut()
            .zip(rows)
            .map(|(t, r)| {
                r.iter()
                    .map(|_| {
                        t.append(&mut self.alloc).unwrap();
                        vec![1.0; VOCAB]
                    })
                    .collect()
            })
            .collect()
    }
}

/// Proposes token 100, inside the nucleus, as many times as asked.
struct Hundred;
impl Drafter for Hundred {
    fn draft(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>> {
        queries.iter().zip(ks).map(|(_, &k)| vec![100; k]).collect()
    }
}

fn wide_nucleus() -> SampleParams {
    SampleParams { temp: 1.0, top_p: 0.9, ..SampleParams::greedy() }
}

fn assert_drawn_from_the_nucleus(drawn: &[u32]) {
    assert!(drawn.iter().any(|&t| t as usize >= LIST_WIDTH), "every draw stayed inside the {LIST_WIDTH}-token candidate list");
    assert!(drawn.iter().all(|&t| t < NUCLEUS), "a draw left the {NUCLEUS}-token nucleus: {drawn:?}");
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1 + i % 50).collect()
}

#[test]
fn speculative_verification_samples_a_nucleus_wider_than_the_candidate_list() {
    let (dec, calls) = Flat::new(true, false);
    let mut sched = Scheduler::new(dec, 4);
    sched.set_drafter(Box::new(Hundred));
    let id = sched.submit_sampled(Request { prompt: prompt(5), max_new: 300, stop: Vec::new() }, wide_nucleus(), 1);
    let drawn = sched.run().remove(&id).expect("the request completes");
    assert_eq!(drawn.len(), 300);
    assert!(calls.verify_rows.load(Ordering::Relaxed) > 0, "no verify step ran, so speculation was not exercised");
    assert_drawn_from_the_nucleus(&drawn);
}

#[test]
fn a_mixed_iteration_samples_a_nucleus_wider_than_the_candidate_list() {
    let (dec, calls) = Flat::new(false, true);
    let mut sched = Scheduler::new(dec, 4);
    let first = sched.submit_sampled(Request { prompt: prompt(5), max_new: 120, stop: Vec::new() }, wide_nucleus(), 1);
    // A long prompt arrives while the first request decodes, so its slices ride mixed iterations.
    let mut out = std::collections::HashMap::new();
    for step in 0..2000 {
        if step == 3 {
            sched.submit_sampled(Request { prompt: prompt(400), max_new: 120, stop: Vec::new() }, wide_nucleus(), 2);
        }
        out.extend(sched.step());
        if !sched.pending() {
            break;
        }
    }
    assert_eq!(out.len(), 2, "both requests complete");
    assert_drawn_from_the_nucleus(&out[&first]);
    assert!(calls.with_a_prompt_slice.load(Ordering::Relaxed) > 0, "no decode step shared an iteration with a prompt slice, so no mixed iteration ran");
}

#[test]
fn a_mixed_speculative_iteration_verifies_against_whole_rows() {
    let (dec, calls) = Flat::new(true, true);
    let mut sched = Scheduler::new(dec, 4);
    sched.set_drafter(Box::new(Hundred));
    let first = sched.submit_sampled(Request { prompt: prompt(5), max_new: 200, stop: Vec::new() }, wide_nucleus(), 1);
    let mut out = std::collections::HashMap::new();
    for step in 0..4000 {
        if step == 2 {
            sched.submit_sampled(Request { prompt: prompt(400), max_new: 40, stop: Vec::new() }, wide_nucleus(), 2);
        }
        out.extend(sched.step());
        if !sched.pending() {
            break;
        }
    }
    assert_eq!(out.len(), 2, "both requests complete");
    assert_drawn_from_the_nucleus(&out[&first]);
    assert!(calls.with_a_prompt_slice.load(Ordering::Relaxed) > 0, "no verify step shared an iteration with a prompt slice, so no mixed iteration ran");
}

#[test]
fn a_rollout_samples_a_nucleus_wider_than_the_candidate_list() {
    let (dec, _) = Flat::new(false, false);
    let mut rollout = PagedRollout::new(dec);
    let params = RolloutParams { max_new: 300, sample: wide_nucleus(), eos: None };
    let done = rollout.sample_n(&prompt(5), 1, &params, &mut Rng::new(3));
    assert_drawn_from_the_nucleus(&done[0].tokens);
    // The reported log-probability is the nucleus's: each of its 270 tokens has mass 1/270.
    for lp in &done[0].logprobs {
        assert!((lp + (NUCLEUS as f32).ln()).abs() < 1e-4, "log-probability {lp}");
    }
}
