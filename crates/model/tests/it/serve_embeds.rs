// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multimodal LLM serving for clients whose
// pipelines feed images, audio or documents into a decoder. If your team needs
// expertise in batching multimodal prompts through one shared scheduler then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of prompts that carry embedding rows, and of the no-repeat n-gram filter.
//!
//! The decoder here keeps real KV contents: at an embedding position it stores
//! a value derived from the row, at a token position the token, and every
//! next token is a function of the whole cache. So a request whose rows were
//! lost, reordered or mixed up with another request's yields different tokens.
//!
//! The spec: rows reach the decoder at their positions through chunked prefill;
//! a decoder that takes tokens only refuses an embedded request cleanly and
//! keeps serving the others; malformed rows never queue; rows survive a
//! preemption by recompute (the pool is oversubscribed so sequences are
//! preempted) with the tokens the request produces alone; and the no-repeat
//! filter bans exactly what the logits form bans, at the cost of extra
//! candidates only on the steps that need them.

use std::collections::HashMap;

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{apply_no_repeat_ngram, NoRepeatNgram, PagedDecoder, PrefillProgress, PromptEmbeds, RejectReason, Request, SampleParams, Scheduler};

const VOCAB: usize = 64;
const BLOCK: u32 = 4;
const WIDTH: usize = 2;
/// Prompt positions prefilled per call, to force several slices.
const SLICE: usize = 5;

struct Mock {
    alloc: BlockAllocator,
    /// The value stored at each slot of each physical block.
    kv: HashMap<u32, Vec<u32>>,
    /// Whether the decoder takes embedding rows.
    embeds: bool,
    /// Candidates asked of each top-k step.
    topk_asked: Vec<usize>,
    /// Token the decoder prefers at each step if nothing is banned (set by the filter spec).
    script: Vec<u32>,
}

impl Mock {
    fn new(blocks: u32, embeds: bool) -> Mock {
        Mock { alloc: BlockAllocator::new(blocks, BLOCK), kv: HashMap::new(), embeds, topk_asked: Vec::new(), script: Vec::new() }
    }

    fn write(&mut self, block: u32, offset: u32, value: u32) {
        let slots = self.kv.entry(block).or_default();
        slots.truncate(offset as usize);
        slots.push(value);
    }

    fn history(&self, table: &BlockTable) -> Vec<u32> {
        (0..table.len()).map(|i| self.kv[&table.blocks()[(i / BLOCK) as usize]][(i % BLOCK) as usize]).collect()
    }

    /// Fill `[start, start + n)` of `table` from the prompt, rows where there are any.
    fn fill(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: Option<&PromptEmbeds>, budget: usize) -> PrefillProgress {
        let start = table.len() as usize;
        let n = budget.min(SLICE).min(prompt.len() - start);
        table.reserve(n as u32, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        for (p, &token) in prompt.iter().enumerate().skip(start).take(n) {
            let value = match embeds.and_then(|e| e.row_at(p as u32)) {
                Some(row) => 1_000_000 + (row[0] * 100.0) as u32 * 31 + (row[1] * 100.0) as u32,
                None => token,
            };
            self.write(table.blocks()[p / BLOCK as usize], p as u32 % BLOCK, value);
        }
        if start + n == prompt.len() {
            PrefillProgress::Done(vec![digest(&self.history(table)) as f32])
        } else {
            PrefillProgress::Partial { consumed: n }
        }
    }
}

fn digest(history: &[u32]) -> u32 {
    let h = history.iter().fold(7u64, |h, &t| h.wrapping_mul(1_000_003) ^ (t as u64 + 1));
    (h >> 5) as u32 & 0xFF_FFFF
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
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        match self.fill(table, prompt, None, usize::MAX) {
            PrefillProgress::Done(h) => h,
            PrefillProgress::Partial { .. } => unreachable!("never called for this mock: prefill_chunk is overridden"),
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(SLICE)
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        self.fill(table, prompt, None, budget)
    }
    fn embed_width(&self) -> Option<usize> {
        self.embeds.then_some(WIDTH)
    }
    fn prefill_chunk_embeds(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: &PromptEmbeds, budget: usize) -> PrefillProgress {
        self.fill(table, prompt, Some(embeds), budget)
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
                if self.script.is_empty() {
                    digest(&self.history(t)) % VOCAB as u32
                } else {
                    self.script[(t.len() as usize - 1) % self.script.len()]
                }
            })
            .collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        self.topk_asked.push(k);
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &input)| {
                let (block, offset) = t.append(&mut self.alloc).expect("the scheduler let a decode step run past the pool");
                self.write(block, offset, input);
                // Best first: the scripted token, then 1, 2, 3 ... in descending logit.
                let best = self.script[(t.len() as usize - 1) % self.script.len()];
                let mut c = vec![(best, 100.0)];
                c.extend((0..VOCAB as u32).filter(|&v| v != best).take(k.saturating_sub(1)).map(|v| (v, 50.0 - v as f32)));
                c.truncate(k);
                c
            })
            .collect()
    }
    fn topk_capacity(&self) -> usize {
        16
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
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (seed * 7 + i) % VOCAB as u32).collect()
}

/// A 12-token prompt whose positions 3..9 are rows derived from `seed`.
fn embedded(seed: u32) -> (Request, PromptEmbeds) {
    let rows: Vec<f32> = (0..6 * WIDTH as u32).map(|i| (seed * 3 + i) as f32 * 0.01 + 0.01).collect();
    (Request { prompt: prompt(seed, 12), max_new: 16, stop: Vec::new() }, PromptEmbeds::run(WIDTH, 3, rows).unwrap())
}

fn alone(seed: u32) -> Vec<u32> {
    let mut sched = Scheduler::new(Mock::new(256, true), 1);
    let (req, e) = embedded(seed);
    let id = sched.submit_embedded(req, e, SampleParams::greedy(), 0);
    sched.run().remove(&id).expect("finished")
}

#[test]
fn rows_reach_the_decoder_at_their_positions_through_chunked_prefill() {
    let mut sched = Scheduler::new(Mock::new(256, true), 4);
    let (ra, ea) = embedded(1);
    let (_, eb) = embedded(2);
    // Same placeholder tokens, different rows, in one batch.
    let rb = ra.clone_request();
    let ia = sched.submit_embedded(ra, ea, SampleParams::greedy(), 0);
    let ib = sched.submit_embedded(rb.clone_request(), eb.clone(), SampleParams::greedy(), 0);
    let done = sched.run();
    let mut solo = Scheduler::new(Mock::new(256, true), 1);
    let id = solo.submit_embedded(rb, eb, SampleParams::greedy(), 0);
    assert_eq!(done[&ia], alone(1), "a batched embedded request decodes as it does alone");
    assert_eq!(done[&ib], solo.run().remove(&id).unwrap());
    assert_ne!(done[&ia], done[&ib], "the rows decide the output, or this mock proves nothing");
}

trait CloneRequest {
    fn clone_request(&self) -> Request;
}
impl CloneRequest for Request {
    fn clone_request(&self) -> Request {
        Request { prompt: self.prompt.clone(), max_new: self.max_new, stop: self.stop.clone() }
    }
}

#[test]
fn a_decoder_without_rows_refuses_an_embedded_request_and_serves_the_others() {
    let mut sched = Scheduler::new(Mock::new(256, false), 4);
    let (req, e) = embedded(1);
    let refused = sched.submit_embedded(req, e, SampleParams::greedy(), 0);
    let plain = sched.submit(Request { prompt: prompt(3, 6), max_new: 4, stop: Vec::new() });
    let report = sched.step_report();
    assert_eq!(report.rejected.len(), 1);
    assert_eq!(report.rejected[0].0, refused);
    assert!(matches!(&report.rejected[0].1, RejectReason::InvalidEmbeds { why } if why.contains("does not take")), "{:?}", report.rejected);
    let done = sched.run();
    assert_eq!(done[&plain].len(), 4, "the token-only request is unaffected");
}

#[test]
fn rows_that_do_not_fit_the_request_are_never_queued() {
    let mut sched = Scheduler::new(Mock::new(256, true), 4);
    let p = Request { prompt: prompt(1, 6), max_new: 4, stop: Vec::new() };
    let past_end = PromptEmbeds::run(WIDTH, 4, vec![0.5; 3 * WIDTH]).unwrap();
    let wrong_width = PromptEmbeds::run(3, 0, vec![0.5; 3]).unwrap();
    let a = sched.submit_embedded(p.clone_request(), past_end, SampleParams::greedy(), 0);
    let b = sched.submit_embedded(p, wrong_width, SampleParams::greedy(), 0);
    let report = sched.step_report();
    let ids: Vec<u64> = report.rejected.iter().map(|r| r.0).collect();
    assert_eq!(ids, vec![a, b]);
    assert!(!sched.pending());
}

/// Six embedded requests of 12 prompt tokens + up to 16 generated (7 blocks
/// each) cannot all live in a 24-block pool: sequences are preempted by
/// recompute and must rebuild their cache, rows included.
#[test]
fn rows_survive_preemption_by_recompute() {
    let seeds = [1u32, 2, 3, 4, 5, 6];
    let mut sched = Scheduler::new(Mock::new(24, true), 8);
    let ids: Vec<u64> = seeds
        .iter()
        .map(|&s| {
            let (req, e) = embedded(s);
            sched.submit_embedded(req, e, SampleParams::greedy(), 0)
        })
        .collect();
    let mut recomputed = 0;
    let mut done = HashMap::new();
    while sched.pending() {
        let report = sched.step_report();
        recomputed += report.recomputed.len();
        done.extend(report.completed);
    }
    assert!(recomputed > 0, "the pool must be small enough that someone is preempted");
    for (&s, id) in seeds.iter().zip(ids) {
        assert_eq!(done[&id], alone(s), "request {s} diverged from its solo run after a recompute");
    }
}

#[test]
fn the_no_repeat_filter_bans_what_the_logits_form_bans() {
    let history: Vec<u32> = [1, 2, 3, 9, 1, 2, 3, 9, 1, 2].to_vec();
    let filter = NoRepeatNgram::new(3, 90, &[]);
    let banned = filter.banned(&history);
    let mut logits = vec![0.0f32; 16];
    apply_no_repeat_ngram(&mut logits, &history, 3, 90, &std::collections::HashSet::new());
    let from_logits: Vec<u32> = (0..16u32).filter(|&t| logits[t as usize] == f32::NEG_INFINITY).collect();
    let mut from_filter = banned.clone();
    from_filter.sort_unstable();
    assert_eq!(from_filter, from_logits);
    assert_eq!(from_filter, vec![3], "the trigram (1,2,3) was seen, so 3 is banned after (1,2)");
    // Allowed tokens are exempt.
    assert!(NoRepeatNgram::new(3, 90, &[3]).banned(&history).is_empty());
}

#[test]
fn a_step_that_bans_nothing_is_plain_greedy_and_a_banned_choice_is_stepped_over() {
    // The decoder wants the repeating script 1,2,3,9 forever; the filter must break the loop.
    let mut sched = Scheduler::new(Mock::new(256, true), 1);
    sched.decoder_mut().script = vec![1, 2, 3, 9];
    let sample = SampleParams { no_repeat: NoRepeatNgram::new(3, 90, &[]), ..SampleParams::greedy() };
    let id = sched.submit_sampled(Request { prompt: prompt(1, 4), max_new: 24, stop: Vec::new() }, sample, 0);
    let out = sched.run().remove(&id).unwrap();
    // No trigram repeats inside the generated text.
    for w in 3..=out.len() {
        let tri = &out[w - 3..w];
        assert!(!out[..w - 1].windows(3).any(|x| x == tri), "trigram {tri:?} repeated in {out:?}");
    }
    let asked = &sched.decoder_mut().topk_asked;
    assert!(!asked.is_empty(), "a banned token must have been stepped over via the candidate list");
    assert!(asked.iter().all(|&k| k <= 3), "candidates are asked only as wide as the ban needs: {asked:?}");
}
